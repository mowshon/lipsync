import numpy as np
import cv2
import os
from lipsync.wav2lip import audio
import subprocess
from tqdm import tqdm
import torch
from lipsync.wav2lip import face_detection
from lipsync.wav2lip.models import Wav2Lip
import tempfile
import pickle
import moviepy.editor as mp


class LipSync:

    # Name of saved checkpoint to load weights from.
    checkpoint_path = ''

    # If True, then use only first video frame for inference.
    static = False

    # Can be specified only if input is a static image (default: 25).
    fps = 25.

    # Padding (top, bottom, left, right). Please adjust to include chin at least.
    pads = [0, 10, 0, 0]

    # Batch size for face detection.
    face_det_batch_size = 16

    # Batch size for Wav2Lip model(s).
    wav2lip_batch_size = 128

    # Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p.
    resize_factor = 1

    # Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
    # 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
    crop = [0, -1, 0, -1]

    # Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
    # 'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
    box = [-1, -1, -1, -1]

    # Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
    # 'Use if you get a flipped result, despite feeding a normal looking video
    rotate = False

    # Prevent smoothing face detections over a short temporal window
    nosmooth = False

    # This option enables caching for face positioning on every frame.
    # Useful if you will be using the same video, but different audio.
    save_cache = True

    cache_dir = tempfile.gettempdir()

    _filepath = ''

    img_size = 96

    mel_step_size = 16

    device = 'cuda'

    # Levels: quiet, panic, fatal, error, warning, info, verbose, debug
    ffmpeg_loglevel = 'verbose'

    def __init__(self, **kwargs):
        if 'device' not in kwargs.keys():
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def get_smoothened_boxes(boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_cache_filename(self):
        filename = os.path.basename(self._filepath)
        return os.path.join(self.cache_dir, f'{filename}.pk')

    def get_from_cache(self):
        if not self.save_cache:
            return False

        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            with open(cache_filename, 'rb') as cached_file:
                print(f'Load cache: {cache_filename}')
                return pickle.load(cached_file)

        return False

    def face_detect(self, images):
        cache = self.get_from_cache()
        if cache:
            return cache

        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=self.device
        )

        batch_size = self.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please set the resize_factor parameter to True'
                    )
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector

        if self.save_cache:
            with open(self.get_cache_filename(), 'wb') as cached_file:
                pickle.dump(results, cached_file)

        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(self.checkpoint_path))
        checkpoint = self._load(self.checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    @staticmethod
    def create_temp_file(ext):
        temp, filename = tempfile.mkstemp()
        os.close(temp)

        return f'{filename}.{ext}'

    def sync(self, face, audio_file, outfile):
        self._filepath = face

        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid path to video/image file')

        if face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.static = True
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            print('Reading video frames...')
            video = mp.VideoFileClip(face)
            full_frames = [cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR) for frame_np in video.iter_frames()]
            fps = video.fps

        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')

            filename = self.create_temp_file('wav')
            command = 'ffmpeg -y -i {} -strict -2 {} -loglevel {}'.format(
                audio_file, filename, self.ffmpeg_loglevel
            )

            subprocess.call(command, shell=True)
            audio_file = filename

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        #full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        temp_result_avi = self.create_temp_file('avi')
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(np.ceil(
                                                                            float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                model = self.load_model()
                print("Model loaded")

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(
                    temp_result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h)
                )

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel {}'.format(
            audio_file, temp_result_avi, outfile, self.ffmpeg_loglevel
        )
        subprocess.call(command, shell=True)

        return outfile
