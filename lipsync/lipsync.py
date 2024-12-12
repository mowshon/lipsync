import numpy as np
import cv2
import os
import subprocess
from tqdm import tqdm
import torch
import tempfile
import pickle

from lipsync import audio
from lipsync.wav2lip import face_detection
from lipsync.helpers import read_frames
from lipsync.models import load_model


class LipSync:
    """
    Class for lip-syncing videos using the Wav2Lip model.

    Attributes:
        checkpoint_path (str): Path to the saved Wav2Lip model checkpoint.
        static (bool): If True, uses only the first video frame for inference.
        fps (float): Frames per second, used when input is a static image.
        pads (List[int]): Padding for face bounding boxes as [top, bottom, left, right].
        face_det_batch_size (int): Batch size for face detection inference.
        wav2lip_batch_size (int): Batch size for Wav2Lip model inference.
        resize_factor (int): Factor by which to reduce the resolution of the input frames.
        crop (List[int]): Crop dimensions [top, bottom, left, right] for input frames.
        box (List[int]): Fixed bounding box coordinates [y1, y2, x1, x2] for the face.
        rotate (bool): If True, rotates the input frames by 90 degrees.
        nosmooth (bool): If True, disables temporal smoothing of face bounding boxes.
        save_cache (bool): If True, enables caching of face detection results.
        cache_dir (str): Directory path to store cached face detection results.
        _filepath (str): Internal path to the input face/video file.
        img_size (int): Size to which the face region is resized for model input.
        mel_step_size (int): Step size for the mel spectrogram chunks fed into the model.
        device (str): Device on which the model runs, either 'cuda' or 'cpu'.
        ffmpeg_loglevel (str): Log level for ffmpeg operations, e.g. 'verbose'.
        model (str): Name of the model to load (e.g., 'wav2lip').
    """

    # Default parameters
    checkpoint_path: str = ''
    static: bool = False
    fps: float = 25.0
    pads: list[int] = [0, 10, 0, 0]
    face_det_batch_size: int = 16
    wav2lip_batch_size: int = 128
    resize_factor: int = 1
    crop: list[int] = [0, -1, 0, -1]
    box: list[int] = [-1, -1, -1, -1]
    rotate: bool = False
    nosmooth: bool = False
    save_cache: bool = True
    cache_dir: str = tempfile.gettempdir()
    _filepath: str = ''
    img_size: int = 96
    mel_step_size: int = 16
    device: str = 'cpu'
    ffmpeg_loglevel: str = 'verbose'
    model: str = 'wav2lip'

    def __init__(self, **kwargs):
        # Check if CUDA is actually available before setting device
        device = kwargs.get('device', self.device)
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        print('Using device:', self.device)

        # Update class attributes with provided keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_smoothened_boxes(boxes: np.ndarray, t: int) -> np.ndarray:
        """
        Smoothens bounding boxes over a temporal window of size t.
        """
        length = len(boxes)
        for i in range(length):
            window_end = min(i + t, length)
            window = boxes[i:window_end]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_cache_filename(self) -> str:
        """
        Generates a filename for caching face detection results.
        """
        filename = os.path.basename(self._filepath)
        return os.path.join(self.cache_dir, f'{filename}.pk')

    def get_from_cache(self):
        """
        Retrieves face detection results from cache if available.
        """
        if not self.save_cache:
            return False

        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            print(f'Loading from cache: {cache_filename}')
            with open(cache_filename, 'rb') as cached_file:
                return pickle.load(cached_file)

        return False

    def face_detect(self, images: list) -> list:
        """
        Performs face detection on a list of images.
        """
        # Attempt to load from cache
        cache = self.get_from_cache()
        if cache:
            return cache

        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=self.device
        )

        batch_size = self.face_det_batch_size
        predictions = []

        # Efficient batch handling with fallback if OOM occurs
        while True:
            try:
                for i in tqdm(range(0, len(images), batch_size), desc="Face Detection"):
                    batch = np.array(images[i:i + batch_size])
                    preds = detector.get_detections_for_batch(batch)
                    predictions.extend(preds)
            except RuntimeError:
                # Decrease batch size if we run into GPU memory issues
                if batch_size == 1:
                    del detector
                    raise RuntimeError('Image too large for GPU. Try reducing resize_factor.')
                batch_size //= 2
                print(f'OOM encountered. Reducing batch size to {batch_size}. Retrying...')
                predictions.clear()
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        img_h, img_w = images[0].shape[:2]

        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                del detector
                raise ValueError('Face not detected! Make sure every frame has a detectable face.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(img_h, rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(img_w, rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, t=5)

        # Extract cropped faces
        cropped_results = []
        for (x1, y1, x2, y2), image in zip(boxes, images):
            face_img = image[int(y1): int(y2), int(x1): int(x2)]
            cropped_results.append([face_img, (int(y1), int(y2), int(x1), int(x2))])

        del detector

        # Save to cache if enabled
        if self.save_cache:
            with open(self.get_cache_filename(), 'wb') as cached_file:
                pickle.dump(cropped_results, cached_file)

        return cropped_results

    def datagen(self, frames: list, mels: list):
        """
        Generator that yields batches of images and mel spectrograms.
        """
        if self.box[0] == -1:
            # Perform detection on all frames if not static, else only on the first frame
            face_det_results = self.face_detect(frames if not self.static else [frames[0]])
        else:
            # Using a fixed bounding box
            print('Using specified bounding box, skipping face detection.')
            y1, y2, x1, x2 = self.box
            face_det_results = [
                [f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames
            ]

        img_batch = []
        mel_batch = []
        frame_batch = []
        coords_batch = []
        batch_size = self.wav2lip_batch_size

        # Pre-allocate arrays if possible (optional optimization)
        # Since mel and frames vary in size, and we might not know exact batch count in advance,
        # we will stick to appending.

        for i, m in enumerate(mels):
            idx = 0 if self.static else (i % len(frames))
            frame_to_save = frames[idx]
            face, coords = face_det_results[idx]
            face_resized = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face_resized)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            # Yield batch if full
            if len(img_batch) >= batch_size:
                yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch.clear()
                mel_batch.clear()
                frame_batch.clear()
                coords_batch.clear()

        # Yield the remaining batch
        if len(img_batch) > 0:
            yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)

    def _prepare_batch(self, img_batch, mel_batch, frame_batch, coords_batch):
        """
        Prepare data batch tensors from lists for model inference.
        """
        img_batch_np = np.asarray(img_batch, dtype=np.uint8)
        mel_batch_np = np.asarray(mel_batch, dtype=np.float32)

        # Mask the lower half of the image
        img_masked = img_batch_np.copy()
        half = self.img_size // 2
        img_masked[:, half:] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
        mel_batch_np = mel_batch_np[..., np.newaxis]  # Add channel dimension

        return img_batch_np, mel_batch_np, frame_batch, coords_batch

    @staticmethod
    def create_temp_file(ext: str) -> str:
        temp_fd, filename = tempfile.mkstemp()
        os.close(temp_fd)
        return f'{filename}.{ext}'

    def sync(self, face: str, audio_file: str, outfile: str) -> str:
        """
        Performs lip-sync on the input video/image using the provided audio.
        """
        self._filepath = face

        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid file path.')

        if face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            self.static = True
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            print('Reading video frames...')
            full_frames, fps = read_frames(face)

        print(f"Frames for inference: {len(full_frames)}")

        # Convert non-wav audio to wav if needed
        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')
            wav_filename = self.create_temp_file('wav')
            command = (
                f'ffmpeg -y -i "{audio_file}" -strict -2 "{wav_filename}" '
                f'-loglevel {self.ffmpeg_loglevel}'
            )
            subprocess.run(command, shell=True, check=True)
            audio_file = wav_filename

        # Load and process audio
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel).any():
            raise ValueError('Mel contains NaN! Add a small epsilon to the audio and try again.')

        print(f"Mel spectrogram shape: {mel.shape}")

        # Split mel spectrogram into chunks
        mel_chunks = self._split_mel_chunks(mel, fps)

        # Load model once
        model = load_model(self.model, self.device, self.checkpoint_path)
        print("Model loaded")

        # Prepare video writer
        frame_h, frame_w = full_frames[0].shape[:2]
        temp_result_avi = self.create_temp_file('avi')
        out = cv2.VideoWriter(
            temp_result_avi,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (frame_w, frame_h),
        )

        # Run inference batches
        data_generator = self.datagen(full_frames.copy(), mel_chunks)
        total_batches = int(np.ceil(float(len(mel_chunks)) / self.wav2lip_batch_size))
        for (img_batch_np, mel_batch_np, frames, coords) in tqdm(data_generator, total=total_batches, desc="Lip-sync Inference"):
            # Convert to torch tensors
            img_batch_t = torch.FloatTensor(np.transpose(img_batch_np, (0, 3, 1, 2))).to(self.device)
            mel_batch_t = torch.FloatTensor(np.transpose(mel_batch_np, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch_t, img_batch_t)

            # Convert predictions to NumPy and write to output
            pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            for p, f, c in zip(pred_np, frames, coords):
                y1, y2, x1, x2 = c
                p_resized = cv2.resize(p, (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p_resized
                out.write(f)

        out.release()

        # Combine output with original audio
        command = (
            f'ffmpeg -y -i "{audio_file}" -i "{temp_result_avi}" -strict -2 '
            f'-q:v 1 "{outfile}" -loglevel {self.ffmpeg_loglevel}'
        )
        subprocess.run(command, shell=True, check=True)

        return outfile

    def _split_mel_chunks(self, mel: np.ndarray, fps: float) -> list:
        """
        Splits the mel spectrogram into fixed-size chunks for inference.
        """
        mel_chunks = []
        mel_length = mel.shape[1]
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            end_idx = start_idx + self.mel_step_size
            if end_idx > mel_length:
                mel_chunks.append(mel[:, -self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:end_idx])
            i += 1
        print(f"Total mel chunks: {len(mel_chunks)}")
        return mel_chunks
