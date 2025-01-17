import numpy as np
import cv2
import os
import subprocess
from tqdm import tqdm
import torch
import tempfile
import pickle
from lipsync import audio
from lipsync.helpers import read_frames, get_face_box
from lipsync.models import load_model
from typing import List, Tuple, Union
import face_alignment


class LipSync:
    """
    Class for lip-syncing videos using the Wav2Lip model.
    """

    # Default parameters
    checkpoint_path: str = ''
    static: bool = False
    fps: float = 25.0
    pads: List[int] = [0, 10, 0, 0]
    wav2lip_batch_size: int = 128
    nosmooth: bool = False
    save_cache: bool = True
    cache_dir: str = tempfile.gettempdir()
    img_size: int = 96
    mel_step_size: int = 16
    device: str = 'cpu'
    ffmpeg_loglevel: str = 'verbose'
    model: str = 'wav2lip'

    _filepath: str = ''

    def __init__(self, **kwargs):
        """
        Initializes LipSync with custom parameters.
        """
        device = kwargs.get('device', self.device)
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'

        # Update class attributes with provided keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_smoothened_boxes(boxes: np.ndarray, t: int) -> np.ndarray:
        """
        Smoothens bounding boxes over a temporal window.
        """
        for i in range(len(boxes)):
            window_end = min(i + t, len(boxes))
            window = boxes[i:window_end]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_cache_filename(self) -> str:
        """
        Generates a filename for caching face detection results.
        """
        filename = os.path.basename(self._filepath)
        return os.path.join(self.cache_dir, f'{filename}.pk')

    def get_from_cache(self) -> Union[List, bool]:
        """
        Retrieves face detection results from cache if available.
        """
        if not self.save_cache:
            return False

        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            with open(cache_filename, 'rb') as cached_file:
                return pickle.load(cached_file)

        return False

    def detect_faces_in_frames(self, images: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frames using face_alignment.
        """
        detector = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.TWO_D,
            face_detector='sfd',
            device=self.device
        )

        predictions = []
        for i in tqdm(range(0, len(images)), desc="Face Detection"):
            landmarks = detector.get_landmarks_from_image(images[i], return_bboxes=True)
            predictions.append(get_face_box(landmarks))

        del detector
        return predictions

    def process_face_boxes(self, predictions: List, images: List[np.ndarray]) -> List[List]:
        """
        Process face bounding boxes, apply smoothing, and crop faces.
        """
        pady1, pady2, padx1, padx2 = self.pads
        img_h, img_w = images[0].shape[:2]

        # Convert predictions to bounding boxes
        results = []
        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('Face not detected! Ensure all frames contain a face.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(img_h, rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(img_w, rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        # Smooth bounding boxes if needed
        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, t=5)

        cropped_results = []
        for (x1, y1, x2, y2), image in zip(boxes, images):
            face_img = image[int(y1): int(y2), int(x1): int(x2)]
            cropped_results.append([face_img, (int(y1), int(y2), int(x1), int(x2))])

        return cropped_results

    def face_detect(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Performs face detection on a list of images.
        """
        cache = self.get_from_cache()
        if cache:
            return cache

        predictions = self.detect_faces_in_frames(images)
        cropped_results = self.process_face_boxes(predictions, images)

        # Cache results if enabled
        if self.save_cache:
            with open(self.get_cache_filename(), 'wb') as cached_file:
                pickle.dump(cropped_results, cached_file)

        return cropped_results

    def datagen(
        self,
        frames: List[np.ndarray],
        mels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Generator that yields batches of images and mel spectrogram chunks.
        """
        face_det_results = self.face_detect(frames)

        batch_size = self.wav2lip_batch_size
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, m in enumerate(mels):
            idx = 0 if self.static else (i % len(frames))
            frame_to_save = frames[idx]
            face, coords = face_det_results[idx]
            face_resized = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face_resized)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= batch_size:
                yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
                img_batch.clear()
                mel_batch.clear()
                frame_batch.clear()
                coords_batch.clear()

        # Yield remaining batch if any
        if len(img_batch) > 0:
            yield self._prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)

    def _prepare_batch(
        self,
        img_batch: List[np.ndarray],
        mel_batch: List[np.ndarray],
        frame_batch: List[np.ndarray],
        coords_batch: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Prepares a batch of images and mel spectrograms for inference.
        """
        img_batch_np = np.asarray(img_batch, dtype=np.uint8)
        mel_batch_np = np.asarray(mel_batch, dtype=np.float32)

        # Mask the lower half of the image
        half = self.img_size // 2
        img_masked = img_batch_np.copy()
        img_masked[:, half:] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
        mel_batch_np = mel_batch_np[..., np.newaxis]

        return img_batch_np, mel_batch_np, frame_batch, coords_batch

    @staticmethod
    def create_temp_file(ext: str) -> str:
        """
        Creates a temporary file with a specific extension.
        """
        temp_fd, filename = tempfile.mkstemp()
        os.close(temp_fd)
        return f'{filename}.{ext}'

    def sync(self, face: str, audio_file: str, outfile: str) -> str:
        """
        Performs lip-syncing on the input video/image using the provided audio.
        """
        self._filepath = face
        full_frames, fps = self._load_input_face(face)
        audio_file = self._prepare_audio(audio_file)
        mel = self._generate_mel_spectrogram(audio_file)
        mel_chunks = self._split_mel_chunks(mel, fps)
        model = self._load_model_for_inference()

        temp_result_avi = self.create_temp_file('avi')
        out = self._prepare_video_writer(temp_result_avi, full_frames[0].shape[:2], fps)

        self._perform_inference(model, full_frames, mel_chunks, out)
        out.release()

        self._merge_audio_video(audio_file, temp_result_avi, outfile)
        return outfile

    def _load_input_face(self, face: str) -> Tuple[List[np.ndarray], float]:
        """
        Loads the input face (video or image) and returns frames and fps.
        """
        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid file path.')

        if face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            self.static = True
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            full_frames, fps = read_frames(face)

        return full_frames, fps

    def _prepare_audio(self, audio_file: str) -> str:
        """
        Prepares (extracts) raw audio if not in .wav format.
        """
        if not audio_file.endswith('.wav'):
            wav_filename = self.create_temp_file('wav')
            command = (
                f'ffmpeg -y -i "{audio_file}" -strict -2 "{wav_filename}" '
                f'-loglevel {self.ffmpeg_loglevel}'
            )
            subprocess.run(command, shell=True, check=True)
            audio_file = wav_filename
        return audio_file

    @staticmethod
    def _generate_mel_spectrogram(audio_file: str) -> np.ndarray:
        """
        Generates the mel spectrogram from the given audio file.
        """
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel).any():
            raise ValueError('Mel contains NaN! Add a small epsilon to the audio and try again.')
        return mel

    def _load_model_for_inference(self) -> torch.nn.Module:
        """
        Loads the lip sync model for inference.
        """
        model = load_model(self.model, self.device, self.checkpoint_path)
        return model

    @staticmethod
    def _prepare_video_writer(filename: str, frame_shape: Tuple[int, int], fps: float) -> cv2.VideoWriter:
        """
        Prepares the VideoWriter for output.
        """
        frame_h, frame_w = frame_shape
        return cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (frame_w, frame_h),
        )

    def _perform_inference(
        self,
        model: torch.nn.Module,
        full_frames: List[np.ndarray],
        mel_chunks: List[np.ndarray],
        out: cv2.VideoWriter,
    ):
        """
        Runs the inference loop: generates data, passes through model, and writes results.
        """
        data_generator = self.datagen(full_frames.copy(), mel_chunks)
        total_batches = int(np.ceil(len(mel_chunks) / self.wav2lip_batch_size))

        steps = tqdm(data_generator, total=total_batches, desc="Lip-sync Inference")
        for (img_batch_np, mel_batch_np, frames, coords) in steps:
            img_batch_t = torch.FloatTensor(np.transpose(img_batch_np, (0, 3, 1, 2))).to(self.device)
            mel_batch_t = torch.FloatTensor(np.transpose(mel_batch_np, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch_t, img_batch_t)

            self._write_predicted_frames(pred, frames, coords, out)

    @staticmethod
    def _write_predicted_frames(
        pred: torch.Tensor,
        frames: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        out: cv2.VideoWriter
    ):
        """
        Writes the predicted frames (lipsynced faces) into the output video.
        """
        pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        for p, f, c in zip(pred_np, frames, coords):
            y1, y2, x1, x2 = c
            p_resized = cv2.resize(p, (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p_resized
            out.write(f)

    def _merge_audio_video(self, audio_file: str, temp_video: str, outfile: str):
        """
        Merges the generated video with the input audio.
        """
        command = (
            f'ffmpeg -y -i "{audio_file}" -i "{temp_video}" -strict -2 '
            f'-q:v 1 "{outfile}" -loglevel {self.ffmpeg_loglevel}'
        )
        subprocess.run(command, shell=True, check=True)

    def _split_mel_chunks(self, mel: np.ndarray, fps: float) -> List[np.ndarray]:
        """
        Splits the mel spectrogram into fixed-size chunks.
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

        return mel_chunks
