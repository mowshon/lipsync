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

    Attributes:
        checkpoint_path (str): Path to the saved Wav2Lip model checkpoint.
        static (bool): If True, uses only the first video frame for inference.
        fps (float): Frames per second, used when input is a static image.
        pads (List[int]): Padding for face bounding boxes as [top, bottom, left, right].
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
    pads: List[int] = [0, 10, 0, 0]
    wav2lip_batch_size: int = 128
    resize_factor: int = 1
    crop: List[int] = [0, -1, 0, -1]
    box: List[int] = [-1, -1, -1, -1]
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
        """
        Initializes LipSync with custom parameters.

        Args:
            **kwargs: Arbitrary keyword arguments to override class default attributes.
        """
        device = kwargs.get('device', self.device)
        self.device = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        print('Using device:', self.device)

        # Update class attributes with provided keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_smoothened_boxes(boxes: np.ndarray, t: int) -> np.ndarray:
        """
        Smoothens bounding boxes over a temporal window.

        Args:
            boxes (np.ndarray): An array of bounding boxes of shape (N,4).
            t (int): Temporal window size for smoothing.

        Returns:
            np.ndarray: Smoothened bounding boxes.
        """
        for i in range(len(boxes)):
            window_end = min(i + t, len(boxes))
            window = boxes[i:window_end]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_cache_filename(self) -> str:
        """
        Generates a filename for caching face detection results.

        Returns:
            str: The full path to the cache file.
        """
        filename = os.path.basename(self._filepath)
        return os.path.join(self.cache_dir, f'{filename}.pk')

    def get_from_cache(self) -> Union[List, bool]:
        """
        Retrieves face detection results from cache if available.

        Returns:
            Union[List, bool]: Cached results if available, False otherwise.
        """
        if not self.save_cache:
            return False
        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            print(f'Loading from cache: {cache_filename}')
            with open(cache_filename, 'rb') as cached_file:
                return pickle.load(cached_file)
        return False

    def face_detect(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Performs face detection on a list of images.

        Args:
            images (List[np.ndarray]): A list of frames (images) in BGR format.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: A list of tuples containing
            the cropped face image and its bounding box coordinates (y1, y2, x1, x2).

        Raises:
            ValueError: If a face is not detected in one or more frames.
            RuntimeError: If the image is too large for GPU processing.
        """
        cache = self.get_from_cache()
        if cache:
            return cache

        detector = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.TWO_D,
            face_detector='sfd',
            device=self.device
        )

        predictions = []
        for i in tqdm(range(0, len(images)), desc="Face Detection"):
            landmarks = detector.get_landmarks_from_image(images[i], return_bboxes=True)
            predictions.append(
                get_face_box(landmarks)
            )

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        img_h, img_w = images[0].shape[:2]

        for rect, image in zip(predictions, images):
            if rect is None:
                del detector
                raise ValueError('Face not detected! Ensure all frames contain a face.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(img_h, rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(img_w, rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, t=5)

        cropped_results = []
        for (x1, y1, x2, y2), image in zip(boxes, images):
            face_img = image[int(y1): int(y2), int(x1): int(x2)]
            cropped_results.append([face_img, (int(y1), int(y2), int(x1), int(x2))])

        del detector

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

        Args:
            frames (List[np.ndarray]): A list of video frames (BGR format).
            mels (List[np.ndarray]): A list of mel spectrogram chunks.

        Yields:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
                img_batch_np: Numpy array of shape (B, H, W, C*2) containing masked and unmasked images.
                mel_batch_np: Numpy array of shape (B, mel_T, mel_F, 1) containing mel spectrograms.
                frame_batch: A list of corresponding original frames.
                coords_batch: A list of coordinates (y1, y2, x1, x2) for each face.
        """
        if self.box[0] == -1:
            face_det_results = self.face_detect(frames if not self.static else [frames[0]])
        else:
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

        Args:
            img_batch (List[np.ndarray]): List of face images (BGR).
            mel_batch (List[np.ndarray]): List of mel spectrogram chunks.
            frame_batch (List[np.ndarray]): Original frames corresponding to the images.
            coords_batch (List[Tuple[int, int, int, int]]): Bounding box coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
                A tuple (img_batch_np, mel_batch_np, frame_batch, coords_batch) ready for model inference.
        """
        img_batch_np = np.asarray(img_batch, dtype=np.uint8)
        mel_batch_np = np.asarray(mel_batch, dtype=np.float32)

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

        Args:
            ext (str): The file extension (without the dot).

        Returns:
            str: The full path to the created temporary file.
        """
        temp_fd, filename = tempfile.mkstemp()
        os.close(temp_fd)
        return f'{filename}.{ext}'

    def sync(self, face: str, audio_file: str, outfile: str) -> str:
        """
        Performs lip-syncing on the input video/image using the provided audio.

        Args:
            face (str): Path to the input video or image file.
            audio_file (str): Path to the input audio file.
            outfile (str): Path to the output video file.

        Returns:
            str: The path to the output video file.

        Raises:
            ValueError: If the input face file is invalid or if the mel spectrogram contains NaN values.
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

        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')
            wav_filename = self.create_temp_file('wav')
            command = (
                f'ffmpeg -y -i "{audio_file}" -strict -2 "{wav_filename}" '
                f'-loglevel {self.ffmpeg_loglevel}'
            )
            subprocess.run(command, shell=True, check=True)
            audio_file = wav_filename

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel).any():
            raise ValueError('Mel contains NaN! Add a small epsilon to the audio and try again.')

        print(f"Mel spectrogram shape: {mel.shape}")

        mel_chunks = self._split_mel_chunks(mel, fps)

        model = load_model(self.model, self.device, self.checkpoint_path)
        print("Model loaded")

        frame_h, frame_w = full_frames[0].shape[:2]
        temp_result_avi = self.create_temp_file('avi')
        out = cv2.VideoWriter(
            temp_result_avi,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (frame_w, frame_h),
        )

        data_generator = self.datagen(full_frames.copy(), mel_chunks)
        total_batches = int(np.ceil(len(mel_chunks) / self.wav2lip_batch_size))

        for (img_batch_np, mel_batch_np, frames, coords) in tqdm(data_generator, total=total_batches, desc="Lip-sync Inference"):
            img_batch_t = torch.FloatTensor(np.transpose(img_batch_np, (0, 3, 1, 2))).to(self.device)
            mel_batch_t = torch.FloatTensor(np.transpose(mel_batch_np, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch_t, img_batch_t)

            pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            for p, f, c in zip(pred_np, frames, coords):
                y1, y2, x1, x2 = c
                p_resized = cv2.resize(p, (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p_resized
                out.write(f)

        out.release()

        command = (
            f'ffmpeg -y -i "{audio_file}" -i "{temp_result_avi}" -strict -2 '
            f'-q:v 1 "{outfile}" -loglevel {self.ffmpeg_loglevel}'
        )
        subprocess.run(command, shell=True, check=True)

        return outfile

    def _split_mel_chunks(self, mel: np.ndarray, fps: float) -> List[np.ndarray]:
        """
        Splits the mel spectrogram into fixed-size chunks.

        Args:
            mel (np.ndarray): The mel spectrogram array of shape (mel_channels, time_frames).
            fps (float): Frames per second of the video.

        Returns:
            List[np.ndarray]: A list of mel chunks, each of shape (mel_channels, mel_step_size).
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
