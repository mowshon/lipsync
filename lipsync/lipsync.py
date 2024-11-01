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
        fps (float): Frames per second; used if input is a static image.
        pads (List[int]): Padding for the face bounding box (top, bottom, left, right).
        face_det_batch_size (int): Batch size for face detection.
        wav2lip_batch_size (int): Batch size for the Wav2Lip model.
        resize_factor (int): Factor to reduce the resolution of the video frames.
        crop (List[int]): Crop dimensions for the video frames (top, bottom, left, right).
        box (List[int]): Fixed bounding box coordinates for the face (top, bottom, left, right).
        rotate (bool): If True, rotates the video frames by 90 degrees.
        nosmooth (bool): If True, disables smoothing of face detections over time.
        save_cache (bool): If True, enables caching of face detection results.
        cache_dir (str): Directory to store cache files.
        img_size (int): Size to which face images are resized.
        mel_step_size (int): Step size for mel spectrogram chunks.
        device (str): Device to run the model on ('cuda' or 'cpu').
        ffmpeg_loglevel (str): Log level for ffmpeg commands.
    """

    # Default parameters
    checkpoint_path = ''
    static = False
    fps = 25.0
    pads = [0, 10, 0, 0]
    face_det_batch_size = 16
    wav2lip_batch_size = 128
    resize_factor = 1
    crop = [0, -1, 0, -1]
    box = [-1, -1, -1, -1]
    rotate = False
    nosmooth = False
    save_cache = True
    cache_dir = tempfile.gettempdir()
    _filepath = ''
    img_size = 96
    mel_step_size = 16
    device = 'cpu'
    ffmpeg_loglevel = 'verbose'
    model = 'wav2lip'

    def __init__(self, **kwargs):
        """
        Initializes LipSync with custom parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for setting class attributes.
        """
        device = kwargs.get('device', self.device)
        if device == 'cuda':
            # Even when ‘cuda’ is chosen, it is not always available.
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('USE Device:', self.device)

        # Update class attributes with provided keyword arguments
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def get_smoothened_boxes(boxes: list, t: int) -> list:
        """
        Smoothens bounding boxes over a temporal window.

        Args:
            boxes (List[List[float]]): List of bounding boxes.
            t (int): Temporal window size.

        Returns:
            List[List[float]]: Smoothened bounding boxes.
        """
        for i in range(len(boxes)):
            if i + t > len(boxes):
                # If window exceeds list length, use the last T boxes
                window = boxes[len(boxes) - t:]
            else:
                # Define the window of t boxes
                window = boxes[i: i + t]
            # Compute the mean of the boxes in the window
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def get_cache_filename(self) -> str:
        """
        Generates a filename for caching face detection results.

        Returns:
            str: Cache filename.
        """
        # Get the base name of the file
        filename = os.path.basename(self._filepath)

        # Construct the cache filename
        return os.path.join(self.cache_dir, f'{filename}.pk')

    def get_from_cache(self) -> list | bool:
        """
        Retrieves face detection results from cache if available.

        Returns:
            List or bool: Cached results if available, else False.
        """
        if not self.save_cache:
            return False

        cache_filename = self.get_cache_filename()
        if os.path.isfile(cache_filename):
            # Load cached face detection results
            with open(cache_filename, 'rb') as cached_file:
                print(f'Load cache: {cache_filename}')
                return pickle.load(cached_file)

        return False

    def face_detect(self, images: list) -> list:
        """
        Performs face detection on a list of images.

        Args:
            images (List[np.ndarray]): List of images in BGR format.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: Detected faces and their coordinates.

        Raises:
            ValueError: If a face is not detected in an image.
            RuntimeError: If an image is too large for GPU processing.
        """
        # Attempt to load face detection results from cache
        cache = self.get_from_cache()
        if cache:
            return cache

        # Initialize the face detector
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=self.device
        )

        batch_size = self.face_det_batch_size

        while True:
            predictions = []
            try:
                # Process images in batches
                for i in tqdm(range(0, len(images), batch_size)):
                    batch = np.array(images[i:i + batch_size])
                    # Get face detections for the batch
                    predictions.extend(detector.get_detections_for_batch(batch))
            except RuntimeError:
                if batch_size == 1:
                    # If batch size is 1 and still failing, raise error
                    raise RuntimeError(
                        'Image too big for GPU. Please set resize_factor to reduce image size.'
                    )
                # Reduce batch size to recover from OOM error
                batch_size //= 2
                print(f'Recovering from OOM error; New batch size: {batch_size}')
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                # Save the frame where face was not detected
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected! Ensure all frames contain a face.')

            # Adjust bounding box with padding
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            # Smooth the bounding boxes over time
            boxes = self.get_smoothened_boxes(boxes, t=5)
        results = [
            # Crop the face region from the image
            [image[int(y1): int(y2), int(x1): int(x2)], (int(y1), int(y2), int(x1), int(x2))]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        # Clean up the detector
        del detector

        if self.save_cache:
            # Save face detection results to cache
            with open(self.get_cache_filename(), 'wb') as cached_file:
                pickle.dump(results, cached_file)

        return results

    def datagen(self, frames: list, mels: list):
        """Generator yielding batches of images and mel spectrograms.

        Args:
            frames (List[np.ndarray]): List of video frames.
            mels (List[np.ndarray]): List of mel spectrogram chunks.

        Yields:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
                - img_batch: Batch of images.
                - mel_batch: Batch of mel spectrograms.
                - frame_batch: List of frames.
                - coords_batch: List of coordinates.
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            # Perform face detection if bounding box is not specified
            face_det_results = self.face_detect(
                frames if not self.static else [frames[0]]
            )
        else:
            # Use the specified bounding box
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [
                [f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames
            ]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            # Resize face image to the model input size
            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch_np = np.asarray(img_batch)
                mel_batch_np = np.asarray(mel_batch)

                # Mask the lower half of the face
                img_masked = img_batch_np.copy()
                img_masked[:, self.img_size // 2:] = 0

                # Combine masked and original images
                img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
                # Add channel dimension to mel spectrograms
                mel_batch_np = mel_batch_np[..., np.newaxis]

                yield img_batch_np, mel_batch_np, frame_batch, coords_batch
                # Reset batches
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)

            # Mask and prepare the final batch
            img_masked = img_batch_np.copy()
            img_masked[:, self.img_size // 2:] = 0
            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
            mel_batch_np = mel_batch_np[..., np.newaxis]

            yield img_batch_np, mel_batch_np, frame_batch, coords_batch

    @staticmethod
    def create_temp_file(ext: str) -> str:
        """Creates a temporary file with a specific extension.

        Args:
            ext (str): File extension.

        Returns:
            str: Path to the temporary file.
        """
        # Create a temporary file
        temp_fd, filename = tempfile.mkstemp()
        os.close(temp_fd)

        # Return the filename with the desired extension
        return f'{filename}.{ext}'

    def sync(self, face: str, audio_file: str, outfile: str) -> str:
        """Performs lip-syncing on the input video/image using the provided audio.

        Args:
            face (str): Path to the input video or image file.
            audio_file (str): Path to the input audio file.
            outfile (str): Path to the output video file.

        Returns:
            str: Path to the output video file.

        Raises:
            ValueError: If the input face file is invalid.
            ValueError: If the mel spectrogram contains NaN values.
        """
        self._filepath = face

        if not os.path.isfile(face):
            raise ValueError('face argument must be a valid path to video/image file')

        if face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            # If input is an image, use static mode
            self.static = True
            # Read the image
            full_frames = [cv2.imread(face)]
            fps = self.fps
        else:
            print('Reading video frames...')
            # Convert frames from RGB to BGR
            full_frames, fps = read_frames(face)

        print(f"Number of frames available for inference: {len(full_frames)}")

        if not audio_file.endswith('.wav'):
            print('Extracting raw audio...')
            # Create a temporary WAV file
            filename = self.create_temp_file('wav')
            # Extract audio from the input file using ffmpeg
            command = (
                f'ffmpeg -y -i {audio_file} -strict -2 {filename} '
                f'-loglevel {self.ffmpeg_loglevel}'
            )

            subprocess.call(command, shell=True)
            audio_file = filename

        # Load the audio file
        wav = audio.load_wav(audio_file, 16000)
        # Generate mel spectrogram from the audio
        mel = audio.melspectrogram(wav)
        print(f"Mel spectrogram shape: {mel.shape}")

        if np.isnan(mel.reshape(-1)).sum() > 0:
            # Check for NaN values in mel spectrogram
            raise ValueError(
                'Mel contains NaN! Add a small epsilon noise to the wav file and try again.'
            )

        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps  # 80 mel frames per second of audio
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                # If end is reached, use the last mel_step_size frames
                mel_chunks.append(mel[:, -self.mel_step_size:])
                break
            # Append a chunk of mel spectrogram
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print(f"Length of mel chunks: {len(mel_chunks)}")

        batch_size = self.wav2lip_batch_size
        # Generate data batches
        gen = self.datagen(full_frames.copy(), mel_chunks)

        temp_result_avi = self.create_temp_file('avi')
        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
        ):
            if i == 0:
                # Load the Wav2Lip model
                model = load_model(self.model, self.device, self.checkpoint_path)
                print("Model loaded")

                # Get frame dimensions
                frame_h, frame_w = full_frames[0].shape[:-1]
                # Initialize the video writer
                out = cv2.VideoWriter(
                    temp_result_avi,
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    (frame_w, frame_h),
                )

            # Convert image and mel batches to torch tensors
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                # Generate predictions from the model
                pred = model(mel_batch, img_batch)

            # Convert predictions to NumPy arrays
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                # Resize the predicted face region to match original
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                # Replace the face region in the frame with the prediction
                f[y1:y2, x1:x2] = p
                # Write the frame to the output video
                out.write(f)

        # Release the video writer
        out.release()

        # Combine the generated video with the original audio using ffmpeg
        command = (
            f'ffmpeg -y -i {audio_file} -i {temp_result_avi} -strict -2 '
            f'-q:v 1 {outfile} -loglevel {self.ffmpeg_loglevel}'
        )
        subprocess.call(command, shell=True)

        return outfile
