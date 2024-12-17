import av
import numpy as np
from typing import Tuple, List


def read_frames(face: str) -> Tuple[List[np.ndarray], int]:
    """
    Reads all frames from a video file and returns them along with the video's FPS.

    Args:
        face (str): Path to the video file.

    Returns:
        Tuple[List[np.ndarray], int]:
            - A list of video frames as NumPy arrays in BGR format.
            - The frames per second (FPS) of the video.

    Raises:
        FileNotFoundError: If the specified video file cannot be opened.
        ValueError: If the video stream does not contain any frames.
    """
    try:
        # Open the video file
        container = av.open(face)

        # Access the first video stream and retrieve its FPS
        stream = container.streams.video[0]
        fps = int(stream.average_rate)

        # Read and decode all frames into a list
        full_frames: List[np.ndarray] = []
        for frame in container.decode(video=0):
            # Convert the frame to a NumPy array in BGR format
            img = frame.to_ndarray(format='bgr24')
            full_frames.append(img)

        if not full_frames:
            raise ValueError("The video contains no frames.")

        return full_frames, fps

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Video file '{face}' not found.") from e
    except Exception as e:
        raise ValueError(f"An error occurred while reading the video file: {e}") from e


def get_face_box(landmarks: list) -> Tuple[int, int, int, int]:
    """
    Extracts and returns the bounding box coordinates of a detected face.

    Args:
        landmarks (list): A list containing facial landmarks where the third element
                          (index 2) represents the bounding box coordinates.

    Returns:
        Tuple[int, int, int, int]: The bounding box coordinates (x1, y1, x2, y2) of the face.

    Raises:
        ValueError: If the landmarks list is improperly structured or does not contain
                    the expected bounding box.
    """
    try:
        # Extract the face box from the landmarks
        face_box = landmarks[2][0]  # Access the bounding box coordinates
        face_box = np.clip(face_box, 0, None)  # Ensure no negative values

        # Convert bounding box values to integers
        x1, y1, x2, y2 = map(int, face_box[:-1])  # Exclude the confidence score (last value)
        return x1, y1, x2, y2
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError("Invalid landmarks structure. Could not extract face box.") from e
