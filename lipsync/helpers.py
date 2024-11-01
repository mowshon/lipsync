import av


def read_frames(face: str) -> [list[av.VideoFrame], int]:
    print('Reading video frames...')
    # Open the video file
    container = av.open(face)

    # Get frames per second (fps) of the video
    stream = container.streams.video[0]
    fps = int(stream.average_rate)

    # Read frames
    full_frames = []
    for frame in container.decode(video=0):
        # Convert frame to numpy array in BGR format
        img = frame.to_ndarray(format='bgr24')
        full_frames.append(img)

    return full_frames, fps
