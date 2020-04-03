import cv2

class VideoFrameReader:
    def __init__(self, video):
        self.src = video
        self.frame = None

    def __iter__(self):
        cap = cv2.VideoCapture(self.src)
        success, self.frame = cap.read()
        while success:
            yield self.frame
            success, self.frame = cap.read()

        

