import cv2
import time
from const_cameras import CAMERAS


class CameraSystem:
    def __init__(self, cameras_number, is_test):
        self.cameras_number = cameras_number
        self.cameras = []
        self.is_test = is_test

        self.init_cameras()

    def init_cameras(self):
        # Create VideoCapture instance for every camera index 0-default, 1-gopro 2... N
        for i in range (self.cameras_number):
            camera_index = i
            camera = cv2.VideoCapture(camera_index)
            self.cameras.append(camera)

        if not self.is_test:
            # Wait for GOPRO to initialize beacuse it restarts after VideoCapture connect
            for _ in range(8):
                self.cameras[1].read()
                time.sleep(1)

    def capture_videos(self):
        for camera in self.cameras:
            ret, frame = camera.read()
            if ret:
                cv2.imwrite('screenshot.jpg', frame)
                print("Screenshot saved as 'screenshot.jpg'")
            else:
                print("Error reading frame")
            camera.release()






