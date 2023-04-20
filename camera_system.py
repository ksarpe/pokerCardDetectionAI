import cv2
import time
from const_cameras import CAMERAS
import numpy as np
import torch
from torchvision import transforms


class CameraSystem:
    def __init__(self, cameras_number, is_test, model_path):
        self.cameras_number = cameras_number
        self.cameras = []
        self.is_test = is_test
        self.model = torch.load(model_path)
        self.model.eval()

        self.init_cameras()

    def init_cameras(self):
        # Create VideoCapture instance for every camera index 0-default, 1-gopro 2... N
        for i in range (self.cameras_number):
            camera_index = i
            camera = cv2.VideoCapture(camera_index)
            self.cameras.append(camera)

        if not self.is_test:
            # Wait for GOPRO to initialize beacuse it restarts after VideoCapture connection
            for _ in range(8):
                self.cameras[1].read()
                time.sleep(1)

    def capture_videos(self):
        for camera in self.cameras:
            ret, frame = camera.read()
            if ret:
                preprocessed_frame = self.preprocess_frame(frame)
                cv2.imwrite('screenshot.jpg', preprocessed_frame)
                print("Screenshot saved as 'screenshot.jpg'")
            else:
                print("Error reading frame")
            camera.release()

    def preprocess_for_model(self, image):
        # Resize, normalize, and convert the image to a PyTorch tensor
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image)

    def recognize_rank_and_suit(self, roi):
        # Preprocess the ROI for the model
        input_tensor = self.preprocess_for_model(roi)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch of size 1

        # Move the input to the device (CPU or GPU)
        input_batch = input_batch.to(self.model.device)

        # Make a prediction with the model
        with torch.no_grad():
            output = self.model(input_batch)

        # Get the predicted class index
        _, predicted_index = torch.max(output, 1)

        # Convert the class index to the corresponding rank and suit
        rank, suit = self.index_to_rank_and_suit(predicted_index.item())

        return rank, suit

    def index_to_rank_and_suit(self, index):
        # Convert the class index to the corresponding rank and suit
        # ...
        pass

    def detect_cards(self, hsv_frame):
        # Define color ranges for red and black cards
        # ...
        pass

    def is_card_contour(self, contour):
        # Check if the contour is a card based on size, aspect ratio, etc.
        # ...
        pass

    def extract_rank_and_suit_roi(self, hsv_frame, contour):
        # Extract the rank and suit region of interest (ROI) from the card contour
        # ...
        pass

    def preprocess_frame(self, frame):
        # Resize the frame
        scale_percent = 50  # Percentage of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        return hsv_frame







