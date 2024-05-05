"""
COMP3065 Computer Vision Coursework

This script is part of the COMP3065 Computer Vision coursework. It implements object detection
and video stabilization functionalities using YOLOv3 model and VidStab library.

Author: Jiaxin Tang (20319441)
Email: scyjt2@nottingham.edu.cn

References:
- VidStab library: https://github.com/AdamSpannbauer/python_video_stab
- YOLO model: https://pjreddie.com/darknet/yolo/

Usage:
You can run this script directly using Python. The command to run the script is:
    python main.py

The output files will be saved in the same directory as the selected video file.

Notes:
- The YOLO model used in this script is YOLOv3-608, and the configuration (cfg) and weights (weights) files are required.
"""


import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading
from tqdm import tqdm
import sys
from vidstab import VidStab

class ObjectDetectionApp:
    def __init__(self, root):
        """
        Initialize the ObjectDetectionApp class.

        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("600x480")
        self.root.configure(bg="lightblue")

        # Create the Select Video button
        self.select_video_button = tk.Button(root, text="Select Video", command=self.select_video, padx=10, pady=5, font=("Arial", 12))
        self.select_video_button.pack(pady=10)

        # Create the Stabilize Video button
        self.stabilize_video_button = tk.Button(root, text="Stabilize Video", command=self.stabilize_video, padx=10, pady=5, font=("Arial", 12))
        self.stabilize_video_button.pack(pady=10)

        # Create the Start Detection button
        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection, padx=10, pady=5, font=("Arial", 12))
        self.start_button.pack(pady=10)

        # Create the Pause/Resume Detection button
        self.pause_detection_var = tk.BooleanVar(value=False)
        self.pause_button = tk.Button(root, text="Pause Detection", command=self.pause_resume_detection, padx=10, pady=5, font=("Arial", 12))
        self.pause_button.pack(pady=10)

        # Create the progress percentage label
        self.progress_label = tk.Label(root, text="", pady=10, font=("Arial", 12), bg="lightblue")
        self.progress_label.pack()

        # Create the text display box
        self.output_text = tk.Text(root, height=5, width=50)
        self.output_text.pack()

        # Create the Quit button
        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app, padx=10, pady=5, font=("Arial", 12))
        self.quit_button.pack(pady=10)

        # Create the warning label
        self.warning_label = tk.Label(root, text="", pady=10, font=("Arial", 12), bg="lightblue", fg="red")
        self.warning_label.pack()

        # Initialize variables
        self.video_path = ""
        self.total_frames = 0
        self.current_frame = tk.IntVar(value=0)
        self.cap = None
        # Flag to track if detection has been started
        self.started_detection = False 

        # Load the YOLO model and class labels
        self.yolo_cfg = "yolov3.cfg"
        self.yolo_weights = "yolov3.weights"
        self.yolo_names = "coco.names"
        self.net, self.output_layers = self.load_yolo_model(self.yolo_cfg, self.yolo_weights)
        self.classes = self.load_class_labels(self.yolo_names)

    def load_yolo_model(self, cfg_file, weights_file):
        """
        Load the YOLO model.

        Args:
            cfg_file: The YOLO configuration file.
            weights_file: The YOLO weights file.

        Returns:
            net: The loaded YOLO model.
            output_layers: The names of the output layers in the YOLO model.
        """
        net = cv2.dnn.readNet(cfg_file, weights_file)
        output_layers = net.getUnconnectedOutLayersNames()
        return net, output_layers

    def load_class_labels(self, names_file):
        """
        Load the class labels.

        Args:
            names_file: The file containing class labels.

        Returns:
            classes: The list of class labels.
        """
        with open(names_file, 'r') as f:
            classes = f.read().strip().split('\n')
        return classes

    def detect_objects(self, image):
        """
        Detect objects in an image using the YOLO model.

        Args:
            image: The input image.

        Returns:
            image: The image with detected objects.
        """
        blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        conf_threshold = 0.5
        nms_threshold = 0.4
        class_ids = []
        boxes = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        indices = np.asarray(indices).reshape(-1)

        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, self.classes[class_ids[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def select_video(self):
        """
        Select a video file using a file dialog.
        """
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.output_text.insert(tk.END, f"---------------\nSelected video: {self.video_path}\n")
            self.warning_label.config(text="")
            self.cap = None
        else:
            self.warning_label.config(text="Please select a video file before starting detection.")

    def start_detection(self):
        """
        Start object detection on the selected video file.
        """
        if not self.video_path:
            self.warning_label.config(text="Please select a video file before starting detection.")
            return

        self.start_button.config(state="disabled")
        # Set flag to True when detection starts
        self.started_detection = True 

        def detect():
            """
            Perform object detection on the selected video.

            This function reads the selected video file frame by frame, performs object detection on each frame,
            and saves the detected frames to a new video file. It also updates the progress label and displays
            completion messages.

            Note:
            This function is executed in a separate thread to prevent the UI from freezing.

            Args:
                None

            Returns:
                None
            """
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.output_text.insert(tk.END, "Error: Failed to open video.\n")
                self.start_button.config(state="normal")
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_dir, video_filename = os.path.split(self.video_path)
            output_video_path = os.path.join(video_dir, 'output_' + video_filename)
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            for frame_index in tqdm(range(self.total_frames), desc="Processing frames", unit="frame"):
                if self.pause_detection_var.get():
                    while self.pause_detection_var.get():
                        if self.cap is None:
                            break
                        continue
                ret, frame = self.cap.read()
                if not ret:
                    break

                detected_frame = self.detect_objects(frame)
                out.write(detected_frame)

                progress_percent = int((frame_index + 1) / self.total_frames * 100)
                self.progress_label.config(text=f"Progress: {progress_percent}%")

            if self.cap:
                self.cap.release()
            out.release()
            cv2.destroyAllWindows()

            self.output_text.insert(tk.END, f"---------------\nDetection Completed.\n---------------\nOutput video saved as {output_video_path}.\n")
            self.start_button.config(state="normal")
            self.progress_label.config(text="")

        threading.Thread(target=detect).start()

    def pause_resume_detection(self):
        """
        Pause or resume object detection.
        """
        if not self.started_detection:
            self.warning_label.config(text="Please start detection first.")
            return

        if self.pause_detection_var.get():
            # Resume detection
            self.pause_detection_var.set(False)
            self.start_button.config(state="disabled")
            self.pause_button.config(text="Pause Detection")
            self.output_text.insert(tk.END, "---------------\nDetection Resumed.\n")
            threading.Thread(target=self.continue_detection).start()
        else:
            # Pause detection
            self.pause_detection_var.set(True)
            self.start_button.config(state="normal")
            self.pause_button.config(text="Resume Detection")
            self.output_text.insert(tk.END, "---------------\nDetection Paused.\n")

    def continue_detection(self, start_frame):
        """
        Continue object detection from the specified frame.

        Args:
            start_frame: The frame index to start detection from.
        """
        if self.cap is None:
            return

        # Set the frame position to start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 

        for frame_index in tqdm(range(start_frame, self.total_frames), desc="Processing frames", unit="frame"):
            if not self.pause_detection_var.get():
                ret, frame = self.cap.read()
                if not ret:
                    break

                detected_frame = self.detect_objects(frame)

                if not self.pause_detection_var.get():
                    progress_percent = int((frame_index + 1) / self.total_frames * 100)
                    self.progress_label.config(text=f"Progress: {progress_percent}%")

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def stabilize_video(self):
        """
        Stabilize the selected video file.
        """
        if not self.video_path:
            self.warning_label.config(text="Please select a video file.")
            return

        self.output_text.insert(tk.END, "---------------\nStabilizing video, please wait a few seconds...\n")
        self.root.update() 

        # Using defaults
        stabilizer = VidStab()
        output_video_path = os.path.join(os.path.dirname(self.video_path), 'stabilized_' + os.path.splitext(os.path.basename(self.video_path))[0] + '.avi')
        stabilizer.stabilize(input_path=self.video_path, output_path=output_video_path)

        self.output_text.insert(tk.END, f"---------------\nVideo Stabilization Completed.\nOutput video saved as {output_video_path}.\n")

        # Reset warning label after stabilization
        self.warning_label.config(text="")

    def quit_app(self):
        """
        Quit the application.
        """
        if self.cap:
            self.cap.release()
        self.root.destroy()
        sys.exit()

# Create the root window
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
