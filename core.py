import cv2
import face_recognition
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import os
import threading
import queue

class TrackedFace:
    def __init__(self, location, name, accuracy):
        self.location = location
        self.name = name
        self.accuracy = accuracy
        self.frames_since_detection = 0
        self.id = None
        self.history = [location]  # Keep a history of recent locations
        self.max_history = 10  # Increased history size for smoother movement
        self.name_history = [(name, accuracy)]  # Track name and accuracy history
        self.name_history_max = 5  # Number of frames to keep for name stability

    def update_location(self, new_location):
        self.location = new_location
        self.frames_since_detection = 0
        self.history.append(new_location)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def update_name(self, new_name, new_accuracy):
        self.name_history.append((new_name, new_accuracy))
        if len(self.name_history) > self.name_history_max:
            self.name_history.pop(0)

        # Use most common name in history
        names = [n for n, _ in self.name_history]
        name_counts = {}
        max_count = 0
        best_name = self.name

        for name in names:
            if name not in name_counts:
                name_counts[name] = 0
            name_counts[name] += 1
            if name_counts[name] > max_count:
                max_count = name_counts[name]
                best_name = name

        # Only update if the new name is stable
        if max_count >= 3:  # Require at least 3 consistent detections
            self.name = best_name
            # Update accuracy to average of instances with this name
            matching_accuracies = [acc for n, acc in self.name_history if n == best_name]
            self.accuracy = sum(matching_accuracies) / len(matching_accuracies)

    def get_smoothed_location(self):
        if not self.history:
            return self.location

        # Apply exponential moving average for smoother transitions
        weights = np.exp(np.linspace(0, 1, len(self.history)))
        weights = weights / np.sum(weights)

        # Convert locations to numpy array for weighted average
        locations = np.array(self.history)
        avg_loc = np.average(locations, axis=0, weights=weights)

        # Apply additional smoothing to reduce jitter
        current_loc = np.array(self.location)
        smoothing_factor = 0.7  # Adjust this value to control smoothing (0.0-1.0)
        smoothed_loc = smoothing_factor * avg_loc + (1 - smoothing_factor) * current_loc

        return tuple(smoothed_loc.astype(int))

class FaceRecognitionCore:
    def __init__(self, update_queue, dataset_path="dataset"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognition_threshold = 0.5
        self.max_faces = 20
        self.process_every_n_frames = 3  # Increased from 2 to 3 for better performance
        self.frame_count = 0
        self.current_frame_small = None
        self.dataset_path = dataset_path
        self.data_file = "known_faces.pkl"
        self.model_loaded = False

        # Face tracking variables
        self.tracked_faces = []
        self.max_frames_to_keep = 45  # Increased to reduce face ID switching
        self.next_face_id = 0

        # Thread-safe queue for updates
        self.update_queue = update_queue

        # Ensure directories exist
        self.ensure_directories_exist()

        # Frame buffer for processing
        self.frame_buffer = []
        self.buffer_size = 3
        self.last_processed_frame = None

    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist"""
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_path=None):
        """Load the face recognition dataset"""
        try:
            if dataset_path:
                self.dataset_path = dataset_path

            # Clear existing data
            self.known_face_encodings = []
            self.known_face_names = []

            # Try to load existing dataset from pickle file
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.model_loaded = True
                    self.update_queue.put(("status", f"Loaded {len(self.known_face_names)} faces from dataset"))
            else:
                # Load dataset from directory if pickle file does not exist
                dataset_dir = Path(self.dataset_path)
                if not dataset_dir.exists():
                    self.update_queue.put(("error", "Dataset Error", f"Directory not found: {dataset_dir}"))
                    return

                for person_dir in dataset_dir.iterdir():
                    if person_dir.is_dir():
                        person_name = person_dir.name
                        # Process each image in the person's directory
                        for img_path in person_dir.glob("*.jpg"):
                            try:
                                # Load and encode face
                                image = face_recognition.load_image_file(str(img_path))
                                face_encodings = face_recognition.face_encodings(image)

                                if face_encodings:
                                    self.known_face_encodings.append(face_encodings[0])
                                    self.known_face_names.append(person_name)
                            except Exception as e:
                                self.update_queue.put(("warning", "Image Error",
                                                       f"Error processing {img_path}: {str(e)}"))
                                continue

                self.model_loaded = True
                self.update_queue.put(("status",
                                       f"Loaded {len(self.known_face_names)} faces from dataset"))

                # Save the updated dataset
                self.save_known_faces()

        except Exception as e:
            self.update_queue.put(("error", "Dataset Error", f"Error loading dataset: {str(e)}"))
            self.model_loaded = True  # Still set to True to allow system to run

    def save_known_faces(self):
        """Save the current face encodings and names"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            self.update_queue.put(("status", "Face data saved successfully"))
        except Exception as e:
            self.update_queue.put(("error", "Save Error", f"Error saving face data: {str(e)}"))

    def add_face(self, frame, name):
        """Add a new face to the dataset"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return None, "No face detected"

        if len(face_locations) > 1:
            return None, "Multiple faces detected"

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

        # Save updated dataset
        self.save_known_faces()

        return face_locations[0], "Face added successfully"

    def recognize_faces(self, frame, recognition_threshold):
        self.frame_count += 1
        self.recognition_threshold = recognition_threshold

        # Store the frame in buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Only process every nth frame
        if self.frame_count % self.process_every_n_frames != 0:
            # Return last processed results with current frame
            if self.last_processed_frame is not None:
                return self.tracked_faces
            return []

        # Process frame
        scale_factor = 0.5  # Maintained for performance
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if not face_locations:
            self.tracked_faces = [face for face in self.tracked_faces
                                  if face.frames_since_detection < self.max_frames_to_keep]
            return self.tracked_faces

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Scale locations back to original size
        face_locations = [(int(top/scale_factor), int(right/scale_factor),
                           int(bottom/scale_factor), int(left/scale_factor))
                          for (top, right, bottom, left) in face_locations]

        # Process each detected face
        processed_faces = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            name, accuracy = self._identify_face(face_encoding)
            processed_faces.append((face_location, name, accuracy))

        # Update tracking with improved matching
        self._update_tracking(processed_faces)

        self.last_processed_frame = frame.copy()
        return self.tracked_faces

    def _identify_face(self, face_encoding):
        """Identify a face encoding against known faces"""
        matches = face_recognition.compare_faces(self.known_face_encodings,
                                                 face_encoding,
                                                 tolerance=self.recognition_threshold)
        name = "Unknown"
        accuracy = 0

        if True in matches and self.known_face_encodings:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            accuracy = 1 - face_distances[best_match_index]

            if matches[best_match_index] and accuracy > (1 - self.recognition_threshold):
                name = self.known_face_names[best_match_index]

        return name, accuracy

    def _update_tracking(self, processed_faces):
        """Update tracking with improved face matching"""
        # Calculate centers for all current and new faces
        current_faces = [(face, self._get_face_center(face.location)) for face in self.tracked_faces]
        new_faces = [(face_data, self._get_face_center(face_data[0])) for face_data in processed_faces]

        # Match faces using distance matrix
        matched_pairs = []
        max_distance = 150  # Maximum pixel distance for matching

        while current_faces and new_faces:
            min_dist = float('inf')
            best_pair = None

            for i, (current_face, current_center) in enumerate(current_faces):
                for j, (new_face_data, new_center) in enumerate(new_faces):
                    dist = np.sqrt((current_center[0] - new_center[0])**2 +
                                   (current_center[1] - new_center[1])**2)
                    if dist < min_dist and dist < max_distance:
                        min_dist = dist
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            current_face, _ = current_faces.pop(i)
            (face_location, name, accuracy), _ = new_faces.pop(j)

            # Update the matched face
            current_face.update_location(face_location)
            current_face.update_name(name, accuracy)
            matched_pairs.append(current_face)

        # Add new faces for unmatched detections
        for (face_location, name, accuracy), _ in new_faces:
            new_face = TrackedFace(face_location, name, accuracy)
            new_face.id = self.next_face_id
            self.next_face_id += 1
            matched_pairs.append(new_face)

        # Update tracked faces list
        self.tracked_faces = matched_pairs

    def _get_face_center(self, face_location):
        """Calculate center point of a face"""
        top, right, bottom, left = face_location
        return ((left + right) // 2, (top + bottom) // 2)
