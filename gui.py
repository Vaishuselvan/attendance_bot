from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
from PIL import Image
import queue
import threading
from datetime import datetime
from pathlib import Path
import pyttsx3
import time
import io
import base64
import numpy as np
import face_recognition
from core import FaceRecognitionCore
from attendance import AttendanceManager
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

# Global variables
recognition_threshold = 0.5
is_recording = False
is_camera_on = False
update_queue = queue.Queue()
attendance_recorded = False
capture_progress = {'total': 0, 'current': 0, 'status': ''}
dataset_loaded = False
global_frame = None
frame_lock = threading.Lock()

# Initialize core components
core = FaceRecognitionCore(update_queue)
attendance = AttendanceManager()
cap = None

class VoiceAnnouncer:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self.speaker_thread.start()

    def _initialize_engine(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine

    def _speaker_loop(self):
        while True:
            try:
                if self.engine is None:
                    self.engine = self._initialize_engine()
                text = self.speech_queue.get()
                if text is None:
                    break
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
                self.speech_queue.task_done()
                time.sleep(0.5)
            except Exception as e:
                print(f"Speech error: {str(e)}")
                self.engine = None
                time.sleep(1)

    def speak(self, text):
        if text:
            self.speech_queue.put(text)

    def announce_absentees(self, names):
        if not names:
            self.speak("All students are present today.")
            return
        self.speak("The following students are absent today:")
        time.sleep(0.3)
        for name in names:
            self.speak(name)

    def wait_until_done(self):
        self.speech_queue.join()

    def shutdown(self):
        self.speech_queue.put(None)
        self.speaker_thread.join(timeout=2)
        if self.engine:
            self.engine.stop()

voice_announcer = VoiceAnnouncer()

def initialize_system():
    global dataset_loaded
    try:
        if not dataset_loaded:
            core.load_dataset()
            dataset_loaded = True
            print("Dataset loaded successfully on startup")
        return initialize_camera()
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        return False

def initialize_camera():
    global cap, is_camera_on
    try:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Camera error: Could not open camera")
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            is_camera_on = True
        return True
    except Exception as e:
        print(f"Camera error: {str(e)}")
        return False

def generate_frames():
    global cap, is_recording, recognition_threshold, is_camera_on, global_frame

    frame_buffer = []
    buffer_size = 3
    last_frame_time = time.time()
    target_fps = 30
    frame_interval = 1.0 / target_fps

    while True:
        if cap is None or not cap.isOpened():
            if not initialize_camera():
                break

        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        success, frame = cap.read()
        if not success:
            break

        with frame_lock:
            global_frame = frame.copy()

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        display_frame = frame_buffer[-1].copy()

        if is_recording:
            tracked_faces = core.recognize_faces(display_frame, recognition_threshold)

            if tracked_faces:
                for face in tracked_faces:
                    top, right, bottom, left = face.get_smoothed_location()
                    color = (0, 255, 0) if face.name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    label = f"{face.name} ({face.accuracy:.2%})" if face.name != "Unknown" else face.name

                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    thickness = 1
                    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    y = top - 15 if top - 15 > 15 else top + 15

                    cv2.rectangle(display_frame, (left, y - label_height - 10),
                                (left + label_width + 10, y + 10),
                                (0, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, label, (left + 5, y), font, font_scale, (255, 255, 255), thickness)

                    if face.name != "Unknown":
                        attendance.mark_present(face.name)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def eye():
    return render_template('eye.html')

@app.route('/face_recognition')
def face_recognition():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    global is_recording, cap

    is_recording = not is_recording

    if is_recording:
        if cap is None or not cap.isOpened():
            initialize_camera()
        if cap is None or not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not access camera'})
        return jsonify({'success': True, 'message': 'Recognition Started'})
    else:
        _, absent = attendance.get_attendance_lists()
        voice_announcer.announce_absentees(absent)
        return jsonify({
            'success': True,
            'message': 'Recognition Stopped',
            'absent': absent if absent else []
        })

@app.route('/manual_capture', methods=['POST'])
def manual_capture():
    try:
        data = request.get_json()
        person_name = data.get('name')
        image_data = data.get('image')
        image_count = data.get('count', 0)
        total_images = data.get('total', 60)

        if not person_name or not image_data:
            return jsonify({'success': False, 'message': 'Missing required data'})

        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Create person directory if it doesn't exist
        person_dir = Path(core.dataset_path) / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image_count}.jpg"
        filepath = person_dir / filename
        cv2.imwrite(str(filepath), frame)

        # If all images are captured, reload the dataset
        if image_count + 1 >= total_images:
            core.load_dataset()

        return jsonify({
            'success': True,
            'message': f'Image {image_count + 1} of {total_images} saved successfully',
            'count': image_count + 1
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving image: {str(e)}'})

@app.route('/record_attendance', methods=['POST'])
def record_attendance():
    global attendance_recorded

    if attendance_recorded:
        return jsonify({'success': False, 'message': 'Attendance already recorded for today'})

    # Update these with your email credentials
    sender_email = "kiruthikabalaji91@gmail.com"  # Replace with your Gmail
    app_password = "uumc afkf wzqt zhyr"     # Replace with your Gmail App Password
    recipient_email = "srcw2226j106@srcw.ac.in" # Replace with recipient's email

    success, message = send_attendance_email(sender_email, app_password, recipient_email)

    if success:
        attendance_recorded = True
        return jsonify({'success': True, 'message': 'Attendance recorded and email sent successfully'})
    else:
        return jsonify({'success': False, 'message': message})

def send_attendance_email(sender_email, app_password, recipient_email):
    try:
        # Create the email content
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Attendance Report'

        present, absent = attendance.get_attendance_lists()
        body = f"Present:\n{', '.join(present)}\n\nAbsent:\n{', '.join(absent)}"
        msg.attach(MIMEText(body, 'plain'))

        # Save the attendance report to a file
        report_dir = Path('attendance_reports')
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(body)

        # Attach the report file to the email
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(open(report_file, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={report_file.name}')
        msg.attach(part)

        # Send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()

        return True, 'Email sent successfully'
    except Exception as e:
        return False, f'Error sending email: {str(e)}'

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global recognition_threshold
    try:
        new_threshold = float(request.form.get('threshold', 0.5))
        if 0 <= new_threshold <= 1:
            recognition_threshold = new_threshold
            return jsonify({'success': True, 'message': f'Threshold updated to {new_threshold}'})
        return jsonify({'success': False, 'message': 'Threshold must be between 0 and 1'})
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid threshold value'})

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global dataset_loaded

    if dataset_loaded:
        return jsonify({'success': True, 'message': 'Dataset already loaded'})

    dataset_dir = request.form.get('directory')
    try:
        core.load_dataset(dataset_dir if dataset_dir else None)
        dataset_loaded = True
        return jsonify({'success': True, 'message': 'Dataset loaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading dataset: {str(e)}'})

@app.route('/capture_frame', methods=['GET'])
def capture_frame():
    global global_frame

    with frame_lock:
        if global_frame is not None:
            ret, buffer = cv2.imencode('.jpg', global_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'image': f'data:image/jpeg;base64,{frame_data}'})

    return jsonify({'success': False, 'message': 'No frame available'})

if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, threaded=True)
