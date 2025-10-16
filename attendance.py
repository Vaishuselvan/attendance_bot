import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path

class AttendanceManager:
    def __init__(self, dataset_path="dataset", reports_dir="attendance_reports"):
        self.dataset_path = Path(dataset_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)  # Create reports directory if it doesn't exist
        self.known_students = self._get_known_students()
        self.present_students = set()

    def _get_known_students(self):
        """Get list of all students from dataset folders"""
        return {folder.name for folder in self.dataset_path.iterdir() if folder.is_dir()}

    def mark_present(self, student_name):
        """Mark a student as present"""
        if student_name in self.known_students:
            self.present_students.add(student_name)

    def get_attendance_lists(self):
        """Get present and absent students lists"""
        present = sorted(list(self.present_students))
        absent = sorted(list(self.known_students - self.present_students))
        return present, absent

    def save_attendance_reports(self):
        """Save attendance to separate Excel files for present and absent students"""
        present, absent = self.get_attendance_lists()
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create DataFrames for present and absent students
        present_df = pd.DataFrame({
            'Date': [date] * len(present),
            'Name': present,
            'Status': ['Present'] * len(present)
        })

        absent_df = pd.DataFrame({
            'Date': [date] * len(absent),
            'Name': absent,
            'Status': ['Absent'] * len(absent)
        })

        # Generate filenames
        present_file = self.reports_dir / f"present_students_{timestamp}.xlsx"
        absent_file = self.reports_dir / f"absent_students_{timestamp}.xlsx"

        # Save to Excel files
        present_df.to_excel(present_file, index=False)
        absent_df.to_excel(absent_file, index=False)

        return present_file, absent_file

    def send_attendance_email(self, sender_email, app_password, recipient_email):
        """Send email with attendance reports as attachments"""
        date = datetime.now().strftime("%Y-%m-%d")
        present, absent = self.get_attendance_lists()

        # Save reports and get file paths
        present_file, absent_file = self.save_attendance_reports()

        # Create email content
        subject = f"Attendance Report - {date}"
        body = f"""
        <html>
        <body>
            <h2>Attendance Report for {date}</h2>
            <p>Please find attached the attendance reports:</p>
            <ul>
                <li>Present Students Report ({len(present)} students)</li>
                <li>Absent Students Report ({len(absent)} students)</li>
            </ul>
            <p>Total Students: {len(self.known_students)}</p>
        </body>
        </html>
        """

        # Setup email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        # Attach Excel files
        for file_path in [present_file, absent_file]:
            with open(file_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='xlsx')
                attachment.add_header('Content-Disposition', 'attachment', filename=file_path.name)
                msg.attach(attachment)

        # Send email using Gmail SMTP
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, app_password)
                server.send_message(msg)
            return True, "Email sent successfully"
        except Exception as e:
            error_message = str(e)
            print(f"Failed to send email: {error_message}")
            return False, f"Failed to send email: {error_message}"
