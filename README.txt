AI-Powered Smart Classroom Monitoring and Intelligent Attendance System
=======================================================================

MAIN FILE
---------
Run:
    python app.py

Recommended Python version:
    Python 3.10

Install packages:
    python -m pip install -r requirements.txt

Login:
    Username: admin
    Password: 1234

UPDATED STUDENT REGISTRATION
----------------------------
1. Open the dashboard.
2. Click Register Student.
3. Enter USN and student name.
4. Click Start Registration.
5. The browser automatically captures 10 accepted face images.
6. All 10 colour face images are stored in SQLite table student_images.
7. The first accepted image is also stored in students.photo for display.
8. Training images are saved in dataset/<USN>/01.jpg through 10.jpg.
9. trainer/trainer.yml and trainer/labels.json are rebuilt automatically.
10. The in-memory LBPH attendance recognizer is updated immediately.

IMPORTANT CAMERA NOTE
---------------------
The registration page opens outside the dashboard iframe to avoid the live
classroom stream and browser registration page using the webcam together.
If the webcam is busy, return to the dashboard, stop other camera programs,
and open registration again.

ATTENDANCE TEST SETTINGS
------------------------
The current app.py uses test values:
    LAST_MINUTES_FOR_ATTENDANCE = 2
    MIN_RECOGNITION_COUNT = 1

For normal classroom use, change them to:
    LAST_MINUTES_FOR_ATTENDANCE = 10
    MIN_RECOGNITION_COUNT = 3

FILES REMOVED
-------------
The old duplicate templates were removed:
    indexs1.html
    registers.html
    registers1.html
    reports.html

The separate door_camera.py was removed because the updated project uses one
classroom camera for monitoring and attendance.
