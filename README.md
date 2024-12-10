# vehicle-detection-and-attendance-system

This project is designed to detect vehicles from images uploaded by users, ensuring that the images are authentic and not scanned versions. It also integrates a vehicle number plate detection feature for the purpose of marking attendance. 

The primary objectives include:
- Verifying whether an image is original or scanned.
- Detecting vehicle number plates for attendance tracking.

## Features
- **Vehicle Detection**: Identify and verify vehicles in uploaded images.
- **Number Plate Recognition**: Extract vehicle number plates from detected vehicles for attendance purposes.
- **Scanned Image Detection**: Differentiate between original and scanned images to ensure data integrity.
- **Attendance Marking**: Automatically update attendance records based on recognized vehicle number plates.

## Repository Contents
- `Attendance_system.ipynb`: Jupyter Notebook for implementing the attendance marking system.
- `app.py`: Python script to run the application.
- `custom_model.pt`: Custom-trained model file for vehicle detection.
- `number_plate_results.csv`: Sample results of number plate detection.
- `vehicle_detection.h5`: Model file used for vehicle detection.
- `vehicle_detection.ipynb`: Jupyter Notebook for training and testing the vehicle detection system.


## How It Works
1. Users upload an image containing a vehicle.
2. The system verifies if the image is scanned or original.
3. If the image is authentic, the system detects the vehicle and extracts its number plate.
4. The number plate is matched with records to mark attendance.
