import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime

# Load pre-trained models
@st.cache_resource
def load_models():
    vehicle_model = load_model("vehicle_detection.h5")
    yolov8_model = YOLO("custom_model.pt")  # Replace with your YOLOv8 number plate detection model
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return vehicle_model, yolov8_model, ocr

vehicle_model, yolov8_model, ocr = load_models()

# Function to classify the image type
def classify_image(image, model):
    resized_image = cv2.resize(image, (224, 224))  # Adjust based on your model input size
    normalized_image = resized_image / 255.0
    input_array = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(input_array)[0]
    return np.argmax(prediction), prediction

# Function to detect number plates and extract text
def detect_number_plate(image, yolov8_model, ocr):
    results = yolov8_model.predict(image)
    if not results or results[0].boxes.data is None:
        return []  # No detection

    detected_boxes = results[0].boxes.data.cpu().numpy()
    plate_texts = []

    for box in detected_boxes:
        try:
            x1, y1, x2, y2, conf, cls = map(int, box)
            cropped_plate = image[y1:y2, x1:x2]
            ocr_result = ocr.ocr(cropped_plate)
            if ocr_result and len(ocr_result[0]) > 0:
                for line in ocr_result[0]:
                    plate_texts.append(line[1][0])  # Extract detected text
        except Exception as e:
            print(f"Error processing box {box}: {e}")
    return plate_texts

# Initialize Streamlit app
st.title("Vehicle Detection and Attendance System")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying the vehicle type...")
    class_idx, predictions = classify_image(image, vehicle_model)

    labels = ["Mahindra Scorpio", "Truck", "Non-Vehicle", "Scanned Image"]
    detected_label = labels[class_idx]
    st.success(f"Detected Vehicle Type: {detected_label}")

    if detected_label in ["Mahindra Scorpio", "Truck"]:
        st.write("Processing number plate detection...")

        plate_texts = detect_number_plate(image, yolov8_model, ocr)
        if plate_texts:
            st.success("Number plate detected successfully!")
            st.write("Detected Number Plate(s):", plate_texts)

            csv_file = "number_plate_results.csv"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Load existing data or create new DataFrame
            try:
                existing_data = pd.read_csv(csv_file)
            except FileNotFoundError:
                existing_data = pd.DataFrame(columns=["Timestamp", "Number Plate", "Vehicle Type"])

            updated_records = []
            for plate in plate_texts:
                if plate in existing_data["Number Plate"].values:
                    # Update timestamp for existing vehicle
                    existing_data.loc[existing_data["Number Plate"] == plate, "Timestamp"] = timestamp
                    st.info(f"Updated attendance for {plate}.")
                else:
                    # Add new record
                    updated_records.append({
                        "Timestamp": timestamp,
                        "Number Plate": plate,
                        "Vehicle Type": detected_label
                    })
                    st.success(f"New vehicle {plate} marked present.")

            # Save updated records if any
            if updated_records:
                updated_data = pd.concat([existing_data, pd.DataFrame(updated_records)], ignore_index=True)
            else:
                updated_data = existing_data

            updated_data.to_csv(csv_file, index=False)
            st.write("Attendance updated and saved to CSV:", csv_file)
        else:
            st.error("No number plate detected.")
    elif detected_label == "Scanned Image":
        st.warning("The image is a scanned vehicle image.")
    else:
        st.warning("The uploaded image is not a vehicle.")

st.write("Upload another image to continue.")
