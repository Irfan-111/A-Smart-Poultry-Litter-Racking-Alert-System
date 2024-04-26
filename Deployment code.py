# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:15:31 2024

@author: Irfan
"""
import cv2
import numpy as np
import os
import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image

# Initialize the YOLOv8 model
model = YOLO('C:/Users/Irfan/OneDrive/Documents/deployment code/best (1).pt')

# Define a function to display the Streamlit app
def main():
    # Set the title and description of the app
    st.set_page_config(page_title="Object Detection using YOLOv8", page_icon="🤖")
    st.title('Object Detection using YOLOv8')
    st.markdown('This is an application for object detection using YOLOv8.')

    # Allow the user to upload an image or video file
    file_upload = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "mp4", "MPEG4"])

    if file_upload is not None:
        # Check the file type and call the appropriate detection function
        if file_upload.type.endswith(('jpg', 'jpeg', 'png')):
            # Load the image
            img = Image.open(file_upload)
            img = np.array(img)

            # Detect objects in the image
            results = detect_image(img)

            # Display the object detection results
            st.write("### Object Detection Results")
            for box in results[0].boxes.xyxy.detach().numpy():
                x1, y1, x2, y2 = map(int, box)
                label_name = model.names[int(results[0].boxes.cls[0].item())]
                confidence = results[0].boxes.conf[0].item()
                st.write(f"{label_name} ({confidence:.2f}): {x1} {y1} {x2} {y2}")

                # Draw a bounding box around the object
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display the image with bounding boxes
            st.image(img, caption="Object Detection Results", use_column_width=True)

        elif file_upload.type.endswith(('mp4', 'MPEG4')):
            # Detect objects in the video
            detect_video(file_upload)

# Define a function to detect objects in an image
def detect_image(img):
    # Run object detection on the image
    results = model(img)

    # Return the results
    return results

# Define a function to detect objects in a video
def detect_video(video_file):
    # Save the uploaded video file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    # Open the video file
    video = cv2.VideoCapture("temp_video.mp4")

    # Get the video metadata
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video
    output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process each frame of the video
    for i in range(frame_count):
        # Read a frame from the video
        ret, frame = video.read()

        # Break the loop if the video is over
        if not ret:
            break

        # Run object detection on the frame
        results = model(frame)

        # Draw bounding boxesaround the detected objects
        for box in results[0].boxes.xyxy.detach().numpy():
            x1, y1, x2, y2 = map(int, box)
            label_name = model.names[int(results[0].boxes.cls[0].item())]
            confidence = results[0].boxes.conf[0].item()
            cv2.rectangle(frame, (x1,y1), (x2, y2), (255, 0, 0), 2)

        # Write the frame to the output video
        output_video.write(frame)

    # Release the video and output objects and close the windows
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    # Remove the temporary video file
    os.remove("temp_video.mp4")

# Run the main function
if __name__ == '__main__':
    main()
