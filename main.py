import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from rembg import remove
import cv2

from utils import equalize_this

"# AI FOR BIOMETRIC PHOTOS"
img_file_buffer = st.camera_input("")

img = Image.open(img_file_buffer or "face.jpg")
img_array = np.array(img)

"## Image and Result"
image_column, result_column = st.columns([2, 1])
image_column.image(img, use_column_width=True)

"# Here's how we did it"
"## Let Ai remove the background"
"We use `u2net` to remove the background for you"
removed_background = remove(img)
st.image(removed_background)

"## Replace the background"
color = st.color_picker("Choose background color", value="#5E5E5E")
color_img = Image.new("RGBA", img.size, color)
replaced_background = Image.alpha_composite(color_img, removed_background)
st.image(replaced_background)

"## Let Ai find all facial features in the image"
face_landmarks_list = face_recognition.face_landmarks(img_array)

f"Found {len(face_landmarks_list)} face(s) in this photograph.\
 Let's trace out each facial feature in the image with a line!"
for face_landmarks in face_landmarks_list:
    draw_img = replaced_background.copy()
    for facial_feature in face_landmarks.keys():
        ImageDraw.Draw(draw_img).line(face_landmarks[facial_feature], width=3)
    st.image(draw_img)

    "## Now we crop your image"
    "Based on the facial landmarks,\
     we know how to position the face for a **biometric** photo"
    x, y, w, h = cv2.boundingRect(np.asarray(face_landmarks["chin"]))
    crop_img = replaced_background.crop((
        int(x - x * 0.1),  # left
        int((y - h) * 0.8),  # upper
        int((x + w) + x * 0.1),  # right
        int((y + h) * 1.1)  # lower
    ))
    st.image(crop_img)

    "## Some (non-ai) color enhancement"
    equalized = Image.fromarray(equalize_this(np.array(crop_img)))
    st.image(equalized)

    with st.form("Send to me"):
        email = st.text_input("Enter email to send this picture")
        if st.form_submit_button():
            st.write(email)

    result_column.image(equalized)
