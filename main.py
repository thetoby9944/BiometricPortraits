from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from rembg import remove
import cv2

from utils import equalize_this

"# AI FOR BIOMETRIC PHOTOS"
img_file_buffer = st.camera_input("Take a photo")

img = Image.open(img_file_buffer or "face.jpg")
img_array = np.array(img)

"## Image and Result"
image_column, result_column = st.columns([2, 1])
image_column.image(img, use_column_width=True)

with st.sidebar:
    "# Here's how we did it"
    "## Let Ai remove the background"
    "We use `u2net` to remove the background for you"
    removed_background = remove(img)
    st.image(removed_background)

    "## Replace the background"
    color = st.color_picker("Choose background color", value="#EEEEEE")
    color_img = Image.new("RGBA", img.size, color)
    replaced_background = Image.alpha_composite(color_img, removed_background)
    st.image(replaced_background)

    "## Let Ai find all facial features in the image"
    face_landmarks_list = face_recognition.face_landmarks(img_array)

    f"Found {len(face_landmarks_list)} face(s) in this photograph.\
     Let's trace out each facial feature in the image with a line!"
for face_landmarks in face_landmarks_list:
    with st.sidebar:
        draw_img = replaced_background.copy()
        for facial_feature in face_landmarks.keys():
            ImageDraw.Draw(draw_img).line(face_landmarks[facial_feature], width=3)
        st.image(draw_img)

        "## Now we crop your image"
        "Based on the facial landmarks,\
         we know how to position the face for a **biometric** photo"
        x, y, w, h = cv2.boundingRect(np.asarray(face_landmarks["chin"]))
        x_center, y_center = np.asarray(face_landmarks["nose_bridge"]).mean(axis=0)
        aspect_ratio = 35 / 45
        upper = int(y - h * 1.2)
        lower = int(y + h * 1.4)
        width = (lower - upper) * aspect_ratio
        left = x_center - width // 2
        right = x_center + width // 2
        crop_img = replaced_background.crop((left, upper, right, lower))

        st.image(crop_img)

        "## Some (non-ai) color enhancement"
        img_yuv = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        equalized = Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB))
        st.image(equalized)

        result_image = equalized.resize((413, 531))  # 35 x 45 mm at 300 DPI
        result_column.image(result_image)

    "# Verify"
    head_column, eye_column = st.columns(2)

    result_image = result_image.convert("RGBA")
    with head_column:
        chin_template = Image.open("assets/templates/Kinnschablone.png").convert("RGBA")
        resizing_factor = 413 / equalized.width
        top_to_chin = (h * 1.2 + h) * resizing_factor
        offset = int(chin_template.height - top_to_chin)

        slide_constant = 100
        template_size = (413, 531 + slide_constant)
        transparent = (255, 0, 0, 0)

        bordered_image = Image.new('RGBA', template_size, transparent)
        bordered_image.paste(result_image, (0, slide_constant))

        bordered_template = Image.new('RGBA', template_size, transparent)
        bordered_template.paste(chin_template, (0, slide_constant - offset))

        st.image(Image.alpha_composite(bordered_image, bordered_template))
    with eye_column:
        "## "
        "## "
        eye_template = Image.open("assets/templates/Augenschablone.png").convert("RGBA")
        st.image(Image.alpha_composite(result_image, eye_template))

    st.image(Image.open("assets/templates/Schablone.png"))


    for image_path in Path("assets/references").glob("*.png"):
        reference_column, comparison_column = st.columns([4, 1])
        reference_column.image(Image.open(image_path))
        comparison_column.image(result_image)

    offsets = (118, 118), (650, 118), (650, 709), (118, 709)
    result_print = Image.open("assets/print.jpeg")
    for offset in offsets:
        result_print.paste(result_image, offset)
    result_print.info["dpi"] = 300
    st.image(result_print)

    with st.form("Send to me"):
        email = st.text_input("Enter email to send this picture")
        if st.form_submit_button():
            st.write(email)
