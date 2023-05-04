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
color = st.color_picker("Choose background color", value="#EEEEEE")
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
        st.write(facial_feature)
        ImageDraw.Draw(draw_img).line(face_landmarks[facial_feature], width=3)
    st.image(draw_img)

    "## Now we crop your image"
    "Based on the facial landmarks,\
     we know how to position the face for a **biometric** photo"
    x, y, w, h = cv2.boundingRect(np.asarray(face_landmarks["chin"]))
    x_center, y_center = np.asarray(face_landmarks["nose_bridge"]).mean(axis=0)
    aspect_ratio = 35 / 45
    upper = int(y - h * 1.2)
    lower = int(y + h * 1.3)
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
    result_image.info["dpi"] = 300
    result_column.image(result_image)

    "# Verify"
    result_image = result_image.convert("RGBA")
    "## Head position"
    chin_template = Image.open("Kinnschablone.png").convert("RGBA")
    resizing_factor = 413 / equalized.width
    top_to_chin = (h * 1.2 + h) * resizing_factor
    offset = int(chin_template.height - top_to_chin)

    template_size = (413, 531 + 200)
    transparent = (255, 0, 0, 0)

    bordered_image = Image.new('RGBA', template_size, transparent)
    bordered_image.paste(result_image, (0, 200))

    bordered_template = Image.new('RGBA', template_size, transparent)
    bordered_template.paste(chin_template, (0, 200 - offset))

    st.image(Image.alpha_composite(bordered_image, bordered_template))

    "## Eye position"
    eye_template = Image.open("Augenschablone.png").convert("RGBA")
    st.image(Image.alpha_composite(result_image, eye_template))

    with st.form("Send to me"):
        email = st.text_input("Enter email to send this picture")
        if st.form_submit_button():
            st.write(email)
