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
        st.write(facial_feature)
        ImageDraw.Draw(draw_img).line(face_landmarks[facial_feature], width=3)
    st.image(draw_img)

    "## Now we crop your image"
    "Based on the facial landmarks,\
     we know how to position the face for a **biometric** photo"
    x, y, w, h = cv2.boundingRect(np.asarray(face_landmarks["chin"]))
    x_center, y_center = np.asarray(face_landmarks["nose_bridge"]).mean(axis=0)
    aspect_ratio = 35 / 45
    upper = int((y - h) * 0.8)
    lower = int((y + h) * 1.1)
    width = (lower - upper) * aspect_ratio
    left = x_center - width // 2
    right = x_center + width // 2
    crop_img = replaced_background.crop((left, upper, right, lower))
    crop_img = crop_img.resize((413, 531)) # 35 x 45 mm at 300 DPI
    crop_img.info["dpi"] = 300

    st.image(crop_img)

    "## Some (non-ai) color enhancement"
    equalized = Image.fromarray(equalize_this(np.array(crop_img)))
    st.image(equalized)


    result_column.image(equalized)

    "# Verify"
    equalized = equalized.convert("RGBA")

    chin_template = Image.open("Kinnschablone.png").convert("RGBA")
    top_to_chin = (y + h) - upper
    offset = chin_template.height - top_to_chin
    bordered_image = Image.new('RGBA', (img.width, img.height + 200), (255, 0, 0, 0))
    bordered_image.paste(equalized, (0, 200))
    bordered_template = Image.new('RGBA', (img.width, img.height + 200), (255, 0, 0, 0))
    bordered_template.paste(chin_template, (0, offset))
    st.image(Image.alpha_composite(bordered_image, bordered_template))

    eye_template = Image.open("Augenschablone.png").convert("RGBA")
    st.image(Image.alpha_composite(equalized, eye_template))


    with st.form("Send to me"):
        email = st.text_input("Enter email to send this picture")
        if st.form_submit_button():
            st.write(email)
