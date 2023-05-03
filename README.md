## Here's the code
# Importing Libraries
The script starts by importing the necessary libraries:

```python
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from rembg import remove
import cv2
from utils import equalize_this
```

- `streamlit` is a library that allows the creation of web apps with Python.
- `numpy` is a library that provides support for large, multi-dimensional arrays and matrices.
- `PIL` (Python Imaging Library) is a library that adds support for opening, manipulating, and saving many different image file formats.
- `face_recognition` is a library that provides face detection and recognition functionality.
- `rembg` is a library that provides background removal functionality for images.
- `cv2` is a library that provides computer vision functionality.
- `utils` is a custom module that contains additional utility functions.

# Promting the photo input
The code uses streamlits capabilities to capture photos.

```python
img_file_buffer = st.camera_input("")
img = Image.open(img_file_buffer or "face.jpg")
```
This prompts the user to upload an image or capture an image from their camera. The uploaded image, or an example image, is then opened using the Image.open() method from the PIL library.


# Displaying the Image
The script then displays the uploaded image to the user:

```python
image_column, result_column = st.columns([2, 1])
image_column.image(img, use_column_width=True)
```
This displays the image in a column using the st.columns() method.

# Removing the Background
The script then removes the background from the image using the remove() function from the rembg library:

```python
removed_background = remove(img)
st.image(removed_background)
```
The resulting image with the background removed is then displayed to the user.

# Replacing the Background
The user can then choose a background color using a color picker, and the background color is applied to the image:

```python
color = st.color_picker("Choose background color", value="#5E5E5E")
color_img = Image.new("RGBA", img.size, color)
replaced_background = Image.alpha_composite(color_img, removed_background)
st.image(replaced_background)
```
The `color_picker()` method from the `st` module creates a color picker widget that allows the user to select a background color. The selected color is then used to create a new Image object using the Image.new() method from the PIL library. The resulting image with the replaced background is then displayed to the user.

# Detecting Facial Features
The script then uses the face_recognition library to detect all the facial features in the image:

```python
face_landmarks_list = face_recognition.face_landmarks(img_array)
```
The `face_landmarks()` method from the `face_recognition` library detects all the facial features in the image and returns a list of facial feature coordinates.

# Tracing Facial Features
The facial features are then traced out with a line, and the resulting image is displayed to the user:

```python
for face_landmarks in face_landmarks_list:
    draw_img = replaced_background.copy()
    for facial_feature in face_landmarks.keys():
        ImageDraw.Draw(draw_img).line(face_landmarks[facial_feature], width=3)
    st.image(draw_img)
```

This loops through all the facial features and draws them onto the image.

# Crop the image
The code then crops the image based on the facial landmarks to obtain a biometric photo. The code uses the cv2.boundingRect() function to obtain the bounding rectangle for the chin facial feature. This bounding rectangle is then used to crop the image. The resulting cropped image is displayed using st.image().

```python
x, y, w, h = cv2.boundingRect(np.asarray(face_landmarks["chin"]))
crop_img = replaced_background.crop((
    int(x - x * 0.1),  # left
    int((y - h) * 0.8),  # upper
    int((x + w) + x * 0.1),  # right
    int((y + h) * 1.1)  # lower
))
st.image(crop_img)
```

# Color enhancement
Finally, the code performs some color enhancement on the cropped image using the equalize_this() function from a separate utils.py file. The resulting enhanced image is displayed using st.image(). The code also includes a form that allows the user to input an email address and send the resulting image to that address.

```python
equalized = Image.fromarray(equalize_this(np.array(crop_img)))
st.image(equalized)
result_column.image(equalized)
```

# Send via Email

At last, we allow the user to send the result via email


```python
with st.form("Send to me"):
    email = st.text_input("Enter email to send this picture")
    if st.form_submit_button():
        st.write(email)
```

