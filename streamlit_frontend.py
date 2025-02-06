import streamlit as st
from PIL import Image, ImageDraw
from typing import Tuple
from streamlit_image_coordinates import streamlit_image_coordinates
import requests
import io
import base64

st.set_page_config(page_title="Segment Anything Model", layout="wide")

"# Segment Anything Model"
st.write("this page uses SAM_vit_h model to perform segmentation based on point prompts")

if "points" not in st.session_state:
    st.session_state["points"] = []

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

"## Upload your image"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.session_state["uploaded_image"] = uploaded_file

if st.session_state["uploaded_image"] is not None:
    # Convert uploaded file to a PIL image
    original_img = Image.open(st.session_state["uploaded_image"]).convert("RGBA")

    # Resize image for display only
    display_size = (original_img.width // 2, original_img.height // 2)
    img_display = original_img.resize(display_size)
    draw = ImageDraw.Draw(img_display)

    # Convert original image points to resized image scale
    for point in st.session_state["points"]:
        scaled_x = int(point[0] * (display_size[0] / original_img.width))
        scaled_y = int(point[1] * (display_size[1] / original_img.height))
        coords = (scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5)
        draw.ellipse(coords, fill="red")

    "## Click on any point on the image"
    value = streamlit_image_coordinates(img_display, key="pil")

    if value is not None:
        # Convert coordinate from resized image back to original size
        point_x = int(value["x"] * (original_img.width / display_size[0]))
        point_y = int(value["y"] * (original_img.height / display_size[1]))
        st.write(f"curious about your coordinates? ;) here there are: (x:{point_x}, y:{point_y})")
        st.write("SAM would segment 3 masks based on your point. you can see the masks and their corresponding score below")
        if st.button("Segment"):
            # Convert image to bytes
            img_bytes = io.BytesIO()
            original_img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            # Send data to FastAPI backend
            files = {"file": ("image.png", img_bytes, "image/png")}
            data = {"points_x": point_x, "points_y": point_y}

            res = requests.post("http://api:8000/predict/", files=files, data=data)

            if res.status_code == 200:
                response_json = res.json()
                masks_base64 = response_json["masks"]  # List of 3 base64 masks
                scores = response_json["scores"]  # List of 3 scores

                # Display all 3 masks side by side
                cols = st.columns(3)
                for i in range(3):
                    mask_data = base64.b64decode(masks_base64[i])
                    mask_img = Image.open(io.BytesIO(mask_data)).convert("RGBA")

                    # Ensure mask is the same size as the original image
                    mask_img = mask_img.resize(original_img.size)

                    # Overlay mask on original image
                    img_with_overlay = Image.alpha_composite(original_img, mask_img)

                    # Show each masked image with its score as caption
                    with cols[i]:
                        st.image(img_with_overlay, caption=f"Mask {i+1} (Score: {scores[i]:.4f})", use_column_width=True)

            else:
                st.error("Error processing the segmentation.")

        point = (point_x, point_y)
        if point not in st.session_state["points"]:
            st.session_state["points"].append(point)
            st.rerun()
else:
    st.warning("Upload an image to continue.")
