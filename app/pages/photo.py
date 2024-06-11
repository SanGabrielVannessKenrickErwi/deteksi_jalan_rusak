from ultralytics import YOLO
import streamlit as st

from PIL import Image

model_path = "app/best.pt"

model = YOLO(model_path)
st.set_page_config(
    page_title="Deteksi Jalan Rusak Dengan Foto",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.page_link("home.py", label="back", icon="🔙")
def infer_uploaded_image(conf, model):

    source_img = st.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the pages with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Submit"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        st.write("")
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def main():
    infer_uploaded_image(0.3, model)



if __name__ == "__main__":
    main()