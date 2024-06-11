from ultralytics import YOLO
import streamlit as st
import cv2

model_path = "runs/detect/train18/weights/best.pt"

model = YOLO(model_path)

st.set_page_config(
    page_title="Deteksi Jalan Rusak Dengan Kamera",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.page_link("home.py", label="back", icon="🔙")

def _display_detected_frames(conf, model, st_frame, image):

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
def infer_uploaded_webcam(conf, model):
    try:
        flag = st.button(
            label="Start"
        )
        flagged = st.button(
            label="Stop"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            if flagged:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")


def main():
    infer_uploaded_webcam(0.3, model)


if __name__ == "__main__":
    main()