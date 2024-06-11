from ultralytics import YOLO
import streamlit as st
import cv2
import tempfile
import os

model_path = "app/best.pt"

model = YOLO(model_path)

st.set_page_config(
    page_title="Deteksi Jalan Rusak Dengan Video",
    page_icon="🎥",
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
def infer_uploaded_video(conf, model):

    source_video = st.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Submit"):
            with st.spinner("Running..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, source_video.name)
                    with open(path, "wb") as f:
                        f.write(source_video.getvalue())
                    vid_cap = cv2.VideoCapture(path)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def main():
    infer_uploaded_video(0.3, model)



if __name__ == "__main__":
    main()