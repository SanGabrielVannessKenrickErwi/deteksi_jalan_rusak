from ultralytics import YOLO
import streamlit as st
import cv2
import tempfile
import os
from ultralytics.utils.plotting import Annotator

model_path = "runs/detect/train18/weights/best.pt"
color_list = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]

model = YOLO(model_path)

st.set_page_config(
    page_title="Deteksi Jalan Rusak Dengan Video",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.page_link("home.py", label="back", icon="🔙")

def infer_uploaded_video(conf, model):

    source_video = st.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Submit"):
            with st.spinner("Running..."):
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, source_video.name)
                    with open(path, "wb") as f:
                        f.write(source_video.getvalue())
                    cap = cv2.VideoCapture(path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    output_frames = []

                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model.predict(frame_rgb)
                        annotator = Annotator(frame_rgb)
                        for label in results[0].boxes.data:
                            label_class = int(label[-1].item())
                            annotator.box_label(
                                label[0:4],
                                f"{model.names[int(label[-1].item())]} {round(float(label[-2]), 2)}",
                                color_list[int(label[-1].item())])

                        annotated_frame_bgr = cv2.cvtColor(annotator.im, cv2.COLOR_RGB2BGR)
                        output_frames.append(annotated_frame_bgr)
                        frame_count += 1

                    cap.release()

                    output = "annotated_video.mp4"

                    output_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'h264'), 30, (width, height))
                    for frame in output_frames:
                        output_video.write(frame)

                    output_video.release()
                    st.video(output)


def main():
    infer_uploaded_video(0.3, model)



if __name__ == "__main__":
    main()