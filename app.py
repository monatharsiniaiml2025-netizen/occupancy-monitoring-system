
import streamlit as st
import cv2
import pandas as pd
import os
import tempfile
import time
import subprocess
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Occupancy Monitoring System",
    page_icon="👥",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# ---------------- VIDEO PROCESSING ----------------

def process_video(input_path, output_path, csv_path):

    model = load_model()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save temporary AVI first for reliable OpenCV writing
    avi_path = output_path.replace(".mp4", ".avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        avi_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create output video.")

    counts = []
    frame_number = 0
    previous_count = 0

    progress = st.progress(0)
    status = st.empty()

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Detect people in every 2nd frame
        if frame_number % 2 == 0:

            scale = min(640 / width, 640 / height, 1.0)

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            small_frame = cv2.resize(
                frame,
                (new_width, new_height)
            )

            results = model.predict(
                small_frame,
                classes=[0],
                conf=0.35,
                imgsz=416,
                verbose=False
            )

            previous_count = len(results[0].boxes)

            annotated_small = results[0].plot()

            annotated_frame = cv2.resize(
                annotated_small,
                (width, height)
            )

        else:
            annotated_frame = frame.copy()

        counts.append(previous_count)

        # Add people count to video
        cv2.putText(
            annotated_frame,
            f"People Count: {previous_count}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        writer.write(annotated_frame)

        frame_number += 1

        if total_frames > 0:
            progress.progress(
                min(frame_number / total_frames, 1.0)
            )

        status.write(
            f"Processing frame {frame_number} / {total_frames}"
        )

    cap.release()
    writer.release()

    if not os.path.exists(avi_path) or os.path.getsize(avi_path) == 0:
        raise ValueError("Processed video file is empty.")

    # Convert AVI to browser-compatible MP4 using FFmpeg
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", avi_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                output_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is not installed. Install FFmpeg and add it to PATH."
        )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Video conversion failed: " +
            error.stderr.decode(errors="replace")[-1000:]
        )

    finally:
        if os.path.exists(avi_path):
            os.remove(avi_path)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("MP4 video conversion failed.")

    # Save count data
    df = pd.DataFrame({
        "Frame": range(1, len(counts) + 1),
        "People Count": counts
    })

    df.to_csv(csv_path, index=False)

    processing_time = time.time() - start_time

    progress.progress(1.0)
    status.success("Video processing completed!")

    return df, processing_time


# ---------------- MAIN WEBSITE ----------------

st.markdown(
    '<div class="main-title">AI-Based Occupancy Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">People Detection and Real-Time Occupancy Analysis using YOLOv8</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- VIDEO UPLOAD ----------------

st.subheader("Upload Video")

uploaded_file = st.file_uploader(
    "Choose an MP4 video",
    type=["mp4"]
)

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    st.subheader("Original Video")
    st.video(uploaded_file)

    if st.button("Start People Detection", type="primary"):

        os.makedirs("reports", exist_ok=True)

        input_path = None

        try:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                input_path = temp_file.name

            output_path = os.path.abspath(
                "reports/detected_people.mp4"
            )

            csv_path = os.path.abspath(
                "reports/people_counts.csv"
            )

            st.subheader("Detection Progress")

            df, processing_time = process_video(
                input_path,
                output_path,
                csv_path
            )

            st.success("People detection completed successfully!")

            # ---------------- STATISTICS ----------------

            st.divider()
            st.subheader("Occupancy Statistics")

            peak_occupancy = int(
                df["People Count"].max()
            )

            average_occupancy = float(
                df["People Count"].mean()
            )

            total_frames = len(df)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Frames Processed",
                total_frames
            )

            col2.metric(
                "Peak Occupancy",
                peak_occupancy
            )

            col3.metric(
                "Average Occupancy",
                f"{average_occupancy:.2f}"
            )

            st.caption(
                f"Processing time: {processing_time:.2f} seconds"
            )

            # ---------------- PROCESSED VIDEO ----------------

            st.divider()
            st.subheader("Detected Video")

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(
                video_bytes,
                format="video/mp4"
            )

            st.download_button(
                label="Download Detected Video",
                data=video_bytes,
                file_name="detected_people.mp4",
                mime="video/mp4"
            )

            # ---------------- GRAPH ----------------

            st.divider()
            st.subheader("People Count Analysis")

            st.line_chart(
                df.set_index("Frame")["People Count"]
            )

            # ---------------- TABLE ----------------

            st.divider()
            st.subheader("People Count Details")

            st.dataframe(
                df,
                use_container_width=True
            )

            # ---------------- CSV DOWNLOAD ----------------

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Occupancy Report (CSV)",
                data=csv_data,
                file_name="people_counts.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(f"Error: {error}")

        finally:

            if input_path and os.path.exists(input_path):
                os.remove(input_path)

else:

    st.info("Upload an MP4 video to start people detection.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Occupancy Monitoring System | Python | YOLOv8 | OpenCV | Streamlit"
)


