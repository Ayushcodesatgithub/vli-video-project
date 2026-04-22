import streamlit as st
from engine import VideoSearchEngine
import datetime

st.set_page_config(page_title="Intelligent Video Search", layout="wide")

st.title("🎬 Intelligent Video Search")


if "engine" not in st.session_state:
    st.session_state.engine = VideoSearchEngine()


video_file = st.sidebar.file_uploader(
    "Upload Video", type=["mp4", "mov", "avi"]
)


if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    st.sidebar.success("Video uploaded!")


if st.sidebar.button("Index Video"):
    if not video_file:
        st.sidebar.error("⚠️ Please upload a video first!")
    else:
        with st.spinner("Indexing video..."):
            st.session_state.engine.index_video("temp_video.mp4")
        st.sidebar.success("✅ Indexing complete!")


query = st.text_input(
    "Search for moments (e.g., 'a red car' or 'person laughing')",
    disabled=(st.session_state.engine.index is None)
)

# Search logic
if query:
    if st.session_state.engine.index is None:
        st.error("⚠️ Please index the video first!")
    else:
        results = st.session_state.engine.search(query)

        st.subheader("🔍 Results")

        cols = st.columns(3)

        for i, res in enumerate(results):
            with cols[i % 3]:
                ts = str(datetime.timedelta(seconds=int(res["timestamp"])))

                st.image(res["frame"], caption=f"Time: {ts}")
                st.metric("Relevance Score", f"{res['score']:.4f}")