import cv2
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image


class VideoSearchEngine:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Storage
        self.index = None
        self.metadata = []   # timestamps
        self.frames = []     # actual frames

    def index_video(self, video_path, sample_rate=1.0):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            raise ValueError("❌ Could not read video FPS")

        interval = int(fps * sample_rate)

        embeddings = []
        timestamps = []
        frames = []

        count = 0

        print("Indexing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(img_tensor)
                    features /= features.norm(dim=-1, keepdim=True)

                embeddings.append(features.cpu().numpy())
                timestamps.append(count / fps)
                frames.append(rgb_frame)

            count += 1

        cap.release()

        if len(embeddings) == 0:
            raise ValueError("❌ No frames extracted from video")

        embeddings = np.vstack(embeddings).astype("float32")

        # FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = timestamps
        self.frames = frames

        print(f"✅ Indexed {len(frames)} frames")

    def search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("❌ Please index a video first.")

        text = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        distances, indices = self.index.search(
            text_features.cpu().numpy().astype("float32"),
            top_k
        )

        results = []

        for i, idx in enumerate(indices[0]):
            results.append({
                "timestamp": self.metadata[idx],
                "score": float(distances[0][i]),
                "frame": self.frames[idx]
            })

        return results