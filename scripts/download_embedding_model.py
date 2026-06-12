from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


MODEL_ID = "keepitreal/vietnamese-sbert"
LOCAL_DIR = "models/vietnamese-sbert"


def main():
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {MODEL_ID} to {path}")
    model = SentenceTransformer(LOCAL_DIR, local_files_only=True)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()
