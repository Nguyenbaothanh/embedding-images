import streamlit as st
import requests
from PIL import Image
import io
from supabase import create_client, Client
import json

# ------------------ Cấu hình Supabase ------------------
SUPABASE_URL = "https://cgcqkhkxhgyhmxfcxlwd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNnY3FraGt4aGd5aG14ZmN4bHdkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM4ODg0MywiZXhwIjoyMDY2OTY0ODQzfQ.NGF2a54134UDnopdJTwAUZ521rzbNBT2UVQU1sWikOE"
BUCKET_NAME = "images"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------ Cấu hình API Embedding ------------------
API_URL = "https://2c8360f55533.ngrok-free.app/embed-image"  # Đổi thành public URL nếu chạy trên Colab/ngrok

st.title("CLIP Image Embedding & Supabase Storage")

uploaded_file = st.file_uploader("Chọn ảnh để embedding", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh đã chọn", use_column_width=True)
    if st.button("Tạo embedding & lưu Supabase"):
        # Gửi ảnh lên API
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        with st.spinner("Đang tạo embedding..."):
            response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            result = response.json()
            st.success("Embedding thành công!")
            st.write("Embedding dimension:", result["embedding_dim"])
            st.write("5 số đầu embedding:", result["embedding"][:5])

            # Lưu embedding vào file tạm rồi upload lên Supabase bucket
            import tempfile, os
            embedding_data = json.dumps(result["embedding"])
            file_name = uploaded_file.name + ".embedding.json"
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
                tmp.write(embedding_data)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                res = supabase.storage.from_(BUCKET_NAME).upload(file_name, f)
            os.remove(tmp_path)
            if hasattr(res, 'error') and res.error is not None:
                st.error(f"Lỗi lưu Supabase: {res.error}")
            else:
                st.success(f"Đã lưu embedding vào Supabase: {file_name}")
                st.write(res)
                # Lưu thêm vào database image_embeddings
                file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_name}"
                db_data = {
                    "filename": uploaded_file.name,
                    "embedding": result["embedding"],
                    "file_url": file_url
                }
                db_res = supabase.table("image_embeddings").insert(db_data).execute()
                if hasattr(db_res, 'error') and db_res.error is not None:
                    st.error(f"Lỗi lưu database: {db_res.error}")
                else:
                    st.success(f"Đã lưu vào database image_embeddings!")
        else:
            st.error(f"Lỗi API: {response.text}")
