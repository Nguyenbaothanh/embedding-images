from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io
import os
import torch
import torch.nn.functional as F
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến global để lưu model
model = None
preprocess = None
device = "cpu"

def load_clip_model():
    """Load CLIP model trực tiếp từ PyTorch"""
    global model, preprocess, device
    try:
        logger.info("Đang load CLIP model...")
        
        # Sử dụng torch hub để load CLIP
        model, preprocess = torch.hub.load('openai/CLIP', 'ViT_B_32', jit=False, trust_repo=True)
        
        # Chọn thiết bị
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        if (not force_cpu) and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Chuyển model về thiết bị và eval mode
        model = model.to(device)
        model.eval()
        
        logger.info("Đã load CLIP model thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi load CLIP model: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    # Load model khi khởi động ứng dụng
    load_clip_model()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(title="CLIP Image Embeddings API", version="1.0.0", lifespan=lifespan)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running", "model_loaded": model is not None and preprocess is not None}

@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    """Tạo embedding cho image sử dụng CLIP"""
    try:
        # Kiểm tra model đã được load chưa
        if model is None or preprocess is None:
            raise HTTPException(status_code=503, detail="Model chưa được load, vui lòng thử lại sau")
        
        # Kiểm tra loại file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file image")
        
        # Đọc và xử lý image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="File rỗng")
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể đọc file image: {str(e)}")
        
        # Xử lý image với CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            
        # Chuẩn hóa embedding
        image_features = F.normalize(image_features, p=2, dim=1)
        
        logger.info(f"Đã tạo CLIP embedding thành công cho file: {file.filename}")
        return {
            "success": True,
            "filename": file.filename,
            "embedding": image_features[0].detach().cpu().tolist(),
            "embedding_dim": len(image_features[0]),
            "method": "CLIP",
            "model": "ViT_B_32",
            "device": device,
        }
        
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CLIP Image Embeddings API",
        "version": "1.0.0",
        "description": "API tạo embedding từ image sử dụng CLIP model",
        "endpoints": {
            "health": "/health",
            "embed_image": "/embed-image"
        },
        "model": "ViT_B_32",
        "note": "Sử dụng CLIP để tạo embedding chất lượng cao",
        "status": "Model đã được load" if (model is not None and preprocess is not None) else "Model chưa được load"
    }
