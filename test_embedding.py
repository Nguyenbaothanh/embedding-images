import requests
import json
import base64
from PIL import Image
import io

def test_embedding_api():
    """Test API embedding với ảnh mẫu"""
    
    # URL API
    api_url = "http://127.0.0.1:9999"
    
    # Kiểm tra health
    print("🔍 Kiểm tra health endpoint...")
    try:
        health_response = requests.get(f"{api_url}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health: {health_data}")
        else:
            print(f"❌ Health failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Không thể kết nối API: {e}")
        return
    
    # Test với ảnh mẫu từ thư mục Coding
    image_path = "../Coding/image.png"
    
    print(f"\n🖼️  Gửi ảnh: {image_path}")
    
    try:
        # Gửi file ảnh
        with open(image_path, 'rb') as f:
            files = {'file': ('image.png', f, 'image/png')}
            response = requests.post(f"{api_url}/embed-image", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Embedding thành công!")
            print(f"📁 Filename: {result['filename']}")
            print(f"🔢 Embedding dimension: {result['embedding_dim']}")
            print(f"🖥️  Device: {result['device']}")
            print(f"📊 Method: {result['method']}")
            print(f"🏷️  Model: {result['model']}")
            
            # Hiển thị 5 số đầu tiên của embedding
            embedding = result['embedding']
            print(f"🔢 5 số đầu tiên của embedding: {embedding[:5]}")
            print(f"📏 Tổng số: {len(embedding)}")
            
        else:
            print(f"❌ Lỗi API: {response.status_code}")
            print(f"📝 Response: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {image_path}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_embedding_api()

