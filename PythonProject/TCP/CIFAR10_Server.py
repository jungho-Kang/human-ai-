from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI()

# CIFAR10 클래스
CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# 모델 로드
print("AI 모델 로딩")
model = tf.keras.models.load_model("./model/cifar10_model.h5")
print("AI 모델 로딩 완료")


# 이미지 전처리 + 예측 함수
def predict_image(image_bytes):

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((32, 32))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image, verbose=0)

    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return {
        "class_id": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "confidence": confidence
    }


# =========================
# API 엔드포인트
# =========================
# pip install python-multipart
# multipart/form-data (파일 업로드 방식)
# file: UploadFile = File(...)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    result = predict_image(image_bytes)

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.24", port=8000)