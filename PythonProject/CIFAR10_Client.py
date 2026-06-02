import requests

IMAGE_PATH = "./cat.jpg"

files = {"file": open(IMAGE_PATH, "rb")}
res = requests.post("http://192.168.0.24:8000/predict", files=files)

# JSON 변환
result = res.json()

print("\n")

print("====================")
print("AI 분류 결과")
print("====================")

print(
    "Class ID :",
    result["class_id"]
)

print(
    "Class Name :",
    result["class_name"]
)

print(
    "Confidence :",
    round(
        result["confidence"],
        4
    )
)

print("====================")