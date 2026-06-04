from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np
from sklearn.ensemble import IsolationForest

# =====================================================
# Consumer (원본 데이터 받기)
# =====================================================
consumer = KafkaConsumer(
    "press-force",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    group_id="isolation-visual-group",
    value_deserializer=lambda data: json.loads(data.decode("utf-8"))
)

# =====================================================
# Producer (이상 데이터 전송)
# =====================================================
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda data: json.dumps(data).encode("utf-8")
)

topic = "alert"

# =====================================================
# AI 모델 설정
# =====================================================
TRAIN_SIZE = 100
train_data = []
model = None

print("IsolationForest 시작")

sample_no = 0

# =====================================================
# Kafka 실시간 루프
# =====================================================
for message in consumer:

    data = message.value
    sample_no += 1

    force = data["force"]
    # AI 입력 형태로 변환
    # 예) [[130]]
    x = np.array([[force]])

    # -----------------------------------------
    # 1. 학습 단계
    # -----------------------------------------
    if model is None:
        # AI는 숫자 리스트가 아니라 “표 형태 데이터”를 받기 때문에 2차원으로 만들어야 한다
        train_data.append([force])
        # train_data가 몇 개 모였는지 확인하는 코드
        print(f"학습 중... {len(train_data)}/{TRAIN_SIZE}")

        # 100개가 다 모였다면 모델로 만듬
        # IsolationForest 생성 (이상 데이터는 5%)
        if len(train_data) >= TRAIN_SIZE:
            model = IsolationForest(
                contamination=0.05,
                # 머신러닝 결과를 항상 똑같이 나오게 고정하는 “랜덤 시드 값”이다
                # 42라는 숫자는 별 의미 없고 그냥 관례이다
                random_state=42
            )
            # fit : AI가 정상 패턴을 학습
            model.fit(np.array(train_data))
            print("\n✔ AI 모델 학습 완료\n")

        continue

    # -----------------------------------------
    # 2. 예측 단계
    # -----------------------------------------
    # predict : AI가 판단시작
    # 결과가 1이면 정상, -1이면 이상
    pred = model.predict(x)

    # -----------------------------------------
    # 3. 이상 데이터 처리
    # -----------------------------------------
    if pred[0] == -1:

        alert_data = {
            "sample_no": sample_no,
            "force": force,
            "status": "ANOMALY"
        }

        print("이상 데이터:", alert_data)

        # Kafka로 전송
        producer.send(topic, alert_data)