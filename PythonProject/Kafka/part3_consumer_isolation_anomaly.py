# ============================================
'''
producer > topic(press-force) > isolationforest consumer
> normal or anomaly 판단 > 이상탐지 결과 실시간 그래프 출력

실행
1. producer_10kHz.py
2. consumer_isolation_anomaly.py
'''
# ============================================

from kafka import KafkaConsumer
import json
import time
import numpy as np
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import signal
import sys

# kafka consumer 객체 생성
consumer = KafkaConsumer(
    "press-force",                              # 구독 topic
    bootstrap_servers="localhost:9092",                # kafka broker 주소
    auto_offset_reset="latest",                        # 최신 데이터부터 읽어오기
    group_id="isolation-visual-group",                 # consumer group
    value_deserializer=lambda data: json.loads(data.decode("utf-8"))     # kafka bytes -> dict 변환
)

# 학습 설정

# 100 개씩 받고 학습을 수행하는 학습데이터 단위
TRAIN_SIZE = 100

# 학습데이터 저장
train_data = []

# AI 모델
model = None

# 결과 저장
all_force = []

# normal/anomaly 저장
all_status = []

# 샘플 번호 저장
all_index = []

# threshold 기준 정의
# threshold는 실무에서 보통 3개를 정한다 (3단계 Warning 1, Warning 2, Critical)
THRESHOLD = 170

# 실시간 그래프 갱신 함수
def update_graph():
    ax.clear()

    ax.plot(
        all_index,
        all_force,
        linewidth = 2,
        color = "blue",
        label = "Force"
    )

    # threshold
    # axhline : 수평선 그래프 (threshold=170 기준으로 수평선)
    ax.axhline(
        y = THRESHOLD,
        color = "orange",
        linestyle = "--",
        linewidth = 2,
        label = f"Threshold: {THRESHOLD}"
    )

    # 이상 데이터 추출
    anomaly_x = []

    anomaly_y = []

    # zip : 한 묶음씩 동시에 꺼내줌
    for idx, force, status in zip(
        all_index,
        all_force,
        all_status,
    ):
        if status == "ANOMALY":
            anomaly_x.append(idx)
            anomaly_y.append(force)

    # 이상 데이터 표시
    # scatter : 점 찍기 그래프 (이상 데이터에 점 찍기)
    ax.scatter(
        anomaly_x,
        anomaly_y,
        color = "red",
        s=100,
        marker = "o",
        label = "Anomaly"
    )

    # 그래프 설정
    ax.set_title(
        "kafka + IsolationForest Anomaly Detection"
    )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Force")

    # grid : 그래프에 격자(줄) 표시
    ax.grid(True)
    # legend : 범례(설명 박스) 표시
    ax.legend()
    # tight_layout : 그래프 자동 여백 정리
    plt.tight_layout()

    # 화면 강제 갱신
    # draw : 그래프 내용을 다시 계산해서 화면에 그림
    fig.canvas.draw()
    # flush_events : 그려진 내용을 실제 화면에 즉시 반영
    fig.canvas.flush_events()

# =====================================================
# 실시간 그래프 설정
# =====================================================

# matplotlib을 실시간 모드로 설정
plt.ion()

# 그래프 창 생성
# subplots = 한 화면 안에 여러 개의 그래프(축 영역)를 나눠서 동시에 볼 수 있게 해주는 기능
fig, ax = plt.subplots(
    figsize=(14, 7)
)

# 그래프 갱신 주기 (10초)
UPDATE_INTERVAL = 10

# 마지막 갱신 시각
last_update_time = time.time()



# =====================================================
# Ctrl+C 처리
# =====================================================

def signal_handler(sig, frame):

    print("\n프로그램 종료")

    # 최종 그래프 한번 더 그림
    update_graph()

    # 실시간 모드 종료
    plt.ioff()

    # 최종 그래프 유지
    plt.show()

    sys.exit(0)


signal.signal(
    signal.SIGINT,
    signal_handler
)

# =====================================================
# 시작
# =====================================================

print(
    "IsolationForest Consumer 시작"
)

sample_no = 0

# =====================================================
# Kafka 데이터 수신
# =====================================================

for message in consumer:

    # Kafka 데이터
    data = message.value

    # 샘플 번호
    sample_no += 1

    # Force 값
    force = data["force"]

    # 저장
    all_index.append(sample_no)

    all_force.append(force)

    # AI 입력
    x = np.array([[force]])

    # ------------------------------------
    # 모델 학습 전
    # ------------------------------------

    if model is None:

        train_data.append([force])

        print(
            f"학습 데이터 수집중 "
            f"{len(train_data)}/{TRAIN_SIZE}"
        )

        # 100개 모이면 학습
        if len(train_data) >= TRAIN_SIZE:
            model = IsolationForest(

                contamination=0.05,

                random_state=42
            )

            model.fit(
                np.array(train_data)
            )

            print(
                "\nAI 모델 학습 완료\n"
            )

        continue

    # ------------------------------------
    # 예측
    # ------------------------------------

    pred = model.predict(x)

    # ------------------------------------
    # 결과 판정 (anomaly 판정 --> isolation -1 이상 / 1 정상)
    # ------------------------------------

    if pred[0] == -1:

        status = "ANOMALY"

    else:

        status = "NORMAL"

    # 결과 저장
    all_status.append(status)

    # ------------------------------------
    # 콘솔 출력
    # ------------------------------------

    print(
        f"Force={force:.2f}, "
        f"Result={status}"
    )

    # =====================================================
    # 10초마다 그래프 자동 갱신
    # =====================================================

    current_time = time.time()

    if current_time - last_update_time >= UPDATE_INTERVAL:
        print(
            "\n===== 그래프 갱신 =====\n"
        )

        update_graph()

        last_update_time = current_time