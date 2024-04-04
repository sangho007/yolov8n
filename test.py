import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# YOLO 모델 및 라벨 파일 경로
model_file = 'yolo_model.tflite'
label_file = 'coco_labels.txt'

# 모델 로드 및 인터프리터 생성
interpreter = make_interpreter(model_file)
interpreter.allocate_tensors()
labels = read_label_file(label_file)

# 추적 대상 객체 변수
target_object = None
target_color = None

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 객체 감지 수행
    _, scale = common.set_resized_input(interpreter, frame.shape[:2], lambda size: cv2.resize(frame, size))
    interpreter.invoke()
    objs = detect.get_objects(interpreter, 0.5, scale)

    # 감지된 객체 처리
    for obj in objs:
        # 사람 객체인 경우
        if labels[obj.id] == 'person':
            # 객체 경계 상자 그리기
            bbox = obj.bbox.scale(scale).astype(int)
            cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)

            # 추적 대상 객체가 없는 경우 현재 객체를 추적 대상으로 설정
            if target_object is None:
                target_object = bbox
                target_color = frame[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax].mean(axis=(0, 1))

            # 추적 대상 객체와 현재 객체 비교
            if target_object is not None:
                # 객체 색상 비교
                current_color = frame[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax].mean(axis=(0, 1))
                color_diff = np.linalg.norm(current_color - target_color)

                # 색상 차이가 작은 경우 동일한 객체로 판단하여 추적 대상 갱신
                if color_diff < 50:
                    target_object = bbox

    # 추적 대상 객체가 있는 경우 로봇 제어 (예: 객체 중심으로 이동)
    if target_object is not None:
        # 로봇 제어 코드 작성
        pass

    # 결과 프레임 출력
    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()