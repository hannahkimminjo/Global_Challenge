from deepface import DeepFace
import cv2

# 비디오 파일 경로
video_path = 'your_video_path_here.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # DeepFace를 사용하여 감정 분석
    result = DeepFace.analyze(frame, actions=['emotion'])

    # 분석된 감정 출력
    print(f"Dominant Emotion: {result[0]['dominant_emotion']}")

    # 비디오 프레임 표시
    cv2.imshow('Video', frame)

    # 종료 조건: 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 종료
cap.release()
cv2.destroyAllWindows()
