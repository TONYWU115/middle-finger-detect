import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)

def is_middle_finger_raised(landmarks):
    # 獲取手指關鍵點
    thumb_tip = landmarks[4].y      # 拇指頂端
    index_tip = landmarks[8].y      # 食指頂端
    middle_tip = landmarks[12].y    # 中指頂端
    ring_tip = landmarks[16].y      # 無名指頂端
    pinky_tip = landmarks[20].y     # 小指頂端
    wrist = landmarks[0].y          # 手腕
    
    # 檢查中指是否伸直且其他手指彎曲
    fingers_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    middle_finger_up = (
        middle_tip < wrist and  # 中指高於手腕
        index_tip > middle_tip and  # 食指低於中指
        ring_tip > middle_tip and   # 無名指低於中指
        pinky_tip > middle_tip      # 小指低於中指
    )
    
    return middle_finger_up

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 水平翻轉影像，取消鏡像效果
    frame = cv2.flip(frame, 1)
        
    # 轉換顏色空間並處理影像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # 繪製手部關鍵點
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 檢測中指手勢
            if is_middle_finger_raised(hand_landmarks.landmark):
                # 在畫面上顯示英文警告
                cv2.putText(frame, "Warning! Middle finger detected!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 終端機也顯示英文警告
                print("Warning: Inappropriate gesture detected!")
    
    # 顯示影像
    cv2.imshow('Hand Gesture Detection', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
hands.close()