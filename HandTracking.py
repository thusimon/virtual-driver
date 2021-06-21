import cv2
import mediapipe as mp
import time
from handDetector.DriverDetector import DriverDetector

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
driver_detector = DriverDetector()

p_time = 0
c_time = 0

while True:
  success, img = cap.read()

  driver_detector.getDriverStatus(img)

  c_time = time.time()
  fps = 1/(c_time - p_time)
  p_time = c_time
  cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1 , cv2.LINE_AA)
  
  cv2.imshow('Hand Capture', img)
  
  k = cv2.waitKey(1)
  if k & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
