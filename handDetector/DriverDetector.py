import cv2
import numpy as np
import mediapipe as mp
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from utils.geometry import landmark2npXY, normalizeDot

class DriverDetector:
  def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
      self.hands = mp_hands.Hands(static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence)

  def getDriverStatus(self, img, draw = True):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.hand_results = self.hands.process(img_rgb)
    status = []
    if self.hand_results.multi_hand_landmarks:
      for hand_lmk in self.hand_results.multi_hand_landmarks:
        status.append(self.getHandStatusFromLandmarks(hand_lmk.landmark))
        if draw:
          mp_draw.draw_landmarks(img, hand_lmk, mp_hands.HAND_CONNECTIONS)
    # sort status left, right
    if len(status) > 1:
      status.sort(key=lambda hand_status: hand_status['hand_front_center'][0])
    return status

  def getHandStatusFromLandmarks(self, landmark):
    wrist = landmark[0]
    thumb_mcp = landmark[2]
    thumb_tip = landmark[4]
    index_mcp = landmark[5]
    index_pip = landmark[6]
    middle_mcp = landmark[9]
    middle_pip = landmark[10]
    ring_mcp = landmark[13]
    ring_pip = landmark[14]
    pinky_mcp = landmark[17]
    pinky_pip = landmark[18]
    # calculate the area from index_mcp -> index_pip -> middle_pip -> ring_pip ->
    #  pinky_pip -> pinky_mcp -> ring_mcp -> middle_mcp
    hand_front_points = [index_mcp, index_pip, middle_pip, ring_pip, pinky_pip, pinky_mcp, ring_mcp, middle_mcp]
    hand_front_points_np = landmark2npXY(hand_front_points)
    # calculate the hand height
    index_front_mean = np.mean([hand_front_points_np[0], hand_front_points_np[1]], axis=0)
    pinky_front_mean = np.mean([hand_front_points_np[4], hand_front_points_np[5]], axis=0)
    hand_front_length = np.linalg.norm(index_front_mean-pinky_front_mean)
    # calculate the center of the hand position
    hand_front_center = np.mean(hand_front_points_np, axis=0)
    # calcuate the hand front area
    hand_front_area = cv2.contourArea(hand_front_points_np)
    # calculate the thumb length
    thumb_points = landmark2npXY([wrist, thumb_mcp, thumb_tip])
    thumb_length = np.linalg.norm(thumb_points[1]-thumb_points[2])
    wrist_thumb_vec = thumb_points[0]-thumb_points[1]
    thumb_tip_vec = thumb_points[1]-thumb_points[2]
    thumb_wrist_angle = normalizeDot(wrist_thumb_vec, thumb_tip_vec)
    return {
      'hand_front_center': hand_front_center,
      'hand_front_length': hand_front_length,
      'hand_front_area': hand_front_area,
      'thumb_length': thumb_length,
      'thumb_wrist_angle': thumb_wrist_angle
    }
