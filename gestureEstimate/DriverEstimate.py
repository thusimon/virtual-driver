import numpy as np

class DriverEstimate:
  def __init__(self, size = 3) -> None:
    self.size = size
    self.left_status = []
    self.right_status = []

  def add_status(self, hands_status):
    if len(hands_status) == 0:
      self.left_status.append(None)
      self.right_status.append(None)
    elif len(hands_status) == 1:
      if hands_status[0]['hand_front_center'][0] < 0.5:
        self.left_status.append(hands_status[0])
        self.right_status.append(None)
      else:
        self.left_status.append(None)
        self.right_status.append(hands_status[0])
    else:
      # we already sorted the hands_status
      self.left_status.append(hands_status[0])
      self.right_status.append(hands_status[1])
    
    # trim the length
    if len(self.left_status) > self.size:
      self.left_status.pop(0)
    if len(self.right_status) > self.size:
      self.right_status.pop(0)
  
  def getHandNumber(self):
    left_hand_number = len(list(filter(lambda s: s != None, self.left_status)))
    right_hand_number = len(list(filter(lambda s: s != None, self.right_status)))
    if left_hand_number == self.size and right_hand_number == self.size:
      # stable two hands
      return 0b11
    elif left_hand_number == self.size and right_hand_number == 0:
      # stable only left hand
      return 0b10
    elif left_hand_number == 0 and right_hand_number == self.size:
      # stable only right hand
      return 0b01
    elif left_hand_number == 0 and right_hand_number == 0:
      # stable no hand
      return 0b00
    else:
      # non stable
      return -1

  def isThumbUp(self, status):
    wrist_thumb_angle = np.mean([s['thumb_wrist_angle'] for s in status])
    return wrist_thumb_angle > 0.9

  def isBothThumbUp(self):
    if self.getHandNumber() != 0b11:
      return False
    return self.isThumbUp(self.left_status) and self.isThumbUp(self.right_status)