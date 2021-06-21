import numpy as np

def landmark2npXY(land_mark_points):
  return np.array([np.array([point.x, point.y], dtype=np.float32) for point in land_mark_points], dtype=np.float32)
