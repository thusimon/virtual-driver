import numpy as np

def landmark2npXY(land_mark_points):
  return np.array([np.array([point.x, point.y], dtype=np.float32) for point in land_mark_points], dtype=np.float32)

def normalizeDot(vec1, vec2):
  return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

def angleInVecs(vec1, vec2):
  normalize_dot = normalizeDot(vec1, vec2)
  return np.arccos(normalize_dot)
