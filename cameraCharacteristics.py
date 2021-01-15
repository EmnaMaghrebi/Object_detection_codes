from enum import Enum

class CameraPose(Enum):
    SIDE = 1
    TOP = 2
    NONE = 3

class CameraDistance(Enum):
    CLOSE = 3
    MEDIUM = 5
    FAR = 7
