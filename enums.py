import enum
class RotationAxis(enum.Enum):
   X = 1
   Y = 2
   Z = 3

class Base(enum.Enum):
   X = 1
   Y = 2
   Z = 3

class MovmentDirection(enum.Enum):
    X = 1
    Y = 2
    Z = 3

class ConnectorShape(enum.Enum):
   Rectangle = 1
   Round = 2

class Directions(enum.Enum):
   Positive_x = 0
   Negative_x = 1

   Positive_y = 2
   Negative_y = 3

   Positive_z = 4
   Negative_z = 5



class Commands(enum.Enum):
   init = 0
   reset = 1
   setp = 2
   get_action = 3
   done = 4
   clear = 5
   close = 6
   is_new_layout = 7


class ActionSpace(enum.Enum):
   DISCRETE = 0
   CONTINUOUS = 1
   MIXED = 2
