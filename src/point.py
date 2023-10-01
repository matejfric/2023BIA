from dataclasses import dataclass
from typing import Tuple


@dataclass
class Point:
    x: float
    y: float
    z: float

    def __contains__(self, item):
        return item in (self.x, self.y, self.z)

    def __iter__(self):
        self._iter_attrs = ['x', 'y', 'z']
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self._iter_attrs):
            attr_name = self._iter_attrs[self._iter_index]
            self._iter_index += 1
            return getattr(self, attr_name)
        else:
            raise StopIteration
