from dataclasses import dataclass


@dataclass
class Interval:
    lb: float
    ub: float
    step: float

    @property
    def bounds(self):
        return (self.lb, self.ub)
