import copy
import numpy as np
import sympy
from sympy import Symbol, Matrix, sqrt, pprint, abc, exp


class Component(object):
    def __init__(self):
        self.matrix = Matrix([[1,0], [0,1]])

    def __mul__(self, other):
        new_component = copy.deepcopy(self)
        new_component.matrix = new_component.matrix * other.matrix
        return new_component


# Coupler
coupler = Component()
coupler.matrix = sympy.sqrt(0.5)*Matrix([[1.0, -1j], [-1j, 1.0]])

# Delay
delay = Component()
zm1 = Symbol("zm1")
delay.matrix = Matrix([[zm1, 0],[0, 1]])

# Phase
phase = Component()
phi = Symbol("phi_k")
phase.matrix = Matrix([[exp(-1j*phi), 0],[0, 1]])

def f1():
    block = phase * delay * coupler
    pprint(block.matrix)

def main():
    f1()
    pass

if __name__ == "__main__":
    main() # pragma: no cover
