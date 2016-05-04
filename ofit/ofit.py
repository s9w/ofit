import copy
import numpy as np
import sympy
from sympy import Symbol, Matrix, sqrt, pprint, abc, exp


class Component(object):
    def __init__(self):
        self.matrix = Matrix([[1,0], [0,1]])

    def __mul__(self, other):
        product = copy.deepcopy(self)
        product.matrix = self.matrix * other.matrix
        return product


def create_coupler():
    comp_coupler = Component()
    comp_coupler.matrix = sympy.sqrt(0.5) * Matrix([[1.0, -1j], [-1j, 1.0]])
    return comp_coupler


def create_delay():
    comp_delay = Component()
    zm1 = Symbol("zm1")
    comp_delay.matrix = Matrix([[zm1, 0], [0, 1]])
    return comp_delay


def create_phase(delay_param):
    comp_phase = Component()
    phi = Symbol(delay_param)
    comp_phase.matrix = Matrix([[exp(-1j * phi), 0], [0, 1]])
    return comp_phase


coupler = create_coupler()
delay = create_delay()
phase = create_phase(delay_param="phi_1")


def f1():
    block = phase * delay * coupler
    pprint(block.matrix)


def main():
    f1()
    pass

if __name__ == "__main__":
    main() # pragma: no cover
