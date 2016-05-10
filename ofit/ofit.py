import copy
import numpy as np
import sympy
from sympy import Symbol, Matrix, sqrt, pprint, exp


class Schematic(object):
    def __init__(self):
        self.draw_fun = None
        self.width = 0


class Component(object):
    def __init__(self):
        self.matrix = np.eye(4, dtype=sympy.symbol.Symbol)
        self.schematics = []

    def __mul__(self, other):
        product = copy.deepcopy(self)
        product.matrix = self.matrix * other.matrix
        return product

    def draw(self):
        curr_pos = np.array([0, 1])
        code = ""
        for schematic in self.schematics:
            code += schematic.draw_fun(top_left=curr_pos)
            curr_pos += np.array([schematic.width, 0])
        pass

    def shift_matrix(self, direction):
        shift_amount = {"up": 1, "down": -1}[direction]

        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=0)
        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=1)


def draw_coupler(top_left):
    top_left+1
    code = """
\draw [thick] (0, 1) to [out=0,in=180] (0.022499999999999992, 1.0) to [in=180,out=0] (0.42749999999999999, 0.55000000000000004) to (0.45000000000000001, 0.55000000000000004);
\draw [thick] (0.45000000000000001, 0.55000000000000004) -- (0.55000000000000004, 0.55000000000000004);
\draw [thick] (0.55000000000000004, 0.55000000000000004) to [out=0,in=180] (0.57250000000000001, 0.55000000000000004) to [in=180,out=0] (0.97750000000000004, 1.0) to (1, 1);
\draw [thick] (0, 0) to [out=0,in=180] (0.022499999999999992, 0.0) to [in=180,out=0] (0.42749999999999999, 0.45000000000000001) to (0.45000000000000001, 0.45000000000000001);
\draw [thick] (0.45000000000000001, 0.45000000000000001) -- (0.55000000000000004, 0.45000000000000001);
\draw [thick] (0.55000000000000004, 0.45000000000000001) to [out=0,in=180] (0.57250000000000001, 0.45000000000000001) to [in=180,out=0] (0.97750000000000004, 0.0) to (1, 0);
"""
    return code


def create_coupler():
    comp_coupler = Component()
    # comp_coupler.matrix = np.eye(4)
    core_matrix = np.sqrt(0.5) * np.array([[1.0, -1j], [-1j, 1.0]])
    comp_coupler.matrix[1:3, 1:3] = core_matrix

    # schematic_coupler = Schematic()
    # schematic_coupler.draw_fun = draw_coupler
    # schematic_coupler.width = 3
    # comp_coupler.schematics.append(schematic_coupler)
    return comp_coupler


def create_delay(location="top"):
    comp_delay = Component()
    # comp_delay.matrix = sympy.eye(4)

    zm1 = Symbol("zm1")
    if location == "top":
        core_matrix = np.array([[zm1, 0], [0, 1]])
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, zm1]])
    else:
        raise ValueError

    comp_delay.matrix[1:3, 1:3] = core_matrix
    # comp_delay.matrix = sympy.matrix2numpy(comp_delay.matrix)

    return comp_delay


def create_phase(delay_param, location="top"):
    comp_phase = Component()
    # comp_phase.matrix = sympy.eye(4)

    phi = Symbol(delay_param)
    if location == "top":
        core_matrix = np.array([[exp(-1j * phi), 0], [0, 1]])
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, exp(-1j * phi)]])
    else:
        raise ValueError

    comp_phase.matrix[1:3, 1:3] = core_matrix
    # comp_phase.matrix = sympy.matrix2numpy(comp_phase.matrix)
    return comp_phase


def f1():
    coupler = create_coupler()
    # coupler.draw()
    delay = create_delay(location="top")
    phase = create_phase(delay_param="phi_1")

    block = phase * delay * coupler
    pprint(block.matrix)


def main():
    f1()
    pass

if __name__ == "__main__":
    main() # pragma: no cover
