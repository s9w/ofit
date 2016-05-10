import copy
import numpy as np
import sympy
from sympy import Symbol, Matrix, pprint


class Schematic(object):
    def __init__(self):
        self.draw_fun = None
        self.width = 0

class Component(object):
    def __init__(self):
        self.matrix = np.eye(4, dtype=sympy.symbol.Symbol)
        self.schematics = []

    def __mul__(self, other) -> "Component":
        product = copy.deepcopy(self)
        product.matrix = self.matrix .dot( other.matrix)
        return product

    def draw(self):
        curr_pos = np.array([0, 1])
        code = ""
        for schematic in self.schematics:
            code += schematic.draw_fun(top_left=curr_pos)
            curr_pos += np.array([schematic.width, 0])
        pass

    def shift_matrix(self, direction):
        shift_amount = {"up": -1, "down": 1}[direction]
        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=0)
        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=1)

    def shift_up(self):
        self.shift_matrix("up")

    def shift_down(self):
        self.shift_matrix("down")

    def __str__(self):
        # print("in __str", type(self.matrix[0,0]))
        return np.array2string(self.matrix, precision=3)
        # with printoptions(precision=3, suppress=True):
        #     # return str(self.matrix)
        #     return np.array2string(self.matrix, precision=3 )


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
    core_matrix = np.sqrt(0.5) * np.array([[1.0, -1j], [-1j, 1.0]], dtype=sympy.symbol.Symbol)
    comp_coupler.matrix[1:3, 1:3] = core_matrix

    # schematic_coupler = Schematic()
    # schematic_coupler.draw_fun = draw_coupler
    # schematic_coupler.width = 3
    # comp_coupler.schematics.append(schematic_coupler)
    return comp_coupler


def create_delay(location="top"):
    comp_delay = Component()

    zm1 = Symbol("zm1")
    if location == "top":
        core_matrix = np.array([[zm1, 0], [0, 1]], dtype=sympy.symbol.Symbol)
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, zm1]], dtype=sympy.symbol.Symbol)
    else:
        raise ValueError

    comp_delay.matrix[1:3, 1:3] = core_matrix

    return comp_delay


def create_phase(delay_param, location="top"):
    comp_phase = Component()

    phi = Symbol(delay_param)
    if location == "top":
        core_matrix = np.array([[sympy.exp(-1j * phi), 0], [0, 1]], dtype=sympy.symbol.Symbol)
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, sympy.exp(-1j * phi)]], dtype=sympy.symbol.Symbol)
    else:
        raise ValueError

    comp_phase.matrix[1:3, 1:3] = core_matrix
    return comp_phase

def create_crosser():
    crosser = Component()
    # core_matrix = np.array([[0,1],[1,0]])
    core_matrix = np.array([[0,1], [1,0]], dtype=sympy.symbol.Symbol)
    crosser.matrix[1:3, 1:3] = core_matrix
    return crosser


def f1():
    coupler = create_coupler()
    delay = create_delay(location="top")
    phase = create_phase(delay_param="phi_1")


    block = coupler * delay * phase
    # block.matrix[0,0] = 0.945672345
    print(block)


def main():
    f1()
    pass

if __name__ == "__main__":
    main() # pragma: no cover
