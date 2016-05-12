import copy
import numpy as np
import sympy
from sympy import Symbol, Matrix, pprint

def a(x, y):
    return np.array([x, y])

def tt(np_array):
    return (np_array[0], np_array[1])

options = {
    "unit_height": 1,

    "coupler_width": 1,

    "phase_width": 1,

    "delay_width": 1,
    "delay_height": 0.3
}


class Schematic(object):
    positions = [0, 0, 0, 0]

    def __init__(self, w, height_slots, draw_fun) -> "Schematic":
        self.draw_fun = draw_fun
        self.width = w
        self.height_slots = height_slots
        self.vpos = 0

        self.id = np.random.randint(0, 99)


class Component(object):
    def __init__(self, schematic: "Schematic") -> "Component":
        self.matrix = np.eye(4, dtype=sympy.symbol.Symbol)
        self.schematics = [schematic]

    def __mul__(self: "Component", other: "Component") -> "Component":
        product = copy.deepcopy(self)
        product.matrix = other.matrix.dot(self.matrix)

        product.schematics = product.schematics + other.schematics
        return product

    def draw(self):
        positions = [0, 0, 0, 0]
        draw_code = ""

        for sch in self.schematics:
            if sch.height_slots == 2:
                pos_x = max(positions[sch.vpos], positions[sch.vpos-1])
                positions[sch.vpos] += sch.width
                positions[sch.vpos-1] += sch.width
                print("pos_x", pos_x)
            elif sch.height_slots == 1:
                pos_x = positions[sch.vpos]
                positions[sch.vpos] += sch.width
                print("pos_x", pos_x)
            else:
                raise ValueError

            pos = a(pos_x, sch.vpos)
            draw_code += sch.draw_fun(pos) + "\n"

        return draw_code

    def shift_matrix(self, direction):
        shift_amount = {"up": -1, "down": 1}[direction]
        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=0)
        self.matrix = np.roll(self.matrix, shift=shift_amount, axis=1)

    def shift_up(self):
        self.shift_matrix("up")
        for sch in self.schematics:
            sch.vpos += 1

    def shift_down(self):
        self.shift_matrix("down")
        for sch in self.schematics:
            sch.vpos -= 1

    def __str__(self):
        return np.array2string(self.matrix, precision=3)

def create_coupler():
    def draw_line(left, right):
        return "\draw [thick] {} -- {};\n".format(
            tt(left), tt(right)
        )

    def draw_arm(start, end):
        total_width = (end - start)[0]
        height = (end - start)[1]
        arm_frac = 0.9
        middle_width = total_width * arm_frac
        lead_width = (total_width - middle_width) / 2

        p1 = start
        p2 = p1 + np.array([lead_width, 0])
        p3 = p2 + np.array([middle_width, height])
        p4 = end
        return "\draw [thick] {} to [out=0,in=180] {} to [in=180,out=0] {} to {};\n".format(
            tt(p1), tt(p2), tt(p3), tt(p4)
        )

    def draw_coupler_arm(left, right, height):
        parallel_width = 0.1 * options["coupler_width"]
        width = (right - left)[0]
        arm_width = (width - parallel_width) / 2
        p1 = left
        p2 = p1 + a(arm_width, height)
        p3 = p2 + a(parallel_width, 0)
        p4 = right

        draw_code = ""
        draw_code += draw_arm(p1, p2)
        draw_code += draw_line(p2, p3)
        draw_code += draw_arm(p3, p4)
        return draw_code

    def draw_coupler(pos: np.ndarray):
        gap_frac = 0.1
        arm_height = (1.0 - gap_frac) / 2 * options["unit_height"]

        p_top_left = pos
        p_top_right = p_top_left + a(options["coupler_width"], 0)
        p_bottom_left = p_top_left + a(0, -options["unit_height"])
        p_bottom_right = pos + a(options["coupler_width"], -options["unit_height"])

        draw_code = "% drawing coupler\n"
        draw_code += draw_coupler_arm(p_top_left, p_top_right, -arm_height)
        draw_code += draw_coupler_arm(p_bottom_left, p_bottom_right, arm_height)
        return draw_code

    comp_coupler = Component(schematic=Schematic(w=options["coupler_width"], height_slots=2, draw_fun=draw_coupler))
    core_matrix = np.sqrt(0.5) * np.array([[1.0, -1j], [-1j, 1.0]], dtype=sympy.symbol.Symbol)
    comp_coupler.matrix[1:3, 1:3] = core_matrix

    return comp_coupler


def create_delay(location="top"):

    def draw_delay(pos: np.ndarray):
        delay_width_frac = 0.5
        delay_main_width = delay_width_frac * options["delay_width"]
        straight_width = (options["delay_width"] - delay_main_width) / 2
        p1 = pos
        p2 = p1 + np.array([straight_width, 0])
        p3 = p1 + np.array([options["delay_width"] / 2, options["delay_height"]])
        p4 = p2 + np.array([delay_main_width, 0])
        p5 = p4 + a(straight_width, 0)
        return "\draw [thick] {} to [out=0,in=180] {} to [in=180,out=0] {} to [in=180,out=0] {} to {};\n".format(
            tt(p1), tt(p2), tt(p3), tt(p4), tt(p5)
        )

    comp_delay = Component(schematic=Schematic(w=options["delay_width"], height_slots=1, draw_fun=draw_delay))

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
    def draw_phase(pos):
        p_left = pos
        p_right = pos + a(options["delay_width"], 0)

        rect_width_frac = 0.5
        rect_width = options["delay_width"] * rect_width_frac
        nothing_width = (options["delay_width"] - rect_width) / 2
        p_rect_tl = p_left + a(nothing_width, options["delay_height"]/2)
        p_rect_br = p_rect_tl + a(rect_width, -options["delay_height"])

        draw_code = "% drawing delay\n"
        draw_code += "\draw [thick] {} to {};\n".format(
            tt(p_right), tt(p_left)
        )
        draw_code += "\\draw [fill=white, draw=black,thick] {} rectangle{};\n".format(
            tt(p_rect_tl), tt(p_rect_br)
        )
        return draw_code

    comp_phase = Component(schematic=Schematic(w=options["phase_width"], height_slots=1, draw_fun=draw_phase))

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
    core_matrix = np.array([[0,1], [1,0]], dtype=sympy.symbol.Symbol)
    crosser.matrix[1:3, 1:3] = core_matrix
    return crosser


def f1():
    block = create_coupler() * create_delay() * create_phase(delay_param="phi")
    block2 = create_coupler()*create_coupler()
    block2.shift_up()
    block3 = create_coupler()
    block3.shift_down()
    block = block * block2 * block3


    # block.matrix[0,0] = 0.945672345
    print(block)
    # print(block.schematics)
    draw_code = block.draw()
    with open("ofit_test.tex", "w") as f:
        write_string = r"""\begin{{tikzpicture}}
{}\end{{tikzpicture}}
""".format(draw_code)
        f.write(write_string)


def main():
    f1()
    pass

if __name__ == "__main__":
    main() # pragma: no cover
