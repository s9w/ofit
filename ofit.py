import copy
import numpy as np
import sympy
import math
from sympy import Symbol, Matrix, I


def a(x, y):
    return np.array([x, y])

def tt(np_array):
    return (np_array[0], np_array[1])


options = {
    "gamma": "gamma",
    "zm1": sympy.exp(-1j*Symbol("omega")*Symbol("tau")),
    "filename": "example_filter_tikz.tex",

    "unit_height": 1,

    # coupler
    "coupler_width": 0.7,

    # phase shifter
    "phase_width": 0.5,

    # delay
    "delay_width": 0.8,
    "delay_height": 0.3,

    # crossing element
    "crosser_width": 1,

    # ring
    "ring_width": 0.8,
    "ring_diameter": 0.8
}

used_names = set()


def draw_line(left, right):
    return "\draw [thick] {} -- {};\n".format(
        tt(left), tt(right)
    )


def draw_rect(middle: np.ndarray, width, height):
    bottom_left = middle + a(-width/2, -height/2)
    top_right = bottom_left + a(width, height)

    draw_code = ""
    draw_code += "\\draw [fill=white, draw=black,thick] {} rectangle{};\n".format(
        tt(bottom_left), tt(top_right)
    )
    return draw_code


def draw_delay(pos: np.ndarray, **kwargs):
    delay_width_frac = 0.5
    delay_main_width = delay_width_frac * options["delay_width"]
    straight_width = (options["delay_width"] - delay_main_width) / 2
    p1 = pos
    p2 = p1 + a(straight_width, 0)
    p3 = p1 + a(options["delay_width"] / 2, options["delay_height"])
    p4 = p2 + a(delay_main_width, 0)
    p5 = p4 + a(straight_width, 0)
    return "\draw [thick] {} to [out=0,in=180] {} to [in=180,out=0] {} to [in=180,out=0] {} to {};\n".format(
        tt(p1), tt(p2), tt(p3), tt(p4), tt(p5)
    )


def draw_phase(pos: np.ndarray, param_name):
    p_left = pos
    p_middle = pos + a(options["phase_width"] / 2, 0)
    p_right = pos + a(options["phase_width"], 0)

    rect_width_frac = 0.8
    rect_width = options["phase_width"] * rect_width_frac

    draw_code = "% drawing delay\n"
    draw_code += "\draw [thick] {} to {};\n".format(
        tt(p_right), tt(p_left)
    )
    draw_code += draw_rect(p_middle, rect_width, options["delay_height"])

    draw_code += "\\node [scale=0.5] at {} {{{}}};\n".format(
        tt(p_middle), texify_param(param_name)
    )
    return draw_code


def draw_arm(start, end):
    total_width = (end - start)[0]
    height = (end - start)[1]
    arm_frac = 0.95
    middle_width = total_width * arm_frac
    lead_width = (total_width - middle_width) / 2

    p1 = start
    p2 = p1 + a(lead_width, 0)
    p3 = p2 + a(middle_width, height)
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


def draw_coupler(pos: np.ndarray, **kwargs):
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


def make_symbol(name=None) -> Symbol:
    # generate automatic name
    if not name:
        for i in range(999):
            name = "phi_{}".format(i)
            if name not in used_names:
                used_names.add(name)
                return Symbol(name)

    # check if name isn't already used
    else:
        if name in used_names:
            raise ValueError("Filter parameter name \"{}\" already used.".format(name))
        else:
            used_names.add(name)
            return Symbol(name)


class Schematic(object):
    def __init__(self, w, height_slots, draw_fun, param_name=None, draw_sep=False) -> "Schematic":
        self.draw_fun = draw_fun
        self.width = w
        self.height_slots = height_slots
        self.vpos = 0
        self.param_name = param_name
        self.draw_sep = draw_sep


class Component(object):
    def __init__(self, schematic: "Schematic", shift) -> "Component":
        self.matrix = np.eye(4, dtype=sympy.symbol.Symbol)
        self.schematics = [schematic]

        # initial shifting
        if shift == 1:
            self.shift_up()
        elif shift == -1:
            self.shift_down()
        elif shift == -2:
            self.shift_down()
            self.shift_down()
        elif shift == 0:
            pass
        else:
            raise ValueError

    def __mul__(self: "Component", other: "Component") -> "Component":
        product = copy.deepcopy(self)
        product.matrix = other.matrix.dot(self.matrix)

        product.schematics = product.schematics + other.schematics
        return product

    def __imul__(self: "Component", other: "Component") -> "Component":
        self.matrix = other.matrix.dot(self.matrix)
        self.schematics = self.schematics + other.schematics
        return self

    def draw(self, filename=options["filename"]):
        def vpos_convert_index(x):
            return 1-x

        positions = [0, 0, 0, 0]
        draw_code = ""

        separator_x_coords = []
        for sch in self.schematics:
            array_index = vpos_convert_index(sch.vpos)
            if sch.height_slots == 2:
                x1 = positions[array_index]
                x2 = positions[array_index+1]
                pos_x = max(positions[array_index], positions[array_index+1])

                # leftover space to cover with line
                if not math.isclose(x1, x2):
                    if x1 < x2:
                        y = 1 - array_index
                    else:
                        y = 1 - (array_index+1)
                    draw_code += draw_line(a(x1, y), a(x2, y))

                positions[array_index] = pos_x + sch.width
                positions[array_index+1] = pos_x + sch.width
            elif sch.height_slots == 1:
                pos_x = positions[array_index]
                positions[array_index] += sch.width
            else:
                raise ValueError

            pos = a(pos_x, sch.vpos)
            draw_code += sch.draw_fun(pos, param_name=sch.param_name) + "\n"
            if sch.draw_sep:
                separator_x_coords.append(pos_x)

        # draw separators
        draw_code += "% drawing separators\n"
        for pos_x in separator_x_coords:
            draw_code += "\draw [dashed] ($ (current bounding box.north west) + ({},0) $) to ($ (current bounding box.south west) + ({},0) $);\n".format(
                pos_x, pos_x
            )

        with open(filename, "w") as f:
            write_string = r"""\usetikzlibrary{{calc}}
\begin{{tikzpicture}}
{}\end{{tikzpicture}}
        """.format(draw_code)
            f.write(write_string)

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

    def matrix_to_ne(self):
        def change_complex(a):
            return str(a).replace("I", "1j")
        vfunc = np.vectorize(change_complex)
        return vfunc(self.matrix[1:3, 1:3])


def make_coupler(shift=0):
    comp_coupler = Component(schematic=Schematic(
        w=options["coupler_width"],
        height_slots=2,
        draw_fun=draw_coupler),
        shift=shift
    )
    core_matrix = sympy.sqrt(0.5) * np.array([[1.0, -1j], [-1j, 1.0]], dtype=sympy.symbol.Symbol)
    comp_coupler.matrix[1:3, 1:3] = core_matrix

    return comp_coupler


def make_delay(location="top", shift=0, draw_sep=False):
    comp_delay = Component(
        schematic=Schematic(
            w=options["delay_width"],
            height_slots=1,
            draw_fun=draw_delay,
            draw_sep=draw_sep
        ),
        shift=shift
    )

    if location == "top":
        core_matrix = np.array([[options["zm1"], 0], [0, 1]], dtype=sympy.symbol.Symbol)
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, options["zm1"]]], dtype=sympy.symbol.Symbol)
    else:
        raise ValueError

    comp_delay.matrix[1:3, 1:3] = core_matrix

    return comp_delay


def texify_param(param_name):
    if param_name.startswith("phi_"):
        return "$\\phi_{{{}}}$".format(param_name[4:])
    else:
        return param_name


def make_phase(phase_param: str =None, shift=0, location="top", draw_sep=False):
    phi = make_symbol(phase_param)
    comp_phase = Component(
        schematic=Schematic(
            w=options["phase_width"],
            height_slots=1,
            draw_fun=draw_phase,
            param_name=phi.name,
            draw_sep=draw_sep
        ),
        shift=shift
    )

    if location == "top":
        core_matrix = np.array([[sympy.exp(-1j * phi), 0], [0, 1]], dtype=sympy.symbol.Symbol)
    elif location == "bottom":
        core_matrix = np.array([[1, 0], [0, sympy.exp(-1j * phi)]], dtype=sympy.symbol.Symbol)
    else:
        raise ValueError

    comp_phase.matrix[1:3, 1:3] = core_matrix
    return comp_phase


def make_mzi():
    return make_coupler() * make_phase() * make_coupler()


def make_crosser(draw_sep=False, shift=0):
    def draw_crosser(pos: np.ndarray, **kwargs):
        top_left = pos
        top_right = pos + a(options["crosser_width"], 0)
        bottom_left = pos + a(0, -options["unit_height"])
        bottom_right = bottom_left + a(options["crosser_width"], 0)

        draw_code = "% drawing delay\n"
        draw_code += draw_arm(top_left, bottom_right)
        draw_code += draw_arm(bottom_left, top_right)
        return draw_code

    crosser = Component(
        schematic=Schematic(
            w=options["crosser_width"],
            height_slots=2,
            draw_fun=draw_crosser,
            draw_sep=draw_sep
        ),
        shift=shift
    )
    core_matrix = np.array([[0, 1], [1, 0]], dtype=sympy.symbol.Symbol)
    crosser.matrix[1:3, 1:3] = core_matrix
    return crosser


def make_ring(phase_param=None, gamma=sympy.S.One, shift=0, draw_sep=False):
    def draw_ring(pos: np.ndarray, param_name):
        left = pos
        right = pos + a(options["ring_width"], 0)
        middle = pos + a(options["ring_width"]/2, 0)
        phase_pos = middle + a(0, options["ring_diameter"])
        ring_center = middle + a(0, options["ring_diameter"]/2)
        ring_radius = options["ring_diameter"]/2

        draw_code = "% drawing ring\n"
        draw_code += draw_line(left, right)
        draw_code += "\draw [thick] {} circle [radius={}];\n".format(
            tt(ring_center), ring_radius
        )
        draw_code += draw_rect(phase_pos, options["ring_diameter"]*0.5, options["ring_diameter"]*0.3)
        draw_code += "\\node [scale=0.5] at {} {{{}}};\n".format(
            tt(phase_pos), texify_param(param_name)
        )

        return draw_code

    # transfer function
    c = sympy.sqrt(0.5)
    phase_factor = make_symbol(phase_param)
    phase_term = sympy.exp(-sympy.I * phase_factor)
    zm1 = options["zm1"]
    transfer_func = (c-gamma*zm1*phase_term)/(1-c*gamma*zm1*phase_term)
    core_matrix = np.array([[transfer_func, 0], [0, 1]], dtype=sympy.symbol.Symbol)

    component = Component(
        schematic=Schematic(
            w=options["ring_width"],
            height_slots=1,
            draw_fun=draw_ring,
            param_name=phase_factor.name,
            draw_sep=draw_sep
        ),
        shift=shift
    )

    component.matrix[1:3, 1:3] = core_matrix
    return component
