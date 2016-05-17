from ofit import *
import sympy

def simple_1():
    filter_1 = make_coupler()
    filter_1 = filter_1 * make_delay()
    filter_1.draw()


def junguji96(n=3):
    def generate_unit():
        return make_phase(draw_sep=True, shift=-1) * make_ring() * make_coupler()

    lattice = make_coupler()
    for i in range(n):
        lattice = lattice *  generate_unit()

    lattice.draw()
    # print(str(lattice))
    h = lattice.matrix[2,2]
    print(h)
    print(sympy.mathematica_code(h))


def OM04(n=3):
    def generate_unit():
        delay = make_delay(draw_sep=True)
        bottom_phase_inner = make_phase(shift=-1)
        return delay * bottom_phase_inner * make_mzi()

    lattice = make_phase(shift=-1) * make_mzi()
    for i in range(n):
        lattice *= generate_unit()

    lattice.draw()


def mimo():
    phase1 = make_phase(shift=1)
    phase2 = make_phase()
    phase3 = make_phase(shift=-1)
    phase4 = make_phase(shift=-2)

    lattice = make_coupler(shift=1) * make_coupler(shift=-1)

    lattice *= phase1 * phase2 * phase3 * phase4
    lattice *= make_crosser()

    lattice *= make_coupler(shift=1) * make_coupler(shift=-1)
    lattice.draw()
    print(str(lattice))


def f1():
    top = make_coupler() * make_phase() * make_coupler()
    top.shift_up()

    lattice = make_coupler() * make_ring(shift=-1) * top * make_coupler()
    lattice.draw()


# simple_1()
junguji96(n=8)
# OM04(n=3)
# f1()
# mimo()
