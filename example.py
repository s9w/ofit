# import ofit
from ofit import *
# from ofit import cr
# from ofit import create_coupler
import sympy

def simple_1():
    filter1 = create_coupler()
    filter1 = filter1 * create_delay()
    filter1.draw()


def junguji96(n=3):
    def generate_unit():
        bottom_phase = create_phase(draw_sep=True)
        bottom_phase.shift_down()
        return bottom_phase * create_ring() * create_coupler()

    lattice = create_coupler()
    for i in range(n):
        lattice = lattice * generate_unit()

    lattice.draw()
    # print(str(lattice))
    h = lattice.matrix[2,2]
    print(h)
    print(sympy.mathematica_code(h))


def OM04(n=3):
    def generate_unit():
        delay = create_delay(draw_sep=True)
        bottom_phase_inner = create_phase()
        bottom_phase_inner.shift_down()
        return delay * bottom_phase_inner * create_mzi()

    bottom_phase = create_phase()
    bottom_phase.shift_down()
    lattice = bottom_phase * create_mzi()
    for i in range(n):
        lattice = lattice * generate_unit()

    lattice.draw()


def mimo():
    phase1 = create_phase(shift=1)
    phase2 = create_phase()
    phase3 = create_phase(shift=-1)
    phase4 = create_phase(shift=-2)

    lattice = create_coupler(shift=1) * create_coupler(shift=-1)

    lattice = lattice * phase1 * phase2 * phase3 * phase4
    lattice = lattice * create_crosser()

    lattice = lattice * create_coupler(shift=1) * create_coupler(shift=-1)
    lattice.draw()


def f1():
    top = create_coupler() * create_phase() * create_coupler()
    top.shift_up()

    ring = create_ring()
    ring.shift_down()
    lattice = create_coupler() * ring * top * create_coupler()
    lattice.draw()


# simple_1()
# junguji96(n=8)
# OM04(n=3)
# f1()
mimo()
