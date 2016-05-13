from ofit import *


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


simple_1()
# junguji96(n=3)
# OM04(n=3)
