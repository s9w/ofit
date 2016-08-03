# from ofit import *
import ofit
import sympy
from sympy import Symbol, Matrix, I, simplify

def simple_1():
    filter_1 = make_coupler()
    filter_1 = filter_1 * make_delay()
    filter_1.draw()


def junguji96(n=3):
    def generate_unit():
        return make_phase(draw_sep=True, shift=-1) * make_ring() * make_coupler()

    lattice = make_coupler()
    for i in range(n):
        lattice *= generate_unit()

    lattice.draw()
    ne_matrix = lattice.matrix_to_ne()
    print(ne_matrix)


    # # print(str(lattice))
    # h = lattice.matrix[2,2]
    # print(lattice.matrix.flatten())
    # # print(type(h))
    # # print(sympy.mathematica_code(h))


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
    lattice = make_coupler(shift=1) * make_coupler(shift=-1)

    phase1 = make_phase(shift=1)
    phase2 = make_phase()
    phase3 = make_phase(shift=-1)
    phase4 = make_phase(shift=-2)
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

def filter1(n=1):
    # def make_unit():
    #     return make_mzi()

    # lattice = make_delay()
    lattice = make_mzi() * make_delay()* make_phase(shift=-1)
    for i in range(n-1):
        lattice *= make_mzi() * make_delay() * make_phase(shift=-1)

    # lattice = make_mzi()
    ne_matrix = lattice.matrix_to_ne()
    print(ne_matrix)
    # lattice.draw()

def test_phis_local():
    # s1 = OfitState()
    # filter1 = s1.make_phase()
    # print(filter1.matrix_to_ne())
    #
    # s2 = OfitState()
    # filter2 = s2.make_phase()
    # filter2 = s2.make_phase()
    # print(filter2.matrix_to_ne())

    filter1 = ofit.make_phase()
    print(filter1.matrix_to_ne())
    ofit.used_names = set()
    filter2 = ofit.make_phase()
    print(filter2.matrix_to_ne())
    print(ofit.get_symbol_count())

def test_dof():
    # m1 = ofit.make_coupler() * ofit.make_coupler(shift=1)
    m1 = ofit.make_ring(shift=-1)
    print(m1.matrix)
    print(m1.matrix_to_ne())
    m1.draw()

    # m1 = c1.matrix[1:3, 1:3]
    # # sympy.simplify(m1)
    # print(m1)
    # # print(c1.matrix_to_ne())
    # # print(c1.matrix_to_ne())
    # # print("dof", c1.dof)

    # m1 = simplify(Matrix([[2,0],[0,2]]))
    # print(m1)


# simple_1()
# junguji96(n=1)
# OM04(n=3)
# f1()
# mimo()
# filter1(n=4)

# test_phis_local()

test_dof()
