import ofit

def example(n=2):
    ofit.options["unit_height"] = 0.5
    f = ofit.make_mzi() * ofit.make_delay() * ofit.make_phase(shift=-1)
    for i in range(n - 1):
        f *= ofit.make_mzi() * ofit.make_delay() * ofit.make_phase(shift=-1)

    print(f.matrix)
    f.draw()


example()
