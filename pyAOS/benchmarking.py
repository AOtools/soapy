'''
Script to benchmark AOSim
'''
import cProfile
import numpy
import sys

import main

def bench():
    sim = main.Sim("conf/testConf.py")
    sim.aoinit()

    print("making IMat")
    print(cProfile.runctx("sim.makeIMat()",globals(),locals(),
            sort="cumulative",filename="benchmarkIMat.p"))

    print("running Loop")
    print(cProfile.runctx("sim.aoloop()",globals(),locals(),
            sort="cumulative",filename="benchmarkLoop.p"))


if __name__=="__main__":
    bench()
