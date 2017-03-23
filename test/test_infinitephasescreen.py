import time
from matplotlib import pyplot

import tqdm
from soapy import atmosphere



if __name__ == "__main__":

    nx_size = 400
    pixel_scale = 8./128
    r0 = 0.16
    L0 = 100.
    wind_speed = pixel_scale/2
    time_step = 1.
    wind_direction = 0

    N_iters = 1000

    print("initialising phase screen...")
    phase_screen = atmosphere.InfinitePhaseScreen(nx_size, pixel_scale, r0, L0, wind_speed, time_step, wind_direction)
    print("Done!")

    fig = pyplot.figure()

    t1 = time.time()
    for i in tqdm.tqdm(range(N_iters)):
        screen = phase_screen.move_screen()
    t2 = time.time()

    elapsed = t2 - t1
    itspersec = N_iters/elapsed

    print("Iterations per second: {} it/s".format(itspersec))