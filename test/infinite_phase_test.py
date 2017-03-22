from matplotlib import pyplot

from soapy import atmosphere


if __name__ == "__main__":

    nx_size = 32
    pixel_scale = 4/32
    r0 = 0.16
    L0 = 100.
    wind_speed = pixel_scale/2
    time_step = 1.
    wind_direction = 10

    phase_screen = atmosphere.InfinitePhaseScreen(nx_size, pixel_scale, r0, L0, wind_speed, time_step, wind_direction)

    fig = pyplot.figure()
    for i in range(100):
        screen = phase_screen.move_screen()

        pyplot.cla()
        pyplot.imshow(screen)
        pyplot.pause(0.001)

