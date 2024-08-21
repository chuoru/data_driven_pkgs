# Standard libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

# Internal libraries
sys.path.append('../..')
from models.free_fall import Fall  # noqa: E402


def main():
    fall = Fall()

    T = 10  # s

    dt = 0.01  # s

    t = np.arange(0, T, dt)

    state = np.array([0, 0, 10.0, 0, 9.81])

    outputs = fall.simulate(state, T, dt)

    figure, (ax1, ax2) = plt.subplots(2)

    ax1.plot(t, outputs[:, 0])

    ax1.set_title('Position x (m)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')

    ax2.plot(t, outputs[:, 2])
    ax2.set_title('Position y (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
