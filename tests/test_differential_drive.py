#!/usr/bin/env python3
##
# @file test_differential_drive.py
#
# @brief Provides a script for running the random sampler
# and estimate "model" using direct method of data-driven.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

# Internal libraries
sys.path.append('../')
from samplers.random_sampler import RandomSampler  # noqa: E402
from models.differential_drive import DifferentialDrive  # noqa: E402


def main():
    model = DifferentialDrive(0.5)

    random_sampler = RandomSampler(model)

    T = 100  # s

    N = 100

    dt = 0.01  # s

    initial_conditions, trajectories = random_sampler.sample(
        T, N, dt, has_input=True)

    new_initial_condition = np.random.rand(3)

    new_trajectories = model.simulate(
        new_initial_condition, T, dt, [0.5, 0.1])

    prior_knowledge = np.vstack([initial_conditions, np.ones(N)])

    g = np.linalg.pinv(prior_knowledge) @ np.array([
        *new_initial_condition, 1])

    predicted_trajectory = trajectories @ g

    plt.plot(predicted_trajectory[0, :], predicted_trajectory[2, :], 'r')

    plt.plot(new_trajectories[:, 0], new_trajectories[:, 2], '--b')

    plt.title('Comparison between predicted and actual trajectories')

    plt.xlabel('Position x (m)')
    
    plt.ylabel('Position y (m)')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
