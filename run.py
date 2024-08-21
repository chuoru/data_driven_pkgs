#!/usr/bin/env python3
##
# @file run.py
#
# @brief Provides a script for running the random sampler
# and estimate "model" using direct method of data-driven.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard libraries
import numpy as np
from matplotlib import pyplot as plt

# Internal libraries
from samplers.random_sampler import RandomSampler
from models.free_fall import Fall


def main():
    fall = Fall()

    random_sampler = RandomSampler(fall)

    T = 10  # s

    N = 5

    dt = 0.01  # s

    initial_conditions, trajectories = random_sampler.sample(T, N, dt)

    initial_position = np.random.rand(2)

    initial_velocity = np.random.rand(2)

    new_initial_condition = np.array([*initial_position, *initial_velocity])

    new_trajectories = fall.simulate(
        np.array([*new_initial_condition, 9.81]), T, dt)

    prior_knowledge = np.vstack([initial_conditions, np.ones(N)])

    g = np.linalg.inv(prior_knowledge) @ np.array([
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
