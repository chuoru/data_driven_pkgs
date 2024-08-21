#!/usr/bin/env python3
##
# @file random_sampler.py
#
# @brief Provides a random sampler for sampling from a dataset.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard library
import numpy as np


class RandomSampler:
    """! Random sampler
    """

    def __init__(self, model):
        """! Constructor
        """
        self._model = model

    def sample(self, T, N, dt):
        """! Sample
        @param T: The time
        @param N: The number of samples
        """
        trajectories = []

        initial_conditions = []

        for _ in range(N):
            initial_position = np.random.rand(2)

            initial_velocity = np.random.rand(2)

            intial_condition = np.array([*initial_position, *initial_velocity])

            state = np.array([*intial_condition, 9.81])

            outputs = self._model.simulate(state, T, dt)

            trajectories.append(outputs)

            initial_conditions.append(intial_condition)

        return np.array(initial_conditions).T, np.array(trajectories).T
