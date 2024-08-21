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
    # ============================================================================
    # PUBLIC METHODS
    # ============================================================================
    def __init__(self, model):
        """! Constructor
        """
        self._model = model

    def sample(self, T, N, dt, has_input=False):
        """! Sample
        @param T: The time
        @param N: The number of samples
        @param dt: The time step
        @param input: The input of the system
        """
        trajectories = []

        initial_conditions = []

        for _ in range(N):
            intial_condition, state = self._model.sample()

            outputs = self._simulate(state, T, dt, has_input)

            trajectories.append(outputs)

            initial_conditions.append(intial_condition)

        return np.array(initial_conditions).T, np.array(trajectories).T

    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================
    def _simulate(self, state, T, dt, has_input):
        """! Simulate
        @param state: The state of the system
        @param T: The time
        @param dt: The time step
        @param has_input: The input of the system
        @return The output of the system
        """
        input = 0

        if has_input:
            input = np.random.rand(self._model._nu)

            input = [0.5, 0.1]

        outputs = self._model.simulate(state, T, dt, input)

        return outputs
