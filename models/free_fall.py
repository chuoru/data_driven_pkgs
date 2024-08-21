#!/usr/bin/env python3
##
# @file free_fall.py
#
# @brief Provides analytical model for free fall with friction.
# This model is used to simulate the free fall of an object with friction.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard libraries
import numpy as np
import scipy.signal as signal


class Fall:
    """! Free fall with friction model
    """

    def __init__(self, gamma=1.0, m=1.0):
        """! Constructor
        @param gamma: The friction coefficient
        @param m: The mass of the object
        """
        self._nx = 5

        self._nu = 0

        self._ny = 4

        self._gamma = gamma

        self._m = m

    def function(self, state, dt, input=0):
        """! Function
        @param state: The state of the system
        @param input: The input of the system
        @param dt: The time step
        @return The output of the system
        """
        A = np.array([
            [0, 1, 0, 0, 0],
            [0, -self._gamma/self._m, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, -self._gamma/self._m, -1],
            [0, 0, 0, 0, 0]
        ])

        B = np.array([
            [0],
            [0],
            [0],
            [1],
            [0]
        ])

        C = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])

        D = np.zeros((4, 1))

        sys = signal.StateSpace(A, B, C, D)

        _, output, state = signal.lsim(sys, U=[input], T=[0, dt], X0=state)

        return output[-1], state[-1]

    def sample(self):
        """! Sample state
        @return A random state
        """
        initial_position = np.random.rand(2)

        initial_velocity = np.random.rand(2)

        intial_condition = np.array([*initial_position, *initial_velocity])

        state = np.array([*intial_condition, 9.81])

        return intial_condition, state

    def simulate(self, state, T, dt, input=0):
        """! Simulate
        @param state: The state of the system
        @param input: The input of the system
        @param T: The time
        @param dt: The time step
        @return The outputs of the system
        """
        t = np.arange(0, T, dt)

        outputs = np.zeros((len(t), self._ny))

        for index in range(len(t)):
            output, state = self.function(state, dt, input)

            outputs[index, :] = output

        return outputs
