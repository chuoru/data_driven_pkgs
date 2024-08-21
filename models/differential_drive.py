#!/usr/bin/env python3
##
# @file differential_drive.py
#
# @brief Provides a model for a differential drive robot.
# This model is used to simulate the motion of a differential drive robot.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard library
import math

# External library
import numpy as np


class DifferentialDrive:
    """! Differential drive robot model

    This class provides a model for a differential drive robot.
    """
    def __init__(self, wheel_base):
        """! Constructor
        @param wheel_base: The distance between the two wheels
        """
        self._wheel_base = wheel_base

        self._v_max = 1.0

        self._nx = 3

        self._nu = 2

        self._ny = 3

    def function(self, state, dt, input):
        """! Function
        @param state: The state of the system
        @param input: The input of the system
        @param dt: The time step
        @return The output of the system
        """
        v = input[0]

        w = input[1]

        dfdt = np.array([math.cos(state[2]) * v, math.sin(state[2]) * v, w])

        next_state = state + dfdt * dt

        return next_state

    def simulate(self, state, input, T, dt):
        """! Simulate
        @param state: The state of the system
        @param input: The input of the system
        @param T: The time
        @param dt: The time step
        @return The output of the system
        """
        t = np.arange(0, T, dt)

        outputs = np.zeros((len(t), self._nx))

        for i in range(len(t)):
            state = self.function(state, dt, input)

            outputs[i, :] = state

        return outputs
