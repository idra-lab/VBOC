from casadi import MX, vertcat, cos, sin
from .abstract import AbstractModel


class PendulumModel(AbstractModel):
    """ Define the pendulum dynamics model. """
    def __init__(self, params):
        super().__init__(params)

    def addDynamicsModel(self, params):
        self.amodel.name = "pendulum"

        self.x = MX.sym("x", 2)
        self.x_dot = MX.sym("x_dot", 2)
        self.u = MX.sym("u", 1)
        self.p = MX.sym("p", 1)

        # Dynamics
        self.f_expl = vertcat(
            self.x[1],
            (params.m1 * params.g * params.l1 * sin(self.x[0]) + self.u - params.b * self.x[1])
            / (params.l1 * params.l1 * params.m1)
        )


class DoublePendulumModel(AbstractModel):
    """ Define the double pendulum dynamics model. """
    def __init__(self, params):
        super().__init__(params)

    def addDynamicsModel(self, params):
        self.amodel.name = "double_pendulum"

        self.x = MX.sym("x", 4)
        self.x_dot = MX.sym("x_dot", 4)
        self.u = MX.sym("u", 2)
        self.p = MX.sym("p", 2)

        # Dynamics
        self.f_expl = vertcat(
            self.x[2],
            self.x[3],
            (
                    params.l1 ** 2
                    * params.l2
                    * params.m2
                    * self.x[2] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    + 2 * self.u[1] * cos(-self.x[1] + self.x[0]) * params.l1
                    + 2
                    * (
                            params.g * sin(-2 * self.x[1] + self.x[0]) * params.l1 * params.m2 / 2
                            + sin(-self.x[1] + self.x[0]) * self.x[3] ** 2 * params.l1 * params.l2 * params.m2
                            + params.g * params.l1 * (params.m1 + params.m2 / 2) * sin(self.x[0])
                            - self.u[0]
                    )
                    * params.l2
            )
            / params.l1 ** 2
            / params.l2
            / (params.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * params.m1 - params.m2),
            (
                    -params.g
                    * params.l1
                    * params.l2
                    * params.m2
                    * (params.m1 + params.m2)
                    * sin(-self.x[1] + 2 * self.x[0])
                    - params.l1
                    * params.l2 ** 2
                    * params.m2 ** 2
                    * self.x[3] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    - 2
                    * self.x[2] ** 2
                    * params.l1 ** 2
                    * params.l2
                    * params.m2
                    * (params.m1 + params.m2)
                    * sin(-self.x[1] + self.x[0])
                    + 2 * self.u[0] * cos(-self.x[1] + self.x[0]) * params.l2 * params.m2
                    + params.l1
                    * (params.m1 + params.m2)
                    * (sin(self.x[1]) * params.g * params.l2 * params.m2 - 2 * self.u[1])
            )
            / params.l2 ** 2
            / params.l1
            / params.m2
            / (params.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * params.m1 - params.m2)
        )


class TriplePendulumModel(AbstractModel):
    """ Define the triple pendulum dynamics model. """
    def __init__(self, params):
        super().__init__(params)

    def addDynamicsModel(self, params):
        self.amodel.name = "triple_pendulum"

        self.x = MX.sym("x", 6)
        self.x_dot = MX.sym("x_dot", 6)
        self.u = MX.sym("u", 3)
        self.p = MX.sym("p", 3)

        # dynamics
        self.f_expl = vertcat(
            self.x[3],
            self.x[4],
            self.x[5],
            (-params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                -2 * self.x[2] + 2 * self.x[1] + self.x[
                    0]) - params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                2 * self.x[2] - 2 * self.x[1] + self.x[0]) + 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + 2 * self.x[
                 3] ** 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(-2 * self.x[1] + 2 * self.x[0]) - 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) * cos(
                -2 * self.x[1] + self.x[0] + self.x[2]) - 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + self.x[1] + self.x[
                    0]) + 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) *
             cos(self.x[0] - self.x[2]) + 2 * (
                     self.u[1] * params.l1 * (params.m3 + 2 * params.m2) * cos(-self.x[1] + self.x[0]) + (
                     params.g * params.l1 * params.m2 * (params.m2 + params.m3) * sin(
                 -2 * self.x[1] + self.x[0]) + 2 * self.x[4] ** 2 * params.l1 * params.l2 * params.m2 * (
                             params.m2 + params.m3) * sin(-self.x[1] + self.x[0]) + params.m3 * self.x[
                         5] ** 2 * sin(
                 self.x[0] - self.x[2]) * params.l1 * params.l2 * params.m2 + params.g * params.l1 * (
                             params.m2 ** 2 + (
                             params.m3 + 2 * params.m1) * params.m2 + params.m1 * params.m3) * sin(
                 self.x[0]) - self.u[0] * (
                             params.m3 + 2 * params.m2)) * params.l2) * params.l2) / params.l1 ** 2 / params.l2 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) - params.m2 ** 2 + (
                            -params.m3 - 2 * params.m1) * params.m2 - params.m1 * params.m3) / params.l2 / 2,
            (-2 * self.u[2] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m2 * params.m3 * self.x[5] ** 2 * sin(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.l1 * params.l2 * params.l2 * params.m1 * params.m3 * sin(
                self.x[1] + 2 * self.x[0] - 2 * self.x[2]) - params.g * params.l1 * params.l2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                -self.x[1] + 2 * self.x[0]) - 2 * self.x[
                 4] ** 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m2 * (
                     params.m2 + params.m3) * sin(
                -2 * self.x[1] + 2 * self.x[0]) + 2 * self.u[1] * params.l1 * params.l2 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[0]) + 2 * params.l1 * params.l2 ** 2 * params.l2 * params.m1 * params.m3 *
             self.x[4] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * cos(
                        -2 * self.x[2] + self.x[1] + self.x[
                            0]) + 2 * params.l1 ** 2 * params.l2 * params.l2 * params.m1 * params.m3 * self.x[
                 3] ** 2 * sin(
                        -2 * self.x[2] +
                        self.x[1] + self.x[0]) - 2 * params.l1 ** 2 * params.l2 * self.x[3] ** 2 * (
                     (params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                     params.m1 + params.m2)) * params.l2 * sin(
                        -self.x[1] + self.x[0]) + 2 * self.u[2] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(
                        -self.x[2] + self.x[1]) + (2 * self.u[0] * params.l2 * (params.m3 + 2 * params.m2) * cos(
                        -self.x[1] + self.x[0]) + params.l1 * (
                                                           4 * self.x[5] ** 2 * params.m3 * params.l2 * (
                                                           params.m1 + params.m2 / 2) * params.l2 * sin(
                                                       -self.x[2] + self.x[
                                                           1]) + params.g * params.m3 * params.l2 * params.m1 * sin(
                                                       -2 * self.x[2] + self.x[1]) + params.g * (
                                                                   (
                                                                           params.m1 + 2 * params.m2) * params.m3 + 2 * params.m2 * (
                                                                           params.m1 + params.m2)) * params.l2 * sin(
                                                       self.x[1]) - 2 * self.u[1] * (
                                                                   params.m3 + 2 * params.m1 + 2 * params.m2))) * params.l2) / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 / params.l2 ** 2 / 2,
            (-2 * params.m3 * self.u[1] * params.l1 * params.l2 * (params.m2 + params.m3) * cos(
                2 * self.x[0] - self.x[2] - self.x[
                    1]) + params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(2 * self.x[0] + self.x[2] - 2 * self.x[1]) + 2 * self.u[
                 2] * params.l1 * params.l2 * (
                     params.m2 + params.m3) ** 2 * cos(
                -2 * self.x[1] + 2 * self.x[
                    0]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                2 * self.x[0] - self.x[2]) - params.g * params.m3 * params.l1 * params.l2 * params.l2 * params.m1 * (
                     params.m2 + params.m3) * sin(
                -self.x[2] + 2 * self.x[1]) - 2 * params.l1 * params.l2 * params.l2 ** 2 * params.m1 * params.m3 ** 2 *
             self.x[5] ** 2 * sin(
                        -2 * self.x[2] + 2 * self.x[1]) - 2 * self.u[0] * params.l2 * params.l2 * params.m3 * (
                     params.m2 + params.m3) * cos(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.x[3] ** 2 * params.l1 ** 2 *
             params.l2 * params.l2 * params.m1 * (params.m2 + params.m3) * sin(
                        -2 * self.x[1] + self.x[0] + self.x[2]) + 2 * params.m3 * self.u[1] * params.l1 * params.l2 * (
                     params.m3 + 2 * params.m1 + params.m2) * cos(-self.x[2] + self.x[1]) + (params.m2 + params.m3) * (
                     2 * self.u[0] * params.l2 * params.m3 * cos(self.x[0] - self.x[2]) + params.l1 * (
                     -2 * params.m3 * self.x[3] ** 2 * params.l1 * params.l2 * params.m1 * sin(
                 self.x[0] - self.x[2]) - 4 * params.m3 * self.x[4] ** 2 * sin(
                 -self.x[2] + self.x[1]) * params.l2 * params.l2 * params.m1 + params.g * params.m3 * sin(
                 self.x[2]) * params.l2 * params.m1 - 2 * self.u[2] * (
                             params.m3 + 2 * params.m1 + params.m2))) * params.l2) / params.m3 / (
                    params.m2 * (params.m2 + params.m3) * cos(
                -2 * self.x[1] + 2 * self.x[0]) + params.m1 * params.m3 * cos(
                -2 * self.x[2] + 2 * self.x[1]) + (
                            -params.m1 - params.m2) * params.m3 - 2 * params.m1 * params.m2 - params.m2 ** 2) / params.l1 / params.l2 ** 2 / params.l2 / 2,
        )
