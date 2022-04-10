from utils import plot_pendulum
import scipy.linalg as lin
from pendulum_model import export_pendulum_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
from numpy.linalg import norm as norm
import numpy as np
from numpy import nan
import time
import sys
sys.path.insert(0, '../common')


class OCPpendulum:

    def __init__(self):
        # create ocp object to formulate the OCP
        self.ocp = AcadosOcp()

        # set model
        model = export_pendulum_ode_model()
        self.ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        self.Tf = 1.0
        self.N = 20

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf

        # set dimensions
        self.ocp.dims.N = self.N

        # set cost
        Q = 2*np.diag([0.0, 1e-2])
        R = 2*np.diag([0.0])

        self.ocp.cost.W_e = Q
        self.ocp.cost.W = lin.block_diag(Q, R)

        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[2, 0] = 1.0
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((ny_e, ))

        # set constraints
        self.Fmax = 10
        self.thetamax = np.pi/2
        self.thetamin = 0.0
        self.dthetamax = 10.

        self.ocp.constraints.lbu = np.array([-self.Fmax])
        self.ocp.constraints.ubu = np.array([+self.Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        self.ocp.constraints.lbx = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx = np.array([0, 1])
        self.ocp.constraints.lbx_e = np.array([self.thetamin, -self.dthetamax])
        self.ocp.constraints.ubx_e = np.array([self.thetamax, self.dthetamax])
        self.ocp.constraints.idxbx_e = np.array([0, 1])

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP'

        self.simX = np.ndarray((self.N+1, 2))

        # Initial cnditions
        self.ocp.constraints.x0 = np.array([0, 0])

        # Solver
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file='acados_ocp.json')

    def compute_problem(self, q0, v0):

        x0 = np.array([q0, v0])

        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        status = self.ocp_solver.solve()

        if status != 0:
            return 0

        # get solution
        for i in range(self.N):
            self.simX[i, :] = self.ocp_solver.get(i, "x")
        self.simX[self.N, :] = self.ocp_solver.get(self.N, "x")

        if norm(self.simX[self.N, 1]) < 0.01:
            return 1
        else:
            return 0