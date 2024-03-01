import re
import numpy as np
from casadi import MX
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcp, AcadosOcpSolver


class AbstractModel:
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        # Dummy dynamics (double integrator)
        self.amodel.name = "double_integrator"
        self.x = MX.sym("x")
        self.x_dot = MX.sym("x_dot")
        self.u = MX.sym("u")
        self.f_expl = self.u
        self.p = MX.sym("p")
        self.addDynamicsModel(params)
        self.amodel.x = self.x
        self.amodel.xdot = self.x_dot
        self.amodel.u = self.u
        self.amodel.f_expl_expr = params.dt * self.f_expl
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = int(self.nx / 2)
        self.nv = self.nx - self.nq

        # Joint limits
        self.u_min = -params.u_max * np.ones(self.nu)
        self.u_max = params.u_max * np.ones(self.nu)
        self.x_min = np.hstack([params.q_min * np.ones(self.nq), -params.dq_max * np.ones(self.nq)])
        self.x_max = np.hstack([params.q_max * np.ones(self.nq), params.dq_max * np.ones(self.nq)])
        self.eps = params.state_tol

    def addDynamicsModel(self, params):
        pass

    def insideStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min + self.eps, x <= self.x_max - self.eps))

    def insideControlConstraints(self, u):
        return np.all(np.logical_and(u >= self.u_min, u <= self.u_max))

    def insideRunningConstraints(self, x, u):
        return self.insideStateConstraints(x) and self.insideControlConstraints(u)

    def checkPositionBounds(self, q):
        return np.logical_or(np.any(q < self.x_min[:self.nq] + self.eps), np.any(q > self.x_max[:self.nq] - self.eps))

    def checkVelocityBounds(self, v):
        return np.logical_or(np.any(v < self.x_min[self.nq:] + self.eps), np.any(v > self.x_max[self.nq:] - self.eps))

    def checkStateBounds(self, x):
        return np.logical_or(np.any(x < self.x_min + self.eps), np.any(x > self.x_max - self.eps))


class SimDynamics:
    def __init__(self, model):
        self.model = model
        self.params = model.params
        sim = AcadosSim()
        sim.model = model.amodel
        # T --> should be dt, but instead we multiply the dynamics by dt
        # this speed up the increment of the OCP horizon
        sim.solver_options.T = 1.
        sim.solver_options.integrator_type = self.params.integrator_type
        sim.solver_options.num_stages = self.params.num_stages
        sim.parameter_values = np.zeros(model.nv)
        gen_name = self.params.GEN_DIR + '/sim_' + sim.model.name
        sim.code_export_directory = gen_name
        self.integrator = AcadosSimSolver(sim, build=self.params.regenerate, json_file=gen_name + '.json')

    def simulate(self, x, u):
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        self.integrator.solve()
        x_next = self.integrator.get("x")
        return x_next

    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.model.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1] = self.simulate(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return np.linalg.norm(x - x_sim) < self.params.state_tol * np.sqrt(n+1)


class AbstractController:
    def __init__(self, simulator):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.simulator = simulator
        self.params = simulator.params
        self.model = simulator.model

        self.N = int(self.params.T / self.params.dt)
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.N
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.addCost()

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)
        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
        # Additional constraints
        self.addConstraint()

        # Solver options
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.qp_solver_tol_stat = self.params.qp_tol_stat
        self.ocp.solver_options.nlp_solver_tol_stat = self.params.nlp_tol_stat
        #   IMPORTANT
        # NLP solver tol stat is the tolerance on the stationary condition
        # Asia used a higher value than default (1e-6) to obtain a higher number of solutions
        # maybe try to find a better trade-off btw 1e-3 and 1e-6
        #########################
        self.ocp.solver_options.alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt

        # Generate OCP solver
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.regenerate)

        # Storage
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def addCost(self):
        pass

    def addConstraint(self):
        pass

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, 1.))
        self.ocp_solver.update_qp_solver_cond_N(N)
