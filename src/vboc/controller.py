import numpy as np
from casadi import dot
from .abstract import AbstractController


class ViabilityController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.C = np.zeros((self.model.nv, self.model.nx))

    def addCost(self):
        # Maximize initial velocity
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_0 = dot(self.model.p, self.model.x[self.model.nq:])
        self.ocp.parameter_values = np.zeros(self.model.nv)

    def addConstraint(self):
        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        self.ocp.constraints.C = np.zeros((self.model.nv, self.model.nx))
        self.ocp.constraints.D = np.zeros((self.model.nv, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv,))
        self.ocp.constraints.ug = np.zeros((self.model.nv,))
    
    def solve(self, q_init, d):
        self.ocp_solver.reset()
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', d)
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', d)

        # Set the initial constraint
        d_arr = np.array([d.tolist()])
        self.C[:, self.model.nq:] = np.eye(self.model.nv) - np.matmul(d_arr.T, d_arr)
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # Set initial bounds -> x0_pos = q_init, x0_vel free; (final bounds already set)
        q_init_lb = np.hstack([q_init, self.model.x_min[self.model.nq:]])
        q_init_ub = np.hstack([q_init, self.model.x_max[self.model.nq:]])
        self.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        self.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        # Solve the OCP
        return self.ocp_solver.solve()
    
    def solveVBOC(self, q, d, N_start, n=1, repeat=10):
        N = N_start
        gamma = 0
        x_sol, u_sol = None, None
        if n == 0:
            # N-BRS --> constant horizon N, no need to repeat the process until convergence 
            repeat = 1
        for _ in range(repeat):
            # Solve the OCP
            status = self.solve(q, d)
            if status == 0:
                # Compare the current cost with the previous one:
                x0 = self.ocp_solver.get(0, "x")
                gamma_new = np.linalg.norm(x0[self.model.nq:])

                if gamma_new < gamma + self.tol:
                    break
                gamma = gamma_new

                # Rollout the solution
                x_sol = np.empty((N + n, self.model.nx))
                u_sol = np.empty((N + n, self.model.nu))    # last control is not used
                for i in range(N):
                    x_sol[i] = self.ocp_solver.get(i, 'x')
                    u_sol[i] = self.ocp_solver.get(i, 'u')
                x_sol[N:] = self.ocp_solver.get(N, 'x')
                u_sol[N:] = np.zeros((n, self.model.nu))

                # Reset the initial guess with the previous solution
                self.setGuess(x_sol, u_sol)
                # Increase the horizon
                N += n
                self.resetHorizon(N)
            else:
                return None, None, None
        return x_sol, u_sol, N
