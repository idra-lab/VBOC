import numpy as np
from VBOC.pendulum_class_vboc import OCPpendulum
import random
from numpy.linalg import norm as norm
from multiprocessing import Pool

def testing(v):
    # Reset the number of steps used in the OCP:
    N = ocp.N

    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran = random.choice([-1, 1])   
    p = np.array([ran, 0.])

    # Bounds on the initial state:
    q_init = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init, v_min, dt_sym])
    q_init_ub = np.array([q_init, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, 0., dt_sym])
    q_fin_ub = np.array([q_max, 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 3), np.array([q_init, 0., dt_sym]))

    # Reset current iterate:
    ocp.ocp_solver.reset()

    # Set parameters, guesses and constraints:
    for i in range(N):
        ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
        ocp.ocp_solver.set(i, 'p', p)
        ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, v_min, dt_sym])) 
        ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, v_max, dt_sym])) 

    ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
    ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)

    ocp.ocp_solver.constraints_set(N, "lbx", q_fin_lb)
    ocp.ocp_solver.constraints_set(N, "ubx", q_fin_ub)
    ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
    ocp.ocp_solver.set(N, 'p', p)

    # Solve the OCP:
    status = ocp.ocp_solver.solve()

    if status == 0: 
        return ocp.ocp_solver.get(0, "x")[:2]
    else:
        return None

# Ocp initialization:
ocp = OCPpendulum()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin

# Test data generation:
cpu_num = 30
num_prob = 1000

with Pool(cpu_num) as p:
    data = p.map(testing, range(num_prob))

X_test = np.array([i for i in data if i is not None])
np.save('data1_test.npy', X_test)