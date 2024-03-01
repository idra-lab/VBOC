import os
import time 
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from multiprocessing import Pool
import vboc.model as models
from vboc.parser import Parameters, parse_args
from vboc.abstract import SimDynamics
from vboc.controller import ViabilityController
from vboc.learning import NeuralNetwork, RegressionNN, plot_viability_kernel


def computeDataOnBorder(N_guess, test_flag=0):
    valid_data = np.empty((0, model.nx))
    controller.resetHorizon(N_guess)

    # Randomize the initial state
    d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])
    q = np.array([
        random.uniform(params.q_min + model.eps, params.q_max - model.eps)
        for _ in range(model.nq)]
    )

    # Set the initial guess
    x_guess = np.zeros((N_guess, model.nx))
    u_guess = np.zeros((N_guess, model.nu))
    x_guess[:, :model.nq] = np.full((N_guess, model.nq), q)

    if not test_flag:
        # Dense sampling near the joint position bounds
        vel_dir = random.choice([-1, 1])
        q_init = params.q_min + model.eps if vel_dir == -1 else params.q_max - model.eps
        q_fin = params.q_max - model.eps if vel_dir == -1 else params.q_min + model.eps
        i = random.choice(range(model.nq))
        d[i] = random.random() * vel_dir
        q[i] = q_init
        x_guess[:, i] = np.linspace(q_init, q_fin, N_guess)

    d /= np.linalg.norm(d)
    controller.setGuess(x_guess, u_guess)

    # Solve the OCP
    x_star, u_star, N = controller.solveVBOC(q, d, N_guess, repeat=50)
    if x_star is None:
        return None
    if test_flag:
        return x_star[0]
    # Add a final node
    x_star = np.vstack([x_star, x_star[-1]])
    # Save the initial state as valid data
    valid_data = np.vstack([valid_data, x_star[0]])

    # Generate unviable sample in the cost direction
    x_tilde = np.full((N + 1, model.nx), None)
    x_tilde[0] = np.copy(x_star[0])
    x_tilde[0, model.nq:] -= model.eps * d

    x_limit = True if model.checkVelocityBounds(x_tilde[0, model.nq:]) else False

    # Iterate along the trajectory to verify the viability of the solution
    for j in range(1, N):
        if x_limit:
            if model.checkStateBounds(x_star[j]):
                x_limit = True
            else:

                if model.checkPositionBounds(x_star[j - 1, :model.nq]):
                    break

                gamma = np.linalg.norm(x_star[j, model.nq:])
                d = - x_star[j, model.nq:]
                d /= np.linalg.norm(d)
                controller.resetHorizon(N - j)
                controller.setGuess(x_star[j:N], u_star[j:N])
                x_new, u_new, _ = controller.solveVBOC(x_star[j, :model.nq], d, N - j, repeat=5)
                if x_new is not None:
                    x0 = controller.ocp_solver.get(0, 'x')
                    gamma_new = np.linalg.norm(x0[model.nq:])
                    if gamma_new > gamma + controller.tol:
                        # Update the optimal trajectory
                        x_star[j:N], u_star[j:N] = x_new[:N - j], u_new[:N - j]

                        # Create unviable state
                        x_tilde[j] = np.copy(x_star[j])
                        x_tilde[j, model.nq:] += model.eps * x_tilde[j, model.nq:] / gamma_new

                        # Check if the unviable state is on bound
                        x_limit = True if model.checkVelocityBounds(x_tilde[j, model.nq:]) else False

                    else:
                        x_limit = False
                        x_tilde[j] = np.copy(x_star[j])
                        x_tilde[j, model.nq:] -= model.eps * d
                else:
                    for k in range(j, N):
                        if model.checkVelocityBounds(x_star[k, model.nq:]):
                            valid_data = np.vstack([valid_data, x_star[k]])
                    break
        else:
            x_tilde[j] = simulator.simulate(x_tilde[j - 1], u_star[j - 1])
            x_limit = True if model.checkStateBounds(x_tilde[j]) else False
        if model.insideStateConstraints(x_star[j]):
            valid_data = np.vstack([valid_data, x_star[j]])
    return valid_data


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    # Define the available systems
    available_systems = {
        'pendulum': 'PendulumModel',
        'double_pendulum': 'DoublePendulumModel',
        'triple_pendulum': 'TriplePendulumModel'
    }
    if args['system'] not in available_systems:
        raise ValueError('System not available. Available: ' + str(list(available_systems.keys())))
    params = Parameters('triple_pendulum')          # contains the parameters for all pendulum models
    model = getattr(models, available_systems[args['system']])(params)
    simulator = SimDynamics(model)
    controller = ViabilityController(simulator)

    # Check if data folder exists, if not create it
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)

    horizon = controller.N

    # DATA GENERATION
    if args['vboc']:
        print('Start data generation')
        with Pool(params.cpu_num) as p:
            # inputs --> (horizon, flag to compute only the initial state)
            data = p.starmap(computeDataOnBorder, [(horizon, 0)] * params.prob_num)

        x_data = [i for i in data if i is not None]
        x_save = np.array([i for f in x_data for i in f])

        solved = len(x_data)
        print('Solved/tot: %.3f' % (solved / params.prob_num))
        print('Saved/tot: %.3f' % (len(x_save) / (solved * horizon)))
        np.save(params.DATA_DIR + str(model.nq) + 'dof_vboc', np.asarray(x_save))

    # TRAINING
    if args['training']:
        # Load the data
        x_train = np.load(params.DATA_DIR + str(model.nq) + 'dof_vboc.npy')
        beta = args['beta']
        batch_size = args['batch_size']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = NeuralNetwork(model.nx, (model.nx - 1) * 100, 1).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=args['learning_rate'])
        regressor = RegressionNN(nn_model, loss_fn, optimizer, beta, batch_size)

        # Compute outputs and inputs
        y_train = np.linalg.norm(x_train[:, model.nq:], axis=1).reshape(len(x_train), 1)
        mean = np.mean(x_train[:, :model.nq])
        std = np.std(x_train[:, :model.nq])
        x_train[:, :model.nq] = (x_train[:, :model.nq] - mean) / std
        x_train[:, model.nq:] /= y_train

        print('Start training\n')
        evolution = regressor.training(x_train, y_train)
        print('Training completed\n')

        print('Evaluate the model')
        _, rmse_train = regressor.testing(x_train, y_train)
        print('RMSE on Training data: %.5f' % rmse_train)

        # Compute the test data
        print('Generation of testing data (only x0*)')
        with Pool(params.cpu_num) as p:
            data = p.starmap(computeDataOnBorder, [(horizon, 1)] * params.test_num)

        x_test = np.array([i for i in data if i is not None])
        y_test = np.linalg.norm(x_test[:, model.nq:], axis=1).reshape(len(x_test), 1)
        x_test[:, :model.nq] = (x_test[:, :model.nq] - mean) / std
        x_test[:, model.nq:] /= y_test
        out_test, rmse_test = regressor.testing(x_test, y_test)
        print('RMSE on Test data: %.5f' % rmse_test)

        # Safety margin
        safety_margin = np.amax(out_test - y_test) / y_test
        print(f'Maximum error wrt test data: {safety_margin:.5f}')

        # Save the model
        torch.save({'mean': mean, 'std': std, 'model': nn_model.state_dict()},
                   params.NN_DIR + 'model_' + str(model.nq) + 'dof.pt')
        
        # Save relevant data
        with(open('new_' + str(model.nq) + 'dof_vboc.pkl', 'wb')) as f:
            pickle.dump({'training_evol': evolution,
                         'outputs': out_test, 
                         'y_test': y_test,
                         'alpha': safety_margin}, f)

        print('\nPlot the loss evolution')
        # Plot the loss evolution
        plt.figure()
        plt.plot(evolution)
        plt.show()

    # PLOT THE VIABILITY KERNEL
    if args['plot']:
        nn_data = torch.load(params.NN_DIR + 'model_' + str(model.nq) + 'dof.pt')
        nn_model = NeuralNetwork(model.nx, (model.nx - 1) * 100, 1)
        nn_model.load_state_dict(nn_data['model'])
        plot_viability_kernel(model.nq, params, nn_model, nn_data['mean'], nn_data['std'])
        plt.show()
