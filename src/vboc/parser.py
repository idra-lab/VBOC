import os
import yaml
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='double_pendulum',
                        help='Systems to test. Available: pendulum, double_pendulum, triple_pendulum')
    parser.add_argument('-v', '--vboc', action='store_true',
                        help='Compute data on border of the viability kernel')
    parser.add_argument('--horizon', type=int, default=False, const=100, nargs='?',
                        help='Horizon of the optimal control problem')
    parser.add_argument('-t', '--training', action='store_true',
                        help='Train the neural network model that approximates the viability kernel')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the approximated viability kernel')
    parser.add_argument('-m', '--memmo', action='store_true',
                        help='Learn a policy that drive to equilibrium using VBOC data')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the neural network')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for the neural network')
    parser.add_argument('--beta', type=float, default=0.95,
                        help='Low-pass filter for the loss function')
    return vars(parser.parse_args())


class Parameters:
    def __init__(self, m_name):
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.join(self.PKG_DIR, '../..')
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + m_name + '/')

        model = yaml.load(open(self.CONF_DIR + 'models/' + m_name + '.yaml'), Loader=yaml.FullLoader)
        self.g = float(model['g'])
        self.l1 = float(model['l1'])
        self.l2 = float(model['l2'])
        self.l3 = float(model['l3'])
        self.m1 = float(model['m1'])
        self.m2 = float(model['m2'])
        self.m3 = float(model['m3'])
        self.b = float(model['b'])
        self.q_min = (1 + float(model['q_min'])) * np.pi
        self.q_max = (1 + float(model['q_max'])) * np.pi
        self.dq_min = float(model['dq_min'])
        self.dq_max = float(model['dq_max'])
        self.u_min = float(model['u_min'])
        self.u_max = float(model['u_max'])
        self.state_tol = float(model['state_tol'])

        simulator = yaml.load(open(self.CONF_DIR + 'simulator.yaml'), Loader=yaml.FullLoader)
        self.dt = float(simulator['dt'])
        self.integrator_type = simulator['integrator_type']
        self.num_stages = int(simulator['num_stages'])

        controller = yaml.load(open(self.CONF_DIR + 'controller.yaml'), Loader=yaml.FullLoader)
        self.T = float(controller['T'])
        self.prob_num = int(controller['prob_num'])
        self.test_num = int(controller['test_num'])
        self.n_steps = int(controller['n_steps'])
        self.cpu_num = int(controller['cpu_num'])
        self.regenerate = bool(controller['regenerate'])

        self.solver_type = 'SQP'
        self.solver_mode = controller['solver_mode']
        self.nlp_max_iter = int(controller['nlp_max_iter'])
        self.qp_max_iter = int(controller['qp_max_iter'])
        self.qp_tol_stat = float(controller['qp_tol_stat'])
        self.nlp_tol_stat = float(controller['nlp_tol_stat'])
        self.alpha_reduction = float(controller['alpha_reduction'])
        self.alpha_min = float(controller['alpha_min'])
        self.levenberg_marquardt = float(controller['levenberg_marquardt'])
        self.alpha = int(controller['alpha'])
        self.conv_tol = float(controller['conv_tol'])
        self.cost_tol = float(controller['cost_tol'])
        self.globalization = 'MERIT_BACKTRACKING'
