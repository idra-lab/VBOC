import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
from VBOC.triplependulum_class_vboc import OCPtriplependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetCLS, NeuralNetDIR
import math
from torch.utils.data import DataLoader

# Ocp initialization:
ocp = OCPtriplependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

n_minibatch_model = pow(2,15)

# Pytorch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test = np.load('data3_test.npy')

# VBOC:
model_dir = NeuralNetDIR(6, 500, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('VBOC/model_3dof_vboc'))
data_reverse = np.load('VBOC/data_3dof_vboc.npy')
mean_dir = torch.load('VBOC/mean_3dof_vboc')
std_dir = torch.load('VBOC/std_3dof_vboc')

# Active Learning:
model_al = NeuralNetCLS(6, 500, 2).to(device)
model_al.load_state_dict(torch.load('AL/model_3dof_al'))
mean_al = torch.load('AL/mean_3dof_al')
std_al = torch.load('AL/std_3dof_al')
data_al = np.load('AL/data_3dof_al.npy')

# Hamilton-Jacoby reachability:
model_hjr = NeuralNetCLS(6, 100, 2).to(device)
model_hjr.load_state_dict(torch.load('HJR/model_3dof_hjr'))
mean_hjr = torch.load('HJR/mean_3dof_hjr')
std_hjr = torch.load('HJR/std_3dof_hjr')

# RMSE evolutions:
times_al = np.load('AL/times_3dof_al.npy')
rmse_al = np.load('AL/rmse_3dof_al.npy')
times_vboc = np.load('VBOC/times_3dof_vboc.npy')
rmse_vboc = np.load('VBOC/rmse_3dof_vboc.npy')
times_hjr = np.load('HJR/times_3dof_hjr.npy')
rmse_hjr = np.load('HJR/rmse_3dof_hjr.npy')

plt.figure(figsize=(6, 4))
plt.plot(times_vboc, rmse_vboc, label='VBOC') #[:-8:2]
plt.plot(times_al, rmse_al, label='AL') #[:-28:4]
plt.plot(times_hjr, rmse_hjr, label='HJR') #[:-1]
plt.title('RMSE evolution')
plt.legend(loc='center right')
plt.ylabel('RMSE (rad/s)')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.grid(True, which="both")

print('RMSE test data wrt VBOCP NN: ', rmse_vboc[-1])
print('RMSE test data wrt AL NN: ', rmse_al[-1])
print('RMSE test data wrt HJR NN: ', rmse_hjr[-2])

# Compute the prediction errors over the training data data:

X_training_dir = np.empty((data_reverse.shape[0],7))
for i in range(X_training_dir.shape[0]):
    X_training_dir[i][0] = (data_reverse[i][0] - mean_dir) / std_dir
    X_training_dir[i][1] = (data_reverse[i][1] - mean_dir) / std_dir
    X_training_dir[i][2] = (data_reverse[i][2] - mean_dir) / std_dir
    vel_norm = norm([data_reverse[i][4],data_reverse[i][3],data_reverse[i][5]])
    if vel_norm != 0:
        X_training_dir[i][5] = data_reverse[i][5] / vel_norm
        X_training_dir[i][4] = data_reverse[i][4] / vel_norm
        X_training_dir[i][3] = data_reverse[i][3] / vel_norm
    X_training_dir[i][6] = vel_norm 

with torch.no_grad():
    
    X_iter_tensor = torch.Tensor(X_training_dir[:,:6]).to(device)
    out = np.empty((len(X_training_dir),1))
    my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
    for (idx, batch) in enumerate(my_dataloader):
        if n_minibatch_model*(idx+1) > len(X_training_dir):
            out[n_minibatch_model*idx:len(X_training_dir)] = model_dir(batch).cpu()
        else:
            out[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = model_dir(batch).cpu()

    print('RMSE train data wrt VBOCP NN: ', math.sqrt(np.sum([(out[i] - X_training_dir[i,6])**2 for i in range(len(X_training_dir))])/len(X_training_dir)))

plt.show()