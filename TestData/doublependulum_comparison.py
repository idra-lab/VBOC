import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
from VBOC.doublependulum_class_vboc import OCPdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetCLS, NeuralNetDIR
import math

# Ocp initialization:
ocp = OCPdoublependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

# Pytorch device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test = np.load('data2_test.npy')

# VBOC:
model_dir = NeuralNetDIR(4, 300, 1).to(device)
criterion_dir = nn.MSELoss()
model_dir.load_state_dict(torch.load('VBOC/model_2dof_vboc'))
data_reverse = np.load('VBOC/2dof_vboc.npy')
mean_dir = torch.load('VBOC/mean_2dof_vboc')
std_dir = torch.load('VBOC/std_2dof_vboc')

# Active Learning:
model_al = NeuralNetCLS(4, 300, 2).to(device)
model_al.load_state_dict(torch.load('AL/model_2dof_al'))
mean_al = torch.load('AL/mean_2dof_al')
std_al = torch.load('AL/std_2dof_al')
data_al = np.load('AL/data_2dof_al.npy')

# HJ Reachability:
model_hjr = NeuralNetCLS(4, 100, 2).to(device)
model_hjr.load_state_dict(torch.load('HJR/model_2dof_hjr'))
mean_hjr = torch.load('HJR/mean_2dof_hjr')
std_hjr = torch.load('HJR/std_2dof_hjr')

# RMSE evolutions:
times_hjr = np.load('HJR/times_2dof_hjr.npy')
rmse_hjr = np.load('HJR/rmse_2dof_hjr.npy')
times_al = np.load('AL/times_2dof_al.npy')
rmse_al = np.load('AL/rmse_2dof_al.npy')
times_vbocp = np.load('VBOC/times_2dof_vboc.npy')
rmse_vbocp = np.load('VBOC/rmse_2dof_vboc.npy')

plt.figure(figsize=(6, 4))
plt.plot(times_vbocp, rmse_vbocp, label='VBOC')
plt.plot(times_al, rmse_al, label='AL')
plt.plot(times_hjr, rmse_hjr, label='HJR')
plt.title('RMSE evolution')
plt.legend(loc='center right')
plt.ylabel('RMSE (rad/s)')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.grid(True, which="both")

# Compute the prediction errors over the test data:

# VBOCP:
X_test_dir = np.empty((X_test.shape[0],5))
for i in range(X_test_dir.shape[0]):
    X_test_dir[i][0] = (X_test[i][0] - mean_dir) / std_dir
    X_test_dir[i][1] = (X_test[i][1] - mean_dir) / std_dir
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    if vel_norm != 0:
        X_test_dir[i][2] = X_test[i][2] / vel_norm
        X_test_dir[i][3] = X_test[i][3] / vel_norm
    X_test_dir[i][4] = vel_norm 

X_training_dir = np.empty((data_reverse.shape[0],5))
for i in range(X_training_dir.shape[0]):
    X_training_dir[i][0] = (data_reverse[i][0] - mean_dir) / std_dir
    X_training_dir[i][1] = (data_reverse[i][1] - mean_dir) / std_dir
    vel_norm = norm([data_reverse[i][2],data_reverse[i][3]])
    if vel_norm != 0:
        X_training_dir[i][2] = data_reverse[i][2] / vel_norm
        X_training_dir[i][3] = data_reverse[i][3] / vel_norm
    X_training_dir[i][4] = vel_norm 

with torch.no_grad():
    X_iter_tensor = torch.Tensor(X_test_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_test_dir[:,4:]).to(device)
    outputs = model_dir(X_iter_tensor).cpu().numpy()
    # print('RRMSE test data wrt VBOC NN in %: ', math.sqrt(np.sum([((outputs[i] - X_test_dir[i,4])/X_test_dir[i,4])**2 for i in range(len(X_test_dir))])/len(X_test_dir))*100)
    print('RMSE test data wrt VBOC NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item())
    X_iter_tensor = torch.Tensor(X_training_dir[:,:4]).to(device)
    y_iter_tensor = torch.Tensor(X_training_dir[:,4:]).to(device)
    print('RMSE train data wrt VBOC NN: ', torch.sqrt(criterion_dir(model_dir(X_iter_tensor), y_iter_tensor)).item())

# Active Learning:
output_al_test = np.argmax(model_al((torch.Tensor(X_test).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)
norm_error_al = np.empty((len(X_test),))
norm_relerror_al = np.empty((len(X_test),))
absnorm_error_al = np.empty((len(X_test),))
absnorm_relerror_al = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    v0 = X_test[i][2]
    v1 = X_test[i][3]

    if output_al_test[i] == 0:
        out = 0
        while out == 0 and norm([v0,v1]) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model_al((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and norm([v0,v1]) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model_al((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean_al) / std_al).cpu().detach().numpy(), axis=1)

    norm_error_al[i] = vel_norm - norm([v0,v1])
    norm_relerror_al[i] = norm_error_al[i]/vel_norm
    absnorm_error_al[i] = abs(norm_error_al[i])
    absnorm_relerror_al[i] = abs(norm_relerror_al[i])

with torch.no_grad():
    # print('RRMSE test data wrt AL NN in %: ', math.sqrt(np.sum(np.power(norm_relerror_al,2))/len(norm_relerror_al))*100)
    print('RMSE test data wrt AL NN: ', math.sqrt(np.sum(np.power(norm_error_al,2))/len(norm_error_al)))

# HJ Reachability:
output_hjr_test = np.argmax(model_hjr((torch.Tensor(X_test).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)
norm_error_hjr = np.empty((len(X_test),))
norm_relerror_hjr = np.empty((len(X_test),))
absnorm_error_hjr = np.empty((len(X_test),))
absnorm_relerror_hjr = np.empty((len(X_test),))
for i in range(len(X_test)):
    vel_norm = norm([X_test[i][2],X_test[i][3]])
    v0 = X_test[i][2]
    v1 = X_test[i][3]

    if output_hjr_test[i] == 0:
        out = 0
        while out == 0 and norm([v0,v1]) > 1e-2:
            v0 = v0 - 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 - 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model_hjr((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)
    else:
        out = 1
        while out == 1 and norm([v0,v1]) > 1e-2:
            v0 = v0 + 1e-2 * X_test[i][2]/vel_norm
            v1 = v1 + 1e-2 * X_test[i][3]/vel_norm
            out = np.argmax(model_hjr((torch.Tensor([[X_test[i][0], X_test[i][1], v0, v1]]).to(device) - mean_hjr) / std_hjr).cpu().detach().numpy(), axis=1)

    norm_error_hjr[i] = vel_norm - norm([v0,v1])
    norm_relerror_hjr[i] = norm_error_hjr[i]/vel_norm
    absnorm_error_hjr[i] = abs(norm_error_hjr[i])
    absnorm_relerror_hjr[i] = abs(norm_relerror_hjr[i])

with torch.no_grad():
    # print('RRMSE test data wrt HJR NN in %: ', math.sqrt(np.sum(np.power(norm_relerror_hjr,2))/len(norm_relerror_hjr))*100)
    print('RMSE test data wrt HJR NN: ', math.sqrt(np.sum(np.power(norm_error_hjr,2))/len(norm_error_hjr)))

# RMSE comparison plots:
plt.figure(figsize=(6, 4))
bins = np.linspace(0, 1, 100)
plt.hist([abs(X_test_dir[i,4] - outputs[i].tolist()[0]) for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOC', cumulative=True) #,density=True
plt.hist(absnorm_error_al, bins, alpha=0.5, label='AL', cumulative=True)
plt.hist(absnorm_error_hjr, bins, alpha=0.5, label='HJR', cumulative=True)
plt.title('Cumulative error distribution')
plt.legend(loc='lower right')
plt.ylabel('# test samples')
plt.xlabel('Error (rad/s)')

plt.figure(figsize=(6, 4))
bins = np.linspace(-1, 1, 200)
plt.hist([X_test_dir[i,4] - outputs[i].tolist()[0] for i in range(len(X_test_dir))], bins, alpha=0.5, label='VBOC') #,density=True
plt.hist(norm_error_al, bins, alpha=0.5, label='AL')
plt.hist(norm_error_hjr, bins, alpha=0.5, label='HJR')
plt.title('Error distribution')
plt.legend(loc='upper right')
plt.ylabel('# test samples')
plt.xlabel('Error (rad/s)')

# X_dir_al = np.empty((data_al.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_al[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_al[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_al[i][2],data_al[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_al[i][2] / vel_norm
#         X_dir_al[i][3] = data_al[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_al[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data: ', correct/X_dir_al.shape[0])

# data_internal = np.array([data_al[i] for i in range(data_al.shape[0]) if model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[1]>0])

# X_dir_al = np.empty((data_internal.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_internal[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_internal[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_internal[i][2],data_internal[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_internal[i][2] / vel_norm
#         X_dir_al[i][3] = data_internal[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_internal[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL internal data: ', correct/X_dir_al.shape[0])

# data_boundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) < 10])

# X_dir_al = np.empty((data_boundary.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_boundary[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_boundary[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_boundary[i][2],data_boundary[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_boundary[i][2] / vel_norm
#         X_dir_al[i][3] = data_boundary[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_boundary[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data on boundary: ', correct/X_dir_al.shape[0])

# data_notboundary = np.array([data_al[i] for i in range(data_al.shape[0]) if abs(model_al((torch.Tensor(data_al[i,:4]).to(device) - mean_al) / std_al)[0]) > 10])

# X_dir_al = np.empty((data_notboundary.shape[0],6))

# for i in range(X_dir_al.shape[0]):
#     X_dir_al[i][0] = (data_notboundary[i][0] - mean_dir) / std_dir
#     X_dir_al[i][1] = (data_notboundary[i][1] - mean_dir) / std_dir
#     vel_norm = norm([data_notboundary[i][2],data_notboundary[i][3]])
#     if vel_norm != 0:
#         X_dir_al[i][2] = data_notboundary[i][2] / vel_norm
#         X_dir_al[i][3] = data_notboundary[i][3] / vel_norm
#     X_dir_al[i][4] = vel_norm 
#     X_dir_al[i][5] = data_notboundary[i][5]

# correct = 0

# with torch.no_grad():
#     X_iter_tensor = torch.Tensor(X_dir_al[:,:4]).to(device)
#     outputs = model_dir(X_iter_tensor).cpu().numpy()
#     for i in range(X_dir_al.shape[0]):
#         if X_dir_al[i][4] < outputs[i] and X_dir_al[i][5] > 0.5:
#             correct += 1
#         if X_dir_al[i][4] > outputs[i] and X_dir_al[i][5] < 0.5:
#             correct += 1

#     print('Accuracy AL data not on boundary: ', correct/X_dir_al.shape[0])

with torch.no_grad():
    # Plots:
    h = 0.02
    xx, yy = np.meshgrid(np.arange(q_min, q_max+h, h), np.arange(v_min, v_max+h, h))
    xrav = xx.ravel()
    yrav = yy.ravel()

    # Plot the results:
    plt.figure()
    inp = np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][2]) < 0.1
    #     ):
    #         xit.append(X_test[i][1])
    #         yit.append(X_test[i][3])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_2$')
    plt.xlabel('$q_2$')
    plt.grid()
    plt.title("Classifier section")

    plt.figure()
    inp = np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
                np.empty(yrav.shape[0]),
            ]
    for i in range(inp.shape[0]):
        vel_norm = norm([inp[i][2],inp[i][3]])
        inp[i][0] = (inp[i][0] - mean_dir) / std_dir
        inp[i][1] = (inp[i][1] - mean_dir) / std_dir
        if vel_norm != 0:
            inp[i][2] = inp[i][2] / vel_norm
            inp[i][3] = inp[i][3] / vel_norm
        inp[i][4] = vel_norm
    out = (model_dir(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    y_pred = np.empty(out.shape)
    for i in range(len(out)):
        if inp[i][4] > out[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][3]) < 0.1
    #     ):
    #         xit.append(X_test[i][0])
    #         yit.append(X_test[i][2])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.ylabel('$\dot{q}_1$')
    plt.xlabel('$q_1$')
    plt.grid()
    plt.title("Classifier section")

    # plt.figure()
    # inp = np.c_[
    #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #             xrav,
    #             np.zeros(yrav.shape[0]),
    #             yrav,
    #             np.empty(yrav.shape[0]),
    #         ]
    # for i in range(inp.shape[0]):
    #     vel_norm = norm([inp[i][2],inp[i][3]])
    #     inp[i][0] = (inp[i][0] - mean_dir_x0) / std_dir_x0
    #     inp[i][1] = (inp[i][1] - mean_dir_x0) / std_dir_x0
    #     if vel_norm != 0:
    #         inp[i][2] = inp[i][2] / vel_norm
    #         inp[i][3] = inp[i][3] / vel_norm
    #     inp[i][4] = vel_norm
    # out = (model_dir_x0(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    # y_pred = np.empty(out.shape)
    # for i in range(len(out)):
    #     if inp[i][4] > out[i]:
    #         y_pred[i] = 0
    #     else:
    #         y_pred[i] = 1
    # Z = y_pred.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][2]) < 0.1
    #     ):
    #         xit.append(X_test[i][1])
    #         yit.append(X_test[i][3])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    # plt.xlim([q_min, q_max])
    # plt.ylim([v_min, v_max])
    # plt.grid()
    # plt.title("Second actuator, RT x0")

    # plt.figure()
    # inp = np.c_[
    #             xrav,
    #             (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
    #             yrav,
    #             np.zeros(yrav.shape[0]),
    #             np.empty(yrav.shape[0]),
    #         ]
    # for i in range(inp.shape[0]):
    #     vel_norm = norm([inp[i][2],inp[i][3]])
    #     inp[i][0] = (inp[i][0] - mean_dir_x0) / std_dir_x0
    #     inp[i][1] = (inp[i][1] - mean_dir_x0) / std_dir_x0
    #     if vel_norm != 0:
    #         inp[i][2] = inp[i][2] / vel_norm
    #         inp[i][3] = inp[i][3] / vel_norm
    #     inp[i][4] = vel_norm
    # out = (model_dir_x0(torch.from_numpy(inp[:,:4].astype(np.float32)).to(device))).cpu().numpy() 
    # y_pred = np.empty(out.shape)
    # for i in range(len(out)):
    #     if inp[i][4] > out[i]:
    #         y_pred[i] = 0
    #     else:
    #         y_pred[i] = 1
    # Z = y_pred.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # xit = []
    # yit = []
    # for i in range(X_test.shape[0]):
    #     if (
    #         norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
    #         and norm(X_test[i][3]) < 0.1
    #     ):
    #         xit.append(X_test[i][0])
    #         yit.append(X_test[i][2])
    # plt.plot(
    #     xit,
    #     yit,
    #     "ko",
    #     markersize=2
    # )
    # plt.xlim([q_min, q_max])
    # plt.ylim([v_min, v_max])
    # plt.grid()
    # plt.title("First actuator, RT x0")

    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
            ]
        )
    ).to(device)
    inp = (inp - mean_al) / std_al
    out = model_al(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][0] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][2]) < 0.1
        ):
            xit.append(X_test[i][1])
            yit.append(X_test[i][3])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    # xit = []
    # yit = []
    # cit = []
    # for i in range(len(data_al)):
    #     if (
    #         norm(data_al[i][0] - (q_min + q_max) / 2) < 0.1
    #         and norm(data_al[i][2]) < 0.1
    #     ):
    #         xit.append(data_al[i][1])
    #         yit.append(data_al[i][3])
    #         if data_al[i][4] == 1:
    #             cit.append(0)
    #         else:
    #             cit.append(1)
    # plt.scatter(
    #     xit,
    #     yit,
    #     c=cit,
    #     marker=".",
    #     alpha=0.5,
    #     cmap=plt.cm.Paired,
    # )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator, AL")

    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
            ]
        )
    ).to(device)
    inp = (inp - mean_al) / std_al
    out = model_al(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    xit = []
    yit = []
    for i in range(X_test.shape[0]):
        if (
            norm(X_test[i][1] - (q_min + q_max) / 2) < 0.01
            and norm(X_test[i][3]) < 0.1
        ):
            xit.append(X_test[i][0])
            yit.append(X_test[i][2])
    plt.plot(
        xit,
        yit,
        "ko",
        markersize=2
    )
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator, AL")

    # Plot the results:
    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                xrav,
                np.zeros(yrav.shape[0]),
                yrav,
            ]
        )
    ).to(device)
    inp = (inp - mean_hjr) / std_hjr
    out = model_hjr(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("Second actuator, HJR")

    plt.figure()
    inp = torch.from_numpy(
        np.float32(
            np.c_[
                xrav,
                (q_min + q_max) / 2 * np.ones(xrav.shape[0]),
                yrav,
                np.zeros(yrav.shape[0]),
            ]
        )
    ).to(device)
    inp = (inp - mean_hjr) / std_hjr
    out = model_hjr(inp)
    y_pred = np.argmax(out.cpu().numpy(), axis=1)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlim([q_min, q_max])
    plt.ylim([v_min, v_max])
    plt.grid()
    plt.title("First actuator, HJR")

    # # Plots:
    # h = 0.05
    # xx, yy = np.meshgrid(np.arange(v_min, v_max+h, h), np.arange(v_min, v_max+h, h))
    # xrav = xx.ravel()
    # yrav = yy.ravel()

    # for l in range(5):
    #     q1ran = q_min + random.random() * (q_max-q_min)
    #     q2ran = q_min + random.random() * (q_max-q_min)

    #     # Plot the results:
    #     plt.figure()
    #     inp = np.float32(
    #             np.c_[
    #                 q1ran * np.ones(xrav.shape[0]),
    #                 q2ran * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 yrav,
    #                 np.empty(yrav.shape[0]),
    #             ]
    #         )
    #     for i in range(inp.shape[0]):
    #         vel_norm = norm([inp[i][2],inp[i][3]])
    #         inp[i][0] = (inp[i][0] - mean_dir) / std_dir
    #         inp[i][1] = (inp[i][1] - mean_dir) / std_dir
    #         if vel_norm != 0:
    #             inp[i][2] = inp[i][2] / vel_norm
    #             inp[i][3] = inp[i][3] / vel_norm
    #         inp[i][4] = vel_norm
    #     out = (model_dir(torch.Tensor(inp[:,:4]).to(device))).cpu().numpy() 
    #     y_pred = np.empty(out.shape)
    #     for i in range(len(out)):
    #         if inp[i][4] > out[i]:
    #             y_pred[i] = 0
    #         else:
    #             y_pred[i] = 1
    #     Z = y_pred.reshape(yy.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     for i in range(X_test.shape[0]):
    #         if (
    #             norm(X_test[i][0] - q1ran) < 0.01
    #             and norm(X_test[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_test[i][2])
    #             yit.append(X_test[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" RT")

    #     plt.figure()
    #     inp = torch.from_numpy(
    #         np.float32(
    #             np.c_[
    #                 q1ran * np.ones(xrav.shape[0]),
    #                 q2ran * np.ones(xrav.shape[0]),
    #                 xrav,
    #                 yrav, 
    #             ]
    #         )
    #     ).to(device)
    #     inp = (inp - mean_al) / std_al
    #     out = model_al(inp)
    #     y_pred = np.argmax(out.cpu().numpy(), axis=1)
    #     Z = y_pred.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #     xit = []
    #     yit = []
    #     for i in range(X_test.shape[0]):
    #         if (
    #             norm(X_test[i][0] - q1ran) < 0.01
    #             and norm(X_test[i][1] - q2ran) < 0.01
    #         ):
    #             xit.append(X_test[i][2])
    #             yit.append(X_test[i][3])
    #     plt.plot(
    #         xit,
    #         yit,
    #         "ko",
    #         markersize=2
    #     )
    #     plt.xlim([v_min, v_max])
    #     plt.ylim([v_min, v_max])
    #     plt.grid()
    #     plt.title("q1="+str(q1ran)+" q2="+str(q2ran)+" AL")

plt.show()