import cProfile
import numpy as np
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from numpy.linalg import norm as norm
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 10
    output_size = 2
    learning_rate = 0.01

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Active learning parameters:
    N_init = pow(10, ocp_dim)  # size of initial labeled set
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.2  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    # etp_ref = 1e-4

    # # Generate low-discrepancy unlabeled samples:
    # sampler = qmc.Halton(d=ocp_dim, scramble=False)
    # sample = sampler.random(n=pow(50, ocp_dim))
    # l_bounds = [q_min, q_min, v_min, v_min]
    # u_bounds = [q_max, q_max, v_max, v_max]
    # data = qmc.scale(sample, l_bounds, u_bounds)

    # Generate random samples:
    data_q1 = np.random.uniform(q_min, q_max, size=pow(50, ocp_dim))
    data_q2 = np.random.uniform(q_min, q_max, size=pow(50, ocp_dim))
    data_v1 = np.random.uniform(v_min, v_max, size=pow(50, ocp_dim))
    data_v2 = np.random.uniform(v_min, v_max, size=pow(50, ocp_dim))
    data = np.transpose(np.array([data_q1[:], data_q2[:], data_v1[:], data_v2[:]]))

    # Generate the initial set of labeled samples:
    X_iter = [[(q_max + q_min) / 2, (q_max + q_min) / 2, 0.0, 0.0]]
    res = ocp.compute_problem([(q_max + q_min) / 2, (q_max + q_min) / 2], [0.0, 0.0])
    if res == 1:
        y_iter = [[0, 1]]
    else:
        y_iter = [[1, 0]]

    Xu_iter = data  # Unlabeled set
    Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
    mean, std = torch.mean(Xu_iter_tensor), torch.std(Xu_iter_tensor)

    # Training of an initial classifier:
    for n in range(N_init):
        q0 = data[n, :2]
        v0 = data[n, 2:]

        # Data testing:
        X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
        res = ocp.compute_problem(q0, v0)
        if res == 1:
            y_iter = np.append(y_iter, [[0, 1]], axis=0)
        else:
            y_iter = np.append(y_iter, [[1, 0]], axis=0)

        # Add intermediate states of succesfull initial conditions:
        if res == 1:
            for f in range(1, ocp.N, int(ocp.N / 3)):
                current_val = ocp.ocp_solver.get(f, "x")
                X_iter = np.append(X_iter, [current_val], axis=0)
                y_iter = np.append(y_iter, [[0, 1]], axis=0)

    # Delete tested data from the unlabeled set:
    Xu_iter = np.delete(Xu_iter, range(N_init), axis=0)

    X_iter_tensor = torch.from_numpy(X_iter.astype(np.float32))
    y_iter_tensor = torch.from_numpy(y_iter.astype(np.float32))
    X_iter_tensor = (X_iter_tensor - mean) / std

    size = X_iter.shape[0]

    val = 1

    # Train the model
    while val > loss_stop:
        # Forward pass
        outputs = model(X_iter_tensor)
        loss = criterion(outputs, y_iter_tensor)

        # Backward and optimize
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()

        val = loss.item()

    print("INITIAL CLASSIFIER TRAINED")

    print(
        "X_iter:",
        X_iter.shape,
        "Xu_iter:",
        Xu_iter.shape,
    )

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.02
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()
        inp = torch.from_numpy(
            np.c_[
                q_min * np.ones(xrav.shape[0]), xrav, np.zeros(yrav.shape[0]), yrav
            ].astype(np.float32)
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        Xit = [X_iter[0, :]]
        yit = [0]
        for i in range(X_iter.shape[0]):
            if X_iter[i, 0] < q_min + 0.1 and norm(X_iter[i, 2]) < 0.1:
                Xit = np.append(Xit, [X_iter[i, :]], axis=0)
                if np.array_equal(y_iter[i], [1, 0]):
                    yit = np.append(yit, 0)
                else:
                    yit = np.append(yit, 1)
        plt.scatter(
            Xit[1:, 1],
            Xit[1:, 3],
            c=yit[1:],
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("Second actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.c_[
                xrav, q_min * np.ones(xrav.shape[0]), yrav, np.zeros(yrav.shape[0])
            ].astype(np.float32)
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        Xit = [X_iter[0, :]]
        yit = [0]
        for i in range(X_iter.shape[0]):
            if X_iter[i, 1] < q_min + 0.1 and norm(X_iter[i, 3]) < 0.1:
                Xit = np.append(Xit, [X_iter[i, :]], axis=0)
                if np.array_equal(y_iter[i], [1, 0]):
                    yit = np.append(yit, 0)
                else:
                    yit = np.append(yit, 1)
        plt.scatter(
            Xit[1:, 0], Xit[1:, 2], c=yit[1:], marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")
        plt.show()

        # # Delete certain data from the labeled set:
        # sigmoid = nn.Sigmoid()
        # x_prob = sigmoid(X_iter_tensor).numpy()
        # etx = entropy(x_prob, axis=1)
        # etx = [1 if x > etp_ref else x / etp_ref for x in etx]
        # sel = [i for i in range(X_iter.shape[0]) if np.random.uniform() < etx[i]]
        # X_iter = X_iter[sel]
        # y_iter = y_iter[sel]

        # # Delete certain data from the unlabeled set:
        # Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
        # Xu_iter_tensor = (Xu_iter_tensor - mean) / std
        # xu_prob = sigmoid(Xu_iter_tensor).numpy()
        # etxu = entropy(xu_prob, axis=1)
        # etxu = [1 if x > etp_ref else x / etp_ref for x in etxu]
        # sel = [i for i in range(Xu_iter.shape[0]) if np.random.uniform() < etxu[i]]
        # Xu_iter = Xu_iter[sel]

    # Active learning:
    k = 0  # iteration number
    etpmax = 1
    performance_history = [etpmax]

    while not (etpmax < etp_stop or Xu_iter.shape[0] == 0):

        if Xu_iter.shape[0] < B:
            B = Xu_iter.shape[0]

        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
            Xu_iter_tensor = (Xu_iter_tensor - mean) / std
            prob_xu = sigmoid(model(Xu_iter_tensor)).numpy()
            etp = entropy(prob_xu, axis=1)

        maxindex = np.argpartition(etp, -B)[-B:]  # indexes of the uncertain samples

        etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
        performance_history.append(etpmax)

        k += 1

        # Add the B most uncertain samples to the labeled set:
        for x in range(B):
            q0 = Xu_iter[maxindex[x], :2]
            v0 = Xu_iter[maxindex[x], 2:]

            # Data testing:
            X_iter = np.append(X_iter, [[q0[0], q0[1], v0[0], v0[1]]], axis=0)
            res = ocp.compute_problem(q0, v0)
            if res == 1:
                y_iter = np.append(y_iter, [[0, 1]], axis=0)
            else:
                y_iter = np.append(y_iter, [[1, 0]], axis=0)

        # Delete tested data from the unlabeled set:
        Xu_iter = np.delete(Xu_iter, maxindex, axis=0)

        selec = [i for i in range(X_iter.shape[0] - B) if np.random.uniform() <= 1 / k]
        selec.extend([i for i in range(X_iter.shape[0] - B, X_iter.shape[0])])

        print(len(selec))
        print(X_iter.shape[0])

        X_iter_tensor = torch.from_numpy(X_iter[selec].astype(np.float32))
        X_iter_tensor = (X_iter_tensor - mean) / std
        y_iter_tensor = torch.from_numpy(y_iter[selec].astype(np.float32))

        size = X_iter.shape[0]

        val = 1

        # Train the model
        while val > loss_stop:
            # Forward pass
            outputs = model(X_iter_tensor)
            loss = criterion(outputs, y_iter_tensor)

            # Backward and optimize
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()

            val = loss.item()

        print("CLASSIFIER", k, "TRAINED")

        print(
            "X_iter:",
            X_iter.shape,
            "Xu_iter:",
            Xu_iter.shape,
        )

        print("etpmax:", etpmax)

        # with torch.no_grad():
        #     # Delete certain data from the labeled set:
        #     x_prob = sigmoid(X_iter_tensor).numpy()
        #     etx = entropy(x_prob, axis=1)
        #     etx = [1 if x > etp_ref else x / etp_ref for x in etx]
        #     sel = [
        #         i
        #         for i in range(X_iter_tensor.numpy().shape[0])
        #         if np.random.uniform() < etx[i]
        #     ]
        #     X_iter = X_iter[sel]
        #     y_iter = y_iter[sel]

        #     # Delete certain data from the unlabeled set:
        #     Xu_iter_tensor = torch.from_numpy(Xu_iter.astype(np.float32))
        #     Xu_iter_tensor = (Xu_iter_tensor - mean) / std
        #     xu_prob = sigmoid(Xu_iter_tensor).numpy()
        #     etxu = entropy(xu_prob, axis=1)
        #     etxu = [1 if x > etp_ref else x / etp_ref for x in etxu]
        #     sel = [
        #         i
        #         for i in range(Xu_iter_tensor.numpy().shape[0])
        #         if np.random.uniform() < etxu[i]
        #     ]
        #     Xu_iter = Xu_iter[sel]

    with torch.no_grad():
        # Plot the results:
        plt.figure()
        h = 0.02
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        xrav = xx.ravel()
        yrav = yy.ravel()
        inp = torch.from_numpy(
            np.c_[
                q_min * np.ones(xrav.shape[0]), xrav, np.zeros(yrav.shape[0]), yrav
            ].astype(np.float32)
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        Xit = [X_iter[0, :]]
        yit = [0]
        for i in range(X_iter.shape[0]):
            if X_iter[i, 0] < q_min + 0.1 and norm(X_iter[i, 2]) < 0.1:
                Xit = np.append(Xit, [X_iter[i, :]], axis=0)
                if np.array_equal(y_iter[i], [1, 0]):
                    yit = np.append(yit, 0)
                else:
                    yit = np.append(yit, 1)
        plt.scatter(
            Xit[1:, 1],
            Xit[1:, 3],
            c=yit[1:],
            marker=".",
            alpha=0.5,
            cmap=plt.cm.Paired,
        )
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("Second actuator")

        plt.figure()
        inp = torch.from_numpy(
            np.c_[
                xrav, q_min * np.ones(xrav.shape[0]), yrav, np.zeros(yrav.shape[0])
            ].astype(np.float32)
        )
        inp = (inp - mean) / std
        out = model(inp)
        y_pred = np.argmax(out.numpy(), axis=1)
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        Xit = [X_iter[0, :]]
        yit = [0]
        for i in range(X_iter.shape[0]):
            if X_iter[i, 1] < q_min + 0.1 and norm(X_iter[i, 3]) < 0.1:
                Xit = np.append(Xit, [X_iter[i, :]], axis=0)
                if np.array_equal(y_iter[i], [1, 0]):
                    yit = np.append(yit, 0)
                else:
                    yit = np.append(yit, 1)
        plt.scatter(
            Xit[1:, 0], Xit[1:, 2], c=yit[1:], marker=".", alpha=0.5, cmap=plt.cm.Paired
        )
        plt.xlim([q_min, q_max - 0.01])
        plt.ylim([v_min, v_max])
        plt.grid()
        plt.title("First actuator")
        plt.show()

    plt.figure()
    plt.plot(performance_history[1:])
    plt.scatter(range(len(performance_history[1:])), performance_history[1:])
    plt.xlabel("Iteration number")
    plt.ylabel("Maximum entropy")
    plt.title("Maximum entropy evolution")

    print("Execution time: %s seconds" % (time.time() - start_time))

pr.print_stats(sort="cumtime")

plt.show()