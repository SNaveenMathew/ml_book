from quadratic_model import x1, y, get_model, x1_extended, y_extended
import numpy as np, matplotlib.pyplot as plt#, pandas as pd
from plot_predictions import plot_pred_matrix
from model_utils import get_formula_rhs, get_summary_df, print_and_subset_summary

unconstrained_wts = np.array([-1.0866661, 0.85507196, -3.102202, -2.7769487, 51.564045, 91.12118, -6.4819527])
# model_u = get_model(bias_constraint = False, learning_rate = 0.1 * np.sqrt(10))
# model_u.set_weights([
#         np.array([[unconstrained_wts[0], unconstrained_wts[1]]], dtype=np.float32),
#         np.array([unconstrained_wts[2], unconstrained_wts[3]], dtype=np.float32),
#         np.array([[unconstrained_wts[4]], [unconstrained_wts[5]]], dtype=np.float32),
#         np.array([unconstrained_wts[6]], dtype=np.float32)
# ])

constrained_wts = np.array([0.18998857, 0.39265335, 0., -1.316958, -559.892, 434.2152, 185.34625])
# model_c = get_model()
# model_c.set_weights([
#         np.array([[constrained_wts[0], constrained_wts[1]]], dtype=np.float32),
#         np.array([constrained_wts[2], constrained_wts[3]], dtype=np.float32),
#         np.array([[constrained_wts[4]], [constrained_wts[5]]], dtype=np.float32),
#         np.array([constrained_wts[6]], dtype=np.float32)
# ])

# Brute force grid of 10 points per weight gives 10^7 points overall to evaluate MSE
# n = len(constrained_wts)
# num_grid_points = 10
# start_wts = np.min(np.vstack([unconstrained_wts, constrained_wts]), axis = 0)
# end_wts = np.max(np.vstack([unconstrained_wts, constrained_wts]), axis = 0)
# diff = (end_wts - start_wts) / (num_grid_points - 1)

# def temp(start_wts, end_wts, diff, i):
#         print(i)
#         return slice(start_wts, end_wts, diff)

# x = [np.linspace(start_wts[i], end_wts[i], 10).tolist() for i in range(len(start_wts))]
# total = 0
# model = get_model(bias_constraint = False, learning_rate = 0.1 * np.sqrt(10))
# for x1_1 in range(10):
#         for x2 in range(10):
#                 for x3 in range(10):
#                         for x4 in range(10):
#                                 for x5 in range(10):
#                                         for x6 in range(10):
#                                                 for x7 in range(10):
#                                                         print(total)
#                                                         model.set_weights([
#                                                                 np.array([[x[0][x1_1], x[1][x2]]], dtype=np.float32),
#                                                                 np.array([x[2][x3], x[3][x4]], dtype=np.float32),
#                                                                 np.array([[x[4][x5]], [x[5][x6]]], dtype=np.float32),
#                                                                 np.array([x[6][x7]], dtype=np.float32)
#                                                         ])
#                                                         preds = model1.predict(x1).reshape(y.shape)
#                                                         res = y - preds
#                                                         mse = np.sum(res**2)
#                                                         total += 1

# Better idea is to choose all combinations of (w_i, w_j) and create a grid between (w_u_i, w_u_j) and (w_c_i, w_c_j)
# There are 7C2 = 21 combinations, for each combination create a grid of 20x20 in (w_i, w_j)
model = get_model(bias_constraint = False, learning_rate = 0.1 * np.sqrt(10))
igrid_points = 50
jgrid_points = 50

def get_mse(x, i, j, w_i, w_j, model, x1, y):
        x[i] = w_i
        x[j] = w_j
        model.set_weights([
                np.array([[x[0], x[1]]], dtype=np.float32),
                np.array([x[2], x[3]], dtype=np.float32),
                np.array([[x[4]], [x[5]]], dtype=np.float32),
                np.array([x[6]], dtype=np.float32)
        ])
        preds = model.predict(x1).reshape(y.shape)
        res = y - preds
        mse = np.mean(res**2)
        return mse

def plot_losses(i_grid, j_grid, losses, i, j, constrained = True, extended = False):
        c_plot = plt.scatter([i_grid[0]], [j_grid[0]], c = "red")
        u_plot = plt.scatter([i_grid[-1]], [j_grid[-1]], c = "blue")
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(i_grid, j_grid)
        # X = X.T
        CS = ax.contour(X, Y, losses)
        ax.clabel(CS, inline=True, fontsize=10)
        min_x = min(i_grid[0], i_grid[-1])
        max_x = max(i_grid[0], i_grid[-1])
        min_y = min(j_grid[0], j_grid[-1])
        max_y = max(j_grid[0], j_grid[-1])
        x_delta = (max_x-min_x)/len(losses[0])
        y_delta = (max_y-min_y)/len(losses[0])
        ax.scatter([i_grid[0], i_grid[-1]], [j_grid[0], j_grid[-1]], c = ["red", "blue"])
        ax.set_xlim((min_x - x_delta, max_x + x_delta))
        ax.set_ylim((min_y - y_delta, max_y + y_delta))
        ax.legend((u_plot, c_plot), ('Unconstrained weights', 'Constrained weights'), scatterpoints = 1, fontsize = 8)
        file = str(i) + ", " + str(j)
        if constrained:
               file = file + "_constrained"
        else:
               file = file + "_unconstrained"
        
        if extended:
               file = file + "_extended"
        else:
               file = file + "_unextended"
        
        fig.savefig(file + ".png")
        return 

total = 0
for i in np.arange(0, len(unconstrained_wts)):
# for i in np.arange(0, 1):
    w_c_i = constrained_wts[i]
    w_u_i = unconstrained_wts[i]
    i_grid = np.linspace(w_c_i, w_u_i, igrid_points)
#     for j in np.arange(i + 1, 2):
    for j in np.arange(i + 1, len(unconstrained_wts)):
        w_c_j = constrained_wts[j]
        w_u_j = unconstrained_wts[j]
        j_grid = np.linspace(w_c_j, w_u_j, jgrid_points)
        c_losses = np.zeros((igrid_points, jgrid_points))
        u_losses = np.zeros((igrid_points, jgrid_points))
        c_losses_ext = np.zeros((igrid_points, jgrid_points))
        u_losses_ext = np.zeros((igrid_points, jgrid_points))
        for i_, w_i in enumerate(i_grid):
                for j_, w_j in enumerate(j_grid):
                        print(total)
                        W_c = constrained_wts.copy()
                        W_u = unconstrained_wts.copy()
                        c_mse = get_mse(W_c, i, j, w_i, w_j, model, x1, y)
                        u_mse = get_mse(W_u, i, j, w_i, w_j, model, x1, y)
                        c_mse_ext = get_mse(W_c, i, j, w_i, w_j, model, x1_extended, y_extended)
                        u_mse_ext = get_mse(W_u, i, j, w_i, w_j, model, x1_extended, y_extended)
                        c_losses[i_, j_] = c_mse
                        u_losses[i_, j_] = u_mse
                        c_losses_ext[i_, j_] = c_mse_ext
                        u_losses_ext[i_, j_] = u_mse_ext
                        total += 1
        
        plot_losses(i_grid, j_grid, c_losses, i, j, constrained = True, extended = False)
        plot_losses(i_grid, j_grid, u_losses, i, j, constrained = False, extended = False)
        plot_losses(i_grid, j_grid, c_losses_ext, i, j, constrained = True, extended = True)
        plot_losses(i_grid, j_grid, u_losses_ext, i, j, constrained = False, extended = True)