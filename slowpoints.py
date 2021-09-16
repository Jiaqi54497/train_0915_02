import pdb
import numpy as np
import scipy
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Reference: https://direct.mit.edu/neco/article/25/3/626/7854/Opening-the-Black-Box-Low-Dimensional-Dynamics-in

# F : callable that defines the time derivative of the dynamics. Must use jax
# x0 : list of initial points to start optimization from
def find_slow_points(F, x0):

    # Jacobian of F
    def jac_F(x):
        return jax.jacfwd(F)(x)

    # Kinetic Energy function
    def q(x):
        return F(x).T @ F(x)

    # Gradient calculated using GradientTape
    def grad_q(x):
        return jax.grad(q)(x)

    def hess_q(x):
        return jax.jacrev(jax.jacfwd(q))(x)

    print(q(x0[0]))
    print(grad_q(x0[0]))
    print(hess_q(x0[0]))

    # # Use newton conjugate gradient algorithm as it makes use of the Hessian
    opt_res = []
    for x_ in tqdm(x0):
        res = minimize(q, x_, method='Newton-CG', jac=grad_q, hess=hess_q)
        opt_res.append(res)

    slow_points = []

    # Verify that the found minima satisfy one of the three criteria given by Barak and Sussilo
    for res in opt_res:
        if len(slow_points) > 0:
            if not np.any([np.allclose(res.x, slow_point, atol=1e-6, rtol=1e-4)
                           for slow_point in slow_points]):
                slow_points.append(res.x)
        else:
            slow_points.append(res.x)


    # For each unique slow point, classify it as either a fixed point, local minimum or saddle of F
    Jeig = []
    for slow_point in slow_points:
        J = jac_F(slow_point)
        eigvals, U = np.linalg.eig(J)
        Jeig.append((eigvals, U))

    return slow_points, Jeig
# F : callable that defines the time derivative of the dynamics. Must accept and return tf.tensor
# x0 : list of initial points to start optimization from
# trajectories: array of system trajectories initialized from the initial conditions
# dim: project onto either dimension 2 or 3 if state space dimension is large than this
def visualize_slow_points(fig, F, x0, trajectories, slow_points=None, J_eig=None, dim=None):

    if slow_points is None:
        slow_points, J_eig = find_slow_points(F, x0)

    # Marker styles
    marker = {'fixed': 'X', 'saddle': '1'}

    if x0[0].size > 3:
        if dim is None:
            dim = 3
    else:
        dim = x0[0].size

    if dim == 2:
        ax = fig.add_axes([0, 1, 0, 1])
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0

        for i, slow_point in enumerate(slow_points):

            if slow_point[0] < xmin:
                xmin = slow_point[0]
            elif slow_point[0] > xmax:
                xmax = slow_point[0]

            if slow_point[1] < ymin:
                ymin = slow_point[1]
            elif slow_point[1] > ymax:
                ymax = slow_point[1]

            if np.all(J_eig[i][0] < 0):
                sp_type = 'fixed'
            else:
                sp_type = 'saddle'
            ax.scatter(slow_point[0], slow_point[1], marker=marker[sp_type])

        ax.set_xlim([xmin - 1e-1, xmax + 1e-1])
        ax.set_ylim([ymin - 1e-1, ymax + 1e-1])

    elif dim == 3:
        pass

    return ax
