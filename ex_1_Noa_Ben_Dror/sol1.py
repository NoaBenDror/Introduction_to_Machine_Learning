import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr


#Q6 - Theoretical question
A = np.array([[5,5], [-1,7]])
AtA = A.T.dot(A)
eigvals, eigvecs = np.linalg.eig(AtA)
print(eigvals)
print(eigvecs)
sig = np.array([[np.sqrt(80), 0], [0, np.sqrt(20)]])
print(sig)
sig_inv = np.linalg.inv(sig)
V = -eigvecs[:, [1, 0]]
U = A.dot(V).dot(sig_inv)
print(V)
print(U)



NSAMPLES = 100000
NTOSSES = 1000

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z, title):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()


def plot_2d(x_y, title):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title(title)
    plt.show()


# Q11
plot_3d(x_y_z, 'Q11 - Generation of random points')

# Q12
S = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2.0]])
scaled_x_y_z = S.dot(x_y_z)
plot_3d(scaled_x_y_z, 'Q12 - Dots after scaling')
cov_analytically_q12 = S.dot(cov.dot(S.T))
cov_numerically_q12 = np.cov(scaled_x_y_z)
print('covariance matrix numerically Q12: ')
print(cov_numerically_q12)
print('covariance matrix analytically Q12: ')
print(cov_analytically_q12)

# Q13
random_ortho_mat = get_orthogonal_matrix(3)
mult_x_y_z_by_ortho = random_ortho_mat.dot(scaled_x_y_z)
plot_3d(mult_x_y_z_by_ortho, 'Q13 - Multiplication by random orthogonal matrix')
cov_analytically_q13 = random_ortho_mat.dot(cov_analytically_q12.dot(random_ortho_mat.T))
cov_numerically_q13 = np.cov(mult_x_y_z_by_ortho)
print('covariance matrix numerically Q13: ')
print(cov_numerically_q13)
print('covariance matrix analytically Q13: ')
print(cov_analytically_q13)

# Q14
x_y = mult_x_y_z_by_ortho[:2]
plot_2d(x_y, 'Q14 - Projection of the data to X,Y axes')

# Q15
x_y_condition = \
    x_y[:, (0.1 > mult_x_y_z_by_ortho[2, :]) & (mult_x_y_z_by_ortho[2, :] > -0.4)]
plot_2d(x_y_condition, 'Q15 - Projection of the data to X,Y axes with condition')


# Q16A
eps = 0.25
data = np.random.binomial(1, eps, (NSAMPLES, NTOSSES))
first_5_rows = data[:5]
cum_sum_arr = np.zeros_like(first_5_rows)
for i in range(5):
    cum_sum_arr[i] = np.cumsum(first_5_rows[i])
index_arr = np.arange(1,1001)
averages = cum_sum_arr / index_arr
plt.title('Q16 A - The estimate mean as a function of m')
for i in range(5):
    plt.plot(averages[i], label='row ' + str(i))
    plt.xlabel('m')
    plt.ylabel('Estimate')
    plt.legend()

plt.show()

# Q16B
eps = [0.5, 0.25, 0.1, 0.01, 0.001]
for e in eps:
    plt.plot(np.minimum(np.ones(1000), 1/(4 * (np.arange(1, 1001)) * e**2)), label='Chebyshev')
    plt.plot(np.minimum(np.ones(1000), 2 * np.exp(-2*(np.arange(1, 1001)*e**2))), label='Hoeffding')
    plt.title('Q16 B : epsilon = ' + str(e))
    plt.xlabel('m')
    plt.ylabel('Upper bound')
    plt.legend()
    plt.show()


# Q16C
averages = np.cumsum(data, axis=1) / np.arange(1, 1001)
eps = [0.5, 0.25, 0.1, 0.01, 0.001]
for e in eps:
    plt.plot(np.minimum(np.ones(1000), 1 / (4 * (np.arange(1, 1001)) * e**2)), label='Chebyshev')
    plt.plot(np.minimum(np.ones(1000), 2 * np.exp(-2 * (np.arange(1, 1001) * e**2))), label='Hoeffding')
    deviate = np.mean(np.abs(averages - 0.25) > e, axis=0)
    plt.plot(deviate, label='Percentage')
    plt.title('Q16 C :  epsilon = ' + str(e))
    plt.xlabel('m')
    plt.ylabel('Upper bound')
    plt.legend()
    plt.show()