import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import tensorflow as tf
from ovito.io import import_file

# authors = {Burke Garza (current edits for personal use), Xie, Tian and France-Lanord, Arthur and Wang, Yanming and #Shao-Horn, Yang and Grossman, Jeffrey C.}

# helper function adapted from gdynet/visualization.ipynb
def training_logs(train_dir: str = None):
    """ Retrieve Training Logs
    Retrieve training logs of the loss and VAMP scores for validation and training over the epochs run
    
    Parameters
    ----------
    train_dir : str
        File path to directory containing GDyNet training outputs

    Returns
    -------
    train_logs : pandas dataframe formed from outputs in train_{i}.log files
    """
    os.chdir(train_dir)
    train_logs = []
    for i in range(3):
        try:
            train_logs.append(pd.read_csv('train_{}.log'.format(i)))
        except FileNotFoundError:
            pass
    train_logs = pd.concat(train_logs, ignore_index=True)
    return train_logs

def scatter3d(x,y,z, cs, title, colorsMap='bwr', angle=30):
    """ Scatter3D
        XYZ plotting of eigenstate probability densities with 3D heatmap.
        Adapted from gdynet/visualization.ipynb
        
        Parameters
        ----------
        xyz : ndarrays
            NumPy arrays containing periodic, cartesian trajectory coordinates
        colorsMap : str
            Matplotlib divergent color code for probability heatmap
    """
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=15, c=scalarMap.to_rgba(cs)[:len(x)], marker='o', alpha=0.8)
    scalarMap.set_array(cs)
    ax.view_init(10, angle)
    fig.colorbar(scalarMap)
    plt.title(title)
    plt.show()

# useful class to show what VAMPNets do at the fundamental level, outputs koopman operator for 'plot_timescales' and 'plot_ck_tests'
# from gdynet/vampnet.py, author = {Xie, Tian and France-Lanord, Arthur and Wang, Yanming and Shao-Horn, Yang and Grossman, Jeffrey C.},
class VampnetTools(object):

    '''Wrapper for the functions used for the development of a VAMPnet.

    Parameters
    ----------

    epsilon: float, optional, default = 1e-10
        threshold for eigenvalues to be considered different from zero,
        used to prevent ill-conditioning problems during the inversion of the
        auto-covariance matrices.

    k_eig: int, optional, default = 0
        the number of eigenvalues, or singular values, to be considered while
        calculating the VAMP score. If k_eig is higher than zero, only the top
        k_eig values will be considered, otherwise teh algorithms will use all
        the available singular/eigen values.
    '''

    def __init__(self, epsilon=1e-10, k_eig=0):
        self._epsilon = epsilon
        self._k_eig = k_eig

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def k_eig(self):
        return self._k_eig

    @k_eig.setter
    def k_eig(self, value):
        self._k_eig = value


    def loss_VAMP(self, y_true, y_pred):
        '''Calculates the gradient of the VAMP-1 score calculated with respect
        to the network lobes. Using the shrinkage algorithm to guarantee that
        the auto-covariance matrices are really positive definite and that their
        inverse square-root exists. Can be used as a losseigval_inv_sqrt function
        for a keras model

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape [batch_size, 2 * output_size].
            gradient of the VAMP-1 score
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))
        D,U,V = tf.linalg.svd(vamp_matrix, full_matrices=True)
        diag = tf.diag(D)

        # Base-changed covariance matrices
        x_base = tf.matmul(cov_00_ir, U)
        y_base = tf.matmul(cov_11_ir, V)

        # Calculate the gradients
        nabla_01 = tf.matmul(x_base, y_base, transpose_b=True)
        nabla_00 = -0.5 * tf.matmul(x_base, tf.matmul(diag, x_base,  transpose_b=True))
        nabla_11 = -0.5 * tf.matmul(y_base, tf.matmul(diag, y_base,  transpose_b=True))


        # Derivative for the output of both networks.
        x_der = 2 * tf.matmul(nabla_00, x) + tf.matmul(nabla_01, y)
        y_der = 2 * tf.matmul(nabla_11, y) + tf.matmul(nabla_01, x,  transpose_a=True)

        x_der = 1/(batch_size - 1) * x_der
        y_der = 1/(batch_size - 1) * y_der

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d, y_1d], axis=-1)

        # Stops the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow minimizes the loss-function
        loss_score = - concat_derivatives * y_pred

        return loss_score


    def loss_VAMP2_autograd(self, y_true, y_pred):
        '''Calculates the VAMP-2 score with respect to the network lobes. Same function
        as loss_VAMP2, but the gradient is computed automatically by tensorflow. Added
        after tensorflow 1.5 introduced gradients for eigenvalue decomposition and SVD

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape [batch_size, 2 * output_size].
            gradient of the VAMP-2 score
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the covariance matrices
        cov_01 = 1/(batch_size - 1) * tf.matmul(x, y, transpose_b=True)
        cov_00 = 1/(batch_size - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1/(batch_size - 1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = self._inv(cov_00, ret_sqrt = True)
        cov_11_inv = self._inv(cov_11, ret_sqrt = True)

        vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)


        vamp_score = tf.norm(vamp_matrix)

        return - tf.square(vamp_score)


    def loss_VAMP2(self, y_true, y_pred):
        '''Calculates the gradient of the VAMP-2 score calculated with respect
        to the network lobes. Using the shrinkage algorithm to guarantee that
        the auto-covariance matrices are really positive definite and that their
        inverse square-root exists. Can be used as a loss function for a keras
        model

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape [batch_size, 2 * output_size].
            gradient of the VAMP-2 score
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the covariance matrices
        cov_01 = 1/(batch_size - 1) * tf.matmul(x, y, transpose_b=True)
        cov_10 = 1/(batch_size - 1) * tf.matmul(y, x, transpose_b=True)
        cov_00 = 1/(batch_size - 1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1/(batch_size - 1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = self._inv(cov_00)
        cov_11_inv = self._inv(cov_11)

        # Split the gradient computation in 2 parts for readability
        # These are reported as Eq. 10, 11 in the VAMPnets paper
        left_part_x = tf.matmul(cov_00_inv, tf.matmul(cov_01, cov_11_inv))
        left_part_y = tf.matmul(cov_11_inv, tf.matmul(cov_10, cov_00_inv))

        right_part_x = y - tf.matmul(cov_10, tf.matmul(cov_00_inv, x))
        right_part_y = x - tf.matmul(cov_01, tf.matmul(cov_11_inv, y))

        # Calculate the dot product of the two matrices
        x_der = 2/(batch_size - 1) * tf.matmul(left_part_x, right_part_x)
        y_der = 2/(batch_size - 1) * tf.matmul(left_part_y, right_part_y)

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d,y_1d], axis=-1)

        # Stop the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow maximizes the loss-function
        loss_score =  - concat_derivatives * y_pred

        return loss_score



    def metric_VAMP(self, y_true, y_pred):
        '''Returns the sum of the top k eigenvalues of the vamp matrix, with k
        determined by the wrapper parameter k_eig, and the vamp matrix defined
        as:
            V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
        Can be used as a metric function in model.fit()

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum: tensorflow float
            sum of the k highest eigenvalues in the vamp matrix
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(tf.linalg.svd(vamp_matrix, compute_uv=False))
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Sum the singular values
        eig_sum = tf.cond(cond, lambda: tf.reduce_sum(top_k_val), lambda: tf.reduce_sum(diag))

        return eig_sum


    def metric_VAMP2(self, y_true, y_pred):
        '''Returns the sum of the squared top k eigenvalues of the vamp matrix,
        with k determined by the wrapper parameter k_eig, and the vamp matrix
        defined as:
            V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
        Can be used as a metric function in model.fit()

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum_sq: tensorflow float
            sum of the squared k highest eigenvalues in the vamp matrix
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        matrices = self._build_vamp_matrices(x, y, batch_size)
        cov_00_ir, cov_11_ir, cov_01 = matrices

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_11_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(tf.linalg.svd(vamp_matrix, compute_uv=False))
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Square the singular values and sum them
        pow2_topk = tf.reduce_sum(tf.multiply(top_k_val,top_k_val))
        pow2_diag = tf.reduce_sum(tf.multiply(diag,diag))
        eig_sum_sq = tf.cond(cond, lambda: pow2_topk, lambda: pow2_diag)

        return eig_sum_sq


    def estimate_koopman_op(self, traj, tau):
        '''Estimates the koopman operator for a given trajectory at the lag time
            specified. The formula for the estimation is:
                K = C00 ^ -1 @ C01

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, n_traj, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which the koopman operator is estimated

        Returns
        -------
        koopman_op: numpy array with shape [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        '''
        n_classes = traj.shape[-1]
        prev = traj[:-tau].reshape(-1, n_classes)
        post = traj[tau:].reshape(-1, n_classes)

        c_0 = np.transpose(prev) @ prev
        c_tau = np.transpose(prev) @ post

        eigv, eigvec = np.linalg.eig(c_0)
        include = eigv > self._epsilon
        eigv = eigv[include]
        eigvec = eigvec[:,include]
        c0_inv = eigvec @ np.diag(1/eigv) @ np.transpose(eigvec)

        koopman_op = c0_inv @ c_tau
        return koopman_op


    def get_its(self, traj, lags):
        ''' Implied timescales from a trajectory estimated at a series of lag times.

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, n_traj, traj_dimensions]
            trajectory data

        lags: numpy array with size [lag_times]
            series of lag times at which the implied timescales are estimated

        Returns
        -------
        its: numpy array with size [traj_dimensions - 1, lag_times]
            Implied timescales estimated for the trajectory.

        '''

        its = np.zeros((traj.shape[-1]-1, len(lags)))

        for t, tau_lag in enumerate(lags):
            koopman_op = self.estimate_koopman_op(traj, tau_lag)
            k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            its[:,t] = (-tau_lag / np.log(k_eigvals))

        return its


    def get_ck_test(self, traj, steps, tau):
        ''' Chapman-Kolmogorov test for the koopman operator
        estimated for the given trajectory at the given lag times

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, n_traj, traj_dimensions]
            trajectory data

        steps: int
            how many lag times the ck test will be evaluated at

        tau: int
            shift between consecutive lag times

        Returns
        -------
        predicted: numpy array with size [traj_dimensions, traj_dimensions, steps]
        estimated: numpy array with size [traj_dimensions, traj_dimensions, steps]
            The predicted and estimated transition probabilities at the
            indicated lag times

        '''

        n_states = traj.shape[-1]

        predicted = np.zeros((n_states, n_states, steps))
        estimated = np.zeros((n_states, n_states, steps))

        predicted[:,:,0] =  np.identity(n_states)
        estimated[:,:,0] =  np.identity(n_states)

        for vector, i  in zip(np.identity(n_states), range(n_states)):
            for n in range(1, steps):

                koop = self.estimate_koopman_op(traj, tau)

                koop_pred = np.linalg.matrix_power(koop,n)

                koop_est = self.estimate_koopman_op(traj, tau*n)

                predicted[i,:,n]= vector @ koop_pred
                estimated[i,:,n]= vector @ koop_est


        return [predicted, estimated]


    def estimate_koopman_constrained(self, traj, tau, th=0):
        ''' Calculate the transition matrix that minimizes the norm of the prediction
        error between the trajectory and the tau-shifted trajectory, using the
        estimate of the non-reversible koopman operator as a starting value.
        The constraints impose that all the values in the matrix are positive, and that
        the row sum equals 1. This is achieved using a COBYLA scipy minimizer.

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which the koopman operator is estimated

        th: float, optional, default = 0
            Parameter used to force the elements of the matrix to be higher than 0.
            Useful to prevent elements of the matrix to have small negative value
            due to numerical issues.

        Returns
        -------
        koop_positive: numpy array with shape [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        '''

        koop_init = self.estimate_koopman_op(traj, tau)

        n_states = traj.shape[1]

        rs = lambda k: np.reshape(k, (n_states, n_states))

        def errfun(k):
            diff_matrix = traj[tau:].T - rs(k) @ traj[:-tau].T
            return np.linalg.norm(diff_matrix)

        constr = []

        for n in range(n_states**2):
            # elements > 0
            constr.append({
                'type':'ineq',
                'fun': lambda x, n = n: x.flatten()[n] - th
                })
            # elements < 1
            constr.append({
                'type':'ineq',
                'fun': lambda x, n = n: 1 - x.flatten()[n] - th
                })

        for n in range(n_states):
            # row sum < 1
            constr.append({
                'type':'ineq',
                'fun': lambda x, n = n: 1 - np.sum(x.flatten()[n:n+n_states])
                })
            # row sum > 1
            constr.append({
                'type':'ineq',
                'fun': lambda x, n = n: np.sum(x.flatten()[n:n+n_states]) - 1
                })

        koop_positive = optimize.minimize(
            errfun,
            koop_init,
            constraints = constr,
            method = 'COBYLA',
            tol = 1e-10,
            options = {'disp':False, 'maxiter':1e5},
            ).x

        return koop_positive


    def plot_its(self, its, lag, ylog=False):
        '''Plots the implied timescales calculated by the function
        'get_its'

        Parameters
        ----------
        its: numpy array
            the its array returned by the function get_its
        lag: numpy array
            lag times array used to estimate the implied timescales
        ylog: Boolean, optional, default = False
            if true, the plot will be a logarithmic plot, otherwise it
            will be a semilogy plot

        '''

        if ylog:
            plt.loglog(lag, its.T[:,::-1]);
            plt.loglog(lag, lag, 'k');
            plt.fill_between(lag, lag, 0.99, alpha=0.2, color='k');
        else:
            plt.semilogy(lag, its.T[:,::-1]);
            plt.semilogy(lag, lag, 'k');
            plt.fill_between(lag, lag, 0.99, alpha=0.2, color='k');
        plt.show()


    def plot_ck_test(self, pred, est, n_states, steps, tau):
        '''Plots the result of the Chapman-Kolmogorov test calculated by the function
        'get_ck_test'

        Parameters
        ----------
        pred: numpy array
        est: numpy array
            pred, est are the two arrays returned by the function get_ck_test
        n_states: int
        steps: int
        tau: int
            values used for the Chapman-Kolmogorov test as parameters in the function
            get_ck_test
        '''

        fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True)
        for index_i in range(n_states):
            for index_j in range(n_states):

                ax[index_i][index_j].plot(range(0, steps*tau, tau),
                                          pred[index_i, index_j], color='b')

                ax[index_i][index_j].plot(range(0, steps*tau, tau),
                                          est[index_i, index_j], color='r', linestyle='--')

                ax[index_i][index_j].set_title(str(index_i+1)+ '->' +str(index_j+1),
                                               fontsize='small')

        ax[0][0].set_ylim((-0.1,1.1));
        ax[0][0].set_xlim((0, steps*tau));
        ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps*tau, 3)));
        plt.show()


    def _inv(self, x, ret_sqrt=False):
        '''Utility function that returns the inverse of a matrix, with the
        option to return the square root of the inverse matrix.

        Parameters
        ----------
        x: numpy array with shape [m,m]
            matrix to be inverted

        ret_sqrt: bool, optional, default = False
            if True, the square root of the inverse matrix is returned instead

        Returns
        -------
        x_inv: numpy array with shape [m,m]
            inverse of the original matrix
        '''

        # Calculate eigvalues and eigvectors
        # one more edit here to correct the tf.self_adjoint_eig to linalg.eigh
        eigval_all, eigvec_all = tf.linalg.eigh(x)

        # Filter out eigvalues below threshold and corresponding eigvectors
        eig_th = tf.constant(self.epsilon, dtype=tf.float32)
        index_eig = tf.where(eigval_all > eig_th)
        eigval = tf.gather_nd(eigval_all, index_eig)
        eigvec = tf.gather_nd(tf.transpose(eigvec_all), index_eig)

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        eigval_inv = tf.linalg.diag(1/eigval)
        eigval_inv_sqrt = tf.linalg.diag(tf.sqrt(1/eigval))

        cond_sqrt = tf.convert_to_tensor(ret_sqrt)

        diag = tf.cond(cond_sqrt, lambda: eigval_inv_sqrt, lambda: eigval_inv)

        # Rebuild the square root of the inverse matrix
        x_inv = tf.matmul(tf.transpose(eigvec), tf.matmul(diag, eigvec))

        return x_inv


    def _prep_data(self, data):
        '''Utility function that transorms the input data from a tensorflow -
        viable format to a structure used by the following functions in the
        pipeline.

        Parameters
        ----------
        data: tensorflow tensor with shape [b, 2*o]
            original format of the data

        Returns
        -------
        x: tensorflow tensor with shape [o, b]
            transposed, mean-free data corresponding to the left, lag-free lobe
            of the network

        y: tensorflow tensor with shape [o, b]
            transposed, mean-free data corresponding to the right, lagged lobe
            of the network

        b: tensorflow float32
            batch size of the data

        o: int
            output size of each lobe of the network

        '''

        shape = tf.shape(data)
        # ad-hoc fix recommended in this closed tensorflow issue: https://github.com/google/tangent/issues/95
        # this was necessary since the code is two years old
        b = tf.cast(shape[0], tf.float32)
        o = shape[1]//2

        # Split the data of the two networks and transpose it
        x_biased = tf.transpose(data[:,:o])
        y_biased = tf.transpose(data[:,o:])

        # Subtract the mean
        x = x_biased - tf.reduce_mean(x_biased, axis=1, keepdims=True)
        y = y_biased - tf.reduce_mean(y_biased, axis=1, keepdims=True)

        return x, y, b, o


    def _reshape_data(self, data):
        """
        Utility function that reshapes the input data for calculating VAMP
        losses/metrics. This is introduced because our prediction is with
        shape [b0, n0, 2*o], while ones accepted in VAMPnets are with
        shape [b, 2*o]


        Parameters
        ----------
        data: tensorflow tensor with shape [b0, n0, 2*o]

        Returns
        -------
        reshaped_data: tensorflow tensor with shape [b, 2*o]
        """

        # combine all li in each traj together
        o0 = tf.shape(data)[-1]
        reshaped_data = tf.reshape(data, (-1, o0))
        return reshaped_data



    def _build_vamp_matrices(self, x, y, b):
        '''Utility function that returns the matrices used to compute the VAMP
        scores and their gradients for non-reversible problems.

        Parameters
        ----------
        x: tensorflow tensor with shape [output_size, b]
            output of the left lobe of the network

        y: tensorflow tensor with shape [output_size, b]
            output of the right lobe of the network

        b: tensorflow float32
            batch size of the data

        Returns
        -------
        cov_00_inv_root: numpy array with shape [output_size, output_size]
            square root of the inverse of the auto-covariance matrix of x

        cov_11_inv_root: numpy array with shape [output_size, output_size]
            square root of the inverse of the auto-covariance matrix of y

        cov_01: numpy array with shape [output_size, output_size]
            cross-covariance matrix of x and y

        '''

        # Calculate the cross-covariance
        cov_01 = 1/(b-1) * tf.matmul(x, y, transpose_b=True)
        # Calculate the auto-correations
        cov_00 = 1/(b-1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1/(b-1) * tf.matmul(y, y, transpose_b=True)

        # Calculate the inverse root of the auto-covariance
        cov_00_inv_root = self._inv(cov_00, ret_sqrt=True)
        cov_11_inv_root = self._inv(cov_11, ret_sqrt=True)

        return cov_00_inv_root, cov_11_inv_root, cov_01


    def _build_vamp_matrices_rev(self, x, y, b):
        '''Utility function that returns the matrices used to compute the VAMP
        scores and their gradients for reversible problems. The matrices are
        transformed into symmetrical matrices by calculating the covariances
        using the mean of the auto- and cross-covariances, so that:
            cross_cov = 1/2*(cov_01 + cov_10)
        and:
            auto_cov = 1/2*(cov_00 + cov_11)


        Parameters
        ----------
        x: tensorflow tensor with shape [output_size, b]
            output of the left lobe of the network

        y: tensorflow tensor with shape [output_size, b]
            output of the right lobe of the network

        b: tensorflow float32
            batch size of the data

        Returns
        -------
        auto_cov_inv_root: numpy array with shape [output_size, output_size]
            square root of the inverse of the mean over the auto-covariance
            matrices of x and y

        cross_cov: numpy array with shape [output_size, output_size]
            mean of the cross-covariance matrices of x and y
        '''

        # Calculate the cross-covariances
        cov_01 = 1/(b-1) * tf.matmul(x, y, transpose_b=True)
        cov_10 = 1/(b-1) * tf.matmul(y, x, transpose_b=True)
        cross_cov = 1/2 * (cov_01 + cov_10)
        # Calculate the auto-covariances
        cov_00 = 1/(b-1) * tf.matmul(x, x, transpose_b=True)
        cov_11 = 1/(b-1) * tf.matmul(y, y, transpose_b=True)
        auto_cov = 1/2 * (cov_00 + cov_11)

        # Calculate the inverse root of the auto-covariance
        auto_cov_inv_root = self._inv(auto_cov, ret_sqrt=True)

        return auto_cov_inv_root, cross_cov



    #### EXPERIMENTAL FUNCTIONS ####


    def _loss_VAMP_sym(self, y_true, y_pred):
        '''WORK IN PROGRESS

        Calculates the gradient of the VAMP-1 score calculated with respect
        to the network lobes. Using the shrinkage algorithm to guarantee that
        the auto-covariance matrices are really positive definite and that their
        inverse square-root exists. Can be used as a loss function for a keras
        model. The difference with the main loss_VAMP function is that here the
        matrices C00, C01, C11 are 'mixed' together:

        C00' = C11' = (C00+C11)/2
        C01 = C10 = (C01 + C10)/2

        There is no mathematical reasoning behind this experimental loss function.
        It performs worse than VAMP-2 with regard to the identification of processes,
        but it also helps the network to converge to a transformation that separates
        more neatly the different states

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        loss_score: tensorflow tensor with shape [batch_size, 2 * output_size].
            gradient of the VAMP-1 score
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrix, and the
        # cross-covariance matrix
        cov_00_ir, cov_01  = self._build_vamp_matrices_rev(x, y, batch_size)

        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_00_ir))

        D,U,V = tf.linalg.svd(vamp_matrix, full_matrices=True)
        diag = tf.linalg.diag(D)

        # Base-changed covariance matrices
        x_base = tf.matmul(cov_00_ir, U)
        y_base = tf.matmul(V, cov_00_ir, transpose_a=True)

        # Derivative for the output of both networks.
        nabla_01 = tf.matmul(x_base, y_base)
        nabla_00 = -0.5 * tf.matmul(x_base, tf.matmul(diag, x_base, transpose_b=True))

        # Derivative for the output of both networks.
        x_der = 2/(batch_size - 1) * (tf.matmul(nabla_00, x) + tf.matmul(nabla_01, y))
        y_der = 2/(batch_size - 1) * (tf.matmul(nabla_00, y) + tf.matmul(nabla_01, x))

        # Transpose back as the input y_pred was
        x_1d = tf.transpose(x_der)
        y_1d = tf.transpose(y_der)

        # Concatenate it again
        concat_derivatives = tf.concat([x_1d,y_1d], axis=-1)

        # Stop the gradient calculations of Tensorflow
        concat_derivatives = tf.stop_gradient(concat_derivatives)

        # With a minus because Tensorflow maximizes the loss-function
        loss_score = - concat_derivatives * y_pred

        return loss_score



    def _metric_VAMP_sym(self, y_true, y_pred):
        '''Metric function relative to the _loss_VAMP_sym function.

        Parameters
        ----------
        y_true: tensorflow tensor.
            parameter not needed for the calculation, added to comply with Keras
            rules for loss fuctions format.

        y_pred: tensorflow tensor with shape [batch_size, 2 * output_size]
            output of the two lobes of the network

        Returns
        -------
        eig_sum: tensorflow float
            sum of the k highest eigenvalues in the vamp matrix
        '''

        # reshape data
        y_pred = self._reshape_data(y_pred)

        # Remove the mean from the data
        x, y, batch_size, output_size = self._prep_data(y_pred)

        # Calculate the inverse root of the auto-covariance matrices, and the
        # cross-covariance matrix
        cov_00_ir, cov_01  = self._build_vamp_matrices_rev(x, y, batch_size)

        # Calculate the VAMP matrix
        vamp_matrix = tf.matmul(cov_00_ir, tf.matmul(cov_01, cov_00_ir))

        # Select the K highest singular values of the VAMP matrix
        diag = tf.convert_to_tensor(tf.linalg.svd(vamp_matrix, compute_uv=False))
        cond = tf.greater(self.k_eig, 0)
        top_k_val = tf.nn.top_k(diag, k=self.k_eig)[0]

        # Sum the singular values
        eig_sum = tf.cond(cond, lambda: tf.reduce_sum(top_k_val), lambda: tf.reduce_sum(diag))

        return eig_sum



    def _estimate_koopman_op(self, traj, tau):
        '''Estimates the koopman operator for a given trajectory at the lag time
            specified. The formula for the estimation is:
                K = C00 ^ -1/2 @ C01 @ C11 ^ -1/2

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which the koopman operator is estimated

        Returns
        -------
        koopman_op: numpy array with shape [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        '''

        c_0 = traj[:-tau].T @ traj[:-tau]
        c_1 = traj[tau:].T @ traj[tau:]
        c_tau = traj[:-tau].T @ traj[tau:]

        eigv0, eigvec0 = np.linalg.eig(c_0)
        include0 = eigv0 > self._epsilon
        eigv0_root = np.sqrt(eigv0[include0])
        eigvec0 = eigvec0[:,include0]
        c0_inv_root = eigvec0 @ np.diag(1/eigv0_root) @ eigvec0.T

        eigv1, eigvec1 = np.linalg.eig(c_1)
        include1 = eigv1 > self._epsilon
        eigv1_root = np.sqrt(eigv1[include1])
        eigvec1 = eigvec1[:,include1]
        c1_inv_root = eigvec1 @ np.diag(1/eigv1_root) @ eigvec1.T

        koopman_op = c0_inv_root @ c_tau @ c1_inv_root
        return koopman_op
# from gdynet/postprocess.py, author = {Xie, Tian and France-Lanord, Arthur and Wang, Yanming and Shao-Horn, Yang and Grossman, Jeffrey C.},
def plot_timescales(predictions, lags, n_splits,
                    split_axis=0, time_unit_in_ns=1.):
    """
    Plot timescales implied by the Koopman dynamical model.

    Parameters
    ----------
    predictions: np.array, shape (F, num_atom, n_classes)
        probability of each state
    lags: np.array, shape (N,)
        lag times where timescales are computed
    n_splits: int
        number of splits to divide the trajectory to estimate uncertainty
    split_axis: int
        choose the axis to split the trajectory
    time_unit_in_ns: float
        the unit of each time step (0.1 means 0.1 ns/timestep)
    """
    if split_axis not in [0, 1]:
        raise ValueError('Split can only happen along axis 0 (time) or '
                         '1 (atom)')
    vamp = VampnetTools(epsilon=1e-5)
    splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
    splited_its = np.stack([vamp.get_its(p, lags) for p in splited_preds])

    # time unit conversion
    lags = lags * time_unit_in_ns
    splited_its = splited_its * time_unit_in_ns

    its_log_mean = np.mean(np.log(splited_its), axis=0)
    its_log_std = np.std(np.log(splited_its), axis=0)
    its_mean = np.exp(its_log_mean)
    its_upper = np.exp(its_log_mean + its_log_std * 1.96 /
                       np.sqrt(n_splits))
    its_lower = np.exp(its_log_mean - its_log_std * 1.96 /
                       np.sqrt(n_splits))

    plt.figure(figsize=(4, 3))
    for i in range(its_mean.shape[0]):
        plt.semilogy(lags, its_mean[i])
        plt.fill_between(lags, its_lower[i], its_upper[i], alpha=0.2)
    plt.semilogy(lags, lags, 'k')
    plt.fill_between(lags, lags, lags[0], alpha=0.2, color='k')
    plt.xlabel('Lag time (ns)')
    plt.ylabel('Timescales (ns)')
# from gdynet/postprocess.py
def plot_ck_tests(predictions, tau_msm, steps, n_splits,
                  split_axis=0, time_unit_in_ns=1.):
    """
    Plot CK tests by the Koopman dynamical model.

    Parameters
    ----------
    predictions: np.array, shape (F, num_atom, n_classes)
        probability of each state
    tau_msm: int
        lag time used to build the Koopman model
    steps: int
        number of steps to validate the Koopman model
    n_splits: int
        number of splits to divide the trajectory to estimate uncertainty
    split_axis: int
        choose the axis to split the trajectory
    time_unit_in_ns: float
        the unit of each time step (0.1 means 0.1 ns/timestep)
    """
    if split_axis not in [0, 1]:
        raise ValueError('Split can only happen along axis 0 (time) or '
                         '1 (atom)')
    n_states = predictions.shape[-1]
    vamp = VampnetTools(epsilon=1e-5)
    splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
    splited_predicted, splited_estimated = [], []
    for p in splited_preds:
        predicted, estimated = vamp.get_ck_test(p, steps, tau_msm)
        splited_predicted.append(predicted)
        splited_estimated.append(estimated)
    splited_predicted = np.stack(splited_predicted)
    splited_estimated = np.stack(splited_estimated)

    # time unit conversion
    tau_msm = tau_msm * time_unit_in_ns

    fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True,
                           figsize=(5, 4))
    for i in range(n_states):
        for j in range(n_states):
            pred_mean = splited_predicted[:, i, j].mean(axis=0)
            pred_std = splited_predicted[:, i, j].std(axis=0)
            ax[i][j].plot(np.arange(0, steps * tau_msm, tau_msm),
                          pred_mean, color='b')
            ax[i][j].fill_between(
                np.arange(0, steps * tau_msm, tau_msm),
                pred_mean - pred_std * 1.96 / np.sqrt(n_splits),
                pred_mean + pred_std * 1.96 / np.sqrt(n_splits),
                alpha=0.2, color='b')
            ax[i][j].errorbar(
                np.arange(0, steps * tau_msm, tau_msm),
                splited_estimated[:, i, j].mean(axis=0),
                splited_estimated[:, i, j].std(axis=0) * 1.96 /
                np.sqrt(n_splits),
                color='r', linestyle='--')
            ax[i][j].set_title(str(i) + '->' + str(j))
    ax[0][0].set_ylim((-0.1, 1.1))
    ax[0][0].set_xlim((0, steps * tau_msm))
    ax[0][0].axes.get_xaxis().set_ticks(
        np.linspace(0, steps * tau_msm, 3))
    fig.text(0.5, 0.0, '(ns)', ha='center')
    #fig.tight_layout()

# from gdynet/vampnet.py, author = {Xie, Tian and France-Lanord, Arthur and Wang, Yanming and Shao-Horn, Yang and Grossman, Jeffrey C.},
def estimate_koopman_op(traj, tau):
        '''Estimates the koopman operator for a given trajectory at the lag time
            specified. The formula for the estimation is:
                K = C00 ^ -1 @ C01

        Parameters
        ----------
        traj: numpy array with size [traj_timesteps, n_traj, traj_dimensions]
            Trajectory described by the returned koopman operator

        tau: int
            Time shift at which the koopman operator is estimated

        Returns
        -------
        koopman_op: numpy array with shape [traj_dimensions, traj_dimensions]
            Koopman operator estimated at timeshift tau

        '''
        _epsilon = 1e-10
        n_classes = traj.shape[-1]
        prev = traj[:-tau].reshape(-1, n_classes)
        post = traj[tau:].reshape(-1, n_classes)

        c_0 = np.transpose(prev) @ prev
        c_tau = np.transpose(prev) @ post

        eigv, eigvec = np.linalg.eig(c_0)
        include = eigv > _epsilon
        eigv = eigv[include]
        eigvec = eigvec[:,include]
        c0_inv = eigvec @ np.diag(1/eigv) @ np.transpose(eigvec)

        koopman_op = c0_inv @ c_tau
        return koopman_op

class MDLearn():
    """ Molecular Dynamics Learning Instance for a Single Trajectory File
    Takes an input file to OVITO and converts the coordinates to numpy arrays,
    then to graphs as compressed .npz files with training, validation, and
    testing separation.

        Attributes
        ----------
            filename : str
                Name of OVITO-readable file including extension
            atom1 : str
                Atomic symbol (case-sensitive) represented as Type 1
            atom2 : str
                Atomic symbol (case-sensitive) represented as Type 2

        Methods
        -------
        process_data_splits(train_split = 0.6, test_split = 0.3)
            Split dataset into training, testing, validation portions with validation percent found from remainder implicitly
        compress_arrays(train_name, test_name, val_name)
            Compress arrays into .npz files for smaller data footprint
        shape_preds(train_dir, n_classes, time_unit)
            Assigns inputs to class attributes
        uncertainty(max_tau = 200, n_splits = 2)
            Estimate model uncertainty over extended timescales for application
        
    """

    def __init__(
        self,
        filename: str = None,
        atom1: str = None,
        atom2: str = None,
        dopant: str = None,
        ):
       
        self.filename = filename
        self.atom1 = atom1
        self.atom2 = atom2
        self.dopant = dopant
        # read file with OVITO
        self.pipeline = import_file(filename, sort_particles = True)
        # assume the cell doesn't change from frame to frame
        data = self.pipeline.compute(0)
        self.cell = np.array(data.cell)[:3,:3]
        self.lattices = np.repeat(cell.reshape)
        print('Lattices Array Dimensions:\n')
        print(lattices.shape)
        # access OVITO data like you would in pandas
        self.atom_types = np.array(data.particles['Particle Type'])
        # the gdynet README says to put the atomic number as the type, not the 
        # ad-hoc label I had in LAMMPS
        if self.atom1 == 'Ni' and self.atom2 == 'Cu':
            self.atom_types[np.where(self.atom_types == 1)] = 28
            self.atom_types[np.where(self.atom_types == 2)] = 29
        elif self.atom1 == 'Cu' and self.atom2 == 'Ni':
            self.atom_types[np.where(self.atom_types == 1)] = 29
            self.atom_types[np.where(self.atom_types == 2)] = 28
        
        print('Atom Types Array Dimension:' + self.atom_types.shape + '\n')
        print('Atom Types Data Type' + self.atom_types.dtype + '\n')
        # the movement of Ni dopant is most important and is our target
        self.target_index = np.array(np.where(self.atom_types == 28)).flatten()
        print('Target Index Shape: ' + self.target_index.shape + '\n')
        # bring in the xyz coordinates, reshaping to we can append all the frames together
        traj_coords = np.array(data.particles['Position'])
        self.nparticles = len(traj_coords)
        print('Original shape: ' + str(traj_coords.shape))
        self.traj_coords = traj_coords.reshape((1,nparticles,3))
        print('New shape: ' + str(self.traj_coords.shape))

    def process_data_splits(train_split = 0.6, test_split = 0.3):
        # this loop takes a while (unfortunately OVITO doesn't go much faster)
        for frame in range(1, self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame)
            current_coords = np.array(data.particles['Position'])
            self.traj_coords = np.append(self.traj_coords, current_coords.reshape((1,self.nparticles,3)), axis = 0)

        self.train_frame = int(self.pipeline.source.num_frames*train_split)
        self.test_frame = int(self.pipeline.source.num_frames*(train_split + test_split))
        print('Train frame is ' + self.train_frame + '\n')
        print('Test frame is ' + self.test_frame + '\n')
        # fun with slicing
        self.train_traj_coords = self.traj_coords[:self.train_frame, :, :]
        self.test_traj_coords = self.traj_coords[self.train_frame:self.test_frame, :, :]
        self.val_traj_coords = self.traj_coords[self.test_frame:,:,:]
        # shape of the new training-testing-validation splits
        print('Training coordinates shape: ' + self.train_traj_coords.shape + '\n')
        print('Test coordinates shape: ' + self.test_traj_coords.shape + '\n')
        print('Validation coordinates shape: ' + self.val_traj_coords.shape + '\n')
        return None
    

    def compress_arrays(train_name='train-traj', test_name='test-traj', val_name='val-traj'):
        """ Compress Arrays
        Compression needed for the model with the appropriate array labels ( slice the lattices to have the same number of frames)
        """
        # training data
        np.savez_compressed(train_name, traj_coords=self.train_traj_coords,
        lattices=self.lattices[:self.train_frame, :, :],
        atom_types=self.atom_types, target_index=self.target_index)
        # test data
        np.savez_compressed(test_name, traj_coords=self.test_traj_coords,
        lattices=self.lattices[self.train_frame:self.test_frame, :, :],
        atom_types=self.atom_types, target_index=self.target_index)
        # validation data
        np.savez_compressed(val_name, traj_coords=self.val_traj_coords,
        lattices=self.lattices[self.test_frame:, :, :],
        atom_types=self.atom_types, target_index=self.target_index)
        print('Array compression complete! ')
        return None

    def which():
        print('python preprocess.py train-traj.npz train-graph.npz')
        return None

    def which_command(n = 4):
        # gives the user a command line hint
        print('python ../../main.py --train-flist train-graph.npz --val-flist val-graph.npz --test-flist test-graph.npz --job-dir ./--n-classes '+str(n))
        return None

    ### critical function for post-processing
    def shape_preds(
        train_dir: str = None,
        n_classes: int = 6,
        time_unit: float = None
        )
        """ Reshape predictions 
        Reshape the predicted trajectories for special atoms into an array indexable by time, # of batches, and # of classes

        Parameters
        ----------
            train_dir : str
                directory path to location of model training run
            n_classes : int
                number of classes to distinguish eigenstates
            time_unit : float
                **IMPORTANT** units of the simulation timestep in NANOSECONDS
        """
        self.train_dir = train_dir
        self.n_classes = n_classes
        self.time_unit = time_unit
        preds = np.load(self.train_dir)
        # preds has shape (num_trajs,num_frames, num_atoms, n_classes)
        preds.shape
        # n_clases parameter chosen at the start of training for natom types
        preds = np.transpose(preds, (1,0,2,3))
        F = preds.shape[0]
        # set internal object attribute for this training run and analysis
        self.preds = preds.reshape(F, -1 , self.n_classes)
        # preds has shape (num_frames, n_batches, n_classes)
        print('You set n_classes equal to ' + str(self.n_classes) + '\n')
        return self.preds.shape

    ### all the following are post-processing commands
    def train_loss(train_dir: str = None):
        # no need to create the logs if they already exist
        if not hasattr(self, 'train_logs'): 
            self.train_logs = training_logs(train_dir)
            self.train_dir = train_dir
        plt.rcParams['font.weight'] = 'bold' 
        plt.rcParams['axes.labelweight'] = 'bold'
        self.train_logs.plot.line(y=['loss', 'val_loss'])
        plt.title('Training and Validation Loss for 3-Stage Training', fontweight='bold')
        plt.xlabel('Epochs', fontweight ='bold')
        plt.ylabel('VAMP-2 Loss', fontweight='bold')
        return None
    
    def train_vamp2(train_dir: str = None):
        if not hasattr(self, 'train_logs'):
            self.train_logs = training_logs(train_dir)
            self.train_dir = train_dir
        plt.rcParams['font.weight'] = 'bold' 
        plt.rcParams['axes.labelweight'] = 'bold'
        self.train_logs.plot.line(y=['metric_VAMP2', 'val_metric_VAMP2'])
        plt.title('VAMP-2 Training/Validation Scores', fontweight = 'bold')
        plt.xlabel('Epochs', fontweight='bold')
        plt.ylabel('VAMP-2 Score', fontweight='bold')
        return None

    def train_vamp(train_dir str = None):
        if not hasattr(self, 'train_logs'):
            self.train_logs = training_logs(train_dir)
            self.train_dir = train_dir
        plt.rcParams['font.weight'] = 'bold' 
        plt.rcParams['axes.labelweight'] = 'bold'
        self.train_logs.plot.line(y=['metric_VAMP', 'val_metric_VAMP'])
        plt.title('VAMP Training/Validation Scores', fontweight='bold')
        plt.xlabel('Epochs', fontweight ='bold')
        plt.ylabel('VAMP Score', fontweight='bold')

    ## the following functions require self.preds to be instantiated so
    ## request some user input 
    def repair_preds()
        """ Reshape Predictions For Study If Missed by Accident
        Reshape the predicted trajectories for special atoms into an array indexable by time, # of batches, and # of classes

        See Also
        --------
        shape_preds method above
        """
        return self.preds = self.shape_preds(train_dir = self.train_dir, n_classes = self.n_classes, self.time_unit)
    def pie():
        if not hasattr(self, 'preds'):
            if not hasattr(self, 'train_dir'):
                self.train_dir = Input('Please input the training directory where outputs are:')
            if not hasattr(self, 'n_classes'):
                self.n_classes = Input('Please input the number of classes in the GDyNet model you trained:')
            if not hasattr(self, 'time_unit'):
                self.time_unit = Input('Please input the timestep length in units of nanoseconds:')
            self.preds = self.shape_preds()
        
        probs = np.sum(preds, axis=(0,1))
        probs = probs / np.sum(probs)
        plt.figure()
        plt.pie(probs, labels=labels,autopct='%1.2f%%')
        plt.axis('image')
        plt.title('Population of States' fontweight='bold')
        plt.show()
        return None

    def uncertainty(max_tau : float = 200, n_splits : int = 2):
        if not hasattr(self, 'preds'):
            self.preds = self.repair_preds()
            
        lag = np.arange(1, max_tau, n_splits)
        plot_timescales(self.preds, lag, n_splits=n_splits, split_axis=0,time_unit_in_ns=self.time_unit)
        return None

    def ck_tests(tau_msm : float = 100):
        """ Chapman-Kolmogorov Tests
            I don't really know what this is yet oops 
        """
        if not hasattr(self, 'preds'):
            self.preds = self.repair_preds()
        plot_ck_tests(self.preds, tau_msm=tau_msm, step=10, n_splits=4, split_axis=0, time_unit_in_ns=self.time_unit)
        plt.suptitle('CK Tests for Koopman Model with n_classes = ' + str(self.n_classes))
        return None

    def form_koop_op(tau_msm : float = 100):
        """Estimate Koopman Operator
        Estimate the Koopman operator and its eigenvstuff. Sets multiple attributes for the class instance
        
        Parameters
        ----------
        tau_msm : (float)
            Tau for the Markov model to be formed from the eigenvalues

        Return
        ------
        Plot of the eigenvalues and their transitions
        """
        self.tau_msm = tau_msm
        if not hasattr(self, 'preds'):
            self.preds = self.repair_preds()
        self.koopman_op = estimate_koopman_op(self.preds, self.tau_msm)
        self.eigvals, self.eigvecs = np.linalg.eig(self.koopman_op.T)
        for i, eigval in sorted(enumerate(self.eigvals), key=lambda x: x[1], reverse=True):
        print('Eig {}'.format(i))
        print('Value:', eigval)
        print('Timescale: {} ns'.format(-self.tau_msm / np.log(np.abs(eigval)) * time_unit_in_ns))
        print('Vector:', self.eigvecs[:, i])
        plot_eigvals(self.eigvecs[:, i])
        return None

    def spatialplot():
        for i in range(self.n_classes):
            scatter3d(self.traj_coords[::50, 0, 0], self.traj_coords[::50, 0, 1], self.traj_coords[::50, 0, 2],
                    cs=self.preds.reshape(-1, self.n_classes)[::50, i], title='State' + str(i),
                    angle=18)
        return None