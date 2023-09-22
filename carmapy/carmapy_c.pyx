import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, log
@cython.boundscheck(True)
# Define the C++ functions using Cython types
cdef double normal_sigma_fixed_marginal_fun_indi(double zSigmaz_S, double tau, double p_S, double y_sigma) nogil:
    cdef double result = p_S / 2.0 * log(tau / (1.0 + tau)) + (zSigmaz_S / (2 * y_sigma * (1.0 + tau)))
    return result

def Normal_fixed_sigma_marginal(np.ndarray[np.int64_t, ndim=1] index_vec_input,
                                np.ndarray[np.float64_t, ndim=2] Sigma,
                                np.ndarray[np.float64_t, ndim=1] z,
                                double tau,
                                double p_S,
                                double y_sigma):
    cdef np.ndarray[np.int64_t, ndim=1] index_vec = index_vec_input
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S = Sigma[np.ix_(index_vec, index_vec)]
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S_inv = np.linalg.pinv(Sigma_S, rcond=0.00001)
    cdef np.ndarray[np.float64_t, ndim=1] sub_z = z[index_vec]
    cdef double zSigmaz_S = np.dot(sub_z.T, np.dot(Sigma_S_inv, sub_z))

    cdef double b = normal_sigma_fixed_marginal_fun_indi(zSigmaz_S, tau, p_S, y_sigma)

    return b


cdef double ind_normal_sigma_fixed_marginal_fun_indi(double zSigmaz_S, double tau,
                                                    double p_S, double det_S):
    cdef double result = p_S / 2.0 * np.log(tau) - 0.5 * np.log(det_S) + (zSigmaz_S / 2.0)
    return result

cpdef double ind_Normal_fixed_sigma_marginal(np.ndarray[np.int64_t, ndim=1] index_vec_input,
                                             np.ndarray[np.float64_t, ndim=2] Sigma,
                                             np.ndarray[np.float64_t, ndim=1] z,
                                             double tau, int p_S, double y_sigma):
    cdef int p = Sigma.shape[0]
    cdef int n = index_vec_input.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S = np.zeros((p_S, p_S), dtype=np.float64)

    for i in range(p_S):
        for j in range(p_S):
            Sigma_S[i, j] = Sigma[index_vec_input[i], index_vec_input[j]]

    cdef np.ndarray[np.float64_t, ndim=2] A = tau * np.eye(p_S, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S_inv = np.linalg.inv(Sigma_S + A)

    cdef np.ndarray[np.float64_t, ndim=1] sub_z = np.zeros(p_S, dtype=np.float64)
    for i in range(p_S):
        sub_z[i] = z[index_vec_input[i]]

    cdef double zSigmaz_S = np.dot(sub_z.T, np.dot(Sigma_S_inv, sub_z))
    cdef double b = ind_normal_sigma_fixed_marginal_fun_indi(zSigmaz_S, tau, p_S, np.linalg.det(Sigma_S + A))
    cdef double results = b
    return results


cpdef double outlier_Normal_fixed_sigma_marginal(int[:] index_vec_input,
                                                double[:, :] Sigma,
                                                double[:] z,
                                                double tau, double p_S, double y_sigma):
    cdef int S = index_vec_input.shape[0]
    cdef int[:] index_vec = index_vec_input
    cdef double[:, :] Sigma_S = np.zeros((S, S))
    cdef double[:, :] Sigma_S_inv = np.zeros((S, S))
    cdef double[:] sub_z = np.zeros(S)

    for i in range(S):
        for j in range(S):
            Sigma_S[i, j] = Sigma[index_vec[i], index_vec[j]]
    Sigma_S_inv = np.linalg.pinv(Sigma_S, rcond=0.00001)

    for i in range(S):
        sub_z[i] = z[index_vec[i]]

    cdef double det_S = np.linalg.det(Sigma_S)
    det_S = abs(det_S)

    cdef double zSigmaz_S = np.dot(sub_z, np.dot(Sigma_S_inv, sub_z))
    cdef double results = -0.5 * log(det_S) - (tau * zSigmaz_S / (2 * y_sigma * (1.0 + tau)))

    return results


cdef double normal_marginal_fun_indi(double zSigmaz_S, double tau, double p, double zSigmaz, double p_S):
    cdef double result
    if tau <= 0 or zSigmaz <= 0 or zSigmaz_S <= 0 or p_S <= 0:
        result = -np.inf
    else:
        result = p_S / 2.00 * np.log(tau / (1.00 + tau)) - p / 2.00 * np.log(1 - zSigmaz_S / ((1.00 + tau) * zSigmaz))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Normal_marginal(np.ndarray[np.int32_t, ndim=1] index_vec_input,
                            np.ndarray[np.float64_t, ndim=2] Sigma,
                            np.ndarray[np.float64_t, ndim=1] z,
                            double zSigmaz,
                            double tau,
                            double p,
                            double p_S):

    # Convert index_vec_input to numpy array and subtract 1
    cdef np.int32_t[:] index_vec = index_vec_input

    # Perform the rest of the calculations
    cdef np.float64_t[:, :] Sigma_S = Sigma[np.ix_(index_vec, index_vec)]
    cdef np.float64_t[:, :] Sigma_S_inv = np.linalg.pinv(Sigma_S)
    cdef np.float64_t[:] sub_z = z[index_vec]

    cdef double zSigmaz_S = np.dot(np.dot(sub_z, Sigma_S_inv), sub_z)
    cdef double b

    b = normal_marginal_fun_indi(zSigmaz_S, tau, p, zSigmaz, p_S)

    return b

cpdef double outlier_ind_Normal_marginal(np.ndarray[int, ndim=1] index_vec_input,
                                         np.ndarray[np.float64_t, ndim=2] Sigma,
                                         np.ndarray[np.float64_t, ndim=1] z,
                                         double tau, int p_S, double y_sigma):
    cdef int p = Sigma.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S = np.zeros((p_S, p_S), dtype=np.float64)

    # Convert Python integer arrays to C integer arrays and subtract 1 from index_vec_input
    cdef int[:] index_vec = index_vec_input - 1

    # Extract the relevant elements from Sigma
    cdef int i, j
    for i in range(p_S):
        for j in range(p_S):
            Sigma_S[i, j] = Sigma[index_vec[i], index_vec[j]]

    cdef np.ndarray[np.float64_t, ndim=2] A = tau * np.eye(p_S, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Sigma_S_inv = np.linalg.inv(Sigma_S + A)

    cdef np.ndarray[np.float64_t, ndim=1] sub_z = np.zeros(p_S, dtype=np.float64)
    for i in range(p_S):
        sub_z[i] = z[index_vec[i]]

    cdef double zSigmaz_S = np.dot(sub_z, np.dot(Sigma_S_inv, sub_z))
    cdef double det_S = np.linalg.det(Sigma_S + A)

    cdef double b = 0.5 * (np.log(abs(det_S)) + np.log(tau)) - 0.5 * zSigmaz_S
    cdef double results = b
    return results


