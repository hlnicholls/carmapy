import sys
sys.path.append('/Users/hn9/Documents/GitHub/carmapy/carmapy')
import carma_normal_fixed_sigma 
import carmapy.carmapy_c
import numpy as np
import pandas as pd
import time
import os
from itertools import combinations
from scipy import sparse, optimize
from scipy.io import mmwrite
from scipy.special import gammaln, betaln
from math import log
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, PoissonRegressor
from sklearn.model_selection import GridSearchCV

sumstats = pd.read_csv('/Users/hn9/Documents/GitHub/carmapy/tests/APOE_locus_sumstats.txt.gz', sep='\t')
ld = pd.read_csv('/Users/hn9/Documents/GitHub/carmapy/tests/APOE_locus_ld.txt.gz', sep='\t')
z_list = sumstats['Z'].tolist()
ld_list = np.asmatrix(ld)

effect_size_prior = 'Normal'

np.random.seed(1)
z_array = np.array(z_list, dtype=np.float64)
n = len(z_list)
p = len(z_list)
beta = np.zeros(p)
beta[0] = 1
lambda_list = []
lambda_list.append(1 / np.sqrt(p))

output_labels='.'
label_list=None
rho_index=0.99
BF_index=10
EM_dist='Logistic'
Max_Model_Dim=2e+5
all_iter=3
all_inner_iter=10
input_alpha=0
epsilon_threshold=1e-4
num_causal=10
y_var=1
tau=0.04
outlier_switch=True
outlier_BF_index=1/3.2

# Setup initial variables in CARMA_fixed_sigma for testing inner functions
np.seterr(divide='ignore', invalid='ignore')
L = len(z_list)
w_list=None
log_2pi = np.log(2 * np.pi)
p_list = []
for i in range(L):
    z_list[i] = np.asmatrix(z_list[i])
    p_list.append(z_list[i].shape[0])

B = Max_Model_Dim
all_B_list = [[np.zeros(0, dtype=int), csr_matrix((0, p_list[i]), dtype=int)] for i in range(L)]
q_list = []
if w_list is not None:
    for i in range(L):
        q_list.append(w_list[i].shape[1])
        invariant_var_index = np.where(np.std(w_list[i][:, 1:], axis=0) == 0)[0]
        if len(invariant_var_index) != 0:
            invariant_var = w_list[i][:, invariant_var_index + 1]
            scaler = StandardScaler()
            w_list[i] = np.c_[1, scaler.fit_transform(w_list[i][:, 1:])]
            w_list[i][:, invariant_var_index + 1] = invariant_var
        else:
            scaler = StandardScaler()
            w_list[i] = np.c_[1, scaler.fit_transform(w_list[i][:, 1:])]

if label_list is None:
    label_list = [f'locus_{i}' for i in range(1, L + 1)]

Sigma_list = ld_list
S_list = [np.zeros(0, dtype=int) for i in range(L)]
all_C_list = [[np.zeros(0, dtype=int), csr_matrix((0, p_list[i]), dtype=int)] for i in range(L)]
all_epsilon_threshold = 0
epsilon_list = [epsilon_threshold * p_list[i] for i in range(L)]
all_epsilon_threshold = sum(epsilon_list)
model_prior = 'Poisson'
standardize_model_space = True

def module_cauchy_shotgun(z, ld_matrix, Max_Model_Dim=1e+4, input_S=None, lambda_val=(1 / np.sqrt(len(z_array))), label=None,
                        num_causal=10, output_labels=None, y_var=1, effect_size_prior=None, model_prior=None,
                        outlier_switch=None, input_conditional_S_list=None, tau=1/0.05**2, C_list=None, prior_prob=None,
                        epsilon=1e-3, inner_all_iter=10, condition_index=None):
    # The prior distributions on the model space
    prob_list = []
    p = len(z)
    if model_prior == 'input.prob':
        posi_log_pro = np.log(prior_prob)
        nega_log_pro = np.log(1 - prior_prob)

        def input_prior_dist(x):
            variable_index = np.where(x == 1)[0]
            other_index = np.setdiff1d(np.arange(1, p + 1) - 1, variable_index)
            if len(variable_index) > 0:
                return np.sum(posi_log_pro[variable_index]) + np.sum(nega_log_pro[other_index]) - np.sum(nega_log_pro)
            else:
                return np.sum(nega_log_pro)
        prior_dist = input_prior_dist

    if model_prior == 'Poisson':
        def poisson_prior_dist(t):
            dim_model = np.sum(t).item()
            result = dim_model * log(lambda_val) + gammaln(p - dim_model + 1) - gammaln(p + 1)
            return result
        prior_dist = poisson_prior_dist

    if model_prior == 'beta-binomial':
        def beta_binomial_dist(t):
            dim_model = np.sum(t)
            result = betaln(dim_model + 1, p - dim_model + 9) - betaln(1, p + 9)
            return result
        prior_dist = beta_binomial_dist

# The marginal likelihood defined by the prior distribution of the effect size
# Calling on the cython c++ functions
    if effect_size_prior == 'Cauchy':
        marginal_likelihood = carmapy.carmapy_c.Cauchy_fixed_sigma_marginal
        tau_sample = np.random.gamma(0.5, scale=2, size=int(1e+5))
        if outlier_switch:
            outlier_likelihood = carmapy.carmapy_c.outlier_Cauchy_fixed_sigma_marginal
            outlier_tau = tau_sample

    if effect_size_prior == 'Hyper-g':
        marginal_likelihood = carmapy.carmapy_c.hyper_g_fixed_sigma_marginal
        tau_sample = np.random.gamma(0.5, scale=2, size=int(1e+5))

    if effect_size_prior == 'Normal':
        marginal_likelihood = carmapy.carmapy_c.Normal_fixed_sigma_marginal
        tau_sample = tau
        if outlier_switch:
            outlier_likelihood = carmapy.carmapy_c.outlier_Normal_fixed_sigma_marginal
            outlier_tau = tau_sample

    if effect_size_prior == 'Spike-slab':
        marginal_likelihood = carmapy.carmapy_c.ind_Normal_fixed_sigma_marginal
        tau_sample = tau
        if outlier_switch:
            outlier_likelihood = carmapy.carmapy_c.outlier_ind_Normal_marginal
            outlier_tau = tau_sample

# Feature learning for the fine-mapping step, such as learning the visited model space from the previous iterations
    p = len(z_array)
    log_2pi = np.log(2 * np.pi)
    B = Max_Model_Dim
    stored_result_prob = np.zeros(p)
    stored_bf = 0
    Sigma = ld_matrix.copy()

    if input_S is not None:
        S = input_S.copy()
    else:
        S = []

    conditional_S = None
    #null_model = sparse.csr_matrix(np.zeros(p))
    null_model = np.zeros(p)
    null_margin = prior_dist(null_model)
    if C_list is None:
        C_list = [[], []]

        B_list = [prior_dist(null_model), sparse.csr_matrix(np.zeros(p))]

        if input_conditional_S_list is None:
            conditional_S_list = []
            conditional_S = None
        else:
            conditional_S_list = input_conditional_S_list.copy()
            conditional_S = input_conditional_S_list['Index'].copy()
            conditional_S = np.unique(conditional_S)
            S = conditional_S

    # Define neighborhood model space
    def set_gamma_func(input_S, condition_index=None, p=None):
        def set_gamma_func_base(S):
            def add_function(y):
                results = [np.sort([x] + y) for x in S_sub]
                return np.array(results)

            #S_sub = [i for i in range(1, p + 1) if i not in S]
            set_gamma = [[], [], []]

            if len(S) == 0:
                print('S is 0')
                S_sub = [i for i in range(1, p + 1)]
                set_gamma[1] = [S + [x] for x in S_sub]
            elif len(S) == 1:
                S_sub = [i for i in range(1, p + 1) if i not in S]
                set_gamma[0] = [list(comb) for comb in combinations(S, len(S) - 1)]
                set_gamma[1] = [sorted([x] + S) for x in S_sub]
                set_gamma[2] = add_function(set_gamma[0][0]).reshape(1, -1)
            else:
                S_sub = [i for i in range(1, p + 1) if i not in S]
                set_gamma[0] = [sorted(list(comb)) for comb in combinations(S, len(S) - 1)] if len(S) > 2 else [list(comb) for comb in combinations(S, len(S) - 1)]
                set_gamma[1] = [sorted([x] + S) for x in S_sub]
                set_gamma[2] = add_function(set_gamma[0][0])
                for i in range(1, len(set_gamma[0])):
                    set_gamma[2] = np.vstack((set_gamma[2], add_function(set_gamma[0][i])))
            return set_gamma

        def set_gamma_func_conditional(input_S, condition_index):
            def add_function(y):
                results = [np.sort([x] + y) for x in S_sub]
                return np.array(results)

            S = [i for i in input_S if i != condition_index]
            S_sub = [i for i in range(1, p + 1) if i not in input_S]
            set_gamma = [[], [], []]

            if len(S) == 0:
                set_gamma[1] = [S + [x] for x in S_sub]
            elif len(S) == 1:
                set_gamma[0] = [list(comb) for comb in combinations(S, len(S) - 1)]
                set_gamma[1] = [sorted([x] + S) for x in S_sub]
                set_gamma[2] = add_function(set_gamma[0][0]).reshape(1, -1)
            else:
                set_gamma[0] = [sorted(list(comb)) for comb in combinations(S, len(S) - 1)] if len(S) > 2 else [list(comb) for comb in combinations(S, len(S) - 1)]
                set_gamma[1] = [sorted([x] + S) for x in S_sub]
                set_gamma[2] = add_function(set_gamma[0][0])
                for i in range(1, len(set_gamma[0])):
                    set_gamma[2] = np.vstack((set_gamma[2], add_function(set_gamma[0][i])))
            return set_gamma

        if condition_index is None:
            results = set_gamma_func_base(input_S)
            for i in range(len(results)):
                print('set_gamma[i] length', len(results[i]))
        else:
            results = set_gamma_func_conditional(input_S, condition_index)

        return results

    def duplicated_dgCMatrix(dgCMat, MARGIN):
        MARGIN = int(MARGIN)
        n = dgCMat.shape[0]
        p = dgCMat.shape[1]
        J = np.repeat(np.arange(1, p + 1), np.diff(dgCMat.indptr))
        I = dgCMat.indices + 1
        x = dgCMat.data
        if MARGIN == 1:
            names_x = {J[i]: x[i] for i in range(len(J))}
            RowLst = np.split(x, I)
            is_empty = np.setdiff1d(np.arange(1, n + 1), I)
            result = np.array([np.any(np.concatenate(RowLst[i])) for i in range(len(RowLst))])
        elif MARGIN == 2:
            names_x = {I[i]: x[i] for i in range(len(I))}
            ColLst = np.split(x, J)
            is_empty = np.setdiff1d(np.arange(1, p + 1), J)
            result = np.array([np.any(np.concatenate(ColLst[i])) for i in range(len(ColLst))])
        else:
            print("Invalid MARGIN; returning None")
            result = None
        if np.any(is_empty):
            out = np.zeros(n) if MARGIN == 1 else np.zeros(p)
            out[~is_empty] = result
            if len(is_empty) > 1:
                out[is_empty[1:]] = True
            result = out
        return result

    def match_dgCMatrix(dgCMat1, dgCMat2):
        n1, p1 = dgCMat1.shape
        J1 = np.repeat(np.arange(p1), np.diff(dgCMat1.indptr))
        I1 = dgCMat1.indices
        x1 = dgCMat1.data
        n2, p2 = dgCMat2.shape
        J2 = np.repeat(np.arange(p2), np.diff(dgCMat2.indptr))
        I2 = dgCMat2.indices
        x2 = dgCMat2.data
        RowLst1 = [set(J1[I1 == i]) for i in range(n1)]
        is_empty1 = set(range(n1)) - set(I1)
        RowLst2 = [set(J2[I2 == i]) for i in range(n2)]
        is_empty2 = set(range(n2)) - set(I2)
        result = [RowLst1.index(row) + 1 if row in RowLst1 else None for row in RowLst2]
        for i in sorted(list(is_empty1), reverse=True):
            result.insert(i, i + 1)
        return result

# Compute posterior inclusion probability based on the marginal likelihood and model space
    def PIP_func(likeli, model_space):
        infi_index = np.where(np.isinf(likeli))[0]
        if len(infi_index) != 0:
            likeli = np.delete(likeli, infi_index)
            model_space = np.delete(model_space, infi_index, axis=0)
        na_index = np.where(np.isnan(likeli))[0]
        if len(na_index) != 0:
            likeli = np.delete(likeli, na_index)
            model_space = np.delete(model_space, na_index, axis=0)
        aa = likeli - np.nanmax(likeli)  # Using np.nanmax to ignore NaN values
        prob_sum = np.sum(np.exp(aa))
        p = model_space.shape[1]  # Number of columns in model_space
        result_prob = np.full(p, np.nan)
        for i in range(p):
            result_prob[i] = np.sum(np.exp(aa[np.where(model_space[:, i] == 1)])) / prob_sum
        return result_prob

    def index_fun_inner(x, p=p):
        n = x.shape[0]
        row_indices = np.repeat(np.arange(n), x.size // n)
        col_indices = x.T.flatten().astype(int) - 1
        data = np.ones(n * x.shape[1], dtype=int)
        m = csc_matrix((data, (row_indices, col_indices)), shape=(n, p))
        return m

    def index_fun(outer_x, max_model_dimins=10):
        outer_x = np.array(outer_x)
        print('outer_x', outer_x.shape[0])
        if outer_x.shape[0] > 1000:
            index_bins = np.where(np.arange(1, outer_x.shape[0] + 1) % (outer_x.shape[0] // max_model_dimins) == 0)[0]
            result_m = index_fun_inner(outer_x[:index_bins[0], :])
            for b in range(len(index_bins) - 1):
                result_m = vstack([result_m, index_fun_inner(outer_x[index_bins[b]:index_bins[b + 1], :])])
            if index_bins[-1] != outer_x.shape[0] - 1:
                result_m = vstack([result_m, index_fun_inner(outer_x[index_bins[-1] + 1:, :])])
        else:
            result_m = index_fun_inner(outer_x)
        return result_m

    def ridge_fun(x, modi_ld_S, test_S_indices, temp_Sigma, z, outlier_tau, outlier_likelihood):
        temp_ld_S = x * modi_ld_S + (1 - x) * np.eye(modi_ld_S.shape[0])
        temp_Sigma[test_S_indices[:, None], test_S_indices] = temp_ld_S
        return outlier_likelihood(test_S_indices, temp_Sigma, z, outlier_tau, len(test_S_indices), 1)

    for l in range(1, inner_all_iter + 1):
        for h in range(1, 11):
            # Shotgun COMPUTATION
            set_gamma = set_gamma_func(S, conditional_S, p=p)
            if conditional_S is None:
                print('conditional_S is None')
                working_S = S
                print('working_S:', working_S)
                base_model = null_model
                base_model_margin = null_margin
            else:
                working_S = S[~np.in1d(S, conditional_S)]
                if len(working_S) != 0:
                    base_model = csr_matrix((np.ones(len(working_S)), ([0] * len(working_S), working_S - 1)), shape=(1, p))
                    p_S = len(working_S)
                    base_model_margin = marginal_likelihood(working_S, Sigma, z, tau_sample, p_S, y_var) + prior_dist(base_model)
                else:
                    base_model = null_model
                    base_model_margin = null_margin

            set_gamma_margin = []
            set_gamma_prior = []
            matrix_gamma = [[], [], []]

            if len(working_S) != 0:
                print('working_S != 0')
                S_model = csr_matrix(([1], ([0], [working_S[0] - 1])), shape=(1, p))
                p_S = len(working_S)
                working_S = np.concatenate([np.array(sublist).flatten() for sublist in S])
                working_S = np.unique(working_S - 1).astype(np.uint32)
                #z = np.array(sumstats['Z'].tolist(), dtype=np.float64)
                Sigma = np.array(ld_list, dtype=np.float64)
                current_log_margin = marginal_likelihood(working_S, Sigma, z, tau_sample, p_S, y_var) + prior_dist(S_model)
            else:
                current_log_margin = prior_dist(null_model)

            if len(working_S) > 1:
                print('working_S > 1')
                for i in range(len(set_gamma)):
                    t0 = time.time()
                    matrix_gamma.append(index_fun(set_gamma[i]))
                    #creating dense_gamma_matrix to avoid errors caused by csr matrices
                    dense_gamma_matrix = matrix_gamma[-1].toarray()
                    col_num = len(set_gamma[i])
                    if len(C_list[1]) <= col_num:
                        print('len(C_list[1]) <= col_num')
                        C_list[1].append(csr_matrix(([], ([], [])), shape=(0, p)))
                        C_list[0].append([])
                        computed_index = []
                    else:
                        print('len(C_list[1]) > col_num')
                        computed_index = match_dgCMatrix(C_list[1][col_num], matrix_gamma[i])

                    p_S = len(set_gamma[i])
                    computed_index = np.array(computed_index)
                    set_gamma_i = np.array([set_gamma[i]], dtype=np.uint32) - 1
                    if np.sum(~np.isnan(computed_index)) == 0:
                        set_gamma_margin.append(np.apply_along_axis(marginal_likelihood, 0, set_gamma_i, Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var))
                        while len(C_list[0]) <= col_num:
                            C_list[0].append([])
                            C_list[1].append([])
                        C_list[0][col_num].extend(set_gamma_margin[-1])
                        C_list[1][col_num] = vstack((C_list[1][col_num], dense_gamma_matrix))
                        set_gamma_prior = [prior_dist(row) for row in dense_gamma_matrix]
                        set_gamma_margin[-1] += set_gamma_prior[-1]
                    else:
                        set_gamma_margin.append(np.full(dense_gamma_matrix.shape[0], np.nan))
                        set_gamma_margin[-1][~np.isnan(computed_index)] = C_list[0][col_num][~np.isnan(computed_index)]
                        if np.sum(np.isnan(computed_index)) != 0:
                            set_gamma_margin[-1][np.isnan(computed_index)] = np.apply_along_axis(marginal_likelihood, 0, set_gamma_i[np.isnan(computed_index)], Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                        C_list[0][col_num].extend(set_gamma_margin[-1][np.isnan(computed_index)])
                        C_list[1][col_num] = vstack((C_list[1][col_num], dense_gamma_matrix[np.isnan(computed_index)]))
                        set_gamma_margin[-1] += np.apply_along_axis(prior_dist, 1, dense_gamma_matrix)

                    t1 = time.time() - t0

                    print(set_gamma_margin)
                    add_B = [None, None]
                    add_B[0] = [set_gamma_margin[0], set_gamma_margin[1], set_gamma_margin[2]]
                    add_B[1] = csr_matrix(([], ([], [])), shape=(0, p))
                    for i in range(3):
                        add_B[1] = vstack((add_B[1], matrix_gamma[i]))

            if len(working_S) == 1:
                print('working_S == 1')
                set_gamma_margin[0] = null_margin
                matrix_gamma[0] = null_model

                for i in range(1, 3):
                    matrix_gamma[i] = index_fun(set_gamma[i])
                    col_num = len(set_gamma[i])

                    if len(C_list[1]) <= col_num:
                        C_list[1].append(csr_matrix(([], ([], [])), shape=(0, p)))
                        C_list[0].append([])
                        computed_index = []
                    else:
                        computed_index = match_dgCMatrix(C_list[1][col_num], matrix_gamma[i])

                    p_S = len(set_gamma[i])
                    computed_index = np.array(computed_index)
                    if np.sum(~np.isnan(computed_index)) == 0:
                        set_gamma_margin[i] = np.apply_along_axis(marginal_likelihood, 0, np.array([set_gamma[i]], dtype=np.uint32) - 1, Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                        C_list[0][col_num].extend(set_gamma_margin[i])
                        C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[i]))
                        set_gamma_prior[i] = np.apply_along_axis(prior_dist, 1, matrix_gamma[i])
                        set_gamma_margin[i] += set_gamma_prior[i]
                    else:
                        set_gamma_margin[i] = np.full(matrix_gamma[i].shape[0], np.nan)
                        set_gamma_margin[i][~np.isnan(computed_index)] = C_list[0][col_num][~np.isnan(computed_index)]
                        if np.sum(np.isnan(computed_index)) != 0:
                            set_gamma_margin[i][np.isnan(computed_index)] = np.apply_along_axis(marginal_likelihood, 0, set_gamma[i][np.isnan(computed_index)], Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                        C_list[0][col_num].extend(set_gamma_margin[i][np.isnan(computed_index)])
                        C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[i][np.isnan(computed_index)]))
                        set_gamma_margin[i] += np.apply_along_axis(prior_dist, 1, matrix_gamma[i])

                    add_B = [[], csr_matrix(([], ([], [])), shape=(0, p))]
                    add_B[0].extend([set_gamma_margin[0], set_gamma_margin[1], set_gamma_margin[2]])
                    add_B[1] = csr_matrix(([], ([], [])), shape=(0, p))
                    for i in range(3):
                        add_B[1] = vstack((add_B[1], matrix_gamma[i]))

            if len(working_S) == 0:
                print('working_S is 0')
                for i in [1]:
                    #print(set_gamma[i])
                    matrix_gamma[i] = index_fun(set_gamma[i])
                    col_num = len(set_gamma[i])
                    if len(C_list[1]) <= col_num:
                        C_list[1].append(csr_matrix(([], ([], [])), shape=(0, p)))
                        C_list[0].append([])
                        computed_index = []
                    else:
                        computed_index = match_dgCMatrix(C_list[1][col_num], matrix_gamma[i])

                    p_S = len(set_gamma[i])
                    computed_index = np.array(computed_index)
                    print('set_gamma', np.array(set_gamma[i], dtype=np.uint32).flatten() - 1)
                    if np.sum(~np.isnan(computed_index)) == 0:
                        # wip here:
                        set_gamma_margin[i] = np.apply_along_axis(marginal_likelihood, 0, np.array(set_gamma[i], dtype=np.uint32).flatten() - 1, Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                        C_list[0][col_num].extend(set_gamma_margin[i])
                        C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[i]))
                        set_gamma_prior[i] = np.apply_along_axis(prior_dist, 1, matrix_gamma[i])
                        set_gamma_margin[i] += set_gamma_prior[i]
                    else:
                        set_gamma_margin[i] = np.full(matrix_gamma[i].shape[0], np.nan)
                        set_gamma_margin[i][~np.isnan(computed_index)] = C_list[0][col_num][~np.isnan(computed_index)]
                        if np.sum(np.isnan(computed_index)) != 0:
                            set_gamma_i = np.array([set_gamma[i]], dtype=np.uint32) - 1
                            set_gamma_margin[i][np.isnan(computed_index)] = np.apply_along_axis(marginal_likelihood, 1, set_gamma_i[np.isnan(computed_index)], Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                        C_list[0][col_num].extend(set_gamma_margin[i][np.isnan(computed_index)])
                        C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[i][np.isnan(computed_index)]))
                        set_gamma_margin[i] += np.apply_along_axis(prior_dist, 1, matrix_gamma[i])

                add_B = [set_gamma_margin[1], matrix_gamma[1]]

# Add visited models into the storage space of models
            add_index = match_dgCMatrix(B_list[1], add_B[1])
            if len(np.where(~np.isnan(add_index))[0]) > 10:
                check_index = np.random.choice(np.where(~np.isnan(add_index))[0], 10)

            if len(add_index[~np.isnan(add_index)]) != 0:
                B_list[0].extend(add_B[0][np.isnan(add_index)])
                B_list[1] = vstack((B_list[1], add_B[1][np.isnan(add_index),]))
            else:
                B_list[0].extend(add_B[0])
                B_list[1] = vstack((B_list[1], add_B[1]))

            sort_order = np.argsort(B_list[0])[::-1]
            B_list[0] = [B_list[0][i] for i in sort_order]
            B_list[1] = B_list[1][sort_order,]
# Select next visiting model 
            if len(working_S) != 0:
                set_star = pd.DataFrame({'set_index': range(1, 4),
                                        'gamma_set_index': [np.nan] * 3,
                                        'margin': [np.nan] * 3})
                for i in range(1):
                    aa = set_gamma_margin[i] - current_log_margin
                    aa = aa - aa[np.argmax(aa)]
                    if np.sum(np.isnan(aa)) != 0:
                        aa[np.isnan(aa)] = np.min(aa)
                    set_star['gamma_set_index'][i] = np.random.choice(range(1, len(set_gamma_margin[i]) + 1), 1, p=np.exp(aa))
                    set_star['margin'][i] = set_gamma_margin[i][set_star['gamma_set_index'][i]]

# The Bayesian hypothesis testing for outliers (Z-scores/LD discrepancies)
            if outlier_switch:
                for i in range(1, len(set_gamma)):
                    while True:
                        aa = set_gamma_margin[i] - current_log_margin
                        aa = aa - aa[np.argmax(aa)]
                        if np.sum(np.isnan(aa)) != 0:
                            aa[np.isnan(aa)] = np.min(aa[~np.isnan(aa)])

                        set_star['gamma_set_index'][i] = np.random.choice(range(1, len(set_gamma_margin[i]) + 1), 1, p=np.exp(aa))
                        set_star['margin'][i] = set_gamma_margin[i][set_star['gamma_set_index'][i]]

                        test_S = set_gamma[i][set_star['gamma_set_index'][i]]
                        # Added these 2 lines to correct input for ridge_fun()
                        test_S = np.concatenate([np.array(sublist).flatten() for sublist in test_S])
                        test_S = np.unique(test_S - 1).astype(np.int32)

                        modi_Sigma = Sigma
                        temp_Sigma = Sigma
                        if len(test_S) > 1:
                            modi_ld_S = modi_Sigma[test_S, test_S]
                            opizer = optimize(ridge_fun, interval=[0, 1], maximum=True)
                            modi_ld_S = opizer['maximum'] * modi_ld_S + (1 - opizer['maximum']) * np.diag(np.diag(modi_ld_S))

                            modi_Sigma[test_S, test_S] = modi_ld_S

                            test_log_BF = outlier_likelihood(test_S, Sigma, z, outlier_tau, len(test_S), 1) - outlier_likelihood(test_S, modi_Sigma, z, outlier_tau, len(test_S), 1)
                            test_log_BF = -abs(test_log_BF)
                            print('Outlier BF:', test_log_BF)
                            print(test_S)
                            print('This is xi hat:', opizer)

                        if np.exp(test_log_BF) < outlier_BF_index:
                            set_gamma[i] = set_gamma[i][~np.isin(set_gamma[i], set_star['gamma_set_index'][i])]
                            set_gamma_margin[i] = set_gamma_margin[i][~np.isin(set_gamma_margin[i], set_star['gamma_set_index'][i])]
                            conditional_S = np.concatenate((conditional_S, test_S[~np.isin(test_S, working_S)]))
                            conditional_S = np.unique(conditional_S)
                        else:
                            break
            else:
                for i in range(1, len(set_gamma)):
                    aa = set_gamma_margin[i] - current_log_margin
                    aa = aa - aa[np.argmax(aa)]
                    if np.sum(np.isnan(aa)) != 0:
                        aa[np.isnan(aa)] = np.min(aa[~np.isnan(aa)])

                    set_star['gamma_set_index'][i] = np.random.choice(range(1, len(set_gamma_margin[i]) + 1), 1, p=np.exp(aa))
                    set_star['margin'][i] = set_gamma_margin[i][set_star['gamma_set_index'][i]]

            if len(working_S) == num_causal:
                set_star = set_star.drop(1)
                aa = set_star['margin'] - current_log_margin - max(set_star['margin'] - current_log_margin)
                sec_sample = np.random.choice([1, 3], 1, p=np.exp(aa))
                S = set_gamma[sec_sample][set_star['gamma_set_index'][np.where(sec_sample == set_star['set_index'])[0][0]], :]
            else:
                aa = set_star['margin'] - current_log_margin - max(set_star['margin'] - current_log_margin)
                sec_sample = np.random.choice(range(1, 4), 1, p=np.exp(aa))
                S = set_gamma[sec_sample][set_star['gamma_set_index'][sec_sample], :]

                set_star = pd.DataFrame({'set_index': [1, 1, 1], 'gamma_set_index': [np.nan, np.nan, np.nan], 'margin': [np.nan, np.nan, np.nan]})
                aa = set_gamma_margin[2] - current_log_margin
                aa = aa - aa[np.argmax(aa)]
                if np.sum(np.isnan(aa)) != 0:
                    aa[np.isnan(aa)] = np.min(aa[~np.isnan(aa)])
                indices = np.argsort(np.exp(aa))[::-1][:min(len(aa), int(p/2))]
                set_star['gamma_set_index'][2] = np.random.choice(indices + 1, 1, p=np.exp(aa)[indices])
                set_star['margin'][2] = set_gamma_margin[2][set_star['gamma_set_index'][2]]
                S = set_gamma[2][set_star['gamma_set_index'][2], :]
                print(set_star)

            print('this is running S:', ','.join(map(str, S)))
            S = np.unique(np.concatenate((S, conditional_S)))


# Output of the results of the module function
        result_B_list = []
        if conditional_S is not None:
            all_c_index = []
            for tt in conditional_S:
                c_index = B_list[1].indices[B_list[1].indptr[tt]:B_list[1].indptr[tt+1]] + 1
                all_c_index.extend(c_index)

            all_c_index = np.unique(all_c_index)
            temp_B_list = []
            temp_B_list.append(B_list[0][np.setdiff1d(range(len(B_list[0])), all_c_index)])
            temp_B_list.append(B_list[1].tocsr()[np.setdiff1d(range(B_list[1].shape[0]), all_c_index), :])
        else:
            temp_B_list = [B_list[0], B_list[1]]

        result_B_list = []
        result_B_list.append(temp_B_list[0][:min(B, temp_B_list[1].shape[0])])
        result_B_list.append(temp_B_list[1][:min(B, temp_B_list[1].shape[0]), :])

        if num_causal == 1:
            single_set = np.arange(1, p+1).reshape(-1, 1)
            single_marginal = np.apply_along_axis(marginal_likelihood, 1, single_set, Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
            aa = single_marginal - np.nanmax(single_marginal)
            prob_sum = np.sum(np.exp(aa))
            result_prob = np.exp(aa) / prob_sum
        else:
            result_prob = PIP_func(result_B_list[0], result_B_list[1])

        conditional_S_list = pd.DataFrame({'Index': conditional_S, 'Z': z[conditional_S, :]})

        stored_bf = 0

        if output_labels is not None:
            if not os.path.exists(output_labels):
                os.makedirs(output_labels)

            np.savetxt(os.path.join(output_labels, f'post_{label}_poi_likeli.txt'), result_B_list[0], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='')
            mmwrite(os.path.join(output_labels, f'post_{label}_poi_gamma.mtx'), result_B_list[1])
            np.savetxt(os.path.join(output_labels, f'post_{label}.txt'), result_prob, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='')
            if outlier_switch:
                conditional_S_list.to_pickle(os.path.join(output_labels, f'post_{label}_outliers.pkl'))

        difference = abs(np.mean(result_B_list[0][:round(np.quantile(np.arange(1, len(result_B_list[0])+1), probs=0.25))]) - stored_bf)
        print(difference)
        if difference < epsilon:
            break
        else:
            stored_bf = np.mean(result_B_list[0][:round(np.quantile(np.arange(1, len(result_B_list[0])+1), probs=0.25))])

        return [result_B_list, C_list, result_prob, conditional_S_list, prob_list]

module_cauchy_shotgun(z_array, ld_list, input_S=None, model_prior='Poisson',effect_size_prior='Normal')
#input_S=list(range(1, len(z_array)+1))
