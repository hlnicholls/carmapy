import carmapy.carmapy_c
import numpy as np
import pandas as pd
import time
import os
from itertools import combinations
from scipy import sparse, optimize
from scipy.special import gammaln, betaln
from math import log
from scipy.sparse import csc_matrix, coo_matrix, vstack
from sklearn.linear_model import LogisticRegressionCV, PoissonRegressor


def EM_M_step_func(input_response=None, w=None, input_alpha=0.5, EM_dist='Logistic'):
    """
    Perform the M-step of the EM algorithm for incorporating functional annotations
    - Maximization (M) step computes Attributes maximizing the expected log-likelihood found on the E step
    - Expectation–maximization algorithm which iterates to find local maximum likelihood

    Attributes:
        input_response (array-like, optional): Response variable for regression. Default is None.
        w (array-like, optional): Design matrix for regression. Default is None.
        input_alpha (float, optional): Regularization parameter for regression. Default is 0.5.
        EM_dist (str, optional): Distribution for EM step, either 'Poisson' or 'Logistic'. Default is 'Logistic'.

    Returns:
        glm_beta: Coefficients for the Generalized Linear Model (GLM) regression.
    """
    if EM_dist == 'Poisson':
        count_index = input_response
        param_grid = {'alpha': [input_alpha]}  # You can add more hyperparameters here if needed
        poisson_model = PoissonRegressor()
        cv_poisson = GridSearchCV(poisson_model, param_grid, cv=3, scoring='neg_mean_squared_error')
        cv_poisson.fit(w, count_index)
        glm_beta = np.concatenate(([cv_poisson.best_estimator_.intercept_], cv_poisson.best_estimator_.coef_))
    elif EM_dist == 'Logistic':
        cv_logistic = LogisticRegressionCV(Cs=10, cv=3, penalty='l2', scoring='neg_log_loss')
        cv_logistic.fit(w, input_response)
        cv_index = np.argmin(-cv_logistic.scores_[1].mean(axis=0)) 
        glm_beta = np.concatenate(([cv_logistic.intercept_[0]], cv_logistic.coef_[0]))

    return glm_beta

def credible_set_fun_improved(pip, ld, true_beta=None, rho=0.99):
    """
    # The computation of the credible set based on the final results of the fine-mapping step.
    Compute an improved credible set based on posterior inclusion probabilities (PIPs) and LD.

    Attributes:
        pip (array-like): Posterior inclusion probabilities.
        ld (array-like): Linkage disequilibrium (LD) matrix.
        true_beta (array-like, optional): True effect sizes. Default is None.
        rho (float, optional): Threshold for credible set. Default is 0.99.

    Returns:
        credible_set: Improved credible set containing the most likely causal variants.
    """
    candidate_r = np.concatenate((np.arange(0.5, 0.95, 0.05), np.arange(0.96, 1.0, 0.01)))
    snp_list = []
    ld = ld.rename(index=dict(zip(ld.index, range(len(ld.index)))))
    for r in candidate_r:
        working_ld = ld.copy()
        cor_threshold = r
        pip_order = np.argsort(-pip)
        snp_list_r = []
        group_index = 0
        s = 0
        while sum(pip[pip_order[s:]]) > rho:
            cor_group = np.abs(working_ld.iloc[pip_order[s], :]) > cor_threshold
            cor_group = cor_group.index[cor_group].tolist()
            cor_group = list(map(int, cor_group))
            if sum(pip[cor_group]) > rho:
                group_pip = pip[cor_group]
                snp_index = [cor_group[i] for i in np.argsort(-group_pip)[:min(np.where(np.cumsum(sorted(group_pip, reverse=True)) > rho)[0]) + 1]]
                snp_list_r.append(snp_index)
                group_index += 1
                pip_order = [x for x in pip_order if x not in snp_index]
                working_ld = working_ld.drop(index=snp_index, columns=snp_index)
            else:
                s += 1
        snp_list.append(snp_list_r)

    if sum(len(x) for x in snp_list) != 0:
        group_index = np.argmax([len(x) for x in snp_list])
        credible_set_list = snp_list[group_index]
        purity = [np.mean(ld.loc[snp, snp]**2) for snp in credible_set_list]
        length_credible = len(credible_set_list)
        if true_beta is not None:
            causal_snp = len([x for x in true_beta if x in np.concatenate(credible_set_list)]) if len([x for x in true_beta if x in np.concatenate(credible_set_list)]) != 0 else 0
            return ([causal_snp, length_credible if length_credible != 0 else None, np.mean([len(x) for x in credible_set_list]), np.mean(purity)], credible_set_list)
        else:
            return ([length_credible if length_credible != 0 else None, np.mean([len(x) for x in credible_set_list]), np.mean(purity)], credible_set_list)
    else:
        return ([0] * 4, [])

def credible_model_fun(likelihood, model_space, bayes_threshold=10):
    """
    # The computation of the credible models based on the final results of the fine-mapping step.
    Identify credible models based on likelihood and a given Bayes threshold.

    Attributes:
        likelihood (array-like): Likelihood values for different models.
        model_space (array-like): Space of possible models.
        bayes_threshold (float, optional): Threshold for Bayes Factor to consider a model credible. Default is 10.

    Returns:
        credible_models: Credible models that meet the specified Bayes threshold.
    """
    na_index = np.where(np.isnan(likelihood))[0]
    if len(na_index) != 0:
        likelihood = np.delete(likelihood, na_index)
        model_space = np.delete(model_space, na_index, axis=0)

    post_like_temp = likelihood - likelihood[0]
    post_prob = np.exp(post_like_temp) / np.sum(np.exp(post_like_temp))
    bayes_f = post_prob[0] / post_prob
    candidate_model = 1

    credible_model_list = []
    credible_model_list.append([])
    input_rs = []

    for ss in np.where(bayes_f < bayes_threshold)[0]:
        credible_model_list[0].append(np.where(model_space[ss] == 1)[0])
        input_rs.extend(np.where(model_space[ss] == 1)[0])

    credible_model_list.append(pd.DataFrame({'Posterior.Prob': post_prob[np.where(bayes_f < bayes_threshold)[0]]}))
    credible_model_list.append(np.unique(input_rs))

    return credible_model_list

###

def module_cauchy_shotgun(z, ld_matrix, Max_Model_Dim=1e+4, input_S=None, lambda_val=(1 / np.sqrt(len(z_list))), label=None,
                        num_causal=10, output_labels=None, y_var=1, effect_size_prior=None, model_prior=None,
                        outlier_switch=None, input_conditional_S_list=None, tau=1/0.05**2, C_list=None, prior_prob=None,
                        epsilon=1e-3, inner_all_iter=10, condition_index=None):
    """
    Shotgun stochastic search algorithm or sampling from the posterior distribution over model space
    Iterative procedure that exhaustively examines three neighborhood sets of the current model:
        1. Algorithm first randomly selects one candidate model from each neighborhood set according to the unnormalized posterior probabilities
        2. Algorithm randomly selects the next model among the three selected models according to the corresponding posterior probabilities
        3. Algorithm stochastically moves toward the high posterior area in the model space

    Attributes:
        z (array-like): Z-scores from GWAS.
        ld_matrix (array-like): Linkage disequilibrium (LD) matrix.
        Max_Model_Dim (float, optional): Maximum model dimension. Default is 1e+4.
        input_S (array-like, optional): Input matrix S. Set of SNPs? Default is None.
        lambda_val (float, optional): Regularization parameter. Default is None.
        label (str, optional): Label for the locus. Default is None.
        num_causal (int, optional): Number of causal SNPs. Default is 10.
        output_labels (str, optional): Output labels. Default is None.
        y_var (float, optional): Variance of the response. Default is 1.
        effect_size_prior (str, optional): Prior distribution for effect sizes. Default is None.
        model_prior (str, optional): Prior distribution for the model. Default is None.
        outlier_switch (bool, optional): Whether to consider outliers. Default is None.
        input_conditional_S_list (list, optional): List of conditional input matrices S. Default is None.
        tau (float, optional): Scaling parameter for priors. Default is 1/0.05**2.
        C_list (list, optional): List of C matrices. Default is None.
        prior_prob (float, optional): Prior probability. Default is None.
        epsilon (float, optional): Convergence threshold. Default is 1e-3.
        inner_all_iter (int, optional): Number of inner iterations. Default is 10.

    Methods:
        input_prior_dist(): Calculate the prior distribution for a given input vector `x`.
        poisson_prior_dist(): Calculate the Poisson prior distribution for a given input vector `t`.
        beta_bionomial_dist(): Calculate the beta-binomial distribution value for a given input vector `t`.
        set_gamma_func(): Defines the neighborhood model space by setting gamma functions based on the input set
            - This function computes three different sets of gamma functions representing various combinations and concatenations of elements from S. 
            - The resulting structure provides a way to explore different possibilities within the neighborhood model space.
            add_function(): Concatenates and sorts the input value y with each element in the S array.
            set_gamma_func_base(): Computes the gamma sets for a given set (S), defining a specific aspect of the neighborhood model space.
            set_gamma_func_conditional(): Computes the gamma sets for an input set with a conditional index, defining a specific aspect of the gamma function/model space neighborhood.
        duplicated_dgCMatrix(): Checks for duplicates in a given sparse matrix (dgCMatrix) along the specified margin.
        match_dgCMatrix(): Matches the rows of two sparse matrices (dgCMat1 and dgCMat2) based on their structure.
        PIP_func(): Computes posterior inclusion probability based on the marginal likelihood and model space
        index_fun_inner(): Constructs a sparse matrix (CSR) based on the given input matrix `x`.
        index_fun(): Constructs a sparse matrix (CSR) based on the given input matrix `outer_x`, with optional binning.
        ridge_fun(): Computes the outlier likelihood based on a modified ridge regression function.
    """
    # The prior distributions on the model space
    print("Z shape:", z.shape)
    print("LD matrix shape:", ld_matrix.shape)
    print("Input S", input_S)
    print("Lambda value", lambda_val)
    prob_list = []
    p = len()
    if model_prior == 'input.prob':
        posi_log_pro = np.log(prior_prob)
        nega_log_pro = np.log(1 - prior_prob)
        prior_dist = input_prior_dist

    if model_prior == 'Poisson':
        prior_dist = poisson_prior_dist

    if model_prior == 'beta-binomial':
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
    log_2pi = np.log(2 * np.pi)
    B = Max_Model_Dim
    stored_result_prob = np.zeros(p)
    stored_bf = 0
    Sigma = ld_matrix

    if input_S is not None:
        S = input_S
    else:
        S = [] # we are here

    conditional_S = None
    #null_model = sparse.csr_matrix(np.zeros(p))
    null_model = np.zeros(p)
    null_margin = prior_dist(null_model)
    print('null margin using prior dist:', null_margin) # 0.0 output matches R equivalent
    if C_list is None:
        C_list = [[], []]
        
        # B_list - Storage space of candidate models
        B_list = [[null_margin], sparse.csr_matrix(np.zeros(p))]

        if input_conditional_S_list is None:
            conditional_S_list = []
            conditional_S = None
        else:
            conditional_S_list = input_conditional_S_list.copy()
            conditional_S = input_conditional_S_list['Index'].copy()
            conditional_S = np.unique(conditional_S)
            S = conditional_S
    
        # Compute posterior inclusion probability based on the marginal likelihood and model space

        for l in range(inner_all_iter):
            print('module_cauchy_shotgun outermost for loop iteration:', l)
            for h in range(10):
                # Shotgun COMPUTATION
                # set_gamma - Model configurations for calculating marginal likelihood
                set_gamma = set_gamma_func(S, conditional_S, p=p)
                if conditional_S is None:
                    working_S = S
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
                    working_S = np.unique(working_S - 1).astype(np.int64)
                    Sigma = np.array(ld_matrix, dtype=np.float64)
                    current_log_margin = marginal_likelihood(working_S, Sigma, z, tau_sample, p_S, y_var) + prior_dist(S_model)
                else:
                    current_log_margin = prior_dist(null_model)
                print('current_log_margin when S!=0:', current_log_margin)

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

                        print('len(set_gamma[i])', len(set_gamma))
                        p_S = len(set_gamma) - 1
                        computed_index = np.array(computed_index)
                        if np.sum(~np.isnan(computed_index)) == 0:
                            set_gamma_margin.append(np.apply_along_axis(marginal_likelihood, 0, np.array([set_gamma[i]], dtype=np.int64), Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var))
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
                            set_gamma_margin[-1] += np.apply_along_axis(prior_dist, 0, dense_gamma_matrix)

                        t1 = time.time() - t0
                        
                        # add_B - Appended storage of visited models (add bayesian?)
                        add_B = [None, None]
                        add_B[0] = [set_gamma_margin[0], set_gamma_margin[1], set_gamma_margin[2]]
                        add_B[1] = csr_matrix(([], ([], [])), shape=(0, p))
                        for i in range(3):
                            add_B[1] = vstack((add_B[1], matrix_gamma[i]))

                if len(working_S) == 1:
                    print('working_S == 1')
                    set_gamma_margin = null_margin
                    matrix_gamma[0] = null_model

                    for i in range(1, 3):
                        matrix_gamma[i] = index_fun(set_gamma[i])
                        print('set gamma into index s==1',len(set_gamma[i]))
                        print('matrix gamma S==1', i, matrix_gamma[i])
                        print(type(matrix_gamma[i]))
                        print(matrix_gamma[i].shape)
                        if i == 1:
                            col_num = len(set_gamma[i])
                            C_list[1] = (csr_matrix(([], ([], [])), shape=(0, p)))
                            #C_list[0].append([]) - not needed as extend() used later
                            computed_index = []
                        else:
                            computed_index = list(range(p))
                        print('len(set_gamma[i])', len(set_gamma))
                        p_S = len(set_gamma) - 1
                        print('s=1 p_S', p_S)
                        computed_index = np.array(computed_index)
                        if np.sum(~np.isnan(computed_index)) == 0:
                            set_gamma_margin = []
                            #set_gamma_margin = np.apply_along_axis(marginal_likelihood, 1, np.array(set_gamma[i], dtype=np.int64), Sigma=np.array(Sigma, dtype=np.float64), z=np.array(z, dtype=np.float64), tau=tau_sample, p_S=p, y_sigma=y_var)
                            set_gamma_margin.append(np.apply_along_axis(marginal_likelihood, 1, np.array(set_gamma[i], dtype=np.int64), Sigma=np.array(Sigma, dtype=np.float64), z=np.array(z, dtype=np.float64), tau=tau_sample, p_S=p_S, y_sigma=y_var))
                            print('set_gamma_margin S==1', len(set_gamma_margin[0]), set_gamma_margin)
                            C_list[0].extend(set_gamma_margin)
                            print('s=1 matrix_gamma[1]', matrix_gamma[i])
                            print(C_list[1].shape, matrix_gamma[i].shape)
                            print(type(C_list[1]))
                            C_list[1] = vstack((C_list[1], matrix_gamma[i]))
                            print(C_list[1])
 
                            for j in range(matrix_gamma[i].shape[0]):
                                row = matrix_gamma[i].getrow(j).toarray()[0]
                                result = prior_dist(row)
                                set_gamma_prior.append(result)
                            set_gamma_margin += set_gamma_prior
                        else:
                            set_gamma_margin[i] = np.full(matrix_gamma[i].shape[0], np.nan)
                            set_gamma_margin= C_list[0]
                            if np.sum(np.isnan(computed_index)) != 0:
                                set_gamma_margin[i][np.isnan(computed_index)] = np.apply_along_axis(marginal_likelihood, 0, set_gamma[i][np.isnan(computed_index)], Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                            C_list[0][col_num].extend(set_gamma_margin[i][np.isnan(computed_index)])
                            C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[i][np.isnan(computed_index)]))
                            for j in range(matrix_gamma[i].shape[0]):
                                row = matrix_gamma[i].getrow(j).toarray()[0]
                                result = prior_dist(row)
                                set_gamma_prior.append(result)
                            set_gamma_margin += np.array(set_gamma_prior)

                        add_B = [[], csr_matrix(([], ([], [])), shape=(0, p))]
                        add_B[0].extend([set_gamma_margin[0], set_gamma_margin[1], set_gamma_margin[2]])
                        for i in range(2):
                            #print('add_B[1] for loop',i, add_B[1])
                            #print(type(add_B[1]))
                            #print('matrix_gamma[i] for loop',i, matrix_gamma[i])
                            add_B[1] = csr_matrix(vstack((add_B[1], matrix_gamma[i])))

                if len(working_S) == 0:
                    for i in [1]:
                        print('set_gamma[i] before index_fun', len(set_gamma[i]), type(set_gamma[i]))
                        matrix_gamma[i] = index_fun(set_gamma[i])
                        print('matrix gamma type', type(matrix_gamma[i]))
                        print('matrix gamma shape after index_fun when S==0:', matrix_gamma[i].shape)
                        print('matrix gamma index_fun when S==0:', matrix_gamma[i])
                        col_num = len(set_gamma[i])
                        print('col_num when S==0', col_num)

                        if len(C_list[1]) < col_num:
                            C_list[1] = []
                            #C_list[0].append([])

                        if len(C_list[1]) == 0:
                            C_list[1] = csr_matrix(([], ([], [])), shape=(0, p))
                            #C_list[0].append([])
                            computed_index = []
                        else:
                            computed_index = match_dgCMatrix(C_list[1][col_num], matrix_gamma[i])

                        print('C_list setup before use (S==0)', len(C_list), C_list)

                        computed_index = np.array(computed_index)
                        p_S = len(set_gamma) - 2
                        if np.sum(~np.isnan(computed_index)) == 0:
                            set_gamma_margin.append(np.apply_along_axis(marginal_likelihood, 1, np.array(set_gamma[i], dtype=np.int64), Sigma=np.array(Sigma, dtype=np.float64), z=np.array(z, dtype=np.float64), tau=tau_sample, p_S=p_S, y_sigma=y_var))
                            C_list[0] = set_gamma_margin
                            C_list[1] = vstack((C_list[1], matrix_gamma[1]))
                            print('C list after first marginal likelihood', C_list)
                            # switched to for loop:
                            for j in range(matrix_gamma[1].shape[0]):
                                row = matrix_gamma[1].getrow(j).toarray()[0]
                                result = prior_dist(row)
                                set_gamma_prior.append(result)
                            set_gamma_margin += np.array(set_gamma_prior)
                        else:
                            set_gamma_margin = np.full(matrix_gamma[1].shape[0], np.nan)
                            set_gamma_margin[~np.isnan(computed_index)] = C_list[0].extend([~np.isnan(computed_index)])
                            if np.sum(np.isnan(computed_index)) != 0:
                                set_gamma_i = np.array([set_gamma[i]], dtype=np.int64)
                                set_gamma_margin[0][np.isnan(computed_index)] = np.apply_along_axis(marginal_likelihood, 0, set_gamma_i[np.isnan(computed_index)], Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
                            C_list[0].extend(set_gamma_margin[np.isnan(computed_index)])
                            C_list[1][col_num] = vstack((C_list[1][col_num], matrix_gamma[1][np.isnan(computed_index)]))
                            #switching from below to for loop due to prior_dist sparse matrix output
                            #set_gamma_margin[i] += np.apply_along_axis(prior_dist, 0, matrix_gamma[i])
                            for j in range(matrix_gamma[1].shape[0]):
                                row = matrix_gamma[1].getrow(j).toarray()[0]
                                result = prior_dist(row)
                                set_gamma_prior.append(result)
                            set_gamma_margin += np.array(set_gamma_prior)
                        print('C.list when S==0', 'length:', len(C_list), C_list)

                    add_B = [set_gamma_margin, matrix_gamma[1]]
                    print('working_S == 0')
                    print('add_B when S==0', 'length of [0]:', len(add_B[0][0]), add_B)

    # Add visited models into the storage space of models
                add_index = match_dgCMatrix(B_list[1], add_B[1])
                add_index = [x if x is not None else np.nan for x in add_index]
                #print('add_index setup:', add_index)
                if len([x for x in add_index if not np.isnan(x)]) > 10:
                    check_index = np.random.choice(np.where(~np.isnan(add_index))[0], 10)

                B_list[0] = np.array(B_list[0])
                add_B[0] = add_B[0].flatten()

                if len([x for x in add_index if not np.isnan(x)]) != 0:
                    B_list[0] = np.concatenate((B_list[0], add_B[0][np.isnan(add_index)]))
                    B_list[1] = vstack((B_list[1], add_B[1][np.isnan(add_index),]))
                else:
                    B_list[0] = B_list[0][1:]
                    B_list[0] = np.concatenate((B_list[0], add_B[0]))
                    B_list[1] = B_list[1][1:, :]
                    B_list[1] = vstack((B_list[1], add_B[1]))

                sort_order = np.argsort(B_list[0])[::-1]
                B_list[0] = [B_list[0][i] for i in sort_order]
                B_list[1] = B_list[1][sort_order,]
                print('working_S ==',working_S, 'B_list:', 'lengths:', len(B_list[0]), B_list[1].shape)

    # Select next visiting model
                if len(working_S) != 0:
                    print('working_S !=0 for selecting next model')
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
                                aa = set_gamma_margin - current_log_margin
                                aa = aa - np.max(aa)
                                if np.sum(np.isnan(aa)) != 0:
                                    aa[np.isnan(aa)] = np.min(aa[~np.isnan(aa)])
                                aa = aa.flatten()
                                set_star['gamma_set_index'][i] = np.random.choice(range(len(set_gamma_margin[0])), 1, p=np.exp(aa))
                                set_star['margin'][i] = set_gamma_margin[set_star['gamma_set_index'][i]]
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
                                    print('np.exp(test_log_BF) < outlier_BF_index')
                                    set_gamma[i] = set_gamma[i][~np.isin(set_gamma[i], set_star['gamma_set_index'][i])]
                                    set_gamma_margin[i] = set_gamma_margin[i][~np.isin(set_gamma_margin[i], set_star['gamma_set_index'][i])]
                                    conditional_S = np.concatenate((conditional_S, test_S[~np.isin(test_S, working_S)]))
                                    conditional_S = np.unique(conditional_S)
                                else:
                                    break
                    else:
                        for i in range(1, len(set_gamma)):
                            aa = set_gamma_margin[i] - current_log_margin
                            aa = aa - np.max(aa)
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
                else:
                    print('setting new S (selecting new model) after S==0')
                    set_star = pd.DataFrame({'set_index': [1, 1, 1], 'gamma_set_index': [np.nan, np.nan, np.nan], 'margin': [np.nan, np.nan, np.nan]})
                    aa = set_gamma_margin - current_log_margin
                    aa = aa - np.max(aa)
                    aa = aa.flatten()
                    if np.sum(np.isnan(aa)) != 0:
                        aa[np.isnan(aa)] = np.min(aa[~np.isnan(aa)])
                    indices = np.argsort(np.exp(aa))[::-1][:min(len(aa), int(p/2))]
                    prob = np.exp(aa)[indices]
                    chosen_index = np.random.choice(indices, 1, p=prob / np.sum(prob))
                    set_star.loc[1, 'gamma_set_index'] = chosen_index
                    set_star.loc[1, 'margin'] = set_gamma_margin[0, int(chosen_index)]
                    idx = int(set_star.loc[1, 'gamma_set_index'])
                    S = [set_gamma[1][idx][0]]
                    print('new S:', S)

            print('this is running S:', ','.join(map(str, S)))
            S = np.unique(np.concatenate((S, conditional_S)))
            print(S)

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
        print('conditional_S is none at output')
        temp_B_list = [B_list[0], B_list[1]]

    result_B_list = []
    result_B_list.append(temp_B_list[0][:min(B, temp_B_list[1].shape[0])])
    result_B_list.append(temp_B_list[1][:min(B, temp_B_list[1].shape[0]), :])

    if num_causal == 1:
        single_set = np.arange(1, p+1).reshape(-1, 1)
        single_marginal = np.apply_along_axis(marginal_likelihood, 0, single_set, Sigma=Sigma, z=z, tau=tau_sample, p_S=p_S, y_sigma=y_var)
        aa = single_marginal - np.nanmax(single_marginal)
        prob_sum = np.sum(np.exp(aa))
        result_prob = np.exp(aa) / prob_sum
    else:
        result_prob = PIP_func(result_B_list[0], result_B_list[1])

    if conditional_S != None:
        conditional_S_list = pd.DataFrame({'Index': conditional_S, 'Z': z[conditional_S, :]})

    stored_bf = 0

    if output_labels is not None:
        if not os.path.exists(output_labels):
            os.makedirs(output_labels)

    difference = abs(np.mean(result_B_list[0][:round(np.quantile(np.arange(1, len(result_B_list[0])+1), q=0.25))]) - stored_bf)
    if difference < epsilon[0]:
        break
    else:
        stored_bf = np.mean(result_B_list[0][:round(np.quantile(np.arange(1, len(result_B_list[0])+1), q=0.25))])

    return [result_B_list, C_list, result_prob, conditional_S_list, prob_list]


### Functions for computing the prior distribution of the model space
def input_prior_dist(x):
    """Calculate the prior distribution for a given input vector `x`."""
    variable_index = np.where(x == 1)[0]
    other_index = np.setdiff1d(np.arange(0, p) - 1, variable_index)
    if len(variable_index) > 0:
        return np.sum(posi_log_pro[variable_index]) + np.sum(nega_log_pro[other_index]) - np.sum(nega_log_pro)
    else:
        return np.sum(nega_log_pro)

def poisson_prior_dist(t):
    """Calculate the Poisson prior distribution for a given input vector `t`."""
    dim_model = np.sum(t).item()
    return dim_model * log(lambda_val) + gammaln(p - dim_model + 1) - gammaln(p + 1)

def beta_binomial_dist(t):
    dim_model = np.sum(t)
    return betaln(dim_model + 1, p - dim_model + 9) - betaln(1, p + 9)

### Function to define the neighbourhood model space
def set_gamma_func(input_S=None, condition_index=None, p=None):
    """Defines the neighborhood model space by setting gamma functions based on the input set.
        - This function computes three different sets of gamma functions representing various combinations and concatenations of elements from S. 
        - The resulting structure provides a way to explore different possibilities within the neighborhood model space.
    """
    def set_gamma_func_base(S):
        """Computes the gamma sets for a given set (S), defining a specific aspect of the neighborhood model space."""
        def add_function(y):
            """Concatenates and sorts the input value y with each element in the S array."""
            results = [np.sort([x] + y) for x in S_sub]
            return np.array(results)

        set_gamma = [[], [], []]

        if len(S) == 0:
            S_sub = [i for i in range(p)]
            set_gamma[1] = [S + [x] for x in S_sub]
        elif len(S) == 1:
            S_sub = [i for i in range(p) if i not in S]
            set_gamma[0] = [list(comb) for comb in combinations(S, len(S) - 1)]
            set_gamma[1] = [sorted([x] + S) for x in S_sub]
            set_gamma[2] = add_function(set_gamma[0][0]).reshape(1, -1)
        else:
            S_sub = [i for i in range(p) if i not in S]
            set_gamma[0] = [sorted(list(comb)) for comb in combinations(S, len(S) - 1)] if len(S) > 2 else [list(comb) for comb in combinations(S, len(S) - 1)]
            set_gamma[1] = [sorted([x] + S) for x in S_sub]
            set_gamma[2] = add_function(set_gamma[0][0])
            for i in range(1, len(set_gamma[0])):
                set_gamma[2] = np.vstack((set_gamma[2], add_function(set_gamma[0][i])))
        return set_gamma

    def set_gamma_func_conditional(input_S, condition_index):
        """Computes the gamma sets for an input set with a conditional index, defining a specific aspect of the gamma function/model space neighborhood."""
        def add_function(y):
            results = [np.sort([x] + y) for x in S_sub]
            return np.array(results)

        S = [i for i in input_S if i != condition_index]
        S_sub = [i for i in range(p) if i not in input_S]
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
    else:
        results = set_gamma_func_conditional(input_S, condition_index)

    return results

def duplicated_dgCMatrix(dgCMat, MARGIN):
    """Checks for duplicates in a given sparse matrix (dgCMatrix) along the specified margin."""
    MARGIN = int(MARGIN)
    n = dgCMat.shape[0]
    p = dgCMat.shape[1]
    J = np.repeat(np.arange(0, p), np.diff(dgCMat.indptr))
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
        is_empty = np.setdiff1d(np.arange(0, p), J)
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
    """Matches the rows of two sparse matrices (dgCMat1 and dgCMat2) based on their structure.."""
    n1, p1 = dgCMat1.shape
    n2, p2 = dgCMat2.shape
    J1 = np.repeat(np.arange(p1), np.diff(dgCMat1.indptr))
    I1 = dgCMat1.indices
    J2 = np.repeat(np.arange(p2), np.diff(dgCMat2.indptr))
    I2 = dgCMat2.indices
    RowLst1 = [set(J1[I1 == i]) for i in range(n1)]
    RowLst2 = [set(J2[I2 == i]) for i in range(n2)]
    is_empty1 = set(range(n1)) - set(I1)
    is_empty2 = set(range(n2)) - set(I2)
    result = [RowLst1.index(row) + 1 if row in RowLst1 else np.nan for row in RowLst2]
    for idx in [i for i, x in enumerate(result) if x in is_empty1]:
        result[idx] += 1

    if any(is_empty1):
        if any(is_empty2):
            result = list(is_empty1) + result
        else:
            result = [np.nan] + result

    return result

# Compute posterior inclusion probability based on the marginal likelihood and model space
def PIP_func(likeli, model_space):
    """Computes posterior inclusion probability based on the marginal likelihood and model space
    Args:
        likeli (vector): All the marginal likelihoods of the candidate models (is 15744 for 481 loci)
        model_space (matrix): 15744 x 481
    """
    infi_index = np.where(np.isinf(likeli))[0]
    if len(infi_index) != 0:
        likeli = np.delete(likeli, infi_index)
        model_space = np.delete(model_space.toarray(), infi_index, axis=0)
    na_index = np.where(np.isnan(likeli))[0]
    if len(na_index) != 0:
        likeli = np.delete(likeli, na_index)
        model_space = np.delete(model_space.toarray(), na_index, axis=0)
    aa = likeli - np.nanmax(likeli)  
    prob_sum = np.sum(np.exp(aa))
    p = model_space.shape[1]
    result_prob = np.full(p, np.nan)
    for i in range(p):
        column_dense = model_space[:, i].toarray().flatten()
        result_prob[i] = np.sum(np.exp(aa[np.where(column_dense == 1)])) / prob_sum
    return result_prob

def index_fun_inner(x, max_col_index):
    """Constructs a sparse matrix (CSR) based on the given input matrix `x`."""
    n_rows, n_cols = x.shape
    row_indices = np.repeat(np.arange(n_rows), n_cols)
    col_indices = x.T.flatten()
    col_indices = np.minimum(col_indices, max_col_index)
    data = np.ones(n_rows * n_cols, dtype=int)
    m = coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, max_col_index + 1)).tocsc()
    return m

def index_fun(outer_x, Max_Model_Dimins=10):
    """Constructs a sparse matrix (CSR) based on the given input matrix `outer_x`, with optional binning."""
    outer_x = np.array(outer_x)
    n_rows, _ = outer_x.shape
    max_col_index = n_rows - 1

    if n_rows > 1000:
        index_bins = np.arange(n_rows // Max_Model_Dimins, n_rows, n_rows // Max_Model_Dimins)
        result_m = index_fun_inner(outer_x[:index_bins[0], :], max_col_index)
        for i in range(len(index_bins) - 1):
            result_m = vstack([result_m, index_fun_inner(outer_x[index_bins[i]:index_bins[i + 1], :], max_col_index)])
        if index_bins[-1] != n_rows:
            result_m = vstack([result_m, index_fun_inner(outer_x[index_bins[-1]:, :], max_col_index)])
    else:
        result_m = index_fun_inner(outer_x, max_col_index)
    result_m = result_m.tocsr()
    return result_m

def ridge_fun(x, modi_ld_S, test_S_indices, temp_Sigma, z, outlier_tau, outlier_likelihood):
    """Computes the outlier likelihood based on a modified ridge regression function."""
    temp_ld_S = x * modi_ld_S + (1 - x) * np.eye(modi_ld_S.shape[0])
    temp_Sigma[test_S_indices[:, None], test_S_indices] = temp_ld_S
    return outlier_likelihood(test_S_indices, temp_Sigma, z, outlier_tau, len(test_S_indices), 1)
