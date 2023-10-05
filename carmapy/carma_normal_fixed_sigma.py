import numpy as np
from scipy.sparse import csr_matrix
import time

from carmapy.carma_normal_fixed_sigma_utils import *


def CARMA_fixed_sigma(
    z_list,
    ld_matrix,
    w_list=None,
    lambda_list=None,
    output_labels=".",
    label_list=None,
    effect_size_prior="Normal",
    rho_index=0.99,
    BF_index=10,
    EM_dist="Logistic",
    Max_Model_Dim=2e5,
    all_iter=3,
    all_inner_iter=10,
    input_alpha=0,
    epsilon_threshold=1e-4,
    num_causal=10,
    y_var=1,
    tau=0.04,
    outlier_switch=True,
    outlier_BF_index=1 / 3.2,
    prior_prob_computation="Logistic",
):
    """
    Apply the CARMA method for fine-mapping with fixed sigma.

    Attributes:
        z_list - Input list of summary statistics at the testing loci; each element of the list is the summary statistics at each individual locus.
        ld_matrix - Input list of LD correlation matrix at the testing loci; each element of the list is the LD matrix at each individual locus.
        w_list - Input list of the functional annotations at the testing loci; each element of the list is the functional annotation matrix at each individual locus.
        lambda_list - Input list of the hyper-parameter \eqn{\eta} at the testing loci; each element of the list is the hyper-parameter of each individual locus.
        label_list - Input list of the names at the testing loci. Default is NULL.
        effect_size_prior - The prior of the effect size. The choice are 'Cauchy' and 'Spike-slab' priors, where the 'Spike-slab' prior is the default prior.
        input_alpha - The elastic net mixing parameter, where \eqn{0\le}\eqn{\alpha}\eqn{\le 1}.
        y_var - The input variance of the summary statistics, the default value is 1 as the summary statistics are standardized.
        rho_index - A number between 0 and 1 specifying \eqn{\rho} for the estimated credible sets.
        BF_index - A number smaller than 1 specifying the threshold of the Bayes factor of the estimated credible models. The default setting is 3.2.
        outlier_switch - The indicator variable for outlier detection. We suggest that the detection should always be turned on if using external LD matrix.
        num_causal - The maximum number of causal variants assumed per locus, which is 10 causal SNPs per locus by default.
        Max_Model_Dim - Maximum number of the top candidate models based on the unnormalized posterior probability.
        all_inner_iter - Maximum iterations for Shotgun algorithm to run per iteration within EM algorithm.
        all_iter - Maximum iterations for EM algorithm to run.
        output_labels - Output directory where output will be written while CARMA is running. Default is the OS root directory ".".
        epsilon_threshold - Convergence threshold measured by average of Bayes factors.
        EM_dist - The distribution used to model the prior probability of being causal as a function of functional annotations. The default distribution is logistic distribution.
    """

    # Feature learning step, such as learning the total number of input loci, the total number of variants at each locus, etc.
    # Additionally, CARMA defines the lists of the model spaces and the likelihood of all input loci

    np.seterr(divide="ignore", invalid="ignore")
    L = 1
    z_list = [np.asmatrix(z_list[0]).T]
    p = len(z_list[0])  # needs to be 481
    p_list = [p]
    assert L == len(z_list), "z_list needs to be a list of 1 array"
    assert z_list[0].shape[1] == 1, f"z_list needs to be a list of 1 array of shape (loci, 1), got {z_list[0].shape[1]} instead"
    assert ld_matrix.shape == (p, p), f"ld_matrix needs to be a matrix of size (n_snps, n_snps), got {ld_matrix.shape} instead"
    log_2pi = np.log(2 * np.pi)
    B = Max_Model_Dim
    all_B_list = [
        [np.zeros(0, dtype=int), csr_matrix((0, p_list[i]), dtype=int)]
        for i in range(L)
    ]

    if label_list is None:
        label_list = [f"locus_{i}" for i in range(1, L + 1)]

    # Sigma_list = ld_matrix
    # S_list = [np.zeros(0, dtype=int) for i in range(L)]
    all_C_list = [
        [np.zeros(0, dtype=int), csr_matrix((0, p_list[i]), dtype=int)]
        for i in range(L)
    ]
    all_epsilon_threshold = 0
    epsilon_list = [
        epsilon_threshold * p_list[i] for i in range(L)
    ]  # TODO: confirm that epsilon_list == epsilon_threshold
    all_epsilon_threshold = sum(
        epsilon_list
    )  # i am summing, but the list is one number
    model_prior = "Poisson"
    standardize_model_space = True

    # Burning step
    previous_result = []

    # Run fine-mapping step (module function) for each locus included in the analysis
    all_C_list = []
    all_C_list.append(
        module_cauchy_shotgun(
            z_list[0],
            ld_matrix,
            lambda_val=1,
            epsilon=epsilon_list,
            input_S=None,
            Max_Model_Dim=Max_Model_Dim,
            outlier_switch=outlier_switch,
            tau=tau,
            num_causal=num_causal,
            y_var=y_var,
            label=label_list,
            output_labels=output_labels,
            effect_size_prior=effect_size_prior,
            model_prior=model_prior,
            inner_all_iter=all_inner_iter,
        )
    )

    # Running CARMA
    delete_list = []

    for g in range(all_iter):
        if outlier_switch:
            delete_list = []
            for i in range(L):
                delete_list.append([])
                if len(all_C_list[i][4]) != 0:
                    temp_delete_list = all_C_list[i][4]["Index"]
                    delete_list[i] = temp_delete_list
        else:
            delete_list = []
            for i in range(L):
                # TODO: check this line, it is not clear what it is doing
                delete_list.append([])

    # If the list of annotations is non-empty, then the PIPs and functional annotations at all loci are aggregated for the M-step of the EM algorithm
    w = None

    if w_list is not None:
        w = np.empty((0, w_list[0].shape[1]))
        w_columns = list(w_list[0].columns)
        for i in range(L):
            if len(delete_list[i]) != 0:
                w = np.vstack((w, w_list[i].drop(delete_list[i]).values))
            else:
                w = np.vstack((w, w_list[i].values))

    for i in range(1, L + 1):
        previous_result.append(
            np.mean(
                all_C_list[i][0][0][0][
                    0 : round(np.quantile(range(1, len(all_C_list[i][0][0][0])), 0.25))
                ]
            )
        )

    model_space_count = []
    if w_list is not None and EM_dist == "Poisson":
        if not standardize_model_space:
            model_space_count = []
            for i in range(L):
                if len(delete_list[i]) != 0:
                    model_space_count.extend(
                        np.sum(
                            all_C_list[i][0][0][1][
                                :,
                                ~np.isin(all_C_list[i][0][0][4], delete_list[i]),
                            ],
                            axis=0,
                        )
                    )
                else:
                    model_space_count.extend(np.sum(all_C_list[i][0][0][1], axis=0))
        else:
            model_space_count = []
            for i in range(L):
                if len(delete_list[i]) != 0:
                    indi_count = np.sum(
                        all_C_list[i][0][0][1][
                            :, ~np.isin(all_C_list[i][0][0][4], delete_list[i])
                        ],
                        axis=0,
                    )
                    indi_count = np.floor(
                        indi_count / all_C_list[i][0][0][1].shape[0] * Max_Model_Dim
                    )
                else:
                    indi_count = np.sum(all_C_list[i][0][0][1], axis=0)
                    indi_count = np.floor(
                        indi_count / all_C_list[i][0][0][1].shape[0] * Max_Model_Dim
                    )
                model_space_count.extend(indi_count)

    M_step_response = model_space_count

    # The M step of the EM algorithm
    if EM_dist == "Logistic":
        M_step_response = []
        for i in range(L):
            if len(delete_list[i]) != 0:
                indi_pip = all_C_list[i][0][0][2][
                    ~np.isin(all_C_list[i][0][0][4], delete_list[i])
                ]
            else:
                indi_pip = all_C_list[i][0][0][2]
            M_step_response.extend(indi_pip)

    try:
        glm_beta = EM_M_step_func(
            input_response=M_step_response,
            w=w,
            input_alpha=input_alpha,
            EM_dist=EM_dist,
        )
        prior_prob_list = []

        for i in range(L):
            if prior_prob_computation == "Intercept.approx":
                glm_beta[0] = np.log(
                    (min(Max_Model_Dim, all_C_list[i][0][0][1].shape[0]))
                    * lambda_list[i]
                    / (lambda_list[i] + p_list[i])
                )
                prior_prob_list.append(
                    np.exp(np.dot(w_list[i], glm_beta))
                    / (1 + np.max(np.exp(np.dot(w_list[i], glm_beta))))[
                        0 : min(Max_Model_Dim, all_C_list[i][0][0][1].shape[0])
                    ]
                )
            if prior_prob_computation == "Logistic":
                prior_prob_list.append(np.logistic(np.dot(w_list[i], glm_beta)))
            if output_labels is not None:
                np.savetxt(
                    f"{output_labels}/post_{label_list[i]}_theta.txt",
                    glm_beta,
                    delimiter=",",
                    fmt="%.6f",
                )
        model_prior = "input.prob"
    except:
        model_prior = "Poisson"
        prior_prob_list = [None] * L

        # Fine-mapping step for each locus, i.e., the E-step in the EM algorithm
        for i in range(L):
            t0 = time.time()
            # Here Hannah used i - 1, but I think it should be i
            all_C_list[i] = module_cauchy_shotgun(
                z_array,
                ld_matrix,
                lambda_val=1,
                input_conditional_S_list=all_C_list[i][0][0][3],
                Max_Model_Dim=Max_Model_Dim,
                y_var=y_var,
                num_causal=num_causal,
                epsilon=epsilon_list,
                C_list=all_C_list[i][0][0][1],
                prior_prob=prior_prob_list[i],
                outlier_switch=outlier_switch,
                tau=tau,
                label=label_list,
                output_labels=output_labels,
                effect_size_prior=effect_size_prior,
                model_prior=model_prior,
                inner_all_iter=all_inner_iter,
            )
            t1 = time.time() - t0
            print(f"This is locus {i} computing time: ", t1)

        difference = 0
        for i in range(L):
            difference += np.sum(
                np.abs(
                    previous_result[i]
                    - np.mean(
                        all_C_list[i][0][0][0][
                            0 : round(
                                np.percentile(
                                    np.arange(1, len(all_C_list[i][0][0][0][0]) + 1),
                                    probs=0.25,
                                )
                            )
                        ]
                    )
                )
            )
            print(f"This is difference: {difference}")
            if difference < all_epsilon_threshold:
                break

    # Output of the results of CARMA
    results_list = []
    for i in range(L):
        results_list.append({})
        pip = all_C_list[i][0][0][2]
        credible_set = credible_set_fun_improved(pip, ld_matrix[i], rho=rho_index)
        credible_model = credible_model_fun(
            all_C_list[i][0][0][0][0],
            all_C_list[i][0][0][0][1],
            bayes_threshold=BF_index,
        )
        results_list[i]["PIPs"] = pip
        results_list[i]["Credible set"] = credible_set
        results_list[i]["Credible model"] = credible_model
        results_list[i]["Outliers"] = all_C_list[i][0][0]

    return results_list
