import sys
sys.path.append('/Users/hn9/Documents/GitHub/carmapy/carmapy')
import carma_normal_fixed_sigma 
import numpy as np
import pandas as pd
import carmapy
import carmapy.carmapy_c
from carmapy.carmapy_c import Normal_fixed_sigma_marginal
import numpy as np
import pandas as pd
import time
import os
from itertools import combinations
from scipy import sparse, optimize
from scipy.io import mmwrite
from scipy.special import gammaln, betaln
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV

#sumstats = pd.read_csv('/Users/hn9/Documents/GitHub/carmapy/tests/APOE_locus_sumstats.txt.gz', sep='\t')
#ld = pd.read_csv('/Users/hn9/Documents/GitHub/carmapy/tests/APOE_locus_ld.txt.gz', sep='\t')

sumstats = pd.read_csv('/Users/hn9/Documents/GitHub/CARMA/Simulation Study/Sample_data/sumstats_chr1_200937832_201937832.txt.gz', sep='\t')
ld = pd.read_csv('/Users/hn9/Documents/GitHub/CARMA/Simulation Study/Sample_data/sumstats_chr1_200937832_201937832_ld.txt.gz', sep='\t', header=None)
z_list = sumstats['Z'].tolist()
ld_matrix = np.asmatrix(ld)

effect_size_prior = 'Normal'

np.random.seed(1)
n = len(sumstats['Z'])
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

results = carma_normal_fixed_sigma.CARMA_fixed_sigma(z_list, ld_matrix)
print(results)