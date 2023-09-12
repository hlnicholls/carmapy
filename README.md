# carmapy

Python implementation of the CARMA R Package<sup>1</sup> for fine-mapping (work in progress).

**Installation:**

- Tested with Python 3.9.6 and cython 0.29.36

```
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install .
```

Sample data to test CARMA available in ```/tests``` or from: https://zenodo.org/record/7772462 (file to download: ```Sample_data.tar.gz```)

**To Test:**

1. Make changes to ```carmapy/carma_normal_fixed_sigma.py``` 
2. Update file paths:
    -  Line 2 ```sys.path.append()``` of ```tests/carma_test.py``` and ```tests/carma_py_functions_test.ipynb```
    - Update file paths to load ```sumstats``` and ```ld``` 
3. Run ```python ./tests/carma_test.py``` to test whole fine-mapping process or test individual functions in ```tests/carma_py_functions_test.ipynb``` 

**Notes:**
- ```tests/carma_py_functions_test.ipynb``` function outputs need to equal to those in ```tests/carma_r_functions_test.ipynb```

- ```tests/carma_cython_test.ipynb``` function outputs need to equal to those in ```tests/carma_rcpp_test.ipynb```

**Reference:**
1. Yang, Z., Wang, C., Liu, L. et al. CARMA is a new Bayesian model for fine-mapping in genome-wide association meta-analyses. Nat Genet 55, 1057–1065 (2023). https://doi.org/10.1038/s41588-023-01392-0
