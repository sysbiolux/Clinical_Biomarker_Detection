########################################################################################################################
# HPC PARALLELIZATION SCRIPT WITH IPYPARALLEL BACKEND ##################################################################
# REMOVING HIGHLY CORRELATED FEATURES, RESAMPLING, FEATURE TRANSFORMATION, PARAMETER GRID SEARCH, DATA SPLIT BY GENDER #
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# November 2021 - January 2023, University of Luxembourg, v.01/25/2023 (M/d/y) #########################################
########################################################################################################################

# SUMMARY: Full clinical cohort data as well as split data based on gender, updated and revised functions and comments,
# split pipeline and grid search, adopted prints, save figures path, performance summary for all 3 data type cases,
# removing constant and near-constant features, feature importance, removing highly correlated features, removing
# features used for engineering, added visualizations for highly correlated features and feature importance evaluation,
# select subgroups, transformed everything into functions, added a configuration script, enable and disable several
# steps, high reproducibility

# /!\ TO CONSIDER DEPENDING IF RUNNING ON HPC OR LOCAL MACHINES: /!\ #
# ------------------------------------------------------------------ #
# First have a look in the script configuration options section of part 1 and adapt if needed!

# Regarding HPC: verbose=0 | False, n_jobs=number of ip-engines, cache_size=200, exhaustive grid search intervals
# Regarding local machines: verbose=[1, 2] or True, n_jobs=-1, cache_size=2000, reduced grid search intervals
# Set parallel_method to ipyparallel to enable HPC client, and threading or multiprocess for local machines

# ON HPC: Check available modules, create python environment, install requirements, create directories, import data e.g.
# ./data/train_imputed.csv and ./data/test_imputed.csv, sync script, config, utils and launcher files.
# Run script on HPC using 'sbatch HPC_SVM_launcher.sh CBD_pipeline_SVM_HPC.py' after the configurations in
# CBDP_config.py are set to your needs.

# REQUIRED FILES: CBD_pipeline_SVM_HPC.py, CBDP_utils.py, CBDP_config.py, HPC_SVM_launcher.sh,
# requirements.txt, ./env/eli5/permutation_importance.py and ./env/mlxtend/evaluate/feature_importance_permutation.py
# and ./env/mlxtend/evaluate/__init__.py adapted for parallelization, shuffle_me.py for explanation of shuffle numbers

# Global session is saved for each kernel in '-global-save.pkl' file, the main script execution output is collected
# in the generated log file if running on HPC. Location: log/job_ID/code_jobID_execution.out

# /!\ CURRENT WARNINGS / ERRORS ENCOUNTERED: /!\ #
# ---------------------------------------------- #
# OutdatedPackageWarning: The package pingouin is out of date. Your version is 0.5.0, the latest is 0.5.1.
# Set the environment variable OUTDATED_IGNORE=1 to disable these warnings. SUPPRESSED, MINOR CHANGES NOT WORTH UPDATING

# ConvergenceWarning: Solver terminated early (max_iter=150000).  Consider pre-processing your data with StandardScaler
# or MinMaxScaler. IGNORED, HAPPENS WHEN USING ROBUST SCALER OR MAX ITER OF THE CLASSIFIER IS REACHED (SEE CONFIG)


########################################################################################################################
# ## PART 0: SCRIPT START IMPORTING LIBRARIES ##########################################################################
########################################################################################################################
################################################
# ## Importing libraries and configuration file
################################################
# Libraries
import argparse
import logging
import math
import matplotlib
import random
import sys
import warnings

import pandas as pd
from eli5.permutation_importance import get_score_importances
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
from joblib import cpu_count, register_parallel_backend
from mlxtend.evaluate import feature_importance_permutation
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# Control output printing destination (original = console)
orig_stdout = sys.__stdout__

# Starting the script and try to load the dependent files
print(f"\n################################################\n################################################")
# Pipeline logo
logo = '\n  _________________  ________         ______\n'\
       ' /  ______/   ___  \\/  ____  \\       /  ___  \\\n'\
       '/  /      |  /__/  /  /    |  | ___ /  /__/  /\n'\
       '|  |     /  ___  </  /    /  / /__//   _____/\n'\
       '\\  \\____/  /___\\  \\ /____/  /     /  /\n'\
       ' \\_____/__________/________/     /__/ v.01/25/2023 (M/d/y)\n'\
       '---=====================================---\n'\
       '  CLINICAL BIOMARKER DETECTION - PIPELINE\n\n'
print(logo)
print(f"For the documentation see the link below:\n"
      f"https://github.com/sysbiolux/Clinical_Biomarker_Detection#readme\n\n"
      f"Starting the Clinical Biomarker Detection Pipeline v.01/25/2023.\n\n")
# Loading the CBD-P utils file
print(f"******************************************\nLOADING DEPENDENT FILES:\n\nLoading the CBD-P utils file ...")
try:
    from source.CBDP_utils import *
    # import all CBDP related functions
    print("CBD-P utils file loaded successfully!\n")
except ImportError('CBD-P utils file could not be found or loaded correctly.'):
    exit()
# Loading the CBD-P configuration file
print(f"Loading the CBD-P configuration file ...")
try:
    from CBDP_config import *
    # import the related configurations
    print("CBD-P configuration file loaded successfully!\n")
except ImportError('CBD-P configuration file could not be found or loaded correctly.'):
    exit()


########################################################################################################################
# ## PART 1: DOUBLE CHECK IF CONFIGURATION VARIABLES ARE LEGAL #########################################################
########################################################################################################################
###################################################################################
# ## Safety measures for disabled parameters and total parameter dictionary update
###################################################################################
# TODO CLEAN SAFETY RESETS AND VARIABLE CHECKS
# Safety measure to surely reset removing highly correlated feature parameters if disabled, else restore
if not enable_rhcf:
    thresh_cramer, thresh_spearman, thresh_pbs = [None] * 3
else:
    thresh_cramer, thresh_spearman, thresh_pbs = thresh_cramer, thresh_spearman, thresh_pbs
# Safety measure for PCA versus LDA (currently only one can be selected)
if pca_tech in ('normal_pca', 'kernel_pca') and da_tech == 'lda' and enable_ft:
    # prioritize pca_tech over da_tech if both are given (currently only one allowed)
    pca_tech = pca_tech
    da_tech = ''
elif da_tech == 'lda' and enable_ft:
    pca_tech = ''
    da_tech = da_tech
elif da_tech == '' and pca_tech in ('normal_pca', 'kernel_pca') and enable_ft:
    pca_tech = pca_tech
    da_tech = da_tech
else:
    pca_tech = ''
    da_tech = ''
    print(f'**No continuous feature transformation technique selected or feature transformation is completely '
          f'disabled.\nStatus of feature transformation step: {enable_ft}**')
# Safety measure for kbest technique, reset to chi2 if not among recommendations or callable function
if kbest_tech not in ('chi2', '', 'cramer'):
    print('**Kbest select technique not among the recommended settings. '
          'Check if a callable score function was given.**')
    if not hasattr(kbest_tech, '__call__'):
        kbest_tech = 'chi2'
        print('**No callable score function given. K best is reset to chi2. Check config is this is not desired.**')
# Safety measure to destroy kbest_tech if ft is not enabled
if enable_ft:
    kbest_tech = kbest_tech
else:
    kbest_tech = ''
    print(f'**No categorical feature transformation technique selected or feature transformation is completely '
          f'disabled.\nStatus of feature transformation step: {enable_ft}**')
# Safety measure to surely reset feature transformation parameters  and dictionary if disabled, else restore
if not (enable_ft and pca_tech == 'normal_pca'):
    pca_lpsr = [None]
    # Update dictionary depending on enabled pipeline steps
    total_params_and_splits.update({'pca_lpsr': pca_lpsr})
else:
    pca_lpsr = pca_lpsr
if not (enable_ft and kbest_tech != ''):
    k_best_lpsr = [None]
    total_params_and_splits.update({'k_best_lpsr': k_best_lpsr})
else:
    k_best_lpsr = k_best_lpsr
# Safety measure to surely reset resampling technique if disabled, else restore
if not enable_resampling:
    resampling_tech = ''
else:
    resampling_tech = resampling_tech
# Safety measure to surely reset resampling parameters and dictionary if disabled, else restore
if resampling_tech != 'smote':
    k_neighbors_smote_lpsr = [None]
    # Update dictionary depending on enabled pipeline steps
    total_params_and_splits.update({'k_neighbors_smote_lpsr': k_neighbors_smote_lpsr})
else:
    k_neighbors_smote_lpsr = k_neighbors_smote_lpsr
# Safety measure to surely reset pipeline order if resampling and feature transformation are disabled
if not (enable_ft or enable_resampling):
    pipeline_order = 'FT and resampling disabled, only standardization applies'
else:
    pipeline_order = pipeline_order
# Safety measure to surely reset engineered input prefix if disabled
if not enable_engineered_input_removal:
    engineered_input_prefix = ''
else:
    engineered_input_prefix = engineered_input_prefix
# Safety measure to surely reset pca tech if disabled
if not enable_ft:
    pca_tech = ''
    pca_kernel_dict.update({key: [None] for key, items in pca_kernel_dict.items()})
    pca_lpsr = [None]
else:
    pca_tech, pca_lpsr = pca_tech, pca_lpsr
# Update kernel pca dictionary to total params and splits if kernel pca activated and reset SVM kernels to linear
if pca_tech == 'kernel_pca':
    kernels = ['linear']  # Change possible SVM kernels to linear if kPCA is on, so only linear SVC parameter are loaded
    non_linear_kernels = [None]
    gamma_psr, coef0_ps, degree_p = [[None]] * 3
    total_params_and_splits.update({'gamma_psr': gamma_psr,
                                    'coef0_ps': coef0_ps,
                                    'degree_p': degree_p})
    if 'poly' in kernel_pca_kernel_lpsr:
        total_params_and_splits.update(pca_kernel_dict)
    elif 'rbf' in kernel_pca_kernel_lpsr:
        total_params_and_splits.update(
            {k: v for k, v in pca_kernel_dict.items() if k not in ('kpca_coef0_lpsr', 'kpca_degree_lpsr')})
    elif 'sigmoid' in kernel_pca_kernel_lpsr:
        total_params_and_splits.update(
            {k: v for k, v in pca_kernel_dict.items() if k not in ['kpca_degree_lpsr']})
else:
    kernels = kernels
    non_linear_kernels = non_linear_kernels
    gamma_psr, coef0_ps, degree_p = gamma_psr, coef0_ps, degree_p
# Be sure that scaler tech is set, cannot be empty, thus reset to default if not set
if scaler_tech not in ('standard', 'minmax', 'robust'):
    scaler_tech = 'standard'
    warnings.warn("**Scaler technique was not set in the configuration file. Default 'standard' is loaded.**")
# If da_tech is set to 'lda', update total_params_and_splits
if enable_ft and da_tech == 'lda' and pca_tech == '':
    total_params_and_splits.update(lda_dict)
# Reset splitting feature if disabled
if not enable_data_split:
    split_feature = ''
else:
    split_feature = split_feature
# Reset subgroups to keep if disabled
if not enable_subgroups:
    subgroups_to_keep = 'all'
else:
    subgroups_to_keep = subgroups_to_keep
# Reset scorer if not among implemented possibilities
if scorer not in ('F.5', 'F1', 'F2', 'F5', 'roc_auc', 'accuracy', 'balanced_accuracy', 'matthews_corrcoef', 'dor'):
    scorer = 'accuracy'
    warnings.warn("**Scorer was not among the possible scores. Default 'accuracy' is loaded.**")
# Reset feature importance method settings
if not enable_feature_importance:
    feature_importance_method = ''
    enable_box_bar_plots = False
elif feature_importance_method not in ('sklearn', 'mlxtend', 'eli5', 'all'):
    feature_importance_method = 'all'
    warnings.warn("**Feature importance method not set correctly. Default 'all' is loaded.**")
else:
    enable_box_bar_plots, feature_importance_method = enable_box_bar_plots, feature_importance_method
# Reset box and bar plot settings dependent on the feature importance
if not enable_box_bar_plots:
    box_bar_figures = ''
elif box_bar_figures not in ('separated', 'combined'):
    box_bar_figures = 'combined'
    warnings.warn("**Plot setting for box and bar plots are not set correctly. Default 'combined' is loaded.**")
else:
    box_bar_figures = box_bar_figures
# check if target feature and positive/negative classes are given
for string in (output_feature, positive_class, negative_class):
    if len(string) == 0:
        raise TypeError("**One or more of the following information is missing in the configuration file to start the "
                        "pipeline: output_feature, positive_class, or negative_class. Got %s instead.**" % string)

# check if kernels and non linear kernels are properly defined
possible_non_linear_svm_kernels = ['poly', 'rbf', 'sigmoid']
tmp = []
for kern in kernels:
    if kern in possible_non_linear_svm_kernels:
        tmp.append(kern)


if set(non_linear_kernels) != set(tmp):
    non_linear_kernels = tmp if len(tmp) > 0 else [None]
else:
    non_linear_kernels = non_linear_kernels

if drop_or_pass_non_treated_features not in ('drop', 'passthrough'):
    drop_or_pass_non_treated_features = 'drop'
    warnings.warn("**Decision to drop or passthrough features in column-transformer that are not transformed is not "
                  "valid. Default 'drop' is loaded.**")

##################################
# ## Configuration variable check
##################################
# Variables check that should strictly be an integer
config_int = [seed, fix_font, imp_font, fig_max_open_warning, pandas_col_display_option, cache_size, grid_verbose,
              hard_iter_cap, splits, shuffle_all, shuffle_male, shuffle_female, n_jobs, tiff_figure_dpi]
if not all(isinstance(i, int) for i in config_int):
    raise TypeError('The following configured variables must be integers: seed, fix_font, imp_font, '
                    'fig_max_open_warning, pandas_col_display_option, cache_size, grid_verbose, hard_iter_cap, splits, '
                    'shuffle_all, shuffle_male, shuffle_female, n_jobs, tiff_figure_dpi. Got %s instead.' % config_int)
# Variables check that should strictly be floating values
config_float = [thresh_near_constant]
if not all(isinstance(i, float) for i in config_float):
    raise TypeError('The following configured variables must be float: thresh_near_constant. Got %s instead.'
                    % config_int)
# Variables check that should strictly be a directory or path
if not os.path.isdir(curr_dir):
    raise IOError('The current directory is not set or recognized as such. Got current directory: %s.' % curr_dir)
if not all(os.path.isfile(i) for i in [train_path, test_path]):
    raise FileNotFoundError("The given train and test set pathways are not set or can't be found. "
                            "Got train: %s and test: %s." % (train_path, test_path))
# Variables check that should strictly be a string
config_str = [plot_style, pipeline_order, output_feature, split_feature, decision_func_shape, parallel_method,
              resampling_tech, folder_prefix, pca_tech, da_tech, scaler_tech, scorer, feature_importance_method,
              box_bar_figures, negative_class, positive_class, kbest_tech, drop_or_pass_non_treated_features,
              sample_tagging_feature]
if not (all(isinstance(i, str) for i in config_str)):
    if not hasattr(kbest_tech, '__call__'):
        raise TypeError('The following configured variables must be single strings: plot_style, pipeline_order, '
                        'output_feature, split_feature, decision_func_shape, parallel_method, folder_prefix, pca_tech, '
                        'da_tech, scaler_tech, scorer, feature_importance_method, box_bar_figures, negative_class, '
                        'positive_class, kbest_tech, drop_or_pass_non_treated_features, sample_tagging_feature. '
                        'Got %s instead.' % config_str)
# Variables check that should strictly be a list of strings or str
if not (all(isinstance(i, str) for i in output_related) or isinstance(output_related, list)):
    raise TypeError('One or multiple of the configured output features were not recognized as str or list of str: '
                    'output_related. Got %s.' % output_related)
kernel_info = [kernels, non_linear_kernels, kernel_pca_kernel_lpsr]
if not (all(isinstance(i, list) for i in kernel_info) or all(isinstance(i, str) for i in kernel_info)):
    raise TypeError('One or multiple of the following configured kernel list information are not strings: kernels, '
                    'non_linear_kernels, kernel_pca_kernel_lpsr. Got %s.' % kernel_info)
# Variables check that should strictly be a boolean
config_bool = [enable_rhcf, enable_resampling, enable_ft, clf_verbose, additional_params, enable_feature_importance,
               enable_engineered_input_removal, enable_data_split, enable_subgroups, enable_box_bar_plots,
               linear_shuffle]
if not all(isinstance(i, bool) for i in config_bool):
    raise TypeError('The following configured variables must be boolean: enable_rhcf, enable_resampling, '
                    'enable_ft, clf_verbose, additional_params, enable_feature_importance, '
                    'enable_engineered_input_removal, enable_data_split, enable_subgroups, enable_box_bar_plots, '
                    'linear_shuffle. '
                    'Got %s instead.' % config_bool)
# Variables that could be str or tuple of str
config_tuples = [engineered_input_prefix, subgroups_to_keep, tag_threshold]
for i in config_tuples:
    if not (isinstance(i, tuple) or all(isinstance(k, str) for k in i)):
        raise TypeError('The following configured variable must be a tuple or str: engineered_input_prefix, '
                        'subgroups_to_keep, tag_threshold. Got %s instead.' % i)
# RHCF threshold variables check if remove highly correlated features is enabled
if enable_rhcf:
    config_thresh_tuple = [thresh_cramer, thresh_spearman, thresh_pbs]
    for i in config_thresh_tuple:
        if not isinstance(i, tuple):
            raise TypeError('The following configured variables must be tuples: '
                            'thresh_cramer, thresh_spearman, thresh_pbs. '
                            'Got %s instead.' % config_thresh_tuple)
        if isinstance(i[1], str) and i[1] == 'decimal':
            if not isinstance(i[0], float):
                raise TypeError('The following threshold variable for decimal cut-off must be float. Got %s '
                                'instead.' % i)
        elif isinstance(i[1], str) and i[1] == 'percentile':
            if not isinstance(i[0], int):
                raise TypeError('The following threshold variable for percentile cut-off must be int. Got %s '
                                'instead.' % i)
        elif i[1] not in ['decimal', 'percentile']:
            raise TypeError("One of the following threshold variable specifications is missing in the configured "
                            "variable: 'decimal' or 'percentile'. Got %s." % i)
# Check for the kernel params and split dict
config_dict = [additional_technique_params, additional_kernel_params, total_params_and_splits, pca_kernel_dict,
               lda_dict]
if not all(isinstance(i, dict) for i in config_dict) and not len(total_params_and_splits) > 0:
    raise TypeError('The following configured variable must be a dictionary (and above zero length if total_params..): '
                    'additional_technique_params, additional_kernel_params total_params_and_splits, pca_kernel_dict, '
                    'lda_dict. Got %s instead.' % config_dict)
# Check if all grid search parameters are legal
# Pipeline parameters
grid_search_clf_params = [regularization_lpsr, shrinking_lpsr, tolerance_lpsr, gamma_psr, degree_p, coef0_ps]
if not all(isinstance(i, list) for i in grid_search_clf_params):
    raise TypeError('The following configured variables must be lists of values or strings: regularization_lpsr, '
                    'shrinking_lpsr, tolerance_lpsr, gamma_psr, degree_p, coef0_ps. Got %s.' % grid_search_clf_params)
# Resampling parameters
if enable_resampling:
    grid_search_samples_params = [k_neighbors_smote_lpsr]
    if not all(isinstance(i, list) for i in grid_search_samples_params):
        raise TypeError('The following configured variable must be lists of values or strings: '
                        'k_neighbors_smote_lpsr. Got %s.' % grid_search_samples_params)
# Feature transformation parameters
if enable_ft:
    grid_search_features_params = [pca_lpsr, k_best_lpsr, kernel_pca_lpsr, kernel_pca_gamma_lpsr,
                                   kernel_pca_tol_lpsr, kernel_pca_degree_lpsr, kernel_pca_coef0_lpsr,
                                   lda_shrinkage_lpsr, lda_priors_lpsr, lda_components_lpsr, lda_tol_lpsr]
    if not all(isinstance(i, list) for i in grid_search_features_params):
        raise TypeError('The following configured variables must be lists of values or strings: '
                        'pca_lpsr, k_best_lpsr, kernel_pca_lpsr, kernel_pca_gamma_lpsr, kernel_pca_tol_lpsr, '
                        'kernel_pca_degree_lpsr, kernel_pca_coef0_lpsr, lda_shrinkage_lpsr, lda_priors_lpsr, '
                        'lda_components_lpsr, lda_tol_lpsr. Got %s' % grid_search_features_params)


########################################################################################################################
# ## PART 2: SETTING UP THE RESULTS FOLDER, AND HPC OPERABILITY ########################################################
########################################################################################################################
###################################################
# ## Creating the results folder and clear content
###################################################
# /!\ Only supported pipeline steps are available and modifications are needed if new technical steps are added
# Name of fully enabled pipeline would be:
# Data split DS, Subgroups SG, Remove Engineered Input REI, Remove Highly Correlated Features RHCF
# Random Under Sampler RUS/Synthetic Minority Over-sampling Technique SMOTE -> 1 step
# Standard Scaler ST/Robust Scaler RO/Min-max Scaler MI, PCA/Kernel PCA kPCA, Feature Transformation FT -> 1 step
# Feature Importance FI, Box Bar Plotting (BBP), Support Vector Machines SVM, High Performance Computing HPC
intermediate_dict = {'SG': enable_subgroups, 'DS': enable_data_split, 'REI': enable_engineered_input_removal,
                     'RHCF': enable_rhcf, 'RUS_SMOTE': enable_resampling, 'PCA-FT_kPCA-FT_LDA-FT': enable_ft,
                     'FI': enable_feature_importance, 'BBP': enable_box_bar_plots}

# Generating the folder intermediate name depending on enabled pipeline steps
folder_intermediate, tmp, tmp1 = '', '', ''
for key, items in intermediate_dict.items():
    if items:
        if key.__contains__('_') and not key.endswith('FT'):  # Resampling first
            tmp = (key.split('_')[0] if resampling_tech == 'rus' else key.split('_')[1])  # Get first or second tech
            folder_intermediate += ('-' + tmp if pipeline_order == 'samples->features' else '')  # Add if order allow it
        elif key.__contains__('_') and key.endswith('FT'):  # Next underscore bearing is FT
            tmp1 = f'{kbest_tech}KBEST-FT' if (kbest_tech in ('chi2',
                                                              'cramer') or hasattr(kbest_tech, '__call__')) else ''
            # Get first, second, or third tech (lda in case of third tech)
            tmp1 += ('-' + key.split('_')[0] if pca_tech == 'normal_pca' else
                     '-' + key.split('_')[1] if pca_tech == 'kernel_pca' else
                     '-' + key.split('_')[2] if da_tech == 'lda' else '')
            if pca_tech == 'kernel_pca' and len(kernel_pca_kernel_lpsr) < 3:
                for pca_kern in kernel_pca_kernel_lpsr:
                    tmp1 = pca_kern + (tmp1 if tmp1 != '' else tmp1)
            folder_intermediate += '-' + scaler_tech[0:2].upper()  # Before adding the FT tech, define & add scaler tech
            folder_intermediate += ('-' + tmp1 if tmp1 != '' and not tmp1.startswith('-') else tmp1)  # FT after scaler
        # Delay resampling tech insertion after FT if pipeline order allows it
        elif folder_intermediate.endswith('FT') and pipeline_order == 'features->samples':
            # As this happens in the following step 'FI' after FT, tmp must be placed in between
            folder_intermediate += (tmp + '-' + key if folder_intermediate.endswith('-') else '-' + tmp + '-' + key)
        else:  # For each true item if not yet called above
            folder_intermediate += '-' + key
    if key == 'PCA-FT_kPCA-FT_LDA-FT' and not enable_ft and pipeline_order == 'samples->features':
        # In case of disabled FT, standard scaler is applied and should also appear in the folder name relative to order
        folder_intermediate += '-' + scaler_tech[0:2].upper()
    elif key == 'PCA-FT_kPCA-FT_LDA-FT' and not enable_ft and pipeline_order == 'features->samples':
        folder_intermediate += '-' + scaler_tech[0:2].upper() + '-' + tmp + ('-' + tmp1 if tmp1 != '' else tmp1)
    # Swap Kbest with PCA/lda if both are used (no need to consider case of kPCA, will be detected with 'PCA' anyhow
    if sum([True for match in ['KBEST', 'PCA', 'LDA'] if match in folder_intermediate]) == 2:
        folder_intermediate = swap_words(folder_intermediate, f'{kbest_tech}KBEST',
                                         'kPCA' if pca_tech == 'kernel_pca' else 'PCA' if pca_tech == 'normal_pca' else
                                         'LDA' if da_tech == 'lda' else '')
    # -FT- might occur twice in the name if double transformation is selected, in that case remove the first -FT-
    if folder_intermediate.count('-FT-') > 1:
        folder_intermediate = folder_intermediate.replace('-FT', '', 1)

# Define the results folder name suffix based on if ipyparallel is activated (HPC-based)
folder_suffix = '-SVM' + ('-lin' if 'linear' in kernels and len(kernels) == 1
                          else '-non-lin' if 'linear' not in kernels else '-both-lin-and-non-lin')
if parallel_method == 'ipyparallel':
    folder_suffix += '-HPC'

# Final results folder will be a combination of given prefix, intermediate name, and HPC and classifier dependent suffix
folder_name = folder_prefix\
              + folder_intermediate\
              + folder_suffix\
              if not folder_prefix.endswith(('/', '\\')) else folder_prefix + folder_intermediate[1:] + folder_suffix

# Create the folder or clear the content of the final folder (if multiple) if already existing
if folder_prefix.__contains__('/'):
    # If a folder in folder is given, first check if the first folder exists, if not create it (do not clear if exists)
    tmp_dir = curr_dir
    for slash in range(folder_prefix.count('/')):
        if os.path.isdir(tmp_dir + '/' + folder_prefix.split('/')[slash]) is False:
            os.mkdir(tmp_dir + '/' + folder_prefix.split('/')[slash])
            tmp_dir += '/' + folder_prefix.split('/')[slash]
        else:
            tmp_dir += '/' + folder_prefix.split('/')[slash]

# Now as all pre folders are created, create the final results folder, if it already exists, append a 2-digit value
folder_name = folder_name + '_00'  # folder _00 until _99 equal 100 folders
if os.path.isdir(curr_dir + '/' + folder_name) is False:
    os.mkdir(curr_dir + '/' + folder_name)
else:
    files = os.listdir(curr_dir + '/' + os.path.split(folder_name)[0])
    size = len(folder_name)
    count = sum([folder_name.split('/')[-1][:-3] in f for f in files])
    if count >= 100:  # if there are 100 folders or more (00-99), add another digit to display _100 in worst cases
        folder_name = folder_name.replace(folder_name[size - 3:], f"{'%03d' % count}", 1)
    else:
        folder_name = folder_name.replace(folder_name[size - 3:], f"_{'%02d' % count}", 1)
    os.mkdir(curr_dir + '/' + folder_name)

###############################################################
# ## HPC parallelization preparations and n_jobs configuration
###############################################################
# Source: https://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/scikit-learn/
# Enable only if running with ipyparallel on HPC
if parallel_method != 'ipyparallel':
    client = None
    # Open output file to redirect output prints if running on local machine
    file_path = curr_dir + '/' + folder_name + '/' + 'CBD-P_output_raw.txt'
    print(f"As parallel method is not set to ipyparallel, it is assumed that you are running the experiment on "
          f"a local machine.\nTherefore, the raw output results of the pipeline will be flushed to the output file "
          f"at the following location:\n** {file_path.replace(chr(92), '/')} **\n\nIf the process is exiting for any "
          f"reason before completing the pipeline,\nplease use the following two lines to redirect the printouts to "
          f"your local console.\nsys.stdout.close()\nsys.stdout = orig_stdout\n")  # with chr(92) being a backslash
    print("\n******************************************")
    buff_size = 1
    sys.stdout = open(file_path, "w", buffering=buff_size)
else:
    # To know the location of the python script
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(FILE_DIR)

    # Prepare the logger
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")
    args = parser.parse_args()
    profile = args.profile
    logging.basicConfig(filename=os.path.join(FILE_DIR, profile + '.log'), filemode='w', level=logging.DEBUG)
    logging.info("number of CPUs found: {0}".format(cpu_count()))
    logging.info("args.profile: {0}".format(profile))

    # Prepare the engines
    client = Client(profile=profile)
    NB_WORKERS = int(os.environ.get("NB_WORKERS", 1))
    # wait for the engines
    client.wait_for_engines(NB_WORKERS)

    # The following command will make sure that each engine is running in the right working directory
    client[:].map(os.chdir, [FILE_DIR] * len(client))
    logging.info("c.ids :{0}".format(str(client.ids)))
    b_view = client.load_balanced_view()
    register_parallel_backend(parallel_method, lambda: IPythonParallelBackend(view=b_view))

# Configure number of jobs
if parallel_method == 'ipyparallel':
    n_jobs = len(client)  # Either len(client) on HPC or -1 on local machine
else:
    n_jobs = n_jobs  # Assume running on local machine with different parallel backend (e.g. threading)


########################################################################################################################
# ## PART 3: MAIN PART OF THE SCRIPT WITH EXECUTABLE CODE ##############################################################
########################################################################################################################
##################################
# ## Script configuration summary
##################################
# Kernel based dictionary of enabled parameters length
kernel_param_lengths = {}
for kern in kernels:
    tmp = []
    for key, items in total_params_and_splits.items():
        # all param lengths with underscore and containing the first letter of the corresponding kernel in the last part
        tmp.append(len(items) if key.__contains__('_') and key.split('_')[-1].__contains__(kern[0])
                   else 1 if key.__contains__('_') or items == [None] else items)
        if key.__contains__('_') and not key.split('_')[-1].__contains__(kern[0]):
            total_params_and_splits.update({key: [None]})
    kernel_param_lengths[kern] = tmp
# Number of total fits for each selected kernel
total_fits = {}
for kern in kernels:
    total_fits[str(kern)] = math.prod(kernel_param_lengths[kern])

# Print a summary of the selected script configurations
backslash = '\\'
newline = '\n'
print("******************************************\nSCRIPT CONFIGURATION SUMMARY OF VARIABLES:\n\n"
      f"Random number generator seed: {seed}\n"
      f"Fixed general figure font: {fix_font}\n"
      f"Font for feature importance figures: {imp_font}\n"
      f"Figure max open warning set to: {fig_max_open_warning}\n"
      f"Matplotlib figure style: {plot_style}\n"
      f"Number of displayed pandas columns: {pandas_col_display_option}\n"
      f"Selected figure tiff format dot-per-inches: {tiff_figure_dpi}\n\n"
      f"Current directory: {curr_dir.replace(backslash, '/')}\n"
      f"Folder name prefix for this analysis: {folder_prefix}\n"
      f"Training set absolute pathway: {train_path.replace(backslash, '/')}\n"
      f"Test set absolute pathway: {test_path.replace(backslash, '/')}\n"
      f"Target output feature: {output_feature}\n"
      f"Names selected for the positive and negative classes respectively: {positive_class, negative_class}\n"
      f"Features directly linked to the target: {output_related}\n\n"
      f"Feature and threshold used to tag specific samples: {sample_tagging_feature, tag_threshold}\n\n"
      f"Near-constant feature threshold: {thresh_near_constant}\n"
      f"Data set splitting enabled based on splitting feature: {enable_data_split, split_feature}\n"
      f"Prefix of engineered input features: {engineered_input_prefix}\n"
      f"Enabled analysis of data subgroups: {enable_subgroups}\n"
      f"The following subgroups will be analyzed: {subgroups_to_keep}\n"
      f"Selected kernels for the SVM classifier: {kernels}\n"
      f"With the following kernels being non linear: {non_linear_kernels}\n"
      f"Cache size (mb) and decision function shape: {cache_size, decision_func_shape}\n"
      f"Classifier and grid search verbosity: {clf_verbose, grid_verbose}\n"
      f"Hard cap stopping criterion (max_iter): {hard_iter_cap}\n"
      f"Number of stratified k fold CV split: {splits}\n"
      f"Scorer selected for the analysis: {scorer}\n"
      f"Feature importance shuffles for all, male and female data: {shuffle_all, shuffle_male, shuffle_female}\n"
      f"Feature importance by permutation in case of linear classification: {linear_shuffle}\n"
      f"Selected parallel backend method and number of jobs: {parallel_method, n_jobs}\n"
      f"Removing features used for feature engineering enabled and selected feature prefixes: "
      f"{enable_engineered_input_removal, engineered_input_prefix}\n"
      f"Removing highly correlated features (RHCF) step enabled: {enable_rhcf}\n"
      f"Correlation threshold for Cramer: {thresh_cramer}\n"
      f"Correlation threshold for Point Bi-serial: {thresh_pbs}\n"
      f"Correlation threshold for Spearman: {thresh_spearman}\n"
      f"Resampling strategy and selected technique enabled: {enable_resampling, resampling_tech}\n"
      f"Feature transformation (FT) step enabled: {enable_ft}\n"
      f"Select k best technique for categorical features: {kbest_tech if kbest_tech != '' and enable_ft else None}\n"
      f"Scaler technique for continuous variables selected: {scaler_tech + ' scaler'}\n"
      f"PCA technique selected: {pca_tech.replace('_', ' ') if pca_tech != '' and enable_ft else None}\n"
      f"If PCA is disabled, DA technique selected: {da_tech if da_tech != '' and enable_ft else None}\n"
      f"Feature importance methods and visualizations enabled: {enable_feature_importance, feature_importance_method}\n"
      f"Box and bar plotting enabled and selected method: {enable_box_bar_plots, box_bar_figures}\n"
      f"Order of steps in the pipeline if FT or resampling are enabled: {pipeline_order}\n"
      f"Decision to drop or pass through features that are not transformed: {drop_or_pass_non_treated_features}\n"
      f"Additional grid search parameters that are not directly supported: {additional_params}\n"
      f"Additional technique parameters: {additional_technique_params}\n"
      f"Additional kernel parameters: {additional_kernel_params}\n")
if enable_resampling & (resampling_tech == 'rus'):
    with_or_without_sampling = 'with rus'
elif resampling_tech == 'smote':
    with_or_without_sampling = 'with smote'
else:
    with_or_without_sampling = 'without'
scale_only = 'including ' + scaler_tech + ' scaling, '
info_pca = \
    (scale_only + (pca_tech.replace('_', ' ') + ', ' if pca_tech != '' else '') +
     (str(kbest_tech) + ' select k best, ' if kbest_tech != '' else ''))
info_da = (scale_only + (da_tech + ', ' if da_tech != '' else '') +
           (str(kbest_tech) + ' select k best, ' if kbest_tech != '' else ''))
for kern in kernels:
    print(f"Total fitting for {kern} kernel, with {splits} fold cross-validation, {'with' if enable_ft else 'without'} "
          f"feature transformation, "
          f"{info_pca if pca_tech != '' else info_da if da_tech != '' else scale_only}\n"
          f"{with_or_without_sampling} resampling, {'with' if enable_feature_importance else 'without'} feature "
          f"importance, and {'with' if additional_params else 'without'} additional\nnon-supported grid search "
          f"parameters: {total_fits[kern]}\n")
    if pca_tech == 'kernel_pca' and len(kernel_pca_kernel_lpsr) > 1:
        poly_fits = int(1/3 * total_fits[kern])
        rbf_fits = int(1/3 * (int((1/len(kernel_pca_degree_lpsr) * 1/len(kernel_pca_coef0_lpsr))*total_fits[kern])))
        sigmoid_fits = int(1/3 * (int((1/len(kernel_pca_degree_lpsr))*total_fits[kern])))
        print(f"With kernelPCA enabled, the above calculated fits are masking the fact that each PCA kernel accepts "
              f"different parameters.\nFor real, as 3 kernels are tested, only a third of above mentioned fits is "
              f"tested for the poly kernelPCA.\nThe rbf and sigmoid kernels do accept less parameters, and therefore "
              f"undergo less number of fits.\nThus, the total fits of this experiment are:\n"
              f"{int(poly_fits + rbf_fits + sigmoid_fits)} total fits, with {int(poly_fits)} poly fits, {int(rbf_fits)}"
              f" rbf fits, and {int(sigmoid_fits)} sigmoid fits.\n")
print(f"Overview of enabled grid search parameters:\n"
      f"{newline.join(f'{key}: {value}' for key, value in total_params_and_splits.items())}\n\n"
      f"Results folder pathway:\n{curr_dir.replace(backslash, '/') + '/' + folder_name}\n\n"
      f"******************************************\n")

########################################################################
# ## Apply random seed, plot style, and pandas configuration parameters
########################################################################
# Random seed
random.seed(seed)
np.random.seed(seed)
# Plot styles
plt.rcParams['font.size'] = fix_font
plt.style.use(plot_style)
# Suppress figure max open warning from matplotlib
plt.rcParams.update({'figure.max_open_warning': fig_max_open_warning})
# Pandas option for number of displayed columns
pd.set_option('display.max_columns', pandas_col_display_option)
# Ensure threading on multiple devices
matplotlib.use('Agg')

#####################################
# ## Loading the train and test data
#####################################
# Read in data as pandas dataframe
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# Print the first rows
print('The first five rows of the original training set are:\n', train.head(5))
print('\nThe first five rows of the original test set are:\n', test.head(5))
# Print the shapes
print('\nThe shape of the original train set is:\n', train.shape)
print('The shape of the original test set is:\n', test.shape)

######################################
# ## Features and labels preparations
######################################
# collecting indices of samples to be tagged if given
if sample_tagging_feature != '':
    if sample_tagging_feature not in train.columns or sample_tagging_feature not in test.columns:
        print(f'Could not find the respective sample tagging feature among the train and test data set: '
              f'{sample_tagging_feature}.')
    else:
        print(f'\nCollecting indices of samples satisfying the following sample tagging condition in the '
              f'test and train set: {sample_tagging_feature, tag_threshold}.')
        sample_tag_idx_train = eval('train[sample_tagging_feature]' + tag_threshold[0] + tag_threshold[1])
        sample_tag_idx_test = eval('test[sample_tagging_feature]' + tag_threshold[0] + tag_threshold[1])
        print(f'Collected indices of {sum(sample_tag_idx_train)} samples that satisfied the given condition in the '
              f'train set and {sum(sample_tag_idx_test)} samples in the test set.')
        if enable_data_split:
            # 1 is male, 2 is female
            sample_tag_idx_male_train = \
                eval('train[sample_tagging_feature][train["PM-sex"] == 1]' + tag_threshold[0] + tag_threshold[1])
            sample_tag_idx_male_test = \
                eval('test[sample_tagging_feature][test["PM-sex"] == 1]' + tag_threshold[0] + tag_threshold[1])
            sample_tag_idx_female_train = \
                eval('train[sample_tagging_feature][train["PM-sex"] == 2]' + tag_threshold[0] + tag_threshold[1])
            sample_tag_idx_female_test = \
                eval('test[sample_tagging_feature][test["PM-sex"] == 2]' + tag_threshold[0] + tag_threshold[1])
            print(f'Collected indices of {sum(sample_tag_idx_male_train)} male samples that satisfied the given '
                  f'condition in the train set and {sum(sample_tag_idx_male_test)} samples in the test set.')
            print(f'Collected indices of {sum(sample_tag_idx_female_train)} female samples that satisfied the given '
                  f'condition in the train set and {sum(sample_tag_idx_female_test)} samples in the test set.')
        else:
            sample_tag_idx_male_train, sample_tag_idx_male_test, \
                sample_tag_idx_female_train, sample_tag_idx_female_test = 4 * [None]
              
# Remove features that were used to calculate the selected output feature
print(f'\nRemoving the following {len(output_related)} output related features:\n{output_related}')
train = train.drop(columns=[out for out in output_related])
test = test.drop(columns=[out for out in output_related])

print(f'\nRetaining the following subgroups: {subgroups_to_keep}')
if enable_subgroups and subgroups_to_keep != 'all':
    tmp_feats = list(train.columns)
    feats_to_keep = \
        [tmp_feats[col] for col in range(len(tmp_feats)) if tmp_feats[col].startswith(subgroups_to_keep)]
    train = train.drop(columns=[group for group in tmp_feats if group not in feats_to_keep])
    test = test.drop(columns=[group for group in tmp_feats if group not in feats_to_keep])
    print('\nThe shape of the train set with selected subgroups including output feature is:\n', train.shape)
    print('The shape of the test set with selected subgroups including output feature is:\n', test.shape)

# Split the data based on the given split feature
if enable_data_split and split_feature in train.columns:
    print(f'\nSplitting the data based on {split_feature} ...')
    train_features, test_features, train_labels, test_labels, feature_list, \
        train_men_features, test_men_features, train_men_labels, test_men_labels, \
        train_female_features, test_female_features, train_female_labels, test_female_labels, \
        feature_list_wo_gender = separate_full_data(full_train=train, full_test=test,
                                                    target_feature=output_feature, splitting_feature=split_feature)
else:  # Continue with full data only, split feature will be turned to None
    print(f'\nFull data analysis without data splitting, either because this step is disabled or\nbecause the feature '
          f'to be split is no longer among the subgroups to keep.\nSubgroup activation: {enable_subgroups}, '
          f'selected subgroups: {subgroups_to_keep}.')
    train_features, test_features, train_labels, test_labels, feature_list = separate_full_data(
        full_train=train, full_test=test, target_feature=output_feature, splitting_feature=split_feature)
    # Pseudo call the below variables to be None to avoid IDE warnings
    train_men_features, test_men_features, train_men_labels, test_men_labels, \
        train_female_features, test_female_features, train_female_labels, test_female_labels, \
        feature_list_wo_gender = [None] * 9

# Print the binary counts and ratio of negative and positive classes in the train and test sets
print(f'\n{negative_class.capitalize()}/{positive_class.capitalize()} counts in the full train set:',
      np.bincount(train_labels), '\nratio:', round(np.bincount(train_labels)[0] / np.bincount(train_labels)[1], 3))
print(f'{negative_class.capitalize()}/{positive_class.capitalize()} counts in the full test set:',
      np.bincount(test_labels), '\nratio:', round(np.bincount(test_labels)[0] / np.bincount(test_labels)[1], 3))
if enable_data_split:
    print(f'\n{negative_class.capitalize()}/{positive_class.capitalize()} counts in the male train set:',
          np.bincount(train_men_labels),
          '\nratio:', round(np.bincount(train_men_labels)[0] / np.bincount(train_men_labels)[1], 3))
    print(f'{negative_class.capitalize()}/{positive_class.capitalize()} counts in the male test set:',
          np.bincount(test_men_labels),
          '\nratio:', round(np.bincount(test_men_labels)[0] / np.bincount(test_men_labels)[1], 3))
    print(f'\n{negative_class.capitalize()}/{positive_class.capitalize()} counts in the female train set:',
          np.bincount(train_female_labels),
          '\nratio:', round(np.bincount(train_female_labels)[0] / np.bincount(train_female_labels)[1], 3))
    print(f'{negative_class.capitalize()}/{positive_class.capitalize()} counts in the female test set:',
          np.bincount(test_female_labels),
          '\nratio:', round(np.bincount(test_female_labels)[0] / np.bincount(test_female_labels)[1], 3))

#####################################
# ## Removing engineered input (REI)
#####################################
# Print the starting shapes
print('\nThe shape of the full train set before pre-processing and without output feature is:\n', train_features.shape)
print('The shape of the full test set before pre-processing and without output feature is:\n', test_features.shape)
if enable_data_split:
    print('\nThe shape of the male train set before pre-processing and without output feature is:\n',
          train_men_features.shape)
    print('The shape of the male test set before pre-processing and without output feature is:\n',
          test_men_features.shape)
    print('\nThe shape of the female train set before pre-processing and without output feature is:\n',
          train_female_features.shape)
    print('The shape of the female test set before pre-processing and without output feature is:\n',
          test_female_features.shape)

if enable_engineered_input_removal:
    print("\nRemoving features that were used for feature engineering ...")
    # Grouped medications GM and grouped devices GD (VitDDef BF or VitD_zscore_merged BF will be removed with RHCF)
    engineered_input_feat = \
        [feature_list[col] for col in range(len(feature_list)) if feature_list[col].startswith(engineered_input_prefix)]

    # Remove those features from full data
    engineered_input_idx_all = [col for col in range(len(feature_list)) if feature_list[col] in engineered_input_feat]
    train_features = np.delete(train_features, engineered_input_idx_all, axis=1)
    test_features = np.delete(test_features, engineered_input_idx_all, axis=1)
    # Update feature list
    feature_list = [x for x in feature_list if x not in engineered_input_feat]

    # Remove those features from gender data
    if enable_data_split:
        engineered_input_sex = \
            [col for col in range(len(feature_list_wo_gender)) if feature_list_wo_gender[col] in engineered_input_feat]
        train_men_features = np.delete(train_men_features, engineered_input_sex, axis=1)
        train_female_features = np.delete(train_female_features, engineered_input_sex, axis=1)
        test_men_features = np.delete(test_men_features, engineered_input_sex, axis=1)
        test_female_features = np.delete(test_female_features, engineered_input_sex, axis=1)
        # Update feature list (at this point both genders still share feature_list_wo_gender)
        feature_list_male = [x for x in feature_list_wo_gender if x not in engineered_input_feat]
        feature_list_female = [x for x in feature_list_wo_gender if x not in engineered_input_feat]
    else:
        feature_list_male, feature_list_female = [None] * 2

    # Print summary
    print(f'The total of {len(engineered_input_feat)} sources of engineered features are removed to avoid high '
          f'correlation.\n')
    if len(engineered_input_feat) == 0:
        print(f'If this step is enabled ({enable_engineered_input_removal}) and 0 sources were removed, '
              f'this means that either no engineered prefix\nwas defined in the config script, '
              f'or the considered features were not among the subgroups defined to keep.\n')

else:  # If engineered features removal is disabled, male and female feature lists are still identical
    feature_list = feature_list
    if enable_data_split:
        feature_list_male = feature_list_wo_gender  # None if data splitting is removed
        feature_list_female = feature_list_wo_gender  # None if data splitting is removed
    else:
        feature_list_male, feature_list_female = [None] * 2


##############################################
# ## Checking for remaining constant features
##############################################
print("\nChecking and removing constant features in the training set ...\n")
# Full data
constant_all = check_constant_features(feature_list, train_features, 'full', nbr_splits=splits,
                                       near_constant_thresh=thresh_near_constant)
# Update the feature lists
feature_list = [feature_list[x] for x in range(len(feature_list)) if x not in constant_all]
# Remove those features
train_features = np.delete(train_features, constant_all, axis=1)
test_features = np.delete(test_features, constant_all, axis=1)

# Male and female data
if enable_data_split:
    print("\nChecking and removing constant features in the male training set ...\n")
    constant_male = check_constant_features(feature_list_male, train_men_features, 'male', nbr_splits=splits,
                                            near_constant_thresh=thresh_near_constant)
    print("\nChecking and removing constant features in the female training set ...\n")
    constant_female = check_constant_features(feature_list_female, train_female_features, 'female', nbr_splits=splits,
                                              near_constant_thresh=thresh_near_constant)
    # Update gender feature list
    feature_list_male = [feature_list_male[x] for x in range(len(feature_list_male)) if x not in constant_male]
    feature_list_female = [feature_list_female[x] for x in range(len(feature_list_female)) if x not in constant_female]
    # Remove those features
    train_men_features = np.delete(train_men_features, constant_male, axis=1)
    test_men_features = np.delete(test_men_features, constant_male, axis=1)
    train_female_features = np.delete(train_female_features, constant_female, axis=1)
    test_female_features = np.delete(test_female_features, constant_female, axis=1)

# Print the new shapes
print('\nThe shape of the full train set after pre-processing is:\n', train_features.shape)
print('The shape of the full test set after pre-processing is:\n', test_features.shape)
if enable_data_split:
    print('\nThe shape of the male train set after pre-processing is:\n', train_men_features.shape)
    print('The shape of the male test set after pre-processing is:\n', test_men_features.shape)
    print('\nThe shape of the female train set after pre-processing is:\n', train_female_features.shape)
    print('The shape of the female test set after pre-processing is:\n', test_female_features.shape)

###########################################################################
# ## Applying computation and removal of highly correlated features (RHCF)
###########################################################################
# If rhcf step is enabled, the different data type associations will be analyzed for their correlations
if enable_rhcf:
    # Begin of the RHCF application
    print(f"\nBeginning to remove highly correlated features (RHCF) with the following associations and thresholds:")
    print(f"Categorical-categorical association with corrected Cramer's V {thresh_cramer} threshold.")
    print(f"Continuous-continuous association with Spearman's Rank Order Correlation {thresh_spearman} threshold.")
    print(f"Categorical-continuous association with Point Bi-Serial Correlation {thresh_pbs} threshold.\n")

    # Full data
    # Get the initial categorical and continuous indices
    continuous_idx, categorical_idx = get_cat_and_cont(train_features, test_features)
    if len(continuous_idx) < 1:
        print('The correlation step for continuous-related associations must be skipped because there are no '
              'continuous features left in the data sets after potentially removing subgroups and engineered features.')
    elif len(categorical_idx) < 1:
        print('The correlation step for categorical-related associations must be skipped because there are no '
              'categorical features left in the data sets after potentially removing subgroups '
              'and engineered features.')

    print('Starting to analyze and remove highly correlated features in the full data set ...\n')

    # Corrected Cramer's V correlation
    if len(categorical_idx) > 1:
        cramer_res, cat_to_drop, cramer_set = applied_cat_rhcf(
            parallel_meth=parallel_method, training_features=train_features, features_list=feature_list,
            categorical=categorical_idx, n_job=n_jobs, cramer_threshold=thresh_cramer, directory=curr_dir,
            folder=folder_name, datatype='full')
        # Heatmap of the cramer matrix (saving process inside function)
        cramer_heatmap(cramer_res, thresh_cramer, 'full', categorical_idx, folder_name, tiff_figure_dpi)
    else:
        cat_to_drop, cramer_set = [], set()

    # Spearman correlation
    if len(continuous_idx) > 1:
        spearman_res, cont_to_drop, spearman_set = applied_cont_rhcf(training_features=train_features,
                                                                     features_list=feature_list,
                                                                     continuous=continuous_idx,
                                                                     spearman_threshold=thresh_spearman,
                                                                     directory=curr_dir, folder=folder_name,
                                                                     datatype='full')
        # Heatmap of the spearman matrix (saving process inside function)
        spearman_heatmap(spearman_res, thresh_spearman, 'full', continuous_idx, folder_name, tiff_figure_dpi)
    else:
        cont_to_drop, spearman_set = [], set()
    # General data update after continuous and categorical correlation features were identified
    rem_train, rem_test, rem_feat, rem_idx = drop_and_update_correlated_data(
        continuous_to_drop=cont_to_drop, categorical_to_drop=cat_to_drop, training_set=train_features,
        test_set=test_features, features_list=feature_list)

    # Point bi-serial correlation
    if len(categorical_idx) >= 1 and len(continuous_idx) >= 1:
        longest, res_pb_r, res_pb_pv, pbs_to_drop, pbs_set, rem_cont, rem_cat = applied_cat_cont_rhcf(
            parallel_meth=parallel_method, training_features=train_features, cont_before_rhcf=continuous_idx,
            cat_before_rhcf=categorical_idx, features_list=feature_list, feat_after_rhcf=rem_feat,
            feat_idx_after_rhcf=rem_idx, n_job=n_jobs, pbs_threshold=thresh_pbs, directory=curr_dir, folder=folder_name,
            datatype='full')
        # Heatmap of the point bi-serial matrix (saving process inside function)
        pbs_heatmap(res_pb_r, thresh_pbs, 'full', rem_cat, rem_cont, longest, folder_name, tiff_figure_dpi)
    else:
        pbs_to_drop, rem_cat, rem_cont, pbs_set = [], [], [], set()

    # Generate the final remaining training set, test set, and feature list
    final_train_features, final_test_features, final_feature_list = final_drop_and_update(
        point_bs_to_drop=pbs_to_drop, training_set=rem_train, test_set=rem_test, features_list=rem_feat)

    # RHCF summary prints and general data update
    original_train, original_test, original_feat, train_features, test_features, feature_list = rhcf_update_summary(
        training_features=train_features, testing_features=test_features, features_list=feature_list,
        fin_train=final_train_features, fin_test=final_test_features, fin_feat=final_feature_list, datatype='full',
        cat_drop=cat_to_drop, cont_drop=cont_to_drop, pbs_drop=pbs_to_drop, cont_idx=continuous_idx,
        cat_idx=categorical_idx, remain_cont=rem_cont, remain_cat=rem_cat, cramer_threshold=thresh_cramer,
        spearman_threshold=thresh_spearman, pbs_threshold=thresh_pbs, remaining_train=rem_train,
        remaining_test=rem_test)

    # Male data
    if enable_data_split:
        # Get the initial categorical and continuous indices
        continuous_idx_male, categorical_idx_male = get_cat_and_cont(train_men_features, test_men_features)
        if len(continuous_idx_male) < 1:
            print('The correlation step for continuous-related associations must be skipped because there are no '
                  'continuous features left in the data sets after potentially removing subgroups and engineered '
                  'features.')
        elif len(categorical_idx_male) < 1:
            print('The correlation step for categorical-related associations must be skipped because there are no '
                  'categorical features left in the data sets after potentially removing subgroups '
                  'and engineered features.')

        print('Starting to analyze and remove highly correlated features in the male data set ...\n')

        # Corrected Cramer's V correlation
        if len(categorical_idx_male) > 1:
            cramer_res_male, cat_to_drop_male, cramer_set_male = applied_cat_rhcf(
                parallel_meth=parallel_method, training_features=train_men_features, features_list=feature_list_male,
                categorical=categorical_idx_male, n_job=n_jobs, cramer_threshold=thresh_cramer, directory=curr_dir,
                folder=folder_name, datatype='male')
            # Heatmap of the male cramer matrix (saving process inside function)
            cramer_heatmap(cramer_res_male, thresh_cramer, 'male', categorical_idx_male, folder_name, tiff_figure_dpi)
        else:
            cat_to_drop_male, cramer_set_male = [], set()

        # Spearman correlation
        if len(continuous_idx_male) > 1:
            spearman_res_male, cont_to_drop_male, spearman_set_male = applied_cont_rhcf(
                training_features=train_men_features, features_list=feature_list_male, continuous=continuous_idx_male,
                spearman_threshold=thresh_spearman, directory=curr_dir, folder=folder_name, datatype='male')
            # Heatmap of the male spearman matrix (saving process inside function)
            spearman_heatmap(spearman_res_male, thresh_spearman, 'male', continuous_idx_male, folder_name,
                             tiff_figure_dpi)
        else:
            cont_to_drop_male, spearman_set_male = [], set()
        # General data update after continuous and categorical correlation features were identified
        rem_train_male, rem_test_male, rem_feat_male, rem_idx_male = drop_and_update_correlated_data(
            continuous_to_drop=cont_to_drop_male, categorical_to_drop=cat_to_drop_male, training_set=train_men_features,
            test_set=test_men_features, features_list=feature_list_male)

        # Point bi-serial correlation
        if len(categorical_idx_male) >= 1 and len(continuous_idx_male) >= 1:
            longest_male, res_pb_r_male, res_pb_pv_male, pbs_to_drop_male, pbs_set_male, \
                rem_cont_male, rem_cat_male = applied_cat_cont_rhcf(
                    parallel_meth=parallel_method, training_features=train_men_features,
                    cont_before_rhcf=continuous_idx_male, cat_before_rhcf=categorical_idx_male,
                    features_list=feature_list_male, feat_after_rhcf=rem_feat_male, feat_idx_after_rhcf=rem_idx_male,
                    n_job=n_jobs, pbs_threshold=thresh_pbs, directory=curr_dir, folder=folder_name, datatype='male')
            # Heatmap of the male point bi-serial matrix (saving process inside function)
            pbs_heatmap(res_pb_r_male, thresh_pbs, 'male', rem_cat_male, rem_cont_male, longest_male, folder_name,
                        tiff_figure_dpi)
        else:
            pbs_to_drop_male, rem_cont_male, rem_cat_male, pbs_set_male = [], [], [], set()

        # Generate the final remaining training set, test set, and feature list
        final_train_features_male, final_test_features_male, final_feature_list_male = final_drop_and_update(
            point_bs_to_drop=pbs_to_drop_male, training_set=rem_train_male, test_set=rem_test_male,
            features_list=rem_feat_male)
        # RHCF summary prints and general data update
        original_train_male, original_test_male, original_feat_male, train_men_features, test_men_features, \
            feature_list_male = rhcf_update_summary(
                training_features=train_men_features, testing_features=test_men_features,
                features_list=feature_list_male, fin_train=final_train_features_male, fin_test=final_test_features_male,
                fin_feat=final_feature_list_male, datatype='male', cat_drop=cat_to_drop_male,
                cont_drop=cont_to_drop_male, pbs_drop=pbs_to_drop_male, cont_idx=continuous_idx_male,
                cat_idx=categorical_idx_male, remain_cont=rem_cont_male, remain_cat=rem_cat_male,
                cramer_threshold=thresh_cramer, spearman_threshold=thresh_spearman, pbs_threshold=thresh_pbs,
                remaining_train=rem_train_male, remaining_test=rem_test_male)

        # Female data
        # Get the initial categorical and continuous indices
        continuous_idx_female, categorical_idx_female = get_cat_and_cont(train_female_features, test_female_features)
        if len(continuous_idx_female) < 1:
            print('The correlation step for continuous-related associations must be skipped because there are no '
                  'continuous features left in the data sets after potentially removing subgroups and engineered '
                  'features.')
        elif len(categorical_idx_female) < 1:
            print('The correlation step for categorical-related associations must be skipped because there are no '
                  'categorical features left in the data sets after potentially removing subgroups '
                  'and engineered features.')

        print('Starting to analyze and remove highly correlated features in the female data set ...\n')

        # Corrected Cramer's V correlation
        if len(categorical_idx_female) > 1:
            cramer_res_female, cat_to_drop_female, cramer_set_female = applied_cat_rhcf(
                parallel_meth=parallel_method, training_features=train_female_features,
                features_list=feature_list_female, categorical=categorical_idx_female, n_job=n_jobs,
                cramer_threshold=thresh_cramer, directory=curr_dir, folder=folder_name, datatype='female')
            # Heatmap of the female cramer matrix (saving process inside function)
            cramer_heatmap(cramer_res_female, thresh_cramer, 'female', categorical_idx_female, folder_name,
                           tiff_figure_dpi)
        else:
            cat_to_drop_female, cramer_set_female = [], set()

        # Spearman correlation
        if len(continuous_idx_female) > 1:
            spearman_res_female, cont_to_drop_female, spearman_set_female = applied_cont_rhcf(
                training_features=train_female_features, features_list=feature_list_female,
                continuous=continuous_idx_female, spearman_threshold=thresh_spearman,
                directory=curr_dir, folder=folder_name, datatype='female')
            # Heatmap of the female spearman matrix (saving process inside function)
            spearman_heatmap(spearman_res_female, thresh_spearman, 'female', continuous_idx_female, folder_name,
                             tiff_figure_dpi)
        else:
            cont_to_drop_female, spearman_set_female = [], set()

        # General data update after continuous and categorical correlation features were identified
        rem_train_female, rem_test_female, rem_feat_female, rem_idx_female = drop_and_update_correlated_data(
            continuous_to_drop=cont_to_drop_female, categorical_to_drop=cat_to_drop_female,
            training_set=train_female_features, test_set=test_female_features, features_list=feature_list_female)

        # Point bi-serial correlation
        if len(categorical_idx_female) >= 1 and len(continuous_idx_female) >= 1:
            longest_female, res_pb_r_female, res_pb_pv_female, pbs_to_drop_female, pbs_set_female, rem_cont_female, \
                rem_cat_female = applied_cat_cont_rhcf(
                    parallel_meth=parallel_method, training_features=train_female_features,
                    cont_before_rhcf=continuous_idx_female, cat_before_rhcf=categorical_idx_female,
                    features_list=feature_list_female, feat_after_rhcf=rem_feat_female,
                    feat_idx_after_rhcf=rem_idx_female, n_job=n_jobs, pbs_threshold=thresh_pbs,
                    directory=curr_dir, folder=folder_name, datatype='female')
            # Heatmap of the female point bi-serial matrix (saving process inside function)
            pbs_heatmap(res_pb_r_female, thresh_pbs, 'female', rem_cat_female, rem_cont_female, longest_female,
                        folder_name, tiff_figure_dpi)
        else:
            pbs_to_drop_female, rem_cont_female, rem_cat_female, pbs_set_female = [], [], [], set()

        # Generate the final remaining training set, test set, and feature list
        final_train_features_female, final_test_features_female, final_feature_list_female = final_drop_and_update(
            point_bs_to_drop=pbs_to_drop_female, training_set=rem_train_female,
            test_set=rem_test_female, features_list=rem_feat_female)
        # RHCF summary prints and general data update
        original_train_female, original_test_female, original_feat_female, train_female_features, test_female_features,\
            feature_list_female = rhcf_update_summary(
                training_features=train_female_features, testing_features=test_female_features,
                features_list=feature_list_female, fin_train=final_train_features_female,
                fin_test=final_test_features_female, fin_feat=final_feature_list_female, datatype='female',
                cat_drop=cat_to_drop_female, cont_drop=cont_to_drop_female, pbs_drop=pbs_to_drop_female,
                cont_idx=continuous_idx_female, cat_idx=categorical_idx_female, remain_cont=rem_cont_female,
                remain_cat=rem_cat_female, cramer_threshold=thresh_cramer, spearman_threshold=thresh_spearman,
                pbs_threshold=thresh_pbs, remaining_train=rem_train_female, remaining_test=rem_test_female)

        # Venn diagram of the highly correlated features between the classes full, male female if data split enabled
        # Cramer's V correlation
        plot_venn(kernel='correlation', datatype="Cramer's V", set1=cramer_set, set2=cramer_set_male,
                  set3=cramer_set_female, tuple_of_names=("Full data", "Male data", "Female data"), label_fontsize=8,
                  feat_info='top Cramer correlated', weighted=False)
        plt.savefig(folder_name + f'/RHCF_cramer_features_venn.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # Spearman's Rank Order correlation
        plot_venn(kernel='correlation', datatype="Spearman's Rank Order", set1=spearman_set, set2=spearman_set_male,
                  set3=spearman_set_female, tuple_of_names=("Full data", "Male data", "Female data"), label_fontsize=8,
                  feat_info='top Spearman correlated', weighted=False)
        plt.savefig(folder_name + f'/RHCF_spearman_features_venn.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # Point Bi-serial correlation
        plot_venn(kernel='correlation', datatype="Point Bi-Serial", set1=pbs_set, set2=pbs_set_male,
                  set3=pbs_set_female, tuple_of_names=("Full data", "Male data", "Female data"), label_fontsize=8,
                  feat_info='top Point bi-serial correlated', weighted=False)
        plt.savefig(folder_name + f'/RHCF_point_biserial_features_venn.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()

    # End of removing highly correlated features
    print(f"The training and test sets as well as the feature lists are updated accordingly and ready to be passed "
          f"to the machine learning classification pipeline ...\n")

    ##############################################
    # Evaluation of the features that passed RHCF
    ##############################################
    # Draw correlation between remaining features and the output_feature (chi2, cramer if cat, else pbs)
    print(f"Evaluating correlation between remaining features and the target feature {output_feature} ...\n")

    # Get the continuous and categorical idx in full, male and female data after RHCF
    continuous_idx, categorical_idx = get_cat_and_cont(train_features, test_features)
    if enable_data_split:
        continuous_idx_male, categorical_idx_male = get_cat_and_cont(train_men_features, test_men_features)
        continuous_idx_female, categorical_idx_female = get_cat_and_cont(train_female_features, test_female_features)
    else:
        continuous_idx_male, categorical_idx_male, continuous_idx_female, categorical_idx_female = [None] * 4

    # Plot the correlations of features remaining after RHCF with the output feature
    if len(categorical_idx) >= 1 and len(continuous_idx) >= 1:
        possible_correlations = ['cramer', 'pbs'] if kbest_tech == 'cramer' else ['chi', 'pbs']
    elif len(categorical_idx) >= 1 and len(continuous_idx) == 0:
        possible_correlations = ['cramer'] if kbest_tech == 'cramer' else ['chi']
    else:
        possible_correlations = ['pbs']
        # start plotting the various correlation plots if valid
    for corr in possible_correlations:
        # mixed
        draw_corr_after_rhcf(train_features, train_labels, feature_list, categorical_idx, output_feature, corr)
        plt.savefig(folder_name + f'/full_RHCF_remaining_correlation_{corr}.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        if enable_data_split:
              # male
              draw_corr_after_rhcf(train_men_features, train_men_labels, feature_list_male, categorical_idx_male,
                                   output_feature, corr)
              plt.savefig(folder_name + f'/male_RHCF_remaining_correlation_{corr}.tiff', bbox_inches='tight',
                          dpi=tiff_figure_dpi)
              plt.close()
              # female
              draw_corr_after_rhcf(train_female_features, train_female_labels, feature_list_female,
                                   categorical_idx_female, output_feature, corr)
              plt.savefig(folder_name + f'/female_RHCF_remaining_correlation_{corr}.tiff', bbox_inches='tight',
                          dpi=tiff_figure_dpi)
              plt.close()

##########################################################################################
# ## Plot PCA or LDA of the continuous features if one of them is selected as transformer
##########################################################################################
# Get the continuous and categorical idx in full, male and female data in case RHCF would be disabled
continuous_idx, categorical_idx = get_cat_and_cont(train_features, test_features)
if enable_data_split:
    continuous_idx_male, categorical_idx_male = get_cat_and_cont(train_men_features, test_men_features)
    continuous_idx_female, categorical_idx_female = get_cat_and_cont(train_female_features, test_female_features)
else:
    continuous_idx_male, categorical_idx_male, continuous_idx_female, categorical_idx_female = [None] * 4

print(f"Plotting PCA or LDA analysis of remaining continuous features depending on transformation technique "
      f"and colorize by {output_feature} ...\n")
# if PCA
if pca_tech == 'normal_pca':
    # mixed
    plot_PCA(train_features=train_features, train_labels=train_labels, col_idx=continuous_idx, color_by=output_feature,
             title='Mixed BASE-II PCA analysis of continuous features that passed RHCF', comp=15, scaler=scaler_tech)
    plt.savefig(folder_name + f'/full_{pca_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                dpi=tiff_figure_dpi)
    plt.close()
    if enable_data_split:
        # male
        plot_PCA(train_features=train_men_features, train_labels=train_men_labels, col_idx=continuous_idx_male,
                 color_by=output_feature, title='Male BASE-II PCA analysis of continuous features that passed RHCF',
                 comp=15, scaler=scaler_tech)
        plt.savefig(folder_name + f'/male_{pca_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        # female
        plot_PCA(train_features=train_female_features, train_labels=train_female_labels, col_idx=continuous_idx_female,
                 color_by=output_feature, title='Female BASE-II PCA analysis of continuous features that passed RHCF',
                 comp=15, scaler=scaler_tech)
        plt.savefig(folder_name + f'/female_{pca_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
# if LDA
if da_tech == 'lda':
    # mixed
    plot_LDA(train_features=train_features, train_labels=train_labels, col_idx=continuous_idx, color_by=output_feature,
             title='Mixed BASE-II LDA analysis of continuous features that passed RHCF', scaler=scaler_tech)
    plt.savefig(folder_name + f'/full_{da_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                dpi=tiff_figure_dpi)
    plt.close()
    if enable_data_split:
        # male
        plot_LDA(train_features=train_men_features, train_labels=train_men_labels, col_idx=continuous_idx_male,
                 color_by=output_feature, title='Male BASE-II LDA analysis of continuous features that passed RHCF',
                 scaler=scaler_tech)
        plt.savefig(folder_name + f'/male_{da_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        # female
        plot_LDA(train_features=train_female_features, train_labels=train_female_labels, col_idx=continuous_idx_female,
                 color_by=output_feature, title='Female BASE-II LDA analysis of continuous features that passed RHCF',
                 scaler=scaler_tech)
        plt.savefig(folder_name + f'/female_{da_tech}_plot_after_RHCF.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
              
#########################################################################
# ## Machine learning preparations including feature transformation (FT)
#########################################################################
# Print the definite shapes of data sets entering the classification pipeline
print('The shape of the full train set entering the pipeline is:\n', train_features.shape)
print('The shape of the full test set entering the pipeline is:\n', test_features.shape)
if enable_data_split:
    print('\nThe shape of the male train set entering the pipeline is:\n', train_men_features.shape)
    print('The shape of the male test set entering the pipeline is:\n', test_men_features.shape)
    print('\nThe shape of the female train set entering the pipeline is:\n', train_female_features.shape)
    print('The shape of the female test set entering the pipeline is:\n', test_female_features.shape)

# Initialize pipeline steps for continuous and categorical features
print(f'\nPreparing instances for the machine learning classification pipeline ...\n')

# Initialize resampling method if enabled
sampler = 'passthrough'  # In case resampling is disabled
if enable_resampling:
    if resampling_tech == 'rus':
        sampler = RandomUnderSampler(sampling_strategy='majority', replacement=False, random_state=seed)  # RUS function
    elif resampling_tech == 'smote':  # SMOTENC if categorical are present else SMOTE
        # In case where pipeline order is features->samples, we let SMOTENC automatically find the categorical
        sampler = SMOTENC(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs,
                          categorical_features=categorical_idx if pipeline_order == 'samples->features' else
                          'find_for_me') if len(categorical_idx) != 0 and len(continuous_idx) != 0 else \
            SMOTEN(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs) if len(categorical_idx) != 0 else \
            SMOTE(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs)

if enable_data_split:
    sampler_male = 'passthrough'  # In case resampling is disabled
    sampler_female = sampler_male  # same as above, no changes
    if resampling_tech == 'rus':
        # male
        sampler_male = RandomUnderSampler(sampling_strategy='majority', replacement=False,
                                          random_state=seed)  # RUS function
        # female
        sampler_female = sampler_male  # no changes to above

    elif resampling_tech == 'smote':  # SMOTENC if categorical are present, else SMOTE
        # male
        # In case where pipeline order is features->samples, we let SMOTENC automatically find the categorical
        sampler_male = SMOTENC(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs,
                               categorical_features=categorical_idx_male if pipeline_order == 'samples->features' else
                               'find_for_me') if len(categorical_idx_male) != 0 and len(continuous_idx_male) != 0 else \
            SMOTEN(random_state=seed,
                   sampling_strategy='minority', n_jobs=n_jobs) if len(categorical_idx_male) != 0 else \
            SMOTE(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs)
        # female
        # In case where pipeline order is features->samples, we let SMOTENC automatically find the categorical
        sampler_female = SMOTENC(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs,
                                 categorical_features=categorical_idx_female if pipeline_order == 'samples->features'
                                 else 'find_for_me'
                                 ) if len(categorical_idx_female) != 0 and len(continuous_idx_female) != 0 else \
            SMOTEN(random_state=seed,
                   sampling_strategy='minority', n_jobs=n_jobs) if len(categorical_idx_female) != 0 else \
            SMOTE(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs)
else:
    sampler_male, sampler_female = [None] * 2

# Setting up scaler technique
if scaler_tech == 'minmax':
    scaler = MinMaxScaler()
elif scaler_tech == 'robust':
    scaler = RobustScaler()
else:
    scaler = StandardScaler()  # Default standard scaler for continuous features if none is selected

# Initialize FT step if enabled, pca/lda and select k best are added as column transformer steps to feature_filtration
if enable_ft:
    # PCA dimensionality reduction on continuous depending on pca technique
    pca = PCA(random_state=seed) if pca_tech == 'normal_pca' else KernelPCA(random_state=seed,
                                                                            n_jobs=n_jobs,
                                                                            max_iter=hard_iter_cap
                                                                            ) if pca_tech == 'kernel_pca' else None
    da = LinearDiscriminantAnalysis(solver='svd',
                                    store_covariance=True,
                                    covariance_estimator=None) if pca_tech == '' and da_tech == 'lda' else None
    # Chi squared k best selection on categorical
    k_filter = SelectKBest(score_func=chi2) if kbest_tech == 'chi2' else \
        SelectKBest(score_func=corr_cramer_kbest) if kbest_tech == 'cramer' else \
        SelectKBest(score_func=kbest_tech) if hasattr(kbest_tech, '__call__') else drop_or_pass_non_treated_features

    continuous_pipeline = Pipeline([('scaler', scaler), ('pca', pca)]) if pca_tech != '' else \
        Pipeline([('scaler', scaler), ('lda', da)]) if da_tech != '' else \
        Pipeline([('scaler', scaler)])
    # Setting up the feature transformation transformer for the full, male, and female data if lengths are above 1
    if len(continuous_idx) >= 1 and len(categorical_idx) >= 1:
        feature_trans = ColumnTransformer(transformers=[('continuous', continuous_pipeline, continuous_idx),
                                                        ('categorical', k_filter, categorical_idx)], n_jobs=n_jobs)
    elif len(continuous_idx) >= 1:
        feature_trans = ColumnTransformer(transformers=[
            ('continuous', continuous_pipeline, continuous_idx)], remainder=drop_or_pass_non_treated_features,
            n_jobs=n_jobs)
    elif len(categorical_idx) >= 1:
        feature_trans = ColumnTransformer(transformers=[('categorical', k_filter, categorical_idx)],
                                          remainder=drop_or_pass_non_treated_features, n_jobs=n_jobs)
    else:
        feature_trans = 'passthrough'

    if enable_data_split:
        if len(continuous_idx_male) >= 1 and len(categorical_idx_male) >= 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_male),
                ('categorical', k_filter, categorical_idx_male)], n_jobs=n_jobs)
        elif len(continuous_idx_male) >= 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_male)], remainder=drop_or_pass_non_treated_features,
                n_jobs=n_jobs)
        elif len(categorical_idx_male) >= 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('categorical', k_filter, categorical_idx_male)], remainder=drop_or_pass_non_treated_features,
                n_jobs=n_jobs)
        else:
            feature_trans_male = 'passthrough'

        if len(continuous_idx_female) >= 1 and len(categorical_idx_female) >= 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_female),
                ('categorical', k_filter, categorical_idx_female)], n_jobs=n_jobs)
        elif len(continuous_idx_female) >= 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_female)], remainder=drop_or_pass_non_treated_features,
                n_jobs=n_jobs)
        elif len(categorical_idx_male) >= 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('categorical', k_filter, categorical_idx_female)], remainder=drop_or_pass_non_treated_features,
                n_jobs=n_jobs)
        else:
            feature_trans_female = 'passthrough'
    else:
        feature_trans_male, feature_trans_female = [None] * 2

else:  # If FT is disabled, we only stick with the scaler for continuous features if length > 1, else skip
    feature_trans = ColumnTransformer(transformers=[
        ('continuous', scaler, continuous_idx)], remainder=drop_or_pass_non_treated_features,
        n_jobs=n_jobs) if len(continuous_idx) >= 1 else 'passthrough'
    if enable_data_split:
        feature_trans_male = ColumnTransformer(transformers=[('continuous', scaler, continuous_idx_male)],
                                               remainder=drop_or_pass_non_treated_features,
                                               n_jobs=n_jobs) if len(continuous_idx_male) >= 1 else 'passthrough'
        feature_trans_female = ColumnTransformer(transformers=[('continuous', scaler, continuous_idx_female)],
                                                 remainder=drop_or_pass_non_treated_features,
                                                 n_jobs=n_jobs) if len(continuous_idx_female) >= 1 else 'passthrough'
    else:
        feature_trans_male, feature_trans_female = [None] * 2

# Additional pipeline properties including scoring, stratified k fold and parameters for all kernel and possible steps
if scorer in ('balanced_accuracy', 'matthews_corrcoef'):
    test_weights = [1 if y == 0 else round(np.bincount(test_labels)[0] / np.bincount(test_labels)[1], 3) for y in test_labels]
    train_weights = [1 if y == 0 else round(np.bincount(train_labels)[0] / np.bincount(train_labels)[1], 3) for y in train_labels]
    if enable_data_split:
        test_weights_male = [1 if y == 0 else round(np.bincount(test_men_labels)[0] / np.bincount(test_men_labels)[1], 3) for y in test_men_labels]
        train_weights_male = [1 if y == 0 else round(np.bincount(train_men_labels)[0] / np.bincount(train_men_labels)[1], 3) for y in train_men_labels]
        test_weights_female = [1 if y == 0 else round(np.bincount(test_female_labels)[0] / np.bincount(test_female_labels)[1], 3) for y in test_female_labels]
        train_weights_female = [1 if y == 0 else round(np.bincount(train_female_labels)[0] / np.bincount(train_female_labels)[1], 3) for y in train_female_labels]
    else:
        test_weights_male, train_weights_male, test_weights_female, train_weights_female = [None] * 4
else:
    test_weights, train_weights, test_weights_male, train_weights_male, test_weights_female, train_weights_female \
        = [None] * 6

scoring_test, scoring_male, scoring_test_male, scoring_female, scoring_test_female = [None] * 5
if scorer == 'F5':
    scoring = make_scorer(fbeta_score, beta=5, average='macro')  # F beta 5
elif scorer == 'F2':
    scoring = make_scorer(fbeta_score, beta=2, average='macro')  # F beta 2
elif scorer == 'F.5':
    scoring = make_scorer(fbeta_score, beta=0.5, average='macro')  # F beta 0.5
elif scorer == 'F1':
    scoring = make_scorer(fbeta_score, beta=1, average='macro')  # F beta 1
elif scorer == 'roc_auc':
    scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=False,
                          needs_proba=False, average='macro')  # roc_auc scorer
elif scorer == 'balanced_accuracy':  # balanced accuracy scorer with sample_weight
    scoring = make_scorer(balanced_accuracy_score, sample_weight=train_weights)
    scoring_test = make_scorer(balanced_accuracy_score, sample_weight=test_weights)
    if enable_data_split:
        scoring_male = make_scorer(balanced_accuracy_score, sample_weight=train_weights_male)
        scoring_test_male = make_scorer(balanced_accuracy_score, sample_weight=test_weights_male)
        scoring_female = make_scorer(balanced_accuracy_score, sample_weight=train_weights_female)
        scoring_test_female = make_scorer(balanced_accuracy_score, sample_weight=test_weights_female)
elif scorer == 'matthews_corrcoef':  # matthews correlation coefficient with sample_weight
    scoring = make_scorer(matthews_corrcoef, sample_weight=train_weights)
    scoring_test = make_scorer(matthews_corrcoef, sample_weight=test_weights)
    if enable_data_split:
        scoring_male = make_scorer(matthews_corrcoef, sample_weight=train_weights_male)
        scoring_test_male = make_scorer(matthews_corrcoef, sample_weight=test_weights_male)
        scoring_female = make_scorer(matthews_corrcoef, sample_weight=train_weights_female)
        scoring_test_female = make_scorer(matthews_corrcoef, sample_weight=test_weights_female)
elif scorer == 'dor':
    scoring = make_scorer(dor_score)
else:
    scoring = make_scorer(accuracy_score)  # Accuracy as default if none is selected

# stratified k fold split of train, shuffle=F to avoid duplicates, no random seed needed if shuffle=False
skf = StratifiedKFold(n_splits=splits, shuffle=False)

# Setting up shared param dictionary between the selected kernels for grid search cross-validation
# SVM parameters
params = {'clf__C': regularization_lpsr,
          'clf__shrinking': shrinking_lpsr,
          'clf__tol': tolerance_lpsr,
          'clf__gamma': gamma_psr}
# Kernel PCA parameters
kernel_pca_params = {'pca__n_components': kernel_pca_lpsr,
                     'pca__kernel': kernel_pca_kernel_lpsr,
                     'pca__gamma': kernel_pca_gamma_lpsr,
                     'pca__tol': kernel_pca_tol_lpsr}
if 'poly' in kernel_pca_kernel_lpsr:
    kernel_pca_params.update({'pca__degree': kernel_pca_degree_lpsr,
                              'pca__coef0': kernel_pca_coef0_lpsr})
if 'sigmoid' in kernel_pca_kernel_lpsr:
    kernel_pca_params.update({'pca__coef0': kernel_pca_coef0_lpsr})
# LDA parameters
lda_dict_params = {'lda__shrinkage': lda_shrinkage_lpsr,
                   'lda__priors': lda_priors_lpsr,
                   'lda__n_components': lda_components_lpsr,
                   'lda__tol': lda_tol_lpsr}

# PCA n components and k best features are added to the parameter dict if the feature transformation step is enabled
# It is enough to check if the one from the complete data is passthrough, features don't change between the three sets
if enable_ft and pca_tech == 'normal_pca' and feature_trans != 'passthrough':
    params.update({'features__continuous__pca__n_components': pca_lpsr})
    # Load the corresponding FT parameters if kernel pca is activated
elif enable_ft and pca_tech == 'kernel_pca' and feature_trans != 'passthrough':
    params.update({'features__continuous__' + key: items for key, items in kernel_pca_params.items()})
elif enable_ft and da_tech == 'lda' and feature_trans != 'passthrough':
    params.update({'features__continuous__' + key: items for key, items in lda_dict_params.items()})

# Kbest related parameters for grid search
if enable_ft and kbest_tech != '' and feature_trans != 'passthrough':
    params.update({'features__categorical__k': k_best_lpsr})

# Resampler related parameters for grid search
if enable_resampling and (resampling_tech == 'smote'):
    params.update({'samples__k_neighbors': k_neighbors_smote_lpsr})

# In case other techniques with specific parameters should be added according to the configuration file
if additional_params:
    params.update(additional_technique_params)

# For loop for the different kernel-based and gender-specific combinations of classifier and grid search parameters
for kern in kernels:
    # Set up kernel-based classifier with fixed parameters, including max_iter=1.5mio as hard cap stopping criterion
    svm_clf = SVC(kernel=kern, cache_size=cache_size, verbose=clf_verbose, class_weight=None, max_iter=hard_iter_cap,
                  decision_function_shape=decision_func_shape, random_state=seed, break_ties=False, probability=True)

    # Create final gender-specific pipeline including resampling, feature transformation and classification
    # If resampling is deactivated, sampler will become 'passthrough' and is ignored
    if pipeline_order == 'samples->features':  # first sampler then feature transformation
        pipeline = Pipeline([('samples', sampler), ('features', feature_trans), ('clf', svm_clf)])
        if enable_data_split:
            pipeline_male = Pipeline([('samples', sampler_male), ('features', feature_trans_male), ('clf', svm_clf)])
            pipeline_female = \
                Pipeline([('samples', sampler_female), ('features', feature_trans_female), ('clf', svm_clf)])
        else:
            pipeline_male, pipeline_female = [None] * 2
    else:  # If pipeline order is reversed if resampling is deactivated (sampler will become 'passthrough' then)
        pipeline = Pipeline([('features', feature_trans), ('samples', sampler), ('clf', svm_clf)])
        if enable_data_split:
            pipeline_male = Pipeline([('features', feature_trans_male), ('samples', sampler_male), ('clf', svm_clf)])
            pipeline_female = \
                Pipeline([('features', feature_trans_female), ('samples', sampler_female), ('clf', svm_clf)])
        else:
            pipeline_male, pipeline_female = [None] * 2

    # Add kernel-based grid search parameters to the dictionary depending on the kernel
    if kern == 'poly':  # If poly kernel
        final_params = params.copy()
        final_params.update({'clf__degree': degree_p,
                             'clf__coef0': coef0_ps})
    elif kern == 'sigmoid':  # If sigmoid kernel
        final_params = params.copy()
        final_params.update({'clf__coef0': coef0_ps})
    elif kern == 'rbf':
        final_params = params.copy()  # No additional parameters for rbf kernel
    elif kern == 'linear':  # Case when kernel PCA is activated
        final_params = params.copy()
        final_params.pop('clf__gamma')  # Linear kernel does not require gamma
    else:  # Parameters remain kernel independent parameters
        final_params = params.copy()

    if additional_params:  # In case other kernels with specific parameters should be added given the config file
        final_params = params.copy()
        final_params.update(additional_kernel_params)

    k_m = 'only with select k best' if kbest_tech in ('chi2', 'cramer') or hasattr(kbest_tech, '__call__') else 'no'
    tech_m = f"{pca_tech if pca_tech !='' and enable_ft else da_tech if da_tech != '' and enable_ft else k_m}"

    print(f"******************************************\n{kern.capitalize()} SVM grid search parameter summary "
          f"{k_m + ' and ' + tech_m if enable_ft else 'with no enabled'} feature transformation technique:\n\n"
          f"{newline.join(f'{key}: {value}' for key, value in final_params.items())}\n\n"
          f"The full classification pipeline is set up as follows:\n\n{pipeline.named_steps}\n")
    if enable_data_split:
        print(f"The male classification pipeline is set up as follows:\n\n{pipeline_male.named_steps}\n\n"
              f"The female classification pipeline is set up as follows:\n\n{pipeline_female.named_steps}\n\n"
              f"******************************************")

    # Initiate the final gender-specific grid search cross-validation pipeline including all the above pre-settings
    grid_imba = GridSearchCV(pipeline, param_grid=final_params,
                             cv=skf,
                             scoring=scoring,
                             return_train_score=True,
                             verbose=grid_verbose,
                             n_jobs=n_jobs)
    # Grid search on male data
    if enable_data_split:
        grid_imba_male = GridSearchCV(pipeline_male, param_grid=final_params,
                                      cv=skf,
                                      scoring=scoring_male if scorer in ('balanced_accuracy',
                                                                         'matthews_corrcoef') else scoring,
                                      return_train_score=True,
                                      verbose=grid_verbose,
                                      n_jobs=n_jobs)
        # Grid search on female data
        grid_imba_female = GridSearchCV(pipeline_female, param_grid=final_params,
                                        cv=skf,
                                        scoring=scoring_female if scorer in ('balanced_accuracy',
                                                                             'matthews_corrcoef') else scoring,
                                        return_train_score=True,
                                        verbose=grid_verbose,
                                        n_jobs=n_jobs)
    else:
        grid_imba_male, grid_imba_female = [None] * 2

    ###############################################################
    # ## Beginning of the machine learning classification pipeline
    ###############################################################
    # Fit the model to the training data
    print(f'Starting the machine learning classification pipeline ...\n')
    with parallel_backend(parallel_method):
        grid_imba.fit(train_features, train_labels)
        if enable_data_split:
            grid_imba_male.fit(train_men_features, train_men_labels)
            grid_imba_female.fit(train_female_features, train_female_labels)

    print(f'\nEvaluating the best fitted model ...\n')
    # Training predictions (to demonstrate over-fitting)
    train_predictions = grid_imba.predict(train_features)
    train_probs = grid_imba.predict_proba(train_features)[:, 1]
    if enable_data_split:
        train_male_predictions = grid_imba_male.predict(train_men_features)
        train_male_probs = grid_imba_male.predict_proba(train_men_features)[:, 1]
        train_female_predictions = grid_imba_female.predict(train_female_features)
        train_female_probs = grid_imba_female.predict_proba(train_female_features)[:, 1]
    else:
        train_male_predictions, train_male_probs, train_female_predictions, train_female_probs = [None] * 4

    # Testing predictions (to determine performance)
    predictions = grid_imba.predict(test_features)
    probs = grid_imba.predict_proba(test_features)[:, 1]
    if enable_data_split:
        male_predictions = grid_imba_male.predict(test_men_features)
        male_probs = grid_imba_male.predict_proba(test_men_features)[:, 1]
        female_predictions = grid_imba_female.predict(test_female_features)
        female_probs = grid_imba_female.predict_proba(test_female_features)[:, 1]
    else:
        male_predictions, male_probs, female_predictions, female_probs = [None] * 4

    #######################
    # ## Model evaluations
    #######################
    # Calculate accuracy, F1 and AUC
    accuracy = accuracy_score(test_labels, predictions)
    f1_test = f1_score(test_labels, predictions, average='macro')
    f1_train = f1_score(train_labels, train_predictions, average='macro')
    auc = roc_auc_score(test_labels, predictions)
    correctly_classified = np.array(test_labels == predictions).sum()
    # male ony
    if enable_data_split:
        accuracy_male = accuracy_score(test_men_labels, male_predictions)
        f1_test_male = f1_score(test_men_labels, male_predictions, average='macro')
        f1_train_male = f1_score(train_men_labels, train_male_predictions, average='macro')
        auc_male = roc_auc_score(test_men_labels, male_predictions)
        correctly_classified_male = np.array(test_men_labels == male_predictions).sum()
        # female ony
        accuracy_female = accuracy_score(test_female_labels, female_predictions)
        f1_test_female = f1_score(test_female_labels, female_predictions, average='macro')
        f1_train_female = f1_score(train_female_labels, train_female_predictions, average='macro')
        auc_female = roc_auc_score(test_female_labels, female_predictions)
        correctly_classified_female = np.array(test_female_labels == female_predictions).sum()
    else:
        accuracy_male, f1_test_male, f1_train_male, auc_male, correctly_classified_male, \
            accuracy_female, f1_test_female, f1_train_female, auc_female, correctly_classified_female = [None] * 10

    ######################
    # ## Evaluation plots
    ######################
    # Readjust font size for roc_auc curve and confusion matrix
    if plt.rcParams['font.size'] != fix_font:
        plt.rcParams['font.size'] = fix_font

    # ROC_AUC curve of cross-validated grid search training and final test set full data
    print(f"Full data model evaluation for {kern.upper()} kernel:")
    # test set roc_auc
    evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels, 16, 'full')
    plt.savefig(folder_name + f'/full_{kern}_roc_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    # test set pr_auc
    plot_pr(probs, test_labels, 16, 'full')
    plt.savefig(folder_name + f'/full_{kern}_pr_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()

    # cross-validated training set roc_auc
    cv_roc_mean, cv_roc_std = plot_roc_validation('full', pd.DataFrame(train_features), pd.DataFrame(train_labels),
                                                  grid_imba.best_estimator_, reps=5, folds=splits, ax=plt)
    plt.savefig(folder_name + f'/full_{kern}_cross_validation_roc_auc.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    print(f"\nTraining ROC_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f"
          % (5, splits, float(cv_roc_mean), float(cv_roc_std)))
    # cross-validated training set pr_curve
    cv_pr_mean, cv_pr_std = plot_pr_validation('full', pd.DataFrame(train_features), pd.DataFrame(train_labels),
                                               grid_imba.best_estimator_, reps=5, folds=splits, ax=plt)
    plt.savefig(folder_name + f'/full_{kern}_cross_validation_pr_auc.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    print(f"\nTraining PR_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f"
          % (5, splits, float(cv_pr_mean), float(cv_pr_std)))

    if enable_data_split:
        # Male data
        print(f"\n\nMale data model evaluation for {kern.upper()} kernel:")
        # test set roc_auc
        evaluate_model(male_predictions, male_probs, train_male_predictions, train_male_probs, test_men_labels,
                       train_men_labels, 16, 'male')
        plt.savefig(folder_name + f'/male_{kern}_roc_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # test set pr_auc
        plot_pr(male_probs, test_men_labels, 16, 'male')
        plt.savefig(folder_name + f'/male_{kern}_pr_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        
        # cross-validated training set roc_auc
        cv_roc_mean_male, cv_roc_std_male = plot_roc_validation('male',
                                                                pd.DataFrame(train_men_features),
                                                                pd.DataFrame(train_men_labels),
                                                                grid_imba_male.best_estimator_,
                                                                reps=5, folds=splits, ax=plt)
        plt.savefig(folder_name + f'/male_{kern}_cross_validation_roc_auc.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print(f"\nTraining ROC_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f"
              % (5, splits, float(cv_roc_mean_male), float(cv_roc_std_male)))
        # cross-validated training set pr_auc
        cv_pr_mean_male, cv_pr_std_male = plot_pr_validation('male',
                                                             pd.DataFrame(train_men_features),
                                                             pd.DataFrame(train_men_labels),
                                                             grid_imba_male.best_estimator_,
                                                             reps=5, folds=splits, ax=plt)
        plt.savefig(folder_name + f'/male_{kern}_cross_validation_pr_auc.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print(f"\nTraining PR_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f"
              % (5, splits, float(cv_pr_mean_male), float(cv_pr_std_male)))
              
        # Female data
        print(f"\n\nFemale data model evaluation for {kern.upper()} kernel:")
        # test set roc_auc
        evaluate_model(female_predictions, female_probs, train_female_predictions, train_female_probs,
                       test_female_labels, train_female_labels, 16, 'female')
        plt.savefig(folder_name + f'/female_{kern}_roc_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # test set pr_auc
        plot_pr(female_probs, test_female_labels, 16, 'female')
        plt.savefig(folder_name + f'/female_{kern}_pr_auc_curve.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        
        # cross-validated training set roc_auc
        cv_roc_mean_female, cv_roc_std_female = plot_roc_validation('female',
                                                                    pd.DataFrame(train_female_features),
                                                                    pd.DataFrame(train_female_labels),
                                                                    grid_imba_female.best_estimator_,
                                                                    reps=5, folds=splits, ax=plt)
        plt.savefig(folder_name + f'/female_{kern}_cross_validation_roc_auc.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print(f"\nTraining ROC_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f"
              % (5, splits, float(cv_roc_mean_female), float(cv_roc_std_female)))
        # cross-validated training set pr_auc
        cv_pr_mean_female, cv_pr_std_female = plot_pr_validation('female',
                                                                 pd.DataFrame(train_female_features),
                                                                 pd.DataFrame(train_female_labels),
                                                                 grid_imba_female.best_estimator_,
                                                                 reps=5, folds=splits, ax=plt)
        plt.savefig(folder_name + f'/female_{kern}_cross_validation_pr_auc.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print(f"\nTraining PR_AUC in %d-times stratified %d-fold CV: %.3f +- %.3f\n"
              % (5, splits, float(cv_pr_mean_female), float(cv_pr_std_female)))
    else:
        cv_roc_mean_male, cv_roc_std_male, cv_roc_mean_female, cv_roc_std_female, cv_pr_mean_male, cv_pr_std_male, \
          cv_pr_mean_female, cv_pr_std_female = [None] * 8

    # Confusion matrix full data
    cm = confusion_matrix(test_labels, predictions)
    print(f"\nFull data confusion matrix for {kern.upper()} kernel:")
    plot_confusion_matrix(cm, classes=[negative_class.capitalize(), positive_class.capitalize()],
                          title='Confusion Matrix', normalize=True)
    plt.savefig(folder_name + f'/full_{kern}_cm.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    if enable_data_split:
        # Male data
        cm_male = confusion_matrix(test_men_labels, male_predictions)
        print(f"\nMale data confusion matrix for {kern.upper()} kernel:")
        plot_confusion_matrix(cm_male, classes=[negative_class.capitalize(), positive_class.capitalize()],
                              title='Confusion Matrix', normalize=True)
        plt.savefig(folder_name + f'/male_{kern}_cm.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # Female data
        cm_female = confusion_matrix(test_female_labels, female_predictions)
        print(f"\nFemale data confusion matrix for {kern.upper()} kernel:")
        plot_confusion_matrix(cm_female, classes=[negative_class.capitalize(), positive_class.capitalize()],
                              title='Confusion Matrix', normalize=True)
        plt.savefig(folder_name + f'/female_{kern}_cm.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
    else:
        cm_male, cm_female = [None] * 2

    # Turn the original feature lists into np arrays with technically just shorter names for later use
    features = np.array(feature_list)
    if enable_data_split:
        features_male = np.array(feature_list_male)
        features_female = np.array(feature_list_female)
    else:
        features_male, features_female = [None] * 2

    ##################################################################################
    # ## Updating feature list for full, male and female after feature transformation
    ##################################################################################
    # This part will only be printed if pca_tech is normal pca. Will probably be removed later
    # This section is only enabled if feature transformation step is enabled
    if enable_ft and (pca_tech == 'normal_pca' or kbest_tech != '') and feature_trans != 'passthrough':
        # Update feature list with the features that remained after feature transformation by PCA and SelectKBest
        # Full data
        new_features_full = update_features(predict_method=grid_imba, named_step='features',
                                            cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                            features_list=feature_list, cat_list=categorical_idx,
                                            cont_list=continuous_idx, pca_tech=pca_tech, cat_tech=kbest_tech)
        if enable_data_split:
            # Male data
            new_features_male = update_features(predict_method=grid_imba_male, named_step='features',
                                                cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                                features_list=feature_list_male, cat_list=categorical_idx_male,
                                                cont_list=continuous_idx_male, pca_tech=pca_tech, cat_tech=kbest_tech)
            # Female data
            new_features_female = update_features(predict_method=grid_imba_female, named_step='features',
                                                  cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                                  features_list=feature_list_female, cat_list=categorical_idx_female,
                                                  cont_list=continuous_idx_female, pca_tech=pca_tech,
                                                  cat_tech=kbest_tech)
        else:
            new_features_male, new_features_female = [None] * 2

        ####################################################################################################
        # ## Extracting the best selected features for later analysis of feature transformation performance
        ####################################################################################################
        # We retrieve the indices of best selected features to see which one were selected
        print(f'\nSelect K best and/or PCA (best feature per component) identified the following top features:\n'
              f'(Please note that the correct sorted top features by PCA summed weighted are printed afterwards.)\n')
        # Get the top feature of n components selected by the pipeline pca and select k best
        idx_of_best = [idx for idx in range(len(features)) if features[idx] in new_features_full]
        # Print the selected features
        print(f'Full data:\n{grid_imba.best_params_["features__categorical__k"] if kbest_tech != "" else "No"} k best '
              f'and '
              f'{grid_imba.best_params_["features__continuous__pca__n_components"] if pca_tech=="normal_pca" else "no"}'
              f' PCA components:\n{features[idx_of_best]}\n')
        # Inform about possible duplicates in case PCA number 1 top features are extracted for each selected component
        if pca_tech == 'normal_pca':
            inform_about_duplicates(new_features_full, idx_of_best, 'full')

        # In male and female
        if enable_data_split:
            cat_string = "features__categorical__k"
            cont_string = "features__continuous__pca__n_components"
            # male
            # Get the top feature of n components selected by the pipeline pca and select k best
            idx_of_best_male = [idx for idx in range(len(features_male)) if features_male[idx] in new_features_male]
            idx_of_best_female = \
                [idx for idx in range(len(features_female)) if features_female[idx] in new_features_female]
            # Print the selected features
            print(f'Male data:\n{grid_imba_male.best_params_[cat_string] if kbest_tech != "" else "No"} k best and '
                  f'{grid_imba_male.best_params_[cont_string] if pca_tech=="normal_pca" else "no"}'
                  f' PCA components:\n{features_male[idx_of_best_male]}\n')
            # Inform about possible duplicates in case the PCA top features are extracted for each selected component
            if pca_tech == 'normal_pca':
                inform_about_duplicates(new_features_male, idx_of_best_male, 'male')
            # female
            print(f'Female data:\n{grid_imba_female.best_params_[cat_string] if kbest_tech != "" else "No"} k best and '
                  f'{grid_imba_female.best_params_[cont_string] if pca_tech=="normal_pca" else "no"}'
                  f' PCA components:\n{features_female[idx_of_best_female]}\n')
            # Inform about possible duplicates in case the PCA top features are extracted for each selected component
            if pca_tech == 'normal_pca':
                inform_about_duplicates(new_features_female, idx_of_best_female, 'female')

    ########################################
    # ## Non linear feature importance (FI)
    ########################################
    # Readjust font size for feature importance figures
    if enable_feature_importance and (kern in non_linear_kernels or pca_tech == 'kernel_pca' or linear_shuffle):
        if plt.rcParams['font.size'] != imp_font:
            plt.rcParams['font.size'] = imp_font

        # FEATURE IMPORTANCE BY SKLEARN.INSPECTION
        if feature_importance_method in ('all', 'sklearn'):
            print("\nStarting feature importance permutation by SKLEARN:")
            # Full data
            print("In full data ...")
            with parallel_backend(parallel_method):
                perm_importance = permutation_importance(estimator=grid_imba.best_estimator_, X=train_features,
                                                         y=train_labels, random_state=seed, n_repeats=shuffle_all,
                                                         n_jobs=n_jobs)
            sorted_idx, sk_above_zero_imp = sorted_above_zero(importance_mean=perm_importance.importances_mean,
                                                              bar_cap=40)
            # Figure of most important features
            importance_plot(datatype='full', method='sklearn', kern=kern, idx_sorted=sorted_idx, features_list=features,
                            importance_mean=perm_importance.importances_mean, importance_above_zero=sk_above_zero_imp,
                            importance_std=perm_importance.importances_std)
            plt.savefig(
                folder_name + f'/full_{kern}_feature_importance_sklearn.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                print("In male data ...")
                with parallel_backend(parallel_method):
                    perm_importance_male = permutation_importance(grid_imba_male.best_estimator_, train_men_features,
                                                                  train_men_labels, random_state=seed,
                                                                  n_repeats=shuffle_male, n_jobs=n_jobs)
                sorted_idx_male, sk_above_zero_imp_male = sorted_above_zero(
                    importance_mean=perm_importance_male.importances_mean, bar_cap=40)
                # Figure of the most important features
                importance_plot(datatype='male', method='sklearn', kern=kern, idx_sorted=sorted_idx_male,
                                features_list=features_male,
                                importance_mean=perm_importance_male.importances_mean,
                                importance_above_zero=sk_above_zero_imp_male,
                                importance_std=perm_importance_male.importances_std)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                print("In female data ...")
                with parallel_backend(parallel_method):
                    perm_importance_female = permutation_importance(grid_imba_female.best_estimator_,
                                                                    train_female_features,
                                                                    train_female_labels, random_state=seed,
                                                                    n_repeats=shuffle_female, n_jobs=n_jobs)
                sorted_idx_female, sk_above_zero_imp_female = sorted_above_zero(
                    importance_mean=perm_importance_female.importances_mean, bar_cap=40)
                # Figure of the most important features
                importance_plot(datatype='female', method='sklearn', kern=kern, idx_sorted=sorted_idx_female,
                                features_list=features_female,
                                importance_mean=perm_importance_female.importances_mean,
                                importance_above_zero=sk_above_zero_imp_female,
                                importance_std=perm_importance_female.importances_std)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
            else:
                sorted_idx_male, sk_above_zero_imp_male, sorted_idx_female, sk_above_zero_imp_female = [None] * 4
        else:
            sorted_idx, sk_above_zero_imp, sorted_idx_male,\
                sk_above_zero_imp_male, sorted_idx_female, sk_above_zero_imp_female = [None] * 6

        # FEATURE IMPORTANCE BY ELI5 (modified ELI5 scripts)
        if feature_importance_method in ('all', 'eli5'):
            print("\nStarting feature importance permutation by ELI5:")
            # Full data
            print("In full data ...")
            with parallel_backend(parallel_method):
                perm_all, perm_mean = get_score_importances(score_func=grid_imba.best_estimator_, X=train_features,
                                                            y=train_labels, n_iter=shuffle_all, random_state=seed,
                                                            n_jobs=n_jobs)  # use classifier score
            sorted_idx_eli, el_above_zero_imp = sorted_above_zero(importance_mean=perm_mean, bar_cap=40)
            std_perm = np.std(perm_all, axis=1)
            # Figure of most important features
            importance_plot(datatype='full', method='eli5', kern=kern, idx_sorted=sorted_idx_eli,
                            features_list=features, importance_mean=perm_mean, importance_above_zero=el_above_zero_imp,
                            importance_std=std_perm)
            plt.savefig(folder_name + f'/full_{kern}_feature_importance_eli5.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                print("In male data ...")
                with parallel_backend(parallel_method):
                    perm_all_male, perm_mean_male = get_score_importances(score_func=grid_imba_male.best_estimator_,
                                                                          X=train_men_features, y=train_men_labels,
                                                                          n_iter=shuffle_male, random_state=seed,
                                                                          n_jobs=n_jobs)  # use classifier score
                sorted_idx_eli_male, el_above_zero_imp_male = sorted_above_zero(importance_mean=perm_mean_male,
                                                                                bar_cap=40)
                std_perm_male = np.std(perm_all_male, axis=1)
                # Figure of most important features
                importance_plot(datatype='male', method='eli5', kern=kern, idx_sorted=sorted_idx_eli_male,
                                features_list=features_male,
                                importance_mean=perm_mean_male,
                                importance_above_zero=el_above_zero_imp_male,
                                importance_std=std_perm_male)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_eli5.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                print("In female data ...")
                with parallel_backend(parallel_method):
                    perm_all_female, perm_mean_female = get_score_importances(
                        score_func=grid_imba_female.best_estimator_, X=train_female_features, y=train_female_labels,
                        n_iter=shuffle_female, random_state=seed, n_jobs=n_jobs)  # use classifier score
                sorted_idx_eli_female, el_above_zero_imp_female = sorted_above_zero(importance_mean=perm_mean_female,
                                                                                    bar_cap=40)
                std_perm_female = np.std(perm_all_female, axis=1)
                # Figure of most important features
                importance_plot(datatype='female', method='eli5', kern=kern, idx_sorted=sorted_idx_eli_female,
                                features_list=features_female, importance_mean=perm_mean_female,
                                importance_above_zero=el_above_zero_imp_female, importance_std=std_perm_female)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_eli5.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
            else:
                sorted_idx_eli_male, el_above_zero_imp_male,\
                    sorted_idx_eli_female, el_above_zero_imp_female = [None] * 4
        else:
            sorted_idx_eli, el_above_zero_imp, sorted_idx_eli_male, el_above_zero_imp_male,\
                sorted_idx_eli_female, el_above_zero_imp_female = [None] * 6

        # FEATURE IMPORTANCE BY MLXTEND (modified MLXTEND scripts)
        if feature_importance_method in ('all', 'mlxtend'):
            print("\nStarting feature importance permutation by MLXTEND:")
            # Full data
            print("In full data ...")
            with parallel_backend(parallel_method):
                imp_all, imp_vals = feature_importance_permutation(
                    X=train_features, y=train_labels, predict_method=grid_imba.best_estimator_, num_rounds=shuffle_all,
                    seed=seed, n_jobs=n_jobs)
            std = np.std(imp_all, axis=1)
            indices, ml_above_zero_imp = sorted_above_zero(importance_mean=imp_vals, bar_cap=40)
            # Figure of most important features
            importance_plot(datatype='full', method='mlxtend', kern=kern, idx_sorted=indices, features_list=features,
                            importance_mean=imp_vals, importance_above_zero=ml_above_zero_imp, importance_std=std)
            plt.savefig(folder_name + f'/full_{kern}_feature_importance_mlxtend.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                print("In male data ...")
                with parallel_backend(parallel_method):
                    imp_all_male, imp_vals_male = feature_importance_permutation(
                        X=train_men_features, y=train_men_labels, predict_method=grid_imba_male.best_estimator_,
                        num_rounds=shuffle_male, seed=seed, n_jobs=n_jobs)
                std_male = np.std(imp_all_male, axis=1)
                indices_male, ml_above_zero_imp_male = sorted_above_zero(importance_mean=imp_vals_male, bar_cap=40)
                # Figure of most important features
                importance_plot(datatype='male', method='mlxtend', kern=kern, idx_sorted=indices_male,
                                features_list=features_male,
                                importance_mean=imp_vals_male,
                                importance_above_zero=ml_above_zero_imp_male,
                                importance_std=std_male)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                print("In female data ...")
                with parallel_backend(parallel_method):
                    imp_all_female, imp_vals_female = feature_importance_permutation(
                        X=train_female_features, y=train_female_labels, predict_method=grid_imba_female.best_estimator_,
                        num_rounds=shuffle_female, seed=seed, n_jobs=n_jobs)
                std_female = np.std(imp_all_female, axis=1)
                indices_female, ml_above_zero_imp_female = sorted_above_zero(importance_mean=imp_vals_female,
                                                                             bar_cap=40)
                # Figure of most important features
                importance_plot(datatype='female', method='mlxtend', kern=kern, idx_sorted=indices_female,
                                features_list=features_female,
                                importance_mean=imp_vals_female,
                                importance_above_zero=ml_above_zero_imp_female,
                                importance_std=std_female)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
            else:
                indices_male, ml_above_zero_imp_male, indices_female, ml_above_zero_imp_female = [None] * 4
        else:
            indices, ml_above_zero_imp, indices_male, ml_above_zero_imp_male,\
                indices_female, ml_above_zero_imp_female = [None] * 6

        # Print summary
        if enable_feature_importance:
            print('\n******************************************\nFull feature importance summary:\n')
            if feature_importance_method in ('all', 'sklearn'):
                print('Top important features with sklearn:', features[sorted_idx[-sk_above_zero_imp:]][::-1], '\n')
            if feature_importance_method in ('all', 'eli5'):
                print('Top important features with eli5:', features[sorted_idx_eli[-el_above_zero_imp:]][::-1], '\n')
            if feature_importance_method in ('all', 'mlxtend'):
                print('Top important features with mlxtend:', features[indices[-ml_above_zero_imp:]][::-1], '\n')
            if enable_data_split:
                print('******************************************')
                print('Male feature importance summary:\n')
                if feature_importance_method in ('all', 'sklearn'):
                    print('Top important features with sklearn:',
                          features_male[sorted_idx_male[-sk_above_zero_imp_male:]][::-1], '\n')
                if feature_importance_method in ('all', 'eli5'):
                    print('Top important features with eli5:',
                          features_male[sorted_idx_eli_male[-el_above_zero_imp_male:]][::-1], '\n')
                if feature_importance_method in ('all', 'mlxtend'):
                    print('Top important features with mlxtend:',
                          features_male[indices_male[-ml_above_zero_imp_male:]][::-1], '\n')

                print('******************************************\nFemale feature importance summary:\n')
                if feature_importance_method in ('all', 'sklearn'):
                    print('Top important features with sklearn:',
                          features_female[sorted_idx_female[-sk_above_zero_imp_female:]][::-1], '\n')
                if feature_importance_method in ('all', 'eli5'):
                    print('Top important features with eli5:',
                          features_female[sorted_idx_eli_female[-el_above_zero_imp_female:]][::-1], '\n')
                if feature_importance_method in ('all', 'mlxtend'):
                    print('Top important features with mlxtend:',
                          features_female[indices_female[-ml_above_zero_imp_female:]][::-1], '\n')

        ####################################################
        # ## Evaluate non-linear feature importance methods
        ####################################################
        # Venn diagram of top important features, only possible with all methods selected
        if feature_importance_method == 'all':
            # Full data
            sklearn = set(features[sorted_idx[-sk_above_zero_imp:]])
            eli5 = set(features[sorted_idx_eli[-el_above_zero_imp:]])
            mlxtend = set(features[indices[-ml_above_zero_imp:]])

            plot_venn(kernel=kern, datatype='Full', set1=sklearn, set2=eli5, set3=mlxtend,
                      tuple_of_names=('sklearn', 'eli5', 'mlxtend'), label_fontsize=8,
                      feat_info='top important', weighted=True)
            plt.savefig(folder_name + f'/full_{kern}_feature_importance_venn_diagram.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                sklearn_male = set(features_male[sorted_idx_male[-sk_above_zero_imp_male:]])
                eli5_male = set(features_male[sorted_idx_eli_male[-el_above_zero_imp_male:]])
                mlxtend_male = set(features_male[indices_male[-ml_above_zero_imp_male:]])

                plot_venn(kernel=kern, datatype='Male', set1=sklearn_male, set2=eli5_male, set3=mlxtend_male,
                          tuple_of_names=('sklearn', 'eli5', 'mlxtend'), label_fontsize=8,
                          feat_info='top important', weighted=True)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_venn_diagram.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                sklearn_female = set(features_female[sorted_idx_female[-sk_above_zero_imp_female:]])
                eli5_female = set(features_female[sorted_idx_eli_female[-el_above_zero_imp_female:]])
                mlxtend_female = set(features_female[indices_female[-ml_above_zero_imp_female:]])

                plot_venn(kernel=kern, datatype='Female', set1=sklearn_female, set2=eli5_female, set3=mlxtend_female,
                          tuple_of_names=('sklearn', 'eli5', 'mlxtend'), label_fontsize=8,
                          feat_info='top important', weighted=True)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_venn_diagram.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

            # Scatter plot comparing the feature importance measuring effect between the three methods
            metrics = ["r", "CI95%", "p-val"]
            # Full data
            scatter_comparison(kernel=kern, datatype='Full', mean1=perm_importance.importances_mean, mean2=perm_mean,
                               mean3=imp_vals,
                               new_feat_idx=range(len(features)), metric_list=metrics)
            plt.savefig(folder_name + f'/full_{kern}_feature_importance_comparison.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                scatter_comparison(kernel=kern, datatype='Male', mean1=perm_importance_male.importances_mean,
                                   mean2=perm_mean_male, mean3=imp_vals_male,
                                   new_feat_idx=range(len(features_male)), metric_list=metrics)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_comparison.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                scatter_comparison(kernel=kern, datatype='Female', mean1=perm_importance_female.importances_mean,
                                   mean2=perm_mean_female, mean3=imp_vals_female,
                                   new_feat_idx=range(len(features_female)), metric_list=metrics)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_comparison.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

            # Scatter plot to check correlation between the three methods using r2 and linear equation
            # Full data
            scatter_r_squared(kernel=kern, datatype='Full', mean1=perm_importance.importances_mean, mean2=perm_mean,
                              mean3=imp_vals,
                              tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                              new_feat_idx=range(len(features)), fontsize=12)
            plt.savefig(folder_name + f'/full_{kern}_feature_importance_r2.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                scatter_r_squared(kernel=kern, datatype='Male', mean1=perm_importance_male.importances_mean,
                                  mean2=perm_mean_male, mean3=imp_vals_male,
                                  tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                                  new_feat_idx=range(len(features_male)), fontsize=12)
                plt.savefig(folder_name + f'/male_{kern}_feature_importance_r2.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                scatter_r_squared(kernel=kern, datatype='Female', mean1=perm_importance_female.importances_mean,
                                  mean2=perm_mean_female, mean3=imp_vals_female,
                                  tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                                  new_feat_idx=range(len(features_female)), fontsize=12)
                plt.savefig(folder_name + f'/female_{kern}_feature_importance_r2.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

        # Violin plot of the shuffling effect on the most important feature scores
        # Sklearn on full data
        if feature_importance_method in ('all', 'sklearn'):
            plot_violin(kern, 'Full sklearn', perm_importance.importances[sorted_idx[-sk_above_zero_imp:]],
                        features[sorted_idx[-sk_above_zero_imp:]], fontsize=7)
            plt.savefig(folder_name + f'/full_{kern}_violin_plot_sklearn.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male sklearn',
                            perm_importance_male.importances[sorted_idx_male[-sk_above_zero_imp_male:]],
                            features_male[sorted_idx_male[-sk_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/male_{kern}_violin_plot_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female sklearn',
                            perm_importance_female.importances[sorted_idx_female[-sk_above_zero_imp_female:]],
                            features_female[sorted_idx_female[-sk_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/female_{kern}_violin_plot_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
        # ELI5 on full data
        if feature_importance_method in ('all', 'eli5'):
            plot_violin(kern, 'Full eli5', perm_all[sorted_idx_eli[-el_above_zero_imp:]],
                        features[sorted_idx_eli[-el_above_zero_imp:]], fontsize=7)
            plt.savefig(folder_name + f'/full_{kern}_violin_plot_eli5.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male eli5', perm_all_male[sorted_idx_eli_male[-el_above_zero_imp_male:]],
                            features_male[sorted_idx_eli_male[-el_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/male_{kern}_violin_plot_eli5.tiff', bbox_inches='tight',
                            dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female eli5', perm_all_female[sorted_idx_eli_female[-el_above_zero_imp_female:]],
                            features_female[sorted_idx_eli_female[-el_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/female_{kern}_violin_plot_eli5.tiff', bbox_inches='tight',
                            dpi=tiff_figure_dpi)
                plt.close()
        # MLXTEND on full data
        if feature_importance_method in ('all', 'mlxtend'):
            plot_violin(kern, 'Full mlxtend',
                        imp_all[indices[-ml_above_zero_imp:]], features[indices[-ml_above_zero_imp:]],
                        fontsize=7)
            plt.savefig(folder_name + f'/full_{kern}_violin_plot_mlxtend.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male mlxtend', imp_all_male[indices_male[-ml_above_zero_imp_male:]],
                            features_male[indices_male[-ml_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/male_{kern}_violin_plot_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female mlxtend', imp_all_female[indices_female[-ml_above_zero_imp_female:]],
                            features_female[indices_female[-ml_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/female_{kern}_violin_plot_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

        #############################################################################
        # ## Box and bar plots of most important continuous and categorical features
        #############################################################################
        if enable_box_bar_plots:
            # Case of sklearn method
            if feature_importance_method in ('all', 'sklearn'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels,
                                 sorted_idx[-sk_above_zero_imp:], features, sk_above_zero_imp, output_feature,
                                 negative_class.capitalize(), positive_class.capitalize(), 'full', kern, folder_name,
                                 importance_method='sklearn', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                 fontsize=fix_font)

                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     sorted_idx_male[-sk_above_zero_imp_male:], features_male, sk_above_zero_imp_male,
                                     output_feature, negative_class.capitalize(), positive_class.capitalize(), 'male',
                                     kern, folder_name, importance_method='sklearn', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, sorted_idx_female[-sk_above_zero_imp_female:], features_female,
                                     sk_above_zero_imp_female, output_feature, negative_class.capitalize(),
                                     positive_class.capitalize(), 'female', kern, folder_name,
                                     importance_method='sklearn', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                     fontsize=fix_font)

                    # Case of eli5 method
            if feature_importance_method in ('all', 'eli5'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels,
                                 sorted_idx_eli[-el_above_zero_imp:], features, el_above_zero_imp, output_feature,
                                 negative_class.capitalize(), positive_class.capitalize(), 'full', kern, folder_name,
                                 importance_method='eli5', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                 fontsize=fix_font)
                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     sorted_idx_eli_male[-el_above_zero_imp_male:], features_male,
                                     el_above_zero_imp_male, output_feature, negative_class.capitalize(),
                                     positive_class.capitalize(), 'male', kern, folder_name, importance_method='eli5',
                                     tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, sorted_idx_eli_female[-el_above_zero_imp_female:],
                                     features_female, el_above_zero_imp_female, output_feature,
                                     negative_class.capitalize(), positive_class.capitalize(), 'female', kern,
                                     folder_name, importance_method='eli5', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
            # Case of mlxtend method
            if feature_importance_method in ('all', 'mlxtend'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels, indices[-ml_above_zero_imp:],
                                 features, ml_above_zero_imp, output_feature, negative_class.capitalize(),
                                 positive_class.capitalize(), 'full', kern, folder_name, importance_method='mlxtend',
                                 tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     indices_male[-ml_above_zero_imp_male:], features_male, ml_above_zero_imp_male,
                                     output_feature, negative_class.capitalize(), positive_class.capitalize(), 'male',
                                     kern, folder_name, importance_method='mlxtend', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, indices_female[-ml_above_zero_imp_female:], features_female,
                                     ml_above_zero_imp_female, output_feature, negative_class.capitalize(),
                                     positive_class.capitalize(), 'female', kern, folder_name,
                                     importance_method='mlxtend', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                     fontsize=fix_font)

            # Reset matplotlib style conflict with seaborn for next plots if data split is skipped
            plt.style.use(plot_style)

    ####################################
    # ## Linear feature importance (FI) (Only running when linear PCA, no PCA or LDA is combined with linear SVM kernel)
    ####################################
    # Readjust font size for importance figure of linear kernel
    if enable_feature_importance and kern == 'linear':
        # Only in case of linear SVM with normal pca or no pca
        if plt.rcParams['font.size'] != imp_font:
            plt.rcParams['font.size'] = imp_font

        print("\n******************************************\nProcessing feature importance with linear kernel...\n")
        # Full data
        print("In the full data...\n")
        lin_imp = grid_imba.best_estimator_.named_steps['clf'].coef_[0]
        lin_idx, lin_above_zero_imp = sorted_above_zero(importance_mean=lin_imp, bar_cap=40)
        # linear output features
        # in case of pca, do it by the function
        # in case of kernel pca, this step is skipped anyhow
        # in case of no FT, the .coef_ is only one vector with coefficients for each feature, thus we can combine the
        # lin_out_features as they enter the pipeline
        lin_out_features, sum_of_variance = \
            linear_svm_get_features(grid_imba.best_estimator_, lin_idx, categorical_idx, continuous_idx,
                                    features, lin_imp, 'pca') if pca_tech == 'normal_pca' else \
            (linear_svm_get_features(grid_imba.best_estimator_, lin_idx, categorical_idx, continuous_idx,
                                     features, lin_imp, 'kernel_pca'), [None]) if pca_tech == 'kernel_pca' else \
            (np.array(list(features[continuous_idx]) + list(features[categorical_idx])),
             [None]) if (pca_tech == '' and da_tech == '' and kbest_tech == '') or not enable_ft else \
            (linear_svm_get_features(grid_imba.best_estimator_, lin_idx, categorical_idx, continuous_idx, features,
                                     lin_imp, 'lda'), [None]) if da_tech == 'lda' else \
            (linear_svm_get_features(grid_imba.best_estimator_, lin_idx, categorical_idx, continuous_idx, features,
                                     lin_imp, 'none'), [None])  # should only be triggered if kbest is alone
        # print sum of variance of selected pca components if enabled
        if pca_tech == 'normal_pca':
            print(f"Full data sum of explained variance by the selected pca components equals {sum_of_variance}")
        # Use list comprehension to get the correct index for later box bar plot
        lin_out_real_idx_for_bbp = \
            [list(features).index(x.split(' (')[0]) for x in lin_out_features if x.split(' (')[0] in features]
        # In case of multiple features extracted from one single lda dimension, concatenate those features to one and
        # put it as label of the ld1 importance bar, which is usually the first one.
        if len(lin_out_features) != len(lin_idx):
            tmp = str()
            all_selected_lda_features = [s for s in lin_out_features if "LD #" in s]
            for pos in range(len(all_selected_lda_features)):
                if pos == len(all_selected_lda_features) - 1:
                    tmp += all_selected_lda_features[pos]
                else:
                    tmp += all_selected_lda_features[pos] + '\n'
            lin_out_features = np.array([tmp] + list(lin_out_features[len(all_selected_lda_features):]))
        # Figure of most important features
        importance_plot(datatype='full', method='SVM_coef', kern=kern, idx_sorted=lin_idx,
                        features_list=lin_out_features, importance_mean=lin_imp,
                        importance_above_zero=lin_above_zero_imp, importance_std=None)
        plt.savefig(folder_name + f'/full_{kern}_feature_importance.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print('Full data top important features with linear kernel (limited to top 40):\n\n',
              lin_out_features[lin_idx][::-1][:40], '\n')

        # male data
        if enable_data_split:
            print("In the male data...\n")
            lin_imp_male = grid_imba_male.best_estimator_.named_steps['clf'].coef_[0]
            lin_idx_male, lin_above_zero_imp_male = sorted_above_zero(importance_mean=lin_imp_male, bar_cap=40)
            lin_out_features_male, sum_of_variance_male = \
                linear_svm_get_features(grid_imba_male.best_estimator_, lin_idx_male, categorical_idx_male,
                                        continuous_idx_male, features_male,
                                        lin_imp_male, 'pca') if pca_tech == 'normal_pca' else \
                (linear_svm_get_features(grid_imba_male.best_estimator_, lin_idx_male, categorical_idx_male,
                                         continuous_idx_male, features_male, lin_imp_male, 'kernel_pca'),
                 [None]) if pca_tech == 'kernel_pca' else \
                (np.array(list(features_male[continuous_idx_male]) + list(features_male[categorical_idx_male])),
                 [None]) if (pca_tech == '' and da_tech == '' and kbest_tech == '') or not enable_ft else \
                (linear_svm_get_features(grid_imba_male.best_estimator_, lin_idx_male, categorical_idx_male,
                                         continuous_idx_male, features_male,
                                         lin_imp_male, 'lda'), [None]) if da_tech == 'lda' else \
                (linear_svm_get_features(grid_imba_male.best_estimator_, lin_idx_male, categorical_idx_male,
                                         continuous_idx_male, features_male, lin_imp_male, 'none'), [None])
            # print sum of variance of selected pca components if enabled
            if pca_tech == 'normal_pca':
                print(f"Male data sum of explained variance by the selected pca components equals "
                      f"{sum_of_variance_male}")
            # Use list comprehension to get the correct index
            lin_out_real_idx_for_bbp_male = \
                [list(features_male).index(x.split(' (')[0]) for x in lin_out_features_male
                 if x.split(' (')[0] in features_male]
            # In case of multiple features extracted from one single lda dimension, concatenate those features to one
            # and put it as label of the ld1 importance bar, which is usually the first one.
            if len(lin_out_features_male) != len(lin_idx_male):
                tmp = str()
                all_selected_lda_features_male = [s for s in lin_out_features_male if "LD #" in s]
                for pos in range(len(all_selected_lda_features_male)):
                    if pos == len(all_selected_lda_features_male) - 1:
                        tmp += all_selected_lda_features_male[pos]
                    else:
                        tmp += all_selected_lda_features_male[pos] + '\n'
                lin_out_features_male = np.array([tmp] +
                                                 list(lin_out_features_male[len(all_selected_lda_features_male):]))
            # Figure of most important features
            importance_plot(datatype='male', method='SVM_coef', kern=kern, idx_sorted=lin_idx_male,
                            features_list=lin_out_features_male, importance_mean=lin_imp_male,
                            importance_above_zero=lin_above_zero_imp_male, importance_std=None)
            plt.savefig(folder_name + f'/male_{kern}_feature_importance.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            print('Male data top important features with linear kernel (limited to top 40):\n\n',
                  lin_out_features_male[lin_idx_male][::-1][:40], '\n')

            # Female data
            print("In the female data...\n")
            lin_imp_female = grid_imba_female.best_estimator_.named_steps['clf'].coef_[0]
            lin_idx_female, lin_above_zero_imp_female = sorted_above_zero(importance_mean=lin_imp_female, bar_cap=40)
            lin_out_features_female, sum_of_variance_female = \
                linear_svm_get_features(grid_imba_female.best_estimator_, lin_idx_female,
                                        categorical_idx_female, continuous_idx_female, features_female,
                                        lin_imp_female, 'pca') if pca_tech == 'normal_pca' else \
                (linear_svm_get_features(grid_imba_female.best_estimator_, lin_idx_female, categorical_idx_female,
                                         continuous_idx_female, features_female, lin_imp_female, 'kernel_pca'),
                 [None]) if pca_tech == 'kernel_pca' else \
                (np.array(list(features_female[continuous_idx_female]) + list(features_female[categorical_idx_female])),
                 [None]) if (pca_tech == '' and da_tech == '' and kbest_tech == '') or not enable_ft else \
                (linear_svm_get_features(grid_imba_female.best_estimator_, lin_idx_female, categorical_idx_female,
                                         continuous_idx_female, features_female,
                                         lin_imp_female, 'lda'), [None]) if da_tech == 'lda' else \
                (linear_svm_get_features(grid_imba_female.best_estimator_, lin_idx_female, categorical_idx_female,
                                         continuous_idx_female, features_female, lin_imp_female, 'none'), [None])
            # print sum of variance of selected pca components if enabled
            if pca_tech == 'normal_pca':
                print(f"Female data sum of explained variance by the selected pca components equals "
                      f"{sum_of_variance_female}")
            # Use list comprehension to get the correct index
            lin_out_real_idx_for_bbp_female = \
                [list(features_female).index(x.split(' (')[0]) for x in lin_out_features_female
                 if x.split(' (')[0] in features_female]
            # In case of multiple features extracted from one single lda dimension, concatenate those features to one
            # and put it as label of the ld1 importance bar, which is usually the first one.
            if len(lin_out_features_female) != len(lin_idx_female):
                tmp = str()
                all_selected_lda_features_female = [s for s in lin_out_features_female if "LD #" in s]
                for pos in range(len(all_selected_lda_features_female)):
                    if pos == len(all_selected_lda_features_female) - 1:
                        tmp += all_selected_lda_features_female[pos]
                    else:
                        tmp += all_selected_lda_features_female[pos] + '\n'
                lin_out_features_female = \
                    np.array([tmp] + list(lin_out_features_female[len(all_selected_lda_features_female):]))
            # Figure of most important features
            importance_plot(datatype='female', method='SVM_coef', kern=kern, idx_sorted=lin_idx_female,
                            features_list=lin_out_features_female, importance_mean=lin_imp_female,
                            importance_above_zero=lin_above_zero_imp_female, importance_std=None)
            plt.savefig(folder_name + f'/female_{kern}_feature_importance.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            print('Female data top important features with linear kernel (limited to top 40):\n\n',
                  lin_out_features_female[lin_idx_female][::-1][:40], '\n')
        else:
            lin_idx_male, lin_above_zero_imp_male,\
                lin_idx_female, lin_above_zero_imp_female,\
                lin_out_features_male, lin_out_features_female,\
                lin_out_real_idx_for_bbp_male, lin_out_real_idx_for_bbp_female,\
                lin_imp_male, lin_imp_female = [None] * 10

        ############################################################
        # ## Box and bar plots in case of linear feature importance
        ############################################################
        if enable_box_bar_plots:
            # In case of LDA, there might be more interesting features than importance measures by SVM, and the box and
            # bar plot should therefore show these supplementary features as well. This can be identified if
            # lin_out_real_idx_for_bbp is larger than the lin_idx. this must also be considered for the split data case.
            # if LDA with multiple features, lin_out_real_idx will have all the idx that belong into features, only
            # lin_idx seems to be flawed then.

            # In case of kernel PCA: lin_idx includes kernel pca components, although info about real feature is lost
            # thus we need to remove those specific lin_idx.
            if pca_tech == 'kernel_pca' and len(lin_idx) != len(lin_out_real_idx_for_bbp):
                # remove the kernel components importance for the final box and bar plot
                to_remove = np.arange(grid_imba.best_params_["features__continuous__pca__n_components"])
                lin_idx, lin_above_zero_imp = sorted_above_zero(importance_mean=lin_imp[len(to_remove):], bar_cap=40)
            # In case of LDA: renew real idx of features including only the first one of the LDA
            if da_tech == 'lda' and any(['#2' not in s for s in lin_out_features]):  # only if 1 comp
                lin_out_real_idx_for_bbp = \
                    [list(features).index(x.split(' (')[0]) for x in lin_out_features if x.split(' (')[0] in features]
            
            # plot
            box_and_bar_plot(train_features, train_labels, test_features, test_labels,
                             np.array(lin_out_real_idx_for_bbp)[lin_idx][-lin_above_zero_imp:],
                             features, lin_above_zero_imp, output_feature, negative_class.capitalize(),
                             positive_class.capitalize(), 'full', kern, folder_name, importance_method='SVM_coef',
                             tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)

            if enable_data_split:
                # Male data
                if pca_tech == 'kernel_pca' and len(lin_idx_male) != len(lin_out_real_idx_for_bbp_male):
                    # remove kernel pca components
                    to_remove_male = np.arange(grid_imba_male.best_params_["features__continuous__pca__n_components"])
                    lin_idx_male, lin_above_zero_imp_male = \
                        sorted_above_zero(importance_mean=lin_imp_male[len(to_remove_male):], bar_cap=40)
                # In case of LDA: renew real idx of features including only the first one of the LDA
                if da_tech == 'lda' and any(['#2' not in s for s in lin_out_features_male]):
                    lin_out_real_idx_for_bbp_male = \
                        [list(features_male).index(x.split(' (')[0]) for x in lin_out_features_male if
                         x.split(' (')[0] in features_male]
                # plot
                box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                 np.array(lin_out_real_idx_for_bbp_male)[lin_idx_male][-lin_above_zero_imp_male:],
                                 features_male, lin_above_zero_imp_male, output_feature, negative_class.capitalize(),
                                 positive_class.capitalize(), 'male', kern, folder_name, importance_method='SVM_coef',
                                 tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                # Female data
                if pca_tech == 'kernel_pca' and len(lin_idx_female) != len(lin_out_real_idx_for_bbp_female):
                    # remove kernel pca components
                    to_remove_female = \
                        np.arange(grid_imba_female.best_params_["features__continuous__pca__n_components"])
                    lin_idx_female, lin_above_zero_imp_female = \
                        sorted_above_zero(importance_mean=lin_imp_female[len(to_remove_female):], bar_cap=40)
                # In case of LDA: renew real idx of features including only the first one of the LDA
                if da_tech == 'lda' and any(['#2' not in s for s in lin_out_features_female]):
                    lin_out_real_idx_for_bbp_female = \
                        [list(features_female).index(x.split(' (')[0]) for x in lin_out_features_female if
                         x.split(' (')[0] in features_female]
                # plot
                box_and_bar_plot(train_female_features, train_female_labels, test_female_features, test_female_labels,
                                 np.array(lin_out_real_idx_for_bbp_female)[lin_idx_female][-lin_above_zero_imp_female:],
                                 features_female, lin_above_zero_imp_female, output_feature,
                                 negative_class.capitalize(), positive_class.capitalize(), 'female', kern, folder_name,
                                 importance_method='SVM_coef', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                 fontsize=fix_font)

            # Reset matplotlib style conflict with seaborn for next plots if data split is skipped
            plt.style.use(plot_style)

        ########################################################################################################
        # ## Scatter plot to compare permutation & SVM_coef if linear_shuffle is enabled with linear SVM kernel
        ########################################################################################################
        # We can have multiple cases where linear feature transformation and linear classification happen
        # For all, linear_shuffle must be enabled (to have second approach to compare) with the linear SVM_coef

        if linear_shuffle and kern == 'linear':
            # in case of kernel pca, we have to clip off the n_components from the beginning of the lin_imp
            if pca_tech == 'kernel_pca':
                to_remove = np.arange(grid_imba.best_params_["features__continuous__pca__n_components"])
                lin_imp = lin_imp[len(to_remove):]  # clip off the kernel_pca components that we can't trace back
            # plot
            scatter_plot_importance_technique(kern, 'Full', mean1=lin_imp,
                                              mean2=perm_importance.importances_mean if
                                              feature_importance_method in ('all',
                                                                            'sklearn') else np.zeros(len(features)),
                                              mean3=perm_mean if
                                              feature_importance_method in ('all',
                                                                            'eli5') else np.zeros(len(features)),
                                              mean4=imp_vals if
                                              feature_importance_method in ('all',
                                                                            'mlxtend') else np.zeros(len(features)),
                                              tuple_of_names=('SVM_coef vs Sklearn', 'SVM_coef vs Eli5',
                                                              'SVM_coef vs Mlxtend'),
                                              fontsize=12, permutation_technique=feature_importance_method,
                                              lin_out_real_idx=lin_out_real_idx_for_bbp)
            plt.savefig(folder_name + f'/full_{kern}_scatter_permutation_versus_svm.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()

            # Same for split data sets if enabled
            if enable_data_split:
                # male
                if pca_tech == 'kernel_pca':
                    to_remove_male = np.arange(grid_imba_male.best_params_["features__continuous__pca__n_components"])
                    lin_imp_male = lin_imp_male[len(to_remove_male):]  # clip off the pca components that can't find
                # plot
                scatter_plot_importance_technique(kern, 'Male', mean1=lin_imp_male,
                                                  mean2=perm_importance_male.importances_mean if
                                                  feature_importance_method in ('all', 'sklearn') else
                                                  np.zeros(len(features_male)),
                                                  mean3=perm_mean_male if
                                                  feature_importance_method in ('all', 'eli5') else
                                                  np.zeros(len(features_male)),
                                                  mean4=imp_vals_male if
                                                  feature_importance_method in ('all', 'mlxtend') else
                                                  np.zeros(len(features_male)),
                                                  tuple_of_names=('SVM_coef vs Sklearn', 'SVM_coef vs Eli5',
                                                                  'SVM_coef vs Mlxtend'),
                                                  fontsize=12, permutation_technique=feature_importance_method,
                                                  lin_out_real_idx=lin_out_real_idx_for_bbp_male)
                plt.savefig(folder_name + f'/male_{kern}_scatter_permutation_versus_svm.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # female
                if pca_tech == 'kernel_pca':
                    to_remove_female = \
                        np.arange(grid_imba_female.best_params_["features__continuous__pca__n_components"])
                    lin_imp_female = lin_imp_female[len(to_remove_female):]  # clip off the kernel_pca components ...
                # plot
                scatter_plot_importance_technique(kern, 'Female', mean1=lin_imp_female,
                                                  mean2=perm_importance_female.importances_mean if
                                                  feature_importance_method in ('all', 'sklearn') else
                                                  np.zeros(len(features_female)),
                                                  mean3=perm_mean_female if
                                                  feature_importance_method in ('all', 'eli5') else
                                                  np.zeros(len(features_female)),
                                                  mean4=imp_vals_female if
                                                  feature_importance_method in ('all', 'mlxtend') else
                                                  np.zeros(len(features_female)),
                                                  tuple_of_names=('SVM_coef vs Sklearn', 'SVM_coef vs Eli5',
                                                                  'SVM_coef vs Mlxtend'),
                                                  fontsize=12, permutation_technique=feature_importance_method,
                                                  lin_out_real_idx=lin_out_real_idx_for_bbp_female)
                plt.savefig(folder_name + f'/female_{kern}_scatter_permutation_versus_svm.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

    #############################################
    # ## Display the summary performance metrics
    #############################################
    # Baseline, training and test precision, recall and roc are displayed with evaluate_model()
    # Confusion matrix is displayed with plot_confusion_matrix()
    print(f"******************************************\nFull data performance summary for {kern.upper()} kernel:\n")
    if scorer == 'dor':
        print(f"Mean GridSearchCV ({scorer.upper()}) train score: %.2f" % grid_imba.best_score_)
        print(f"Overall ({scorer.upper()}) train score: %.2f" % scoring(grid_imba.best_estimator_,
                                                                        train_features, train_labels))
        print(f"Overall ({scorer.upper()}) test score: %.2f" % scoring(grid_imba.best_estimator_,
                                                                       test_features, test_labels))
    else:
        print(f"Mean GridSearchCV ({scorer}) train score: %.2f" % (grid_imba.best_score_ * 100), '%.')
        print(f"Overall ({scorer}) train score: %.2f" % (scoring(grid_imba.best_estimator_,
                                                                 train_features, train_labels) * 100), '%.')
        print(f"Overall ({scorer}) test score: %.2f" % (
            scoring_test(grid_imba.best_estimator_,
                         test_features, test_labels) * 100 if scorer in ('balanced_accuracy', 'matthews_corrcoef') else
            scoring(grid_imba.best_estimator_, test_features, test_labels) * 100), '%.')
    print('Mean GridSearchCV PR-AUC train score: %.2f' % (cv_pr_mean * 100), '%.', '(+- %.2f)' % (cv_pr_std * 100))
    print('Mean GridSearchCV ROC-AUC train score: %.2f' % (cv_roc_mean * 100), '%.', '(+- %.2f)' % (cv_roc_std * 100))
    print('Overall ROC-AUC test score: %.2f' % (auc * 100), '%.')
    print('Overall F1 train score: %.2f' % (f1_train * 100), '%.')
    print('Overall F1 test score: %.2f' % (f1_test * 100), '%.')
    print('Overall DOR train score: %.2f' % dor_score(train_labels, train_predictions), '.')
    print('Overall DOR test score: %.2f' % dor_score(test_labels, predictions), '.')
    print('Accuracy: %.2f' % (accuracy * 100), '%.')
    print('Correctly classified samples:', correctly_classified, 'of', len(test_labels))
    print('True negatives detail:', cm[0, 0], 'correctly classified of', cm[0, 0] + cm[0, 1])
    print('True positives detail:', cm[1, 1], 'correctly classified of', cm[1, 1] + cm[1, 0])
    print('Best fitting parameters after grid search:', grid_imba.best_params_, '\n')
    print('******************************************')

    if enable_data_split:
        print(f"Male data performance summary for {kern.upper()} kernel:\n")
        if scorer == 'dor':
            print(f"Mean GridSearchCV ({scorer.upper()}) train score: %.2f" % grid_imba_male.best_score_)
            print(f"Overall ({scorer.upper()}) train score: %.2f" % scoring(grid_imba_male.best_estimator_,
                                                                            train_men_features, train_men_labels))
            print(f"Overall ({scorer.upper()}) test score: %.2f" % scoring(grid_imba_male.best_estimator_,
                                                                           test_men_features, test_men_labels))
        else:
            print(f"Mean GridSearchCV ({scorer}) train score: %.2f" % (grid_imba_male.best_score_ * 100), '%.')
            print(f"Overall ({scorer}) train score: %.2f" % (
                scoring_male(grid_imba_male.best_estimator_, train_men_features,
                             train_men_labels) * 100 if scorer in ('balanced_accuracy', 'matthews_corrcoef') else
                scoring(grid_imba_male.best_estimator_, train_men_features, train_men_labels) * 100), '%.')
            print(f"Overall ({scorer}) test score: %.2f" % (
                scoring_test_male(grid_imba_male.best_estimator_, test_men_features,
                                  test_men_labels) * 100 if scorer in ('balanced_accuracy', 'matthews_corrcoef') else
                scoring(grid_imba_male.best_estimator_, test_men_features, test_men_labels) * 100), '%.')
        print('Mean GridSearchCV PR-AUC train score: %.2f' % (cv_pr_mean_male * 100),
              '%.', '(+- %.2f)' % (cv_pr_std_male * 100))
        print('Mean GridSearchCV ROC-AUC train score: %.2f' % (cv_roc_mean_male * 100),
              '%.', '(+- %.2f)' % (cv_roc_std_male * 100))
        print('Overall ROC-AUC test score: %.2f' % (auc_male * 100), '%.')
        print('Overall F1 train score: %.2f' % (f1_train_male * 100), '%.')
        print('Overall F1 test score: %.2f' % (f1_test_male * 100), '%.')
        print('Overall DOR train score: %.2f' % dor_score(train_men_labels, train_male_predictions), '.')
        print('Overall DOR test score: %.2f' % dor_score(test_men_labels, male_predictions), '.')
        print('Accuracy: %.2f' % (accuracy_male * 100), '%.')
        print('Correctly classified samples:', correctly_classified_male, 'of', len(test_men_labels))
        print('True negatives detail:', cm_male[0, 0], 'correctly classified of', cm_male[0, 0] + cm_male[0, 1])
        print('True positives detail:', cm_male[1, 1], 'correctly classified of', cm_male[1, 1] + cm_male[1, 0])
        print('Best fitting parameters after grid search:', grid_imba_male.best_params_, '\n')

        print(f"******************************************\nFemale data performance summary "
              f"for {kern.upper()} kernel:\n")
        if scorer == 'dor':
            print(f"Mean GridSearchCV ({scorer.upper()}) train score: %.2f" % grid_imba_female.best_score_)
            print(f"Overall ({scorer.upper()}) train score: %.2f" % scoring(grid_imba_female.best_estimator_,
                                                                            train_female_features,
                                                                            train_female_labels))
            print(f"Overall ({scorer.upper()}) test score: %.2f" % scoring(grid_imba_female.best_estimator_,
                                                                           test_female_features,
                                                                           test_female_labels))
        else:
            print(f"Mean GridSearchCV ({scorer}) train score: %.2f" % (grid_imba_female.best_score_ * 100), '%.')
            print(f"Overall ({scorer}) train score: %.2f" % (
                scoring_female(grid_imba_female.best_estimator_, train_female_features,
                               train_female_labels) * 100 if scorer in ('balanced_accuracy', 'matthews_corrcoef') else
                scoring(grid_imba_female.best_estimator_, train_female_features, train_female_labels) * 100), '%.')
            print(f"Overall ({scorer}) test score: %.2f" % (
                scoring_test_female(grid_imba_female.best_estimator_, test_female_features,
                                    test_female_labels) * 100 if scorer in ('balanced_accuracy',
                                                                            'matthews_corrcoef') else
                scoring(grid_imba_female.best_estimator_, test_female_features, test_female_labels) * 100), '%.')
        print('Mean GridSearchCV PR-AUC train score: %.2f' % (cv_pr_mean_female * 100),
              '%.', '(+- %.2f)' % (cv_pr_std_female * 100))
        print('Mean GridSearchCV ROC-AUC train score: %.2f' % (cv_roc_mean_female * 100),
              '%.', '(+- %.2f)' % (cv_roc_std_female * 100))
        print('Overall ROC-AUC test score: %.2f' % (auc_female * 100), '%.')
        print('Overall F1 train score: %.2f' % (f1_train_female * 100), '%.')
        print('Overall F1 test score: %.2f' % (f1_test_female * 100), '%.')
        print('Overall DOR train score: %.2f' % dor_score(train_female_labels, train_female_predictions), '.')
        print('Overall DOR test score: %.2f' % dor_score(test_female_labels, female_predictions), '.')
        print('Accuracy: %.2f' % (accuracy_female * 100), '%.')
        print('Correctly classified samples:', correctly_classified_female, 'of', len(test_female_labels))
        print('True negatives detail:', cm_female[0, 0], 'correctly classified of', cm_female[0, 0] + cm_female[0, 1])
        print('True positives detail:', cm_female[1, 1], 'correctly classified of', cm_female[1, 1] + cm_female[1, 0])
        print('Best fitting parameters after grid search:', grid_imba_female.best_params_, '\n')
        print('******************************************')

    ##########################
    # ## Save python sessions
    ##########################
    # Collect all picklable variables
    bk = {}
    for ks in dir():
        obj = globals()[ks]
        if is_picklable(obj):
            try:
                bk.update({ks: obj})
            except TypeError:
                pass

    # Save session based on kernel
    filename = folder_name + f'/{kern}-global-save.pkl'
    with open(filename, 'wb') as f:
        print('Saving the resulting variables of this experimental session...\n')
        pickle.dump(bk, f)
        print('Saving done!\n')

    # Load your pipeline session (option 1)
    # file_to_restore = filename
    # if __name__ == '__main__':
    # with open(file_to_restore, 'rb') as f:
    #     bk_restore = pickle.load(f)
    # AttributeError: Can't get attribute 'adjacent_values' on <module '__main__'>

    # Load your pipeline session (option 2, requires custom overridden unpickler class)
    # file_to_restore = filename
    # pickle_data = CustomUnpickler(open(file_to_restore, 'rb')).load()
    # AttributeError: Can't get attribute 'adjacent_values' on <module '__main__'>

################################################
# ## Shutdown the worker engines and client hub
################################################

# Force shutting down the ipengine workers and client hub if the client exists
print('******************************************\nPipeline experiment completed successfully!')
if client:
    print('\nShutting down HPC client hub...')
    client.shutdown(hub=True)
    print('\nHPC client hub shut down!')

print('\n################################################\n################################################\n')

# close writeable output file and reset to console
sys.stdout.close()
sys.stdout = orig_stdout

########################################################################################################################
# END OF CLINICAL BIOMARKER DETECTION PIPELINE #########################################################################
########################################################################################################################
