########################################################################################################################
# HPC PARALLELIZATION SCRIPT WITH IPYPARALLEL BACKEND ##################################################################
# REMOVING HIGHLY CORRELATED FEATURES, RESAMPLING, FEATURE TRANSFORMATION, PARAMETER GRID SEARCH, DATA SPLIT BY GENDER #
# Berlin Aging Study-II (BASE-II): Gender-Specific Frailty Machine Learning Support Vector Machine Classification ######
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# November 2021 - March 2022, University of Luxembourg #################################################################
########################################################################################################################

# SUMMARY: Full clinical cohort data as well as split data based on gender, updated and revised functions and comments,
# split pipeline and grid search, adopted prints, save figures path, performance summary for all 3 data type cases,
# removing constant features, feature importance, removing highly correlated features, removing features used for
# engineering, added visualizations for highly correlated features and feature importance evaluation, select subgroups,
# transformed everything into functions, added a configuration script, enable and disable several steps, high
# reproducibility

# /!\ TO CONSIDER DEPENDING IF RUNNING ON HPC OR LOCAL MACHINES: /!\ #
# ------------------------------------------------------------------ #
# First have a look in the script configuration options section of part 1 and adapt if needed!

# Regarding HPC: verbose=0 | False, n_jobs=number of ip-engines, cache_size=200, exhaustive grid search intervals
# Regarding local machines: verbose=[1, 2] or True, n_jobs=-1, cache_size=2000, reduced grid search intervals
# Set parallel_method to ipyparallel to enable HPC client, and threading or multiprocess for local machines

# ON HPC: Check available modules, create python environment, install requirements, create directories, import data e.g.
# ./data/train_imputed.csv and ./data/test_imputed.csv, sync script, config, utils and launcher files.
# Run script on HPC using 'sbatch HPC_SVM_launcher.sh base_II_pipeline_SVM_HPC.py' after the configurations in
# base_II_config.py are set to your needs.

# REQUIRED FILES: base_II_pipeline_SVM_HPC.py, base_II_utils.py, base_II_config.py, HPC_SVM_launcher.sh,
# requirements.txt, ./env/eli5/permutation_importance.py and ./env/mlxtend/evaluate/feature_importance_permutation.py
# and ./env/mlxtend/evaluate/__init__.py adapted for parallelization, shuffle_me.py for explanation of shuffle numbers

# Global session is saved for each kernel in '-global-save.pkl' file, the main script execution output is collected
# in the generated log file if running on HPC. Location: log/job_ID/code_jobID_execution.out

# /!\ CURRENT WARNINGS / ERRORS ENCOUNTERED: /!\ #
# ---------------------------------------------- #
# RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface
# (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory.
# (To control this warning, see the rcParam `figure.max_open_warning`). SOLVED, WITH PLT.CLOSE() AND CHANGE PARAM TO 0

# PearsonRNearConstantInputWarning: An input array is nearly constant; the computed correlation coefficient may
# be inaccurate. warnings.warn(PearsonRNearConstantInputWarning()). IGNORED, ONLY SEEN ONCE IN MALE DATA

# OutdatedPackageWarning: The package pingouin is out of date. Your version is 0.5.0, the latest is 0.5.1.
# Set the environment variable OUTDATED_IGNORE=1 to disable these warnings. IGNORED, MINOR CHANGE NOT WORTH UPDATING

# ConvergenceWarning: Solver terminated early (max_iter=150000).  Consider pre-processing your data with StandardScaler
# or MinMaxScaler. IGNORED, HAPPENS WHEN USING ROBUST SCALER OR MAX ITER IS REACHED

# FitFailedWarning: X fits failed out of a total of Y. The score on these train-test partitions for these
# parameters will be set to nan. IGNORED, OCCURS WHEN LINEAR SVC IS APPLIED TOGETHER WITH KERNEL PCA, HAPPENS IN 3 CASES


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
import random
import sys
from eli5.permutation_importance import get_score_importances
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
from joblib import cpu_count, register_parallel_backend
from mlxtend.evaluate import feature_importance_permutation
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC

# Starting the script and try to load the dependent files
print(f"\n################################################\n################################################")
# Pipeline logo
logo = '  _________________  ________         ______\n'\
       ' /  ______/   ___  \/  ____  \       /  ___  \\\n'\
       '/  /      |  /__/  /  /    |  | ___ /  /__/  /\n'\
       '| |      /  __   <   /    /  / /__//   _____/\n'\
       '\  \____/  /_/    \ /____/  /     /  /\n'\
       ' \_____/__________/________/     /__/ v.3/21/2022\n'\
       '---=====================================---\n'\
       '  CLINICAL BIOMARKER DETECTION - PIPELINE\n'
print(logo)
print(f"\nStarting the Clinical Biomarker Detection Pipeline ...\n")
# Loading the BASE-II utils file
print(f"******************************************\nLOADING DEPENDENT FILES:\n\nLoading the BASE-II utils file ...")
try:
    from source.base_II_utils import *
    # from base_II_utils import *
    print("BASE-II utils file loaded successfully!\n")
except ImportError('BASE-II utils file could not be found or loaded correctly.'):
    exit()
# Loading the BASE-II configuration file
print(f"Loading the BASE-II configuration file ...")
try:
    from base_II_config import *
    # from base_II_config import *
    print("BASE-II configuration file loaded successfully!\n")
except ImportError('BASE-II configuration file could not be found or loaded correctly.'):
    exit()


########################################################################################################################
# ## PART 1: DOUBLE CHECK IF CONFIGURATION VARIABLES ARE LEGAL #########################################################
########################################################################################################################
###################################################################################
# ## Safety measures for disabled parameters and total parameter dictionary update
###################################################################################
# Safety measure to surely reset removing highly correlated feature parameters if disabled, else restore
if not enable_rhcf:
    thresh_cramer, thresh_spearman, thresh_pbs = [None] * 3
else:
    thresh_cramer, thresh_spearman, thresh_pbs = thresh_cramer, thresh_spearman, thresh_pbs
# Safety measure to surely reset feature transformation parameters  and dictionary if disabled, else restore
if not (enable_ft and pca_tech == 'normal_pca'):
    pca_lpsr = [None]
    k_best_lpsr = k_best_lpsr
    # Update dictionary depending on enabled pipeline steps
    total_params_and_splits.update({'pca_lpsr': pca_lpsr})
elif not enable_ft:
    k_best_lpsr = [None]
    total_params_and_splits.update({'k_best_lpsr': k_best_lpsr})
else:
    pca_lpsr, k_best_lpsr = pca_lpsr, k_best_lpsr
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
if not (enable_ft & enable_resampling):
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
    raise Warning("Scaler technique was not set in the configuration file. Default 'standard' is loaded.")
# Reset splitting feature if disabled
if not enable_data_split:
    split_feature = [None]
else:
    split_feature = split_feature
# Reset subgroups to keep if disabled
if not enable_subgroups:
    subgroups_to_keep = 'all'
else:
    subgroups_to_keep = subgroups_to_keep
# Reset scorer if not among implemented possibilities
if scorer not in ('F.5', 'F1', 'F2'):
    scorer = 'accuracy'
    raise Warning("Scorer was not among the possible scores. Default 'accuracy' is loaded.")
# Reset feature importance method settings
if not enable_feature_importance:
    feature_importance_method = ''
    enable_box_bar_plots = False
elif feature_importance_method not in ('sklearn', 'mlxtend', 'eli5', 'all'):
    feature_importance_method = 'all'
    raise Warning("Feature importance method not set correctly. Default 'all' is loaded.")
else:
    enable_box_bar_plots, feature_importance_method = enable_box_bar_plots, feature_importance_method
# Reset box and bar plot settings dependent on the feature importance
if not enable_box_bar_plots:
    box_bar_figures = ''
elif box_bar_figures not in ('separated', 'combined'):
    box_bar_figures = 'combined'
    raise Warning("Plot setting for box and bar plots are not set correctly. Default 'combined' is loaded.")
else:
    box_bar_figures = box_bar_figures
# check if target feature and positive/negative classes are given
for string in (output_feature, positive_class, negative_class):
    if len(string) == 0:
        raise TypeError("One or more of the following information is missing in the configuration file to start the "
                        "pipeline: output_feature, positive_class, or negative_class. Got %s instead." % string)

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
# Variables check that should strictly be a directory or path
if not os.path.isdir(curr_dir):
    raise IOError('The current directory is not set or recognized as such. Got current directory: %s.' % curr_dir)
if not all(os.path.isfile(i) for i in [train_path, test_path]):
    raise FileNotFoundError("The given train and test set pathways are not set or can't be found. "
                            "Got train: %s and test: %s." % (train_path, test_path))
# Variables check that should strictly be a string
config_str = [plot_style, pipeline_order, output_feature, split_feature, decision_func_shape, parallel_method,
              resampling_tech, folder_prefix, pca_tech, scaler_tech, scorer, feature_importance_method, box_bar_figures,
              negative_class, positive_class]
if not all(isinstance(i, str) for i in config_str):
    raise TypeError('The following configured variables must be single strings: plot_style, pipeline_order, '
                    'output_feature, split_feature, decision_func_shape, parallel_method, folder_prefix, pca_tech, '
                    'scaler_tech, scorer, feature_importance_method, box_bar_figures, negative_class, positive_class. '
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
               enable_engineered_input_removal, enable_data_split, enable_subgroups, enable_box_bar_plots]
if not all(isinstance(i, bool) for i in config_bool):
    raise TypeError('The following configured variables must be boolean: enable_rhcf, enable_resampling, '
                    'enable_ft, clf_verbose, additional_params, enable_feature_importance, '
                    'enable_engineered_input_removal, enable_data_split, enable_subgroups, enable_box_bar_plots. '
                    'Got %s instead.' % config_bool)
# Variables that could be str or tuple of str
if not (isinstance(engineered_input_prefix, tuple) or all(isinstance(i, str) for i in engineered_input_prefix)):
    raise TypeError('The following configured variable must be a tuple or str: engineered_input_prefix. Got %s instead.'
                    % engineered_input_prefix)
if not (isinstance(subgroups_to_keep, tuple) or all(isinstance(i, str) for i in subgroups_to_keep)):
    raise TypeError('The following configured variable must be a tuple or str: subgroups_to_keep. Got %s instead.'
                    % subgroups_to_keep)
# RHCF threshold variables check if remove highly correlated features is enabled
if enable_rhcf:
    config_thresh_tuple = [thresh_cramer, thresh_spearman, thresh_pbs]
    for i in config_thresh_tuple:
        if not isinstance(i, tuple):
            raise TypeError('The following configured variables must be tuples: '
                            'thresh_cramer, thresh_spearman, thresh_pbs. Got %s instead.' % config_thresh_tuple)
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
config_dict = [additional_technique_params, additional_kernel_params, total_params_and_splits, pca_kernel_dict]
if not all(isinstance(i, dict) for i in config_dict) and not len(total_params_and_splits) > 0:
    raise TypeError('The following configured variable must be a dictionary (and above zero length if total_params..): '
                    'additional_technique_params, additional_kernel_params total_params_and_splits, pca_kernel_dict. '
                    'Got %s instead.' % config_dict)
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
                                   kernel_pca_tol_lpsr, kernel_pca_degree_lpsr, kernel_pca_coef0_lpsr]
    if not all(isinstance(i, list) for i in grid_search_features_params):
        raise TypeError('The following configured variables must be lists of values or strings: '
                        'pca_lpsr, k_best_lpsr, kernel_pca_lpsr, kernel_pca_gamma_lpsr, kernel_pca_tol_lpsr, '
                        'kernel_pca_degree_lpsr, kernel_pca_coef0_lpsr. Got %s' % grid_search_features_params)


########################################################################################################################
# ## PART 2: SETTING UP THE RESULTS FOLDER, HPC OPERABILITY, AND FUNCTIONS #############################################
########################################################################################################################
###################################################
# ## Creating the results folder and clear content
###################################################
# /!\ Only supported pipeline steps are available and modifications are needed if new technical steps are added
# Name of fully enabled pipeline would be:
# Data split DS, Subgroups SG, Remove Engineered Input REI, Remove Highly Correlated Features RHCF
# Random Under Sampler RUS/Synthetic Minority Over-sampling Technique SMOTE -> 1 step
# Standard Scaler ST/Robust Scaler RO/Min-max Scaler MI, PCA/Kernel PCA KPCA, Feature Transformation FT -> 1 step
# Feature Importance FI, Support Vector Machines SVM, High Performance Computing HPC
intermediate_dict = {'SG': enable_subgroups, 'DS': enable_data_split, 'REI': enable_engineered_input_removal,
                     'RHCF': enable_rhcf, 'RUS_SMOTE': enable_resampling, 'PCA-FT_KPCA-FT': enable_ft,
                     'FI': enable_feature_importance, 'BBP': enable_box_bar_plots}
# Generating the folder intermediate name depending on enabled pipeline steps
folder_intermediate, tmp, tmp1 = '', '', ''
for key, items in intermediate_dict.items():
    if items:
        if key.__contains__('_') and not key.endswith('FT'):  # Resampling first
            tmp = (key.split('_')[0] if resampling_tech == 'rus' else key.split('_')[1])  # Get first or second tech
            folder_intermediate += ('-' + tmp if pipeline_order == 'samples->features' else '')  # Add if order allow it
        elif key.__contains__('_') and key.endswith('FT'):  # Next underscore bearing is FT
            tmp1 = (key.split('_')[0] if pca_tech == 'normal_pca' else key.split('_')[1])  # Get first or second tech
            if pca_tech == 'kernel_pca' and len(kernel_pca_kernel_lpsr) < 3:
                for pca_kern in kernel_pca_kernel_lpsr:
                    tmp1 = pca_kern + tmp1
            folder_intermediate += '-' + scaler_tech[0:2].upper()  # Before adding the FT tech, define & add scaler tech
            folder_intermediate += '-' + tmp1  # Add FT after scaler
        # Delay resampling tech insertion after FT if pipeline order allows it
        elif folder_intermediate.endswith('FT') and pipeline_order == 'features->samples':
            # As this happens in the following step 'FI' after FT, tmp must be placed in between
            folder_intermediate += '-' + tmp + '-' + key
        else:
            folder_intermediate += '-' + key  # For each true item if not yet called above
# Define the results folder name suffix based on if ipyparallel is activated (HPC-based)
folder_suffix = '-SVM'
if parallel_method == 'ipyparallel':
    folder_suffix += '-HPC'
# Final results folder will be a combination of given prefix, intermediate name, and HPC and classifier dependent suffix
folder_name = folder_prefix + folder_intermediate + folder_suffix

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

# Now as all pre folders are created, create the final results folder, if it already exists, clear it for new results
if os.path.isdir(curr_dir + '/' + folder_name) is False:
    os.mkdir(curr_dir + '/' + folder_name)
else:
    files = os.listdir(curr_dir + '/' + folder_name)
    for f in files:
        os.remove(curr_dir + '/' + folder_name + '/' + f)

###############################################################
# ## HPC parallelization preparations and n_jobs configuration
###############################################################
# Source: https://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/scikit-learn/
# Enable only if running with ipyparallel on HPC
if parallel_method != 'ipyparallel':
    client = None
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
      f"Selected figure tiff format dot-per-inches: {tiff_figure_dpi}\n"
      f"Current directory: {curr_dir.replace(backslash, '/')}\n"
      f"Folder name prefix for this analysis: {folder_prefix}\n"
      f"Training set absolute pathway: {train_path.replace(backslash, '/')}\n"
      f"Test set absolute pathway: {test_path.replace(backslash, '/')}\n"
      f"Target output feature: {output_feature}\n"
      f"Names selected for the positive and negative classes respectively: {positive_class, negative_class}\n"
      f"Features directly linked to the target: {output_related}\n"
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
      f"Selected parallel backend method and number of jobs: {parallel_method, n_jobs}\n"
      f"Removing features used for feature engineering enabled and selected feature prefixes: "
      f"{enable_engineered_input_removal, engineered_input_prefix}\n"
      f"Removing highly correlated features (RHCF) step enabled: {enable_rhcf}\n"
      f"Correlation specification for Cramer: {thresh_cramer}\n"
      f"Correlation specification for Point Bi-serial: {thresh_pbs}\n"
      f"Correlation specification for Spearman: {thresh_spearman}\n"
      f"Resampling strategy and selected technique enabled: {enable_resampling, resampling_tech}\n"
      f"Feature transformation (FT) step enabled: {enable_ft}\n"
      f"Scaler technique for continuous variables selected: {scaler_tech + ' scaler'}\n"
      f"PCA technique selected: {pca_tech.replace('_', ' ')}\n"
      f"Feature importance methods and visualizations enabled: {enable_feature_importance, feature_importance_method}\n"
      f"Box and bar plotting enabled and selected method: {enable_box_bar_plots, box_bar_figures}\n"
      f"Order of steps in the pipeline if FT or resampling are enabled: {pipeline_order}\n"
      f"Additional grid search parameters that are not directly supported: {additional_params}\n"
      f"Additional technique parameters: {additional_technique_params}\n"
      f"Additional kernel parameters: {additional_kernel_params}\n")
if enable_resampling & (resampling_tech == 'rus'):
    with_or_without_sampling = 'with rus'
elif resampling_tech == 'smote':
    with_or_without_sampling = 'with smote'
else:
    with_or_without_sampling = 'without'
scale_only = 'including '+scaler_tech+' scaling,\n'
for kern in kernels:
    print(f"Total fitting for {kern} kernel, with {splits} fold cross-validation, {'with' if enable_ft else 'without'} "
          f"feature transformation, "
          f"{(scale_only + pca_tech.replace('_', ' ') + ', select k best' if enable_ft else scale_only)}, "
          f"{with_or_without_sampling} resampling, {'with' if enable_feature_importance else 'without'} feature "
          f"importance, and {'with' if additional_params else 'without'} additional\nnon-supported grid search "
          f"parameters: {total_fits[kern]}")
    if pca_tech == 'kernel_pca' and len(kernel_pca_kernel_lpsr) > 1:
        poly_fits = int(1/3 * total_fits[kern])
        rbf_fits = int(1/3 * (int((1/len(kernel_pca_degree_lpsr) * 1/len(kernel_pca_coef0_lpsr))*total_fits[kern])))
        sigmoid_fits = int(1/3 * (int((1/len(kernel_pca_degree_lpsr))*total_fits[kern])))
        print(f"With kernelPCA enabled, the above calculated fits are masking the fact that each PCA kernel accepts "
              f"different parameters.\nFor real, as 3 kernels are tested, only a third of above mentioned fits is "
              f"tested for the poly kernelPCA.\nThe rbf and sigmoid kernels do accept less parameters, and therefore "
              f"undergo less number of fits.\nThus, the total fits of this experiment are: "
              f"{int(poly_fits + rbf_fits + sigmoid_fits)}, with {int(poly_fits)} poly fits, {int(rbf_fits)} rbf fits, "
              f"and {int(sigmoid_fits)} sigmoid fits.")
print(f"\nOverview of enabled grid search parameters "
      f"{'without' if pca_tech == 'normal_pca' else 'with' if pca_tech == 'kernel_pca' else ''} "
      f"additional kernel pca parameters:\n"
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
    print('\nThe shape of the train set with selected subgroups is:\n', train.shape)
    print('The shape of the test set with selected subgroups is:\n', test.shape)

# Split the data based on the given split feature
if enable_data_split and split_feature in train.columns:
    print(f'\nSplitting the data based on {split_feature} ...')
    train_features, test_features, train_labels, test_labels, feature_list, \
        train_men_features, test_men_features, train_men_labels, test_men_labels, \
        train_female_features, test_female_features, train_female_labels, test_female_labels, \
        feature_list_wo_gender = separate_full_data(full_train=train, full_test=test,
                                                    target_feature=output_feature, splitting_feature=split_feature)
else:  # Continue with full data only, split feature will be turned to None
    print('\nFull data analysis without data splitting, either because this step is disabled or because the feature to '
          'be split is no longer among the subgroups to keep. Subgroup activation: %s, selected subgroups: %s.'
          % enable_subgroups, subgroups_to_keep)
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
print('\nThe shape of the full train set before pre-processing is:\n', train_features.shape)
print('The shape of the full test set before pre-processing is:\n', test_features.shape)
if enable_data_split:
    print('\nThe shape of the male train set before pre-processing is:\n', train_men_features.shape)
    print('The shape of the male test set before pre-processing is:\n', test_men_features.shape)
    print('\nThe shape of the female train set before pre-processing is:\n', train_female_features.shape)
    print('The shape of the female test set before pre-processing is:\n', test_female_features.shape)

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
print("Checking and removing constant features in the training set ...\n")
# Full data
constant_all = check_constant_features(feature_list, train_features, 'full')
# Update the feature lists
feature_list = [feature_list[x] for x in range(len(feature_list)) if x not in constant_all]
# Remove those features
train_features = np.delete(train_features, constant_all, axis=1)
test_features = np.delete(test_features, constant_all, axis=1)

# Male and female data
if enable_data_split:
    constant_male = check_constant_features(feature_list_male, train_men_features, 'male')
    constant_female = check_constant_features(feature_list_female, train_female_features, 'female')
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
            categorical=categorical_idx, n_job=n_jobs, cramer_threshold=thresh_cramer)
        # Heatmap of the cramer matrix (saving process inside function)
        cramer_heatmap(cramer_res, thresh_cramer, 'full', categorical_idx, folder_name, tiff_figure_dpi)
    else:
        cat_to_drop, cramer_set = [], []

    # Spearman correlation
    if len(continuous_idx) > 1:
        spearman_res, cont_to_drop, spearman_set = applied_cont_rhcf(training_features=train_features,
                                                                     features_list=feature_list,
                                                                     continuous=continuous_idx,
                                                                     spearman_threshold=thresh_spearman)
        # Heatmap of the spearman matrix (saving process inside function)
        spearman_heatmap(spearman_res, thresh_spearman, 'full', continuous_idx, folder_name, tiff_figure_dpi)
    else:
        cont_to_drop, spearman_set = [], []
    # General data update after continuous and categorical correlation features were identified
    rem_train, rem_test, rem_feat, rem_idx = drop_and_update_correlated_data(
        continuous_to_drop=cont_to_drop, categorical_to_drop=cat_to_drop, training_set=train_features,
        test_set=test_features, features_list=feature_list)

    # Point bi-serial correlation
    if len(categorical_idx) >= 1 and len(continuous_idx) >= 1:
        longest, res_pb_r, res_pb_pv, pbs_to_drop, pbs_set, rem_cont, rem_cat = applied_cat_cont_rhcf(
            parallel_meth=parallel_method, training_features=train_features, cont_before_rhcf=continuous_idx,
            cat_before_rhcf=categorical_idx, features_list=feature_list, feat_after_rhcf=rem_feat,
            feat_idx_after_rhcf=rem_idx, n_job=n_jobs, pbs_threshold=thresh_pbs)
        # Heatmap of the point bi-serial matrix (saving process inside function)
        pbs_heatmap(res_pb_r, thresh_pbs, 'full', rem_cat, rem_cont, longest, folder_name, tiff_figure_dpi)
    else:
        pbs_to_drop, rem_cat, rem_cont, pbs_set = [], [], [], []

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
                categorical=categorical_idx_male, n_job=n_jobs, cramer_threshold=thresh_cramer)
            # Heatmap of the male cramer matrix (saving process inside function)
            cramer_heatmap(cramer_res_male, thresh_cramer, 'male', categorical_idx_male, folder_name, tiff_figure_dpi)
        else:
            cat_to_drop_male, cramer_set_male = [], []

        # Spearman correlation
        if len(continuous_idx_male) > 1:
            spearman_res_male, cont_to_drop_male, spearman_set_male = applied_cont_rhcf(
                training_features=train_men_features, features_list=feature_list_male, continuous=continuous_idx_male,
                spearman_threshold=thresh_spearman)
            # Heatmap of the male spearman matrix (saving process inside function)
            spearman_heatmap(spearman_res_male, thresh_spearman, 'male', continuous_idx_male, folder_name,
                             tiff_figure_dpi)
        else:
            cont_to_drop_male, spearman_set_male = [], []
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
                    n_job=n_jobs, pbs_threshold=thresh_pbs)
            # Heatmap of the male point bi-serial matrix (saving process inside function)
            pbs_heatmap(res_pb_r_male, thresh_pbs, 'male', rem_cat_male, rem_cont_male, longest_male, folder_name,
                        tiff_figure_dpi)
        else:
            pbs_to_drop_male, rem_cont_male, rem_cat_male, pbs_set_male = [], [], [], []

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
                cramer_threshold=thresh_cramer)
            # Heatmap of the female cramer matrix (saving process inside function)
            cramer_heatmap(cramer_res_female, thresh_cramer, 'female', categorical_idx_female, folder_name,
                           tiff_figure_dpi)
        else:
            cat_to_drop_female, cramer_set_female = [], []

        # Spearman correlation
        if len(continuous_idx_female) > 1:
            spearman_res_female, cont_to_drop_female, spearman_set_female = applied_cont_rhcf(
                training_features=train_female_features, features_list=feature_list_female,
                continuous=continuous_idx_female, spearman_threshold=thresh_spearman)
            # Heatmap of the female spearman matrix (saving process inside function)
            spearman_heatmap(spearman_res_female, thresh_spearman, 'female', continuous_idx_female, folder_name,
                             tiff_figure_dpi)
        else:
            cont_to_drop_female, spearman_set_female = [], []

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
                    feat_idx_after_rhcf=rem_idx_female, n_job=n_jobs, pbs_threshold=thresh_pbs)
            # Heatmap of the female point bi-serial matrix (saving process inside function)
            pbs_heatmap(res_pb_r_female, thresh_pbs, 'female', rem_cat_female, rem_cont_female, longest_female,
                        folder_name, tiff_figure_dpi)
        else:
            pbs_to_drop_female, rem_cont_female, rem_cat_female, pbs_set_female = [], [], [], []

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

#########################################################################
# ## Machine learning preparations including feature transformation (FT)
#########################################################################
# Print the definite shapes of data sets entering the classification pipeline
print('The shape of the full train entering the pipeline is:\n', train_features.shape)
print('The shape of the full test set entering the pipeline is:\n', test_features.shape)
if enable_data_split:
    print('\nThe shape of the male train set entering the pipeline is:\n', train_men_features.shape)
    print('The shape of the male test set entering the pipeline is:\n', test_men_features.shape)
    print('\nThe shape of the female train set entering the pipeline is:\n', train_female_features.shape)
    print('The shape of the female test set entering the pipeline is:\n', test_female_features.shape)

# Initialize pipeline steps for continuous and categorical features
print(f'\nPreparing instances for the machine learning classification pipeline ...\n')

# Get the continuous and categorical idx in full, male and female data
continuous_idx, categorical_idx = get_cat_and_cont(train_features, test_features)
if enable_data_split:
    continuous_idx_male, categorical_idx_male = get_cat_and_cont(train_men_features, test_men_features)
    continuous_idx_female, categorical_idx_female = get_cat_and_cont(train_female_features, test_female_features)
else:
    continuous_idx_male, categorical_idx_male, continuous_idx_female, categorical_idx_female = [None] * 4

# Initialize resampling method if enabled
sampler = 'passthrough'  # In case resampling is disabled
if enable_resampling:
    if resampling_tech == 'rus':
        sampler = RandomUnderSampler(sampling_strategy='majority', replacement=False, random_state=seed)  # RUS function
    elif resampling_tech == 'smote':
        sampler = SMOTE(random_state=seed, sampling_strategy='minority', n_jobs=n_jobs)  # SMOTE function

# Setting up scaler technique
if scaler_tech == 'minmax':
    scaler = MinMaxScaler()
elif scaler_tech == 'robust':
    scaler = RobustScaler()
else:
    scaler = StandardScaler()

# Initialize FT step if enabled, pca and select k best are added as column transformer steps to feature_filtration
if enable_ft:
    # PCA dimensionality reduction on continuous depending on pca technique
    pca = PCA(random_state=seed) if pca_tech == 'normal_pca' else KernelPCA(random_state=seed,
                                                                            n_jobs=n_jobs,
                                                                            max_iter=hard_iter_cap)
    k_filter = SelectKBest(score_func=chi2)  # Chi squared k best selection on categorical
    continuous_pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
    # Setting up the feature transformation transformer for the full, male, and female data if lengths are above 1
    if len(continuous_idx) > 1 and len(categorical_idx) > 1:
        feature_trans = ColumnTransformer(transformers=[('continuous', continuous_pipeline, continuous_idx),
                                                        ('categorical', k_filter, categorical_idx)], n_jobs=n_jobs)
    elif len(continuous_idx) > 1:
        feature_trans = ColumnTransformer(transformers=[
            ('continuous', continuous_pipeline, continuous_idx)], n_jobs=n_jobs)
    elif len(categorical_idx) > 1:
        feature_trans = ColumnTransformer(transformers=[('categorical', k_filter, categorical_idx)], n_jobs=n_jobs)
    else:
        feature_trans = 'passthrough'

    if enable_data_split:
        if len(continuous_idx_male) > 1 and len(categorical_idx_male) > 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_male),
                ('categorical', k_filter, categorical_idx_male)], n_jobs=n_jobs)
        elif len(continuous_idx_male) > 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_male)], n_jobs=n_jobs)
        elif len(categorical_idx_male) > 1:
            feature_trans_male = ColumnTransformer(transformers=[
                ('categorical', k_filter, categorical_idx_male)], n_jobs=n_jobs)
        else:
            feature_trans_male = 'passthrough'

        if len(continuous_idx_female) > 1 and len(categorical_idx_female) > 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_female),
                ('categorical', k_filter, categorical_idx_female)], n_jobs=n_jobs)
        elif len(continuous_idx_female) > 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('continuous', continuous_pipeline, continuous_idx_female)], n_jobs=n_jobs)
        elif len(categorical_idx_male) > 1:
            feature_trans_female = ColumnTransformer(transformers=[
                ('categorical', k_filter, categorical_idx_female)], n_jobs=n_jobs)
        else:
            feature_trans_female = 'passthrough'
    else:
        feature_trans_male, feature_trans_female = [None] * 2

else:  # If FT is disabled, we only stick with the scaler for continuous features if length > 1, else skip
    feature_trans = ColumnTransformer(transformers=[
        ('continuous', scaler, continuous_idx)], n_jobs=n_jobs) if len(continuous_idx) > 1 else 'passthrough'
    if enable_data_split:
        feature_trans_male = ColumnTransformer(transformers=[('continuous', scaler, continuous_idx_male)],
                                               n_jobs=n_jobs) if len(continuous_idx_male) > 1 else 'passthrough'
        feature_trans_female = ColumnTransformer(transformers=[('continuous', scaler, continuous_idx_female)],
                                                 n_jobs=n_jobs) if len(continuous_idx_female) > 1 else 'passthrough'
    else:
        feature_trans_male, feature_trans_female = [None] * 2

# Additional pipeline properties including scoring, stratified k fold and parameters for all kernel and possible steps
if scorer == 'F2':
    scoring = make_scorer(fbeta_score, beta=2, average='macro')  # F beta 2
elif scorer == 'F.5':
    scoring = make_scorer(fbeta_score, beta=0.5, average='macro')  # F beta 0.5
elif scorer == 'F1':
    scoring = make_scorer(fbeta_score, beta=1, average='macro')  # F beta 1
else:
    scoring = make_scorer(accuracy_score)  # Accuracy as default
skf = StratifiedKFold(n_splits=splits, shuffle=False)  # stratified k fold split of train, shuffle=F to avoid duplicates

# Setting up shared param dictionary between the selected kernels for grid search cross-validation
params = {'clf__C': regularization_lpsr,
          'clf__shrinking': shrinking_lpsr,
          'clf__tol': tolerance_lpsr,
          'clf__gamma': gamma_psr}
kernel_pca_params = {'pca__n_components': kernel_pca_lpsr,
                     'pca__kernel': kernel_pca_kernel_lpsr,
                     'pca__gamma': kernel_pca_gamma_lpsr,
                     'pca__tol': kernel_pca_tol_lpsr}
if 'poly' in kernel_pca_kernel_lpsr:
    kernel_pca_params.update({'pca__degree': kernel_pca_degree_lpsr,
                              'pca__coef0': kernel_pca_coef0_lpsr})
if 'sigmoid' in kernel_pca_kernel_lpsr:
    kernel_pca_params.update({'pca__coef0': kernel_pca_coef0_lpsr})

# PCA n components and k best features are added to the parameter dict if the feature transformation step is enabled
# It is enough to check if the one from the complete data is passthrough, features don't change between the three sets
if enable_ft and pca_tech == 'normal_pca' and feature_trans != 'passthrough':
    params.update({'features__continuous__pca__n_components': pca_lpsr,
                   'features__categorical__k': k_best_lpsr})
    # Load the corresponding FT parameters if kernel pca is activated
elif enable_ft and pca_tech == 'kernel_pca' and feature_trans != 'passthrough':
    params.update({'features__categorical__k': k_best_lpsr})
    params.update({'features__continuous__' + key: items for key, items in kernel_pca_params.items()})
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
            pipeline_male = Pipeline([('samples', sampler), ('features', feature_trans_male), ('clf', svm_clf)])
            pipeline_female = Pipeline([('samples', sampler), ('features', feature_trans_female), ('clf', svm_clf)])
        else:
            pipeline_male, pipeline_female = [None] * 2
    else:  # If pipeline order is reversed if resampling is deactivated (sampler will become 'passthrough' then)
        pipeline = Pipeline([('features', feature_trans), ('samples', sampler), ('clf', svm_clf)])
        if enable_data_split:
            pipeline_male = Pipeline([('features', feature_trans_male), ('samples', sampler), ('clf', svm_clf)])
            pipeline_female = Pipeline([('features', feature_trans_female), ('samples', sampler), ('clf', svm_clf)])
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

    print(f"******************************************\n{kern.capitalize()} SVM grid search parameter summary with "
          f"{pca_tech} technique:\n\n"
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
                                      scoring=scoring,
                                      return_train_score=True,
                                      verbose=grid_verbose,
                                      n_jobs=n_jobs)
        # Grid search on female data
        grid_imba_female = GridSearchCV(pipeline_female, param_grid=final_params,
                                        cv=skf,
                                        scoring=scoring,
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
    misclassified = np.array(test_labels != predictions).sum()
    # male ony
    if enable_data_split:
        accuracy_male = accuracy_score(test_men_labels, male_predictions)
        f1_test_male = f1_score(test_men_labels, male_predictions, average='macro')
        f1_train_male = f1_score(train_men_labels, train_male_predictions, average='macro')
        auc_male = roc_auc_score(test_men_labels, male_predictions)
        misclassified_male = np.array(test_men_labels != male_predictions).sum()
        # female ony
        accuracy_female = accuracy_score(test_female_labels, female_predictions)
        f1_test_female = f1_score(test_female_labels, female_predictions, average='macro')
        f1_train_female = f1_score(train_female_labels, train_female_predictions, average='macro')
        auc_female = roc_auc_score(test_female_labels, female_predictions)
        misclassified_female = np.array(test_female_labels != female_predictions).sum()
    else:
        accuracy_male, f1_test_male, f1_train_male, auc_male, misclassified_male, \
            accuracy_female, f1_test_female, f1_train_female, auc_female, misclassified_female = [None] * 10

    ######################
    # ## Evaluation plots
    ######################
    # Readjust font size for roc_auc curve and confusion matrix
    if plt.rcParams['font.size'] != fix_font:
        plt.rcParams['font.size'] = fix_font

    # ROC_AUC curve full data
    print(f"Full data model evaluation for {kern.upper()} kernel:")
    evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels, fontsize=16)
    plt.savefig(folder_name + f'/{kern}_roc_auc_curve_ALL.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    if enable_data_split:
        # Male data
        print(f"\nMale data model evaluation for {kern.upper()} kernel:")
        evaluate_model(male_predictions, male_probs, train_male_predictions, train_male_probs, test_men_labels,
                       train_men_labels, fontsize=16)
        plt.savefig(folder_name + f'/{kern}_roc_auc_curve_male.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # Female data
        print(f"\nFemale data model evaluation for {kern.upper()} kernel:")
        evaluate_model(female_predictions, female_probs, train_female_predictions, train_female_probs,
                       test_female_labels, train_female_labels, fontsize=16)
        plt.savefig(folder_name + f'/{kern}_roc_auc_curve_female.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()

    # Confusion matrix full data
    cm = confusion_matrix(test_labels, predictions)
    print(f"\nFull data confusion matrix for {kern.upper()} kernel:")
    plot_confusion_matrix(cm, classes=[negative_class.capitalize(), positive_class.capitalize()],
                          title='Confusion Matrix', normalize=True)
    plt.savefig(folder_name + f'/{kern}_cm_ALL.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
    plt.close()
    if enable_data_split:
        # Male data
        cm_male = confusion_matrix(test_men_labels, male_predictions)
        print(f"\nMale data confusion matrix for {kern.upper()} kernel:")
        plot_confusion_matrix(cm_male, classes=[negative_class.capitalize(), positive_class.capitalize()],
                              title='Confusion Matrix', normalize=True)
        plt.savefig(folder_name + f'/{kern}_cm_male.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()
        # Female data
        cm_female = confusion_matrix(test_female_labels, female_predictions)
        print(f"\nFemale data confusion matrix for {kern.upper()} kernel:")
        plot_confusion_matrix(cm_female, classes=[negative_class.capitalize(), positive_class.capitalize()],
                              title='Confusion Matrix', normalize=True)
        plt.savefig(folder_name + f'/{kern}_cm_female.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
        plt.close()

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
    # This section is only enabled if feature transformation step is enabled
    if enable_ft and pca_tech == 'normal_pca' and feature_trans != 'passthrough':
        # Update feature list with the features that remained after feature transformation by PCA and SelectKBest
        # Full data
        new_features_full = update_features(predict_method=grid_imba, named_step='features',
                                            cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                            features_list=feature_list, cat_list=categorical_idx,
                                            cont_list=continuous_idx, pca_tech=pca_tech)
        if enable_data_split:
            # Male data
            new_features_male = update_features(predict_method=grid_imba_male, named_step='features',
                                                cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                                features_list=feature_list_male, cat_list=categorical_idx_male,
                                                cont_list=continuous_idx_male, pca_tech=pca_tech)
            # Female data
            new_features_female = update_features(predict_method=grid_imba_female, named_step='features',
                                                  cat_transformer='categorical', cont_transformer=['continuous', 'pca'],
                                                  features_list=feature_list_female, cat_list=categorical_idx_female,
                                                  cont_list=continuous_idx_female, pca_tech=pca_tech)
        else:
            new_features_male, new_features_female = [None] * 2

        ####################################################################################################
        # ## Extracting the best selected features for later analysis of feature transformation performance
        ####################################################################################################
        # We retrieve the indices of best selected features to see which one were selected
        print(f'\nSelect K best and PCA identified the following top features:\n')
        # Get the top feature of n components selected by the pipeline pca and select k best
        idx_of_best = [idx for idx in range(len(features)) if features[idx] in new_features_full]
        # Print the selected features
        print(f'Full data:\n{grid_imba.best_params_["features__categorical__k"]} k best and '
              f'{grid_imba.best_params_["features__continuous__pca__n_components"]} PCA '
              f'components:\n{features[idx_of_best]}\n')
        # Inform about possible duplicates in case PCA number 1 top features are extracted for each selected component
        inform_about_duplicates(new_features_full, idx_of_best, 'full')

        # In male and female
        if enable_data_split:
            # Get the top feature of n components selected by the pipeline pca and select k best
            idx_of_best_male = [idx for idx in range(len(features_male)) if features_male[idx] in new_features_male]
            idx_of_best_female = \
                [idx for idx in range(len(features_female)) if features_female[idx] in new_features_female]
            # Print the selected features
            print(f'Male data:\n{grid_imba_male.best_params_["features__categorical__k"]} k best and '
                  f'{grid_imba_male.best_params_["features__continuous__pca__n_components"]} PCA '
                  f'components:\n{features_male[idx_of_best_male]}\n')
            # Inform about possible duplicates in case the PCA top features are extracted for each selected component
            inform_about_duplicates(new_features_male, idx_of_best_male, 'male')
            print(f'Female data:\n{grid_imba_female.best_params_["features__categorical__k"]} k best and '
                  f'{grid_imba_female.best_params_["features__continuous__pca__n_components"]} PCA '
                  f'components:\n{features_female[idx_of_best_female]}\n')
            # Inform about possible duplicates in case the PCA top features are extracted for each selected component
            inform_about_duplicates(new_features_female, idx_of_best_female, 'female')

    ########################################
    # ## Non linear feature importance (FI)
    ########################################
    # Readjust font size for feature importance figures
    if enable_feature_importance and (kern in non_linear_kernels):
        if plt.rcParams['font.size'] != imp_font:
            plt.rcParams['font.size'] = imp_font

        # FEATURE IMPORTANCE BY SKLEARN.INSPECTION
        if feature_importance_method in ('all', 'sklearn'):
            print("Starting feature importance permutation by SKLEARN:")
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
                folder_name + f'/{kern}_full_feature_importance_sklearn.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
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
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_sklearn.tiff',
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
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_sklearn.tiff',
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
            plt.savefig(folder_name + f'/{kern}_full_feature_importance_eli5.tiff',
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
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_eli5.tiff',
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
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_eli5.tiff',
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
            plt.savefig(folder_name + f'/{kern}_full_feature_importance_mlxtend.tiff',
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
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_mlxtend.tiff',
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
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_mlxtend.tiff',
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
            print('******************************************')
            if enable_data_split:
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
            plt.savefig(folder_name + f'/{kern}_full_feature_importance_venn_diagram.tiff',
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
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_venn_diagram.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                sklearn_female = set(features_female[sorted_idx_female[-sk_above_zero_imp_female:]])
                eli5_female = set(features_female[sorted_idx_eli_female[-el_above_zero_imp_female:]])
                mlxtend_female = set(features_female[indices_female[-ml_above_zero_imp_female:]])

                plot_venn(kernel=kern, datatype='Female', set1=sklearn_female, set2=eli5_female, set3=mlxtend_female,
                          tuple_of_names=('sklearn', 'eli5', 'mlxtend'), label_fontsize=8,
                          feat_info='top important', weighted=True)
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_venn_diagram.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

            # Scatter plot comparing the feature importance measuring effect between the three methods
            metrics = ["r", "CI95%", "p-val"]
            # Full data
            scatter_comparison(kernel=kern, datatype='Full', mean1=perm_importance.importances_mean, mean2=perm_mean,
                               mean3=imp_vals,
                               new_feat_idx=range(len(features)), metric_list=metrics)
            plt.savefig(folder_name + f'/{kern}_full_feature_importance_comparison.tiff',
                        bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                scatter_comparison(kernel=kern, datatype='Male', mean1=perm_importance_male.importances_mean,
                                   mean2=perm_mean_male, mean3=imp_vals_male,
                                   new_feat_idx=range(len(features_male)), metric_list=metrics)
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_comparison.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                scatter_comparison(kernel=kern, datatype='Female', mean1=perm_importance_female.importances_mean,
                                   mean2=perm_mean_female, mean3=imp_vals_female,
                                   new_feat_idx=range(len(features_female)), metric_list=metrics)
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_comparison.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

            # Scatter plot to check correlation between the three methods using r2 and linear equation
            # Full data
            scatter_r_squared(kernel=kern, datatype='Full', mean1=perm_importance.importances_mean, mean2=perm_mean,
                              mean3=imp_vals,
                              tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                              new_feat_idx=range(len(features)), fontsize=12)
            plt.savefig(folder_name + f'/{kern}_full_feature_importance_r2.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                scatter_r_squared(kernel=kern, datatype='Male', mean1=perm_importance_male.importances_mean,
                                  mean2=perm_mean_male, mean3=imp_vals_male,
                                  tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                                  new_feat_idx=range(len(features_male)), fontsize=12)
                plt.savefig(folder_name + f'/{kern}_male_feature_importance_r2.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                scatter_r_squared(kernel=kern, datatype='Female', mean1=perm_importance_female.importances_mean,
                                  mean2=perm_mean_female, mean3=imp_vals_female,
                                  tuple_of_names=('Sklearn vs Eli5', 'Sklearn vs Mlxtend', 'Eli5 vs Mlxtend'),
                                  new_feat_idx=range(len(features_female)), fontsize=12)
                plt.savefig(folder_name + f'/{kern}_female_feature_importance_r2.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

        # Violin plot of the shuffling effect on the most important feature scores
        # Sklearn on full data
        if feature_importance_method in ('all', 'sklearn'):
            plot_violin(kern, 'Full sklearn', perm_importance.importances[sorted_idx[-sk_above_zero_imp:]],
                        features[sorted_idx[-sk_above_zero_imp:]], fontsize=7)
            plt.savefig(folder_name + f'/{kern}_full_violin_plot_sklearn.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male sklearn',
                            perm_importance_male.importances[sorted_idx_male[-sk_above_zero_imp_male:]],
                            features_male[sorted_idx_male[-sk_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_male_violin_plot_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female sklearn',
                            perm_importance_female.importances[sorted_idx_female[-sk_above_zero_imp_female:]],
                            features_female[sorted_idx_female[-sk_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_female_violin_plot_sklearn.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
        # ELI5 on full data
        if feature_importance_method in ('all', 'eli5'):
            plot_violin(kern, 'Full eli5', perm_all[sorted_idx_eli[-el_above_zero_imp:]],
                        features[sorted_idx_eli[-el_above_zero_imp:]], fontsize=7)
            plt.savefig(folder_name + f'/{kern}_full_violin_plot_eli5.tiff', bbox_inches='tight', dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male eli5', perm_all_male[sorted_idx_eli_male[-el_above_zero_imp_male:]],
                            features_male[sorted_idx_eli_male[-el_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_male_violin_plot_eli5.tiff', bbox_inches='tight',
                            dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female eli5', perm_all_female[sorted_idx_eli_female[-el_above_zero_imp_female:]],
                            features_female[sorted_idx_eli_female[-el_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_female_violin_plot_eli5.tiff', bbox_inches='tight',
                            dpi=tiff_figure_dpi)
                plt.close()
        # MLXTEND on full data
        if feature_importance_method in ('all', 'mlxtend'):
            plot_violin(kern, 'Full mlxtend',
                        imp_all[indices[-ml_above_zero_imp:]], features[indices[-ml_above_zero_imp:]],
                        fontsize=7)
            plt.savefig(folder_name + f'/{kern}_full_violin_plot_mlxtend.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            # Male data
            if enable_data_split:
                plot_violin(kern, 'Male mlxtend', imp_all_male[indices_male[-ml_above_zero_imp_male:]],
                            features_male[indices_male[-ml_above_zero_imp_male:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_male_violin_plot_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()
                # Female data
                plot_violin(kern, 'Female mlxtend', imp_all_female[indices_female[-ml_above_zero_imp_female:]],
                            features_female[indices_female[-ml_above_zero_imp_female:]], fontsize=7)
                plt.savefig(folder_name + f'/{kern}_female_violin_plot_mlxtend.tiff',
                            bbox_inches='tight', dpi=tiff_figure_dpi)
                plt.close()

        #############################################################################
        # ## Box and bar plots of most important continuous and categorical features
        #############################################################################
        if enable_box_bar_plots:
            # Case of sklearn method
            if feature_importance_method in ('all', 'sklearn'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels, sorted_idx,
                                 features, sk_above_zero_imp, output_feature, negative_class.capitalize(),
                                 positive_class.capitalize(), 'full', kern, folder_name, importance_method='sklearn',
                                 tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     sorted_idx_male, features_male, sk_above_zero_imp_male, output_feature,
                                     negative_class.capitalize(), positive_class.capitalize(), 'male', kern,
                                     folder_name, importance_method='sklearn', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, sorted_idx_female, features_female, sk_above_zero_imp_female,
                                     output_feature, negative_class.capitalize(), positive_class.capitalize(), 'female',
                                     kern, folder_name, importance_method='sklearn', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
            # Case of eli5 method
            if feature_importance_method in ('all', 'eli5'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels, sorted_idx_eli,
                                 features, el_above_zero_imp, output_feature, negative_class.capitalize(),
                                 positive_class.capitalize(), 'full', kern, folder_name, importance_method='eli5',
                                 tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     sorted_idx_eli_male, features_male, el_above_zero_imp_male, output_feature,
                                     negative_class.capitalize(), positive_class.capitalize(), 'male', kern,
                                     folder_name, importance_method='eli5', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, sorted_idx_eli_female, features_female,
                                     el_above_zero_imp_female, output_feature, negative_class.capitalize(),
                                     positive_class.capitalize(), 'female', kern, folder_name, importance_method='eli5',
                                     tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
            # Case of mlxtend method
            if feature_importance_method in ('all', 'mlxtend'):
                # Full data
                box_and_bar_plot(train_features, train_labels, test_features, test_labels, indices,
                                 features, ml_above_zero_imp, output_feature, negative_class.capitalize(),
                                 positive_class.capitalize(), 'full', kern, folder_name, importance_method='mlxtend',
                                 tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
                if enable_data_split:
                    # Male data
                    box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                     indices_male, features_male, ml_above_zero_imp_male, output_feature,
                                     negative_class.capitalize(), positive_class.capitalize(), 'male', kern,
                                     folder_name, importance_method='mlxtend', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)
                    # Female data
                    box_and_bar_plot(train_female_features, train_female_labels, test_female_features,
                                     test_female_labels, indices_female, features_female, ml_above_zero_imp_female,
                                     output_feature, negative_class.capitalize(), positive_class.capitalize(), 'female',
                                     kern, folder_name, importance_method='mlxtend', tiff_size=tiff_figure_dpi,
                                     graphs=box_bar_figures, fontsize=fix_font)

    ####################################
    # ## Linear feature importance (FI)
    ####################################
    # Readjust font size for importance figure of linear kernel
    if enable_feature_importance and kern == 'linear':  # In case of linear SVM kernel because of non-linear pca
        if plt.rcParams['font.size'] != imp_font:
            plt.rcParams['font.size'] = imp_font

        print("\n******************************************\nProcessing feature importance with linear kernel...\n")
        # Full data
        lin_imp = grid_imba.best_estimator_.named_steps['clf'].coef_[0]
        lin_idx, lin_above_zero_imp = sorted_above_zero(importance_mean=lin_imp, bar_cap=40)
        # Figure of most important features
        importance_plot(datatype='full', method='LINEAR SVC', kern=kern, idx_sorted=lin_idx, features_list=features,
                        importance_mean=lin_imp, importance_above_zero=lin_above_zero_imp, importance_std=None)
        plt.savefig(folder_name + f'/{kern}_full_feature_importance.tiff', bbox_inches='tight',
                    dpi=tiff_figure_dpi)
        plt.close()
        print('Full data top important features with linear kernel:\n',
              features[lin_idx[-lin_above_zero_imp:]][::-1], '\n')

        # male data
        if enable_data_split:
            lin_imp_male = grid_imba_male.best_estimator_.named_steps['clf'].coef_[0]
            lin_idx_male, lin_above_zero_imp_male = sorted_above_zero(importance_mean=lin_imp_male, bar_cap=40)
            # Figure of most important features
            importance_plot(datatype='male', method='LINEAR SVC', kern=kern, idx_sorted=lin_idx_male,
                            features_list=features_male, importance_mean=lin_imp_male,
                            importance_above_zero=lin_above_zero_imp_male, importance_std=None)
            plt.savefig(folder_name + f'/{kern}_male_feature_importance.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            print('Male data top important features with linear kernel:\n',
                  features_male[lin_idx_male[-lin_above_zero_imp_male:]][::-1], '\n')

            # Female data
            lin_imp_female = grid_imba_female.best_estimator_.named_steps['clf'].coef_[0]
            lin_idx_female, lin_above_zero_imp_female = sorted_above_zero(importance_mean=lin_imp_female, bar_cap=40)
            # Figure of most important features
            importance_plot(datatype='female', method='LINEAR SVC', kern=kern, idx_sorted=lin_idx_female,
                            features_list=features_female, importance_mean=lin_imp_female,
                            importance_above_zero=lin_above_zero_imp_female, importance_std=None)
            plt.savefig(folder_name + f'/{kern}_female_feature_importance.tiff', bbox_inches='tight',
                        dpi=tiff_figure_dpi)
            plt.close()
            print('Female data top important features with linear kernel:\n',
                  features_female[lin_idx_female[-lin_above_zero_imp_female:]][::-1], '\n')
        else:
            lin_idx_male, lin_above_zero_imp_male, lin_idx_female, lin_above_zero_imp_female = [None] * 4

        ############################################################
        # ## Box and bar plots in case of linear feature importance
        ############################################################
        if enable_box_bar_plots:
            # Full data
            box_and_bar_plot(train_features, train_labels, test_features, test_labels, lin_idx,
                             features, lin_above_zero_imp, output_feature, negative_class.capitalize(),
                             positive_class.capitalize(), 'full', kern, folder_name, importance_method='',
                             tiff_size=tiff_figure_dpi, graphs=box_bar_figures, fontsize=fix_font)
            if enable_data_split:
                # Male data
                box_and_bar_plot(train_men_features, train_men_labels, test_men_features, test_men_labels,
                                 lin_idx_male, features_male, lin_above_zero_imp_male, output_feature,
                                 negative_class.capitalize(), positive_class.capitalize(), 'male', kern, folder_name,
                                 importance_method='', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                 fontsize=fix_font)
                # Female data
                box_and_bar_plot(train_female_features, train_female_labels, test_female_features, test_female_labels,
                                 lin_idx_female, features_female, lin_above_zero_imp_female, output_feature,
                                 negative_class.capitalize(), positive_class.capitalize(), 'female', kern, folder_name,
                                 importance_method='', tiff_size=tiff_figure_dpi, graphs=box_bar_figures,
                                 fontsize=fix_font)

    #############################################
    # ## Display the summary performance metrics
    #############################################
    # Baseline, training and test precision, recall and roc are displayed with evaluate_model()
    # Confusion matrix is displayed with plot_confusion_matrix()
    print(f"******************************************\nFull data performance summary for {kern.upper()} kernel:\n")
    print("Mean CV (F-beta-2) score:", round(grid_imba.best_score_, 5) * 100, '%.')
    print('Accuracy:', round(accuracy, 4) * 100, '%.')
    print('F1 test score:', round(f1_test, 4) * 100, '%.')
    print('F1 train score:', round(f1_train, 4) * 100, '%.')
    print('AUC:', round(auc, 4) * 100, '%.')
    print('Misclassified samples:', misclassified, 'of', len(test_labels))
    print('Best fitting parameters after grid search:', grid_imba.best_params_, '\n')
    print('******************************************')

    if enable_data_split:
        print(f"Male data performance summary for {kern.upper()} kernel:\n")
        print("Mean CV (F-beta-2) score:", round(grid_imba_male.best_score_, 5) * 100, '%.')
        print('Accuracy:', round(accuracy_male, 4) * 100, '%.')
        print('F1 test score:', round(f1_test_male, 4) * 100, '%.')
        print('F1 train score:', round(f1_train_male, 4) * 100, '%.')
        print('AUC:', round(auc_male, 4) * 100, '%.')
        print('Misclassified samples:', misclassified_male, 'of', len(test_men_labels))
        print('Best fitting parameters after grid search:', grid_imba_male.best_params_, '\n')

        print(f"******************************************\nFemale data performance summary "
              f"for {kern.upper()} kernel:\n")
        print("Mean CV (F-beta-2) score:", round(grid_imba_female.best_score_, 5) * 100, '%.')
        print('Accuracy:', round(accuracy_female, 4) * 100, '%.')
        print('F1 test score:', round(f1_test_female, 4) * 100, '%.')
        print('F1 train score:', round(f1_train_female, 4) * 100, '%.')
        print('AUC:', round(auc_female, 4) * 100, '%.')
        print('Misclassified samples:', misclassified_female, 'of', len(test_female_labels))
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

########################################################################################################################
# END OF BASE II SVM PIPELINE HPC ######################################################################################
########################################################################################################################
