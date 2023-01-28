########################################################################################################################
# CLINICAL BIOMARKER DETECTION - PIPELINE (CBD-P) CONFIGURATION FILE ###################################################
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# November 2021 - January 2023, University of Luxembourg, v.01/25/2023 (M/d/y) #########################################
########################################################################################################################

# Configure up to 83 variables specific for the CBD_pipeline_SVM_HPC.py script. Configured variables will undergo
# legal violation checks in the main script and will be adopted to the correctly enabled pipeline steps if necessary.

# /!\ When launching a given configuration, wait until the job has started running before modifying the config file. /!\


########################################################################################################################
# ## PART 0: CONFIGURATION START IMPORTING LIBRARIES ###################################################################
########################################################################################################################
#########################
# ## Importing libraries
#########################
import os
import numpy as np


########################################################################################################################
# ## PART 1: CONFIGURATION OPTIONS #####################################################################################
########################################################################################################################
########################################################################################
# ## General settings like random seed, plotting style, library options and figures dpi
########################################################################################
# TODO REWRITE VARIABLE DESCRIPTION PROPERLY
seed = 42  # Random seed, 42, int
fix_font = 18  # Fix font size for general plots, 18, int
imp_font = 8  # Specific font size for feature importance plots, 8, int
plot_style = 'fivethirtyeight'  # Matplotlib plot style, str
fig_max_open_warning = 0  # Warning shown by matplotlib after number of open figures, int
pandas_col_display_option = 5  # Number of columns displayed in pandas dataframe, int
tiff_figure_dpi = 300  # Dot per inches resolution of the result figures, int
# Other font sizes for titles, ticks, or labels are defined in the corresponding function
# Later all fonts will be generalized
debug = False  # Debug statement to decide if figures should be displayed for debugging, select False to avoid
# 'RuntimeError: main thread is not in main loop' errors when the complete pipeline runs on local machine with threading

#######################################################################################################
# ## Data specific configurations like pathways, target features, related target features and prefixes
#######################################################################################################
# DATA SET RULES: Data should be provided in imputed training and imputed test set, containing mixed structure of
# continuous, binary and ordinal features
curr_dir = os.getcwd()  # Pathway to current directory, str (dir)
folder_prefix = 'results/BASE-II'  # Prefix of results folder name can be a folder in folder, str (suffixed in code)
# It is recommended to set a folder prefix that refers to the experimental configuration of interest
# Please adapt the following variables to your data set and research topic, e.g. Frailty in BASE-II
train_path = curr_dir + '/data/train_imputed.csv'  # Path to imputed training set, str (file)
test_path = curr_dir + '/data/test_imputed.csv'  # Path to imputed training set, str (file)
output_feature = 'PM-Frailty_Index'  # Target output feature, str (must be binary vector)
positive_class = 'frail'  # Name to give the positive class of your target feature, str
negative_class = 'non-frail'  # Name to give the negative class of your target feature, str
output_related = ['PM-Frailty_Score',
                  'PM-Frailty_gait',
                  'SV-Frailty_exhaustion',
                  'SV-Frailty_physicalactivity',
                  'PM-Frailty_gripstrength',
                  'PM-Frailty_weightloss',
                  'PM-Gripstrength_max']  # Output-related features, str (can be list)
sample_tagging_feature = ['PM-Frailty_Score', 'BF-VitDDef']  # Feature used to define samples to tag specifically, str or list of str
tag_threshold = (('>=', '3'), ('==', '1'))  # Threshold to define samples to tag, tuple of str or tuple of tuples if multiple tagging features exist, first position must be math operator
features_of_interest = ['']  # Features of interest to investigate distributions in TP, TN, FP and FN, list of str or str

###################################################################################################################
# ## Machine learning classification estimator specific configurations (in this case Support Vector Machine (SVM))
###################################################################################################################
# /!\ SUPPORTED KERNELS: linear, poly, rbf, sigmoid
# Classification variables
kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel to use for the Support Vector Machine classifier, str (or list)
# Will be replaced by 'linear' only if kernelPCA is activated
non_linear_kernels = ['poly', 'rbf', 'sigmoid']  # Repeat with the above kernels that are non_linear, str (can be list)
cache_size = 200  # Cache size of SVM classifier, 200 (HPC), 2000 (local), int
decision_func_shape = 'ovr'  # Decision function shape of classifier, one vs rest 'ovr' or one vs one 'ovo', 'ovr', str
clf_verbose = False  # Classifier verbose either True or False, False, bool
grid_verbose = 1  # Grid search verbose, either 1 or 2, 1, int
hard_iter_cap = 150000  # Hard stopping criterion, 150000, int
splits = 10  # Stratified k fold splits, 10, int
scorer = 'F2'  # Scorer used, either F.5, F1, F2, F5, roc_auc, dor, matthews_corrcoef, balanced_accuracy or accuracy,
# def acc.,str, if scorer is balanced_accuracy or matthews_corrcoef, the current weights are 10 to 1 for positive class.
# /!\ Kernels will be changed to only 'linear' if non-linear PCA is activated in the pipeline

# Feature importance number of shuffles
shuffle_all = 1000  # Proven 1000 for a set of 1200 samples that each sample receives at least half of the other values
shuffle_male = 500  # Proven 500 for a set of 600 samples
shuffle_female = 500  # Proven 500 for a set of 600 samples
# See shuffle_me.py file in detail for proof
linear_shuffle = True  # In case of linear SVM combined with linear pca, get feature importance by shuffling, True, bool

######################################################################################
# ## Select parallel method and enabling various pipeline steps with given techniques
######################################################################################
parallel_method = 'ipyparallel'  # Parallel agent, 'ipyparallel' (HPC), 'threading', 'multiprocess', 'loki' (local), str
n_jobs = -1  # Number of jobs for distributed tasks, will be adjusted if ipyparallel is enabled, default -1, int
thresh_near_constant = 0.001  # Thresh for a continuous feature near-constance by variance-mean-ratio, def 0.001, float
# near-constant categorical feature threshold will be based on the number of stratified k fold splits defined above
enable_data_split = True  # True if data should be split based on the binary split feature below, default True, bool
split_feature = 'PM-sex'  # Feature based on which data is split, str (will be set to None if disabled)
enable_subgroups = False  # True if data shall be limited to subgroups, else full feature input, default False, bool
subgroups_to_keep = 'all'  # Prefix of subgroups to keep for the analysis, default 'all', tuple of str, str or 'all'
# /!\ If specific prefixes are selected, make sure that they do not conflict with the engineered_input_prefix below and
# that the output feature defined above is included .e.g. subgroups_to_keep = ('BF-', 'SV-', output_feature)
# Possible subgroups are prefixed: BF- (body fluids), PM- (physical measurements), IM- (individual medications), ID-
# (individual devices), GM- (grouped medications), GD- (grouped devices), SV- (survey), CH- (chronic), NT- (nutrients),
# CG- (cognitive), ST- (study-related).
enable_engineered_input_removal = True  # Change to enable or disable removal of engineered input features, True, bool
engineered_input_prefix = ('IM-', 'ID-')  # Prefix of features used in engineering, IM, ID, str, tuple of str, can be ''
enable_rhcf = True  # Change to True to enable & False to disable RHCF, default True, bool
# Set threshold variables for RHCF with Cramer's V, Spearman's Rank and Point bi-serial, will be None if disabled,
# # can be a decimal threshold or a given percentile threshold
thresh_cramer = (0.6, 'decimal')  # Corrected Cramer's V correlation threshold, default (0.6, 'decimal'), tuple
thresh_spearman = (95, 'percentile')  # Spearman's Rank correlation threshold, default (95, 'percentile'), tuple
thresh_pbs = (0.6, 'decimal')  # Point bi-serial correlation threshold, default (0.6, 'decimal'), tuple
enable_resampling = True  # Change to True to enable & False to disable resampling, default True, bool
resampling_tech = 'rus'  # 'rus' (random under-sampling), 'smote' (synthetic minority over-sampling technique), str
enable_ft = True  # Change to True to enable & False to disable feature transformation, default True, bool
scaler_tech = 'standard'  # Change scaler function to 'standard', 'minmax' or 'robust' scaler, default 'standard', str
# /!\ If PCA is enabled or given, then LDA/QDA will be disabled, and vice versa
pca_tech = 'normal_pca'  # Select pca technique to choose between 'normal_pca' and 'kernel_pca', def 'normal_pca', str
# /!\ Currently, either LDA or PCA can be selected as tech for continuous features, later maybe enable LDA after PCA
da_tech = ''  # Select discriminant analysis tech for continuous features, 'lda' (LDA, later QDA), def 'lda', str
# /!\ Resampling_tech and pca_tech will be set to '' anyhow if disabled, scaler_tech must be given
kbest_tech = 'cramer'  # Select score function for kbest technique, 'chi2', or callable score func, str, func
pipeline_order = 'samples->features'  # Order of the steps either 'samples->features' or 'features->samples', first, str
drop_or_pass_non_treated_features = 'drop'  # Either 'drop' or 'passthrough' untransformed features, str
enable_feature_importance = True  # Change to True to enable & False to disable feature importance, default True, bool
feature_importance_method = 'all'  # Change to 'eli5', 'mlxtend', 'sklearn', or 'all' to enable methods, def. 'all', str
enable_box_bar_plots = True  # True to enable box and bar plots of most important features, default True, bool
box_bar_figures = 'combined'  # Whether the box and bar plots should be separated or combined figure, 'combined', str

###############################################################
# ## Grid search parameter intervals (Support Vector Machines)
###############################################################
# /!\ VARIABLE RULES: in the form of 'name_abbr' with abbr being the first letter of the corresponding kernel
# E.g. poly=p, rbf=r, sigmoid=s, the abbreviation must be preceded by an underscore '_', name shall be parameter name
# SVM classifier specific variables
regularization_lpsr = [x for x in np.logspace(-2, 6, 9)]  # Regularization parameter, default 1, int
shrinking_lpsr = [True, False]  # Shrinking heuristic, default True, bool
tolerance_lpsr = [x for x in np.logspace(-4, -2, 3)]  # Stopping criterion tolerance, default 0.001, float
gamma_psr = ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]  # Single training influence, default 'scale'
degree_p = [2, 3, 4, 5]  # Polynomial degree, default 3, int
coef0_ps = [0.0, 0.01, 0.1, 0.5]  # Independent term in kernel function, default 0.0, float

# Pipeline step specific variables
# With resampling enabled and resampling_tech='smote'
k_neighbors_smote_lpsr = [3, 5, 7]  # K nearest neighbor for smote resampling, default 5 (or a kneighborsmixin)
# With FT enabled regarding categorical features
k_best_lpsr = [1, 2, 5, 10, 15]  # Number of k best features to select, default 10, int
# With FT enabled and pca_tech='normal_pca', regarding continuous features
pca_lpsr = [2, 5, 10, 15, 20]  # Number of PCA components, default None, int
# With FT enabled and pca_tech='kernel_pca', regarding continuous features
kernel_pca_kernel_lpsr = ['poly', 'rbf', 'sigmoid']  # kernels for kernelPCA, default 'linear', str
kernel_pca_lpsr = [2, 5, 10, 15, 20]  # Number of components, default None, int
kernel_pca_tol_lpsr = [0.0, 0.001, 0.01]  # Tolerance, default 0, float
kernel_pca_gamma_lpsr = [None, 0.1, 1.0, 10.0]  # Gamma, default None, float
kernel_pca_degree_lpsr = [2, 3, 4, 5]  # Degree, default 3, int
kernel_pca_coef0_lpsr = [0.1, 0.5, 1.0]  # Coef0, default 1, float
# With FT enabled, pca_tech = '' and da_tech = 'lda', regarding continuous features
lda_shrinkage_lpsr = [None]  # This should be left to None if no covariance estimator is used, default None, float
lda_priors_lpsr = [None]  # Class prior prob., proportions are inferred from train data if def, default, None, np.array
lda_components_lpsr = [1]  # LDA components, if None, will be set to min(n_classes-1, n_features), default, None, int
lda_tol_lpsr = [0.0001, 0.001, 0.01]  # Tolerance for singular value x to be considered significant, default, 0.0001

########################################################################################################################
# ## Generating various parameter dictionaries based on above configuration, please double check and adapt if necessary
########################################################################################################################
# Collecting the total length of possible grid search parameters and cross validation k fold split
# /!\ DICTIONARY RULES: grid search parameters should be arranged in the same order as called above, and should contain
# an underscore (except number of splits), which should be followed by the first letter of the enabled SVC kernels
# The values of the parameter dict will be updated if any of the given parameters violates against the requirements for
# the enabled steps, i.e. pca and select k best only for feature transformation, k neighbors only for smote resampling
total_params_and_splits = {'regularization_lpsr': regularization_lpsr, 'shrinking_lpsr': shrinking_lpsr,
                           'tolerance_lpsr': tolerance_lpsr, 'gamma_psr': gamma_psr, 'coef0_ps': coef0_ps,
                           'degree_p': degree_p, 'pca_lpsr': pca_lpsr, 'k_best_lpsr': k_best_lpsr,
                           'k_neighbors_smote_lpsr': k_neighbors_smote_lpsr, 'splits': splits}

# Kernel PCA specific dictionary enabled with FT and kernel pca technique, using same rules as above
pca_kernel_dict = {'kpca_components_lpsr': kernel_pca_lpsr, 'kpca_kernel_lpsr': kernel_pca_kernel_lpsr,
                   'kpca_gamma_lpsr': kernel_pca_gamma_lpsr, 'kpca_tol_lpsr': kernel_pca_tol_lpsr,
                   'kpca_degree_lpsr': kernel_pca_degree_lpsr, 'kpca_coef0_lpsr': kernel_pca_coef0_lpsr}

# LDA specific dictionary enabled with FT and pca_tech ='', da_tech = 'lda', using same rules as above
lda_dict = {'lda_shrinkage_lpsr': lda_shrinkage_lpsr, 'lda_priors_lpsr': lda_priors_lpsr,
            'lda_components_lpsr': lda_components_lpsr, 'lda_tol_lpsr': lda_tol_lpsr}

# If any additional kernel or technique is added with a parameter interval required in grid search
# /!\ These additional TECHNIQUE parameters may require modifications in the main script to ensure correct activation
additional_params = False  # Change to True if additional non supported parameters are added, default False, bool
additional_kernel_params = {}  # Add additional kernel parameter to introduce here if not supported already
additional_technique_params = {}  # Add additional technique parameter to introduce here if not supported already

########################################################################################################################
# END OF CBD-P CONFIG ##################################################################################################
########################################################################################################################
