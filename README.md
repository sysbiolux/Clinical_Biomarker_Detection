# Clinical Biomarker Detection - Pipeline
DRIVEN-DTU WP13:  
Biomarker Detection In Clinical Cohort Data Using Machine Learning
---
version 06/08/2022 (M/d/y)

---
## Description
The Clinical Biomarker Detection pipeline presented in this repository applies pre-processing and machine learning-based approaches to identify strong biomarkers for a given disease in clinical cohort data. The pipeline is currently designed to apply Support Vector Machine Classification to predict a binary target feature (e.g. disease) in combination with other configurable techniques and processing steps that can drastically improve the prediction power.

---
## Getting Started
The repository is composed of the main pipeline script [CBD_pipeline_SVM_HPC.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/CBD_pipeline_SVM_HPC.py) and the configuration file [CBDP_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/CBDP_config.py) that needs to be configured in respect to the clinical data and the research topic.  
Furthermore, the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) contains all necessary functions used by the pipeline, including modified files for the python packages `imblearn`, `eli5` and `mlxtend`.

### Requirements (necessary for both local machine and HPC application)
* The python packages necessary for this analysis can be found and installed to your working environment via the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) file using `pip install -r requirements.txt`. Python version used: `Python version 3.8.6`.

* **/!\\ /!\\ /!\\** After installing the required packages, files in the *imblearn*, *eli5* and *mlxtend* package folders need to be replaced with the modified files in the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) in order to enable them for parallelized computing and additional automations.

  - In *eli5*: The file *permutation_importance.py* in the ***./env/lib/.../eli5/*** folder must be replaced by this [permutation_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/eli5_mod) file.
  - In *mlxtend*: The files *feature_importance.py* and *\__init__.py* in the ***./env/lib/.../mlxtend/evaluate/*** folder must be replaced by the two files [feature_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod) and [\__init__.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod).
  - In *imblearn*: The file *base.py* in the ***./env/lib/.../imblearn/over_sampling/_smote/*** folder must be replaced by this [base.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/imblearn_mod) file.

---
## The Pipeline Flow
![pipeline_flowchart_legend](https://user-images.githubusercontent.com/38098941/159273463-e41be6e8-d8d8-41d3-96d9-89216e5d5c55.svg)
* Step (#9): Refers to the configurable processing steps
* Technique (#16): Point where a technique must be selected
* Specification (#43): Possible configurable technique specifications
* Starting point (#1): Pipeline entry point
* Pipe funnel point (#1): Pipeline funnel exit
* Ending point (#1): End of pipeline and results output

![pipeline_flowchart gv](https://user-images.githubusercontent.com/38098941/178989590-0669bb23-d0f5-401f-98dc-03c5e4c7107f.svg)
* Abbreviations:  
  - T/F: True/False
  - rus: random under-sampling  
  - smote: synthetic minority over-sampling technique
  - chi_sq: chi squared
  - PCA: principal component analysis
  - LDA: linear discriminant analysis

* Note:  
  - Only one possibility of pipeline-order is shown in detail in the figure above, namely *samples->features*. In case of *features->samples*, the pipeline steps IR and FT are swapped, meaning that FT is performed before IR. In case of IR and FT being both disabled in the configuration file, these steps will be skipped except the standard scaling mechanism of continuous features during FT which is the minimum of transformation one should at least pass to a Support Vector Machine classifier. The two mentioned cases are depicted as black dashed lines. Regarding the feature transformation techniques, it is possible to select a combination of continuous and categorical techniques (e.g. PCA + Select K Best) as well as to select one single transformation, e.g. PCA for the continuous features. In that case any categorical features will be passed through the pipeline without any transformation. Later update of the pipeline may include the choice between passing these features through or dropping them.
---
## The Pipeline Steps and Relevant Techniques Briefly Explained

* **Input:**  
Entry of the pipeline. The pipeline is desinged to take as input a preferrably cleaned and imputed version of the data, which is already split into training and test sets.  
* **Data splitting (DS):**  
Step of splitting the input data sets based on a given binary feature, e.g. gender, specific disease,... If no splitting feature is given or the step being disabled, it will be ignored.  
* **Subgroups (SG):**  
Step of data subgroup selections based on priorly added prefixes to the features, e.g. BF- for body fluids, PM- for physical measurements,... If no subgroups are selected or the step being disabled, all input features will be passed to the next step. This step is specifically designed to match and select given subgroups, but individual feature names could be added as well to the selection.  
* **Remove engineered input (REI):**  
Step to remove features or group of features that are already represented in an engineered feature of the data set to avoid having redundant information and correlated features in the data set. Such features can include entire subgroups that were engineered into other features, but also individual feature names could be added to this list. If no features are selected or the step being disabled, this step will be ignored.  
* **Constancy check (CC):**  
Step to remove features that are constant or near-constant. Regarding near-constants, there are three different applications involved depending on the feature data type. In case of continuous feature, the variance-to-mean ratio is calculated and defined as near-constant if this ratio does not meed the minimum threshold given in the configuration file. In case of categorical features, binary types are defined as near-constant if the sum of positive class does not exceed the number of stratified splits during cross-validation as given in the configuration file. Non-binary categorical features are defined as near-constant if the sum of all non-zero classes does not exceed the number of stratified splits during cross-validation.  
* **Remove highly correlated features (RHCF):**  
Step to remove the highly correlated features in the data set for the following associations of possible feature data types: continuous-continuous, categorical-categorical, and continuous-categorical correlation. If disabled, the step will be ignored. The following techniques will be applied.
  - continuous-continuous: Spearman's Rank Order Correlation using decimal or percentile threshold
  - categorical-categorical: Corrected Cramer's V Correlation using decimal or percentile threshold
  - continuous-categorical: Point Bi-serial correlation using decimal or percentile threshold (correlated features will be removed from the longer list)  
* **Imbalance resampling (IR):**  
Step of the classification pipeline to resample imbalanced data. The order of steps in the classification pipeline can be defined in the configuration file, e.g. resampling before feature transformation or vice versa. If disabled, the step will be ignored. If enabled, the following techniques can be selected.
  - Random under-sampling (RUS): Randomly select majority class samples to equal the number of minority class samples.
  - Synthetic minority over-sampling technique (SMOTE): Creation of synthetic minority class samples using k-nearest neighbors algorithm to equal the number of majority class samples. In more detail, the variant SMOTENC is used which is specifically designed to over-sample continuous and categorical features together (original SMOTE does not make a difference between the two types). To ensure proper functioning of SMOTENC in all pipeline-order cases, it is recommended to update the base SMOTENC function as explained above in the 'Requirements' section.  
* **Feature transformation (FT):**  
Step of the classification pipeline to transform the features. The order of steps in the classification pipeline can be defined in the configuration file, e.g. resampling before feature transformation or vice versa. If disabled, the step will be ignored. If enabled, the following techniques can be defined.
  - Scaler technique: Select between the standard scaler (distribution centered around 0, standard deviation of 1, mean removed), robust scaler (median and scales removed according to the quantile range), or Minmax scaler (scaling each feature to a specific range like \[0, 1]). The scaler technique will only be applied on the continuous features, with standard scaler being the default if none is selected. The default scaling technique will also be applied alone in case this step is disabled.
  - Feature technique: Select between linear PCA, LDA and non-linear kernel PCA for continuous features. For categorical features, currently only the *select k best method* using chi squared and Cramer's V correlation is available. If the step is disabled, the features will not be transformed. Please note that in case of non-linear PCA, the classifier kernel will be forced to be linear in order to avoid applying non-linear kernel transformations twice (if linear PCA or LDA is selected, non-linear classifier kernels are allowed).  
* **Feature importance (FI):**  
Step to identify the most important features selected by the classification model if this step is enabled. In case of linear classification, feature importance by permutation is not necessary and the information can be retrieved directly from the trained estimator using the built-in `SVC.coef_` attribute. In case of non-linear classification, the feature importance is measured using the feature permutation algorithm. In this case, it is possible to choose between three different methods that do show consistent results. It is also possible to select all and validate the result's consistency by yourself.
  - sklearn's *permutation_importance* function of the sklearn.inspection group.
  - eli5's *get_score_importance* function of the eli5.permutation_importance group (mod files required).
  - mlxtend's *feature_importance_permutation* function of the mlxtend.evaluate group (mod files required).
  - 'all' to run the permuted feature importance with all three methods and to plot comparisons (mod files required).  
* **Box and bar plotting (BBP):**  
Step to visualize the most important features in a ranked order between the negative and positive classes. If disabled, the step will be ignored. If enabled, the most important categorical and continuous features can be plotted separately or combined as defined in the configuration file.  
* **Output:**  
Classification model of the selected output-target, model evaluation summaries and plots, e.g. confusion matrices, ROC-AUC curves, performance metrics, summary plots for the various enabled steps like heatmaps of the highly correlated features, venn diagrams of removed features if data is split, comparison plots of feature importance methods if all enabled, list of features ranked by their importancy, ...

---
## Usage
Depending of the configured setup and user preferences, the pipeline can either be deployed using a local machine or using HPC clusters. Please note that this choice will have large effects on the required computational time for the analysis, and therefore the configuration settings should be selected appropriately and with care. The input data must exist as training and test data, preferrably cleaned and imputed (no empty values). The feature names in the data set should be preceeded by a prefix that refers to the subgroup of clinical data, e.g. body fluids (BF-), physical measurements (PM-), survey (SV-), individual medications (IM-), individual devices (ID-), ...

### Pipeline Configuration
The configuration file [CBDP_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/CBDP_config.py) presents 78 configurable variables and parameters that define the enabled steps, techniques, and specifications that should be highly specific to the clinical data of interest. The table below summarises the configurable variables, and more precise descriptions are available in the configuration file.

#### General Settings

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| seed | 42  | Random seed | int |
| fix_font | 18 | Fix font size for general plots | int |
| imp_font | 8 | Specific font size for feature importance plots | int |
| plot_style | 'fivethirtyeight' | Matplotlib plot style | str |
| fig_max_open_warning | 0 | Warning shown by matplotlib after number of open figures | int |
| pandas_col_display_option | 5 | Number of columns displayed in pandas dataframe | int |
| tiff_figure_dpi | 300 | Dot per inches resolution of the result figures | int |

#### Data and Topic-specific Settings

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| curr_dir | os.getcwd() | Pathway to current directory | str, directory |
| folder_prefix | 'results/' | Folder name for results can be a folder in folder or prefix | str |
| train_path | curr_dir + '/data/train_imputed.csv' | Path to imputed training set | str, file |
| test_path | curr_dir + '/data/test_imputed.csv' | Path to imputed training set | str, file |
| output_feature | 'PM-Frailty_Index' | Target output feature | str, binary feature |
| positive_class | 'frail' | Name to give the positive class of the output feature | str |
| negative_class | 'non-frail' | Name to give the negative class of the output feature | str |
| output_related | \['PM-Frailty_Score', 'PM-Frailty_gait', 'SV-Frailty_exhaustion', 'SV-Frailty_physicalactivity', 'PM-Frailty_gripstrength', 'PM-Gripstrength_max', 'PM-Frailty_weightloss'] | Output-related features | str, list |

#### Machine Learning Classifier-specific Fixed Parameters

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| kernels | \['linear', 'poly', 'rbf', 'sigmoid'] | Kernels to use for the Support Vector Machine classifier | str, list |
| non_linear_kernels | \['poly', 'rbf', 'sigmoid'] | Repeat with the above kernels that are non_linear | str, list |
| cache_size | 200 | Cache size of SVM classifier, 200 (HPC) - 2000 (local) | int |
| decision_func_shape | 'ovr' | Decision function shape of classifier, one vs rest 'ovr' or one vs one 'ovo' | str |
| clf_verbose | False | Classifier verbose | bool |
| grid_verbose | 1 | Grid search verbose | int |
| hard_iter_cap | 150000 | Hard stopping criterion | int |
| splits | 10 | Stratified k fold splits | int |
| scorer | 'F2' | Scorer used during the experimental steps, F.5, F1, F2, or accuracy | str |
| shuffle_all | 1000 | Proven 1000 for a set of 1200 samples that each sample receives at least half of the other values  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/shuffle_proof/shuffle_me.py)) | int |
| shuffle_male | 500 | Proven 500 for a set of 600 samples  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/shuffle_proof/shuffle_me.py)) | int |
| shuffle_female | 500 | Proven 500 for a set of 600 samples  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/shuffle_proof/shuffle_me.py)) | int |
| linear_shuffle | True | Feature importance by shuffling in case of PCA+linear SVM if true, else the .`coef_` attribute of the linear SVM is used and the sorted averaged loadings of features across all selected PCA components are used to determine the most important features | bool |

#### Selecting Parallel Backend, Enabled Steps and Technical Specifications

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| parallel_method | 'ipyparallel' | Parallel backend agent, 'ipyparallel' (HPC), 'threading', 'multiprocess', 'loki' (local) | str |
| n_jobs | -1 | Number of jobs for distributed tasks, will be adjusted if ipyparallel is enabled | int |
| thresh_near_constant | 0.001 | Thresh for a continuous feature near-constance by variance-mean-ratio | float |
| enable_data_split | True | True if data should be split based on the binary split feature below | bool |
| split_feature | 'PM-sex' | Feature based on which data is split | str |
| enable_subgroups | False | True if data shall be limited to subgroups, else full feature input | bool |
| subgroups_to_keep | 'all' | Prefix of subgroups to keep for the analysis | tuple of str, str or 'all' |
| enable_engineered_input_removal | True | Change to enable or disable removal of engineered input features | bool |
| engineered_input_prefix | ('IM-', 'ID-') | Prefix of features used in engineering | str, tuple of str, or empty |
| enable_rhcf | True | Change to True to enable & False to disable removing highly correlated features | bool |
| thresh_cramer | (0.6, 'decimal') | Corrected Cramer's V correlation threshold, choose decimal or percentile | tuple of int or float and str |
| thresh_spearman | (95, 'percentile') | Spearman's Rank correlation threshold, choose decimal or percentile | tuple of int or float and str |
| thresh_pbs | (0.6, 'decimal') | Point bi-serial correlation threshold, choose decimal or percentile | tuple of int or float and str |
| enable_resampling | True | Change to True to enable & False to disable resampling | bool |
| resampling_tech | 'rus' | 'rus' (random under-sampling), 'smote' (synthetic minority over-sampling technique) or empty | str |
| enable_ft | True | Change to True to enable & False to disable feature transformation | bool |
| scaler_tech | 'standard' | Change scaler function to 'standard', 'minmax' or 'robust' scaler | str |
| pca_tech | 'normal_pca' | Select pca technique to choose between 'normal_pca' and 'kernel_pca' | str |
| da_tech | 'lda' | Select discriminant analysis tech for continuous features, 'lda' (LDA, later QDA) | str |
| kbest_tech | 'cramer' | Select score function for kbest technique, 'chi2', 'cramer', or callable score func | str or callable |
| pipeline_order | 'samples->features' | Order of the steps either 'samples->features' or 'features->samples' | str |
| enable_feature_importance | True | Change to True to enable & False to disable feature importance | bool |
| feature_importance_method | 'all' | Change to 'eli5', 'mlxtend', 'sklearn', or 'all' to enable methods, default 'all' | str |
| enable_box_bar_plots | True | True to enable box and bar plots of most important features or False to disable, default True | bool |
| box_bar_figures | 'combined' | Whether the box and bar plots should be separated or combined figure, 'separated' or 'combined' | str |

#### Machine Learning Classifier-specific Parameters For Grid Search

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| regularization_lpsr | \[x for x in np.logspace(-2, 6, 9)] | Regularization parameter, default 1 | int |
| shrinking_lpsr | \[True, False] | Shrinking heuristic, default True | bool |
| tolerance_lpsr | \[x for x in np.logspace(-4, -2, 3)] | Stopping criterion tolerance, default 0.001 | float |
| gamma_psr | \['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10] | Single training influence, default | 'scale' |
| degree_p | \[2, 3, 4, 5] | Polynomial degree, default 3 | int |
| coef0_ps | \[0.0, 0.01, 0.1, 0.5] | Independent term in kernel function, default 0.0 | float |
| k_neighbors_smote_lpsr | \[2, 3, 5] | K nearest neighbor for smote resampling, default 5 | int or a kneighborsmixin func |
| k_best_lpsr | \[1, 2, 5, 10, 15] | Number of k best features to select by chi squared, default 10 | int |
| pca_lpsr | \[2, 5, 10, 15, 20] | Number of PCA components, default None | int |
| kernel_pca_kernel_lpsr | \['poly', 'rbf', 'sigmoid'] | kernels for kernelPCA, default 'linear' | str |
| kernel_pca_lpsr | \[2, 5, 10, 15, 20] | Number of components, default None | int |
| kernel_pca_tol_lpsr | \[0.0, 0.001, 0.01] | Tolerance, default 0 | float |
| kernel_pca_gamma_lpsr | \[None, 0.1, 1.0, 10.0] | Gamma parameter, default None | float |
| kernel_pca_degree_lpsr | \[2, 3, 4, 5] | Polynomial degree, default 3 | int |
| kernel_pca_coef0_lpsr | \[0.1, 0.5, 1.0] | Coef0 parameter, default 1 | float |
| lda_shrinkage_lpsr | \[None] | This should be left to None if no covariance estimator is used, default None | float |
| lda_priors_lpsr | \[None] | Class prior prob., proportions are inferred from train data if def, default, None | np.array |
| lda_components_lpsr | \[1] | LDA components, if None, will be set to min(n_classes-1, n_features), default, None | int |
| lda_tol_lpsr | \[0.0001, 0.001, 0.01] | Tolerance for singular value x to be considered significant, default, 0.0001 | float |


#### Dictionaries Based on the Above Configuration For Summaries

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| total_params_and_splits | {'regularization_lpsr': regularization_lpsr, 'shrinking_lpsr': shrinking_lpsr, 'tolerance_lpsr': tolerance_lpsr, 'gamma_psr': gamma_psr, 'coef0_ps': coef0_ps, 'degree_p': degree_p, 'pca_lpsr': pca_lpsr, 'k_best_lpsr': k_best_lpsr, 'k_neighbors_smote_lpsr': k_neighbors_smote_lpsr, 'splits': splits} | Dictionary of parameters for SVC and normal PCA | dict |
| pca_kernel_dict | {'kpca_components_lpsr': kernel_pca_lpsr, 'kpca_kernel_lpsr': kernel_pca_kernel_lpsr, 'kpca_gamma_lpsr': kernel_pca_gamma_lpsr, 'kpca_tol_lpsr': kernel_pca_tol_lpsr, 'kpca_degree_lpsr': kernel_pca_degree_lpsr, 'kpca_coef0_lpsr': kernel_pca_coef0_lpsr} | Dictionary of parameters specific to the kernel PCA technique | dict |
| lda_dict | {'lda_shrinkage_lpsr': lda_shrinkage_lpsr, 'lda_priors_lpsr': lda_priors_lpsr, 'lda_components_lpsr': lda_components_lpsr, 'lda_tol_lpsr': lda_tol_lpsr} | Dictionary of parameters specific to the DA technique (currently only LDA) | dict |
| additional_params | False | Change to True if additional non pre-supported parameters are added | bool |
| additional_kernel_params | {} | Add additional kernel parameter to introduce here if not supported already | dict |
| additional_technique_params | {} | Add additional technique parameter to introduce here if not supported already | dict |

---
### Run On Local Machine
For running the pipeline on a local machine it is recommended to reduce the grid search parameter intervals accordingly to guarantee reasonable computational time. For a better overview on the progress of the classification step, the `clf_verbose` can be set to `true` and the `grid_verbose` to `2`. Also, if enough computational memory is available, the SVM parameter `cache_size` can be increased to up to `2000` (mb).  
Currently, the `parallel_method` supported for local machine analysis is limited to `threading` and `multiprocess`, as the method `ipyparallel` is reserved for the analysis on the HPC clusters. the number of available workers `n_jobs` can be set to `-1` for all CPUs or to a specific number of CPUs available to the local machine.

---
### Run On HPC Cluster
For running the pipeline on HPC clusters, it is first necessary to set up and activate the appropriate environment including the `Python version 3.8.6` and the required python packages that are listed in the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) using `pip install -r requirements.txt`.  
In the provided [regular HPC launcher script](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/HPC_SVM_launcher.sh) and the [long HPC launcher script](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/HPC_SVM_long_launcher.sh), the following information may need to be adjusted to your settings:  
* `--mail-use` for job status notifications (fill-in valid e-mail address and uncomment if needed)
* `language_to_load` to specify the python language module to load
* `environment_to_load` to specify the path to the environment source
* Please also verify the resource allocation before submitting a job and adjust if necessary.  

When the environment is set up, packages installed and files in eli5, mlxtend, and smote package directories replaced, you can start adjusting the configuration file to the needs of your data and experimental setting. Please note that on HPC, parallelization via the `parallel_method` `ipyparallel` is highly recommended. The number of jobs will be automatically set to the number of available workers based on the launcher script. The `clf_verbose` can be set to `false` and the `grid_verbose` to `1` to avoid massive printouts, and the SVM parameter `cache_size` can be adjusted to `200` (mb).  

On the HPC node, the files should be accessible and stored in the same way as found in this repository, and verify the path to the cleaned imputed training and test data set and the path where results should be stored so that it matches the variables in the configuration file.  

If everything is set and ready, run the pipeline with the configured experimental settings on HPC clusters using the below command:  
`sbatch HPC_SVM_launcher.sh CBD_pipeline_SVM_HPC.py`

---
## Results
The results will be stored in the configured `folder_prefix` folder and bear the combined and sorted abbreviations of enabled steps, e.g. `<possible_prefix>-DS-REI-RHCF-ST-chi2KBEST-PCA-FT-RUS-FI-BBP-SVM-both-lin-and-non-lin-HPC` for a pipeline deployed with data splitting (DS), removing engineered input (REI), removing highly correlated features (RHCF), standard scaler (ST), chi2 SelectKBest (chi2KBEST) normal PCA (PCA), feature transformation (FT), imbalance resampling with random under-sampling (RUS), calculated feature importance (FI), with box and bar plots (BBP), using support vector machine classification (SVM) with linear and non-linear kernels (both-lin-and-non-lin) and run on the high performance computing clusters (HPC). Note the pipeline-order in the name being features first by FT, then resampling by RUS.  

Other possible abbreviations are: MI minmax scaler, RO robust scaler, SMOTE synthetic minority over-sampling technique, kPCA kernel PCA (which will be preceeded by the actual kernel if one analyses them one by one, e.g. polykPCA to save computational time). 

The results will consist of confusion matrices, roc_auc curves, summarising heatmap and venn diagram plots for RHCF, summarising plots for feature importance and shuffling effects, comparison scatter plots for the different feature importance methods, and the code execution output file generated either in the terminal (local machine) or in a readable .out file (HPC).

---
## Planned Updates
- [x] ~Continue editing this README file~ 03/14/2022
- [x] ~Add boxplot of most important features in Original data~ 03/18/2022
- [x] ~Make pipeline generate a similar .out file of the code execution when running locally compared to HPC .out~ 03/23/2022
- [x] ~In case of linear SVM kernel combined with linear PCA, enable extraction of feature importance by shuffling and by the `.coef_` attribute of the linear classifier~ 05/30/2022
- [x] ~Near-constancy check added for continuous and binary features~ 05/30/2022
- [x] ~Enable launching analysis with any combination of the supported techniques, including single or double feature transformation~ 05/30/2022
- [x] ~Generate detailed list of highly correlated features during that pre processing step~ 05/30/2022
- [x] ~Added corrected Cramer's V correlation as a possible score function for selectKBest~ 05/30/2022
- [x] ~Updated the pipeline flow-chart and README descriptions~ 07/14/2022
- [ ] Extend the pipeline to allow tree-based classification
- [ ] Make the pipeline compatible with additional processing techniques, e.g. dimensionality reduction, feature selection, ...
- [ ] Add delight to the experience when all tasks are complete :tada:
