# Clinical Biomarker Detection - Pipeline
DRIVEN-DTU WP13: Biomarker Detection In Clinical Cohort Data Using Machine Learning

---
## Description
The Clinical Biomarker Detection pipeline presented in this repository applies pre-processing and machine learning-based approaches to identify strong biomarkers for a given disease in clinical cohort data. The pipeline is currently designed to apply Support Vector Machine Classification to predict a binary target feature (e.g. disease) in combination with other configurable techniques and processing steps that can drastically improve the prediction power.

---
## Getting Started
The repository is composed of the main pipeline script [base_II_pipeline_SVM_HPC.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_pipeline_SVM_HPC.py) and the configuration file [base_II_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_config.py) that needs to be configured in respect to the clinical data and the research topic.  
Furthermore, the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) contains all necessary functions used by the pipeline, including two modified files for both python packages `eli5` and `mlxtend`.

### Requirements (necessary for both local machine and HPC application)
* The python packages necessary for this analysis can be found and installed to your working environment via the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) file using `pip install -r requirements.txt`. Python version used: `Python version 3.8.6`.

* **/!\\ /!\\ /!\\** After installing the required packages, files in the *eli5* and *mlxtend* package folders need to be replaced with the modified files in the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) in order to enable them for parallelized computing.

  - In *eli5*: The file *permutation_importance.py* in the ***./env/lib/.../eli5/*** folder must be replaced by this [permutation_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/eli5_mod) file.
  - In *mlxtend*: The files *feature_importance.py* and *\__init__.py* in the ***./env/lib/.../mlxtend/evaluate/*** folder must be replaced by the two files [feature_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod) and [__init__.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod).

---
## The Pipeline Steps
![pipeline_flowchart_legend gv](https://user-images.githubusercontent.com/38098941/157884373-e0fc6fee-623c-4ca1-a8dd-47ba5260cbf3.svg)
* Step: Refers to the configurable processing steps
* Technique: Point where a technique must be selected
* Specification: Possible configurable technique specifications
* Starting point: Pipeline entry point
* Pipe funnel point: Pipeline funnel exit

![pipeline_flowchart gv](https://user-images.githubusercontent.com/38098941/157884274-6d565e24-1b3c-4512-b1a0-a26705945ffc.svg)
* Abbreviations:  
  - T/F: True/False
  - rus: random under-sampling  
  - smote: synthetic minority over-sampling technique

* Note:  
  - Only one possibility of pipe-order is shown in the figure above, namely *samples->features*. In case of *features->samples*, the pipeline steps IR and FT are swapped, meaning that FT is perfromed before IR. In case of IR and FT being both disabled in the configuration file, these steps will be skipped except the standard scaling mechanism of continuous features during FT which is the minimum of transformation one should at least pass to a Support Vector Machine classifier.

---
## Usage
Depending of the configured setup and user preferences, the pipeline can either be deployed using a local machine or using HPC clusters. Please note that this choice will have large effects on the required computational time for the analysis, and therefore the configuration settings should be selected appropriately and with care. The input data must exist as training and test data, preferrably cleaned and imputed (no empty values). The feature names in the data set should be preceeded by a prefix that refers to the subgroup of clinical data, e.g. body fluids (BF-), physical measurements (PM-), survey (SV-), individual medications (IM-), individual devices (ID-), ...

### Pipeline Configuration
The configuration file [base_II_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_config.py) presents 64 configurable variables and parameters that define the enabled steps, techniques, and specifications that should be highly specific to the clinical data of interest. The table below summarises the configurable variables, and more precise descriptions are available in the configuration file.

| Variable | Example | Description | Type |
| :--- | :--- | :--- | :--- |
| seed | 42  | Random seed | int |
| fix_font | 18 | Fix font size for general plots | int |
| imp_font | 8 | Specific font size for feature importance plots | int |
| plot_style | 'fivethirtyeight' | Matplotlib plot style | str |
| fig_max_open_warning | 0 | Warning shown by matplotlib after number of open figures | int |
| pandas_col_display_option | 5 | Number of columns displayed in pandas dataframe | int |
| tiff_figure_dpi | 300 | Dot per inches resolution of the result figures | int |
| curr_dir | os.getcwd() | Pathway to current directory | str, directory |
| folder_prefix | 'results/BASE-II' | Prefix of folder name for results can be a folder in folder | str |
| train_path | curr_dir + '/' + 'data/train_imputed.csv' | Path to imputed training set | str, file |
| test_path | curr_dir + '/' + 'data/test_imputed.csv' | Path to imputed training set | str, file |
| output_feature | 'PM-Frailty_Index' | Target output feature | str, binary feature |
| output_related | \['PM-Frailty_Score', 'PM-Frailty_gait', 'SV-Frailty_exhaustion', 'SV-Frailty_physicalactivity', 'PM-Frailty_gripstrength', 'PM-Gripstrength_max', 'PM-Frailty_weightloss'] | Output-related features | str, list |
| kernels | \['poly', 'rbf', 'sigmoid'] | Kernels to use for the Support Vector Machine classifier | str, list |
| non_linear_kernels | \['poly', 'rbf', 'sigmoid'] | Repeat with the above kernels that are non_linear | str, list |
| cache_size | 200 | Cache size of SVM classifier, 200 (HPC) - 2000 (local) | int |
| decision_func_shape | 'ovr' | Decision function shape of classifier, one vs rest 'ovr' or one vs one 'ovo' | str |
| clf_verbose | False | Classifier verbose | bool |
| grid_verbose | 1 | Grid search verbose | int |
| hard_iter_cap | 150000 | Hard stopping criterion | int |
| splits | 10 | Stratified k fold splits | int |
| scorer | 'F2' | Scorer used during the experimental steps, F.5, F1, F2, or accuracy | str |
| shuffle_all | 1000 | Proven 1000 for a set of 1200 samples that each sample receives at least half of the other values  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/shuffle_proof)) | int |
| shuffle_male | 500 | Proven 500 for a set of 600 samples  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/shuffle_proof)) | int |
| shuffle_female | 500 | Proven 500 for a set of 600 samples  (see [proof](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/shuffle_proof)) | int |
| parallel_method | 'ipyparallel' | Parallel backend agent, 'ipyparallel' (HPC), 'threading', 'multiprocess', 'loki' (local) | str |
| n_jobs | -1 | Number of jobs for distributed tasks, will be adjusted if ipyparallel is enabled | int |
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
| pipeline_order | 'samples->features' | Order of the steps either 'samples->features' or 'features->samples' | str |
| enable_feature_importance | True | Change to True to enable & False to disable feature importance | bool |
| regularization_lpsr | \[x for x in np.logspace(-2, 6, 9)] | Regularization parameter, default 1 | int |
| shrinking_lpsr | \[True, False] | Shrinking heuristic, default True | bool |
| tolerance_lpsr | \[x for x in np.logspace(-4, -2, 3)] | Stopping criterion tolerance, default 0.001 | float |
| gamma_psr | \['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10] | Single training influence, default | 'scale' |
| degree_p | \[2, 3, 4, 5] | Polynomial degree, default 3 | int |
| coef0_ps | \[0.0, 0.01, 0.1, 0.5] | Independent term in kernel function, default 0.0 | float |
| k_neighbors_smote_lpsr | \[3, 5] | K nearest neighbor for smote resampling, default 5 | int or a kneighborsmixin func |
| k_best_lpsr | \[5, 10, 20, 30, 45] | Number of k best features to select by chi squared, default 10 | int |
| pca_lpsr | \[5, 10, 20, 30, 45] | Number of PCA components, default None | int |
| kernel_pca_kernel_lpsr | \['poly', 'rbf', 'sigmoid'] | kernels for kernelPCA, default 'linear' | str |
| kernel_pca_lpsr | \[5, 10, 20, 30, 45] | Number of components, default None | int |
| kernel_pca_tol_lpsr | \[0.0, 0.001, 0.01] | Tolerance, default 0 | float |
| kernel_pca_gamma_lpsr | \[None, 0.1, 1.0, 10.0] | Gamma parameter, default None | float |
| kernel_pca_degree_lpsr | \[2, 3, 4, 5] | Polynomial degree, default 3 | int |
| kernel_pca_coef0_lpsr | \[0.1, 0.5, 1.0] | Coef0 parameter, default 1 | float |
| total_params_and_splits | {'regularization_lpsr': regularization_lpsr, 'shrinking_lpsr': shrinking_lpsr, 'tolerance_lpsr': tolerance_lpsr, 'gamma_psr': gamma_psr, 'coef0_ps': coef0_ps, 'degree_p': degree_p, 'pca_lpsr': pca_lpsr, 'k_best_lpsr': k_best_lpsr, 'k_neighbors_smote_lpsr': k_neighbors_smote_lpsr, 'splits': splits} | Dictionary of parameters for SVC and normal PCA | dict |
| pca_kernel_dict | {'kpca_components_lpsr': kernel_pca_lpsr, 'kpca_kernel_lpsr': kernel_pca_kernel_lpsr, 'kpca_gamma_lpsr': kernel_pca_gamma_lpsr, 'kpca_tol_lpsr': kernel_pca_tol_lpsr, 'kpca_degree_lpsr': kernel_pca_degree_lpsr, 'kpca_coef0_lpsr': kernel_pca_coef0_lpsr} | Dictionary of parameters specific to the kernel PCA technique | dict |
| additional_params | False | Change to True if additional non pre-supported parameters are added | bool |
| additional_kernel_params | {} | Add additional kernel parameter to introduce here if not supported already | dict |
| additional_technique_params | {} | Add additional technique parameter to introduce here if not supported already | dict |

---
### Run On Local Machine
For running the pipeline on a local machine it is recommended to reduce the grid search parameter intervals accordingly to guarantee reasonable computational time. For a better overview on the progress of the classification step, the `clf_verbose` can be set to `true` and the `grid_verbose` to `2`. Also, if enough computational memory is available, the SVM parameter `cache_size` can be increased to up to `2000` (mb).  
Currently, the `parallel_method` supported for local machine analysis is limited to `threading` and `multiprocess`, as the method `ipyparallel` is reserved for the analysis on the HPC clusters. the number of available workers `n_jobs` can be set to `-1` for all CPUs or to a specific number of CPUs available to the local machine.

---
### Run On HPC Cluster
For running the pipeline on HPC clusters, it is first necessary to set up and activate the appropriate environment including `Python version 3.8.6` and the required python packages that are listed in the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) using `pip install -r requirements.txt`. In the provided [regular HPC launcher script](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/HPC_SVM_launcher.sh) and the [long HPC launcher script](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/HPC_SVM_long_launcher.sh), the following information may need to be adjusted to your settings: `--mail-use` for job status notifications, `language-to-load` to specify the python language to load, and `environment-to-load` to specify the path to the environment source. Please also verify the resource allocation before submitting a job and adjust if necessary. When the environment is set up, packages installed and files in eli5 and mlxtend replaced, you can start adjusting the configuration file to the needs of your data and experimental setting. Please note that on HPC, parallelization via the `parallel_method` `ipyparallel` is recommended. The number of of jobs will be automatically set to the number of available workers based on the launcher script. The `clf_verbose` can be set to `false` and the `grid_verbose` to `1` to avoid massive printouts, and the SVM parameter `cache_size` can be adjusted to `200` (mb). On the HPC node, the files should be accessible and stored in the same way as found in this repository, and verify the path to the cleaned imputed training and test data set and the path where results should be stored so that it matches the variables in the configuration file.  

If everything is set and ready, run the pipeline with the configured experimental settings on HPC clusters using the command 'sbatch HPC_SVM_launcher.sh base_II_pipeline_SVM_HPC.py'.

---
## Results
The results will be stored in the configured `folder_prefix` folder and bear the combined and sorted abbreviations of enabled steps, e.g. `BASE-II-DS-REI-RHCF-ST-PCA-FT-RUS-FI-SVM-HPC` for a pipeline applied on BASE-II with data splitting (DS), removing engineered input (REI), removing highly correlated features (RHCF), standard scaler (ST), normal PCA (PCA), feature transformation (FT), imbalance resampling with random under-sampling (RUS), calculated feature importance (FI), using support vector machine classification (SVM) and run on the high performance computing clusters (HPC). Note the pipeline-order in the name being features first by FT, then resampling by RUS.  

Other possible abbreviations are: MI minmax scaler, RO robust scaler, SMOTE synthetic minority over-sampling technique, kPCA kernel PCA (which will be preceeded by the actual kernel if one analyses them one by one, e.g. polykPCA to save computational time).  

The results will consist of confusion matrices, roc_auc curves, summarising heatmap and venn diagram plots for RHCF, summarising plots for feature importance and shuffling effects, comparison scatter plots for the different feature importance methods, and the code execution output file generated either in the terminal (local machine) or in a readable .out file (HPC).

---
## Planned Updates
- [x] Continue editing this README file
- [ ] Extend the pipeline to allow tree-based classification
- [ ] Make pipeline generate a similar .out file of the code execution when running locally compared to HPC .out
- [ ] Add boxplot of most important features in Original data
- [ ] Make the pipeline compatible with additional processing techniques, e.g. dimensionality reduction, feature selection, ...
- [ ] Add delight to the experience when all tasks are complete :tada:
