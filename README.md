# Clinical Biomarker Detection
DRIVEN-DTU WP13: Biomarker Detection In Clinical Cohort Data Using Machine Learning

## Description
The Clinical Biomarker Detection pipeline presented in this repository applies pre-processing and machine learning-based approaches to identify strong biomarkers for a given disease in clinical cohort data. The pipeline is currently designed to apply Support Vector Machine Classification to predict a binary target feature (e.g. disease) in combination with other configurable techniques and processing steps that can drastically improve the prediction power.

## Getting Started
The repository is composed of the main pipeline script [base_II_pipeline_SVM_HPC.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_pipeline_SVM_HPC.py) and the configuration file [base_II_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_config.py) that needs to be configured in respect to the clinical data and the research topic.  
Furthermore, the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) contains all necessary functions used by the pipeline, including two modified files for both python packages `eli5` and `mlxtend`.

### Requirements
* The python packages necessary for this analysis can be found and installed to your working environment via the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) file using `pip install -r requirements.txt`.

* After installing the required packages, files in the *eli5* and *mlxtend* package folders need to be replaced with the modified files in the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) in order to enable them for parallelized computing.

  - In *eli5*: The file *permutation_importance.py* in the ***./env/lib/.../eli5/*** folder must be replaced by this [permutation_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/eli5_mod) file.
  - In *mlxtend*: The files *feature_importance.py* and *__init__.py* in the ***./env/lib/.../mlxtend/evaluate/*** folder must be replaced by the two files [feature_importance.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod) and [__init__.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/tree/main/source/mlxtend_mod).

## The Pipeline Steps
![pipeline_flowchart_legend gv](https://user-images.githubusercontent.com/38098941/157884373-e0fc6fee-623c-4ca1-a8dd-47ba5260cbf3.svg)
* Step: Refers to the configurable processing steps
* Technique: Point where a technique must be selected
* Specification: Possible configurable technique specifications
* Starting point: Pipeline entry point
* Pipe funnel point: Pipeline end point
![pipeline_flowchart gv](https://user-images.githubusercontent.com/38098941/157884274-6d565e24-1b3c-4512-b1a0-a26705945ffc.svg)
* Abbreviations:  
  - rus: random under-sampling  
  - smote: synthetic minority over-sampling technique

## Usage

### Pipeline Configuration

### Run On Local Machine

### Run On HPC Cluster

## Results

## Planned Updates
