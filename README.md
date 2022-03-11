# Clinical Biomarker Detection
DRIVEN-DTU WP13: Biomarker Detection In Clinical Cohort Data Using Machine Learning

## Description
The Clinical Biomarker Detection pipeline presented in this repository applies pre-processing and machine learning-based approaches to identify strong biomarkers for a given disease in clinical cohort data. The pipeline is currently designed to apply Support Vector Machine Classification to predict a binary target feature (e.g. disease) in combination with other configurable techniques and processing steps that can drastically improve the prediction power.

## Getting Started
The repository is composed of the main pipeline script [base_II_pipeline_SVM_HPC.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_pipeline_SVM_HPC.py) and the configuration file [base_II_config.py](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/base_II_config.py) that needs to be configured in respect to the clinical data and the research topic. Furthermore, the [source folder](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/source/) contains all necessary functions used by the pipeline, including two modified files for both python packages `eli5` and `mlxtend`.

### Requirements
The python packages necessary for this analysis can be found and installed via the [requirements.txt](https://github.com/sysbiolux/Clinical_Biomarker_Detection/blob/main/requirements.txt) file using `pip install -r requirements.txt` to your working environment. The pipeline is designed to either run on local machine or on HPC clusters
which can be defined in the configuration file. 

## The Pipeline Steps

## Usage

### Pipeline Configuration

### Run On Local Machine

### Run On HPC Cluster

## Results

## Planned Updates
