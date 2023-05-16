########################################################################################################################
# BASE-II CLUSTERING ANALYSIS  #########################################################################################
# USING PCA, LDA, t-SNE and UMAP TO CLUSTER FRAILTY INDEX IN BASE-II MIXED, MALE, AND FEMALE ###########################
# After constancy check, before and after RHCF, for the complete data, the different continuous subgroups, and scalers #
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# February 2023, University of Luxembourg, v.02/26/2023 (M/d/y) ########################################################
########################################################################################################################

# Testing for: Mixed, male, female, complete, train, test, constancy threshold 0.01, 0.001, ALL, PM, BF, NT, CG,
# before and after RHCF, standard, robust, minmax scaler, PCA, LDA, t-SNE perplexities, UMAP neighbors

# Takes 24-30 hours

############################################
# ## Importing libraries and util functions
############################################
# housekeeping libraries
import matplotlib
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# util functions
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sys.path.append('../')
from source.CBDP_utils import separate_full_data, \
    get_cat_and_cont, check_constant_features, applied_cont_rhcf, spearman_heatmap  # from BASE-II project
from plot_functions import plot_PCA, plot_LDA, plot_tSNE, plot_UMAP  # from Circ-TNBC project

########################
# ## Constant variables
########################
seed = 42
fix_font = 18
plot_style = 'fivethirtyeight'
fig_max_open_warning = 0
pandas_col_display_option = 5
# Pathway to current directory and data
curr_dir = os.getcwd()
train_path = '../data/train_imputed.csv'
test_path = '../data/test_imputed.csv'
# target feature and target related features
output_feature = 'PM-Frailty_Index'
positive_class = 'frail'
negative_class = 'non-frail'
split_feature = 'PM-sex'
output_related = ['PM-Frailty_Score',
                  'PM-Frailty_gait',
                  'SV-Frailty_exhaustion',
                  'SV-Frailty_physicalactivity',
                  'PM-Frailty_gripstrength',
                  'PM-Frailty_weightloss',
                  'PM-Gripstrength_max']
splits = 10
tiff_figure_dpi = 300  # only for spearman heatmap, cluster plots will be in png format
debug = False  # debug, if you wish to see all figures pop-up ...

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
matplotlib.use('Agg' if not debug else 'TkAgg')
# suppress FutureWarning from t-SNE and UserWarning from UMAP
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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

# Split the data based on the given split feature
print(f'\nSplitting the data based on {split_feature} ...')
train_features, test_features, train_labels, test_labels, feature_list, \
    train_men_features, test_men_features, train_men_labels, test_men_labels, \
    train_female_features, test_female_features, train_female_labels, test_female_labels, \
    feature_list_wo_gender = separate_full_data(full_train=train, full_test=test,
                                                target_feature=output_feature, splitting_feature=split_feature)

# Print the binary counts and ratio of negative and positive classes in the train and test sets
print(f'\n{negative_class.capitalize()}/{positive_class.capitalize()} counts in the full train set:',
      np.bincount(train_labels), '\nratio:', round(np.bincount(train_labels)[0] / np.bincount(train_labels)[1], 3))
print(f'{negative_class.capitalize()}/{positive_class.capitalize()} counts in the full test set:',
      np.bincount(test_labels), '\nratio:', round(np.bincount(test_labels)[0] / np.bincount(test_labels)[1], 3))
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

# keep only the continuous features to reduce input data and fasten up below application
cont_mixed, _ = get_cat_and_cont(train_features, test_features)
cont_male, _ = get_cat_and_cont(train_men_features, test_men_features)
cont_female, _ = get_cat_and_cont(train_female_features, test_female_features)

# update data sets and generate a complete one for each
# mixed
train_features = train_features[:, cont_mixed]
test_features = test_features[:, cont_mixed]
comp_features = np.concatenate((train_features, test_features))
# male
train_men_features = train_men_features[:, cont_male]
test_men_features = test_men_features[:, cont_male]
comp_features_male = np.concatenate((train_men_features, test_men_features))
# female
train_female_features = train_female_features[:, cont_female]
test_female_features = test_female_features[:, cont_female]
comp_features_female = np.concatenate((train_female_features, test_female_features))

# update feature lists
feature_list = [feature_list[x] for x in range(len(feature_list)) if x in cont_mixed]
feature_list_male = [feature_list_wo_gender[x] for x in range(len(feature_list_wo_gender)) if x in cont_male]
feature_list_female = [feature_list_wo_gender[x] for x in range(len(feature_list_wo_gender)) if x in cont_female]

# Use the following 3 feature list variables if you wish to analyse for a specific combination of features instead
# of subgroups (If you wish so, you can also adapt the 'subgroups' variable further below to ['SPECIFIC'] only
#feature_list = []
#feature_list_male = []
#feature_list_female = []

# Testing for: Mixed, male, female, complete, train, test, constancy threshold 0.01, 0.001, ALL, PM, BF, NT, CG,
# before and after RHCF, standard, robust, minmax scaler, PCA, LDA, t-SNE perplexities, UMAP neighbors
data_sets = [train_features, test_features, comp_features,
             train_men_features, test_men_features, comp_features_male,
             train_female_features, test_female_features, comp_features_female]
target_labels = [train_labels, test_labels, np.concatenate((train_labels, test_labels)),
                 train_men_labels, test_men_labels, np.concatenate((train_men_labels, test_men_labels)),
                 train_female_labels, test_female_labels, np.concatenate((train_female_labels, test_female_labels))]
data_sets_verbosity = ['mixed training', 'mixed test', 'mixed complete',
                       'male training', 'male test', 'male complete',
                       'female training', 'female test', 'female complete']
feature_lists = [feature_list, feature_list, feature_list,
                 feature_list_male, feature_list_male, feature_list_male,
                 feature_list_female, feature_list_female, feature_list_female]
scalers = ['standard', 'robust', 'minmax']
near_constancy_thresholds = [0.01, 0.001]
subgroups = ['ALL', 'PM-', 'BF-', 'NT-', 'CG-']  # all other subgroups have no or only 1 continuous feature
with_rhcf = ['before RHCF', 'after RHCF']
rhcf_thresholds = [(0.9, 'decimal'), (95, 'percentile')]  # only need cont-cont Spearman' Rank Order here
cluster_test = ['PCA', 'LDA', 't-SNE', 'UMAP']
n_perplexity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_neighbors = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


#########################################################
# ## Loop to test only No transformation and Scaler Only
#########################################################
for num, data in enumerate(data_sets):
    print(f'\n\n*** Analysing {data_sets_verbosity[num]}... *** \n\n')
    folder_name = data_sets_verbosity[num].replace(' ', '_')
    if os.path.isdir(curr_dir + '/' + folder_name) is False:
        os.mkdir(curr_dir + '/' + folder_name)
    #########################
    # ## Retaining subgroups
    #########################
    for subs in subgroups:
        print(f'Retaining the following subgroups: {subs}\n')
        if subs == 'ALL':
            data_sg = data.copy()
            new_feature_list_sg = feature_lists[num].copy()
        if subs == 'SPECIFIC':  # specific combination of features
            data_sg = pd.DataFrame(data, columns=feature_lists[num])
            new_feature_list_sg = feature_lists[num].copy()
        else:  # defined subgroups
            data_sg = pd.DataFrame(data, columns=feature_lists[num])  # from array in to pandas to easily drop
            new_feature_list_sg = \
                [feature_lists[num][x] for x in range(len(feature_lists[num])) if feature_lists[num][x].startswith(subs)]
            data_sg = np.array(data_sg[new_feature_list_sg])  # drop and back to np array
            print(f'The shape of the {data_sets_verbosity[num]} set with selected subgroups is:\n{data_sg.shape}\n')
        # plot PCA While nothing is transformed
        plot_PCA(X=pd.DataFrame(data_sg, columns=new_feature_list_sg),
                 y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                 label=output_feature,
                 title=f'BASE-II PCA analysis\n'
                       f'{data_sets_verbosity[num]}, before RHCF, no continuous threshold, {subs} subgroup(s), '
                       f'no scaler', seed=seed)
        plt.savefig(f'./{folder_name}/Untransformed_PCA_{subs}',
                    bbox_inches='tight')
        plt.close()
        for scaler in scalers:
            data_scaled = None
            if scaler == 'standard':
                data_scaled = StandardScaler().fit_transform(data_sg)
                print('Processing standard scaling.\n')
            if scaler == 'robust':
                data_scaled = RobustScaler().fit_transform(data_sg)
                print('Processing robust scaling.\n')
            if scaler == 'minmax':
                data_scaled = MinMaxScaler().fit_transform(data_sg)
                print('Processing minmax scaling.\n')
            plot_PCA(X=pd.DataFrame(data_scaled, columns=new_feature_list_sg),
                     y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                     label=output_feature,
                     title=f'BASE-II PCA analysis\n'
                           f'{data_sets_verbosity[num]}, before RHCF, '
                           f'no continuous threshold, {subs} subgroup(s), {scaler} scaled', seed=seed)
            plt.savefig(f'./{folder_name}/{scaler}_scaled_only_PCA_{subs}',
                        bbox_inches='tight')
            plt.close()
print('Analysis of untransformed or scaled only PCA in all subgroups done!')
###################################
# ## End None and Scaler Only Loop
###################################


########################################
# ## Starting the Cluster analysis loop
########################################
for num, data in enumerate(data_sets):
    print(f'\n\n*** Analysing {data_sets_verbosity[num]}... *** \n\n')
    folder_name = data_sets_verbosity[num].replace(' ', '_')
    if os.path.isdir(curr_dir + '/' + folder_name) is False:
        os.mkdir(curr_dir + '/' + folder_name)
    for threshold in near_constancy_thresholds:
        ##############################################
        # ## Checking for remaining constant features
        ##############################################
        print(f'\n**Check and remove near constant continuous feature with treshold {threshold}...**\n')
        constants = check_constant_features(feature_lists[num], data, data_sets_verbosity[num], nbr_splits=splits,
                                            near_constant_thresh=threshold)
        # Update the feature lists
        new_feature_list = [feature_lists[num][x] for x in range(len(feature_lists[num])) if x not in constants]
        # Remove those features
        data_rem = data.copy()
        data_rem = np.delete(data_rem, constants, axis=1)
        print(f'The shape of the {data_sets_verbosity[num]} set after pre-processing is:\n {data_rem.shape}\n')
        for subs in subgroups:
            #########################
            # ## Retaining subgroups
            #########################
            print(f'Retaining the following subgroups: {subs}\n')
            if subs == 'ALL':
                data_sg = data_rem.copy()
                new_feature_list_sg = new_feature_list.copy()
            if subs == 'SPECIFIC':  # specific combination of features
                data_sg = pd.DataFrame(data, columns=feature_lists[num])
                new_feature_list_sg = feature_lists[num].copy()
            else:  # defined subgroups
                data_sg = pd.DataFrame(data_rem, columns=new_feature_list)  # from array in to pandas to easily drop
                new_feature_list_sg = \
                    [new_feature_list[x] for x in range(len(new_feature_list)) if new_feature_list[x].startswith(subs)]
                data_sg = np.array(data_sg[new_feature_list_sg])  # drop and back to np array
                print(f'The shape of the {data_sets_verbosity[num]} set with selected subgroups is:\n{data_sg.shape}\n')
            for rhcf_status in with_rhcf:
                ###########################################################################
                # ## Applying computation and removal of highly correlated features (RHCF)
                ###########################################################################
                print(f'Analysing clusterisation {rhcf_status}.\n')
                if rhcf_status == 'before RHCF':
                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}') is False:
                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}')
                    data_rhcf = data_sg.copy()
                    new_feature_list_rhcf = new_feature_list_sg.copy()
                    # Apply standard, minmax, or robust scaler
                    for scaler in scalers:
                        data_scaled = None
                        if scaler == 'standard':
                            data_scaled = StandardScaler().fit_transform(data_rhcf)
                            print('Processing standard scaling.\n')
                        if scaler == 'robust':
                            data_scaled = RobustScaler().fit_transform(data_rhcf)
                            print('Processing robust scaling.\n')
                        if scaler == 'minmax':
                            data_scaled = MinMaxScaler().fit_transform(data_rhcf)
                            print('Processing minmax scaling.\n')
                        # start analysis plotting here
                        for analysis in cluster_test:
                            ################
                            # ## Clustering
                            ################
                            print(f"Starting {analysis} cluster analysis for {data_sets_verbosity[num]}.\n")
                            if analysis == 'PCA':
                                if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is False:
                                    os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                plot_PCA(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                         y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                         label=output_feature,
                                         title=f'BASE-II PCA analysis\n'
                                               f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                               f'continuous threshold {threshold}, {subs} subgroup(s), {scaler} scaled',
                                         seed=seed)
                                plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_PCA_'
                                            f'{subs}_{rhcf_status}_cont_thresh_{str(threshold).replace(".", "-")}',
                                            bbox_inches='tight')
                                plt.close()
                            if analysis == 'LDA':
                                if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is False:
                                    os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                plot_LDA(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                         y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                         label=output_feature,
                                         title=f'BASE-II LDA analysis\n'
                                               f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                               f'continuous threshold {threshold}, {subs} subgroup(s), {scaler} scaled',
                                         lda_output_target=output_feature,
                                         seed=seed)
                                plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_LDA_'
                                            f'{subs}_{rhcf_status}_cont_thresh_{str(threshold).replace(".", "-")}',
                                            bbox_inches='tight')
                                plt.close()
                            if analysis == 't-SNE':
                                if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is False:
                                    os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                for perplexity in n_perplexity:
                                    plot_tSNE(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                              y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                              label=output_feature,
                                              title=f'BASE-II t-SNE analysis\n'
                                                    f'{data_sets_verbosity[num]}, {rhcf_status},'
                                                    f'continuous threshold {threshold}, {subs} subgroup(s), {scaler} '
                                                    f'scaled, p={perplexity}', seed=seed, perplexity=perplexity)
                                    plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_'
                                                f'tSNE_perplexity_{perplexity}_{subs}_{rhcf_status}_cont_thresh_'
                                                f'{str(threshold).replace(".", "-")}', bbox_inches='tight')
                                    plt.close()
                            if analysis == 'UMAP':
                                if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is False:
                                    os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                for neighbors in n_neighbors:
                                    plot_UMAP(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                              y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                              label=output_feature,
                                              title=f'BASE-II UMAP analysis\n'
                                                    f'{data_sets_verbosity[num]}, {rhcf_status},'
                                                    f'continuous threshold {threshold}, {subs} subgroup(s), {scaler} '
                                                    f'scaled, n={neighbors}', seed=seed, n_neighbors=neighbors)
                                    plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_'
                                                f'UMAP_neighbors_{neighbors}_{subs}_{rhcf_status}_cont_thresh_'
                                                f'{str(threshold).replace(".", "-")}', bbox_inches='tight')
                                    plt.close()
                        print(f'Cluster analysis completed in {data_sets_verbosity[num]}, using {subs} subgroup(s), '
                              f'with continuous near-constant threshold {threshold}, {scaler} scaler, '
                              f'and {rhcf_status}.\n')
                # else if RHCF is enabled, go through threshold possibilities first
                if rhcf_status == 'after RHCF':
                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}') is False:
                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}')
                    # Begin of the RHCF application
                    print(f"Remove highly correlated features (RHCF) with the following associations and thresholds:")
                    for rhcf_thresh in rhcf_thresholds:
                        print(
                            f"Continuous-continuous with Spearman's Rank Order Correlation {rhcf_thresh} threshold.\n")
                        # Get the initial continuous indices
                        continuous_idx, _ = get_cat_and_cont(data_sg)
                        spearman_res, cont_to_drop, spearman_set = \
                            applied_cont_rhcf(training_features=data_sg, features_list=new_feature_list_sg,
                                              continuous=continuous_idx, spearman_threshold=rhcf_thresh,
                                              directory=curr_dir,
                                              folder=folder_name + '/' + rhcf_status.replace(" ", "") + '/',
                                              datatype=f'{subs}_{str(threshold).replace(".", "-")}_{rhcf_thresh[1]}')
                        # Heatmap of the spearman matrix (saving process inside function)
                        spearman_heatmap(spearman_res, rhcf_thresh,
                                         f'{subs}_{str(threshold).replace(".", "-")}_{rhcf_thresh[1]}',
                                         continuous_idx, folder_name + '/' + rhcf_status.replace(" ", "") + '/',
                                         tiff_figure_dpi)
                        # Drop and update correlated data
                        data_rhcf = np.delete(data_sg, cont_to_drop, axis=1)
                        new_feature_list_rhcf = \
                            [new_feature_list_sg[x] for x in range(len(new_feature_list_sg)) if x not in cont_to_drop]
                        print(f'The following features were dropped for {data_sets_verbosity[num]}:\n'
                              f'{spearman_set}\n')
                        # Apply standard, minmax, or robust scaler
                        for scaler in scalers:
                            data_scaled = None
                            if scaler == 'standard':
                                data_scaled = StandardScaler().fit_transform(data_rhcf)
                                print('Processing standard scaling.\n')
                            if scaler == 'robust':
                                data_scaled = RobustScaler().fit_transform(data_rhcf)
                                print('Processing robust scaling.\n')
                            if scaler == 'minmax':
                                data_scaled = MinMaxScaler().fit_transform(data_rhcf)
                                print('Processing minmax scaling.\n')
                            # Start analysis plotting here
                            for analysis in cluster_test:
                                ################
                                # ## Clustering
                                ################
                                print(f"Starting {analysis} cluster analysis for {data_sets_verbosity[num]}.\n")
                                if analysis == 'PCA':
                                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is \
                                            False:
                                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                    plot_PCA(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                             y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                             label=output_feature, seed=seed,
                                             title=f'BASE-II PCA analysis\n'
                                                   f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                                   f'{rhcf_thresh}, continuous threshold {threshold}, {subs} '
                                                   f'subgroup(s), {scaler} scaled')
                                    plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_'
                                                f'PCA_{subs}_{rhcf_status}_{rhcf_thresh[1]}_cont_thresh_'
                                                f'{str(threshold).replace(".", "-")}', bbox_inches='tight')
                                    plt.close()
                                if analysis == 'LDA':
                                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is \
                                            False:
                                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                    plot_LDA(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                             y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                             label=output_feature,
                                             title=f'BASE-II LDA analysis\n'
                                                   f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                                   f'{rhcf_thresh}, '
                                                   f'continuous threshold {threshold}, {subs} subgroup(s), {scaler} '
                                                   f'scaled', lda_output_target=output_feature, seed=seed)
                                    plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/{scaler}_'
                                                f'LDA_{subs}_{rhcf_status}_{rhcf_thresh[1]}_cont_thresh_'
                                                f'{str(threshold).replace(".", "-")}', bbox_inches='tight')
                                    plt.close()
                                if analysis == 't-SNE':
                                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is \
                                            False:
                                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                    for perplexity in n_perplexity:
                                        plot_tSNE(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                                  y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                                  label=output_feature,
                                                  title=f'BASE-II t-SNE analysis\n'
                                                        f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                                        f'{rhcf_thresh}, continuous threshold {threshold}, '
                                                        f'{subs} subgroup(s), {scaler} scaled, p={perplexity}',
                                                  seed=seed, perplexity=perplexity)
                                        plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/'
                                                    f'{scaler}_tSNE_perplexity_{perplexity}_{subs}_{rhcf_status}_'
                                                    f'{rhcf_thresh[1]}_cont_thresh_{str(threshold).replace(".", "-")}',
                                                    bbox_inches='tight')
                                        plt.close()
                                if analysis == 'UMAP':
                                    if os.path.isdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}') is \
                                            False:
                                        os.mkdir(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}')
                                    for neighbors in n_neighbors:
                                        plot_UMAP(X=pd.DataFrame(data_scaled, columns=new_feature_list_rhcf),
                                                  y=pd.DataFrame(target_labels[num], columns=[output_feature]),
                                                  label=output_feature,
                                                  title=f'BASE-II UMAP analysis\n'
                                                        f'{data_sets_verbosity[num]}, {rhcf_status}, '
                                                        f'{rhcf_thresh}, continuous threshold {threshold}, '
                                                        f'{subs} subgroup(s), {scaler} scaled, n={neighbors}',
                                                  seed=seed, n_neighbors=neighbors)
                                        plt.savefig(f'./{folder_name}/{rhcf_status.replace(" ", "")}/{analysis}/'
                                                    f'{scaler}_UMAP_neighbors_{neighbors}_{subs}_{rhcf_status}_'
                                                    f'{rhcf_thresh[1]}_cont_thresh_{str(threshold).replace(".", "-")}',
                                                    bbox_inches='tight')
                                        plt.close()
                            print(f'Cluster analysis completed in {data_sets_verbosity[num]}, using {subs} subgroup(s),'
                                  f'with continuous near-constancy threshold {threshold} and {rhcf_status} with '
                                  f'{rhcf_thresh} and {scaler} scaler.\n')
    print(f'Complete cluster analysis in {data_sets_verbosity[num]} done!\n')
print(f'\n***BASE-II Cluster Analysis successfully completed!***')
###############################
# ## Cluster analysis loop end
###############################


########################################################################################################################
# ## END OF CLUSTER ANALYSIS SCRIPT FOR BASE-II ########################################################################
########################################################################################################################
