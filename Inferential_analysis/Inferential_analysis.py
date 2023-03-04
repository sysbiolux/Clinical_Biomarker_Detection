########################################################################################################################
# BASE-II INFERENTIAL ANALYSIS  ########################################################################################
# USING VARIOUS STATISTICAL METHODS TO INFER FRAILTY RELATIONSHIPS IN BASE-II MIXED, MALE, AND FEMALE ##################
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# March 2023, University of Luxembourg, v.03/04/2023 (M/d/y) ###########################################################
########################################################################################################################

# Testing for: Statistical correlation tests in mixed, male, female for frailty index as well as for the 5 frailty
# phenotypes in relation to age, WHR, height, BMI, ALM, ALM-BMI ratio, among others (could go through all continuous
# features, and select the ones with the best correlations or p values). Welch's t test because the variance are very
# likely un identical

############################################
# ## Importing libraries and util functions
############################################
# housekeeping libraries
import matplotlib
import os
import random
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# util functions
sys.path.append('../')
# from source.CBDP_utils import separate_full_data, get_cat_and_cont
from machine_learning.Remastered_pipeline_with_utils_and_config_210222.CBDP_utils import separate_full_data, \
    get_cat_and_cont


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
train_path = curr_dir + '/data/train_imputed.csv'
test_path = curr_dir + '/data/test_imputed.csv'

# target feature and target related features
output_feature = 'PM-Frailty_Index'
positive_class = 'frail'
negative_class = 'non-frail'
split_feature = 'PM-sex'
statistical_targets = ['PM-Frailty_Index', 'PM-Frailty_gait', 'SV-Frailty_exhaustion', 'SV-Frailty_physicalactivity',
                       'PM-Frailty_gripstrength', 'PM-Frailty_weightloss']
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

# create frailty dataframe of mixed, mail and female for all statistical targets and combine them at the end
frailty_feature_list = \
    [feature_list[k] for k in [feature_list.index(k) for k in feature_list if k in statistical_targets]]
train_frailty_mixed = \
    train_features[:, [feature_list.index(k) for k in feature_list if k in statistical_targets]]
train_frailty_male = \
    train_men_features[:, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]
train_frailty_female = train_female_features[
                       :, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]
test_frailty_mixed = test_features[:, [feature_list.index(k) for k in feature_list if k in statistical_targets]]
test_frailty_male = \
    test_men_features[:, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]
test_frailty_female = test_female_features[
                      :, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]
comp_frailty_mixed = np.concatenate((train_features, test_features))[
                     :, [feature_list.index(k) for k in feature_list if k in statistical_targets]]
comp_frailty_male = np.concatenate((train_men_features, test_men_features))[
                    :, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]
comp_frailty_female = np.concatenate((train_female_features, test_female_features))[
                      :, [feature_list_wo_gender.index(k) for k in feature_list_wo_gender if k in statistical_targets]]

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

# Combine feature list and data sets of continuous features with the frailty data sets and feature list
feature_list = feature_list + frailty_feature_list
feature_list_male = feature_list_male + frailty_feature_list
feature_list_female = feature_list_female + frailty_feature_list
train_features = np.concatenate((train_features, train_frailty_mixed), axis=1)
test_features = np.concatenate((test_features, test_frailty_mixed), axis=1)
comp_features = np.concatenate((comp_features, comp_frailty_mixed), axis=1)
train_men_features = np.concatenate((train_men_features, train_frailty_male), axis=1)
test_men_features = np.concatenate((test_men_features, test_frailty_male), axis=1)
comp_features_male = np.concatenate((comp_features_male, comp_frailty_male), axis=1)
train_female_features = np.concatenate((train_female_features, train_frailty_female), axis=1)
test_female_features = np.concatenate((test_female_features, test_frailty_female), axis=1)
comp_features_female = np.concatenate((comp_features_female, comp_frailty_female), axis=1)

# Testing for: Statistical correlation tests in mixed, male, female for frailty index as well as for the 5 frailty
# phenotypes in relation to age, WHR, height, BMI, ALM, ALM-BMI ratio, among others (could go through all continuous
# features, and select the ones with the best correlations or p values).
data_sets = [# train_features, test_features,
             comp_features,
             # train_men_features, test_men_features,
             comp_features_male,
             # train_female_features, test_female_features,
             comp_features_female]
data_sets_verbosity = [# 'mixed training', 'mixed test',
                       'mixed complete',
                       # 'male training', 'male test',
                       'male complete',
                       # 'female training', 'female test',
                       'female complete']
target_labels = [# train_labels, test_labels,
                 np.concatenate((train_labels, test_labels)),
                 # train_men_labels, test_men_labels,
                 np.concatenate((train_men_labels, test_men_labels)),
                 # train_female_labels, test_female_labels,
                 np.concatenate((train_female_labels, test_female_labels))]
feature_lists = [# feature_list, feature_list,
                 feature_list,
                 # feature_list_male, feature_list_male,
                 feature_list_male,
                 # feature_list_female, feature_list_female,
                 feature_list_female]
features_of_interest = ['PM-age_Charite', 'PM-WHR', 'PM-height', 'PM-BMI', 'PM-ALM', 'PM-ALM_BMI']
y_label_boxes = ['Age [years]', 'Waist-hip ratio', 'Height [cm]', 'Body Mass Index [kg/m\N{SUPERSCRIPT TWO}]',
                 'Appendicular Lean Mass [kg]', 'BMI-adjusted ALM']
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"

#########################################
# Starting loop for inferential analysis
#########################################
for num, data in enumerate(data_sets):
    print(f'Processing inferential analysis in {data_sets_verbosity[num]}...\n')
    folder_name = data_sets_verbosity[num].replace(' ', '_') + '_inferential'
    correl_dict_per_data = dict.fromkeys(['correlations'], list())
    p_val_dict_per_data = dict.fromkeys(['p-values'], list())
    if os.path.isdir(curr_dir + '/' + folder_name) is False:
        os.mkdir(curr_dir + '/' + folder_name)
    # starting to use a two group t-test (Note that the t-test is still appropriate in this case because we are
    # comparing the means of two independent groups, even though one of the groups is binary) Welch's t test because
    # the variance are very likely un identical
    for num_int, interest in enumerate(features_of_interest):
        print(f'Analysing relationship between {interest} and ...')
        data_int = pd.DataFrame(data, columns=feature_lists[num])
        continuous_feature = data_int[interest]
        for target in statistical_targets:
            print(f'... {target}.')
            if os.path.isdir(f'./{folder_name}/{target}') is False:
                os.mkdir(f'./{folder_name}/{target}')
            if target != 'PM-Frailty_Index':
                data_tar = pd.DataFrame(data, columns=feature_lists[num])
                binary_outcome = data_tar[target]
            else:
                binary_outcome = pd.DataFrame(target_labels[num], columns=[target])
            # split the continuous feature into two groups based on the binary outcome
            unaffected = continuous_feature[binary_outcome[target] == 0] if target == 'PM-Frailty_Index' else \
                continuous_feature[binary_outcome == 0]
            affected = continuous_feature[binary_outcome[target] == 1] if target == 'PM-Frailty_Index' else \
                continuous_feature[binary_outcome == 1]
            # perform a two-sample t-test
            t_statistic, p_value = stats.ttest_ind(unaffected, affected, equal_var=False)
            correl_dict_per_data['correlations'].append({target: (interest, t_statistic)})
            p_val_dict_per_data['p-values'].append({target: (interest, p_value)})
            print('t-statistic:', t_statistic)
            print('p-value:', p_value, '\n')
            if p_value < 0.05:
                print(f'Plotting box plot of {interest} split by {target} condition.\n')
                # create a box plot of the continuous feature split by the binary outcome if pval < 0.05
                plt.figure(figsize=(8, 7))
                box = plt.boxplot([unaffected, affected], notch=True, patch_artist=True)
                colors = [['#D8D3D3', '#a9a9a9'] if data_sets_verbosity[num] == 'mixed complete' else
                          ['#FAEEAF', '#ffd500'] if data_sets_verbosity[num] == 'male complete' else
                          ['#92B1D5', '#005bbb']]
                for patch, color in zip(box['boxes'], colors[0]):
                    patch.set_facecolor(color)
                plt.title(f'{data_sets_verbosity[num].capitalize()}: {interest} split by conditions in {target}\n'
                          f'p-value: {p_value:e}, t-statistics: {round(t_statistic, 3)}',
                          fontsize=12, fontweight='bold')
                plt.xticks([1, 2], ['Not Affected', 'Affected'], fontsize=9, fontweight='bold')
                plt.ylabel(y_label_boxes[num_int], fontsize=9, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'./{folder_name}/{target}/{interest}.png')
                plt.close()
    # create a bar plot of p-values for each target in each data group, which we find within the dictionaries
    for tar in statistical_targets:
        tar_pval_list = []
        for feat in features_of_interest:
            for item in p_val_dict_per_data['p-values']:
                if tar in item:
                    if feat in item[tar]:
                        tar_pval_list.append(item[tar][1])
        # plot for each target feature
        plt.figure(figsize=(8, 7))
        colors = ['#a9a9a9' if data_sets_verbosity[num] == 'mixed complete' else
                  '#ffd500' if data_sets_verbosity[num] == 'male complete' else
                  '#005bbb']
        plt.bar([features_of_interest[x] for x in np.argsort(-np.log10(tar_pval_list))[::-1]],
                sorted(-np.log10(tar_pval_list), reverse=True), color=colors)
        plt.hlines(-np.log10(0.05), xmin=-0.5, xmax=len(features_of_interest)-0.5, color='black',
                   linestyles='--', linewidth=1.5, label='sign. threshold')
        plt.title(f'{data_sets_verbosity[num].capitalize()}: Bar plot of log base 10 transformed p-values\n'
                  f'between {tar} conditions', fontsize=12, fontweight='bold')
        plt.xlabel(None)
        plt.ylabel('-log10(p-value)', fontsize=9, fontweight='bold')
        plt.xticks(fontsize=9, fontweight='bold')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'./{folder_name}/{tar}/bar_plot_of_p_values.png')
        plt.close()
    print(f'Processing inferential analysis ended in {data_sets_verbosity[num]}.\n')
print('Inferential analysis in BASE-II completed!')
####################################
# Loop for inferential analysis end
####################################


########################################################################################################################
# ## END OF CLUSTER ANALYSIS SCRIPT FOR BASE-II ########################################################################
########################################################################################################################
