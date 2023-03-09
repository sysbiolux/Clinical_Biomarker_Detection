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
from matplotlib.patches import Patch
import scipy.stats as stats

# util functions
sys.path.append('../')
from source.CBDP_utils import separate_full_data, get_cat_and_cont


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
data_sets = [train_features, test_features,
             comp_features,
             train_men_features, test_men_features,
             comp_features_male,
             train_female_features, test_female_features,
             comp_features_female]
data_sets_verbosity = ['mixed training', 'mixed test',
                       'mixed complete',
                       'male training', 'male test',
                       'male complete',
                       'female training', 'female test',
                       'female complete']
target_labels = [train_labels, test_labels,
                 np.concatenate((train_labels, test_labels)),
                 train_men_labels, test_men_labels,
                 np.concatenate((train_men_labels, test_men_labels)),
                 train_female_labels, test_female_labels,
                 np.concatenate((train_female_labels, test_female_labels))]
feature_lists = [feature_list, feature_list,
                 feature_list,
                 feature_list_male, feature_list_male,
                 feature_list_male,
                 feature_list_female, feature_list_female,
                 feature_list_female]
features_of_interest = ['PM-age_Charite', 'PM-WHR', 'PM-height', 'PM-BMI', 'PM-ALM', 'PM-ALM_BMI']
y_label_boxes = ['Age [years]', 'Waist-hip ratio', 'Height [cm]', 'Body Mass Index [kg/m\N{SUPERSCRIPT TWO}]',
                 'Appendicular Lean Mass [kg]', 'BMI-adjusted ALM']

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"

#########################################
# Starting loop for inferential analysis
#########################################
dict_to_capture_all_corr = dict.fromkeys(data_sets_verbosity, [])
dict_to_capture_all_pval = dict.fromkeys(data_sets_verbosity, [])
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
    # update the capturing dict
    dict_to_capture_all_corr[data_sets_verbosity[num]].append({data_sets_verbosity[num]: correl_dict_per_data})
    dict_to_capture_all_pval[data_sets_verbosity[num]].append({data_sets_verbosity[num]: p_val_dict_per_data})
    print(f'Processing inferential analysis ended in {data_sets_verbosity[num]}.\n')
print('Inferential analysis in BASE-II completed!')
####################################
# Loop for inferential analysis end
####################################


##################################################
# Combining the interesting ones after inspection
##################################################
# create a bar plot of p-values for each target in combined figure, sorted by the overall stronges signal
# Dictionary captured each run individually for each key, so it is enough to look into one single key to get all data

mixed_frailty = []
male_frailty = []
female_frailty = []
lists = [mixed_frailty, male_frailty, female_frailty]
for pos in range(len(dict_to_capture_all_pval['mixed complete'])):
    for key, item in dict_to_capture_all_pval['mixed complete'][pos].items():
        for tar in statistical_targets:
            for feat in features_of_interest:
                for it in item['p-values']:
                    if tar in it and tar == 'PM-Frailty_Index':
                        if feat in it[tar]:
                            lists[pos].append(it[tar][1])
# plot for frailty index only feature
colors = dict({0: '#a9a9a9',
              1: '#ffd500',
              2: '#005bbb'})
bar_height = 0.25
# Set position of bar on X axis
ref = np.arange(len(features_of_interest))
r1 = [x - 0.25 for x in ref]
r2 = [x + 0.25 for x in ref]
# get order (say, order of highest in all)
all = []
for data in lists:
    tmp = -np.log10(data)
    all.append(tmp)
highest_per_feature = np.zeros(len(features_of_interest))
final_rank = np.zeros(len(features_of_interest))
for case in range(np.array(all).shape[0]):
    for fe in range(len(features_of_interest)):
        if all[case][fe] >= highest_per_feature[fe]:
            highest_per_feature[fe] = all[case][fe]
final_rank = np.argsort(highest_per_feature)[::-1]
# plot
plt.subplots(figsize=(12, 4))
ax1 = plt.subplot()
ax1.bar(r1,
        -np.log10(lists[0])[final_rank],
        bar_height, label='Mixed',
        color=colors[0],
        edgecolor='black')
ax1.bar(ref,
        -np.log10(male_frailty)[final_rank],
        bar_height, label='Male',
        color=colors[1],
        edgecolor='black')
ax1.bar(r2,
        -np.log10(female_frailty)[final_rank],
        bar_height, label='Female',
        color=colors[2],
        edgecolor='black')
ax1.hlines(-np.log10(0.05), xmin=-0.5, xmax=len(features_of_interest) - 0.5, color='black',
           linestyles='--', linewidth=1.5, label='sign. threshold')
ax1.set_xticks(ref)
ax1.set_xticklabels(labels=np.array(features_of_interest)[final_rank], fontsize=12, fontweight='bold')
ax1.set_ylabel(ylabel='-log10(p-value)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10, fontweight='bold', rotation=15 if len(features_of_interest) > 6 else 0)
plt.title("Welch's unequal variance T-test of hallmark risk factors between non-frail and frail patients",
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(f'./combined_bar_plot_of_p_values.png')
plt.close()

# after inspection , we want to show some that are common and unique in male and female
# sorted by p value: PM-ALM_BMI, PM-age_Charite, PM-ALM, PM-WHR
features_to_boxplot_in_order = ['PM-ALM_BMI', 'PM-age_Charite', 'PM-ALM', 'PM-WHR']
y_label_boxplot = ['BMI-adjusted ALM', 'Age [years]', 'Appendicular Lean Mass [kg]', 'Waist-hip ratio']
# get that data
data_sets_boxplot = [comp_features,
                     comp_features_male,
                     comp_features_female]

mixed_frailty_bp = []
male_frailty_bp = []
female_frailty_bp = []
lists_bp = [mixed_frailty_bp, male_frailty_bp, female_frailty_bp]
for num, data in enumerate(data_sets_boxplot):
    for num_int, interest in enumerate(features_to_boxplot_in_order):
        data_int = pd.DataFrame(data, columns=feature_lists[num])
        continuous_feature = data_int[interest]
        for target in statistical_targets:
            if target == 'PM-Frailty_Index':
                binary_outcome = pd.DataFrame(target_labels[num], columns=[target])
                # split the continuous feature into two groups based on the binary outcome
                unaffected = continuous_feature[binary_outcome[target] == 0]
                affected = continuous_feature[binary_outcome[target] == 1]
                lists_bp[num].append([unaffected, affected])
# create combined box plots
colors = ['#a9a9a9', '#ffd500', '#005bbb']
fig, axes = plt.subplots(1, len(features_to_boxplot_in_order), figsize=(16, 6))
for num, ax in enumerate(axes):
    box1 = ax.boxplot(lists_bp[0][num][0], positions=[0], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    box2 = ax.boxplot(lists_bp[1][num][0], positions=[1.25], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    box3 = ax.boxplot(lists_bp[2][num][0], positions=[2.5], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    box4 = ax.boxplot(lists_bp[0][num][1], positions=[0.5], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    box5 = ax.boxplot(lists_bp[1][num][1], positions=[1.75], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    box6 = ax.boxplot(lists_bp[2][num][1], positions=[3.0], widths=0.4, notch=True, patch_artist=True, showfliers=False)
    for (patch1, patch2, patch3, patch4, patch5, patch6) in \
            zip(box1['boxes'], box2['boxes'], box3['boxes'], box4['boxes'], box5['boxes'], box6['boxes']):
        # patch 1-3 unaffected, patch 4-6 affected
        patch1.set_facecolor(colors[0])
        patch2.set_facecolor(colors[1])
        patch3.set_facecolor(colors[2])
        patch4.set_facecolor(colors[0])
        patch4.set_hatch('++++')
        patch5.set_facecolor(colors[1])
        patch5.set_hatch('++++')
        patch6.set_facecolor(colors[2])
        patch6.set_hatch('++++')
    ax.set_xticks([])
    if 'PM-ALM' in features_to_boxplot_in_order:
        if num == features_to_boxplot_in_order.index('PM-ALM'):
            ticks_transformed = ax.get_yticks() / 1000
            ax.set_yticklabels([int(x) for x in ticks_transformed])
    ax.set_ylabel(ylabel=y_label_boxplot[num], fontsize=12, fontweight='bold')
    ax.set_title(f'{features_to_boxplot_in_order[num]}', fontsize=14, fontweight='bold')
# generate a legend based on legend from any of the above acxes
lgd = plt.legend(loc='lower right', bbox_to_anchor=(1, -0.15), fontsize=12)
handles, labs = lgd.axes.get_legend_handles_labels()
handles.append(Patch(facecolor='white', edgecolor='black', hatch='++++'))
labs.append('Frail')
handles.append(Patch(facecolor='white', edgecolor='black'))
labs.append('Non-frail')
lgd.set_title('Conditions')
lgd._legend_box = None
lgd._init_legend_box(handles, labs)
lgd._set_loc(lgd._loc)
lgd.set_title(lgd.get_title().get_text())
for text in lgd.get_texts():
    text.set_weight('bold')
fig.tight_layout()
plt.savefig(f'./combined_box_plot_of_significant_features.png')
plt.close()
######################################################
# END Combining the interesting ones after inspection
######################################################


########################################################################################################################
# ## END OF CLUSTER ANALYSIS SCRIPT FOR BASE-II ########################################################################
########################################################################################################################
