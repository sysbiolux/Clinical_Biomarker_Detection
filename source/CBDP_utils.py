########################################################################################################################
# SCRIPT WITH ALL UTIL FUNCTIONS FOR CLINICAL DATA ANALYSIS ############################################################
# Jeff DIDIER - Faculty of Science, Technology and Medicine (FSTM), Department of Life Sciences and Medicine (DLSM) ####
# November 2021 - October 2022, University of Luxembourg, v.10/13/2022 (M/d/y) #########################################
########################################################################################################################

# Script of util functions for the classification and evaluation of predictive machine learning models to detect
# biomarkers in clinical cohort data. Functions are constraint to mixed data structure and binary classification.


########################################################################################################################
# ## PART 0: SCRIPT START IMPORTING LIBRARIES ##########################################################################
########################################################################################################################
#########################
# ## Importing libraries
#########################
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import scipy.stats as ss
from joblib import parallel_backend, Parallel, delayed
from matplotlib_venn import venn3, venn3_unweighted
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold


########################################################################################################################
# ## PART 1: CUSTOM FUNCTIONS USED IN THIS ANALYSIS ####################################################################
########################################################################################################################
####################################
# ## Features and labels separation
####################################


def separate_full_data(full_train, full_test, target_feature, splitting_feature=False):
    """
    Function to split the training and test data sets based on a binary feature and returns split data, labels, and
    feature lists.

    Parameters
    ----------
    full_train : pandas.DataFrame
        the full training data set
    full_test : pandas.DataFrame
        The full test data set
    target_feature : str
        the target feature (binary) based on which the data should be split
    splitting_feature : bool
        whether or not the full data should be split (default False), if false, only the full data labels and feature
        list will be returned

    Returns
    -------
    training_feat : np.array
        train set features of the full data set
    testing_feat : np.array
        test set features of the full data set
    training_labels : np.array
        train set labels of the full data set
    testing_labels : np.array
        test set labels of the full data set
    feature_names : list
        list of feature names of the full data set
    training_feat_class1 : np.array
        train set features of the first class of split features
    testing_feat_class1 : np.array
        test set features of the first class of split features
    training_labels_class1 : np.array
        train set labels of the first class of split features
    testing_labels_class1 : np.array
        test set labels of the first class of split features
    training_feat_class2 : np.array
        train set features of the second class of split features
    testing_feat_class2 : np.array
        test set features of the second class of split features
    training_labels_class2 : np.array
        train set labels of the second class of split features
    testing_labels_class2 : np.array
        test set labels of the second class of split features
    feature_names_classes : list
        list of feature names of the first and second classes
    """
    # Training and test features and labels of the full data are returned if no splitting feature is given
    training_feat = full_train.drop(target_feature, axis=1)
    testing_feat = full_test.drop(target_feature, axis=1)
    training_labels = np.array(full_train[target_feature])
    testing_labels = np.array(full_test[target_feature])
    feature_names = list(training_feat.columns)
    if not splitting_feature:
        return np.array(training_feat), np.array(testing_feat), training_labels, testing_labels, feature_names
    else:
        labels = np.unique(full_train[splitting_feature])
        # First binary class data
        class1_train = full_train[full_train[splitting_feature] == labels[0]]
        class1_test = full_test[full_test[splitting_feature] == labels[0]]
        class1_train_feat = class1_train.drop([target_feature, splitting_feature], axis=1)
        class1_test_feat = class1_test.drop([target_feature, splitting_feature], axis=1)
        class1_train_labels = np.array(class1_train[target_feature])
        class1_test_labels = np.array(class1_test[target_feature])
        # Second binary class data
        class2_train = full_train[full_train[splitting_feature] == labels[1]]
        class2_test = full_test[full_test[splitting_feature] == labels[1]]
        class2_train_feat = class2_train.drop([target_feature, splitting_feature], axis=1)
        class2_test_feat = class2_test.drop([target_feature, splitting_feature], axis=1)
        class2_train_labels = np.array(class2_train[target_feature])
        class2_test_labels = np.array(class2_test[target_feature])
        # Feature list for the split data (is equal for both)
        feature_names_gender = list(class1_train_feat.columns)
        return np.array(training_feat), np.array(testing_feat), training_labels, testing_labels, feature_names, \
            np.array(class1_train_feat), np.array(class1_test_feat), class1_train_labels, class1_test_labels, \
            np.array(class2_train_feat), np.array(class2_test_feat), class2_train_labels, class2_test_labels, \
            feature_names_gender


###################
# ## ROC-AUC curve
###################
# Create model evaluation function
def evaluate_model(pred, prob, train_pred, train_prob, testlabels, trainlabels, fontsize, data_group):
    """
    Function to compare and evaluate machine learning models to baseline performance.
    Computes statistics and shows ROC curve.

    Parameters
    ----------
    pred : np.array
        model predictions of the test data set
    prob : np.array
        model prediction probabilities of the test data set
    train_pred : np.array
        model predictions of the training data set
    train_prob : np.array
        model prediction probabilities of the training data set
    testlabels : np.array
        labels of the test data set
    trainlabels : np.array
        labels of the training data set
    fontsize : int
        fontsize for the ROC curve figure
    data_group : str
        name of the data subgroup to be analysed.     
    """
    # Calculate and store baseline, test and train results
    baseline = {'Recall': recall_score(testlabels, [1 for _ in range(len(testlabels))]),
                'Precision': precision_score(testlabels, [1 for _ in range(len(testlabels))]),
                'ROC': 0.5}
    results = {'Recall': recall_score(testlabels, pred),
               'Precision': precision_score(testlabels, pred),
               'ROC': roc_auc_score(testlabels, prob)}
    train_results = {'Recall': recall_score(trainlabels, train_pred),
                     'Precision': precision_score(trainlabels, train_pred),
                     'ROC': roc_auc_score(trainlabels, train_prob)}
    # Print the results
    for metric in ['Recall', 'Precision', 'ROC']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} '
              f'Train: {round(train_results[metric], 2)}')
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(testlabels, [1 for _ in range(len(testlabels))])
    model_fpr, model_tpr, _ = roc_curve(testlabels, prob)
    # Plot both curves
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = fontsize
    plt.plot(base_fpr, base_tpr, 'r--', linewidth=1.5, label='baseline')
    plt.plot(model_fpr, model_tpr, 'b', label='model')
    plt.legend(fontsize=8, loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{data_group.capitalize()} ROC AUC Curve')

    
def plot_pr(prob, testlabels, fontsize, data_group):
    """
    Function to plot the precision - recall curve on validation set.

    Parameters
    ----------
    prob : np.array
        model prediction probabilities of the test data set
    testlabels : np.array
        labels of the test data set
    fontsize : int
        fontsize for the ROC curve figure
    data_group : str
        name of the data subgroup to be analysed.
    """
    precision, recall, _ = precision_recall_curve(testlabels, prob)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = fontsize
    plt.plot([0, 1], [1, 0], 'r--', linewidth=1.5, label='baseline')
    plt.plot(recall, precision, 'b', label='model')
    plt.legend(fontsize=8, loc='lower right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{data_group.capitalize()} PR AUC Curve')


#####################################
# ## ROC_AUC during cross-validation
#####################################
# function to plot ROC_AUC curves of the stratified k-fold cross-validation during GridSearchCV
def plot_roc_validation(data_group, x, y, final_model, reps=5, folds=5, ax=None, fontsize=14):
    """ performs (reps) times (folds)-fold cross-validation and displays
    the average ROC curve, its confidence interval as well as each ROC curve
    for each validation fold for an out-of-bag stacked model.
    Parameters
    ----------
    data_group : String
        name of the data subgroup to be analysed.
    x : DataFrame
        Predictors
    y : Series
        Targets
    final_model : model
        sklearn model type for stacking
    reps : int, optional
        number of shuffled repeats. The default is 5.
    folds : int, optional
        number of folds in the cross-validation. The default is 5.
    ax : axis, optional
        the axis to be plotted. The default is None.
    fontsize : int, optional
        plot text font size.

    Returns
    -------
    None.
    """
    tprs = []
    scores = []
    base_fpr = np.linspace(0, 1, 101)
    idx = np.arange(0, len(y))
    np.random.seed(42)
    ax.figure(figsize=(8, 6))
    ax.rcParams['font.size'] = fontsize
    for j in np.random.randint(0, high=10000, size=reps):
        np.random.shuffle(idx)
        kf = StratifiedKFold(folds, shuffle=True, random_state=j)
        x_shuff = x.iloc[idx, :]
        y_shuff = y.iloc[idx]
        for train, test in kf.split(x_shuff, y_shuff):
            x_test = x_shuff.iloc[test]
            y_test = y_shuff.iloc[test]
            trained_model = final_model
            y_score = trained_model.predict_proba(x_test)
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            score = roc_auc_score(y_test, y_score[:, 1])
            scores.append(score)
            ax.plot(fpr, tpr, "b", alpha=0.1, linewidth=2)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
    ax.plot(0, 0, "b", alpha=0.1, label='n times k-fold cross-validation')  # force roc curve at 0,0
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    mean_scores = np.mean(scores)
    std_scores = np.std(scores)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    ax.plot(base_fpr, mean_tprs, "b", linewidth=3, label='average ROC curve')
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3, label='confidence interval')
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label='baseline')
    ax.xlim([-0.01, 1.01])
    ax.ylim([-0.01, 1.01])
    ax.title(f"{data_group.capitalize()} ROC AUC in %d-times %d-fold CV: %.3f +- %.3f"
             % (reps, folds, float(mean_scores), float(std_scores)))
    ax.legend(fontsize=8, loc='lower right')
    return mean_scores, std_scores


# function to plot ROC_AUC curves of the stratified k-fold cross-validation during GridSearchCV
def plot_pr_validation(data_group, x, y, final_model, reps=5, folds=5, ax=None, fontsize=14):
    """ performs (reps) times (folds)-fold cross-validation and displays
    the average PR (precision-recall) curve, its confidence interval as well as each PR curve
    for each validation fold for an out-of-bag stacked model.
    Parameters
    ----------
    data_group : String
        name of the data subgroup to be analysed.
    x : DataFrame
        Predictors
    y : Series
        Targets
    final_model : model
        sklearn model type for stacking
    reps : int, optional
        number of shuffled repeats. The default is 5.
    folds : int, optional
        number of folds in the cross-validation. The default is 5.
    ax : axis, optional
        the axis to be plotted. The default is None.
    fontsize : int, optional
        plot text font size.

    Returns
    -------
    None.
    """

    y_real = []
    y_proba = []
    precision_array = []
    recall_array = np.linspace(0, 1, 100)
    idx = np.arange(0, len(y))
    np.random.seed(42)
    ax.figure(figsize=(8, 6))
    ax.rcParams['font.size'] = fontsize
    for j in np.random.randint(0, high=10000, size=reps):
        np.random.shuffle(idx)
        kf = StratifiedKFold(folds, shuffle=True, random_state=j)
        x_shuff = x.iloc[idx, :]
        y_shuff = y.iloc[idx]
        for train, test in kf.split(x_shuff, y_shuff):
            x_test = x_shuff.iloc[test]
            y_test = y_shuff.iloc[test]
            trained_model = final_model
            y_score = trained_model.predict_proba(x_test)
            prec, rec, _ = precision_recall_curve(y_test, y_score[:, 1])
            prec, rec = prec[::-1], rec[::-1]  # reverse order of results
            prec_array = np.interp(recall_array, rec, prec)
            prec_array[0] = 1.0
            precision_array.append(prec_array)
            ax.plot(rec, prec, "b", alpha=0.1, linewidth=2)
            y_real.append(y_test)
            y_proba.append(y_score[:, 1])

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ax.plot(0, 1, "b", alpha=0.1, label='n times k-fold cross-validation')  # force pr curve at 0,1
    mean_precision = np.mean(precision_array, axis=0)
    std_precision = np.std(precision_array, axis=0)
    mean_scores = np.mean(mean_precision)
    std_scores = np.mean(std_precision)
    ax.plot(recall, precision, "b", linewidth=3, label='average PR curve')
    ax.fill_between(recall_array, mean_precision - std_precision, mean_precision + std_precision,
                    color="grey", alpha=0.3, label='confidence interval')
    ax.plot([0, 1], [1, 0], "r--", linewidth=1.2, label='baseline')
    ax.xlim([-0.01, 1.01])
    ax.ylim([-0.01, 1.01])
    ax.title(f"{data_group.capitalize()} PR AUC in %d-times %d-fold CV: %.3f +- %.3f"
             % (reps, folds, float(mean_scores), float(std_scores)))
    ax.legend(fontsize=8, loc='lower right')
    return mean_scores, std_scores


######################
# ## Confusion matrix
######################
def plot_confusion_matrix(c_matrix, classes, normalize=False, title='Confusion matrix', c_map=plt.cm.Oranges):
    """
    Function to print and plot the confusion matrix. Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Parameters
    ----------
    c_matrix : np.array
        calculated confusion matrix array returned by sklearn.metrics.confusion_matrix function
    classes : list
        list of class names for the confusion matrix (order: negative class, positive class)
    normalize : bool
        normalization of the confusion matrix results in the figure (default False)
    title : str
        title of the confusion matrix plot
    c_map : matplotlib.colors.Colormap
        matplotlib color map for the confusion matrix plot
    """
    true_matrix = c_matrix
    if normalize:
        c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(np.round(c_matrix, 2))
    else:
        print('Confusion matrix, without normalization')
        print(c_matrix)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(c_matrix, interpolation='nearest', cmap=c_map)
    plt.title(title, size=24)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)
    fmt = '.2f' if normalize else 'd'
    thresh = np.nanpercentile(c_matrix, 50)
    # Labeling the plot
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, format(c_matrix[i, j] * 100, fmt) + '%' + '\n\n' +
                 format(true_matrix[i, j], 'd') if normalize else format(c_matrix[i, j], fmt), fontsize=20,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if c_matrix[i, j] >= thresh else "black")
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


###############################################################
# ## Feature update after feature selection by fitted pipeline
###############################################################
def update_features(predict_method, named_step, cat_transformer, cont_transformer,
                    features_list, cat_list, cont_list, pca_tech, cat_tech):
    """
    Function to update the feature lists according to the feature transformation step of the best prediction method for
    categorical and continuous features. The best components are selected for normal PCA and select k best and are
    returned.

    Parameters
    ----------
    predict_method : GridSearchCV
        model to use as prediction method, best estimator will be selected to update the feature lists
    named_step : str
        name of the pipeline step in the prediction method that deals with the feature transformation
    cat_transformer : str
        name of the categorical transformer of that particular pipeline step
    cont_transformer : list
        name of the continuous transformer steps of that particular pipeline step
    features_list : list
        list of feature names from which you want an updated version
    cat_list : list
        list of categorical indices in the data set
    cont_list : list
        list of continuous indices in the data set
    pca_tech : str
        the PCA technique used by the prediction pipeline (if normal_pca, continuous features can be updated, else only
        categorical features can be)
    cat_tech : str
        the select k best technique used by the prediction pipeline

    Returns
    -------
    end_feature_list : list
        new updated feature list with first the remaining continuous then the categorical features
    """
    if hasattr(predict_method, 'best_estimator_'):
        # Get feature index of k best selected features among categorical idx
        if cat_tech != '':
            feat_k_best = \
                predict_method.best_estimator_.named_steps[named_step].named_transformers_[
                    cat_transformer].get_support() if hasattr(predict_method.best_estimator_.named_steps[
                                                                            named_step].named_transformers_[
                                                                            cat_transformer], 'get_support') else []
            most_important_cat = np.where(feat_k_best == 1)[0] if len(feat_k_best) > 0 else []
        else:
            most_important_cat = []
        if pca_tech == 'normal_pca':
            # Get feature index of n components pca selected features among continuous idx
            n_pcs = predict_method.best_estimator_.named_steps[named_step].named_transformers_[
                cont_transformer[0]].named_steps[cont_transformer[1]].components_.shape[0]
            # Get the index of the most important feature on EACH component ((un)comment 3 next lines if not desired)
            most_important_con = np.array([np.abs(
                predict_method.best_estimator_.named_steps[named_step].named_transformers_[
                    cont_transformer[0]].named_steps[
                        cont_transformer[1]].components_[i]).argmax() for i in range(n_pcs)])
        elif pca_tech == 'kernel_pca':
            most_important_con = []
            print('Kernel PCA best component feature ranking can not be tracked down. '
                  'Important continuous idx set to zero.')
        else:
            most_important_con = []

        # Fetch the real feature index in the original feature_list using the categorical and continuous indices
        feat_imp_cat = [cat_list[x] for x in np.sort(most_important_cat)] if len(most_important_cat) > 0 else []
        feat_imp_con = [cont_list[x] for x in np.sort(most_important_con)]
        feat_cat_names = [features_list[x] for x in feat_imp_cat] if len(feat_imp_cat) > 0 else []
        feat_con_names = [features_list[x] for x in feat_imp_con]
        # Create the final feature list after feature transformation and combine according to the combination strategy
        end_feature_list = feat_con_names + feat_cat_names
    else:
        raise TypeError('The estimator must first be fitted to the dataset.')
    return end_feature_list


# The pca components could yield duplicate features
def inform_about_duplicates(new_features, real_bests, datatype):
    """
    Function to inform about and print any duplicates when updating the features after the feature transformation step.
    Selecting the top feature of each PCA component can lead to duplicates.

    Parameters
    ----------
    new_features : list
        list of new updated features that were selected for feature transformation
    real_bests : list
        list of indices of unique features in the initial feature list from the updated feature list
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    """
    duplicates = []
    for fx in new_features:
        if new_features.count(fx) > 1 and fx not in duplicates:
            duplicates.append(fx)
    if len(duplicates) > 0:
        print(
            f"Found {len(duplicates)} duplicated feature(s) in the {datatype} data among the updated feature list "
            f"after Feature Transformation, namely\n{duplicates}.\nThe length of the initially extracted features "
            f"is {len(new_features)}, and {len(real_bests)} without the duplicate(s).\n")
    else:
        print(f"No duplicates found among the extracted features in the {datatype} data.\n")


##################################################################
# ## Venn diagram to compare three sets of lists with annotations
##################################################################
def plot_venn(kernel, datatype, set1, set2, set3, tuple_of_names, label_fontsize, feat_info, weighted=True):
    """
    Function to plot venn diagram and compare three sets of lists with annotations as long as the number of features
    is below 40. Circles can be weighted or unweighted.

    Parameters
    ----------
    kernel : str
        name of current Support Vector Machines Classifier kernel
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    set1 : set
        first set to be compared
    set2 : set
        second set to be compared
    set3 : set
        third set to be compared
    tuple_of_names : tuple
        tuple of names to attribute to the three sets in the venn diagram
    label_fontsize : int
        fontsize of the annotation labels in the venn diagram circles
    feat_info : str
        information conveyed by the venn diagram (e.g. top important features, top correlated features, ...)
    weighted : bool
        boolean flag whether or not to use weighted venn diagram (default True)
    """
    plt.figure(figsize=(10, 10))
    if weighted:
        venn = venn3([set1, set2, set3], tuple_of_names)
    else:
        venn = venn3_unweighted([set1, set2, set3], tuple_of_names)
    plt.title(f'{datatype} {kernel} Venn diagram of {feat_info} features', fontsize=18)
    # Create dict with settings of possible overlaps between 3 sets
    settings_dict = {'111': (set1 & set2 & set3),
                     '110': (set1 & set2 - set3),
                     '011': (set2 & set3 - set1),
                     '101': (set1 & set3 - set2),
                     '100': (set1 - set2 - set3),
                     '010': (set2 - set1 - set3),
                     '001': (set3 - set1 - set2)}
    for key, value in settings_dict.items():
        if hasattr(venn.get_label_by_id(key), 'get_text'):
            number_of_features = venn.get_label_by_id(key).get_text()
            if int(number_of_features) > 40:
                venn.get_label_by_id(key).set_text(f'\n{number_of_features}')
            else:
                venn.get_label_by_id(key).set_text('\n'.join(value).__add__(f'\n{number_of_features}'))
            venn.get_label_by_id(key).set_fontsize(label_fontsize)
    plt.tight_layout()


#############################################################
# ## Scatter plot to compare the means of the three matrices
#############################################################
def array_for_legend(array, num_metrics, correlation_type):
    """
    Function to transform a informative array into a multi-line string to be used as legend inside plotted figures. It
    will be used inside the scatter comparison plot function.

    Parameters
    ----------
    array : pandas.DataFrame
        model to use as prediction method, best estimator will be selected to update the feature lists
    num_metrics : int
        number of correlation metrics to show in the legend
    correlation_type : str
        correlation method to use (e.g. pearson)

    Returns
    -------
    multi_line_string : str
        a multi-line string of the correlation array that can be plotted without being truncated
    """
    header = "\t".join(array).expandtabs()
    vals = ' '
    for n in range(num_metrics):
        if isinstance(array.values[0][n], float):
            vals += f'{array.values[0][n]} '
        else:
            vals += '['
            for m in range(len(array.values[0][n])):
                vals += f'{array.values[0][n][m]}'
                if m < len(array.values[0][n]) - 1:
                    vals += ' '
            vals += '] '
    multi_line_string = f'{header}\n{correlation_type}{vals[:-1]}'
    return multi_line_string


def scatter_comparison(kernel, datatype, mean1, mean2, mean3, new_feat_idx, metric_list):
    """
    Function to scatter plot and compare the results of three different methods that produce a calculated mean.

    Parameters
    ----------
    kernel : str
        name of current Support Vector Machines Classifier kernel
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    mean1 : np.array
        first array of means to be compared
    mean2 : np.array
        second array of means to be compared
    mean3 : np.array
        third array of means to be compared
    new_feat_idx : range
        range of feature indices (e.g. 'range(len(features))')
    metric_list : list
        list of correlation metrics to show in the plot legend
    """
    fig, axe = plt.subplots(figsize=(12, 12))
    axe.scatter(range(len(new_feat_idx)), mean1[new_feat_idx], marker='o', s=12, label='sklearn, S.')
    axe.scatter(range(len(new_feat_idx)), mean2[new_feat_idx], marker='^', s=12, label='eli5, E.')
    axe.scatter(range(len(new_feat_idx)), mean3[new_feat_idx], marker='+', s=12, label='mlxtend, M.')
    axe.legend(loc='best')
    corr1 = pg.corr(mean1[new_feat_idx], mean2[new_feat_idx]).round(3)[metric_list]
    corr2 = pg.corr(mean1[new_feat_idx], mean3[new_feat_idx]).round(3)[metric_list]
    corr3 = pg.corr(mean2[new_feat_idx], mean3[new_feat_idx]).round(3)[metric_list]
    text1 = f'SvsE\t\t{array_for_legend(corr1, 3, "pearson")}\n'.expandtabs()
    text2 = f'SvsM\t\t{array_for_legend(corr2, 3, "pearson")}\n'.expandtabs()
    text3 = f'EvsM\t\t{array_for_legend(corr3, 3, "pearson")}'.expandtabs()
    text = '\n' + text1 + '\n' + text2 + '\n' + text3
    plt.plot([], label=text)
    plt.legend(handlelength=0, fontsize=12, borderpad=2)
    plt.title(f'{datatype} {kernel} combined scatter plot of feature importance methods', size=18)
    plt.xlabel("Feature number", fontsize=12)
    plt.ylabel("Feature importance mean", fontsize=12)
    plt.tight_layout()


################################################################################################
# ## Scatter plot to investigate correlation between three feature importance matrices using r2
################################################################################################
def scatter_r_squared(kernel, datatype, mean1, mean2, mean3, tuple_of_names, new_feat_idx, fontsize):
    """
    Function to scatter plot and compare one method against the other and show the linear equation and r squared
    coefficient.

    Parameters
    ----------
    kernel : str
        name of current Support Vector Machines Classifier kernel
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    mean1 : np.array
        first array of means to be compared
    mean2 : np.array
        second array of means to be compared
    mean3 : np.array
        third array of means to be compared
    tuple_of_names : tuple
        tuple of names to attribute to the three methods
    new_feat_idx : range
        list of feature indices (e.g. 'range(len(features))')
    fontsize : int
        fontsize of the legend to be plotted
    """
    figs, axes = plt.subplots(1, 3, sharey='row', figsize=(16, 9))
    dicts = {f'{tuple_of_names[0]}': [mean1[new_feat_idx], mean2[new_feat_idx]],
             f'{tuple_of_names[1]}': [mean1[new_feat_idx], mean3[new_feat_idx]],
             f'{tuple_of_names[2]}': [mean2[new_feat_idx], mean3[new_feat_idx]]}
    figs.suptitle(f'{datatype} {kernel} pairwise feature importance scatter plots', size=18)
    figs.supylabel('First method', fontsize=12)
    figs.supxlabel('Second method', fontsize=12)
    # Define the axis content and plot
    for axx, _ in zip(axes, range(len(dicts))):
        [m, b] = np.polyfit(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1], 1)
        axx.scatter(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1], s=8, label=f'{tuple_of_names[_]}')
        axx.plot(dicts[tuple_of_names[_]][0], m * dicts[tuple_of_names[_]][0] + b,
                 linestyle=':', linewidth=1, color='m')
        # Define axis text
        text = r"$r^2$ = " \
               + f'{round(pg.corr(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1])["r"].values[0] ** 2, 4)}' \
               + '\nequation = ' + f'{round(m, 4)}*x + {round(b, 4)}'
        axx.plot([], label=text)
        axx.legend(handlelength=0, loc='upper left', fontsize=fontsize)
    figs.tight_layout()


########################################
# ## Violin plot for feature importance
########################################
def adjacent_values(vals, q1, q3):
    """
    Function to calculate the lower and upper adjacent values based on 1st and 3rd inter-quartile range. Upper adjacent
    value is 3rd quartile plus 1.5 times inter-quartile range. Lower adjacent value is 1st quartile minus 1.5 times
    inter-quartile range.

    Parameters
    ----------
    vals : np.array
        array of the values to be treated
    q1 : np.array
        the value of the first quartile in the above array
    q3 : np.array
        the value of the third quartile in the above array

    Returns
    -------
    lower_adjacent_value : np.array
        lower adjacent value as the first quartile minus 1.5 times inter-quartile range
    upper_adjacent_value : np.array
        upper adjacent value as the third quartile plus 1.5 times inter-quartile range
    """
    # Calculate upper and lower adjacent values based on 1st and 3rd inter quartile ranges
    # Upper is 3rd + 1.5 * inter quartile range
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    # Lower is 1st - 1.5 * inter quartile range
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


# Truncate feature name if too long
def trunc_feature(feature=None, stop=None, solve_duplicates=False):
    """
    Function to truncate the name of long features, especially in the violin plot.

    Parameters
    ----------
    feature : str
        name of the real feature (default None)
    stop : int
        number of feature letters to print before adding ellipsis (count of ellipsis points included, default None)
    solve_duplicates : bool
        whether or not the last two letters should be added after ellipsis to avoid duplicates in feature lists or
        dictionaries

    Returns
    -------
    feature : str
        truncated feature name with ellipsis at the end
    """
    if feature != feature[0:stop]:
        if solve_duplicates and len(feature):
            return feature[0:stop-3].__add__('...').__add__(feature[-2:])
        else:
            return feature[0:stop-3].__add__('...')
    else:
        return feature


# Function for customized violin plot, takes the top sorted feature importance matrix and features
def plot_violin(kernel, datatype, top_all, top_feat, fontsize):
    """
    Function to generate violin plot of the score differences after shuffling each feature and taking the score 
     difference between baseline score (best score on data) and new score achieved by best prediction method on the 
     data with shuffled feature.

    Parameters
    ----------
    kernel : str
        name of current Support Vector Machines Classifier kernel
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    top_all : np.array
        array of all obtained score differences for each feature and times shuffled
    top_feat : np.array
        array of the feature names corresponding to the top important features (=< 40)
    fontsize : int
        fontsize of the figure title
    """
    poss_cols = [i for i in range(1, len(top_feat) + 1) if len(top_feat) % i == 0]
    cols = int(np.round(np.median(poss_cols)))
    rows = int(np.ceil(len(top_feat) / cols))
    # In case of a prime number
    if len(poss_cols) == 2 and max(poss_cols) > 3:
        cols = int(np.ceil(1 / 4 * len(top_feat)))
        rows = int(np.ceil(len(top_feat) / cols))
    # In case if only one feature is showing changes (rare but can happen)
    if len(poss_cols) in np.array([1, 2]) and max(poss_cols) < 3:
        cols = 2
        rows = 2
    # Reverse the order of the vectors
    top_all = top_all[::-1]
    top_feat = top_feat[::-1]
    plt.rcParams['font.size'] = fontsize
    # Start figure
    fig, axxx = plt.subplots(nrows=rows, ncols=cols, figsize=(9, 7), sharey='row', sharex='col')
    for k in range(rows):
        for p in range(cols):
            if (p + k * cols) < len(top_feat):
                axxx[k, p].set_title(f'{trunc_feature(top_feat[p + k * cols], 10, False)}', fontsize=fontsize)
                parts = axxx[k, p].violinplot(top_all[p + k * cols, :], showmeans=False, showmedians=False,
                                              showextrema=False)
                # Set parts properties
                for pc in parts['bodies']:
                    pc.set_facecolor('#D43F3A')
                    pc.set_edgecolor('black')
                    pc.set_alpha(1)
                # Calculate quartiles, median, whiskers and mean for each feature
                quartile1, medians, quartile3 = np.percentile(top_all[p + k * cols, :], [25, 50, 75])
                whiskers = np.array([adjacent_values(top_all[p + k * cols, :], quartile1, quartile3)])
                whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
                mean = np.mean(top_all[p + k * cols, :])
                # Plot the statistical properties of the feature
                axxx[k, p].scatter(1, medians, marker='o', color='white', s=20, zorder=4)
                axxx[k, p].vlines(1, quartile1, quartile3, color='k', linestyle='-', lw=3, zorder=3)
                axxx[k, p].vlines(1, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1, zorder=2)
                axxx[k, p].hlines(mean, 0.9, 1.1, color='b', linestyle='-', lw=1, zorder=1)
                axxx[k, p].set_xticks(range(0))
                axxx[k, p].axhline(0, color='k', linestyle='--', lw=1, zorder=0.9)
            else:
                axxx[k, p].set_axis_off()
    # Add title and adjust subplots
    fig.suptitle(f'{datatype} feature importance violin plot of {len(top_feat)} best features {kernel} kernel', size=18)
    plt.subplots_adjust(bottom=0.01, wspace=0.05)
    fig.supylabel('Feature importance', fontsize=fontsize + 2)
    fig.tight_layout()


###################################################
# ## Removing highly correlated features functions
###################################################
# Check for constant features in the training set before applying RHCF
def check_constant_features(features_list, training_features, datatype, nbr_splits, near_constant_thresh=0.01):
    """
    Function to collect and print the constant features detected in the training data set. Beside strictly constant
    features, the function also scans for near constant features measured by the variance-to-mean ratio of the values
    in that feature. Features which variance of unique values does not surpass the near_constant_thresh will be added
    to the list of constant feature. Specific messages will be displayed for strictly constant and near constant
    features.

    Parameters
    ----------
    features_list : np.array
        array of the all the feature names
    training_features : np.array
        array of the training features data set
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    nbr_splits : int
        number of stratified k fold splits to use as threshold for categorical features
    near_constant_thresh : float
        variance threshold for near constant continuous feature, default 0.01

    Returns
    -------
    constants : list
        list of constant features detected in the training data set
    """
    constants = []
    near_constants_cont = []
    near_constants_bool = []
    near_constants_cat = []
    cont, cat = get_cat_and_cont(training_features, None)
    for col in range(len(features_list)):
        if len(np.unique(training_features[:, col])) == 1:
            constants.append(col)
        # check for near-constant features in cont and cat separately
        elif len(np.unique(training_features[:, col])) > 1 and col in cont:
            # if variance divided by mean is below threshold: near-constant cont feature
            if abs(np.var(training_features[:, col],
                   dtype=np.float64) / np.mean(training_features[:, col],
                                               dtype=np.float64)) < near_constant_thresh:
                constants.append(col)
                near_constants_cont.append(col)
        elif len(np.unique(training_features[:, col])) == 2 and col in cat:  # only for binary factors
            # if the minority class is below the number of stratified k fold split: near-constant bool feature
            _, counts = np.unique(training_features[:, col], return_counts=True)
            if min(counts) < nbr_splits:
                constants.append(col)
                near_constants_bool.append(col)
        elif len(np.unique(training_features[:, col])) > 2 and col in cat:  # only for categorical features
            array, counts = np.unique(training_features[:, col], return_counts=True)
            tmp = 0
            for k in range(len(array)):
                if k != 0:
                    tmp += counts[k]
            if tmp < nbr_splits:
                constants.append(col)
                near_constants_cat.append(col)
                
    if len(near_constants_cont) > 0:
        print(f'Found {len(near_constants_cont)} near-constant continuous feature(s) where the variance-to-mean ratio '
              f'is below the {near_constant_thresh} threshold, thus we assume them as being constant features, '
              f'namely:\n{[features_list[var] for var in near_constants_cont]}.\n')
    if len(near_constants_bool) > 0:
        print(f'Found {len(near_constants_bool)} near-constant boolean feature(s) where the number of the minority '
              f'class is below the number of stratified k fold splits {nbr_splits} , thus we assume them as being '
              f'constant features, namely:\n{[features_list[var] for var in near_constants_bool]}.\n')
    if len(near_constants_cat) > 0:
        print(f'Found {len(near_constants_cat)} near-constant categorical feature(s) where the number of the non-zero '
              f'classes is below the number of stratified k fold splits {nbr_splits} , thus we assume them as being '
              f'constant features, namely:\n{[features_list[var] for var in near_constants_cat]}.\n')
    print(f'In total, found {len(constants)} constant and near-constant features in the {datatype} training data set:\n'
          f'{[features_list[var] for var in constants]}.\n')
    return constants


# Get the indices of continuous and categorical features
def get_cat_and_cont(training_features, testing_features=None):
    """
    Function to separately collect the indices of categorical and continuous features.

    Parameters
    ----------
    training_features : np.array
        array of the training features data set
    testing_features : np.array
        array of the test features data set

    Returns
    -------
    cont_list : list
        list of indices of continuous features
    cat_list : list
        list of indices of categorical features
    """
    cont_list = []
    cat_list = []
    if testing_features is None:
        data = training_features
    else:
        data = np.concatenate((training_features, testing_features), axis=0)
    for col in range(training_features.shape[1]):
        if any(np.array(data[:, col].round() != data[:, col])):
            cont_list.append(col)
        else:
            cat_list.append(col)
    return cont_list, cat_list


# Corrected Cramer's V correlation between categorical features
def cramers_corrected_stat(x, y):
    """
    Function to calculate corrected Cramers V statistic for categorical-categorical association. Uses correction 
    from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
    
    Parameters
    ----------
    x : np.array
        array of first vector or column to analyze Cramers V correlation with the second
    y : np.array
        array of second vector or column to analyze Cramers V correlation with the first

    Returns
    -------
    result : float
        float value of the corrected Cramers V correlation coefficient between x and y
    """
    result = -1
    if len(np.unique(x)) == 1:
        print("First variable is constant")
    elif len(np.unique(y)) == 1:
        print("Target variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)
        if conf_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True
        chi_2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]
        n = sum(conf_matrix.sum())
        phi2 = chi_2 / n
        r, k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))
    return round(result, 6)


def corr_cramer_kbest(x, y):
    """
    Function to calculate corrected Cramers V statistic for categorical-categorical association and adapted for
    the use within Sklearn SelectKBest Feature selection technique. Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328.

    Parameters
    ----------
    x : np.array
        training features data set
    y : np.array
        array of target output vector to be compared with the categorical features from X

    Returns
    -------
    cramer_list : list
        list of cramer V coefficient for each of the training features compared to the target
    cramer_pv_list : list
        list of cramer V p-values for each of the training features compared to the target
    """
    if len(np.unique(y)) == 1:
        print("Target variable is constant")

    cramer_list = []
    cramer_pv_list = []
    cat = get_cat_and_cont(x, None)[1]
    for col in cat:
        if len(np.unique(x[:, col])) == 1:
            print(f"First variable is constant, feature position: {cat[col]}")
        else:
            conf_matrix = pd.crosstab(x[:, col], y)
            if conf_matrix.shape[0] == 2:
                correct = False
            else:
                correct = True
            chi_2, p = ss.chi2_contingency(conf_matrix, correction=correct)[0:2]  # first two outputs are chi2 and p-v
            n = sum(conf_matrix.sum())
            phi2 = chi_2 / n
            r, k = conf_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            r_corr = r - ((r - 1) ** 2) / (n - 1)
            k_corr = k - ((k - 1) ** 2) / (n - 1)
            result = np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))
            cramer_list.append(result)
            cramer_pv_list.append(p)
    return cramer_list, cramer_pv_list


def applied_cramer(train_features, categorical_idx, col_idx):
    """
    Function to apply corrected Cramers V correlation on the training data set given the indices of the categorical 
    features and adapted for parallel computation.

    Parameters
    ----------
    train_features : np.array
        array of the training features data set
    categorical_idx : list
        list of categorical indices in the data set
    col_idx : int
        integer of categorical index list to be distributed to the available parallel engines

    Returns
    -------
    res : list
        list of corrected Cramers V correlation between the distributed col_idx and all other categorical index,
        the list for each categorical feature will be stacked at the end of the parallel step, resulting in a matrix of
        shape (n_categorical_features, n_categorical_features), absolute values are returned
    """
    res = []
    for var in categorical_idx:
        cramers = cramers_corrected_stat(train_features[:, col_idx], train_features[:, var])  # Cramer's V test
        res.append(round(cramers, 6))  # Keeping of the rounded value of the Cramer's V
    return np.array(res).__abs__()


def parallel_cramer(train_features, categorical_idx, n_jobs):
    """
    Parallel function to effectively distribute the tasks for calculating corrected Cramers V for each feature with all
    other categorical features.

    Parameters
    ----------
    train_features : np.array
        array of the training features data set
    categorical_idx : list
        list of categorical indices in the data set
    n_jobs : int
        number of jobs for distributed work

    Returns
    -------
    result : np.array
        stacked array of results by the parallel application of corrected Cramers V on the given data set
    """
    result = Parallel(n_jobs=n_jobs)(delayed(applied_cramer)(train_features,
                                                             categorical_idx, col_idx) for col_idx in categorical_idx)
    return np.array(result)


def cramer_threshold_to_drop(cramer_matrix, thresh, categorical, feats, directory, folder, datatype, verbosity=False):
    """
    Function to identify the categorical features in the unitriangular correlation matrix that show correlations with
    other categorical features above a given threshold.

    Parameters
    ----------
    cramer_matrix : np.array
        resulting array of the Cramers V correlation coefficients
    thresh : tuple
        defined threshold to identify correlated features, either percentile value or decimal value  e.g.
        (value, 'decimal')
    categorical : list
        list of categorical indices in the data set
    feats : np.array
        list of feature names if verbosity output is desired
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix
    verbosity : bool
        whether or not the information about who is correlated with whom should be displayed. Could be a very long list.

    Returns
    -------
    categorical_to_drop : list
        list of indices of the categorical features that are above the given correlation threshold
    """
    cramer_panda = pd.DataFrame(cramer_matrix)
    upper = cramer_panda.where(np.triu(np.ones(cramer_panda.shape), k=1).astype(bool))
    if thresh[1] == 'decimal':
        categorical_to_drop = [categorical[col] for col in upper.columns if any(np.array(upper[col] > thresh[0]))]
        message = f"surpassing the threshold of {thresh[0]}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Corrected Cramer's V Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Cramer_correlation_states.txt', 'w')
            f.write(f"Corrected Cramer's V Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(cramer_matrix)):
                tmp = np.where(np.array(upper[col]) > thresh[0])
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(categorical)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(categorical)][tmp]}\ncorrelation: '
                            f'{np.array(upper)[tmp, col][0]}\n\n')
            f.close()
            print("Writing done!")
    else:  # If thresh[1] == 'percentile'
        thresh_percentile = np.nanpercentile(upper, thresh[0])
        categorical_to_drop = \
            [categorical[col] for col in upper.columns if any(upper[col] > thresh_percentile)]
        message = f"surpassing the {thresh[0]}th percentile threshold of {round(thresh_percentile, 4)}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Corrected Cramer's V Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Cramer_correlation_states.txt', 'w')
            f.write(f"Corrected Cramer's V Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(cramer_matrix)):
                tmp = np.where(np.array(upper[col]) > thresh_percentile)
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(categorical)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(categorical)][tmp]}\ncorrelation: '
                            f'{np.array(upper)[tmp, col][0]}\n\n')
            f.close()
            print("Writing done!")
    print(f"\nCorrected Cramer's V correlation of {len(categorical)} categorical features identified "
          f"{len(categorical_to_drop)} features that are correlated, {message}.\nThose features are "
          f"indexed {categorical_to_drop} in the original feature list.")
    return categorical_to_drop


def applied_cat_rhcf(parallel_meth, training_features, features_list, categorical, n_job, cramer_threshold,
                     directory, folder, datatype):
    """
    Function to deploy the parallelized analysis of highly correlated categorical features with corrected Cramers V
    method, adapted for parallelization.

    Parameters
    ----------
    parallel_meth : str
        string referring to the parallel backend method
    training_features : np.array
        array of the training features data set
    features_list : np.array
        array of the feature names
    categorical : list
        list of categorical indices
    n_job : int
        number of jobs for distributed work
    cramer_threshold : tuple
        defined threshold to identify correlated features, either percentile value or decimal value e.g.
        (value, 'decimal')
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix

    Returns
    -------
    cat_matrix : np.array
        resulting array of the Cramers V correlation coefficients
    cat_drop : list
        list of indices of the categorical features that are above the given correlation threshold
    cat_set : set
        set of the feature names that can be dropped
    """
    with parallel_backend(parallel_meth):
        cat_matrix = parallel_cramer(train_features=training_features, categorical_idx=categorical, n_jobs=n_job)
    cat_drop = cramer_threshold_to_drop(cramer_matrix=cat_matrix, thresh=cramer_threshold, categorical=categorical,
                                        feats=features_list, verbosity=True, directory=directory, folder=folder,
                                        datatype=datatype)
    cat_set = set([features_list[idx] for idx in cat_drop])
    print(list(cat_set), '\n')
    return cat_matrix, cat_drop, cat_set


# Spearman's Rank Order correlation between continuous features
def applied_spearman(training_features, continuous):
    """
    Function to apply Spearman's Rank Order Correlation on the training data set given the indices of the continuous
    features.

    Parameters
    ----------
    training_features : np.array
        array of the training features data set
    continuous : list
        list of continuous indices in the data set

    Returns
    -------
    scc : np.array
        array of calculated Spearman's Rank Order correlation coefficient for each continuous feature with every other
        continuous feature, absolute values are returned
    """
    scc, pval = ss.spearmanr(training_features[:, continuous], axis=0)
    scc = pd.DataFrame(scc, index=pd.DataFrame(training_features[:, continuous]).T.index,
                       columns=pd.DataFrame(training_features[:, continuous]).T.index)
    return np.array(scc.abs())


def spearman_threshold_to_drop(spearman_matrix, thresh, continuous, feats, directory, folder, datatype,
                               verbosity=False):
    """
    Function to identify the continuous features in the unitriangular correlation matrix that show correlations with
    other continuous features above a given threshold.

    Parameters
    ----------
    spearman_matrix : np.array
        resulting array of the Spearman's Rank Order correlation coefficients
    thresh : tuple
        defined threshold to identify correlated features, either percentile value or decimal value e.g.
        (value, 'decimal')
    continuous : list
        list of continuous indices in the data set
    feats : np.array
        list of feature names if verbosity output is desired
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix
    verbosity : bool
        whether or not the information about who is correlated with whom should be displayed. Could be a very long list.

    Returns
    -------
    continuous_to_drop : list
        list of indices of the continuous features that are above the given correlation threshold
    """
    spearman_panda = pd.DataFrame(spearman_matrix)
    upper = spearman_panda.where(np.triu(np.ones(spearman_panda.shape), k=1).astype(bool))
    if thresh[1] == 'decimal':
        continuous_to_drop = [continuous[col] for col in upper.columns if any(np.array(upper[col] > thresh[0]))]
        message = f"surpassing the threshold of {thresh[0]}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Spearman's Rank Order Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Spearman_correlation_states.txt', 'w')
            f.write(f"Spearman's Rank Order Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(spearman_matrix)):
                tmp = np.where(np.array(upper[col]) > thresh[0])
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(continuous)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(continuous)][tmp]}\ncorrelation: '
                            f'{np.array(upper)[tmp, col][0]}\n\n')
            f.close()
            print("Writing done!")
    else:  # If thresh[1] == 'percentile'
        thresh_percentile = np.nanpercentile(upper, thresh[0])
        continuous_to_drop = [continuous[col] for col in upper.columns if any(upper[col] > thresh_percentile)]
        message = f"surpassing the {thresh[0]}th percentile threshold of {round(thresh_percentile, 4)}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Spearman's Rank Order Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Spearman_correlation_states.txt', 'w')
            f.write(f"Spearman's Rank Order Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(spearman_matrix)):
                tmp = np.where(np.array(upper[col]) > thresh_percentile)
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(continuous)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(continuous)][tmp]}\ncorrelation: '
                            f'{np.array(upper)[tmp, col][0]}\n\n')
            f.close()
            print("Writing done!")
    print(f"\nSpearman correlation of {len(continuous)} continuous features identified "
          f"{len(continuous_to_drop)} features that are correlated, {message}.\nThose features are "
          f"indexed {continuous_to_drop} in the original feature list.")
    return continuous_to_drop


def applied_cont_rhcf(training_features, features_list, continuous, spearman_threshold,
                      directory, folder, datatype):
    """
    Function to deploy the analysis of highly correlated continuous features with Spearman's Rank Order method.

    Parameters
    ----------
    training_features : np.array
        array of the training features data set
    features_list : np.array
        array of the feature names
    continuous : list
        list of continuous indices
    spearman_threshold : tuple
        defined threshold to identify correlated features, either percentile value or decimal value e.g.
        (value, 'decimal')
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix

    Returns
    -------
    cont_matrix : np.array
        resulting array of the Spearman's Rank Order correlation coefficients
    cont_drop : list
        list of indices of the continuous features that are above the given correlation threshold
    cont_set : set
        set of the feature names that can be dropped
    """
    cont_matrix = applied_spearman(training_features=training_features, continuous=continuous)
    cont_drop = spearman_threshold_to_drop(spearman_matrix=cont_matrix, thresh=spearman_threshold,
                                           continuous=continuous, feats=features_list, directory=directory,
                                           folder=folder, datatype=datatype, verbosity=True)
    cont_set = set([features_list[idx] for idx in cont_drop])
    print(list(cont_set), '\n')
    return cont_matrix, cont_drop, cont_set


# Updating data before going on with point bi-serial
def drop_and_update_correlated_data(continuous_to_drop, categorical_to_drop, training_set, test_set, features_list):
    """
    Function to update the training data set, test data set and feature names list by dropping the previously
    identified correlated categorical and continuous features.

    Parameters
    ----------
    continuous_to_drop : list
        list of identified continuous feature indices to be dropped
    categorical_to_drop : list
        list of identified categorical feature indices to be dropped
    training_set : np.array
        array of the training features data set
    test_set : np.array
        array of the test features data set
    features_list : np.array
        list of the feature names

    Returns
    -------
    cleared_train : np.array
        new training data set cleared from correlated continuous and categorical features
    cleared_test : np.array
        new test data set cleared from correlated continuous and categorical features
    remaining_feat : list
        list of the remaining feature names
    remaining_idx : list
        list of the remaining feature indices
    """
    to_drop = sorted(continuous_to_drop + categorical_to_drop)
    cleared_train = np.delete(training_set, to_drop, axis=1)
    cleared_test = np.delete(test_set, to_drop, axis=1)
    retained_features = [(features_list[p], p) for p in range(len(features_list)) if p not in to_drop]
    remaining_feat, remaining_idx = zip(*retained_features)
    print(f"Checking the correlation among the categorical features and among the continuous features lead to the "
          f"removal of {len(to_drop)} features in both the training and test set.\n{len(retained_features)} features "
          f"were retained and will be passed to the final point bi-serial correlation step.\n\nThe shape of training "
          f"set changed from {training_set.shape} to {cleared_train.shape}, and the test set from {test_set.shape} "
          f"to {cleared_test.shape}.\n")
    return cleared_train, cleared_test, remaining_feat, remaining_idx


# Point Biserial correlation threshold between filtered categorical (var1) and continuous (var2) features
def applied_point_bs(train_features, short, col_idx):
    """
    Function to apply point bi-serial correlation on the training data set given the indices of the shortest remaining
    feature type (categorical or continuous) and adapted for parallel computation.

    Parameters
    ----------
    train_features : np.array
        array of the training features data set
    short : list
        shortest list of either categorical or continuous indices in the data set
    col_idx : int
        integer of longest categorical or continuous index list to be distributed to the available parallel engines

    Returns
    -------
    res_pb : list
        list of point bi-serial correlation coefficient between the distributed col_idx of the longer feature type and
        and all other indices of the shorter feature type the list for each distributed feature will be stacked at the
        end of the parallel step, resulting in a matrix of shape (n_longest_feature_type, n_shortest_feature_type),
        absolute values are returned
    res_pv : list
        list of point bi-serial correlation p-values between the distributed col_idx of the longer feature type and
        and all other indices of the shorter feature type the list for each distributed feature will be stacked at the
        end of the parallel step, resulting in a matrix of shape (n_longest_feature_type, n_shortest_feature_type),
        absolute values are returned
    """
    res_pb = []
    res_pv = []
    for var in short:
        pbserial, pv = ss.pointbiserialr(train_features[:, col_idx], train_features[:, var])
        res_pb.append(round(pbserial, 6))  # find out why r and not p-value?
        res_pv.append(round(pv, 6))  # just in case, p-value
    return np.array(res_pb).__abs__(), np.array(res_pv)


def parallel_point_bs(train_features, longest, short, n_jobs):
    """
    Parallel function to effectively distribute the tasks for calculating point bi-serial correlation for each feature
    of one type with all other features of the other type (either categorical or continuous, longest will be
    distributed).

    Parameters
    ----------
    train_features : np.array
        array of the training features data set
    longest : list
        longest list of feature indices between categorical and continuous
    short : list
        shortest list of feature indices between categorical and continuous
    n_jobs : int
        number of jobs for distributed work

    Returns
    -------
    res_pb_r : np.array
        stacked array of the correlation coefficient results by the parallel application of point bi-serial correlation
        on the given data set
    res_pb_pv : np.array
        stacked array of the p-values results by the parallel application of point bi-serial correlation on the given
        data set
    """
    result = Parallel(n_jobs=n_jobs)(delayed(applied_point_bs)(train_features, short, col_idx) for col_idx in longest)
    res_pb_r, res_pb_pv = zip(*result)
    return np.array(res_pb_r), np.array(res_pb_pv)


def point_bs_threshold_to_drop(point_bs_matrix, thresh, longer, shorter, remaining_feature_idx, feats,
                               directory, folder, datatype, verbosity=False):
    """
    Function to identify the categorical and continuous features in the unitriangular correlation matrix that show
    correlations with the other type of features above a given threshold.

    Parameters
    ----------
    point_bs_matrix : np.array
        resulting array of the point bi-serial correlation coefficients
    thresh : tuple
        defined threshold to identify correlated features, either percentile value or decimal value e.g.
        (value, 'decimal')
    longer : list
        list of feature indices corresponding to the longer type of features (continuous or categorical)
    shorter : list
        list of feature indices corresponding to the shorter type of features (continuous or categorical)
    remaining_feature_idx : list
        list of feature indices remaining in the original data set after the highly correlated continuous and
        categorical features were removed
    feats : np.array
        list of feature names if verbosity output is desired
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix
    verbosity : bool
        whether or not the information about who is correlated with whom should be displayed. Could be a very long list.

    Returns
    -------
    real_cols_to_drop : list
        list of indices of the longer feature type (continuous or categorical) that are above the given correlation
        threshold
    """
    point_bs_panda = pd.DataFrame(point_bs_matrix)
    # Index of pbs cols to drop from original feature_list
    if thresh[1] == 'decimal':
        pbs_cols_to_drop_from_train = [longer[col] for col in point_bs_panda.T.columns if
                                       any(np.array(point_bs_panda.T[col] > thresh[0]))]
        message = f"surpassing the threshold of {thresh[0]}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Point Bi-Serial Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Point_biserial_correlation_states.txt', 'w')
            f.write(f"Point Bi-Serial Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(point_bs_matrix)):
                tmp = np.where(np.array(point_bs_matrix[col, :]) > thresh[0])
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(longer)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(shorter)][tmp]}\ncorrelation: '
                            f'{np.array(point_bs_matrix)[col, tmp][0]}\n\n')
            f.close()
            print("Writing done!")
    else:  # If thresh[1] == 'percentile'
        thresh_percentile = np.nanpercentile(point_bs_panda, thresh[0])
        pbs_cols_to_drop_from_train = [longer[col] for col in point_bs_panda.T.columns if
                                       any(point_bs_panda.T[col] > thresh_percentile)]
        message = f"surpassing the {thresh[0]}th percentile threshold of {round(thresh_percentile, 4)}"
        # print who is correlated with whom
        if verbosity:
            print("Writing the Point Bi-Serial Correlation States into a separate file...")
            f = open(directory + '/' + folder + '/' + f'{datatype}_Point_biserial_correlation_states.txt', 'w')
            f.write(f"Point Bi-Serial Correlation States for the removed highly correlated features in "
                    f"{datatype} data set\n\n\n")
            for col in range(len(point_bs_matrix)):
                tmp = np.where(np.array(point_bs_matrix[col, :]) > thresh_percentile)
                if len(tmp[0]) > 0:
                    f.write(f'* {[np.array(feats)[np.array(longer)][col]]} is highly correlated with '
                            f'{np.array(feats)[np.array(shorter)][tmp]}\ncorrelation: '
                            f'{np.array(point_bs_matrix)[col, tmp][0]}\n\n')
            f.close()
            print("Writing done!")
    # Find the real cols to drop from the retained features after updating for Cramer and spearman
    real_cols_to_drop = [col for col in range(len(remaining_feature_idx)) if
                         remaining_feature_idx[col] in pbs_cols_to_drop_from_train]
    print(f"\nPoint bi-serial correlation between the remaining continuous and categorical features identified "
          f"{len(real_cols_to_drop)} features that are correlated, {message}.\nThose features are indexed "
          f"{pbs_cols_to_drop_from_train} in the original feature list and {real_cols_to_drop} in the retained "
          f"feature list after the two previous correlation checks.")
    return real_cols_to_drop


def applied_cat_cont_rhcf(parallel_meth, training_features, cont_before_rhcf, cat_before_rhcf, features_list,
                          feat_after_rhcf, feat_idx_after_rhcf, n_job, pbs_threshold, directory, folder, datatype):
    """
    Function to deploy the analysis of highly correlated features with Point Bi-serial method, adapted for
    parallelization. The function decides which is the longest of both feature types and distributes the tasks
    along the longest list.

    Parameters
    ----------
    parallel_meth : str
        string referring to the parallel backend method
    training_features : np.array
        array of the training features data set
    cont_before_rhcf : list
        list of continuous indices in the original data set before removal of highly correlated features started
    cat_before_rhcf : list
        list of categorical indices in the original data set before removal of highly correlated features started
    features_list : np.array
        array of the feature names
    feat_after_rhcf : list
        list of feature names after highly correlated continuous and categorical features were removed
    feat_idx_after_rhcf : list
        list of feature indices after highly correlated continuous and categorical features were removed
    n_job : int
        number of jobs for distributed work
    pbs_threshold : tuple
        defined threshold to identify correlated features, either percentile value or decimal value e.g.
        (value, 'decimal')
    directory : str
        current directory
    folder : str
        destination folder of the correlation states
    datatype : str
        type of data being analysed, will be added as prefix

    Returns
    -------
    longer : list
        list of indices of the longer feature type
    pb_corr : np.array
        array of point bi-serial correlation results of shape (n_longer_feature_type, n_shorter_feature_type)
    pb_pv : np.array
        array of point bi-serial p-value results of shape (n_longer_feature_type, n_shorter_feature_type)
    pbs_drop : list
        list of feature indices to drop from the data set that are above the given correlation threshold
    cat_cont_set : set
        set of feature names to be removed
    remaining_cont : list
        list of remaining continuous features before point bi-serial application, to be used in summary function
    remaining_cat : list
        list of remaining categorical features before point bi-serial application, to be used in summary function
    """
    # Fetch the remaining continuous and categorical feature idx and find out which is the longest
    remaining_cont = [idx for idx in cont_before_rhcf if features_list[idx] in feat_after_rhcf]
    remaining_cat = [idx for idx in cat_before_rhcf if features_list[idx] in feat_after_rhcf]
    # The longest of both list will be parallelized, and is the list from which correlated features will be removed
    if len(remaining_cat) >= len(remaining_cont):
        longer = remaining_cat
        shorter = remaining_cont
    else:
        longer = remaining_cont
        shorter = remaining_cat

    # Point bi-serial correlation
    with parallel_backend(parallel_meth):
        pb_corr, pb_pv = parallel_point_bs(train_features=training_features, longest=longer, short=shorter,
                                           n_jobs=n_job)
    pbs_drop = point_bs_threshold_to_drop(pb_corr, pbs_threshold, longer, shorter,
                                          feat_idx_after_rhcf, features_list, directory, folder, datatype, True)
    cat_cont_set = set([feat_after_rhcf[idx] for idx in pbs_drop])
    print(list(cat_cont_set), '\n')
    return longer, pb_corr, pb_pv, pbs_drop, cat_cont_set, remaining_cont, remaining_cat


# Final update for features and data
def final_drop_and_update(point_bs_to_drop, training_set, test_set, features_list):
    """
    Function to update the training data set, test data set and feature names list by dropping the correlated features
    identified with the final point bi-serial method.

    Parameters
    ----------
    point_bs_to_drop : list
        list of identified feature indices to be dropped from the longer list (categorical or continuous)
    training_set : np.array
        array of the training features data set
    test_set : np.array
        array of the test features data set
    features_list : np.array
        array of the feature names

    Returns
    -------
    final_train : np.array
        final training data set after point bi-serial was applied
    final_test : np.array
        final test data set after point bi-serial was applied
    final_feat : list
        list of the remaining feature names after removing all highly correlated features
    """
    # Final update of the remaining train set, test set and feature lists after the two previous correlation checks
    # The idx are either referring to categorical or continuous features, depending on who is the longest list
    final_train = np.delete(training_set, point_bs_to_drop, axis=1)
    final_test = np.delete(test_set, point_bs_to_drop, axis=1)
    final_feat = [features_list[t] for t in range(len(features_list)) if t not in point_bs_to_drop]
    print(f"Checking the correlation between categorical features and continuous features lead to the removal "
          f"of {len(point_bs_to_drop)} features in both the training and test set.\n{len(final_feat)} features were "
          f"retained and will be passed to the machine learning classification pipeline.\n\nThe shape of training set "
          f"changed from {training_set.shape} to {final_train.shape}, and the test set from {test_set.shape} "
          f"to {final_test.shape}.\n")
    return final_train, final_test, final_feat


# RHCF summary
def rhcf_update_summary(training_features, testing_features, features_list, fin_train, fin_test, fin_feat, datatype,
                        cat_drop, cont_drop, pbs_drop, cont_idx, cat_idx, remain_cont, remain_cat, cramer_threshold,
                        spearman_threshold, pbs_threshold, remaining_train, remaining_test):
    """
    Function to print a complete summary of the RHCF effects on the training and test data set. Includes the changes in
    shape, the number of removed features by method, the configured thresholds, and which was the longest or shortest
    feature type.

    Parameters
    ----------
    training_features : np.array
        array of the training features data set
    testing_features : np.array
        array of the test features data set
    features_list : np.array
        array of the feature names
    fin_train : np.array
        array of the training features data set after RHCF
    fin_test : np.array
        array of the test features data set after RHCF
    fin_feat : np.array
        array of the remaining feature names after RHCF
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    cat_drop : list
        list of categorical feature indices to drop after corrected Cramers V correlation
    cont_drop : list
        list of continuous feature indices to drop after Spearman's Rank Order correlation
    pbs_drop : list
        list of longest feature type indices to drop after point bi-serial correlation (categorical or continuous)
    cont_idx : list
        list of all continuous feature indices in the data set
    cat_idx : list
        list of all categorical feature indices in the data set
    remain_cont : list
        list of remaining continuous feature indices in the data set after Spearman's Rank Order
    remain_cat : list
        list of remaining categorical feature indices in the data set after Cramers V
    cramer_threshold : tuple
        defined threshold to identify correlated features with Cramers V, either percentile value or decimal value e.g.
        (value, 'decimal')
    spearman_threshold : tuple
        defined threshold to identify correlated features with Spearman's Rank Order, either percentile value or
        decimal value e.g. (value, 'decimal')
    pbs_threshold : tuple
        defined threshold to identify correlated features with point bi-serial correlation, either percentile value or
        decimal value e.g. (value, 'decimal')
    remaining_train : np.array
        array of the remaining training features set after Cramers V and Spearman's Rank Order correlations
    remaining_test : np.array
        array of the remaining test features set after Cramers V and Spearman's Rank Order correlations

    Returns
    -------
    origin_train : np.array
        original array of training features data set before RHCF started
    origin_test : np.array
        original array of test features data set before RHCF started
    origin_feat : np.array
        original array of feature names before RHCF started
    training_features : np.array
        final training data set after RHCF is done
    testing_features : np.array
        final test data set after RHCF is done
    features_list : np.array
        final array of feature names after RHCF is done
    """
    # To be safe and recover original information later and avoid modifications, the original variables are replaced
    origin_train = training_features
    origin_test = testing_features
    origin_feat = features_list
    training_features = fin_train
    testing_features = fin_test
    features_list = fin_feat
    # Print the results
    print(f"Removing highly correlated features done in the {datatype} data with corrected Cramer's V, "
          f"Spearman's Rank Order Correlation, and Point Bi-Serial Correlation.\n"
          f"\n{datatype.capitalize()} data RHCF Summary:\n"
          f"******************************************\n"
          f"The original train and test shapes were reduced from {origin_train.shape}, {origin_test.shape} to "
          f"{(len(origin_train), len(origin_feat) - len(cat_drop))}, "
          f"{(len(origin_test), len(origin_feat) - len(cat_drop))} after corrected Cramer's V "
          f"between {len(cat_idx)} categorical features identified {len(cat_drop)} highly correlated "
          f"features above the {cramer_threshold} threshold.\n"
          f"The train and test shapes were further reduced from "
          f"{(len(origin_train), len(origin_feat) - len(cat_drop))}, "
          f"{(len(origin_test), len(origin_feat) - len(cat_drop))} to {remaining_train.shape}, {remaining_test.shape} "
          f"after spearman correlation between {len(cont_idx)} continuous features "
          f"identified {len(cont_drop)} highly correlated features above the {spearman_threshold} threshold.\n"
          f"Finally, the train and test shapes were further reduced from {remaining_train.shape}, "
          f"{remaining_test.shape} to the final shape {training_features.shape}, {testing_features.shape} after "
          f"point bi-serial correlation between {len(remain_cat)} remaining categorical and {len(remain_cont)} "
          f"remaining continuous features identified {len(pbs_drop)} features above the {pbs_threshold} threshold "
          f"to drop from the longest list.\n\n"
          f"The longest remaining data type was "
          f"'{'categorical' if len(remain_cat) >= len(remain_cont) else 'continuous'}':\n"
          f"Length remaining continuous="
          f"{len(remain_cont)- len(pbs_drop) if len(remain_cont) >= len(remain_cat) else len(remain_cont)}\n"
          f"Length remaining categorical="
          f"{len(remain_cat)- len(pbs_drop) if len(remain_cat) >= len(remain_cont) else len(remain_cat)}."
          f"\n******************************************\n")
    return origin_train, origin_test, origin_feat, training_features, testing_features, features_list


#############################################
# Visualizing the highly correlated heatmaps
#############################################
def cramer_heatmap(cramer_corr, cramer_threshold, datatype, categorical, folder_dir, tiff_size):
    """
    Function to plot the heatmap of the corrected Cramers V correlation results.

    Parameters
    ----------
    cramer_corr : np.array
        array of the Cramers V correlation results
    cramer_threshold : tuple
        defined threshold to identify correlated features either percentile value or decimal value e.g.
        (value, 'decimal')
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    categorical : list
        list of indices of the categorical features that underwent the correlation calculations
    folder_dir : str
        string referring to the folder directory where the plots should be saved
    tiff_size : int
        dot-per-inch size when saving the figure as .tiff file
    """
    if cramer_threshold[1] == 'decimal':
        cramer_hm = sns.heatmap(cramer_corr, center=cramer_threshold[0],
                                vmin=cramer_corr.min(), vmax=cramer_corr.max(),
                                cmap=sns.color_palette("vlag", as_cmap=True), xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Corrected Cramer's V correlation between {len(categorical)}\ncategorical "
            f"features, threshold: {cramer_threshold}",
            fontsize=14)
    else:  # If percentile
        cramer_panda = pd.DataFrame(cramer_corr)
        upper = cramer_panda.where(np.triu(np.ones(cramer_panda.shape), k=1).astype(bool))
        cramer_hm = sns.heatmap(cramer_corr, center=round(np.nanpercentile(upper, cramer_threshold[0]), 4),
                                vmin=cramer_corr.min(), vmax=cramer_corr.max(),
                                cmap=sns.color_palette("vlag", as_cmap=True), xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Corrected Cramer's V correlation between {len(categorical)}\ncategorical "
            f"features, threshold: {round(np.nanpercentile(upper, cramer_threshold[0]), 4)}, {cramer_threshold}",
            fontsize=14)
    cramer_hm.set_xticklabels(cramer_hm.get_xmajorticklabels(), fontsize=8)
    cramer_hm.set_yticklabels(cramer_hm.get_ymajorticklabels(), fontsize=8)
    cramer_hm.figure.axes[-1].set_ylabel('Correlation coefficient', size=10)
    cbar = cramer_hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel('# Categorical feature', fontsize=10)
    plt.ylabel('# Categorical feature', fontsize=10)
    cramer_hm.figure.savefig(folder_dir + f"/{datatype}_cramer_hm.tiff", dpi=tiff_size, bbox_inches='tight')
    plt.close(cramer_hm.figure)


def spearman_heatmap(spearman_corr, spearman_thresh, datatype, continuous, folder_dir, tiff_size):
    """
    Function to plot the heatmap of the Spearman's Rank Order correlation results.

    Parameters
    ----------
    spearman_corr : np.array
        array of the Spearman's Rank Order correlation results
    spearman_thresh : tuple
        defined threshold to identify correlated features either percentile value or decimal value e.g.
        (value, 'decimal')
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    continuous : list
        list of indices of the continuous features that underwent the correlation calculations
    folder_dir : str
        string referring to the folder directory where the plots should be saved
    tiff_size : int
        dot-per-inch size when saving the figure as .tiff file
    """
    if spearman_thresh[1] == 'decimal':
        spearman_hm = sns.heatmap(spearman_corr, center=spearman_thresh[0], vmin=spearman_corr.min(),
                                  vmax=spearman_corr.max(), cmap=sns.color_palette("vlag", as_cmap=True),
                                  xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Spearman correlation between {len(continuous)}\ncontinuous features, threshold: "
            f"{spearman_thresh}",
            fontsize=14)
    else:  # If percentile
        spearman_upper = pd.DataFrame(spearman_corr).where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
        spearman_hm = sns.heatmap(spearman_corr, center=round(np.nanpercentile(spearman_upper, spearman_thresh[0]), 4),
                                  vmin=spearman_corr.min(), vmax=spearman_corr.max(),
                                  cmap=sns.color_palette("vlag", as_cmap=True),
                                  xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Spearman correlation between {len(continuous)}\ncontinuous features, threshold: "
            f"{round(np.nanpercentile(spearman_upper, spearman_thresh[0]), 4)} {spearman_thresh}",
            fontsize=14)
    spearman_hm.set_xticklabels(spearman_hm.get_xmajorticklabels(), fontsize=8)
    spearman_hm.set_yticklabels(spearman_hm.get_ymajorticklabels(), fontsize=8)
    spearman_hm.figure.axes[-1].set_ylabel('Correlation coefficient', size=10)
    cbar = spearman_hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel('# Continuous feature', fontsize=10)
    plt.ylabel('# Continuous feature', fontsize=10)
    spearman_hm.figure.savefig(folder_dir + f"/{datatype}_spearman_hm.tiff", dpi=tiff_size, bbox_inches='tight')
    plt.close(spearman_hm.figure)


def pbs_heatmap(pbs_corr, pbs_threshold, datatype, remaining_cats, remaining_cont, longer, folder_dir, tiff_size):
    """
    Function to plot the heatmap of the Point Bi-Serial correlation results.

    Parameters
    ----------
    pbs_corr : np.array
        array of the point bi-serial correlation results
    pbs_threshold : tuple
        defined threshold to identify correlated features either percentile value or decimal value e.g.
        (value, 'decimal')
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    remaining_cats : list
        list of remaining categorical feature indices that undergo the correlation calculations after Cramers V and
        Spearman's Rank Order
    remaining_cont : list
        list of remaining continuous feature indices that undergo the correlation calculations after Cramers V and
        Spearman's Rank Order
    longer : list
        list of the feature type indices that is the longest (continuous or categorical)
    folder_dir : str
        string referring to the folder directory where the plots should be saved
    tiff_size : int
        dot-per-inch size when saving the figure as .tiff file
    """
    if pbs_threshold[1] == 'decimal':
        pbserial_hm = sns.heatmap(pbs_corr, center=pbs_threshold[0],
                                  vmin=pbs_corr.min(), vmax=pbs_corr.max(),
                                  cmap=sns.color_palette("vlag", as_cmap=True),
                                  xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Point bi-serial correlation between {len(remaining_cats)} cat.\n"
            f"and {len(remaining_cont)} continuous features, threshold: {pbs_threshold}",
            fontsize=14)
    else:  # If percentile
        # No need to take upper matrix here
        pbserial_hm = sns.heatmap(pbs_corr, center=round(np.nanpercentile(pbs_corr, pbs_threshold[0]), 4),
                                  vmin=pbs_corr.min(), vmax=pbs_corr.max(),
                                  cmap=sns.color_palette("vlag", as_cmap=True),
                                  xticklabels=10, yticklabels=10)
        plt.title(
            f"{datatype.capitalize()} Point bi-serial correlation between {len(remaining_cats)} cat.\n"
            f"and {len(remaining_cont)} continuous features, threshold: "
            f"{round(np.nanpercentile(pbs_corr, pbs_threshold[0]), 4)}, {pbs_threshold}",
            fontsize=14)
    pbserial_hm.set_xticklabels(pbserial_hm.get_xmajorticklabels(), fontsize=8)
    pbserial_hm.set_yticklabels(pbserial_hm.get_ymajorticklabels(), fontsize=8)
    pbserial_hm.figure.axes[-1].set_ylabel('Correlation coefficient', size=10)
    cbar = pbserial_hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    if longer == remaining_cats:
        plt.xlabel('# Continuous feature', fontsize=10)
        plt.ylabel('# Categorical feature', fontsize=10)
    else:
        plt.xlabel('# Categorical feature', fontsize=10)
        plt.ylabel('# Continuous feature', fontsize=10)
    pbserial_hm.figure.savefig(folder_dir + f"/{datatype}_pbserial_hm.tiff", dpi=tiff_size, bbox_inches='tight')
    plt.close(pbserial_hm.figure)


###################################################################
# Function to plot top important features after feature importance
###################################################################
def sorted_above_zero(importance_mean, bar_cap=20):
    """
    Function to sort the mean feature importance and to get the number of features which absolute importance is above 0.

    Parameters
    ----------
    importance_mean : np.array
        array of calculated mean feature importance after shuffling with sklearn, mlxtend, and eli5
    bar_cap : int
        integer to set the maximum of important features above zero to not overload the importance plot hereafter

    Returns
    -------
    sorted_idx : list
        list of sorted indices of mean feature importance (from lowest to highest)
    above_zero : int
        number of important features with importance mean above zero
    """
    sorted_idx = importance_mean.__abs__().argsort()
    above_zero = int(sum(abs(importance_mean) > 0) if sum(abs(importance_mean) > 0) < bar_cap else bar_cap)
    return sorted_idx, above_zero


def importance_plot(datatype, method, kern, idx_sorted, features_list, importance_mean,
                    importance_above_zero, importance_std=None):
    """
    Function to plot the sorted permuted feature importance of the most important features.

    Parameters
    ----------
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    method : str
        string referring to the feature importance method to be analysed (sklearn, mlxtend, eli5)
    kern : str
        string referring to the current SVC kernel tested, appears in title
    idx_sorted : list
        list of sorted indices of mean feature importance (from lowest to highest)
    features_list : np.array
        array of feature names
    importance_mean : np.array
        array of calculated mean feature importance after shuffling with sklearn, mlxtend, and eli5
    importance_above_zero : int
        number of important features with importance above zero
    importance_std : np.array
        array of calculated feature importance standard deviation if enabled (disabled by setting to None)
    """
    plt.figure()
    ax = plt.subplot()
    if importance_std is None:  # In case of linear kernel, no standard deviation available
        plt.barh(features_list[idx_sorted[-importance_above_zero:]],
                 importance_mean[idx_sorted[-importance_above_zero:]])
    else:  # Plot with error bars
        plt.barh(features_list[idx_sorted[-importance_above_zero:]],
                 importance_mean[idx_sorted[-importance_above_zero:]],
                 xerr=importance_std[idx_sorted[-importance_above_zero:]])
    plt.setp(ax.get_yticklabels(), rotation=30, ha='right', fontsize=6)  # to avoid labels overlapping
    plt.title(f'{datatype.capitalize()} feature importance {kern} {method}')
    plt.xlabel("Permuted Feature Importance" if method in ('mlxtend', 'sklearn', 'eli5') else "Feature Importance")
    plt.tight_layout()


###################################################################
# Function to plot box and bar plot of the most important features
###################################################################
def box_and_bar_plot(x_features, x_labels, y_features, y_labels, sorted_top_feature, feature_names, features_above_zero,
                     target_feature, negative_class, positive_class, datatype, kernel, folder_dir, tiff_size,
                     importance_method, graphs='combined', fontsize=18):
    """
    Function to create either separate or combined box and bar plots of the most important continuous and categorical
    features.

    Parameters
    ----------
    x_features : np.array
        array of the training features data set
    x_labels : np.array
        array of the training labels
    y_features : np.array
        array of the test features data set
    y_labels : np.array
        array of the test labels
    sorted_top_feature : list
        list of sorted feature indices after applied feature importance
    feature_names : np.array
        array of the feature names
    features_above_zero : int
        number of important features with importance above zero
    target_feature : str
        string referring to the target output feature of the analysis
    negative_class : str
        name of the negative target feature class
    positive_class : str
        name of the positive target feature class
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    kernel : str
        string referring to the current SVC kernel tested, appears in title
    folder_dir : str
        string referring to the folder direction where figures should be saved
    tiff_size : int
        integer of desired .tiff size
    importance_method : str
        string referring to the feature importance calculation method, appears in title
    graphs : str
        string referring to whether the box and bar plots should be created separately or combined, default 'combined'
    fontsize : int
        integer for desired font size of plot titles and axis labels, default 18
    """
    # Combine the train and test data and labels together
    full_data = np.concatenate((x_features, y_features), axis=0)
    full_labels = np.concatenate((x_labels, y_labels), axis=0)
    # Create a column with target feature information
    tmp = []
    for k in range(len(full_labels)):
        tmp += [positive_class if full_labels[k] == 1 else negative_class]
    # Transform combined data and truncated feature names into pandas dataframe via dictionary
    data_frame = pd.DataFrame({key: train_value for (key, train_value) in zip(feature_names,
                                                                              full_data.transpose())})
    # Truncate feature name for plot titles
    truncated_feature_names = [trunc_feature(k, 30, True) for k in feature_names]
    data_frame.columns = truncated_feature_names
    data_frame[target_feature] = tmp
    # Get the continuous and categorical indices of the most important features
    cont, cat = get_cat_and_cont(x_features[:, sorted_top_feature[-features_above_zero:][::-1]],
                                 y_features[:, sorted_top_feature[-features_above_zero:][::-1]])
    # Separate box and bar plot figures if graphs is set to 2 and if continuous features are present
    sns.set_style("whitegrid")  # Set seaborn plot style with grid
    if len(cont) > 0:
        # Merging only the data of continuous features for box plotting
        melted_cont_data_frame = pd.melt(data_frame.iloc[:,
                                         [sorted_top_feature[-features_above_zero:][::-1][k] for k in cont] + [-1]],
                                         id_vars=[target_feature], var_name=['Most important cont feature'])
        # Starting box plot
        if graphs == 'separated':
            cont_col_wrap_definer = round(len(cont) / int(np.sqrt(len(cont))))  # Number of columns
            ax = sns.catplot(data=melted_cont_data_frame, x=target_feature, y='value', hue=target_feature,
                             col='Most important cont feature', kind='box', col_wrap=cont_col_wrap_definer,
                             sharey=False, sharex=True, saturation=.5, aspect=1.2, height=2.5, linewidth=.8,
                             fliersize=.8, legend_out=True, legend=True, hue_order=[negative_class, positive_class])
            # Titles
            ax.set_titles("{col_name}")
            ax.figure.suptitle(f'{datatype.capitalize()} {kernel} box plot of most important continuous '
                               f'features {importance_method}',
                               fontsize=fontsize)
            # Labels and ticks
            ax.set(xticklabels=[], xticks=[])
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.figure.supylabel('Biomedical metrics', fontsize=int(3/4 * fontsize))
            ax.figure.supxlabel('Target classes', fontsize=int(3/4 * fontsize))
            # Legend
            ax.add_legend(loc='upper right', title=f'{target_feature} classes')
            sns.move_legend(ax, bbox_to_anchor=(0.99, 0.95), loc='upper right')
            # Layout, save and close
            ax.tight_layout(rect=[0.01, 0.03, 0.99, 0.9], h_pad=4)
            plt.savefig(folder_dir + f"/{datatype}_{kernel}_continuous_boxplot_{importance_method}.tiff",
                        dpi=tiff_size, bbox_inches='tight')
            plt.close(ax.figure)
    else:
        melted_cont_data_frame = None
    # Separate bar plot figures if graphs is set to 2 and if categorical features are present
    if len(cat) > 0:
        # Categorical features among the most important ones (melt by feature, group by feature and target, normalize
        # value counts, multiply by 100, call it percent and reset the index of the pandas frame in one line)
        melted_cat_data_frame = \
            pd.melt(data_frame.iloc[:, [sorted_top_feature[-features_above_zero:][::-1][k] for k in cat] + [-1]],
                    id_vars=[target_feature], var_name=['Most important cat feature']).groupby(
                ['Most important cat feature',
                 target_feature])['value'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        # Starting bar plot
        if graphs == 'separated':
            cat_col_wrap_definer = round(len(cat) / int(np.sqrt(len(cat))))  # Number of columns
            ax = sns.catplot(data=melted_cat_data_frame, x='value', y='percent', hue=target_feature,
                             col='Most important cat feature', kind='bar', col_wrap=cat_col_wrap_definer, sharey=True,
                             sharex=False, legend=True, legend_out=True, saturation=.5, aspect=1.6, height=2.5,
                             linewidth=.8, hue_order=[negative_class, positive_class])
            # Titles
            ax.set_titles("{col_name}")
            ax.figure.suptitle(f'{datatype.capitalize()} {kernel} bar plot of most important categorical '
                               f'features {importance_method}',
                               fontsize=fontsize)
            # Labels and ticks
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.figure.supylabel('Normalized counts [%]', fontsize=int(3/4 * fontsize))
            ax.figure.supxlabel('Feature categories', fontsize=int(3/4 * fontsize))
            ax.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
            # Collect percentage values to plot on top of bars
            for axis in ax.axes:
                labels = ['{:,.0f}'.format(x) for x in axis.get_xticks()]
                axis.set_xticklabels(labels, fontsize=8)
                for p in axis.patches:
                    txt = "%.2f" % p.get_height().round(2) + '%'
                    txt_x = p.get_x().round(2) + 0.17  # to put text in the middle of the bar, not the beginning
                    txt_y = p.get_height().round(2) + 2.50  # avoid touching the patches
                    if np.isnan(p.get_height()):  # case where no value exists
                        txt = '0.00%'
                        txt_y = 0 + 2.50  # avoid touching the x-axis
                    if p.get_height() > 70:  # decrease text position on y axis if above 70%
                        txt_y = p.get_height().round(2) - 25.00
                    # finally add the text with its position
                    axis.text(txt_x, txt_y, txt, fontsize=8, rotation='vertical')
            # Legend
            sns.move_legend(ax, loc='upper right', title=f'{target_feature} classes', bbox_to_anchor=(0.99, 0.95))
            # Layout, save and close
            ax.tight_layout(rect=[0.01, 0.03, 0.99, 0.9], h_pad=2)
            plt.savefig(folder_dir + f"/{datatype}_{kernel}_categorical_barplot_{importance_method}.tiff",
                        dpi=tiff_size, bbox_inches='tight')
            plt.close(ax.figure)
    else:
        melted_cat_data_frame = None
    # Combined box and bar plot figures if graphs is set to 1, ordering from most to least important feature
    if graphs == 'combined':
        # Define possible columns and rows number
        poss_cols = [i for i in range(1, len(sorted_top_feature) + 1) if len(sorted_top_feature) % i == 0]
        cols = int(np.round(np.median(poss_cols)))
        rows = int(np.ceil(len(sorted_top_feature) / cols))
        # In case of a prime number above 3
        if len(poss_cols) == 2 and max(poss_cols) > 3:
            cols = int(np.ceil(1 / 4 * len(sorted_top_feature)))
            rows = int(np.ceil(len(sorted_top_feature) / cols))
        # In case where only 1 feature shows up after permutation importance
        if len(poss_cols) in np.array([1, 2]) and max(poss_cols) < 3:
            cols = 2
            rows = 2
        if rows > cols:
            rows, cols = cols, rows
        # Start combined box and bar plot figure
        fig, axe = plt.subplots(nrows=rows, ncols=cols, figsize=(3*cols, 3*rows), sharex='none', sharey='none')
        for k in range(rows):
            for p in range(cols):
                if (p + k * cols) < features_above_zero:
                    # Create truncated feature name for the subplot titles and the feature to look for
                    trunc_name = \
                        trunc_feature(feature_names[sorted_top_feature[-features_above_zero:][::-1]][p + k * cols],
                                      30, True)
                    axe[k, p].set_title(f'{trunc_name}', fontsize=8)
                    # If the next most important feature is continuous
                    if (p + k * cols) in cont:
                        sns.boxplot(ax=axe[k, p],
                                    data=melted_cont_data_frame[melted_cont_data_frame['Most important cont feature']
                                                                == trunc_name],
                                    x=target_feature, y='value',
                                    hue=target_feature, hue_order=[negative_class, positive_class],
                                    saturation=.5, linewidth=.8, fliersize=.8)
                        # Remove legend and labels for each plot
                        axe[k, p].legend([], [], frameon=False)
                        axe[k, p].set(xticklabels=[], xticks=[])
                        axe[k, p].set(xlabel=None)
                        axe[k, p].set(ylabel=None)
                    # If the next most important feature is categorical
                    elif (p + k * cols) in cat:
                        sns.barplot(ax=axe[k, p],
                                    data=melted_cat_data_frame[melted_cat_data_frame['Most important cat feature']
                                                               == trunc_name],
                                    x='value', y='percent',
                                    hue=target_feature, hue_order=[negative_class, positive_class],
                                    saturation=.5, linewidth=.8)
                        # Remove legend and labels for each plot
                        axe[k, p].legend([], [], frameon=False)
                        axe[k, p].set(xlabel=None)
                        axe[k, p].set(ylabel=None)
                        # Reformat y axis
                        axe[k, p].set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
                        labels = ['{:,.0f}'.format(x) for x in axe[k, p].get_xticks()]
                        axe[k, p].set_xticklabels(labels, fontsize=8)
                        # Get percentage to put on top of bar
                        for patch in axe[k, p].patches:
                            txt = "%.2f" % patch.get_height().round(2) + '%'
                            txt_x = patch.get_x().round(2) + 0.17
                            txt_y = patch.get_height().round(2) + 2.50  # avoid touching the patches
                            if np.isnan(patch.get_height()):
                                txt = '0.00%'
                                txt_y = 0 + 2.50  # avoid touching the x-axis
                            if patch.get_height() > 60:  # decrease text position on y axis if above 60%
                                txt_y = patch.get_height().round(2) - 50.00
                            axe[k, p].text(txt_x, txt_y, txt, fontsize=7, rotation='vertical')
                else:
                    axe[k, p].set_axis_off()  # Remove axis of unnecessary subplots in figure
        handles, labels = axe[0, 0].get_legend_handles_labels()  # Fetch the legend handles of the first axis only
        fig.legend(handles, labels, loc='upper right', title=f'{target_feature} classes', bbox_to_anchor=(0.98, 0.98),
                   frameon=False)  # Use that legend as overall figure legend
        # Figure titles
        fig.suptitle(f'{datatype.capitalize()} {kernel} box and bar plot of most important '
                     f'features {importance_method}', fontsize=fontsize)
        fig.supylabel('Normalized counts [%] in bar plots; biomedical metrics in box plots',
                      fontsize=int(3/4 * fontsize))
        fig.supxlabel('Feature categories in bar plots; target classes in box plots', fontsize=int(3/4 * fontsize))
        # Layout, save and close
        fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.9], h_pad=4)
        plt.savefig(folder_dir + f"/{datatype}_{kernel}_combined_box_and_barplot_{importance_method}.tiff",
                    dpi=tiff_size, bbox_inches='tight')
        plt.close(fig)


############################################################################################
# Function to determine if session variables are picklable for storage and custom unpickler
############################################################################################
def is_picklable(objects):
    """
    Function to determine if the session variables are picklable for storage when running on HPC clusters.

    Parameters
    ----------
    objects : object
        session variable object to check whether or not it can be dumped by pickle

    Returns
    -------
    checker : bool
        boolean checker if the object is picklable or not
    """
    try:
        pickle.dumps(objects)
    except (Exception,):
        return False
    return True


# Custom unpickler with overridden class name finder (to load session option 2)
class CustomUnpickler(pickle.Unpickler):
    """
    A custom unpickler class that will override the find_class method in pickle.Unpickler.

    Methods
    -------
    find_class(self, module, name)
        the same find class function of pickle.Unpickler that we will override
    """
    def find_class(self, module, name):
        """
        Function to find the module and class name of the session variable.

        Parameters
        ----------
        module : str
            string referring to the module name
        name : str
            string referring to the variable name

        Returns
        -------
        super : object
            cooperative superclass method of the corresponding found class for each session variable
        """
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

        
#############################################################################################
# Function to process built-in feature importance by the linear SVM classifier for PCA & LDA
#############################################################################################
def linear_pca_avg_features_loadings(best_estimator, feature_step_name, cont_trans_name, pca_step_name, input_features,
                                     cont_idx, lin_importance_by_svm):
    """
    Function to retrieve and calculate the summed weighted pca loadings after linear PCA application.

    Parameters
    ----------
    best_estimator : GridSearchCV.best_estimator_
        previously fitted GridSearchCV estimator
    feature_step_name : str
        name of the feature transforming step inside the pipeline
    cont_trans_name : str
        name of the continuous transformer inside the column transformer of the pipeline
    pca_step_name : str
        name of the pca specific step inside the continuous transformer of the column transformer of that pipeline
    input_features : np.array
        array of total input features that were loaded to the GridSearchCV pipeline
    lin_importance_by_svm : np.array
        array of importance coefficient as returned by the .coef_ attribute of linear SVM classifier
    cont_idx : list
        list of continuous feature indices

    Returns
    -------
    sorted_summed_weighted_loadings : pd.DataFrame
        data frame of the sorted and summed weighted pca loadings including the corresponding feature names
    sum_of_explained_variance : str
        string in xx.xx% format informing about the sum of explained variance by the selected n_components of pca
    """
    # get access to the pca meta data
    enter_pca = best_estimator.named_steps[
        feature_step_name].named_transformers_[cont_trans_name].named_steps[pca_step_name]
    # get the number of selected components
    n_pca = enter_pca.components_.shape[0]
    sum_of_explained_variance = format(sum(enter_pca.explained_variance_), '.2f') + '%'
    pca_loadings = pd.DataFrame(enter_pca.components_.T, columns=[f'pca_component_{i}' for i in range(1, n_pca + 1)],
                                index=input_features[cont_idx])
    importance_of_continuous_features = abs(lin_importance_by_svm[0:n_pca])
    weighted_pca_loadings = pd.DataFrame()
    for k in pca_loadings.columns:
        imp_coef_position = 0
        weighted_pca_loadings[k] = abs(pca_loadings[k]) * importance_of_continuous_features[imp_coef_position]
        imp_coef_position += 1

    weighted_pca_loadings['sum'] = np.sum(weighted_pca_loadings, axis=1)
    sorted_summed_weighted_loadings = weighted_pca_loadings.sort_values(by='sum', ascending=False)
    return sorted_summed_weighted_loadings, sum_of_explained_variance


def linear_svm_get_features(best_estimator, lin_idx, categorical_trans_idx, continuous_trans_idx, input_features,
                            lin_importance_by_svm, ft_tech):
    """
    Function to find the correct feature names in case of linear SVM coef_ applied after ColumnTransformer.
    We know that ColumnTransformer is going step by step and concatenating the resulting transformation in our feature
    transformation case, continuous transformation are called before categorical transformation. Thus, in case of
    linear SVM (e.g. in combination with normal_pca at least) we can try to get the most important features back
    (at least the k best selected features,...)

    Parameters
    ----------
    best_estimator : GridSearchCV.best_estimator_
        previously fitted GridSearchCV estimator
    lin_idx : np.array
        array of sorted linear importance by SVM.coef_
    categorical_trans_idx : np.array
        array of input feature indices that undergo the categorical column transformation if present as the feature
        names can be directly retrieved by the way its implementation is working (SelectKBest), in contrast to PCA and
        kernelPCA where it is not directly possible to retrieve the feature names, but rather the number of most
        important principal component. With this information it could then be possible to get the most important
        features that affected each of the important principal component by looking at the sorted average loading of
        each feature across the selected PCA components
    continuous_trans_idx : np.array
        array of input feature indices that undergo the continuous column transformation if present
    input_features : np.array
        array of total input features that were loaded to the GridSearchCV pipeline
    lin_importance_by_svm : np.array
        array of importance coefficient as returned by the .coef_ attribute of linear SVM classifier
    ft_tech : str
        feature transformation tech from where to gather the coefficients, e.g. pca or lda

    Returns
    -------
    important_feature_names : np.array
        array of names for the most important features after column transformation and using linear SVM with its coef_
    """
    if 'features' not in best_estimator.named_steps:
        print('No feature transformation step found in the estimator pipeline, unable to retrieve most important '
              'feature names. Please note that for the reason this might be intended, the full feature list is returned'
              'assuming that no feature transformation takes place and n_input_features equal n_output_features.')
        return input_features
    else:
        if 'categorical' not in best_estimator.named_steps['features'].named_transformers_:
            print('No categorical transformer found inside the feature transformation step of the estimator pipeline, '
                  'unable to retrieve most important feature names. If a continuous transformation is present and it '
                  'is linear pca, than we can at least attribute the most important number of component')
            if 'continuous' not in best_estimator.named_steps['features'].named_transformers_:
                print('No categorical nor continuous feature transformation step found despite the presence of a step '
                      'called features. This assumes that no feature transformation takes place and returns the full '
                      'feature list. If this is not the case, the function will be passed and there might be an issue '
                      'with the step names, please revise.')
                return input_features
            else:
                pass
        else:
            feat_k_best = best_estimator.named_steps['features'].named_transformers_[
                'categorical'].get_support() if hasattr(best_estimator.named_steps['features'].named_transformers_[
                    'categorical'], 'get_support') else input_features[np.array(categorical_trans_idx)]
            most_important_cat = \
                np.where(feat_k_best == 1)[0] if feat_k_best.dtype == 'bool' else np.arange(len(feat_k_best))
            # these features are added after the pca transformation, so most_important_cat last features can be known
            best_cat_feat_appended = input_features[np.array(categorical_trans_idx)[most_important_cat]]
            # so the feat_k_best last values of lin_imp or arranged lin_idx are the best_cat_feat_appended in same order
            # The remaining len(lin_idx) - feat_k_best should then represent top n_components descending averaged
            # importance for each feature, as yielded by the above function linear_pca_avg_features_loadings
            remaining_cont_features = len(lin_idx) - len(most_important_cat)
            remaining_cont_features_sorted, sum_of_variance = \
                linear_pca_avg_features_loadings(best_estimator, 'features', 'continuous', ft_tech, input_features,
                                                 continuous_trans_idx, lin_importance_by_svm) if ft_tech == 'pca' else \
                (linear_lda_features_loadings(best_estimator, 'features', 'continuous', ft_tech,
                                              continuous_trans_idx, lin_importance_by_svm, input_features),
                 [None]) if ft_tech == 'lda' else (input_features[continuous_trans_idx], [None])
            # append 'PC contribution #_' to the remaining_cont_features_sorted if pca
            if ft_tech == 'pca':
                best_pc_feats = list(remaining_cont_features_sorted.index[0:remaining_cont_features])
                for pos in range(len(best_pc_feats)):
                    best_pc_feats[pos] = best_pc_feats[pos] + f' (PC #{pos + 1})'
                important_feature_names = best_pc_feats + list(best_cat_feat_appended)
                return np.array(important_feature_names), sum_of_variance
            elif ft_tech == 'lda':
                if len(remaining_cont_features_sorted) == 1:
                    best_ld_feats = input_features[tuple(remaining_cont_features_sorted)]
                else:
                    best_ld_feats = input_features[np.array(remaining_cont_features_sorted)]
                for pos in range(len(best_ld_feats)):
                    best_ld_feats[pos] = best_ld_feats[pos] + f' (LD #1, pos.{pos + 1})'
                important_feature_names = list(best_ld_feats) + list(best_cat_feat_appended)
                return np.array(important_feature_names)
            elif ft_tech == 'kernel_pca':
                kernel_components = best_estimator.get_params()['features__continuous__pca__n_components']
                best_kpc_feats = []
                for pos in range(kernel_components):
                    best_kpc_feats.append(f'Kernel PCA component #{pos + 1}')
                important_feature_names = list(best_kpc_feats) + list(best_cat_feat_appended)
                return np.array(important_feature_names)
            elif ft_tech == 'none':
                important_feature_names = list(input_features[continuous_trans_idx]) + list(best_cat_feat_appended)
                return np.array(important_feature_names)


def linear_lda_features_loadings(best_estimator, feature_step_name, cont_trans_name, lda_step_name, cont_idx,
                                 lin_importance_by_svm, input_features):
    """
    Function to retrieve and calculate the summed weighted lda loadings after LDA application in case of multiple
    n_components. If LDA components is 1 (e.g. if classification problem has only 2 classes), only the index of the most
    important feature will be returned. In that case of LDA components equal 1, the LDA coefficients will be normalized
    to 1 and expressed in percent for each feature. Then we can select the number of top features that explain over 95%
    of LDA, which might not always be just one feature.

    Parameters
    ----------
    best_estimator : GridSearchCV.best_estimator_
        previously fitted GridSearchCV estimator
    feature_step_name : str
        name of the feature transforming step inside the pipeline
    cont_trans_name : str
        name of the continuous transformer inside the column transformer of the pipeline
    lda_step_name : str
        name of the pca specific step inside the continuous transformer of the column transformer of that pipeline
    cont_idx : list
        list of continuous feature indices
    lin_importance_by_svm : np.array
        array of importance coefficient as returned by the .coef_ attribute of linear SVM classifier
    input_features : np.array
        array of total input features that were loaded to the GridSearchCV pipeline

    Returns
    -------
    most_important_feature_idx : list
        list of the most important features indices as found in the input features
    """
    # get access to the lda meta data
    enter_lda = best_estimator.named_steps[
        feature_step_name].named_transformers_[cont_trans_name].named_steps[lda_step_name]
    # get the number of selected components
    n_lda = enter_lda.n_components
    lda_loadings = enter_lda.coef_
    if n_lda > 1:
        importance_of_continuous_features = abs(lin_importance_by_svm[0:n_lda])
        weighted_lda_loadings = pd.DataFrame()
        for k in pd.DataFrame(lda_loadings).columns:
            imp_coef_position = 0
            weighted_lda_loadings[k] = abs(lda_loadings[k]) * importance_of_continuous_features[imp_coef_position]
            imp_coef_position += 1

        weighted_lda_loadings['sum'] = np.sum(weighted_lda_loadings, axis=1)
        sorted_summed_weighted_loadings = weighted_lda_loadings.sort_values(by='sum', ascending=False)
        return sorted_summed_weighted_loadings
    else:
        # If LDA only results in 1 linear discriminant dimension, we need to make sure that not only the first feature
        # is reported, but also those that may have similar loading scores (c.f. PCA summed weights). As we only have
        # one weight from the linear importance .coef_ by SVM, the summed weights in LDA only make sense if dimensions
        # are higher than 1. Thus, to still capture the most important feature in a single linear discriminant
        # dimension, we will first normalize the LDA loads and scan for features which sum explains 95% of LD1.
        normalized_lda_loading = abs(lda_loadings[0]) / sum(abs(lda_loadings[0]))
        sorted_normalized_idx = np.argsort(normalized_lda_loading)[::-1]
        tmp_score = float()
        counter = 0
        for pos in sorted_normalized_idx:
            if tmp_score < 0.95:
                tmp_score += normalized_lda_loading[pos]
                counter += 1
                if tmp_score >= 0.95:  # if above selection surpasses 95, remove last one
                    counter -= 1
        most_important_feature_idx = np.array(cont_idx)[sorted_normalized_idx[0:counter if counter < 5 else 3]]
        if len(most_important_feature_idx) > 1:
            print(f'\nLDA feature transformation resulted in one single dimension with more than one feature\nthat '
                  f'together explain 95% of the variance. In total, {counter} features are necessary to explain these '
                  f'95%. For better visualization, we select only the top 3 to top 5 features.\nThose features are: '
                  f'{input_features[most_important_feature_idx]}.\nAs it is currently not clear how to bypass this for '
                  f'the importance plot and box-bar-plot visualization,\nwe will stick with the most important one '
                  f'for the box and bar plot.\n')
        else:
            print(f'\nLDA feature transformation resulted in one single dimension with one single feature\nthat '
                  f'explains 95% of the variance. That feature is: {input_features[most_important_feature_idx]}.\n')
        return [most_important_feature_idx]


def scatter_plot_importance_technique(kernel, datatype, mean1, mean2, mean3, mean4,
                                      tuple_of_names, fontsize, permutation_technique, lin_out_real_idx):
    """
    Function to scatter plot and compare one method against the other and show the linear equation and r squared
    coefficient in case of multiple feature importance techniques, e.g. by permutation and linear SVM .coef_ attribute.

    Parameters
    ----------
    kernel : str
        name of current Support Vector Machines Classifier kernel
    datatype : str
        string referring to the data set being analyzed (e.g. full, male, female)
    mean1 : np.array
        first array of means to be compared with the other methods
    mean2 : np.array
        second array of means to be compared
    mean3 : np.array
        third array of means to be compared
    mean4 : np.array
        Fourth array of means to be compared
    tuple_of_names : tuple
        tuple of names to attribute to the three methods
    fontsize : int
        fontsize of the legend to be plotted
    permutation_technique : str
        name of selected permutation technique to know which means were produced
    lin_out_real_idx : np.array
        array of real indices of linear SVM_coef importance inside the full feature list
    """
    # In case where mean1 is not equal to the length of the other means that came from permutation importance, we need
    # to add zeros at the right position for mean1 as well as the other means
    if len(mean1) not in {len(mean2), len(mean3), len(mean4)}:
        mean1_tmp = mean1.copy()
        # as the first mean (lin_imp) is reduced, we fill the missing positions with 0
        mean1 = np.array([0.] * len(mean2))
        mean1[lin_out_real_idx] = mean1_tmp
        # as the other 3 means have all input features, we replace the surpluses with 0
        index_to_zero = [x for x in range(len(mean2)) if x not in lin_out_real_idx]
        mean2[index_to_zero] = 0.
        mean3[index_to_zero] = 0.
        mean4[index_to_zero] = 0.

    if permutation_technique == 'sklearn':  # only first case
        dicts = {f'{tuple_of_names[0]}': [mean1, mean2]}
    elif permutation_technique == 'eli5':  # only second case
        dicts = {f'{tuple_of_names[1]}': [mean1, mean3]}
    elif permutation_technique == 'mlxtend':  # only third case
        dicts = {f'{tuple_of_names[2]}': [mean1, mean4]}
    else:  # if all permutation techniques are selected
        dicts = {f'{tuple_of_names[0]}': [mean1, mean2],
                 f'{tuple_of_names[1]}': [mean1, mean3],
                 f'{tuple_of_names[2]}': [mean1, mean4]}

    figs, axes = plt.subplots(1, len(dicts), sharey='row', figsize=(16, 9))
    figs.suptitle(f'{datatype} {kernel} pairwise feature importance scatter plots between {len(lin_out_real_idx)} '
                  f'retained features limited by linear SVM (SVM_coef vs permutation)', size=18)
    figs.supylabel('Feature Importance by permutation', fontsize=12)
    figs.supxlabel('Feature Importance by SVM_coef', fontsize=12)
    # Define the axis content and plot
    if len(dicts) > 1:
        for axx, _ in zip(axes, range(len(dicts))):
            # Uncomment next line if you wish to calculate trend line equation
            [m, b] = np.polyfit(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1], 1)
            axx.scatter(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1], s=8, label=f'{tuple_of_names[_]}')
            # Uncomment next 2 lines if you wish to display the linear trend line
            axx.plot(dicts[tuple_of_names[_]][0], m * dicts[tuple_of_names[_]][0] + b,
                     linestyle=':', linewidth=1, color='m')
            # Define axis text, uncomment next 3 lines if you wish to have r squared and equation displayed
            if len(dicts[tuple_of_names[_]][0]) > 1:
                text = r"$r^2$ = " \
                  + f'{round(pg.corr(dicts[tuple_of_names[_]][0], dicts[tuple_of_names[_]][1])["r"].values[0] ** 2, 4)}' \
                  + '\nequation = ' + f'{round(m, 4)}*x + {round(b, 4)}'
                axx.plot([], label=text)
            axx.legend(handlelength=0, loc='upper left', fontsize=fontsize)
        figs.tight_layout()
    else:
        for key in dicts.keys():
            # Uncomment next line if you wish to calculate trend line equation
            [m, b] = np.polyfit(dicts[key][0], dicts[key][1], 1)
            axes.scatter(dicts[key][0], dicts[key][1], s=8, label=f'{key}')
            # Uncomment next 2 lines if you wish to display the linear trend line
            axes.plot(dicts[key][0], m * dicts[key][0] + b,
                      linestyle=':', linewidth=1, color='m')
            # Define axis text, uncomment next 3 lines if you wish to have r squared and equation displayed
            if len(dicts[key][0]) > 1:
                text = r"$r^2$ = " \
                  + f'{round(pg.corr(dicts[key][0], dicts[key][1])["r"].values[0] ** 2, 4)}' \
                  + '\nequation = ' + f'{round(m, 4)}*x + {round(b, 4)}'
                axes.plot([], label=text)
            axes.legend(handlelength=0, loc='upper left', fontsize=fontsize)
        figs.tight_layout()


def swap_words(string, first_to_swap, second_to_swap):
    """
    Function used during folder-name generator mechanism to swap KBEST and PCA/kPCA/LDA names if both are present in
    order to mimic the same order as is being applied inside the column transformer pipeline, where the continuous
    features are transformed first before the categorical features.

    Parameters
    ----------
    string : str
        string referring to the intermediate folder name
    first_to_swap : str
        first substring to be swapped with the one below (usually KBEST)
    second_to_swap : str
        second substring to be swapped with the one above (usually PCA, kPCA or LDA)

    Returns
    -------
    new_string : str
        resulting string where two substrings have been swapped
    """
    new_string = \
        second_to_swap.join(part.replace(second_to_swap, first_to_swap) for part in string.split(first_to_swap))
    return new_string


def dor_score(y_true, y_pred):
    """
    Diagnostic odds ratio scorer for medical binary classification scoring.

    Parameters
    ----------
    y_true : np.array
        array of ground truth labels
    y_pred : np.array
        predicted labels

    Returns
    -------
    dor : np.float
        floating dor score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # correcting contingency table to ensure DOR approximation in case we encounter division by zero
    tn = tn + .5
    fp = fp + .5
    fn = fn + .5
    tp = tp + .5
    # continue rates and likelihood calculations
    tpr = (tp/(tp+fn))
    fpr = (fp/(fp+tn))
    fnr = (fn/(tp+fn))
    tnr = (tn/(fp+tn))
    lr_plus = tpr/fpr
    lr_minus = fnr/tnr
    dor = lr_plus/lr_minus
    return dor


########################################################################################################################
# END OF CBD-P UTILS ###################################################################################################
########################################################################################################################
