# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Feature Importance Estimation Through Permutation
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
# Modified by Jeff DIDIER to support parallelization

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
#from dask.array.slicing import shuffle_slice
import dask.array as da
import dask


def feature_importance_permutation(X, y, predict_method,
                                   num_rounds=1, seed=None, n_jobs=None):
    """Feature importance imputation via permutation importance

    Parameters
    ----------

    X : NumPy array, shape = [n_samples, n_features]
        Dataset, where n_samples is the number of samples and
        n_features is the number of features.

    y : NumPy array, shape = [n_samples]
        Target values.

    predict_method : prediction function
        A callable function that predicts the target values
        from X.

    metric : str, callable
        The metric for evaluating the feature importance through
        permutation. By default, the strings 'accuracy' is
        recommended for classifiers and the string 'r2' is
        recommended for regressors. Optionally, a custom
        scoring function (e.g., `metric=scoring_func`) that
        accepts two arguments, y_true and y_pred, which have
        similar shape to the `y` array.

    num_rounds : int (default=1)
        Number of rounds the feature columns are permuted to
        compute the permutation importance.

    seed : int or None (default=None)
        Random seed for permuting the feature columns.

    n_jobs : int or None (default=None)
        Number of parallel jobs.

    Returns
    ---------

    mean_importance_vals, all_importance_vals : NumPy arrays.
      The first array, mean_importance_vals has shape [n_features, ] and
      contains the importance values for all features.
      The shape of the second array is [n_features, num_rounds] and contains
      the feature importance for each repetition. If num_rounds=1,
      it contains the same values as the first array, mean_importance_vals.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/

    """

    if not isinstance(num_rounds, int):
        raise ValueError('num_rounds must be an integer.')
    if num_rounds < 1:
        raise ValueError('num_rounds must be greater than 1.')

    # baseline score, does not need to be computed by each worker
    baseline = predict_method.score(X, y)

    # random number generator outside parallelized function and precompute random seed to get a fresh randomstate call for each worker
    rng = check_random_state(seed)
    # if dask parallel backend
    if hasattr(X, 'compute'):
    # parallel results do always end up in one zipped variable
        result = []
        for col_idx in range(X.shape[1]):
            res = dask.delayed(_parallel_mlxtend)(X, y, predict_method, baseline, num_rounds, rng, col_idx, distribution='dask')
            result.append(res.persist())
        # prepare vector for delayed task results
        result_dask = []
        for delay in result:
            result_dask.append(delay.compute())
        zipped_result = result_dask
    # else if another backend (e.g. ipyparallel, threading,...)
    else:
        zipped_result = Parallel(n_jobs=n_jobs)(delayed(_parallel_mlxtend)(X, y, predict_method, baseline, num_rounds, rng, col_idx, distribution=None) for col_idx in range(X.shape[1]))


    # unzip the results and turn back into arrays
    all_importance, mean_importance = zip(*zipped_result)
    all_importance_array = np.array(all_importance)
    mean_importance_array = np.array(mean_importance)

    return all_importance_array, mean_importance_array


def _parallel_mlxtend(X, y, predict_method, baseline, num_rounds, rng, col_idx, distribution=None):
    """Function to parallelize the columns to shuffle (also possible to parallelize the number of permutations, or both)
       Author: Jeff DIDIER

    Parameters
    ----------

    X : NumPy array, shape = [n_samples, n_features]
        Dataset, where n_samples is the number of samples and
        n_features is the number of features.

    y : NumPy array, shape = [n_samples]
        Target values.

    predict_method : prediction function
        A callable function that predicts the target values
        from X.

    score_func : str, callable
        The scoring function, defined by the metric parameter in
        the feature_importance function. It represents the metric
        for evaluating the feature importance through
        permutation. By default, the strings 'accuracy' is
        recommended for classifiers and the string 'r2' is
        recommended for regressors. Optionally, a custom
        scoring function (e.g., `metric=scoring_func`) that
        accepts two arguments, y_true and y_pred, which have
        similar shape to the `y` array.

    baseline : float
        The baseline score serves as reference to compare the true model score
        with the score yielded after permuting a given column.

    num_rounds : int (default=1)
        Number of rounds the feature columns are permuted to
        compute the permutation importance.

    rng : callable random number generator
        The random number generator is fetched from outside the function,
        assuring that each worker is operating a differnt random indexing.
        If the rng function would be initialized inside this parallel function,
        then each worker would start operating with the same random seed set.
        From sklearn's function:
        'Precompute random seed from the random state to be used to get a fresh independent RandomState instance for each
        parallel call to _calculate_permutation_scores, irrespective of the fact that variables are shared or not depending
        on the active joblib backend (sequential, thread-based or process-based).'

    col_idx : int
        The parameter col_idx is fetched from the for loop after the parallel function,
        which is distributed to the enabled workers, thus each worker is operating the
        permutation `num_rounds` times on one specific column.

    distribution : None or 'dask' (default=None)
        Whether dask or another parallel backend is used. As dask is treating arrays differently than most,
        the procedure of feature permutation importance for dask distribution is slightly different.

    Returns
    -------

    tasks_results : list
        As this function will be split to each worker, each worker will output 1 list
        of `tasks_results`, which are then stacked  when the parallel process has finished.
        The stacked lists will then be turned into numpy arrays by the main feature_importance function.

    mean : float
        Mean of the feature importance after `num_rounds` permutation of each column. Again, as this function
        is distributed to the workers, each worker will output 1 floating mean value.

    /!\ NOTE /!\

        As the function to parallelize is creating multiple output variables (`tasks_results` and `mean`) whereas
        joblib's Parallel function only returns the full results in zipped form, it is necessary in the feature_importance
        function to explicitly unzip the results (`zip(*results)`), which is then returning the `tasks_results` and
        `mean` variables as tupled 2d matrix and tupled list respectively. Both are then transformed into numpy
        arrays by the feature_importance function.

 """
    # What each worker will do to its distributed column:

    tasks_results = []

    # in case of dask distribution:
    if distribution == 'dask':
        for round_idx in range(num_rounds):
            save_col = X[:, col_idx].copy()
            shuffle_idx = np.arange(len(X))
            # to avoid that dask bags are shuffled bag-wise, we define the vector of shuffled idx first
            rng.shuffle(shuffle_idx)
            # to be sure that the shuffled idx are set correctly for the shuffled array, we use shuffle slice from dask
            #shuffled_array = shuffle_slice(X[:, col_idx], shuffle_idx)
            shuffled_array = X[shuffle_idx, col_idx]
            # the resulting shuffled array is computed to get the full vector and replace it in X
            X[:, col_idx] = shuffled_array
            # the new score is calculated based on the entire y and X datasets, to avoid bag-wise scoring, use compute()
            new_score = predict_method.score(X, y)
            # the shuffled column is transformed back to inital state
            X[:, col_idx] = save_col
            importance = baseline - new_score
            tasks_results.append(importance)

        mean = np.mean(tasks_results)

        return tasks_results, mean

    elif (distribution is None) or (distribution != 'dask'):
        for round_idx in range(num_rounds):
            # generate safe copy of the column to treat
            save_col = X[:, col_idx].copy()
            # now shuffle that column each round
            rng.shuffle(X[:, col_idx])
            # compute new score using the best estimator from gridsearch for each round
            new_score = predict_method.score(X, y)
            # put the shuffled column back in its original order before second shuffle
            X[:, col_idx] = save_col
            # compute the scoring difference between baseline (before shuffling) and the shuffled column
            importance = baseline - new_score
            # each worker will generate its own list containing the importance
            tasks_results.append(importance)

        # after the worker operated 'num_rounds' times, the mean of importance for that column is computed
        mean = np.mean(tasks_results)

        # each worker will return its 'tasks_results' and 'mean' variables as stacked lists, final output are both variables in zipped format
        return tasks_results, mean
