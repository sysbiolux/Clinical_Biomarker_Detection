# Functions for the analyses related to TNBC and circadian rythmicity
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
import umap
import seaborn as sns


def grad_color_map(cluster_df, targets):
    """
    Function to generate gradient color map for continuous target vectors.

    Parameters
    ----------
    cluster_df : pandas.core.frame.DataFrame
        Feature matrix
    targets : list
        list of target values to be mapped to a fix color

    Returns
    -------
    colors : list
        list of mapped colors
    """
    # create gradient color map if continuous targets
    cmap = {cluster_df.index[m]: targets[m] for m in range(len(cluster_df))}  # mapping correct color to correct point
    sm = ScalarMappable(norm=Normalize(vmin=min(list(cmap.values())), vmax=max(list(cmap.values()))),
                        cmap=sns.cubehelix_palette(as_cmap=True))
    colors = [sm.to_rgba(cmap[obsv_id]) for obsv_id in cluster_df.index]
    return colors


def plot_PCA(X=None, y=None, label=None, title='my_plot', seed=42):
    """
    Function to calculate and plot PCA and to colorize by multiple target labels.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Feature matrix
    y : pandas.core.frame.DataFrame
        Target matrix
    label : string
        Column name of target matrix to colorize points
    title : string
        Figure title
   seed : int
        Random number generator seed for reproducibility
    """
    if X is not None and y is not None and label is not None:
        # fit PCA
        pca = PCA(n_components=14, random_state=seed)  # we now we only need 14 here, the 15th is very close to zero
        pca.fit(X)
        PCs = pca.fit_transform(X)
        PCdf = pd.DataFrame(data=PCs, columns=["PC"+str(i) for i in range(1, PCs.shape[1]+1)])

        targets = list(y[label])
        all_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        if len(np.unique(targets)) <= 7:
            colors = all_colors[:len(np.unique(targets))]
        else:
            # create gradient color map if continuous targets
            colors = grad_color_map(PCdf, targets)

        # draw the 4 PCA plots (PC1 vs PC2, PC2 vs PC3, PC1 vs PC3, % variance)
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        for target, color in zip(np.unique(targets) if len(np.unique(targets)) <= 7 else targets, colors):
            idx = y[label] == target
            # pca plots
            ax1.scatter(PCdf.loc[idx.tolist(), 'PC1'], PCdf.loc[idx.tolist(), 'PC2'], color=color, s=30)
            ax1.set_title('PC1 v PC2', fontsize=10)
            ax1.set_xlabel(f'PC1 ({"{:.2f}".format(round(pca.explained_variance_ratio_[0] * 100, 2))}%)')
            ax1.set_ylabel(f'PC2 ({"{:.2f}".format(round(pca.explained_variance_ratio_[1] * 100, 2))}%)')
            ax2.scatter(PCdf.loc[idx.tolist(), 'PC2'], PCdf.loc[idx.tolist(), 'PC3'], color=color, s=30)
            ax2.set_title('PC2 v PC3', fontsize=10)
            ax2.set_xlabel(f'PC2 ({"{:.2f}".format(round(pca.explained_variance_ratio_[1] * 100, 2))}%)')
            ax2.set_ylabel(f'PC3 ({"{:.2f}".format(round(pca.explained_variance_ratio_[2] * 100, 2))}%)')
            ax3.scatter(PCdf.loc[idx.tolist(), 'PC1'], PCdf.loc[idx.tolist(), 'PC3'], color=color, s=30)
            ax3.set_title('PC1 v PC3', fontsize=10)
            ax3.set_xlabel(f'PC1 ({"{:.2f}".format(round(pca.explained_variance_ratio_[0] * 100, 2))}%)')
            ax3.set_ylabel(f'PC3 ({"{:.2f}".format(round(pca.explained_variance_ratio_[2] * 100, 2))}%)')
            # variance bar plot
            ax4.bar(range(1, PCs.shape[1] + 1), pca.explained_variance_ratio_ * 100, color='skyblue')
            for i in range(PCs.shape[1]):
                ax4.annotate(str("{:.2f}".format(round(pca.explained_variance_ratio_[i] * 100, 2))),
                             xy=(i + 1, pca.explained_variance_ratio_[i] * 100), ha='center', va='bottom',
                             size=8, weight='normal')
            ax4.set_title('Explained variance by principal components', fontsize=10)
            ax4.set_xlabel('Principal components')
            ax4.set_ylabel('Variance [%]')
            # control ticks of axis 4
            plt.sca(ax4)
            plt.xticks(range(1, PCs.shape[1] + 1))
            plt.suptitle(title, fontsize=14)
            if len(np.unique(targets)) <= 7:
                f.legend(np.unique(targets), loc='upper right', ncol=2, fontsize=7)  # other font size to not overlap with titles
            else:
                leg = f.legend([f'min: {"{:.2f}".format(round(min(targets), 2))}',
                                f'max: {"{:.2f}".format(round(max(targets), 2))}'],
                               labelcolor=[min(zip(targets, colors))[1], max(zip(targets, colors))[1]],
                               loc='upper right', ncol=1, fontsize=9)
                leg.legendHandles[0].set_color(min(zip(targets, colors))[1])
                leg.legendHandles[1].set_color(max(zip(targets, colors))[1])
            f.tight_layout()
    else:
        raise ValueError('Either X or y or label is not set.')


def plot_UMAP(X=None, y=None, label=None, n_neighbors=15, seed=42, title='my_plot'):
    """
    Function to calculate and plot UMAP and to colorize by multiple target labels.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Feature matrix
    y : pandas.core.frame.DataFrame
        Target matrix
    label : string
        Column name of target matrix to colorize points
    n_neighbors : int
        Control how UMAP balances local versus global structure in the data
    seed : int
        Random number generator seed for reproducibility
    title : string
        Figure title
    """
    if X is not None and y is not None and label is not None:
        # fit UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=seed)
        UMAPs = reducer.fit_transform(X)
        UMAPdf = pd.DataFrame(data=UMAPs, columns=["UMAP1", "UMAP2"])

        targets = list(y[label])
        all_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        if len(np.unique(targets)) <= 7:
            colors = all_colors[:len(np.unique(targets))]
        else:
            # create gradient color map if continuous targets
            colors = grad_color_map(UMAPdf, targets)
        # draw the UMAP plot
        plt.figure(figsize=(10, 10))
        for target, color in zip(np.unique(targets) if len(np.unique(targets)) <= 7 else targets, colors):
            idx = y[label] == target
            # UMAP plot
            plt.scatter(UMAPdf.loc[idx.tolist(), 'UMAP1'], UMAPdf.loc[idx.tolist(), 'UMAP2'], color=color, s=30)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.xticks([])
            plt.yticks([])
            plt.title(title, fontsize=14)
            if len(np.unique(targets)) <= 7:
                plt.legend(np.unique(targets), loc='best', ncol=2, fontsize=9)
            else:
                # in this case, we weirdly have to create the legend handles ourselves, else it will only yield 1 handle
                mini = mlines.Line2D([], [], color=min(zip(targets, colors))[1], marker='o', ls='',
                                     label=f'min: {"{:.2f}".format(round(min(targets), 2))}')
                maxi = mlines.Line2D([], [], color=max(zip(targets, colors))[1], marker='o', ls='',
                                     label=f'min: {"{:.2f}".format(round(max(targets), 2))}')
                plt.legend(handles=[mini, maxi],
                           labelcolor=[min(zip(targets, colors))[1], max(zip(targets, colors))[1]],
                           loc='best', fontsize=9)
            plt.tight_layout()
    else:
        raise ValueError('Either X or y or label is not set.')


def plot_tSNE(X=None, y=None, label=None, seed=42, title='my_plot', **kwargs):
    """
    Function to calculate and plot t-SNE and to colorize by multiple target labels.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Feature matrix
    y : pandas.core.frame.DataFrame
        Target matrix
    label : string
        Column name of target matrix to colorize points
    seed : int
        Random number generator seed for reproducibility
    title : string
        Figure title
    """
    if X is not None and y is not None and label is not None:
        # fit t-SNE
        tsne = TSNE(random_state=seed, **kwargs)  # default 2 components
        tsne_res = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(data=tsne_res, columns=["t-SNE1", "t-SNE2"])

        targets = list(y[label])
        all_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        if len(np.unique(targets)) <= 7:
            colors = all_colors[:len(np.unique(targets))]
        else:
            # create gradient color map if continuous targets
            colors = grad_color_map(tsne_df, targets)
        # draw the tSNE plot
        plt.figure(figsize=(10, 10))
        for target, color in zip(np.unique(targets) if len(np.unique(targets)) <= 7 else targets, colors):
            idx = y[label] == target
            # tSNE plot
            plt.scatter(tsne_df.loc[idx.tolist(), 't-SNE1'], tsne_df.loc[idx.tolist(), 't-SNE2'], color=color, s=30)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(title, fontsize=14)
            if len(np.unique(targets)) <= 7:
                plt.legend(np.unique(targets), loc='upper right', ncol=2, fontsize=9)
            else:
                # in this case, we weirdly have to create the legend handles ourselves, else it will only yield 1 handle
                mini = mlines.Line2D([], [], color=min(zip(targets, colors))[1], marker='o', ls='',
                                     label=f'min: {"{:.2f}".format(round(min(targets), 2))}')
                maxi = mlines.Line2D([], [], color=max(zip(targets, colors))[1], marker='o', ls='',
                                     label=f'min: {"{:.2f}".format(round(max(targets), 2))}')
                plt.legend(handles=[mini, maxi],
                           labelcolor=[min(zip(targets, colors))[1], max(zip(targets, colors))[1]],
                           loc='best', fontsize=9)
            plt.tight_layout()
    else:
        raise ValueError('Either X or y or label is not set.')


def plot_LDA(X=None, y=None, label=None, title='my_plot', lda_output_target=None, seed=42):
    """
    Function to calculate and plot LDA and to colorize by multiple target labels.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Feature matrix
    y : pandas.core.frame.DataFrame
        Target matrix
    label : string
        Column name of target matrix to colorize points
    title : string
        Figure title
    lda_output_target : string
        LDA-specific output target to control on what feature the supervised LDA is fitted, as it would not work for
        continuous features
    seed : int
        Random number generator seed for reproducibility
    """
    if X is not None and y is not None and label is not None:
        # fit LDA as supervised or unsupervised classifier!
        comp = len(y[lda_output_target].unique()) - 1
        LDA = LinearDiscriminantAnalysis(n_components=comp)
        # lda components = number of classes - 1 (only for subtypes)
        LDAs = LDA.fit_transform(X, y[lda_output_target])  # Make sure LDA is only fit to categorical subtypes
        LDAdf = pd.DataFrame(data=LDAs, columns=[f'LD{number + 1}' for number in np.arange(comp)])

        targets = list(y[label])
        all_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        if len(np.unique(targets)) <= 7:
            colors = all_colors[:len(np.unique(targets))]
        else:
            # create gradient color map if continuous targets
            colors = grad_color_map(LDAdf, targets)

        if comp >= 3:
            # draw the 4 LDA plots (LD1 vs LD2, LD2 vs LD3, LD1 vs LD3, % variance)
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
            for target, color in zip(np.unique(targets) if len(np.unique(targets)) <= 7 else targets, colors):
                idx = y[label] == target
                # pca plots
                ax1.scatter(LDAdf.loc[idx.tolist(), 'LD1'], LDAdf.loc[idx.tolist(), 'LD2'], color=color, s=30)
                ax1.set_title('LD1 v LD2', fontsize=10)
                ax1.set_xlabel(f'LD1 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[0] * 100, 2))}%)')
                ax1.set_ylabel(f'LD2 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[1] * 100, 2))}%)')
                ax2.scatter(LDAdf.loc[idx.tolist(), 'LD2'], LDAdf.loc[idx.tolist(), 'LD3'], color=color, s=30)
                ax2.set_title('LD2 v LD3', fontsize=10)
                ax2.set_xlabel(f'LD2 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[1] * 100, 2))}%)')
                ax2.set_ylabel(f'LD3 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[2] * 100, 2))}%)')
                ax3.scatter(LDAdf.loc[idx.tolist(), 'LD1'], LDAdf.loc[idx.tolist(), 'LD3'], color=color, s=30)
                ax3.set_title('LD1 v LD3', fontsize=10)
                ax3.set_xlabel(f'LD1 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[0] * 100, 2))}%)')
                ax3.set_ylabel(f'LD3 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[2] * 100, 2))}%)')
        else:
            # draw only 1 LDA in case of binarized feature
            np.random.seed(seed)
            jittered_y = pd.DataFrame(np.zeros(len(LDAdf)) + 0.1 * np.random.rand(len(LDAdf)) - 0.05,
                                      columns=[f'LD{number + 1}' for number in np.arange(comp)])
            f, (ax1, ax4) = plt.subplots(1, 2, figsize=(10, 5))
            for target, color in zip(np.unique(targets) if len(np.unique(targets)) <= 7 else targets, colors):
                idx = y[label] == target
                ax1.scatter(LDAdf.loc[idx.tolist(), 'LD1'], jittered_y.loc[idx.tolist(), 'LD1'],  color=color, s=30,
                            alpha=0.5)
                ax1.set_title('LD1', fontsize=10)
                ax1.set_xlabel(f'LD1 ({"{:.2f}".format(round(LDA.explained_variance_ratio_[0] * 100, 2))}%)')
                ax1.set_ylabel('')
                ax1.set_ylim([-0.1, 0.1])
                ax1.set_yticks([])
        # variance bar plot
        ax4.bar(range(1, LDAs.shape[1] + 1), LDA.explained_variance_ratio_ * 100, color='skyblue')
        for i in range(LDAs.shape[1]):
            ax4.annotate(str("{:.2f}".format(round(LDA.explained_variance_ratio_[i] * 100, 2))),
                         xy=(i + 1, LDA.explained_variance_ratio_[i] * 100), ha='center', va='bottom',
                         size=8, weight='normal')
            ax4.set_title('Explained variance by linear discriminant components', fontsize=10)
            ax4.set_xlabel('Discriminant components')
            ax4.set_ylabel('Variance [%]')
        # control ticks of axis 4
        plt.sca(ax4)
        plt.xticks(range(1, LDAs.shape[1] + 1))
        plt.suptitle(title, fontsize=14)
        if len(np.unique(targets)) <= 7:
            f.legend(np.unique(targets), loc='upper right', ncol=2, fontsize=7)
        else:
            leg = f.legend([f'min: {"{:.2f}".format(round(min(targets), 2))}',
                            f'max: {"{:.2f}".format(round(max(targets), 2))}'],
                           labelcolor=[min(zip(targets, colors))[1], max(zip(targets, colors))[1]],
                           loc='upper right', ncol=1, fontsize=9)
            leg.legendHandles[0].set_color(min(zip(targets, colors))[1])
            leg.legendHandles[1].set_color(max(zip(targets, colors))[1])
        f.tight_layout()
    else:
        raise ValueError('Either X or y or label is not set.')
