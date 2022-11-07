# ## Python 3.8
#############################33
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# from os import path
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA, FastICA
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.metrics import mean_squared_error


#############################33
import sys
# assert sys.version_info >= (3, 8)
# import mlrose_hiive
import numpy as np
import pandas as pd
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator, FlipFlopGenerator, KnapsackGenerator,ContinuousPeaksGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner
# # import itertools as it
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, KFold
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import svm
# # from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KNeighborsClassifier
#
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
# from sklearn.metrics import confusion_matrix
#
# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
import time as tm
# # from sklearn import metrics
from os import path


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

from sklearn.datasets import fetch_openml

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel  # <<<  ***
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


def km(X,cl=0): # k-means clustering

    if cl == 2:
        clusterer = KMeans(n_clusters=2, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        return cluster_labels
    elif cl == 4:
        clusterer = KMeans(n_clusters=4, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        return cluster_labels


    distortions = []
    inertias = []

    for n_clusters in range(1, 11):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        clusterer.fit(X)
        distortions.append(sum(np.min(cdist(X, clusterer.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
        inertias.append(clusterer.inertia_)

    plt.plot(range(1, 11), distortions, 'og-')
    plt.xlabel('The number of clusters K')
    plt.ylabel('Distortion')
    plt.title('The Elbow plot analysis for the various clusters using Distortion')
    # plt.show()
    plt.savefig('Raisin_KM_Elbow_dis.png')  # save plot
    plt.close()

    plt.plot(range(1, 11), inertias, 'og-')
    plt.xlabel('The number of clusters K')
    plt.ylabel('Inertia')
    plt.title('The Elbow plot analysis for the various clusters using Inertia')
    # plt.show()
    plt.savefig('Raisin_KM_Elbow_in.png')  # save plot



    range_n_clusters = [2, 3, 4, 5]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        # fig, (ax1, ax2) = plt.subplots(1, 2)  # <<<<<<<<
        fig, (ax1) = plt.subplots(1)
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


        plt.suptitle(
            "Silhouette analysis for KMeans clustering with n_clusters = %d"
            # "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            # fontsize=14,
            # fontweight="bold",
        )
        # plt.show()
        plt.savefig('Raisin_KM_Silhouette_'+ str(n_clusters) +'.png')  # save plot

    # plt.show()





def em(X, cl=0): # Expectation Maximization
    if cl == 2:
        clusterer = GaussianMixture(n_components=2, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        return cluster_labels
    elif cl == 4:
        clusterer = GaussianMixture(n_components=4, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        return cluster_labels


    range_n_clusters = [2, 3, 4, 5]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        # fig, (ax1, ax2) = plt.subplots(1, 2)  # <<<<<<<<
        fig, (ax1) = plt.subplots(1)
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        # cluster_labels = clusterer.fit_predict(X)
        gmm = GaussianMixture(n_components=n_clusters, random_state=10)
        # gmm.fit(X)
        cluster_labels = gmm.fit_predict(X)



        # gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        # gm.means_
        # gm.predict([[0, 0], [12, 3]])
        #
        # gmm = GaussianMixture(n_components=4)
        # gmm.fit(X)
        # labels = gmm.predict(X)
        # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis');




        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


        plt.suptitle(
            "Silhouette analysis for Expectation Maximization with n_components = %d"
            # "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            # fontsize=14,
            # fontweight="bold",
        )
        # plt.show()
        plt.savefig('Raisin_EM_Silhouette_'+ str(n_clusters) +'.png')  # save plot




def pca(X,y):


    # X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1 / 7.0, random_state=0)

    # X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)



    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.

    # X_train = scaler.transform(X_train)
    # X_test  = scaler.transform(X_test)


    # scale std
    X_train_std = scaler.transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # Note:
    ######## A pca 0.95 on unscaled data reduced to 1 dimension, while
    ## standardized data reduced to 3 dimensions from a total of 8.

    # pca     = PCA(.95)        #<<<<<<<<<<<<<< <<<<<<<<<<<<
    # pca_std = PCA(.95)
    pca     = PCA(7)
    pca_std = PCA(7)

    pca.fit(X_train)
    pca_std.fit(X_train_std)

    # Show first principal components
    print(f"\nPC 1 without scaling:\n{pca.components_[0]}")
    print(f"\nPC 1 with scaling:\n{pca_std.components_[0]}")

    # Unscaled
    X_train_transformed  = pca.transform(X_train)
    X_test               = pca.transform(X_test)

    # Standard scaled
    X_train_std_transformed = pca_std.transform(X_train_std)
    X_test_std_transformed  = pca_std.transform(X_test_std)


    # Eigen values
    print('Explained variance ratio:',pca_std.explained_variance_ratio_)
    cummulative = np.cumsum(pca_std.explained_variance_ratio_)
    # plt.plot(range(1, 8), pca_std.explained_variance_ratio_, 'og-')
    plt.plot(range(1, 8), cummulative, 'og-')
    plt.xlabel('n-th Principal Component')
    plt.ylabel('Explained variance ratio (cummulative)')
    plt.title('The Eigen value analysis of components generated by PCA')
    # plt.show()
    plt.savefig('Raisin_PCA_Eigen_cumm.png')  # save plot
    plt.close()


    # Unscaled
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X_train_transformed, y_train)
    score = logisticRegr.score(X_test, y_test)

    # # Predict for One /Multiple Observation (image)
    # one = logisticRegr.predict(X_test[0].reshape(1, -1))
    # many = logisticRegr.predict(X_test[0:10])

    # Standard scaled
    logisticRegr_std = LogisticRegression(solver='lbfgs')
    logisticRegr_std.fit(X_train_std_transformed, y_train)
    score_std = logisticRegr_std.score(X_test_std_transformed, y_test)

    print("\nAccuracy:", score)
    print("Accuracy Std:", score_std)


    # visualize standardized vs. untouched dataset with PCA performed
    FIG_SIZE = (10, 7)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)

    target_classes = range(0, 2)
    colors = ("blue", "red")
    markers = ("s", "o")

    for target_class, color, marker in zip(target_classes, colors, markers):
        # x1 = X_train_transformed[y_train == target_class, 0]
        # y1 = X_train_transformed[y_train == target_class, 1]
        ax1.scatter(
            x=X_train_transformed[y_train == target_class, 0],
            y=X_train_transformed[y_train == target_class, 1],
            color=color,
            label=f"class {target_class}",
            alpha=0.5,
            marker=marker,
        )

        ax2.scatter(
            x=X_train_std_transformed[y_train == target_class, 0],
            y=X_train_std_transformed[y_train == target_class, 1],
            color=color,
            label=f"class {target_class}",
            alpha=0.5,
            marker=marker,
        )

    ax1.set_title("Training dataset after PCA")
    ax2.set_title("Standardized training dataset after PCA")

    for ax in (ax1, ax2):
        ax.set_xlabel("1st principal component")
        ax.set_ylabel("2nd principal component")
        ax.legend(loc="upper right")
        ax.grid()

    plt.tight_layout()
    plt.savefig('Raisin_PCA_.png')  # save plot
    # plt.show()

    return X_train_std_transformed, X_test_std_transformed, y_train, y_test


def ica(X,y):


    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)


    # scale std
    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.
    X_train_std = scaler.transform(X_train)
    X_test_std  = scaler.transform(X_test)


    # ica_std = FastICA(n_components=2)
    ica_std = FastICA(n_components=7)
    ica_std.fit(X_train_std)
    X_train_std_transformed = ica_std.transform(X_train_std)
    X_test_std_transformed  = ica_std.transform(X_test_std)
    A_ = ica_std.mixing_  # Get estimated mixing matrix


    # Kurtosis
    from scipy.stats import norm, kurtosis
    kurt_orignal = kurtosis(X_train_std, fisher=True)
    kurt = kurtosis(X_train_std_transformed, fisher=True)

    # tips = sns.load_dataset("tips")
    # sns.kdeplot(data=tips, x="total_bill")

    print('ICA Kurtosis original:',kurt_orignal)
    print('ICA Kurtosis:',kurt)

    plt.plot(range(0, 7), kurt_orignal, 'og-')
    plt.plot(range(0, 7), kurt, 'xb-')
    ax = plt.gca()
    ax.legend(['Original', 'ICA Tranformed'])
    plt.xlabel('Component')
    plt.ylabel('Kurtosis')
    plt.title('The Kurtosis of components generated by ICA')
    # plt.show()
    plt.savefig('Raisin_ICA_kurtosis.png')
    plt.close()

    # print(f"\nPC 1 with scaling:\n{ica_std.components_[0]}")


    # ############    ############
    # Standard scaled
    logisticRegr_std = LogisticRegression(solver='lbfgs')
    logisticRegr_std.fit(X_train_std_transformed, y_train)
    score_std = logisticRegr_std.score(X_test_std_transformed, y_test)

    print("Accuracy Std:", score_std)
    ##################    ##################


    # visualize standardized vs. untouched dataset with PCA performed
    FIG_SIZE = (10, 7)
    fig, (ax2) = plt.subplots(ncols=1, figsize=FIG_SIZE)


    target_classes = range(0, 2)
    colors = ("blue", "red")
    markers = ("s", "o")

    for target_class, color, marker in zip(target_classes, colors, markers):
        ax2.scatter(
            x=X_train_std_transformed[y_train == target_class, 0],
            y=X_train_std_transformed[y_train == target_class, 1],
            color=color,
            label=f"class {target_class}",
            alpha=0.5,
            marker=marker,
        )

    ax2.set_title("Standardized training dataset after ICA")
    ax2.set_xlabel("1st principal component")
    ax2.set_ylabel("2nd principal component")
    ax2.legend(loc="upper right")
    ax2.grid()

    plt.tight_layout()
    plt.savefig('Raisin_ICA_.png')  # save plot
    # plt.show()

    return X_train_std_transformed, X_test_std_transformed, y_train, y_test

def rp(X,y): # Randomized Projections


    # rng = np.random.RandomState(42)
    # X = rng.rand(25, 3000)
    # transformer = GaussianRandomProjection(random_state=rng)
    # X_new = transformer.fit_transform(X)
    # X_new.shape

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)


    # scale std
    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.
    X_train_std = scaler.transform(X_train)
    X_test_std  = scaler.transform(X_test)

    #############################33
    rmse_all=[]
    rmse_all.append(0.0) # Dummy val to shift
    for n_component in range(1, 8):

        # rp_std = GaussianRandomProjection(n_components=2)
        rp_std = GaussianRandomProjection(n_components=n_component,compute_inverse_components=True)
        rp_std.fit(X_train_std)
        X_train_std_transformed = rp_std.transform(X_train_std)
        X_train_std_transformed_inversed = rp_std.inverse_transform(X_train_std_transformed)

        # rmse_comp = mean_squared_error(X_train_std, X_train_std_transformed_inversed, multioutput='raw_values',squared=False)
        rmse = mean_squared_error(X_train_std, X_train_std_transformed_inversed, squared=False)

        rmse_all.append(rmse)
    print('RMSE All:',rmse_all)
    plt.plot(range(1, 8), rmse_all[1:], 'og-')
    ax = plt.gca()
    plt.xlabel('Components')
    plt.ylabel('RMSE')
    plt.title('The RMSE of orignal vs inverse of feature generated by RP')
    # plt.show()
    plt.savefig('Raisin_RP_rmse.png')
    plt.close()

    # X_train_std_transformed_again = rp_std.transform(X_train_std_transformed_inversed)
    # # test1 = np.allclose(X_train_std_transformed, X_train_std_transformed_again)
    # # # test = np.allclose(X_train_std, X_train_std_transformed_inversed)
    # #############################33




    # print(f"\nPC 1 with scaling:\n{rp_std.components_[0]}")


    # ############    ############
    # Standard scaled
    logisticRegr_std = LogisticRegression(solver='lbfgs')
    logisticRegr_std.fit(X_train_std_transformed, y_train)
    score_std = logisticRegr_std.score(X_test_std_transformed, y_test)

    print("Accuracy Std:", score_std)
    ##################    ##################


    # visualize standardized vs. untouched dataset with PCA performed
    FIG_SIZE = (10, 7)
    fig, (ax2) = plt.subplots(ncols=1, figsize=FIG_SIZE)


    target_classes = range(0, 2)
    colors = ("blue", "red")
    markers = ("s", "o")

    for target_class, color, marker in zip(target_classes, colors, markers):
        ax2.scatter(
            x=X_train_std_transformed[y_train == target_class, 0],
            y=X_train_std_transformed[y_train == target_class, 1],
            color=color,
            label=f"class {target_class}",
            alpha=0.5,
            marker=marker,
        )

    ax2.set_title("Standardized training dataset after RP")
    ax2.set_xlabel("1st principal component")
    ax2.set_ylabel("2nd principal component")
    ax2.legend(loc="upper right")
    ax2.grid()

    plt.tight_layout()
    plt.savefig('Raisin_RP_.png')  # save plot
    # plt.show()

    return X_train_std_transformed, X_test_std_transformed, y_train, y_test


def ofs(X,y): # other feature selection algorithm


    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)


    # scale std
    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.
    X_train_std = scaler.transform(X_train)
    X_test_std  = scaler.transform(X_test)


    #
    # X, y = load_iris(return_X_y=True)
    # X.shape

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train_std, y_train)     # clf = clf.fit(X, y)
    print(clf.feature_importances_, clf.feature_importances_.argsort())

    first = clf.feature_importances_.argsort()[-1:]  #6
    second = clf.feature_importances_.argsort()[-2:-1] #1
    #    [0.1481483  0.19205667 0.08683144 0.11135282 0.13380407 0.07531707
    # 0.25248963] [5 2 3 4 0 1 6]

    ofs_std = SelectFromModel(clf, prefit=True)  # <<<  ***
    X_train_std_transformed = ofs_std.transform(X_train_std)
    X_test_std_transformed  = ofs_std.transform(X_test_std)



    # visualize standardized vs. untouched dataset with PCA performed
    FIG_SIZE = (10, 7)
    fig, (ax2) = plt.subplots(ncols=1, figsize=FIG_SIZE)


    target_classes = range(0, 2)
    colors = ("blue", "red")
    markers = ("s", "o")

    for target_class, color, marker in zip(target_classes, colors, markers):
        # x = X_train_std[y_train == target_class, first]  # 0],
        # y = X_train_std[y_train == target_class, second]  # 1],
        ax2.scatter(
            x=X_train_std[y_train == target_class, first], #0],
            y=X_train_std[y_train == target_class, second], #1],
            color=color,
            label=f"class {target_class}",
            alpha=0.5,
            marker=marker,
        )

    ax2.set_title("Standardized training dataset after RP")
    ax2.set_xlabel("1st principal component")
    ax2.set_ylabel("2nd principal component")
    ax2.legend(loc="upper right")
    ax2.grid()

    plt.tight_layout()
    plt.savefig('Raisin_OFS_.png')  # save plot
    # plt.show()


    return X_train_std_transformed, X_test_std_transformed, y_train, y_test




def pca_full(X):

    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X)  # Apply transform to both the training set and the test set.
    X_std = scaler.transform(X)

    pca_std = PCA(0.95)
    pca_std.fit(X_std)
    X_std_transformed = pca_std.transform(X_std)  # 3 components out

    return X_std_transformed


def ica_full(X):

    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X)  # Apply transform to both the training set and the test set.
    X_std = scaler.transform(X)

    ica_std = FastICA(n_components=3)
    # ica_std = FastICA()
    ica_std.fit(X_std)
    X_std_transformed = ica_std.transform(X_std)

    return X_std_transformed


def rp_full(X): # Randomized Projections

    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X)  # Apply transform to both the training set and the test set.
    X_std = scaler.transform(X)

    rp_std = GaussianRandomProjection(n_components=3)
    rp_std.fit(X_std)
    X_std_transformed = rp_std.transform(X_std)  # Out Dime > 7 (~167) Error on eps [0.0 - 0.999]

    return X_std_transformed

def ofs_full(X,y): # other feature selection algorithm

    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X)  # Apply transform to both the training set and the test set.
    X_std = scaler.transform(X)

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_std,y)

    ofs_std = SelectFromModel(clf, prefit=True)  # <<<  ***
    X_std_transformed = ofs_std.transform(X_std) # 4 components out

    return X_std_transformed


def neural_network(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3240)

    # Grid Search
    # # simplefilter("ignore", category=ConvergenceWarning)
    # # parameter_list = {'hidden_layer_sizes': [(2,5),(5,5),(8,5),(15,5), (20,5), (30,5)],
    # parameter_list = {'hidden_layer_sizes': [(15,5)],
    #                 # 'learning_rate_init'  : [0.002] ,
    #                 'learning_rate_init'  : [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1] ,
    #                 'max_iter'  : range(200,500,100)
    #                   }
    # grid = GridSearchCV(estimator=MLPClassifier(), param_grid = parameter_list, cv =8, n_jobs=2)   #<<<<         <<<<
    # grid.fit(X_train, y_train)
    # print(" Results from Grid Search " )
    # print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
    # print("\n The best score across ALL searched params:\n",grid.best_score_)
    # print("\n The best parameters across ALL searched params:\n",grid.best_params_)


    learner_name = "Neural Network"
    print(learner_name)
    learner = MLPClassifier(hidden_layer_sizes=(15,5), learning_rate_init=0.0001, max_iter=200)

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()

    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_loss=log_loss(y_test, y_pred)
    print("Accuracy",test_accuracy)
    print("Loss",test_loss)
    print('\n')

    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Hidden Layers'                                 # <<
    param="hidden_layer_sizes"                                      # <<
    param_range = [(2,5),(5,5),(8,5),(15,5), (20,5), (30,5)]                             # <<
    # param_range = [(2,10),(5,10),(8,10),(15,10), (20,10,10), (30,10), (40,10)]# (50,10), (60,10)]                             # <<
    scoring_metric = 'accuracy'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=MLPClassifier(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=8,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})
    # df = df.rename(columns={'rmsein': 'In Sample Data', 'rmseout': 'Out of Sample Data'})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols           # <<
    # plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))  # 3 decimal places
    plt.grid(True)
    plt.savefig('1Chart_nn_VC1.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Initial Learning Rate'                               # <<
    param="learning_rate_init"                                    # <<
    param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1]# 0.15, 0.2, 0.25, 0.3, 0.4]  # <<
    scoring_metric = 'accuracy'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=MLPClassifier(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=8,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='line',logx=True) # Positions of cols      # <<
    # plt.xticks(param_range, param_range)
    # plt.xticks(rotation='horizontal')
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.7f}'))  # 3 decimal places
    plt.grid(True)
    plt.savefig('1Chart_nn_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #3     # Loss Curve
    vc3_name = 'Iterations'                               # <<
    param="max_iter"                                    # <<
    param_range = range(25,350,50)                  # <<
    scoring_metric = 'neg_log_loss'  # // accuracy, precision
    train_scores_vc3, test_scores_vc3 = \
        validation_curve(estimator=MLPClassifier(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=8,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc3 = np.mean(train_scores_vc3, axis=1)
    test_scores_mean_vc3 = np.mean(test_scores_vc3, axis=1)
    df = pd.DataFrame({vc3_name: param_range, 'Training Score': train_scores_mean_vc3, 'Testing Score': test_scores_mean_vc3})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols      # <<
    # plt.xticks(param_range, param_range)
    # plt.xticks(rotation='horizontal')
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.ylabel('Log loss (-ve)')
    plt.suptitle("Validation Curve"+': '+learner_name)
    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))  # 3 decimal places
    plt.grid(True)
    plt.savefig('1Chart_nn_VC3.png')     # save plot                      # <<
    # plt.show()
    plt.close()



    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'accuracy'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=MLPClassifier(hidden_layer_sizes=(15,5), learning_rate_init=0.001, max_iter=200),                # <<
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=8,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    # train_scores_std = np.std(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    # test_scores_std = np.std(test_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)
    # fit_times_std = np.std(fit_times_lc, axis=1)
    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})
    # df = df.rename(columns={'rmsein': 'In Sample Data', 'rmseout': 'Out of Sample Data'})


    df.plot(x=0, y=[1,2]) # Positions of cols
    # plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('1Chart_nn_LC.png')     # save plot
    # plt.show()
    plt.close()




def main():

    #################################################################################
    # Data set #1
    #################################################################################
    # Area	MajorAxisLength	MinorAxisLength	Eccentricity	ConvexArea	Extent	Perimeter	Class

    raisin_df = pd.read_csv(path.join('data','Raisin_Dataset.csv'))
    raisin_df.rename(columns={'Class':'Class_category'}, inplace=True)
    dataset = 1

    le = preprocessing.LabelEncoder()
    raisin_df['Class'] = le.fit_transform(raisin_df.Class_category)

    y = raisin_df['Class']
    X = raisin_df.drop(['Class', 'Class_category'], axis=1)

    # Step 1 ##################
    # Clustering
    km(X)  # k-means clustering
    # print('em:')
    em(X)  # Expectation Maximization


    # Step 2 ##################
    # Dimentionality reduction
    pca(X,y)
    ica(X,y)
    rp(X,y)  # Randomized Projections
    ofs(X,y)  # other feature selection algorithm


    # # pca_full(X)
    # # ica_full(X)
    # # rp_full(X)
    # # ofs_full(X,y)


    # # Step 3 ##################
    km(pca_full(X))
    km(ica_full(X))
    km(rp_full(X))
    km(ofs_full(X,y))

    em(pca_full(X))
    em(ica_full(X))
    em(rp_full(X))
    em(ofs_full(X,y))


    # Step 4 #########
    # NN on Dim red raisin from Step 2
    # Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
    # and rerun your neural network learner on the newly projected data.
    neural_network(X,y)
    neural_network(pca_full(X),y)
    neural_network(ica_full(X),y)
    neural_network(rp_full(X),y)
    neural_network(ofs_full(X,y),y)


    # Step 5 #########
    # NN on Dim red. clustered raisin
    # Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms
    # ), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if
    # they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.


    # #cluster count:   # pca 2    # ica 4    # rp 2    # ofs 2
    ret = km(pca_full(X),2)
    ret = km(ica_full(X),4)
    ret =  km(rp_full(X),2)
    ret =  km(ofs_full(X,y),2)

    ret = em(pca_full(X),2)
    ret = em(ica_full(X),4)
    ret = em(rp_full(X),2)
    ret = em(ofs_full(X,y),4)

    # Add the cluster label as a new projection/dimension.
    ret1 = pd.Series(ret)
    XCL = pd.concat([X, ret1], axis=1)

    neural_network(XCL,y)





if __name__ == "__main__":
    main()

