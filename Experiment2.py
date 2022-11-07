# ## Python 3.8
# ############################33
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
# from sklearn.preprocessing import OneHotEncoder

#############################33
# import mlrose_hiive
import numpy as np
import pandas as pd
# from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator, FlipFlopGenerator, KnapsackGenerator,ContinuousPeaksGenerator
# from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner
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
    plt.savefig('Customer_KM_Elbow_dis.png')  # save plot
    plt.close()

    plt.plot(range(1, 11), inertias, 'og-')
    plt.xlabel('The number of clusters K')
    plt.ylabel('Inertia')
    plt.title('The Elbow plot analysis for the various clusters using Inertia')
    # plt.show()
    plt.savefig('Customer_KM_Elbow_in.png')  # save plot



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

        # # 2nd Plot showing the actual clusters formed
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax2.scatter(
        #     X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        # )
        #
        # # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # # Draw white circles at cluster centers
        # ax2.scatter(
        #     centers[:, 0],
        #     centers[:, 1],
        #     marker="o",
        #     c="white",
        #     alpha=1,
        #     s=200,
        #     edgecolor="k",
        # )
        #
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        #
        # ax2.set_title("The visualization of the clustered data.")
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering with n_clusters = %d"
            # "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            # fontsize=14,
            # fontweight="bold",
        )
        # plt.show()
        plt.savefig('Customer_KM_Silhouette_'+ str(n_clusters) +'.png')  # save plot

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
        plt.savefig('Customer_EM_Silhouette_'+ str(n_clusters) +'.png')  # save plot




def pca(X):

    X_train =X
    # X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1 / 7.0, random_state=0)

    # X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)



    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.

    # X_train = scaler.transform(X_train)
    # X_test  = scaler.transform(X_test)


    # scale std
    X_train_std = scaler.transform(X_train)
    # X_test_std  = scaler.transform(X_test)

    # Note:
    ######## A pca 0.95 on unscaled data reduced to 1 dimension, while
    ## standardized data reduced to 3 dimensions from a total of 8.

    # pca     = PCA(.95)        #<<<<<<<<<<<<<< <<<<<<<<<<<<
    # pca_std = PCA(.95)
    pca     = PCA(11)
    pca_std = PCA(11)

    pca.fit(X_train)
    pca_std.fit(X_train_std)

    # Show first principal components
    print(f"\nPC 1 without scaling:\n{pca.components_[0]}")
    print(f"\nPC 1 with scaling:\n{pca_std.components_[0]}")

    # Unscaled
    X_train_transformed  = pca.transform(X_train)
    # X_test               = pca.transform(X_test)

    # Standard scaled
    X_train_std_transformed = pca_std.transform(X_train_std)
    # X_test_std_transformed  = pca_std.transform(X_test_std)


    # Eigen values
    print('Explained variance ratio:',pca_std.explained_variance_ratio_)
    cummulative = np.cumsum(pca_std.explained_variance_ratio_)
    # plt.plot(range(1, 12), pca_std.explained_variance_ratio_, 'og-')
    plt.plot(range(1, 12), cummulative, 'og-')
    plt.xlabel('n-th Principal Component')
    # plt.ylabel('Explained variance ratio ')
    plt.ylabel('Explained variance ratio (cummulative)')
    # plt.ylabel('Explained variance ratio')
    plt.title('The Eigen value analysis of components generated by PCA')
    # plt.show()
    plt.savefig('Customer_PCA_Eigen_.png')  # save plot
    plt.close()



    return X_train_std_transformed


def ica(X):

    X_train = X
    # X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=3240)

    # scale std
    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.
    X_train_std = scaler.transform(X_train)
    # X_test_std  = scaler.transform(X_test)


    # ica_std = FastICA(n_components=2)
    ica_std = FastICA(n_components=11)
    ica_std.fit(X_train_std)
    X_train_std_transformed = ica_std.transform(X_train_std)
    # X_test_std_transformed  = ica_std.transform(X_test_std)
    A_ = ica_std.mixing_  # Get estimated mixing matrix


    # Kurtosis
    from scipy.stats import norm, kurtosis
    kurt_orignal = kurtosis(X_train_std, fisher=True)
    kurt = kurtosis(X_train_std_transformed, fisher=True)

    # tips = sns.load_dataset("tips")
    # sns.kdeplot(data=tips, x="total_bill")

    print('ICA Kurtosis original:',kurt_orignal)
    print('ICA Kurtosis:',kurt)

    plt.plot(range(0, 11), kurt_orignal, 'og-')
    plt.plot(range(0, 11), kurt, 'xb-')
    ax = plt.gca()
    ax.legend(['Original', 'ICA Tranformed'])
    plt.xlabel('Component')
    plt.ylabel('Kurtosis')
    plt.title('The Kurtosis of components generated by ICA')
    # plt.show()
    plt.savefig('Customer_ICA_kurtosis.png')
    plt.close()




    return X_train_std_transformed

def rp(X): # Randomized Projections

    X_train =X


    # scale std
    scaler = StandardScaler()  # Fit on training set only.
    scaler.fit(X_train)  # Apply transform to both the training set and the test set.
    X_train_std = scaler.transform(X_train)
    # X_test_std  = scaler.transform(X_test)


    #############################33
    rmse_all=[]
    rmse_all.append(0.0) # Dummy val to shift
    for n_component in range(1, 12):

        # rp_std = GaussianRandomProjection(n_components=2)
        rp_std = GaussianRandomProjection(n_components=n_component,compute_inverse_components=True)
        rp_std.fit(X_train_std)
        X_train_std_transformed = rp_std.transform(X_train_std)
        X_train_std_transformed_inversed = rp_std.inverse_transform(X_train_std_transformed)

        # rmse_comp = mean_squared_error(X_train_std, X_train_std_transformed_inversed, multioutput='raw_values',squared=False)
        rmse = mean_squared_error(X_train_std, X_train_std_transformed_inversed, squared=False)

        rmse_all.append(rmse)
    print('RMSE All:',rmse_all)
    plt.plot(range(1, 12), rmse_all[1:], 'og-')
    ax = plt.gca()
    plt.xlabel('Components')
    plt.ylabel('RMSE')
    plt.title('The RMSE of orignal vs inverse of feature generated by RP')
    # plt.show()
    plt.savefig('Customer_RP_rmse.png')
    plt.close()

    # X_train_std_transformed_again = rp_std.transform(X_train_std_transformed_inversed)
    # # test1 = np.allclose(X_train_std_transformed, X_train_std_transformed_again)
    # # # test = np.allclose(X_train_std, X_train_std_transformed_inversed)
    #############################33




    return X_train_std_transformed


def ofs(X): # other feature selection algorithm


    y = X['Channel1']
    X = X.drop(['Channel1'], axis=1)
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
    plt.savefig('Customer_OFS_.png')  # save plot
    # plt.show()


    return X_train_std_transformed, X_test_std_transformed, y_train, y_test



#######################################
#######################################
#######################################
#######################################

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



def main():

    #################################################################################
    # Data set #2
    #################################################################################
    # Channel	Region	Fresh	Milk	Grocery	Frozen	Detergents_Paper	Delicassen

    customer_df = pd.read_csv(path.join('data','Wholesale_customers_data.csv'))
    # customer_df.rename(columns={'Class':'Class_category'}, inplace=True)
    dataset = 2


    enc = OneHotEncoder(handle_unknown='ignore')

    # Channel
    enc_df = pd.DataFrame(enc.fit_transform(customer_df[['Channel']]).toarray())
    enc_df.rename(columns={0: 'Channel1', 1: 'Channel2'}, inplace=True)
    customer_df = pd.concat([customer_df, enc_df], axis=1)

    # Region
    enc_df = pd.DataFrame(enc.fit_transform(customer_df[['Region']]).toarray())
    enc_df.rename(columns={0: 'Region1', 1: 'Region2', 2: 'Region3'}, inplace=True)
    customer_df = pd.concat([customer_df, enc_df], axis=1)

    X = customer_df.drop(['Channel', 'Region'], axis=1)




    # Step 1 ##################
    # Clustering
    km(X)  # k-means clustering
    # print('em:')
    em(X)  # Expectation Maximization


    # Step 2 ##################
    # Dimentionality reduction

    pca(X)
    ica(X)
    rp(X)  # Randomized Projections
    ofs(X)  # other feature selection algorithm


    # pca_full(X)
    # # ica_full(X)
    # rp_full(X)
    # # ofs_full(X,y)


    # # Step 3 ##################
    km(pca_full(X))
    km(ica_full(X))
    km(rp_full(X))
    km(ofs_full(X))

    em(pca_full(X))
    em(ica_full(X))
    em(rp_full(X))
    em(ofs_full(X))









if __name__ == "__main__":
    main()

