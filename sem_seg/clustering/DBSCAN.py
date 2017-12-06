import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

print (type(X[0]))
print (X.shape)
#print (X)



# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

bbox_co = {}
for k in unique_labels:
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    #print (xy.shape)
    if len(xy) > 2:
        bbox_co[k] = [[int(xy[0][0]), int(xy[1][0])], [int(xy[0][1]), int(xy[1][1])]]

#print (bbox_co)


for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]


    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]

    print (type(xy))
    print (xy.shape)

    if k != -1:
        xy_transpose = xy.reshape(xy.shape[1], xy.shape[0])
        bbox_co[k][0][0] = min(xy_transpose[0])
        bbox_co[k][0][1] = max(xy_transpose[0])

        bbox_co[k][1][0] = min(xy_transpose[1])
        bbox_co[k][1][1] = max(xy_transpose[1])

    # for point in xy:
    #     print (point[0])
    #     print (bbox_co[k][0][0])
    #     print (bbox_co[k][0][1])
    #     print ("-----------------")
    #
    #     if int(point[0]) < bbox_co[k][0][0]:
    #         bbox_co[k][0][0] = point[0]
    #     elif int(point[0]) > bbox_co[k][0][1]:
    #         bbox_co[k][0][1] = point[0]
    #
    #
    #
    #     if int(point[1]) < bbox_co[k][1][0]:
    #         bbox_co[k][1][0] = point[1]
    #     elif int(point[1]) > bbox_co[k][1][1]:
    #         bbox_co[k][1][1] = point[1]

    #plt.plot(xy[:, 0], xy[:, 1], 'o')

    # xy = X[class_member_mask & ~core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #          markeredgecolor='k', markersize=6)

print (bbox_co)

def backtracking_co(list):
    res = [[], []]
    for x in list[0]:
        for y in list[1]:
                res[0].append(x)
                res[1].append(y)
    return res

bbox_co_final = {}
for k,v in bbox_co.items():
    bbox_co_final[k] = backtracking_co(v)
print (bbox_co_final)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()


