import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Data points
X = np.array([[-1, 3], [0, 2], [0, 1], [0, 0], [-2, 0], [-2, 1], [-2, 3], 
              [-1, 0], [-1, 1], [0, 0], [1, 5], [1, 6], [3, 3]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

# Fit model to data points
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# Find separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

print("Value of w vector: ")
print(w)
print("Value of intercept")
print(clf.intercept_[0])

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# Plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
