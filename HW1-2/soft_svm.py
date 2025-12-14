from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})  # safer for Windows
# rc('text', usetex=True)   # enable only if you have LaTeX

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.svm import SVC

# Reproducibility
np.random.seed(22)

# Generate 2 Gaussian clusters
means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

# Build dataset
X = np.vstack((X0, X1))
y = np.hstack((np.zeros(N), np.ones(N)))

# Train linear SVM
C = 10
clf = SVC(kernel='linear', C=C)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

print("Weights:", w)
print("Bias:", b)

# Create plot
with PdfPages("data.pdf") as pdf:
    plt.figure()
    plt.scatter(X0[:, 0], X0[:, 1], c='b', marker='s', label='Class 0')
    plt.scatter(X1[:, 0], X1[:, 1], c='r', marker='o', label='Class 1')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=120, facecolors='none', edgecolors='k', label='Support Vectors')

    # Fix axes
    plt.xlim(0, 5)
    plt.ylim(0, 4)

    # Decision boundary
    xx = np.linspace(0, 5, 50)
    yy = -(w[0] * xx + b) / w[1]
    yy_down = -(w[0] * xx + b - 1) / w[1]
    yy_up   = -(w[0] * xx + b + 1) / w[1]

    plt.plot(xx, yy, 'k-')      # decision boundary
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.xlabel('$x_1$', fontsize=20)
    plt.ylabel('$x_2$', fontsize=20)
    plt.legend()
    plt.tight_layout()

    pdf.savefig()  # save into PDF
    plt.show()
