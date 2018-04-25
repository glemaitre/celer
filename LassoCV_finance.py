import os
import time
import numpy as np

from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV as LassoCVsklearn
from celer.examples_utils import download_preprocess_finance

from celer import LassoCV

print(__doc__)

print("Loading data...")
X_fname = "./data/finance_data_preprocessed.npz"
y_fname = "./data/finance_target_preprocessed.npy"

if not os.path.exists(X_fname):
    print("Downloading and preprocessing the Finance dataset...")
    print("*** Warning: This may take several minutes ***")
    download_preprocess_finance()

X = sparse.load_npz(X_fname)
X.sort_indices()
y = np.load(y_fname)

X_train, X_test, y_train, y_test = train_test_split(X, y)

n_samples = len(y)
params = dict(eps=1e-1, cv=2, verbose=1, n_alphas=10)

t0 = time.time()
model = LassoCV(**params)
model.fit(X_train, y_train)
print("Estimated regularization parameter alpha: %s" % model.alpha_)
t_me = time.time() - t0
print("me: %s s" % t_me)

y_pred_me = model.predict(X_test)
mse_me = mean_squared_error(y_test, y_pred_me)
print("mse celer %s" % mse_me)

t0 = time.time()
model_sklearn = LassoCVsklearn(**params)
model_sklearn.fit(X_train, y_train)
print("Estimated regularization parameter alpha: %s" % model_sklearn.alpha_)
t_sklearn = time.time() - t0
print("sklearn: %s s" % t_sklearn)

y_pred_sklearn = model_sklearn.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print("mse sklearn %s" % mse_sklearn)
