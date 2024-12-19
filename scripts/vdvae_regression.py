import sys
import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import KFold
import argparse
import pickle

# Argument parser
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1, 2, 5, 7]

# Load data
nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
train_latents = nsd_features['train_latents']
test_latents = nsd_features['test_latents']

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub, sub)
test_fmri = np.load(test_path)

train_fmri = train_fmri[:train_latents.shape[0]]
test_fmri = test_fmri[:test_latents.shape[0]]
print(train_fmri.shape[0])
print(train_latents.shape[0])
print(test_fmri.shape[0])
print(test_latents.shape[0])
# Ensure dimensions match
assert train_fmri.shape[0] == train_latents.shape[0], "Mismatch in train samples!"
assert test_fmri.shape[0] == test_latents.shape[0], "Mismatch in test samples!"
# Preprocess fMRI
train_fmri = train_fmri / 300
test_fmri = test_fmri / 300

norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

# Configurations
num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)
num_targets = train_latents.shape[1]
num_splits = 5  # Cross-validation splits
lambda_candidates = [1e4, 5e4, 1e5]  # Candidate hyperparameters
num_jobs = 10 # Number of parallel jobs

# Function to train and evaluate a single sub-problem
def train_subproblem(X, Y, lambdas, num_splits):
    best_lambda = None
    best_score = -np.inf
    best_model = None

    kf = KFold(n_splits=num_splits)
    for lam in lambdas:
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Ridge regression
            reg = skl.Ridge(alpha=lam, max_iter=10000, fit_intercept=True)
            reg.fit(X_train, Y_train)

            # Evaluate on validation set
            score = reg.score(X_val, Y_val)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam
            best_model = skl.Ridge(alpha=best_lambda, max_iter=10000, fit_intercept=True)
            best_model.fit(X, Y)

    return best_model, best_lambda

# Main B-MOR process
n_subproblems = min(num_targets, num_jobs)
sub_problem_size = num_targets // n_subproblems
models = []
for i in range(n_subproblems):
    start_idx = i * sub_problem_size
    end_idx = (i + 1) * sub_problem_size if i != n_subproblems - 1 else num_targets
    Y_sub = train_latents[:, start_idx:end_idx]

    print(f"Training subproblem {i + 1}/{n_subproblems}...")
    model, best_lambda = train_subproblem(train_fmri, Y_sub, lambda_candidates, num_splits)
    models.append(model)
    print(f"Best lambda for subproblem {i + 1}: {best_lambda}")

# Make predictions for all sub-problems
pred_latents = np.hstack([model.predict(test_fmri) for model in models])

# Save predictions
np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_bmor.npy'.format(sub, sub), pred_latents)

# Save regression weights
weights = [model.coef_ for model in models]
biases = [model.intercept_ for model in models]
datadict = {
    'weights': weights,
    'biases': biases,
}
with open('data/regression_weights/subj{:02d}/vdvae_regression_weights_bmor.pkl'.format(sub), "wb") as f:
    pickle.dump(datadict, f)
