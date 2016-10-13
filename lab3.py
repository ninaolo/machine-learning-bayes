#!/usr/bin/python
# coding: utf-8

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def compute_prior(labels, weights=None):
    n_pts = labels.shape[0]
    if weights is None:
        weights = np.ones((n_pts, 1)) / n_pts
    else:
        assert (weights.shape[0] == n_pts)
    unique_labels = np.unique(labels)

    prior = np.zeros((np.size(unique_labels), 1))

    # TODO: compute the values of prior for each class!

    sum_of_all_weights = np.sum(weights)

    for k, label in enumerate(unique_labels):
        label_count = 0
        for i in range(len(labels)):
            if labels[i] == label:
                label_count += weights[i]
        prior[k] = float(label_count / sum_of_all_weights)

    return prior


# in:      X - N x d matrix of N data points
#     labels - N vector of class labels¢
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def ml_params_old(X, labels, weights=None):
    assert (X.shape[0] == labels.shape[0])
    n_pts, n_dims = np.shape(X)
    unique_labels = np.unique(labels)
    n_classes = np.size(unique_labels)

    if weights is None:
        weights = np.ones((n_pts, 1)) / float(n_pts)

    mu = np.zeros((n_classes, n_dims))
    sigma = np.zeros((n_classes, n_dims, n_dims))

    # Computation of mu
    for k, label in enumerate(unique_labels):
        all_x_for_label = logical_indexing(X, labels, label)
        x_sum = np.zeros(n_dims)

        for xi in all_x_for_label:
            x_sum += xi

        mu[k] = x_sum / float(number_of_rows(all_x_for_label))

    # Computation of sigma
    for k, label in enumerate(unique_labels):
        all_x_for_label = logical_indexing(X, labels, label)

        # This is the sum part.
        for m in range(n_dims):
            v_temp = 0
            for xi in all_x_for_label:
                v_temp += np.power(xi[m] - mu[k][m], 2)

            sigma[k][m][m] = float(v_temp / number_of_rows(all_x_for_label))

    return mu, sigma


# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def ml_params_new(X, labels, weights=None):
    assert (X.shape[0] == labels.shape[0])
    n_pts, n_dims = np.shape(X)
    unique_labels = np.unique(labels)
    n_classes = np.size(unique_labels)

    if weights is None:
        weights = np.ones((n_pts, 1)) / float(n_pts)

    mu = np.zeros((n_classes, n_dims))
    sigma = np.zeros((n_classes, n_dims, n_dims))

    # Computation of mu with weights.
    for k, label in enumerate(unique_labels):
        wi_xi_sum = 0
        wi_sum = 0
        for i in range(len(labels)):
            if labels[i] == k:
                wi_xi_sum += X[i] * weights[i]
                wi_sum += weights[i]
        mu[k] = wi_xi_sum / float(wi_sum)

    # Computation of sigma with weights.
    for k, label in enumerate(unique_labels):

        # This computes the diagonal - sigma(m, m).
        for m in range(n_dims):
            v_temp = 0
            wi_sum = 0
            for i in range(len(labels)):
                if labels[i] == k:
                    v_temp += np.power(X[i][m] - mu[k][m], 2) * weights[i]
                    wi_sum += weights[i]

            sigma[k][m][m] = float(v_temp / wi_sum)

    return mu, sigma


def number_of_rows(X):
    return X.shape[0]


def number_of_columns(X):
    return X.shape[1]


# in: x_stars - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classify_bayes(x_stars, prior, mu, sigma):
    n_pts = number_of_rows(x_stars)
    n_classes, n_dims = np.shape(mu)
    log_prob = np.zeros((n_classes, n_pts))

    # TODO: fill in the code to compute the log posterior logProb!

    # Go through all unseen vectors x_star to be classified.
    for k in range(n_classes):
        sigma_det = np.linalg.det(sigma[k])
        sigma_inv = np.linalg.inv(sigma[k])
        for i, x_star in enumerate(x_stars):
            v = (x_star - mu[k])
            v_trans = np.transpose(v)
            log_prob[k][i] = - 0.5 * np.log(sigma_det) - 0.5 * np.dot(np.dot(v, sigma_inv), v_trans) + np.log(prior[k])

    # The max a posteriori probability.
    h = np.argmax(log_prob, axis=0)
    return h


class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def train_classifier(self, X, labels, weights=None):
        rtn = BayesClassifier()
        rtn.prior = compute_prior(labels, weights)
        rtn.mu, rtn.sigma = ml_params_new(X, labels, weights)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classify_bayes(X, self.prior, self.mu, self.sigma)


def test_ml_estimates():
    X, labels = genBlobs(centers=5)
    mu, sigma = ml_params_new(X, labels)
    plotGaussian(X, labels, mu, sigma)


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#          iterations - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def train_boost(base_classifier, X, labels, iterations=10):
    n_pts, n_dims = np.shape(X)

    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    current_weights = np.ones((n_pts, 1)) / float(n_pts)

    for t in range(0, iterations):
        # A new classifier can be trained like this, given the current weights.
        classifiers.append(base_classifier.train_classifier(X, labels, current_weights))

        # Do classification for each point.
        vote_t = classifiers[-1].classify(X)

        # Compute the error for iteration t.
        error_t = 0.00001  # If we have 0 here, some alphas will be inf.
        for i in range(n_pts):
            if vote_t[i] != labels[i]:
                error_t += current_weights[i]

        # Compute alpha for iteration t.
        alpha_t = 0.5 * (np.log(1 - error_t) - np.log(error_t))
        alphas.append(alpha_t)

        # Update the weights for next iteration, t + 1.
        for i in range(n_pts):
            factor = np.exp(-alpha_t) if vote_t[i] == labels[i] else np.exp(alpha_t)
            current_weights[i] *= factor
        current_weights / np.sum(current_weights)

    return classifiers, alphas


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    n_labels - the number of different classes
# out:  y_pred - N vector of class predictions for test points
def classify_boost(X, classifiers, alphas, n_labels):
    n_pts = X.shape[0]
    n_classifiers = len(classifiers)

    # If we only have one classifier, we may just classify directly.
    if n_classifiers == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((n_pts, n_labels))

        for t in range(n_classifiers):
            h = classifiers[t].classify(X)
            for i, xi_label in enumerate(h):
                votes[i][xi_label] += alphas[t]

        return np.argmax(votes, axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def train_classifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = train_boost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classify_boost(X, self.classifiers, self.alphas, self.nbr_classes)


def logical_indexing(X, labels, wanted_label):
    """ Gets the row vectors in X corresponding to the wanted label. """
    return X[labels == wanted_label, :]


def test_bayes_iris():
    testClassifier(BayesClassifier(), dataset='iris', split=0.7)
    plotBoundary(BayesClassifier(), dataset='iris', split=0.7)


def test_bayes_vowel():
    testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
    plotBoundary(BayesClassifier(), dataset='vowel', split=0.7)


def test_boost_iris():
    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris', split=0.7)


def test_boost_vowel():
    testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel', split=0.7)


def test_tree_iris():
    testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris', split=0.7)


def test_tree_boost_iris():
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
    plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris', split=0.7)


def test_tree_vowel():
    testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel', split=0.7)


def test_tree_boost_vowel():
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
    plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel', split=0.7)


test_tree_boost_iris()