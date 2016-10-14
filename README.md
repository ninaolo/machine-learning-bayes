# Bayesian Learning and Boosting

## Assignment 1
Run `test_ml_estimates()`.

## Assignment 2
Look at `compute_prior(labels)` and `classifyBayes(X, prior, mu, sigma)`.

## Assignment 3
Run `test_bayes_iris()` and `test_bayes_vowels()`.

**Q1: When can a feature independence assumption be reasonable and when not?**

**A1:** It is reasonable when the features are conditionally independent given classification, or at least reasonably independent, still works with little dependence. However if features a strictly correlated and not cancel each other out, NB assumption do not perform as good. Naïve Bayes can perform surprisingly well even on complex tasks where it is clear that the strong independence assumptions are false. No matter how strong the dependencies among attributes are, naive Bayes can still be optimal if the they distribute evenly in classes, or cancel each other out.

**Q2: How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classifier or, alternatively, manipulating the data?**

**A2:** The decision boundary could become more complex (see figure). 

## Assignment 4
Look at `ml_params(X, labels, W)`.

## Assignment 5
Run `test_boost_iris()` and `test_boost_vowels()`.

**Q1: Is there any improvement in classification accuracy? Why/why not?**

**A1:** Yes. Iris: 89 --> 94.2. Vowels: 64.7 --> 77.1. The reason is that boosting has taken multiple weak learners and combined their predictions into a strong learner. Also, boosting reduces variance and this was the case since the std.dev was reduced in both cases (and std.dev = sqrt(var)).

**Q2: Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

**A2:** Yes, it is more complex. It first looks like a parabola, but after combining multiple classifiers it looses the parabola shape and becomes more complex.

**Q3: Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

**A3:**


## Assignment 6
Run `test_tree_boost_iris()` and `test_tree_boost_vowels()`.

## Assignment 7
**Q1: If you had to pick a classifier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria:**

**Bayes**
* **Outliers**
* **Irrelevant inputs: part of the feature space is irrelevant**
* **Predictive power -->** NB classifiers estimate badly, but often classify well.
* **Mixed types of data: binary, categorical or continuous features, etc.**
* **Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both. -->** NB's main strength is its efficiency: Training and classification can be accomplished with one pass over the data. 

**A1:**


