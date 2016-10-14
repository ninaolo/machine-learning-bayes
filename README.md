# Bayesian Learning and Boosting

## Assignment 1
Run `test_ml_estimates()`.

## Assignment 2
Look at `compute_prior(labels)` and `classifyBayes(X, prior, mu, sigma)`.

## Assignment 3
Run `test_bayes_iris()` and `test_bayes_vowels()`.

**Q1: When can a feature independence assumption be reasonable and when not?**

**A1:** Naïve Bayes can perform surprisingly well even on complex tasks where it is clear that the strong independence assumptions are false. No matter how strong the dependencies among attributes are, naive Bayes can still be optimal if the they distribute evenly in classes, or cancel each other out.

**Q2: How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classifier or, alternatively, manipulating the data?**

**A2:**

## Assignment 4
Look at `ml_params(X, labels, W)`.

## Assignment 5
Run `test_boost_iris()` and `test_boost_vowels()`.

**Q1: Is there any improvement in classification accuracy? Why/why not?**

**A1:**

**Q2: Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?**

**A2:**

**Q3: Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?**

**A3:**


## Assignment 6
Run `test_tree_boost_iris()` and `test_tree_boost_vowels()`.

## Assignment 7
**Q1: If you had to pick a classifier, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria:**

**Bayes**
* **Outliers**
* **Irrelevant inputs: part of the feature space is irrelevant**
* **Predictive power --> **NB classifiers estimate badly, but often classify well.
* **Mixed types of data: binary, categorical or continuous features, etc.**
* **Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both. --> **NB's main strength is its efficiency: Training and classification can be accomplished with one pass over the data. 

**A1:**


