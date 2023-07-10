# essential knowledge
- type 1 error: reject a true null hypothesis
- type 2 error: fail reject a false null hypothesis
- p-value: probability of obtaining a sample as extreme as or more extreme that the sample we have obtained, assuming H0 is true
- beta_1 of logistic: when x increases by one unit, the log odds increases by beta_1 units
- in AWS (S3, RDS, EC2):
	- S3: Simple Storage Service 
		- for storing model output
			- evaluation metrics
			- log file
			- figures
			- excel, xlsx files
				- feature importance
	- RDS: Relational Database Service (RDS)
		- for storing database 
		- we used Postgresql at MM
		- use `pd.read_sql(con, sql, chunksize)` to read data
			- but before this, need to get the connection
		- use `pd.to_sql(name, con, schema)` to write data to RDS
	- EC2 (Elastic Compute Cloud)
		- allows users to rent virtual computers 
		- we as data scientist were given limited access level. That means we can't change the configure of each virtual machine
	- we use Jenkins in combination with EC2, to track the progress of each build of our model
		- each time, load repos from github
		- Jenkins is used today along the entire software development lifecycle, enabling the integration and automation of different processes, including building, testing, and deployment. Jenkins creates ‘pipelines’, which define the series of steps that the server will take to perform the required tasks.
	- sagemaker
		- tried it, but don't like it
		- they have specialized xgboost package, but it's just a thin wrapper of the original xgboost
- important git commands
	- `git init`
	- `git clone`
	- `git branch`
		- list all branches or create new branch
	- `git checkout branch`
		- change to another branch
	- `git log`
	- `git status`
	- `git add -A`
	- `git commit -m <>`
	- `git pull origin main`
	- `git push origin main`
# behavioral questions

# programming questions
- inheritance = a class is derived from another class called parent class 
- decorator
	- three kinds of OOP decorator
		- @classmethod
			- takes class as input. Access class-level variables
		- @property
			- access method like an attribute
		- @staticmethod
			- like a normal function outside of the class. call like `class.method(x,y)`, where x and y may not be objects of the class
			- do not take `cls` as first input
	- @functools.cache decorator to turn recursion into memoization
- difference between class method and static method
	- class method: 
		- use `@classmethod` property
		- bound to the class, not object
		- receive the class itself as first argument
		- can access variables specific to the class
	- static method
		- use `staticmethod` property
		- also bound to the class, not object
		- do not receive class itself as first argument
		- can do class-independent stuff
- difference between list and tuple in python:
	- mutability: list is mutable, that means you can add, remove, or modify elements after the list is created. However, tuple is immutable
	- syntax: use `[]` for list, and `()` for tuple. Also, often we can omit `()` entirely. But if our tuple has one element only, we have to add a trailing comma
	- performance: Tuples are generally faster than lists for accessing elements, especially for large collections of data. This is because tuples are stored in a more compact format in memory.
- what is decorator in python
	- a function that takes another function/class as input
	- can modify and extend the behavior of the given function/class
	- e.g., record the running time of the given function
	- uses: logging, timing, caching, authentication
- What is the difference between a shallow copy and a deep copy in Python?
	- shallow copy: use `.copy`
	- deep copy: use `copy.deepcopy()`
	- if we use shallow copy of a list, whose elements are also lists, then each element of shallow copy and the corresponding element of the original will point to the same list. So, if that list changes, it will affect both the original and the copy
	- A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.
	- A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
	- when we do `t1 = t2`, it's pointing to the same memory location!
	- basically, for shallow copy, each entry of the copy is still pointing to the same object. So, if we modify an entry of the original, then the copy's entry will also be affected.


# ML questions
- how to detect data drift?
	- cross entropy
	- KL / KS: Statistical tests: Statistical tests can be used to compare the statistical properties of the training data and the test data. For example, the Kolmogorov-Smirnov test can be used to compare the distributions of two datasets. If the distributions are significantly different, it may be an indication of data drift.
	- Drift detection algorithms: There are several drift detection algorithms that can be used to detect data drift. These algorithms typically monitor the statistical properties of the data over time and raise an alert if significant changes are detected. Some popular drift detection algorithms include ADWIN, DDM, and EDDM.
	- Visualization: Data visualization techniques can be used to detect data drift. For example, you can plot the distribution of a particular feature over time and look for any significant changes. If the distribution starts to shift significantly, it may be an indication of data drift.
- What is ROC curve?
    - The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).
    - Since the ROC curve is so similar to the precision/recall (PR) curve, you may wonder how to decide which one to use. As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives. Otherwise, use the ROC curve. For example, looking at the previous ROC curve (and the ROC AUC score), you may think that the classifier is really good. But this is mostly because there are few positives (5s) compared to the negatives (non-5s). In contrast, the PR curve makes it clear that the classifier has room for improvement (the curve could be closer to the topleft corner).
- What’s the F1 score? When is it useful?
    - Answer: The F1 score is a measure of a model’s performance. It is the harmonic mean of the precision and recall of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would use it in classification tests where true negatives don’t matter much.
- How would you handle an imbalanced dataset?
    - An imbalanced dataset is when you have, for example, a classification test and 90% of the data is in one class. That leads to problems: an accuracy of 90% can be skewed if you have no predictive power on the other category of data! Here are a few tactics to get over the hump:
- what is bias and variance? What is bias-variance trade off? How to reduce bias? How to reduce variance?
    - we have a model with high train performance but low test performance. What problem are we facing?
- what is overfitting and underfitting? How to avoid them?
- Suppose we want to build a predictive model to predict whether a person is infected with COVID-19 or not. What metric would be most appropriate to evaluate our model performance?
	- AUC-ROC or AUC-PR
- we have features like `salary`, `height`, and we want to predict whether a customer would buy an insurance or not. We want to use kNN to do so. 
    - explain the process?
    - pros and cons of kNN?
- what is word2vec? Difference between CBOW and Skip-gram?  
    - CBOW: given context words, predict center word
    - skip gram: given a center word, predict all context words
    - how many parameters are there? (2|V|d)
- precision and recall trade-off?
    - if increase the threshold (prob increases from 0.5 to 0.7) of classifying as positive, then 
    - recall will decrease
    - precision will increase
    - f1 score may increase or decrease
- In linear regression, why use gradient descent instead of the closed-form solution?
- give a picture with classes, draw the decision boundary
- How to use decision tree to do regression?
    - take the average of the points in that leaf node
    - objective function = mse 
- if our tree is experiencing overfitting, what parameters can we tune in sklearn?
	- x
- what are the `random` in random forest
	- two sources of randomness! For each tree:
		- randomly select examples to train each tree => higher bias, lower variance
		- randomly select a subset of features to train each tree => higher bias, lower variance
- what are bagging and pasting?
    - bagging: use same model, but sampling of the training examples is done with replacement
    - pasting: use same model, but sampling of the training examples is done without replacement
- what is gradient boosting in ensemble learning?
    - when making prediction, sum over the prediction results of all predictors
    - Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.
- what are the meaning of the following hyperparameters?
    - `learning_rate`:
# neural network questions
- if underfitting (i.e., reduce bias)
    - train bigger model
    - train longer/better optimization algorithm
    - NN architecture/hyperparameters search
- if overfitting (i.e., reduce variance)
    - in general, reduce # parameters: A model with millions of parameters would severely risk overfitting the training set, especially if there are not enough training instances or if they are too noisy.
    - get more data
    - regularization of weights and increase regularization parameters
    - regularization by dropout and increase dropout rate 
    - regularization by early stopping
    - reduce # hidden layers 
    - reduce # neurons in each layer 
    - data augmentation (https://www.coursera.org/learn/deep-neural-network/lecture/Pa53F/other-regularization-methods)
        - that is, flip, rotate the images, etc.
    - NN architecture/hyperparameters search
- if we see that dev error and test error differ a lot, we're overfitting the dev set
    - make sure dev and test have the same distribution 
    - sol: increase the size of dev set
- what is batch-norm?
    - x
- what is drop-out?
    - x
- why use convolutional layer? Why we need convolutional layer?
- pros and cons of ReLU and sigmoid?
    - ReLU (non-saturated): exploding gradient (unstable gradient)
        - faster to compute
        - has dying ReLU problem: during training, some neurons effectively “die,” meaning they stop outputting anything other than 0. 
    - sigmoid (saturated): vanishing gradient. When value is large, the gradient is almost 0
        - slower to compute
- how to deal with exploding gradient?
    - gradient clippings

# statistics + math questions
https://www.springboard.com/blog/ai-machine-learning/machine-learning-interview-questions/
## probability: interview questions
- duplicate all data once
- for linear regression
    - coefficient will not change
    - std err will decrease, because typically, std err will have sqrt(n) in the denominator
        - the higher `n`, the lower std err is
    - t-statistic will increase, since t-statistic is just coeff / (std err), and numerator won't change
- for population parameter estimation:
    - the standard error will decrease
- that's why we should never duplicate our data, because it will make standard error unreasonably low!
    - instead, use weight instead of duplicate.
- to think about this problem, think of how duplicate affects sample variance, std error
    - sample mean, sample variance remain the same
    - so, std error (i.e., variance of sample mean) will decrease due to sqrt(n)!
    - so, for CI, it will have a smaller ME
- we have n students. Randomly select n with replacement, what is the expected percentage of students being sample? In fact, when n tends to infinity, percentage tends to 63.212%
This is the same as "percentage of bins having >=1 balls"
- when we want to maintain maximum from an array, this is the hiring problem!
    - P(update exactly twice) is a bit tricky.
- There’s one box — has 12 black and 12 red cards, 2nd box has 24 black and 24 red; if you want to draw 2 cards at random from one of the 2 boxes, which box has the higher probability of getting the same color? 
    - Can you tell intuitively why the 2nd box has a higher probability
    - think of the extreme: 2 vs 24. Then after drawing one color, then to draw another same color, the prob is 1/3, but for second case, it's 23/47.
- (toss two coins, then multiply them) vs (toss one coin, then square it), which one has larger expected value?
    - E(X)^2 vs E(X^2)
    - E(X^2) = V(X) + E(X)^2, so E(X^2) larger
- Suppose you roll a die and earn whatever face you get. Now suppose you have a chance to roll a second die. If you roll, you earn whatever face you get but you forfeit earnings from the first round. When should you roll the second time?   
    - when the first roll has value less than 3.5, the expected value
## stats interview Q
- https://towardsdatascience.com/40-statistics-interview-problems-and-answers-for-data-scientists-6971a02b7eee
- https://www.springboard.com/blog/ai-machine-learning/machine-learning-interview-questions/
- what is kurtosis? 
	- relate to the fourth moment. It measures the heaviness of the tails.
- regression, if unit changed, what is its effect?
	- if y is increased a times (i.e., unit of y changed), then all parameters increased by a times
		- intuitively, to make y increase by a, we need the RHS to increase by a times as well => each parameter increases by a times 
	- if a feature increase by x times (i.e., the unit of one feature is changed), the corresponding coefficient decreased by x times
		- intuitively, since y should not be affected by the unit change of a feature, the parameter value must be divided by x to balance the effect made to the feature
- when to use z-test and t-test
    - A Z-test is a hypothesis test with a normal distribution that uses a z-statistic. A z-test is used when you know the population variance or if you don’t know the population variance but have a large sample size.
    - A T-test is a hypothesis test with a t-distribution that uses a t-statistic. You would use a t-test when you don’t know the population variance and have a small sample size.
        - if sample size is large, then technically using t-test will be very similar to z test, since t approaches z, and so people just use z when n is large, regardless of whether population variance is known or not.
- what is rule of 72?
	- the time needed to double investment is = 72/r%
	- if accuracy is needed, use 69.3 instead of 72
- what is empirical rule?
    - The empirical rule states that if a dataset is normally distributed, 68% of the data will fall within one standard deviation, 95% of the data will fall within two standard deviations, and 99.7% of the data will fall within 3 standard deviations.
- what general conditions must be satisfied for the central limit theorem to hold?
    - IID assumptions. That is, each sample is iid. This implies the data must be sampled randomly.
    - The sample size must be sufficiently large, generally it should be greater or equal than 30
- What is the `Pareto principle`?
    - The Pareto principle, also known as the 80/20 rule, states that 80% of the effects come from 20% of the causes. Eg. 80% of sales come from 20% of customers.
- What is a `confounding variable`?
    - A confounding variable, or a confounder, is a variable that influences both the dependent variable and the independent variable, causing a spurious association, a mathematical relationship in which two or more variables are associated but not causally related.
- What does interpolation and extrapolation mean? Which is generally more accurate?
    - Interpolation is a prediction made using inputs that lie within the set of observed values. Extrapolation is when a prediction is made using an input that’s outside the set of observed values.
    - Generally, interpolations are more accurate.
- Is mean imputation of missing data acceptable practice? Why or why not?
- Mean imputation is the practice of replacing null values in a data set with the mean of the data.
    - Mean imputation is generally bad practice because it doesn’t take into account feature correlation. 
		- imagine we have a table showing age and fitness score and imagine that an eighty-year-old has a missing fitness score. If we took the average fitness score from an age range of 15 to 80, then the eighty-year-old will appear to have a much higher fitness score that he actually should.
    - Second, mean imputation reduces the variance of the data and increases bias in our data. This leads to a less accurate model and a narrower confidence interval due to a smaller variance.
- Give an example where the median is a better measure than the mean
    - When there are a number of outliers that positively or negatively skew the data.
- If a distribution is skewed to the right and has a median of 30, will the mean be greater than or less than 30?
    - mean will be larger than 30, since there are many data points with high values, and they will increase the mean a lot
- Give examples of data that does not have a Gaussian distribution, nor log-normal.
    - Any type of categorical data won’t have a gaussian distribution or lognormal distribution.
    - Exponential distributions — eg. the amount of time that a car battery lasts or the amount of time until an earthquake occurs.
- difference between one-hot encoding and dummy encoding?
    - one-hot encoding creates dependent columns, whereas dummy encoding will not!
		- since for one-hot, the last column can be deduced, after knowing the first n-1 columns
- if features are standardized and we're using linear regression, then how do we interpret the coefficients?
    - the interpretation will be in terms of standard deviation. When we increase 1sd, which corresponds to increasing the standardized variables by 1, then target will increase by the coefficient value
- when doing 2 population hypothesis test, will using t-test or normality test make a difference?
    - no, if the sample size is large enough
    - if someone tells u that the dataset is actually the population, then it doesn't make sense to do a hypothesis test, since the sample mean is the population mean in this case!
- what is power
    - power = P(reject H0 | H0 is false) = 1 - beta = 1 - Type I error
- connection between p-value and test statistic?
    - left tail: `p-value = P(t<test statistic)`
    - right tail: `p-value = P(t>test statistic)`
    - two tails: `p-value = 2*min(P(t<test statistic), P(t>test statistic))`
- probability vs likelihood?
    - https://www.statology.org/likelihood-vs-probability/
    - probability: the chance that a particular outcome occurs based on the values of parameters in a model/distribution.
    - likelihood: refers to how well a sample provides support for a particular value of a parameter in a model. That is, likelihood tells us the probability that we will obtain the observed sample that we've already observed, assuming that each sample is a IID RV from a known distribution with some parameter value. 
        - essentially, tells us how likely to obtain the given sample, if the distribution parameter is set at some value
        - the higher the likelihood, the more plausible the parameter is.
- difference between Z and t distribution:
    - t = Z/sqrt(chi-square(n)/n)
    - var(t) = v/v-2 => larger than 1 variance => larger variance than Z
    - as n tends to infinity, var(t) = 1, and Z = t, since the denominator of t becomes 1
- why correlation is between -1 and 1?
    - correlation of two vectors is cos(theta), where theta is the angle between x_demean and y_demean, and so correlation must be between -1 and 1, since cos(theta) is
    - we can first write down the formula for sample correlation, and then define x_demean as a vector whose term is x_i - x_mean
    - for correlation of random variables, we will need to define inner products for random variables and then use CS inequality 
- For sample size n, the margin of error is 3. How many more samples do we need to bring the margin of error down to 0.3?
    - suppose we're doing CI for population mean
    - ME = Z * s/sqrt(n) = 3
    - so, Z * s/sqrt(100n) = 0.3
    - need 100n-n = 99n more samples!
- In what cases does R^2 take negative values?
    - when it's worse than a model that always predict the mean of the target values
    - remember that R^2 = 1 - SSE/SST
- What is the standard error of the mean?
    - standard error of an estimator
        - i.e., se of sample mean, sample proportion, etc
    - remember standard error of the mean = standard deviation of sample mean!
    - this is the standard deviation of the sample mean
- There are 100 products and 25 of them are bad. What is the confidence interval?
    - `ci = 0.25 +- 1.96*sqrt(0.25(1-0.25)/100)` 
- What is a p-value? Would your interpretation of p-value change if you had a different (much bigger, 3 mil records for ex.) data set?
    - yes, because the t-statistic depends on sqrt(n). The larger n, the more likely that any deviation of the sample mean from the population mean is due to wrong H0, because when the sample size is sufficiently large, sample mean should be extremely close to is expected value, which is the population mean, according to the law of large number. Therefore, if the sample size increases, absolute value of test statistics will become larger, and thus p-value will be smaller
- in logistic regression, what is the cost function we want to maximize?
	- ans: cross entropy. 
		- why don't we use the sum of squares error (SSE) as the cost function?
		- ans: in logistic regression, SSE is not convex (and thus has multiple minimum pts). Also, minimizing cross entropy corresponds to minimizing negative of log likelihood 
		- do we have a closed-form solution for this minimization problem?
- in neural network, what is ReLU? Why we prefer using ReLU instead of sigmoid activation unit?
    - using ReLU is faster for training neutral network, since there is no vanishing gradient problem
- Why is “Naive” Bayes naive?
    - Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. This implies the absolute independence of features — a condition probably never met in real life.


# behavioral quesstions
- introduce yourself 
	- Hello, my name is Wing Ho Lin. You can call me Wing. Let's first of all talk about my current employment. I'm now working as a part-time lecturer at Hong Kong Polytechnic University, teaching data science courses, such as statistical inference. And in the past, I have also taught some other major data science courses, such as machine learning and data mining.
	- Before I took up the teaching position, I worked as a full-time data scientist at MassMutual, which is an insurance company headquartered in the US.
	- And my main responsibility at Massmutual was building predictive models for their insurance products.
		- and in fact, the purpose of building models at MassMutual was not for internal use, but we want to sell our model as a SaaS product, Software as a Service product, to other insurance companies around the world, so that other insurance companies can use our model to predict on their own dataset.
		- and FWD was also one of our clients interested in our predictive models.
		- I was involved in the whole development process, from ideation, research, data engineering, model building, performance evaluation. 
	- For example, I'm responsible for building regression models to predict the loss ratio for health insurance products, such as VHIS, Voluntary Health Insurance Scheme.
	- Also, I built decision models to predict the underwriting decisions in order to automate the underwriting process for health insurance products
	- And let's talk about my education. I received my MPhil and Bachelor's degrees from HKUST, and I majored in computer science.
	- when I'm not working, I love reading math textbooks, especially statistics-related topics, because I really love math.	
	- I also enjoying doing many kinds of exercises when I'm free, such as jogging, hiking and cycling. 
- weakness:
	- as a job candidate, I have relatively fewer years of industrial experience, because my work experience is mainly on teaching as a lecturer of data science, but this also allows me to have a deeper understanding of data science, because I teach data science.
	- I also only have exposure to the insurance industry and academia.
- strength:
	- my deep understanding of machine learning, statistics and math, due to my teaching experience in data science 
	- I'm a humble and cooperative person and I'm receptive to other people's ideas. I think these personality traits are important in a company running in a startup style, because collaboration is the key to success for startups 
- describe the objective, steps, and results of a data science project you worked for before
	- x 

## important things of behavioral
- for part-time jobs, do not say that I am available earlier because just marking (for the teaching job) is left. We must say the time when we have finished everything!
- important: do not mention work-life balance! This might have the company thinking you are lazy!

## specific company interview
### FWD interview
- 45 mins
- english intro
- project experience
- ML tools - high level
- customer data example from university project

- why want to work for FWD?
- FWD seems to be eager to use data science to distinguish itself from other insurance companies, and I think it's a good thing, because that means data science is considered important at FWD
- i think FWD has a promising future. It's going to be publicly listed in HK, and it's headquartered in HK, and so it should be able to provide a stable environment for its employees in HK.
- i think FWD is doing data science projects that are similar to what we did at MassMutual, and I really enjoy what I did at MassMutual, and so I think working at FWD will be enjoyable too.


### Intact interview
- why want to join my company
	- intact is a well-known insurance company with strong financials, so I think it can provide stability to its staff
	- I believe from the job description that the data science team cares about the understanding of underlying Machine Learning theory, and I think this is a good thing
	- the startup culture: the intact lab has a startup culture but also has the stability of a multinational company

## your teaching experience
- ML: teach PCA, bayesian network, more theoretical, 
DM: more on software. Teach pandas to do data analytics, in addition to predictive modelling
- common topics for data mining and ML:
	- DT
	- RF
	- ENSEMBLE
	- LDA
	- Linear regression
	- logistic regression
	- Naive Bayes
	- ROC curve, PR curve, precision, recall, F1
	- K-means
	- knn
- for data mining, extra topics:
	- data management
		- pandas and sklearn
		- missing value management
		- one-hot encoding, target encoding, data normalization etc
- for ML, extra topis:
	- bayesian network
	- PCA
	- more mathematical proofs of theorem (e.g., proof of why the cost function of logistic regression is cross entropy function, but not sum of squares function, as in linear regression)

# questions
- lightgbm, xgboost, catboost?
	- catboost: Category Boosting
	- lightgbm: Light Gradient Boosted Machine
	- xgboost: eXtreme Gradient Boosting
	- https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924
	- xgboost vs lightgbm
		- https://lightgbm.readthedocs.io/en/latest/Features.html
		- lightgbm: `level-wise` to grow tree. All other tree algorithms use `leaf-wise`
		- `Speed`: LightGBM is generally faster than XGBoost due to its use of histogram-based algorithms for finding the best split points. This allows LightGBM to handle large datasets more efficiently than XGBoost.
		- `Memory` usage: LightGBM uses less memory than XGBoost because it only stores the non-zero values of the feature matrix. This can be particularly useful when dealing with high-dimensional data.
		- `Handling categorical features`: Optimal Split for Categorical Features. LightGBM can handle categorical features directly, while XGBoost requires one-hot encoding of categorical features. This can be a disadvantage for XGBoost when dealing with datasets with a large number of categorical features.
			- if the cardinality is n, then when we divide it into 2 groups, how many ways are there?
			- It is common to represent categorical features with one-hot encoding, but this approach is suboptimal for tree learners. Particularly for high-cardinality categorical features, a tree built on one-hot features tends to be unbalanced and needs to grow very deep to achieve good accuracy.
			- Instead of one-hot encoding, the optimal solution is to split on a categorical feature by partitioning its categories into 2 subsets. If the feature has k categories, there are 2^(k-1) - 1 possible partitions. But there is an efficient solution for regression trees[8]. It needs about O(k * log(k)) to find the optimal partition.
			- The basic idea is to sort the categories according to the training objective at each split. More specifically, LightGBM sorts the histogram (for a categorical feature) according to its accumulated values (sum_gradient / sum_hessian) and then finds the best split on the sorted histogram.
		- `Regularization`: XGBoost offers more regularization options than LightGBM, including L1 and L2 regularization, as well as tree pruning. LightGBM only offers L2 regularization.
		- `Tuning parameters`: XGBoost has more tuning parameters than LightGBM, which can make it more difficult to optimize. LightGBM has fewer tuning parameters, which can make it easier to use and optimize.
	- `lightgbm`: also can handle categorical features, without first using to one-hot encoding 
	
- important parameters of tree-based algorithms
	- max_depth
	- min_samples_leaf
	- n_estimators
	- learning_rate
		- scales the contribution of each tree. If you set it to a low value, such as 0.05, you will need more trees in the ensemble to fit the training set, but the predictions will usually generalize better. This is a regularization technique called shrinkage.
	- subsampling?

- explain decision tree
	- no need feature scale and centering
	- gini index: another measure for decision tree
	- entropy = weighted surprise
	- IMPORTANT: in CART decision tree, we grow the tree in a depth-first manner. In lightGBM, we grow in a breath-first manner
	- `min_*` features: the higher, the higher the regularization
		- `min_samples_split`
		- `min_samples_leaf`
		- `min_weight_fraction_leaf`
		- `min_impurity_decrease`
	- `max_*` features: the lower, the higher the regularization
		- `max_depth`
		- `max_features`
		- `max_leaf_nodes`
	- if doing regression, then the measure becomes MSE instead of entropy

## reason of leaving
- reason of leaving MM
    - because of a change of company strategy (or policy direction), MassMutual decided to shut down the operation of the entire HK office, and so everyone in the HK office has been made redundant
    - (no need to mention funding problem)
- why leave my part-time teaching position?
	- I'm currently teaching as a part-time lecturer at Hong Kong Polytechnic University. This position is meant to be temporary only, since it's part time.
- how can you contribute:
	- my experience in building predictive models 
	- I also have strong background in statistics, because I teach statistics courses at university as well.
	- I have experience working at a startup and multinational company, because at MassMutual I was assigned to help with the startup of MassMutual called Haven Tech Asia, but I still report to the data science team of MassMutual in the US. 

# post-interview questions to ask
- what is the size of the data science team?
- are we responsible for the Hong Kong business only? 
	- So, our model will only be used in HK region?
- Can we use the data from other regional office of FWD?
	- will there be data governance issue that disallow us to use data from other regions?
- will there be another round of interview?
- when can I expect to hear from you about the decision?
- what does it take to switch from a contract role to a permanent role?

## job responsibility at Massmutual
- SaaS = Software as a service
- IMPORTANT: we also collect 3rd party data (e.g., demographic data from census department) to improve our data
	- we know the address of each insured
	- we then find the neighbor each insured belongs to. A neighbor is defined in the census department
	- then, from the census dept, find the median income, percent of people going to uni, etc. 
		- so, we have approximately known the income, education level, etc
### loss ratio model
- loss ratio model:
	- objective: given a customer wanting to purchase a health insurance product, predict his loss ratio in the coming 5 years
		- loss ratio = (amount claimed) / (insurance premium)
		- in the train set, loss ratio is defined as:
			- (sum of claim amount in 5 yrs) / (sum of premium in 5 yrs)
		- we don't do prediction on the first 2 years, because in the first two years, there are usually exclusion period / waiting period
	- but when doing evaluation, we focus on evaluating the ranking power of the model, using Spearman's rank correlation coefficient, decile plot, gini coefficient
		- for underwriting purposes, we often just want to decline a certain percent of customer who would claim the most, and may not be interested in the exact loss ratio.
	- benefit: automate the underwriting process
		- can automatically decide whether to let a client to purchase an insurance
		- also, can adjust the client's insurance premium based on the predicted loss ratio
	- steps:
		- selecting useful features from database
			- we have thousands of tables in our database, we need to find the useful features and make it a dataset for building model. 
			- we select common features 
				- `height, weight, age, sex, bmi, alcohol, cigarette consumption, dangerous activity, addictive substance`
				- personal disease variables, family disease variables
				- heart rate, blood pressure (one model has these variables whereas others don't, because some insurance companies may not collect these as features)
		- data cleansing and sanity check
			- many features have missing values and wrongly input values (e.g., unit of height is mixed up, the height of a 2 year old is 1.8m etc )
			- we also need to do sanity check
				- e.g., some records show very low claiming amount (e.g., $20)
					- that's because the client may have another insurance that already almost cover the cost. 
			- premium and claim amount also suffer from inflation problems, so need to do discounting
		- feature engineering
			- one-hot encoding
			- no target encoding, since we actually don't have features with higher-than-2 cardinality 
			- make features that summarizes lots of features
				- clean_record, clean_record_fam
		- feature selection
			- use two stage method: binary variable filtering, and RFECV (Recursive Feature Elimination with Cross-Validation)
		- build several models, can say we try ensemble as well
			- but feature importance decomposition will have problems if using ensemble
			- XGBoost with Tweedie loss function
			- frequency-severity model 
				- build two models: 
					- predict frequency of claim 
						- XGBoost with Poisson loss
						- y is `freq = total_cnt / exposure`
					- loss ratio of each claim
						- Gamma regression
						- y is `loss_ratio / freq`
			- classification-regression model
				- build two models:
					- predicting prob of having a claim
						- xgboost with logistic loss
					- loss ratio, given a person will claim
						- the train set consists of those who made >= 1 claim
		- performance evaluation
			- we use standard regression metrics 
				- but performance isn't good, since our model is weak in predicting the exact loss ratio
			- instead, evaluate the model based on the ranking power in terms of loss ratio (i.e., risk)
				- sometimes, our model may not be able to predict the exact loss ratio, but our model does recognize that this record with have a relatively high loss ratio, compared with other records.
			- metrics for evaluating the ranking power:
				- Decile plot: using cohort deciles
					- rank the test set record based on predicted loss ratio
					- calculate the mean loss ratio for each cohort
					- compare with the true mean loss ratio for each cohort
					- the closer the predicted curve is to the true curve, the better
				- Gini index
					- this is a common evaluation metric, particularly in the actuarial science community, for assessing the discriminatory power of a model (e.g., whether the model can differentiate between the high-risk customers and the low-risk customers). Gini index is a single number between 0 and 1. In our case, the higher the gini index, the better our model in ranking the contracts by their loss ratio. 
					- gini index = gini coefficient = `2 * (area bounded by the line of equality and lorenz curve)`
					- x-axis: fraction of policyholders, ranked by models from lowest loss ratio to highest loss ratio
					- y-axis: fraction of total claim amount
				- Spearman’s rank correlation
					- https://towardsdatascience.com/clearly-explained-pearson-v-s-spearman-correlation-coefficient-ada2f473b8
					- just convert the two sets of data into rank, and then run pearson's correlation
					- It assesses how well the relationship between two variables can be described using a monotonic function. In our case, the two variables are the true loss ratio and the predicted loss ratio. The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables. While Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not). A perfect Spearman correlation of 1 occurs when each of the two variables is a perfect monotonic increasing function of the other.
					- check whether the ranking is a linear function or not
- problems with building a SaaS, but not for internal use:
	- subject to data drift, concept drift
	- can't use specialized features (e.g., education background, living location, etc)
- how to know if your model is better than actuary's estimation of loss ratio?

### challenges faced with loss ratio model
- the time period can't be easily adjusted. We now use 5 yrs, but some other companies may prefer some other time period. 
	- we need to rebuild a model if we change the time period
- we are restricted with what features we can use, since we need to make sure the features we use are common to other insurance companies, because our aim is for other insurance companies to use our model
	- ex: we generally can't use income, education background, heart rate etc as features, since these features may not be available for other companies
- we'll face concept drift and data drift problem
	- concept drift: MM may be more rigid when it comes to BMI. A high BMI may lead to a decline in the underwriting process, but in some other companies, they may not even look at the person's BMI.
	- data drift: MM has a high proportion of customers with age between 35 - 50. But some other companies may target people with younger age, and so the proportion of age between 35 - 50 may be lower
- some claiming cases can't be predicted, since they could be accidents
	- so, predicting exact loss ratio accurately is often difficult


## details of data mining/ML courses at POLYU
### project on predictive modelling
- classification project from kaggle competition, with the feature names masked to avoid
- dataset obtained from openML, Kaggle
- `French Motor Claims Datasets`: freMTPL2freq
	- from kaggle
	- 677,991 records
	- y = claim amount of coming year
	- regression problem
	- insurance claims dataset
- `Census Income dataset`: predict whether an adult will have income >50k or not
	- from kaggle
	- binary classification problem
- use google colab for students without sufficient computing device
### data mining assignment on data wrangling and EDA
- `Titanic dataset`
	- missing value management
	- data cleansing (the salary unit is mixed, need to use regular expression)
	- one hot encode categorical features
	- standardization of features 
	- find correlation between each pair of continuous features
	- find the survival rate by cohort (e.g., gender + ticket class)
	- mean age by cohort (gender + ticket class)
	- does people have siblings/spouses more likely to survive?
	- find highest and lowest top 3 fare for each ticket class
- `MovieLens` moving rating dataset


# important parameters for gradient-boosting based algo
- `scale_pos_weight`
	- for imbalanced classification dataset
- `learning_rate`
	- Boosting learning rate (xgb’s “eta”)
- `n_estimators`
	- number of trees
- `subsample`
	- in RF, it's called `max_samples`
	- Subsample ratio of the training instance.
- `max_features`
	- it's called `colsample_by*` for xgboost
- `max_depth`
	- Maximum tree depth for base learners
- `colsample_bytree`
	- Subsample ratio of columns when constructing each tree.
	- basically, proportion of features to consider when constructing a tree, just like random forest
- `min_child_weight`
	- Minimum sum of instance weight(hessian) needed in a child.
- `min_samples_split`
	- The minimum number of samples required to split an internal node:
	- it's called `min_child_weight` for xgboost
- `reg_lambda`
	- L2 regularization term on weights (xgb’s lambda).
- `tweedie_variance_power`
- `early_stopping`

- for random forest, we have `max_samples`, which defaults to `m`. Note: `max_samples` in RF = `subsample` in boosting algorithm
- for prediction of RF and Extra-Trees are both based on averaging!
- remember the sampling method is w.r!
- The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice the variance reduction is often significant hence yielding an overall better model.
- In contrast to the original publication [B2001], the scikit-learn implementation combines classifiers by `averaging their probabilistic prediction, instead of letting each classifier vote for a single class.`
- In extremely randomized trees (see ExtraTreesClassifier and ExtraTreesRegressor classes), randomness goes one step further in the way splits are computed. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias:
	- i.e., an extra source of randomness: the threshold of each feature is randomly chosen
- remember: random forest and extra-tree takes average, whereas boosting algorithms take the sum!

# leaf-wise vs level-wise
- level-wise (breath-first): grows the tree level by level. In this strategy, each node splits the data prioritizing the nodes closer to the tree root. 
	- most common. Used by xgboost
- leaf-wise (depth-first): grows the tree by splitting the data at the nodes with the highest loss change.
	- used by lightGBM



## Feature importance evaluation for tree algorithm
- The relative rank (i.e. depth) of a feature used as a decision node in a tree can be used to assess the relative importance of that feature with respect to the predictability of the target variable. Features used at the top of the tree contribute to the final prediction decision of a larger fraction of the input samples. The expected fraction of the samples they contribute to can thus be used as an estimate of the relative importance of the features. In scikit-learn, the fraction of samples a feature contributes to is combined with the decrease in impurity from splitting them to create a normalized estimate of the predictive power of that feature.
- By averaging the estimates of predictive ability over several randomized trees one can reduce the variance of such an estimate and use it for feature selection. This is known as the mean decrease in impurity, or MDI. Refer to [L2014] for more information on MDI and feature importance evaluation with Random Forests.
- Warning: The impurity-based feature importances computed on tree-based models suffer from two flaws that can lead to misleading conclusions. First they are computed on statistics derived from the training dataset and therefore do not necessarily inform us on which features are most important to make good predictions on held-out dataset. Secondly, they favor high cardinality features, that is features with many unique values. Permutation feature importance is an alternative to impurity-based feature importance that does not suffer from these flaws. These two methods of obtaining feature importance are explored in: Permutation Importance vs Random Forest Feature Importance (MDI).
- The parameter learning_rate strongly interacts with the parameter n_estimators, the number of weak learners to fit. Smaller values of learning_rate require larger numbers of weak learners to maintain a constant training error. Empirical evidence suggests that small values of learning_rate favor better test error. [HTF] recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1) and choose n_estimators by early stopping. For a more detailed discussion of the interaction between learning_rate and n_estimators see [R2007].
	- important: use small `learning_rate` and choose `n_estimaotrs` by `early_stopping_rounds = 20`!
	- usually, we use `subsample`, `learning_rate`, `max_features`, `early_stopping`, `n_estimators` 
	
## histogram-based gradient boosting
- idea: use binning for the feature values
	- e.g., features with values in [0,10] are put in first bin, etc
	- so, number of possible values of the features = # bins!
	- it reduces the possible split points needed to evaluate!
- Histogram-based boosting is a type of gradient boosting algorithm that uses histograms to represent the distribution of feature values in the training data. In traditional gradient boosting algorithms, the feature values are sorted and split at each node of the decision tree, which can be computationally expensive for large datasets. Histogram-based boosting, on the other hand, uses histograms to group the feature values into discrete bins, which reduces the number of split points and speeds up the training process.
- In histogram-based boosting, the training data is first divided into bins based on the feature values. The bins are then used to construct histograms that represent the distribution of feature values in each bin. The histograms are used to calculate the gradients and Hessians for each split point, which are then used to determine the best split point for each node of the decision tree.
- Histogram-based boosting has several advantages over traditional gradient boosting algorithms. First, it is faster and more memory-efficient, since it reduces the number of split points and eliminates the need for sorting the feature values. Second, it can handle categorical features and missing values more efficiently, since the bins can be constructed to handle these types of data. Finally, it can improve the accuracy of the model, since the histograms can capture the distribution of the feature values more accurately than traditional split points.
- main advantage: speed. Can be orders of magnitude faster!
- have built-in support for missing values 
- native support for categorical features
- lower memory usage, since continuous values are replaced with discrete bins 

# things to remember
- in regularizing linear/logistic regression, we're regularizing the weights of the models!
	- so, the values of the weights will be low after regularization

# terms
- data drift
	- 
- data mismatch
	
# collinearity vs multicollinearity vs correlation?
- perfect collinearity: X1 and X2 can be polynomial relationship, not a straight line relationship
- https://quantifyinghealth.com/correlation-collinearity-multicollinearity/
- example of exactly three RVs having collinearity, but no pair of RVs are correlated
	- X_i=score_i, i=1,2,..,100
	- X_i and X_j are independent
	- S = weighted average of the X_i
	- S and each X_i are correlated, but not high
	- but S and all the X_i have perfect multicollinearity
		- NOTE: X_1 and the rest of X_i are still not linearly related!
- However, because collinearity can also occur between 3 variables or more, EVEN when no pair of variables is highly correlated (a situation often referred to as “multicollinearity”), the correlation matrix cannot be used to detect all cases of collinearity.
- to detect multicollinearity, for each X_i and S, calculate VIF_i
	- for each X_i, build a regression model, treating X_i as dependent variable and all other variables as independent.
	- if R^2 is high, that means X_i can be explained by the rest of the variables
		- and so, multicollinearity exists between X_i and the rest of the variables, since X_i can be predicted by other variables
		- however, we only know there are multicollineary. There could be some X_j that have no effect on X_i. That is, the rest of the variables together is predictive of X_i
	- As a rule of thumb, a VIF > 10 is a sign of multicollinearity


# statistics
- https://quantifyinghealth.com/p-value-explanation/
- https://quantifyinghealth.com/make-results-statistically-significant/
- ![](media/collinearity.jpg)
Misinterpretation #3: A p-value of 0.001 means that the effect is stronger than with a p-value of 0.03
A p-value says nothing about the size of the effect. One important thing to note is that p-values are sensitive of the size of the sample you are working with — all other things held constant, a larger study will yield lower p-values. This however would not change the practical significance of the results.


Misinterpretation #4: Blind confidence in statistically significant results (believing all results that have p-value < 0.05)
In fact, if we examine the relationship between any 2 unrelated random variables, we have a chance of 5% of getting a p-value < 0.05. This means that by running 20 statistical test, on average 1 of them will be statistically significant.

So how can we be sure that the researchers who conducted the study didn’t run many tests in order to increase their chances of getting a statistically significant result?

The answer is that we cannot rule out multiple testing.

This, among many other reasons, such as publication bias and conflicts of interest, is why we must be very conservative when we interpret p-values.

- Regression coefficient = beta1, in simple linear regression
	- NOTE: after standardizing the features, correlation coefficient = regression coefficient

- should we standardize the dependent variable, when doing standardization?
	- yes: https://quantifyinghealth.com/standardized-vs-unstandardized-regression-coefficients/
	- if both kinds of variables are standardized, then 
		- A change of 1 standard deviation in X is associated with a change of β standard deviations in Y
	- if only independent variables are standardized, then
		- A change of 1 standard deviation in X is associated with a change of β units in Y
	- for categorical variables, we will have trouble interpreting the coefficients
- P(type I) = alpha = P(reject a true H0) = P(FP)
- P(type II) = beta = P(fail to reject a false H0) = P(FN)
- P(power) = 1 - P(type II) = P(TP)
- usually, H0 = no effect; H1 = has some effect


## f test in linear regression
- https://quantifyinghealth.com/f-statistic-in-linear-regression/
- The F-statistic provides us with a way for globally testing if ANY of the independent variables X1, X2, X3, X4… is related to the outcome Y.
	- If the p-value associated with the F-statistic is ≥ 0.05: Then there is no relationship between ANY of the independent variables and Y
	- If the p-value associated with the F-statistic < 0.05: Then, AT LEAST 1 independent variable is related to Y

- why when doing ANOVA, we only need to check if F is larger than F_alpha?
	- at least one of the parameters is larger than 0, we have the test statistic being large
	- at least one of the parameters is smaller than 0, we have the test statistic being large as well!
	- so, under two cases, the statistic will be large
	- so, should reject H0 if the test statistic is large



# ANOVA:
- when we have more than 2 populations, we need to use ANOVA to test if the mean of all the populations are the same
- 





- statistical power = 1 - P(type 1 error) = 1 - P(reject true H0) = 1- FP
	










- P(type II) = P(not reject false H0) = P(FN)
- P(Type I) = P(reject true H0) = P(FP)
- power = 1 - P(type I) = 1 - P(FP) = P(TP) = P(reject false H0)


- Homoscedasticity (ho-mo-sec-das-ti-ci-ty): the sequence of RV have constant variance

- interaction terms:
	- https://quantifyinghealth.com/why-and-when-to-include-interactions-in-a-regression-model/
	- Variables that have a large influence on the outcome are more likely to have a statistically significant interaction with other factors that influence this outcome.

# transform
- square root transform / log transform for right skew
	- https://quantifyinghealth.com/square-root-transformation/
	- IMPORTANT: for regression, we apply square root on the target, not on the features!
		- when we see homoscedasticity is violated
		- when we see the linear relationship of x and y is violated
	- to make right-skew distribution become more normal 
	- compressing high values and stretching out the ones on the lower end.
	- log transform will have more severe effect than square root transform
	- If your variable has a right skew, you can try a square root transformation in order to normalize it.
	- Examples of variables with a right skew include: income distribution, age, height and weight.
	- in regression, we assume there is a linear relationship between x and y
		- One solution to fix a non-linear relationship between X and Y, is to try a log or square root transformation.
		- 
- quadratic, cube or exponential transform for left skew
- sometimes, if we see the relationship between x and y is quadratic, instead of using a quadratic term, we can do a square root transform!


## regularization
- Because regularization is trying to shrink coefficients, it will affect larger coefficients more than smaller ones. So the scale on which each variable is measured will play a very important role on how much the coefficient will be shrunk. Standardizing helps deal with this problem by setting all variables on the same scale.
- LASSO (L1 regularization) is better when we want to select variables from a larger subset, for instance for exploratory analysis or when we want a simple interpretable model. It will also perform better (have a higher prediction accuracy) than ridge regression in situations where a small number of independent variables are good predictors of the outcome and the rest are not that important.
- Ridge regression (L2 regularization) performs better than LASSO when we have a large number of variables (or even all of them) each contributing a little bit in predicting the outcome.


## feature importance
- IMPORTANT: comparing the standardized coefficients is not good, since feature with higher variance will have higher coefficients!

## statistics notes
- decide by n-1 in sample sd: this is only because we want the sample sd to be a better estimate of the population sd. If our purpose of calculating sample sd is not to estimate the population sd, we need not use `n-1`!
	- https://quantifyinghealth.com/why-divide-sample-standard-deviation-by-n-1/
	- If we want to estimate the average distance of the data from µ (i.e. when we want to estimate the population standard deviation), we will have to account for 2 things:
		- the average distance of the data from x̄ (i.e. the sample standard deviation), and
		- the average distance of x̄ from µ.
	- so, dividing by `n` will be an underestimate of the true variation as an estimate of the population sd!
- when we say marginal distribution, that means unrestricted, no by cohort, distribution
	- conditional distribution: the height distribution, given sex is male
	- marginal distribution: the height distribution of both sexes
- missingness: https://quantifyinghealth.com/handle-missing-data/
	- when we have a variable with missing, ask ourselves:
		- is the missing value depends on the value of the variable itself?
			- if so, then missing not at random
			- older students may not want to disclose their age => MNAR
			- salary: lower salary people tend to omit it => MNAR
		- is the missing value depends on the values of other variables?
			- e.g., 
- for chi-squared test, we have three usages:
	- goodness of fit
	- homogeneity
		- it's just like independence test
		- basically, test p_1 = p_2 = p_3 = ... = p_k
	- independence
- for testing all proportions are equal, we can use chi-squared test, instead of ANOVA!
- to interpret the coefficient of a quadratic term of linear regression, we consider the derivative of y wrt x:
	- y' = `theta_1 + 2x * theta_2`
		- so, a unit increase of x will make y increase `theta_1 + 2x * theta_2`
- odds ratio = odds(x+1) / odds(x)
- relative risk = probability ratio = `sigmod(theta*(x+1))/sigmod(theta*(x)) `
- remember, when we have a function such that `ln y = a + b*x`, if x increases by 1, then y is multiplied by e^b
	- we can know this by considering the ratio: `odd(x+1) / odd(x)`
- be careful about the multiple testing problem!
	- we need to understand false discovery rate
	- 



type 1 = reject a true H0 = FP
type 2 = fail to reject false H0 = FN


recall = TPR = TP / (TP+FN)
precision = TP / (TP + FP)


- macro: unweighted mean of each class
- micro: find globally
- weighted: macro, but weight by support
- 

# how to deal with dataset too large for memory (out-of-core learning)
- aka `out-of-core learning`, `online learning`, `incremental learning`
- classic example: stochastic gradient descent (SGD) and mini-batch gradient descent
- when loading data (e.g., `pd.read_sql` and `pd.read_csv`), use `chunksize` parameters
- some algorithms has `partial_fit` method
- https://scikit-learn.org/0.15/modules/scaling_strategies.html
- some methods have `partial_fit`
	- multinomialNB
- for k-means, can use `MiniBatchKMeans` to do incremental learning!
```python
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    minibatch_kmeans.fit(X)
```


# interview questions
- https://www.simplilearn.com/tutorials/data-science-tutorial/data-science-interview-questions
- What are dimensionality reduction and its benefits?
	- The Dimensionality reduction refers to the process of converting a data set with vast dimensions into data with fewer dimensions (fields) to convey similar information concisely. 
	- This reduction helps in compressing data and reducing storage space. It also reduces computation time as fewer dimensions lead to less computing. It removes redundant features; for example, there's no point in storing a value in two different units (meters and inches). 	
	- if feature interpretation is important, then we can't use dimensionality reduction!
		- e.g., we need to explain how each feature contribute to the final prediction
- how to deal with outliers?
	- if the outliers naturally occur in the data distribution, then it's fine to have
	- if the outliers are due to input error, 
	- use algorithms that don't suffer some outlier problem, such as tree-based algorithms
- how to deal with missing values?
	- impute by median by cohort (stratified by features)
	- build a linear model to predict the missing values, possibly by cohort 
- how to do feature selection
	- `binary variable filtering using variance threshold`
		- Q: can we do a entropy threshold selection instead?
	- `Recursive Feature Elimination with Cross-Validation (RFECV)`
		- required:
			- choose a model that can output feature importance
			- choose scoring, which is used to determine the optimal number of features. E.g., if `accuracy`, then the number of features resulting in the highest scoring will be used 
		- start with all features. Train the model based on all features. At each step, remove the feature with the lowest score. Repeat this until `min_feature` is reached
		- Finally, check which number of feature results in the highest cross-validated scoring.
		- https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
	- `Boruta`
		- a
	- `permutation importance`
		- 
- how to find feature importance
	- if tree models, use the built-in `mean decrease in impurity` (`feature_importances_`)
		- decrease in impurity = information gain
		- the average is over the trees and also over the number of times the feature is chosen in each tree
		- 
	- for other models, use 
		- `permutation importance`
			- Permutation feature importance is a model inspection technique that can be used for any fitted estimator when the data is tabular. This is especially useful for non-linear or opaque estimators. The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled [1]. This procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature. This technique benefits from being model agnostic and can be calculated many times with different permutations of the feature.
			- start with a trained model (e.g., random forest) and calculate the metric score
			- permute one feature, then find the new metric score (e.g., roc-auc) 
			- calculate decrease_in_score = old_score - new_score
			- pros:
				- no need to do the refitting. In fact, the idea is similar to backward elimination in regression. But backward elimination requires refitting of model after removing one feature, and thus very slow
			- cons:
				- can't deal with correlated features: suppose we have `height in cm` and `height in m`. Then, each of the feature importance will be low?
					- not necessarily, since we do not rebuild the model! If `height in cm` is identify as useful features by the model, it'll continue so. Think of regression! It will have a large coefficient!
					- situation: originally, our model gives equal importance to the two features.  
				- some features, after shuffling, may become unrealistic as a whole record! Think of shuffling `height` and now there could be a baby with 2m tall!
				- 
		- `SHAP`


# explain ML models
## explain PCA step by step
- objective: find a k-dimensional space to project each record to this space such that the projection error is minimized
	- `projection error = sum_{i=1}^{k}||p_i-x_i||`, where p_i is the projection vector of x_i
- result: the space should be span by the eigenvectors of the covariance matrix
- first, de-mean the data for each feature (so, each feature has zero mean)
- Then, find the eigenvectors of the covariance matrix of the demeaned matrix. This is just `1/m * M'M`
- The principal components are the eigenvectors corresponding to the largest eigenvalues
- if we want to reduce the dimension to k, then find the scalar projection of each record to the orthogonal subspace spanned by the first k eigenvectors 
	- `x_proj = \sum_{i=1}^k (x^Tq_i)q_i`
- to determine $k$, we look at the total variance explained
	- `total variance = \sum (eigenvalues)`
	- set a percent: e.g., find $k$ such that 90% of the variance is explained by the projection
- if we pick the first k eigenvectors, then the error is the sum of the remaining n-k eigenvalues
## explain decision tree step by step
- 
## explain k-means step by step
- objective: minimize inertia:
	- `inertia = \sum_{i=1}^{n} (x_i - centroid(x_i))**2`
	- suppose have k clusters
	- initialize k centroids randomly
	- `cluster assignment`: assign each record to the closest centroid
	- `centroid update`: centroid_i = mean(all x_i assigned to centroid i)
  	- in sklearn, `k-means++` is chosen as the default initialization method for cluster centroids 
		- choose an existing instance as centroid, but with higher probability for those instances whose distance from the nearest centroid is high
## explain EM algorithm step by step
- start with empty tree. 
- in each step, choose a feature, then for each possible value of that feature:
	- calculate the entropy wrt the class label
	- then, find the weighted mean of the entropy, over the possible values of that feature
	- optionally, find the information gain = (entropy of root) - (weighted entropy)
	- pick the feature with the lowest weighted entropy, or equivalently, highest information gain
## explain decision tree step by step
- `entropy of a node = \sum_{i=1}^{k} -(p_i * log p_i)`
	- k is the number of classes
- in each step, we select a node to split the data
- for each feature, calculate its information gain
	- if a feature has three possible values, then it will generate three child nodes
		- for each child node, calculate the entropy
	- calculate the weight mean of the entropy of the child nodes
	- optionally, calculate the `information gain = entropy(parent) - entropy(children)`
- pick the feature with the highest information gain
## explain random forest
- build multiple decision tree, but for each decision tree:
	- randomly select m records with replacement
	- randomly select a subset of features (usually sqrt(n)), and only pick the best features to split 
- take average of the predictions made by the trees
	- QUESTION: do voting or averaging?
		- for `predict_proba`, do averaging
		- for `predict`, That is, the predicted class is the one with highest mean probability estimate across the trees.
## explain gradient boosting
- the next weak learner learn from mistakes of ALL previous learners


## time series problems
- what is a stationary time series?
	- It is stationary when the variance and mean of the series are constant with time. 
	- 

- feature importance of linear regression with standardized features may still have problems because:
	- features with higher variance are given higher value of coefficients
	- the features may have multicollinarity


- TPR = P(flag as positive | positive)
	- TP / (TP + FN)
- FPR = P(flag as positive | negative)
	- FP / (TN + FP)
- TNR = P(flag as negative | negative)
	- TN / (TN + FP)
- FNR = P(flag as negative | positive)
	- FN / (TP + FN)
	

# regression
## autocorrelation
- if error terms not independent (i.e., have autocorrelation, or the y's are not indp), then 
	- estimated parameters still unbiased, but not the most efficient
	- MSE may underestimate the variance of the error
	- Variances of least squares estimators under the usual model tend to under-estimate the true variances of these estimators when autocorrelation is present. This, in turn, will over-state the statistical significance of some regression parameters.
	- Confidence interval estimates and hypothesis tests (based on t- or F-distribution) for regression parameters are no longer reliable.





## ensemble
- doing aggregating on multiple models can reduce both bias and variance
- question: how to find feature importance for ensemble?
	- how to use SHAP for ensemble
	- https://github.com/slundberg/shap/issues/457
	- It turns out that a linear combination of models over the same set of input features has SHAP values that are the same linear combination of the SHAP values of each individual model (linearity is one of the properties of Shapley values). So yes, if you are really just linearly combining the output of the model (in the log odds space if you are doing classification), then just use the same weighted average of each of the individual explanations.
- question: when doing ensemble for classification, if all the models can output probability, would it be better to do average of the probability than doing voting?
- `hard voting and soft voting`: If all classifiers are able to estimate class probabilities (i.e., if they all have a predict_proba() method), then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers. This is called soft voting. It often achieves higher performance than hard voting because it gives more weight to highly confident votes. All you need to do is set the voting classi‐ fier’s voting hyperparameter to "soft", and ensure that all classifiers can estimate class probabilities. This is not the case for the SVC class by default, so you need to set its probability hyperparameter to True (this will make the SVC class use cross-validation to estimate class probabilities, slowing down training, and it will add a predict_proba() method).
- `bagging`: bootstrap aggregating. Train the same model, but trained of different subset of the train set
	- with replacement
- `pasting`: without replacement
	- In other words, both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor.
- `random patches`: sample both train set and features
- `random subspace`: sample only features
- `boosting`: hypothesis boosting. 
	- sequentially build models, not in-parallel!
	- (i+1)th model tries correct on ith model
	- The general idea of most boosting methods is to train predictors sequentially, each trying to correct its prede‐ cessor. There are many boosting methods available, but by far the most popular are AdaBoost13 (short for adaptive boosting) and gradient boosting. 
- `ada boost`: One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfit. This results in new predictors focusing more and more on the hard cases. This is the technique used by AdaBoost.
	- For example, when training an AdaBoost classifier, the algorithm first trains a base classifier (such as a decision tree) and uses it to make predictions on the training set. The algorithm then increases the relative weight of misclassified training instances. Then it trains a second classifier, using the updated weights, and again makes predic‐ tions on the training set, updates the instance weights, and so on
- The AdaBoost algorithm works as follows:
	- Initialize the weights of the training examples to be equal.
	- Train a weak classifier on the training data.
	- Calculate the error rate of the weak classifier on the training data.
	- Increase the weights of the misclassified examples and decrease the weights of the correctly classified examples.
	- Repeat steps 2-4 for a specified number of iterations or until the error rate is below a certain threshold.
	- Combine the weak classifiers into a strong classifier using a weighted sum.
- The final strong classifier is a weighted sum of the weak classifiers, where the weights are determined by the accuracy of each weak classifier. The strong classifier is able to classify new examples by taking a weighted sum of the predictions of the weak classifiers.


## feature importance for tree-based model: mean decrease in impurity (MDI)
- 
Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature reduce impurity on average, across all trees in the forest. More precisely, it is a weighted average, where each node’s weight is equal to the number of training samples that are associated with it

covariates

# what is generative model
- is a model of the conditional probability P(X|y=y)
	- basically, we want to model the data distribution of each class 
- there are two purposes of modelling the conditional probability P(X|y):
	- generate new data from each class
	- do classification, but we additionally need the prior of y, p(y), to do so.
- A generative model in machine learning is a type of model that can generate new data that is similar to the training data. It is based on the idea of modeling the underlying probability distribution of the data, which can then be used to generate new data that is similar to the training data.
- Generative models can be used for a wide range of applications, including image and video synthesis, text generation, and music composition. Some popular generative models include:
- Generative Adversarial Networks (GANs): GANs are a type of neural network that consists of two parts: a generator and a discriminator. The generator generates new data, while the discriminator tries to distinguish between the generated data and the real data. The two parts are trained together, with the goal of improving the quality of the generated data over time.
- Variational Autoencoders (VAEs): VAEs are a type of neural network that can learn a compressed representation of the input data. The compressed representation can then be used to generate new data that is similar to the training data.
- Autoregressive models: Autoregressive models are a type of generative model that can generate new data by predicting the next value in a sequence based on the previous values. For example, a language model can generate new text by predicting the next word in a sentence based on the previous words.
- Generative models have several advantages over other types of models. First, they can be used to create new data that is similar to the training data, which can be useful for data augmentation and other applications. Second, they can be used to generate new ideas and insights that may not be apparent from the original data. Finally, they can be used to create new art and music, which can be a valuable creative tool for artists and musicians.


# how to generate data from a probability distribution
- use inverse transform
- https://stats.stackexchange.com/questions/307686/generating-data-from-arbitrary-distribution
- disadvantage: need to know the inverse of cdf

# what is a transformer
- uses `self-attention` to capture long-range dependencies in the input sequence
- A transformer is a type of neural network architecture that was introduced in a 2017 paper by Vaswani et al. called "Attention Is All You Need". The transformer architecture is based on the idea of self-attention, which allows the model to weigh the importance of different parts of the input sequence when making predictions.
The transformer architecture consists of an encoder and a decoder. The encoder takes an input sequence and generates a sequence of hidden states, while the decoder takes the hidden states and generates an output sequence. The key innovation of the transformer architecture is the use of self-attention mechanisms in both the encoder and the decoder.
Self-attention allows the model to weigh the importance of different parts of the input sequence when making predictions. This is done by computing a weighted sum of the input sequence, where the weights are determined by the similarity between each input element and the current element being processed. The similarity is computed using a learned attention function.
The transformer architecture has several advantages over other neural network architectures. First, it is able to handle variable-length input sequences, which is important for natural language processing tasks. Second, it is able to capture long-range dependencies in the input sequence, which is important for tasks such as machine translation. Finally, it is computationally efficient and can be easily parallelized, which makes it well-suited for large-scale applications.
The transformer architecture has been shown to achieve state-of-the-art performance on a wide range of natural language processing tasks, including machine translation, language modeling, and text classification. Some popular implementations of the transformer architecture include the Transformer model in the original paper, BERT (Bidirectional Encoder Representations from Transformers), and GPT-2 (Generative Pre-trained Transformer 2).

# what is a ML model with high variance
- there is a large performance difference between prediction on train set and on test set. That is, it can perform well on the train set, but poorly on test set, suggesting overfitting
- A machine learning model with high variance is a model that is overfitting to the training data. Overfitting occurs when a model is too complex and captures noise in the training data, rather than the underlying patterns. As a result, the model performs well on the training data but poorly on new, unseen data.
- A high variance model is characterized by a large gap between the training error and the validation error. The training error is the error rate of the model on the training data, while the validation error is the error rate of the model on a separate validation dataset. If the training error is much lower than the validation error, it is an indication that the model is overfitting to the training data.
- Some common causes of high variance in machine learning models include:
	- Using a complex model: Models that are too complex, such as deep neural networks with many layers, are more prone to overfitting.
	- Insufficient training data: If the training data is too small, the model may not be able to capture the underlying patterns and may overfit to the noise in the data.
	- Lack of regularization: Regularization techniques, such as L1 and L2 regularization, can help prevent overfitting by adding a penalty term to the loss function.
	- Incorrect hyperparameters: Hyperparameters, such as the learning rate and the number of hidden units in a neural network, can have a significant impact on the performance of the model. If the hyperparameters are not tuned correctly, the model may overfit to the training data.

# what is a ML model with high bias
A machine learning model with high bias is a model that is underfitting to the training data. Underfitting occurs when a model is too simple and is not able to capture the underlying patterns in the data. As a result, the model performs poorly on both the training data and new, unseen data.
A high bias model is characterized by a small gap between the training error and the validation error, with both errors being high. The training error is the error rate of the model on the training data, while the validation error is the error rate of the model on a separate validation dataset. If both errors are high, it is an indication that the model is underfitting to the training data.
Some common causes of high bias in machine learning models include:
	- Using a simple model: Models that are too simple, such as linear regression models with few features, may not be able to capture the underlying patterns in the data.
	- Insufficient training data: If the training data is too small, the model may not be able to capture the underlying patterns and may underfit to the data.
	- Incorrect features: If the features used in the model are not relevant to the problem, the model may not be able to capture the underlying patterns in the data.
	- Incorrect hyperparameters: Hyperparameters, such as the learning rate and the number of hidden units in a neural network, can have a significant impact on the performance of the model. If the hyperparameters are not tuned correctly, the model may underfit to the training data.
To address high bias in machine learning models, it is important to use techniques such as feature engineering, adding more features, and increasing the complexity of the model. It is also important to ensure that the model has sufficient training data and that the data is representative of the underlying distribution.


# explain the different kinds of join
- example: 
	- left table: customers
	- right table: purchase records made by customers
- left join: return all customers and the matching purchase records that each customer made. Those customers who have not made any purchases will also be shown, but the attributes from the right table will be NULL.
- right join: return all purchase records and the matching customer records with the purchase. 
- inner join: return only the rows with the matching values in both table
- outer join: A full outer join returns all the rows from both tables, including the rows that do not have a match in the other table. If there is no match in one of the tables, the result will contain NULL values for the columns of the other table.
- cross join: A cross join returns the Cartesian product of the two tables, which means that it returns all possible combinations of rows from both tables. It does not require a common column between the tables.
- natural join: the "matching" is implicitly defined as those common columns of the two tables. By default, it's inner join. Can change to left or outer join too.



