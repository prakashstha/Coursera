# Probability

- Study of uncertainty

- Can be thought as

	- fraction of times an event occurs

	- degree of belief about an event

- Quantifying uncertainty requires notion of *Random Variable*

	- a function that maps outcomes from a random experiment to a set of properties that we are interested in

	- Associated with random variable, there is a function that measures the probability that a particular outcome(s) will occur -- called *probability distribution*
- Probability Distributions are used as building block for other concepts -- probabilistic modeling, graphical modeling and model selection.

- Theory of Probability aims at defining mathematical structure to describe random outcomes of experiments.
- In ML and statistics, there are two major interpretations of probability
	- **Bayesian Interpretation** 
		-  the degree of uncertainty that the user has about an event
		- sometimes referred as "subjective probability" or "degree of belief"
	- **Frequentist Interpretation**
		- the relative frequencies of events of interest to the total number of events that occurred

## Concepts
- **Sample Space** $\Omega$
	- a set of all possible outcomes of the experiment
	- two consecutive coin tosses have a sample space of $\text{\{hh, tt, ht, th\}}$
	- is referred to by different names -- "state space", "sample description space", "possibility space", "event space"
- **Event Space** $\mathcal{A}$
	- the collection of all possible events that can occur in a given experiment, essentially representing the set of all subsets of the sample space, where each subset is considered a potential event that can be assigned a probability

- **Probability of an Event**
	- With each event $A \in \mathcal{A}$, we associate a number $P(A)$ that measures the probability or degree of belief that the event will occur.
	- $P(A)$ is called probability of $A$, such that 
		- $0 \le P(A) \le 1$
		- $\sum P(A) = 1$
		


## Probability Theory and Statistics
-  Are often presented together, but they concern different aspects of uncertainty. 

-   **Probability Theory:** Focuses on modeling uncertainty in processes by defining random variables and applying probability rules to predict possible outcomes. It starts with a theoretical model of how events might happen.
    
-   **Statistics:** Centers on analyzing observed data to infer the underlying processes that generated it. Instead of starting with a theoretical model, it works backward from what has already happened to uncover explanations.
    
-   **Machine Learning:** Closely aligns with statistics in its goal to build models that describe or predict data patterns. It aims to generalize from observed data to capture the processes that generated it.
    
-   **Relationship Between Probability and Statistics:** Probability provides the mathematical foundation for constructing and evaluating models. In machine learning and statistics, probability rules are used to find models that best fit the data, enabling accurate predictions or insights into underlying processes.

## Discrete and Continuous Probabilities
- *Probability Mass Function (PMF)*, i.e., $P(X=x)$,  defines the probability that a **discrete random variable** $X$ takes a particular value $x \in \Large\tau$  
- When target space $\tau$ is continuous, it is more natural to specify the probability that a random variable $X$ is in interval, i.e., $P(a \le X \le b$ for $a<b$.
- *Cumulative Probability (CDF)*, i.e., $P(X \le x)$,  defines the probability that a **continuous random variable** $X$ is less than a particular value $x$

	| Type       | Point Probability                   | Interval Probability                         |
	|------------|-------------------------------------|----------------------------------------------|
	| Discrete   | $P(X=x)$  Probability Mass Function | NA                                           |
	| Continuous | $p(x)$ Probability Density Function | $P(X \le x$ Cumulative Distribution Function |

## Sum Rule, Product Rule, and Bayes' Theorem

### Sum Rule
- $p(x, y)$ &emsp; &emsp;&nbsp; &emsp; $\rightarrow$ the joint distribution of the two random variables $x, y$.
- $p(x)$ and $p(y)\rightarrow$ corresponding marginal distribution
- $p(y|x)$ &emsp; &emsp; &emsp;&emsp;$\rightarrow$  conditional distribution of $y$ given $x$
- *sum rule*, aka *marginalization property*, states that

$$
p(x) = 
\begin{cases}
	\mathop{\Large\sum}\limits_{y \in Y}p(x, y) & \text{if} ~y~\text{is descrete} \\
    \mathop{\Large\int}\limits_{y} p(x, y) \text{d}y & \text{if} ~y~\text{is continuous}
\end{cases}
$$

-  If $\mathbf{x} = [x_1, ..., x_D]^T$, we obtain marginal

$$p(x_i) = \mathop{\Large \int} p(x_1, ..., x_D) \text{d}x_{|i}$$

- where, we sum out all random variables except $x_i$
-   **Computational Challenges:** Probabilistic modeling faces significant difficulties due to the application of the sum rule.
-   **High-Dimensional Sums/Integrals:**
    -   The sum rule often requires calculating sums or integrals across many variables or states.
    -   This becomes computationally complex in high dimensions.
-   **No Efficient Solution:** There is no known polynomial-time algorithm to compute high-dimensional sums or integrals exactly, making these tasks computationally hard.

### Product Rule
- relates the join distribution to the conditional distribution via $p(x, y) = p(y|x)p(x)$
- says that every joint distribution of two random variables can be factorized

### Bayes' Theorem
- Bayes' theorem states
$$p(x|y) = \dfrac{p(y|x)p(x)}{p(y)}$$

- which is direct consequence of the product rule

$$p(x, y) = p(x|y)p(y)~~\text{and}~~p(x, y) = p(y|x)p(x)$$
	 
$$⇒ p(x|y)p(y) = p(y|x)p(x) \rightarrow p(x|y) = \dfrac{p(y|x)p(x)}{p(y)}$$

- where 
	- $p(x)$ 
		- is the prior, which encapsulates our subjective prior knowledge of the unobserved (latent) variable $x$ before observing any data
		- we can choose any prior that makes sense, but we need to ensure that the prior has non-zero pdf (or pmf) on all plausible $x$, even if they are rare
	- $p(y|x)$ 
		- is the likelihood that describes how $x$ and $y$ are related.
		- note that the likelihood is the distribution in $y$ and we call it the "likelihood of $x$ ( given $y$)" or the "probability of $y$ given $x$
	- $p(x|y)$ 
		- is the posterior, the quantity of interest in Bayesian statistics -- what we know about $x$ after having observed $y$
	- The quantity
	
    $$p(y) := \mathop{\Large \int} p(y|x)p(x) \text{d}x = \mathop{\mathbb{E}_x}[p(y|x)]$$ 
    
    - is the *marginal likelihood/evidence*
	- By definition, the marginal likelihood integrates the numerator w.r.t. $x$, so it is independent of $x$
	- And it ensures that the posterior $p(x|y)$ is normalized.
	- It also plays an important role in Bayesian model selection

- allows us to invert the relationship between $x$ and $y$ given by the likelihood ⇒ sometimes called *probabilistic inverse*

## Summary Statistics and Independence

### Means and Covariance

-   Means and covariance are valuable for describing the characteristics of probabilistic distributions, such as their expected values and variability.
-   The expected value is a fundamental concept in probability and plays a central role in machine learning, with many foundational principles derived from it.

- **Expected Value:** The expected value of a function $g:\mathop{\mathbb{R}}\rightarrow\mathop{\mathbb{R}}$ of a univariate random variable $X \sim p(x)$ is given by
$$\mathop{\mathbb{E}}[g(x)] = 
\begin{cases}
	\int_\mathcal{X} g(x)p(x) \text{d}x & \text{if} ~x~\text{is continuous} \\
	\sum_\mathcal{X} g(x)p(x) & \text{if} ~x~\text{is discrete} 
\end{cases}
$$
 
- where $\mathcal{X}$ is the set of possible outcomes (the target space) of random variable $X$

- For multivariate random variables $[X_1, ..., X_D]^T$, we define expected value element wise 
$$\mathbb{E}[g(x)] = \begin{bmatrix}  
\mathbb[E]_{X_1}[g(x_1)] \\  
⋮\\
\mathbb[E]_{X_D}[g(x_D)]
\end{bmatrix}
 $$
	
- where 
    - the subscript $\mathbb{E}_{X_d}$ indicates that we are taking the expected value wrt the $d$th element of the vector $x$.

    - and, operator $\mathbb{E}_X$ acts as operator indicating that we should take integral wrt probability density (continuous distribution) or the sum over all states (for discrete distributions).

* **Mean**
	- Definition of *mean* is a special case of the expected value, obtained by choosing *g* as an identity function. 
	- The **mean** of a random variable $X$ with states $x \in \mathbb{R}^{D}$ is an average and is defined as 

$$\mathbb{E}_X[x] = 
\begin{bmatrix}
\mathbb{E}_{X_1}[x_1]\\
⋮\\
\mathbb{E}_{X_D}[x_D]
\end{bmatrix} \in \mathbb{R}^{D}
$$	

where 

$$\mathbb{E}_{X_d}[x_d] = 
\begin{cases}
	\int_\mathcal{X} x_dp(x_d) \text{d}x_d & \text{if} ~X~\text{is continuous} \\
	\sum_\mathcal{X} x_dp(x_d) & \text{if} ~X~\text{is discrete} 
\end{cases}
$$



### Median and Mode
- There are two notions of "average" --*median and mode*
- **Median** is the "middle" value if we sort the values, i.e., 50% of the values are greater than the median and 50% are smaller.
- For continuous values, median is the value where the **cdf** is 0.5.
- Median is more robust to outliers than the mean. 
- However, generalization of median to higher dimension is non-trivial as there is no obvious way to sort in more than one dimension.
- **Mode** is most frequently occurring value 
	- for discrete random variable, it is the value with highest frequency of occurrence
	- for continuous random variable, it is defined as a peak in the density $p(x)$


### Covariance 

**Covariance (Univariate)**
- the covariance between two univariate random variable $X, Y \in \mathbb{R}$ is given the expected product of their deviations from their respective means, i.e., 

$$\text{Cov}_{X, Y}[x, y]:=\mathbb{E}_{X, Y}[(x - \mathbb{E}_x[x])(y - \mathbb{E}_Y[y])]$$

- By using the lineary of expectations, the equation can be rewritten as 

$$\text{Cov}[x, y] = \mathbb{E}[xy]  - \mathbb{E}[x]\mathbb{E}[y]$$

- The covariance of a variable with itself is called *variance* $\mathbb{V}_X[x]$
- the square root of variance is called standard deviation $\sigma(x)$

**Covariance (Multivariate)**
- If we consider two multivariate random variables $X$ and $Y$ with states $x \in \mathbb{R}^{D}$ and $y \in \mathbb{R}^{E}$ respectively, the covariance between $X$ and $Y$ is defined as

$$\text{Cov}[x, y] = \mathbb{E}[xy^T] - \mathbb{E}[x]\mathbb{E}[y]^T = \text{Cov}[y, x]^T \in \mathbb{R}^{D \times E}$$


**Variance**
- The Covariance definition above can be applied to the same multivariate random variabe, that describes the relation between individual dimensions of the random variable.

- The variance of a random variable $X$ with states $x \in \mathbb{R}^D$ and a mean vector $\boldsymbol{\mu} \in \mathbb{R}^D$ is defined as 
$$
\begin{align*}  
\mathbb{V}_X[\boldsymbol{x}] &= \text{Cov}_X[\boldsymbol{x}, \boldsymbol{x}] \\  
 &= \mathbb{E}[(\boldsymbol{x} - \boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^T] = \mathbb{E}_X[\boldsymbol{x}\boldsymbol{x}^T] - \mathbb{E}_X[\boldsymbol{x}]\mathbb{E}_X[\boldsymbol{x}]^T
\\
 &=
 \begin{bmatrix}
\text{Cov}[x_1, x_1] & \text{Cov}[x_1, x_2] ~~... ~~ \text{Cov}[x_1, x_D]\\
\text{Cov}[x_2, x_1] & \text{Cov}[x_2, x_2] ~~... ~~ \text{Cov}[x_2, x_D]\\
⋮\\
\text{Cov}[x_D, x_1] & \text{Cov}[x_D, x_2] ~~... ~~ \text{Cov}[x_D, x_D]
 \end{bmatrix}
\end{align*}
$$

- which is called the covariance matrix of the multivariate random number $X$
- is symmetrical and positive semidefinite
- its diagonal contains variances of the marginals

**Correlation**
- The *correlation* between two random variables $X, Y$ is given by 

$$\text{corr}[x, y] = \dfrac{\text{Cov}[x, y]}{\sqrt{\mathbb{V}[x]\mathbb{V}[y]}} \in [-1, 1]$$

- Thus, correlation matrix is the covariance matrix of standardized random variables $\dfrac{x}{\sigma({x})}$
- Both covariance and correlation indicate how two random variables are related

### Empirical Means and Covariances

- Means and covariance considered so far are true statistics for the population and are often also called *population mean and covariance*
- In ML, we need to learn these statistics from empirical observation of data, referred to as *empirical means and covariances*
- Empirical mean vector is the arithmetic average of the observations for each variable, and it is defined as

$$\bar{x} := \mathop{\large\sum}\limits_{n = 1}^{N}\boldsymbol{x}_n$$

- where $\boldsymbol{x} \in \mathbb{R}^D$
- Empirical covariance matrix is a $D \times D$ matrix 

$$\Sigma := \dfrac{1}{N}\mathop{\large\sum}\limits_{n = 1}^{N} (\boldsymbol{x}_n - \bar{\boldsymbol{x}})(\boldsymbol{x}_n - \bar{\boldsymbol{x}})^T$$

- Empirical covariance matrices are symmetric and positive semidefinite.


### Three Expressions for the Variance

**1: Standard Definition**
- The standard definition of variance is expectation of squared deviation of a random variable $X$ from its expected value 

    $\mathbb{V}_{X}[x] = \mathbb{E}_X[x - \mu]^2$ where $\mu = \mathbb{E}_X[x]$

- Thus, the variance is the mean of a new random variable $Z:=(X - \mu)^2$
- When estimating the variance empirically, we need to resort to a two-pass algorithm -- one pass through the data to calculate the mean \mu and, then a second pass using this estimate $\hat{\mu}$ calculate the variance.

- To estimate variance empirically, a two-pass algorithm is required: the first pass calculates the mean ($\mu$), and the second pass uses this estimate ($\hat{\mu}$) to compute the variance.

**Raw Score Formulat**
- Rearranging the terms in the standard definition allows us to derive the raw-score formula for variance, eliminating the need for two passes.

$$\mathbb{V}_X[x] = \mathbb{E}_X[x^2] - (\mathbb{E}_X[x])^2$$
- which is basically "the mean of the square minus the square of the mean"
- Variance can be computed in a single pass through the data by simultaneously accumulating $x_i$ and $x_i^2$
- Unfortunately, this approach may suffer from numerical instability. 
- Despite this, the raw-score formula for variance is valuable in machine learning, particularly for deriving the bias-variance decomposition.


**Sum of Pairwise Differences**
- Variance can be understand as the sum of pairwise differences between all pairs of observations.
- Consider a sample $x_1, ..., x_N$ of realizations of a random variable $X$
- We can compute squared difference between pairs of $x_i$ and $x_j$
- By expanding the square, we can show that the sum of $N^2$ pairwise differences is the empirical variance of the observation

$$\dfrac{1}{N^2} \mathop{\large\sum}\limits_{i, j = 1}^{N} (x_i - x_j)^2 = 
2  \left [ \dfrac{1}{N}\mathop{\large\sum}\limits_{i = 1}^{N} x_i- \left (\dfrac{1}{N}\mathop{\large\sum}\limits_{i = 1}^{N}x_i \right )^2  \right ]$$

- Thus, by computing the mean (N terms in the summation) and computing variance (again N terms in the summation)
, we can obtain an expression  that has $N^2$ terms.

### Sums and Transformations of Random Variables

- Consider two random variables $X, Y$ with states $x, y \in \mathbb{R}^D$. Then

    $\mathbb{E}[x + y] = \mathbb{E}[x] + \mathbb{E}[y]$

    $\mathbb{E}[x - y] = \mathbb{E}[x] - \mathbb{E}[y]$

    $\mathbb{V}[x + y] = \mathbb{V}[x] + \mathbb{V}[y] + \text{Cov}[x, y] + \text{Cov}[y, x]$

    $\mathbb{V}[x - y] = \mathbb{V}[x] + \mathbb{V}[y] - \text{Cov}[x, y] - \text{Cov}[y, x]$


- Mean and (co)variance exhibit some useful properties when it comes to affine transformation of random varaibles
- Consider a random variable $X$ with 
    - mean $\boldsymbol{\mu}$  
    - covariance matrix $\Sigma$  
    - a determinisitc transformation $\boldsymbol{y} = A\boldsymbol{x} + \boldsymbol{b}$
- $\boldsymbol{y}$ itself is a random variable whose mean vector and covariace matrix are given by 

    $\mathbb{E}_Y[\boldsymbol{y}] = \mathbb{E}_X[\boldsymbol{Ax} + \boldsymbol{b}] = \boldsymbol{A}\mathbb{E}_X[\boldsymbol{x}]+ \boldsymbol{b} = \boldsymbol{A}\boldsymbol{\mu}+ \boldsymbol{b} $

    $\mathbb{V}_Y[\boldsymbol{y}] = \mathbb{V}_X[\boldsymbol{Ax} + \boldsymbol{b}] = \mathbb{V}[\boldsymbol{Ax}] =  \boldsymbol{A}\mathbb{V}[\boldsymbol{x}]\boldsymbol{A}^T =  \boldsymbol{A}\Sigma\boldsymbol{A}^T$

- Furthermore, 

$$
    \begin{align*}
    \text{Cov}[x, y] &= \mathbb{E}[x(Ax + b)^T] - \mathbb{E}[x]\mathbb{E}[Ax + b]^T\\
    &= \mathbb{E}[x]b^T + \mathbb{E}[xx^T]A^T - \mu b^T - \mu\mu^TA^T\\
    &=\mu b^T - \mu b^T +(\mathbb{E}[xx^T]  - \mu\mu^T)A^T\\
    &= \Sigma \boldsymbol{A}^T
    \end{align*}
    $$


### Statistical Independence

- Two random variables X, Y are statistically independent if and only if

$$p(x, y) = p(x)p(y)$$

- If $X$ and $Y$ are statistically independent then

    - $p(y|x) = p(y)$
    - $p(x|y) = p(x)$
    - $\mathbb{V}[x + y] = \mathbb[V][x] + \mathbb{V}[y]$
    - $\mathbb{V}_{X, Y}[x + y] = \mathbb[V]_X[x] + \mathbb{V}_Y[y]$
    - $\text{Cov}_{X, Y}[x, y] = 0$
- The last point may not hold in converse, i.e., $X$ and $Y$ can have zero covariance but still not be statistically independent.
- Covariance captures only linear dependence.
- Variables with non-linear dependencies can still exhibit zero covariance.


**Independent and Identically Distributed (i.i.d)**
- In ML, we often consider problems that can be modeled as i.i.d. random variables
- For more than two random variables, they are referred to as from iid if they are
    - *Independent* -- mutually independent, where all subsets are independent, and
    - *Identically Distributed* -- are from the same distribution

**Conditional Independence**
- Two random variables $X$ and $Y$ are conditionally independent given $Z$ if and only if

    $p(x, y|z) = p(x|z)p(y|z)~~\text{for all}~~z \in \mathbb{Z}$

- Denoted as $X \perp Y |Z\rightarrow$ $X$ is conditionally independent of $Y$ given $Z$
    - can be understood as "given knowledge about $z$, the distribution of $x$ and $y$ factorizes"

- Using product rule, we can write

    $p(x,y|z) = p(x|y, z)p(y|z)$

- Comparing above two equations, we get 

    $p(x|y, z) = p(x|z)$
- which gives alternative definition -- 'given that we know $z$, knowledge about $y$ does not change our knowledge of $x$'