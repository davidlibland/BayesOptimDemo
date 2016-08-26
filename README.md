# Bayesian Optimization Demo
A short demonstration where bayesian optimization is used to make a good choice of hyperparameters.

Consider the following data set (pictured at two separate scales):

![Scatter Plot of Data](readme_imgs/data_scatter_plot.png)

There seems to be a linear relationship between x and y, and the y-values seem to concentrate near x=0 and disperse for large values of x.
We want to model the data near x=0 via the following model

![y=ax+b+\epsilon](latex_imgs/y=ax+b.png)

where ![\epsilon](latex_imgs/epsilon.png) is noise which depends on x.
Notice that there are some extreme outliers, so using a least-squares approach doesn't lead to a good fit:

![unregularized_fit](readme_imgs/lst_sqr.png)

We need ![\epsilon](latex_imgs/epsilon.png) to *heavy-tailed*; so we fit a student t distribution (where the mode, scale, and shape all depend on x) using gradient descent.

Of course, this is a toy problem, which we are playing with because it is simple to visualize; this tutorial is really about Bayesian optimization:
The challenge is that we won't acheive a good fit without proper regularization, and we then need to choose hyperparameters ![\lambda_1,\dots,\lambda_n](latex_imgs/lambdas.png) to control the regularization. For any given choice of hyperparameters, we can fit our model on a training subset of the data, and then evaluate the fit on a cross-validation subset of data leading to an error function:

![\varepsilon_{CV}(\lambda_1,\dots,\lambda_n):=\textrm{CrossValidationError}(\lambda_1,\dots,\lambda_n)](latex_imgs/varepsdef.png)

which we want to minimize. To minimize this we could use:

1. A grid search for optimal values of ![\lambda_1,\dots,\lambda_n](latex_imgs/lambdas.png),
2. A random search for optimal values of ![\lambda_1,\dots,\lambda_n](latex_imgs/lambdas.png),
3. Numerical Optimization (such as Nelder-Mead),
4. Bayesian Optimization.

Note that sampling ![\varepsilon_{CV}](latex_imgs/varepsilon.png) at a choice of hyperparameters can be costly (since we need to fit our model each time we sample); so rather than sampling ![\varepsilon_{CV}](latex_imgs/varepsilon.png) either randomly or on a grid, we'd like to make informed decisions about the best places at whcih to sample ![\varepsilon_{CV}](latex_imgs/varepsilon.png). Numerical Optimization and Bayesian Optimization both attempt to make these informed decisions, and we focus on Bayesian Optimization in this tutorial.

The basic idea is as follows: we will sample ![\varepsilon_{CV}](latex_imgs/varepsilon.png) at a relatively small number of points, and then fit a gaussian process to that sample: i.e. we model the function ![\varepsilon_{CV}(\lambda_1,\dots,\lambda_n)](latex_imgs/varepsilonatlambdas.png) (pictured in red):

![Gaussian Process fit to two samples](readme_imgs/gp1.png)

This model give us estimates of both 

1. the expected (mean) value of ![\varepsilon_{CV}](latex_imgs/varepsilon.png) if we were to sample it at novel points (pictured in green), as well as
2.  our uncertainty (or expected deviation) from that mean (the region pictured in grey),

and we use this information to choose where to sample ![\varepsilon_{CV}](latex_imgs/varepsilon.png) next. Now it is important to note that our primary concern is not to accurately model ![\varepsilon_{CV}](latex_imgs/varepsilon.png) *everywhere* with our gaussian process; our primary concern is to accurately model ![\varepsilon_{CV}](latex_imgs/varepsilon.png) near it's **minimums**. So we sample ![\varepsilon_{CV}](latex_imgs/varepsilon.png) at points where we have the greatest *expected improvement* of fitting our model to the minimums of ![\varepsilon_{CV}](latex_imgs/varepsilon.png):

![Gaussian Process fit to three samples](readme_imgs/gp2.png)

and we repeat until our model fits ![\varepsilon_{CV}](latex_imgs/varepsilon.png) accurately enough near it's minimums:

![Gaussian Process fit to four samples](readme_imgs/gp3.png)

Finally, we use the resulting model to make an optimal choice for our hyperparameters ![\lambda_1,\dots,\lambda_n](latex_imgs/lambdas.png).

This leads to a much better fit (green is the probability density, purple is one standard deviation - only when defined):

![regularized_fit](readme_imgs/reg.png)

The full tutorial can be found in the jupyter notebook `RegressionWithBayesOpt.ipynb`.