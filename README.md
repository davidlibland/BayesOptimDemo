# Bayesian Optimization Demo
A short demonstration where bayesian optimization is used to make a good choice of hyperparameters.

Consider the following data set (pictured at two separate scales):
![Scatter Plot of Data](data_scatter_plot.png)
There seems to be a linear relationship between $$x$$ and $$y$$, and the $$y$$-values seem to concentrate near $$x=0$$ and disperse for large values of $$x$$.
We want to model the data near $$x=0$$ via the following model

$$y = a x+b+\epsilon$$

where <blockquote style="color:rgba(0,0,0,255);color:rgb(0,0,0);font-size:36.00pt;"><!--latexit:AAAIWnjapVVdjBNVFL4Xltulhd3u8v8/y7YKyk93UcqPi+wPBYUtSLtL2basd2Zu
22Hnp5m5BcrYeKORF40xhqgxamR5EUL8ixpD9MXEGENIdCEmJiY++Gbig/JmjPFM
W6A7y5t30s6557vnzDnnnvtduaxrDo/FbuN589sWkLPnMpEzzHY0y8xELPk0U7hz
MkJtpaSBOh3hVrkb4QvvfxTuCbT3bOyNRB96eNPmRx7t27V7YPDgoeTRE5ncpKKW
dMvhYxGzouu3FgZDi5ZLJ5KpbVOs6kzAu+l3PKLo1HGuLO7oDHd1L1m6TMwT80Wb
WCCICIh2sVAEr6xYuWr1mrXr1m8QIbFIdIhO0SVWizVinVgvpIxMHaZrJhtTLN2y
04alsjGucZ1lyjajhqyzvEGLplbQFMohpbRKOZsMd8pUmSraVsVUhz3DCceq2ApL
s3O8FzXHrS1bQ9u2x8aSqeMHh3LJVH1hqkwVlohJ8KAwFotn+nc89vhEIw2TGizT
EJlzomkwvTN+V8wkU0freYfbZkJ79j5xEgrhcFsziyJ8DBztexICS6ZGK5xC3Kk6
cnnf/jjYNSa3hkLDIwcS9+YQ1CAHSa5w5ohusUosmexqi+VUS6kYzOT1ULJ9sTLP
u9TmmqKzWjBXcRgkMUWLLAuiF7STd+vlq0lR0KhSwbLhZ3Kprp1lwS1O9QF3h2Zu
2amZtbxbZJbBuF2d5dilhmNQXqpF/VrPrTNX7VQNeY4HXjJ84fLCrryrmWXI11Qa
0RYqusQtiVfLTFI1G6qrV0Ggiq1BwpJSojZVOPRzMJhzGDSGWeSlXJlC/VQokQvF
AUcp8FBiNpM0RzItqYHVGyYYjG79HyMYlYYrDrcM7TwUdtgyDGqqTjAK0ZjsrNKY
uzldrrnZXE+2Nlttgzqf68n71MPQrLVsX96zk3r7vGU+d7Tm/ZnF+obP8ugh9oMQ
owCIt2kFm075MVlugrI8B7prB4IPUpqIQnUfZJkAWWVmU27ZXgv6cd3DgW+8s+3D
jmhMve/YPVLz4VS9F5ALsg8dbEEH56CspQYu86OtFXILfrTYihb9aKkVLc2xtb18
TRfe/mxMswF5gg9zoNtr9S1wcwcpKL0FNmtdAtl6C6xzLuQKja4y6XiJGnBCCgXo
d5NrtM7PTx0evfS0WHrpiFgmVo4nUwk4p7ePPXM8lEqPjcM8Bf0LvFNI6LTowDwJ
e9a7f1ODKcNhsVysGB+1TKpYQGQnJ5oeprPxpgTa/CngqxFN8U4VtavT+Tgon6WT
4e4WGlMbzDYtxx+o790/1PjkDAsVit6NwjUIZPBr67nNhzN3xFpwqZ2Gb44AzU9P
xZuSR+XG9liiT4IHicUekZYPJMQGjzIzOVZ2NN0ywbZyZjK85B4F3w/2ciUfr5Pz
YVZl6mDzMvyxej5tWxYXGLWjLrQa9aKtqB/FURZRVEJl5KIX0CvoVfQaeh1dRG+g
t9Db6B30HvoAXUXX0IfoY/Qp+gx9gb5E19FX6Ft0A/2AfkK/oj/QX7gNd+KVeAOO
4n68G+/FA3gIj+JxfApTrGEbc1zFL+KX8AX8Mn4XX8JX8Sf4Ov4Of49v4JsEk3YS
JItIBwmTbrKKbCSbST/ZTYbICEmQQ+QoSZMsOUUYOU3OkvPkeXKRvEmukc/JdfIN
uUFukhnyM/mF/EZ+J3+SO+Rv8g/5N4ACCwMdgUhgU2BPYCDQ3Ip5uHlhumjWCIz8
BxPe2V8=
--><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle><mrow><mtable columnspacing="0.167em" columnalign="right center left" displaystyle="true"><mrow><mi>&#x3B5;</mi></mrow></mtable></mrow></mstyle></math></blockquote> $$\epsilon$$ is noise which depends on $$x$$.
Notice that there are some extreme outliers, so using a least-squares approach doesn't lead to a good fit. we need $$\epsilon$$ to *heavy-tailed*; so we fit a student t distribution (where the mode, scale, and shape all depend on $$x$$) using gradient descent.

Of course, this is a toy problem, which we are playing with because it is simple to visualize; this tutorial is really about Bayesian optimization:
The challenge is that we won't acheive a good fit without proper regularization, and we then need to choose hyperparameters $$\lambda_1,\dots,\lambda_n$$ to control the regularization. For any given choice of hyperparameters, we can fit our model on a training subset of the data, and then evaluate the fit on a cross-validation subset of data leading to an error function:

$$\varepsilon_{CV}(\lambda_1,\dots,\lambda_n):=\textrm{CrossValidationError}(\lambda_1,\dots,\lambda_n)$$

which we want to minimize. To minimize this we could use:

1. A grid search for optimal values of $$\lambda_1,\dots,\lambda_n$$,
2. A random search for optimal values of $$\lambda_1,\dots,\lambda_n$$,
3. Numerical Optimization (such as Nelder-Mead),
4. Bayesian Optimization.

Note that sampling $$\varepsilon_{CV}$$ at a choice of hyperparameters can be costly (since we need to fit our model each time we sample); so rather than sampling $$\varepsilon_{CV}$$ either randomly or on a grid, we'd like to make informed decisions about the best places at whcih to sample $$\varepsilon_{CV}$$. Numerical Optimization and Bayesian Optimization both attempt to make these informed decisions, and we focus on Bayesian Optimization in this tutorial.

The basic idea is as follows: we will sample $$\varepsilon_{CV}$$ at a relatively small number of points, and then fit a gaussian process to that sample: i.e. we model the function $$\varepsilon_{CV}(\lambda_1,\dots,\lambda_n)$$ (pictured in red):

![Gaussian Process fit to two samples](gp1.png)

This model give us estimates of both 

1. the expected (mean) value of $$\varepsilon_{CV}$$ if we were to sample it at novel points (pictured in green), as well as
2.  our uncertainty (or expected deviation) from that mean (the region pictured in grey),

and we use this information to choose where to sample $$\varepsilon_{CV}$$ next. Now it is important to note that our primary concern is not to accurately model $$\varepsilon_{CV}$$ *everywhere* with our gaussian process; our primary concern is to accurately model $$\varepsilon_{CV}$$ near it's **minimums**. So we sample $$\varepsilon_{CV}$$ at points where we have the greatest *expected improvement* of fitting our model to the minimums of $$\varepsilon_{CV}$$:

![Gaussian Process fit to three samples](gp2.png)

and we repeat until our model fits $$\varepsilon_{CV}$$ accurately enough near it's minimums:

![Gaussian Process fit to four samples](gp3.png)

Finally, we use the resulting model to make an optimal choice for our hyperparameters $$(\lambda_1,\dots,\lambda_n)$$.

The full tutorial can be found in the jupyter notebook `RegressionWithBayesOpt.ipynb`.