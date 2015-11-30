# Hierarchical modeling course notes
Max Joseph  
November 16, 2015  


Chapter 1: Linear models
========================

## Big picture

This course builds on an understanding of the mechanics of linear models. 
Here we introduce some key topics that will facilitate future understanding hierarchical models.

#### Learning goals

<!---
I haven't added anything about interaction effects or centering covariates yet. 
Maybe this would fit well in the general linear model section?
Centering covariates might also be good in the linear regression section. 
Or, maybe it's worth deferring to Gelman and Hill on centering and streamlining the notes?
I also was not sure what we were thinking about varying structure and standard errors. 
Did you have something in mind here?
I have avoided talking much about standard errors yet because it's a fairly 
frequentist notion (sd of the sampling distribution)
-->

- linear regression with `lm`
- intercepts, "categorical" effects
- varying model structure to estimate effects and standard errors
- interactions as variation in slope estimates for different groups
- centering input variables and intepreting resulting parameters
- assumptions and unarticulated priors
- understanding residual variance (Gaussian)
- understanding all of the above graphically
- understanding and plotting output of lm
- notation and linear algebra review: $X\beta$

Linear regression, ANOVA, ANCOVA, and multiple regression models are all species cases of general linear models (hereafter "linear models"). 
In all of these cases, we have observed some response variable $y$, which is potentially modeled as a function of some covariate(s) $x_1, x_2, ..., x_p$.

## Model of the mean

If we have no covariates of interest, then we may be interested in estimating the population mean and variance of the random variable $Y$ based on $n$ observations, corresponding to the values $y_1, ..., y_n$. 
Here, capital letters indicate the random variable, and lowercase corresponds to realizations of that variable. 
This model is sometimes referred to as the "model of the mean". 


```r
# simulating a sample of y values from a normal distribution
y <- rnorm(20)
plot(y)
```

![A set of observed $y$ values, $n=20$.](main_files/figure-html/chunk1-1.png) 

We have two parameters to estimate: the mean of $Y$, which we'll refer to as $\mu$, and the variance of $Y$, which we'll refer to as $\sigma^2$. 
Here, and in general, we will use greek letters to refer to parameters. 
If $Y$ is normally distributed, then we can assume that the realizations or samples $y$ that we observe are also normally distributed: $y \sim N(\mu, \sigma^2)$. 
Here and elsewhere, the $\sim$ symbol represents that some quantity "is distributed as" something else (usually a probability distribution). 
You can also think of $\sim$ as meaning "is sampled from".
A key concept here is that we are performing statistical inference, meaning we are trying to learn about (estimate) population-level parameters with sample data. 
In other words, we are not trying to learn about the sample mean $\bar{y}$ or sample variance of $y$. 
These can be calculated and treated as known once we have observed a particular collection of $y$ values. 
The unknown quantities $\mu$ and $\sigma^2$ are the targets of inference. 

Fitting this model (and linear models in general) is possible in R with the `lm` function. 
For this rather simple model, we can estimate the parameters as follows:


```r
# fitting a model of the mean with lm
m <- lm(y ~ 1)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ 1)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -1.4899 -0.6403 -0.1107  0.8012  2.4723 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)
## (Intercept)  0.07801    0.22385   0.348    0.731
## 
## Residual standard error: 1.001 on 19 degrees of freedom
```

The summary of our model object `m` provides a lot of information. 
For reasons that will become clear shortly, the estimated population mean is referred to as the "Intercept". 
Here, we get a point estimate for the population mean $\mu$: 0.078 and an estimate of the residual standard deviation $\sigma$: 1.001, which we can square to get an estimate of the residual variance $\sigma^2$: 1.002.

## Linear regression

Often, we are interested in estimating the mean of $Y$ as a function of some other variable, say $X$. 
Simple linear regression assumes that $y$ is again sampled from a normal distribution, but this time the mean or expected value of $y$ is a function of $x$:

$$y_i \sim N(\mu_i, \sigma^2)$$

$$\mu_i = \alpha + \beta x_i$$

Here, subscripts indicate which particular value of $y$ and $x$ we're talking about. 
Specifically, we observe $n$ pairs of values: $(y_i, x_i), ..., (y_n, x_n)$, with all $x$ values known exactly.
Linear regression models can equivalently be written as follows:

$$y_i = \alpha + \beta x_i + \epsilon_i$$

$$\epsilon_i \sim N(0, \sigma^2)$$

Key assumptions here are that each of the error terms $\epsilon_1, ..., \epsilon_n$ are normally distributed around zero with some variance (i.e., the error terms are identically distributed), and that the value of $\epsilon_1$ does not affect the value of any other $\epsilon$ (i.e., the errors are independent).
This combination of assumptions is often referred to as "independent and identically distributed" or i.i.d. 
Equivalently, given some particular $x_i$ and a set of linear regression parameters, the distribution of $y_i$ is normal. 
A common misconception is that linear regression assumes the distribution of $y$ is normal. 
This is wrong - linear regression assumes that the error terms are normally distributed. 
The assumption that the variance $\sigma^2$ is constant for all values of $x$ is referred to as homoskedasticity. 
Rural readers may find it useful to think of skedasticity as the amount of "skedaddle" away from the regression line in the $y$ values. 
If the variance is changing across values of $x$, then the assumption of homoskedasticity is violated and you've got a heteroskedasticity problem. 


```r
# simulate and plot x and y values
n <- 50
x <- runif(n)
alpha <- -2
beta <- 3
sigma <- .4
y <- rnorm(n, mean = alpha + beta * x, sd = sigma)
plot(x, y)

# add known mean function 
lines(x = x, y = alpha + beta * x, col='blue')
legend('topleft', 
       pch = c(1, NA), lty = c(NA, 1), 
       col = c('black', 'blue'), 
       legend = c('Observed data', 'E(y | x)'), 
       bty = 'n')
```

![Simulated data from a linear regression model. The true expected value of y given x, E(y | x), is shown as a line.](main_files/figure-html/unnamed-chunk-11-1.png) 

The normality assumption means that the probability density of $y$ is highest at the value $\alpha + \beta x$, where the regression line is, and falls off away from the line according to the normal probablity density. 
This graphically looks like a bell 'tube' along the regression line, adding a dimension along $x$ to the classic bell 'curve'.

![Graphical depiction of the linear regression normality assumption. The probability density of y is shown in color. Higher probabilities are shown as more intense colors, and regions with low probabilities are lighter.](main_files/figure-html/unnamed-chunk-12-1.png) 

### Model fitting

Linear regression parameters $\alpha$, $\beta$, and $\sigma^2$ can be estimated with `lm`.
The syntax is very similar to the previous model, except now we need to include our covariate `x` in the formula (the first argument to the `lm` function). 


```r
m <- lm(y ~ x)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ x)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -1.14973 -0.26184 -0.03679  0.26521  0.94473 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -2.0750     0.1271  -16.32   <2e-16 ***
## x             3.1161     0.2099   14.85   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.438 on 48 degrees of freedom
## Multiple R-squared:  0.8212,	Adjusted R-squared:  0.8175 
## F-statistic: 220.4 on 1 and 48 DF,  p-value: < 2.2e-16
```

The point estimate for the parameter $\alpha$ is called "(Intercept)". 
This is because our estimate for $\alpha$ is the y-intercept of the estimated regression line when $x=0$ (recall that $y_i = \alpha + \beta x_i + \epsilon_i$).
The estimate for $\beta$ is called "x", because it is a coefficient associated with the variable "x" in this model. 
This parameter is often referred to as the "slope", because it represents the increase in the expected value of $y$ for a one unit increase in $x$ (the rise in y over run in x).
Point estimates for the standard deviation and variance of $\epsilon$ can be extracted as before (`summary(m)$sigma` and `summary(m)$sigma^2`).

### Centering and scaling covariates

Often, it's a good idea to "center" covariates so that they have a mean of zero ($\bar{x} = 0$). 
This is achieved by subtracting the sample mean of a covariate from the vector of covariate values ($x - \bar{x}$).
It's also useful to additionally scale covariates so that they are all on a common and unitless scale. 
While many will divide each covariate by its standard deviation, Gelman and Hill (pg. 57) recommend dividing by twice the standard deviation ($s_x$) so that binary covariates are transformed from $x \in \{0, 1\}$ to $x_t \in \{-0.5, 0.5\}$, where $x_t$ is the transformed covariate: $x_t = \frac{x - \bar{x}}{2 s_x}$. 
If covariates are not centered and scaled, then it is common to observe correlations between estimated slopes and intercepts. 

![Linear regression line of best fit with 95% confidence intervals for the line. Notice that if the slope is lower (the upper dashed line) then the intercept necessarily goes up, and if the slope is higher (the lower dashed line), the intercept must decrease.](main_files/figure-html/unnamed-chunk-14-1.png) 

So, we expect that in this case, the estimates for the intercept and slope must be negatively correlated.
This is bourne out in the confidence region for our estimates of $\alpha$ and $\beta$. 
Usually, people inspect univariate confidence intervals for parameters, e.g.,


```r
confint(m)
```

```
##                  2.5 %    97.5 %
## (Intercept) -3.5876915 -2.451254
## x            0.8371536  1.166814
```

This is misleading because our estimates for these parameters are correlated. 
For any given value of the intercept, there are only certain values of the slope that are supported.
To assess this possibility, we might also be interested in the bivariate confidence ellipse for these two parameters. 
We can evaluate this quantity graphically as follows with some help from the `car` package:


```r
library(car)
confidenceEllipse(m)
```

![](main_files/figure-html/unnamed-chunk-16-1.png) 

This is not great. 
We want to be able to directly use the univariate confidence intervals. 
Our problem can be solved by centering $x$:

![](main_files/figure-html/unnamed-chunk-17-1.png) 

Now there is no serious correlation in the estimates and we are free to use the univariate confidence intervals without needing to consider the joint distribution of the slope and intercept. 
This trick helps with interpretation, but it will also prove useful later in the course in the context of Markov chain Monte Carlo (MCMC) sampling. 

### Checking assumptions

We have assumed that the distribution of error terms is normally distributed, and this assumption is worth checking. 
Below, we plot a histogram of the residuals (another name for the $\epsilon$ parameters) along with a superimposed normal probability density so that we can check normality. 


```r
hist(resid(m), breaks = 20, freq = F, 
     main = 'Histogram of model residuals')
curve_x <- seq(min(resid(m)), max(resid(m)), .01)
lines(curve_x, dnorm(curve_x, 0, summary(m)$sigma))
```

![Simulated data from a linear regression model.](main_files/figure-html/unnamed-chunk-18-1.png) 

Even when the assumption of normality is correct, it is not always obvious that the residuals are normally distributed. 
Another useful plot for assessing normality of errors is a quantile-quantile or Q-Q plot. 
If the residuals do not deviate much from normality, then the points in a Q-Q plot won't deviate much from the dashed one-to-one line. 
If points lie above or below the line, then the residual is larger or smaller, respectively, than expected based on a normal distribution. 


```r
plot(m, 2)
```

![A quantile-quantile plot to assess normality of residuals.](main_files/figure-html/unnamed-chunk-19-1.png) 

To assess heteroskedasticity, it is useful to inspect a plot of the residuals vs. fitted values, e.g. `plot(m, 1)`. 
If it seems as though the spread or variance of residuals varies across the range of fitted values, then it may be worth worrying about homoskedasticity and trying some transformations to fix the problem. 

## Analysis of variance

Sometimes, the covariate of interest is not continuous but instead categorical (e.g., "chocolate", "strawberry", or "vanilla"). 
We might again wonder whether the mean of a random variable $Y$ depends on the value of this covariate. 
However, we cannot really estimate a meaningful "slope" parameter, because in this case $x$ is not continuous. 
Instead, we might formulate the model as follows:

$$y_i \sim N(\alpha_{j[i]}, \sigma^2)$$

Where $\alpha_j$ is the mean of group $j$, and we have $J$ groups total. 
The notation $\alpha_{j[i]}$ represents the notion that the $i^{th}$ observation corresponds to group $j$, and we are going to assume that all observations in the $j^{th}$ group have the same mean, $\alpha_j$. 
The above model is perfectly legitimate, and our parameters to estimate are the group means $\alpha_1, ..., \alpha_J$ and the residual variance $\sigma^2$. 
This parameterization is called the "means" parameterization, and though it is perhaps easier to understand than the following alternative, it is less often used. 

This model is usually parameterized not in terms of the group means, but rather in terms of an intercept (corresponding to the mean of one "reference" group), and deviations from the intercept (differences between a group of interest and the intercept). 
For instance, in R, the group whose mean is the intercept (the "reference" group) will be the group whose name comes first alphabetically. 
Either way, we will estimate the same number of parameters. 
So if our groups are "chocolate", "strawberry", and "vanilla", R will assign the group "chocolate" to be the intercept, and provide 2 more coefficient estimates for the difference between the estimated group mean of strawberry vs. chocoloate, and vanilla vs. chocolate. 

This parameterization can be written as

$$y_i \stackrel{iid}{\sim} N(\mu_0 + \beta_{j[i]}, \sigma^2)$$

where $\mu_0$ is the "intercept" or mean of the reference group, and $\beta_j$ represents the difference in the population mean of group $j$ compared to the reference group (if $j$ is the reference group, the $\beta_j = 0$). 
Traditionally this model is called simple one-way analysis of variance, but we view it simply as another special case of a linear model.

The following example illustrates some data simulation, visualization, and parameter estimation in this context. 
Specifically, we assess 60 humans for their taste response to three flavors of iced cream. 
We want to extrapolate from our sample to the broader population of all ice cream eating humans to learn whether in general people think ice cream tastiness varies as a function of flavor. 


```r
# simulate and visualize data
n <- 60
x <- rep(c("chocolate", "strawberry", "vanilla"), length.out = n)
x <- factor(x)
sigma <- 1
mu_y <- c(chocolate = 3.352, strawberry = .93, vanilla = 1.5)
y <- rnorm(n, mu_y[x], sigma)

library(ggplot2)
ggplot(data.frame(x, y), aes(x, y)) + 
  geom_jitter(position = position_jitter(width=.1)) + 
  xlab('Group') + 
  ylab('Tastiness')
```

![](main_files/figure-html/unnamed-chunk-20-1.png) 

### Model fitting

We can estimate our parameters with the `lm` function (this should be a strong hint that there are not huge differences between linear regression and ANOVA). 
The syntax is exactly the same as with linear regression. 
The only difference is that our input `x` is not numeric, it's a character vector. 


```r
m <- lm(y ~ x)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ x)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -2.13646 -0.57239 -0.00602  0.53962  2.36177 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   3.1641     0.1998  15.836  < 2e-16 ***
## xstrawberry  -1.5507     0.2826  -5.488 9.75e-07 ***
## xvanilla     -1.2210     0.2826  -4.321 6.29e-05 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.8936 on 57 degrees of freedom
## Multiple R-squared:  0.3697,	Adjusted R-squared:  0.3476 
## F-statistic: 16.72 on 2 and 57 DF,  p-value: 1.937e-06
```

Because chocolate comes first alphabetically, it is the reference group and the "(Intercept)" estimate corresponds to the estimate of the group-level mean for chocolate. 
The other two estimates are contrasts between the other groups and this reference group, i.e.  "xstrawberry" is the estimated difference between the group mean for strawberry and the reference group. 

If we wish instead to use a means paramaterization, we need to supress the intercept term in our model as follows:


```r
m <- lm(y ~ 0 + x)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ 0 + x)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -2.13646 -0.57239 -0.00602  0.53962  2.36177 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## xchocolate    3.1641     0.1998  15.836  < 2e-16 ***
## xstrawberry   1.6133     0.1998   8.074 5.17e-11 ***
## xvanilla      1.9431     0.1998   9.725 1.04e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.8936 on 57 degrees of freedom
## Multiple R-squared:  0.8781,	Adjusted R-squared:  0.8717 
## F-statistic: 136.8 on 3 and 57 DF,  p-value: < 2.2e-16
```

Arguably, this approach is more useful because it simplifies the construction of confidence intervals for the group means:


```r
confint(m)
```

```
##                2.5 %   97.5 %
## xchocolate  2.763954 3.564172
## xstrawberry 1.213226 2.013444
## xvanilla    1.542958 2.343176
```

### Checking assumptions

Our assumptions in this simple one way ANOVA context are identical to our assumptions with linear regression. 
Specifically, we assumed that our errors are independently and identically distributed, and that the variance is constant for each group (homoskedasticity). 
The built in `plot` method for `lm` objects is designed to return diagnostic plots that help to check these assumptions. 


```r
par(mfrow=c(2, 2))
plot(m)
```

![](main_files/figure-html/unnamed-chunk-24-1.png) 

## General linear models

We have covered a few special cases of general linear models, which are usually written as follows:

$$y \stackrel{iid}{\sim} N(X \beta, \sigma^2)$$

Where $y$ is a vector consisting of $n$ observations, $X$ is a "design" matrix with $n$ rows and $p$ columns, and $\beta$ is a vector of $p$ parameters. 
There are multivariate general linear models (e.g., MANOVA) where the response variable is a matrix and a covariance matrix is used in place of a scalar variance parameter, but we'll stick to univariate models for simplicity.
The key point here is that the producct of $X$ and $\beta$ provides the mean of the normal distribution from which $y$ is drawn. 
From this perspective, the difference between the model of the mean, linear regression, ANOVA, etc., lies in the structure of $X$ and subsequent interpretation of the parameters $\beta$. 
This is a very powerful idea that unites many superficially disparate approaches. 
It also is the reason that these models are considered "linear", even though a regression line might by quite non-linear (e.g., polynomial regression). 
These models are linear in their parameters, meaning that our expected value for the response $y$ is a **linear combination** (formal notion) of the parameters. 
If a vector of expected values for $y$ in some model cannot be represented as $X \beta$, then it is not a linear model. 

In the model of the mean, $X$ is an $n$ by $1$ matrix, with each element equal to $1$ (i.e. a vector of ones). 
With linear regression, $X$'s first column is all ones (corresponding to the intercept parameter), and the second column contains the values of the covariate $x$. 
In ANOVA, the design matrix $X$ will differ between the means and effects parameterizations. 
With a means parameterization, the entries in column $j$ will equal one if observation (row) $i$ is in group $j$, and entries are zero otherwise. 
If you are not comfortable with matrix multiplication, it's worth investing some effort so that you can understand why $X\beta$ is such a powerful construct. 

>Can you figure out the structure of $X$ with R's default effects parameterization?
>You can check your work with `model.matrix(m)`, where `m` is a model that you've fitted with `lm`.

## Interactions between covariates

Often, the effect of one covariate depends on the value of another covariate. 
This is referred to as "interaction" between the covariates. 
Interactions can exist between two or more continuous and/or nominal covariates. 
These situations have special names in the classical statistics literature. 
For example, models with interactions between nominal covariates fall under "factorial ANOVA", those with interactions between a continuous and a nominal covariate are referred to as "analysis of covariance (ANCOVA)". 
Here we prefer to consider these all as special cases of general linear models. 

### Interactions between two continuous covariates

Here we demonstrate simulation and estimation for a model with an interaction between two continuous covariates. 
Notice that in the simulation, we have exploited the $X \beta$ construct to generate a vector of expected values for $y$. 


```r
n <- 50
x1 <- rnorm(n)
x2 <- rnorm(n)
beta <- c(.5, 1, -1, 2)
sigma <- 1
X <- matrix(c(rep(1, n), x1, x2, x1 * x2), nrow=n)
mu_y <- X %*% beta
y <- rnorm(n, mu_y, sigma)
m <- lm(y ~ x1 * x2)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ x1 * x2)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -1.8837 -0.7082 -0.1602  0.7519  1.9890 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.5541     0.1449   3.824 0.000394 ***
## x1            1.1270     0.1325   8.506 5.36e-11 ***
## x2           -0.6053     0.1370  -4.419 5.98e-05 ***
## x1:x2         1.9240     0.1364  14.110  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.015 on 46 degrees of freedom
## Multiple R-squared:  0.8712,	Adjusted R-squared:  0.8628 
## F-statistic: 103.7 on 3 and 46 DF,  p-value: < 2.2e-16
```

Visualizing these models is tricky, because we are in 3d space (with dimensions $x_1$, $x_2$, and $y$), but contour plots can be effective and leverage peoples' understanding of topographic maps. 


```r
# visualizing the results in terms of the linear predictor
lo <- 40
x1seq <- seq(min(x1), max(x1), length.out = lo)
x2seq <- seq(min(x2), max(x2), length.out = lo)
g <- expand.grid(x1=x1seq, x2=x2seq)
g$e_y <- beta[1] + beta[2] * g$x1 + beta[3] * g$x2 + beta[4] * g$x1 * g$x2
ggplot(g, aes(x=x1, y=x2)) + 
  geom_tile(aes(fill=e_y)) + 
  stat_contour(aes(z=e_y), col='grey') + 
  scale_fill_gradient2() + 
  geom_point(data=data.frame(x1, x2))
```

![](main_files/figure-html/unnamed-chunk-26-1.png) 

Alternatively, you might check out the `effects` package:


```r
library(effects)
plot(allEffects(m))
```

![](main_files/figure-html/unnamed-chunk-27-1.png) 

### Interactions between two categorical covariates

Here we demonstrate interaction between two categorical covariates, using the `diamonds` dataset which is in the `ggplot2` package.t
We are interested in the relationship between diamond price, cut quality, and color.


```r
str(ToothGrowth)
```

```
## 'data.frame':	60 obs. of  3 variables:
##  $ len : num  4.2 11.5 7.3 5.8 6.4 10 11.2 11.2 5.2 7 ...
##  $ supp: Factor w/ 2 levels "OJ","VC": 2 2 2 2 2 2 2 2 2 2 ...
##  $ dose: num  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 ...
```

```r
ToothGrowth$dose <- factor(ToothGrowth$dose)
ggplot(ToothGrowth, aes(x=interaction(dose, supp), y=len)) + 
  geom_point()
```

![](main_files/figure-html/unnamed-chunk-28-1.png) 

In general, visualizing the raw data is a good idea. 
However, we might also be interested in a table with group-wise summaries, such as the sample means, standard deviations, and sample sizes. 


```r
library(dplyr)
ToothGrowth %>%
  group_by(dose, supp) %>%
  summarize(mean = mean(len), 
            sd = sd(len), 
            n = n())
```

```
## Source: local data frame [6 x 5]
## Groups: dose [?]
## 
##     dose   supp  mean       sd     n
##   (fctr) (fctr) (dbl)    (dbl) (int)
## 1    0.5     OJ 13.23 4.459709    10
## 2    0.5     VC  7.98 2.746634    10
## 3      1     OJ 22.70 3.910953    10
## 4      1     VC 16.77 2.515309    10
## 5      2     OJ 26.06 2.655058    10
## 6      2     VC 26.14 4.797731    10
```

We can construct a model to estimate the effect of dose, supplement, and their interaction. 


```r
m <- lm(len ~ dose * supp, data = ToothGrowth)
summary(m)
```

```
## 
## Call:
## lm(formula = len ~ dose * supp, data = ToothGrowth)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
##  -8.20  -2.72  -0.27   2.65   8.27 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)    13.230      1.148  11.521 3.60e-16 ***
## dose1           9.470      1.624   5.831 3.18e-07 ***
## dose2          12.830      1.624   7.900 1.43e-10 ***
## suppVC         -5.250      1.624  -3.233  0.00209 ** 
## dose1:suppVC   -0.680      2.297  -0.296  0.76831    
## dose2:suppVC    5.330      2.297   2.321  0.02411 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 3.631 on 54 degrees of freedom
## Multiple R-squared:  0.7937,	Adjusted R-squared:  0.7746 
## F-statistic: 41.56 on 5 and 54 DF,  p-value: < 2.2e-16
```

This summary gives the effects-parameterization version of the summary. 
The "(Intercept)" refers to the combination of factor levels that occur first alphabetically: in this case, a dose of 0.5 with the "OJ" supplement. 
The coefficients for `dose1` and `dose2` represent estimated contrasts for these two groups relative to the intercept. 
The coefficient for `suppVC` represents the contrast between the "VC" and "OJ" levels of supplement when the dose is 0.5.
The interaction terms represent the difference in the effect of VC for `dose1` and `dose2` relative to a dose of 0.5. 
None of this is particularly intuitive, but this information can be gleaned by inspecting the design matrix $X$ produced by `lm` in the process of fitting the model. 
Inspecting the design matrix along with the dataset to gives a better sense for how $X$ relates to the factor levels:


```r
cbind(model.matrix(m), ToothGrowth)
```

```
##    (Intercept) dose1 dose2 suppVC dose1:suppVC dose2:suppVC  len supp dose
## 1            1     0     0      1            0            0  4.2   VC  0.5
## 2            1     0     0      1            0            0 11.5   VC  0.5
## 3            1     0     0      1            0            0  7.3   VC  0.5
## 4            1     0     0      1            0            0  5.8   VC  0.5
## 5            1     0     0      1            0            0  6.4   VC  0.5
## 6            1     0     0      1            0            0 10.0   VC  0.5
## 7            1     0     0      1            0            0 11.2   VC  0.5
## 8            1     0     0      1            0            0 11.2   VC  0.5
## 9            1     0     0      1            0            0  5.2   VC  0.5
## 10           1     0     0      1            0            0  7.0   VC  0.5
## 11           1     1     0      1            1            0 16.5   VC    1
## 12           1     1     0      1            1            0 16.5   VC    1
## 13           1     1     0      1            1            0 15.2   VC    1
## 14           1     1     0      1            1            0 17.3   VC    1
## 15           1     1     0      1            1            0 22.5   VC    1
## 16           1     1     0      1            1            0 17.3   VC    1
## 17           1     1     0      1            1            0 13.6   VC    1
## 18           1     1     0      1            1            0 14.5   VC    1
## 19           1     1     0      1            1            0 18.8   VC    1
## 20           1     1     0      1            1            0 15.5   VC    1
## 21           1     0     1      1            0            1 23.6   VC    2
## 22           1     0     1      1            0            1 18.5   VC    2
## 23           1     0     1      1            0            1 33.9   VC    2
## 24           1     0     1      1            0            1 25.5   VC    2
## 25           1     0     1      1            0            1 26.4   VC    2
## 26           1     0     1      1            0            1 32.5   VC    2
## 27           1     0     1      1            0            1 26.7   VC    2
## 28           1     0     1      1            0            1 21.5   VC    2
## 29           1     0     1      1            0            1 23.3   VC    2
## 30           1     0     1      1            0            1 29.5   VC    2
## 31           1     0     0      0            0            0 15.2   OJ  0.5
## 32           1     0     0      0            0            0 21.5   OJ  0.5
## 33           1     0     0      0            0            0 17.6   OJ  0.5
## 34           1     0     0      0            0            0  9.7   OJ  0.5
## 35           1     0     0      0            0            0 14.5   OJ  0.5
## 36           1     0     0      0            0            0 10.0   OJ  0.5
## 37           1     0     0      0            0            0  8.2   OJ  0.5
## 38           1     0     0      0            0            0  9.4   OJ  0.5
## 39           1     0     0      0            0            0 16.5   OJ  0.5
## 40           1     0     0      0            0            0  9.7   OJ  0.5
## 41           1     1     0      0            0            0 19.7   OJ    1
## 42           1     1     0      0            0            0 23.3   OJ    1
## 43           1     1     0      0            0            0 23.6   OJ    1
## 44           1     1     0      0            0            0 26.4   OJ    1
## 45           1     1     0      0            0            0 20.0   OJ    1
## 46           1     1     0      0            0            0 25.2   OJ    1
## 47           1     1     0      0            0            0 25.8   OJ    1
## 48           1     1     0      0            0            0 21.2   OJ    1
## 49           1     1     0      0            0            0 14.5   OJ    1
## 50           1     1     0      0            0            0 27.3   OJ    1
## 51           1     0     1      0            0            0 25.5   OJ    2
## 52           1     0     1      0            0            0 26.4   OJ    2
## 53           1     0     1      0            0            0 22.4   OJ    2
## 54           1     0     1      0            0            0 24.5   OJ    2
## 55           1     0     1      0            0            0 24.8   OJ    2
## 56           1     0     1      0            0            0 30.9   OJ    2
## 57           1     0     1      0            0            0 26.4   OJ    2
## 58           1     0     1      0            0            0 27.3   OJ    2
## 59           1     0     1      0            0            0 29.4   OJ    2
## 60           1     0     1      0            0            0 23.0   OJ    2
```

Often, researchers want to know if interactions need to be included in the model. 
From a null hypothesis significance testing perspective, we can evaluate the 'significance' of the interaction term as follows: 


```r
anova(m)
```

```
## Analysis of Variance Table
## 
## Response: len
##           Df  Sum Sq Mean Sq F value    Pr(>F)    
## dose       2 2426.43 1213.22  92.000 < 2.2e-16 ***
## supp       1  205.35  205.35  15.572 0.0002312 ***
## dose:supp  2  108.32   54.16   4.107 0.0218603 *  
## Residuals 54  712.11   13.19                      
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We find that the interaction between dose and supplement is statistically significant, meaning that if we assume that there is no interaction, it is unlikely to observe data that are as or more extreme as what we have observed over the course of infinitely many replicated experiments that will probably never occur.
Although this is far from intuitive, this approach has been widely used.
We will introduce a more streamlined procedure in chapter 3 that 1) does not assume that the effect is zero to begin with, and 2) does not necessarily invoke a hypothetical infinite number of replicated realizations of the data, conditional on one particular parameter value. 
An alternative approach would be to use information theoretics to decide whether the interaction is warranted:


```r
m2 <- lm(len ~ dose + supp, data = ToothGrowth)
AIC(m, m2)
```

```
##    df      AIC
## m   7 332.7056
## m2  5 337.2013
```

In the past decade following Burnham and Anderson's book on the topic, ecologists have leaned heavily on Akaike's information criterion (AIC), which is a relative measure of model quality (balancing goodness of fit with model complexity). 
Here we see that the original model `m` with interaction has a lower AIC value, and is therefore better supported. 
AIC can be considered to be similar to cross validation, approximating the ability of a model to predict future data.

Being somewhat lazy, we might again choose to plot the results of this model using the `effects` package. 


```r
plot(allEffects(m))
```

![](main_files/figure-html/unnamed-chunk-34-1.png) 

This is less than satisfying, as it does not show any data. 
All we see is model output. 
If the model is crap, then the output and these plots are also crap. 
But, evaluating the crappiness of the model is difficult when there are no data shown.
Ideally, the data can be shown along with the estimated group means and some indication of uncertainty. 
If we weren't quite so lazy, we could use the `predict` function to obtain confidence intervals for the means of each group. 


```r
# construct a new data frame for predictions
g <- expand.grid(supp = levels(ToothGrowth$supp), 
                 dose = levels(ToothGrowth$dose))
p <- predict(m, g, interval = 'confidence', type='response')
predictions <- cbind(g, data.frame(p))

ggplot(ToothGrowth, aes(x=interaction(dose, supp), y=len)) + 
  geom_segment(data=predictions, 
               aes(y=lwr, yend=upr, 
                   xend=interaction(dose, supp)), col='red') + 
  geom_point(data=predictions, aes(y=fit), color='red', size=2, shape=2) + 
  geom_jitter(position = position_jitter(width=.1), shape=1) + 
  ylab("Length")
```

![](main_files/figure-html/unnamed-chunk-35-1.png) 

This plot is nice because we can observe the data along with the model output. 
This makes it easier for readers to understand how the model relates to, fits, and does not fit the data.
If you wish to obscure the data, you could make a bar plot with error pars to represent the standard errors. 
Although "dynamite" plots are common, we shall not include one here and we strongly recommend that you never produce such a plot ([more here](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/TatsukiRcode/Poster3.pdf)). 

### Interactions between continuous and categorical covariates

Sometimes, we're interested in interactions between continuous or numeric covariates and another covariates with discrete categorical levels. 
Again, this falls under the broad class of models used in analysis of covariance (ANCOVA). 


```r
x1 <- rnorm(n)
x2 <- factor(sample(c('A', 'B'), n, replace=TRUE))

# generate slopes and intercepts for the first and second groups
a <- rnorm(2)
b <- rnorm(2)
sigma <- .4

X <- matrix(c(ifelse(x2 == 'A', 1, 0), 
              ifelse(x2 == 'B', 1, 0), 
              ifelse(x2 == 'A', x1, 0), 
              ifelse(x2 == 'B', x1, 0)
            ), nrow=n)

mu_y <- X %*% c(a, b)
y <- rnorm(n, mu_y, sigma)
plot(x1, y, col=x2, pch=19)
legend('topright', col=1:2, legend=c('Group A', 'Group B'), pch=19)
```

![](main_files/figure-html/unnamed-chunk-36-1.png) 

Here the intercepts and slopes are allowed to vary for two groups. 
We can fit a model with an interaction between these covariates. 
The intercepts and slopes are estimated separately for the two groups. 


```r
m <- lm(y ~ x1 * x2)
summary(m)
```

```
## 
## Call:
## lm(formula = y ~ x1 * x2)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -1.1005 -0.2183  0.0147  0.2479  1.0519 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  1.11755    0.08965  12.466 2.36e-16 ***
## x1          -0.39243    0.11732  -3.345  0.00165 ** 
## x2B         -1.19500    0.11257 -10.616 5.89e-14 ***
## x1:x2B       3.13087    0.13683  22.882  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3788 on 46 degrees of freedom
## Multiple R-squared:  0.9722,	Adjusted R-squared:  0.9704 
## F-statistic: 536.6 on 3 and 46 DF,  p-value: < 2.2e-16
```

Let's plot the lines of best fit along with the data. 


```r
plot(x1, y, col=x2, pch=19)
legend('topright', col=1:2, legend=c('Group A', 'Group B'), pch=19)
abline(coef(m)[1], coef(m)[2])
abline(coef(m)[1] + coef(m)[3], coef(m)[2] + coef(m)[4], col='red')
```

![](main_files/figure-html/unnamed-chunk-38-1.png) 

The `abline` function, used above, adds lines to plots based on a y-intercept (first argument) and a slope (second argument). 
Do you understand why the particular coefficients that we used as inputs provide the desired intercepts and slopes for each group? 

## Further reading

Schielzeth, H. 2010. Simple means to improve the interpretability of regression coefficients. Methods in Ecology and Evolution 1:103–113.  

Enqvist, L. 2005. The mistreatment of covariate interaction terms in linear model analyses of behavioural and evolutionary ecology studies. Animal Behaviour 70:967–971.  

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 3-4.


Chapter 2: Maximum likelihood estimation
=============================

## Big picture

The likelihood is defined as the probability of the data, conditional on some parameter(s). 
Having observed some data, we often want to know which particular parameter values maximize the probability of those data. 
These parameter values are referred to as the maximum likelihood estimates. 

The goal here is to connect the notion of a likelihood to probability distributions and models. 
We can obtain maximum likelihood estimates (MLEs) in a few ways: analytically, with brute force (direct search), and via optimization (e.g., the `optim` function).

#### Learning goals

- definition of likelihood
- single parameter models: obtaining a MLE with optim
- model of the mean with unknown variance 
- fitting simple linear models with likelihood
- assumptions (especially related to independence)
- inference (what probability does the likelihood function represent?)

## What is likelihood?

The likelihood function represents the probability of the data $y$, conditioned on the parameter(s) $\theta$.
Mathematically, the likelihood is $p(\pmb{y}|\theta) = \mathcal{L}(\theta | \pmb{y})$, where $\pmb{y}$ is a (possibly) vector-valued sample of observations from the random variable $\pmb{Y} = (Y_1, Y_2, ..., Y_n)$. 
More casually, the likelihood function tells us the probability of having observed the sample that we did under different values of the parameter(s) $\theta$.
It is important to recognize that $\theta$ is not treated as a random variable in the likelihood function (the data are treated as random variables).
The likelihood is not the probability of $\theta$ conditional on the data $\pmb{y}$; $p(y | \theta) \neq p(\theta | y)$. 
To calculate $p(\theta | y)$, we'll need to invert the above logic, and we can do so later with Bayes' theorem (also known as the law of inverse probability).

### Joint probabilities of independent events

You may recall that if we have two events $A$ and $B$, and we want to know the joint probability that both events $A$ and $B$ occur, we can generally obtain the joint probability as: $P(A, B) = P(A|B)P(B)$ or $P(A, B) = P(B|A)P(A)$. 
However, if the events $A$ and $B$ are independent, then $P(A|B) = P(A)$ and $P(B|A) = P(B)$. 
In other words, having observed that one event has happened, the probability of the other event is unchanged. 
In this case, we obtain the joint probability of two independent events as $P(A, B)=P(A)P(B)$. 
This logic extends to more than two independent events: $P(E_1, E_2, ..., E_n) = \prod_{i=1}^{n} E_i$, where $E_i$ is the $i^{th}$ event.

Why does this matter? 
Recall the independence assumption that we made in the construction of our linear models in the previous chapters: the error terms $\epsilon_i \sim N(0, \sigma^2)$, or equivalently the conditional distribution of y values $y_i$, $[y_i | \beta, \sigma^2] \sim N(X \beta, \sigma^2)$ are independent. 
Here the square brackets are used as a more compact version of probability notation, we could have also written $P(Y_i = y_i | \beta, \sigma^2)$, the probability that the random variable $Y_i$ equals a particular value $y_i$ conditional on the parameters.
The residual error term of observation $i=1$ tells us nothing about the error term for $i=2$, and conditional on a particular $\beta$ and $\sigma^2$, $y_{1}$ tells us nothing about $y_2$. 
If we assume that our observations are conditionally independent (conditioning on our parameter vector $\theta = (\beta, \sigma^2)$), then we can simply multiply all of the point-wise likelihoods together to find the joint probability of our sample $\pmb{y}$ conditional on the parameters (the likelihood of our sample):

$$p(y_1, y_2, ..., y_n |\theta) = p(y_1 | \theta) p(y_2 | \theta) ... p(y_n | \theta)$$
$$p(\pmb{y} | \theta) = \prod_{i=1}^{n} p(y_i | \theta)$$
$$\mathcal{L}(\theta | \pmb{y}) = \prod_{i=1}^{n} p(y_i | \theta)$$

If the observations $y_1, ..., y_n$ are not conditionally independent (or if you like, if the error terms are not independent), then a likelihood function that multiplies the point-wise probabilities together as if they are independent events is no longer valid. 
This is the problem underlying many discussions of non-independence, psuedoreplication, and autocorrelation (spatial, temporal, phylogenetic): all of these lead to violations of this independence assumption, meaning that it is not correct to work with the product of all the point-wise likelihoods unless terms are added to the model (e.g., blocking factors, autoregressive terms, spatial random effects) so that the observations are conditionally indepenent.

## Obtaining maximum likelihood estimates

We have already obtained quite a few maximum likelihood estimates (MLEs) in the previous chapter with the `lm()` function. 
Here, we provide a more general treatment of estimation.

Assuming that we have a valid likelihood function $\mathcal{L}(\theta | \pmb{y})$, we often seek to find the parameter values that maximize the probability of having observed our sample $\pmb{y}$. 
We can proceed in a few different ways, analytically, by direct search, and by optimization for example. 
Usually the likelihood function is computationally and analytically more tractable on a log scale, so that we often end up working with the log-likelihood rather than the likelihood directly.
This is fine, because any parameter(s) $\theta$ that maximize the likelihood will also maximize the log-likelihood and vice versa, because the log function is strictly increasing. 
Mathematically, we might refer to a maximum likelihood estimate as the value of $\theta$ that maximizes $p(\pmb{y} | \theta)$.
Recalling some calculus, it is reasonable to think that we might attempt to differentiate $p(\pmb{y} | \theta)$ with respect to $\theta$, and find the points at which the derivative equal zero to identify candidate maxima. 
The first derivative will be zero at a maximum, but also at any minima or inflection points, so in practice first-order differentiation alone is not sufficient to identify MLEs. 
In this class, we won't worry about analytically deriving MLEs, but for those who are interested and have some multivariate calculus chops, see Casella and Berger's 2002 book *Statistical Inference*. 

So, we've established that the likelihood is: $p(y | \theta) = \prod_{i=1}^n p(y_i | \theta)$. 
Computationally, this is challenging because we are working with really small numbers (products of small numbers) - so small that our computers have a hard time keeping track of them with much precision. 
Summing logs of small numbers is more computationally stable than multiplying many small numbers together. 
So, let's instead work with the log likelihood by taking the log of both sides of the previous equation. 

$$log(p(y|\theta)) = log \big(\prod_{i=1}^n p(y_i | \theta) \big)$$

Because $log(ab) = log(a) + log(b)$, we can sum up the log likelihoods on the right side of the equation: 

$$log(p(y|\theta)) = \sum_{i=1}^n log(p(y_i | \theta))$$

### Direct search

Here we'll illustrate two methods to find MLEs for normal models: direct search and optimization. 
Returning to our simplest normal model (the model of the mean), we have two parameters: $\theta = (\mu, \sigma^2)$ and $y \sim N(\mu, \sigma^2)$. 
As an aside, maximizing the likelihood is equivalent to minimizing the sum of squared error with the normal distribution. 
Below, we simulate a small dataset with known parameters, and then use a direct search over a bivariate grid of parameters ($\mu$ and $\sigma$). 


```r
# set parameters
mu <- 6
sigma <- 3

# simulate observations
n <- 200
y <- rnorm(n, mu, sigma)

# generate a grid of parameter values to search over
g <- expand.grid(mu = seq(4, 8, length.out=100), 
                 sigma=seq(2, 7, length.out=100))

# evaluate the log-likelihood of the data for each parameter combination
g$loglik <- rep(NA, nrow(g))
for (i in 1:nrow(g)){
  g$loglik[i] <- sum(dnorm(y, g$mu[i], g$sigma[i], log = TRUE))
}

# plot results
library(ggplot2)
ggplot(g, aes(x = mu, y = sigma)) + 
  geom_tile(aes(fill = loglik)) + 
  stat_contour(aes(z = loglik), bins=40) + 
  scale_fill_gradient(low="white", high="red")
```

![](main_files/figure-html/unnamed-chunk-39-1.png) 

This is a contour plot of the log-likelihood surface. 
The black lines are log-likelihood isoclines, corresponding to particular values of the log-likelihood.
If we are lucky, there is only one global maximum on the surface (this can be assessed analytically), and we've found it. 
If the contour plot is hard to understand, here is a 3d representation of the surface (or, at least the points that we evaluated). 




```r
cols <- colorRampPalette(c('purple', 'blue', 'red', 'yellow'))
g$color <- cols(30)[as.numeric(cut(g$loglik, breaks = 30))]
plot3d(x = g$mu, y = g$sigma, z = g$loglik, col = g$color, 
       xlab = 'mu', ylab = 'sigma', zlab = 'loglik')
```

<script src="CanvasMatrix.js" type="text/javascript"></script>
<canvas id="unnamed_chunk_41textureCanvas" style="display: none;" width="256" height="256">
<img src="unnamed_chunk_41snapshot.png" alt="unnamed_chunk_41snapshot" width=673/><br>
Your browser does not support the HTML5 canvas element.</canvas>
<script type="text/javascript">
var min = Math.min,
max = Math.max,
sqrt = Math.sqrt,
sin = Math.sin,
acos = Math.acos,
tan = Math.tan,
SQRT2 = Math.SQRT2,
PI = Math.PI,
log = Math.log,
exp = Math.exp,
vshader, fshader,
rglClass = function() {
this.zoom = [];
this.FOV  = [];
this.userMatrix = [];
this.viewport = [];
this.listeners = [];
this.clipplanes = [];
this.opaque = [];
this.transparent = [];
this.subscenes = [];
this.vshaders = [];
this.fshaders = [];
this.flags = [];
this.prog = [];
this.ofsLoc = [];
this.origLoc = [];
this.sizeLoc = [];
this.usermatLoc = [];
this.vClipplane = [];
this.texture = [];
this.texLoc = [];
this.sampler = [];
this.origsize = [];
this.values = [];
this.offsets = [];
this.normLoc = [];
this.clipLoc = [];
this.centers = [];
this.f = [];
this.buf = [];
this.ibuf = [];
this.mvMatLoc = [];
this.prMatLoc = [];
this.textScaleLoc = [];
this.normMatLoc = [];
this.IMVClip = [];
this.drawFns = [];
this.clipFns = [];
this.prMatrix = new CanvasMatrix4();
this.mvMatrix = new CanvasMatrix4();
this.vp = null;
this.prmvMatrix = null;
this.origs = null;
this.gl = null;
};
(function() {
this.getShader = function( gl, shaderType, code ){
var shader;
shader = gl.createShader ( shaderType );
gl.shaderSource(shader, code);
gl.compileShader(shader);
if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === 0)
alert(gl.getShaderInfoLog(shader));
return shader;
};
this.multMV = function(M, v) {
return [M.m11*v[0] + M.m12*v[1] + M.m13*v[2] + M.m14*v[3],
M.m21*v[0] + M.m22*v[1] + M.m23*v[2] + M.m24*v[3],
M.m31*v[0] + M.m32*v[1] + M.m33*v[2] + M.m34*v[3],
M.m41*v[0] + M.m42*v[1] + M.m43*v[2] + M.m44*v[3]];
};
this.f_is_lit = 1;
this.f_is_smooth = 2;
this.f_has_texture = 4;
this.f_is_indexed = 8;
this.f_depth_sort = 16;
this.f_fixed_quads = 32;
this.f_is_transparent = 64;
this.f_is_lines = 128;
this.f_sprites_3d = 256;
this.f_sprite_3d = 512;
this.f_is_subscene = 1024;
this.f_is_clipplanes = 2048;
this.f_reuse = 4096;
this.whichList = function(id) {
if (this.flags[id] & this.f_is_subscene)
return "subscenes";
else if (this.flags[id] & this.f_is_clipplanes)
return "clipplanes";
else if (this.flags[id] & this.f_is_transparent)
return "transparent";
else
return "opaque";
};
this.inSubscene = function(id, subscene) {
var thelist = this.whichList(id);
return this[thelist][subscene].indexOf(id) > -1;
};
this.addToSubscene = function(id, subscene) {
var thelist = this.whichList(id);
if (this[thelist][subscene].indexOf(id) == -1)
this[thelist][subscene].push(id);
};
this.delFromSubscene = function(id, subscene) {
var thelist = this.whichList(id),
i = this[thelist][subscene].indexOf(id);
if (i > -1)
this[thelist][subscene].splice(i, 1);
};
this.setSubsceneEntries = function(ids, subscene) {
this.subscenes[subscene] = [];
this.clipplanes[subscene] = [];
this.transparent[subscene] = [];
this.opaque[subscene] = [];
for (var i = 0; i < ids.length; i++)
this.addToSubscene(ids[i], subscene);
};
this.getSubsceneEntries = function(subscene) {
return(this.subscenes[subscene].concat(this.clipplanes[subscene]).
concat(this.transparent[subscene]).concat(this.opaque[subscene]));
};
this.getPowerOfTwo = function(value) {
var pow = 1;
while(pow<value) {
pow *= 2;
}
return pow;
};
this.handleLoadedTexture = function(id) {
var gl = this.gl, textureCanvas = this.textureCanvas;
gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textureCanvas);
gl.generateMipmap(gl.TEXTURE_2D);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
gl.bindTexture(gl.TEXTURE_2D, null);
};
this.loadImageNowOrLater = function(name, id) {
var image = document.getElementById(name),
self = this;
if (image.rglTextureLoaded) {
this.loadImageToTexture(image, id);
} else {
image.addEventListener("load", 
function() {
self.loadImageToTexture(this, id);
}); 
}
};
this.loadImageToTexture = function(image, id) {
var canvas = this.textureCanvas,
ctx = canvas.getContext("2d"),
w = image.width,
h = image.height,
canvasX = this.getPowerOfTwo(w),
canvasY = this.getPowerOfTwo(h),
gl = this.gl,
maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
image.rglTextureLoaded = true;
while (canvasX > 1 && (canvasX > maxTexSize || canvasY > maxTexSize)) {
canvasX /= 2;
canvasY /= 2;
}
canvas.width = canvasX;
canvas.height = canvasY;
ctx.imageSmoothingEnabled = true;
ctx.drawImage(image, 0, 0, canvasX, canvasY);
image.style.display = "none";
this.handleLoadedTexture(id);
this.drawScene();
};
}).call(rglClass.prototype);
var unnamed_chunk_41rgl = new rglClass();
unnamed_chunk_41rgl.start = function() {
var i, j, v, ind, texts, f, texinfo, canvas;
var debug = function(msg) {
document.getElementById("unnamed_chunk_41debug").innerHTML = msg;
};
debug("");
canvas = this.canvas = document.getElementById("unnamed_chunk_41canvas");
this.textureCanvas = document.getElementById("unnamed_chunk_41textureCanvas");
if (!window.WebGLRenderingContext){
debug("<img src=\"unnamed_chunk_41snapshot.png\" alt=\"unnamed_chunk_41snapshot\" width=673/><br> Your browser does not support WebGL. See <a href=\"http://get.webgl.org\">http://get.webgl.org</a>");
return;
}
try {
// Try to grab the standard context. If it fails, fallback to experimental.
this.gl = canvas.getContext("webgl") ||
canvas.getContext("experimental-webgl");
}
catch(e) {}
if ( !this.gl ) {
debug("<img src=\"unnamed_chunk_41snapshot.png\" alt=\"unnamed_chunk_41snapshot\" width=673/><br> Your browser appears to support WebGL, but did not create a WebGL context.  See <a href=\"http://get.webgl.org\">http://get.webgl.org</a>");
return;
}
var gl = this.gl,
width = 673, height = 481;
canvas.width = width;   canvas.height = height;
var normMatrix = new CanvasMatrix4(),
saveMat = {},
distance,
posLoc = 0,
colLoc = 1;
var activeSubscene = 1;
this.flags[1] = 1192;
this.zoom[1] = 1;
this.FOV[1] = 30;
this.viewport[1] = [0, 0, 672, 480];
this.userMatrix[1] = new CanvasMatrix4();
this.userMatrix[1].load([
1, 0, 0, 0,
0, 0.3420201, -0.9396926, 0,
0, 0.9396926, 0.3420201, 0,
0, 0, 0, 1
]);
this.clipplanes[1] = [];
this.opaque[1] = [7,9,10,11,12,13,14,15,16,17,18];
this.transparent[1] = [];
this.subscenes[1] = [];
function drawTextToCanvas(text, cex) {
var canvasX, canvasY,
textX, textY,
textHeight = 20 * cex,
textColour = "white",
fontFamily = "Arial",
backgroundColour = "rgba(0,0,0,0)",
canvas = document.getElementById("unnamed_chunk_41textureCanvas"),
ctx = canvas.getContext("2d"),
i;
ctx.font = textHeight+"px "+fontFamily;
canvasX = 1;
var widths = [];
for (i = 0; i < text.length; i++)  {
widths[i] = ctx.measureText(text[i]).width;
canvasX = (widths[i] > canvasX) ? widths[i] : canvasX;
}
canvasX = unnamed_chunk_41rgl.getPowerOfTwo(canvasX);
var offset = 2*textHeight, // offset to first baseline
skip = 2*textHeight;   // skip between baselines
canvasY = unnamed_chunk_41rgl.getPowerOfTwo(offset + text.length*skip);
canvas.width = canvasX;
canvas.height = canvasY;
ctx.fillStyle = backgroundColour;
ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
ctx.fillStyle = textColour;
ctx.textAlign = "left";
ctx.textBaseline = "alphabetic";
ctx.font = textHeight+"px "+fontFamily;
for(i = 0; i < text.length; i++) {
textY = i*skip + offset;
ctx.fillText(text[i], 0,  textY);
}
return {canvasX:canvasX, canvasY:canvasY,
widths:widths, textHeight:textHeight,
offset:offset, skip:skip};
}
// ****** points object 7 ******
this.flags[7] = 0;
this.vshaders[7] = "	/* ****** points object 7 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	void main(void) {\n	  vPosition = mvMatrix * vec4(aPos, 1.);\n	  gl_Position = prMatrix * vPosition;\n	  gl_PointSize = 3.;\n	  vCol = aCol;\n	}";
this.fshaders[7] = "	/* ****** points object 7 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  gl_FragColor = lighteffect;\n	}";
this.prog[7]  = gl.createProgram();
gl.attachShader(this.prog[7], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[7] ));
gl.attachShader(this.prog[7], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[7] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[7], 0, "aPos");
gl.bindAttribLocation(this.prog[7], 1, "aCol");
gl.linkProgram(this.prog[7]);
this.offsets[7]={vofs:0, cofs:3, nofs:-1, radofs:-1, oofs:-1, tofs:-1, stride:7};
v=new Float32Array([
4, 2, -669.9025, 0.627451, 0.1254902, 0.9411765, 1,
4.040404, 2, -665.7001, 0.627451, 0.1254902, 0.9411765, 1,
4.080808, 2, -661.5795, 0.5607843, 0.1098039, 0.945098, 1,
4.121212, 2, -657.5404, 0.4941176, 0.09803922, 0.9529412, 1,
4.161616, 2, -653.583, 0.4313726, 0.08627451, 0.9568627, 1,
4.20202, 2, -649.7072, 0.4313726, 0.08627451, 0.9568627, 1,
4.242424, 2, -645.913, 0.3647059, 0.07058824, 0.9647059, 1,
4.282828, 2, -642.2004, 0.3019608, 0.05882353, 0.9686275, 1,
4.323232, 2, -638.5695, 0.3019608, 0.05882353, 0.9686275, 1,
4.363636, 2, -635.0202, 0.2352941, 0.04705882, 0.9764706, 1,
4.40404, 2, -631.5525, 0.172549, 0.03137255, 0.9803922, 1,
4.444445, 2, -628.1664, 0.172549, 0.03137255, 0.9803922, 1,
4.484848, 2, -624.862, 0.1058824, 0.01960784, 0.9882353, 1,
4.525252, 2, -621.6392, 0.1058824, 0.01960784, 0.9882353, 1,
4.565657, 2, -618.498, 0.04313726, 0.007843138, 0.9921569, 1,
4.606061, 2, -615.4384, 0.03137255, 0, 0.9647059, 1,
4.646465, 2, -612.4604, 0.03137255, 0, 0.9647059, 1,
4.686869, 2, -609.5641, 0.1372549, 0, 0.8588235, 1,
4.727273, 2, -606.7495, 0.1372549, 0, 0.8588235, 1,
4.767677, 2, -604.0164, 0.2392157, 0, 0.7568628, 1,
4.808081, 2, -601.3649, 0.2392157, 0, 0.7568628, 1,
4.848485, 2, -598.7951, 0.3411765, 0, 0.654902, 1,
4.888889, 2, -596.3069, 0.3411765, 0, 0.654902, 1,
4.929293, 2, -593.9003, 0.4470588, 0, 0.5490196, 1,
4.969697, 2, -591.5754, 0.4470588, 0, 0.5490196, 1,
5.010101, 2, -589.3321, 0.5490196, 0, 0.4470588, 1,
5.050505, 2, -587.1704, 0.5490196, 0, 0.4470588, 1,
5.090909, 2, -585.0903, 0.5490196, 0, 0.4470588, 1,
5.131313, 2, -583.0919, 0.654902, 0, 0.3411765, 1,
5.171717, 2, -581.175, 0.654902, 0, 0.3411765, 1,
5.212121, 2, -579.3398, 0.654902, 0, 0.3411765, 1,
5.252525, 2, -577.5862, 0.7568628, 0, 0.2392157, 1,
5.292929, 2, -575.9143, 0.7568628, 0, 0.2392157, 1,
5.333333, 2, -574.324, 0.7568628, 0, 0.2392157, 1,
5.373737, 2, -572.8152, 0.8588235, 0, 0.1372549, 1,
5.414141, 2, -571.3882, 0.8588235, 0, 0.1372549, 1,
5.454545, 2, -570.0427, 0.8588235, 0, 0.1372549, 1,
5.494949, 2, -568.7789, 0.8588235, 0, 0.1372549, 1,
5.535354, 2, -567.5967, 0.9647059, 0, 0.03137255, 1,
5.575758, 2, -566.4961, 0.9647059, 0, 0.03137255, 1,
5.616162, 2, -565.4772, 0.9647059, 0, 0.03137255, 1,
5.656566, 2, -564.5398, 0.9647059, 0, 0.03137255, 1,
5.69697, 2, -563.6841, 0.9647059, 0, 0.03137255, 1,
5.737374, 2, -562.91, 0.9647059, 0, 0.03137255, 1,
5.777778, 2, -562.2176, 1, 0.06666667, 0, 1,
5.818182, 2, -561.6068, 1, 0.06666667, 0, 1,
5.858586, 2, -561.0775, 1, 0.06666667, 0, 1,
5.89899, 2, -560.6299, 1, 0.06666667, 0, 1,
5.939394, 2, -560.264, 1, 0.06666667, 0, 1,
5.979798, 2, -559.9796, 1, 0.06666667, 0, 1,
6.020202, 2, -559.7769, 1, 0.06666667, 0, 1,
6.060606, 2, -559.6558, 1, 0.06666667, 0, 1,
6.10101, 2, -559.6163, 1, 0.06666667, 0, 1,
6.141414, 2, -559.6585, 1, 0.06666667, 0, 1,
6.181818, 2, -559.7823, 1, 0.06666667, 0, 1,
6.222222, 2, -559.9877, 1, 0.06666667, 0, 1,
6.262626, 2, -560.2747, 1, 0.06666667, 0, 1,
6.30303, 2, -560.6434, 1, 0.06666667, 0, 1,
6.343434, 2, -561.0937, 1, 0.06666667, 0, 1,
6.383838, 2, -561.6255, 1, 0.06666667, 0, 1,
6.424242, 2, -562.2391, 1, 0.06666667, 0, 1,
6.464646, 2, -562.9343, 0.9647059, 0, 0.03137255, 1,
6.505051, 2, -563.711, 0.9647059, 0, 0.03137255, 1,
6.545455, 2, -564.5694, 0.9647059, 0, 0.03137255, 1,
6.585859, 2, -565.5095, 0.9647059, 0, 0.03137255, 1,
6.626263, 2, -566.5311, 0.9647059, 0, 0.03137255, 1,
6.666667, 2, -567.6344, 0.9647059, 0, 0.03137255, 1,
6.707071, 2, -568.8193, 0.8588235, 0, 0.1372549, 1,
6.747475, 2, -570.0858, 0.8588235, 0, 0.1372549, 1,
6.787879, 2, -571.434, 0.8588235, 0, 0.1372549, 1,
6.828283, 2, -572.8637, 0.8588235, 0, 0.1372549, 1,
6.868687, 2, -574.3751, 0.7568628, 0, 0.2392157, 1,
6.909091, 2, -575.9681, 0.7568628, 0, 0.2392157, 1,
6.949495, 2, -577.6428, 0.7568628, 0, 0.2392157, 1,
6.989899, 2, -579.399, 0.654902, 0, 0.3411765, 1,
7.030303, 2, -581.2369, 0.654902, 0, 0.3411765, 1,
7.070707, 2, -583.1564, 0.654902, 0, 0.3411765, 1,
7.111111, 2, -585.1576, 0.5490196, 0, 0.4470588, 1,
7.151515, 2, -587.2404, 0.5490196, 0, 0.4470588, 1,
7.191919, 2, -589.4047, 0.4470588, 0, 0.5490196, 1,
7.232323, 2, -591.6508, 0.4470588, 0, 0.5490196, 1,
7.272727, 2, -593.9784, 0.4470588, 0, 0.5490196, 1,
7.313131, 2, -596.3876, 0.3411765, 0, 0.654902, 1,
7.353535, 2, -598.8785, 0.3411765, 0, 0.654902, 1,
7.393939, 2, -601.451, 0.2392157, 0, 0.7568628, 1,
7.434343, 2, -604.1052, 0.2392157, 0, 0.7568628, 1,
7.474748, 2, -606.8409, 0.1372549, 0, 0.8588235, 1,
7.515152, 2, -609.6583, 0.1372549, 0, 0.8588235, 1,
7.555555, 2, -612.5574, 0.03137255, 0, 0.9647059, 1,
7.59596, 2, -615.538, 0.03137255, 0, 0.9647059, 1,
7.636364, 2, -618.6002, 0.04313726, 0.007843138, 0.9921569, 1,
7.676768, 2, -621.7441, 0.1058824, 0.01960784, 0.9882353, 1,
7.717172, 2, -624.9696, 0.1058824, 0.01960784, 0.9882353, 1,
7.757576, 2, -628.2767, 0.172549, 0.03137255, 0.9803922, 1,
7.79798, 2, -631.6655, 0.172549, 0.03137255, 0.9803922, 1,
7.838384, 2, -635.1359, 0.2352941, 0.04705882, 0.9764706, 1,
7.878788, 2, -638.6879, 0.3019608, 0.05882353, 0.9686275, 1,
7.919192, 2, -642.3215, 0.3019608, 0.05882353, 0.9686275, 1,
7.959596, 2, -646.0368, 0.3647059, 0.07058824, 0.9647059, 1,
8, 2, -649.8337, 0.4313726, 0.08627451, 0.9568627, 1,
4, 2.050505, -657.9836, 0.4941176, 0.09803922, 0.9529412, 1,
4.040404, 2.050505, -653.9857, 0.4941176, 0.09803922, 0.9529412, 1,
4.080808, 2.050505, -650.0655, 0.4313726, 0.08627451, 0.9568627, 1,
4.121212, 2.050505, -646.223, 0.3647059, 0.07058824, 0.9647059, 1,
4.161616, 2.050505, -642.4581, 0.3019608, 0.05882353, 0.9686275, 1,
4.20202, 2.050505, -638.7709, 0.3019608, 0.05882353, 0.9686275, 1,
4.242424, 2.050505, -635.1613, 0.2352941, 0.04705882, 0.9764706, 1,
4.282828, 2.050505, -631.6293, 0.172549, 0.03137255, 0.9803922, 1,
4.323232, 2.050505, -628.175, 0.172549, 0.03137255, 0.9803922, 1,
4.363636, 2.050505, -624.7985, 0.1058824, 0.01960784, 0.9882353, 1,
4.40404, 2.050505, -621.4995, 0.04313726, 0.007843138, 0.9921569, 1,
4.444445, 2.050505, -618.2781, 0.04313726, 0.007843138, 0.9921569, 1,
4.484848, 2.050505, -615.1345, 0.03137255, 0, 0.9647059, 1,
4.525252, 2.050505, -612.0685, 0.03137255, 0, 0.9647059, 1,
4.565657, 2.050505, -609.0801, 0.1372549, 0, 0.8588235, 1,
4.606061, 2.050505, -606.1694, 0.1372549, 0, 0.8588235, 1,
4.646465, 2.050505, -603.3364, 0.2392157, 0, 0.7568628, 1,
4.686869, 2.050505, -600.581, 0.2392157, 0, 0.7568628, 1,
4.727273, 2.050505, -597.9032, 0.3411765, 0, 0.654902, 1,
4.767677, 2.050505, -595.3031, 0.3411765, 0, 0.654902, 1,
4.808081, 2.050505, -592.7807, 0.4470588, 0, 0.5490196, 1,
4.848485, 2.050505, -590.3359, 0.4470588, 0, 0.5490196, 1,
4.888889, 2.050505, -587.9688, 0.5490196, 0, 0.4470588, 1,
4.929293, 2.050505, -585.6793, 0.5490196, 0, 0.4470588, 1,
4.969697, 2.050505, -583.4675, 0.654902, 0, 0.3411765, 1,
5.010101, 2.050505, -581.3333, 0.654902, 0, 0.3411765, 1,
5.050505, 2.050505, -579.2767, 0.654902, 0, 0.3411765, 1,
5.090909, 2.050505, -577.2979, 0.7568628, 0, 0.2392157, 1,
5.131313, 2.050505, -575.3967, 0.7568628, 0, 0.2392157, 1,
5.171717, 2.050505, -573.5731, 0.7568628, 0, 0.2392157, 1,
5.212121, 2.050505, -571.8272, 0.8588235, 0, 0.1372549, 1,
5.252525, 2.050505, -570.1589, 0.8588235, 0, 0.1372549, 1,
5.292929, 2.050505, -568.5684, 0.8588235, 0, 0.1372549, 1,
5.333333, 2.050505, -567.0554, 0.9647059, 0, 0.03137255, 1,
5.373737, 2.050505, -565.6201, 0.9647059, 0, 0.03137255, 1,
5.414141, 2.050505, -564.2625, 0.9647059, 0, 0.03137255, 1,
5.454545, 2.050505, -562.9824, 0.9647059, 0, 0.03137255, 1,
5.494949, 2.050505, -561.7801, 1, 0.06666667, 0, 1,
5.535354, 2.050505, -560.6554, 1, 0.06666667, 0, 1,
5.575758, 2.050505, -559.6084, 1, 0.06666667, 0, 1,
5.616162, 2.050505, -558.639, 1, 0.06666667, 0, 1,
5.656566, 2.050505, -557.7473, 1, 0.06666667, 0, 1,
5.69697, 2.050505, -556.9332, 1, 0.1686275, 0, 1,
5.737374, 2.050505, -556.1968, 1, 0.1686275, 0, 1,
5.777778, 2.050505, -555.538, 1, 0.1686275, 0, 1,
5.818182, 2.050505, -554.9569, 1, 0.1686275, 0, 1,
5.858586, 2.050505, -554.4534, 1, 0.1686275, 0, 1,
5.89899, 2.050505, -554.0276, 1, 0.1686275, 0, 1,
5.939394, 2.050505, -553.6794, 1, 0.1686275, 0, 1,
5.979798, 2.050505, -553.4089, 1, 0.1686275, 0, 1,
6.020202, 2.050505, -553.2161, 1, 0.1686275, 0, 1,
6.060606, 2.050505, -553.1009, 1, 0.1686275, 0, 1,
6.10101, 2.050505, -553.0634, 1, 0.1686275, 0, 1,
6.141414, 2.050505, -553.1035, 1, 0.1686275, 0, 1,
6.181818, 2.050505, -553.2212, 1, 0.1686275, 0, 1,
6.222222, 2.050505, -553.4166, 1, 0.1686275, 0, 1,
6.262626, 2.050505, -553.6897, 1, 0.1686275, 0, 1,
6.30303, 2.050505, -554.0404, 1, 0.1686275, 0, 1,
6.343434, 2.050505, -554.4688, 1, 0.1686275, 0, 1,
6.383838, 2.050505, -554.9748, 1, 0.1686275, 0, 1,
6.424242, 2.050505, -555.5585, 1, 0.1686275, 0, 1,
6.464646, 2.050505, -556.2198, 1, 0.1686275, 0, 1,
6.505051, 2.050505, -556.9588, 1, 0.1686275, 0, 1,
6.545455, 2.050505, -557.7755, 1, 0.06666667, 0, 1,
6.585859, 2.050505, -558.6697, 1, 0.06666667, 0, 1,
6.626263, 2.050505, -559.6417, 1, 0.06666667, 0, 1,
6.666667, 2.050505, -560.6913, 1, 0.06666667, 0, 1,
6.707071, 2.050505, -561.8185, 1, 0.06666667, 0, 1,
6.747475, 2.050505, -563.0234, 0.9647059, 0, 0.03137255, 1,
6.787879, 2.050505, -564.306, 0.9647059, 0, 0.03137255, 1,
6.828283, 2.050505, -565.6661, 0.9647059, 0, 0.03137255, 1,
6.868687, 2.050505, -567.104, 0.9647059, 0, 0.03137255, 1,
6.909091, 2.050505, -568.6196, 0.8588235, 0, 0.1372549, 1,
6.949495, 2.050505, -570.2127, 0.8588235, 0, 0.1372549, 1,
6.989899, 2.050505, -571.8835, 0.8588235, 0, 0.1372549, 1,
7.030303, 2.050505, -573.632, 0.7568628, 0, 0.2392157, 1,
7.070707, 2.050505, -575.4581, 0.7568628, 0, 0.2392157, 1,
7.111111, 2.050505, -577.3619, 0.7568628, 0, 0.2392157, 1,
7.151515, 2.050505, -579.3433, 0.654902, 0, 0.3411765, 1,
7.191919, 2.050505, -581.4024, 0.654902, 0, 0.3411765, 1,
7.232323, 2.050505, -583.5391, 0.654902, 0, 0.3411765, 1,
7.272727, 2.050505, -585.7535, 0.5490196, 0, 0.4470588, 1,
7.313131, 2.050505, -588.0456, 0.5490196, 0, 0.4470588, 1,
7.353535, 2.050505, -590.4153, 0.4470588, 0, 0.5490196, 1,
7.393939, 2.050505, -592.8626, 0.4470588, 0, 0.5490196, 1,
7.434343, 2.050505, -595.3876, 0.3411765, 0, 0.654902, 1,
7.474748, 2.050505, -597.9903, 0.3411765, 0, 0.654902, 1,
7.515152, 2.050505, -600.6706, 0.2392157, 0, 0.7568628, 1,
7.555555, 2.050505, -603.4285, 0.2392157, 0, 0.7568628, 1,
7.59596, 2.050505, -606.2642, 0.1372549, 0, 0.8588235, 1,
7.636364, 2.050505, -609.1774, 0.1372549, 0, 0.8588235, 1,
7.676768, 2.050505, -612.1683, 0.03137255, 0, 0.9647059, 1,
7.717172, 2.050505, -615.2369, 0.03137255, 0, 0.9647059, 1,
7.757576, 2.050505, -618.3831, 0.04313726, 0.007843138, 0.9921569, 1,
7.79798, 2.050505, -621.607, 0.1058824, 0.01960784, 0.9882353, 1,
7.838384, 2.050505, -624.9085, 0.1058824, 0.01960784, 0.9882353, 1,
7.878788, 2.050505, -628.2877, 0.172549, 0.03137255, 0.9803922, 1,
7.919192, 2.050505, -631.7446, 0.172549, 0.03137255, 0.9803922, 1,
7.959596, 2.050505, -635.2791, 0.2352941, 0.04705882, 0.9764706, 1,
8, 2.050505, -638.8912, 0.3019608, 0.05882353, 0.9686275, 1,
4, 2.10101, -647.1478, 0.3647059, 0.07058824, 0.9647059, 1,
4.040404, 2.10101, -643.3398, 0.3647059, 0.07058824, 0.9647059, 1,
4.080808, 2.10101, -639.6058, 0.3019608, 0.05882353, 0.9686275, 1,
4.121212, 2.10101, -635.9459, 0.2352941, 0.04705882, 0.9764706, 1,
4.161616, 2.10101, -632.3598, 0.2352941, 0.04705882, 0.9764706, 1,
4.20202, 2.10101, -628.8477, 0.172549, 0.03137255, 0.9803922, 1,
4.242424, 2.10101, -625.4095, 0.1058824, 0.01960784, 0.9882353, 1,
4.282828, 2.10101, -622.0454, 0.1058824, 0.01960784, 0.9882353, 1,
4.323232, 2.10101, -618.7552, 0.04313726, 0.007843138, 0.9921569, 1,
4.363636, 2.10101, -615.5389, 0.03137255, 0, 0.9647059, 1,
4.40404, 2.10101, -612.3967, 0.03137255, 0, 0.9647059, 1,
4.444445, 2.10101, -609.3284, 0.1372549, 0, 0.8588235, 1,
4.484848, 2.10101, -606.334, 0.1372549, 0, 0.8588235, 1,
4.525252, 2.10101, -603.4136, 0.2392157, 0, 0.7568628, 1,
4.565657, 2.10101, -600.5673, 0.2392157, 0, 0.7568628, 1,
4.606061, 2.10101, -597.7948, 0.3411765, 0, 0.654902, 1,
4.646465, 2.10101, -595.0963, 0.3411765, 0, 0.654902, 1,
4.686869, 2.10101, -592.4718, 0.4470588, 0, 0.5490196, 1,
4.727273, 2.10101, -589.9212, 0.4470588, 0, 0.5490196, 1,
4.767677, 2.10101, -587.4446, 0.5490196, 0, 0.4470588, 1,
4.808081, 2.10101, -585.042, 0.5490196, 0, 0.4470588, 1,
4.848485, 2.10101, -582.7133, 0.654902, 0, 0.3411765, 1,
4.888889, 2.10101, -580.4586, 0.654902, 0, 0.3411765, 1,
4.929293, 2.10101, -578.2779, 0.7568628, 0, 0.2392157, 1,
4.969697, 2.10101, -576.1711, 0.7568628, 0, 0.2392157, 1,
5.010101, 2.10101, -574.1384, 0.7568628, 0, 0.2392157, 1,
5.050505, 2.10101, -572.1795, 0.8588235, 0, 0.1372549, 1,
5.090909, 2.10101, -570.2946, 0.8588235, 0, 0.1372549, 1,
5.131313, 2.10101, -568.4837, 0.8588235, 0, 0.1372549, 1,
5.171717, 2.10101, -566.7468, 0.9647059, 0, 0.03137255, 1,
5.212121, 2.10101, -565.0838, 0.9647059, 0, 0.03137255, 1,
5.252525, 2.10101, -563.4948, 0.9647059, 0, 0.03137255, 1,
5.292929, 2.10101, -561.9797, 1, 0.06666667, 0, 1,
5.333333, 2.10101, -560.5386, 1, 0.06666667, 0, 1,
5.373737, 2.10101, -559.1715, 1, 0.06666667, 0, 1,
5.414141, 2.10101, -557.8784, 1, 0.06666667, 0, 1,
5.454545, 2.10101, -556.6591, 1, 0.1686275, 0, 1,
5.494949, 2.10101, -555.5139, 1, 0.1686275, 0, 1,
5.535354, 2.10101, -554.4426, 1, 0.1686275, 0, 1,
5.575758, 2.10101, -553.4454, 1, 0.1686275, 0, 1,
5.616162, 2.10101, -552.522, 1, 0.1686275, 0, 1,
5.656566, 2.10101, -551.6727, 1, 0.2745098, 0, 1,
5.69697, 2.10101, -550.8972, 1, 0.2745098, 0, 1,
5.737374, 2.10101, -550.1958, 1, 0.2745098, 0, 1,
5.777778, 2.10101, -549.5683, 1, 0.2745098, 0, 1,
5.818182, 2.10101, -549.0148, 1, 0.2745098, 0, 1,
5.858586, 2.10101, -548.5352, 1, 0.2745098, 0, 1,
5.89899, 2.10101, -548.1296, 1, 0.2745098, 0, 1,
5.939394, 2.10101, -547.798, 1, 0.2745098, 0, 1,
5.979798, 2.10101, -547.5404, 1, 0.2745098, 0, 1,
6.020202, 2.10101, -547.3567, 1, 0.2745098, 0, 1,
6.060606, 2.10101, -547.2469, 1, 0.2745098, 0, 1,
6.10101, 2.10101, -547.2112, 1, 0.2745098, 0, 1,
6.141414, 2.10101, -547.2494, 1, 0.2745098, 0, 1,
6.181818, 2.10101, -547.3616, 1, 0.2745098, 0, 1,
6.222222, 2.10101, -547.5477, 1, 0.2745098, 0, 1,
6.262626, 2.10101, -547.8078, 1, 0.2745098, 0, 1,
6.30303, 2.10101, -548.1418, 1, 0.2745098, 0, 1,
6.343434, 2.10101, -548.5499, 1, 0.2745098, 0, 1,
6.383838, 2.10101, -549.0319, 1, 0.2745098, 0, 1,
6.424242, 2.10101, -549.5878, 1, 0.2745098, 0, 1,
6.464646, 2.10101, -550.2177, 1, 0.2745098, 0, 1,
6.505051, 2.10101, -550.9216, 1, 0.2745098, 0, 1,
6.545455, 2.10101, -551.6995, 1, 0.2745098, 0, 1,
6.585859, 2.10101, -552.5513, 1, 0.1686275, 0, 1,
6.626263, 2.10101, -553.4771, 1, 0.1686275, 0, 1,
6.666667, 2.10101, -554.4768, 1, 0.1686275, 0, 1,
6.707071, 2.10101, -555.5505, 1, 0.1686275, 0, 1,
6.747475, 2.10101, -556.6982, 1, 0.1686275, 0, 1,
6.787879, 2.10101, -557.9198, 1, 0.06666667, 0, 1,
6.828283, 2.10101, -559.2154, 1, 0.06666667, 0, 1,
6.868687, 2.10101, -560.585, 1, 0.06666667, 0, 1,
6.909091, 2.10101, -562.0285, 1, 0.06666667, 0, 1,
6.949495, 2.10101, -563.546, 0.9647059, 0, 0.03137255, 1,
6.989899, 2.10101, -565.1375, 0.9647059, 0, 0.03137255, 1,
7.030303, 2.10101, -566.8029, 0.9647059, 0, 0.03137255, 1,
7.070707, 2.10101, -568.5422, 0.8588235, 0, 0.1372549, 1,
7.111111, 2.10101, -570.3556, 0.8588235, 0, 0.1372549, 1,
7.151515, 2.10101, -572.2429, 0.8588235, 0, 0.1372549, 1,
7.191919, 2.10101, -574.2042, 0.7568628, 0, 0.2392157, 1,
7.232323, 2.10101, -576.2394, 0.7568628, 0, 0.2392157, 1,
7.272727, 2.10101, -578.3486, 0.7568628, 0, 0.2392157, 1,
7.313131, 2.10101, -580.5318, 0.654902, 0, 0.3411765, 1,
7.353535, 2.10101, -582.7889, 0.654902, 0, 0.3411765, 1,
7.393939, 2.10101, -585.1201, 0.5490196, 0, 0.4470588, 1,
7.434343, 2.10101, -587.5251, 0.5490196, 0, 0.4470588, 1,
7.474748, 2.10101, -590.0042, 0.4470588, 0, 0.5490196, 1,
7.515152, 2.10101, -592.5571, 0.4470588, 0, 0.5490196, 1,
7.555555, 2.10101, -595.1841, 0.3411765, 0, 0.654902, 1,
7.59596, 2.10101, -597.885, 0.3411765, 0, 0.654902, 1,
7.636364, 2.10101, -600.6599, 0.2392157, 0, 0.7568628, 1,
7.676768, 2.10101, -603.5087, 0.2392157, 0, 0.7568628, 1,
7.717172, 2.10101, -606.4316, 0.1372549, 0, 0.8588235, 1,
7.757576, 2.10101, -609.4283, 0.1372549, 0, 0.8588235, 1,
7.79798, 2.10101, -612.4991, 0.03137255, 0, 0.9647059, 1,
7.838384, 2.10101, -615.6438, 0.03137255, 0, 0.9647059, 1,
7.878788, 2.10101, -618.8625, 0.04313726, 0.007843138, 0.9921569, 1,
7.919192, 2.10101, -622.1552, 0.1058824, 0.01960784, 0.9882353, 1,
7.959596, 2.10101, -625.5217, 0.1058824, 0.01960784, 0.9882353, 1,
8, 2.10101, -628.9623, 0.172549, 0.03137255, 0.9803922, 1,
4, 2.151515, -637.2892, 0.2352941, 0.04705882, 0.9764706, 1,
4.040404, 2.151515, -633.6579, 0.2352941, 0.04705882, 0.9764706, 1,
4.080808, 2.151515, -630.0972, 0.172549, 0.03137255, 0.9803922, 1,
4.121212, 2.151515, -626.607, 0.1058824, 0.01960784, 0.9882353, 1,
4.161616, 2.151515, -623.1873, 0.1058824, 0.01960784, 0.9882353, 1,
4.20202, 2.151515, -619.8381, 0.04313726, 0.007843138, 0.9921569, 1,
4.242424, 2.151515, -616.5596, 0.04313726, 0.007843138, 0.9921569, 1,
4.282828, 2.151515, -613.3514, 0.03137255, 0, 0.9647059, 1,
4.323232, 2.151515, -610.2139, 0.1372549, 0, 0.8588235, 1,
4.363636, 2.151515, -607.1469, 0.1372549, 0, 0.8588235, 1,
4.40404, 2.151515, -604.1505, 0.2392157, 0, 0.7568628, 1,
4.444445, 2.151515, -601.2245, 0.2392157, 0, 0.7568628, 1,
4.484848, 2.151515, -598.3691, 0.3411765, 0, 0.654902, 1,
4.525252, 2.151515, -595.5842, 0.3411765, 0, 0.654902, 1,
4.565657, 2.151515, -592.8698, 0.4470588, 0, 0.5490196, 1,
4.606061, 2.151515, -590.226, 0.4470588, 0, 0.5490196, 1,
4.646465, 2.151515, -587.6528, 0.5490196, 0, 0.4470588, 1,
4.686869, 2.151515, -585.15, 0.5490196, 0, 0.4470588, 1,
4.727273, 2.151515, -582.7178, 0.654902, 0, 0.3411765, 1,
4.767677, 2.151515, -580.3561, 0.654902, 0, 0.3411765, 1,
4.808081, 2.151515, -578.0649, 0.7568628, 0, 0.2392157, 1,
4.848485, 2.151515, -575.8443, 0.7568628, 0, 0.2392157, 1,
4.888889, 2.151515, -573.6942, 0.7568628, 0, 0.2392157, 1,
4.929293, 2.151515, -571.6147, 0.8588235, 0, 0.1372549, 1,
4.969697, 2.151515, -569.6057, 0.8588235, 0, 0.1372549, 1,
5.010101, 2.151515, -567.6672, 0.9647059, 0, 0.03137255, 1,
5.050505, 2.151515, -565.7992, 0.9647059, 0, 0.03137255, 1,
5.090909, 2.151515, -564.0018, 0.9647059, 0, 0.03137255, 1,
5.131313, 2.151515, -562.2749, 1, 0.06666667, 0, 1,
5.171717, 2.151515, -560.6185, 1, 0.06666667, 0, 1,
5.212121, 2.151515, -559.0327, 1, 0.06666667, 0, 1,
5.252525, 2.151515, -557.5175, 1, 0.06666667, 0, 1,
5.292929, 2.151515, -556.0727, 1, 0.1686275, 0, 1,
5.333333, 2.151515, -554.6984, 1, 0.1686275, 0, 1,
5.373737, 2.151515, -553.3948, 1, 0.1686275, 0, 1,
5.414141, 2.151515, -552.1616, 1, 0.1686275, 0, 1,
5.454545, 2.151515, -550.999, 1, 0.2745098, 0, 1,
5.494949, 2.151515, -549.9069, 1, 0.2745098, 0, 1,
5.535354, 2.151515, -548.8853, 1, 0.2745098, 0, 1,
5.575758, 2.151515, -547.9343, 1, 0.2745098, 0, 1,
5.616162, 2.151515, -547.0538, 1, 0.2745098, 0, 1,
5.656566, 2.151515, -546.2438, 1, 0.3764706, 0, 1,
5.69697, 2.151515, -545.5044, 1, 0.3764706, 0, 1,
5.737374, 2.151515, -544.8355, 1, 0.3764706, 0, 1,
5.777778, 2.151515, -544.2371, 1, 0.3764706, 0, 1,
5.818182, 2.151515, -543.7093, 1, 0.3764706, 0, 1,
5.858586, 2.151515, -543.252, 1, 0.3764706, 0, 1,
5.89899, 2.151515, -542.8652, 1, 0.3764706, 0, 1,
5.939394, 2.151515, -542.549, 1, 0.3764706, 0, 1,
5.979798, 2.151515, -542.3033, 1, 0.3764706, 0, 1,
6.020202, 2.151515, -542.1281, 1, 0.3764706, 0, 1,
6.060606, 2.151515, -542.0235, 1, 0.3764706, 0, 1,
6.10101, 2.151515, -541.9894, 1, 0.3764706, 0, 1,
6.141414, 2.151515, -542.0258, 1, 0.3764706, 0, 1,
6.181818, 2.151515, -542.1328, 1, 0.3764706, 0, 1,
6.222222, 2.151515, -542.3102, 1, 0.3764706, 0, 1,
6.262626, 2.151515, -542.5583, 1, 0.3764706, 0, 1,
6.30303, 2.151515, -542.8768, 1, 0.3764706, 0, 1,
6.343434, 2.151515, -543.2659, 1, 0.3764706, 0, 1,
6.383838, 2.151515, -543.7256, 1, 0.3764706, 0, 1,
6.424242, 2.151515, -544.2557, 1, 0.3764706, 0, 1,
6.464646, 2.151515, -544.8564, 1, 0.3764706, 0, 1,
6.505051, 2.151515, -545.5276, 1, 0.3764706, 0, 1,
6.545455, 2.151515, -546.2694, 1, 0.3764706, 0, 1,
6.585859, 2.151515, -547.0817, 1, 0.2745098, 0, 1,
6.626263, 2.151515, -547.9645, 1, 0.2745098, 0, 1,
6.666667, 2.151515, -548.9178, 1, 0.2745098, 0, 1,
6.707071, 2.151515, -549.9418, 1, 0.2745098, 0, 1,
6.747475, 2.151515, -551.0362, 1, 0.2745098, 0, 1,
6.787879, 2.151515, -552.2011, 1, 0.1686275, 0, 1,
6.828283, 2.151515, -553.4366, 1, 0.1686275, 0, 1,
6.868687, 2.151515, -554.7426, 1, 0.1686275, 0, 1,
6.909091, 2.151515, -556.1192, 1, 0.1686275, 0, 1,
6.949495, 2.151515, -557.5663, 1, 0.06666667, 0, 1,
6.989899, 2.151515, -559.0839, 1, 0.06666667, 0, 1,
7.030303, 2.151515, -560.6721, 1, 0.06666667, 0, 1,
7.070707, 2.151515, -562.3307, 1, 0.06666667, 0, 1,
7.111111, 2.151515, -564.0599, 0.9647059, 0, 0.03137255, 1,
7.151515, 2.151515, -565.8597, 0.9647059, 0, 0.03137255, 1,
7.191919, 2.151515, -567.73, 0.9647059, 0, 0.03137255, 1,
7.232323, 2.151515, -569.6708, 0.8588235, 0, 0.1372549, 1,
7.272727, 2.151515, -571.6821, 0.8588235, 0, 0.1372549, 1,
7.313131, 2.151515, -573.764, 0.7568628, 0, 0.2392157, 1,
7.353535, 2.151515, -575.9164, 0.7568628, 0, 0.2392157, 1,
7.393939, 2.151515, -578.1393, 0.7568628, 0, 0.2392157, 1,
7.434343, 2.151515, -580.4329, 0.654902, 0, 0.3411765, 1,
7.474748, 2.151515, -582.7969, 0.654902, 0, 0.3411765, 1,
7.515152, 2.151515, -585.2314, 0.5490196, 0, 0.4470588, 1,
7.555555, 2.151515, -587.7365, 0.5490196, 0, 0.4470588, 1,
7.59596, 2.151515, -590.3121, 0.4470588, 0, 0.5490196, 1,
7.636364, 2.151515, -592.9582, 0.4470588, 0, 0.5490196, 1,
7.676768, 2.151515, -595.6749, 0.3411765, 0, 0.654902, 1,
7.717172, 2.151515, -598.4621, 0.3411765, 0, 0.654902, 1,
7.757576, 2.151515, -601.3198, 0.2392157, 0, 0.7568628, 1,
7.79798, 2.151515, -604.2481, 0.2392157, 0, 0.7568628, 1,
7.838384, 2.151515, -607.2469, 0.1372549, 0, 0.8588235, 1,
7.878788, 2.151515, -610.3162, 0.1372549, 0, 0.8588235, 1,
7.919192, 2.151515, -613.4561, 0.03137255, 0, 0.9647059, 1,
7.959596, 2.151515, -616.6665, 0.04313726, 0.007843138, 0.9921569, 1,
8, 2.151515, -619.9474, 0.04313726, 0.007843138, 0.9921569, 1,
4, 2.20202, -628.314, 0.172549, 0.03137255, 0.9803922, 1,
4.040404, 2.20202, -624.8474, 0.1058824, 0.01960784, 0.9882353, 1,
4.080808, 2.20202, -621.4481, 0.04313726, 0.007843138, 0.9921569, 1,
4.121212, 2.20202, -618.1161, 0.04313726, 0.007843138, 0.9921569, 1,
4.161616, 2.20202, -614.8516, 0.03137255, 0, 0.9647059, 1,
4.20202, 2.20202, -611.6543, 0.03137255, 0, 0.9647059, 1,
4.242424, 2.20202, -608.5244, 0.1372549, 0, 0.8588235, 1,
4.282828, 2.20202, -605.4617, 0.1372549, 0, 0.8588235, 1,
4.323232, 2.20202, -602.4665, 0.2392157, 0, 0.7568628, 1,
4.363636, 2.20202, -599.5385, 0.3411765, 0, 0.654902, 1,
4.40404, 2.20202, -596.6779, 0.3411765, 0, 0.654902, 1,
4.444445, 2.20202, -593.8846, 0.4470588, 0, 0.5490196, 1,
4.484848, 2.20202, -591.1588, 0.4470588, 0, 0.5490196, 1,
4.525252, 2.20202, -588.5001, 0.5490196, 0, 0.4470588, 1,
4.565657, 2.20202, -585.9089, 0.5490196, 0, 0.4470588, 1,
4.606061, 2.20202, -583.3849, 0.654902, 0, 0.3411765, 1,
4.646465, 2.20202, -580.9283, 0.654902, 0, 0.3411765, 1,
4.686869, 2.20202, -578.5391, 0.7568628, 0, 0.2392157, 1,
4.727273, 2.20202, -576.2172, 0.7568628, 0, 0.2392157, 1,
4.767677, 2.20202, -573.9626, 0.7568628, 0, 0.2392157, 1,
4.808081, 2.20202, -571.7753, 0.8588235, 0, 0.1372549, 1,
4.848485, 2.20202, -569.6554, 0.8588235, 0, 0.1372549, 1,
4.888889, 2.20202, -567.6028, 0.9647059, 0, 0.03137255, 1,
4.929293, 2.20202, -565.6176, 0.9647059, 0, 0.03137255, 1,
4.969697, 2.20202, -563.6996, 0.9647059, 0, 0.03137255, 1,
5.010101, 2.20202, -561.8491, 1, 0.06666667, 0, 1,
5.050505, 2.20202, -560.0658, 1, 0.06666667, 0, 1,
5.090909, 2.20202, -558.3499, 1, 0.06666667, 0, 1,
5.131313, 2.20202, -556.7013, 1, 0.1686275, 0, 1,
5.171717, 2.20202, -555.1201, 1, 0.1686275, 0, 1,
5.212121, 2.20202, -553.6061, 1, 0.1686275, 0, 1,
5.252525, 2.20202, -552.1595, 1, 0.1686275, 0, 1,
5.292929, 2.20202, -550.7803, 1, 0.2745098, 0, 1,
5.333333, 2.20202, -549.4684, 1, 0.2745098, 0, 1,
5.373737, 2.20202, -548.2238, 1, 0.2745098, 0, 1,
5.414141, 2.20202, -547.0466, 1, 0.2745098, 0, 1,
5.454545, 2.20202, -545.9366, 1, 0.3764706, 0, 1,
5.494949, 2.20202, -544.8941, 1, 0.3764706, 0, 1,
5.535354, 2.20202, -543.9188, 1, 0.3764706, 0, 1,
5.575758, 2.20202, -543.0109, 1, 0.3764706, 0, 1,
5.616162, 2.20202, -542.1703, 1, 0.3764706, 0, 1,
5.656566, 2.20202, -541.3972, 1, 0.3764706, 0, 1,
5.69697, 2.20202, -540.6912, 1, 0.4823529, 0, 1,
5.737374, 2.20202, -540.0527, 1, 0.4823529, 0, 1,
5.777778, 2.20202, -539.4814, 1, 0.4823529, 0, 1,
5.818182, 2.20202, -538.9775, 1, 0.4823529, 0, 1,
5.858586, 2.20202, -538.541, 1, 0.4823529, 0, 1,
5.89899, 2.20202, -538.1718, 1, 0.4823529, 0, 1,
5.939394, 2.20202, -537.8699, 1, 0.4823529, 0, 1,
5.979798, 2.20202, -537.6353, 1, 0.4823529, 0, 1,
6.020202, 2.20202, -537.4681, 1, 0.4823529, 0, 1,
6.060606, 2.20202, -537.3682, 1, 0.4823529, 0, 1,
6.10101, 2.20202, -537.3356, 1, 0.4823529, 0, 1,
6.141414, 2.20202, -537.3704, 1, 0.4823529, 0, 1,
6.181818, 2.20202, -537.4725, 1, 0.4823529, 0, 1,
6.222222, 2.20202, -537.642, 1, 0.4823529, 0, 1,
6.262626, 2.20202, -537.8787, 1, 0.4823529, 0, 1,
6.30303, 2.20202, -538.1829, 1, 0.4823529, 0, 1,
6.343434, 2.20202, -538.5543, 1, 0.4823529, 0, 1,
6.383838, 2.20202, -538.9931, 1, 0.4823529, 0, 1,
6.424242, 2.20202, -539.4992, 1, 0.4823529, 0, 1,
6.464646, 2.20202, -540.0726, 1, 0.4823529, 0, 1,
6.505051, 2.20202, -540.7134, 1, 0.4823529, 0, 1,
6.545455, 2.20202, -541.4216, 1, 0.3764706, 0, 1,
6.585859, 2.20202, -542.197, 1, 0.3764706, 0, 1,
6.626263, 2.20202, -543.0398, 1, 0.3764706, 0, 1,
6.666667, 2.20202, -543.95, 1, 0.3764706, 0, 1,
6.707071, 2.20202, -544.9274, 1, 0.3764706, 0, 1,
6.747475, 2.20202, -545.9722, 1, 0.3764706, 0, 1,
6.787879, 2.20202, -547.0843, 1, 0.2745098, 0, 1,
6.828283, 2.20202, -548.2638, 1, 0.2745098, 0, 1,
6.868687, 2.20202, -549.5106, 1, 0.2745098, 0, 1,
6.909091, 2.20202, -550.8247, 1, 0.2745098, 0, 1,
6.949495, 2.20202, -552.2062, 1, 0.1686275, 0, 1,
6.989899, 2.20202, -553.655, 1, 0.1686275, 0, 1,
7.030303, 2.20202, -555.1711, 1, 0.1686275, 0, 1,
7.070707, 2.20202, -556.7546, 1, 0.1686275, 0, 1,
7.111111, 2.20202, -558.4054, 1, 0.06666667, 0, 1,
7.151515, 2.20202, -560.1235, 1, 0.06666667, 0, 1,
7.191919, 2.20202, -561.909, 1, 0.06666667, 0, 1,
7.232323, 2.20202, -563.7618, 0.9647059, 0, 0.03137255, 1,
7.272727, 2.20202, -565.6819, 0.9647059, 0, 0.03137255, 1,
7.313131, 2.20202, -567.6694, 0.9647059, 0, 0.03137255, 1,
7.353535, 2.20202, -569.7242, 0.8588235, 0, 0.1372549, 1,
7.393939, 2.20202, -571.8464, 0.8588235, 0, 0.1372549, 1,
7.434343, 2.20202, -574.0358, 0.7568628, 0, 0.2392157, 1,
7.474748, 2.20202, -576.2927, 0.7568628, 0, 0.2392157, 1,
7.515152, 2.20202, -578.6168, 0.654902, 0, 0.3411765, 1,
7.555555, 2.20202, -581.0083, 0.654902, 0, 0.3411765, 1,
7.59596, 2.20202, -583.4671, 0.654902, 0, 0.3411765, 1,
7.636364, 2.20202, -585.9932, 0.5490196, 0, 0.4470588, 1,
7.676768, 2.20202, -588.5867, 0.5490196, 0, 0.4470588, 1,
7.717172, 2.20202, -591.2476, 0.4470588, 0, 0.5490196, 1,
7.757576, 2.20202, -593.9757, 0.4470588, 0, 0.5490196, 1,
7.79798, 2.20202, -596.7712, 0.3411765, 0, 0.654902, 1,
7.838384, 2.20202, -599.634, 0.3411765, 0, 0.654902, 1,
7.878788, 2.20202, -602.5641, 0.2392157, 0, 0.7568628, 1,
7.919192, 2.20202, -605.5616, 0.1372549, 0, 0.8588235, 1,
7.959596, 2.20202, -608.6265, 0.1372549, 0, 0.8588235, 1,
8, 2.20202, -611.7587, 0.03137255, 0, 0.9647059, 1,
4, 2.252525, -620.1392, 0.04313726, 0.007843138, 0.9921569, 1,
4.040404, 2.252525, -616.8262, 0.04313726, 0.007843138, 0.9921569, 1,
4.080808, 2.252525, -613.5777, 0.03137255, 0, 0.9647059, 1,
4.121212, 2.252525, -610.3935, 0.1372549, 0, 0.8588235, 1,
4.161616, 2.252525, -607.2736, 0.1372549, 0, 0.8588235, 1,
4.20202, 2.252525, -604.2181, 0.2392157, 0, 0.7568628, 1,
4.242424, 2.252525, -601.227, 0.2392157, 0, 0.7568628, 1,
4.282828, 2.252525, -598.3002, 0.3411765, 0, 0.654902, 1,
4.323232, 2.252525, -595.4377, 0.3411765, 0, 0.654902, 1,
4.363636, 2.252525, -592.6396, 0.4470588, 0, 0.5490196, 1,
4.40404, 2.252525, -589.9058, 0.4470588, 0, 0.5490196, 1,
4.444445, 2.252525, -587.2364, 0.5490196, 0, 0.4470588, 1,
4.484848, 2.252525, -584.6313, 0.5490196, 0, 0.4470588, 1,
4.525252, 2.252525, -582.0906, 0.654902, 0, 0.3411765, 1,
4.565657, 2.252525, -579.6143, 0.654902, 0, 0.3411765, 1,
4.606061, 2.252525, -577.2023, 0.7568628, 0, 0.2392157, 1,
4.646465, 2.252525, -574.8546, 0.7568628, 0, 0.2392157, 1,
4.686869, 2.252525, -572.5712, 0.8588235, 0, 0.1372549, 1,
4.727273, 2.252525, -570.3523, 0.8588235, 0, 0.1372549, 1,
4.767677, 2.252525, -568.1977, 0.8588235, 0, 0.1372549, 1,
4.808081, 2.252525, -566.1074, 0.9647059, 0, 0.03137255, 1,
4.848485, 2.252525, -564.0815, 0.9647059, 0, 0.03137255, 1,
4.888889, 2.252525, -562.1199, 1, 0.06666667, 0, 1,
4.929293, 2.252525, -560.2227, 1, 0.06666667, 0, 1,
4.969697, 2.252525, -558.3898, 1, 0.06666667, 0, 1,
5.010101, 2.252525, -556.6213, 1, 0.1686275, 0, 1,
5.050505, 2.252525, -554.9171, 1, 0.1686275, 0, 1,
5.090909, 2.252525, -553.2772, 1, 0.1686275, 0, 1,
5.131313, 2.252525, -551.7017, 1, 0.2745098, 0, 1,
5.171717, 2.252525, -550.1906, 1, 0.2745098, 0, 1,
5.212121, 2.252525, -548.7438, 1, 0.2745098, 0, 1,
5.252525, 2.252525, -547.3614, 1, 0.2745098, 0, 1,
5.292929, 2.252525, -546.0433, 1, 0.3764706, 0, 1,
5.333333, 2.252525, -544.7896, 1, 0.3764706, 0, 1,
5.373737, 2.252525, -543.6002, 1, 0.3764706, 0, 1,
5.414141, 2.252525, -542.4752, 1, 0.3764706, 0, 1,
5.454545, 2.252525, -541.4144, 1, 0.3764706, 0, 1,
5.494949, 2.252525, -540.4181, 1, 0.4823529, 0, 1,
5.535354, 2.252525, -539.4861, 1, 0.4823529, 0, 1,
5.575758, 2.252525, -538.6185, 1, 0.4823529, 0, 1,
5.616162, 2.252525, -537.8151, 1, 0.4823529, 0, 1,
5.656566, 2.252525, -537.0762, 1, 0.4823529, 0, 1,
5.69697, 2.252525, -536.4016, 1, 0.4823529, 0, 1,
5.737374, 2.252525, -535.7913, 1, 0.4823529, 0, 1,
5.777778, 2.252525, -535.2454, 1, 0.5843138, 0, 1,
5.818182, 2.252525, -534.7639, 1, 0.5843138, 0, 1,
5.858586, 2.252525, -534.3467, 1, 0.5843138, 0, 1,
5.89899, 2.252525, -533.9938, 1, 0.5843138, 0, 1,
5.939394, 2.252525, -533.7053, 1, 0.5843138, 0, 1,
5.979798, 2.252525, -533.4811, 1, 0.5843138, 0, 1,
6.020202, 2.252525, -533.3214, 1, 0.5843138, 0, 1,
6.060606, 2.252525, -533.2259, 1, 0.5843138, 0, 1,
6.10101, 2.252525, -533.1948, 1, 0.5843138, 0, 1,
6.141414, 2.252525, -533.228, 1, 0.5843138, 0, 1,
6.181818, 2.252525, -533.3256, 1, 0.5843138, 0, 1,
6.222222, 2.252525, -533.4875, 1, 0.5843138, 0, 1,
6.262626, 2.252525, -533.7138, 1, 0.5843138, 0, 1,
6.30303, 2.252525, -534.0045, 1, 0.5843138, 0, 1,
6.343434, 2.252525, -534.3594, 1, 0.5843138, 0, 1,
6.383838, 2.252525, -534.7787, 1, 0.5843138, 0, 1,
6.424242, 2.252525, -535.2624, 1, 0.5843138, 0, 1,
6.464646, 2.252525, -535.8104, 1, 0.4823529, 0, 1,
6.505051, 2.252525, -536.4228, 1, 0.4823529, 0, 1,
6.545455, 2.252525, -537.0995, 1, 0.4823529, 0, 1,
6.585859, 2.252525, -537.8406, 1, 0.4823529, 0, 1,
6.626263, 2.252525, -538.6461, 1, 0.4823529, 0, 1,
6.666667, 2.252525, -539.5158, 1, 0.4823529, 0, 1,
6.707071, 2.252525, -540.4499, 1, 0.4823529, 0, 1,
6.747475, 2.252525, -541.4484, 1, 0.3764706, 0, 1,
6.787879, 2.252525, -542.5112, 1, 0.3764706, 0, 1,
6.828283, 2.252525, -543.6384, 1, 0.3764706, 0, 1,
6.868687, 2.252525, -544.8299, 1, 0.3764706, 0, 1,
6.909091, 2.252525, -546.0858, 1, 0.3764706, 0, 1,
6.949495, 2.252525, -547.4059, 1, 0.2745098, 0, 1,
6.989899, 2.252525, -548.7905, 1, 0.2745098, 0, 1,
7.030303, 2.252525, -550.2394, 1, 0.2745098, 0, 1,
7.070707, 2.252525, -551.7527, 1, 0.2745098, 0, 1,
7.111111, 2.252525, -553.3303, 1, 0.1686275, 0, 1,
7.151515, 2.252525, -554.9722, 1, 0.1686275, 0, 1,
7.191919, 2.252525, -556.6785, 1, 0.1686275, 0, 1,
7.232323, 2.252525, -558.4492, 1, 0.06666667, 0, 1,
7.272727, 2.252525, -560.2842, 1, 0.06666667, 0, 1,
7.313131, 2.252525, -562.1835, 1, 0.06666667, 0, 1,
7.353535, 2.252525, -564.1472, 0.9647059, 0, 0.03137255, 1,
7.393939, 2.252525, -566.1753, 0.9647059, 0, 0.03137255, 1,
7.434343, 2.252525, -568.2677, 0.8588235, 0, 0.1372549, 1,
7.474748, 2.252525, -570.4244, 0.8588235, 0, 0.1372549, 1,
7.515152, 2.252525, -572.6455, 0.8588235, 0, 0.1372549, 1,
7.555555, 2.252525, -574.931, 0.7568628, 0, 0.2392157, 1,
7.59596, 2.252525, -577.2808, 0.7568628, 0, 0.2392157, 1,
7.636364, 2.252525, -579.6949, 0.654902, 0, 0.3411765, 1,
7.676768, 2.252525, -582.1734, 0.654902, 0, 0.3411765, 1,
7.717172, 2.252525, -584.7162, 0.5490196, 0, 0.4470588, 1,
7.757576, 2.252525, -587.3234, 0.5490196, 0, 0.4470588, 1,
7.79798, 2.252525, -589.9949, 0.4470588, 0, 0.5490196, 1,
7.838384, 2.252525, -592.7308, 0.4470588, 0, 0.5490196, 1,
7.878788, 2.252525, -595.5311, 0.3411765, 0, 0.654902, 1,
7.919192, 2.252525, -598.3956, 0.3411765, 0, 0.654902, 1,
7.959596, 2.252525, -601.3246, 0.2392157, 0, 0.7568628, 1,
8, 2.252525, -604.3179, 0.2392157, 0, 0.7568628, 1,
4, 2.30303, -612.6907, 0.03137255, 0, 0.9647059, 1,
4.040404, 2.30303, -609.5215, 0.1372549, 0, 0.8588235, 1,
4.080808, 2.30303, -606.4139, 0.1372549, 0, 0.8588235, 1,
4.121212, 2.30303, -603.3678, 0.2392157, 0, 0.7568628, 1,
4.161616, 2.30303, -600.3833, 0.2392157, 0, 0.7568628, 1,
4.20202, 2.30303, -597.4603, 0.3411765, 0, 0.654902, 1,
4.242424, 2.30303, -594.5989, 0.4470588, 0, 0.5490196, 1,
4.282828, 2.30303, -591.7991, 0.4470588, 0, 0.5490196, 1,
4.323232, 2.30303, -589.0608, 0.5490196, 0, 0.4470588, 1,
4.363636, 2.30303, -586.384, 0.5490196, 0, 0.4470588, 1,
4.40404, 2.30303, -583.7689, 0.654902, 0, 0.3411765, 1,
4.444445, 2.30303, -581.2153, 0.654902, 0, 0.3411765, 1,
4.484848, 2.30303, -578.7232, 0.654902, 0, 0.3411765, 1,
4.525252, 2.30303, -576.2927, 0.7568628, 0, 0.2392157, 1,
4.565657, 2.30303, -573.9238, 0.7568628, 0, 0.2392157, 1,
4.606061, 2.30303, -571.6163, 0.8588235, 0, 0.1372549, 1,
4.646465, 2.30303, -569.3705, 0.8588235, 0, 0.1372549, 1,
4.686869, 2.30303, -567.1863, 0.9647059, 0, 0.03137255, 1,
4.727273, 2.30303, -565.0635, 0.9647059, 0, 0.03137255, 1,
4.767677, 2.30303, -563.0024, 0.9647059, 0, 0.03137255, 1,
4.808081, 2.30303, -561.0027, 1, 0.06666667, 0, 1,
4.848485, 2.30303, -559.0648, 1, 0.06666667, 0, 1,
4.888889, 2.30303, -557.1882, 1, 0.06666667, 0, 1,
4.929293, 2.30303, -555.3733, 1, 0.1686275, 0, 1,
4.969697, 2.30303, -553.6199, 1, 0.1686275, 0, 1,
5.010101, 2.30303, -551.9282, 1, 0.1686275, 0, 1,
5.050505, 2.30303, -550.2979, 1, 0.2745098, 0, 1,
5.090909, 2.30303, -548.7292, 1, 0.2745098, 0, 1,
5.131313, 2.30303, -547.222, 1, 0.2745098, 0, 1,
5.171717, 2.30303, -545.7764, 1, 0.3764706, 0, 1,
5.212121, 2.30303, -544.3924, 1, 0.3764706, 0, 1,
5.252525, 2.30303, -543.0699, 1, 0.3764706, 0, 1,
5.292929, 2.30303, -541.809, 1, 0.3764706, 0, 1,
5.333333, 2.30303, -540.6097, 1, 0.4823529, 0, 1,
5.373737, 2.30303, -539.4719, 1, 0.4823529, 0, 1,
5.414141, 2.30303, -538.3956, 1, 0.4823529, 0, 1,
5.454545, 2.30303, -537.381, 1, 0.4823529, 0, 1,
5.494949, 2.30303, -536.4279, 1, 0.4823529, 0, 1,
5.535354, 2.30303, -535.5363, 1, 0.5843138, 0, 1,
5.575758, 2.30303, -534.7062, 1, 0.5843138, 0, 1,
5.616162, 2.30303, -533.9378, 1, 0.5843138, 0, 1,
5.656566, 2.30303, -533.2309, 1, 0.5843138, 0, 1,
5.69697, 2.30303, -532.5856, 1, 0.5843138, 0, 1,
5.737374, 2.30303, -532.0018, 1, 0.5843138, 0, 1,
5.777778, 2.30303, -531.4796, 1, 0.5843138, 0, 1,
5.818182, 2.30303, -531.0189, 1, 0.5843138, 0, 1,
5.858586, 2.30303, -530.6198, 1, 0.5843138, 0, 1,
5.89899, 2.30303, -530.2822, 1, 0.5843138, 0, 1,
5.939394, 2.30303, -530.0063, 1, 0.6862745, 0, 1,
5.979798, 2.30303, -529.7918, 1, 0.6862745, 0, 1,
6.020202, 2.30303, -529.6389, 1, 0.6862745, 0, 1,
6.060606, 2.30303, -529.5476, 1, 0.6862745, 0, 1,
6.10101, 2.30303, -529.5179, 1, 0.6862745, 0, 1,
6.141414, 2.30303, -529.5497, 1, 0.6862745, 0, 1,
6.181818, 2.30303, -529.643, 1, 0.6862745, 0, 1,
6.222222, 2.30303, -529.7979, 1, 0.6862745, 0, 1,
6.262626, 2.30303, -530.0144, 1, 0.6862745, 0, 1,
6.30303, 2.30303, -530.2924, 1, 0.5843138, 0, 1,
6.343434, 2.30303, -530.632, 1, 0.5843138, 0, 1,
6.383838, 2.30303, -531.0331, 1, 0.5843138, 0, 1,
6.424242, 2.30303, -531.4958, 1, 0.5843138, 0, 1,
6.464646, 2.30303, -532.0201, 1, 0.5843138, 0, 1,
6.505051, 2.30303, -532.6059, 1, 0.5843138, 0, 1,
6.545455, 2.30303, -533.2532, 1, 0.5843138, 0, 1,
6.585859, 2.30303, -533.9622, 1, 0.5843138, 0, 1,
6.626263, 2.30303, -534.7327, 1, 0.5843138, 0, 1,
6.666667, 2.30303, -535.5647, 1, 0.5843138, 0, 1,
6.707071, 2.30303, -536.4583, 1, 0.4823529, 0, 1,
6.747475, 2.30303, -537.4135, 1, 0.4823529, 0, 1,
6.787879, 2.30303, -538.4302, 1, 0.4823529, 0, 1,
6.828283, 2.30303, -539.5084, 1, 0.4823529, 0, 1,
6.868687, 2.30303, -540.6483, 1, 0.4823529, 0, 1,
6.909091, 2.30303, -541.8496, 1, 0.3764706, 0, 1,
6.949495, 2.30303, -543.1125, 1, 0.3764706, 0, 1,
6.989899, 2.30303, -544.4371, 1, 0.3764706, 0, 1,
7.030303, 2.30303, -545.8231, 1, 0.3764706, 0, 1,
7.070707, 2.30303, -547.2708, 1, 0.2745098, 0, 1,
7.111111, 2.30303, -548.7799, 1, 0.2745098, 0, 1,
7.151515, 2.30303, -550.3506, 1, 0.2745098, 0, 1,
7.191919, 2.30303, -551.9829, 1, 0.1686275, 0, 1,
7.232323, 2.30303, -553.6768, 1, 0.1686275, 0, 1,
7.272727, 2.30303, -555.4322, 1, 0.1686275, 0, 1,
7.313131, 2.30303, -557.2491, 1, 0.06666667, 0, 1,
7.353535, 2.30303, -559.1276, 1, 0.06666667, 0, 1,
7.393939, 2.30303, -561.0677, 1, 0.06666667, 0, 1,
7.434343, 2.30303, -563.0693, 0.9647059, 0, 0.03137255, 1,
7.474748, 2.30303, -565.1326, 0.9647059, 0, 0.03137255, 1,
7.515152, 2.30303, -567.2573, 0.9647059, 0, 0.03137255, 1,
7.555555, 2.30303, -569.4436, 0.8588235, 0, 0.1372549, 1,
7.59596, 2.30303, -571.6915, 0.8588235, 0, 0.1372549, 1,
7.636364, 2.30303, -574.0009, 0.7568628, 0, 0.2392157, 1,
7.676768, 2.30303, -576.3718, 0.7568628, 0, 0.2392157, 1,
7.717172, 2.30303, -578.8044, 0.654902, 0, 0.3411765, 1,
7.757576, 2.30303, -581.2985, 0.654902, 0, 0.3411765, 1,
7.79798, 2.30303, -583.8541, 0.654902, 0, 0.3411765, 1,
7.838384, 2.30303, -586.4713, 0.5490196, 0, 0.4470588, 1,
7.878788, 2.30303, -589.1501, 0.5490196, 0, 0.4470588, 1,
7.919192, 2.30303, -591.8904, 0.4470588, 0, 0.5490196, 1,
7.959596, 2.30303, -594.6923, 0.4470588, 0, 0.5490196, 1,
8, 2.30303, -597.5557, 0.3411765, 0, 0.654902, 1,
4, 2.353535, -605.9028, 0.1372549, 0, 0.8588235, 1,
4.040404, 2.353535, -602.8682, 0.2392157, 0, 0.7568628, 1,
4.080808, 2.353535, -599.8925, 0.3411765, 0, 0.654902, 1,
4.121212, 2.353535, -596.9758, 0.3411765, 0, 0.654902, 1,
4.161616, 2.353535, -594.1179, 0.4470588, 0, 0.5490196, 1,
4.20202, 2.353535, -591.3191, 0.4470588, 0, 0.5490196, 1,
4.242424, 2.353535, -588.5792, 0.5490196, 0, 0.4470588, 1,
4.282828, 2.353535, -585.8982, 0.5490196, 0, 0.4470588, 1,
4.323232, 2.353535, -583.2762, 0.654902, 0, 0.3411765, 1,
4.363636, 2.353535, -580.7131, 0.654902, 0, 0.3411765, 1,
4.40404, 2.353535, -578.2089, 0.7568628, 0, 0.2392157, 1,
4.444445, 2.353535, -575.7637, 0.7568628, 0, 0.2392157, 1,
4.484848, 2.353535, -573.3775, 0.7568628, 0, 0.2392157, 1,
4.525252, 2.353535, -571.0502, 0.8588235, 0, 0.1372549, 1,
4.565657, 2.353535, -568.7819, 0.8588235, 0, 0.1372549, 1,
4.606061, 2.353535, -566.5724, 0.9647059, 0, 0.03137255, 1,
4.646465, 2.353535, -564.4219, 0.9647059, 0, 0.03137255, 1,
4.686869, 2.353535, -562.3304, 1, 0.06666667, 0, 1,
4.727273, 2.353535, -560.2978, 1, 0.06666667, 0, 1,
4.767677, 2.353535, -558.3242, 1, 0.06666667, 0, 1,
4.808081, 2.353535, -556.4095, 1, 0.1686275, 0, 1,
4.848485, 2.353535, -554.5537, 1, 0.1686275, 0, 1,
4.888889, 2.353535, -552.7569, 1, 0.1686275, 0, 1,
4.929293, 2.353535, -551.019, 1, 0.2745098, 0, 1,
4.969697, 2.353535, -549.3401, 1, 0.2745098, 0, 1,
5.010101, 2.353535, -547.7201, 1, 0.2745098, 0, 1,
5.050505, 2.353535, -546.1591, 1, 0.3764706, 0, 1,
5.090909, 2.353535, -544.657, 1, 0.3764706, 0, 1,
5.131313, 2.353535, -543.2138, 1, 0.3764706, 0, 1,
5.171717, 2.353535, -541.8296, 1, 0.3764706, 0, 1,
5.212121, 2.353535, -540.5043, 1, 0.4823529, 0, 1,
5.252525, 2.353535, -539.238, 1, 0.4823529, 0, 1,
5.292929, 2.353535, -538.0306, 1, 0.4823529, 0, 1,
5.333333, 2.353535, -536.8822, 1, 0.4823529, 0, 1,
5.373737, 2.353535, -535.7927, 1, 0.4823529, 0, 1,
5.414141, 2.353535, -534.7622, 1, 0.5843138, 0, 1,
5.454545, 2.353535, -533.7906, 1, 0.5843138, 0, 1,
5.494949, 2.353535, -532.8779, 1, 0.5843138, 0, 1,
5.535354, 2.353535, -532.0242, 1, 0.5843138, 0, 1,
5.575758, 2.353535, -531.2294, 1, 0.5843138, 0, 1,
5.616162, 2.353535, -530.4937, 1, 0.5843138, 0, 1,
5.656566, 2.353535, -529.8168, 1, 0.6862745, 0, 1,
5.69697, 2.353535, -529.1988, 1, 0.6862745, 0, 1,
5.737374, 2.353535, -528.6398, 1, 0.6862745, 0, 1,
5.777778, 2.353535, -528.1398, 1, 0.6862745, 0, 1,
5.818182, 2.353535, -527.6987, 1, 0.6862745, 0, 1,
5.858586, 2.353535, -527.3165, 1, 0.6862745, 0, 1,
5.89899, 2.353535, -526.9933, 1, 0.6862745, 0, 1,
5.939394, 2.353535, -526.729, 1, 0.6862745, 0, 1,
5.979798, 2.353535, -526.5237, 1, 0.6862745, 0, 1,
6.020202, 2.353535, -526.3773, 1, 0.6862745, 0, 1,
6.060606, 2.353535, -526.2899, 1, 0.6862745, 0, 1,
6.10101, 2.353535, -526.2614, 1, 0.6862745, 0, 1,
6.141414, 2.353535, -526.2917, 1, 0.6862745, 0, 1,
6.181818, 2.353535, -526.3812, 1, 0.6862745, 0, 1,
6.222222, 2.353535, -526.5295, 1, 0.6862745, 0, 1,
6.262626, 2.353535, -526.7368, 1, 0.6862745, 0, 1,
6.30303, 2.353535, -527.003, 1, 0.6862745, 0, 1,
6.343434, 2.353535, -527.3281, 1, 0.6862745, 0, 1,
6.383838, 2.353535, -527.7123, 1, 0.6862745, 0, 1,
6.424242, 2.353535, -528.1553, 1, 0.6862745, 0, 1,
6.464646, 2.353535, -528.6573, 1, 0.6862745, 0, 1,
6.505051, 2.353535, -529.2183, 1, 0.6862745, 0, 1,
6.545455, 2.353535, -529.8381, 1, 0.6862745, 0, 1,
6.585859, 2.353535, -530.517, 1, 0.5843138, 0, 1,
6.626263, 2.353535, -531.2547, 1, 0.5843138, 0, 1,
6.666667, 2.353535, -532.0515, 1, 0.5843138, 0, 1,
6.707071, 2.353535, -532.9071, 1, 0.5843138, 0, 1,
6.747475, 2.353535, -533.8217, 1, 0.5843138, 0, 1,
6.787879, 2.353535, -534.7952, 1, 0.5843138, 0, 1,
6.828283, 2.353535, -535.8277, 1, 0.4823529, 0, 1,
6.868687, 2.353535, -536.9191, 1, 0.4823529, 0, 1,
6.909091, 2.353535, -538.0695, 1, 0.4823529, 0, 1,
6.949495, 2.353535, -539.2789, 1, 0.4823529, 0, 1,
6.989899, 2.353535, -540.5471, 1, 0.4823529, 0, 1,
7.030303, 2.353535, -541.8743, 1, 0.3764706, 0, 1,
7.070707, 2.353535, -543.2605, 1, 0.3764706, 0, 1,
7.111111, 2.353535, -544.7056, 1, 0.3764706, 0, 1,
7.151515, 2.353535, -546.2096, 1, 0.3764706, 0, 1,
7.191919, 2.353535, -547.7726, 1, 0.2745098, 0, 1,
7.232323, 2.353535, -549.3945, 1, 0.2745098, 0, 1,
7.272727, 2.353535, -551.0754, 1, 0.2745098, 0, 1,
7.313131, 2.353535, -552.8152, 1, 0.1686275, 0, 1,
7.353535, 2.353535, -554.614, 1, 0.1686275, 0, 1,
7.393939, 2.353535, -556.4717, 1, 0.1686275, 0, 1,
7.434343, 2.353535, -558.3883, 1, 0.06666667, 0, 1,
7.474748, 2.353535, -560.3639, 1, 0.06666667, 0, 1,
7.515152, 2.353535, -562.3984, 1, 0.06666667, 0, 1,
7.555555, 2.353535, -564.4919, 0.9647059, 0, 0.03137255, 1,
7.59596, 2.353535, -566.6443, 0.9647059, 0, 0.03137255, 1,
7.636364, 2.353535, -568.8557, 0.8588235, 0, 0.1372549, 1,
7.676768, 2.353535, -571.126, 0.8588235, 0, 0.1372549, 1,
7.717172, 2.353535, -573.4553, 0.7568628, 0, 0.2392157, 1,
7.757576, 2.353535, -575.8434, 0.7568628, 0, 0.2392157, 1,
7.79798, 2.353535, -578.2906, 0.7568628, 0, 0.2392157, 1,
7.838384, 2.353535, -580.7966, 0.654902, 0, 0.3411765, 1,
7.878788, 2.353535, -583.3617, 0.654902, 0, 0.3411765, 1,
7.919192, 2.353535, -585.9857, 0.5490196, 0, 0.4470588, 1,
7.959596, 2.353535, -588.6686, 0.5490196, 0, 0.4470588, 1,
8, 2.353535, -591.4105, 0.4470588, 0, 0.5490196, 1,
4, 2.40404, -599.7167, 0.3411765, 0, 0.654902, 1,
4.040404, 2.40404, -596.8082, 0.3411765, 0, 0.654902, 1,
4.080808, 2.40404, -593.9562, 0.4470588, 0, 0.5490196, 1,
4.121212, 2.40404, -591.1608, 0.4470588, 0, 0.5490196, 1,
4.161616, 2.40404, -588.4218, 0.5490196, 0, 0.4470588, 1,
4.20202, 2.40404, -585.7393, 0.5490196, 0, 0.4470588, 1,
4.242424, 2.40404, -583.1133, 0.654902, 0, 0.3411765, 1,
4.282828, 2.40404, -580.5438, 0.654902, 0, 0.3411765, 1,
4.323232, 2.40404, -578.0308, 0.7568628, 0, 0.2392157, 1,
4.363636, 2.40404, -575.5742, 0.7568628, 0, 0.2392157, 1,
4.40404, 2.40404, -573.1742, 0.8588235, 0, 0.1372549, 1,
4.444445, 2.40404, -570.8307, 0.8588235, 0, 0.1372549, 1,
4.484848, 2.40404, -568.5436, 0.8588235, 0, 0.1372549, 1,
4.525252, 2.40404, -566.313, 0.9647059, 0, 0.03137255, 1,
4.565657, 2.40404, -564.139, 0.9647059, 0, 0.03137255, 1,
4.606061, 2.40404, -562.0214, 1, 0.06666667, 0, 1,
4.646465, 2.40404, -559.9604, 1, 0.06666667, 0, 1,
4.686869, 2.40404, -557.9558, 1, 0.06666667, 0, 1,
4.727273, 2.40404, -556.0077, 1, 0.1686275, 0, 1,
4.767677, 2.40404, -554.1161, 1, 0.1686275, 0, 1,
4.808081, 2.40404, -552.281, 1, 0.1686275, 0, 1,
4.848485, 2.40404, -550.5024, 1, 0.2745098, 0, 1,
4.888889, 2.40404, -548.7803, 1, 0.2745098, 0, 1,
4.929293, 2.40404, -547.1147, 1, 0.2745098, 0, 1,
4.969697, 2.40404, -545.5056, 1, 0.3764706, 0, 1,
5.010101, 2.40404, -543.9529, 1, 0.3764706, 0, 1,
5.050505, 2.40404, -542.4568, 1, 0.3764706, 0, 1,
5.090909, 2.40404, -541.0172, 1, 0.4823529, 0, 1,
5.131313, 2.40404, -539.634, 1, 0.4823529, 0, 1,
5.171717, 2.40404, -538.3074, 1, 0.4823529, 0, 1,
5.212121, 2.40404, -537.0372, 1, 0.4823529, 0, 1,
5.252525, 2.40404, -535.8235, 1, 0.4823529, 0, 1,
5.292929, 2.40404, -534.6663, 1, 0.5843138, 0, 1,
5.333333, 2.40404, -533.5657, 1, 0.5843138, 0, 1,
5.373737, 2.40404, -532.5214, 1, 0.5843138, 0, 1,
5.414141, 2.40404, -531.5338, 1, 0.5843138, 0, 1,
5.454545, 2.40404, -530.6025, 1, 0.5843138, 0, 1,
5.494949, 2.40404, -529.7278, 1, 0.6862745, 0, 1,
5.535354, 2.40404, -528.9096, 1, 0.6862745, 0, 1,
5.575758, 2.40404, -528.1479, 1, 0.6862745, 0, 1,
5.616162, 2.40404, -527.4426, 1, 0.6862745, 0, 1,
5.656566, 2.40404, -526.7939, 1, 0.6862745, 0, 1,
5.69697, 2.40404, -526.2017, 1, 0.6862745, 0, 1,
5.737374, 2.40404, -525.6659, 1, 0.6862745, 0, 1,
5.777778, 2.40404, -525.1866, 1, 0.6862745, 0, 1,
5.818182, 2.40404, -524.7639, 1, 0.7921569, 0, 1,
5.858586, 2.40404, -524.3976, 1, 0.7921569, 0, 1,
5.89899, 2.40404, -524.0878, 1, 0.7921569, 0, 1,
5.939394, 2.40404, -523.8345, 1, 0.7921569, 0, 1,
5.979798, 2.40404, -523.6378, 1, 0.7921569, 0, 1,
6.020202, 2.40404, -523.4974, 1, 0.7921569, 0, 1,
6.060606, 2.40404, -523.4136, 1, 0.7921569, 0, 1,
6.10101, 2.40404, -523.3863, 1, 0.7921569, 0, 1,
6.141414, 2.40404, -523.4155, 1, 0.7921569, 0, 1,
6.181818, 2.40404, -523.5012, 1, 0.7921569, 0, 1,
6.222222, 2.40404, -523.6433, 1, 0.7921569, 0, 1,
6.262626, 2.40404, -523.842, 1, 0.7921569, 0, 1,
6.30303, 2.40404, -524.0972, 1, 0.7921569, 0, 1,
6.343434, 2.40404, -524.4088, 1, 0.7921569, 0, 1,
6.383838, 2.40404, -524.7769, 1, 0.7921569, 0, 1,
6.424242, 2.40404, -525.2015, 1, 0.6862745, 0, 1,
6.464646, 2.40404, -525.6827, 1, 0.6862745, 0, 1,
6.505051, 2.40404, -526.2203, 1, 0.6862745, 0, 1,
6.545455, 2.40404, -526.8144, 1, 0.6862745, 0, 1,
6.585859, 2.40404, -527.465, 1, 0.6862745, 0, 1,
6.626263, 2.40404, -528.1721, 1, 0.6862745, 0, 1,
6.666667, 2.40404, -528.9357, 1, 0.6862745, 0, 1,
6.707071, 2.40404, -529.7558, 1, 0.6862745, 0, 1,
6.747475, 2.40404, -530.6323, 1, 0.5843138, 0, 1,
6.787879, 2.40404, -531.5654, 1, 0.5843138, 0, 1,
6.828283, 2.40404, -532.555, 1, 0.5843138, 0, 1,
6.868687, 2.40404, -533.601, 1, 0.5843138, 0, 1,
6.909091, 2.40404, -534.7036, 1, 0.5843138, 0, 1,
6.949495, 2.40404, -535.8626, 1, 0.4823529, 0, 1,
6.989899, 2.40404, -537.0782, 1, 0.4823529, 0, 1,
7.030303, 2.40404, -538.3502, 1, 0.4823529, 0, 1,
7.070707, 2.40404, -539.6787, 1, 0.4823529, 0, 1,
7.111111, 2.40404, -541.0637, 1, 0.3764706, 0, 1,
7.151515, 2.40404, -542.5052, 1, 0.3764706, 0, 1,
7.191919, 2.40404, -544.0032, 1, 0.3764706, 0, 1,
7.232323, 2.40404, -545.5577, 1, 0.3764706, 0, 1,
7.272727, 2.40404, -547.1687, 1, 0.2745098, 0, 1,
7.313131, 2.40404, -548.8362, 1, 0.2745098, 0, 1,
7.353535, 2.40404, -550.5602, 1, 0.2745098, 0, 1,
7.393939, 2.40404, -552.3406, 1, 0.1686275, 0, 1,
7.434343, 2.40404, -554.1776, 1, 0.1686275, 0, 1,
7.474748, 2.40404, -556.071, 1, 0.1686275, 0, 1,
7.515152, 2.40404, -558.021, 1, 0.06666667, 0, 1,
7.555555, 2.40404, -560.0274, 1, 0.06666667, 0, 1,
7.59596, 2.40404, -562.0904, 1, 0.06666667, 0, 1,
7.636364, 2.40404, -564.2098, 0.9647059, 0, 0.03137255, 1,
7.676768, 2.40404, -566.3857, 0.9647059, 0, 0.03137255, 1,
7.717172, 2.40404, -568.6181, 0.8588235, 0, 0.1372549, 1,
7.757576, 2.40404, -570.907, 0.8588235, 0, 0.1372549, 1,
7.79798, 2.40404, -573.2524, 0.7568628, 0, 0.2392157, 1,
7.838384, 2.40404, -575.6543, 0.7568628, 0, 0.2392157, 1,
7.878788, 2.40404, -578.1127, 0.7568628, 0, 0.2392157, 1,
7.919192, 2.40404, -580.6276, 0.654902, 0, 0.3411765, 1,
7.959596, 2.40404, -583.199, 0.654902, 0, 0.3411765, 1,
8, 2.40404, -585.8268, 0.5490196, 0, 0.4470588, 1,
4, 2.454545, -594.0796, 0.4470588, 0, 0.5490196, 1,
4.040404, 2.454545, -591.2896, 0.4470588, 0, 0.5490196, 1,
4.080808, 2.454545, -588.5538, 0.5490196, 0, 0.4470588, 1,
4.121212, 2.454545, -585.8721, 0.5490196, 0, 0.4470588, 1,
4.161616, 2.454545, -583.2447, 0.654902, 0, 0.3411765, 1,
4.20202, 2.454545, -580.6714, 0.654902, 0, 0.3411765, 1,
4.242424, 2.454545, -578.1524, 0.7568628, 0, 0.2392157, 1,
4.282828, 2.454545, -575.6876, 0.7568628, 0, 0.2392157, 1,
4.323232, 2.454545, -573.2769, 0.7568628, 0, 0.2392157, 1,
4.363636, 2.454545, -570.9204, 0.8588235, 0, 0.1372549, 1,
4.40404, 2.454545, -568.6182, 0.8588235, 0, 0.1372549, 1,
4.444445, 2.454545, -566.3701, 0.9647059, 0, 0.03137255, 1,
4.484848, 2.454545, -564.1761, 0.9647059, 0, 0.03137255, 1,
4.525252, 2.454545, -562.0364, 1, 0.06666667, 0, 1,
4.565657, 2.454545, -559.9509, 1, 0.06666667, 0, 1,
4.606061, 2.454545, -557.9196, 1, 0.06666667, 0, 1,
4.646465, 2.454545, -555.9425, 1, 0.1686275, 0, 1,
4.686869, 2.454545, -554.0196, 1, 0.1686275, 0, 1,
4.727273, 2.454545, -552.1508, 1, 0.1686275, 0, 1,
4.767677, 2.454545, -550.3363, 1, 0.2745098, 0, 1,
4.808081, 2.454545, -548.5759, 1, 0.2745098, 0, 1,
4.848485, 2.454545, -546.8698, 1, 0.2745098, 0, 1,
4.888889, 2.454545, -545.2178, 1, 0.3764706, 0, 1,
4.929293, 2.454545, -543.62, 1, 0.3764706, 0, 1,
4.969697, 2.454545, -542.0764, 1, 0.3764706, 0, 1,
5.010101, 2.454545, -540.587, 1, 0.4823529, 0, 1,
5.050505, 2.454545, -539.1519, 1, 0.4823529, 0, 1,
5.090909, 2.454545, -537.7708, 1, 0.4823529, 0, 1,
5.131313, 2.454545, -536.444, 1, 0.4823529, 0, 1,
5.171717, 2.454545, -535.1714, 1, 0.5843138, 0, 1,
5.212121, 2.454545, -533.9529, 1, 0.5843138, 0, 1,
5.252525, 2.454545, -532.7887, 1, 0.5843138, 0, 1,
5.292929, 2.454545, -531.6786, 1, 0.5843138, 0, 1,
5.333333, 2.454545, -530.6228, 1, 0.5843138, 0, 1,
5.373737, 2.454545, -529.6212, 1, 0.6862745, 0, 1,
5.414141, 2.454545, -528.6736, 1, 0.6862745, 0, 1,
5.454545, 2.454545, -527.7804, 1, 0.6862745, 0, 1,
5.494949, 2.454545, -526.9413, 1, 0.6862745, 0, 1,
5.535354, 2.454545, -526.1564, 1, 0.6862745, 0, 1,
5.575758, 2.454545, -525.4257, 1, 0.6862745, 0, 1,
5.616162, 2.454545, -524.7492, 1, 0.7921569, 0, 1,
5.656566, 2.454545, -524.1269, 1, 0.7921569, 0, 1,
5.69697, 2.454545, -523.5588, 1, 0.7921569, 0, 1,
5.737374, 2.454545, -523.0448, 1, 0.7921569, 0, 1,
5.777778, 2.454545, -522.5851, 1, 0.7921569, 0, 1,
5.818182, 2.454545, -522.1795, 1, 0.7921569, 0, 1,
5.858586, 2.454545, -521.8282, 1, 0.7921569, 0, 1,
5.89899, 2.454545, -521.531, 1, 0.7921569, 0, 1,
5.939394, 2.454545, -521.288, 1, 0.7921569, 0, 1,
5.979798, 2.454545, -521.0992, 1, 0.7921569, 0, 1,
6.020202, 2.454545, -520.9647, 1, 0.7921569, 0, 1,
6.060606, 2.454545, -520.8843, 1, 0.7921569, 0, 1,
6.10101, 2.454545, -520.8581, 1, 0.7921569, 0, 1,
6.141414, 2.454545, -520.886, 1, 0.7921569, 0, 1,
6.181818, 2.454545, -520.9683, 1, 0.7921569, 0, 1,
6.222222, 2.454545, -521.1046, 1, 0.7921569, 0, 1,
6.262626, 2.454545, -521.2952, 1, 0.7921569, 0, 1,
6.30303, 2.454545, -521.5399, 1, 0.7921569, 0, 1,
6.343434, 2.454545, -521.8389, 1, 0.7921569, 0, 1,
6.383838, 2.454545, -522.192, 1, 0.7921569, 0, 1,
6.424242, 2.454545, -522.5994, 1, 0.7921569, 0, 1,
6.464646, 2.454545, -523.0609, 1, 0.7921569, 0, 1,
6.505051, 2.454545, -523.5766, 1, 0.7921569, 0, 1,
6.545455, 2.454545, -524.1465, 1, 0.7921569, 0, 1,
6.585859, 2.454545, -524.7706, 1, 0.7921569, 0, 1,
6.626263, 2.454545, -525.4489, 1, 0.6862745, 0, 1,
6.666667, 2.454545, -526.1815, 1, 0.6862745, 0, 1,
6.707071, 2.454545, -526.9681, 1, 0.6862745, 0, 1,
6.747475, 2.454545, -527.809, 1, 0.6862745, 0, 1,
6.787879, 2.454545, -528.704, 1, 0.6862745, 0, 1,
6.828283, 2.454545, -529.6533, 1, 0.6862745, 0, 1,
6.868687, 2.454545, -530.6567, 1, 0.5843138, 0, 1,
6.909091, 2.454545, -531.7144, 1, 0.5843138, 0, 1,
6.949495, 2.454545, -532.8262, 1, 0.5843138, 0, 1,
6.989899, 2.454545, -533.9922, 1, 0.5843138, 0, 1,
7.030303, 2.454545, -535.2125, 1, 0.5843138, 0, 1,
7.070707, 2.454545, -536.4869, 1, 0.4823529, 0, 1,
7.111111, 2.454545, -537.8155, 1, 0.4823529, 0, 1,
7.151515, 2.454545, -539.1983, 1, 0.4823529, 0, 1,
7.191919, 2.454545, -540.6353, 1, 0.4823529, 0, 1,
7.232323, 2.454545, -542.1265, 1, 0.3764706, 0, 1,
7.272727, 2.454545, -543.6718, 1, 0.3764706, 0, 1,
7.313131, 2.454545, -545.2714, 1, 0.3764706, 0, 1,
7.353535, 2.454545, -546.9252, 1, 0.2745098, 0, 1,
7.393939, 2.454545, -548.6331, 1, 0.2745098, 0, 1,
7.434343, 2.454545, -550.3953, 1, 0.2745098, 0, 1,
7.474748, 2.454545, -552.2116, 1, 0.1686275, 0, 1,
7.515152, 2.454545, -554.0821, 1, 0.1686275, 0, 1,
7.555555, 2.454545, -556.0068, 1, 0.1686275, 0, 1,
7.59596, 2.454545, -557.9858, 1, 0.06666667, 0, 1,
7.636364, 2.454545, -560.0189, 1, 0.06666667, 0, 1,
7.676768, 2.454545, -562.1061, 1, 0.06666667, 0, 1,
7.717172, 2.454545, -564.2476, 0.9647059, 0, 0.03137255, 1,
7.757576, 2.454545, -566.4433, 0.9647059, 0, 0.03137255, 1,
7.79798, 2.454545, -568.6932, 0.8588235, 0, 0.1372549, 1,
7.838384, 2.454545, -570.9973, 0.8588235, 0, 0.1372549, 1,
7.878788, 2.454545, -573.3555, 0.7568628, 0, 0.2392157, 1,
7.919192, 2.454545, -575.7679, 0.7568628, 0, 0.2392157, 1,
7.959596, 2.454545, -578.2346, 0.7568628, 0, 0.2392157, 1,
8, 2.454545, -580.7554, 0.654902, 0, 0.3411765, 1,
4, 2.50505, -588.9443, 0.5490196, 0, 0.4470588, 1,
4.040404, 2.50505, -586.2656, 0.5490196, 0, 0.4470588, 1,
4.080808, 2.50505, -583.639, 0.654902, 0, 0.3411765, 1,
4.121212, 2.50505, -581.0644, 0.654902, 0, 0.3411765, 1,
4.161616, 2.50505, -578.5419, 0.7568628, 0, 0.2392157, 1,
4.20202, 2.50505, -576.0714, 0.7568628, 0, 0.2392157, 1,
4.242424, 2.50505, -573.6528, 0.7568628, 0, 0.2392157, 1,
4.282828, 2.50505, -571.2864, 0.8588235, 0, 0.1372549, 1,
4.323232, 2.50505, -568.9719, 0.8588235, 0, 0.1372549, 1,
4.363636, 2.50505, -566.7095, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.50505, -564.4991, 0.9647059, 0, 0.03137255, 1,
4.444445, 2.50505, -562.3408, 1, 0.06666667, 0, 1,
4.484848, 2.50505, -560.2345, 1, 0.06666667, 0, 1,
4.525252, 2.50505, -558.1802, 1, 0.06666667, 0, 1,
4.565657, 2.50505, -556.1779, 1, 0.1686275, 0, 1,
4.606061, 2.50505, -554.2277, 1, 0.1686275, 0, 1,
4.646465, 2.50505, -552.3295, 1, 0.1686275, 0, 1,
4.686869, 2.50505, -550.4833, 1, 0.2745098, 0, 1,
4.727273, 2.50505, -548.6891, 1, 0.2745098, 0, 1,
4.767677, 2.50505, -546.947, 1, 0.2745098, 0, 1,
4.808081, 2.50505, -545.257, 1, 0.3764706, 0, 1,
4.848485, 2.50505, -543.6189, 1, 0.3764706, 0, 1,
4.888889, 2.50505, -542.0328, 1, 0.3764706, 0, 1,
4.929293, 2.50505, -540.4988, 1, 0.4823529, 0, 1,
4.969697, 2.50505, -539.0168, 1, 0.4823529, 0, 1,
5.010101, 2.50505, -537.5869, 1, 0.4823529, 0, 1,
5.050505, 2.50505, -536.209, 1, 0.4823529, 0, 1,
5.090909, 2.50505, -534.8831, 1, 0.5843138, 0, 1,
5.131313, 2.50505, -533.6093, 1, 0.5843138, 0, 1,
5.171717, 2.50505, -532.3875, 1, 0.5843138, 0, 1,
5.212121, 2.50505, -531.2177, 1, 0.5843138, 0, 1,
5.252525, 2.50505, -530.0999, 1, 0.6862745, 0, 1,
5.292929, 2.50505, -529.0341, 1, 0.6862745, 0, 1,
5.333333, 2.50505, -528.0204, 1, 0.6862745, 0, 1,
5.373737, 2.50505, -527.0588, 1, 0.6862745, 0, 1,
5.414141, 2.50505, -526.1491, 1, 0.6862745, 0, 1,
5.454545, 2.50505, -525.2914, 1, 0.6862745, 0, 1,
5.494949, 2.50505, -524.4859, 1, 0.7921569, 0, 1,
5.535354, 2.50505, -523.7323, 1, 0.7921569, 0, 1,
5.575758, 2.50505, -523.0308, 1, 0.7921569, 0, 1,
5.616162, 2.50505, -522.3813, 1, 0.7921569, 0, 1,
5.656566, 2.50505, -521.7838, 1, 0.7921569, 0, 1,
5.69697, 2.50505, -521.2383, 1, 0.7921569, 0, 1,
5.737374, 2.50505, -520.7449, 1, 0.7921569, 0, 1,
5.777778, 2.50505, -520.3035, 1, 0.7921569, 0, 1,
5.818182, 2.50505, -519.9142, 1, 0.7921569, 0, 1,
5.858586, 2.50505, -519.5768, 1, 0.7921569, 0, 1,
5.89899, 2.50505, -519.2916, 1, 0.8941177, 0, 1,
5.939394, 2.50505, -519.0583, 1, 0.8941177, 0, 1,
5.979798, 2.50505, -518.877, 1, 0.8941177, 0, 1,
6.020202, 2.50505, -518.7478, 1, 0.8941177, 0, 1,
6.060606, 2.50505, -518.6707, 1, 0.8941177, 0, 1,
6.10101, 2.50505, -518.6454, 1, 0.8941177, 0, 1,
6.141414, 2.50505, -518.6724, 1, 0.8941177, 0, 1,
6.181818, 2.50505, -518.7512, 1, 0.8941177, 0, 1,
6.222222, 2.50505, -518.8821, 1, 0.8941177, 0, 1,
6.262626, 2.50505, -519.0651, 1, 0.8941177, 0, 1,
6.30303, 2.50505, -519.3001, 1, 0.8941177, 0, 1,
6.343434, 2.50505, -519.5872, 1, 0.7921569, 0, 1,
6.383838, 2.50505, -519.9262, 1, 0.7921569, 0, 1,
6.424242, 2.50505, -520.3173, 1, 0.7921569, 0, 1,
6.464646, 2.50505, -520.7604, 1, 0.7921569, 0, 1,
6.505051, 2.50505, -521.2555, 1, 0.7921569, 0, 1,
6.545455, 2.50505, -521.8027, 1, 0.7921569, 0, 1,
6.585859, 2.50505, -522.4019, 1, 0.7921569, 0, 1,
6.626263, 2.50505, -523.0531, 1, 0.7921569, 0, 1,
6.666667, 2.50505, -523.7563, 1, 0.7921569, 0, 1,
6.707071, 2.50505, -524.5116, 1, 0.7921569, 0, 1,
6.747475, 2.50505, -525.3189, 1, 0.6862745, 0, 1,
6.787879, 2.50505, -526.1783, 1, 0.6862745, 0, 1,
6.828283, 2.50505, -527.0896, 1, 0.6862745, 0, 1,
6.868687, 2.50505, -528.053, 1, 0.6862745, 0, 1,
6.909091, 2.50505, -529.0684, 1, 0.6862745, 0, 1,
6.949495, 2.50505, -530.1359, 1, 0.6862745, 0, 1,
6.989899, 2.50505, -531.2554, 1, 0.5843138, 0, 1,
7.030303, 2.50505, -532.4269, 1, 0.5843138, 0, 1,
7.070707, 2.50505, -533.6505, 1, 0.5843138, 0, 1,
7.111111, 2.50505, -534.926, 1, 0.5843138, 0, 1,
7.151515, 2.50505, -536.2536, 1, 0.4823529, 0, 1,
7.191919, 2.50505, -537.6332, 1, 0.4823529, 0, 1,
7.232323, 2.50505, -539.0649, 1, 0.4823529, 0, 1,
7.272727, 2.50505, -540.5486, 1, 0.4823529, 0, 1,
7.313131, 2.50505, -542.0843, 1, 0.3764706, 0, 1,
7.353535, 2.50505, -543.6721, 1, 0.3764706, 0, 1,
7.393939, 2.50505, -545.3118, 1, 0.3764706, 0, 1,
7.434343, 2.50505, -547.0037, 1, 0.2745098, 0, 1,
7.474748, 2.50505, -548.7475, 1, 0.2745098, 0, 1,
7.515152, 2.50505, -550.5433, 1, 0.2745098, 0, 1,
7.555555, 2.50505, -552.3912, 1, 0.1686275, 0, 1,
7.59596, 2.50505, -554.2911, 1, 0.1686275, 0, 1,
7.636364, 2.50505, -556.2431, 1, 0.1686275, 0, 1,
7.676768, 2.50505, -558.2471, 1, 0.06666667, 0, 1,
7.717172, 2.50505, -560.3031, 1, 0.06666667, 0, 1,
7.757576, 2.50505, -562.4111, 1, 0.06666667, 0, 1,
7.79798, 2.50505, -564.5712, 0.9647059, 0, 0.03137255, 1,
7.838384, 2.50505, -566.7833, 0.9647059, 0, 0.03137255, 1,
7.878788, 2.50505, -569.0474, 0.8588235, 0, 0.1372549, 1,
7.919192, 2.50505, -571.3636, 0.8588235, 0, 0.1372549, 1,
7.959596, 2.50505, -573.7318, 0.7568628, 0, 0.2392157, 1,
8, 2.50505, -576.152, 0.7568628, 0, 0.2392157, 1,
4, 2.555556, -584.2682, 0.5490196, 0, 0.4470588, 1,
4.040404, 2.555556, -581.6944, 0.654902, 0, 0.3411765, 1,
4.080808, 2.555556, -579.1705, 0.654902, 0, 0.3411765, 1,
4.121212, 2.555556, -576.6967, 0.7568628, 0, 0.2392157, 1,
4.161616, 2.555556, -574.2729, 0.7568628, 0, 0.2392157, 1,
4.20202, 2.555556, -571.899, 0.8588235, 0, 0.1372549, 1,
4.242424, 2.555556, -569.5752, 0.8588235, 0, 0.1372549, 1,
4.282828, 2.555556, -567.3013, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.555556, -565.0775, 0.9647059, 0, 0.03137255, 1,
4.363636, 2.555556, -562.9036, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.555556, -560.7797, 1, 0.06666667, 0, 1,
4.444445, 2.555556, -558.7059, 1, 0.06666667, 0, 1,
4.484848, 2.555556, -556.6819, 1, 0.1686275, 0, 1,
4.525252, 2.555556, -554.7081, 1, 0.1686275, 0, 1,
4.565657, 2.555556, -552.7841, 1, 0.1686275, 0, 1,
4.606061, 2.555556, -550.9102, 1, 0.2745098, 0, 1,
4.646465, 2.555556, -549.0863, 1, 0.2745098, 0, 1,
4.686869, 2.555556, -547.3124, 1, 0.2745098, 0, 1,
4.727273, 2.555556, -545.5884, 1, 0.3764706, 0, 1,
4.767677, 2.555556, -543.9145, 1, 0.3764706, 0, 1,
4.808081, 2.555556, -542.2906, 1, 0.3764706, 0, 1,
4.848485, 2.555556, -540.7166, 1, 0.4823529, 0, 1,
4.888889, 2.555556, -539.1926, 1, 0.4823529, 0, 1,
4.929293, 2.555556, -537.7187, 1, 0.4823529, 0, 1,
4.969697, 2.555556, -536.2947, 1, 0.4823529, 0, 1,
5.010101, 2.555556, -534.9207, 1, 0.5843138, 0, 1,
5.050505, 2.555556, -533.5967, 1, 0.5843138, 0, 1,
5.090909, 2.555556, -532.3228, 1, 0.5843138, 0, 1,
5.131313, 2.555556, -531.0988, 1, 0.5843138, 0, 1,
5.171717, 2.555556, -529.9247, 1, 0.6862745, 0, 1,
5.212121, 2.555556, -528.8007, 1, 0.6862745, 0, 1,
5.252525, 2.555556, -527.7267, 1, 0.6862745, 0, 1,
5.292929, 2.555556, -526.7026, 1, 0.6862745, 0, 1,
5.333333, 2.555556, -525.7286, 1, 0.6862745, 0, 1,
5.373737, 2.555556, -524.8046, 1, 0.7921569, 0, 1,
5.414141, 2.555556, -523.9305, 1, 0.7921569, 0, 1,
5.454545, 2.555556, -523.1064, 1, 0.7921569, 0, 1,
5.494949, 2.555556, -522.3324, 1, 0.7921569, 0, 1,
5.535354, 2.555556, -521.6083, 1, 0.7921569, 0, 1,
5.575758, 2.555556, -520.9342, 1, 0.7921569, 0, 1,
5.616162, 2.555556, -520.3101, 1, 0.7921569, 0, 1,
5.656566, 2.555556, -519.736, 1, 0.7921569, 0, 1,
5.69697, 2.555556, -519.212, 1, 0.8941177, 0, 1,
5.737374, 2.555556, -518.7379, 1, 0.8941177, 0, 1,
5.777778, 2.555556, -518.3137, 1, 0.8941177, 0, 1,
5.818182, 2.555556, -517.9396, 1, 0.8941177, 0, 1,
5.858586, 2.555556, -517.6155, 1, 0.8941177, 0, 1,
5.89899, 2.555556, -517.3413, 1, 0.8941177, 0, 1,
5.939394, 2.555556, -517.1172, 1, 0.8941177, 0, 1,
5.979798, 2.555556, -516.9431, 1, 0.8941177, 0, 1,
6.020202, 2.555556, -516.8189, 1, 0.8941177, 0, 1,
6.060606, 2.555556, -516.7447, 1, 0.8941177, 0, 1,
6.10101, 2.555556, -516.7205, 1, 0.8941177, 0, 1,
6.141414, 2.555556, -516.7463, 1, 0.8941177, 0, 1,
6.181818, 2.555556, -516.8222, 1, 0.8941177, 0, 1,
6.222222, 2.555556, -516.948, 1, 0.8941177, 0, 1,
6.262626, 2.555556, -517.1238, 1, 0.8941177, 0, 1,
6.30303, 2.555556, -517.3495, 1, 0.8941177, 0, 1,
6.343434, 2.555556, -517.6254, 1, 0.8941177, 0, 1,
6.383838, 2.555556, -517.9511, 1, 0.8941177, 0, 1,
6.424242, 2.555556, -518.3269, 1, 0.8941177, 0, 1,
6.464646, 2.555556, -518.7527, 1, 0.8941177, 0, 1,
6.505051, 2.555556, -519.2285, 1, 0.8941177, 0, 1,
6.545455, 2.555556, -519.7542, 1, 0.7921569, 0, 1,
6.585859, 2.555556, -520.33, 1, 0.7921569, 0, 1,
6.626263, 2.555556, -520.9557, 1, 0.7921569, 0, 1,
6.666667, 2.555556, -521.6314, 1, 0.7921569, 0, 1,
6.707071, 2.555556, -522.3571, 1, 0.7921569, 0, 1,
6.747475, 2.555556, -523.1328, 1, 0.7921569, 0, 1,
6.787879, 2.555556, -523.9586, 1, 0.7921569, 0, 1,
6.828283, 2.555556, -524.8342, 1, 0.7921569, 0, 1,
6.868687, 2.555556, -525.7599, 1, 0.6862745, 0, 1,
6.909091, 2.555556, -526.7356, 1, 0.6862745, 0, 1,
6.949495, 2.555556, -527.7613, 1, 0.6862745, 0, 1,
6.989899, 2.555556, -528.837, 1, 0.6862745, 0, 1,
7.030303, 2.555556, -529.9626, 1, 0.6862745, 0, 1,
7.070707, 2.555556, -531.1383, 1, 0.5843138, 0, 1,
7.111111, 2.555556, -532.364, 1, 0.5843138, 0, 1,
7.151515, 2.555556, -533.6396, 1, 0.5843138, 0, 1,
7.191919, 2.555556, -534.9652, 1, 0.5843138, 0, 1,
7.232323, 2.555556, -536.3409, 1, 0.4823529, 0, 1,
7.272727, 2.555556, -537.7665, 1, 0.4823529, 0, 1,
7.313131, 2.555556, -539.2421, 1, 0.4823529, 0, 1,
7.353535, 2.555556, -540.7677, 1, 0.4823529, 0, 1,
7.393939, 2.555556, -542.3433, 1, 0.3764706, 0, 1,
7.434343, 2.555556, -543.9689, 1, 0.3764706, 0, 1,
7.474748, 2.555556, -545.6445, 1, 0.3764706, 0, 1,
7.515152, 2.555556, -547.3701, 1, 0.2745098, 0, 1,
7.555555, 2.555556, -549.1456, 1, 0.2745098, 0, 1,
7.59596, 2.555556, -550.9713, 1, 0.2745098, 0, 1,
7.636364, 2.555556, -552.8468, 1, 0.1686275, 0, 1,
7.676768, 2.555556, -554.7723, 1, 0.1686275, 0, 1,
7.717172, 2.555556, -556.7479, 1, 0.1686275, 0, 1,
7.757576, 2.555556, -558.7734, 1, 0.06666667, 0, 1,
7.79798, 2.555556, -560.8489, 1, 0.06666667, 0, 1,
7.838384, 2.555556, -562.9745, 0.9647059, 0, 0.03137255, 1,
7.878788, 2.555556, -565.15, 0.9647059, 0, 0.03137255, 1,
7.919192, 2.555556, -567.3755, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.555556, -569.651, 0.8588235, 0, 0.1372549, 1,
8, 2.555556, -571.9765, 0.8588235, 0, 0.1372549, 1,
4, 2.606061, -580.0131, 0.654902, 0, 0.3411765, 1,
4.040404, 2.606061, -577.538, 0.7568628, 0, 0.2392157, 1,
4.080808, 2.606061, -575.1111, 0.7568628, 0, 0.2392157, 1,
4.121212, 2.606061, -572.7322, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.606061, -570.4014, 0.8588235, 0, 0.1372549, 1,
4.20202, 2.606061, -568.1187, 0.8588235, 0, 0.1372549, 1,
4.242424, 2.606061, -565.884, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.606061, -563.6974, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.606061, -561.559, 1, 0.06666667, 0, 1,
4.363636, 2.606061, -559.4685, 1, 0.06666667, 0, 1,
4.40404, 2.606061, -557.4261, 1, 0.06666667, 0, 1,
4.444445, 2.606061, -555.4319, 1, 0.1686275, 0, 1,
4.484848, 2.606061, -553.4857, 1, 0.1686275, 0, 1,
4.525252, 2.606061, -551.5875, 1, 0.2745098, 0, 1,
4.565657, 2.606061, -549.7375, 1, 0.2745098, 0, 1,
4.606061, 2.606061, -547.9355, 1, 0.2745098, 0, 1,
4.646465, 2.606061, -546.1816, 1, 0.3764706, 0, 1,
4.686869, 2.606061, -544.4758, 1, 0.3764706, 0, 1,
4.727273, 2.606061, -542.818, 1, 0.3764706, 0, 1,
4.767677, 2.606061, -541.2083, 1, 0.3764706, 0, 1,
4.808081, 2.606061, -539.6467, 1, 0.4823529, 0, 1,
4.848485, 2.606061, -538.1332, 1, 0.4823529, 0, 1,
4.888889, 2.606061, -536.6677, 1, 0.4823529, 0, 1,
4.929293, 2.606061, -535.2503, 1, 0.5843138, 0, 1,
4.969697, 2.606061, -533.881, 1, 0.5843138, 0, 1,
5.010101, 2.606061, -532.5598, 1, 0.5843138, 0, 1,
5.050505, 2.606061, -531.2866, 1, 0.5843138, 0, 1,
5.090909, 2.606061, -530.0615, 1, 0.6862745, 0, 1,
5.131313, 2.606061, -528.8845, 1, 0.6862745, 0, 1,
5.171717, 2.606061, -527.7555, 1, 0.6862745, 0, 1,
5.212121, 2.606061, -526.6746, 1, 0.6862745, 0, 1,
5.252525, 2.606061, -525.6418, 1, 0.6862745, 0, 1,
5.292929, 2.606061, -524.6571, 1, 0.7921569, 0, 1,
5.333333, 2.606061, -523.7205, 1, 0.7921569, 0, 1,
5.373737, 2.606061, -522.8319, 1, 0.7921569, 0, 1,
5.414141, 2.606061, -521.9914, 1, 0.7921569, 0, 1,
5.454545, 2.606061, -521.199, 1, 0.7921569, 0, 1,
5.494949, 2.606061, -520.4546, 1, 0.7921569, 0, 1,
5.535354, 2.606061, -519.7583, 1, 0.7921569, 0, 1,
5.575758, 2.606061, -519.1101, 1, 0.8941177, 0, 1,
5.616162, 2.606061, -518.51, 1, 0.8941177, 0, 1,
5.656566, 2.606061, -517.9579, 1, 0.8941177, 0, 1,
5.69697, 2.606061, -517.4539, 1, 0.8941177, 0, 1,
5.737374, 2.606061, -516.998, 1, 0.8941177, 0, 1,
5.777778, 2.606061, -516.5902, 1, 0.8941177, 0, 1,
5.818182, 2.606061, -516.2304, 1, 0.8941177, 0, 1,
5.858586, 2.606061, -515.9188, 1, 0.8941177, 0, 1,
5.89899, 2.606061, -515.6552, 1, 0.8941177, 0, 1,
5.939394, 2.606061, -515.4396, 1, 0.8941177, 0, 1,
5.979798, 2.606061, -515.2721, 1, 0.8941177, 0, 1,
6.020202, 2.606061, -515.1527, 1, 0.8941177, 0, 1,
6.060606, 2.606061, -515.0814, 1, 0.8941177, 0, 1,
6.10101, 2.606061, -515.0582, 1, 0.8941177, 0, 1,
6.141414, 2.606061, -515.083, 1, 0.8941177, 0, 1,
6.181818, 2.606061, -515.1559, 1, 0.8941177, 0, 1,
6.222222, 2.606061, -515.2769, 1, 0.8941177, 0, 1,
6.262626, 2.606061, -515.4459, 1, 0.8941177, 0, 1,
6.30303, 2.606061, -515.663, 1, 0.8941177, 0, 1,
6.343434, 2.606061, -515.9282, 1, 0.8941177, 0, 1,
6.383838, 2.606061, -516.2415, 1, 0.8941177, 0, 1,
6.424242, 2.606061, -516.6028, 1, 0.8941177, 0, 1,
6.464646, 2.606061, -517.0123, 1, 0.8941177, 0, 1,
6.505051, 2.606061, -517.4698, 1, 0.8941177, 0, 1,
6.545455, 2.606061, -517.9753, 1, 0.8941177, 0, 1,
6.585859, 2.606061, -518.529, 1, 0.8941177, 0, 1,
6.626263, 2.606061, -519.1307, 1, 0.8941177, 0, 1,
6.666667, 2.606061, -519.7805, 1, 0.7921569, 0, 1,
6.707071, 2.606061, -520.4784, 1, 0.7921569, 0, 1,
6.747475, 2.606061, -521.2243, 1, 0.7921569, 0, 1,
6.787879, 2.606061, -522.0183, 1, 0.7921569, 0, 1,
6.828283, 2.606061, -522.8604, 1, 0.7921569, 0, 1,
6.868687, 2.606061, -523.7506, 1, 0.7921569, 0, 1,
6.909091, 2.606061, -524.6888, 1, 0.7921569, 0, 1,
6.949495, 2.606061, -525.6751, 1, 0.6862745, 0, 1,
6.989899, 2.606061, -526.7095, 1, 0.6862745, 0, 1,
7.030303, 2.606061, -527.792, 1, 0.6862745, 0, 1,
7.070707, 2.606061, -528.9225, 1, 0.6862745, 0, 1,
7.111111, 2.606061, -530.1011, 1, 0.6862745, 0, 1,
7.151515, 2.606061, -531.3278, 1, 0.5843138, 0, 1,
7.191919, 2.606061, -532.6025, 1, 0.5843138, 0, 1,
7.232323, 2.606061, -533.9254, 1, 0.5843138, 0, 1,
7.272727, 2.606061, -535.2963, 1, 0.5843138, 0, 1,
7.313131, 2.606061, -536.7153, 1, 0.4823529, 0, 1,
7.353535, 2.606061, -538.1823, 1, 0.4823529, 0, 1,
7.393939, 2.606061, -539.6974, 1, 0.4823529, 0, 1,
7.434343, 2.606061, -541.2606, 1, 0.3764706, 0, 1,
7.474748, 2.606061, -542.8719, 1, 0.3764706, 0, 1,
7.515152, 2.606061, -544.5312, 1, 0.3764706, 0, 1,
7.555555, 2.606061, -546.2386, 1, 0.3764706, 0, 1,
7.59596, 2.606061, -547.9941, 1, 0.2745098, 0, 1,
7.636364, 2.606061, -549.7977, 1, 0.2745098, 0, 1,
7.676768, 2.606061, -551.6494, 1, 0.2745098, 0, 1,
7.717172, 2.606061, -553.5491, 1, 0.1686275, 0, 1,
7.757576, 2.606061, -555.4969, 1, 0.1686275, 0, 1,
7.79798, 2.606061, -557.4927, 1, 0.06666667, 0, 1,
7.838384, 2.606061, -559.5367, 1, 0.06666667, 0, 1,
7.878788, 2.606061, -561.6287, 1, 0.06666667, 0, 1,
7.919192, 2.606061, -563.7688, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.606061, -565.957, 0.9647059, 0, 0.03137255, 1,
8, 2.606061, -568.1932, 0.8588235, 0, 0.1372549, 1,
4, 2.656566, -576.1442, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.656566, -573.7625, 0.7568628, 0, 0.2392157, 1,
4.080808, 2.656566, -571.4269, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.656566, -569.1376, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.656566, -566.8946, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.656566, -564.6979, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.656566, -562.5474, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.656566, -560.4431, 1, 0.06666667, 0, 1,
4.323232, 2.656566, -558.3852, 1, 0.06666667, 0, 1,
4.363636, 2.656566, -556.3735, 1, 0.1686275, 0, 1,
4.40404, 2.656566, -554.408, 1, 0.1686275, 0, 1,
4.444445, 2.656566, -552.4889, 1, 0.1686275, 0, 1,
4.484848, 2.656566, -550.616, 1, 0.2745098, 0, 1,
4.525252, 2.656566, -548.7893, 1, 0.2745098, 0, 1,
4.565657, 2.656566, -547.0089, 1, 0.2745098, 0, 1,
4.606061, 2.656566, -545.2748, 1, 0.3764706, 0, 1,
4.646465, 2.656566, -543.587, 1, 0.3764706, 0, 1,
4.686869, 2.656566, -541.9454, 1, 0.3764706, 0, 1,
4.727273, 2.656566, -540.35, 1, 0.4823529, 0, 1,
4.767677, 2.656566, -538.801, 1, 0.4823529, 0, 1,
4.808081, 2.656566, -537.2982, 1, 0.4823529, 0, 1,
4.848485, 2.656566, -535.8416, 1, 0.4823529, 0, 1,
4.888889, 2.656566, -534.4313, 1, 0.5843138, 0, 1,
4.929293, 2.656566, -533.0673, 1, 0.5843138, 0, 1,
4.969697, 2.656566, -531.7496, 1, 0.5843138, 0, 1,
5.010101, 2.656566, -530.4781, 1, 0.5843138, 0, 1,
5.050505, 2.656566, -529.2529, 1, 0.6862745, 0, 1,
5.090909, 2.656566, -528.0739, 1, 0.6862745, 0, 1,
5.131313, 2.656566, -526.9412, 1, 0.6862745, 0, 1,
5.171717, 2.656566, -525.8548, 1, 0.6862745, 0, 1,
5.212121, 2.656566, -524.8146, 1, 0.7921569, 0, 1,
5.252525, 2.656566, -523.8207, 1, 0.7921569, 0, 1,
5.292929, 2.656566, -522.8731, 1, 0.7921569, 0, 1,
5.333333, 2.656566, -521.9717, 1, 0.7921569, 0, 1,
5.373737, 2.656566, -521.1166, 1, 0.7921569, 0, 1,
5.414141, 2.656566, -520.3078, 1, 0.7921569, 0, 1,
5.454545, 2.656566, -519.5452, 1, 0.7921569, 0, 1,
5.494949, 2.656566, -518.8289, 1, 0.8941177, 0, 1,
5.535354, 2.656566, -518.1588, 1, 0.8941177, 0, 1,
5.575758, 2.656566, -517.535, 1, 0.8941177, 0, 1,
5.616162, 2.656566, -516.9575, 1, 0.8941177, 0, 1,
5.656566, 2.656566, -516.4262, 1, 0.8941177, 0, 1,
5.69697, 2.656566, -515.9412, 1, 0.8941177, 0, 1,
5.737374, 2.656566, -515.5024, 1, 0.8941177, 0, 1,
5.777778, 2.656566, -515.11, 1, 0.8941177, 0, 1,
5.818182, 2.656566, -514.7638, 1, 0.8941177, 0, 1,
5.858586, 2.656566, -514.4638, 1, 0.8941177, 0, 1,
5.89899, 2.656566, -514.2101, 1, 0.8941177, 0, 1,
5.939394, 2.656566, -514.0027, 1, 1, 0, 1,
5.979798, 2.656566, -513.8416, 1, 1, 0, 1,
6.020202, 2.656566, -513.7267, 1, 1, 0, 1,
6.060606, 2.656566, -513.658, 1, 1, 0, 1,
6.10101, 2.656566, -513.6357, 1, 1, 0, 1,
6.141414, 2.656566, -513.6595, 1, 1, 0, 1,
6.181818, 2.656566, -513.7297, 1, 1, 0, 1,
6.222222, 2.656566, -513.8461, 1, 1, 0, 1,
6.262626, 2.656566, -514.0088, 1, 1, 0, 1,
6.30303, 2.656566, -514.2178, 1, 0.8941177, 0, 1,
6.343434, 2.656566, -514.473, 1, 0.8941177, 0, 1,
6.383838, 2.656566, -514.7745, 1, 0.8941177, 0, 1,
6.424242, 2.656566, -515.1222, 1, 0.8941177, 0, 1,
6.464646, 2.656566, -515.5162, 1, 0.8941177, 0, 1,
6.505051, 2.656566, -515.9565, 1, 0.8941177, 0, 1,
6.545455, 2.656566, -516.443, 1, 0.8941177, 0, 1,
6.585859, 2.656566, -516.9758, 1, 0.8941177, 0, 1,
6.626263, 2.656566, -517.5549, 1, 0.8941177, 0, 1,
6.666667, 2.656566, -518.1802, 1, 0.8941177, 0, 1,
6.707071, 2.656566, -518.8517, 1, 0.8941177, 0, 1,
6.747475, 2.656566, -519.5696, 1, 0.7921569, 0, 1,
6.787879, 2.656566, -520.3337, 1, 0.7921569, 0, 1,
6.828283, 2.656566, -521.1441, 1, 0.7921569, 0, 1,
6.868687, 2.656566, -522.0007, 1, 0.7921569, 0, 1,
6.909091, 2.656566, -522.9036, 1, 0.7921569, 0, 1,
6.949495, 2.656566, -523.8528, 1, 0.7921569, 0, 1,
6.989899, 2.656566, -524.8482, 1, 0.7921569, 0, 1,
7.030303, 2.656566, -525.8899, 1, 0.6862745, 0, 1,
7.070707, 2.656566, -526.9778, 1, 0.6862745, 0, 1,
7.111111, 2.656566, -528.1121, 1, 0.6862745, 0, 1,
7.151515, 2.656566, -529.2925, 1, 0.6862745, 0, 1,
7.191919, 2.656566, -530.5193, 1, 0.5843138, 0, 1,
7.232323, 2.656566, -531.7923, 1, 0.5843138, 0, 1,
7.272727, 2.656566, -533.1116, 1, 0.5843138, 0, 1,
7.313131, 2.656566, -534.4771, 1, 0.5843138, 0, 1,
7.353535, 2.656566, -535.8889, 1, 0.4823529, 0, 1,
7.393939, 2.656566, -537.347, 1, 0.4823529, 0, 1,
7.434343, 2.656566, -538.8513, 1, 0.4823529, 0, 1,
7.474748, 2.656566, -540.4019, 1, 0.4823529, 0, 1,
7.515152, 2.656566, -541.9987, 1, 0.3764706, 0, 1,
7.555555, 2.656566, -543.6418, 1, 0.3764706, 0, 1,
7.59596, 2.656566, -545.3312, 1, 0.3764706, 0, 1,
7.636364, 2.656566, -547.0669, 1, 0.2745098, 0, 1,
7.676768, 2.656566, -548.8488, 1, 0.2745098, 0, 1,
7.717172, 2.656566, -550.6769, 1, 0.2745098, 0, 1,
7.757576, 2.656566, -552.5514, 1, 0.1686275, 0, 1,
7.79798, 2.656566, -554.4721, 1, 0.1686275, 0, 1,
7.838384, 2.656566, -556.4391, 1, 0.1686275, 0, 1,
7.878788, 2.656566, -558.4523, 1, 0.06666667, 0, 1,
7.919192, 2.656566, -560.5118, 1, 0.06666667, 0, 1,
7.959596, 2.656566, -562.6176, 0.9647059, 0, 0.03137255, 1,
8, 2.656566, -564.7695, 0.9647059, 0, 0.03137255, 1,
4, 2.707071, -572.6306, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.707071, -570.3368, 0.8588235, 0, 0.1372549, 1,
4.080808, 2.707071, -568.0875, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.707071, -565.8829, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.707071, -563.7228, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.707071, -561.6072, 1, 0.06666667, 0, 1,
4.242424, 2.707071, -559.5363, 1, 0.06666667, 0, 1,
4.282828, 2.707071, -557.5098, 1, 0.06666667, 0, 1,
4.323232, 2.707071, -555.5279, 1, 0.1686275, 0, 1,
4.363636, 2.707071, -553.5906, 1, 0.1686275, 0, 1,
4.40404, 2.707071, -551.6978, 1, 0.2745098, 0, 1,
4.444445, 2.707071, -549.8495, 1, 0.2745098, 0, 1,
4.484848, 2.707071, -548.0459, 1, 0.2745098, 0, 1,
4.525252, 2.707071, -546.2867, 1, 0.3764706, 0, 1,
4.565657, 2.707071, -544.5722, 1, 0.3764706, 0, 1,
4.606061, 2.707071, -542.9022, 1, 0.3764706, 0, 1,
4.646465, 2.707071, -541.2767, 1, 0.3764706, 0, 1,
4.686869, 2.707071, -539.6958, 1, 0.4823529, 0, 1,
4.727273, 2.707071, -538.1594, 1, 0.4823529, 0, 1,
4.767677, 2.707071, -536.6677, 1, 0.4823529, 0, 1,
4.808081, 2.707071, -535.2204, 1, 0.5843138, 0, 1,
4.848485, 2.707071, -533.8177, 1, 0.5843138, 0, 1,
4.888889, 2.707071, -532.4595, 1, 0.5843138, 0, 1,
4.929293, 2.707071, -531.146, 1, 0.5843138, 0, 1,
4.969697, 2.707071, -529.877, 1, 0.6862745, 0, 1,
5.010101, 2.707071, -528.6525, 1, 0.6862745, 0, 1,
5.050505, 2.707071, -527.4725, 1, 0.6862745, 0, 1,
5.090909, 2.707071, -526.3372, 1, 0.6862745, 0, 1,
5.131313, 2.707071, -525.2463, 1, 0.6862745, 0, 1,
5.171717, 2.707071, -524.2001, 1, 0.7921569, 0, 1,
5.212121, 2.707071, -523.1984, 1, 0.7921569, 0, 1,
5.252525, 2.707071, -522.2411, 1, 0.7921569, 0, 1,
5.292929, 2.707071, -521.3286, 1, 0.7921569, 0, 1,
5.333333, 2.707071, -520.4605, 1, 0.7921569, 0, 1,
5.373737, 2.707071, -519.637, 1, 0.7921569, 0, 1,
5.414141, 2.707071, -518.858, 1, 0.8941177, 0, 1,
5.454545, 2.707071, -518.1237, 1, 0.8941177, 0, 1,
5.494949, 2.707071, -517.4338, 1, 0.8941177, 0, 1,
5.535354, 2.707071, -516.7885, 1, 0.8941177, 0, 1,
5.575758, 2.707071, -516.1878, 1, 0.8941177, 0, 1,
5.616162, 2.707071, -515.6316, 1, 0.8941177, 0, 1,
5.656566, 2.707071, -515.12, 1, 0.8941177, 0, 1,
5.69697, 2.707071, -514.6529, 1, 0.8941177, 0, 1,
5.737374, 2.707071, -514.2304, 1, 0.8941177, 0, 1,
5.777778, 2.707071, -513.8524, 1, 1, 0, 1,
5.818182, 2.707071, -513.519, 1, 1, 0, 1,
5.858586, 2.707071, -513.2302, 1, 1, 0, 1,
5.89899, 2.707071, -512.9858, 1, 1, 0, 1,
5.939394, 2.707071, -512.7861, 1, 1, 0, 1,
5.979798, 2.707071, -512.6309, 1, 1, 0, 1,
6.020202, 2.707071, -512.5202, 1, 1, 0, 1,
6.060606, 2.707071, -512.4541, 1, 1, 0, 1,
6.10101, 2.707071, -512.4326, 1, 1, 0, 1,
6.141414, 2.707071, -512.4556, 1, 1, 0, 1,
6.181818, 2.707071, -512.5231, 1, 1, 0, 1,
6.222222, 2.707071, -512.6353, 1, 1, 0, 1,
6.262626, 2.707071, -512.7919, 1, 1, 0, 1,
6.30303, 2.707071, -512.9932, 1, 1, 0, 1,
6.343434, 2.707071, -513.239, 1, 1, 0, 1,
6.383838, 2.707071, -513.5293, 1, 1, 0, 1,
6.424242, 2.707071, -513.8642, 1, 1, 0, 1,
6.464646, 2.707071, -514.2436, 1, 0.8941177, 0, 1,
6.505051, 2.707071, -514.6676, 1, 0.8941177, 0, 1,
6.545455, 2.707071, -515.1362, 1, 0.8941177, 0, 1,
6.585859, 2.707071, -515.6492, 1, 0.8941177, 0, 1,
6.626263, 2.707071, -516.2069, 1, 0.8941177, 0, 1,
6.666667, 2.707071, -516.8091, 1, 0.8941177, 0, 1,
6.707071, 2.707071, -517.4559, 1, 0.8941177, 0, 1,
6.747475, 2.707071, -518.1472, 1, 0.8941177, 0, 1,
6.787879, 2.707071, -518.8831, 1, 0.8941177, 0, 1,
6.828283, 2.707071, -519.6635, 1, 0.7921569, 0, 1,
6.868687, 2.707071, -520.4884, 1, 0.7921569, 0, 1,
6.909091, 2.707071, -521.358, 1, 0.7921569, 0, 1,
6.949495, 2.707071, -522.272, 1, 0.7921569, 0, 1,
6.989899, 2.707071, -523.2307, 1, 0.7921569, 0, 1,
7.030303, 2.707071, -524.2338, 1, 0.7921569, 0, 1,
7.070707, 2.707071, -525.2816, 1, 0.6862745, 0, 1,
7.111111, 2.707071, -526.3738, 1, 0.6862745, 0, 1,
7.151515, 2.707071, -527.5107, 1, 0.6862745, 0, 1,
7.191919, 2.707071, -528.6921, 1, 0.6862745, 0, 1,
7.232323, 2.707071, -529.9181, 1, 0.6862745, 0, 1,
7.272727, 2.707071, -531.1886, 1, 0.5843138, 0, 1,
7.313131, 2.707071, -532.5036, 1, 0.5843138, 0, 1,
7.353535, 2.707071, -533.8632, 1, 0.5843138, 0, 1,
7.393939, 2.707071, -535.2674, 1, 0.5843138, 0, 1,
7.434343, 2.707071, -536.7161, 1, 0.4823529, 0, 1,
7.474748, 2.707071, -538.2094, 1, 0.4823529, 0, 1,
7.515152, 2.707071, -539.7472, 1, 0.4823529, 0, 1,
7.555555, 2.707071, -541.3296, 1, 0.3764706, 0, 1,
7.59596, 2.707071, -542.9565, 1, 0.3764706, 0, 1,
7.636364, 2.707071, -544.628, 1, 0.3764706, 0, 1,
7.676768, 2.707071, -546.3441, 1, 0.3764706, 0, 1,
7.717172, 2.707071, -548.1047, 1, 0.2745098, 0, 1,
7.757576, 2.707071, -549.9098, 1, 0.2745098, 0, 1,
7.79798, 2.707071, -551.7595, 1, 0.1686275, 0, 1,
7.838384, 2.707071, -553.6537, 1, 0.1686275, 0, 1,
7.878788, 2.707071, -555.5925, 1, 0.1686275, 0, 1,
7.919192, 2.707071, -557.5759, 1, 0.06666667, 0, 1,
7.959596, 2.707071, -559.6038, 1, 0.06666667, 0, 1,
8, 2.707071, -561.6763, 1, 0.06666667, 0, 1,
4, 2.757576, -569.4435, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.757576, -567.233, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.757576, -565.0654, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.757576, -562.9408, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.757576, -560.8591, 1, 0.06666667, 0, 1,
4.20202, 2.757576, -558.8203, 1, 0.06666667, 0, 1,
4.242424, 2.757576, -556.8245, 1, 0.1686275, 0, 1,
4.282828, 2.757576, -554.8716, 1, 0.1686275, 0, 1,
4.323232, 2.757576, -552.9616, 1, 0.1686275, 0, 1,
4.363636, 2.757576, -551.0946, 1, 0.2745098, 0, 1,
4.40404, 2.757576, -549.2705, 1, 0.2745098, 0, 1,
4.444445, 2.757576, -547.4894, 1, 0.2745098, 0, 1,
4.484848, 2.757576, -545.7512, 1, 0.3764706, 0, 1,
4.525252, 2.757576, -544.0559, 1, 0.3764706, 0, 1,
4.565657, 2.757576, -542.4036, 1, 0.3764706, 0, 1,
4.606061, 2.757576, -540.7941, 1, 0.4823529, 0, 1,
4.646465, 2.757576, -539.2277, 1, 0.4823529, 0, 1,
4.686869, 2.757576, -537.7042, 1, 0.4823529, 0, 1,
4.727273, 2.757576, -536.2236, 1, 0.4823529, 0, 1,
4.767677, 2.757576, -534.7859, 1, 0.5843138, 0, 1,
4.808081, 2.757576, -533.3912, 1, 0.5843138, 0, 1,
4.848485, 2.757576, -532.0394, 1, 0.5843138, 0, 1,
4.888889, 2.757576, -530.7305, 1, 0.5843138, 0, 1,
4.929293, 2.757576, -529.4646, 1, 0.6862745, 0, 1,
4.969697, 2.757576, -528.2416, 1, 0.6862745, 0, 1,
5.010101, 2.757576, -527.0616, 1, 0.6862745, 0, 1,
5.050505, 2.757576, -525.9245, 1, 0.6862745, 0, 1,
5.090909, 2.757576, -524.8303, 1, 0.7921569, 0, 1,
5.131313, 2.757576, -523.7791, 1, 0.7921569, 0, 1,
5.171717, 2.757576, -522.7708, 1, 0.7921569, 0, 1,
5.212121, 2.757576, -521.8054, 1, 0.7921569, 0, 1,
5.252525, 2.757576, -520.883, 1, 0.7921569, 0, 1,
5.292929, 2.757576, -520.0035, 1, 0.7921569, 0, 1,
5.333333, 2.757576, -519.167, 1, 0.8941177, 0, 1,
5.373737, 2.757576, -518.3734, 1, 0.8941177, 0, 1,
5.414141, 2.757576, -517.6227, 1, 0.8941177, 0, 1,
5.454545, 2.757576, -516.915, 1, 0.8941177, 0, 1,
5.494949, 2.757576, -516.2501, 1, 0.8941177, 0, 1,
5.535354, 2.757576, -515.6283, 1, 0.8941177, 0, 1,
5.575758, 2.757576, -515.0494, 1, 0.8941177, 0, 1,
5.616162, 2.757576, -514.5134, 1, 0.8941177, 0, 1,
5.656566, 2.757576, -514.0203, 1, 1, 0, 1,
5.69697, 2.757576, -513.5702, 1, 1, 0, 1,
5.737374, 2.757576, -513.163, 1, 1, 0, 1,
5.777778, 2.757576, -512.7987, 1, 1, 0, 1,
5.818182, 2.757576, -512.4774, 1, 1, 0, 1,
5.858586, 2.757576, -512.199, 1, 1, 0, 1,
5.89899, 2.757576, -511.9636, 1, 1, 0, 1,
5.939394, 2.757576, -511.7711, 1, 1, 0, 1,
5.979798, 2.757576, -511.6215, 1, 1, 0, 1,
6.020202, 2.757576, -511.5149, 1, 1, 0, 1,
6.060606, 2.757576, -511.4512, 1, 1, 0, 1,
6.10101, 2.757576, -511.4304, 1, 1, 0, 1,
6.141414, 2.757576, -511.4526, 1, 1, 0, 1,
6.181818, 2.757576, -511.5177, 1, 1, 0, 1,
6.222222, 2.757576, -511.6258, 1, 1, 0, 1,
6.262626, 2.757576, -511.7768, 1, 1, 0, 1,
6.30303, 2.757576, -511.9707, 1, 1, 0, 1,
6.343434, 2.757576, -512.2075, 1, 1, 0, 1,
6.383838, 2.757576, -512.4873, 1, 1, 0, 1,
6.424242, 2.757576, -512.8101, 1, 1, 0, 1,
6.464646, 2.757576, -513.1757, 1, 1, 0, 1,
6.505051, 2.757576, -513.5844, 1, 1, 0, 1,
6.545455, 2.757576, -514.0359, 1, 1, 0, 1,
6.585859, 2.757576, -514.5303, 1, 0.8941177, 0, 1,
6.626263, 2.757576, -515.0677, 1, 0.8941177, 0, 1,
6.666667, 2.757576, -515.6481, 1, 0.8941177, 0, 1,
6.707071, 2.757576, -516.2714, 1, 0.8941177, 0, 1,
6.747475, 2.757576, -516.9376, 1, 0.8941177, 0, 1,
6.787879, 2.757576, -517.6468, 1, 0.8941177, 0, 1,
6.828283, 2.757576, -518.3989, 1, 0.8941177, 0, 1,
6.868687, 2.757576, -519.1939, 1, 0.8941177, 0, 1,
6.909091, 2.757576, -520.0319, 1, 0.7921569, 0, 1,
6.949495, 2.757576, -520.9128, 1, 0.7921569, 0, 1,
6.989899, 2.757576, -521.8366, 1, 0.7921569, 0, 1,
7.030303, 2.757576, -522.8033, 1, 0.7921569, 0, 1,
7.070707, 2.757576, -523.813, 1, 0.7921569, 0, 1,
7.111111, 2.757576, -524.8657, 1, 0.7921569, 0, 1,
7.151515, 2.757576, -525.9613, 1, 0.6862745, 0, 1,
7.191919, 2.757576, -527.0999, 1, 0.6862745, 0, 1,
7.232323, 2.757576, -528.2813, 1, 0.6862745, 0, 1,
7.272727, 2.757576, -529.5057, 1, 0.6862745, 0, 1,
7.313131, 2.757576, -530.773, 1, 0.5843138, 0, 1,
7.353535, 2.757576, -532.0833, 1, 0.5843138, 0, 1,
7.393939, 2.757576, -533.4365, 1, 0.5843138, 0, 1,
7.434343, 2.757576, -534.8326, 1, 0.5843138, 0, 1,
7.474748, 2.757576, -536.2717, 1, 0.4823529, 0, 1,
7.515152, 2.757576, -537.7537, 1, 0.4823529, 0, 1,
7.555555, 2.757576, -539.2786, 1, 0.4823529, 0, 1,
7.59596, 2.757576, -540.8465, 1, 0.4823529, 0, 1,
7.636364, 2.757576, -542.4573, 1, 0.3764706, 0, 1,
7.676768, 2.757576, -544.1111, 1, 0.3764706, 0, 1,
7.717172, 2.757576, -545.8078, 1, 0.3764706, 0, 1,
7.757576, 2.757576, -547.5474, 1, 0.2745098, 0, 1,
7.79798, 2.757576, -549.33, 1, 0.2745098, 0, 1,
7.838384, 2.757576, -551.1555, 1, 0.2745098, 0, 1,
7.878788, 2.757576, -553.0239, 1, 0.1686275, 0, 1,
7.919192, 2.757576, -554.9353, 1, 0.1686275, 0, 1,
7.959596, 2.757576, -556.8896, 1, 0.1686275, 0, 1,
8, 2.757576, -558.8868, 1, 0.06666667, 0, 1,
4, 2.808081, -566.5575, 0.9647059, 0, 0.03137255, 1,
4.040404, 2.808081, -564.4258, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.808081, -562.3354, 1, 0.06666667, 0, 1,
4.121212, 2.808081, -560.2866, 1, 0.06666667, 0, 1,
4.161616, 2.808081, -558.2791, 1, 0.06666667, 0, 1,
4.20202, 2.808081, -556.313, 1, 0.1686275, 0, 1,
4.242424, 2.808081, -554.3883, 1, 0.1686275, 0, 1,
4.282828, 2.808081, -552.505, 1, 0.1686275, 0, 1,
4.323232, 2.808081, -550.6631, 1, 0.2745098, 0, 1,
4.363636, 2.808081, -548.8627, 1, 0.2745098, 0, 1,
4.40404, 2.808081, -547.1036, 1, 0.2745098, 0, 1,
4.444445, 2.808081, -545.386, 1, 0.3764706, 0, 1,
4.484848, 2.808081, -543.7097, 1, 0.3764706, 0, 1,
4.525252, 2.808081, -542.0749, 1, 0.3764706, 0, 1,
4.565657, 2.808081, -540.4814, 1, 0.4823529, 0, 1,
4.606061, 2.808081, -538.9294, 1, 0.4823529, 0, 1,
4.646465, 2.808081, -537.4188, 1, 0.4823529, 0, 1,
4.686869, 2.808081, -535.9496, 1, 0.4823529, 0, 1,
4.727273, 2.808081, -534.5217, 1, 0.5843138, 0, 1,
4.767677, 2.808081, -533.1353, 1, 0.5843138, 0, 1,
4.808081, 2.808081, -531.7903, 1, 0.5843138, 0, 1,
4.848485, 2.808081, -530.4868, 1, 0.5843138, 0, 1,
4.888889, 2.808081, -529.2245, 1, 0.6862745, 0, 1,
4.929293, 2.808081, -528.0038, 1, 0.6862745, 0, 1,
4.969697, 2.808081, -526.8244, 1, 0.6862745, 0, 1,
5.010101, 2.808081, -525.6864, 1, 0.6862745, 0, 1,
5.050505, 2.808081, -524.5898, 1, 0.7921569, 0, 1,
5.090909, 2.808081, -523.5347, 1, 0.7921569, 0, 1,
5.131313, 2.808081, -522.5209, 1, 0.7921569, 0, 1,
5.171717, 2.808081, -521.5486, 1, 0.7921569, 0, 1,
5.212121, 2.808081, -520.6176, 1, 0.7921569, 0, 1,
5.252525, 2.808081, -519.7281, 1, 0.7921569, 0, 1,
5.292929, 2.808081, -518.8799, 1, 0.8941177, 0, 1,
5.333333, 2.808081, -518.0732, 1, 0.8941177, 0, 1,
5.373737, 2.808081, -517.3079, 1, 0.8941177, 0, 1,
5.414141, 2.808081, -516.584, 1, 0.8941177, 0, 1,
5.454545, 2.808081, -515.9014, 1, 0.8941177, 0, 1,
5.494949, 2.808081, -515.2604, 1, 0.8941177, 0, 1,
5.535354, 2.808081, -514.6606, 1, 0.8941177, 0, 1,
5.575758, 2.808081, -514.1024, 1, 1, 0, 1,
5.616162, 2.808081, -513.5854, 1, 1, 0, 1,
5.656566, 2.808081, -513.11, 1, 1, 0, 1,
5.69697, 2.808081, -512.6759, 1, 1, 0, 1,
5.737374, 2.808081, -512.2833, 1, 1, 0, 1,
5.777778, 2.808081, -511.932, 1, 1, 0, 1,
5.818182, 2.808081, -511.6221, 1, 1, 0, 1,
5.858586, 2.808081, -511.3537, 1, 1, 0, 1,
5.89899, 2.808081, -511.1266, 1, 1, 0, 1,
5.939394, 2.808081, -510.9409, 1, 1, 0, 1,
5.979798, 2.808081, -510.7967, 1, 1, 0, 1,
6.020202, 2.808081, -510.6939, 1, 1, 0, 1,
6.060606, 2.808081, -510.6325, 1, 1, 0, 1,
6.10101, 2.808081, -510.6125, 1, 1, 0, 1,
6.141414, 2.808081, -510.6338, 1, 1, 0, 1,
6.181818, 2.808081, -510.6966, 1, 1, 0, 1,
6.222222, 2.808081, -510.8008, 1, 1, 0, 1,
6.262626, 2.808081, -510.9464, 1, 1, 0, 1,
6.30303, 2.808081, -511.1334, 1, 1, 0, 1,
6.343434, 2.808081, -511.3618, 1, 1, 0, 1,
6.383838, 2.808081, -511.6317, 1, 1, 0, 1,
6.424242, 2.808081, -511.9429, 1, 1, 0, 1,
6.464646, 2.808081, -512.2955, 1, 1, 0, 1,
6.505051, 2.808081, -512.6896, 1, 1, 0, 1,
6.545455, 2.808081, -513.125, 1, 1, 0, 1,
6.585859, 2.808081, -513.6019, 1, 1, 0, 1,
6.626263, 2.808081, -514.1201, 1, 1, 0, 1,
6.666667, 2.808081, -514.6797, 1, 0.8941177, 0, 1,
6.707071, 2.808081, -515.2808, 1, 0.8941177, 0, 1,
6.747475, 2.808081, -515.9233, 1, 0.8941177, 0, 1,
6.787879, 2.808081, -516.6072, 1, 0.8941177, 0, 1,
6.828283, 2.808081, -517.3325, 1, 0.8941177, 0, 1,
6.868687, 2.808081, -518.0991, 1, 0.8941177, 0, 1,
6.909091, 2.808081, -518.9072, 1, 0.8941177, 0, 1,
6.949495, 2.808081, -519.7567, 1, 0.7921569, 0, 1,
6.989899, 2.808081, -520.6476, 1, 0.7921569, 0, 1,
7.030303, 2.808081, -521.58, 1, 0.7921569, 0, 1,
7.070707, 2.808081, -522.5536, 1, 0.7921569, 0, 1,
7.111111, 2.808081, -523.5688, 1, 0.7921569, 0, 1,
7.151515, 2.808081, -524.6253, 1, 0.7921569, 0, 1,
7.191919, 2.808081, -525.7233, 1, 0.6862745, 0, 1,
7.232323, 2.808081, -526.8626, 1, 0.6862745, 0, 1,
7.272727, 2.808081, -528.0433, 1, 0.6862745, 0, 1,
7.313131, 2.808081, -529.2655, 1, 0.6862745, 0, 1,
7.353535, 2.808081, -530.5291, 1, 0.5843138, 0, 1,
7.393939, 2.808081, -531.834, 1, 0.5843138, 0, 1,
7.434343, 2.808081, -533.1804, 1, 0.5843138, 0, 1,
7.474748, 2.808081, -534.5682, 1, 0.5843138, 0, 1,
7.515152, 2.808081, -535.9973, 1, 0.4823529, 0, 1,
7.555555, 2.808081, -537.468, 1, 0.4823529, 0, 1,
7.59596, 2.808081, -538.9799, 1, 0.4823529, 0, 1,
7.636364, 2.808081, -540.5333, 1, 0.4823529, 0, 1,
7.676768, 2.808081, -542.1281, 1, 0.3764706, 0, 1,
7.717172, 2.808081, -543.7643, 1, 0.3764706, 0, 1,
7.757576, 2.808081, -545.442, 1, 0.3764706, 0, 1,
7.79798, 2.808081, -547.1609, 1, 0.2745098, 0, 1,
7.838384, 2.808081, -548.9214, 1, 0.2745098, 0, 1,
7.878788, 2.808081, -550.7232, 1, 0.2745098, 0, 1,
7.919192, 2.808081, -552.5665, 1, 0.1686275, 0, 1,
7.959596, 2.808081, -554.4511, 1, 0.1686275, 0, 1,
8, 2.808081, -556.3771, 1, 0.1686275, 0, 1,
4, 2.858586, -563.949, 0.9647059, 0, 0.03137255, 1,
4.040404, 2.858586, -561.892, 1, 0.06666667, 0, 1,
4.080808, 2.858586, -559.8749, 1, 0.06666667, 0, 1,
4.121212, 2.858586, -557.8978, 1, 0.06666667, 0, 1,
4.161616, 2.858586, -555.9606, 1, 0.1686275, 0, 1,
4.20202, 2.858586, -554.0634, 1, 0.1686275, 0, 1,
4.242424, 2.858586, -552.2061, 1, 0.1686275, 0, 1,
4.282828, 2.858586, -550.3888, 1, 0.2745098, 0, 1,
4.323232, 2.858586, -548.6114, 1, 0.2745098, 0, 1,
4.363636, 2.858586, -546.874, 1, 0.2745098, 0, 1,
4.40404, 2.858586, -545.1765, 1, 0.3764706, 0, 1,
4.444445, 2.858586, -543.519, 1, 0.3764706, 0, 1,
4.484848, 2.858586, -541.9015, 1, 0.3764706, 0, 1,
4.525252, 2.858586, -540.3239, 1, 0.4823529, 0, 1,
4.565657, 2.858586, -538.7863, 1, 0.4823529, 0, 1,
4.606061, 2.858586, -537.2886, 1, 0.4823529, 0, 1,
4.646465, 2.858586, -535.8309, 1, 0.4823529, 0, 1,
4.686869, 2.858586, -534.4131, 1, 0.5843138, 0, 1,
4.727273, 2.858586, -533.0353, 1, 0.5843138, 0, 1,
4.767677, 2.858586, -531.6974, 1, 0.5843138, 0, 1,
4.808081, 2.858586, -530.3996, 1, 0.5843138, 0, 1,
4.848485, 2.858586, -529.1417, 1, 0.6862745, 0, 1,
4.888889, 2.858586, -527.9236, 1, 0.6862745, 0, 1,
4.929293, 2.858586, -526.7456, 1, 0.6862745, 0, 1,
4.969697, 2.858586, -525.6075, 1, 0.6862745, 0, 1,
5.010101, 2.858586, -524.5094, 1, 0.7921569, 0, 1,
5.050505, 2.858586, -523.4513, 1, 0.7921569, 0, 1,
5.090909, 2.858586, -522.433, 1, 0.7921569, 0, 1,
5.131313, 2.858586, -521.4548, 1, 0.7921569, 0, 1,
5.171717, 2.858586, -520.5165, 1, 0.7921569, 0, 1,
5.212121, 2.858586, -519.6182, 1, 0.7921569, 0, 1,
5.252525, 2.858586, -518.7598, 1, 0.8941177, 0, 1,
5.292929, 2.858586, -517.9413, 1, 0.8941177, 0, 1,
5.333333, 2.858586, -517.1629, 1, 0.8941177, 0, 1,
5.373737, 2.858586, -516.4244, 1, 0.8941177, 0, 1,
5.414141, 2.858586, -515.7258, 1, 0.8941177, 0, 1,
5.454545, 2.858586, -515.0672, 1, 0.8941177, 0, 1,
5.494949, 2.858586, -514.4485, 1, 0.8941177, 0, 1,
5.535354, 2.858586, -513.8698, 1, 1, 0, 1,
5.575758, 2.858586, -513.3311, 1, 1, 0, 1,
5.616162, 2.858586, -512.8323, 1, 1, 0, 1,
5.656566, 2.858586, -512.3735, 1, 1, 0, 1,
5.69697, 2.858586, -511.9546, 1, 1, 0, 1,
5.737374, 2.858586, -511.5757, 1, 1, 0, 1,
5.777778, 2.858586, -511.2367, 1, 1, 0, 1,
5.818182, 2.858586, -510.9377, 1, 1, 0, 1,
5.858586, 2.858586, -510.6786, 1, 1, 0, 1,
5.89899, 2.858586, -510.4596, 1, 1, 0, 1,
5.939394, 2.858586, -510.2804, 1, 1, 0, 1,
5.979798, 2.858586, -510.1412, 1, 1, 0, 1,
6.020202, 2.858586, -510.042, 1, 1, 0, 1,
6.060606, 2.858586, -509.9827, 1, 1, 0, 1,
6.10101, 2.858586, -509.9634, 1, 1, 0, 1,
6.141414, 2.858586, -509.984, 1, 1, 0, 1,
6.181818, 2.858586, -510.0446, 1, 1, 0, 1,
6.222222, 2.858586, -510.1452, 1, 1, 0, 1,
6.262626, 2.858586, -510.2857, 1, 1, 0, 1,
6.30303, 2.858586, -510.4662, 1, 1, 0, 1,
6.343434, 2.858586, -510.6866, 1, 1, 0, 1,
6.383838, 2.858586, -510.9469, 1, 1, 0, 1,
6.424242, 2.858586, -511.2473, 1, 1, 0, 1,
6.464646, 2.858586, -511.5876, 1, 1, 0, 1,
6.505051, 2.858586, -511.9678, 1, 1, 0, 1,
6.545455, 2.858586, -512.388, 1, 1, 0, 1,
6.585859, 2.858586, -512.8481, 1, 1, 0, 1,
6.626263, 2.858586, -513.3482, 1, 1, 0, 1,
6.666667, 2.858586, -513.8883, 1, 1, 0, 1,
6.707071, 2.858586, -514.4683, 1, 0.8941177, 0, 1,
6.747475, 2.858586, -515.0883, 1, 0.8941177, 0, 1,
6.787879, 2.858586, -515.7482, 1, 0.8941177, 0, 1,
6.828283, 2.858586, -516.4481, 1, 0.8941177, 0, 1,
6.868687, 2.858586, -517.1879, 1, 0.8941177, 0, 1,
6.909091, 2.858586, -517.9677, 1, 0.8941177, 0, 1,
6.949495, 2.858586, -518.7875, 1, 0.8941177, 0, 1,
6.989899, 2.858586, -519.6472, 1, 0.7921569, 0, 1,
7.030303, 2.858586, -520.5468, 1, 0.7921569, 0, 1,
7.070707, 2.858586, -521.4865, 1, 0.7921569, 0, 1,
7.111111, 2.858586, -522.466, 1, 0.7921569, 0, 1,
7.151515, 2.858586, -523.4855, 1, 0.7921569, 0, 1,
7.191919, 2.858586, -524.545, 1, 0.7921569, 0, 1,
7.232323, 2.858586, -525.6444, 1, 0.6862745, 0, 1,
7.272727, 2.858586, -526.7838, 1, 0.6862745, 0, 1,
7.313131, 2.858586, -527.9632, 1, 0.6862745, 0, 1,
7.353535, 2.858586, -529.1825, 1, 0.6862745, 0, 1,
7.393939, 2.858586, -530.4417, 1, 0.5843138, 0, 1,
7.434343, 2.858586, -531.741, 1, 0.5843138, 0, 1,
7.474748, 2.858586, -533.0801, 1, 0.5843138, 0, 1,
7.515152, 2.858586, -534.4592, 1, 0.5843138, 0, 1,
7.555555, 2.858586, -535.8783, 1, 0.4823529, 0, 1,
7.59596, 2.858586, -537.3373, 1, 0.4823529, 0, 1,
7.636364, 2.858586, -538.8364, 1, 0.4823529, 0, 1,
7.676768, 2.858586, -540.3753, 1, 0.4823529, 0, 1,
7.717172, 2.858586, -541.9542, 1, 0.3764706, 0, 1,
7.757576, 2.858586, -543.5731, 1, 0.3764706, 0, 1,
7.79798, 2.858586, -545.2319, 1, 0.3764706, 0, 1,
7.838384, 2.858586, -546.9307, 1, 0.2745098, 0, 1,
7.878788, 2.858586, -548.6694, 1, 0.2745098, 0, 1,
7.919192, 2.858586, -550.4481, 1, 0.2745098, 0, 1,
7.959596, 2.858586, -552.2667, 1, 0.1686275, 0, 1,
8, 2.858586, -554.1253, 1, 0.1686275, 0, 1,
4, 2.909091, -561.5969, 1, 0.06666667, 0, 1,
4.040404, 2.909091, -559.6107, 1, 0.06666667, 0, 1,
4.080808, 2.909091, -557.663, 1, 0.06666667, 0, 1,
4.121212, 2.909091, -555.754, 1, 0.1686275, 0, 1,
4.161616, 2.909091, -553.8834, 1, 0.1686275, 0, 1,
4.20202, 2.909091, -552.0515, 1, 0.1686275, 0, 1,
4.242424, 2.909091, -550.2582, 1, 0.2745098, 0, 1,
4.282828, 2.909091, -548.5034, 1, 0.2745098, 0, 1,
4.323232, 2.909091, -546.7872, 1, 0.2745098, 0, 1,
4.363636, 2.909091, -545.1096, 1, 0.3764706, 0, 1,
4.40404, 2.909091, -543.4706, 1, 0.3764706, 0, 1,
4.444445, 2.909091, -541.8701, 1, 0.3764706, 0, 1,
4.484848, 2.909091, -540.3083, 1, 0.4823529, 0, 1,
4.525252, 2.909091, -538.785, 1, 0.4823529, 0, 1,
4.565657, 2.909091, -537.3003, 1, 0.4823529, 0, 1,
4.606061, 2.909091, -535.8542, 1, 0.4823529, 0, 1,
4.646465, 2.909091, -534.4466, 1, 0.5843138, 0, 1,
4.686869, 2.909091, -533.0776, 1, 0.5843138, 0, 1,
4.727273, 2.909091, -531.7473, 1, 0.5843138, 0, 1,
4.767677, 2.909091, -530.4555, 1, 0.5843138, 0, 1,
4.808081, 2.909091, -529.2023, 1, 0.6862745, 0, 1,
4.848485, 2.909091, -527.9876, 1, 0.6862745, 0, 1,
4.888889, 2.909091, -526.8115, 1, 0.6862745, 0, 1,
4.929293, 2.909091, -525.6741, 1, 0.6862745, 0, 1,
4.969697, 2.909091, -524.5752, 1, 0.7921569, 0, 1,
5.010101, 2.909091, -523.5148, 1, 0.7921569, 0, 1,
5.050505, 2.909091, -522.4931, 1, 0.7921569, 0, 1,
5.090909, 2.909091, -521.5099, 1, 0.7921569, 0, 1,
5.131313, 2.909091, -520.5654, 1, 0.7921569, 0, 1,
5.171717, 2.909091, -519.6594, 1, 0.7921569, 0, 1,
5.212121, 2.909091, -518.7919, 1, 0.8941177, 0, 1,
5.252525, 2.909091, -517.9631, 1, 0.8941177, 0, 1,
5.292929, 2.909091, -517.1729, 1, 0.8941177, 0, 1,
5.333333, 2.909091, -516.4211, 1, 0.8941177, 0, 1,
5.373737, 2.909091, -515.7081, 1, 0.8941177, 0, 1,
5.414141, 2.909091, -515.0336, 1, 0.8941177, 0, 1,
5.454545, 2.909091, -514.3976, 1, 0.8941177, 0, 1,
5.494949, 2.909091, -513.8002, 1, 1, 0, 1,
5.535354, 2.909091, -513.2415, 1, 1, 0, 1,
5.575758, 2.909091, -512.7213, 1, 1, 0, 1,
5.616162, 2.909091, -512.2397, 1, 1, 0, 1,
5.656566, 2.909091, -511.7966, 1, 1, 0, 1,
5.69697, 2.909091, -511.3922, 1, 1, 0, 1,
5.737374, 2.909091, -511.0263, 1, 1, 0, 1,
5.777778, 2.909091, -510.699, 1, 1, 0, 1,
5.818182, 2.909091, -510.4103, 1, 1, 0, 1,
5.858586, 2.909091, -510.1602, 1, 1, 0, 1,
5.89899, 2.909091, -509.9486, 1, 1, 0, 1,
5.939394, 2.909091, -509.7756, 1, 1, 0, 1,
5.979798, 2.909091, -509.6412, 1, 1, 0, 1,
6.020202, 2.909091, -509.5454, 1, 1, 0, 1,
6.060606, 2.909091, -509.4882, 1, 1, 0, 1,
6.10101, 2.909091, -509.4695, 1, 1, 0, 1,
6.141414, 2.909091, -509.4894, 1, 1, 0, 1,
6.181818, 2.909091, -509.5479, 1, 1, 0, 1,
6.222222, 2.909091, -509.6451, 1, 1, 0, 1,
6.262626, 2.909091, -509.7807, 1, 1, 0, 1,
6.30303, 2.909091, -509.955, 1, 1, 0, 1,
6.343434, 2.909091, -510.1678, 1, 1, 0, 1,
6.383838, 2.909091, -510.4192, 1, 1, 0, 1,
6.424242, 2.909091, -510.7092, 1, 1, 0, 1,
6.464646, 2.909091, -511.0378, 1, 1, 0, 1,
6.505051, 2.909091, -511.4049, 1, 1, 0, 1,
6.545455, 2.909091, -511.8106, 1, 1, 0, 1,
6.585859, 2.909091, -512.2549, 1, 1, 0, 1,
6.626263, 2.909091, -512.7378, 1, 1, 0, 1,
6.666667, 2.909091, -513.2593, 1, 1, 0, 1,
6.707071, 2.909091, -513.8193, 1, 1, 0, 1,
6.747475, 2.909091, -514.418, 1, 0.8941177, 0, 1,
6.787879, 2.909091, -515.0552, 1, 0.8941177, 0, 1,
6.828283, 2.909091, -515.731, 1, 0.8941177, 0, 1,
6.868687, 2.909091, -516.4453, 1, 0.8941177, 0, 1,
6.909091, 2.909091, -517.1983, 1, 0.8941177, 0, 1,
6.949495, 2.909091, -517.9898, 1, 0.8941177, 0, 1,
6.989899, 2.909091, -518.8199, 1, 0.8941177, 0, 1,
7.030303, 2.909091, -519.6886, 1, 0.7921569, 0, 1,
7.070707, 2.909091, -520.5959, 1, 0.7921569, 0, 1,
7.111111, 2.909091, -521.5417, 1, 0.7921569, 0, 1,
7.151515, 2.909091, -522.5262, 1, 0.7921569, 0, 1,
7.191919, 2.909091, -523.5492, 1, 0.7921569, 0, 1,
7.232323, 2.909091, -524.6108, 1, 0.7921569, 0, 1,
7.272727, 2.909091, -525.7109, 1, 0.6862745, 0, 1,
7.313131, 2.909091, -526.8497, 1, 0.6862745, 0, 1,
7.353535, 2.909091, -528.027, 1, 0.6862745, 0, 1,
7.393939, 2.909091, -529.243, 1, 0.6862745, 0, 1,
7.434343, 2.909091, -530.4974, 1, 0.5843138, 0, 1,
7.474748, 2.909091, -531.7905, 1, 0.5843138, 0, 1,
7.515152, 2.909091, -533.1222, 1, 0.5843138, 0, 1,
7.555555, 2.909091, -534.4924, 1, 0.5843138, 0, 1,
7.59596, 2.909091, -535.9012, 1, 0.4823529, 0, 1,
7.636364, 2.909091, -537.3486, 1, 0.4823529, 0, 1,
7.676768, 2.909091, -538.8346, 1, 0.4823529, 0, 1,
7.717172, 2.909091, -540.3591, 1, 0.4823529, 0, 1,
7.757576, 2.909091, -541.9223, 1, 0.3764706, 0, 1,
7.79798, 2.909091, -543.524, 1, 0.3764706, 0, 1,
7.838384, 2.909091, -545.1643, 1, 0.3764706, 0, 1,
7.878788, 2.909091, -546.8432, 1, 0.2745098, 0, 1,
7.919192, 2.909091, -548.5607, 1, 0.2745098, 0, 1,
7.959596, 2.909091, -550.3167, 1, 0.2745098, 0, 1,
8, 2.909091, -552.1113, 1, 0.1686275, 0, 1,
4, 2.959596, -559.4817, 1, 0.06666667, 0, 1,
4.040404, 2.959596, -557.5627, 1, 0.06666667, 0, 1,
4.080808, 2.959596, -555.6809, 1, 0.1686275, 0, 1,
4.121212, 2.959596, -553.8364, 1, 0.1686275, 0, 1,
4.161616, 2.959596, -552.0292, 1, 0.1686275, 0, 1,
4.20202, 2.959596, -550.2593, 1, 0.2745098, 0, 1,
4.242424, 2.959596, -548.5266, 1, 0.2745098, 0, 1,
4.282828, 2.959596, -546.8312, 1, 0.2745098, 0, 1,
4.323232, 2.959596, -545.1731, 1, 0.3764706, 0, 1,
4.363636, 2.959596, -543.5522, 1, 0.3764706, 0, 1,
4.40404, 2.959596, -541.9687, 1, 0.3764706, 0, 1,
4.444445, 2.959596, -540.4224, 1, 0.4823529, 0, 1,
4.484848, 2.959596, -538.9134, 1, 0.4823529, 0, 1,
4.525252, 2.959596, -537.4417, 1, 0.4823529, 0, 1,
4.565657, 2.959596, -536.0072, 1, 0.4823529, 0, 1,
4.606061, 2.959596, -534.61, 1, 0.5843138, 0, 1,
4.646465, 2.959596, -533.2501, 1, 0.5843138, 0, 1,
4.686869, 2.959596, -531.9275, 1, 0.5843138, 0, 1,
4.727273, 2.959596, -530.6421, 1, 0.5843138, 0, 1,
4.767677, 2.959596, -529.394, 1, 0.6862745, 0, 1,
4.808081, 2.959596, -528.1832, 1, 0.6862745, 0, 1,
4.848485, 2.959596, -527.0096, 1, 0.6862745, 0, 1,
4.888889, 2.959596, -525.8734, 1, 0.6862745, 0, 1,
4.929293, 2.959596, -524.7744, 1, 0.7921569, 0, 1,
4.969697, 2.959596, -523.7127, 1, 0.7921569, 0, 1,
5.010101, 2.959596, -522.6882, 1, 0.7921569, 0, 1,
5.050505, 2.959596, -521.701, 1, 0.7921569, 0, 1,
5.090909, 2.959596, -520.7512, 1, 0.7921569, 0, 1,
5.131313, 2.959596, -519.8386, 1, 0.7921569, 0, 1,
5.171717, 2.959596, -518.9632, 1, 0.8941177, 0, 1,
5.212121, 2.959596, -518.1252, 1, 0.8941177, 0, 1,
5.252525, 2.959596, -517.3243, 1, 0.8941177, 0, 1,
5.292929, 2.959596, -516.5609, 1, 0.8941177, 0, 1,
5.333333, 2.959596, -515.8346, 1, 0.8941177, 0, 1,
5.373737, 2.959596, -515.1456, 1, 0.8941177, 0, 1,
5.414141, 2.959596, -514.494, 1, 0.8941177, 0, 1,
5.454545, 2.959596, -513.8795, 1, 1, 0, 1,
5.494949, 2.959596, -513.3024, 1, 1, 0, 1,
5.535354, 2.959596, -512.7625, 1, 1, 0, 1,
5.575758, 2.959596, -512.2599, 1, 1, 0, 1,
5.616162, 2.959596, -511.7946, 1, 1, 0, 1,
5.656566, 2.959596, -511.3665, 1, 1, 0, 1,
5.69697, 2.959596, -510.9758, 1, 1, 0, 1,
5.737374, 2.959596, -510.6223, 1, 1, 0, 1,
5.777778, 2.959596, -510.3061, 1, 1, 0, 1,
5.818182, 2.959596, -510.0271, 1, 1, 0, 1,
5.858586, 2.959596, -509.7855, 1, 1, 0, 1,
5.89899, 2.959596, -509.5811, 1, 1, 0, 1,
5.939394, 2.959596, -509.4139, 1, 1, 0, 1,
5.979798, 2.959596, -509.2841, 1, 1, 0, 1,
6.020202, 2.959596, -509.1915, 1, 1, 0, 1,
6.060606, 2.959596, -509.1362, 1, 1, 0, 1,
6.10101, 2.959596, -509.1182, 1, 1, 0, 1,
6.141414, 2.959596, -509.1375, 1, 1, 0, 1,
6.181818, 2.959596, -509.194, 1, 1, 0, 1,
6.222222, 2.959596, -509.2878, 1, 1, 0, 1,
6.262626, 2.959596, -509.4189, 1, 1, 0, 1,
6.30303, 2.959596, -509.5872, 1, 1, 0, 1,
6.343434, 2.959596, -509.7928, 1, 1, 0, 1,
6.383838, 2.959596, -510.0357, 1, 1, 0, 1,
6.424242, 2.959596, -510.3159, 1, 1, 0, 1,
6.464646, 2.959596, -510.6333, 1, 1, 0, 1,
6.505051, 2.959596, -510.9881, 1, 1, 0, 1,
6.545455, 2.959596, -511.3801, 1, 1, 0, 1,
6.585859, 2.959596, -511.8094, 1, 1, 0, 1,
6.626263, 2.959596, -512.2759, 1, 1, 0, 1,
6.666667, 2.959596, -512.7797, 1, 1, 0, 1,
6.707071, 2.959596, -513.3208, 1, 1, 0, 1,
6.747475, 2.959596, -513.8992, 1, 1, 0, 1,
6.787879, 2.959596, -514.5148, 1, 0.8941177, 0, 1,
6.828283, 2.959596, -515.1678, 1, 0.8941177, 0, 1,
6.868687, 2.959596, -515.858, 1, 0.8941177, 0, 1,
6.909091, 2.959596, -516.5854, 1, 0.8941177, 0, 1,
6.949495, 2.959596, -517.3502, 1, 0.8941177, 0, 1,
6.989899, 2.959596, -518.1522, 1, 0.8941177, 0, 1,
7.030303, 2.959596, -518.9915, 1, 0.8941177, 0, 1,
7.070707, 2.959596, -519.868, 1, 0.7921569, 0, 1,
7.111111, 2.959596, -520.7819, 1, 0.7921569, 0, 1,
7.151515, 2.959596, -521.733, 1, 0.7921569, 0, 1,
7.191919, 2.959596, -522.7214, 1, 0.7921569, 0, 1,
7.232323, 2.959596, -523.7471, 1, 0.7921569, 0, 1,
7.272727, 2.959596, -524.8101, 1, 0.7921569, 0, 1,
7.313131, 2.959596, -525.9103, 1, 0.6862745, 0, 1,
7.353535, 2.959596, -527.0477, 1, 0.6862745, 0, 1,
7.393939, 2.959596, -528.2225, 1, 0.6862745, 0, 1,
7.434343, 2.959596, -529.4346, 1, 0.6862745, 0, 1,
7.474748, 2.959596, -530.6839, 1, 0.5843138, 0, 1,
7.515152, 2.959596, -531.9705, 1, 0.5843138, 0, 1,
7.555555, 2.959596, -533.2944, 1, 0.5843138, 0, 1,
7.59596, 2.959596, -534.6555, 1, 0.5843138, 0, 1,
7.636364, 2.959596, -536.0539, 1, 0.4823529, 0, 1,
7.676768, 2.959596, -537.4896, 1, 0.4823529, 0, 1,
7.717172, 2.959596, -538.9626, 1, 0.4823529, 0, 1,
7.757576, 2.959596, -540.4728, 1, 0.4823529, 0, 1,
7.79798, 2.959596, -542.0203, 1, 0.3764706, 0, 1,
7.838384, 2.959596, -543.6051, 1, 0.3764706, 0, 1,
7.878788, 2.959596, -545.2272, 1, 0.3764706, 0, 1,
7.919192, 2.959596, -546.8865, 1, 0.2745098, 0, 1,
7.959596, 2.959596, -548.5831, 1, 0.2745098, 0, 1,
8, 2.959596, -550.317, 1, 0.2745098, 0, 1,
4, 3.010101, -557.5856, 1, 0.06666667, 0, 1,
4.040404, 3.010101, -555.7304, 1, 0.1686275, 0, 1,
4.080808, 3.010101, -553.9113, 1, 0.1686275, 0, 1,
4.121212, 3.010101, -552.1282, 1, 0.1686275, 0, 1,
4.161616, 3.010101, -550.3811, 1, 0.2745098, 0, 1,
4.20202, 3.010101, -548.67, 1, 0.2745098, 0, 1,
4.242424, 3.010101, -546.9951, 1, 0.2745098, 0, 1,
4.282828, 3.010101, -545.3561, 1, 0.3764706, 0, 1,
4.323232, 3.010101, -543.7531, 1, 0.3764706, 0, 1,
4.363636, 3.010101, -542.1862, 1, 0.3764706, 0, 1,
4.40404, 3.010101, -540.6553, 1, 0.4823529, 0, 1,
4.444445, 3.010101, -539.1605, 1, 0.4823529, 0, 1,
4.484848, 3.010101, -537.7017, 1, 0.4823529, 0, 1,
4.525252, 3.010101, -536.279, 1, 0.4823529, 0, 1,
4.565657, 3.010101, -534.8922, 1, 0.5843138, 0, 1,
4.606061, 3.010101, -533.5416, 1, 0.5843138, 0, 1,
4.646465, 3.010101, -532.2269, 1, 0.5843138, 0, 1,
4.686869, 3.010101, -530.9482, 1, 0.5843138, 0, 1,
4.727273, 3.010101, -529.7056, 1, 0.6862745, 0, 1,
4.767677, 3.010101, -528.4991, 1, 0.6862745, 0, 1,
4.808081, 3.010101, -527.3286, 1, 0.6862745, 0, 1,
4.848485, 3.010101, -526.1941, 1, 0.6862745, 0, 1,
4.888889, 3.010101, -525.0956, 1, 0.6862745, 0, 1,
4.929293, 3.010101, -524.0332, 1, 0.7921569, 0, 1,
4.969697, 3.010101, -523.0068, 1, 0.7921569, 0, 1,
5.010101, 3.010101, -522.0165, 1, 0.7921569, 0, 1,
5.050505, 3.010101, -521.0621, 1, 0.7921569, 0, 1,
5.090909, 3.010101, -520.1439, 1, 0.7921569, 0, 1,
5.131313, 3.010101, -519.2616, 1, 0.8941177, 0, 1,
5.171717, 3.010101, -518.4154, 1, 0.8941177, 0, 1,
5.212121, 3.010101, -517.6052, 1, 0.8941177, 0, 1,
5.252525, 3.010101, -516.8311, 1, 0.8941177, 0, 1,
5.292929, 3.010101, -516.093, 1, 0.8941177, 0, 1,
5.333333, 3.010101, -515.3909, 1, 0.8941177, 0, 1,
5.373737, 3.010101, -514.7249, 1, 0.8941177, 0, 1,
5.414141, 3.010101, -514.0948, 1, 1, 0, 1,
5.454545, 3.010101, -513.5009, 1, 1, 0, 1,
5.494949, 3.010101, -512.9429, 1, 1, 0, 1,
5.535354, 3.010101, -512.421, 1, 1, 0, 1,
5.575758, 3.010101, -511.9352, 1, 1, 0, 1,
5.616162, 3.010101, -511.4853, 1, 1, 0, 1,
5.656566, 3.010101, -511.0715, 1, 1, 0, 1,
5.69697, 3.010101, -510.6938, 1, 1, 0, 1,
5.737374, 3.010101, -510.352, 1, 1, 0, 1,
5.777778, 3.010101, -510.0463, 1, 1, 0, 1,
5.818182, 3.010101, -509.7766, 1, 1, 0, 1,
5.858586, 3.010101, -509.543, 1, 1, 0, 1,
5.89899, 3.010101, -509.3454, 1, 1, 0, 1,
5.939394, 3.010101, -509.1839, 1, 1, 0, 1,
5.979798, 3.010101, -509.0583, 1, 1, 0, 1,
6.020202, 3.010101, -508.9688, 1, 1, 0, 1,
6.060606, 3.010101, -508.9154, 1, 1, 0, 1,
6.10101, 3.010101, -508.8979, 1, 1, 0, 1,
6.141414, 3.010101, -508.9166, 1, 1, 0, 1,
6.181818, 3.010101, -508.9712, 1, 1, 0, 1,
6.222222, 3.010101, -509.0619, 1, 1, 0, 1,
6.262626, 3.010101, -509.1886, 1, 1, 0, 1,
6.30303, 3.010101, -509.3513, 1, 1, 0, 1,
6.343434, 3.010101, -509.5501, 1, 1, 0, 1,
6.383838, 3.010101, -509.785, 1, 1, 0, 1,
6.424242, 3.010101, -510.0558, 1, 1, 0, 1,
6.464646, 3.010101, -510.3627, 1, 1, 0, 1,
6.505051, 3.010101, -510.7056, 1, 1, 0, 1,
6.545455, 3.010101, -511.0846, 1, 1, 0, 1,
6.585859, 3.010101, -511.4996, 1, 1, 0, 1,
6.626263, 3.010101, -511.9506, 1, 1, 0, 1,
6.666667, 3.010101, -512.4376, 1, 1, 0, 1,
6.707071, 3.010101, -512.9608, 1, 1, 0, 1,
6.747475, 3.010101, -513.5199, 1, 1, 0, 1,
6.787879, 3.010101, -514.1151, 1, 1, 0, 1,
6.828283, 3.010101, -514.7462, 1, 0.8941177, 0, 1,
6.868687, 3.010101, -515.4135, 1, 0.8941177, 0, 1,
6.909091, 3.010101, -516.1167, 1, 0.8941177, 0, 1,
6.949495, 3.010101, -516.856, 1, 0.8941177, 0, 1,
6.989899, 3.010101, -517.6313, 1, 0.8941177, 0, 1,
7.030303, 3.010101, -518.4427, 1, 0.8941177, 0, 1,
7.070707, 3.010101, -519.2901, 1, 0.8941177, 0, 1,
7.111111, 3.010101, -520.1736, 1, 0.7921569, 0, 1,
7.151515, 3.010101, -521.093, 1, 0.7921569, 0, 1,
7.191919, 3.010101, -522.0485, 1, 0.7921569, 0, 1,
7.232323, 3.010101, -523.0401, 1, 0.7921569, 0, 1,
7.272727, 3.010101, -524.0677, 1, 0.7921569, 0, 1,
7.313131, 3.010101, -525.1313, 1, 0.6862745, 0, 1,
7.353535, 3.010101, -526.2309, 1, 0.6862745, 0, 1,
7.393939, 3.010101, -527.3666, 1, 0.6862745, 0, 1,
7.434343, 3.010101, -528.5383, 1, 0.6862745, 0, 1,
7.474748, 3.010101, -529.746, 1, 0.6862745, 0, 1,
7.515152, 3.010101, -530.9898, 1, 0.5843138, 0, 1,
7.555555, 3.010101, -532.2697, 1, 0.5843138, 0, 1,
7.59596, 3.010101, -533.5855, 1, 0.5843138, 0, 1,
7.636364, 3.010101, -534.9374, 1, 0.5843138, 0, 1,
7.676768, 3.010101, -536.3253, 1, 0.4823529, 0, 1,
7.717172, 3.010101, -537.7493, 1, 0.4823529, 0, 1,
7.757576, 3.010101, -539.2092, 1, 0.4823529, 0, 1,
7.79798, 3.010101, -540.7053, 1, 0.4823529, 0, 1,
7.838384, 3.010101, -542.2373, 1, 0.3764706, 0, 1,
7.878788, 3.010101, -543.8054, 1, 0.3764706, 0, 1,
7.919192, 3.010101, -545.4095, 1, 0.3764706, 0, 1,
7.959596, 3.010101, -547.0497, 1, 0.2745098, 0, 1,
8, 3.010101, -548.7259, 1, 0.2745098, 0, 1,
4, 3.060606, -555.8924, 1, 0.1686275, 0, 1,
4.040404, 3.060606, -554.098, 1, 0.1686275, 0, 1,
4.080808, 3.060606, -552.3384, 1, 0.1686275, 0, 1,
4.121212, 3.060606, -550.6136, 1, 0.2745098, 0, 1,
4.161616, 3.060606, -548.9237, 1, 0.2745098, 0, 1,
4.20202, 3.060606, -547.2687, 1, 0.2745098, 0, 1,
4.242424, 3.060606, -545.6485, 1, 0.3764706, 0, 1,
4.282828, 3.060606, -544.0632, 1, 0.3764706, 0, 1,
4.323232, 3.060606, -542.5127, 1, 0.3764706, 0, 1,
4.363636, 3.060606, -540.9971, 1, 0.4823529, 0, 1,
4.40404, 3.060606, -539.5164, 1, 0.4823529, 0, 1,
4.444445, 3.060606, -538.0704, 1, 0.4823529, 0, 1,
4.484848, 3.060606, -536.6594, 1, 0.4823529, 0, 1,
4.525252, 3.060606, -535.2832, 1, 0.5843138, 0, 1,
4.565657, 3.060606, -533.9418, 1, 0.5843138, 0, 1,
4.606061, 3.060606, -532.6354, 1, 0.5843138, 0, 1,
4.646465, 3.060606, -531.3637, 1, 0.5843138, 0, 1,
4.686869, 3.060606, -530.127, 1, 0.6862745, 0, 1,
4.727273, 3.060606, -528.925, 1, 0.6862745, 0, 1,
4.767677, 3.060606, -527.7579, 1, 0.6862745, 0, 1,
4.808081, 3.060606, -526.6257, 1, 0.6862745, 0, 1,
4.848485, 3.060606, -525.5284, 1, 0.6862745, 0, 1,
4.888889, 3.060606, -524.4659, 1, 0.7921569, 0, 1,
4.929293, 3.060606, -523.4382, 1, 0.7921569, 0, 1,
4.969697, 3.060606, -522.4454, 1, 0.7921569, 0, 1,
5.010101, 3.060606, -521.4875, 1, 0.7921569, 0, 1,
5.050505, 3.060606, -520.5644, 1, 0.7921569, 0, 1,
5.090909, 3.060606, -519.6762, 1, 0.7921569, 0, 1,
5.131313, 3.060606, -518.8228, 1, 0.8941177, 0, 1,
5.171717, 3.060606, -518.0043, 1, 0.8941177, 0, 1,
5.212121, 3.060606, -517.2206, 1, 0.8941177, 0, 1,
5.252525, 3.060606, -516.4719, 1, 0.8941177, 0, 1,
5.292929, 3.060606, -515.7579, 1, 0.8941177, 0, 1,
5.333333, 3.060606, -515.0788, 1, 0.8941177, 0, 1,
5.373737, 3.060606, -514.4346, 1, 0.8941177, 0, 1,
5.414141, 3.060606, -513.8251, 1, 1, 0, 1,
5.454545, 3.060606, -513.2506, 1, 1, 0, 1,
5.494949, 3.060606, -512.7109, 1, 1, 0, 1,
5.535354, 3.060606, -512.2061, 1, 1, 0, 1,
5.575758, 3.060606, -511.7361, 1, 1, 0, 1,
5.616162, 3.060606, -511.3011, 1, 1, 0, 1,
5.656566, 3.060606, -510.9008, 1, 1, 0, 1,
5.69697, 3.060606, -510.5354, 1, 1, 0, 1,
5.737374, 3.060606, -510.2048, 1, 1, 0, 1,
5.777778, 3.060606, -509.9091, 1, 1, 0, 1,
5.818182, 3.060606, -509.6483, 1, 1, 0, 1,
5.858586, 3.060606, -509.4223, 1, 1, 0, 1,
5.89899, 3.060606, -509.2312, 1, 1, 0, 1,
5.939394, 3.060606, -509.0749, 1, 1, 0, 1,
5.979798, 3.060606, -508.9535, 1, 1, 0, 1,
6.020202, 3.060606, -508.8669, 1, 1, 0, 1,
6.060606, 3.060606, -508.8152, 1, 1, 0, 1,
6.10101, 3.060606, -508.7984, 1, 1, 0, 1,
6.141414, 3.060606, -508.8164, 1, 1, 0, 1,
6.181818, 3.060606, -508.8692, 1, 1, 0, 1,
6.222222, 3.060606, -508.9569, 1, 1, 0, 1,
6.262626, 3.060606, -509.0795, 1, 1, 0, 1,
6.30303, 3.060606, -509.2369, 1, 1, 0, 1,
6.343434, 3.060606, -509.4292, 1, 1, 0, 1,
6.383838, 3.060606, -509.6563, 1, 1, 0, 1,
6.424242, 3.060606, -509.9183, 1, 1, 0, 1,
6.464646, 3.060606, -510.2152, 1, 1, 0, 1,
6.505051, 3.060606, -510.5469, 1, 1, 0, 1,
6.545455, 3.060606, -510.9134, 1, 1, 0, 1,
6.585859, 3.060606, -511.3148, 1, 1, 0, 1,
6.626263, 3.060606, -511.7511, 1, 1, 0, 1,
6.666667, 3.060606, -512.2222, 1, 1, 0, 1,
6.707071, 3.060606, -512.7282, 1, 1, 0, 1,
6.747475, 3.060606, -513.269, 1, 1, 0, 1,
6.787879, 3.060606, -513.8447, 1, 1, 0, 1,
6.828283, 3.060606, -514.4552, 1, 0.8941177, 0, 1,
6.868687, 3.060606, -515.1006, 1, 0.8941177, 0, 1,
6.909091, 3.060606, -515.7809, 1, 0.8941177, 0, 1,
6.949495, 3.060606, -516.496, 1, 0.8941177, 0, 1,
6.989899, 3.060606, -517.2459, 1, 0.8941177, 0, 1,
7.030303, 3.060606, -518.0308, 1, 0.8941177, 0, 1,
7.070707, 3.060606, -518.8504, 1, 0.8941177, 0, 1,
7.111111, 3.060606, -519.7049, 1, 0.7921569, 0, 1,
7.151515, 3.060606, -520.5943, 1, 0.7921569, 0, 1,
7.191919, 3.060606, -521.5186, 1, 0.7921569, 0, 1,
7.232323, 3.060606, -522.4776, 1, 0.7921569, 0, 1,
7.272727, 3.060606, -523.4716, 1, 0.7921569, 0, 1,
7.313131, 3.060606, -524.5004, 1, 0.7921569, 0, 1,
7.353535, 3.060606, -525.564, 1, 0.6862745, 0, 1,
7.393939, 3.060606, -526.6625, 1, 0.6862745, 0, 1,
7.434343, 3.060606, -527.7959, 1, 0.6862745, 0, 1,
7.474748, 3.060606, -528.9641, 1, 0.6862745, 0, 1,
7.515152, 3.060606, -530.1672, 1, 0.6862745, 0, 1,
7.555555, 3.060606, -531.4051, 1, 0.5843138, 0, 1,
7.59596, 3.060606, -532.6779, 1, 0.5843138, 0, 1,
7.636364, 3.060606, -533.9855, 1, 0.5843138, 0, 1,
7.676768, 3.060606, -535.328, 1, 0.5843138, 0, 1,
7.717172, 3.060606, -536.7053, 1, 0.4823529, 0, 1,
7.757576, 3.060606, -538.1176, 1, 0.4823529, 0, 1,
7.79798, 3.060606, -539.5646, 1, 0.4823529, 0, 1,
7.838384, 3.060606, -541.0465, 1, 0.3764706, 0, 1,
7.878788, 3.060606, -542.5633, 1, 0.3764706, 0, 1,
7.919192, 3.060606, -544.1149, 1, 0.3764706, 0, 1,
7.959596, 3.060606, -545.7014, 1, 0.3764706, 0, 1,
8, 3.060606, -547.3227, 1, 0.2745098, 0, 1,
4, 3.111111, -554.3873, 1, 0.1686275, 0, 1,
4.040404, 3.111111, -552.6506, 1, 0.1686275, 0, 1,
4.080808, 3.111111, -550.9477, 1, 0.2745098, 0, 1,
4.121212, 3.111111, -549.2785, 1, 0.2745098, 0, 1,
4.161616, 3.111111, -547.6431, 1, 0.2745098, 0, 1,
4.20202, 3.111111, -546.0413, 1, 0.3764706, 0, 1,
4.242424, 3.111111, -544.4733, 1, 0.3764706, 0, 1,
4.282828, 3.111111, -542.939, 1, 0.3764706, 0, 1,
4.323232, 3.111111, -541.4385, 1, 0.3764706, 0, 1,
4.363636, 3.111111, -539.9717, 1, 0.4823529, 0, 1,
4.40404, 3.111111, -538.5386, 1, 0.4823529, 0, 1,
4.444445, 3.111111, -537.1393, 1, 0.4823529, 0, 1,
4.484848, 3.111111, -535.7737, 1, 0.4823529, 0, 1,
4.525252, 3.111111, -534.4418, 1, 0.5843138, 0, 1,
4.565657, 3.111111, -533.1436, 1, 0.5843138, 0, 1,
4.606061, 3.111111, -531.8792, 1, 0.5843138, 0, 1,
4.646465, 3.111111, -530.6486, 1, 0.5843138, 0, 1,
4.686869, 3.111111, -529.4516, 1, 0.6862745, 0, 1,
4.727273, 3.111111, -528.2884, 1, 0.6862745, 0, 1,
4.767677, 3.111111, -527.1589, 1, 0.6862745, 0, 1,
4.808081, 3.111111, -526.0631, 1, 0.6862745, 0, 1,
4.848485, 3.111111, -525.0011, 1, 0.6862745, 0, 1,
4.888889, 3.111111, -523.9728, 1, 0.7921569, 0, 1,
4.929293, 3.111111, -522.9783, 1, 0.7921569, 0, 1,
4.969697, 3.111111, -522.0175, 1, 0.7921569, 0, 1,
5.010101, 3.111111, -521.0904, 1, 0.7921569, 0, 1,
5.050505, 3.111111, -520.197, 1, 0.7921569, 0, 1,
5.090909, 3.111111, -519.3374, 1, 0.8941177, 0, 1,
5.131313, 3.111111, -518.5115, 1, 0.8941177, 0, 1,
5.171717, 3.111111, -517.7194, 1, 0.8941177, 0, 1,
5.212121, 3.111111, -516.9609, 1, 0.8941177, 0, 1,
5.252525, 3.111111, -516.2362, 1, 0.8941177, 0, 1,
5.292929, 3.111111, -515.5453, 1, 0.8941177, 0, 1,
5.333333, 3.111111, -514.8881, 1, 0.8941177, 0, 1,
5.373737, 3.111111, -514.2645, 1, 0.8941177, 0, 1,
5.414141, 3.111111, -513.6748, 1, 1, 0, 1,
5.454545, 3.111111, -513.1188, 1, 1, 0, 1,
5.494949, 3.111111, -512.5964, 1, 1, 0, 1,
5.535354, 3.111111, -512.1079, 1, 1, 0, 1,
5.575758, 3.111111, -511.6531, 1, 1, 0, 1,
5.616162, 3.111111, -511.232, 1, 1, 0, 1,
5.656566, 3.111111, -510.8446, 1, 1, 0, 1,
5.69697, 3.111111, -510.491, 1, 1, 0, 1,
5.737374, 3.111111, -510.1711, 1, 1, 0, 1,
5.777778, 3.111111, -509.8849, 1, 1, 0, 1,
5.818182, 3.111111, -509.6324, 1, 1, 0, 1,
5.858586, 3.111111, -509.4138, 1, 1, 0, 1,
5.89899, 3.111111, -509.2288, 1, 1, 0, 1,
5.939394, 3.111111, -509.0775, 1, 1, 0, 1,
5.979798, 3.111111, -508.96, 1, 1, 0, 1,
6.020202, 3.111111, -508.8763, 1, 1, 0, 1,
6.060606, 3.111111, -508.8262, 1, 1, 0, 1,
6.10101, 3.111111, -508.8099, 1, 1, 0, 1,
6.141414, 3.111111, -508.8273, 1, 1, 0, 1,
6.181818, 3.111111, -508.8785, 1, 1, 0, 1,
6.222222, 3.111111, -508.9633, 1, 1, 0, 1,
6.262626, 3.111111, -509.082, 1, 1, 0, 1,
6.30303, 3.111111, -509.2343, 1, 1, 0, 1,
6.343434, 3.111111, -509.4204, 1, 1, 0, 1,
6.383838, 3.111111, -509.6402, 1, 1, 0, 1,
6.424242, 3.111111, -509.8938, 1, 1, 0, 1,
6.464646, 3.111111, -510.1811, 1, 1, 0, 1,
6.505051, 3.111111, -510.5021, 1, 1, 0, 1,
6.545455, 3.111111, -510.8568, 1, 1, 0, 1,
6.585859, 3.111111, -511.2453, 1, 1, 0, 1,
6.626263, 3.111111, -511.6675, 1, 1, 0, 1,
6.666667, 3.111111, -512.1235, 1, 1, 0, 1,
6.707071, 3.111111, -512.6132, 1, 1, 0, 1,
6.747475, 3.111111, -513.1365, 1, 1, 0, 1,
6.787879, 3.111111, -513.6937, 1, 1, 0, 1,
6.828283, 3.111111, -514.2845, 1, 0.8941177, 0, 1,
6.868687, 3.111111, -514.9092, 1, 0.8941177, 0, 1,
6.909091, 3.111111, -515.5675, 1, 0.8941177, 0, 1,
6.949495, 3.111111, -516.2596, 1, 0.8941177, 0, 1,
6.989899, 3.111111, -516.9854, 1, 0.8941177, 0, 1,
7.030303, 3.111111, -517.7449, 1, 0.8941177, 0, 1,
7.070707, 3.111111, -518.5382, 1, 0.8941177, 0, 1,
7.111111, 3.111111, -519.3652, 1, 0.8941177, 0, 1,
7.151515, 3.111111, -520.226, 1, 0.7921569, 0, 1,
7.191919, 3.111111, -521.1204, 1, 0.7921569, 0, 1,
7.232323, 3.111111, -522.0486, 1, 0.7921569, 0, 1,
7.272727, 3.111111, -523.0106, 1, 0.7921569, 0, 1,
7.313131, 3.111111, -524.0062, 1, 0.7921569, 0, 1,
7.353535, 3.111111, -525.0356, 1, 0.6862745, 0, 1,
7.393939, 3.111111, -526.0988, 1, 0.6862745, 0, 1,
7.434343, 3.111111, -527.1956, 1, 0.6862745, 0, 1,
7.474748, 3.111111, -528.3262, 1, 0.6862745, 0, 1,
7.515152, 3.111111, -529.4905, 1, 0.6862745, 0, 1,
7.555555, 3.111111, -530.6886, 1, 0.5843138, 0, 1,
7.59596, 3.111111, -531.9203, 1, 0.5843138, 0, 1,
7.636364, 3.111111, -533.1859, 1, 0.5843138, 0, 1,
7.676768, 3.111111, -534.4852, 1, 0.5843138, 0, 1,
7.717172, 3.111111, -535.8181, 1, 0.4823529, 0, 1,
7.757576, 3.111111, -537.1849, 1, 0.4823529, 0, 1,
7.79798, 3.111111, -538.5853, 1, 0.4823529, 0, 1,
7.838384, 3.111111, -540.0195, 1, 0.4823529, 0, 1,
7.878788, 3.111111, -541.4874, 1, 0.3764706, 0, 1,
7.919192, 3.111111, -542.9891, 1, 0.3764706, 0, 1,
7.959596, 3.111111, -544.5245, 1, 0.3764706, 0, 1,
8, 3.111111, -546.0936, 1, 0.3764706, 0, 1,
4, 3.161616, -553.0566, 1, 0.1686275, 0, 1,
4.040404, 3.161616, -551.3751, 1, 0.2745098, 0, 1,
4.080808, 3.161616, -549.7261, 1, 0.2745098, 0, 1,
4.121212, 3.161616, -548.1098, 1, 0.2745098, 0, 1,
4.161616, 3.161616, -546.5261, 1, 0.2745098, 0, 1,
4.20202, 3.161616, -544.9752, 1, 0.3764706, 0, 1,
4.242424, 3.161616, -543.4568, 1, 0.3764706, 0, 1,
4.282828, 3.161616, -541.9712, 1, 0.3764706, 0, 1,
4.323232, 3.161616, -540.5182, 1, 0.4823529, 0, 1,
4.363636, 3.161616, -539.0979, 1, 0.4823529, 0, 1,
4.40404, 3.161616, -537.7103, 1, 0.4823529, 0, 1,
4.444445, 3.161616, -536.3553, 1, 0.4823529, 0, 1,
4.484848, 3.161616, -535.033, 1, 0.5843138, 0, 1,
4.525252, 3.161616, -533.7433, 1, 0.5843138, 0, 1,
4.565657, 3.161616, -532.4863, 1, 0.5843138, 0, 1,
4.606061, 3.161616, -531.262, 1, 0.5843138, 0, 1,
4.646465, 3.161616, -530.0703, 1, 0.6862745, 0, 1,
4.686869, 3.161616, -528.9113, 1, 0.6862745, 0, 1,
4.727273, 3.161616, -527.7849, 1, 0.6862745, 0, 1,
4.767677, 3.161616, -526.6912, 1, 0.6862745, 0, 1,
4.808081, 3.161616, -525.6302, 1, 0.6862745, 0, 1,
4.848485, 3.161616, -524.6018, 1, 0.7921569, 0, 1,
4.888889, 3.161616, -523.6061, 1, 0.7921569, 0, 1,
4.929293, 3.161616, -522.6431, 1, 0.7921569, 0, 1,
4.969697, 3.161616, -521.7127, 1, 0.7921569, 0, 1,
5.010101, 3.161616, -520.815, 1, 0.7921569, 0, 1,
5.050505, 3.161616, -519.95, 1, 0.7921569, 0, 1,
5.090909, 3.161616, -519.1176, 1, 0.8941177, 0, 1,
5.131313, 3.161616, -518.3179, 1, 0.8941177, 0, 1,
5.171717, 3.161616, -517.5508, 1, 0.8941177, 0, 1,
5.212121, 3.161616, -516.8165, 1, 0.8941177, 0, 1,
5.252525, 3.161616, -516.1147, 1, 0.8941177, 0, 1,
5.292929, 3.161616, -515.4457, 1, 0.8941177, 0, 1,
5.333333, 3.161616, -514.8093, 1, 0.8941177, 0, 1,
5.373737, 3.161616, -514.2056, 1, 0.8941177, 0, 1,
5.414141, 3.161616, -513.6345, 1, 1, 0, 1,
5.454545, 3.161616, -513.0961, 1, 1, 0, 1,
5.494949, 3.161616, -512.5903, 1, 1, 0, 1,
5.535354, 3.161616, -512.1172, 1, 1, 0, 1,
5.575758, 3.161616, -511.6768, 1, 1, 0, 1,
5.616162, 3.161616, -511.2691, 1, 1, 0, 1,
5.656566, 3.161616, -510.894, 1, 1, 0, 1,
5.69697, 3.161616, -510.5515, 1, 1, 0, 1,
5.737374, 3.161616, -510.2418, 1, 1, 0, 1,
5.777778, 3.161616, -509.9647, 1, 1, 0, 1,
5.818182, 3.161616, -509.7202, 1, 1, 0, 1,
5.858586, 3.161616, -509.5085, 1, 1, 0, 1,
5.89899, 3.161616, -509.3294, 1, 1, 0, 1,
5.939394, 3.161616, -509.1829, 1, 1, 0, 1,
5.979798, 3.161616, -509.0691, 1, 1, 0, 1,
6.020202, 3.161616, -508.988, 1, 1, 0, 1,
6.060606, 3.161616, -508.9395, 1, 1, 0, 1,
6.10101, 3.161616, -508.9238, 1, 1, 0, 1,
6.141414, 3.161616, -508.9406, 1, 1, 0, 1,
6.181818, 3.161616, -508.9902, 1, 1, 0, 1,
6.222222, 3.161616, -509.0724, 1, 1, 0, 1,
6.262626, 3.161616, -509.1872, 1, 1, 0, 1,
6.30303, 3.161616, -509.3347, 1, 1, 0, 1,
6.343434, 3.161616, -509.5149, 1, 1, 0, 1,
6.383838, 3.161616, -509.7278, 1, 1, 0, 1,
6.424242, 3.161616, -509.9733, 1, 1, 0, 1,
6.464646, 3.161616, -510.2515, 1, 1, 0, 1,
6.505051, 3.161616, -510.5623, 1, 1, 0, 1,
6.545455, 3.161616, -510.9058, 1, 1, 0, 1,
6.585859, 3.161616, -511.282, 1, 1, 0, 1,
6.626263, 3.161616, -511.6908, 1, 1, 0, 1,
6.666667, 3.161616, -512.1323, 1, 1, 0, 1,
6.707071, 3.161616, -512.6064, 1, 1, 0, 1,
6.747475, 3.161616, -513.1133, 1, 1, 0, 1,
6.787879, 3.161616, -513.6528, 1, 1, 0, 1,
6.828283, 3.161616, -514.2249, 1, 0.8941177, 0, 1,
6.868687, 3.161616, -514.8297, 1, 0.8941177, 0, 1,
6.909091, 3.161616, -515.4672, 1, 0.8941177, 0, 1,
6.949495, 3.161616, -516.1373, 1, 0.8941177, 0, 1,
6.989899, 3.161616, -516.8401, 1, 0.8941177, 0, 1,
7.030303, 3.161616, -517.5756, 1, 0.8941177, 0, 1,
7.070707, 3.161616, -518.3438, 1, 0.8941177, 0, 1,
7.111111, 3.161616, -519.1445, 1, 0.8941177, 0, 1,
7.151515, 3.161616, -519.978, 1, 0.7921569, 0, 1,
7.191919, 3.161616, -520.8441, 1, 0.7921569, 0, 1,
7.232323, 3.161616, -521.7429, 1, 0.7921569, 0, 1,
7.272727, 3.161616, -522.6743, 1, 0.7921569, 0, 1,
7.313131, 3.161616, -523.6384, 1, 0.7921569, 0, 1,
7.353535, 3.161616, -524.6352, 1, 0.7921569, 0, 1,
7.393939, 3.161616, -525.6647, 1, 0.6862745, 0, 1,
7.434343, 3.161616, -526.7267, 1, 0.6862745, 0, 1,
7.474748, 3.161616, -527.8215, 1, 0.6862745, 0, 1,
7.515152, 3.161616, -528.9489, 1, 0.6862745, 0, 1,
7.555555, 3.161616, -530.109, 1, 0.6862745, 0, 1,
7.59596, 3.161616, -531.3018, 1, 0.5843138, 0, 1,
7.636364, 3.161616, -532.5272, 1, 0.5843138, 0, 1,
7.676768, 3.161616, -533.7853, 1, 0.5843138, 0, 1,
7.717172, 3.161616, -535.076, 1, 0.5843138, 0, 1,
7.757576, 3.161616, -536.3994, 1, 0.4823529, 0, 1,
7.79798, 3.161616, -537.7555, 1, 0.4823529, 0, 1,
7.838384, 3.161616, -539.1442, 1, 0.4823529, 0, 1,
7.878788, 3.161616, -540.5656, 1, 0.4823529, 0, 1,
7.919192, 3.161616, -542.0197, 1, 0.3764706, 0, 1,
7.959596, 3.161616, -543.5064, 1, 0.3764706, 0, 1,
8, 3.161616, -545.0258, 1, 0.3764706, 0, 1,
4, 3.212121, -551.888, 1, 0.1686275, 0, 1,
4.040404, 3.212121, -550.2588, 1, 0.2745098, 0, 1,
4.080808, 3.212121, -548.6613, 1, 0.2745098, 0, 1,
4.121212, 3.212121, -547.0954, 1, 0.2745098, 0, 1,
4.161616, 3.212121, -545.5612, 1, 0.3764706, 0, 1,
4.20202, 3.212121, -544.0586, 1, 0.3764706, 0, 1,
4.242424, 3.212121, -542.5876, 1, 0.3764706, 0, 1,
4.282828, 3.212121, -541.1484, 1, 0.3764706, 0, 1,
4.323232, 3.212121, -539.7407, 1, 0.4823529, 0, 1,
4.363636, 3.212121, -538.3647, 1, 0.4823529, 0, 1,
4.40404, 3.212121, -537.0204, 1, 0.4823529, 0, 1,
4.444445, 3.212121, -535.7076, 1, 0.4823529, 0, 1,
4.484848, 3.212121, -534.4266, 1, 0.5843138, 0, 1,
4.525252, 3.212121, -533.1771, 1, 0.5843138, 0, 1,
4.565657, 3.212121, -531.9594, 1, 0.5843138, 0, 1,
4.606061, 3.212121, -530.7732, 1, 0.5843138, 0, 1,
4.646465, 3.212121, -529.6187, 1, 0.6862745, 0, 1,
4.686869, 3.212121, -528.4958, 1, 0.6862745, 0, 1,
4.727273, 3.212121, -527.4047, 1, 0.6862745, 0, 1,
4.767677, 3.212121, -526.3451, 1, 0.6862745, 0, 1,
4.808081, 3.212121, -525.3172, 1, 0.6862745, 0, 1,
4.848485, 3.212121, -524.3209, 1, 0.7921569, 0, 1,
4.888889, 3.212121, -523.3563, 1, 0.7921569, 0, 1,
4.929293, 3.212121, -522.4233, 1, 0.7921569, 0, 1,
4.969697, 3.212121, -521.522, 1, 0.7921569, 0, 1,
5.010101, 3.212121, -520.6522, 1, 0.7921569, 0, 1,
5.050505, 3.212121, -519.8142, 1, 0.7921569, 0, 1,
5.090909, 3.212121, -519.0078, 1, 0.8941177, 0, 1,
5.131313, 3.212121, -518.233, 1, 0.8941177, 0, 1,
5.171717, 3.212121, -517.4899, 1, 0.8941177, 0, 1,
5.212121, 3.212121, -516.7784, 1, 0.8941177, 0, 1,
5.252525, 3.212121, -516.0986, 1, 0.8941177, 0, 1,
5.292929, 3.212121, -515.4504, 1, 0.8941177, 0, 1,
5.333333, 3.212121, -514.8339, 1, 0.8941177, 0, 1,
5.373737, 3.212121, -514.249, 1, 0.8941177, 0, 1,
5.414141, 3.212121, -513.6957, 1, 1, 0, 1,
5.454545, 3.212121, -513.1741, 1, 1, 0, 1,
5.494949, 3.212121, -512.6841, 1, 1, 0, 1,
5.535354, 3.212121, -512.2258, 1, 1, 0, 1,
5.575758, 3.212121, -511.7991, 1, 1, 0, 1,
5.616162, 3.212121, -511.4041, 1, 1, 0, 1,
5.656566, 3.212121, -511.0407, 1, 1, 0, 1,
5.69697, 3.212121, -510.709, 1, 1, 0, 1,
5.737374, 3.212121, -510.4089, 1, 1, 0, 1,
5.777778, 3.212121, -510.1404, 1, 1, 0, 1,
5.818182, 3.212121, -509.9036, 1, 1, 0, 1,
5.858586, 3.212121, -509.6985, 1, 1, 0, 1,
5.89899, 3.212121, -509.5249, 1, 1, 0, 1,
5.939394, 3.212121, -509.3831, 1, 1, 0, 1,
5.979798, 3.212121, -509.2728, 1, 1, 0, 1,
6.020202, 3.212121, -509.1942, 1, 1, 0, 1,
6.060606, 3.212121, -509.1473, 1, 1, 0, 1,
6.10101, 3.212121, -509.132, 1, 1, 0, 1,
6.141414, 3.212121, -509.1483, 1, 1, 0, 1,
6.181818, 3.212121, -509.1963, 1, 1, 0, 1,
6.222222, 3.212121, -509.2759, 1, 1, 0, 1,
6.262626, 3.212121, -509.3872, 1, 1, 0, 1,
6.30303, 3.212121, -509.5302, 1, 1, 0, 1,
6.343434, 3.212121, -509.7047, 1, 1, 0, 1,
6.383838, 3.212121, -509.9109, 1, 1, 0, 1,
6.424242, 3.212121, -510.1488, 1, 1, 0, 1,
6.464646, 3.212121, -510.4183, 1, 1, 0, 1,
6.505051, 3.212121, -510.7194, 1, 1, 0, 1,
6.545455, 3.212121, -511.0522, 1, 1, 0, 1,
6.585859, 3.212121, -511.4166, 1, 1, 0, 1,
6.626263, 3.212121, -511.8127, 1, 1, 0, 1,
6.666667, 3.212121, -512.2404, 1, 1, 0, 1,
6.707071, 3.212121, -512.6998, 1, 1, 0, 1,
6.747475, 3.212121, -513.1908, 1, 1, 0, 1,
6.787879, 3.212121, -513.7134, 1, 1, 0, 1,
6.828283, 3.212121, -514.2678, 1, 0.8941177, 0, 1,
6.868687, 3.212121, -514.8537, 1, 0.8941177, 0, 1,
6.909091, 3.212121, -515.4713, 1, 0.8941177, 0, 1,
6.949495, 3.212121, -516.1205, 1, 0.8941177, 0, 1,
6.989899, 3.212121, -516.8014, 1, 0.8941177, 0, 1,
7.030303, 3.212121, -517.5139, 1, 0.8941177, 0, 1,
7.070707, 3.212121, -518.2581, 1, 0.8941177, 0, 1,
7.111111, 3.212121, -519.0339, 1, 0.8941177, 0, 1,
7.151515, 3.212121, -519.8413, 1, 0.7921569, 0, 1,
7.191919, 3.212121, -520.6804, 1, 0.7921569, 0, 1,
7.232323, 3.212121, -521.5511, 1, 0.7921569, 0, 1,
7.272727, 3.212121, -522.4536, 1, 0.7921569, 0, 1,
7.313131, 3.212121, -523.3876, 1, 0.7921569, 0, 1,
7.353535, 3.212121, -524.3533, 1, 0.7921569, 0, 1,
7.393939, 3.212121, -525.3506, 1, 0.6862745, 0, 1,
7.434343, 3.212121, -526.3795, 1, 0.6862745, 0, 1,
7.474748, 3.212121, -527.4401, 1, 0.6862745, 0, 1,
7.515152, 3.212121, -528.5324, 1, 0.6862745, 0, 1,
7.555555, 3.212121, -529.6562, 1, 0.6862745, 0, 1,
7.59596, 3.212121, -530.8118, 1, 0.5843138, 0, 1,
7.636364, 3.212121, -531.999, 1, 0.5843138, 0, 1,
7.676768, 3.212121, -533.2178, 1, 0.5843138, 0, 1,
7.717172, 3.212121, -534.4683, 1, 0.5843138, 0, 1,
7.757576, 3.212121, -535.7504, 1, 0.4823529, 0, 1,
7.79798, 3.212121, -537.0642, 1, 0.4823529, 0, 1,
7.838384, 3.212121, -538.4096, 1, 0.4823529, 0, 1,
7.878788, 3.212121, -539.7866, 1, 0.4823529, 0, 1,
7.919192, 3.212121, -541.1953, 1, 0.3764706, 0, 1,
7.959596, 3.212121, -542.6357, 1, 0.3764706, 0, 1,
8, 3.212121, -544.1077, 1, 0.3764706, 0, 1,
4, 3.262626, -550.8698, 1, 0.2745098, 0, 1,
4.040404, 3.262626, -549.2906, 1, 0.2745098, 0, 1,
4.080808, 3.262626, -547.7422, 1, 0.2745098, 0, 1,
4.121212, 3.262626, -546.2244, 1, 0.3764706, 0, 1,
4.161616, 3.262626, -544.7373, 1, 0.3764706, 0, 1,
4.20202, 3.262626, -543.2809, 1, 0.3764706, 0, 1,
4.242424, 3.262626, -541.8552, 1, 0.3764706, 0, 1,
4.282828, 3.262626, -540.4601, 1, 0.4823529, 0, 1,
4.323232, 3.262626, -539.0956, 1, 0.4823529, 0, 1,
4.363636, 3.262626, -537.7619, 1, 0.4823529, 0, 1,
4.40404, 3.262626, -536.4589, 1, 0.4823529, 0, 1,
4.444445, 3.262626, -535.1865, 1, 0.5843138, 0, 1,
4.484848, 3.262626, -533.9448, 1, 0.5843138, 0, 1,
4.525252, 3.262626, -532.7337, 1, 0.5843138, 0, 1,
4.565657, 3.262626, -531.5533, 1, 0.5843138, 0, 1,
4.606061, 3.262626, -530.4036, 1, 0.5843138, 0, 1,
4.646465, 3.262626, -529.2846, 1, 0.6862745, 0, 1,
4.686869, 3.262626, -528.1962, 1, 0.6862745, 0, 1,
4.727273, 3.262626, -527.1385, 1, 0.6862745, 0, 1,
4.767677, 3.262626, -526.1115, 1, 0.6862745, 0, 1,
4.808081, 3.262626, -525.1152, 1, 0.6862745, 0, 1,
4.848485, 3.262626, -524.1495, 1, 0.7921569, 0, 1,
4.888889, 3.262626, -523.2145, 1, 0.7921569, 0, 1,
4.929293, 3.262626, -522.3102, 1, 0.7921569, 0, 1,
4.969697, 3.262626, -521.4366, 1, 0.7921569, 0, 1,
5.010101, 3.262626, -520.5936, 1, 0.7921569, 0, 1,
5.050505, 3.262626, -519.7812, 1, 0.7921569, 0, 1,
5.090909, 3.262626, -518.9996, 1, 0.8941177, 0, 1,
5.131313, 3.262626, -518.2487, 1, 0.8941177, 0, 1,
5.171717, 3.262626, -517.5284, 1, 0.8941177, 0, 1,
5.212121, 3.262626, -516.8387, 1, 0.8941177, 0, 1,
5.252525, 3.262626, -516.1798, 1, 0.8941177, 0, 1,
5.292929, 3.262626, -515.5515, 1, 0.8941177, 0, 1,
5.333333, 3.262626, -514.9539, 1, 0.8941177, 0, 1,
5.373737, 3.262626, -514.387, 1, 0.8941177, 0, 1,
5.414141, 3.262626, -513.8508, 1, 1, 0, 1,
5.454545, 3.262626, -513.3452, 1, 1, 0, 1,
5.494949, 3.262626, -512.8702, 1, 1, 0, 1,
5.535354, 3.262626, -512.426, 1, 1, 0, 1,
5.575758, 3.262626, -512.0125, 1, 1, 0, 1,
5.616162, 3.262626, -511.6295, 1, 1, 0, 1,
5.656566, 3.262626, -511.2773, 1, 1, 0, 1,
5.69697, 3.262626, -510.9557, 1, 1, 0, 1,
5.737374, 3.262626, -510.6649, 1, 1, 0, 1,
5.777778, 3.262626, -510.4047, 1, 1, 0, 1,
5.818182, 3.262626, -510.1751, 1, 1, 0, 1,
5.858586, 3.262626, -509.9763, 1, 1, 0, 1,
5.89899, 3.262626, -509.8081, 1, 1, 0, 1,
5.939394, 3.262626, -509.6706, 1, 1, 0, 1,
5.979798, 3.262626, -509.5637, 1, 1, 0, 1,
6.020202, 3.262626, -509.4875, 1, 1, 0, 1,
6.060606, 3.262626, -509.442, 1, 1, 0, 1,
6.10101, 3.262626, -509.4272, 1, 1, 0, 1,
6.141414, 3.262626, -509.4431, 1, 1, 0, 1,
6.181818, 3.262626, -509.4896, 1, 1, 0, 1,
6.222222, 3.262626, -509.5667, 1, 1, 0, 1,
6.262626, 3.262626, -509.6746, 1, 1, 0, 1,
6.30303, 3.262626, -509.8131, 1, 1, 0, 1,
6.343434, 3.262626, -509.9823, 1, 1, 0, 1,
6.383838, 3.262626, -510.1822, 1, 1, 0, 1,
6.424242, 3.262626, -510.4128, 1, 1, 0, 1,
6.464646, 3.262626, -510.674, 1, 1, 0, 1,
6.505051, 3.262626, -510.9659, 1, 1, 0, 1,
6.545455, 3.262626, -511.2885, 1, 1, 0, 1,
6.585859, 3.262626, -511.6417, 1, 1, 0, 1,
6.626263, 3.262626, -512.0256, 1, 1, 0, 1,
6.666667, 3.262626, -512.4402, 1, 1, 0, 1,
6.707071, 3.262626, -512.8854, 1, 1, 0, 1,
6.747475, 3.262626, -513.3613, 1, 1, 0, 1,
6.787879, 3.262626, -513.8679, 1, 1, 0, 1,
6.828283, 3.262626, -514.4052, 1, 0.8941177, 0, 1,
6.868687, 3.262626, -514.9731, 1, 0.8941177, 0, 1,
6.909091, 3.262626, -515.5718, 1, 0.8941177, 0, 1,
6.949495, 3.262626, -516.201, 1, 0.8941177, 0, 1,
6.989899, 3.262626, -516.861, 1, 0.8941177, 0, 1,
7.030303, 3.262626, -517.5516, 1, 0.8941177, 0, 1,
7.070707, 3.262626, -518.2729, 1, 0.8941177, 0, 1,
7.111111, 3.262626, -519.0249, 1, 0.8941177, 0, 1,
7.151515, 3.262626, -519.8076, 1, 0.7921569, 0, 1,
7.191919, 3.262626, -520.6208, 1, 0.7921569, 0, 1,
7.232323, 3.262626, -521.4648, 1, 0.7921569, 0, 1,
7.272727, 3.262626, -522.3395, 1, 0.7921569, 0, 1,
7.313131, 3.262626, -523.2449, 1, 0.7921569, 0, 1,
7.353535, 3.262626, -524.1808, 1, 0.7921569, 0, 1,
7.393939, 3.262626, -525.1476, 1, 0.6862745, 0, 1,
7.434343, 3.262626, -526.1449, 1, 0.6862745, 0, 1,
7.474748, 3.262626, -527.1729, 1, 0.6862745, 0, 1,
7.515152, 3.262626, -528.2316, 1, 0.6862745, 0, 1,
7.555555, 3.262626, -529.321, 1, 0.6862745, 0, 1,
7.59596, 3.262626, -530.441, 1, 0.5843138, 0, 1,
7.636364, 3.262626, -531.5917, 1, 0.5843138, 0, 1,
7.676768, 3.262626, -532.7731, 1, 0.5843138, 0, 1,
7.717172, 3.262626, -533.9852, 1, 0.5843138, 0, 1,
7.757576, 3.262626, -535.2279, 1, 0.5843138, 0, 1,
7.79798, 3.262626, -536.5013, 1, 0.4823529, 0, 1,
7.838384, 3.262626, -537.8054, 1, 0.4823529, 0, 1,
7.878788, 3.262626, -539.1401, 1, 0.4823529, 0, 1,
7.919192, 3.262626, -540.5056, 1, 0.4823529, 0, 1,
7.959596, 3.262626, -541.9017, 1, 0.3764706, 0, 1,
8, 3.262626, -543.3284, 1, 0.3764706, 0, 1,
4, 3.313131, -549.9914, 1, 0.2745098, 0, 1,
4.040404, 3.313131, -548.46, 1, 0.2745098, 0, 1,
4.080808, 3.313131, -546.9584, 1, 0.2745098, 0, 1,
4.121212, 3.313131, -545.4866, 1, 0.3764706, 0, 1,
4.161616, 3.313131, -544.0445, 1, 0.3764706, 0, 1,
4.20202, 3.313131, -542.6321, 1, 0.3764706, 0, 1,
4.242424, 3.313131, -541.2495, 1, 0.3764706, 0, 1,
4.282828, 3.313131, -539.8967, 1, 0.4823529, 0, 1,
4.323232, 3.313131, -538.5735, 1, 0.4823529, 0, 1,
4.363636, 3.313131, -537.2802, 1, 0.4823529, 0, 1,
4.40404, 3.313131, -536.0165, 1, 0.4823529, 0, 1,
4.444445, 3.313131, -534.7826, 1, 0.5843138, 0, 1,
4.484848, 3.313131, -533.5784, 1, 0.5843138, 0, 1,
4.525252, 3.313131, -532.4041, 1, 0.5843138, 0, 1,
4.565657, 3.313131, -531.2594, 1, 0.5843138, 0, 1,
4.606061, 3.313131, -530.1445, 1, 0.6862745, 0, 1,
4.646465, 3.313131, -529.0593, 1, 0.6862745, 0, 1,
4.686869, 3.313131, -528.0038, 1, 0.6862745, 0, 1,
4.727273, 3.313131, -526.9782, 1, 0.6862745, 0, 1,
4.767677, 3.313131, -525.9822, 1, 0.6862745, 0, 1,
4.808081, 3.313131, -525.0161, 1, 0.6862745, 0, 1,
4.848485, 3.313131, -524.0796, 1, 0.7921569, 0, 1,
4.888889, 3.313131, -523.1729, 1, 0.7921569, 0, 1,
4.929293, 3.313131, -522.2959, 1, 0.7921569, 0, 1,
4.969697, 3.313131, -521.4487, 1, 0.7921569, 0, 1,
5.010101, 3.313131, -520.6312, 1, 0.7921569, 0, 1,
5.050505, 3.313131, -519.8435, 1, 0.7921569, 0, 1,
5.090909, 3.313131, -519.0855, 1, 0.8941177, 0, 1,
5.131313, 3.313131, -518.3573, 1, 0.8941177, 0, 1,
5.171717, 3.313131, -517.6588, 1, 0.8941177, 0, 1,
5.212121, 3.313131, -516.99, 1, 0.8941177, 0, 1,
5.252525, 3.313131, -516.351, 1, 0.8941177, 0, 1,
5.292929, 3.313131, -515.7418, 1, 0.8941177, 0, 1,
5.333333, 3.313131, -515.1622, 1, 0.8941177, 0, 1,
5.373737, 3.313131, -514.6124, 1, 0.8941177, 0, 1,
5.414141, 3.313131, -514.0924, 1, 1, 0, 1,
5.454545, 3.313131, -513.6021, 1, 1, 0, 1,
5.494949, 3.313131, -513.1416, 1, 1, 0, 1,
5.535354, 3.313131, -512.7108, 1, 1, 0, 1,
5.575758, 3.313131, -512.3097, 1, 1, 0, 1,
5.616162, 3.313131, -511.9384, 1, 1, 0, 1,
5.656566, 3.313131, -511.5968, 1, 1, 0, 1,
5.69697, 3.313131, -511.285, 1, 1, 0, 1,
5.737374, 3.313131, -511.0029, 1, 1, 0, 1,
5.777778, 3.313131, -510.7506, 1, 1, 0, 1,
5.818182, 3.313131, -510.528, 1, 1, 0, 1,
5.858586, 3.313131, -510.3352, 1, 1, 0, 1,
5.89899, 3.313131, -510.1721, 1, 1, 0, 1,
5.939394, 3.313131, -510.0387, 1, 1, 0, 1,
5.979798, 3.313131, -509.9351, 1, 1, 0, 1,
6.020202, 3.313131, -509.8612, 1, 1, 0, 1,
6.060606, 3.313131, -509.8171, 1, 1, 0, 1,
6.10101, 3.313131, -509.8027, 1, 1, 0, 1,
6.141414, 3.313131, -509.8181, 1, 1, 0, 1,
6.181818, 3.313131, -509.8632, 1, 1, 0, 1,
6.222222, 3.313131, -509.938, 1, 1, 0, 1,
6.262626, 3.313131, -510.0426, 1, 1, 0, 1,
6.30303, 3.313131, -510.177, 1, 1, 0, 1,
6.343434, 3.313131, -510.341, 1, 1, 0, 1,
6.383838, 3.313131, -510.5349, 1, 1, 0, 1,
6.424242, 3.313131, -510.7585, 1, 1, 0, 1,
6.464646, 3.313131, -511.0117, 1, 1, 0, 1,
6.505051, 3.313131, -511.2948, 1, 1, 0, 1,
6.545455, 3.313131, -511.6076, 1, 1, 0, 1,
6.585859, 3.313131, -511.9502, 1, 1, 0, 1,
6.626263, 3.313131, -512.3224, 1, 1, 0, 1,
6.666667, 3.313131, -512.7245, 1, 1, 0, 1,
6.707071, 3.313131, -513.1563, 1, 1, 0, 1,
6.747475, 3.313131, -513.6178, 1, 1, 0, 1,
6.787879, 3.313131, -514.1091, 1, 1, 0, 1,
6.828283, 3.313131, -514.6301, 1, 0.8941177, 0, 1,
6.868687, 3.313131, -515.1808, 1, 0.8941177, 0, 1,
6.909091, 3.313131, -515.7614, 1, 0.8941177, 0, 1,
6.949495, 3.313131, -516.3716, 1, 0.8941177, 0, 1,
6.989899, 3.313131, -517.0116, 1, 0.8941177, 0, 1,
7.030303, 3.313131, -517.6813, 1, 0.8941177, 0, 1,
7.070707, 3.313131, -518.3808, 1, 0.8941177, 0, 1,
7.111111, 3.313131, -519.11, 1, 0.8941177, 0, 1,
7.151515, 3.313131, -519.869, 1, 0.7921569, 0, 1,
7.191919, 3.313131, -520.6577, 1, 0.7921569, 0, 1,
7.232323, 3.313131, -521.4761, 1, 0.7921569, 0, 1,
7.272727, 3.313131, -522.3243, 1, 0.7921569, 0, 1,
7.313131, 3.313131, -523.2023, 1, 0.7921569, 0, 1,
7.353535, 3.313131, -524.11, 1, 0.7921569, 0, 1,
7.393939, 3.313131, -525.0474, 1, 0.6862745, 0, 1,
7.434343, 3.313131, -526.0146, 1, 0.6862745, 0, 1,
7.474748, 3.313131, -527.0115, 1, 0.6862745, 0, 1,
7.515152, 3.313131, -528.0382, 1, 0.6862745, 0, 1,
7.555555, 3.313131, -529.0946, 1, 0.6862745, 0, 1,
7.59596, 3.313131, -530.1808, 1, 0.6862745, 0, 1,
7.636364, 3.313131, -531.2966, 1, 0.5843138, 0, 1,
7.676768, 3.313131, -532.4423, 1, 0.5843138, 0, 1,
7.717172, 3.313131, -533.6177, 1, 0.5843138, 0, 1,
7.757576, 3.313131, -534.8228, 1, 0.5843138, 0, 1,
7.79798, 3.313131, -536.0577, 1, 0.4823529, 0, 1,
7.838384, 3.313131, -537.3223, 1, 0.4823529, 0, 1,
7.878788, 3.313131, -538.6167, 1, 0.4823529, 0, 1,
7.919192, 3.313131, -539.9408, 1, 0.4823529, 0, 1,
7.959596, 3.313131, -541.2946, 1, 0.3764706, 0, 1,
8, 3.313131, -542.6782, 1, 0.3764706, 0, 1,
4, 3.363636, -549.2432, 1, 0.2745098, 0, 1,
4.040404, 3.363636, -547.7574, 1, 0.2745098, 0, 1,
4.080808, 3.363636, -546.3006, 1, 0.3764706, 0, 1,
4.121212, 3.363636, -544.8726, 1, 0.3764706, 0, 1,
4.161616, 3.363636, -543.4735, 1, 0.3764706, 0, 1,
4.20202, 3.363636, -542.1033, 1, 0.3764706, 0, 1,
4.242424, 3.363636, -540.7618, 1, 0.4823529, 0, 1,
4.282828, 3.363636, -539.4493, 1, 0.4823529, 0, 1,
4.323232, 3.363636, -538.1656, 1, 0.4823529, 0, 1,
4.363636, 3.363636, -536.9108, 1, 0.4823529, 0, 1,
4.40404, 3.363636, -535.6848, 1, 0.4823529, 0, 1,
4.444445, 3.363636, -534.4877, 1, 0.5843138, 0, 1,
4.484848, 3.363636, -533.3194, 1, 0.5843138, 0, 1,
4.525252, 3.363636, -532.18, 1, 0.5843138, 0, 1,
4.565657, 3.363636, -531.0695, 1, 0.5843138, 0, 1,
4.606061, 3.363636, -529.9878, 1, 0.6862745, 0, 1,
4.646465, 3.363636, -528.9349, 1, 0.6862745, 0, 1,
4.686869, 3.363636, -527.9109, 1, 0.6862745, 0, 1,
4.727273, 3.363636, -526.9158, 1, 0.6862745, 0, 1,
4.767677, 3.363636, -525.9496, 1, 0.6862745, 0, 1,
4.808081, 3.363636, -525.0122, 1, 0.6862745, 0, 1,
4.848485, 3.363636, -524.1036, 1, 0.7921569, 0, 1,
4.888889, 3.363636, -523.224, 1, 0.7921569, 0, 1,
4.929293, 3.363636, -522.3732, 1, 0.7921569, 0, 1,
4.969697, 3.363636, -521.5512, 1, 0.7921569, 0, 1,
5.010101, 3.363636, -520.7581, 1, 0.7921569, 0, 1,
5.050505, 3.363636, -519.9938, 1, 0.7921569, 0, 1,
5.090909, 3.363636, -519.2584, 1, 0.8941177, 0, 1,
5.131313, 3.363636, -518.5519, 1, 0.8941177, 0, 1,
5.171717, 3.363636, -517.8742, 1, 0.8941177, 0, 1,
5.212121, 3.363636, -517.2254, 1, 0.8941177, 0, 1,
5.252525, 3.363636, -516.6054, 1, 0.8941177, 0, 1,
5.292929, 3.363636, -516.0143, 1, 0.8941177, 0, 1,
5.333333, 3.363636, -515.4521, 1, 0.8941177, 0, 1,
5.373737, 3.363636, -514.9187, 1, 0.8941177, 0, 1,
5.414141, 3.363636, -514.4141, 1, 0.8941177, 0, 1,
5.454545, 3.363636, -513.9385, 1, 1, 0, 1,
5.494949, 3.363636, -513.4916, 1, 1, 0, 1,
5.535354, 3.363636, -513.0737, 1, 1, 0, 1,
5.575758, 3.363636, -512.6846, 1, 1, 0, 1,
5.616162, 3.363636, -512.3243, 1, 1, 0, 1,
5.656566, 3.363636, -511.993, 1, 1, 0, 1,
5.69697, 3.363636, -511.6904, 1, 1, 0, 1,
5.737374, 3.363636, -511.4167, 1, 1, 0, 1,
5.777778, 3.363636, -511.1719, 1, 1, 0, 1,
5.818182, 3.363636, -510.956, 1, 1, 0, 1,
5.858586, 3.363636, -510.7689, 1, 1, 0, 1,
5.89899, 3.363636, -510.6106, 1, 1, 0, 1,
5.939394, 3.363636, -510.4813, 1, 1, 0, 1,
5.979798, 3.363636, -510.3807, 1, 1, 0, 1,
6.020202, 3.363636, -510.3091, 1, 1, 0, 1,
6.060606, 3.363636, -510.2662, 1, 1, 0, 1,
6.10101, 3.363636, -510.2523, 1, 1, 0, 1,
6.141414, 3.363636, -510.2672, 1, 1, 0, 1,
6.181818, 3.363636, -510.3109, 1, 1, 0, 1,
6.222222, 3.363636, -510.3836, 1, 1, 0, 1,
6.262626, 3.363636, -510.485, 1, 1, 0, 1,
6.30303, 3.363636, -510.6154, 1, 1, 0, 1,
6.343434, 3.363636, -510.7746, 1, 1, 0, 1,
6.383838, 3.363636, -510.9626, 1, 1, 0, 1,
6.424242, 3.363636, -511.1795, 1, 1, 0, 1,
6.464646, 3.363636, -511.4253, 1, 1, 0, 1,
6.505051, 3.363636, -511.6999, 1, 1, 0, 1,
6.545455, 3.363636, -512.0034, 1, 1, 0, 1,
6.585859, 3.363636, -512.3358, 1, 1, 0, 1,
6.626263, 3.363636, -512.697, 1, 1, 0, 1,
6.666667, 3.363636, -513.087, 1, 1, 0, 1,
6.707071, 3.363636, -513.5059, 1, 1, 0, 1,
6.747475, 3.363636, -513.9537, 1, 1, 0, 1,
6.787879, 3.363636, -514.4303, 1, 0.8941177, 0, 1,
6.828283, 3.363636, -514.9358, 1, 0.8941177, 0, 1,
6.868687, 3.363636, -515.4702, 1, 0.8941177, 0, 1,
6.909091, 3.363636, -516.0333, 1, 0.8941177, 0, 1,
6.949495, 3.363636, -516.6254, 1, 0.8941177, 0, 1,
6.989899, 3.363636, -517.2463, 1, 0.8941177, 0, 1,
7.030303, 3.363636, -517.8961, 1, 0.8941177, 0, 1,
7.070707, 3.363636, -518.5747, 1, 0.8941177, 0, 1,
7.111111, 3.363636, -519.2822, 1, 0.8941177, 0, 1,
7.151515, 3.363636, -520.0186, 1, 0.7921569, 0, 1,
7.191919, 3.363636, -520.7838, 1, 0.7921569, 0, 1,
7.232323, 3.363636, -521.5778, 1, 0.7921569, 0, 1,
7.272727, 3.363636, -522.4008, 1, 0.7921569, 0, 1,
7.313131, 3.363636, -523.2525, 1, 0.7921569, 0, 1,
7.353535, 3.363636, -524.1332, 1, 0.7921569, 0, 1,
7.393939, 3.363636, -525.0427, 1, 0.6862745, 0, 1,
7.434343, 3.363636, -525.981, 1, 0.6862745, 0, 1,
7.474748, 3.363636, -526.9482, 1, 0.6862745, 0, 1,
7.515152, 3.363636, -527.9443, 1, 0.6862745, 0, 1,
7.555555, 3.363636, -528.9692, 1, 0.6862745, 0, 1,
7.59596, 3.363636, -530.0229, 1, 0.6862745, 0, 1,
7.636364, 3.363636, -531.1056, 1, 0.5843138, 0, 1,
7.676768, 3.363636, -532.2171, 1, 0.5843138, 0, 1,
7.717172, 3.363636, -533.3575, 1, 0.5843138, 0, 1,
7.757576, 3.363636, -534.5267, 1, 0.5843138, 0, 1,
7.79798, 3.363636, -535.7247, 1, 0.4823529, 0, 1,
7.838384, 3.363636, -536.9517, 1, 0.4823529, 0, 1,
7.878788, 3.363636, -538.2075, 1, 0.4823529, 0, 1,
7.919192, 3.363636, -539.4921, 1, 0.4823529, 0, 1,
7.959596, 3.363636, -540.8056, 1, 0.4823529, 0, 1,
8, 3.363636, -542.1479, 1, 0.3764706, 0, 1,
4, 3.414141, -548.6161, 1, 0.2745098, 0, 1,
4.040404, 3.414141, -547.174, 1, 0.2745098, 0, 1,
4.080808, 3.414141, -545.7599, 1, 0.3764706, 0, 1,
4.121212, 3.414141, -544.3739, 1, 0.3764706, 0, 1,
4.161616, 3.414141, -543.0159, 1, 0.3764706, 0, 1,
4.20202, 3.414141, -541.6859, 1, 0.3764706, 0, 1,
4.242424, 3.414141, -540.3839, 1, 0.4823529, 0, 1,
4.282828, 3.414141, -539.1098, 1, 0.4823529, 0, 1,
4.323232, 3.414141, -537.8638, 1, 0.4823529, 0, 1,
4.363636, 3.414141, -536.6459, 1, 0.4823529, 0, 1,
4.40404, 3.414141, -535.4559, 1, 0.5843138, 0, 1,
4.444445, 3.414141, -534.2939, 1, 0.5843138, 0, 1,
4.484848, 3.414141, -533.16, 1, 0.5843138, 0, 1,
4.525252, 3.414141, -532.054, 1, 0.5843138, 0, 1,
4.565657, 3.414141, -530.9761, 1, 0.5843138, 0, 1,
4.606061, 3.414141, -529.9261, 1, 0.6862745, 0, 1,
4.646465, 3.414141, -528.9042, 1, 0.6862745, 0, 1,
4.686869, 3.414141, -527.9103, 1, 0.6862745, 0, 1,
4.727273, 3.414141, -526.9445, 1, 0.6862745, 0, 1,
4.767677, 3.414141, -526.0066, 1, 0.6862745, 0, 1,
4.808081, 3.414141, -525.0967, 1, 0.6862745, 0, 1,
4.848485, 3.414141, -524.2148, 1, 0.7921569, 0, 1,
4.888889, 3.414141, -523.361, 1, 0.7921569, 0, 1,
4.929293, 3.414141, -522.5352, 1, 0.7921569, 0, 1,
4.969697, 3.414141, -521.7373, 1, 0.7921569, 0, 1,
5.010101, 3.414141, -520.9675, 1, 0.7921569, 0, 1,
5.050505, 3.414141, -520.2257, 1, 0.7921569, 0, 1,
5.090909, 3.414141, -519.5119, 1, 0.8941177, 0, 1,
5.131313, 3.414141, -518.8261, 1, 0.8941177, 0, 1,
5.171717, 3.414141, -518.1683, 1, 0.8941177, 0, 1,
5.212121, 3.414141, -517.5386, 1, 0.8941177, 0, 1,
5.252525, 3.414141, -516.9368, 1, 0.8941177, 0, 1,
5.292929, 3.414141, -516.363, 1, 0.8941177, 0, 1,
5.333333, 3.414141, -515.8173, 1, 0.8941177, 0, 1,
5.373737, 3.414141, -515.2996, 1, 0.8941177, 0, 1,
5.414141, 3.414141, -514.8099, 1, 0.8941177, 0, 1,
5.454545, 3.414141, -514.3482, 1, 0.8941177, 0, 1,
5.494949, 3.414141, -513.9145, 1, 1, 0, 1,
5.535354, 3.414141, -513.5088, 1, 1, 0, 1,
5.575758, 3.414141, -513.1311, 1, 1, 0, 1,
5.616162, 3.414141, -512.7814, 1, 1, 0, 1,
5.656566, 3.414141, -512.4598, 1, 1, 0, 1,
5.69697, 3.414141, -512.1661, 1, 1, 0, 1,
5.737374, 3.414141, -511.9005, 1, 1, 0, 1,
5.777778, 3.414141, -511.6629, 1, 1, 0, 1,
5.818182, 3.414141, -511.4533, 1, 1, 0, 1,
5.858586, 3.414141, -511.2717, 1, 1, 0, 1,
5.89899, 3.414141, -511.1181, 1, 1, 0, 1,
5.939394, 3.414141, -510.9925, 1, 1, 0, 1,
5.979798, 3.414141, -510.8949, 1, 1, 0, 1,
6.020202, 3.414141, -510.8253, 1, 1, 0, 1,
6.060606, 3.414141, -510.7838, 1, 1, 0, 1,
6.10101, 3.414141, -510.7703, 1, 1, 0, 1,
6.141414, 3.414141, -510.7847, 1, 1, 0, 1,
6.181818, 3.414141, -510.8272, 1, 1, 0, 1,
6.222222, 3.414141, -510.8977, 1, 1, 0, 1,
6.262626, 3.414141, -510.9962, 1, 1, 0, 1,
6.30303, 3.414141, -511.1227, 1, 1, 0, 1,
6.343434, 3.414141, -511.2772, 1, 1, 0, 1,
6.383838, 3.414141, -511.4597, 1, 1, 0, 1,
6.424242, 3.414141, -511.6703, 1, 1, 0, 1,
6.464646, 3.414141, -511.9088, 1, 1, 0, 1,
6.505051, 3.414141, -512.1754, 1, 1, 0, 1,
6.545455, 3.414141, -512.47, 1, 1, 0, 1,
6.585859, 3.414141, -512.7925, 1, 1, 0, 1,
6.626263, 3.414141, -513.1431, 1, 1, 0, 1,
6.666667, 3.414141, -513.5217, 1, 1, 0, 1,
6.707071, 3.414141, -513.9283, 1, 1, 0, 1,
6.747475, 3.414141, -514.363, 1, 0.8941177, 0, 1,
6.787879, 3.414141, -514.8256, 1, 0.8941177, 0, 1,
6.828283, 3.414141, -515.3162, 1, 0.8941177, 0, 1,
6.868687, 3.414141, -515.8349, 1, 0.8941177, 0, 1,
6.909091, 3.414141, -516.3815, 1, 0.8941177, 0, 1,
6.949495, 3.414141, -516.9562, 1, 0.8941177, 0, 1,
6.989899, 3.414141, -517.5589, 1, 0.8941177, 0, 1,
7.030303, 3.414141, -518.1896, 1, 0.8941177, 0, 1,
7.070707, 3.414141, -518.8483, 1, 0.8941177, 0, 1,
7.111111, 3.414141, -519.535, 1, 0.8941177, 0, 1,
7.151515, 3.414141, -520.2497, 1, 0.7921569, 0, 1,
7.191919, 3.414141, -520.9924, 1, 0.7921569, 0, 1,
7.232323, 3.414141, -521.7632, 1, 0.7921569, 0, 1,
7.272727, 3.414141, -522.562, 1, 0.7921569, 0, 1,
7.313131, 3.414141, -523.3887, 1, 0.7921569, 0, 1,
7.353535, 3.414141, -524.2435, 1, 0.7921569, 0, 1,
7.393939, 3.414141, -525.1263, 1, 0.6862745, 0, 1,
7.434343, 3.414141, -526.037, 1, 0.6862745, 0, 1,
7.474748, 3.414141, -526.9759, 1, 0.6862745, 0, 1,
7.515152, 3.414141, -527.9427, 1, 0.6862745, 0, 1,
7.555555, 3.414141, -528.9375, 1, 0.6862745, 0, 1,
7.59596, 3.414141, -529.9603, 1, 0.6862745, 0, 1,
7.636364, 3.414141, -531.0112, 1, 0.5843138, 0, 1,
7.676768, 3.414141, -532.09, 1, 0.5843138, 0, 1,
7.717172, 3.414141, -533.1969, 1, 0.5843138, 0, 1,
7.757576, 3.414141, -534.3318, 1, 0.5843138, 0, 1,
7.79798, 3.414141, -535.4947, 1, 0.5843138, 0, 1,
7.838384, 3.414141, -536.6855, 1, 0.4823529, 0, 1,
7.878788, 3.414141, -537.9045, 1, 0.4823529, 0, 1,
7.919192, 3.414141, -539.1514, 1, 0.4823529, 0, 1,
7.959596, 3.414141, -540.4263, 1, 0.4823529, 0, 1,
8, 3.414141, -541.7292, 1, 0.3764706, 0, 1,
4, 3.464647, -548.1019, 1, 0.2745098, 0, 1,
4.040404, 3.464647, -546.7015, 1, 0.2745098, 0, 1,
4.080808, 3.464647, -545.3284, 1, 0.3764706, 0, 1,
4.121212, 3.464647, -543.9824, 1, 0.3764706, 0, 1,
4.161616, 3.464647, -542.6637, 1, 0.3764706, 0, 1,
4.20202, 3.464647, -541.3722, 1, 0.3764706, 0, 1,
4.242424, 3.464647, -540.1078, 1, 0.4823529, 0, 1,
4.282828, 3.464647, -538.8707, 1, 0.4823529, 0, 1,
4.323232, 3.464647, -537.6608, 1, 0.4823529, 0, 1,
4.363636, 3.464647, -536.4781, 1, 0.4823529, 0, 1,
4.40404, 3.464647, -535.3225, 1, 0.5843138, 0, 1,
4.444445, 3.464647, -534.1942, 1, 0.5843138, 0, 1,
4.484848, 3.464647, -533.0931, 1, 0.5843138, 0, 1,
4.525252, 3.464647, -532.0191, 1, 0.5843138, 0, 1,
4.565657, 3.464647, -530.9724, 1, 0.5843138, 0, 1,
4.606061, 3.464647, -529.9529, 1, 0.6862745, 0, 1,
4.646465, 3.464647, -528.9605, 1, 0.6862745, 0, 1,
4.686869, 3.464647, -527.9954, 1, 0.6862745, 0, 1,
4.727273, 3.464647, -527.0574, 1, 0.6862745, 0, 1,
4.767677, 3.464647, -526.1467, 1, 0.6862745, 0, 1,
4.808081, 3.464647, -525.2632, 1, 0.6862745, 0, 1,
4.848485, 3.464647, -524.4069, 1, 0.7921569, 0, 1,
4.888889, 3.464647, -523.5777, 1, 0.7921569, 0, 1,
4.929293, 3.464647, -522.7758, 1, 0.7921569, 0, 1,
4.969697, 3.464647, -522.001, 1, 0.7921569, 0, 1,
5.010101, 3.464647, -521.2535, 1, 0.7921569, 0, 1,
5.050505, 3.464647, -520.5331, 1, 0.7921569, 0, 1,
5.090909, 3.464647, -519.84, 1, 0.7921569, 0, 1,
5.131313, 3.464647, -519.1741, 1, 0.8941177, 0, 1,
5.171717, 3.464647, -518.5353, 1, 0.8941177, 0, 1,
5.212121, 3.464647, -517.9238, 1, 0.8941177, 0, 1,
5.252525, 3.464647, -517.3394, 1, 0.8941177, 0, 1,
5.292929, 3.464647, -516.7823, 1, 0.8941177, 0, 1,
5.333333, 3.464647, -516.2524, 1, 0.8941177, 0, 1,
5.373737, 3.464647, -515.7496, 1, 0.8941177, 0, 1,
5.414141, 3.464647, -515.274, 1, 0.8941177, 0, 1,
5.454545, 3.464647, -514.8257, 1, 0.8941177, 0, 1,
5.494949, 3.464647, -514.4046, 1, 0.8941177, 0, 1,
5.535354, 3.464647, -514.0106, 1, 1, 0, 1,
5.575758, 3.464647, -513.6439, 1, 1, 0, 1,
5.616162, 3.464647, -513.3043, 1, 1, 0, 1,
5.656566, 3.464647, -512.992, 1, 1, 0, 1,
5.69697, 3.464647, -512.7068, 1, 1, 0, 1,
5.737374, 3.464647, -512.4489, 1, 1, 0, 1,
5.777778, 3.464647, -512.2181, 1, 1, 0, 1,
5.818182, 3.464647, -512.0146, 1, 1, 0, 1,
5.858586, 3.464647, -511.8383, 1, 1, 0, 1,
5.89899, 3.464647, -511.6891, 1, 1, 0, 1,
5.939394, 3.464647, -511.5672, 1, 1, 0, 1,
5.979798, 3.464647, -511.4724, 1, 1, 0, 1,
6.020202, 3.464647, -511.4049, 1, 1, 0, 1,
6.060606, 3.464647, -511.3645, 1, 1, 0, 1,
6.10101, 3.464647, -511.3513, 1, 1, 0, 1,
6.141414, 3.464647, -511.3654, 1, 1, 0, 1,
6.181818, 3.464647, -511.4066, 1, 1, 0, 1,
6.222222, 3.464647, -511.4751, 1, 1, 0, 1,
6.262626, 3.464647, -511.5707, 1, 1, 0, 1,
6.30303, 3.464647, -511.6936, 1, 1, 0, 1,
6.343434, 3.464647, -511.8437, 1, 1, 0, 1,
6.383838, 3.464647, -512.0209, 1, 1, 0, 1,
6.424242, 3.464647, -512.2253, 1, 1, 0, 1,
6.464646, 3.464647, -512.457, 1, 1, 0, 1,
6.505051, 3.464647, -512.7158, 1, 1, 0, 1,
6.545455, 3.464647, -513.0019, 1, 1, 0, 1,
6.585859, 3.464647, -513.3151, 1, 1, 0, 1,
6.626263, 3.464647, -513.6556, 1, 1, 0, 1,
6.666667, 3.464647, -514.0232, 1, 1, 0, 1,
6.707071, 3.464647, -514.418, 1, 0.8941177, 0, 1,
6.747475, 3.464647, -514.8401, 1, 0.8941177, 0, 1,
6.787879, 3.464647, -515.2893, 1, 0.8941177, 0, 1,
6.828283, 3.464647, -515.7657, 1, 0.8941177, 0, 1,
6.868687, 3.464647, -516.2694, 1, 0.8941177, 0, 1,
6.909091, 3.464647, -516.8002, 1, 0.8941177, 0, 1,
6.949495, 3.464647, -517.3583, 1, 0.8941177, 0, 1,
6.989899, 3.464647, -517.9435, 1, 0.8941177, 0, 1,
7.030303, 3.464647, -518.556, 1, 0.8941177, 0, 1,
7.070707, 3.464647, -519.1956, 1, 0.8941177, 0, 1,
7.111111, 3.464647, -519.8624, 1, 0.7921569, 0, 1,
7.151515, 3.464647, -520.5565, 1, 0.7921569, 0, 1,
7.191919, 3.464647, -521.2777, 1, 0.7921569, 0, 1,
7.232323, 3.464647, -522.0261, 1, 0.7921569, 0, 1,
7.272727, 3.464647, -522.8018, 1, 0.7921569, 0, 1,
7.313131, 3.464647, -523.6046, 1, 0.7921569, 0, 1,
7.353535, 3.464647, -524.4346, 1, 0.7921569, 0, 1,
7.393939, 3.464647, -525.2919, 1, 0.6862745, 0, 1,
7.434343, 3.464647, -526.1763, 1, 0.6862745, 0, 1,
7.474748, 3.464647, -527.088, 1, 0.6862745, 0, 1,
7.515152, 3.464647, -528.0268, 1, 0.6862745, 0, 1,
7.555555, 3.464647, -528.9928, 1, 0.6862745, 0, 1,
7.59596, 3.464647, -529.986, 1, 0.6862745, 0, 1,
7.636364, 3.464647, -531.0065, 1, 0.5843138, 0, 1,
7.676768, 3.464647, -532.0541, 1, 0.5843138, 0, 1,
7.717172, 3.464647, -533.129, 1, 0.5843138, 0, 1,
7.757576, 3.464647, -534.231, 1, 0.5843138, 0, 1,
7.79798, 3.464647, -535.3602, 1, 0.5843138, 0, 1,
7.838384, 3.464647, -536.5166, 1, 0.4823529, 0, 1,
7.878788, 3.464647, -537.7003, 1, 0.4823529, 0, 1,
7.919192, 3.464647, -538.9111, 1, 0.4823529, 0, 1,
7.959596, 3.464647, -540.1491, 1, 0.4823529, 0, 1,
8, 3.464647, -541.4144, 1, 0.3764706, 0, 1,
4, 3.515152, -547.6928, 1, 0.2745098, 0, 1,
4.040404, 3.515152, -546.3324, 1, 0.3764706, 0, 1,
4.080808, 3.515152, -544.9985, 1, 0.3764706, 0, 1,
4.121212, 3.515152, -543.6909, 1, 0.3764706, 0, 1,
4.161616, 3.515152, -542.4098, 1, 0.3764706, 0, 1,
4.20202, 3.515152, -541.1552, 1, 0.3764706, 0, 1,
4.242424, 3.515152, -539.9269, 1, 0.4823529, 0, 1,
4.282828, 3.515152, -538.725, 1, 0.4823529, 0, 1,
4.323232, 3.515152, -537.5496, 1, 0.4823529, 0, 1,
4.363636, 3.515152, -536.4006, 1, 0.4823529, 0, 1,
4.40404, 3.515152, -535.2781, 1, 0.5843138, 0, 1,
4.444445, 3.515152, -534.1819, 1, 0.5843138, 0, 1,
4.484848, 3.515152, -533.1122, 1, 0.5843138, 0, 1,
4.525252, 3.515152, -532.0689, 1, 0.5843138, 0, 1,
4.565657, 3.515152, -531.0521, 1, 0.5843138, 0, 1,
4.606061, 3.515152, -530.0616, 1, 0.6862745, 0, 1,
4.646465, 3.515152, -529.0976, 1, 0.6862745, 0, 1,
4.686869, 3.515152, -528.16, 1, 0.6862745, 0, 1,
4.727273, 3.515152, -527.2488, 1, 0.6862745, 0, 1,
4.767677, 3.515152, -526.364, 1, 0.6862745, 0, 1,
4.808081, 3.515152, -525.5057, 1, 0.6862745, 0, 1,
4.848485, 3.515152, -524.6738, 1, 0.7921569, 0, 1,
4.888889, 3.515152, -523.8683, 1, 0.7921569, 0, 1,
4.929293, 3.515152, -523.0892, 1, 0.7921569, 0, 1,
4.969697, 3.515152, -522.3366, 1, 0.7921569, 0, 1,
5.010101, 3.515152, -521.6104, 1, 0.7921569, 0, 1,
5.050505, 3.515152, -520.9106, 1, 0.7921569, 0, 1,
5.090909, 3.515152, -520.2372, 1, 0.7921569, 0, 1,
5.131313, 3.515152, -519.5903, 1, 0.7921569, 0, 1,
5.171717, 3.515152, -518.9698, 1, 0.8941177, 0, 1,
5.212121, 3.515152, -518.3757, 1, 0.8941177, 0, 1,
5.252525, 3.515152, -517.808, 1, 0.8941177, 0, 1,
5.292929, 3.515152, -517.2668, 1, 0.8941177, 0, 1,
5.333333, 3.515152, -516.752, 1, 0.8941177, 0, 1,
5.373737, 3.515152, -516.2635, 1, 0.8941177, 0, 1,
5.414141, 3.515152, -515.8016, 1, 0.8941177, 0, 1,
5.454545, 3.515152, -515.366, 1, 0.8941177, 0, 1,
5.494949, 3.515152, -514.9569, 1, 0.8941177, 0, 1,
5.535354, 3.515152, -514.5742, 1, 0.8941177, 0, 1,
5.575758, 3.515152, -514.2179, 1, 0.8941177, 0, 1,
5.616162, 3.515152, -513.8881, 1, 1, 0, 1,
5.656566, 3.515152, -513.5846, 1, 1, 0, 1,
5.69697, 3.515152, -513.3076, 1, 1, 0, 1,
5.737374, 3.515152, -513.057, 1, 1, 0, 1,
5.777778, 3.515152, -512.8328, 1, 1, 0, 1,
5.818182, 3.515152, -512.6351, 1, 1, 0, 1,
5.858586, 3.515152, -512.4638, 1, 1, 0, 1,
5.89899, 3.515152, -512.3189, 1, 1, 0, 1,
5.939394, 3.515152, -512.2004, 1, 1, 0, 1,
5.979798, 3.515152, -512.1084, 1, 1, 0, 1,
6.020202, 3.515152, -512.0427, 1, 1, 0, 1,
6.060606, 3.515152, -512.0035, 1, 1, 0, 1,
6.10101, 3.515152, -511.9908, 1, 1, 0, 1,
6.141414, 3.515152, -512.0044, 1, 1, 0, 1,
6.181818, 3.515152, -512.0445, 1, 1, 0, 1,
6.222222, 3.515152, -512.111, 1, 1, 0, 1,
6.262626, 3.515152, -512.2039, 1, 1, 0, 1,
6.30303, 3.515152, -512.3232, 1, 1, 0, 1,
6.343434, 3.515152, -512.469, 1, 1, 0, 1,
6.383838, 3.515152, -512.6412, 1, 1, 0, 1,
6.424242, 3.515152, -512.8398, 1, 1, 0, 1,
6.464646, 3.515152, -513.0649, 1, 1, 0, 1,
6.505051, 3.515152, -513.3163, 1, 1, 0, 1,
6.545455, 3.515152, -513.5942, 1, 1, 0, 1,
6.585859, 3.515152, -513.8985, 1, 1, 0, 1,
6.626263, 3.515152, -514.2292, 1, 0.8941177, 0, 1,
6.666667, 3.515152, -514.5864, 1, 0.8941177, 0, 1,
6.707071, 3.515152, -514.97, 1, 0.8941177, 0, 1,
6.747475, 3.515152, -515.3799, 1, 0.8941177, 0, 1,
6.787879, 3.515152, -515.8164, 1, 0.8941177, 0, 1,
6.828283, 3.515152, -516.2792, 1, 0.8941177, 0, 1,
6.868687, 3.515152, -516.7685, 1, 0.8941177, 0, 1,
6.909091, 3.515152, -517.2842, 1, 0.8941177, 0, 1,
6.949495, 3.515152, -517.8263, 1, 0.8941177, 0, 1,
6.989899, 3.515152, -518.3948, 1, 0.8941177, 0, 1,
7.030303, 3.515152, -518.9898, 1, 0.8941177, 0, 1,
7.070707, 3.515152, -519.6112, 1, 0.7921569, 0, 1,
7.111111, 3.515152, -520.259, 1, 0.7921569, 0, 1,
7.151515, 3.515152, -520.9333, 1, 0.7921569, 0, 1,
7.191919, 3.515152, -521.6339, 1, 0.7921569, 0, 1,
7.232323, 3.515152, -522.361, 1, 0.7921569, 0, 1,
7.272727, 3.515152, -523.1145, 1, 0.7921569, 0, 1,
7.313131, 3.515152, -523.8945, 1, 0.7921569, 0, 1,
7.353535, 3.515152, -524.7008, 1, 0.7921569, 0, 1,
7.393939, 3.515152, -525.5336, 1, 0.6862745, 0, 1,
7.434343, 3.515152, -526.3928, 1, 0.6862745, 0, 1,
7.474748, 3.515152, -527.2784, 1, 0.6862745, 0, 1,
7.515152, 3.515152, -528.1904, 1, 0.6862745, 0, 1,
7.555555, 3.515152, -529.1289, 1, 0.6862745, 0, 1,
7.59596, 3.515152, -530.0938, 1, 0.6862745, 0, 1,
7.636364, 3.515152, -531.0851, 1, 0.5843138, 0, 1,
7.676768, 3.515152, -532.1029, 1, 0.5843138, 0, 1,
7.717172, 3.515152, -533.147, 1, 0.5843138, 0, 1,
7.757576, 3.515152, -534.2177, 1, 0.5843138, 0, 1,
7.79798, 3.515152, -535.3146, 1, 0.5843138, 0, 1,
7.838384, 3.515152, -536.4381, 1, 0.4823529, 0, 1,
7.878788, 3.515152, -537.588, 1, 0.4823529, 0, 1,
7.919192, 3.515152, -538.7642, 1, 0.4823529, 0, 1,
7.959596, 3.515152, -539.967, 1, 0.4823529, 0, 1,
8, 3.515152, -541.1961, 1, 0.3764706, 0, 1,
4, 3.565657, -547.3818, 1, 0.2745098, 0, 1,
4.040404, 3.565657, -546.0597, 1, 0.3764706, 0, 1,
4.080808, 3.565657, -544.7633, 1, 0.3764706, 0, 1,
4.121212, 3.565657, -543.4926, 1, 0.3764706, 0, 1,
4.161616, 3.565657, -542.2474, 1, 0.3764706, 0, 1,
4.20202, 3.565657, -541.0281, 1, 0.3764706, 0, 1,
4.242424, 3.565657, -539.8344, 1, 0.4823529, 0, 1,
4.282828, 3.565657, -538.6663, 1, 0.4823529, 0, 1,
4.323232, 3.565657, -537.524, 1, 0.4823529, 0, 1,
4.363636, 3.565657, -536.4073, 1, 0.4823529, 0, 1,
4.40404, 3.565657, -535.3163, 1, 0.5843138, 0, 1,
4.444445, 3.565657, -534.251, 1, 0.5843138, 0, 1,
4.484848, 3.565657, -533.2114, 1, 0.5843138, 0, 1,
4.525252, 3.565657, -532.1974, 1, 0.5843138, 0, 1,
4.565657, 3.565657, -531.2092, 1, 0.5843138, 0, 1,
4.606061, 3.565657, -530.2466, 1, 0.6862745, 0, 1,
4.646465, 3.565657, -529.3096, 1, 0.6862745, 0, 1,
4.686869, 3.565657, -528.3984, 1, 0.6862745, 0, 1,
4.727273, 3.565657, -527.5129, 1, 0.6862745, 0, 1,
4.767677, 3.565657, -526.653, 1, 0.6862745, 0, 1,
4.808081, 3.565657, -525.8188, 1, 0.6862745, 0, 1,
4.848485, 3.565657, -525.0103, 1, 0.6862745, 0, 1,
4.888889, 3.565657, -524.2275, 1, 0.7921569, 0, 1,
4.929293, 3.565657, -523.4703, 1, 0.7921569, 0, 1,
4.969697, 3.565657, -522.7389, 1, 0.7921569, 0, 1,
5.010101, 3.565657, -522.0331, 1, 0.7921569, 0, 1,
5.050505, 3.565657, -521.353, 1, 0.7921569, 0, 1,
5.090909, 3.565657, -520.6985, 1, 0.7921569, 0, 1,
5.131313, 3.565657, -520.0698, 1, 0.7921569, 0, 1,
5.171717, 3.565657, -519.4667, 1, 0.8941177, 0, 1,
5.212121, 3.565657, -518.8893, 1, 0.8941177, 0, 1,
5.252525, 3.565657, -518.3376, 1, 0.8941177, 0, 1,
5.292929, 3.565657, -517.8116, 1, 0.8941177, 0, 1,
5.333333, 3.565657, -517.3113, 1, 0.8941177, 0, 1,
5.373737, 3.565657, -516.8366, 1, 0.8941177, 0, 1,
5.414141, 3.565657, -516.3876, 1, 0.8941177, 0, 1,
5.454545, 3.565657, -515.9644, 1, 0.8941177, 0, 1,
5.494949, 3.565657, -515.5667, 1, 0.8941177, 0, 1,
5.535354, 3.565657, -515.1948, 1, 0.8941177, 0, 1,
5.575758, 3.565657, -514.8485, 1, 0.8941177, 0, 1,
5.616162, 3.565657, -514.528, 1, 0.8941177, 0, 1,
5.656566, 3.565657, -514.233, 1, 0.8941177, 0, 1,
5.69697, 3.565657, -513.9638, 1, 1, 0, 1,
5.737374, 3.565657, -513.7203, 1, 1, 0, 1,
5.777778, 3.565657, -513.5024, 1, 1, 0, 1,
5.818182, 3.565657, -513.3102, 1, 1, 0, 1,
5.858586, 3.565657, -513.1437, 1, 1, 0, 1,
5.89899, 3.565657, -513.0029, 1, 1, 0, 1,
5.939394, 3.565657, -512.8878, 1, 1, 0, 1,
5.979798, 3.565657, -512.7983, 1, 1, 0, 1,
6.020202, 3.565657, -512.7346, 1, 1, 0, 1,
6.060606, 3.565657, -512.6965, 1, 1, 0, 1,
6.10101, 3.565657, -512.684, 1, 1, 0, 1,
6.141414, 3.565657, -512.6973, 1, 1, 0, 1,
6.181818, 3.565657, -512.7363, 1, 1, 0, 1,
6.222222, 3.565657, -512.8009, 1, 1, 0, 1,
6.262626, 3.565657, -512.8912, 1, 1, 0, 1,
6.30303, 3.565657, -513.0071, 1, 1, 0, 1,
6.343434, 3.565657, -513.1488, 1, 1, 0, 1,
6.383838, 3.565657, -513.3162, 1, 1, 0, 1,
6.424242, 3.565657, -513.5092, 1, 1, 0, 1,
6.464646, 3.565657, -513.7279, 1, 1, 0, 1,
6.505051, 3.565657, -513.9723, 1, 1, 0, 1,
6.545455, 3.565657, -514.2424, 1, 0.8941177, 0, 1,
6.585859, 3.565657, -514.5381, 1, 0.8941177, 0, 1,
6.626263, 3.565657, -514.8596, 1, 0.8941177, 0, 1,
6.666667, 3.565657, -515.2067, 1, 0.8941177, 0, 1,
6.707071, 3.565657, -515.5794, 1, 0.8941177, 0, 1,
6.747475, 3.565657, -515.9779, 1, 0.8941177, 0, 1,
6.787879, 3.565657, -516.402, 1, 0.8941177, 0, 1,
6.828283, 3.565657, -516.8519, 1, 0.8941177, 0, 1,
6.868687, 3.565657, -517.3274, 1, 0.8941177, 0, 1,
6.909091, 3.565657, -517.8286, 1, 0.8941177, 0, 1,
6.949495, 3.565657, -518.3555, 1, 0.8941177, 0, 1,
6.989899, 3.565657, -518.908, 1, 0.8941177, 0, 1,
7.030303, 3.565657, -519.4862, 1, 0.8941177, 0, 1,
7.070707, 3.565657, -520.0901, 1, 0.7921569, 0, 1,
7.111111, 3.565657, -520.7197, 1, 0.7921569, 0, 1,
7.151515, 3.565657, -521.375, 1, 0.7921569, 0, 1,
7.191919, 3.565657, -522.056, 1, 0.7921569, 0, 1,
7.232323, 3.565657, -522.7626, 1, 0.7921569, 0, 1,
7.272727, 3.565657, -523.4949, 1, 0.7921569, 0, 1,
7.313131, 3.565657, -524.2529, 1, 0.7921569, 0, 1,
7.353535, 3.565657, -525.0366, 1, 0.6862745, 0, 1,
7.393939, 3.565657, -525.8459, 1, 0.6862745, 0, 1,
7.434343, 3.565657, -526.681, 1, 0.6862745, 0, 1,
7.474748, 3.565657, -527.5417, 1, 0.6862745, 0, 1,
7.515152, 3.565657, -528.428, 1, 0.6862745, 0, 1,
7.555555, 3.565657, -529.3401, 1, 0.6862745, 0, 1,
7.59596, 3.565657, -530.2779, 1, 0.6862745, 0, 1,
7.636364, 3.565657, -531.2413, 1, 0.5843138, 0, 1,
7.676768, 3.565657, -532.2305, 1, 0.5843138, 0, 1,
7.717172, 3.565657, -533.2452, 1, 0.5843138, 0, 1,
7.757576, 3.565657, -534.2857, 1, 0.5843138, 0, 1,
7.79798, 3.565657, -535.3519, 1, 0.5843138, 0, 1,
7.838384, 3.565657, -536.4437, 1, 0.4823529, 0, 1,
7.878788, 3.565657, -537.5612, 1, 0.4823529, 0, 1,
7.919192, 3.565657, -538.7044, 1, 0.4823529, 0, 1,
7.959596, 3.565657, -539.8733, 1, 0.4823529, 0, 1,
8, 3.565657, -541.0679, 1, 0.3764706, 0, 1,
4, 3.616162, -547.1624, 1, 0.2745098, 0, 1,
4.040404, 3.616162, -545.877, 1, 0.3764706, 0, 1,
4.080808, 3.616162, -544.6165, 1, 0.3764706, 0, 1,
4.121212, 3.616162, -543.381, 1, 0.3764706, 0, 1,
4.161616, 3.616162, -542.1704, 1, 0.3764706, 0, 1,
4.20202, 3.616162, -540.9849, 1, 0.4823529, 0, 1,
4.242424, 3.616162, -539.8243, 1, 0.4823529, 0, 1,
4.282828, 3.616162, -538.6886, 1, 0.4823529, 0, 1,
4.323232, 3.616162, -537.5779, 1, 0.4823529, 0, 1,
4.363636, 3.616162, -536.4922, 1, 0.4823529, 0, 1,
4.40404, 3.616162, -535.4315, 1, 0.5843138, 0, 1,
4.444445, 3.616162, -534.3958, 1, 0.5843138, 0, 1,
4.484848, 3.616162, -533.385, 1, 0.5843138, 0, 1,
4.525252, 3.616162, -532.3992, 1, 0.5843138, 0, 1,
4.565657, 3.616162, -531.4383, 1, 0.5843138, 0, 1,
4.606061, 3.616162, -530.5024, 1, 0.5843138, 0, 1,
4.646465, 3.616162, -529.5915, 1, 0.6862745, 0, 1,
4.686869, 3.616162, -528.7056, 1, 0.6862745, 0, 1,
4.727273, 3.616162, -527.8445, 1, 0.6862745, 0, 1,
4.767677, 3.616162, -527.0085, 1, 0.6862745, 0, 1,
4.808081, 3.616162, -526.1975, 1, 0.6862745, 0, 1,
4.848485, 3.616162, -525.4114, 1, 0.6862745, 0, 1,
4.888889, 3.616162, -524.6503, 1, 0.7921569, 0, 1,
4.929293, 3.616162, -523.9141, 1, 0.7921569, 0, 1,
4.969697, 3.616162, -523.203, 1, 0.7921569, 0, 1,
5.010101, 3.616162, -522.5168, 1, 0.7921569, 0, 1,
5.050505, 3.616162, -521.8555, 1, 0.7921569, 0, 1,
5.090909, 3.616162, -521.2192, 1, 0.7921569, 0, 1,
5.131313, 3.616162, -520.608, 1, 0.7921569, 0, 1,
5.171717, 3.616162, -520.0216, 1, 0.7921569, 0, 1,
5.212121, 3.616162, -519.4603, 1, 0.8941177, 0, 1,
5.252525, 3.616162, -518.9238, 1, 0.8941177, 0, 1,
5.292929, 3.616162, -518.4124, 1, 0.8941177, 0, 1,
5.333333, 3.616162, -517.926, 1, 0.8941177, 0, 1,
5.373737, 3.616162, -517.4645, 1, 0.8941177, 0, 1,
5.414141, 3.616162, -517.0279, 1, 0.8941177, 0, 1,
5.454545, 3.616162, -516.6163, 1, 0.8941177, 0, 1,
5.494949, 3.616162, -516.2297, 1, 0.8941177, 0, 1,
5.535354, 3.616162, -515.8682, 1, 0.8941177, 0, 1,
5.575758, 3.616162, -515.5315, 1, 0.8941177, 0, 1,
5.616162, 3.616162, -515.2198, 1, 0.8941177, 0, 1,
5.656566, 3.616162, -514.9331, 1, 0.8941177, 0, 1,
5.69697, 3.616162, -514.6713, 1, 0.8941177, 0, 1,
5.737374, 3.616162, -514.4345, 1, 0.8941177, 0, 1,
5.777778, 3.616162, -514.2227, 1, 0.8941177, 0, 1,
5.818182, 3.616162, -514.0359, 1, 1, 0, 1,
5.858586, 3.616162, -513.874, 1, 1, 0, 1,
5.89899, 3.616162, -513.7371, 1, 1, 0, 1,
5.939394, 3.616162, -513.6251, 1, 1, 0, 1,
5.979798, 3.616162, -513.5381, 1, 1, 0, 1,
6.020202, 3.616162, -513.4761, 1, 1, 0, 1,
6.060606, 3.616162, -513.4391, 1, 1, 0, 1,
6.10101, 3.616162, -513.4271, 1, 1, 0, 1,
6.141414, 3.616162, -513.4399, 1, 1, 0, 1,
6.181818, 3.616162, -513.4778, 1, 1, 0, 1,
6.222222, 3.616162, -513.5406, 1, 1, 0, 1,
6.262626, 3.616162, -513.6284, 1, 1, 0, 1,
6.30303, 3.616162, -513.7412, 1, 1, 0, 1,
6.343434, 3.616162, -513.8789, 1, 1, 0, 1,
6.383838, 3.616162, -514.0416, 1, 1, 0, 1,
6.424242, 3.616162, -514.2293, 1, 0.8941177, 0, 1,
6.464646, 3.616162, -514.442, 1, 0.8941177, 0, 1,
6.505051, 3.616162, -514.6796, 1, 0.8941177, 0, 1,
6.545455, 3.616162, -514.9421, 1, 0.8941177, 0, 1,
6.585859, 3.616162, -515.2297, 1, 0.8941177, 0, 1,
6.626263, 3.616162, -515.5422, 1, 0.8941177, 0, 1,
6.666667, 3.616162, -515.8796, 1, 0.8941177, 0, 1,
6.707071, 3.616162, -516.2421, 1, 0.8941177, 0, 1,
6.747475, 3.616162, -516.6295, 1, 0.8941177, 0, 1,
6.787879, 3.616162, -517.0419, 1, 0.8941177, 0, 1,
6.828283, 3.616162, -517.4792, 1, 0.8941177, 0, 1,
6.868687, 3.616162, -517.9416, 1, 0.8941177, 0, 1,
6.909091, 3.616162, -518.4289, 1, 0.8941177, 0, 1,
6.949495, 3.616162, -518.9411, 1, 0.8941177, 0, 1,
6.989899, 3.616162, -519.4783, 1, 0.8941177, 0, 1,
7.030303, 3.616162, -520.0405, 1, 0.7921569, 0, 1,
7.070707, 3.616162, -520.6277, 1, 0.7921569, 0, 1,
7.111111, 3.616162, -521.2398, 1, 0.7921569, 0, 1,
7.151515, 3.616162, -521.877, 1, 0.7921569, 0, 1,
7.191919, 3.616162, -522.539, 1, 0.7921569, 0, 1,
7.232323, 3.616162, -523.226, 1, 0.7921569, 0, 1,
7.272727, 3.616162, -523.938, 1, 0.7921569, 0, 1,
7.313131, 3.616162, -524.675, 1, 0.7921569, 0, 1,
7.353535, 3.616162, -525.437, 1, 0.6862745, 0, 1,
7.393939, 3.616162, -526.2238, 1, 0.6862745, 0, 1,
7.434343, 3.616162, -527.0357, 1, 0.6862745, 0, 1,
7.474748, 3.616162, -527.8726, 1, 0.6862745, 0, 1,
7.515152, 3.616162, -528.7344, 1, 0.6862745, 0, 1,
7.555555, 3.616162, -529.6212, 1, 0.6862745, 0, 1,
7.59596, 3.616162, -530.5329, 1, 0.5843138, 0, 1,
7.636364, 3.616162, -531.4696, 1, 0.5843138, 0, 1,
7.676768, 3.616162, -532.4313, 1, 0.5843138, 0, 1,
7.717172, 3.616162, -533.4179, 1, 0.5843138, 0, 1,
7.757576, 3.616162, -534.4295, 1, 0.5843138, 0, 1,
7.79798, 3.616162, -535.4661, 1, 0.5843138, 0, 1,
7.838384, 3.616162, -536.5276, 1, 0.4823529, 0, 1,
7.878788, 3.616162, -537.6142, 1, 0.4823529, 0, 1,
7.919192, 3.616162, -538.7256, 1, 0.4823529, 0, 1,
7.959596, 3.616162, -539.8621, 1, 0.4823529, 0, 1,
8, 3.616162, -541.0236, 1, 0.3764706, 0, 1,
4, 3.666667, -547.0284, 1, 0.2745098, 0, 1,
4.040404, 3.666667, -545.7781, 1, 0.3764706, 0, 1,
4.080808, 3.666667, -544.5521, 1, 0.3764706, 0, 1,
4.121212, 3.666667, -543.3504, 1, 0.3764706, 0, 1,
4.161616, 3.666667, -542.173, 1, 0.3764706, 0, 1,
4.20202, 3.666667, -541.0198, 1, 0.3764706, 0, 1,
4.242424, 3.666667, -539.891, 1, 0.4823529, 0, 1,
4.282828, 3.666667, -538.7864, 1, 0.4823529, 0, 1,
4.323232, 3.666667, -537.7062, 1, 0.4823529, 0, 1,
4.363636, 3.666667, -536.6501, 1, 0.4823529, 0, 1,
4.40404, 3.666667, -535.6185, 1, 0.5843138, 0, 1,
4.444445, 3.666667, -534.611, 1, 0.5843138, 0, 1,
4.484848, 3.666667, -533.6279, 1, 0.5843138, 0, 1,
4.525252, 3.666667, -532.6691, 1, 0.5843138, 0, 1,
4.565657, 3.666667, -531.7345, 1, 0.5843138, 0, 1,
4.606061, 3.666667, -530.8242, 1, 0.5843138, 0, 1,
4.646465, 3.666667, -529.9382, 1, 0.6862745, 0, 1,
4.686869, 3.666667, -529.0765, 1, 0.6862745, 0, 1,
4.727273, 3.666667, -528.239, 1, 0.6862745, 0, 1,
4.767677, 3.666667, -527.4259, 1, 0.6862745, 0, 1,
4.808081, 3.666667, -526.637, 1, 0.6862745, 0, 1,
4.848485, 3.666667, -525.8724, 1, 0.6862745, 0, 1,
4.888889, 3.666667, -525.1322, 1, 0.6862745, 0, 1,
4.929293, 3.666667, -524.4161, 1, 0.7921569, 0, 1,
4.969697, 3.666667, -523.7244, 1, 0.7921569, 0, 1,
5.010101, 3.666667, -523.057, 1, 0.7921569, 0, 1,
5.050505, 3.666667, -522.4139, 1, 0.7921569, 0, 1,
5.090909, 3.666667, -521.795, 1, 0.7921569, 0, 1,
5.131313, 3.666667, -521.2004, 1, 0.7921569, 0, 1,
5.171717, 3.666667, -520.6301, 1, 0.7921569, 0, 1,
5.212121, 3.666667, -520.0841, 1, 0.7921569, 0, 1,
5.252525, 3.666667, -519.5624, 1, 0.7921569, 0, 1,
5.292929, 3.666667, -519.0649, 1, 0.8941177, 0, 1,
5.333333, 3.666667, -518.5918, 1, 0.8941177, 0, 1,
5.373737, 3.666667, -518.1429, 1, 0.8941177, 0, 1,
5.414141, 3.666667, -517.7183, 1, 0.8941177, 0, 1,
5.454545, 3.666667, -517.3181, 1, 0.8941177, 0, 1,
5.494949, 3.666667, -516.942, 1, 0.8941177, 0, 1,
5.535354, 3.666667, -516.5903, 1, 0.8941177, 0, 1,
5.575758, 3.666667, -516.2628, 1, 0.8941177, 0, 1,
5.616162, 3.666667, -515.9597, 1, 0.8941177, 0, 1,
5.656566, 3.666667, -515.6808, 1, 0.8941177, 0, 1,
5.69697, 3.666667, -515.4262, 1, 0.8941177, 0, 1,
5.737374, 3.666667, -515.1959, 1, 0.8941177, 0, 1,
5.777778, 3.666667, -514.9899, 1, 0.8941177, 0, 1,
5.818182, 3.666667, -514.8082, 1, 0.8941177, 0, 1,
5.858586, 3.666667, -514.6507, 1, 0.8941177, 0, 1,
5.89899, 3.666667, -514.5175, 1, 0.8941177, 0, 1,
5.939394, 3.666667, -514.4086, 1, 0.8941177, 0, 1,
5.979798, 3.666667, -514.324, 1, 0.8941177, 0, 1,
6.020202, 3.666667, -514.2637, 1, 0.8941177, 0, 1,
6.060606, 3.666667, -514.2277, 1, 0.8941177, 0, 1,
6.10101, 3.666667, -514.2159, 1, 0.8941177, 0, 1,
6.141414, 3.666667, -514.2285, 1, 0.8941177, 0, 1,
6.181818, 3.666667, -514.2653, 1, 0.8941177, 0, 1,
6.222222, 3.666667, -514.3265, 1, 0.8941177, 0, 1,
6.262626, 3.666667, -514.4119, 1, 0.8941177, 0, 1,
6.30303, 3.666667, -514.5215, 1, 0.8941177, 0, 1,
6.343434, 3.666667, -514.6555, 1, 0.8941177, 0, 1,
6.383838, 3.666667, -514.8137, 1, 0.8941177, 0, 1,
6.424242, 3.666667, -514.9963, 1, 0.8941177, 0, 1,
6.464646, 3.666667, -515.2031, 1, 0.8941177, 0, 1,
6.505051, 3.666667, -515.4342, 1, 0.8941177, 0, 1,
6.545455, 3.666667, -515.6896, 1, 0.8941177, 0, 1,
6.585859, 3.666667, -515.9693, 1, 0.8941177, 0, 1,
6.626263, 3.666667, -516.2733, 1, 0.8941177, 0, 1,
6.666667, 3.666667, -516.6015, 1, 0.8941177, 0, 1,
6.707071, 3.666667, -516.954, 1, 0.8941177, 0, 1,
6.747475, 3.666667, -517.3309, 1, 0.8941177, 0, 1,
6.787879, 3.666667, -517.7319, 1, 0.8941177, 0, 1,
6.828283, 3.666667, -518.1573, 1, 0.8941177, 0, 1,
6.868687, 3.666667, -518.607, 1, 0.8941177, 0, 1,
6.909091, 3.666667, -519.0809, 1, 0.8941177, 0, 1,
6.949495, 3.666667, -519.5792, 1, 0.7921569, 0, 1,
6.989899, 3.666667, -520.1017, 1, 0.7921569, 0, 1,
7.030303, 3.666667, -520.6486, 1, 0.7921569, 0, 1,
7.070707, 3.666667, -521.2196, 1, 0.7921569, 0, 1,
7.111111, 3.666667, -521.815, 1, 0.7921569, 0, 1,
7.151515, 3.666667, -522.4347, 1, 0.7921569, 0, 1,
7.191919, 3.666667, -523.0786, 1, 0.7921569, 0, 1,
7.232323, 3.666667, -523.7469, 1, 0.7921569, 0, 1,
7.272727, 3.666667, -524.4394, 1, 0.7921569, 0, 1,
7.313131, 3.666667, -525.1562, 1, 0.6862745, 0, 1,
7.353535, 3.666667, -525.8973, 1, 0.6862745, 0, 1,
7.393939, 3.666667, -526.6627, 1, 0.6862745, 0, 1,
7.434343, 3.666667, -527.4523, 1, 0.6862745, 0, 1,
7.474748, 3.666667, -528.2662, 1, 0.6862745, 0, 1,
7.515152, 3.666667, -529.1045, 1, 0.6862745, 0, 1,
7.555555, 3.666667, -529.967, 1, 0.6862745, 0, 1,
7.59596, 3.666667, -530.8538, 1, 0.5843138, 0, 1,
7.636364, 3.666667, -531.7649, 1, 0.5843138, 0, 1,
7.676768, 3.666667, -532.7003, 1, 0.5843138, 0, 1,
7.717172, 3.666667, -533.6599, 1, 0.5843138, 0, 1,
7.757576, 3.666667, -534.6439, 1, 0.5843138, 0, 1,
7.79798, 3.666667, -535.6521, 1, 0.4823529, 0, 1,
7.838384, 3.666667, -536.6846, 1, 0.4823529, 0, 1,
7.878788, 3.666667, -537.7414, 1, 0.4823529, 0, 1,
7.919192, 3.666667, -538.8224, 1, 0.4823529, 0, 1,
7.959596, 3.666667, -539.9279, 1, 0.4823529, 0, 1,
8, 3.666667, -541.0575, 1, 0.3764706, 0, 1,
4, 3.717172, -546.9741, 1, 0.2745098, 0, 1,
4.040404, 3.717172, -545.7576, 1, 0.3764706, 0, 1,
4.080808, 3.717172, -544.5647, 1, 0.3764706, 0, 1,
4.121212, 3.717172, -543.3954, 1, 0.3764706, 0, 1,
4.161616, 3.717172, -542.2498, 1, 0.3764706, 0, 1,
4.20202, 3.717172, -541.1277, 1, 0.3764706, 0, 1,
4.242424, 3.717172, -540.0294, 1, 0.4823529, 0, 1,
4.282828, 3.717172, -538.9547, 1, 0.4823529, 0, 1,
4.323232, 3.717172, -537.9035, 1, 0.4823529, 0, 1,
4.363636, 3.717172, -536.876, 1, 0.4823529, 0, 1,
4.40404, 3.717172, -535.8721, 1, 0.4823529, 0, 1,
4.444445, 3.717172, -534.8919, 1, 0.5843138, 0, 1,
4.484848, 3.717172, -533.9353, 1, 0.5843138, 0, 1,
4.525252, 3.717172, -533.0023, 1, 0.5843138, 0, 1,
4.565657, 3.717172, -532.093, 1, 0.5843138, 0, 1,
4.606061, 3.717172, -531.2073, 1, 0.5843138, 0, 1,
4.646465, 3.717172, -530.3452, 1, 0.5843138, 0, 1,
4.686869, 3.717172, -529.5067, 1, 0.6862745, 0, 1,
4.727273, 3.717172, -528.6919, 1, 0.6862745, 0, 1,
4.767677, 3.717172, -527.9007, 1, 0.6862745, 0, 1,
4.808081, 3.717172, -527.1331, 1, 0.6862745, 0, 1,
4.848485, 3.717172, -526.3892, 1, 0.6862745, 0, 1,
4.888889, 3.717172, -525.6689, 1, 0.6862745, 0, 1,
4.929293, 3.717172, -524.9722, 1, 0.6862745, 0, 1,
4.969697, 3.717172, -524.2991, 1, 0.7921569, 0, 1,
5.010101, 3.717172, -523.6497, 1, 0.7921569, 0, 1,
5.050505, 3.717172, -523.0239, 1, 0.7921569, 0, 1,
5.090909, 3.717172, -522.4218, 1, 0.7921569, 0, 1,
5.131313, 3.717172, -521.8433, 1, 0.7921569, 0, 1,
5.171717, 3.717172, -521.2883, 1, 0.7921569, 0, 1,
5.212121, 3.717172, -520.7571, 1, 0.7921569, 0, 1,
5.252525, 3.717172, -520.2495, 1, 0.7921569, 0, 1,
5.292929, 3.717172, -519.7654, 1, 0.7921569, 0, 1,
5.333333, 3.717172, -519.3051, 1, 0.8941177, 0, 1,
5.373737, 3.717172, -518.8683, 1, 0.8941177, 0, 1,
5.414141, 3.717172, -518.4551, 1, 0.8941177, 0, 1,
5.454545, 3.717172, -518.0657, 1, 0.8941177, 0, 1,
5.494949, 3.717172, -517.6998, 1, 0.8941177, 0, 1,
5.535354, 3.717172, -517.3575, 1, 0.8941177, 0, 1,
5.575758, 3.717172, -517.0389, 1, 0.8941177, 0, 1,
5.616162, 3.717172, -516.744, 1, 0.8941177, 0, 1,
5.656566, 3.717172, -516.4726, 1, 0.8941177, 0, 1,
5.69697, 3.717172, -516.2249, 1, 0.8941177, 0, 1,
5.737374, 3.717172, -516.0008, 1, 0.8941177, 0, 1,
5.777778, 3.717172, -515.8004, 1, 0.8941177, 0, 1,
5.818182, 3.717172, -515.6235, 1, 0.8941177, 0, 1,
5.858586, 3.717172, -515.4703, 1, 0.8941177, 0, 1,
5.89899, 3.717172, -515.3408, 1, 0.8941177, 0, 1,
5.939394, 3.717172, -515.2348, 1, 0.8941177, 0, 1,
5.979798, 3.717172, -515.1525, 1, 0.8941177, 0, 1,
6.020202, 3.717172, -515.0938, 1, 0.8941177, 0, 1,
6.060606, 3.717172, -515.0587, 1, 0.8941177, 0, 1,
6.10101, 3.717172, -515.0473, 1, 0.8941177, 0, 1,
6.141414, 3.717172, -515.0595, 1, 0.8941177, 0, 1,
6.181818, 3.717172, -515.0953, 1, 0.8941177, 0, 1,
6.222222, 3.717172, -515.1548, 1, 0.8941177, 0, 1,
6.262626, 3.717172, -515.2379, 1, 0.8941177, 0, 1,
6.30303, 3.717172, -515.3446, 1, 0.8941177, 0, 1,
6.343434, 3.717172, -515.475, 1, 0.8941177, 0, 1,
6.383838, 3.717172, -515.629, 1, 0.8941177, 0, 1,
6.424242, 3.717172, -515.8066, 1, 0.8941177, 0, 1,
6.464646, 3.717172, -516.0078, 1, 0.8941177, 0, 1,
6.505051, 3.717172, -516.2327, 1, 0.8941177, 0, 1,
6.545455, 3.717172, -516.4812, 1, 0.8941177, 0, 1,
6.585859, 3.717172, -516.7533, 1, 0.8941177, 0, 1,
6.626263, 3.717172, -517.0491, 1, 0.8941177, 0, 1,
6.666667, 3.717172, -517.3685, 1, 0.8941177, 0, 1,
6.707071, 3.717172, -517.7115, 1, 0.8941177, 0, 1,
6.747475, 3.717172, -518.0781, 1, 0.8941177, 0, 1,
6.787879, 3.717172, -518.4684, 1, 0.8941177, 0, 1,
6.828283, 3.717172, -518.8823, 1, 0.8941177, 0, 1,
6.868687, 3.717172, -519.3198, 1, 0.8941177, 0, 1,
6.909091, 3.717172, -519.781, 1, 0.7921569, 0, 1,
6.949495, 3.717172, -520.2658, 1, 0.7921569, 0, 1,
6.989899, 3.717172, -520.7742, 1, 0.7921569, 0, 1,
7.030303, 3.717172, -521.3063, 1, 0.7921569, 0, 1,
7.070707, 3.717172, -521.8619, 1, 0.7921569, 0, 1,
7.111111, 3.717172, -522.4413, 1, 0.7921569, 0, 1,
7.151515, 3.717172, -523.0442, 1, 0.7921569, 0, 1,
7.191919, 3.717172, -523.6708, 1, 0.7921569, 0, 1,
7.232323, 3.717172, -524.321, 1, 0.7921569, 0, 1,
7.272727, 3.717172, -524.9948, 1, 0.6862745, 0, 1,
7.313131, 3.717172, -525.6923, 1, 0.6862745, 0, 1,
7.353535, 3.717172, -526.4133, 1, 0.6862745, 0, 1,
7.393939, 3.717172, -527.1581, 1, 0.6862745, 0, 1,
7.434343, 3.717172, -527.9264, 1, 0.6862745, 0, 1,
7.474748, 3.717172, -528.7184, 1, 0.6862745, 0, 1,
7.515152, 3.717172, -529.534, 1, 0.6862745, 0, 1,
7.555555, 3.717172, -530.3732, 1, 0.5843138, 0, 1,
7.59596, 3.717172, -531.2361, 1, 0.5843138, 0, 1,
7.636364, 3.717172, -532.1226, 1, 0.5843138, 0, 1,
7.676768, 3.717172, -533.0327, 1, 0.5843138, 0, 1,
7.717172, 3.717172, -533.9665, 1, 0.5843138, 0, 1,
7.757576, 3.717172, -534.9239, 1, 0.5843138, 0, 1,
7.79798, 3.717172, -535.9049, 1, 0.4823529, 0, 1,
7.838384, 3.717172, -536.9095, 1, 0.4823529, 0, 1,
7.878788, 3.717172, -537.9378, 1, 0.4823529, 0, 1,
7.919192, 3.717172, -538.9897, 1, 0.4823529, 0, 1,
7.959596, 3.717172, -540.0652, 1, 0.4823529, 0, 1,
8, 3.717172, -541.1644, 1, 0.3764706, 0, 1,
4, 3.767677, -546.9944, 1, 0.2745098, 0, 1,
4.040404, 3.767677, -545.8103, 1, 0.3764706, 0, 1,
4.080808, 3.767677, -544.6492, 1, 0.3764706, 0, 1,
4.121212, 3.767677, -543.511, 1, 0.3764706, 0, 1,
4.161616, 3.767677, -542.3959, 1, 0.3764706, 0, 1,
4.20202, 3.767677, -541.3038, 1, 0.3764706, 0, 1,
4.242424, 3.767677, -540.2346, 1, 0.4823529, 0, 1,
4.282828, 3.767677, -539.1885, 1, 0.4823529, 0, 1,
4.323232, 3.767677, -538.1653, 1, 0.4823529, 0, 1,
4.363636, 3.767677, -537.1652, 1, 0.4823529, 0, 1,
4.40404, 3.767677, -536.1881, 1, 0.4823529, 0, 1,
4.444445, 3.767677, -535.2339, 1, 0.5843138, 0, 1,
4.484848, 3.767677, -534.3029, 1, 0.5843138, 0, 1,
4.525252, 3.767677, -533.3947, 1, 0.5843138, 0, 1,
4.565657, 3.767677, -532.5096, 1, 0.5843138, 0, 1,
4.606061, 3.767677, -531.6475, 1, 0.5843138, 0, 1,
4.646465, 3.767677, -530.8083, 1, 0.5843138, 0, 1,
4.686869, 3.767677, -529.9922, 1, 0.6862745, 0, 1,
4.727273, 3.767677, -529.199, 1, 0.6862745, 0, 1,
4.767677, 3.767677, -528.4289, 1, 0.6862745, 0, 1,
4.808081, 3.767677, -527.6818, 1, 0.6862745, 0, 1,
4.848485, 3.767677, -526.9576, 1, 0.6862745, 0, 1,
4.888889, 3.767677, -526.2565, 1, 0.6862745, 0, 1,
4.929293, 3.767677, -525.5784, 1, 0.6862745, 0, 1,
4.969697, 3.767677, -524.9233, 1, 0.6862745, 0, 1,
5.010101, 3.767677, -524.2911, 1, 0.7921569, 0, 1,
5.050505, 3.767677, -523.682, 1, 0.7921569, 0, 1,
5.090909, 3.767677, -523.0959, 1, 0.7921569, 0, 1,
5.131313, 3.767677, -522.5328, 1, 0.7921569, 0, 1,
5.171717, 3.767677, -521.9927, 1, 0.7921569, 0, 1,
5.212121, 3.767677, -521.4755, 1, 0.7921569, 0, 1,
5.252525, 3.767677, -520.9814, 1, 0.7921569, 0, 1,
5.292929, 3.767677, -520.5103, 1, 0.7921569, 0, 1,
5.333333, 3.767677, -520.0621, 1, 0.7921569, 0, 1,
5.373737, 3.767677, -519.637, 1, 0.7921569, 0, 1,
5.414141, 3.767677, -519.2349, 1, 0.8941177, 0, 1,
5.454545, 3.767677, -518.8558, 1, 0.8941177, 0, 1,
5.494949, 3.767677, -518.4996, 1, 0.8941177, 0, 1,
5.535354, 3.767677, -518.1665, 1, 0.8941177, 0, 1,
5.575758, 3.767677, -517.8564, 1, 0.8941177, 0, 1,
5.616162, 3.767677, -517.5693, 1, 0.8941177, 0, 1,
5.656566, 3.767677, -517.3052, 1, 0.8941177, 0, 1,
5.69697, 3.767677, -517.064, 1, 0.8941177, 0, 1,
5.737374, 3.767677, -516.8459, 1, 0.8941177, 0, 1,
5.777778, 3.767677, -516.6508, 1, 0.8941177, 0, 1,
5.818182, 3.767677, -516.4786, 1, 0.8941177, 0, 1,
5.858586, 3.767677, -516.3295, 1, 0.8941177, 0, 1,
5.89899, 3.767677, -516.2034, 1, 0.8941177, 0, 1,
5.939394, 3.767677, -516.1003, 1, 0.8941177, 0, 1,
5.979798, 3.767677, -516.0201, 1, 0.8941177, 0, 1,
6.020202, 3.767677, -515.9631, 1, 0.8941177, 0, 1,
6.060606, 3.767677, -515.929, 1, 0.8941177, 0, 1,
6.10101, 3.767677, -515.9178, 1, 0.8941177, 0, 1,
6.141414, 3.767677, -515.9297, 1, 0.8941177, 0, 1,
6.181818, 3.767677, -515.9645, 1, 0.8941177, 0, 1,
6.222222, 3.767677, -516.0225, 1, 0.8941177, 0, 1,
6.262626, 3.767677, -516.1033, 1, 0.8941177, 0, 1,
6.30303, 3.767677, -516.2072, 1, 0.8941177, 0, 1,
6.343434, 3.767677, -516.3341, 1, 0.8941177, 0, 1,
6.383838, 3.767677, -516.4839, 1, 0.8941177, 0, 1,
6.424242, 3.767677, -516.6569, 1, 0.8941177, 0, 1,
6.464646, 3.767677, -516.8527, 1, 0.8941177, 0, 1,
6.505051, 3.767677, -517.0716, 1, 0.8941177, 0, 1,
6.545455, 3.767677, -517.3135, 1, 0.8941177, 0, 1,
6.585859, 3.767677, -517.5784, 1, 0.8941177, 0, 1,
6.626263, 3.767677, -517.8663, 1, 0.8941177, 0, 1,
6.666667, 3.767677, -518.1771, 1, 0.8941177, 0, 1,
6.707071, 3.767677, -518.511, 1, 0.8941177, 0, 1,
6.747475, 3.767677, -518.8679, 1, 0.8941177, 0, 1,
6.787879, 3.767677, -519.2478, 1, 0.8941177, 0, 1,
6.828283, 3.767677, -519.6507, 1, 0.7921569, 0, 1,
6.868687, 3.767677, -520.0765, 1, 0.7921569, 0, 1,
6.909091, 3.767677, -520.5255, 1, 0.7921569, 0, 1,
6.949495, 3.767677, -520.9973, 1, 0.7921569, 0, 1,
6.989899, 3.767677, -521.4922, 1, 0.7921569, 0, 1,
7.030303, 3.767677, -522.0101, 1, 0.7921569, 0, 1,
7.070707, 3.767677, -522.551, 1, 0.7921569, 0, 1,
7.111111, 3.767677, -523.1149, 1, 0.7921569, 0, 1,
7.151515, 3.767677, -523.7017, 1, 0.7921569, 0, 1,
7.191919, 3.767677, -524.3116, 1, 0.7921569, 0, 1,
7.232323, 3.767677, -524.9445, 1, 0.6862745, 0, 1,
7.272727, 3.767677, -525.6004, 1, 0.6862745, 0, 1,
7.313131, 3.767677, -526.2793, 1, 0.6862745, 0, 1,
7.353535, 3.767677, -526.9812, 1, 0.6862745, 0, 1,
7.393939, 3.767677, -527.7061, 1, 0.6862745, 0, 1,
7.434343, 3.767677, -528.4539, 1, 0.6862745, 0, 1,
7.474748, 3.767677, -529.2249, 1, 0.6862745, 0, 1,
7.515152, 3.767677, -530.0187, 1, 0.6862745, 0, 1,
7.555555, 3.767677, -530.8356, 1, 0.5843138, 0, 1,
7.59596, 3.767677, -531.6755, 1, 0.5843138, 0, 1,
7.636364, 3.767677, -532.5384, 1, 0.5843138, 0, 1,
7.676768, 3.767677, -533.4243, 1, 0.5843138, 0, 1,
7.717172, 3.767677, -534.3332, 1, 0.5843138, 0, 1,
7.757576, 3.767677, -535.2651, 1, 0.5843138, 0, 1,
7.79798, 3.767677, -536.2199, 1, 0.4823529, 0, 1,
7.838384, 3.767677, -537.1978, 1, 0.4823529, 0, 1,
7.878788, 3.767677, -538.1987, 1, 0.4823529, 0, 1,
7.919192, 3.767677, -539.2226, 1, 0.4823529, 0, 1,
7.959596, 3.767677, -540.2695, 1, 0.4823529, 0, 1,
8, 3.767677, -541.3394, 1, 0.3764706, 0, 1,
4, 3.818182, -547.0844, 1, 0.2745098, 0, 1,
4.040404, 3.818182, -545.9313, 1, 0.3764706, 0, 1,
4.080808, 3.818182, -544.8007, 1, 0.3764706, 0, 1,
4.121212, 3.818182, -543.6925, 1, 0.3764706, 0, 1,
4.161616, 3.818182, -542.6067, 1, 0.3764706, 0, 1,
4.20202, 3.818182, -541.5432, 1, 0.3764706, 0, 1,
4.242424, 3.818182, -540.5022, 1, 0.4823529, 0, 1,
4.282828, 3.818182, -539.4836, 1, 0.4823529, 0, 1,
4.323232, 3.818182, -538.4873, 1, 0.4823529, 0, 1,
4.363636, 3.818182, -537.5135, 1, 0.4823529, 0, 1,
4.40404, 3.818182, -536.562, 1, 0.4823529, 0, 1,
4.444445, 3.818182, -535.633, 1, 0.5843138, 0, 1,
4.484848, 3.818182, -534.7263, 1, 0.5843138, 0, 1,
4.525252, 3.818182, -533.842, 1, 0.5843138, 0, 1,
4.565657, 3.818182, -532.9802, 1, 0.5843138, 0, 1,
4.606061, 3.818182, -532.1407, 1, 0.5843138, 0, 1,
4.646465, 3.818182, -531.3236, 1, 0.5843138, 0, 1,
4.686869, 3.818182, -530.5289, 1, 0.5843138, 0, 1,
4.727273, 3.818182, -529.7567, 1, 0.6862745, 0, 1,
4.767677, 3.818182, -529.0068, 1, 0.6862745, 0, 1,
4.808081, 3.818182, -528.2793, 1, 0.6862745, 0, 1,
4.848485, 3.818182, -527.5742, 1, 0.6862745, 0, 1,
4.888889, 3.818182, -526.8915, 1, 0.6862745, 0, 1,
4.929293, 3.818182, -526.2311, 1, 0.6862745, 0, 1,
4.969697, 3.818182, -525.5933, 1, 0.6862745, 0, 1,
5.010101, 3.818182, -524.9777, 1, 0.6862745, 0, 1,
5.050505, 3.818182, -524.3846, 1, 0.7921569, 0, 1,
5.090909, 3.818182, -523.8139, 1, 0.7921569, 0, 1,
5.131313, 3.818182, -523.2656, 1, 0.7921569, 0, 1,
5.171717, 3.818182, -522.7396, 1, 0.7921569, 0, 1,
5.212121, 3.818182, -522.2361, 1, 0.7921569, 0, 1,
5.252525, 3.818182, -521.7549, 1, 0.7921569, 0, 1,
5.292929, 3.818182, -521.2962, 1, 0.7921569, 0, 1,
5.333333, 3.818182, -520.8599, 1, 0.7921569, 0, 1,
5.373737, 3.818182, -520.4459, 1, 0.7921569, 0, 1,
5.414141, 3.818182, -520.0544, 1, 0.7921569, 0, 1,
5.454545, 3.818182, -519.6852, 1, 0.7921569, 0, 1,
5.494949, 3.818182, -519.3384, 1, 0.8941177, 0, 1,
5.535354, 3.818182, -519.014, 1, 0.8941177, 0, 1,
5.575758, 3.818182, -518.7121, 1, 0.8941177, 0, 1,
5.616162, 3.818182, -518.4325, 1, 0.8941177, 0, 1,
5.656566, 3.818182, -518.1754, 1, 0.8941177, 0, 1,
5.69697, 3.818182, -517.9406, 1, 0.8941177, 0, 1,
5.737374, 3.818182, -517.7281, 1, 0.8941177, 0, 1,
5.777778, 3.818182, -517.5381, 1, 0.8941177, 0, 1,
5.818182, 3.818182, -517.3705, 1, 0.8941177, 0, 1,
5.858586, 3.818182, -517.2253, 1, 0.8941177, 0, 1,
5.89899, 3.818182, -517.1025, 1, 0.8941177, 0, 1,
5.939394, 3.818182, -517.0021, 1, 0.8941177, 0, 1,
5.979798, 3.818182, -516.9241, 1, 0.8941177, 0, 1,
6.020202, 3.818182, -516.8685, 1, 0.8941177, 0, 1,
6.060606, 3.818182, -516.8353, 1, 0.8941177, 0, 1,
6.10101, 3.818182, -516.8245, 1, 0.8941177, 0, 1,
6.141414, 3.818182, -516.836, 1, 0.8941177, 0, 1,
6.181818, 3.818182, -516.87, 1, 0.8941177, 0, 1,
6.222222, 3.818182, -516.9263, 1, 0.8941177, 0, 1,
6.262626, 3.818182, -517.0051, 1, 0.8941177, 0, 1,
6.30303, 3.818182, -517.1063, 1, 0.8941177, 0, 1,
6.343434, 3.818182, -517.2298, 1, 0.8941177, 0, 1,
6.383838, 3.818182, -517.3757, 1, 0.8941177, 0, 1,
6.424242, 3.818182, -517.5441, 1, 0.8941177, 0, 1,
6.464646, 3.818182, -517.7348, 1, 0.8941177, 0, 1,
6.505051, 3.818182, -517.9479, 1, 0.8941177, 0, 1,
6.545455, 3.818182, -518.1835, 1, 0.8941177, 0, 1,
6.585859, 3.818182, -518.4413, 1, 0.8941177, 0, 1,
6.626263, 3.818182, -518.7217, 1, 0.8941177, 0, 1,
6.666667, 3.818182, -519.0244, 1, 0.8941177, 0, 1,
6.707071, 3.818182, -519.3495, 1, 0.8941177, 0, 1,
6.747475, 3.818182, -519.697, 1, 0.7921569, 0, 1,
6.787879, 3.818182, -520.0669, 1, 0.7921569, 0, 1,
6.828283, 3.818182, -520.4592, 1, 0.7921569, 0, 1,
6.868687, 3.818182, -520.8739, 1, 0.7921569, 0, 1,
6.909091, 3.818182, -521.311, 1, 0.7921569, 0, 1,
6.949495, 3.818182, -521.7704, 1, 0.7921569, 0, 1,
6.989899, 3.818182, -522.2523, 1, 0.7921569, 0, 1,
7.030303, 3.818182, -522.7566, 1, 0.7921569, 0, 1,
7.070707, 3.818182, -523.2833, 1, 0.7921569, 0, 1,
7.111111, 3.818182, -523.8323, 1, 0.7921569, 0, 1,
7.151515, 3.818182, -524.4038, 1, 0.7921569, 0, 1,
7.191919, 3.818182, -524.9977, 1, 0.6862745, 0, 1,
7.232323, 3.818182, -525.614, 1, 0.6862745, 0, 1,
7.272727, 3.818182, -526.2526, 1, 0.6862745, 0, 1,
7.313131, 3.818182, -526.9136, 1, 0.6862745, 0, 1,
7.353535, 3.818182, -527.597, 1, 0.6862745, 0, 1,
7.393939, 3.818182, -528.3029, 1, 0.6862745, 0, 1,
7.434343, 3.818182, -529.0311, 1, 0.6862745, 0, 1,
7.474748, 3.818182, -529.7817, 1, 0.6862745, 0, 1,
7.515152, 3.818182, -530.5548, 1, 0.5843138, 0, 1,
7.555555, 3.818182, -531.3502, 1, 0.5843138, 0, 1,
7.59596, 3.818182, -532.168, 1, 0.5843138, 0, 1,
7.636364, 3.818182, -533.0082, 1, 0.5843138, 0, 1,
7.676768, 3.818182, -533.8708, 1, 0.5843138, 0, 1,
7.717172, 3.818182, -534.7559, 1, 0.5843138, 0, 1,
7.757576, 3.818182, -535.6633, 1, 0.4823529, 0, 1,
7.79798, 3.818182, -536.593, 1, 0.4823529, 0, 1,
7.838384, 3.818182, -537.5452, 1, 0.4823529, 0, 1,
7.878788, 3.818182, -538.5198, 1, 0.4823529, 0, 1,
7.919192, 3.818182, -539.5168, 1, 0.4823529, 0, 1,
7.959596, 3.818182, -540.5362, 1, 0.4823529, 0, 1,
8, 3.818182, -541.5779, 1, 0.3764706, 0, 1,
4, 3.868687, -547.2394, 1, 0.2745098, 0, 1,
4.040404, 3.868687, -546.1163, 1, 0.3764706, 0, 1,
4.080808, 3.868687, -545.015, 1, 0.3764706, 0, 1,
4.121212, 3.868687, -543.9355, 1, 0.3764706, 0, 1,
4.161616, 3.868687, -542.8779, 1, 0.3764706, 0, 1,
4.20202, 3.868687, -541.842, 1, 0.3764706, 0, 1,
4.242424, 3.868687, -540.828, 1, 0.4823529, 0, 1,
4.282828, 3.868687, -539.8358, 1, 0.4823529, 0, 1,
4.323232, 3.868687, -538.8654, 1, 0.4823529, 0, 1,
4.363636, 3.868687, -537.9168, 1, 0.4823529, 0, 1,
4.40404, 3.868687, -536.9901, 1, 0.4823529, 0, 1,
4.444445, 3.868687, -536.0851, 1, 0.4823529, 0, 1,
4.484848, 3.868687, -535.2019, 1, 0.5843138, 0, 1,
4.525252, 3.868687, -534.3406, 1, 0.5843138, 0, 1,
4.565657, 3.868687, -533.5011, 1, 0.5843138, 0, 1,
4.606061, 3.868687, -532.6834, 1, 0.5843138, 0, 1,
4.646465, 3.868687, -531.8875, 1, 0.5843138, 0, 1,
4.686869, 3.868687, -531.1135, 1, 0.5843138, 0, 1,
4.727273, 3.868687, -530.3612, 1, 0.5843138, 0, 1,
4.767677, 3.868687, -529.6307, 1, 0.6862745, 0, 1,
4.808081, 3.868687, -528.9221, 1, 0.6862745, 0, 1,
4.848485, 3.868687, -528.2353, 1, 0.6862745, 0, 1,
4.888889, 3.868687, -527.5703, 1, 0.6862745, 0, 1,
4.929293, 3.868687, -526.9271, 1, 0.6862745, 0, 1,
4.969697, 3.868687, -526.3058, 1, 0.6862745, 0, 1,
5.010101, 3.868687, -525.7062, 1, 0.6862745, 0, 1,
5.050505, 3.868687, -525.1285, 1, 0.6862745, 0, 1,
5.090909, 3.868687, -524.5726, 1, 0.7921569, 0, 1,
5.131313, 3.868687, -524.0385, 1, 0.7921569, 0, 1,
5.171717, 3.868687, -523.5262, 1, 0.7921569, 0, 1,
5.212121, 3.868687, -523.0357, 1, 0.7921569, 0, 1,
5.252525, 3.868687, -522.5671, 1, 0.7921569, 0, 1,
5.292929, 3.868687, -522.1202, 1, 0.7921569, 0, 1,
5.333333, 3.868687, -521.6952, 1, 0.7921569, 0, 1,
5.373737, 3.868687, -521.292, 1, 0.7921569, 0, 1,
5.414141, 3.868687, -520.9106, 1, 0.7921569, 0, 1,
5.454545, 3.868687, -520.551, 1, 0.7921569, 0, 1,
5.494949, 3.868687, -520.2132, 1, 0.7921569, 0, 1,
5.535354, 3.868687, -519.8973, 1, 0.7921569, 0, 1,
5.575758, 3.868687, -519.6031, 1, 0.7921569, 0, 1,
5.616162, 3.868687, -519.3308, 1, 0.8941177, 0, 1,
5.656566, 3.868687, -519.0803, 1, 0.8941177, 0, 1,
5.69697, 3.868687, -518.8516, 1, 0.8941177, 0, 1,
5.737374, 3.868687, -518.6447, 1, 0.8941177, 0, 1,
5.777778, 3.868687, -518.4597, 1, 0.8941177, 0, 1,
5.818182, 3.868687, -518.2964, 1, 0.8941177, 0, 1,
5.858586, 3.868687, -518.155, 1, 0.8941177, 0, 1,
5.89899, 3.868687, -518.0353, 1, 0.8941177, 0, 1,
5.939394, 3.868687, -517.9375, 1, 0.8941177, 0, 1,
5.979798, 3.868687, -517.8615, 1, 0.8941177, 0, 1,
6.020202, 3.868687, -517.8073, 1, 0.8941177, 0, 1,
6.060606, 3.868687, -517.775, 1, 0.8941177, 0, 1,
6.10101, 3.868687, -517.7644, 1, 0.8941177, 0, 1,
6.141414, 3.868687, -517.7757, 1, 0.8941177, 0, 1,
6.181818, 3.868687, -517.8088, 1, 0.8941177, 0, 1,
6.222222, 3.868687, -517.8636, 1, 0.8941177, 0, 1,
6.262626, 3.868687, -517.9404, 1, 0.8941177, 0, 1,
6.30303, 3.868687, -518.0389, 1, 0.8941177, 0, 1,
6.343434, 3.868687, -518.1592, 1, 0.8941177, 0, 1,
6.383838, 3.868687, -518.3014, 1, 0.8941177, 0, 1,
6.424242, 3.868687, -518.4654, 1, 0.8941177, 0, 1,
6.464646, 3.868687, -518.6512, 1, 0.8941177, 0, 1,
6.505051, 3.868687, -518.8588, 1, 0.8941177, 0, 1,
6.545455, 3.868687, -519.0882, 1, 0.8941177, 0, 1,
6.585859, 3.868687, -519.3394, 1, 0.8941177, 0, 1,
6.626263, 3.868687, -519.6125, 1, 0.7921569, 0, 1,
6.666667, 3.868687, -519.9073, 1, 0.7921569, 0, 1,
6.707071, 3.868687, -520.224, 1, 0.7921569, 0, 1,
6.747475, 3.868687, -520.5625, 1, 0.7921569, 0, 1,
6.787879, 3.868687, -520.9228, 1, 0.7921569, 0, 1,
6.828283, 3.868687, -521.3049, 1, 0.7921569, 0, 1,
6.868687, 3.868687, -521.7089, 1, 0.7921569, 0, 1,
6.909091, 3.868687, -522.1346, 1, 0.7921569, 0, 1,
6.949495, 3.868687, -522.5822, 1, 0.7921569, 0, 1,
6.989899, 3.868687, -523.0515, 1, 0.7921569, 0, 1,
7.030303, 3.868687, -523.5427, 1, 0.7921569, 0, 1,
7.070707, 3.868687, -524.0557, 1, 0.7921569, 0, 1,
7.111111, 3.868687, -524.5906, 1, 0.7921569, 0, 1,
7.151515, 3.868687, -525.1472, 1, 0.6862745, 0, 1,
7.191919, 3.868687, -525.7256, 1, 0.6862745, 0, 1,
7.232323, 3.868687, -526.3259, 1, 0.6862745, 0, 1,
7.272727, 3.868687, -526.948, 1, 0.6862745, 0, 1,
7.313131, 3.868687, -527.5919, 1, 0.6862745, 0, 1,
7.353535, 3.868687, -528.2576, 1, 0.6862745, 0, 1,
7.393939, 3.868687, -528.9451, 1, 0.6862745, 0, 1,
7.434343, 3.868687, -529.6545, 1, 0.6862745, 0, 1,
7.474748, 3.868687, -530.3856, 1, 0.5843138, 0, 1,
7.515152, 3.868687, -531.1386, 1, 0.5843138, 0, 1,
7.555555, 3.868687, -531.9134, 1, 0.5843138, 0, 1,
7.59596, 3.868687, -532.71, 1, 0.5843138, 0, 1,
7.636364, 3.868687, -533.5284, 1, 0.5843138, 0, 1,
7.676768, 3.868687, -534.3687, 1, 0.5843138, 0, 1,
7.717172, 3.868687, -535.2307, 1, 0.5843138, 0, 1,
7.757576, 3.868687, -536.1146, 1, 0.4823529, 0, 1,
7.79798, 3.868687, -537.0203, 1, 0.4823529, 0, 1,
7.838384, 3.868687, -537.9478, 1, 0.4823529, 0, 1,
7.878788, 3.868687, -538.897, 1, 0.4823529, 0, 1,
7.919192, 3.868687, -539.8682, 1, 0.4823529, 0, 1,
7.959596, 3.868687, -540.8611, 1, 0.4823529, 0, 1,
8, 3.868687, -541.8759, 1, 0.3764706, 0, 1,
4, 3.919192, -547.4554, 1, 0.2745098, 0, 1,
4.040404, 3.919192, -546.361, 1, 0.3764706, 0, 1,
4.080808, 3.919192, -545.288, 1, 0.3764706, 0, 1,
4.121212, 3.919192, -544.2361, 1, 0.3764706, 0, 1,
4.161616, 3.919192, -543.2056, 1, 0.3764706, 0, 1,
4.20202, 3.919192, -542.1962, 1, 0.3764706, 0, 1,
4.242424, 3.919192, -541.2081, 1, 0.3764706, 0, 1,
4.282828, 3.919192, -540.2413, 1, 0.4823529, 0, 1,
4.323232, 3.919192, -539.2958, 1, 0.4823529, 0, 1,
4.363636, 3.919192, -538.3715, 1, 0.4823529, 0, 1,
4.40404, 3.919192, -537.4684, 1, 0.4823529, 0, 1,
4.444445, 3.919192, -536.5867, 1, 0.4823529, 0, 1,
4.484848, 3.919192, -535.7261, 1, 0.4823529, 0, 1,
4.525252, 3.919192, -534.8869, 1, 0.5843138, 0, 1,
4.565657, 3.919192, -534.0688, 1, 0.5843138, 0, 1,
4.606061, 3.919192, -533.2721, 1, 0.5843138, 0, 1,
4.646465, 3.919192, -532.4966, 1, 0.5843138, 0, 1,
4.686869, 3.919192, -531.7424, 1, 0.5843138, 0, 1,
4.727273, 3.919192, -531.0093, 1, 0.5843138, 0, 1,
4.767677, 3.919192, -530.2976, 1, 0.5843138, 0, 1,
4.808081, 3.919192, -529.6072, 1, 0.6862745, 0, 1,
4.848485, 3.919192, -528.9379, 1, 0.6862745, 0, 1,
4.888889, 3.919192, -528.29, 1, 0.6862745, 0, 1,
4.929293, 3.919192, -527.6633, 1, 0.6862745, 0, 1,
4.969697, 3.919192, -527.0578, 1, 0.6862745, 0, 1,
5.010101, 3.919192, -526.4736, 1, 0.6862745, 0, 1,
5.050505, 3.919192, -525.9106, 1, 0.6862745, 0, 1,
5.090909, 3.919192, -525.369, 1, 0.6862745, 0, 1,
5.131313, 3.919192, -524.8486, 1, 0.7921569, 0, 1,
5.171717, 3.919192, -524.3494, 1, 0.7921569, 0, 1,
5.212121, 3.919192, -523.8715, 1, 0.7921569, 0, 1,
5.252525, 3.919192, -523.4148, 1, 0.7921569, 0, 1,
5.292929, 3.919192, -522.9794, 1, 0.7921569, 0, 1,
5.333333, 3.919192, -522.5652, 1, 0.7921569, 0, 1,
5.373737, 3.919192, -522.1724, 1, 0.7921569, 0, 1,
5.414141, 3.919192, -521.8007, 1, 0.7921569, 0, 1,
5.454545, 3.919192, -521.4504, 1, 0.7921569, 0, 1,
5.494949, 3.919192, -521.1212, 1, 0.7921569, 0, 1,
5.535354, 3.919192, -520.8134, 1, 0.7921569, 0, 1,
5.575758, 3.919192, -520.5268, 1, 0.7921569, 0, 1,
5.616162, 3.919192, -520.2614, 1, 0.7921569, 0, 1,
5.656566, 3.919192, -520.0173, 1, 0.7921569, 0, 1,
5.69697, 3.919192, -519.7945, 1, 0.7921569, 0, 1,
5.737374, 3.919192, -519.5929, 1, 0.7921569, 0, 1,
5.777778, 3.919192, -519.4125, 1, 0.8941177, 0, 1,
5.818182, 3.919192, -519.2535, 1, 0.8941177, 0, 1,
5.858586, 3.919192, -519.1157, 1, 0.8941177, 0, 1,
5.89899, 3.919192, -518.9991, 1, 0.8941177, 0, 1,
5.939394, 3.919192, -518.9038, 1, 0.8941177, 0, 1,
5.979798, 3.919192, -518.8298, 1, 0.8941177, 0, 1,
6.020202, 3.919192, -518.777, 1, 0.8941177, 0, 1,
6.060606, 3.919192, -518.7454, 1, 0.8941177, 0, 1,
6.10101, 3.919192, -518.7352, 1, 0.8941177, 0, 1,
6.141414, 3.919192, -518.7462, 1, 0.8941177, 0, 1,
6.181818, 3.919192, -518.7784, 1, 0.8941177, 0, 1,
6.222222, 3.919192, -518.8318, 1, 0.8941177, 0, 1,
6.262626, 3.919192, -518.9066, 1, 0.8941177, 0, 1,
6.30303, 3.919192, -519.0026, 1, 0.8941177, 0, 1,
6.343434, 3.919192, -519.1199, 1, 0.8941177, 0, 1,
6.383838, 3.919192, -519.2584, 1, 0.8941177, 0, 1,
6.424242, 3.919192, -519.4182, 1, 0.8941177, 0, 1,
6.464646, 3.919192, -519.5992, 1, 0.7921569, 0, 1,
6.505051, 3.919192, -519.8015, 1, 0.7921569, 0, 1,
6.545455, 3.919192, -520.025, 1, 0.7921569, 0, 1,
6.585859, 3.919192, -520.2698, 1, 0.7921569, 0, 1,
6.626263, 3.919192, -520.5359, 1, 0.7921569, 0, 1,
6.666667, 3.919192, -520.8232, 1, 0.7921569, 0, 1,
6.707071, 3.919192, -521.1318, 1, 0.7921569, 0, 1,
6.747475, 3.919192, -521.4615, 1, 0.7921569, 0, 1,
6.787879, 3.919192, -521.8127, 1, 0.7921569, 0, 1,
6.828283, 3.919192, -522.185, 1, 0.7921569, 0, 1,
6.868687, 3.919192, -522.5786, 1, 0.7921569, 0, 1,
6.909091, 3.919192, -522.9934, 1, 0.7921569, 0, 1,
6.949495, 3.919192, -523.4295, 1, 0.7921569, 0, 1,
6.989899, 3.919192, -523.8869, 1, 0.7921569, 0, 1,
7.030303, 3.919192, -524.3655, 1, 0.7921569, 0, 1,
7.070707, 3.919192, -524.8654, 1, 0.7921569, 0, 1,
7.111111, 3.919192, -525.3865, 1, 0.6862745, 0, 1,
7.151515, 3.919192, -525.9289, 1, 0.6862745, 0, 1,
7.191919, 3.919192, -526.4926, 1, 0.6862745, 0, 1,
7.232323, 3.919192, -527.0775, 1, 0.6862745, 0, 1,
7.272727, 3.919192, -527.6836, 1, 0.6862745, 0, 1,
7.313131, 3.919192, -528.311, 1, 0.6862745, 0, 1,
7.353535, 3.919192, -528.9597, 1, 0.6862745, 0, 1,
7.393939, 3.919192, -529.6296, 1, 0.6862745, 0, 1,
7.434343, 3.919192, -530.3207, 1, 0.5843138, 0, 1,
7.474748, 3.919192, -531.0332, 1, 0.5843138, 0, 1,
7.515152, 3.919192, -531.7669, 1, 0.5843138, 0, 1,
7.555555, 3.919192, -532.5219, 1, 0.5843138, 0, 1,
7.59596, 3.919192, -533.298, 1, 0.5843138, 0, 1,
7.636364, 3.919192, -534.0955, 1, 0.5843138, 0, 1,
7.676768, 3.919192, -534.9142, 1, 0.5843138, 0, 1,
7.717172, 3.919192, -535.7542, 1, 0.4823529, 0, 1,
7.757576, 3.919192, -536.6154, 1, 0.4823529, 0, 1,
7.79798, 3.919192, -537.4979, 1, 0.4823529, 0, 1,
7.838384, 3.919192, -538.4016, 1, 0.4823529, 0, 1,
7.878788, 3.919192, -539.3267, 1, 0.4823529, 0, 1,
7.919192, 3.919192, -540.2729, 1, 0.4823529, 0, 1,
7.959596, 3.919192, -541.2404, 1, 0.3764706, 0, 1,
8, 3.919192, -542.2292, 1, 0.3764706, 0, 1,
4, 3.969697, -547.7283, 1, 0.2745098, 0, 1,
4.040404, 3.969697, -546.6617, 1, 0.2745098, 0, 1,
4.080808, 3.969697, -545.6157, 1, 0.3764706, 0, 1,
4.121212, 3.969697, -544.5905, 1, 0.3764706, 0, 1,
4.161616, 3.969697, -543.5859, 1, 0.3764706, 0, 1,
4.20202, 3.969697, -542.6021, 1, 0.3764706, 0, 1,
4.242424, 3.969697, -541.639, 1, 0.3764706, 0, 1,
4.282828, 3.969697, -540.6967, 1, 0.4823529, 0, 1,
4.323232, 3.969697, -539.775, 1, 0.4823529, 0, 1,
4.363636, 3.969697, -538.8741, 1, 0.4823529, 0, 1,
4.40404, 3.969697, -537.9939, 1, 0.4823529, 0, 1,
4.444445, 3.969697, -537.1344, 1, 0.4823529, 0, 1,
4.484848, 3.969697, -536.2957, 1, 0.4823529, 0, 1,
4.525252, 3.969697, -535.4776, 1, 0.5843138, 0, 1,
4.565657, 3.969697, -534.6802, 1, 0.5843138, 0, 1,
4.606061, 3.969697, -533.9036, 1, 0.5843138, 0, 1,
4.646465, 3.969697, -533.1478, 1, 0.5843138, 0, 1,
4.686869, 3.969697, -532.4126, 1, 0.5843138, 0, 1,
4.727273, 3.969697, -531.6981, 1, 0.5843138, 0, 1,
4.767677, 3.969697, -531.0044, 1, 0.5843138, 0, 1,
4.808081, 3.969697, -530.3314, 1, 0.5843138, 0, 1,
4.848485, 3.969697, -529.6791, 1, 0.6862745, 0, 1,
4.888889, 3.969697, -529.0475, 1, 0.6862745, 0, 1,
4.929293, 3.969697, -528.4366, 1, 0.6862745, 0, 1,
4.969697, 3.969697, -527.8465, 1, 0.6862745, 0, 1,
5.010101, 3.969697, -527.277, 1, 0.6862745, 0, 1,
5.050505, 3.969697, -526.7283, 1, 0.6862745, 0, 1,
5.090909, 3.969697, -526.2003, 1, 0.6862745, 0, 1,
5.131313, 3.969697, -525.6931, 1, 0.6862745, 0, 1,
5.171717, 3.969697, -525.2065, 1, 0.6862745, 0, 1,
5.212121, 3.969697, -524.7407, 1, 0.7921569, 0, 1,
5.252525, 3.969697, -524.2956, 1, 0.7921569, 0, 1,
5.292929, 3.969697, -523.8712, 1, 0.7921569, 0, 1,
5.333333, 3.969697, -523.4675, 1, 0.7921569, 0, 1,
5.373737, 3.969697, -523.0845, 1, 0.7921569, 0, 1,
5.414141, 3.969697, -522.7223, 1, 0.7921569, 0, 1,
5.454545, 3.969697, -522.3808, 1, 0.7921569, 0, 1,
5.494949, 3.969697, -522.06, 1, 0.7921569, 0, 1,
5.535354, 3.969697, -521.7599, 1, 0.7921569, 0, 1,
5.575758, 3.969697, -521.4805, 1, 0.7921569, 0, 1,
5.616162, 3.969697, -521.2219, 1, 0.7921569, 0, 1,
5.656566, 3.969697, -520.984, 1, 0.7921569, 0, 1,
5.69697, 3.969697, -520.7668, 1, 0.7921569, 0, 1,
5.737374, 3.969697, -520.5703, 1, 0.7921569, 0, 1,
5.777778, 3.969697, -520.3945, 1, 0.7921569, 0, 1,
5.818182, 3.969697, -520.2395, 1, 0.7921569, 0, 1,
5.858586, 3.969697, -520.1052, 1, 0.7921569, 0, 1,
5.89899, 3.969697, -519.9915, 1, 0.7921569, 0, 1,
5.939394, 3.969697, -519.8986, 1, 0.7921569, 0, 1,
5.979798, 3.969697, -519.8265, 1, 0.7921569, 0, 1,
6.020202, 3.969697, -519.775, 1, 0.7921569, 0, 1,
6.060606, 3.969697, -519.7443, 1, 0.7921569, 0, 1,
6.10101, 3.969697, -519.7343, 1, 0.7921569, 0, 1,
6.141414, 3.969697, -519.7449, 1, 0.7921569, 0, 1,
6.181818, 3.969697, -519.7764, 1, 0.7921569, 0, 1,
6.222222, 3.969697, -519.8285, 1, 0.7921569, 0, 1,
6.262626, 3.969697, -519.9014, 1, 0.7921569, 0, 1,
6.30303, 3.969697, -519.9949, 1, 0.7921569, 0, 1,
6.343434, 3.969697, -520.1093, 1, 0.7921569, 0, 1,
6.383838, 3.969697, -520.2443, 1, 0.7921569, 0, 1,
6.424242, 3.969697, -520.4, 1, 0.7921569, 0, 1,
6.464646, 3.969697, -520.5764, 1, 0.7921569, 0, 1,
6.505051, 3.969697, -520.7736, 1, 0.7921569, 0, 1,
6.545455, 3.969697, -520.9915, 1, 0.7921569, 0, 1,
6.585859, 3.969697, -521.2301, 1, 0.7921569, 0, 1,
6.626263, 3.969697, -521.4894, 1, 0.7921569, 0, 1,
6.666667, 3.969697, -521.7695, 1, 0.7921569, 0, 1,
6.707071, 3.969697, -522.0703, 1, 0.7921569, 0, 1,
6.747475, 3.969697, -522.3917, 1, 0.7921569, 0, 1,
6.787879, 3.969697, -522.7339, 1, 0.7921569, 0, 1,
6.828283, 3.969697, -523.0969, 1, 0.7921569, 0, 1,
6.868687, 3.969697, -523.4805, 1, 0.7921569, 0, 1,
6.909091, 3.969697, -523.8848, 1, 0.7921569, 0, 1,
6.949495, 3.969697, -524.3099, 1, 0.7921569, 0, 1,
6.989899, 3.969697, -524.7557, 1, 0.7921569, 0, 1,
7.030303, 3.969697, -525.2222, 1, 0.6862745, 0, 1,
7.070707, 3.969697, -525.7095, 1, 0.6862745, 0, 1,
7.111111, 3.969697, -526.2174, 1, 0.6862745, 0, 1,
7.151515, 3.969697, -526.7461, 1, 0.6862745, 0, 1,
7.191919, 3.969697, -527.2955, 1, 0.6862745, 0, 1,
7.232323, 3.969697, -527.8656, 1, 0.6862745, 0, 1,
7.272727, 3.969697, -528.4564, 1, 0.6862745, 0, 1,
7.313131, 3.969697, -529.068, 1, 0.6862745, 0, 1,
7.353535, 3.969697, -529.7003, 1, 0.6862745, 0, 1,
7.393939, 3.969697, -530.3532, 1, 0.5843138, 0, 1,
7.434343, 3.969697, -531.0269, 1, 0.5843138, 0, 1,
7.474748, 3.969697, -531.7213, 1, 0.5843138, 0, 1,
7.515152, 3.969697, -532.4365, 1, 0.5843138, 0, 1,
7.555555, 3.969697, -533.1724, 1, 0.5843138, 0, 1,
7.59596, 3.969697, -533.9289, 1, 0.5843138, 0, 1,
7.636364, 3.969697, -534.7062, 1, 0.5843138, 0, 1,
7.676768, 3.969697, -535.5042, 1, 0.5843138, 0, 1,
7.717172, 3.969697, -536.323, 1, 0.4823529, 0, 1,
7.757576, 3.969697, -537.1624, 1, 0.4823529, 0, 1,
7.79798, 3.969697, -538.0226, 1, 0.4823529, 0, 1,
7.838384, 3.969697, -538.9035, 1, 0.4823529, 0, 1,
7.878788, 3.969697, -539.8051, 1, 0.4823529, 0, 1,
7.919192, 3.969697, -540.7274, 1, 0.4823529, 0, 1,
7.959596, 3.969697, -541.6705, 1, 0.3764706, 0, 1,
8, 3.969697, -542.6342, 1, 0.3764706, 0, 1,
4, 4.020202, -548.0546, 1, 0.2745098, 0, 1,
4.040404, 4.020202, -547.0145, 1, 0.2745098, 0, 1,
4.080808, 4.020202, -545.9947, 1, 0.3764706, 0, 1,
4.121212, 4.020202, -544.9951, 1, 0.3764706, 0, 1,
4.161616, 4.020202, -544.0156, 1, 0.3764706, 0, 1,
4.20202, 4.020202, -543.0564, 1, 0.3764706, 0, 1,
4.242424, 4.020202, -542.1173, 1, 0.3764706, 0, 1,
4.282828, 4.020202, -541.1985, 1, 0.3764706, 0, 1,
4.323232, 4.020202, -540.2999, 1, 0.4823529, 0, 1,
4.363636, 4.020202, -539.4214, 1, 0.4823529, 0, 1,
4.40404, 4.020202, -538.5632, 1, 0.4823529, 0, 1,
4.444445, 4.020202, -537.7252, 1, 0.4823529, 0, 1,
4.484848, 4.020202, -536.9073, 1, 0.4823529, 0, 1,
4.525252, 4.020202, -536.1097, 1, 0.4823529, 0, 1,
4.565657, 4.020202, -535.3323, 1, 0.5843138, 0, 1,
4.606061, 4.020202, -534.5751, 1, 0.5843138, 0, 1,
4.646465, 4.020202, -533.8381, 1, 0.5843138, 0, 1,
4.686869, 4.020202, -533.1212, 1, 0.5843138, 0, 1,
4.727273, 4.020202, -532.4246, 1, 0.5843138, 0, 1,
4.767677, 4.020202, -531.7482, 1, 0.5843138, 0, 1,
4.808081, 4.020202, -531.092, 1, 0.5843138, 0, 1,
4.848485, 4.020202, -530.4559, 1, 0.5843138, 0, 1,
4.888889, 4.020202, -529.8401, 1, 0.6862745, 0, 1,
4.929293, 4.020202, -529.2445, 1, 0.6862745, 0, 1,
4.969697, 4.020202, -528.6691, 1, 0.6862745, 0, 1,
5.010101, 4.020202, -528.1139, 1, 0.6862745, 0, 1,
5.050505, 4.020202, -527.5789, 1, 0.6862745, 0, 1,
5.090909, 4.020202, -527.0641, 1, 0.6862745, 0, 1,
5.131313, 4.020202, -526.5695, 1, 0.6862745, 0, 1,
5.171717, 4.020202, -526.0951, 1, 0.6862745, 0, 1,
5.212121, 4.020202, -525.6409, 1, 0.6862745, 0, 1,
5.252525, 4.020202, -525.2069, 1, 0.6862745, 0, 1,
5.292929, 4.020202, -524.7931, 1, 0.7921569, 0, 1,
5.333333, 4.020202, -524.3995, 1, 0.7921569, 0, 1,
5.373737, 4.020202, -524.0261, 1, 0.7921569, 0, 1,
5.414141, 4.020202, -523.6729, 1, 0.7921569, 0, 1,
5.454545, 4.020202, -523.3399, 1, 0.7921569, 0, 1,
5.494949, 4.020202, -523.0272, 1, 0.7921569, 0, 1,
5.535354, 4.020202, -522.7346, 1, 0.7921569, 0, 1,
5.575758, 4.020202, -522.4622, 1, 0.7921569, 0, 1,
5.616162, 4.020202, -522.21, 1, 0.7921569, 0, 1,
5.656566, 4.020202, -521.978, 1, 0.7921569, 0, 1,
5.69697, 4.020202, -521.7662, 1, 0.7921569, 0, 1,
5.737374, 4.020202, -521.5746, 1, 0.7921569, 0, 1,
5.777778, 4.020202, -521.4033, 1, 0.7921569, 0, 1,
5.818182, 4.020202, -521.2521, 1, 0.7921569, 0, 1,
5.858586, 4.020202, -521.1211, 1, 0.7921569, 0, 1,
5.89899, 4.020202, -521.0103, 1, 0.7921569, 0, 1,
5.939394, 4.020202, -520.9197, 1, 0.7921569, 0, 1,
5.979798, 4.020202, -520.8494, 1, 0.7921569, 0, 1,
6.020202, 4.020202, -520.7992, 1, 0.7921569, 0, 1,
6.060606, 4.020202, -520.7692, 1, 0.7921569, 0, 1,
6.10101, 4.020202, -520.7595, 1, 0.7921569, 0, 1,
6.141414, 4.020202, -520.7699, 1, 0.7921569, 0, 1,
6.181818, 4.020202, -520.8005, 1, 0.7921569, 0, 1,
6.222222, 4.020202, -520.8514, 1, 0.7921569, 0, 1,
6.262626, 4.020202, -520.9224, 1, 0.7921569, 0, 1,
6.30303, 4.020202, -521.0137, 1, 0.7921569, 0, 1,
6.343434, 4.020202, -521.1251, 1, 0.7921569, 0, 1,
6.383838, 4.020202, -521.2567, 1, 0.7921569, 0, 1,
6.424242, 4.020202, -521.4086, 1, 0.7921569, 0, 1,
6.464646, 4.020202, -521.5806, 1, 0.7921569, 0, 1,
6.505051, 4.020202, -521.7729, 1, 0.7921569, 0, 1,
6.545455, 4.020202, -521.9853, 1, 0.7921569, 0, 1,
6.585859, 4.020202, -522.218, 1, 0.7921569, 0, 1,
6.626263, 4.020202, -522.4708, 1, 0.7921569, 0, 1,
6.666667, 4.020202, -522.7439, 1, 0.7921569, 0, 1,
6.707071, 4.020202, -523.0371, 1, 0.7921569, 0, 1,
6.747475, 4.020202, -523.3506, 1, 0.7921569, 0, 1,
6.787879, 4.020202, -523.6843, 1, 0.7921569, 0, 1,
6.828283, 4.020202, -524.0381, 1, 0.7921569, 0, 1,
6.868687, 4.020202, -524.4122, 1, 0.7921569, 0, 1,
6.909091, 4.020202, -524.8064, 1, 0.7921569, 0, 1,
6.949495, 4.020202, -525.2209, 1, 0.6862745, 0, 1,
6.989899, 4.020202, -525.6556, 1, 0.6862745, 0, 1,
7.030303, 4.020202, -526.1104, 1, 0.6862745, 0, 1,
7.070707, 4.020202, -526.5855, 1, 0.6862745, 0, 1,
7.111111, 4.020202, -527.0807, 1, 0.6862745, 0, 1,
7.151515, 4.020202, -527.5963, 1, 0.6862745, 0, 1,
7.191919, 4.020202, -528.1319, 1, 0.6862745, 0, 1,
7.232323, 4.020202, -528.6878, 1, 0.6862745, 0, 1,
7.272727, 4.020202, -529.2639, 1, 0.6862745, 0, 1,
7.313131, 4.020202, -529.8601, 1, 0.6862745, 0, 1,
7.353535, 4.020202, -530.4766, 1, 0.5843138, 0, 1,
7.393939, 4.020202, -531.1133, 1, 0.5843138, 0, 1,
7.434343, 4.020202, -531.7701, 1, 0.5843138, 0, 1,
7.474748, 4.020202, -532.4473, 1, 0.5843138, 0, 1,
7.515152, 4.020202, -533.1445, 1, 0.5843138, 0, 1,
7.555555, 4.020202, -533.862, 1, 0.5843138, 0, 1,
7.59596, 4.020202, -534.5997, 1, 0.5843138, 0, 1,
7.636364, 4.020202, -535.3576, 1, 0.5843138, 0, 1,
7.676768, 4.020202, -536.1357, 1, 0.4823529, 0, 1,
7.717172, 4.020202, -536.934, 1, 0.4823529, 0, 1,
7.757576, 4.020202, -537.7525, 1, 0.4823529, 0, 1,
7.79798, 4.020202, -538.5912, 1, 0.4823529, 0, 1,
7.838384, 4.020202, -539.4501, 1, 0.4823529, 0, 1,
7.878788, 4.020202, -540.3292, 1, 0.4823529, 0, 1,
7.919192, 4.020202, -541.2285, 1, 0.3764706, 0, 1,
7.959596, 4.020202, -542.1479, 1, 0.3764706, 0, 1,
8, 4.020202, -543.0877, 1, 0.3764706, 0, 1,
4, 4.070707, -548.4307, 1, 0.2745098, 0, 1,
4.040404, 4.070707, -547.4163, 1, 0.2745098, 0, 1,
4.080808, 4.070707, -546.4216, 1, 0.2745098, 0, 1,
4.121212, 4.070707, -545.4467, 1, 0.3764706, 0, 1,
4.161616, 4.070707, -544.4913, 1, 0.3764706, 0, 1,
4.20202, 4.070707, -543.5558, 1, 0.3764706, 0, 1,
4.242424, 4.070707, -542.6399, 1, 0.3764706, 0, 1,
4.282828, 4.070707, -541.7437, 1, 0.3764706, 0, 1,
4.323232, 4.070707, -540.8672, 1, 0.4823529, 0, 1,
4.363636, 4.070707, -540.0105, 1, 0.4823529, 0, 1,
4.40404, 4.070707, -539.1734, 1, 0.4823529, 0, 1,
4.444445, 4.070707, -538.356, 1, 0.4823529, 0, 1,
4.484848, 4.070707, -537.5583, 1, 0.4823529, 0, 1,
4.525252, 4.070707, -536.7804, 1, 0.4823529, 0, 1,
4.565657, 4.070707, -536.0222, 1, 0.4823529, 0, 1,
4.606061, 4.070707, -535.2836, 1, 0.5843138, 0, 1,
4.646465, 4.070707, -534.5648, 1, 0.5843138, 0, 1,
4.686869, 4.070707, -533.8656, 1, 0.5843138, 0, 1,
4.727273, 4.070707, -533.1862, 1, 0.5843138, 0, 1,
4.767677, 4.070707, -532.5264, 1, 0.5843138, 0, 1,
4.808081, 4.070707, -531.8864, 1, 0.5843138, 0, 1,
4.848485, 4.070707, -531.2661, 1, 0.5843138, 0, 1,
4.888889, 4.070707, -530.6655, 1, 0.5843138, 0, 1,
4.929293, 4.070707, -530.0845, 1, 0.6862745, 0, 1,
4.969697, 4.070707, -529.5233, 1, 0.6862745, 0, 1,
5.010101, 4.070707, -528.9818, 1, 0.6862745, 0, 1,
5.050505, 4.070707, -528.46, 1, 0.6862745, 0, 1,
5.090909, 4.070707, -527.9579, 1, 0.6862745, 0, 1,
5.131313, 4.070707, -527.4755, 1, 0.6862745, 0, 1,
5.171717, 4.070707, -527.0128, 1, 0.6862745, 0, 1,
5.212121, 4.070707, -526.5698, 1, 0.6862745, 0, 1,
5.252525, 4.070707, -526.1465, 1, 0.6862745, 0, 1,
5.292929, 4.070707, -525.7429, 1, 0.6862745, 0, 1,
5.333333, 4.070707, -525.359, 1, 0.6862745, 0, 1,
5.373737, 4.070707, -524.9948, 1, 0.6862745, 0, 1,
5.414141, 4.070707, -524.6503, 1, 0.7921569, 0, 1,
5.454545, 4.070707, -524.3255, 1, 0.7921569, 0, 1,
5.494949, 4.070707, -524.0204, 1, 0.7921569, 0, 1,
5.535354, 4.070707, -523.7351, 1, 0.7921569, 0, 1,
5.575758, 4.070707, -523.4694, 1, 0.7921569, 0, 1,
5.616162, 4.070707, -523.2234, 1, 0.7921569, 0, 1,
5.656566, 4.070707, -522.9972, 1, 0.7921569, 0, 1,
5.69697, 4.070707, -522.7906, 1, 0.7921569, 0, 1,
5.737374, 4.070707, -522.6038, 1, 0.7921569, 0, 1,
5.777778, 4.070707, -522.4366, 1, 0.7921569, 0, 1,
5.818182, 4.070707, -522.2892, 1, 0.7921569, 0, 1,
5.858586, 4.070707, -522.1614, 1, 0.7921569, 0, 1,
5.89899, 4.070707, -522.0533, 1, 0.7921569, 0, 1,
5.939394, 4.070707, -521.965, 1, 0.7921569, 0, 1,
5.979798, 4.070707, -521.8964, 1, 0.7921569, 0, 1,
6.020202, 4.070707, -521.8475, 1, 0.7921569, 0, 1,
6.060606, 4.070707, -521.8182, 1, 0.7921569, 0, 1,
6.10101, 4.070707, -521.8087, 1, 0.7921569, 0, 1,
6.141414, 4.070707, -521.8188, 1, 0.7921569, 0, 1,
6.181818, 4.070707, -521.8488, 1, 0.7921569, 0, 1,
6.222222, 4.070707, -521.8983, 1, 0.7921569, 0, 1,
6.262626, 4.070707, -521.9677, 1, 0.7921569, 0, 1,
6.30303, 4.070707, -522.0566, 1, 0.7921569, 0, 1,
6.343434, 4.070707, -522.1653, 1, 0.7921569, 0, 1,
6.383838, 4.070707, -522.2937, 1, 0.7921569, 0, 1,
6.424242, 4.070707, -522.4418, 1, 0.7921569, 0, 1,
6.464646, 4.070707, -522.6096, 1, 0.7921569, 0, 1,
6.505051, 4.070707, -522.7971, 1, 0.7921569, 0, 1,
6.545455, 4.070707, -523.0043, 1, 0.7921569, 0, 1,
6.585859, 4.070707, -523.2313, 1, 0.7921569, 0, 1,
6.626263, 4.070707, -523.4778, 1, 0.7921569, 0, 1,
6.666667, 4.070707, -523.7442, 1, 0.7921569, 0, 1,
6.707071, 4.070707, -524.0302, 1, 0.7921569, 0, 1,
6.747475, 4.070707, -524.3359, 1, 0.7921569, 0, 1,
6.787879, 4.070707, -524.6614, 1, 0.7921569, 0, 1,
6.828283, 4.070707, -525.0065, 1, 0.6862745, 0, 1,
6.868687, 4.070707, -525.3713, 1, 0.6862745, 0, 1,
6.909091, 4.070707, -525.7559, 1, 0.6862745, 0, 1,
6.949495, 4.070707, -526.1601, 1, 0.6862745, 0, 1,
6.989899, 4.070707, -526.584, 1, 0.6862745, 0, 1,
7.030303, 4.070707, -527.0277, 1, 0.6862745, 0, 1,
7.070707, 4.070707, -527.491, 1, 0.6862745, 0, 1,
7.111111, 4.070707, -527.9741, 1, 0.6862745, 0, 1,
7.151515, 4.070707, -528.4769, 1, 0.6862745, 0, 1,
7.191919, 4.070707, -528.9993, 1, 0.6862745, 0, 1,
7.232323, 4.070707, -529.5415, 1, 0.6862745, 0, 1,
7.272727, 4.070707, -530.1034, 1, 0.6862745, 0, 1,
7.313131, 4.070707, -530.6849, 1, 0.5843138, 0, 1,
7.353535, 4.070707, -531.2862, 1, 0.5843138, 0, 1,
7.393939, 4.070707, -531.9072, 1, 0.5843138, 0, 1,
7.434343, 4.070707, -532.5479, 1, 0.5843138, 0, 1,
7.474748, 4.070707, -533.2083, 1, 0.5843138, 0, 1,
7.515152, 4.070707, -533.8884, 1, 0.5843138, 0, 1,
7.555555, 4.070707, -534.5881, 1, 0.5843138, 0, 1,
7.59596, 4.070707, -535.3077, 1, 0.5843138, 0, 1,
7.636364, 4.070707, -536.0469, 1, 0.4823529, 0, 1,
7.676768, 4.070707, -536.8058, 1, 0.4823529, 0, 1,
7.717172, 4.070707, -537.5844, 1, 0.4823529, 0, 1,
7.757576, 4.070707, -538.3827, 1, 0.4823529, 0, 1,
7.79798, 4.070707, -539.2007, 1, 0.4823529, 0, 1,
7.838384, 4.070707, -540.0384, 1, 0.4823529, 0, 1,
7.878788, 4.070707, -540.8958, 1, 0.4823529, 0, 1,
7.919192, 4.070707, -541.7729, 1, 0.3764706, 0, 1,
7.959596, 4.070707, -542.6698, 1, 0.3764706, 0, 1,
8, 4.070707, -543.5863, 1, 0.3764706, 0, 1,
4, 4.121212, -548.8536, 1, 0.2745098, 0, 1,
4.040404, 4.121212, -547.8639, 1, 0.2745098, 0, 1,
4.080808, 4.121212, -546.8934, 1, 0.2745098, 0, 1,
4.121212, 4.121212, -545.9422, 1, 0.3764706, 0, 1,
4.161616, 4.121212, -545.0102, 1, 0.3764706, 0, 1,
4.20202, 4.121212, -544.0974, 1, 0.3764706, 0, 1,
4.242424, 4.121212, -543.2038, 1, 0.3764706, 0, 1,
4.282828, 4.121212, -542.3295, 1, 0.3764706, 0, 1,
4.323232, 4.121212, -541.4743, 1, 0.3764706, 0, 1,
4.363636, 4.121212, -540.6384, 1, 0.4823529, 0, 1,
4.40404, 4.121212, -539.8218, 1, 0.4823529, 0, 1,
4.444445, 4.121212, -539.0243, 1, 0.4823529, 0, 1,
4.484848, 4.121212, -538.2461, 1, 0.4823529, 0, 1,
4.525252, 4.121212, -537.4871, 1, 0.4823529, 0, 1,
4.565657, 4.121212, -536.7473, 1, 0.4823529, 0, 1,
4.606061, 4.121212, -536.0267, 1, 0.4823529, 0, 1,
4.646465, 4.121212, -535.3254, 1, 0.5843138, 0, 1,
4.686869, 4.121212, -534.6432, 1, 0.5843138, 0, 1,
4.727273, 4.121212, -533.9804, 1, 0.5843138, 0, 1,
4.767677, 4.121212, -533.3367, 1, 0.5843138, 0, 1,
4.808081, 4.121212, -532.7123, 1, 0.5843138, 0, 1,
4.848485, 4.121212, -532.1071, 1, 0.5843138, 0, 1,
4.888889, 4.121212, -531.5211, 1, 0.5843138, 0, 1,
4.929293, 4.121212, -530.9543, 1, 0.5843138, 0, 1,
4.969697, 4.121212, -530.4067, 1, 0.5843138, 0, 1,
5.010101, 4.121212, -529.8784, 1, 0.6862745, 0, 1,
5.050505, 4.121212, -529.3693, 1, 0.6862745, 0, 1,
5.090909, 4.121212, -528.8795, 1, 0.6862745, 0, 1,
5.131313, 4.121212, -528.4088, 1, 0.6862745, 0, 1,
5.171717, 4.121212, -527.9573, 1, 0.6862745, 0, 1,
5.212121, 4.121212, -527.5251, 1, 0.6862745, 0, 1,
5.252525, 4.121212, -527.1121, 1, 0.6862745, 0, 1,
5.292929, 4.121212, -526.7184, 1, 0.6862745, 0, 1,
5.333333, 4.121212, -526.3439, 1, 0.6862745, 0, 1,
5.373737, 4.121212, -525.9885, 1, 0.6862745, 0, 1,
5.414141, 4.121212, -525.6525, 1, 0.6862745, 0, 1,
5.454545, 4.121212, -525.3356, 1, 0.6862745, 0, 1,
5.494949, 4.121212, -525.0379, 1, 0.6862745, 0, 1,
5.535354, 4.121212, -524.7595, 1, 0.7921569, 0, 1,
5.575758, 4.121212, -524.5003, 1, 0.7921569, 0, 1,
5.616162, 4.121212, -524.2603, 1, 0.7921569, 0, 1,
5.656566, 4.121212, -524.0396, 1, 0.7921569, 0, 1,
5.69697, 4.121212, -523.8381, 1, 0.7921569, 0, 1,
5.737374, 4.121212, -523.6558, 1, 0.7921569, 0, 1,
5.777778, 4.121212, -523.4927, 1, 0.7921569, 0, 1,
5.818182, 4.121212, -523.3488, 1, 0.7921569, 0, 1,
5.858586, 4.121212, -523.2242, 1, 0.7921569, 0, 1,
5.89899, 4.121212, -523.1188, 1, 0.7921569, 0, 1,
5.939394, 4.121212, -523.0326, 1, 0.7921569, 0, 1,
5.979798, 4.121212, -522.9656, 1, 0.7921569, 0, 1,
6.020202, 4.121212, -522.9178, 1, 0.7921569, 0, 1,
6.060606, 4.121212, -522.8893, 1, 0.7921569, 0, 1,
6.10101, 4.121212, -522.8801, 1, 0.7921569, 0, 1,
6.141414, 4.121212, -522.89, 1, 0.7921569, 0, 1,
6.181818, 4.121212, -522.9191, 1, 0.7921569, 0, 1,
6.222222, 4.121212, -522.9675, 1, 0.7921569, 0, 1,
6.262626, 4.121212, -523.0351, 1, 0.7921569, 0, 1,
6.30303, 4.121212, -523.1219, 1, 0.7921569, 0, 1,
6.343434, 4.121212, -523.228, 1, 0.7921569, 0, 1,
6.383838, 4.121212, -523.3532, 1, 0.7921569, 0, 1,
6.424242, 4.121212, -523.4977, 1, 0.7921569, 0, 1,
6.464646, 4.121212, -523.6614, 1, 0.7921569, 0, 1,
6.505051, 4.121212, -523.8444, 1, 0.7921569, 0, 1,
6.545455, 4.121212, -524.0466, 1, 0.7921569, 0, 1,
6.585859, 4.121212, -524.2679, 1, 0.7921569, 0, 1,
6.626263, 4.121212, -524.5085, 1, 0.7921569, 0, 1,
6.666667, 4.121212, -524.7684, 1, 0.7921569, 0, 1,
6.707071, 4.121212, -525.0474, 1, 0.6862745, 0, 1,
6.747475, 4.121212, -525.3457, 1, 0.6862745, 0, 1,
6.787879, 4.121212, -525.6632, 1, 0.6862745, 0, 1,
6.828283, 4.121212, -525.9999, 1, 0.6862745, 0, 1,
6.868687, 4.121212, -526.3559, 1, 0.6862745, 0, 1,
6.909091, 4.121212, -526.7311, 1, 0.6862745, 0, 1,
6.949495, 4.121212, -527.1254, 1, 0.6862745, 0, 1,
6.989899, 4.121212, -527.5391, 1, 0.6862745, 0, 1,
7.030303, 4.121212, -527.9719, 1, 0.6862745, 0, 1,
7.070707, 4.121212, -528.424, 1, 0.6862745, 0, 1,
7.111111, 4.121212, -528.8953, 1, 0.6862745, 0, 1,
7.151515, 4.121212, -529.3858, 1, 0.6862745, 0, 1,
7.191919, 4.121212, -529.8955, 1, 0.6862745, 0, 1,
7.232323, 4.121212, -530.4245, 1, 0.5843138, 0, 1,
7.272727, 4.121212, -530.9727, 1, 0.5843138, 0, 1,
7.313131, 4.121212, -531.5401, 1, 0.5843138, 0, 1,
7.353535, 4.121212, -532.1267, 1, 0.5843138, 0, 1,
7.393939, 4.121212, -532.7325, 1, 0.5843138, 0, 1,
7.434343, 4.121212, -533.3576, 1, 0.5843138, 0, 1,
7.474748, 4.121212, -534.002, 1, 0.5843138, 0, 1,
7.515152, 4.121212, -534.6655, 1, 0.5843138, 0, 1,
7.555555, 4.121212, -535.3482, 1, 0.5843138, 0, 1,
7.59596, 4.121212, -536.0502, 1, 0.4823529, 0, 1,
7.636364, 4.121212, -536.7714, 1, 0.4823529, 0, 1,
7.676768, 4.121212, -537.5118, 1, 0.4823529, 0, 1,
7.717172, 4.121212, -538.2714, 1, 0.4823529, 0, 1,
7.757576, 4.121212, -539.0503, 1, 0.4823529, 0, 1,
7.79798, 4.121212, -539.8484, 1, 0.4823529, 0, 1,
7.838384, 4.121212, -540.6657, 1, 0.4823529, 0, 1,
7.878788, 4.121212, -541.5022, 1, 0.3764706, 0, 1,
7.919192, 4.121212, -542.358, 1, 0.3764706, 0, 1,
7.959596, 4.121212, -543.233, 1, 0.3764706, 0, 1,
8, 4.121212, -544.1271, 1, 0.3764706, 0, 1,
4, 4.171717, -549.3201, 1, 0.2745098, 0, 1,
4.040404, 4.171717, -548.3542, 1, 0.2745098, 0, 1,
4.080808, 4.171717, -547.4072, 1, 0.2745098, 0, 1,
4.121212, 4.171717, -546.4788, 1, 0.2745098, 0, 1,
4.161616, 4.171717, -545.5692, 1, 0.3764706, 0, 1,
4.20202, 4.171717, -544.6784, 1, 0.3764706, 0, 1,
4.242424, 4.171717, -543.8063, 1, 0.3764706, 0, 1,
4.282828, 4.171717, -542.9531, 1, 0.3764706, 0, 1,
4.323232, 4.171717, -542.1185, 1, 0.3764706, 0, 1,
4.363636, 4.171717, -541.3027, 1, 0.3764706, 0, 1,
4.40404, 4.171717, -540.5057, 1, 0.4823529, 0, 1,
4.444445, 4.171717, -539.7274, 1, 0.4823529, 0, 1,
4.484848, 4.171717, -538.9679, 1, 0.4823529, 0, 1,
4.525252, 4.171717, -538.2272, 1, 0.4823529, 0, 1,
4.565657, 4.171717, -537.5052, 1, 0.4823529, 0, 1,
4.606061, 4.171717, -536.802, 1, 0.4823529, 0, 1,
4.646465, 4.171717, -536.1176, 1, 0.4823529, 0, 1,
4.686869, 4.171717, -535.4518, 1, 0.5843138, 0, 1,
4.727273, 4.171717, -534.8049, 1, 0.5843138, 0, 1,
4.767677, 4.171717, -534.1767, 1, 0.5843138, 0, 1,
4.808081, 4.171717, -533.5673, 1, 0.5843138, 0, 1,
4.848485, 4.171717, -532.9766, 1, 0.5843138, 0, 1,
4.888889, 4.171717, -532.4048, 1, 0.5843138, 0, 1,
4.929293, 4.171717, -531.8516, 1, 0.5843138, 0, 1,
4.969697, 4.171717, -531.3173, 1, 0.5843138, 0, 1,
5.010101, 4.171717, -530.8016, 1, 0.5843138, 0, 1,
5.050505, 4.171717, -530.3048, 1, 0.5843138, 0, 1,
5.090909, 4.171717, -529.8267, 1, 0.6862745, 0, 1,
5.131313, 4.171717, -529.3674, 1, 0.6862745, 0, 1,
5.171717, 4.171717, -528.9268, 1, 0.6862745, 0, 1,
5.212121, 4.171717, -528.505, 1, 0.6862745, 0, 1,
5.252525, 4.171717, -528.1019, 1, 0.6862745, 0, 1,
5.292929, 4.171717, -527.7177, 1, 0.6862745, 0, 1,
5.333333, 4.171717, -527.3522, 1, 0.6862745, 0, 1,
5.373737, 4.171717, -527.0054, 1, 0.6862745, 0, 1,
5.414141, 4.171717, -526.6774, 1, 0.6862745, 0, 1,
5.454545, 4.171717, -526.3681, 1, 0.6862745, 0, 1,
5.494949, 4.171717, -526.0776, 1, 0.6862745, 0, 1,
5.535354, 4.171717, -525.8059, 1, 0.6862745, 0, 1,
5.575758, 4.171717, -525.553, 1, 0.6862745, 0, 1,
5.616162, 4.171717, -525.3188, 1, 0.6862745, 0, 1,
5.656566, 4.171717, -525.1033, 1, 0.6862745, 0, 1,
5.69697, 4.171717, -524.9067, 1, 0.7921569, 0, 1,
5.737374, 4.171717, -524.7288, 1, 0.7921569, 0, 1,
5.777778, 4.171717, -524.5696, 1, 0.7921569, 0, 1,
5.818182, 4.171717, -524.4292, 1, 0.7921569, 0, 1,
5.858586, 4.171717, -524.3076, 1, 0.7921569, 0, 1,
5.89899, 4.171717, -524.2047, 1, 0.7921569, 0, 1,
5.939394, 4.171717, -524.1205, 1, 0.7921569, 0, 1,
5.979798, 4.171717, -524.0552, 1, 0.7921569, 0, 1,
6.020202, 4.171717, -524.0086, 1, 0.7921569, 0, 1,
6.060606, 4.171717, -523.9808, 1, 0.7921569, 0, 1,
6.10101, 4.171717, -523.9717, 1, 0.7921569, 0, 1,
6.141414, 4.171717, -523.9814, 1, 0.7921569, 0, 1,
6.181818, 4.171717, -524.0098, 1, 0.7921569, 0, 1,
6.222222, 4.171717, -524.0571, 1, 0.7921569, 0, 1,
6.262626, 4.171717, -524.123, 1, 0.7921569, 0, 1,
6.30303, 4.171717, -524.2078, 1, 0.7921569, 0, 1,
6.343434, 4.171717, -524.3113, 1, 0.7921569, 0, 1,
6.383838, 4.171717, -524.4335, 1, 0.7921569, 0, 1,
6.424242, 4.171717, -524.5745, 1, 0.7921569, 0, 1,
6.464646, 4.171717, -524.7343, 1, 0.7921569, 0, 1,
6.505051, 4.171717, -524.9128, 1, 0.6862745, 0, 1,
6.545455, 4.171717, -525.1101, 1, 0.6862745, 0, 1,
6.585859, 4.171717, -525.3262, 1, 0.6862745, 0, 1,
6.626263, 4.171717, -525.561, 1, 0.6862745, 0, 1,
6.666667, 4.171717, -525.8146, 1, 0.6862745, 0, 1,
6.707071, 4.171717, -526.0869, 1, 0.6862745, 0, 1,
6.747475, 4.171717, -526.3781, 1, 0.6862745, 0, 1,
6.787879, 4.171717, -526.6879, 1, 0.6862745, 0, 1,
6.828283, 4.171717, -527.0165, 1, 0.6862745, 0, 1,
6.868687, 4.171717, -527.3639, 1, 0.6862745, 0, 1,
6.909091, 4.171717, -527.73, 1, 0.6862745, 0, 1,
6.949495, 4.171717, -528.1149, 1, 0.6862745, 0, 1,
6.989899, 4.171717, -528.5186, 1, 0.6862745, 0, 1,
7.030303, 4.171717, -528.941, 1, 0.6862745, 0, 1,
7.070707, 4.171717, -529.3822, 1, 0.6862745, 0, 1,
7.111111, 4.171717, -529.8422, 1, 0.6862745, 0, 1,
7.151515, 4.171717, -530.3209, 1, 0.5843138, 0, 1,
7.191919, 4.171717, -530.8184, 1, 0.5843138, 0, 1,
7.232323, 4.171717, -531.3346, 1, 0.5843138, 0, 1,
7.272727, 4.171717, -531.8696, 1, 0.5843138, 0, 1,
7.313131, 4.171717, -532.4233, 1, 0.5843138, 0, 1,
7.353535, 4.171717, -532.9958, 1, 0.5843138, 0, 1,
7.393939, 4.171717, -533.5871, 1, 0.5843138, 0, 1,
7.434343, 4.171717, -534.1971, 1, 0.5843138, 0, 1,
7.474748, 4.171717, -534.8259, 1, 0.5843138, 0, 1,
7.515152, 4.171717, -535.4735, 1, 0.5843138, 0, 1,
7.555555, 4.171717, -536.1398, 1, 0.4823529, 0, 1,
7.59596, 4.171717, -536.8249, 1, 0.4823529, 0, 1,
7.636364, 4.171717, -537.5287, 1, 0.4823529, 0, 1,
7.676768, 4.171717, -538.2513, 1, 0.4823529, 0, 1,
7.717172, 4.171717, -538.9927, 1, 0.4823529, 0, 1,
7.757576, 4.171717, -539.7528, 1, 0.4823529, 0, 1,
7.79798, 4.171717, -540.5317, 1, 0.4823529, 0, 1,
7.838384, 4.171717, -541.3293, 1, 0.3764706, 0, 1,
7.878788, 4.171717, -542.1457, 1, 0.3764706, 0, 1,
7.919192, 4.171717, -542.9809, 1, 0.3764706, 0, 1,
7.959596, 4.171717, -543.8348, 1, 0.3764706, 0, 1,
8, 4.171717, -544.7075, 1, 0.3764706, 0, 1,
4, 4.222222, -549.8276, 1, 0.2745098, 0, 1,
4.040404, 4.222222, -548.8848, 1, 0.2745098, 0, 1,
4.080808, 4.222222, -547.9601, 1, 0.2745098, 0, 1,
4.121212, 4.222222, -547.0539, 1, 0.2745098, 0, 1,
4.161616, 4.222222, -546.166, 1, 0.3764706, 0, 1,
4.20202, 4.222222, -545.2963, 1, 0.3764706, 0, 1,
4.242424, 4.222222, -544.4449, 1, 0.3764706, 0, 1,
4.282828, 4.222222, -543.6119, 1, 0.3764706, 0, 1,
4.323232, 4.222222, -542.7972, 1, 0.3764706, 0, 1,
4.363636, 4.222222, -542.0009, 1, 0.3764706, 0, 1,
4.40404, 4.222222, -541.2228, 1, 0.3764706, 0, 1,
4.444445, 4.222222, -540.4631, 1, 0.4823529, 0, 1,
4.484848, 4.222222, -539.7216, 1, 0.4823529, 0, 1,
4.525252, 4.222222, -538.9985, 1, 0.4823529, 0, 1,
4.565657, 4.222222, -538.2937, 1, 0.4823529, 0, 1,
4.606061, 4.222222, -537.6072, 1, 0.4823529, 0, 1,
4.646465, 4.222222, -536.939, 1, 0.4823529, 0, 1,
4.686869, 4.222222, -536.2891, 1, 0.4823529, 0, 1,
4.727273, 4.222222, -535.6576, 1, 0.4823529, 0, 1,
4.767677, 4.222222, -535.0443, 1, 0.5843138, 0, 1,
4.808081, 4.222222, -534.4494, 1, 0.5843138, 0, 1,
4.848485, 4.222222, -533.8728, 1, 0.5843138, 0, 1,
4.888889, 4.222222, -533.3145, 1, 0.5843138, 0, 1,
4.929293, 4.222222, -532.7745, 1, 0.5843138, 0, 1,
4.969697, 4.222222, -532.2529, 1, 0.5843138, 0, 1,
5.010101, 4.222222, -531.7495, 1, 0.5843138, 0, 1,
5.050505, 4.222222, -531.2645, 1, 0.5843138, 0, 1,
5.090909, 4.222222, -530.7978, 1, 0.5843138, 0, 1,
5.131313, 4.222222, -530.3494, 1, 0.5843138, 0, 1,
5.171717, 4.222222, -529.9193, 1, 0.6862745, 0, 1,
5.212121, 4.222222, -529.5075, 1, 0.6862745, 0, 1,
5.252525, 4.222222, -529.114, 1, 0.6862745, 0, 1,
5.292929, 4.222222, -528.7389, 1, 0.6862745, 0, 1,
5.333333, 4.222222, -528.3821, 1, 0.6862745, 0, 1,
5.373737, 4.222222, -528.0435, 1, 0.6862745, 0, 1,
5.414141, 4.222222, -527.7233, 1, 0.6862745, 0, 1,
5.454545, 4.222222, -527.4214, 1, 0.6862745, 0, 1,
5.494949, 4.222222, -527.1379, 1, 0.6862745, 0, 1,
5.535354, 4.222222, -526.8726, 1, 0.6862745, 0, 1,
5.575758, 4.222222, -526.6257, 1, 0.6862745, 0, 1,
5.616162, 4.222222, -526.397, 1, 0.6862745, 0, 1,
5.656566, 4.222222, -526.1867, 1, 0.6862745, 0, 1,
5.69697, 4.222222, -525.9947, 1, 0.6862745, 0, 1,
5.737374, 4.222222, -525.821, 1, 0.6862745, 0, 1,
5.777778, 4.222222, -525.6656, 1, 0.6862745, 0, 1,
5.818182, 4.222222, -525.5286, 1, 0.6862745, 0, 1,
5.858586, 4.222222, -525.4099, 1, 0.6862745, 0, 1,
5.89899, 4.222222, -525.3094, 1, 0.6862745, 0, 1,
5.939394, 4.222222, -525.2273, 1, 0.6862745, 0, 1,
5.979798, 4.222222, -525.1635, 1, 0.6862745, 0, 1,
6.020202, 4.222222, -525.118, 1, 0.6862745, 0, 1,
6.060606, 4.222222, -525.0909, 1, 0.6862745, 0, 1,
6.10101, 4.222222, -525.082, 1, 0.6862745, 0, 1,
6.141414, 4.222222, -525.0915, 1, 0.6862745, 0, 1,
6.181818, 4.222222, -525.1193, 1, 0.6862745, 0, 1,
6.222222, 4.222222, -525.1653, 1, 0.6862745, 0, 1,
6.262626, 4.222222, -525.2297, 1, 0.6862745, 0, 1,
6.30303, 4.222222, -525.3124, 1, 0.6862745, 0, 1,
6.343434, 4.222222, -525.4135, 1, 0.6862745, 0, 1,
6.383838, 4.222222, -525.5328, 1, 0.6862745, 0, 1,
6.424242, 4.222222, -525.6705, 1, 0.6862745, 0, 1,
6.464646, 4.222222, -525.8265, 1, 0.6862745, 0, 1,
6.505051, 4.222222, -526.0007, 1, 0.6862745, 0, 1,
6.545455, 4.222222, -526.1934, 1, 0.6862745, 0, 1,
6.585859, 4.222222, -526.4043, 1, 0.6862745, 0, 1,
6.626263, 4.222222, -526.6335, 1, 0.6862745, 0, 1,
6.666667, 4.222222, -526.881, 1, 0.6862745, 0, 1,
6.707071, 4.222222, -527.1469, 1, 0.6862745, 0, 1,
6.747475, 4.222222, -527.4311, 1, 0.6862745, 0, 1,
6.787879, 4.222222, -527.7336, 1, 0.6862745, 0, 1,
6.828283, 4.222222, -528.0544, 1, 0.6862745, 0, 1,
6.868687, 4.222222, -528.3936, 1, 0.6862745, 0, 1,
6.909091, 4.222222, -528.751, 1, 0.6862745, 0, 1,
6.949495, 4.222222, -529.1267, 1, 0.6862745, 0, 1,
6.989899, 4.222222, -529.5208, 1, 0.6862745, 0, 1,
7.030303, 4.222222, -529.9332, 1, 0.6862745, 0, 1,
7.070707, 4.222222, -530.3638, 1, 0.5843138, 0, 1,
7.111111, 4.222222, -530.8129, 1, 0.5843138, 0, 1,
7.151515, 4.222222, -531.2802, 1, 0.5843138, 0, 1,
7.191919, 4.222222, -531.7658, 1, 0.5843138, 0, 1,
7.232323, 4.222222, -532.2698, 1, 0.5843138, 0, 1,
7.272727, 4.222222, -532.7921, 1, 0.5843138, 0, 1,
7.313131, 4.222222, -533.3326, 1, 0.5843138, 0, 1,
7.353535, 4.222222, -533.8915, 1, 0.5843138, 0, 1,
7.393939, 4.222222, -534.4688, 1, 0.5843138, 0, 1,
7.434343, 4.222222, -535.0643, 1, 0.5843138, 0, 1,
7.474748, 4.222222, -535.6781, 1, 0.4823529, 0, 1,
7.515152, 4.222222, -536.3102, 1, 0.4823529, 0, 1,
7.555555, 4.222222, -536.9608, 1, 0.4823529, 0, 1,
7.59596, 4.222222, -537.6295, 1, 0.4823529, 0, 1,
7.636364, 4.222222, -538.3166, 1, 0.4823529, 0, 1,
7.676768, 4.222222, -539.022, 1, 0.4823529, 0, 1,
7.717172, 4.222222, -539.7458, 1, 0.4823529, 0, 1,
7.757576, 4.222222, -540.4878, 1, 0.4823529, 0, 1,
7.79798, 4.222222, -541.2482, 1, 0.3764706, 0, 1,
7.838384, 4.222222, -542.0269, 1, 0.3764706, 0, 1,
7.878788, 4.222222, -542.8239, 1, 0.3764706, 0, 1,
7.919192, 4.222222, -543.6391, 1, 0.3764706, 0, 1,
7.959596, 4.222222, -544.4727, 1, 0.3764706, 0, 1,
8, 4.222222, -545.3246, 1, 0.3764706, 0, 1,
4, 4.272727, -550.3735, 1, 0.2745098, 0, 1,
4.040404, 4.272727, -549.4528, 1, 0.2745098, 0, 1,
4.080808, 4.272727, -548.5499, 1, 0.2745098, 0, 1,
4.121212, 4.272727, -547.6649, 1, 0.2745098, 0, 1,
4.161616, 4.272727, -546.7979, 1, 0.2745098, 0, 1,
4.20202, 4.272727, -545.9486, 1, 0.3764706, 0, 1,
4.242424, 4.272727, -545.1173, 1, 0.3764706, 0, 1,
4.282828, 4.272727, -544.3038, 1, 0.3764706, 0, 1,
4.323232, 4.272727, -543.5083, 1, 0.3764706, 0, 1,
4.363636, 4.272727, -542.7307, 1, 0.3764706, 0, 1,
4.40404, 4.272727, -541.9709, 1, 0.3764706, 0, 1,
4.444445, 4.272727, -541.2289, 1, 0.3764706, 0, 1,
4.484848, 4.272727, -540.5049, 1, 0.4823529, 0, 1,
4.525252, 4.272727, -539.7988, 1, 0.4823529, 0, 1,
4.565657, 4.272727, -539.1106, 1, 0.4823529, 0, 1,
4.606061, 4.272727, -538.4402, 1, 0.4823529, 0, 1,
4.646465, 4.272727, -537.7877, 1, 0.4823529, 0, 1,
4.686869, 4.272727, -537.1531, 1, 0.4823529, 0, 1,
4.727273, 4.272727, -536.5364, 1, 0.4823529, 0, 1,
4.767677, 4.272727, -535.9376, 1, 0.4823529, 0, 1,
4.808081, 4.272727, -535.3566, 1, 0.5843138, 0, 1,
4.848485, 4.272727, -534.7936, 1, 0.5843138, 0, 1,
4.888889, 4.272727, -534.2484, 1, 0.5843138, 0, 1,
4.929293, 4.272727, -533.7211, 1, 0.5843138, 0, 1,
4.969697, 4.272727, -533.2117, 1, 0.5843138, 0, 1,
5.010101, 4.272727, -532.7202, 1, 0.5843138, 0, 1,
5.050505, 4.272727, -532.2466, 1, 0.5843138, 0, 1,
5.090909, 4.272727, -531.7908, 1, 0.5843138, 0, 1,
5.131313, 4.272727, -531.353, 1, 0.5843138, 0, 1,
5.171717, 4.272727, -530.933, 1, 0.5843138, 0, 1,
5.212121, 4.272727, -530.5309, 1, 0.5843138, 0, 1,
5.252525, 4.272727, -530.1467, 1, 0.6862745, 0, 1,
5.292929, 4.272727, -529.7803, 1, 0.6862745, 0, 1,
5.333333, 4.272727, -529.4319, 1, 0.6862745, 0, 1,
5.373737, 4.272727, -529.1013, 1, 0.6862745, 0, 1,
5.414141, 4.272727, -528.7886, 1, 0.6862745, 0, 1,
5.454545, 4.272727, -528.4938, 1, 0.6862745, 0, 1,
5.494949, 4.272727, -528.2169, 1, 0.6862745, 0, 1,
5.535354, 4.272727, -527.9579, 1, 0.6862745, 0, 1,
5.575758, 4.272727, -527.7168, 1, 0.6862745, 0, 1,
5.616162, 4.272727, -527.4935, 1, 0.6862745, 0, 1,
5.656566, 4.272727, -527.2881, 1, 0.6862745, 0, 1,
5.69697, 4.272727, -527.1006, 1, 0.6862745, 0, 1,
5.737374, 4.272727, -526.931, 1, 0.6862745, 0, 1,
5.777778, 4.272727, -526.7793, 1, 0.6862745, 0, 1,
5.818182, 4.272727, -526.6455, 1, 0.6862745, 0, 1,
5.858586, 4.272727, -526.5295, 1, 0.6862745, 0, 1,
5.89899, 4.272727, -526.4315, 1, 0.6862745, 0, 1,
5.939394, 4.272727, -526.3513, 1, 0.6862745, 0, 1,
5.979798, 4.272727, -526.289, 1, 0.6862745, 0, 1,
6.020202, 4.272727, -526.2446, 1, 0.6862745, 0, 1,
6.060606, 4.272727, -526.218, 1, 0.6862745, 0, 1,
6.10101, 4.272727, -526.2094, 1, 0.6862745, 0, 1,
6.141414, 4.272727, -526.2186, 1, 0.6862745, 0, 1,
6.181818, 4.272727, -526.2457, 1, 0.6862745, 0, 1,
6.222222, 4.272727, -526.2908, 1, 0.6862745, 0, 1,
6.262626, 4.272727, -526.3536, 1, 0.6862745, 0, 1,
6.30303, 4.272727, -526.4344, 1, 0.6862745, 0, 1,
6.343434, 4.272727, -526.5331, 1, 0.6862745, 0, 1,
6.383838, 4.272727, -526.6496, 1, 0.6862745, 0, 1,
6.424242, 4.272727, -526.7841, 1, 0.6862745, 0, 1,
6.464646, 4.272727, -526.9363, 1, 0.6862745, 0, 1,
6.505051, 4.272727, -527.1066, 1, 0.6862745, 0, 1,
6.545455, 4.272727, -527.2946, 1, 0.6862745, 0, 1,
6.585859, 4.272727, -527.5006, 1, 0.6862745, 0, 1,
6.626263, 4.272727, -527.7244, 1, 0.6862745, 0, 1,
6.666667, 4.272727, -527.9662, 1, 0.6862745, 0, 1,
6.707071, 4.272727, -528.2258, 1, 0.6862745, 0, 1,
6.747475, 4.272727, -528.5033, 1, 0.6862745, 0, 1,
6.787879, 4.272727, -528.7986, 1, 0.6862745, 0, 1,
6.828283, 4.272727, -529.1119, 1, 0.6862745, 0, 1,
6.868687, 4.272727, -529.4431, 1, 0.6862745, 0, 1,
6.909091, 4.272727, -529.7921, 1, 0.6862745, 0, 1,
6.949495, 4.272727, -530.1591, 1, 0.6862745, 0, 1,
6.989899, 4.272727, -530.5438, 1, 0.5843138, 0, 1,
7.030303, 4.272727, -530.9465, 1, 0.5843138, 0, 1,
7.070707, 4.272727, -531.3671, 1, 0.5843138, 0, 1,
7.111111, 4.272727, -531.8055, 1, 0.5843138, 0, 1,
7.151515, 4.272727, -532.2619, 1, 0.5843138, 0, 1,
7.191919, 4.272727, -532.7361, 1, 0.5843138, 0, 1,
7.232323, 4.272727, -533.2283, 1, 0.5843138, 0, 1,
7.272727, 4.272727, -533.7382, 1, 0.5843138, 0, 1,
7.313131, 4.272727, -534.2661, 1, 0.5843138, 0, 1,
7.353535, 4.272727, -534.8119, 1, 0.5843138, 0, 1,
7.393939, 4.272727, -535.3755, 1, 0.5843138, 0, 1,
7.434343, 4.272727, -535.957, 1, 0.4823529, 0, 1,
7.474748, 4.272727, -536.5565, 1, 0.4823529, 0, 1,
7.515152, 4.272727, -537.1738, 1, 0.4823529, 0, 1,
7.555555, 4.272727, -537.809, 1, 0.4823529, 0, 1,
7.59596, 4.272727, -538.462, 1, 0.4823529, 0, 1,
7.636364, 4.272727, -539.133, 1, 0.4823529, 0, 1,
7.676768, 4.272727, -539.8218, 1, 0.4823529, 0, 1,
7.717172, 4.272727, -540.5285, 1, 0.4823529, 0, 1,
7.757576, 4.272727, -541.2531, 1, 0.3764706, 0, 1,
7.79798, 4.272727, -541.9956, 1, 0.3764706, 0, 1,
7.838384, 4.272727, -542.756, 1, 0.3764706, 0, 1,
7.878788, 4.272727, -543.5342, 1, 0.3764706, 0, 1,
7.919192, 4.272727, -544.3304, 1, 0.3764706, 0, 1,
7.959596, 4.272727, -545.1444, 1, 0.3764706, 0, 1,
8, 4.272727, -545.9763, 1, 0.3764706, 0, 1,
4, 4.323232, -550.9552, 1, 0.2745098, 0, 1,
4.040404, 4.323232, -550.0558, 1, 0.2745098, 0, 1,
4.080808, 4.323232, -549.174, 1, 0.2745098, 0, 1,
4.121212, 4.323232, -548.3096, 1, 0.2745098, 0, 1,
4.161616, 4.323232, -547.4626, 1, 0.2745098, 0, 1,
4.20202, 4.323232, -546.6331, 1, 0.2745098, 0, 1,
4.242424, 4.323232, -545.8211, 1, 0.3764706, 0, 1,
4.282828, 4.323232, -545.0266, 1, 0.3764706, 0, 1,
4.323232, 4.323232, -544.2495, 1, 0.3764706, 0, 1,
4.363636, 4.323232, -543.4899, 1, 0.3764706, 0, 1,
4.40404, 4.323232, -542.7477, 1, 0.3764706, 0, 1,
4.444445, 4.323232, -542.0231, 1, 0.3764706, 0, 1,
4.484848, 4.323232, -541.3159, 1, 0.3764706, 0, 1,
4.525252, 4.323232, -540.6262, 1, 0.4823529, 0, 1,
4.565657, 4.323232, -539.9539, 1, 0.4823529, 0, 1,
4.606061, 4.323232, -539.2991, 1, 0.4823529, 0, 1,
4.646465, 4.323232, -538.6618, 1, 0.4823529, 0, 1,
4.686869, 4.323232, -538.0419, 1, 0.4823529, 0, 1,
4.727273, 4.323232, -537.4396, 1, 0.4823529, 0, 1,
4.767677, 4.323232, -536.8547, 1, 0.4823529, 0, 1,
4.808081, 4.323232, -536.2872, 1, 0.4823529, 0, 1,
4.848485, 4.323232, -535.7372, 1, 0.4823529, 0, 1,
4.888889, 4.323232, -535.2047, 1, 0.5843138, 0, 1,
4.929293, 4.323232, -534.6897, 1, 0.5843138, 0, 1,
4.969697, 4.323232, -534.1921, 1, 0.5843138, 0, 1,
5.010101, 4.323232, -533.712, 1, 0.5843138, 0, 1,
5.050505, 4.323232, -533.2494, 1, 0.5843138, 0, 1,
5.090909, 4.323232, -532.8042, 1, 0.5843138, 0, 1,
5.131313, 4.323232, -532.3765, 1, 0.5843138, 0, 1,
5.171717, 4.323232, -531.9662, 1, 0.5843138, 0, 1,
5.212121, 4.323232, -531.5735, 1, 0.5843138, 0, 1,
5.252525, 4.323232, -531.1982, 1, 0.5843138, 0, 1,
5.292929, 4.323232, -530.8404, 1, 0.5843138, 0, 1,
5.333333, 4.323232, -530.5001, 1, 0.5843138, 0, 1,
5.373737, 4.323232, -530.1772, 1, 0.6862745, 0, 1,
5.414141, 4.323232, -529.8718, 1, 0.6862745, 0, 1,
5.454545, 4.323232, -529.5838, 1, 0.6862745, 0, 1,
5.494949, 4.323232, -529.3133, 1, 0.6862745, 0, 1,
5.535354, 4.323232, -529.0603, 1, 0.6862745, 0, 1,
5.575758, 4.323232, -528.8248, 1, 0.6862745, 0, 1,
5.616162, 4.323232, -528.6067, 1, 0.6862745, 0, 1,
5.656566, 4.323232, -528.4061, 1, 0.6862745, 0, 1,
5.69697, 4.323232, -528.223, 1, 0.6862745, 0, 1,
5.737374, 4.323232, -528.0573, 1, 0.6862745, 0, 1,
5.777778, 4.323232, -527.9091, 1, 0.6862745, 0, 1,
5.818182, 4.323232, -527.7784, 1, 0.6862745, 0, 1,
5.858586, 4.323232, -527.6651, 1, 0.6862745, 0, 1,
5.89899, 4.323232, -527.5693, 1, 0.6862745, 0, 1,
5.939394, 4.323232, -527.491, 1, 0.6862745, 0, 1,
5.979798, 4.323232, -527.4302, 1, 0.6862745, 0, 1,
6.020202, 4.323232, -527.3868, 1, 0.6862745, 0, 1,
6.060606, 4.323232, -527.3608, 1, 0.6862745, 0, 1,
6.10101, 4.323232, -527.3524, 1, 0.6862745, 0, 1,
6.141414, 4.323232, -527.3615, 1, 0.6862745, 0, 1,
6.181818, 4.323232, -527.3879, 1, 0.6862745, 0, 1,
6.222222, 4.323232, -527.4319, 1, 0.6862745, 0, 1,
6.262626, 4.323232, -527.4933, 1, 0.6862745, 0, 1,
6.30303, 4.323232, -527.5722, 1, 0.6862745, 0, 1,
6.343434, 4.323232, -527.6686, 1, 0.6862745, 0, 1,
6.383838, 4.323232, -527.7824, 1, 0.6862745, 0, 1,
6.424242, 4.323232, -527.9137, 1, 0.6862745, 0, 1,
6.464646, 4.323232, -528.0625, 1, 0.6862745, 0, 1,
6.505051, 4.323232, -528.2287, 1, 0.6862745, 0, 1,
6.545455, 4.323232, -528.4124, 1, 0.6862745, 0, 1,
6.585859, 4.323232, -528.6136, 1, 0.6862745, 0, 1,
6.626263, 4.323232, -528.8323, 1, 0.6862745, 0, 1,
6.666667, 4.323232, -529.0684, 1, 0.6862745, 0, 1,
6.707071, 4.323232, -529.322, 1, 0.6862745, 0, 1,
6.747475, 4.323232, -529.593, 1, 0.6862745, 0, 1,
6.787879, 4.323232, -529.8815, 1, 0.6862745, 0, 1,
6.828283, 4.323232, -530.1875, 1, 0.6862745, 0, 1,
6.868687, 4.323232, -530.511, 1, 0.5843138, 0, 1,
6.909091, 4.323232, -530.8519, 1, 0.5843138, 0, 1,
6.949495, 4.323232, -531.2103, 1, 0.5843138, 0, 1,
6.989899, 4.323232, -531.5862, 1, 0.5843138, 0, 1,
7.030303, 4.323232, -531.9795, 1, 0.5843138, 0, 1,
7.070707, 4.323232, -532.3903, 1, 0.5843138, 0, 1,
7.111111, 4.323232, -532.8186, 1, 0.5843138, 0, 1,
7.151515, 4.323232, -533.2643, 1, 0.5843138, 0, 1,
7.191919, 4.323232, -533.7275, 1, 0.5843138, 0, 1,
7.232323, 4.323232, -534.2083, 1, 0.5843138, 0, 1,
7.272727, 4.323232, -534.7064, 1, 0.5843138, 0, 1,
7.313131, 4.323232, -535.222, 1, 0.5843138, 0, 1,
7.353535, 4.323232, -535.7551, 1, 0.4823529, 0, 1,
7.393939, 4.323232, -536.3056, 1, 0.4823529, 0, 1,
7.434343, 4.323232, -536.8737, 1, 0.4823529, 0, 1,
7.474748, 4.323232, -537.4592, 1, 0.4823529, 0, 1,
7.515152, 4.323232, -538.0621, 1, 0.4823529, 0, 1,
7.555555, 4.323232, -538.6826, 1, 0.4823529, 0, 1,
7.59596, 4.323232, -539.3204, 1, 0.4823529, 0, 1,
7.636364, 4.323232, -539.9758, 1, 0.4823529, 0, 1,
7.676768, 4.323232, -540.6486, 1, 0.4823529, 0, 1,
7.717172, 4.323232, -541.3389, 1, 0.3764706, 0, 1,
7.757576, 4.323232, -542.0467, 1, 0.3764706, 0, 1,
7.79798, 4.323232, -542.772, 1, 0.3764706, 0, 1,
7.838384, 4.323232, -543.5146, 1, 0.3764706, 0, 1,
7.878788, 4.323232, -544.2748, 1, 0.3764706, 0, 1,
7.919192, 4.323232, -545.0525, 1, 0.3764706, 0, 1,
7.959596, 4.323232, -545.8476, 1, 0.3764706, 0, 1,
8, 4.323232, -546.6602, 1, 0.2745098, 0, 1,
4, 4.373737, -551.5706, 1, 0.2745098, 0, 1,
4.040404, 4.373737, -550.6918, 1, 0.2745098, 0, 1,
4.080808, 4.373737, -549.8302, 1, 0.2745098, 0, 1,
4.121212, 4.373737, -548.9857, 1, 0.2745098, 0, 1,
4.161616, 4.373737, -548.1581, 1, 0.2745098, 0, 1,
4.20202, 4.373737, -547.3477, 1, 0.2745098, 0, 1,
4.242424, 4.373737, -546.5543, 1, 0.2745098, 0, 1,
4.282828, 4.373737, -545.7781, 1, 0.3764706, 0, 1,
4.323232, 4.373737, -545.0188, 1, 0.3764706, 0, 1,
4.363636, 4.373737, -544.2767, 1, 0.3764706, 0, 1,
4.40404, 4.373737, -543.5516, 1, 0.3764706, 0, 1,
4.444445, 4.373737, -542.8436, 1, 0.3764706, 0, 1,
4.484848, 4.373737, -542.1526, 1, 0.3764706, 0, 1,
4.525252, 4.373737, -541.4787, 1, 0.3764706, 0, 1,
4.565657, 4.373737, -540.8218, 1, 0.4823529, 0, 1,
4.606061, 4.373737, -540.1821, 1, 0.4823529, 0, 1,
4.646465, 4.373737, -539.5594, 1, 0.4823529, 0, 1,
4.686869, 4.373737, -538.9538, 1, 0.4823529, 0, 1,
4.727273, 4.373737, -538.3652, 1, 0.4823529, 0, 1,
4.767677, 4.373737, -537.7938, 1, 0.4823529, 0, 1,
4.808081, 4.373737, -537.2393, 1, 0.4823529, 0, 1,
4.848485, 4.373737, -536.702, 1, 0.4823529, 0, 1,
4.888889, 4.373737, -536.1817, 1, 0.4823529, 0, 1,
4.929293, 4.373737, -535.6785, 1, 0.4823529, 0, 1,
4.969697, 4.373737, -535.1923, 1, 0.5843138, 0, 1,
5.010101, 4.373737, -534.7233, 1, 0.5843138, 0, 1,
5.050505, 4.373737, -534.2712, 1, 0.5843138, 0, 1,
5.090909, 4.373737, -533.8363, 1, 0.5843138, 0, 1,
5.131313, 4.373737, -533.4185, 1, 0.5843138, 0, 1,
5.171717, 4.373737, -533.0176, 1, 0.5843138, 0, 1,
5.212121, 4.373737, -532.6339, 1, 0.5843138, 0, 1,
5.252525, 4.373737, -532.2672, 1, 0.5843138, 0, 1,
5.292929, 4.373737, -531.9176, 1, 0.5843138, 0, 1,
5.333333, 4.373737, -531.5851, 1, 0.5843138, 0, 1,
5.373737, 4.373737, -531.2696, 1, 0.5843138, 0, 1,
5.414141, 4.373737, -530.9712, 1, 0.5843138, 0, 1,
5.454545, 4.373737, -530.6899, 1, 0.5843138, 0, 1,
5.494949, 4.373737, -530.4256, 1, 0.5843138, 0, 1,
5.535354, 4.373737, -530.1784, 1, 0.6862745, 0, 1,
5.575758, 4.373737, -529.9482, 1, 0.6862745, 0, 1,
5.616162, 4.373737, -529.7352, 1, 0.6862745, 0, 1,
5.656566, 4.373737, -529.5392, 1, 0.6862745, 0, 1,
5.69697, 4.373737, -529.3603, 1, 0.6862745, 0, 1,
5.737374, 4.373737, -529.1984, 1, 0.6862745, 0, 1,
5.777778, 4.373737, -529.0536, 1, 0.6862745, 0, 1,
5.818182, 4.373737, -528.9259, 1, 0.6862745, 0, 1,
5.858586, 4.373737, -528.8152, 1, 0.6862745, 0, 1,
5.89899, 4.373737, -528.7216, 1, 0.6862745, 0, 1,
5.939394, 4.373737, -528.6451, 1, 0.6862745, 0, 1,
5.979798, 4.373737, -528.5856, 1, 0.6862745, 0, 1,
6.020202, 4.373737, -528.5433, 1, 0.6862745, 0, 1,
6.060606, 4.373737, -528.5179, 1, 0.6862745, 0, 1,
6.10101, 4.373737, -528.5097, 1, 0.6862745, 0, 1,
6.141414, 4.373737, -528.5185, 1, 0.6862745, 0, 1,
6.181818, 4.373737, -528.5444, 1, 0.6862745, 0, 1,
6.222222, 4.373737, -528.5873, 1, 0.6862745, 0, 1,
6.262626, 4.373737, -528.6473, 1, 0.6862745, 0, 1,
6.30303, 4.373737, -528.7244, 1, 0.6862745, 0, 1,
6.343434, 4.373737, -528.8186, 1, 0.6862745, 0, 1,
6.383838, 4.373737, -528.9298, 1, 0.6862745, 0, 1,
6.424242, 4.373737, -529.0581, 1, 0.6862745, 0, 1,
6.464646, 4.373737, -529.2035, 1, 0.6862745, 0, 1,
6.505051, 4.373737, -529.3659, 1, 0.6862745, 0, 1,
6.545455, 4.373737, -529.5454, 1, 0.6862745, 0, 1,
6.585859, 4.373737, -529.7419, 1, 0.6862745, 0, 1,
6.626263, 4.373737, -529.9556, 1, 0.6862745, 0, 1,
6.666667, 4.373737, -530.1863, 1, 0.6862745, 0, 1,
6.707071, 4.373737, -530.434, 1, 0.5843138, 0, 1,
6.747475, 4.373737, -530.6989, 1, 0.5843138, 0, 1,
6.787879, 4.373737, -530.9808, 1, 0.5843138, 0, 1,
6.828283, 4.373737, -531.2797, 1, 0.5843138, 0, 1,
6.868687, 4.373737, -531.5958, 1, 0.5843138, 0, 1,
6.909091, 4.373737, -531.9288, 1, 0.5843138, 0, 1,
6.949495, 4.373737, -532.2791, 1, 0.5843138, 0, 1,
6.989899, 4.373737, -532.6462, 1, 0.5843138, 0, 1,
7.030303, 4.373737, -533.0306, 1, 0.5843138, 0, 1,
7.070707, 4.373737, -533.4319, 1, 0.5843138, 0, 1,
7.111111, 4.373737, -533.8504, 1, 0.5843138, 0, 1,
7.151515, 4.373737, -534.2859, 1, 0.5843138, 0, 1,
7.191919, 4.373737, -534.7385, 1, 0.5843138, 0, 1,
7.232323, 4.373737, -535.2081, 1, 0.5843138, 0, 1,
7.272727, 4.373737, -535.6948, 1, 0.4823529, 0, 1,
7.313131, 4.373737, -536.1986, 1, 0.4823529, 0, 1,
7.353535, 4.373737, -536.7194, 1, 0.4823529, 0, 1,
7.393939, 4.373737, -537.2573, 1, 0.4823529, 0, 1,
7.434343, 4.373737, -537.8123, 1, 0.4823529, 0, 1,
7.474748, 4.373737, -538.3844, 1, 0.4823529, 0, 1,
7.515152, 4.373737, -538.9735, 1, 0.4823529, 0, 1,
7.555555, 4.373737, -539.5797, 1, 0.4823529, 0, 1,
7.59596, 4.373737, -540.2029, 1, 0.4823529, 0, 1,
7.636364, 4.373737, -540.8433, 1, 0.4823529, 0, 1,
7.676768, 4.373737, -541.5006, 1, 0.3764706, 0, 1,
7.717172, 4.373737, -542.1751, 1, 0.3764706, 0, 1,
7.757576, 4.373737, -542.8666, 1, 0.3764706, 0, 1,
7.79798, 4.373737, -543.5752, 1, 0.3764706, 0, 1,
7.838384, 4.373737, -544.3008, 1, 0.3764706, 0, 1,
7.878788, 4.373737, -545.0436, 1, 0.3764706, 0, 1,
7.919192, 4.373737, -545.8033, 1, 0.3764706, 0, 1,
7.959596, 4.373737, -546.5803, 1, 0.2745098, 0, 1,
8, 4.373737, -547.3741, 1, 0.2745098, 0, 1,
4, 4.424242, -552.2173, 1, 0.1686275, 0, 1,
4.040404, 4.424242, -551.3586, 1, 0.2745098, 0, 1,
4.080808, 4.424242, -550.5165, 1, 0.2745098, 0, 1,
4.121212, 4.424242, -549.6912, 1, 0.2745098, 0, 1,
4.161616, 4.424242, -548.8824, 1, 0.2745098, 0, 1,
4.20202, 4.424242, -548.0904, 1, 0.2745098, 0, 1,
4.242424, 4.424242, -547.3151, 1, 0.2745098, 0, 1,
4.282828, 4.424242, -546.5563, 1, 0.2745098, 0, 1,
4.323232, 4.424242, -545.8144, 1, 0.3764706, 0, 1,
4.363636, 4.424242, -545.0891, 1, 0.3764706, 0, 1,
4.40404, 4.424242, -544.3804, 1, 0.3764706, 0, 1,
4.444445, 4.424242, -543.6885, 1, 0.3764706, 0, 1,
4.484848, 4.424242, -543.0132, 1, 0.3764706, 0, 1,
4.525252, 4.424242, -542.3546, 1, 0.3764706, 0, 1,
4.565657, 4.424242, -541.7127, 1, 0.3764706, 0, 1,
4.606061, 4.424242, -541.0875, 1, 0.3764706, 0, 1,
4.646465, 4.424242, -540.4789, 1, 0.4823529, 0, 1,
4.686869, 4.424242, -539.887, 1, 0.4823529, 0, 1,
4.727273, 4.424242, -539.3118, 1, 0.4823529, 0, 1,
4.767677, 4.424242, -538.7533, 1, 0.4823529, 0, 1,
4.808081, 4.424242, -538.2115, 1, 0.4823529, 0, 1,
4.848485, 4.424242, -537.6863, 1, 0.4823529, 0, 1,
4.888889, 4.424242, -537.1779, 1, 0.4823529, 0, 1,
4.929293, 4.424242, -536.6861, 1, 0.4823529, 0, 1,
4.969697, 4.424242, -536.2109, 1, 0.4823529, 0, 1,
5.010101, 4.424242, -535.7525, 1, 0.4823529, 0, 1,
5.050505, 4.424242, -535.3108, 1, 0.5843138, 0, 1,
5.090909, 4.424242, -534.8857, 1, 0.5843138, 0, 1,
5.131313, 4.424242, -534.4773, 1, 0.5843138, 0, 1,
5.171717, 4.424242, -534.0856, 1, 0.5843138, 0, 1,
5.212121, 4.424242, -533.7106, 1, 0.5843138, 0, 1,
5.252525, 4.424242, -533.3522, 1, 0.5843138, 0, 1,
5.292929, 4.424242, -533.0106, 1, 0.5843138, 0, 1,
5.333333, 4.424242, -532.6855, 1, 0.5843138, 0, 1,
5.373737, 4.424242, -532.3773, 1, 0.5843138, 0, 1,
5.414141, 4.424242, -532.0856, 1, 0.5843138, 0, 1,
5.454545, 4.424242, -531.8107, 1, 0.5843138, 0, 1,
5.494949, 4.424242, -531.5524, 1, 0.5843138, 0, 1,
5.535354, 4.424242, -531.3109, 1, 0.5843138, 0, 1,
5.575758, 4.424242, -531.0859, 1, 0.5843138, 0, 1,
5.616162, 4.424242, -530.8777, 1, 0.5843138, 0, 1,
5.656566, 4.424242, -530.6862, 1, 0.5843138, 0, 1,
5.69697, 4.424242, -530.5113, 1, 0.5843138, 0, 1,
5.737374, 4.424242, -530.3531, 1, 0.5843138, 0, 1,
5.777778, 4.424242, -530.2116, 1, 0.6862745, 0, 1,
5.818182, 4.424242, -530.0867, 1, 0.6862745, 0, 1,
5.858586, 4.424242, -529.9786, 1, 0.6862745, 0, 1,
5.89899, 4.424242, -529.8871, 1, 0.6862745, 0, 1,
5.939394, 4.424242, -529.8124, 1, 0.6862745, 0, 1,
5.979798, 4.424242, -529.7543, 1, 0.6862745, 0, 1,
6.020202, 4.424242, -529.7128, 1, 0.6862745, 0, 1,
6.060606, 4.424242, -529.6881, 1, 0.6862745, 0, 1,
6.10101, 4.424242, -529.68, 1, 0.6862745, 0, 1,
6.141414, 4.424242, -529.6887, 1, 0.6862745, 0, 1,
6.181818, 4.424242, -529.7139, 1, 0.6862745, 0, 1,
6.222222, 4.424242, -529.7559, 1, 0.6862745, 0, 1,
6.262626, 4.424242, -529.8146, 1, 0.6862745, 0, 1,
6.30303, 4.424242, -529.8899, 1, 0.6862745, 0, 1,
6.343434, 4.424242, -529.9819, 1, 0.6862745, 0, 1,
6.383838, 4.424242, -530.0906, 1, 0.6862745, 0, 1,
6.424242, 4.424242, -530.216, 1, 0.6862745, 0, 1,
6.464646, 4.424242, -530.358, 1, 0.5843138, 0, 1,
6.505051, 4.424242, -530.5168, 1, 0.5843138, 0, 1,
6.545455, 4.424242, -530.6922, 1, 0.5843138, 0, 1,
6.585859, 4.424242, -530.8843, 1, 0.5843138, 0, 1,
6.626263, 4.424242, -531.0931, 1, 0.5843138, 0, 1,
6.666667, 4.424242, -531.3185, 1, 0.5843138, 0, 1,
6.707071, 4.424242, -531.5607, 1, 0.5843138, 0, 1,
6.747475, 4.424242, -531.8195, 1, 0.5843138, 0, 1,
6.787879, 4.424242, -532.095, 1, 0.5843138, 0, 1,
6.828283, 4.424242, -532.3871, 1, 0.5843138, 0, 1,
6.868687, 4.424242, -532.696, 1, 0.5843138, 0, 1,
6.909091, 4.424242, -533.0215, 1, 0.5843138, 0, 1,
6.949495, 4.424242, -533.3638, 1, 0.5843138, 0, 1,
6.989899, 4.424242, -533.7227, 1, 0.5843138, 0, 1,
7.030303, 4.424242, -534.0983, 1, 0.5843138, 0, 1,
7.070707, 4.424242, -534.4905, 1, 0.5843138, 0, 1,
7.111111, 4.424242, -534.8995, 1, 0.5843138, 0, 1,
7.151515, 4.424242, -535.3251, 1, 0.5843138, 0, 1,
7.191919, 4.424242, -535.7674, 1, 0.4823529, 0, 1,
7.232323, 4.424242, -536.2264, 1, 0.4823529, 0, 1,
7.272727, 4.424242, -536.702, 1, 0.4823529, 0, 1,
7.313131, 4.424242, -537.1943, 1, 0.4823529, 0, 1,
7.353535, 4.424242, -537.7034, 1, 0.4823529, 0, 1,
7.393939, 4.424242, -538.2291, 1, 0.4823529, 0, 1,
7.434343, 4.424242, -538.7715, 1, 0.4823529, 0, 1,
7.474748, 4.424242, -539.3305, 1, 0.4823529, 0, 1,
7.515152, 4.424242, -539.9062, 1, 0.4823529, 0, 1,
7.555555, 4.424242, -540.4987, 1, 0.4823529, 0, 1,
7.59596, 4.424242, -541.1078, 1, 0.3764706, 0, 1,
7.636364, 4.424242, -541.7336, 1, 0.3764706, 0, 1,
7.676768, 4.424242, -542.376, 1, 0.3764706, 0, 1,
7.717172, 4.424242, -543.0352, 1, 0.3764706, 0, 1,
7.757576, 4.424242, -543.711, 1, 0.3764706, 0, 1,
7.79798, 4.424242, -544.4035, 1, 0.3764706, 0, 1,
7.838384, 4.424242, -545.1127, 1, 0.3764706, 0, 1,
7.878788, 4.424242, -545.8386, 1, 0.3764706, 0, 1,
7.919192, 4.424242, -546.5811, 1, 0.2745098, 0, 1,
7.959596, 4.424242, -547.3403, 1, 0.2745098, 0, 1,
8, 4.424242, -548.1163, 1, 0.2745098, 0, 1,
4, 4.474748, -552.8937, 1, 0.1686275, 0, 1,
4.040404, 4.474748, -552.0542, 1, 0.1686275, 0, 1,
4.080808, 4.474748, -551.231, 1, 0.2745098, 0, 1,
4.121212, 4.474748, -550.4241, 1, 0.2745098, 0, 1,
4.161616, 4.474748, -549.6335, 1, 0.2745098, 0, 1,
4.20202, 4.474748, -548.8593, 1, 0.2745098, 0, 1,
4.242424, 4.474748, -548.1014, 1, 0.2745098, 0, 1,
4.282828, 4.474748, -547.3597, 1, 0.2745098, 0, 1,
4.323232, 4.474748, -546.6344, 1, 0.2745098, 0, 1,
4.363636, 4.474748, -545.9254, 1, 0.3764706, 0, 1,
4.40404, 4.474748, -545.2326, 1, 0.3764706, 0, 1,
4.444445, 4.474748, -544.5562, 1, 0.3764706, 0, 1,
4.484848, 4.474748, -543.8961, 1, 0.3764706, 0, 1,
4.525252, 4.474748, -543.2523, 1, 0.3764706, 0, 1,
4.565657, 4.474748, -542.6248, 1, 0.3764706, 0, 1,
4.606061, 4.474748, -542.0135, 1, 0.3764706, 0, 1,
4.646465, 4.474748, -541.4186, 1, 0.3764706, 0, 1,
4.686869, 4.474748, -540.8401, 1, 0.4823529, 0, 1,
4.727273, 4.474748, -540.2778, 1, 0.4823529, 0, 1,
4.767677, 4.474748, -539.7318, 1, 0.4823529, 0, 1,
4.808081, 4.474748, -539.2021, 1, 0.4823529, 0, 1,
4.848485, 4.474748, -538.6888, 1, 0.4823529, 0, 1,
4.888889, 4.474748, -538.1917, 1, 0.4823529, 0, 1,
4.929293, 4.474748, -537.711, 1, 0.4823529, 0, 1,
4.969697, 4.474748, -537.2465, 1, 0.4823529, 0, 1,
5.010101, 4.474748, -536.7984, 1, 0.4823529, 0, 1,
5.050505, 4.474748, -536.3666, 1, 0.4823529, 0, 1,
5.090909, 4.474748, -535.951, 1, 0.4823529, 0, 1,
5.131313, 4.474748, -535.5518, 1, 0.5843138, 0, 1,
5.171717, 4.474748, -535.1689, 1, 0.5843138, 0, 1,
5.212121, 4.474748, -534.8022, 1, 0.5843138, 0, 1,
5.252525, 4.474748, -534.452, 1, 0.5843138, 0, 1,
5.292929, 4.474748, -534.118, 1, 0.5843138, 0, 1,
5.333333, 4.474748, -533.8003, 1, 0.5843138, 0, 1,
5.373737, 4.474748, -533.4989, 1, 0.5843138, 0, 1,
5.414141, 4.474748, -533.2138, 1, 0.5843138, 0, 1,
5.454545, 4.474748, -532.945, 1, 0.5843138, 0, 1,
5.494949, 4.474748, -532.6926, 1, 0.5843138, 0, 1,
5.535354, 4.474748, -532.4564, 1, 0.5843138, 0, 1,
5.575758, 4.474748, -532.2365, 1, 0.5843138, 0, 1,
5.616162, 4.474748, -532.033, 1, 0.5843138, 0, 1,
5.656566, 4.474748, -531.8457, 1, 0.5843138, 0, 1,
5.69697, 4.474748, -531.6748, 1, 0.5843138, 0, 1,
5.737374, 4.474748, -531.5201, 1, 0.5843138, 0, 1,
5.777778, 4.474748, -531.3818, 1, 0.5843138, 0, 1,
5.818182, 4.474748, -531.2598, 1, 0.5843138, 0, 1,
5.858586, 4.474748, -531.1541, 1, 0.5843138, 0, 1,
5.89899, 4.474748, -531.0646, 1, 0.5843138, 0, 1,
5.939394, 4.474748, -530.9915, 1, 0.5843138, 0, 1,
5.979798, 4.474748, -530.9348, 1, 0.5843138, 0, 1,
6.020202, 4.474748, -530.8942, 1, 0.5843138, 0, 1,
6.060606, 4.474748, -530.8701, 1, 0.5843138, 0, 1,
6.10101, 4.474748, -530.8622, 1, 0.5843138, 0, 1,
6.141414, 4.474748, -530.8706, 1, 0.5843138, 0, 1,
6.181818, 4.474748, -530.8953, 1, 0.5843138, 0, 1,
6.222222, 4.474748, -530.9363, 1, 0.5843138, 0, 1,
6.262626, 4.474748, -530.9937, 1, 0.5843138, 0, 1,
6.30303, 4.474748, -531.0673, 1, 0.5843138, 0, 1,
6.343434, 4.474748, -531.1573, 1, 0.5843138, 0, 1,
6.383838, 4.474748, -531.2635, 1, 0.5843138, 0, 1,
6.424242, 4.474748, -531.3861, 1, 0.5843138, 0, 1,
6.464646, 4.474748, -531.525, 1, 0.5843138, 0, 1,
6.505051, 4.474748, -531.6802, 1, 0.5843138, 0, 1,
6.545455, 4.474748, -531.8516, 1, 0.5843138, 0, 1,
6.585859, 4.474748, -532.0394, 1, 0.5843138, 0, 1,
6.626263, 4.474748, -532.2435, 1, 0.5843138, 0, 1,
6.666667, 4.474748, -532.4639, 1, 0.5843138, 0, 1,
6.707071, 4.474748, -532.7006, 1, 0.5843138, 0, 1,
6.747475, 4.474748, -532.9536, 1, 0.5843138, 0, 1,
6.787879, 4.474748, -533.223, 1, 0.5843138, 0, 1,
6.828283, 4.474748, -533.5085, 1, 0.5843138, 0, 1,
6.868687, 4.474748, -533.8105, 1, 0.5843138, 0, 1,
6.909091, 4.474748, -534.1287, 1, 0.5843138, 0, 1,
6.949495, 4.474748, -534.4633, 1, 0.5843138, 0, 1,
6.989899, 4.474748, -534.8141, 1, 0.5843138, 0, 1,
7.030303, 4.474748, -535.1813, 1, 0.5843138, 0, 1,
7.070707, 4.474748, -535.5647, 1, 0.5843138, 0, 1,
7.111111, 4.474748, -535.9645, 1, 0.4823529, 0, 1,
7.151515, 4.474748, -536.3806, 1, 0.4823529, 0, 1,
7.191919, 4.474748, -536.8129, 1, 0.4823529, 0, 1,
7.232323, 4.474748, -537.2616, 1, 0.4823529, 0, 1,
7.272727, 4.474748, -537.7266, 1, 0.4823529, 0, 1,
7.313131, 4.474748, -538.2079, 1, 0.4823529, 0, 1,
7.353535, 4.474748, -538.7054, 1, 0.4823529, 0, 1,
7.393939, 4.474748, -539.2194, 1, 0.4823529, 0, 1,
7.434343, 4.474748, -539.7496, 1, 0.4823529, 0, 1,
7.474748, 4.474748, -540.2961, 1, 0.4823529, 0, 1,
7.515152, 4.474748, -540.8589, 1, 0.4823529, 0, 1,
7.555555, 4.474748, -541.438, 1, 0.3764706, 0, 1,
7.59596, 4.474748, -542.0334, 1, 0.3764706, 0, 1,
7.636364, 4.474748, -542.6452, 1, 0.3764706, 0, 1,
7.676768, 4.474748, -543.2733, 1, 0.3764706, 0, 1,
7.717172, 4.474748, -543.9176, 1, 0.3764706, 0, 1,
7.757576, 4.474748, -544.5782, 1, 0.3764706, 0, 1,
7.79798, 4.474748, -545.2552, 1, 0.3764706, 0, 1,
7.838384, 4.474748, -545.9485, 1, 0.3764706, 0, 1,
7.878788, 4.474748, -546.658, 1, 0.2745098, 0, 1,
7.919192, 4.474748, -547.3839, 1, 0.2745098, 0, 1,
7.959596, 4.474748, -548.1261, 1, 0.2745098, 0, 1,
8, 4.474748, -548.8846, 1, 0.2745098, 0, 1,
4, 4.525252, -553.5975, 1, 0.1686275, 0, 1,
4.040404, 4.525252, -552.7767, 1, 0.1686275, 0, 1,
4.080808, 4.525252, -551.9718, 1, 0.1686275, 0, 1,
4.121212, 4.525252, -551.1828, 1, 0.2745098, 0, 1,
4.161616, 4.525252, -550.4098, 1, 0.2745098, 0, 1,
4.20202, 4.525252, -549.6527, 1, 0.2745098, 0, 1,
4.242424, 4.525252, -548.9116, 1, 0.2745098, 0, 1,
4.282828, 4.525252, -548.1864, 1, 0.2745098, 0, 1,
4.323232, 4.525252, -547.4772, 1, 0.2745098, 0, 1,
4.363636, 4.525252, -546.7839, 1, 0.2745098, 0, 1,
4.40404, 4.525252, -546.1066, 1, 0.3764706, 0, 1,
4.444445, 4.525252, -545.4451, 1, 0.3764706, 0, 1,
4.484848, 4.525252, -544.7997, 1, 0.3764706, 0, 1,
4.525252, 4.525252, -544.1702, 1, 0.3764706, 0, 1,
4.565657, 4.525252, -543.5566, 1, 0.3764706, 0, 1,
4.606061, 4.525252, -542.9589, 1, 0.3764706, 0, 1,
4.646465, 4.525252, -542.3773, 1, 0.3764706, 0, 1,
4.686869, 4.525252, -541.8115, 1, 0.3764706, 0, 1,
4.727273, 4.525252, -541.2617, 1, 0.3764706, 0, 1,
4.767677, 4.525252, -540.7278, 1, 0.4823529, 0, 1,
4.808081, 4.525252, -540.21, 1, 0.4823529, 0, 1,
4.848485, 4.525252, -539.7079, 1, 0.4823529, 0, 1,
4.888889, 4.525252, -539.2219, 1, 0.4823529, 0, 1,
4.929293, 4.525252, -538.7518, 1, 0.4823529, 0, 1,
4.969697, 4.525252, -538.2977, 1, 0.4823529, 0, 1,
5.010101, 4.525252, -537.8595, 1, 0.4823529, 0, 1,
5.050505, 4.525252, -537.4373, 1, 0.4823529, 0, 1,
5.090909, 4.525252, -537.0309, 1, 0.4823529, 0, 1,
5.131313, 4.525252, -536.6406, 1, 0.4823529, 0, 1,
5.171717, 4.525252, -536.2662, 1, 0.4823529, 0, 1,
5.212121, 4.525252, -535.9077, 1, 0.4823529, 0, 1,
5.252525, 4.525252, -535.5652, 1, 0.5843138, 0, 1,
5.292929, 4.525252, -535.2386, 1, 0.5843138, 0, 1,
5.333333, 4.525252, -534.928, 1, 0.5843138, 0, 1,
5.373737, 4.525252, -534.6332, 1, 0.5843138, 0, 1,
5.414141, 4.525252, -534.3545, 1, 0.5843138, 0, 1,
5.454545, 4.525252, -534.0917, 1, 0.5843138, 0, 1,
5.494949, 4.525252, -533.8448, 1, 0.5843138, 0, 1,
5.535354, 4.525252, -533.6139, 1, 0.5843138, 0, 1,
5.575758, 4.525252, -533.3989, 1, 0.5843138, 0, 1,
5.616162, 4.525252, -533.1999, 1, 0.5843138, 0, 1,
5.656566, 4.525252, -533.0168, 1, 0.5843138, 0, 1,
5.69697, 4.525252, -532.8497, 1, 0.5843138, 0, 1,
5.737374, 4.525252, -532.6984, 1, 0.5843138, 0, 1,
5.777778, 4.525252, -532.5632, 1, 0.5843138, 0, 1,
5.818182, 4.525252, -532.4438, 1, 0.5843138, 0, 1,
5.858586, 4.525252, -532.3405, 1, 0.5843138, 0, 1,
5.89899, 4.525252, -532.2531, 1, 0.5843138, 0, 1,
5.939394, 4.525252, -532.1816, 1, 0.5843138, 0, 1,
5.979798, 4.525252, -532.126, 1, 0.5843138, 0, 1,
6.020202, 4.525252, -532.0864, 1, 0.5843138, 0, 1,
6.060606, 4.525252, -532.0628, 1, 0.5843138, 0, 1,
6.10101, 4.525252, -532.0551, 1, 0.5843138, 0, 1,
6.141414, 4.525252, -532.0633, 1, 0.5843138, 0, 1,
6.181818, 4.525252, -532.0875, 1, 0.5843138, 0, 1,
6.222222, 4.525252, -532.1276, 1, 0.5843138, 0, 1,
6.262626, 4.525252, -532.1837, 1, 0.5843138, 0, 1,
6.30303, 4.525252, -532.2557, 1, 0.5843138, 0, 1,
6.343434, 4.525252, -532.3436, 1, 0.5843138, 0, 1,
6.383838, 4.525252, -532.4476, 1, 0.5843138, 0, 1,
6.424242, 4.525252, -532.5674, 1, 0.5843138, 0, 1,
6.464646, 4.525252, -532.7032, 1, 0.5843138, 0, 1,
6.505051, 4.525252, -532.8549, 1, 0.5843138, 0, 1,
6.545455, 4.525252, -533.0226, 1, 0.5843138, 0, 1,
6.585859, 4.525252, -533.2062, 1, 0.5843138, 0, 1,
6.626263, 4.525252, -533.4058, 1, 0.5843138, 0, 1,
6.666667, 4.525252, -533.6213, 1, 0.5843138, 0, 1,
6.707071, 4.525252, -533.8527, 1, 0.5843138, 0, 1,
6.747475, 4.525252, -534.1001, 1, 0.5843138, 0, 1,
6.787879, 4.525252, -534.3635, 1, 0.5843138, 0, 1,
6.828283, 4.525252, -534.6427, 1, 0.5843138, 0, 1,
6.868687, 4.525252, -534.9379, 1, 0.5843138, 0, 1,
6.909091, 4.525252, -535.2491, 1, 0.5843138, 0, 1,
6.949495, 4.525252, -535.5762, 1, 0.5843138, 0, 1,
6.989899, 4.525252, -535.9193, 1, 0.4823529, 0, 1,
7.030303, 4.525252, -536.2783, 1, 0.4823529, 0, 1,
7.070707, 4.525252, -536.6532, 1, 0.4823529, 0, 1,
7.111111, 4.525252, -537.0441, 1, 0.4823529, 0, 1,
7.151515, 4.525252, -537.4509, 1, 0.4823529, 0, 1,
7.191919, 4.525252, -537.8737, 1, 0.4823529, 0, 1,
7.232323, 4.525252, -538.3124, 1, 0.4823529, 0, 1,
7.272727, 4.525252, -538.7671, 1, 0.4823529, 0, 1,
7.313131, 4.525252, -539.2377, 1, 0.4823529, 0, 1,
7.353535, 4.525252, -539.7242, 1, 0.4823529, 0, 1,
7.393939, 4.525252, -540.2267, 1, 0.4823529, 0, 1,
7.434343, 4.525252, -540.7452, 1, 0.4823529, 0, 1,
7.474748, 4.525252, -541.2796, 1, 0.3764706, 0, 1,
7.515152, 4.525252, -541.8299, 1, 0.3764706, 0, 1,
7.555555, 4.525252, -542.3962, 1, 0.3764706, 0, 1,
7.59596, 4.525252, -542.9784, 1, 0.3764706, 0, 1,
7.636364, 4.525252, -543.5765, 1, 0.3764706, 0, 1,
7.676768, 4.525252, -544.1907, 1, 0.3764706, 0, 1,
7.717172, 4.525252, -544.8207, 1, 0.3764706, 0, 1,
7.757576, 4.525252, -545.4667, 1, 0.3764706, 0, 1,
7.79798, 4.525252, -546.1286, 1, 0.3764706, 0, 1,
7.838384, 4.525252, -546.8065, 1, 0.2745098, 0, 1,
7.878788, 4.525252, -547.5003, 1, 0.2745098, 0, 1,
7.919192, 4.525252, -548.2101, 1, 0.2745098, 0, 1,
7.959596, 4.525252, -548.9358, 1, 0.2745098, 0, 1,
8, 4.525252, -549.6774, 1, 0.2745098, 0, 1,
4, 4.575758, -554.3272, 1, 0.1686275, 0, 1,
4.040404, 4.575758, -553.5244, 1, 0.1686275, 0, 1,
4.080808, 4.575758, -552.7372, 1, 0.1686275, 0, 1,
4.121212, 4.575758, -551.9655, 1, 0.1686275, 0, 1,
4.161616, 4.575758, -551.2095, 1, 0.2745098, 0, 1,
4.20202, 4.575758, -550.4691, 1, 0.2745098, 0, 1,
4.242424, 4.575758, -549.7442, 1, 0.2745098, 0, 1,
4.282828, 4.575758, -549.0349, 1, 0.2745098, 0, 1,
4.323232, 4.575758, -548.3412, 1, 0.2745098, 0, 1,
4.363636, 4.575758, -547.6631, 1, 0.2745098, 0, 1,
4.40404, 4.575758, -547.0007, 1, 0.2745098, 0, 1,
4.444445, 4.575758, -546.3538, 1, 0.3764706, 0, 1,
4.484848, 4.575758, -545.7225, 1, 0.3764706, 0, 1,
4.525252, 4.575758, -545.1068, 1, 0.3764706, 0, 1,
4.565657, 4.575758, -544.5067, 1, 0.3764706, 0, 1,
4.606061, 4.575758, -543.9222, 1, 0.3764706, 0, 1,
4.646465, 4.575758, -543.3533, 1, 0.3764706, 0, 1,
4.686869, 4.575758, -542.7999, 1, 0.3764706, 0, 1,
4.727273, 4.575758, -542.2622, 1, 0.3764706, 0, 1,
4.767677, 4.575758, -541.7401, 1, 0.3764706, 0, 1,
4.808081, 4.575758, -541.2335, 1, 0.3764706, 0, 1,
4.848485, 4.575758, -540.7426, 1, 0.4823529, 0, 1,
4.888889, 4.575758, -540.2672, 1, 0.4823529, 0, 1,
4.929293, 4.575758, -539.8075, 1, 0.4823529, 0, 1,
4.969697, 4.575758, -539.3633, 1, 0.4823529, 0, 1,
5.010101, 4.575758, -538.9348, 1, 0.4823529, 0, 1,
5.050505, 4.575758, -538.5217, 1, 0.4823529, 0, 1,
5.090909, 4.575758, -538.1244, 1, 0.4823529, 0, 1,
5.131313, 4.575758, -537.7426, 1, 0.4823529, 0, 1,
5.171717, 4.575758, -537.3763, 1, 0.4823529, 0, 1,
5.212121, 4.575758, -537.0258, 1, 0.4823529, 0, 1,
5.252525, 4.575758, -536.6907, 1, 0.4823529, 0, 1,
5.292929, 4.575758, -536.3713, 1, 0.4823529, 0, 1,
5.333333, 4.575758, -536.0675, 1, 0.4823529, 0, 1,
5.373737, 4.575758, -535.7793, 1, 0.4823529, 0, 1,
5.414141, 4.575758, -535.5067, 1, 0.5843138, 0, 1,
5.454545, 4.575758, -535.2496, 1, 0.5843138, 0, 1,
5.494949, 4.575758, -535.0082, 1, 0.5843138, 0, 1,
5.535354, 4.575758, -534.7823, 1, 0.5843138, 0, 1,
5.575758, 4.575758, -534.572, 1, 0.5843138, 0, 1,
5.616162, 4.575758, -534.3774, 1, 0.5843138, 0, 1,
5.656566, 4.575758, -534.1983, 1, 0.5843138, 0, 1,
5.69697, 4.575758, -534.0349, 1, 0.5843138, 0, 1,
5.737374, 4.575758, -533.887, 1, 0.5843138, 0, 1,
5.777778, 4.575758, -533.7546, 1, 0.5843138, 0, 1,
5.818182, 4.575758, -533.6379, 1, 0.5843138, 0, 1,
5.858586, 4.575758, -533.5369, 1, 0.5843138, 0, 1,
5.89899, 4.575758, -533.4514, 1, 0.5843138, 0, 1,
5.939394, 4.575758, -533.3814, 1, 0.5843138, 0, 1,
5.979798, 4.575758, -533.3271, 1, 0.5843138, 0, 1,
6.020202, 4.575758, -533.2884, 1, 0.5843138, 0, 1,
6.060606, 4.575758, -533.2653, 1, 0.5843138, 0, 1,
6.10101, 4.575758, -533.2577, 1, 0.5843138, 0, 1,
6.141414, 4.575758, -533.2657, 1, 0.5843138, 0, 1,
6.181818, 4.575758, -533.2894, 1, 0.5843138, 0, 1,
6.222222, 4.575758, -533.3287, 1, 0.5843138, 0, 1,
6.262626, 4.575758, -533.3835, 1, 0.5843138, 0, 1,
6.30303, 4.575758, -533.4539, 1, 0.5843138, 0, 1,
6.343434, 4.575758, -533.5399, 1, 0.5843138, 0, 1,
6.383838, 4.575758, -533.6415, 1, 0.5843138, 0, 1,
6.424242, 4.575758, -533.7588, 1, 0.5843138, 0, 1,
6.464646, 4.575758, -533.8916, 1, 0.5843138, 0, 1,
6.505051, 4.575758, -534.04, 1, 0.5843138, 0, 1,
6.545455, 4.575758, -534.204, 1, 0.5843138, 0, 1,
6.585859, 4.575758, -534.3835, 1, 0.5843138, 0, 1,
6.626263, 4.575758, -534.5787, 1, 0.5843138, 0, 1,
6.666667, 4.575758, -534.7895, 1, 0.5843138, 0, 1,
6.707071, 4.575758, -535.0159, 1, 0.5843138, 0, 1,
6.747475, 4.575758, -535.2578, 1, 0.5843138, 0, 1,
6.787879, 4.575758, -535.5154, 1, 0.5843138, 0, 1,
6.828283, 4.575758, -535.7885, 1, 0.4823529, 0, 1,
6.868687, 4.575758, -536.0773, 1, 0.4823529, 0, 1,
6.909091, 4.575758, -536.3816, 1, 0.4823529, 0, 1,
6.949495, 4.575758, -536.7015, 1, 0.4823529, 0, 1,
6.989899, 4.575758, -537.037, 1, 0.4823529, 0, 1,
7.030303, 4.575758, -537.3882, 1, 0.4823529, 0, 1,
7.070707, 4.575758, -537.7549, 1, 0.4823529, 0, 1,
7.111111, 4.575758, -538.1372, 1, 0.4823529, 0, 1,
7.151515, 4.575758, -538.5351, 1, 0.4823529, 0, 1,
7.191919, 4.575758, -538.9486, 1, 0.4823529, 0, 1,
7.232323, 4.575758, -539.3777, 1, 0.4823529, 0, 1,
7.272727, 4.575758, -539.8224, 1, 0.4823529, 0, 1,
7.313131, 4.575758, -540.2827, 1, 0.4823529, 0, 1,
7.353535, 4.575758, -540.7585, 1, 0.4823529, 0, 1,
7.393939, 4.575758, -541.25, 1, 0.3764706, 0, 1,
7.434343, 4.575758, -541.757, 1, 0.3764706, 0, 1,
7.474748, 4.575758, -542.2797, 1, 0.3764706, 0, 1,
7.515152, 4.575758, -542.8179, 1, 0.3764706, 0, 1,
7.555555, 4.575758, -543.3718, 1, 0.3764706, 0, 1,
7.59596, 4.575758, -543.9412, 1, 0.3764706, 0, 1,
7.636364, 4.575758, -544.5262, 1, 0.3764706, 0, 1,
7.676768, 4.575758, -545.1268, 1, 0.3764706, 0, 1,
7.717172, 4.575758, -545.7431, 1, 0.3764706, 0, 1,
7.757576, 4.575758, -546.3749, 1, 0.3764706, 0, 1,
7.79798, 4.575758, -547.0223, 1, 0.2745098, 0, 1,
7.838384, 4.575758, -547.6853, 1, 0.2745098, 0, 1,
7.878788, 4.575758, -548.3639, 1, 0.2745098, 0, 1,
7.919192, 4.575758, -549.058, 1, 0.2745098, 0, 1,
7.959596, 4.575758, -549.7678, 1, 0.2745098, 0, 1,
8, 4.575758, -550.4932, 1, 0.2745098, 0, 1,
4, 4.626263, -555.0811, 1, 0.1686275, 0, 1,
4.040404, 4.626263, -554.2957, 1, 0.1686275, 0, 1,
4.080808, 4.626263, -553.5256, 1, 0.1686275, 0, 1,
4.121212, 4.626263, -552.7707, 1, 0.1686275, 0, 1,
4.161616, 4.626263, -552.0311, 1, 0.1686275, 0, 1,
4.20202, 4.626263, -551.3067, 1, 0.2745098, 0, 1,
4.242424, 4.626263, -550.5976, 1, 0.2745098, 0, 1,
4.282828, 4.626263, -549.9037, 1, 0.2745098, 0, 1,
4.323232, 4.626263, -549.2251, 1, 0.2745098, 0, 1,
4.363636, 4.626263, -548.5618, 1, 0.2745098, 0, 1,
4.40404, 4.626263, -547.9136, 1, 0.2745098, 0, 1,
4.444445, 4.626263, -547.2808, 1, 0.2745098, 0, 1,
4.484848, 4.626263, -546.6632, 1, 0.2745098, 0, 1,
4.525252, 4.626263, -546.0609, 1, 0.3764706, 0, 1,
4.565657, 4.626263, -545.4738, 1, 0.3764706, 0, 1,
4.606061, 4.626263, -544.902, 1, 0.3764706, 0, 1,
4.646465, 4.626263, -544.3455, 1, 0.3764706, 0, 1,
4.686869, 4.626263, -543.8041, 1, 0.3764706, 0, 1,
4.727273, 4.626263, -543.2781, 1, 0.3764706, 0, 1,
4.767677, 4.626263, -542.7673, 1, 0.3764706, 0, 1,
4.808081, 4.626263, -542.2717, 1, 0.3764706, 0, 1,
4.848485, 4.626263, -541.7914, 1, 0.3764706, 0, 1,
4.888889, 4.626263, -541.3264, 1, 0.3764706, 0, 1,
4.929293, 4.626263, -540.8766, 1, 0.4823529, 0, 1,
4.969697, 4.626263, -540.4421, 1, 0.4823529, 0, 1,
5.010101, 4.626263, -540.0228, 1, 0.4823529, 0, 1,
5.050505, 4.626263, -539.6188, 1, 0.4823529, 0, 1,
5.090909, 4.626263, -539.23, 1, 0.4823529, 0, 1,
5.131313, 4.626263, -538.8566, 1, 0.4823529, 0, 1,
5.171717, 4.626263, -538.4984, 1, 0.4823529, 0, 1,
5.212121, 4.626263, -538.1553, 1, 0.4823529, 0, 1,
5.252525, 4.626263, -537.8276, 1, 0.4823529, 0, 1,
5.292929, 4.626263, -537.5151, 1, 0.4823529, 0, 1,
5.333333, 4.626263, -537.2179, 1, 0.4823529, 0, 1,
5.373737, 4.626263, -536.9359, 1, 0.4823529, 0, 1,
5.414141, 4.626263, -536.6692, 1, 0.4823529, 0, 1,
5.454545, 4.626263, -536.4177, 1, 0.4823529, 0, 1,
5.494949, 4.626263, -536.1815, 1, 0.4823529, 0, 1,
5.535354, 4.626263, -535.9606, 1, 0.4823529, 0, 1,
5.575758, 4.626263, -535.7549, 1, 0.4823529, 0, 1,
5.616162, 4.626263, -535.5645, 1, 0.5843138, 0, 1,
5.656566, 4.626263, -535.3893, 1, 0.5843138, 0, 1,
5.69697, 4.626263, -535.2294, 1, 0.5843138, 0, 1,
5.737374, 4.626263, -535.0847, 1, 0.5843138, 0, 1,
5.777778, 4.626263, -534.9553, 1, 0.5843138, 0, 1,
5.818182, 4.626263, -534.8411, 1, 0.5843138, 0, 1,
5.858586, 4.626263, -534.7422, 1, 0.5843138, 0, 1,
5.89899, 4.626263, -534.6585, 1, 0.5843138, 0, 1,
5.939394, 4.626263, -534.5901, 1, 0.5843138, 0, 1,
5.979798, 4.626263, -534.537, 1, 0.5843138, 0, 1,
6.020202, 4.626263, -534.4991, 1, 0.5843138, 0, 1,
6.060606, 4.626263, -534.4765, 1, 0.5843138, 0, 1,
6.10101, 4.626263, -534.4691, 1, 0.5843138, 0, 1,
6.141414, 4.626263, -534.477, 1, 0.5843138, 0, 1,
6.181818, 4.626263, -534.5001, 1, 0.5843138, 0, 1,
6.222222, 4.626263, -534.5385, 1, 0.5843138, 0, 1,
6.262626, 4.626263, -534.5922, 1, 0.5843138, 0, 1,
6.30303, 4.626263, -534.6611, 1, 0.5843138, 0, 1,
6.343434, 4.626263, -534.7452, 1, 0.5843138, 0, 1,
6.383838, 4.626263, -534.8446, 1, 0.5843138, 0, 1,
6.424242, 4.626263, -534.9593, 1, 0.5843138, 0, 1,
6.464646, 4.626263, -535.0892, 1, 0.5843138, 0, 1,
6.505051, 4.626263, -535.2344, 1, 0.5843138, 0, 1,
6.545455, 4.626263, -535.3948, 1, 0.5843138, 0, 1,
6.585859, 4.626263, -535.5705, 1, 0.5843138, 0, 1,
6.626263, 4.626263, -535.7614, 1, 0.4823529, 0, 1,
6.666667, 4.626263, -535.9677, 1, 0.4823529, 0, 1,
6.707071, 4.626263, -536.1891, 1, 0.4823529, 0, 1,
6.747475, 4.626263, -536.4258, 1, 0.4823529, 0, 1,
6.787879, 4.626263, -536.6777, 1, 0.4823529, 0, 1,
6.828283, 4.626263, -536.9449, 1, 0.4823529, 0, 1,
6.868687, 4.626263, -537.2274, 1, 0.4823529, 0, 1,
6.909091, 4.626263, -537.5251, 1, 0.4823529, 0, 1,
6.949495, 4.626263, -537.8381, 1, 0.4823529, 0, 1,
6.989899, 4.626263, -538.1664, 1, 0.4823529, 0, 1,
7.030303, 4.626263, -538.5099, 1, 0.4823529, 0, 1,
7.070707, 4.626263, -538.8687, 1, 0.4823529, 0, 1,
7.111111, 4.626263, -539.2427, 1, 0.4823529, 0, 1,
7.151515, 4.626263, -539.6319, 1, 0.4823529, 0, 1,
7.191919, 4.626263, -540.0364, 1, 0.4823529, 0, 1,
7.232323, 4.626263, -540.4562, 1, 0.4823529, 0, 1,
7.272727, 4.626263, -540.8912, 1, 0.4823529, 0, 1,
7.313131, 4.626263, -541.3415, 1, 0.3764706, 0, 1,
7.353535, 4.626263, -541.807, 1, 0.3764706, 0, 1,
7.393939, 4.626263, -542.2878, 1, 0.3764706, 0, 1,
7.434343, 4.626263, -542.7839, 1, 0.3764706, 0, 1,
7.474748, 4.626263, -543.2952, 1, 0.3764706, 0, 1,
7.515152, 4.626263, -543.8217, 1, 0.3764706, 0, 1,
7.555555, 4.626263, -544.3635, 1, 0.3764706, 0, 1,
7.59596, 4.626263, -544.9206, 1, 0.3764706, 0, 1,
7.636364, 4.626263, -545.4929, 1, 0.3764706, 0, 1,
7.676768, 4.626263, -546.0805, 1, 0.3764706, 0, 1,
7.717172, 4.626263, -546.6833, 1, 0.2745098, 0, 1,
7.757576, 4.626263, -547.3015, 1, 0.2745098, 0, 1,
7.79798, 4.626263, -547.9348, 1, 0.2745098, 0, 1,
7.838384, 4.626263, -548.5834, 1, 0.2745098, 0, 1,
7.878788, 4.626263, -549.2473, 1, 0.2745098, 0, 1,
7.919192, 4.626263, -549.9263, 1, 0.2745098, 0, 1,
7.959596, 4.626263, -550.6207, 1, 0.2745098, 0, 1,
8, 4.626263, -551.3303, 1, 0.2745098, 0, 1,
4, 4.676768, -555.8576, 1, 0.1686275, 0, 1,
4.040404, 4.676768, -555.0891, 1, 0.1686275, 0, 1,
4.080808, 4.676768, -554.3354, 1, 0.1686275, 0, 1,
4.121212, 4.676768, -553.5968, 1, 0.1686275, 0, 1,
4.161616, 4.676768, -552.873, 1, 0.1686275, 0, 1,
4.20202, 4.676768, -552.1642, 1, 0.1686275, 0, 1,
4.242424, 4.676768, -551.4703, 1, 0.2745098, 0, 1,
4.282828, 4.676768, -550.7914, 1, 0.2745098, 0, 1,
4.323232, 4.676768, -550.1274, 1, 0.2745098, 0, 1,
4.363636, 4.676768, -549.4783, 1, 0.2745098, 0, 1,
4.40404, 4.676768, -548.8441, 1, 0.2745098, 0, 1,
4.444445, 4.676768, -548.2249, 1, 0.2745098, 0, 1,
4.484848, 4.676768, -547.6205, 1, 0.2745098, 0, 1,
4.525252, 4.676768, -547.0311, 1, 0.2745098, 0, 1,
4.565657, 4.676768, -546.4567, 1, 0.2745098, 0, 1,
4.606061, 4.676768, -545.8972, 1, 0.3764706, 0, 1,
4.646465, 4.676768, -545.3525, 1, 0.3764706, 0, 1,
4.686869, 4.676768, -544.8229, 1, 0.3764706, 0, 1,
4.727273, 4.676768, -544.3081, 1, 0.3764706, 0, 1,
4.767677, 4.676768, -543.8083, 1, 0.3764706, 0, 1,
4.808081, 4.676768, -543.3234, 1, 0.3764706, 0, 1,
4.848485, 4.676768, -542.8534, 1, 0.3764706, 0, 1,
4.888889, 4.676768, -542.3984, 1, 0.3764706, 0, 1,
4.929293, 4.676768, -541.9583, 1, 0.3764706, 0, 1,
4.969697, 4.676768, -541.5331, 1, 0.3764706, 0, 1,
5.010101, 4.676768, -541.1228, 1, 0.3764706, 0, 1,
5.050505, 4.676768, -540.7275, 1, 0.4823529, 0, 1,
5.090909, 4.676768, -540.347, 1, 0.4823529, 0, 1,
5.131313, 4.676768, -539.9816, 1, 0.4823529, 0, 1,
5.171717, 4.676768, -539.631, 1, 0.4823529, 0, 1,
5.212121, 4.676768, -539.2954, 1, 0.4823529, 0, 1,
5.252525, 4.676768, -538.9747, 1, 0.4823529, 0, 1,
5.292929, 4.676768, -538.6689, 1, 0.4823529, 0, 1,
5.333333, 4.676768, -538.3781, 1, 0.4823529, 0, 1,
5.373737, 4.676768, -538.1022, 1, 0.4823529, 0, 1,
5.414141, 4.676768, -537.8412, 1, 0.4823529, 0, 1,
5.454545, 4.676768, -537.5952, 1, 0.4823529, 0, 1,
5.494949, 4.676768, -537.364, 1, 0.4823529, 0, 1,
5.535354, 4.676768, -537.1478, 1, 0.4823529, 0, 1,
5.575758, 4.676768, -536.9465, 1, 0.4823529, 0, 1,
5.616162, 4.676768, -536.7602, 1, 0.4823529, 0, 1,
5.656566, 4.676768, -536.5887, 1, 0.4823529, 0, 1,
5.69697, 4.676768, -536.4323, 1, 0.4823529, 0, 1,
5.737374, 4.676768, -536.2907, 1, 0.4823529, 0, 1,
5.777778, 4.676768, -536.1641, 1, 0.4823529, 0, 1,
5.818182, 4.676768, -536.0524, 1, 0.4823529, 0, 1,
5.858586, 4.676768, -535.9556, 1, 0.4823529, 0, 1,
5.89899, 4.676768, -535.8737, 1, 0.4823529, 0, 1,
5.939394, 4.676768, -535.8068, 1, 0.4823529, 0, 1,
5.979798, 4.676768, -535.7548, 1, 0.4823529, 0, 1,
6.020202, 4.676768, -535.7177, 1, 0.4823529, 0, 1,
6.060606, 4.676768, -535.6956, 1, 0.4823529, 0, 1,
6.10101, 4.676768, -535.6884, 1, 0.4823529, 0, 1,
6.141414, 4.676768, -535.696, 1, 0.4823529, 0, 1,
6.181818, 4.676768, -535.7187, 1, 0.4823529, 0, 1,
6.222222, 4.676768, -535.7563, 1, 0.4823529, 0, 1,
6.262626, 4.676768, -535.8088, 1, 0.4823529, 0, 1,
6.30303, 4.676768, -535.8762, 1, 0.4823529, 0, 1,
6.343434, 4.676768, -535.9586, 1, 0.4823529, 0, 1,
6.383838, 4.676768, -536.0558, 1, 0.4823529, 0, 1,
6.424242, 4.676768, -536.168, 1, 0.4823529, 0, 1,
6.464646, 4.676768, -536.2952, 1, 0.4823529, 0, 1,
6.505051, 4.676768, -536.4372, 1, 0.4823529, 0, 1,
6.545455, 4.676768, -536.5942, 1, 0.4823529, 0, 1,
6.585859, 4.676768, -536.7661, 1, 0.4823529, 0, 1,
6.626263, 4.676768, -536.9529, 1, 0.4823529, 0, 1,
6.666667, 4.676768, -537.1547, 1, 0.4823529, 0, 1,
6.707071, 4.676768, -537.3714, 1, 0.4823529, 0, 1,
6.747475, 4.676768, -537.603, 1, 0.4823529, 0, 1,
6.787879, 4.676768, -537.8495, 1, 0.4823529, 0, 1,
6.828283, 4.676768, -538.111, 1, 0.4823529, 0, 1,
6.868687, 4.676768, -538.3875, 1, 0.4823529, 0, 1,
6.909091, 4.676768, -538.6788, 1, 0.4823529, 0, 1,
6.949495, 4.676768, -538.985, 1, 0.4823529, 0, 1,
6.989899, 4.676768, -539.3062, 1, 0.4823529, 0, 1,
7.030303, 4.676768, -539.6423, 1, 0.4823529, 0, 1,
7.070707, 4.676768, -539.9934, 1, 0.4823529, 0, 1,
7.111111, 4.676768, -540.3594, 1, 0.4823529, 0, 1,
7.151515, 4.676768, -540.7402, 1, 0.4823529, 0, 1,
7.191919, 4.676768, -541.1361, 1, 0.3764706, 0, 1,
7.232323, 4.676768, -541.5468, 1, 0.3764706, 0, 1,
7.272727, 4.676768, -541.9725, 1, 0.3764706, 0, 1,
7.313131, 4.676768, -542.4131, 1, 0.3764706, 0, 1,
7.353535, 4.676768, -542.8687, 1, 0.3764706, 0, 1,
7.393939, 4.676768, -543.3391, 1, 0.3764706, 0, 1,
7.434343, 4.676768, -543.8245, 1, 0.3764706, 0, 1,
7.474748, 4.676768, -544.3248, 1, 0.3764706, 0, 1,
7.515152, 4.676768, -544.8401, 1, 0.3764706, 0, 1,
7.555555, 4.676768, -545.3702, 1, 0.3764706, 0, 1,
7.59596, 4.676768, -545.9153, 1, 0.3764706, 0, 1,
7.636364, 4.676768, -546.4754, 1, 0.2745098, 0, 1,
7.676768, 4.676768, -547.0504, 1, 0.2745098, 0, 1,
7.717172, 4.676768, -547.6402, 1, 0.2745098, 0, 1,
7.757576, 4.676768, -548.2451, 1, 0.2745098, 0, 1,
7.79798, 4.676768, -548.8647, 1, 0.2745098, 0, 1,
7.838384, 4.676768, -549.4995, 1, 0.2745098, 0, 1,
7.878788, 4.676768, -550.149, 1, 0.2745098, 0, 1,
7.919192, 4.676768, -550.8135, 1, 0.2745098, 0, 1,
7.959596, 4.676768, -551.493, 1, 0.2745098, 0, 1,
8, 4.676768, -552.1874, 1, 0.1686275, 0, 1,
4, 4.727273, -556.6552, 1, 0.1686275, 0, 1,
4.040404, 4.727273, -555.903, 1, 0.1686275, 0, 1,
4.080808, 4.727273, -555.1655, 1, 0.1686275, 0, 1,
4.121212, 4.727273, -554.4425, 1, 0.1686275, 0, 1,
4.161616, 4.727273, -553.7341, 1, 0.1686275, 0, 1,
4.20202, 4.727273, -553.0403, 1, 0.1686275, 0, 1,
4.242424, 4.727273, -552.3613, 1, 0.1686275, 0, 1,
4.282828, 4.727273, -551.6967, 1, 0.2745098, 0, 1,
4.323232, 4.727273, -551.0468, 1, 0.2745098, 0, 1,
4.363636, 4.727273, -550.4115, 1, 0.2745098, 0, 1,
4.40404, 4.727273, -549.7908, 1, 0.2745098, 0, 1,
4.444445, 4.727273, -549.1847, 1, 0.2745098, 0, 1,
4.484848, 4.727273, -548.5933, 1, 0.2745098, 0, 1,
4.525252, 4.727273, -548.0164, 1, 0.2745098, 0, 1,
4.565657, 4.727273, -547.4541, 1, 0.2745098, 0, 1,
4.606061, 4.727273, -546.9065, 1, 0.2745098, 0, 1,
4.646465, 4.727273, -546.3734, 1, 0.3764706, 0, 1,
4.686869, 4.727273, -545.855, 1, 0.3764706, 0, 1,
4.727273, 4.727273, -545.3512, 1, 0.3764706, 0, 1,
4.767677, 4.727273, -544.862, 1, 0.3764706, 0, 1,
4.808081, 4.727273, -544.3874, 1, 0.3764706, 0, 1,
4.848485, 4.727273, -543.9274, 1, 0.3764706, 0, 1,
4.888889, 4.727273, -543.4821, 1, 0.3764706, 0, 1,
4.929293, 4.727273, -543.0513, 1, 0.3764706, 0, 1,
4.969697, 4.727273, -542.6351, 1, 0.3764706, 0, 1,
5.010101, 4.727273, -542.2336, 1, 0.3764706, 0, 1,
5.050505, 4.727273, -541.8467, 1, 0.3764706, 0, 1,
5.090909, 4.727273, -541.4744, 1, 0.3764706, 0, 1,
5.131313, 4.727273, -541.1166, 1, 0.3764706, 0, 1,
5.171717, 4.727273, -540.7735, 1, 0.4823529, 0, 1,
5.212121, 4.727273, -540.445, 1, 0.4823529, 0, 1,
5.252525, 4.727273, -540.1312, 1, 0.4823529, 0, 1,
5.292929, 4.727273, -539.8319, 1, 0.4823529, 0, 1,
5.333333, 4.727273, -539.5472, 1, 0.4823529, 0, 1,
5.373737, 4.727273, -539.2772, 1, 0.4823529, 0, 1,
5.414141, 4.727273, -539.0217, 1, 0.4823529, 0, 1,
5.454545, 4.727273, -538.7809, 1, 0.4823529, 0, 1,
5.494949, 4.727273, -538.5547, 1, 0.4823529, 0, 1,
5.535354, 4.727273, -538.3431, 1, 0.4823529, 0, 1,
5.575758, 4.727273, -538.1461, 1, 0.4823529, 0, 1,
5.616162, 4.727273, -537.9637, 1, 0.4823529, 0, 1,
5.656566, 4.727273, -537.7959, 1, 0.4823529, 0, 1,
5.69697, 4.727273, -537.6428, 1, 0.4823529, 0, 1,
5.737374, 4.727273, -537.5042, 1, 0.4823529, 0, 1,
5.777778, 4.727273, -537.3802, 1, 0.4823529, 0, 1,
5.818182, 4.727273, -537.2709, 1, 0.4823529, 0, 1,
5.858586, 4.727273, -537.1762, 1, 0.4823529, 0, 1,
5.89899, 4.727273, -537.0961, 1, 0.4823529, 0, 1,
5.939394, 4.727273, -537.0306, 1, 0.4823529, 0, 1,
5.979798, 4.727273, -536.9797, 1, 0.4823529, 0, 1,
6.020202, 4.727273, -536.9434, 1, 0.4823529, 0, 1,
6.060606, 4.727273, -536.9217, 1, 0.4823529, 0, 1,
6.10101, 4.727273, -536.9147, 1, 0.4823529, 0, 1,
6.141414, 4.727273, -536.9222, 1, 0.4823529, 0, 1,
6.181818, 4.727273, -536.9443, 1, 0.4823529, 0, 1,
6.222222, 4.727273, -536.9811, 1, 0.4823529, 0, 1,
6.262626, 4.727273, -537.0325, 1, 0.4823529, 0, 1,
6.30303, 4.727273, -537.0985, 1, 0.4823529, 0, 1,
6.343434, 4.727273, -537.1791, 1, 0.4823529, 0, 1,
6.383838, 4.727273, -537.2743, 1, 0.4823529, 0, 1,
6.424242, 4.727273, -537.3841, 1, 0.4823529, 0, 1,
6.464646, 4.727273, -537.5085, 1, 0.4823529, 0, 1,
6.505051, 4.727273, -537.6476, 1, 0.4823529, 0, 1,
6.545455, 4.727273, -537.8012, 1, 0.4823529, 0, 1,
6.585859, 4.727273, -537.9695, 1, 0.4823529, 0, 1,
6.626263, 4.727273, -538.1523, 1, 0.4823529, 0, 1,
6.666667, 4.727273, -538.3499, 1, 0.4823529, 0, 1,
6.707071, 4.727273, -538.5619, 1, 0.4823529, 0, 1,
6.747475, 4.727273, -538.7886, 1, 0.4823529, 0, 1,
6.787879, 4.727273, -539.0299, 1, 0.4823529, 0, 1,
6.828283, 4.727273, -539.2858, 1, 0.4823529, 0, 1,
6.868687, 4.727273, -539.5564, 1, 0.4823529, 0, 1,
6.909091, 4.727273, -539.8415, 1, 0.4823529, 0, 1,
6.949495, 4.727273, -540.1413, 1, 0.4823529, 0, 1,
6.989899, 4.727273, -540.4556, 1, 0.4823529, 0, 1,
7.030303, 4.727273, -540.7846, 1, 0.4823529, 0, 1,
7.070707, 4.727273, -541.1282, 1, 0.3764706, 0, 1,
7.111111, 4.727273, -541.4864, 1, 0.3764706, 0, 1,
7.151515, 4.727273, -541.8592, 1, 0.3764706, 0, 1,
7.191919, 4.727273, -542.2466, 1, 0.3764706, 0, 1,
7.232323, 4.727273, -542.6486, 1, 0.3764706, 0, 1,
7.272727, 4.727273, -543.0652, 1, 0.3764706, 0, 1,
7.313131, 4.727273, -543.4965, 1, 0.3764706, 0, 1,
7.353535, 4.727273, -543.9423, 1, 0.3764706, 0, 1,
7.393939, 4.727273, -544.4028, 1, 0.3764706, 0, 1,
7.434343, 4.727273, -544.8779, 1, 0.3764706, 0, 1,
7.474748, 4.727273, -545.3676, 1, 0.3764706, 0, 1,
7.515152, 4.727273, -545.8719, 1, 0.3764706, 0, 1,
7.555555, 4.727273, -546.3907, 1, 0.2745098, 0, 1,
7.59596, 4.727273, -546.9243, 1, 0.2745098, 0, 1,
7.636364, 4.727273, -547.4724, 1, 0.2745098, 0, 1,
7.676768, 4.727273, -548.0352, 1, 0.2745098, 0, 1,
7.717172, 4.727273, -548.6125, 1, 0.2745098, 0, 1,
7.757576, 4.727273, -549.2045, 1, 0.2745098, 0, 1,
7.79798, 4.727273, -549.811, 1, 0.2745098, 0, 1,
7.838384, 4.727273, -550.4322, 1, 0.2745098, 0, 1,
7.878788, 4.727273, -551.068, 1, 0.2745098, 0, 1,
7.919192, 4.727273, -551.7184, 1, 0.2745098, 0, 1,
7.959596, 4.727273, -552.3834, 1, 0.1686275, 0, 1,
8, 4.727273, -553.063, 1, 0.1686275, 0, 1,
4, 4.777778, -557.4726, 1, 0.06666667, 0, 1,
4.040404, 4.777778, -556.7363, 1, 0.1686275, 0, 1,
4.080808, 4.777778, -556.0142, 1, 0.1686275, 0, 1,
4.121212, 4.777778, -555.3064, 1, 0.1686275, 0, 1,
4.161616, 4.777778, -554.613, 1, 0.1686275, 0, 1,
4.20202, 4.777778, -553.9338, 1, 0.1686275, 0, 1,
4.242424, 4.777778, -553.2689, 1, 0.1686275, 0, 1,
4.282828, 4.777778, -552.6184, 1, 0.1686275, 0, 1,
4.323232, 4.777778, -551.9821, 1, 0.1686275, 0, 1,
4.363636, 4.777778, -551.3602, 1, 0.2745098, 0, 1,
4.40404, 4.777778, -550.7526, 1, 0.2745098, 0, 1,
4.444445, 4.777778, -550.1592, 1, 0.2745098, 0, 1,
4.484848, 4.777778, -549.5802, 1, 0.2745098, 0, 1,
4.525252, 4.777778, -549.0154, 1, 0.2745098, 0, 1,
4.565657, 4.777778, -548.465, 1, 0.2745098, 0, 1,
4.606061, 4.777778, -547.9289, 1, 0.2745098, 0, 1,
4.646465, 4.777778, -547.407, 1, 0.2745098, 0, 1,
4.686869, 4.777778, -546.8995, 1, 0.2745098, 0, 1,
4.727273, 4.777778, -546.4063, 1, 0.2745098, 0, 1,
4.767677, 4.777778, -545.9274, 1, 0.3764706, 0, 1,
4.808081, 4.777778, -545.4628, 1, 0.3764706, 0, 1,
4.848485, 4.777778, -545.0125, 1, 0.3764706, 0, 1,
4.888889, 4.777778, -544.5765, 1, 0.3764706, 0, 1,
4.929293, 4.777778, -544.1548, 1, 0.3764706, 0, 1,
4.969697, 4.777778, -543.7474, 1, 0.3764706, 0, 1,
5.010101, 4.777778, -543.3543, 1, 0.3764706, 0, 1,
5.050505, 4.777778, -542.9755, 1, 0.3764706, 0, 1,
5.090909, 4.777778, -542.611, 1, 0.3764706, 0, 1,
5.131313, 4.777778, -542.2608, 1, 0.3764706, 0, 1,
5.171717, 4.777778, -541.9249, 1, 0.3764706, 0, 1,
5.212121, 4.777778, -541.6033, 1, 0.3764706, 0, 1,
5.252525, 4.777778, -541.2961, 1, 0.3764706, 0, 1,
5.292929, 4.777778, -541.0031, 1, 0.4823529, 0, 1,
5.333333, 4.777778, -540.7244, 1, 0.4823529, 0, 1,
5.373737, 4.777778, -540.46, 1, 0.4823529, 0, 1,
5.414141, 4.777778, -540.21, 1, 0.4823529, 0, 1,
5.454545, 4.777778, -539.9742, 1, 0.4823529, 0, 1,
5.494949, 4.777778, -539.7527, 1, 0.4823529, 0, 1,
5.535354, 4.777778, -539.5456, 1, 0.4823529, 0, 1,
5.575758, 4.777778, -539.3527, 1, 0.4823529, 0, 1,
5.616162, 4.777778, -539.1742, 1, 0.4823529, 0, 1,
5.656566, 4.777778, -539.0099, 1, 0.4823529, 0, 1,
5.69697, 4.777778, -538.86, 1, 0.4823529, 0, 1,
5.737374, 4.777778, -538.7244, 1, 0.4823529, 0, 1,
5.777778, 4.777778, -538.603, 1, 0.4823529, 0, 1,
5.818182, 4.777778, -538.496, 1, 0.4823529, 0, 1,
5.858586, 4.777778, -538.4033, 1, 0.4823529, 0, 1,
5.89899, 4.777778, -538.3248, 1, 0.4823529, 0, 1,
5.939394, 4.777778, -538.2607, 1, 0.4823529, 0, 1,
5.979798, 4.777778, -538.2109, 1, 0.4823529, 0, 1,
6.020202, 4.777778, -538.1754, 1, 0.4823529, 0, 1,
6.060606, 4.777778, -538.1541, 1, 0.4823529, 0, 1,
6.10101, 4.777778, -538.1472, 1, 0.4823529, 0, 1,
6.141414, 4.777778, -538.1546, 1, 0.4823529, 0, 1,
6.181818, 4.777778, -538.1763, 1, 0.4823529, 0, 1,
6.222222, 4.777778, -538.2123, 1, 0.4823529, 0, 1,
6.262626, 4.777778, -538.2626, 1, 0.4823529, 0, 1,
6.30303, 4.777778, -538.3271, 1, 0.4823529, 0, 1,
6.343434, 4.777778, -538.4061, 1, 0.4823529, 0, 1,
6.383838, 4.777778, -538.4993, 1, 0.4823529, 0, 1,
6.424242, 4.777778, -538.6068, 1, 0.4823529, 0, 1,
6.464646, 4.777778, -538.7286, 1, 0.4823529, 0, 1,
6.505051, 4.777778, -538.8647, 1, 0.4823529, 0, 1,
6.545455, 4.777778, -539.0151, 1, 0.4823529, 0, 1,
6.585859, 4.777778, -539.1799, 1, 0.4823529, 0, 1,
6.626263, 4.777778, -539.3589, 1, 0.4823529, 0, 1,
6.666667, 4.777778, -539.5522, 1, 0.4823529, 0, 1,
6.707071, 4.777778, -539.7598, 1, 0.4823529, 0, 1,
6.747475, 4.777778, -539.9818, 1, 0.4823529, 0, 1,
6.787879, 4.777778, -540.218, 1, 0.4823529, 0, 1,
6.828283, 4.777778, -540.4685, 1, 0.4823529, 0, 1,
6.868687, 4.777778, -540.7334, 1, 0.4823529, 0, 1,
6.909091, 4.777778, -541.0125, 1, 0.4823529, 0, 1,
6.949495, 4.777778, -541.306, 1, 0.3764706, 0, 1,
6.989899, 4.777778, -541.6137, 1, 0.3764706, 0, 1,
7.030303, 4.777778, -541.9358, 1, 0.3764706, 0, 1,
7.070707, 4.777778, -542.2721, 1, 0.3764706, 0, 1,
7.111111, 4.777778, -542.6228, 1, 0.3764706, 0, 1,
7.151515, 4.777778, -542.9877, 1, 0.3764706, 0, 1,
7.191919, 4.777778, -543.367, 1, 0.3764706, 0, 1,
7.232323, 4.777778, -543.7606, 1, 0.3764706, 0, 1,
7.272727, 4.777778, -544.1685, 1, 0.3764706, 0, 1,
7.313131, 4.777778, -544.5906, 1, 0.3764706, 0, 1,
7.353535, 4.777778, -545.0271, 1, 0.3764706, 0, 1,
7.393939, 4.777778, -545.4779, 1, 0.3764706, 0, 1,
7.434343, 4.777778, -545.943, 1, 0.3764706, 0, 1,
7.474748, 4.777778, -546.4224, 1, 0.2745098, 0, 1,
7.515152, 4.777778, -546.9161, 1, 0.2745098, 0, 1,
7.555555, 4.777778, -547.424, 1, 0.2745098, 0, 1,
7.59596, 4.777778, -547.9464, 1, 0.2745098, 0, 1,
7.636364, 4.777778, -548.4829, 1, 0.2745098, 0, 1,
7.676768, 4.777778, -549.0338, 1, 0.2745098, 0, 1,
7.717172, 4.777778, -549.5991, 1, 0.2745098, 0, 1,
7.757576, 4.777778, -550.1785, 1, 0.2745098, 0, 1,
7.79798, 4.777778, -550.7723, 1, 0.2745098, 0, 1,
7.838384, 4.777778, -551.3805, 1, 0.2745098, 0, 1,
7.878788, 4.777778, -552.0029, 1, 0.1686275, 0, 1,
7.919192, 4.777778, -552.6396, 1, 0.1686275, 0, 1,
7.959596, 4.777778, -553.2906, 1, 0.1686275, 0, 1,
8, 4.777778, -553.9559, 1, 0.1686275, 0, 1,
4, 4.828283, -558.3085, 1, 0.06666667, 0, 1,
4.040404, 4.828283, -557.5875, 1, 0.06666667, 0, 1,
4.080808, 4.828283, -556.8804, 1, 0.1686275, 0, 1,
4.121212, 4.828283, -556.1874, 1, 0.1686275, 0, 1,
4.161616, 4.828283, -555.5084, 1, 0.1686275, 0, 1,
4.20202, 4.828283, -554.8433, 1, 0.1686275, 0, 1,
4.242424, 4.828283, -554.1923, 1, 0.1686275, 0, 1,
4.282828, 4.828283, -553.5553, 1, 0.1686275, 0, 1,
4.323232, 4.828283, -552.9323, 1, 0.1686275, 0, 1,
4.363636, 4.828283, -552.3233, 1, 0.1686275, 0, 1,
4.40404, 4.828283, -551.7283, 1, 0.2745098, 0, 1,
4.444445, 4.828283, -551.1473, 1, 0.2745098, 0, 1,
4.484848, 4.828283, -550.5803, 1, 0.2745098, 0, 1,
4.525252, 4.828283, -550.0273, 1, 0.2745098, 0, 1,
4.565657, 4.828283, -549.4883, 1, 0.2745098, 0, 1,
4.606061, 4.828283, -548.9634, 1, 0.2745098, 0, 1,
4.646465, 4.828283, -548.4525, 1, 0.2745098, 0, 1,
4.686869, 4.828283, -547.9554, 1, 0.2745098, 0, 1,
4.727273, 4.828283, -547.4725, 1, 0.2745098, 0, 1,
4.767677, 4.828283, -547.0035, 1, 0.2745098, 0, 1,
4.808081, 4.828283, -546.5486, 1, 0.2745098, 0, 1,
4.848485, 4.828283, -546.1077, 1, 0.3764706, 0, 1,
4.888889, 4.828283, -545.6807, 1, 0.3764706, 0, 1,
4.929293, 4.828283, -545.2678, 1, 0.3764706, 0, 1,
4.969697, 4.828283, -544.8689, 1, 0.3764706, 0, 1,
5.010101, 4.828283, -544.484, 1, 0.3764706, 0, 1,
5.050505, 4.828283, -544.1131, 1, 0.3764706, 0, 1,
5.090909, 4.828283, -543.7562, 1, 0.3764706, 0, 1,
5.131313, 4.828283, -543.4133, 1, 0.3764706, 0, 1,
5.171717, 4.828283, -543.0844, 1, 0.3764706, 0, 1,
5.212121, 4.828283, -542.7695, 1, 0.3764706, 0, 1,
5.252525, 4.828283, -542.4686, 1, 0.3764706, 0, 1,
5.292929, 4.828283, -542.1817, 1, 0.3764706, 0, 1,
5.333333, 4.828283, -541.9089, 1, 0.3764706, 0, 1,
5.373737, 4.828283, -541.65, 1, 0.3764706, 0, 1,
5.414141, 4.828283, -541.4051, 1, 0.3764706, 0, 1,
5.454545, 4.828283, -541.1743, 1, 0.3764706, 0, 1,
5.494949, 4.828283, -540.9574, 1, 0.4823529, 0, 1,
5.535354, 4.828283, -540.7546, 1, 0.4823529, 0, 1,
5.575758, 4.828283, -540.5657, 1, 0.4823529, 0, 1,
5.616162, 4.828283, -540.3909, 1, 0.4823529, 0, 1,
5.656566, 4.828283, -540.23, 1, 0.4823529, 0, 1,
5.69697, 4.828283, -540.0833, 1, 0.4823529, 0, 1,
5.737374, 4.828283, -539.9504, 1, 0.4823529, 0, 1,
5.777778, 4.828283, -539.8316, 1, 0.4823529, 0, 1,
5.818182, 4.828283, -539.7268, 1, 0.4823529, 0, 1,
5.858586, 4.828283, -539.636, 1, 0.4823529, 0, 1,
5.89899, 4.828283, -539.5592, 1, 0.4823529, 0, 1,
5.939394, 4.828283, -539.4964, 1, 0.4823529, 0, 1,
5.979798, 4.828283, -539.4476, 1, 0.4823529, 0, 1,
6.020202, 4.828283, -539.4128, 1, 0.4823529, 0, 1,
6.060606, 4.828283, -539.392, 1, 0.4823529, 0, 1,
6.10101, 4.828283, -539.3853, 1, 0.4823529, 0, 1,
6.141414, 4.828283, -539.3925, 1, 0.4823529, 0, 1,
6.181818, 4.828283, -539.4138, 1, 0.4823529, 0, 1,
6.222222, 4.828283, -539.449, 1, 0.4823529, 0, 1,
6.262626, 4.828283, -539.4982, 1, 0.4823529, 0, 1,
6.30303, 4.828283, -539.5615, 1, 0.4823529, 0, 1,
6.343434, 4.828283, -539.6387, 1, 0.4823529, 0, 1,
6.383838, 4.828283, -539.73, 1, 0.4823529, 0, 1,
6.424242, 4.828283, -539.8353, 1, 0.4823529, 0, 1,
6.464646, 4.828283, -539.9546, 1, 0.4823529, 0, 1,
6.505051, 4.828283, -540.0878, 1, 0.4823529, 0, 1,
6.545455, 4.828283, -540.2351, 1, 0.4823529, 0, 1,
6.585859, 4.828283, -540.3964, 1, 0.4823529, 0, 1,
6.626263, 4.828283, -540.5717, 1, 0.4823529, 0, 1,
6.666667, 4.828283, -540.761, 1, 0.4823529, 0, 1,
6.707071, 4.828283, -540.9644, 1, 0.4823529, 0, 1,
6.747475, 4.828283, -541.1816, 1, 0.3764706, 0, 1,
6.787879, 4.828283, -541.413, 1, 0.3764706, 0, 1,
6.828283, 4.828283, -541.6583, 1, 0.3764706, 0, 1,
6.868687, 4.828283, -541.9176, 1, 0.3764706, 0, 1,
6.909091, 4.828283, -542.191, 1, 0.3764706, 0, 1,
6.949495, 4.828283, -542.4783, 1, 0.3764706, 0, 1,
6.989899, 4.828283, -542.7797, 1, 0.3764706, 0, 1,
7.030303, 4.828283, -543.095, 1, 0.3764706, 0, 1,
7.070707, 4.828283, -543.4244, 1, 0.3764706, 0, 1,
7.111111, 4.828283, -543.7677, 1, 0.3764706, 0, 1,
7.151515, 4.828283, -544.1251, 1, 0.3764706, 0, 1,
7.191919, 4.828283, -544.4965, 1, 0.3764706, 0, 1,
7.232323, 4.828283, -544.8818, 1, 0.3764706, 0, 1,
7.272727, 4.828283, -545.2812, 1, 0.3764706, 0, 1,
7.313131, 4.828283, -545.6946, 1, 0.3764706, 0, 1,
7.353535, 4.828283, -546.122, 1, 0.3764706, 0, 1,
7.393939, 4.828283, -546.5634, 1, 0.2745098, 0, 1,
7.434343, 4.828283, -547.0188, 1, 0.2745098, 0, 1,
7.474748, 4.828283, -547.4882, 1, 0.2745098, 0, 1,
7.515152, 4.828283, -547.9716, 1, 0.2745098, 0, 1,
7.555555, 4.828283, -548.4691, 1, 0.2745098, 0, 1,
7.59596, 4.828283, -548.9805, 1, 0.2745098, 0, 1,
7.636364, 4.828283, -549.5059, 1, 0.2745098, 0, 1,
7.676768, 4.828283, -550.0453, 1, 0.2745098, 0, 1,
7.717172, 4.828283, -550.5988, 1, 0.2745098, 0, 1,
7.757576, 4.828283, -551.1663, 1, 0.2745098, 0, 1,
7.79798, 4.828283, -551.7477, 1, 0.2745098, 0, 1,
7.838384, 4.828283, -552.3431, 1, 0.1686275, 0, 1,
7.878788, 4.828283, -552.9526, 1, 0.1686275, 0, 1,
7.919192, 4.828283, -553.576, 1, 0.1686275, 0, 1,
7.959596, 4.828283, -554.2136, 1, 0.1686275, 0, 1,
8, 4.828283, -554.8651, 1, 0.1686275, 0, 1,
4, 4.878788, -559.1617, 1, 0.06666667, 0, 1,
4.040404, 4.878788, -558.4554, 1, 0.06666667, 0, 1,
4.080808, 4.878788, -557.763, 1, 0.06666667, 0, 1,
4.121212, 4.878788, -557.0842, 1, 0.1686275, 0, 1,
4.161616, 4.878788, -556.4192, 1, 0.1686275, 0, 1,
4.20202, 4.878788, -555.7678, 1, 0.1686275, 0, 1,
4.242424, 4.878788, -555.1302, 1, 0.1686275, 0, 1,
4.282828, 4.878788, -554.5063, 1, 0.1686275, 0, 1,
4.323232, 4.878788, -553.8962, 1, 0.1686275, 0, 1,
4.363636, 4.878788, -553.2997, 1, 0.1686275, 0, 1,
4.40404, 4.878788, -552.717, 1, 0.1686275, 0, 1,
4.444445, 4.878788, -552.1479, 1, 0.1686275, 0, 1,
4.484848, 4.878788, -551.5927, 1, 0.2745098, 0, 1,
4.525252, 4.878788, -551.051, 1, 0.2745098, 0, 1,
4.565657, 4.878788, -550.5232, 1, 0.2745098, 0, 1,
4.606061, 4.878788, -550.009, 1, 0.2745098, 0, 1,
4.646465, 4.878788, -549.5086, 1, 0.2745098, 0, 1,
4.686869, 4.878788, -549.0219, 1, 0.2745098, 0, 1,
4.727273, 4.878788, -548.5488, 1, 0.2745098, 0, 1,
4.767677, 4.878788, -548.0895, 1, 0.2745098, 0, 1,
4.808081, 4.878788, -547.644, 1, 0.2745098, 0, 1,
4.848485, 4.878788, -547.2122, 1, 0.2745098, 0, 1,
4.888889, 4.878788, -546.794, 1, 0.2745098, 0, 1,
4.929293, 4.878788, -546.3896, 1, 0.2745098, 0, 1,
4.969697, 4.878788, -545.9988, 1, 0.3764706, 0, 1,
5.010101, 4.878788, -545.6219, 1, 0.3764706, 0, 1,
5.050505, 4.878788, -545.2586, 1, 0.3764706, 0, 1,
5.090909, 4.878788, -544.9091, 1, 0.3764706, 0, 1,
5.131313, 4.878788, -544.5732, 1, 0.3764706, 0, 1,
5.171717, 4.878788, -544.2511, 1, 0.3764706, 0, 1,
5.212121, 4.878788, -543.9427, 1, 0.3764706, 0, 1,
5.252525, 4.878788, -543.648, 1, 0.3764706, 0, 1,
5.292929, 4.878788, -543.367, 1, 0.3764706, 0, 1,
5.333333, 4.878788, -543.0998, 1, 0.3764706, 0, 1,
5.373737, 4.878788, -542.8463, 1, 0.3764706, 0, 1,
5.414141, 4.878788, -542.6064, 1, 0.3764706, 0, 1,
5.454545, 4.878788, -542.3803, 1, 0.3764706, 0, 1,
5.494949, 4.878788, -542.1679, 1, 0.3764706, 0, 1,
5.535354, 4.878788, -541.9692, 1, 0.3764706, 0, 1,
5.575758, 4.878788, -541.7843, 1, 0.3764706, 0, 1,
5.616162, 4.878788, -541.6131, 1, 0.3764706, 0, 1,
5.656566, 4.878788, -541.4556, 1, 0.3764706, 0, 1,
5.69697, 4.878788, -541.3118, 1, 0.3764706, 0, 1,
5.737374, 4.878788, -541.1817, 1, 0.3764706, 0, 1,
5.777778, 4.878788, -541.0653, 1, 0.3764706, 0, 1,
5.818182, 4.878788, -540.9626, 1, 0.4823529, 0, 1,
5.858586, 4.878788, -540.8737, 1, 0.4823529, 0, 1,
5.89899, 4.878788, -540.7985, 1, 0.4823529, 0, 1,
5.939394, 4.878788, -540.737, 1, 0.4823529, 0, 1,
5.979798, 4.878788, -540.6892, 1, 0.4823529, 0, 1,
6.020202, 4.878788, -540.6552, 1, 0.4823529, 0, 1,
6.060606, 4.878788, -540.6348, 1, 0.4823529, 0, 1,
6.10101, 4.878788, -540.6282, 1, 0.4823529, 0, 1,
6.141414, 4.878788, -540.6353, 1, 0.4823529, 0, 1,
6.181818, 4.878788, -540.6561, 1, 0.4823529, 0, 1,
6.222222, 4.878788, -540.6906, 1, 0.4823529, 0, 1,
6.262626, 4.878788, -540.7388, 1, 0.4823529, 0, 1,
6.30303, 4.878788, -540.8008, 1, 0.4823529, 0, 1,
6.343434, 4.878788, -540.8765, 1, 0.4823529, 0, 1,
6.383838, 4.878788, -540.9658, 1, 0.4823529, 0, 1,
6.424242, 4.878788, -541.0689, 1, 0.3764706, 0, 1,
6.464646, 4.878788, -541.1857, 1, 0.3764706, 0, 1,
6.505051, 4.878788, -541.3163, 1, 0.3764706, 0, 1,
6.545455, 4.878788, -541.4605, 1, 0.3764706, 0, 1,
6.585859, 4.878788, -541.6185, 1, 0.3764706, 0, 1,
6.626263, 4.878788, -541.7902, 1, 0.3764706, 0, 1,
6.666667, 4.878788, -541.9756, 1, 0.3764706, 0, 1,
6.707071, 4.878788, -542.1747, 1, 0.3764706, 0, 1,
6.747475, 4.878788, -542.3876, 1, 0.3764706, 0, 1,
6.787879, 4.878788, -542.6141, 1, 0.3764706, 0, 1,
6.828283, 4.878788, -542.8544, 1, 0.3764706, 0, 1,
6.868687, 4.878788, -543.1084, 1, 0.3764706, 0, 1,
6.909091, 4.878788, -543.3761, 1, 0.3764706, 0, 1,
6.949495, 4.878788, -543.6575, 1, 0.3764706, 0, 1,
6.989899, 4.878788, -543.9526, 1, 0.3764706, 0, 1,
7.030303, 4.878788, -544.2615, 1, 0.3764706, 0, 1,
7.070707, 4.878788, -544.584, 1, 0.3764706, 0, 1,
7.111111, 4.878788, -544.9203, 1, 0.3764706, 0, 1,
7.151515, 4.878788, -545.2704, 1, 0.3764706, 0, 1,
7.191919, 4.878788, -545.6341, 1, 0.3764706, 0, 1,
7.232323, 4.878788, -546.0115, 1, 0.3764706, 0, 1,
7.272727, 4.878788, -546.4027, 1, 0.2745098, 0, 1,
7.313131, 4.878788, -546.8076, 1, 0.2745098, 0, 1,
7.353535, 4.878788, -547.2261, 1, 0.2745098, 0, 1,
7.393939, 4.878788, -547.6584, 1, 0.2745098, 0, 1,
7.434343, 4.878788, -548.1045, 1, 0.2745098, 0, 1,
7.474748, 4.878788, -548.5642, 1, 0.2745098, 0, 1,
7.515152, 4.878788, -549.0377, 1, 0.2745098, 0, 1,
7.555555, 4.878788, -549.5248, 1, 0.2745098, 0, 1,
7.59596, 4.878788, -550.0258, 1, 0.2745098, 0, 1,
7.636364, 4.878788, -550.5403, 1, 0.2745098, 0, 1,
7.676768, 4.878788, -551.0687, 1, 0.2745098, 0, 1,
7.717172, 4.878788, -551.6107, 1, 0.2745098, 0, 1,
7.757576, 4.878788, -552.1665, 1, 0.1686275, 0, 1,
7.79798, 4.878788, -552.736, 1, 0.1686275, 0, 1,
7.838384, 4.878788, -553.3192, 1, 0.1686275, 0, 1,
7.878788, 4.878788, -553.9161, 1, 0.1686275, 0, 1,
7.919192, 4.878788, -554.5267, 1, 0.1686275, 0, 1,
7.959596, 4.878788, -555.1511, 1, 0.1686275, 0, 1,
8, 4.878788, -555.7891, 1, 0.1686275, 0, 1,
4, 4.929293, -560.0309, 1, 0.06666667, 0, 1,
4.040404, 4.929293, -559.3391, 1, 0.06666667, 0, 1,
4.080808, 4.929293, -558.6608, 1, 0.06666667, 0, 1,
4.121212, 4.929293, -557.9958, 1, 0.06666667, 0, 1,
4.161616, 4.929293, -557.3444, 1, 0.06666667, 0, 1,
4.20202, 4.929293, -556.7063, 1, 0.1686275, 0, 1,
4.242424, 4.929293, -556.0817, 1, 0.1686275, 0, 1,
4.282828, 4.929293, -555.4705, 1, 0.1686275, 0, 1,
4.323232, 4.929293, -554.8728, 1, 0.1686275, 0, 1,
4.363636, 4.929293, -554.2885, 1, 0.1686275, 0, 1,
4.40404, 4.929293, -553.7177, 1, 0.1686275, 0, 1,
4.444445, 4.929293, -553.1602, 1, 0.1686275, 0, 1,
4.484848, 4.929293, -552.6162, 1, 0.1686275, 0, 1,
4.525252, 4.929293, -552.0857, 1, 0.1686275, 0, 1,
4.565657, 4.929293, -551.5685, 1, 0.2745098, 0, 1,
4.606061, 4.929293, -551.0649, 1, 0.2745098, 0, 1,
4.646465, 4.929293, -550.5746, 1, 0.2745098, 0, 1,
4.686869, 4.929293, -550.0978, 1, 0.2745098, 0, 1,
4.727273, 4.929293, -549.6345, 1, 0.2745098, 0, 1,
4.767677, 4.929293, -549.1846, 1, 0.2745098, 0, 1,
4.808081, 4.929293, -548.748, 1, 0.2745098, 0, 1,
4.848485, 4.929293, -548.325, 1, 0.2745098, 0, 1,
4.888889, 4.929293, -547.9154, 1, 0.2745098, 0, 1,
4.929293, 4.929293, -547.5192, 1, 0.2745098, 0, 1,
4.969697, 4.929293, -547.1365, 1, 0.2745098, 0, 1,
5.010101, 4.929293, -546.7672, 1, 0.2745098, 0, 1,
5.050505, 4.929293, -546.4113, 1, 0.2745098, 0, 1,
5.090909, 4.929293, -546.0689, 1, 0.3764706, 0, 1,
5.131313, 4.929293, -545.7399, 1, 0.3764706, 0, 1,
5.171717, 4.929293, -545.4243, 1, 0.3764706, 0, 1,
5.212121, 4.929293, -545.1223, 1, 0.3764706, 0, 1,
5.252525, 4.929293, -544.8336, 1, 0.3764706, 0, 1,
5.292929, 4.929293, -544.5583, 1, 0.3764706, 0, 1,
5.333333, 4.929293, -544.2965, 1, 0.3764706, 0, 1,
5.373737, 4.929293, -544.0482, 1, 0.3764706, 0, 1,
5.414141, 4.929293, -543.8132, 1, 0.3764706, 0, 1,
5.454545, 4.929293, -543.5917, 1, 0.3764706, 0, 1,
5.494949, 4.929293, -543.3837, 1, 0.3764706, 0, 1,
5.535354, 4.929293, -543.189, 1, 0.3764706, 0, 1,
5.575758, 4.929293, -543.0079, 1, 0.3764706, 0, 1,
5.616162, 4.929293, -542.8401, 1, 0.3764706, 0, 1,
5.656566, 4.929293, -542.6858, 1, 0.3764706, 0, 1,
5.69697, 4.929293, -542.5449, 1, 0.3764706, 0, 1,
5.737374, 4.929293, -542.4175, 1, 0.3764706, 0, 1,
5.777778, 4.929293, -542.3035, 1, 0.3764706, 0, 1,
5.818182, 4.929293, -542.2029, 1, 0.3764706, 0, 1,
5.858586, 4.929293, -542.1158, 1, 0.3764706, 0, 1,
5.89899, 4.929293, -542.0421, 1, 0.3764706, 0, 1,
5.939394, 4.929293, -541.9819, 1, 0.3764706, 0, 1,
5.979798, 4.929293, -541.9351, 1, 0.3764706, 0, 1,
6.020202, 4.929293, -541.9017, 1, 0.3764706, 0, 1,
6.060606, 4.929293, -541.8818, 1, 0.3764706, 0, 1,
6.10101, 4.929293, -541.8753, 1, 0.3764706, 0, 1,
6.141414, 4.929293, -541.8822, 1, 0.3764706, 0, 1,
6.181818, 4.929293, -541.9026, 1, 0.3764706, 0, 1,
6.222222, 4.929293, -541.9364, 1, 0.3764706, 0, 1,
6.262626, 4.929293, -541.9836, 1, 0.3764706, 0, 1,
6.30303, 4.929293, -542.0444, 1, 0.3764706, 0, 1,
6.343434, 4.929293, -542.1185, 1, 0.3764706, 0, 1,
6.383838, 4.929293, -542.2061, 1, 0.3764706, 0, 1,
6.424242, 4.929293, -542.3071, 1, 0.3764706, 0, 1,
6.464646, 4.929293, -542.4215, 1, 0.3764706, 0, 1,
6.505051, 4.929293, -542.5494, 1, 0.3764706, 0, 1,
6.545455, 4.929293, -542.6907, 1, 0.3764706, 0, 1,
6.585859, 4.929293, -542.8454, 1, 0.3764706, 0, 1,
6.626263, 4.929293, -543.0136, 1, 0.3764706, 0, 1,
6.666667, 4.929293, -543.1953, 1, 0.3764706, 0, 1,
6.707071, 4.929293, -543.3903, 1, 0.3764706, 0, 1,
6.747475, 4.929293, -543.5988, 1, 0.3764706, 0, 1,
6.787879, 4.929293, -543.8207, 1, 0.3764706, 0, 1,
6.828283, 4.929293, -544.0561, 1, 0.3764706, 0, 1,
6.868687, 4.929293, -544.3049, 1, 0.3764706, 0, 1,
6.909091, 4.929293, -544.5671, 1, 0.3764706, 0, 1,
6.949495, 4.929293, -544.8428, 1, 0.3764706, 0, 1,
6.989899, 4.929293, -545.132, 1, 0.3764706, 0, 1,
7.030303, 4.929293, -545.4345, 1, 0.3764706, 0, 1,
7.070707, 4.929293, -545.7505, 1, 0.3764706, 0, 1,
7.111111, 4.929293, -546.08, 1, 0.3764706, 0, 1,
7.151515, 4.929293, -546.4229, 1, 0.2745098, 0, 1,
7.191919, 4.929293, -546.7791, 1, 0.2745098, 0, 1,
7.232323, 4.929293, -547.1489, 1, 0.2745098, 0, 1,
7.272727, 4.929293, -547.532, 1, 0.2745098, 0, 1,
7.313131, 4.929293, -547.9287, 1, 0.2745098, 0, 1,
7.353535, 4.929293, -548.3387, 1, 0.2745098, 0, 1,
7.393939, 4.929293, -548.7623, 1, 0.2745098, 0, 1,
7.434343, 4.929293, -549.1992, 1, 0.2745098, 0, 1,
7.474748, 4.929293, -549.6495, 1, 0.2745098, 0, 1,
7.515152, 4.929293, -550.1133, 1, 0.2745098, 0, 1,
7.555555, 4.929293, -550.5906, 1, 0.2745098, 0, 1,
7.59596, 4.929293, -551.0813, 1, 0.2745098, 0, 1,
7.636364, 4.929293, -551.5854, 1, 0.2745098, 0, 1,
7.676768, 4.929293, -552.103, 1, 0.1686275, 0, 1,
7.717172, 4.929293, -552.634, 1, 0.1686275, 0, 1,
7.757576, 4.929293, -553.1783, 1, 0.1686275, 0, 1,
7.79798, 4.929293, -553.7363, 1, 0.1686275, 0, 1,
7.838384, 4.929293, -554.3076, 1, 0.1686275, 0, 1,
7.878788, 4.929293, -554.8923, 1, 0.1686275, 0, 1,
7.919192, 4.929293, -555.4905, 1, 0.1686275, 0, 1,
7.959596, 4.929293, -556.1021, 1, 0.1686275, 0, 1,
8, 4.929293, -556.7271, 1, 0.1686275, 0, 1,
4, 4.979798, -560.9152, 1, 0.06666667, 0, 1,
4.040404, 4.979798, -560.2374, 1, 0.06666667, 0, 1,
4.080808, 4.979798, -559.5728, 1, 0.06666667, 0, 1,
4.121212, 4.979798, -558.9212, 1, 0.06666667, 0, 1,
4.161616, 4.979798, -558.2829, 1, 0.06666667, 0, 1,
4.20202, 4.979798, -557.6577, 1, 0.06666667, 0, 1,
4.242424, 4.979798, -557.0457, 1, 0.1686275, 0, 1,
4.282828, 4.979798, -556.4469, 1, 0.1686275, 0, 1,
4.323232, 4.979798, -555.8612, 1, 0.1686275, 0, 1,
4.363636, 4.979798, -555.2887, 1, 0.1686275, 0, 1,
4.40404, 4.979798, -554.7294, 1, 0.1686275, 0, 1,
4.444445, 4.979798, -554.1832, 1, 0.1686275, 0, 1,
4.484848, 4.979798, -553.6501, 1, 0.1686275, 0, 1,
4.525252, 4.979798, -553.1303, 1, 0.1686275, 0, 1,
4.565657, 4.979798, -552.6237, 1, 0.1686275, 0, 1,
4.606061, 4.979798, -552.1301, 1, 0.1686275, 0, 1,
4.646465, 4.979798, -551.6498, 1, 0.2745098, 0, 1,
4.686869, 4.979798, -551.1826, 1, 0.2745098, 0, 1,
4.727273, 4.979798, -550.7286, 1, 0.2745098, 0, 1,
4.767677, 4.979798, -550.2878, 1, 0.2745098, 0, 1,
4.808081, 4.979798, -549.86, 1, 0.2745098, 0, 1,
4.848485, 4.979798, -549.4456, 1, 0.2745098, 0, 1,
4.888889, 4.979798, -549.0442, 1, 0.2745098, 0, 1,
4.929293, 4.979798, -548.656, 1, 0.2745098, 0, 1,
4.969697, 4.979798, -548.281, 1, 0.2745098, 0, 1,
5.010101, 4.979798, -547.9192, 1, 0.2745098, 0, 1,
5.050505, 4.979798, -547.5705, 1, 0.2745098, 0, 1,
5.090909, 4.979798, -547.235, 1, 0.2745098, 0, 1,
5.131313, 4.979798, -546.9126, 1, 0.2745098, 0, 1,
5.171717, 4.979798, -546.6035, 1, 0.2745098, 0, 1,
5.212121, 4.979798, -546.3074, 1, 0.3764706, 0, 1,
5.252525, 4.979798, -546.0245, 1, 0.3764706, 0, 1,
5.292929, 4.979798, -545.7549, 1, 0.3764706, 0, 1,
5.333333, 4.979798, -545.4984, 1, 0.3764706, 0, 1,
5.373737, 4.979798, -545.255, 1, 0.3764706, 0, 1,
5.414141, 4.979798, -545.0248, 1, 0.3764706, 0, 1,
5.454545, 4.979798, -544.8078, 1, 0.3764706, 0, 1,
5.494949, 4.979798, -544.6039, 1, 0.3764706, 0, 1,
5.535354, 4.979798, -544.4132, 1, 0.3764706, 0, 1,
5.575758, 4.979798, -544.2357, 1, 0.3764706, 0, 1,
5.616162, 4.979798, -544.0714, 1, 0.3764706, 0, 1,
5.656566, 4.979798, -543.9202, 1, 0.3764706, 0, 1,
5.69697, 4.979798, -543.7821, 1, 0.3764706, 0, 1,
5.737374, 4.979798, -543.6573, 1, 0.3764706, 0, 1,
5.777778, 4.979798, -543.5456, 1, 0.3764706, 0, 1,
5.818182, 4.979798, -543.447, 1, 0.3764706, 0, 1,
5.858586, 4.979798, -543.3617, 1, 0.3764706, 0, 1,
5.89899, 4.979798, -543.2895, 1, 0.3764706, 0, 1,
5.939394, 4.979798, -543.2305, 1, 0.3764706, 0, 1,
5.979798, 4.979798, -543.1846, 1, 0.3764706, 0, 1,
6.020202, 4.979798, -543.1519, 1, 0.3764706, 0, 1,
6.060606, 4.979798, -543.1324, 1, 0.3764706, 0, 1,
6.10101, 4.979798, -543.126, 1, 0.3764706, 0, 1,
6.141414, 4.979798, -543.1328, 1, 0.3764706, 0, 1,
6.181818, 4.979798, -543.1528, 1, 0.3764706, 0, 1,
6.222222, 4.979798, -543.1859, 1, 0.3764706, 0, 1,
6.262626, 4.979798, -543.2322, 1, 0.3764706, 0, 1,
6.30303, 4.979798, -543.2916, 1, 0.3764706, 0, 1,
6.343434, 4.979798, -543.3643, 1, 0.3764706, 0, 1,
6.383838, 4.979798, -543.4501, 1, 0.3764706, 0, 1,
6.424242, 4.979798, -543.5491, 1, 0.3764706, 0, 1,
6.464646, 4.979798, -543.6612, 1, 0.3764706, 0, 1,
6.505051, 4.979798, -543.7865, 1, 0.3764706, 0, 1,
6.545455, 4.979798, -543.9249, 1, 0.3764706, 0, 1,
6.585859, 4.979798, -544.0765, 1, 0.3764706, 0, 1,
6.626263, 4.979798, -544.2413, 1, 0.3764706, 0, 1,
6.666667, 4.979798, -544.4193, 1, 0.3764706, 0, 1,
6.707071, 4.979798, -544.6104, 1, 0.3764706, 0, 1,
6.747475, 4.979798, -544.8147, 1, 0.3764706, 0, 1,
6.787879, 4.979798, -545.0322, 1, 0.3764706, 0, 1,
6.828283, 4.979798, -545.2628, 1, 0.3764706, 0, 1,
6.868687, 4.979798, -545.5066, 1, 0.3764706, 0, 1,
6.909091, 4.979798, -545.7635, 1, 0.3764706, 0, 1,
6.949495, 4.979798, -546.0337, 1, 0.3764706, 0, 1,
6.989899, 4.979798, -546.317, 1, 0.3764706, 0, 1,
7.030303, 4.979798, -546.6134, 1, 0.2745098, 0, 1,
7.070707, 4.979798, -546.923, 1, 0.2745098, 0, 1,
7.111111, 4.979798, -547.2458, 1, 0.2745098, 0, 1,
7.151515, 4.979798, -547.5818, 1, 0.2745098, 0, 1,
7.191919, 4.979798, -547.9309, 1, 0.2745098, 0, 1,
7.232323, 4.979798, -548.2932, 1, 0.2745098, 0, 1,
7.272727, 4.979798, -548.6686, 1, 0.2745098, 0, 1,
7.313131, 4.979798, -549.0573, 1, 0.2745098, 0, 1,
7.353535, 4.979798, -549.459, 1, 0.2745098, 0, 1,
7.393939, 4.979798, -549.874, 1, 0.2745098, 0, 1,
7.434343, 4.979798, -550.3021, 1, 0.2745098, 0, 1,
7.474748, 4.979798, -550.7433, 1, 0.2745098, 0, 1,
7.515152, 4.979798, -551.1978, 1, 0.2745098, 0, 1,
7.555555, 4.979798, -551.6654, 1, 0.2745098, 0, 1,
7.59596, 4.979798, -552.1462, 1, 0.1686275, 0, 1,
7.636364, 4.979798, -552.6401, 1, 0.1686275, 0, 1,
7.676768, 4.979798, -553.1473, 1, 0.1686275, 0, 1,
7.717172, 4.979798, -553.6675, 1, 0.1686275, 0, 1,
7.757576, 4.979798, -554.201, 1, 0.1686275, 0, 1,
7.79798, 4.979798, -554.7476, 1, 0.1686275, 0, 1,
7.838384, 4.979798, -555.3074, 1, 0.1686275, 0, 1,
7.878788, 4.979798, -555.8803, 1, 0.1686275, 0, 1,
7.919192, 4.979798, -556.4664, 1, 0.1686275, 0, 1,
7.959596, 4.979798, -557.0657, 1, 0.1686275, 0, 1,
8, 4.979798, -557.6781, 1, 0.06666667, 0, 1,
4, 5.030303, -561.8136, 1, 0.06666667, 0, 1,
4.040404, 5.030303, -561.1493, 1, 0.06666667, 0, 1,
4.080808, 5.030303, -560.4979, 1, 0.06666667, 0, 1,
4.121212, 5.030303, -559.8594, 1, 0.06666667, 0, 1,
4.161616, 5.030303, -559.2338, 1, 0.06666667, 0, 1,
4.20202, 5.030303, -558.6212, 1, 0.06666667, 0, 1,
4.242424, 5.030303, -558.0214, 1, 0.06666667, 0, 1,
4.282828, 5.030303, -557.4345, 1, 0.06666667, 0, 1,
4.323232, 5.030303, -556.8605, 1, 0.1686275, 0, 1,
4.363636, 5.030303, -556.2994, 1, 0.1686275, 0, 1,
4.40404, 5.030303, -555.7513, 1, 0.1686275, 0, 1,
4.444445, 5.030303, -555.216, 1, 0.1686275, 0, 1,
4.484848, 5.030303, -554.6937, 1, 0.1686275, 0, 1,
4.525252, 5.030303, -554.1842, 1, 0.1686275, 0, 1,
4.565657, 5.030303, -553.6876, 1, 0.1686275, 0, 1,
4.606061, 5.030303, -553.204, 1, 0.1686275, 0, 1,
4.646465, 5.030303, -552.7333, 1, 0.1686275, 0, 1,
4.686869, 5.030303, -552.2754, 1, 0.1686275, 0, 1,
4.727273, 5.030303, -551.8304, 1, 0.1686275, 0, 1,
4.767677, 5.030303, -551.3984, 1, 0.2745098, 0, 1,
4.808081, 5.030303, -550.9793, 1, 0.2745098, 0, 1,
4.848485, 5.030303, -550.5731, 1, 0.2745098, 0, 1,
4.888889, 5.030303, -550.1797, 1, 0.2745098, 0, 1,
4.929293, 5.030303, -549.7993, 1, 0.2745098, 0, 1,
4.969697, 5.030303, -549.4318, 1, 0.2745098, 0, 1,
5.010101, 5.030303, -549.0771, 1, 0.2745098, 0, 1,
5.050505, 5.030303, -548.7354, 1, 0.2745098, 0, 1,
5.090909, 5.030303, -548.4066, 1, 0.2745098, 0, 1,
5.131313, 5.030303, -548.0907, 1, 0.2745098, 0, 1,
5.171717, 5.030303, -547.7877, 1, 0.2745098, 0, 1,
5.212121, 5.030303, -547.4976, 1, 0.2745098, 0, 1,
5.252525, 5.030303, -547.2204, 1, 0.2745098, 0, 1,
5.292929, 5.030303, -546.9561, 1, 0.2745098, 0, 1,
5.333333, 5.030303, -546.7047, 1, 0.2745098, 0, 1,
5.373737, 5.030303, -546.4662, 1, 0.2745098, 0, 1,
5.414141, 5.030303, -546.2406, 1, 0.3764706, 0, 1,
5.454545, 5.030303, -546.028, 1, 0.3764706, 0, 1,
5.494949, 5.030303, -545.8281, 1, 0.3764706, 0, 1,
5.535354, 5.030303, -545.6413, 1, 0.3764706, 0, 1,
5.575758, 5.030303, -545.4673, 1, 0.3764706, 0, 1,
5.616162, 5.030303, -545.3062, 1, 0.3764706, 0, 1,
5.656566, 5.030303, -545.158, 1, 0.3764706, 0, 1,
5.69697, 5.030303, -545.0228, 1, 0.3764706, 0, 1,
5.737374, 5.030303, -544.9004, 1, 0.3764706, 0, 1,
5.777778, 5.030303, -544.791, 1, 0.3764706, 0, 1,
5.818182, 5.030303, -544.6944, 1, 0.3764706, 0, 1,
5.858586, 5.030303, -544.6107, 1, 0.3764706, 0, 1,
5.89899, 5.030303, -544.54, 1, 0.3764706, 0, 1,
5.939394, 5.030303, -544.4821, 1, 0.3764706, 0, 1,
5.979798, 5.030303, -544.4372, 1, 0.3764706, 0, 1,
6.020202, 5.030303, -544.4052, 1, 0.3764706, 0, 1,
6.060606, 5.030303, -544.386, 1, 0.3764706, 0, 1,
6.10101, 5.030303, -544.3798, 1, 0.3764706, 0, 1,
6.141414, 5.030303, -544.3864, 1, 0.3764706, 0, 1,
6.181818, 5.030303, -544.406, 1, 0.3764706, 0, 1,
6.222222, 5.030303, -544.4385, 1, 0.3764706, 0, 1,
6.262626, 5.030303, -544.4838, 1, 0.3764706, 0, 1,
6.30303, 5.030303, -544.5421, 1, 0.3764706, 0, 1,
6.343434, 5.030303, -544.6133, 1, 0.3764706, 0, 1,
6.383838, 5.030303, -544.6974, 1, 0.3764706, 0, 1,
6.424242, 5.030303, -544.7944, 1, 0.3764706, 0, 1,
6.464646, 5.030303, -544.9042, 1, 0.3764706, 0, 1,
6.505051, 5.030303, -545.027, 1, 0.3764706, 0, 1,
6.545455, 5.030303, -545.1627, 1, 0.3764706, 0, 1,
6.585859, 5.030303, -545.3113, 1, 0.3764706, 0, 1,
6.626263, 5.030303, -545.4728, 1, 0.3764706, 0, 1,
6.666667, 5.030303, -545.6472, 1, 0.3764706, 0, 1,
6.707071, 5.030303, -545.8345, 1, 0.3764706, 0, 1,
6.747475, 5.030303, -546.0347, 1, 0.3764706, 0, 1,
6.787879, 5.030303, -546.2479, 1, 0.3764706, 0, 1,
6.828283, 5.030303, -546.4739, 1, 0.2745098, 0, 1,
6.868687, 5.030303, -546.7128, 1, 0.2745098, 0, 1,
6.909091, 5.030303, -546.9646, 1, 0.2745098, 0, 1,
6.949495, 5.030303, -547.2293, 1, 0.2745098, 0, 1,
6.989899, 5.030303, -547.507, 1, 0.2745098, 0, 1,
7.030303, 5.030303, -547.7975, 1, 0.2745098, 0, 1,
7.070707, 5.030303, -548.101, 1, 0.2745098, 0, 1,
7.111111, 5.030303, -548.4172, 1, 0.2745098, 0, 1,
7.151515, 5.030303, -548.7465, 1, 0.2745098, 0, 1,
7.191919, 5.030303, -549.0886, 1, 0.2745098, 0, 1,
7.232323, 5.030303, -549.4437, 1, 0.2745098, 0, 1,
7.272727, 5.030303, -549.8116, 1, 0.2745098, 0, 1,
7.313131, 5.030303, -550.1925, 1, 0.2745098, 0, 1,
7.353535, 5.030303, -550.5862, 1, 0.2745098, 0, 1,
7.393939, 5.030303, -550.9929, 1, 0.2745098, 0, 1,
7.434343, 5.030303, -551.4125, 1, 0.2745098, 0, 1,
7.474748, 5.030303, -551.8449, 1, 0.1686275, 0, 1,
7.515152, 5.030303, -552.2903, 1, 0.1686275, 0, 1,
7.555555, 5.030303, -552.7485, 1, 0.1686275, 0, 1,
7.59596, 5.030303, -553.2197, 1, 0.1686275, 0, 1,
7.636364, 5.030303, -553.7038, 1, 0.1686275, 0, 1,
7.676768, 5.030303, -554.2008, 1, 0.1686275, 0, 1,
7.717172, 5.030303, -554.7107, 1, 0.1686275, 0, 1,
7.757576, 5.030303, -555.2335, 1, 0.1686275, 0, 1,
7.79798, 5.030303, -555.7692, 1, 0.1686275, 0, 1,
7.838384, 5.030303, -556.3177, 1, 0.1686275, 0, 1,
7.878788, 5.030303, -556.8792, 1, 0.1686275, 0, 1,
7.919192, 5.030303, -557.4536, 1, 0.06666667, 0, 1,
7.959596, 5.030303, -558.041, 1, 0.06666667, 0, 1,
8, 5.030303, -558.6411, 1, 0.06666667, 0, 1,
4, 5.080808, -562.725, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.080808, -562.0738, 1, 0.06666667, 0, 1,
4.080808, 5.080808, -561.4353, 1, 0.06666667, 0, 1,
4.121212, 5.080808, -560.8094, 1, 0.06666667, 0, 1,
4.161616, 5.080808, -560.1962, 1, 0.06666667, 0, 1,
4.20202, 5.080808, -559.5957, 1, 0.06666667, 0, 1,
4.242424, 5.080808, -559.0078, 1, 0.06666667, 0, 1,
4.282828, 5.080808, -558.4325, 1, 0.06666667, 0, 1,
4.323232, 5.080808, -557.8699, 1, 0.06666667, 0, 1,
4.363636, 5.080808, -557.3199, 1, 0.06666667, 0, 1,
4.40404, 5.080808, -556.7826, 1, 0.1686275, 0, 1,
4.444445, 5.080808, -556.2579, 1, 0.1686275, 0, 1,
4.484848, 5.080808, -555.7459, 1, 0.1686275, 0, 1,
4.525252, 5.080808, -555.2465, 1, 0.1686275, 0, 1,
4.565657, 5.080808, -554.7598, 1, 0.1686275, 0, 1,
4.606061, 5.080808, -554.2857, 1, 0.1686275, 0, 1,
4.646465, 5.080808, -553.8243, 1, 0.1686275, 0, 1,
4.686869, 5.080808, -553.3755, 1, 0.1686275, 0, 1,
4.727273, 5.080808, -552.9393, 1, 0.1686275, 0, 1,
4.767677, 5.080808, -552.5159, 1, 0.1686275, 0, 1,
4.808081, 5.080808, -552.105, 1, 0.1686275, 0, 1,
4.848485, 5.080808, -551.7068, 1, 0.2745098, 0, 1,
4.888889, 5.080808, -551.3213, 1, 0.2745098, 0, 1,
4.929293, 5.080808, -550.9484, 1, 0.2745098, 0, 1,
4.969697, 5.080808, -550.5881, 1, 0.2745098, 0, 1,
5.010101, 5.080808, -550.2405, 1, 0.2745098, 0, 1,
5.050505, 5.080808, -549.9056, 1, 0.2745098, 0, 1,
5.090909, 5.080808, -549.5833, 1, 0.2745098, 0, 1,
5.131313, 5.080808, -549.2736, 1, 0.2745098, 0, 1,
5.171717, 5.080808, -548.9766, 1, 0.2745098, 0, 1,
5.212121, 5.080808, -548.6922, 1, 0.2745098, 0, 1,
5.252525, 5.080808, -548.4205, 1, 0.2745098, 0, 1,
5.292929, 5.080808, -548.1614, 1, 0.2745098, 0, 1,
5.333333, 5.080808, -547.915, 1, 0.2745098, 0, 1,
5.373737, 5.080808, -547.6812, 1, 0.2745098, 0, 1,
5.414141, 5.080808, -547.4601, 1, 0.2745098, 0, 1,
5.454545, 5.080808, -547.2516, 1, 0.2745098, 0, 1,
5.494949, 5.080808, -547.0558, 1, 0.2745098, 0, 1,
5.535354, 5.080808, -546.8726, 1, 0.2745098, 0, 1,
5.575758, 5.080808, -546.702, 1, 0.2745098, 0, 1,
5.616162, 5.080808, -546.5442, 1, 0.2745098, 0, 1,
5.656566, 5.080808, -546.3989, 1, 0.2745098, 0, 1,
5.69697, 5.080808, -546.2664, 1, 0.3764706, 0, 1,
5.737374, 5.080808, -546.1464, 1, 0.3764706, 0, 1,
5.777778, 5.080808, -546.0391, 1, 0.3764706, 0, 1,
5.818182, 5.080808, -545.9445, 1, 0.3764706, 0, 1,
5.858586, 5.080808, -545.8624, 1, 0.3764706, 0, 1,
5.89899, 5.080808, -545.7931, 1, 0.3764706, 0, 1,
5.939394, 5.080808, -545.7364, 1, 0.3764706, 0, 1,
5.979798, 5.080808, -545.6923, 1, 0.3764706, 0, 1,
6.020202, 5.080808, -545.6609, 1, 0.3764706, 0, 1,
6.060606, 5.080808, -545.6422, 1, 0.3764706, 0, 1,
6.10101, 5.080808, -545.636, 1, 0.3764706, 0, 1,
6.141414, 5.080808, -545.6426, 1, 0.3764706, 0, 1,
6.181818, 5.080808, -545.6617, 1, 0.3764706, 0, 1,
6.222222, 5.080808, -545.6935, 1, 0.3764706, 0, 1,
6.262626, 5.080808, -545.738, 1, 0.3764706, 0, 1,
6.30303, 5.080808, -545.7952, 1, 0.3764706, 0, 1,
6.343434, 5.080808, -545.8649, 1, 0.3764706, 0, 1,
6.383838, 5.080808, -545.9473, 1, 0.3764706, 0, 1,
6.424242, 5.080808, -546.0424, 1, 0.3764706, 0, 1,
6.464646, 5.080808, -546.1501, 1, 0.3764706, 0, 1,
6.505051, 5.080808, -546.2705, 1, 0.3764706, 0, 1,
6.545455, 5.080808, -546.4035, 1, 0.2745098, 0, 1,
6.585859, 5.080808, -546.5492, 1, 0.2745098, 0, 1,
6.626263, 5.080808, -546.7075, 1, 0.2745098, 0, 1,
6.666667, 5.080808, -546.8784, 1, 0.2745098, 0, 1,
6.707071, 5.080808, -547.062, 1, 0.2745098, 0, 1,
6.747475, 5.080808, -547.2583, 1, 0.2745098, 0, 1,
6.787879, 5.080808, -547.4672, 1, 0.2745098, 0, 1,
6.828283, 5.080808, -547.6887, 1, 0.2745098, 0, 1,
6.868687, 5.080808, -547.9229, 1, 0.2745098, 0, 1,
6.909091, 5.080808, -548.1697, 1, 0.2745098, 0, 1,
6.949495, 5.080808, -548.4293, 1, 0.2745098, 0, 1,
6.989899, 5.080808, -548.7014, 1, 0.2745098, 0, 1,
7.030303, 5.080808, -548.9861, 1, 0.2745098, 0, 1,
7.070707, 5.080808, -549.2836, 1, 0.2745098, 0, 1,
7.111111, 5.080808, -549.5937, 1, 0.2745098, 0, 1,
7.151515, 5.080808, -549.9164, 1, 0.2745098, 0, 1,
7.191919, 5.080808, -550.2518, 1, 0.2745098, 0, 1,
7.232323, 5.080808, -550.5998, 1, 0.2745098, 0, 1,
7.272727, 5.080808, -550.9604, 1, 0.2745098, 0, 1,
7.313131, 5.080808, -551.3338, 1, 0.2745098, 0, 1,
7.353535, 5.080808, -551.7197, 1, 0.2745098, 0, 1,
7.393939, 5.080808, -552.1183, 1, 0.1686275, 0, 1,
7.434343, 5.080808, -552.5296, 1, 0.1686275, 0, 1,
7.474748, 5.080808, -552.9536, 1, 0.1686275, 0, 1,
7.515152, 5.080808, -553.3901, 1, 0.1686275, 0, 1,
7.555555, 5.080808, -553.8393, 1, 0.1686275, 0, 1,
7.59596, 5.080808, -554.3011, 1, 0.1686275, 0, 1,
7.636364, 5.080808, -554.7756, 1, 0.1686275, 0, 1,
7.676768, 5.080808, -555.2628, 1, 0.1686275, 0, 1,
7.717172, 5.080808, -555.7626, 1, 0.1686275, 0, 1,
7.757576, 5.080808, -556.275, 1, 0.1686275, 0, 1,
7.79798, 5.080808, -556.8001, 1, 0.1686275, 0, 1,
7.838384, 5.080808, -557.3378, 1, 0.06666667, 0, 1,
7.878788, 5.080808, -557.8882, 1, 0.06666667, 0, 1,
7.919192, 5.080808, -558.4513, 1, 0.06666667, 0, 1,
7.959596, 5.080808, -559.027, 1, 0.06666667, 0, 1,
8, 5.080808, -559.6153, 1, 0.06666667, 0, 1,
4, 5.131313, -563.6486, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.131313, -563.0101, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.131313, -562.3842, 1, 0.06666667, 0, 1,
4.121212, 5.131313, -561.7706, 1, 0.06666667, 0, 1,
4.161616, 5.131313, -561.1694, 1, 0.06666667, 0, 1,
4.20202, 5.131313, -560.5806, 1, 0.06666667, 0, 1,
4.242424, 5.131313, -560.0042, 1, 0.06666667, 0, 1,
4.282828, 5.131313, -559.4402, 1, 0.06666667, 0, 1,
4.323232, 5.131313, -558.8885, 1, 0.06666667, 0, 1,
4.363636, 5.131313, -558.3494, 1, 0.06666667, 0, 1,
4.40404, 5.131313, -557.8226, 1, 0.06666667, 0, 1,
4.444445, 5.131313, -557.3082, 1, 0.06666667, 0, 1,
4.484848, 5.131313, -556.8062, 1, 0.1686275, 0, 1,
4.525252, 5.131313, -556.3166, 1, 0.1686275, 0, 1,
4.565657, 5.131313, -555.8394, 1, 0.1686275, 0, 1,
4.606061, 5.131313, -555.3746, 1, 0.1686275, 0, 1,
4.646465, 5.131313, -554.9222, 1, 0.1686275, 0, 1,
4.686869, 5.131313, -554.4822, 1, 0.1686275, 0, 1,
4.727273, 5.131313, -554.0546, 1, 0.1686275, 0, 1,
4.767677, 5.131313, -553.6394, 1, 0.1686275, 0, 1,
4.808081, 5.131313, -553.2366, 1, 0.1686275, 0, 1,
4.848485, 5.131313, -552.8462, 1, 0.1686275, 0, 1,
4.888889, 5.131313, -552.4682, 1, 0.1686275, 0, 1,
4.929293, 5.131313, -552.1026, 1, 0.1686275, 0, 1,
4.969697, 5.131313, -551.7494, 1, 0.2745098, 0, 1,
5.010101, 5.131313, -551.4086, 1, 0.2745098, 0, 1,
5.050505, 5.131313, -551.0802, 1, 0.2745098, 0, 1,
5.090909, 5.131313, -550.7642, 1, 0.2745098, 0, 1,
5.131313, 5.131313, -550.4606, 1, 0.2745098, 0, 1,
5.171717, 5.131313, -550.1694, 1, 0.2745098, 0, 1,
5.212121, 5.131313, -549.8906, 1, 0.2745098, 0, 1,
5.252525, 5.131313, -549.6243, 1, 0.2745098, 0, 1,
5.292929, 5.131313, -549.3702, 1, 0.2745098, 0, 1,
5.333333, 5.131313, -549.1287, 1, 0.2745098, 0, 1,
5.373737, 5.131313, -548.8995, 1, 0.2745098, 0, 1,
5.414141, 5.131313, -548.6827, 1, 0.2745098, 0, 1,
5.454545, 5.131313, -548.4783, 1, 0.2745098, 0, 1,
5.494949, 5.131313, -548.2863, 1, 0.2745098, 0, 1,
5.535354, 5.131313, -548.1067, 1, 0.2745098, 0, 1,
5.575758, 5.131313, -547.9395, 1, 0.2745098, 0, 1,
5.616162, 5.131313, -547.7847, 1, 0.2745098, 0, 1,
5.656566, 5.131313, -547.6423, 1, 0.2745098, 0, 1,
5.69697, 5.131313, -547.5123, 1, 0.2745098, 0, 1,
5.737374, 5.131313, -547.3947, 1, 0.2745098, 0, 1,
5.777778, 5.131313, -547.2895, 1, 0.2745098, 0, 1,
5.818182, 5.131313, -547.1967, 1, 0.2745098, 0, 1,
5.858586, 5.131313, -547.1163, 1, 0.2745098, 0, 1,
5.89899, 5.131313, -547.0483, 1, 0.2745098, 0, 1,
5.939394, 5.131313, -546.9927, 1, 0.2745098, 0, 1,
5.979798, 5.131313, -546.9495, 1, 0.2745098, 0, 1,
6.020202, 5.131313, -546.9187, 1, 0.2745098, 0, 1,
6.060606, 5.131313, -546.9003, 1, 0.2745098, 0, 1,
6.10101, 5.131313, -546.8943, 1, 0.2745098, 0, 1,
6.141414, 5.131313, -546.9008, 1, 0.2745098, 0, 1,
6.181818, 5.131313, -546.9196, 1, 0.2745098, 0, 1,
6.222222, 5.131313, -546.9507, 1, 0.2745098, 0, 1,
6.262626, 5.131313, -546.9943, 1, 0.2745098, 0, 1,
6.30303, 5.131313, -547.0504, 1, 0.2745098, 0, 1,
6.343434, 5.131313, -547.1188, 1, 0.2745098, 0, 1,
6.383838, 5.131313, -547.1996, 1, 0.2745098, 0, 1,
6.424242, 5.131313, -547.2928, 1, 0.2745098, 0, 1,
6.464646, 5.131313, -547.3984, 1, 0.2745098, 0, 1,
6.505051, 5.131313, -547.5164, 1, 0.2745098, 0, 1,
6.545455, 5.131313, -547.6468, 1, 0.2745098, 0, 1,
6.585859, 5.131313, -547.7896, 1, 0.2745098, 0, 1,
6.626263, 5.131313, -547.9448, 1, 0.2745098, 0, 1,
6.666667, 5.131313, -548.1124, 1, 0.2745098, 0, 1,
6.707071, 5.131313, -548.2924, 1, 0.2745098, 0, 1,
6.747475, 5.131313, -548.4848, 1, 0.2745098, 0, 1,
6.787879, 5.131313, -548.6896, 1, 0.2745098, 0, 1,
6.828283, 5.131313, -548.9068, 1, 0.2745098, 0, 1,
6.868687, 5.131313, -549.1364, 1, 0.2745098, 0, 1,
6.909091, 5.131313, -549.3784, 1, 0.2745098, 0, 1,
6.949495, 5.131313, -549.6328, 1, 0.2745098, 0, 1,
6.989899, 5.131313, -549.8997, 1, 0.2745098, 0, 1,
7.030303, 5.131313, -550.1788, 1, 0.2745098, 0, 1,
7.070707, 5.131313, -550.4705, 1, 0.2745098, 0, 1,
7.111111, 5.131313, -550.7745, 1, 0.2745098, 0, 1,
7.151515, 5.131313, -551.0909, 1, 0.2745098, 0, 1,
7.191919, 5.131313, -551.4197, 1, 0.2745098, 0, 1,
7.232323, 5.131313, -551.7609, 1, 0.1686275, 0, 1,
7.272727, 5.131313, -552.1145, 1, 0.1686275, 0, 1,
7.313131, 5.131313, -552.4805, 1, 0.1686275, 0, 1,
7.353535, 5.131313, -552.8589, 1, 0.1686275, 0, 1,
7.393939, 5.131313, -553.2497, 1, 0.1686275, 0, 1,
7.434343, 5.131313, -553.6529, 1, 0.1686275, 0, 1,
7.474748, 5.131313, -554.0685, 1, 0.1686275, 0, 1,
7.515152, 5.131313, -554.4965, 1, 0.1686275, 0, 1,
7.555555, 5.131313, -554.9369, 1, 0.1686275, 0, 1,
7.59596, 5.131313, -555.3897, 1, 0.1686275, 0, 1,
7.636364, 5.131313, -555.8549, 1, 0.1686275, 0, 1,
7.676768, 5.131313, -556.3325, 1, 0.1686275, 0, 1,
7.717172, 5.131313, -556.8225, 1, 0.1686275, 0, 1,
7.757576, 5.131313, -557.325, 1, 0.06666667, 0, 1,
7.79798, 5.131313, -557.8397, 1, 0.06666667, 0, 1,
7.838384, 5.131313, -558.3669, 1, 0.06666667, 0, 1,
7.878788, 5.131313, -558.9066, 1, 0.06666667, 0, 1,
7.919192, 5.131313, -559.4586, 1, 0.06666667, 0, 1,
7.959596, 5.131313, -560.0229, 1, 0.06666667, 0, 1,
8, 5.131313, -560.5998, 1, 0.06666667, 0, 1,
4, 5.181818, -564.5834, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.181818, -563.9574, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.181818, -563.3436, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.181818, -562.7419, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.181818, -562.1523, 1, 0.06666667, 0, 1,
4.20202, 5.181818, -561.575, 1, 0.06666667, 0, 1,
4.242424, 5.181818, -561.0097, 1, 0.06666667, 0, 1,
4.282828, 5.181818, -560.4567, 1, 0.06666667, 0, 1,
4.323232, 5.181818, -559.9158, 1, 0.06666667, 0, 1,
4.363636, 5.181818, -559.387, 1, 0.06666667, 0, 1,
4.40404, 5.181818, -558.8705, 1, 0.06666667, 0, 1,
4.444445, 5.181818, -558.366, 1, 0.06666667, 0, 1,
4.484848, 5.181818, -557.8738, 1, 0.06666667, 0, 1,
4.525252, 5.181818, -557.3937, 1, 0.06666667, 0, 1,
4.565657, 5.181818, -556.9257, 1, 0.1686275, 0, 1,
4.606061, 5.181818, -556.47, 1, 0.1686275, 0, 1,
4.646465, 5.181818, -556.0264, 1, 0.1686275, 0, 1,
4.686869, 5.181818, -555.5949, 1, 0.1686275, 0, 1,
4.727273, 5.181818, -555.1756, 1, 0.1686275, 0, 1,
4.767677, 5.181818, -554.7684, 1, 0.1686275, 0, 1,
4.808081, 5.181818, -554.3735, 1, 0.1686275, 0, 1,
4.848485, 5.181818, -553.9907, 1, 0.1686275, 0, 1,
4.888889, 5.181818, -553.62, 1, 0.1686275, 0, 1,
4.929293, 5.181818, -553.2615, 1, 0.1686275, 0, 1,
4.969697, 5.181818, -552.9151, 1, 0.1686275, 0, 1,
5.010101, 5.181818, -552.5809, 1, 0.1686275, 0, 1,
5.050505, 5.181818, -552.2589, 1, 0.1686275, 0, 1,
5.090909, 5.181818, -551.949, 1, 0.1686275, 0, 1,
5.131313, 5.181818, -551.6513, 1, 0.2745098, 0, 1,
5.171717, 5.181818, -551.3658, 1, 0.2745098, 0, 1,
5.212121, 5.181818, -551.0924, 1, 0.2745098, 0, 1,
5.252525, 5.181818, -550.8312, 1, 0.2745098, 0, 1,
5.292929, 5.181818, -550.5821, 1, 0.2745098, 0, 1,
5.333333, 5.181818, -550.3452, 1, 0.2745098, 0, 1,
5.373737, 5.181818, -550.1204, 1, 0.2745098, 0, 1,
5.414141, 5.181818, -549.9078, 1, 0.2745098, 0, 1,
5.454545, 5.181818, -549.7074, 1, 0.2745098, 0, 1,
5.494949, 5.181818, -549.5192, 1, 0.2745098, 0, 1,
5.535354, 5.181818, -549.343, 1, 0.2745098, 0, 1,
5.575758, 5.181818, -549.1791, 1, 0.2745098, 0, 1,
5.616162, 5.181818, -549.0273, 1, 0.2745098, 0, 1,
5.656566, 5.181818, -548.8876, 1, 0.2745098, 0, 1,
5.69697, 5.181818, -548.7602, 1, 0.2745098, 0, 1,
5.737374, 5.181818, -548.6449, 1, 0.2745098, 0, 1,
5.777778, 5.181818, -548.5417, 1, 0.2745098, 0, 1,
5.818182, 5.181818, -548.4507, 1, 0.2745098, 0, 1,
5.858586, 5.181818, -548.3719, 1, 0.2745098, 0, 1,
5.89899, 5.181818, -548.3052, 1, 0.2745098, 0, 1,
5.939394, 5.181818, -548.2507, 1, 0.2745098, 0, 1,
5.979798, 5.181818, -548.2083, 1, 0.2745098, 0, 1,
6.020202, 5.181818, -548.1782, 1, 0.2745098, 0, 1,
6.060606, 5.181818, -548.1601, 1, 0.2745098, 0, 1,
6.10101, 5.181818, -548.1542, 1, 0.2745098, 0, 1,
6.141414, 5.181818, -548.1605, 1, 0.2745098, 0, 1,
6.181818, 5.181818, -548.179, 1, 0.2745098, 0, 1,
6.222222, 5.181818, -548.2095, 1, 0.2745098, 0, 1,
6.262626, 5.181818, -548.2523, 1, 0.2745098, 0, 1,
6.30303, 5.181818, -548.3072, 1, 0.2745098, 0, 1,
6.343434, 5.181818, -548.3743, 1, 0.2745098, 0, 1,
6.383838, 5.181818, -548.4536, 1, 0.2745098, 0, 1,
6.424242, 5.181818, -548.5449, 1, 0.2745098, 0, 1,
6.464646, 5.181818, -548.6485, 1, 0.2745098, 0, 1,
6.505051, 5.181818, -548.7642, 1, 0.2745098, 0, 1,
6.545455, 5.181818, -548.8921, 1, 0.2745098, 0, 1,
6.585859, 5.181818, -549.0321, 1, 0.2745098, 0, 1,
6.626263, 5.181818, -549.1843, 1, 0.2745098, 0, 1,
6.666667, 5.181818, -549.3486, 1, 0.2745098, 0, 1,
6.707071, 5.181818, -549.5251, 1, 0.2745098, 0, 1,
6.747475, 5.181818, -549.7138, 1, 0.2745098, 0, 1,
6.787879, 5.181818, -549.9147, 1, 0.2745098, 0, 1,
6.828283, 5.181818, -550.1277, 1, 0.2745098, 0, 1,
6.868687, 5.181818, -550.3528, 1, 0.2745098, 0, 1,
6.909091, 5.181818, -550.5901, 1, 0.2745098, 0, 1,
6.949495, 5.181818, -550.8396, 1, 0.2745098, 0, 1,
6.989899, 5.181818, -551.1012, 1, 0.2745098, 0, 1,
7.030303, 5.181818, -551.375, 1, 0.2745098, 0, 1,
7.070707, 5.181818, -551.6609, 1, 0.2745098, 0, 1,
7.111111, 5.181818, -551.959, 1, 0.1686275, 0, 1,
7.151515, 5.181818, -552.2693, 1, 0.1686275, 0, 1,
7.191919, 5.181818, -552.5917, 1, 0.1686275, 0, 1,
7.232323, 5.181818, -552.9263, 1, 0.1686275, 0, 1,
7.272727, 5.181818, -553.2731, 1, 0.1686275, 0, 1,
7.313131, 5.181818, -553.632, 1, 0.1686275, 0, 1,
7.353535, 5.181818, -554.0031, 1, 0.1686275, 0, 1,
7.393939, 5.181818, -554.3863, 1, 0.1686275, 0, 1,
7.434343, 5.181818, -554.7817, 1, 0.1686275, 0, 1,
7.474748, 5.181818, -555.1892, 1, 0.1686275, 0, 1,
7.515152, 5.181818, -555.6089, 1, 0.1686275, 0, 1,
7.555555, 5.181818, -556.0408, 1, 0.1686275, 0, 1,
7.59596, 5.181818, -556.4848, 1, 0.1686275, 0, 1,
7.636364, 5.181818, -556.941, 1, 0.1686275, 0, 1,
7.676768, 5.181818, -557.4093, 1, 0.06666667, 0, 1,
7.717172, 5.181818, -557.8898, 1, 0.06666667, 0, 1,
7.757576, 5.181818, -558.3825, 1, 0.06666667, 0, 1,
7.79798, 5.181818, -558.8873, 1, 0.06666667, 0, 1,
7.838384, 5.181818, -559.4043, 1, 0.06666667, 0, 1,
7.878788, 5.181818, -559.9334, 1, 0.06666667, 0, 1,
7.919192, 5.181818, -560.4747, 1, 0.06666667, 0, 1,
7.959596, 5.181818, -561.0282, 1, 0.06666667, 0, 1,
8, 5.181818, -561.5938, 1, 0.06666667, 0, 1,
4, 5.232323, -565.5288, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.232323, -564.9148, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.232323, -564.3127, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.232323, -563.7226, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.232323, -563.1444, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.232323, -562.5781, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.232323, -562.0238, 1, 0.06666667, 0, 1,
4.282828, 5.232323, -561.4813, 1, 0.06666667, 0, 1,
4.323232, 5.232323, -560.9509, 1, 0.06666667, 0, 1,
4.363636, 5.232323, -560.4323, 1, 0.06666667, 0, 1,
4.40404, 5.232323, -559.9256, 1, 0.06666667, 0, 1,
4.444445, 5.232323, -559.4309, 1, 0.06666667, 0, 1,
4.484848, 5.232323, -558.9481, 1, 0.06666667, 0, 1,
4.525252, 5.232323, -558.4772, 1, 0.06666667, 0, 1,
4.565657, 5.232323, -558.0182, 1, 0.06666667, 0, 1,
4.606061, 5.232323, -557.5712, 1, 0.06666667, 0, 1,
4.646465, 5.232323, -557.1361, 1, 0.06666667, 0, 1,
4.686869, 5.232323, -556.713, 1, 0.1686275, 0, 1,
4.727273, 5.232323, -556.3017, 1, 0.1686275, 0, 1,
4.767677, 5.232323, -555.9024, 1, 0.1686275, 0, 1,
4.808081, 5.232323, -555.515, 1, 0.1686275, 0, 1,
4.848485, 5.232323, -555.1395, 1, 0.1686275, 0, 1,
4.888889, 5.232323, -554.776, 1, 0.1686275, 0, 1,
4.929293, 5.232323, -554.4244, 1, 0.1686275, 0, 1,
4.969697, 5.232323, -554.0847, 1, 0.1686275, 0, 1,
5.010101, 5.232323, -553.7569, 1, 0.1686275, 0, 1,
5.050505, 5.232323, -553.4411, 1, 0.1686275, 0, 1,
5.090909, 5.232323, -553.1371, 1, 0.1686275, 0, 1,
5.131313, 5.232323, -552.8452, 1, 0.1686275, 0, 1,
5.171717, 5.232323, -552.5651, 1, 0.1686275, 0, 1,
5.212121, 5.232323, -552.297, 1, 0.1686275, 0, 1,
5.252525, 5.232323, -552.0408, 1, 0.1686275, 0, 1,
5.292929, 5.232323, -551.7965, 1, 0.1686275, 0, 1,
5.333333, 5.232323, -551.5641, 1, 0.2745098, 0, 1,
5.373737, 5.232323, -551.3437, 1, 0.2745098, 0, 1,
5.414141, 5.232323, -551.1352, 1, 0.2745098, 0, 1,
5.454545, 5.232323, -550.9386, 1, 0.2745098, 0, 1,
5.494949, 5.232323, -550.754, 1, 0.2745098, 0, 1,
5.535354, 5.232323, -550.5812, 1, 0.2745098, 0, 1,
5.575758, 5.232323, -550.4204, 1, 0.2745098, 0, 1,
5.616162, 5.232323, -550.2715, 1, 0.2745098, 0, 1,
5.656566, 5.232323, -550.1346, 1, 0.2745098, 0, 1,
5.69697, 5.232323, -550.0096, 1, 0.2745098, 0, 1,
5.737374, 5.232323, -549.8965, 1, 0.2745098, 0, 1,
5.777778, 5.232323, -549.7953, 1, 0.2745098, 0, 1,
5.818182, 5.232323, -549.7061, 1, 0.2745098, 0, 1,
5.858586, 5.232323, -549.6287, 1, 0.2745098, 0, 1,
5.89899, 5.232323, -549.5634, 1, 0.2745098, 0, 1,
5.939394, 5.232323, -549.5098, 1, 0.2745098, 0, 1,
5.979798, 5.232323, -549.4683, 1, 0.2745098, 0, 1,
6.020202, 5.232323, -549.4387, 1, 0.2745098, 0, 1,
6.060606, 5.232323, -549.421, 1, 0.2745098, 0, 1,
6.10101, 5.232323, -549.4152, 1, 0.2745098, 0, 1,
6.141414, 5.232323, -549.4214, 1, 0.2745098, 0, 1,
6.181818, 5.232323, -549.4395, 1, 0.2745098, 0, 1,
6.222222, 5.232323, -549.4695, 1, 0.2745098, 0, 1,
6.262626, 5.232323, -549.5114, 1, 0.2745098, 0, 1,
6.30303, 5.232323, -549.5653, 1, 0.2745098, 0, 1,
6.343434, 5.232323, -549.6311, 1, 0.2745098, 0, 1,
6.383838, 5.232323, -549.7088, 1, 0.2745098, 0, 1,
6.424242, 5.232323, -549.7985, 1, 0.2745098, 0, 1,
6.464646, 5.232323, -549.9, 1, 0.2745098, 0, 1,
6.505051, 5.232323, -550.0135, 1, 0.2745098, 0, 1,
6.545455, 5.232323, -550.1389, 1, 0.2745098, 0, 1,
6.585859, 5.232323, -550.2762, 1, 0.2745098, 0, 1,
6.626263, 5.232323, -550.4255, 1, 0.2745098, 0, 1,
6.666667, 5.232323, -550.5867, 1, 0.2745098, 0, 1,
6.707071, 5.232323, -550.7598, 1, 0.2745098, 0, 1,
6.747475, 5.232323, -550.9449, 1, 0.2745098, 0, 1,
6.787879, 5.232323, -551.1418, 1, 0.2745098, 0, 1,
6.828283, 5.232323, -551.3508, 1, 0.2745098, 0, 1,
6.868687, 5.232323, -551.5716, 1, 0.2745098, 0, 1,
6.909091, 5.232323, -551.8043, 1, 0.1686275, 0, 1,
6.949495, 5.232323, -552.049, 1, 0.1686275, 0, 1,
6.989899, 5.232323, -552.3056, 1, 0.1686275, 0, 1,
7.030303, 5.232323, -552.5742, 1, 0.1686275, 0, 1,
7.070707, 5.232323, -552.8546, 1, 0.1686275, 0, 1,
7.111111, 5.232323, -553.147, 1, 0.1686275, 0, 1,
7.151515, 5.232323, -553.4513, 1, 0.1686275, 0, 1,
7.191919, 5.232323, -553.7675, 1, 0.1686275, 0, 1,
7.232323, 5.232323, -554.0957, 1, 0.1686275, 0, 1,
7.272727, 5.232323, -554.4358, 1, 0.1686275, 0, 1,
7.313131, 5.232323, -554.7878, 1, 0.1686275, 0, 1,
7.353535, 5.232323, -555.1517, 1, 0.1686275, 0, 1,
7.393939, 5.232323, -555.5276, 1, 0.1686275, 0, 1,
7.434343, 5.232323, -555.9153, 1, 0.1686275, 0, 1,
7.474748, 5.232323, -556.3151, 1, 0.1686275, 0, 1,
7.515152, 5.232323, -556.7267, 1, 0.1686275, 0, 1,
7.555555, 5.232323, -557.1503, 1, 0.06666667, 0, 1,
7.59596, 5.232323, -557.5858, 1, 0.06666667, 0, 1,
7.636364, 5.232323, -558.0332, 1, 0.06666667, 0, 1,
7.676768, 5.232323, -558.4926, 1, 0.06666667, 0, 1,
7.717172, 5.232323, -558.9638, 1, 0.06666667, 0, 1,
7.757576, 5.232323, -559.447, 1, 0.06666667, 0, 1,
7.79798, 5.232323, -559.9421, 1, 0.06666667, 0, 1,
7.838384, 5.232323, -560.4492, 1, 0.06666667, 0, 1,
7.878788, 5.232323, -560.9681, 1, 0.06666667, 0, 1,
7.919192, 5.232323, -561.499, 1, 0.06666667, 0, 1,
7.959596, 5.232323, -562.0419, 1, 0.06666667, 0, 1,
8, 5.232323, -562.5966, 0.9647059, 0, 0.03137255, 1,
4, 5.282828, -566.4839, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.282828, -565.8817, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.282828, -565.291, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.282828, -564.7122, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.282828, -564.1449, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.282828, -563.5894, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.282828, -563.0456, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.282828, -562.5135, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.282828, -561.9931, 1, 0.06666667, 0, 1,
4.363636, 5.282828, -561.4844, 1, 0.06666667, 0, 1,
4.40404, 5.282828, -560.9874, 1, 0.06666667, 0, 1,
4.444445, 5.282828, -560.5021, 1, 0.06666667, 0, 1,
4.484848, 5.282828, -560.0284, 1, 0.06666667, 0, 1,
4.525252, 5.282828, -559.5665, 1, 0.06666667, 0, 1,
4.565657, 5.282828, -559.1163, 1, 0.06666667, 0, 1,
4.606061, 5.282828, -558.6778, 1, 0.06666667, 0, 1,
4.646465, 5.282828, -558.251, 1, 0.06666667, 0, 1,
4.686869, 5.282828, -557.8359, 1, 0.06666667, 0, 1,
4.727273, 5.282828, -557.4324, 1, 0.06666667, 0, 1,
4.767677, 5.282828, -557.0407, 1, 0.1686275, 0, 1,
4.808081, 5.282828, -556.6607, 1, 0.1686275, 0, 1,
4.848485, 5.282828, -556.2924, 1, 0.1686275, 0, 1,
4.888889, 5.282828, -555.9357, 1, 0.1686275, 0, 1,
4.929293, 5.282828, -555.5908, 1, 0.1686275, 0, 1,
4.969697, 5.282828, -555.2576, 1, 0.1686275, 0, 1,
5.010101, 5.282828, -554.936, 1, 0.1686275, 0, 1,
5.050505, 5.282828, -554.6262, 1, 0.1686275, 0, 1,
5.090909, 5.282828, -554.3281, 1, 0.1686275, 0, 1,
5.131313, 5.282828, -554.0417, 1, 0.1686275, 0, 1,
5.171717, 5.282828, -553.7669, 1, 0.1686275, 0, 1,
5.212121, 5.282828, -553.5039, 1, 0.1686275, 0, 1,
5.252525, 5.282828, -553.2526, 1, 0.1686275, 0, 1,
5.292929, 5.282828, -553.0129, 1, 0.1686275, 0, 1,
5.333333, 5.282828, -552.785, 1, 0.1686275, 0, 1,
5.373737, 5.282828, -552.5688, 1, 0.1686275, 0, 1,
5.414141, 5.282828, -552.3642, 1, 0.1686275, 0, 1,
5.454545, 5.282828, -552.1714, 1, 0.1686275, 0, 1,
5.494949, 5.282828, -551.9902, 1, 0.1686275, 0, 1,
5.535354, 5.282828, -551.8208, 1, 0.1686275, 0, 1,
5.575758, 5.282828, -551.663, 1, 0.2745098, 0, 1,
5.616162, 5.282828, -551.517, 1, 0.2745098, 0, 1,
5.656566, 5.282828, -551.3827, 1, 0.2745098, 0, 1,
5.69697, 5.282828, -551.26, 1, 0.2745098, 0, 1,
5.737374, 5.282828, -551.149, 1, 0.2745098, 0, 1,
5.777778, 5.282828, -551.0498, 1, 0.2745098, 0, 1,
5.818182, 5.282828, -550.9623, 1, 0.2745098, 0, 1,
5.858586, 5.282828, -550.8864, 1, 0.2745098, 0, 1,
5.89899, 5.282828, -550.8223, 1, 0.2745098, 0, 1,
5.939394, 5.282828, -550.7698, 1, 0.2745098, 0, 1,
5.979798, 5.282828, -550.7291, 1, 0.2745098, 0, 1,
6.020202, 5.282828, -550.7, 1, 0.2745098, 0, 1,
6.060606, 5.282828, -550.6827, 1, 0.2745098, 0, 1,
6.10101, 5.282828, -550.677, 1, 0.2745098, 0, 1,
6.141414, 5.282828, -550.683, 1, 0.2745098, 0, 1,
6.181818, 5.282828, -550.7008, 1, 0.2745098, 0, 1,
6.222222, 5.282828, -550.7302, 1, 0.2745098, 0, 1,
6.262626, 5.282828, -550.7714, 1, 0.2745098, 0, 1,
6.30303, 5.282828, -550.8242, 1, 0.2745098, 0, 1,
6.343434, 5.282828, -550.8887, 1, 0.2745098, 0, 1,
6.383838, 5.282828, -550.965, 1, 0.2745098, 0, 1,
6.424242, 5.282828, -551.0529, 1, 0.2745098, 0, 1,
6.464646, 5.282828, -551.1525, 1, 0.2745098, 0, 1,
6.505051, 5.282828, -551.2639, 1, 0.2745098, 0, 1,
6.545455, 5.282828, -551.3869, 1, 0.2745098, 0, 1,
6.585859, 5.282828, -551.5217, 1, 0.2745098, 0, 1,
6.626263, 5.282828, -551.6681, 1, 0.2745098, 0, 1,
6.666667, 5.282828, -551.8262, 1, 0.1686275, 0, 1,
6.707071, 5.282828, -551.996, 1, 0.1686275, 0, 1,
6.747475, 5.282828, -552.1776, 1, 0.1686275, 0, 1,
6.787879, 5.282828, -552.3708, 1, 0.1686275, 0, 1,
6.828283, 5.282828, -552.5757, 1, 0.1686275, 0, 1,
6.868687, 5.282828, -552.7923, 1, 0.1686275, 0, 1,
6.909091, 5.282828, -553.0206, 1, 0.1686275, 0, 1,
6.949495, 5.282828, -553.2607, 1, 0.1686275, 0, 1,
6.989899, 5.282828, -553.5124, 1, 0.1686275, 0, 1,
7.030303, 5.282828, -553.7758, 1, 0.1686275, 0, 1,
7.070707, 5.282828, -554.0509, 1, 0.1686275, 0, 1,
7.111111, 5.282828, -554.3378, 1, 0.1686275, 0, 1,
7.151515, 5.282828, -554.6362, 1, 0.1686275, 0, 1,
7.191919, 5.282828, -554.9465, 1, 0.1686275, 0, 1,
7.232323, 5.282828, -555.2684, 1, 0.1686275, 0, 1,
7.272727, 5.282828, -555.602, 1, 0.1686275, 0, 1,
7.313131, 5.282828, -555.9473, 1, 0.1686275, 0, 1,
7.353535, 5.282828, -556.3043, 1, 0.1686275, 0, 1,
7.393939, 5.282828, -556.673, 1, 0.1686275, 0, 1,
7.434343, 5.282828, -557.0535, 1, 0.1686275, 0, 1,
7.474748, 5.282828, -557.4456, 1, 0.06666667, 0, 1,
7.515152, 5.282828, -557.8494, 1, 0.06666667, 0, 1,
7.555555, 5.282828, -558.2648, 1, 0.06666667, 0, 1,
7.59596, 5.282828, -558.6921, 1, 0.06666667, 0, 1,
7.636364, 5.282828, -559.131, 1, 0.06666667, 0, 1,
7.676768, 5.282828, -559.5815, 1, 0.06666667, 0, 1,
7.717172, 5.282828, -560.0439, 1, 0.06666667, 0, 1,
7.757576, 5.282828, -560.5179, 1, 0.06666667, 0, 1,
7.79798, 5.282828, -561.0035, 1, 0.06666667, 0, 1,
7.838384, 5.282828, -561.501, 1, 0.06666667, 0, 1,
7.878788, 5.282828, -562.0101, 1, 0.06666667, 0, 1,
7.919192, 5.282828, -562.5309, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.282828, -563.0634, 0.9647059, 0, 0.03137255, 1,
8, 5.282828, -563.6075, 0.9647059, 0, 0.03137255, 1,
4, 5.333333, -567.4481, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.333333, -566.8572, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.333333, -566.2777, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.333333, -565.7097, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.333333, -565.1532, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.333333, -564.6082, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.333333, -564.0746, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.333333, -563.5525, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.333333, -563.0419, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.333333, -562.5428, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.333333, -562.0552, 1, 0.06666667, 0, 1,
4.444445, 5.333333, -561.579, 1, 0.06666667, 0, 1,
4.484848, 5.333333, -561.1143, 1, 0.06666667, 0, 1,
4.525252, 5.333333, -560.6611, 1, 0.06666667, 0, 1,
4.565657, 5.333333, -560.2194, 1, 0.06666667, 0, 1,
4.606061, 5.333333, -559.7891, 1, 0.06666667, 0, 1,
4.646465, 5.333333, -559.3704, 1, 0.06666667, 0, 1,
4.686869, 5.333333, -558.963, 1, 0.06666667, 0, 1,
4.727273, 5.333333, -558.5672, 1, 0.06666667, 0, 1,
4.767677, 5.333333, -558.1829, 1, 0.06666667, 0, 1,
4.808081, 5.333333, -557.81, 1, 0.06666667, 0, 1,
4.848485, 5.333333, -557.4487, 1, 0.06666667, 0, 1,
4.888889, 5.333333, -557.0988, 1, 0.1686275, 0, 1,
4.929293, 5.333333, -556.7603, 1, 0.1686275, 0, 1,
4.969697, 5.333333, -556.4333, 1, 0.1686275, 0, 1,
5.010101, 5.333333, -556.1179, 1, 0.1686275, 0, 1,
5.050505, 5.333333, -555.8139, 1, 0.1686275, 0, 1,
5.090909, 5.333333, -555.5214, 1, 0.1686275, 0, 1,
5.131313, 5.333333, -555.2404, 1, 0.1686275, 0, 1,
5.171717, 5.333333, -554.9708, 1, 0.1686275, 0, 1,
5.212121, 5.333333, -554.7128, 1, 0.1686275, 0, 1,
5.252525, 5.333333, -554.4661, 1, 0.1686275, 0, 1,
5.292929, 5.333333, -554.231, 1, 0.1686275, 0, 1,
5.333333, 5.333333, -554.0074, 1, 0.1686275, 0, 1,
5.373737, 5.333333, -553.7952, 1, 0.1686275, 0, 1,
5.414141, 5.333333, -553.5945, 1, 0.1686275, 0, 1,
5.454545, 5.333333, -553.4053, 1, 0.1686275, 0, 1,
5.494949, 5.333333, -553.2276, 1, 0.1686275, 0, 1,
5.535354, 5.333333, -553.0613, 1, 0.1686275, 0, 1,
5.575758, 5.333333, -552.9066, 1, 0.1686275, 0, 1,
5.616162, 5.333333, -552.7633, 1, 0.1686275, 0, 1,
5.656566, 5.333333, -552.6315, 1, 0.1686275, 0, 1,
5.69697, 5.333333, -552.5112, 1, 0.1686275, 0, 1,
5.737374, 5.333333, -552.4023, 1, 0.1686275, 0, 1,
5.777778, 5.333333, -552.3049, 1, 0.1686275, 0, 1,
5.818182, 5.333333, -552.2191, 1, 0.1686275, 0, 1,
5.858586, 5.333333, -552.1446, 1, 0.1686275, 0, 1,
5.89899, 5.333333, -552.0817, 1, 0.1686275, 0, 1,
5.939394, 5.333333, -552.0302, 1, 0.1686275, 0, 1,
5.979798, 5.333333, -551.9902, 1, 0.1686275, 0, 1,
6.020202, 5.333333, -551.9617, 1, 0.1686275, 0, 1,
6.060606, 5.333333, -551.9447, 1, 0.1686275, 0, 1,
6.10101, 5.333333, -551.9391, 1, 0.1686275, 0, 1,
6.141414, 5.333333, -551.9451, 1, 0.1686275, 0, 1,
6.181818, 5.333333, -551.9625, 1, 0.1686275, 0, 1,
6.222222, 5.333333, -551.9913, 1, 0.1686275, 0, 1,
6.262626, 5.333333, -552.0317, 1, 0.1686275, 0, 1,
6.30303, 5.333333, -552.0836, 1, 0.1686275, 0, 1,
6.343434, 5.333333, -552.1469, 1, 0.1686275, 0, 1,
6.383838, 5.333333, -552.2217, 1, 0.1686275, 0, 1,
6.424242, 5.333333, -552.308, 1, 0.1686275, 0, 1,
6.464646, 5.333333, -552.4057, 1, 0.1686275, 0, 1,
6.505051, 5.333333, -552.515, 1, 0.1686275, 0, 1,
6.545455, 5.333333, -552.6357, 1, 0.1686275, 0, 1,
6.585859, 5.333333, -552.7678, 1, 0.1686275, 0, 1,
6.626263, 5.333333, -552.9115, 1, 0.1686275, 0, 1,
6.666667, 5.333333, -553.0667, 1, 0.1686275, 0, 1,
6.707071, 5.333333, -553.2333, 1, 0.1686275, 0, 1,
6.747475, 5.333333, -553.4114, 1, 0.1686275, 0, 1,
6.787879, 5.333333, -553.601, 1, 0.1686275, 0, 1,
6.828283, 5.333333, -553.8021, 1, 0.1686275, 0, 1,
6.868687, 5.333333, -554.0146, 1, 0.1686275, 0, 1,
6.909091, 5.333333, -554.2386, 1, 0.1686275, 0, 1,
6.949495, 5.333333, -554.4741, 1, 0.1686275, 0, 1,
6.989899, 5.333333, -554.7211, 1, 0.1686275, 0, 1,
7.030303, 5.333333, -554.9796, 1, 0.1686275, 0, 1,
7.070707, 5.333333, -555.2495, 1, 0.1686275, 0, 1,
7.111111, 5.333333, -555.5309, 1, 0.1686275, 0, 1,
7.151515, 5.333333, -555.8237, 1, 0.1686275, 0, 1,
7.191919, 5.333333, -556.1281, 1, 0.1686275, 0, 1,
7.232323, 5.333333, -556.444, 1, 0.1686275, 0, 1,
7.272727, 5.333333, -556.7713, 1, 0.1686275, 0, 1,
7.313131, 5.333333, -557.1101, 1, 0.1686275, 0, 1,
7.353535, 5.333333, -557.4604, 1, 0.06666667, 0, 1,
7.393939, 5.333333, -557.8221, 1, 0.06666667, 0, 1,
7.434343, 5.333333, -558.1954, 1, 0.06666667, 0, 1,
7.474748, 5.333333, -558.5801, 1, 0.06666667, 0, 1,
7.515152, 5.333333, -558.9763, 1, 0.06666667, 0, 1,
7.555555, 5.333333, -559.384, 1, 0.06666667, 0, 1,
7.59596, 5.333333, -559.8031, 1, 0.06666667, 0, 1,
7.636364, 5.333333, -560.2338, 1, 0.06666667, 0, 1,
7.676768, 5.333333, -560.6758, 1, 0.06666667, 0, 1,
7.717172, 5.333333, -561.1295, 1, 0.06666667, 0, 1,
7.757576, 5.333333, -561.5945, 1, 0.06666667, 0, 1,
7.79798, 5.333333, -562.071, 1, 0.06666667, 0, 1,
7.838384, 5.333333, -562.5591, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.333333, -563.0586, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.333333, -563.5695, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.333333, -564.092, 0.9647059, 0, 0.03137255, 1,
8, 5.333333, -564.6259, 0.9647059, 0, 0.03137255, 1,
4, 5.383838, -568.4207, 0.8588235, 0, 0.1372549, 1,
4.040404, 5.383838, -567.8408, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.383838, -567.2721, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.383838, -566.7147, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.383838, -566.1686, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.383838, -565.6337, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.383838, -565.1101, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.383838, -564.5978, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.383838, -564.0967, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.383838, -563.6069, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.383838, -563.1284, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.383838, -562.6611, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.383838, -562.2051, 1, 0.06666667, 0, 1,
4.525252, 5.383838, -561.7604, 1, 0.06666667, 0, 1,
4.565657, 5.383838, -561.3269, 1, 0.06666667, 0, 1,
4.606061, 5.383838, -560.9047, 1, 0.06666667, 0, 1,
4.646465, 5.383838, -560.4937, 1, 0.06666667, 0, 1,
4.686869, 5.383838, -560.094, 1, 0.06666667, 0, 1,
4.727273, 5.383838, -559.7056, 1, 0.06666667, 0, 1,
4.767677, 5.383838, -559.3284, 1, 0.06666667, 0, 1,
4.808081, 5.383838, -558.9625, 1, 0.06666667, 0, 1,
4.848485, 5.383838, -558.6079, 1, 0.06666667, 0, 1,
4.888889, 5.383838, -558.2645, 1, 0.06666667, 0, 1,
4.929293, 5.383838, -557.9324, 1, 0.06666667, 0, 1,
4.969697, 5.383838, -557.6116, 1, 0.06666667, 0, 1,
5.010101, 5.383838, -557.302, 1, 0.06666667, 0, 1,
5.050505, 5.383838, -557.0037, 1, 0.1686275, 0, 1,
5.090909, 5.383838, -556.7167, 1, 0.1686275, 0, 1,
5.131313, 5.383838, -556.4409, 1, 0.1686275, 0, 1,
5.171717, 5.383838, -556.1763, 1, 0.1686275, 0, 1,
5.212121, 5.383838, -555.9231, 1, 0.1686275, 0, 1,
5.252525, 5.383838, -555.6811, 1, 0.1686275, 0, 1,
5.292929, 5.383838, -555.4504, 1, 0.1686275, 0, 1,
5.333333, 5.383838, -555.2309, 1, 0.1686275, 0, 1,
5.373737, 5.383838, -555.0227, 1, 0.1686275, 0, 1,
5.414141, 5.383838, -554.8258, 1, 0.1686275, 0, 1,
5.454545, 5.383838, -554.6401, 1, 0.1686275, 0, 1,
5.494949, 5.383838, -554.4657, 1, 0.1686275, 0, 1,
5.535354, 5.383838, -554.3026, 1, 0.1686275, 0, 1,
5.575758, 5.383838, -554.1507, 1, 0.1686275, 0, 1,
5.616162, 5.383838, -554.0101, 1, 0.1686275, 0, 1,
5.656566, 5.383838, -553.8807, 1, 0.1686275, 0, 1,
5.69697, 5.383838, -553.7626, 1, 0.1686275, 0, 1,
5.737374, 5.383838, -553.6558, 1, 0.1686275, 0, 1,
5.777778, 5.383838, -553.5602, 1, 0.1686275, 0, 1,
5.818182, 5.383838, -553.476, 1, 0.1686275, 0, 1,
5.858586, 5.383838, -553.4029, 1, 0.1686275, 0, 1,
5.89899, 5.383838, -553.3411, 1, 0.1686275, 0, 1,
5.939394, 5.383838, -553.2906, 1, 0.1686275, 0, 1,
5.979798, 5.383838, -553.2514, 1, 0.1686275, 0, 1,
6.020202, 5.383838, -553.2234, 1, 0.1686275, 0, 1,
6.060606, 5.383838, -553.2067, 1, 0.1686275, 0, 1,
6.10101, 5.383838, -553.2013, 1, 0.1686275, 0, 1,
6.141414, 5.383838, -553.2071, 1, 0.1686275, 0, 1,
6.181818, 5.383838, -553.2242, 1, 0.1686275, 0, 1,
6.222222, 5.383838, -553.2525, 1, 0.1686275, 0, 1,
6.262626, 5.383838, -553.2921, 1, 0.1686275, 0, 1,
6.30303, 5.383838, -553.343, 1, 0.1686275, 0, 1,
6.343434, 5.383838, -553.4052, 1, 0.1686275, 0, 1,
6.383838, 5.383838, -553.4785, 1, 0.1686275, 0, 1,
6.424242, 5.383838, -553.5632, 1, 0.1686275, 0, 1,
6.464646, 5.383838, -553.6591, 1, 0.1686275, 0, 1,
6.505051, 5.383838, -553.7664, 1, 0.1686275, 0, 1,
6.545455, 5.383838, -553.8848, 1, 0.1686275, 0, 1,
6.585859, 5.383838, -554.0145, 1, 0.1686275, 0, 1,
6.626263, 5.383838, -554.1555, 1, 0.1686275, 0, 1,
6.666667, 5.383838, -554.3077, 1, 0.1686275, 0, 1,
6.707071, 5.383838, -554.4713, 1, 0.1686275, 0, 1,
6.747475, 5.383838, -554.6461, 1, 0.1686275, 0, 1,
6.787879, 5.383838, -554.8321, 1, 0.1686275, 0, 1,
6.828283, 5.383838, -555.0294, 1, 0.1686275, 0, 1,
6.868687, 5.383838, -555.238, 1, 0.1686275, 0, 1,
6.909091, 5.383838, -555.4578, 1, 0.1686275, 0, 1,
6.949495, 5.383838, -555.6889, 1, 0.1686275, 0, 1,
6.989899, 5.383838, -555.9313, 1, 0.1686275, 0, 1,
7.030303, 5.383838, -556.1849, 1, 0.1686275, 0, 1,
7.070707, 5.383838, -556.4498, 1, 0.1686275, 0, 1,
7.111111, 5.383838, -556.726, 1, 0.1686275, 0, 1,
7.151515, 5.383838, -557.0134, 1, 0.1686275, 0, 1,
7.191919, 5.383838, -557.3121, 1, 0.06666667, 0, 1,
7.232323, 5.383838, -557.622, 1, 0.06666667, 0, 1,
7.272727, 5.383838, -557.9432, 1, 0.06666667, 0, 1,
7.313131, 5.383838, -558.2757, 1, 0.06666667, 0, 1,
7.353535, 5.383838, -558.6194, 1, 0.06666667, 0, 1,
7.393939, 5.383838, -558.9744, 1, 0.06666667, 0, 1,
7.434343, 5.383838, -559.3407, 1, 0.06666667, 0, 1,
7.474748, 5.383838, -559.7182, 1, 0.06666667, 0, 1,
7.515152, 5.383838, -560.107, 1, 0.06666667, 0, 1,
7.555555, 5.383838, -560.5071, 1, 0.06666667, 0, 1,
7.59596, 5.383838, -560.9184, 1, 0.06666667, 0, 1,
7.636364, 5.383838, -561.341, 1, 0.06666667, 0, 1,
7.676768, 5.383838, -561.7748, 1, 0.06666667, 0, 1,
7.717172, 5.383838, -562.22, 1, 0.06666667, 0, 1,
7.757576, 5.383838, -562.6763, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.383838, -563.144, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.383838, -563.6229, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.383838, -564.1131, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.383838, -564.6145, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.383838, -565.1272, 0.9647059, 0, 0.03137255, 1,
8, 5.383838, -565.6512, 0.9647059, 0, 0.03137255, 1,
4, 5.434343, -569.4009, 0.8588235, 0, 0.1372549, 1,
4.040404, 5.434343, -568.8317, 0.8588235, 0, 0.1372549, 1,
4.080808, 5.434343, -568.2736, 0.8588235, 0, 0.1372549, 1,
4.121212, 5.434343, -567.7265, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.434343, -567.1905, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.434343, -566.6655, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.434343, -566.1516, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.434343, -565.6488, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.434343, -565.157, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.434343, -564.6763, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.434343, -564.2065, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.434343, -563.7479, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.434343, -563.3004, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.434343, -562.8638, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.434343, -562.4384, 1, 0.06666667, 0, 1,
4.606061, 5.434343, -562.024, 1, 0.06666667, 0, 1,
4.646465, 5.434343, -561.6206, 1, 0.06666667, 0, 1,
4.686869, 5.434343, -561.2283, 1, 0.06666667, 0, 1,
4.727273, 5.434343, -560.8471, 1, 0.06666667, 0, 1,
4.767677, 5.434343, -560.4769, 1, 0.06666667, 0, 1,
4.808081, 5.434343, -560.1178, 1, 0.06666667, 0, 1,
4.848485, 5.434343, -559.7697, 1, 0.06666667, 0, 1,
4.888889, 5.434343, -559.4327, 1, 0.06666667, 0, 1,
4.929293, 5.434343, -559.1068, 1, 0.06666667, 0, 1,
4.969697, 5.434343, -558.7918, 1, 0.06666667, 0, 1,
5.010101, 5.434343, -558.488, 1, 0.06666667, 0, 1,
5.050505, 5.434343, -558.1952, 1, 0.06666667, 0, 1,
5.090909, 5.434343, -557.9135, 1, 0.06666667, 0, 1,
5.131313, 5.434343, -557.6428, 1, 0.06666667, 0, 1,
5.171717, 5.434343, -557.3831, 1, 0.06666667, 0, 1,
5.212121, 5.434343, -557.1346, 1, 0.06666667, 0, 1,
5.252525, 5.434343, -556.897, 1, 0.1686275, 0, 1,
5.292929, 5.434343, -556.6706, 1, 0.1686275, 0, 1,
5.333333, 5.434343, -556.4552, 1, 0.1686275, 0, 1,
5.373737, 5.434343, -556.2509, 1, 0.1686275, 0, 1,
5.414141, 5.434343, -556.0576, 1, 0.1686275, 0, 1,
5.454545, 5.434343, -555.8753, 1, 0.1686275, 0, 1,
5.494949, 5.434343, -555.7042, 1, 0.1686275, 0, 1,
5.535354, 5.434343, -555.544, 1, 0.1686275, 0, 1,
5.575758, 5.434343, -555.395, 1, 0.1686275, 0, 1,
5.616162, 5.434343, -555.257, 1, 0.1686275, 0, 1,
5.656566, 5.434343, -555.1299, 1, 0.1686275, 0, 1,
5.69697, 5.434343, -555.014, 1, 0.1686275, 0, 1,
5.737374, 5.434343, -554.9092, 1, 0.1686275, 0, 1,
5.777778, 5.434343, -554.8154, 1, 0.1686275, 0, 1,
5.818182, 5.434343, -554.7327, 1, 0.1686275, 0, 1,
5.858586, 5.434343, -554.661, 1, 0.1686275, 0, 1,
5.89899, 5.434343, -554.6004, 1, 0.1686275, 0, 1,
5.939394, 5.434343, -554.5508, 1, 0.1686275, 0, 1,
5.979798, 5.434343, -554.5123, 1, 0.1686275, 0, 1,
6.020202, 5.434343, -554.4849, 1, 0.1686275, 0, 1,
6.060606, 5.434343, -554.4684, 1, 0.1686275, 0, 1,
6.10101, 5.434343, -554.4631, 1, 0.1686275, 0, 1,
6.141414, 5.434343, -554.4688, 1, 0.1686275, 0, 1,
6.181818, 5.434343, -554.4856, 1, 0.1686275, 0, 1,
6.222222, 5.434343, -554.5134, 1, 0.1686275, 0, 1,
6.262626, 5.434343, -554.5523, 1, 0.1686275, 0, 1,
6.30303, 5.434343, -554.6022, 1, 0.1686275, 0, 1,
6.343434, 5.434343, -554.6632, 1, 0.1686275, 0, 1,
6.383838, 5.434343, -554.7352, 1, 0.1686275, 0, 1,
6.424242, 5.434343, -554.8184, 1, 0.1686275, 0, 1,
6.464646, 5.434343, -554.9125, 1, 0.1686275, 0, 1,
6.505051, 5.434343, -555.0177, 1, 0.1686275, 0, 1,
6.545455, 5.434343, -555.134, 1, 0.1686275, 0, 1,
6.585859, 5.434343, -555.2613, 1, 0.1686275, 0, 1,
6.626263, 5.434343, -555.3997, 1, 0.1686275, 0, 1,
6.666667, 5.434343, -555.5491, 1, 0.1686275, 0, 1,
6.707071, 5.434343, -555.7096, 1, 0.1686275, 0, 1,
6.747475, 5.434343, -555.8812, 1, 0.1686275, 0, 1,
6.787879, 5.434343, -556.0637, 1, 0.1686275, 0, 1,
6.828283, 5.434343, -556.2574, 1, 0.1686275, 0, 1,
6.868687, 5.434343, -556.4621, 1, 0.1686275, 0, 1,
6.909091, 5.434343, -556.6779, 1, 0.1686275, 0, 1,
6.949495, 5.434343, -556.9047, 1, 0.1686275, 0, 1,
6.989899, 5.434343, -557.1426, 1, 0.06666667, 0, 1,
7.030303, 5.434343, -557.3915, 1, 0.06666667, 0, 1,
7.070707, 5.434343, -557.6515, 1, 0.06666667, 0, 1,
7.111111, 5.434343, -557.9225, 1, 0.06666667, 0, 1,
7.151515, 5.434343, -558.2047, 1, 0.06666667, 0, 1,
7.191919, 5.434343, -558.4978, 1, 0.06666667, 0, 1,
7.232323, 5.434343, -558.8021, 1, 0.06666667, 0, 1,
7.272727, 5.434343, -559.1173, 1, 0.06666667, 0, 1,
7.313131, 5.434343, -559.4436, 1, 0.06666667, 0, 1,
7.353535, 5.434343, -559.781, 1, 0.06666667, 0, 1,
7.393939, 5.434343, -560.1295, 1, 0.06666667, 0, 1,
7.434343, 5.434343, -560.489, 1, 0.06666667, 0, 1,
7.474748, 5.434343, -560.8595, 1, 0.06666667, 0, 1,
7.515152, 5.434343, -561.2411, 1, 0.06666667, 0, 1,
7.555555, 5.434343, -561.6337, 1, 0.06666667, 0, 1,
7.59596, 5.434343, -562.0375, 1, 0.06666667, 0, 1,
7.636364, 5.434343, -562.4522, 1, 0.06666667, 0, 1,
7.676768, 5.434343, -562.8781, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.434343, -563.3149, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.434343, -563.7629, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.434343, -564.2219, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.434343, -564.6919, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.434343, -565.173, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.434343, -565.6652, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.434343, -566.1684, 0.9647059, 0, 0.03137255, 1,
8, 5.434343, -566.6827, 0.9647059, 0, 0.03137255, 1,
4, 5.484848, -570.3883, 0.8588235, 0, 0.1372549, 1,
4.040404, 5.484848, -569.8295, 0.8588235, 0, 0.1372549, 1,
4.080808, 5.484848, -569.2816, 0.8588235, 0, 0.1372549, 1,
4.121212, 5.484848, -568.7446, 0.8588235, 0, 0.1372549, 1,
4.161616, 5.484848, -568.2184, 0.8588235, 0, 0.1372549, 1,
4.20202, 5.484848, -567.7031, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.484848, -567.1985, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.484848, -566.705, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.484848, -566.2222, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.484848, -565.7502, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.484848, -565.2892, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.484848, -564.8389, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.484848, -564.3995, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.484848, -563.9711, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.484848, -563.5534, 0.9647059, 0, 0.03137255, 1,
4.606061, 5.484848, -563.1466, 0.9647059, 0, 0.03137255, 1,
4.646465, 5.484848, -562.7506, 0.9647059, 0, 0.03137255, 1,
4.686869, 5.484848, -562.3655, 1, 0.06666667, 0, 1,
4.727273, 5.484848, -561.9913, 1, 0.06666667, 0, 1,
4.767677, 5.484848, -561.6279, 1, 0.06666667, 0, 1,
4.808081, 5.484848, -561.2753, 1, 0.06666667, 0, 1,
4.848485, 5.484848, -560.9337, 1, 0.06666667, 0, 1,
4.888889, 5.484848, -560.6028, 1, 0.06666667, 0, 1,
4.929293, 5.484848, -560.2828, 1, 0.06666667, 0, 1,
4.969697, 5.484848, -559.9737, 1, 0.06666667, 0, 1,
5.010101, 5.484848, -559.6754, 1, 0.06666667, 0, 1,
5.050505, 5.484848, -559.388, 1, 0.06666667, 0, 1,
5.090909, 5.484848, -559.1114, 1, 0.06666667, 0, 1,
5.131313, 5.484848, -558.8457, 1, 0.06666667, 0, 1,
5.171717, 5.484848, -558.5908, 1, 0.06666667, 0, 1,
5.212121, 5.484848, -558.3468, 1, 0.06666667, 0, 1,
5.252525, 5.484848, -558.1136, 1, 0.06666667, 0, 1,
5.292929, 5.484848, -557.8914, 1, 0.06666667, 0, 1,
5.333333, 5.484848, -557.6799, 1, 0.06666667, 0, 1,
5.373737, 5.484848, -557.4793, 1, 0.06666667, 0, 1,
5.414141, 5.484848, -557.2896, 1, 0.06666667, 0, 1,
5.454545, 5.484848, -557.1107, 1, 0.1686275, 0, 1,
5.494949, 5.484848, -556.9426, 1, 0.1686275, 0, 1,
5.535354, 5.484848, -556.7854, 1, 0.1686275, 0, 1,
5.575758, 5.484848, -556.639, 1, 0.1686275, 0, 1,
5.616162, 5.484848, -556.5036, 1, 0.1686275, 0, 1,
5.656566, 5.484848, -556.379, 1, 0.1686275, 0, 1,
5.69697, 5.484848, -556.2652, 1, 0.1686275, 0, 1,
5.737374, 5.484848, -556.1622, 1, 0.1686275, 0, 1,
5.777778, 5.484848, -556.0702, 1, 0.1686275, 0, 1,
5.818182, 5.484848, -555.989, 1, 0.1686275, 0, 1,
5.858586, 5.484848, -555.9186, 1, 0.1686275, 0, 1,
5.89899, 5.484848, -555.8591, 1, 0.1686275, 0, 1,
5.939394, 5.484848, -555.8104, 1, 0.1686275, 0, 1,
5.979798, 5.484848, -555.7726, 1, 0.1686275, 0, 1,
6.020202, 5.484848, -555.7457, 1, 0.1686275, 0, 1,
6.060606, 5.484848, -555.7296, 1, 0.1686275, 0, 1,
6.10101, 5.484848, -555.7243, 1, 0.1686275, 0, 1,
6.141414, 5.484848, -555.7299, 1, 0.1686275, 0, 1,
6.181818, 5.484848, -555.7464, 1, 0.1686275, 0, 1,
6.222222, 5.484848, -555.7737, 1, 0.1686275, 0, 1,
6.262626, 5.484848, -555.8118, 1, 0.1686275, 0, 1,
6.30303, 5.484848, -555.8608, 1, 0.1686275, 0, 1,
6.343434, 5.484848, -555.9207, 1, 0.1686275, 0, 1,
6.383838, 5.484848, -555.9915, 1, 0.1686275, 0, 1,
6.424242, 5.484848, -556.0731, 1, 0.1686275, 0, 1,
6.464646, 5.484848, -556.1655, 1, 0.1686275, 0, 1,
6.505051, 5.484848, -556.2687, 1, 0.1686275, 0, 1,
6.545455, 5.484848, -556.3829, 1, 0.1686275, 0, 1,
6.585859, 5.484848, -556.5079, 1, 0.1686275, 0, 1,
6.626263, 5.484848, -556.6437, 1, 0.1686275, 0, 1,
6.666667, 5.484848, -556.7904, 1, 0.1686275, 0, 1,
6.707071, 5.484848, -556.9479, 1, 0.1686275, 0, 1,
6.747475, 5.484848, -557.1163, 1, 0.1686275, 0, 1,
6.787879, 5.484848, -557.2956, 1, 0.06666667, 0, 1,
6.828283, 5.484848, -557.4857, 1, 0.06666667, 0, 1,
6.868687, 5.484848, -557.6867, 1, 0.06666667, 0, 1,
6.909091, 5.484848, -557.8985, 1, 0.06666667, 0, 1,
6.949495, 5.484848, -558.1212, 1, 0.06666667, 0, 1,
6.989899, 5.484848, -558.3547, 1, 0.06666667, 0, 1,
7.030303, 5.484848, -558.5991, 1, 0.06666667, 0, 1,
7.070707, 5.484848, -558.8542, 1, 0.06666667, 0, 1,
7.111111, 5.484848, -559.1204, 1, 0.06666667, 0, 1,
7.151515, 5.484848, -559.3973, 1, 0.06666667, 0, 1,
7.191919, 5.484848, -559.6851, 1, 0.06666667, 0, 1,
7.232323, 5.484848, -559.9837, 1, 0.06666667, 0, 1,
7.272727, 5.484848, -560.2932, 1, 0.06666667, 0, 1,
7.313131, 5.484848, -560.6135, 1, 0.06666667, 0, 1,
7.353535, 5.484848, -560.9447, 1, 0.06666667, 0, 1,
7.393939, 5.484848, -561.2868, 1, 0.06666667, 0, 1,
7.434343, 5.484848, -561.6397, 1, 0.06666667, 0, 1,
7.474748, 5.484848, -562.0034, 1, 0.06666667, 0, 1,
7.515152, 5.484848, -562.3781, 1, 0.06666667, 0, 1,
7.555555, 5.484848, -562.7635, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.484848, -563.1598, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.484848, -563.567, 0.9647059, 0, 0.03137255, 1,
7.676768, 5.484848, -563.985, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.484848, -564.4139, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.484848, -564.8536, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.484848, -565.3042, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.484848, -565.7656, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.484848, -566.2379, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.484848, -566.7211, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.484848, -567.215, 0.9647059, 0, 0.03137255, 1,
8, 5.484848, -567.7198, 0.9647059, 0, 0.03137255, 1,
4, 5.535354, -571.3822, 0.8588235, 0, 0.1372549, 1,
4.040404, 5.535354, -570.8336, 0.8588235, 0, 0.1372549, 1,
4.080808, 5.535354, -570.2957, 0.8588235, 0, 0.1372549, 1,
4.121212, 5.535354, -569.7684, 0.8588235, 0, 0.1372549, 1,
4.161616, 5.535354, -569.2517, 0.8588235, 0, 0.1372549, 1,
4.20202, 5.535354, -568.7457, 0.8588235, 0, 0.1372549, 1,
4.242424, 5.535354, -568.2504, 0.8588235, 0, 0.1372549, 1,
4.282828, 5.535354, -567.7657, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.535354, -567.2917, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.535354, -566.8284, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.535354, -566.3757, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.535354, -565.9337, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.535354, -565.5023, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.535354, -565.0815, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.535354, -564.6714, 0.9647059, 0, 0.03137255, 1,
4.606061, 5.535354, -564.272, 0.9647059, 0, 0.03137255, 1,
4.646465, 5.535354, -563.8833, 0.9647059, 0, 0.03137255, 1,
4.686869, 5.535354, -563.5052, 0.9647059, 0, 0.03137255, 1,
4.727273, 5.535354, -563.1378, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.535354, -562.7809, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.535354, -562.4348, 1, 0.06666667, 0, 1,
4.848485, 5.535354, -562.0993, 1, 0.06666667, 0, 1,
4.888889, 5.535354, -561.7745, 1, 0.06666667, 0, 1,
4.929293, 5.535354, -561.4603, 1, 0.06666667, 0, 1,
4.969697, 5.535354, -561.1568, 1, 0.06666667, 0, 1,
5.010101, 5.535354, -560.864, 1, 0.06666667, 0, 1,
5.050505, 5.535354, -560.5817, 1, 0.06666667, 0, 1,
5.090909, 5.535354, -560.3102, 1, 0.06666667, 0, 1,
5.131313, 5.535354, -560.0493, 1, 0.06666667, 0, 1,
5.171717, 5.535354, -559.7991, 1, 0.06666667, 0, 1,
5.212121, 5.535354, -559.5594, 1, 0.06666667, 0, 1,
5.252525, 5.535354, -559.3306, 1, 0.06666667, 0, 1,
5.292929, 5.535354, -559.1122, 1, 0.06666667, 0, 1,
5.333333, 5.535354, -558.9047, 1, 0.06666667, 0, 1,
5.373737, 5.535354, -558.7077, 1, 0.06666667, 0, 1,
5.414141, 5.535354, -558.5214, 1, 0.06666667, 0, 1,
5.454545, 5.535354, -558.3458, 1, 0.06666667, 0, 1,
5.494949, 5.535354, -558.1808, 1, 0.06666667, 0, 1,
5.535354, 5.535354, -558.0264, 1, 0.06666667, 0, 1,
5.575758, 5.535354, -557.8828, 1, 0.06666667, 0, 1,
5.616162, 5.535354, -557.7497, 1, 0.06666667, 0, 1,
5.656566, 5.535354, -557.6274, 1, 0.06666667, 0, 1,
5.69697, 5.535354, -557.5156, 1, 0.06666667, 0, 1,
5.737374, 5.535354, -557.4146, 1, 0.06666667, 0, 1,
5.777778, 5.535354, -557.3242, 1, 0.06666667, 0, 1,
5.818182, 5.535354, -557.2444, 1, 0.06666667, 0, 1,
5.858586, 5.535354, -557.1754, 1, 0.06666667, 0, 1,
5.89899, 5.535354, -557.1169, 1, 0.1686275, 0, 1,
5.939394, 5.535354, -557.0692, 1, 0.1686275, 0, 1,
5.979798, 5.535354, -557.032, 1, 0.1686275, 0, 1,
6.020202, 5.535354, -557.0056, 1, 0.1686275, 0, 1,
6.060606, 5.535354, -556.9897, 1, 0.1686275, 0, 1,
6.10101, 5.535354, -556.9846, 1, 0.1686275, 0, 1,
6.141414, 5.535354, -556.9901, 1, 0.1686275, 0, 1,
6.181818, 5.535354, -557.0063, 1, 0.1686275, 0, 1,
6.222222, 5.535354, -557.0331, 1, 0.1686275, 0, 1,
6.262626, 5.535354, -557.0706, 1, 0.1686275, 0, 1,
6.30303, 5.535354, -557.1187, 1, 0.1686275, 0, 1,
6.343434, 5.535354, -557.1775, 1, 0.06666667, 0, 1,
6.383838, 5.535354, -557.2469, 1, 0.06666667, 0, 1,
6.424242, 5.535354, -557.327, 1, 0.06666667, 0, 1,
6.464646, 5.535354, -557.4177, 1, 0.06666667, 0, 1,
6.505051, 5.535354, -557.5192, 1, 0.06666667, 0, 1,
6.545455, 5.535354, -557.6312, 1, 0.06666667, 0, 1,
6.585859, 5.535354, -557.754, 1, 0.06666667, 0, 1,
6.626263, 5.535354, -557.8873, 1, 0.06666667, 0, 1,
6.666667, 5.535354, -558.0314, 1, 0.06666667, 0, 1,
6.707071, 5.535354, -558.186, 1, 0.06666667, 0, 1,
6.747475, 5.535354, -558.3514, 1, 0.06666667, 0, 1,
6.787879, 5.535354, -558.5273, 1, 0.06666667, 0, 1,
6.828283, 5.535354, -558.7141, 1, 0.06666667, 0, 1,
6.868687, 5.535354, -558.9113, 1, 0.06666667, 0, 1,
6.909091, 5.535354, -559.1193, 1, 0.06666667, 0, 1,
6.949495, 5.535354, -559.3379, 1, 0.06666667, 0, 1,
6.989899, 5.535354, -559.5672, 1, 0.06666667, 0, 1,
7.030303, 5.535354, -559.8071, 1, 0.06666667, 0, 1,
7.070707, 5.535354, -560.0577, 1, 0.06666667, 0, 1,
7.111111, 5.535354, -560.319, 1, 0.06666667, 0, 1,
7.151515, 5.535354, -560.5909, 1, 0.06666667, 0, 1,
7.191919, 5.535354, -560.8734, 1, 0.06666667, 0, 1,
7.232323, 5.535354, -561.1666, 1, 0.06666667, 0, 1,
7.272727, 5.535354, -561.4705, 1, 0.06666667, 0, 1,
7.313131, 5.535354, -561.785, 1, 0.06666667, 0, 1,
7.353535, 5.535354, -562.1102, 1, 0.06666667, 0, 1,
7.393939, 5.535354, -562.446, 1, 0.06666667, 0, 1,
7.434343, 5.535354, -562.7925, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.535354, -563.1497, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.535354, -563.5175, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.535354, -563.8959, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.535354, -564.285, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.535354, -564.6848, 0.9647059, 0, 0.03137255, 1,
7.676768, 5.535354, -565.0953, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.535354, -565.5164, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.535354, -565.9481, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.535354, -566.3904, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.535354, -566.8435, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.535354, -567.3072, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.535354, -567.7816, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.535354, -568.2666, 0.8588235, 0, 0.1372549, 1,
8, 5.535354, -568.7623, 0.8588235, 0, 0.1372549, 1,
4, 5.585859, -572.3821, 0.8588235, 0, 0.1372549, 1,
4.040404, 5.585859, -571.8434, 0.8588235, 0, 0.1372549, 1,
4.080808, 5.585859, -571.3152, 0.8588235, 0, 0.1372549, 1,
4.121212, 5.585859, -570.7974, 0.8588235, 0, 0.1372549, 1,
4.161616, 5.585859, -570.29, 0.8588235, 0, 0.1372549, 1,
4.20202, 5.585859, -569.7932, 0.8588235, 0, 0.1372549, 1,
4.242424, 5.585859, -569.3068, 0.8588235, 0, 0.1372549, 1,
4.282828, 5.585859, -568.8308, 0.8588235, 0, 0.1372549, 1,
4.323232, 5.585859, -568.3654, 0.8588235, 0, 0.1372549, 1,
4.363636, 5.585859, -567.9103, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.585859, -567.4658, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.585859, -567.0317, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.585859, -566.608, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.585859, -566.1949, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.585859, -565.7922, 0.9647059, 0, 0.03137255, 1,
4.606061, 5.585859, -565.4, 0.9647059, 0, 0.03137255, 1,
4.646465, 5.585859, -565.0182, 0.9647059, 0, 0.03137255, 1,
4.686869, 5.585859, -564.6469, 0.9647059, 0, 0.03137255, 1,
4.727273, 5.585859, -564.2861, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.585859, -563.9357, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.585859, -563.5958, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.585859, -563.2664, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.585859, -562.9474, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.585859, -562.6389, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.585859, -562.3408, 1, 0.06666667, 0, 1,
5.010101, 5.585859, -562.0532, 1, 0.06666667, 0, 1,
5.050505, 5.585859, -561.7761, 1, 0.06666667, 0, 1,
5.090909, 5.585859, -561.5094, 1, 0.06666667, 0, 1,
5.131313, 5.585859, -561.2532, 1, 0.06666667, 0, 1,
5.171717, 5.585859, -561.0075, 1, 0.06666667, 0, 1,
5.212121, 5.585859, -560.7722, 1, 0.06666667, 0, 1,
5.252525, 5.585859, -560.5474, 1, 0.06666667, 0, 1,
5.292929, 5.585859, -560.3331, 1, 0.06666667, 0, 1,
5.333333, 5.585859, -560.1292, 1, 0.06666667, 0, 1,
5.373737, 5.585859, -559.9358, 1, 0.06666667, 0, 1,
5.414141, 5.585859, -559.7529, 1, 0.06666667, 0, 1,
5.454545, 5.585859, -559.5804, 1, 0.06666667, 0, 1,
5.494949, 5.585859, -559.4183, 1, 0.06666667, 0, 1,
5.535354, 5.585859, -559.2668, 1, 0.06666667, 0, 1,
5.575758, 5.585859, -559.1257, 1, 0.06666667, 0, 1,
5.616162, 5.585859, -558.9951, 1, 0.06666667, 0, 1,
5.656566, 5.585859, -558.8749, 1, 0.06666667, 0, 1,
5.69697, 5.585859, -558.7652, 1, 0.06666667, 0, 1,
5.737374, 5.585859, -558.666, 1, 0.06666667, 0, 1,
5.777778, 5.585859, -558.5772, 1, 0.06666667, 0, 1,
5.818182, 5.585859, -558.4989, 1, 0.06666667, 0, 1,
5.858586, 5.585859, -558.431, 1, 0.06666667, 0, 1,
5.89899, 5.585859, -558.3737, 1, 0.06666667, 0, 1,
5.939394, 5.585859, -558.3267, 1, 0.06666667, 0, 1,
5.979798, 5.585859, -558.2903, 1, 0.06666667, 0, 1,
6.020202, 5.585859, -558.2643, 1, 0.06666667, 0, 1,
6.060606, 5.585859, -558.2488, 1, 0.06666667, 0, 1,
6.10101, 5.585859, -558.2437, 1, 0.06666667, 0, 1,
6.141414, 5.585859, -558.2491, 1, 0.06666667, 0, 1,
6.181818, 5.585859, -558.265, 1, 0.06666667, 0, 1,
6.222222, 5.585859, -558.2913, 1, 0.06666667, 0, 1,
6.262626, 5.585859, -558.3281, 1, 0.06666667, 0, 1,
6.30303, 5.585859, -558.3754, 1, 0.06666667, 0, 1,
6.343434, 5.585859, -558.4331, 1, 0.06666667, 0, 1,
6.383838, 5.585859, -558.5013, 1, 0.06666667, 0, 1,
6.424242, 5.585859, -558.58, 1, 0.06666667, 0, 1,
6.464646, 5.585859, -558.6691, 1, 0.06666667, 0, 1,
6.505051, 5.585859, -558.7687, 1, 0.06666667, 0, 1,
6.545455, 5.585859, -558.8787, 1, 0.06666667, 0, 1,
6.585859, 5.585859, -558.9992, 1, 0.06666667, 0, 1,
6.626263, 5.585859, -559.1302, 1, 0.06666667, 0, 1,
6.666667, 5.585859, -559.2716, 1, 0.06666667, 0, 1,
6.707071, 5.585859, -559.4235, 1, 0.06666667, 0, 1,
6.747475, 5.585859, -559.5859, 1, 0.06666667, 0, 1,
6.787879, 5.585859, -559.7587, 1, 0.06666667, 0, 1,
6.828283, 5.585859, -559.942, 1, 0.06666667, 0, 1,
6.868687, 5.585859, -560.1357, 1, 0.06666667, 0, 1,
6.909091, 5.585859, -560.34, 1, 0.06666667, 0, 1,
6.949495, 5.585859, -560.5547, 1, 0.06666667, 0, 1,
6.989899, 5.585859, -560.7798, 1, 0.06666667, 0, 1,
7.030303, 5.585859, -561.0154, 1, 0.06666667, 0, 1,
7.070707, 5.585859, -561.2615, 1, 0.06666667, 0, 1,
7.111111, 5.585859, -561.5181, 1, 0.06666667, 0, 1,
7.151515, 5.585859, -561.785, 1, 0.06666667, 0, 1,
7.191919, 5.585859, -562.0625, 1, 0.06666667, 0, 1,
7.232323, 5.585859, -562.3505, 1, 0.06666667, 0, 1,
7.272727, 5.585859, -562.6489, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.585859, -562.9577, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.585859, -563.277, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.585859, -563.6068, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.585859, -563.9471, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.585859, -564.2978, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.585859, -564.659, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.585859, -565.0306, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.585859, -565.4127, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.585859, -565.8053, 0.9647059, 0, 0.03137255, 1,
7.676768, 5.585859, -566.2084, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.585859, -566.6219, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.585859, -567.0458, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.585859, -567.4803, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.585859, -567.9252, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.585859, -568.3805, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.585859, -568.8463, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.585859, -569.3226, 0.8588235, 0, 0.1372549, 1,
8, 5.585859, -569.8094, 0.8588235, 0, 0.1372549, 1,
4, 5.636364, -573.3876, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.636364, -572.8585, 0.8588235, 0, 0.1372549, 1,
4.080808, 5.636364, -572.3397, 0.8588235, 0, 0.1372549, 1,
4.121212, 5.636364, -571.8311, 0.8588235, 0, 0.1372549, 1,
4.161616, 5.636364, -571.3328, 0.8588235, 0, 0.1372549, 1,
4.20202, 5.636364, -570.8448, 0.8588235, 0, 0.1372549, 1,
4.242424, 5.636364, -570.3671, 0.8588235, 0, 0.1372549, 1,
4.282828, 5.636364, -569.8996, 0.8588235, 0, 0.1372549, 1,
4.323232, 5.636364, -569.4424, 0.8588235, 0, 0.1372549, 1,
4.363636, 5.636364, -568.9955, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.636364, -568.5589, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.636364, -568.1326, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.636364, -567.7165, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.636364, -567.3107, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.636364, -566.9152, 0.9647059, 0, 0.03137255, 1,
4.606061, 5.636364, -566.53, 0.9647059, 0, 0.03137255, 1,
4.646465, 5.636364, -566.155, 0.9647059, 0, 0.03137255, 1,
4.686869, 5.636364, -565.7903, 0.9647059, 0, 0.03137255, 1,
4.727273, 5.636364, -565.436, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.636364, -565.0919, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.636364, -564.758, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.636364, -564.4344, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.636364, -564.1212, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.636364, -563.8181, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.636364, -563.5254, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.636364, -563.2429, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.636364, -562.9708, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.636364, -562.7089, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.636364, -562.4572, 1, 0.06666667, 0, 1,
5.171717, 5.636364, -562.2159, 1, 0.06666667, 0, 1,
5.212121, 5.636364, -561.9848, 1, 0.06666667, 0, 1,
5.252525, 5.636364, -561.764, 1, 0.06666667, 0, 1,
5.292929, 5.636364, -561.5535, 1, 0.06666667, 0, 1,
5.333333, 5.636364, -561.3533, 1, 0.06666667, 0, 1,
5.373737, 5.636364, -561.1633, 1, 0.06666667, 0, 1,
5.414141, 5.636364, -560.9836, 1, 0.06666667, 0, 1,
5.454545, 5.636364, -560.8142, 1, 0.06666667, 0, 1,
5.494949, 5.636364, -560.6551, 1, 0.06666667, 0, 1,
5.535354, 5.636364, -560.5062, 1, 0.06666667, 0, 1,
5.575758, 5.636364, -560.3676, 1, 0.06666667, 0, 1,
5.616162, 5.636364, -560.2393, 1, 0.06666667, 0, 1,
5.656566, 5.636364, -560.1213, 1, 0.06666667, 0, 1,
5.69697, 5.636364, -560.0135, 1, 0.06666667, 0, 1,
5.737374, 5.636364, -559.9161, 1, 0.06666667, 0, 1,
5.777778, 5.636364, -559.8289, 1, 0.06666667, 0, 1,
5.818182, 5.636364, -559.752, 1, 0.06666667, 0, 1,
5.858586, 5.636364, -559.6854, 1, 0.06666667, 0, 1,
5.89899, 5.636364, -559.629, 1, 0.06666667, 0, 1,
5.939394, 5.636364, -559.5829, 1, 0.06666667, 0, 1,
5.979798, 5.636364, -559.5471, 1, 0.06666667, 0, 1,
6.020202, 5.636364, -559.5216, 1, 0.06666667, 0, 1,
6.060606, 5.636364, -559.5063, 1, 0.06666667, 0, 1,
6.10101, 5.636364, -559.5014, 1, 0.06666667, 0, 1,
6.141414, 5.636364, -559.5067, 1, 0.06666667, 0, 1,
6.181818, 5.636364, -559.5223, 1, 0.06666667, 0, 1,
6.222222, 5.636364, -559.5482, 1, 0.06666667, 0, 1,
6.262626, 5.636364, -559.5843, 1, 0.06666667, 0, 1,
6.30303, 5.636364, -559.6307, 1, 0.06666667, 0, 1,
6.343434, 5.636364, -559.6874, 1, 0.06666667, 0, 1,
6.383838, 5.636364, -559.7544, 1, 0.06666667, 0, 1,
6.424242, 5.636364, -559.8316, 1, 0.06666667, 0, 1,
6.464646, 5.636364, -559.9192, 1, 0.06666667, 0, 1,
6.505051, 5.636364, -560.017, 1, 0.06666667, 0, 1,
6.545455, 5.636364, -560.1251, 1, 0.06666667, 0, 1,
6.585859, 5.636364, -560.2434, 1, 0.06666667, 0, 1,
6.626263, 5.636364, -560.3721, 1, 0.06666667, 0, 1,
6.666667, 5.636364, -560.5109, 1, 0.06666667, 0, 1,
6.707071, 5.636364, -560.6602, 1, 0.06666667, 0, 1,
6.747475, 5.636364, -560.8196, 1, 0.06666667, 0, 1,
6.787879, 5.636364, -560.9894, 1, 0.06666667, 0, 1,
6.828283, 5.636364, -561.1694, 1, 0.06666667, 0, 1,
6.868687, 5.636364, -561.3597, 1, 0.06666667, 0, 1,
6.909091, 5.636364, -561.5602, 1, 0.06666667, 0, 1,
6.949495, 5.636364, -561.7711, 1, 0.06666667, 0, 1,
6.989899, 5.636364, -561.9922, 1, 0.06666667, 0, 1,
7.030303, 5.636364, -562.2236, 1, 0.06666667, 0, 1,
7.070707, 5.636364, -562.4653, 1, 0.06666667, 0, 1,
7.111111, 5.636364, -562.7173, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.636364, -562.9796, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.636364, -563.2521, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.636364, -563.5349, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.636364, -563.8279, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.636364, -564.1313, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.636364, -564.4449, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.636364, -564.7689, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.636364, -565.103, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.636364, -565.4474, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.636364, -565.8022, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.636364, -566.1672, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.636364, -566.5425, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.636364, -566.9281, 0.9647059, 0, 0.03137255, 1,
7.676768, 5.636364, -567.3239, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.636364, -567.73, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.636364, -568.1465, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.636364, -568.5731, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.636364, -569.0101, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.636364, -569.4573, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.636364, -569.9149, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.636364, -570.3826, 0.8588235, 0, 0.1372549, 1,
8, 5.636364, -570.8607, 0.8588235, 0, 0.1372549, 1,
4, 5.686869, -574.3981, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.686869, -573.8783, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.686869, -573.3687, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.686869, -572.8691, 0.8588235, 0, 0.1372549, 1,
4.161616, 5.686869, -572.3796, 0.8588235, 0, 0.1372549, 1,
4.20202, 5.686869, -571.9002, 0.8588235, 0, 0.1372549, 1,
4.242424, 5.686869, -571.4309, 0.8588235, 0, 0.1372549, 1,
4.282828, 5.686869, -570.9717, 0.8588235, 0, 0.1372549, 1,
4.323232, 5.686869, -570.5226, 0.8588235, 0, 0.1372549, 1,
4.363636, 5.686869, -570.0837, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.686869, -569.6548, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.686869, -569.236, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.686869, -568.8273, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.686869, -568.4286, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.686869, -568.0401, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.686869, -567.6617, 0.9647059, 0, 0.03137255, 1,
4.646465, 5.686869, -567.2934, 0.9647059, 0, 0.03137255, 1,
4.686869, 5.686869, -566.9352, 0.9647059, 0, 0.03137255, 1,
4.727273, 5.686869, -566.587, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.686869, -566.249, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.686869, -565.921, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.686869, -565.6032, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.686869, -565.2955, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.686869, -564.9978, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.686869, -564.7103, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.686869, -564.4328, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.686869, -564.1654, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.686869, -563.9081, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.686869, -563.6609, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.686869, -563.4239, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.686869, -563.1969, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.686869, -562.98, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.686869, -562.7732, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.686869, -562.5765, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.686869, -562.3899, 1, 0.06666667, 0, 1,
5.414141, 5.686869, -562.2134, 1, 0.06666667, 0, 1,
5.454545, 5.686869, -562.047, 1, 0.06666667, 0, 1,
5.494949, 5.686869, -561.8907, 1, 0.06666667, 0, 1,
5.535354, 5.686869, -561.7444, 1, 0.06666667, 0, 1,
5.575758, 5.686869, -561.6083, 1, 0.06666667, 0, 1,
5.616162, 5.686869, -561.4823, 1, 0.06666667, 0, 1,
5.656566, 5.686869, -561.3664, 1, 0.06666667, 0, 1,
5.69697, 5.686869, -561.2606, 1, 0.06666667, 0, 1,
5.737374, 5.686869, -561.1648, 1, 0.06666667, 0, 1,
5.777778, 5.686869, -561.0792, 1, 0.06666667, 0, 1,
5.818182, 5.686869, -561.0036, 1, 0.06666667, 0, 1,
5.858586, 5.686869, -560.9381, 1, 0.06666667, 0, 1,
5.89899, 5.686869, -560.8828, 1, 0.06666667, 0, 1,
5.939394, 5.686869, -560.8375, 1, 0.06666667, 0, 1,
5.979798, 5.686869, -560.8024, 1, 0.06666667, 0, 1,
6.020202, 5.686869, -560.7773, 1, 0.06666667, 0, 1,
6.060606, 5.686869, -560.7623, 1, 0.06666667, 0, 1,
6.10101, 5.686869, -560.7574, 1, 0.06666667, 0, 1,
6.141414, 5.686869, -560.7626, 1, 0.06666667, 0, 1,
6.181818, 5.686869, -560.778, 1, 0.06666667, 0, 1,
6.222222, 5.686869, -560.8033, 1, 0.06666667, 0, 1,
6.262626, 5.686869, -560.8389, 1, 0.06666667, 0, 1,
6.30303, 5.686869, -560.8845, 1, 0.06666667, 0, 1,
6.343434, 5.686869, -560.9401, 1, 0.06666667, 0, 1,
6.383838, 5.686869, -561.0059, 1, 0.06666667, 0, 1,
6.424242, 5.686869, -561.0818, 1, 0.06666667, 0, 1,
6.464646, 5.686869, -561.1678, 1, 0.06666667, 0, 1,
6.505051, 5.686869, -561.2639, 1, 0.06666667, 0, 1,
6.545455, 5.686869, -561.3701, 1, 0.06666667, 0, 1,
6.585859, 5.686869, -561.4863, 1, 0.06666667, 0, 1,
6.626263, 5.686869, -561.6127, 1, 0.06666667, 0, 1,
6.666667, 5.686869, -561.7491, 1, 0.06666667, 0, 1,
6.707071, 5.686869, -561.8957, 1, 0.06666667, 0, 1,
6.747475, 5.686869, -562.0523, 1, 0.06666667, 0, 1,
6.787879, 5.686869, -562.2191, 1, 0.06666667, 0, 1,
6.828283, 5.686869, -562.3959, 1, 0.06666667, 0, 1,
6.868687, 5.686869, -562.5828, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.686869, -562.7798, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.686869, -562.987, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.686869, -563.2042, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.686869, -563.4315, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.686869, -563.6689, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.686869, -563.9164, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.686869, -564.1741, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.686869, -564.4418, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.686869, -564.7195, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.686869, -565.0074, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.686869, -565.3054, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.686869, -565.6135, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.686869, -565.9317, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.686869, -566.2599, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.686869, -566.5983, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.686869, -566.9468, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.686869, -567.3054, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.686869, -567.674, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.686869, -568.0528, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.686869, -568.4417, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.686869, -568.8406, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.686869, -569.2496, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.686869, -569.6688, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.686869, -570.098, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.686869, -570.5373, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.686869, -570.9867, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.686869, -571.4462, 0.8588235, 0, 0.1372549, 1,
8, 5.686869, -571.9158, 0.8588235, 0, 0.1372549, 1,
4, 5.737374, -575.4131, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.737374, -574.9024, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.737374, -574.4017, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.737374, -573.9109, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.737374, -573.43, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.737374, -572.959, 0.8588235, 0, 0.1372549, 1,
4.242424, 5.737374, -572.498, 0.8588235, 0, 0.1372549, 1,
4.282828, 5.737374, -572.0468, 0.8588235, 0, 0.1372549, 1,
4.323232, 5.737374, -571.6056, 0.8588235, 0, 0.1372549, 1,
4.363636, 5.737374, -571.1743, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.737374, -570.7529, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.737374, -570.3415, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.737374, -569.9399, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.737374, -569.5483, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.737374, -569.1666, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.737374, -568.7948, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.737374, -568.4329, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.737374, -568.081, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.737374, -567.739, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.737374, -567.4069, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.737374, -567.0847, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.737374, -566.7724, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.737374, -566.47, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.737374, -566.1776, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.737374, -565.8951, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.737374, -565.6225, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.737374, -565.3598, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.737374, -565.1071, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.737374, -564.8642, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.737374, -564.6313, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.737374, -564.4083, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.737374, -564.1952, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.737374, -563.992, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.737374, -563.7988, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.737374, -563.6154, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.737374, -563.442, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.737374, -563.2785, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.737374, -563.1249, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.737374, -562.9813, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.737374, -562.8475, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.737374, -562.7237, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.737374, -562.6098, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.737374, -562.5059, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.737374, -562.4117, 1, 0.06666667, 0, 1,
5.777778, 5.737374, -562.3276, 1, 0.06666667, 0, 1,
5.818182, 5.737374, -562.2534, 1, 0.06666667, 0, 1,
5.858586, 5.737374, -562.1891, 1, 0.06666667, 0, 1,
5.89899, 5.737374, -562.1347, 1, 0.06666667, 0, 1,
5.939394, 5.737374, -562.0902, 1, 0.06666667, 0, 1,
5.979798, 5.737374, -562.0557, 1, 0.06666667, 0, 1,
6.020202, 5.737374, -562.0311, 1, 0.06666667, 0, 1,
6.060606, 5.737374, -562.0164, 1, 0.06666667, 0, 1,
6.10101, 5.737374, -562.0115, 1, 0.06666667, 0, 1,
6.141414, 5.737374, -562.0167, 1, 0.06666667, 0, 1,
6.181818, 5.737374, -562.0317, 1, 0.06666667, 0, 1,
6.222222, 5.737374, -562.0566, 1, 0.06666667, 0, 1,
6.262626, 5.737374, -562.0916, 1, 0.06666667, 0, 1,
6.30303, 5.737374, -562.1364, 1, 0.06666667, 0, 1,
6.343434, 5.737374, -562.191, 1, 0.06666667, 0, 1,
6.383838, 5.737374, -562.2557, 1, 0.06666667, 0, 1,
6.424242, 5.737374, -562.3303, 1, 0.06666667, 0, 1,
6.464646, 5.737374, -562.4147, 1, 0.06666667, 0, 1,
6.505051, 5.737374, -562.5091, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.737374, -562.6134, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.737374, -562.7277, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.737374, -562.8518, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.737374, -562.9858, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.737374, -563.1298, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.737374, -563.2838, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.737374, -563.4476, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.737374, -563.6213, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.737374, -563.8049, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.737374, -563.9985, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.737374, -564.202, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.737374, -564.4155, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.737374, -564.6388, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.737374, -564.872, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.737374, -565.1152, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.737374, -565.3683, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.737374, -565.6313, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.737374, -565.9042, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.737374, -566.1871, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.737374, -566.4799, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.737374, -566.7825, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.737374, -567.0952, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.737374, -567.4177, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.737374, -567.7501, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.737374, -568.0925, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.737374, -568.4447, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.737374, -568.8069, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.737374, -569.179, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.737374, -569.561, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.737374, -569.953, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.737374, -570.3549, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.737374, -570.7667, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.737374, -571.1884, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.737374, -571.62, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.737374, -572.0615, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.737374, -572.513, 0.8588235, 0, 0.1372549, 1,
8, 5.737374, -572.9744, 0.8588235, 0, 0.1372549, 1,
4, 5.787879, -576.4323, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.787879, -575.9305, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.787879, -575.4384, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.787879, -574.9562, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.787879, -574.4836, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.787879, -574.0208, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.787879, -573.5677, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.787879, -573.1245, 0.8588235, 0, 0.1372549, 1,
4.323232, 5.787879, -572.6909, 0.8588235, 0, 0.1372549, 1,
4.363636, 5.787879, -572.2672, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.787879, -571.8531, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.787879, -571.4487, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.787879, -571.0542, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.787879, -570.6694, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.787879, -570.2943, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.787879, -569.929, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.787879, -569.5734, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.787879, -569.2275, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.787879, -568.8915, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.787879, -568.5651, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.787879, -568.2485, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.787879, -567.9417, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.787879, -567.6446, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.787879, -567.3572, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.787879, -567.0797, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.787879, -566.8118, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.787879, -566.5536, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.787879, -566.3053, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.787879, -566.0667, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.787879, -565.8378, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.787879, -565.6187, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.787879, -565.4092, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.787879, -565.2096, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.787879, -565.0197, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.787879, -564.8396, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.787879, -564.6692, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.787879, -564.5085, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.787879, -564.3576, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.787879, -564.2164, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.787879, -564.085, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.787879, -563.9634, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.787879, -563.8514, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.787879, -563.7493, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.787879, -563.6569, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.787879, -563.5742, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.787879, -563.5012, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.787879, -563.438, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.787879, -563.3846, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.787879, -563.3409, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.787879, -563.3069, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.787879, -563.2827, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.787879, -563.2682, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.787879, -563.2635, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.787879, -563.2686, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.787879, -563.2834, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.787879, -563.3079, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.787879, -563.3422, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.787879, -563.3862, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.787879, -563.4399, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.787879, -563.5035, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.787879, -563.5767, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.787879, -563.6597, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.787879, -563.7525, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.787879, -563.855, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.787879, -563.9672, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.787879, -564.0892, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.787879, -564.2209, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.787879, -564.3624, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.787879, -564.5137, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.787879, -564.6746, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.787879, -564.8453, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.787879, -565.0258, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.787879, -565.2161, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.787879, -565.416, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.787879, -565.6257, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.787879, -565.8452, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.787879, -566.0743, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.787879, -566.3133, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.787879, -566.562, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.787879, -566.8204, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.787879, -567.0886, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.787879, -567.3666, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.787879, -567.6542, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.787879, -567.9517, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.787879, -568.2588, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.787879, -568.5757, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.787879, -568.9024, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.787879, -569.2388, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.787879, -569.585, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.787879, -569.9409, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.787879, -570.3065, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.787879, -570.6819, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.787879, -571.067, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.787879, -571.4619, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.787879, -571.8666, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.787879, -572.2809, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.787879, -572.7051, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.787879, -573.1389, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.787879, -573.5826, 0.7568628, 0, 0.2392157, 1,
8, 5.787879, -574.0359, 0.7568628, 0, 0.2392157, 1,
4, 5.838384, -577.4551, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.838384, -576.962, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.838384, -576.4785, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.838384, -576.0045, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.838384, -575.5401, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.838384, -575.0853, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.838384, -574.64, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.838384, -574.2043, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.838384, -573.7783, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.838384, -573.3618, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.838384, -572.9548, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.838384, -572.5575, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.838384, -572.1697, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.838384, -571.7916, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.838384, -571.4229, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.838384, -571.0639, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.838384, -570.7144, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.838384, -570.3746, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.838384, -570.0443, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.838384, -569.7236, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.838384, -569.4124, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.838384, -569.1108, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.838384, -568.8188, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.838384, -568.5364, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.838384, -568.2636, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.838384, -568.0004, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.838384, -567.7467, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.838384, -567.5026, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.838384, -567.2681, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.838384, -567.0432, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.838384, -566.8278, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.838384, -566.622, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.838384, -566.4258, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.838384, -566.2392, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.838384, -566.0621, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.838384, -565.8947, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.838384, -565.7368, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.838384, -565.5885, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.838384, -565.4498, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.838384, -565.3206, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.838384, -565.201, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.838384, -565.0911, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.838384, -564.9907, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.838384, -564.8998, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.838384, -564.8185, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.838384, -564.7469, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.838384, -564.6848, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.838384, -564.6323, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.838384, -564.5893, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.838384, -564.5559, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.838384, -564.5322, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.838384, -564.5179, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.838384, -564.5133, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.838384, -564.5182, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.838384, -564.5328, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.838384, -564.5569, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.838384, -564.5906, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.838384, -564.6338, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.838384, -564.6866, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.838384, -564.7491, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.838384, -564.821, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.838384, -564.9026, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.838384, -564.9938, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.838384, -565.0945, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.838384, -565.2048, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.838384, -565.3247, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.838384, -565.4542, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.838384, -565.5933, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.838384, -565.7419, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.838384, -565.9001, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.838384, -566.0679, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.838384, -566.2452, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.838384, -566.4321, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.838384, -566.6287, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.838384, -566.8348, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.838384, -567.0504, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.838384, -567.2757, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.838384, -567.5105, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.838384, -567.7549, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.838384, -568.0089, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.838384, -568.2725, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.838384, -568.5456, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.838384, -568.8283, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.838384, -569.1206, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.838384, -569.4225, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.838384, -569.7339, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.838384, -570.055, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.838384, -570.3856, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.838384, -570.7258, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.838384, -571.0756, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.838384, -571.4349, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.838384, -571.8038, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.838384, -572.1824, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.838384, -572.5704, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.838384, -572.9681, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.838384, -573.3754, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.838384, -573.7922, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.838384, -574.2186, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.838384, -574.6545, 0.7568628, 0, 0.2392157, 1,
8, 5.838384, -575.1001, 0.7568628, 0, 0.2392157, 1,
4, 5.888889, -578.4814, 0.7568628, 0, 0.2392157, 1,
4.040404, 5.888889, -577.9966, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.888889, -577.5214, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.888889, -577.0555, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.888889, -576.599, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.888889, -576.152, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.888889, -575.7144, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.888889, -575.2861, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.888889, -574.8673, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.888889, -574.4579, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.888889, -574.0579, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.888889, -573.6674, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.888889, -573.2863, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.888889, -572.9145, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.888889, -572.5522, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.888889, -572.1993, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.888889, -571.8558, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.888889, -571.5217, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.888889, -571.1971, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.888889, -570.8818, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.888889, -570.576, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.888889, -570.2796, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.888889, -569.9926, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.888889, -569.715, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.888889, -569.4468, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.888889, -569.1881, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.888889, -568.9387, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.888889, -568.6989, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.888889, -568.4683, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.888889, -568.2473, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.888889, -568.0355, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.888889, -567.8333, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.888889, -567.6404, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.888889, -567.457, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.888889, -567.283, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.888889, -567.1183, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.888889, -566.9632, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.888889, -566.8174, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.888889, -566.681, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.888889, -566.5541, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.888889, -566.4366, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.888889, -566.3284, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.888889, -566.2297, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.888889, -566.1405, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.888889, -566.0606, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.888889, -565.9902, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.888889, -565.9291, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.888889, -565.8775, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.888889, -565.8353, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.888889, -565.8025, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.888889, -565.7791, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.888889, -565.7651, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.888889, -565.7606, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.888889, -565.7654, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.888889, -565.7797, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.888889, -565.8034, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.888889, -565.8365, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.888889, -565.879, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.888889, -565.931, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.888889, -565.9923, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.888889, -566.0631, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.888889, -566.1432, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.888889, -566.2328, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.888889, -566.3318, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.888889, -566.4403, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.888889, -566.5582, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.888889, -566.6854, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.888889, -566.8221, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.888889, -566.9681, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.888889, -567.1237, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.888889, -567.2886, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.888889, -567.4629, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.888889, -567.6467, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.888889, -567.8398, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.888889, -568.0424, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.888889, -568.2544, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.888889, -568.4758, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.888889, -568.7066, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.888889, -568.9468, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.888889, -569.1965, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.888889, -569.4555, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.888889, -569.724, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.888889, -570.0019, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.888889, -570.2892, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.888889, -570.5859, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.888889, -570.8921, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.888889, -571.2076, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.888889, -571.5326, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.888889, -571.8669, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.888889, -572.2108, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.888889, -572.564, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.888889, -572.9266, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.888889, -573.2986, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.888889, -573.6801, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.888889, -574.071, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.888889, -574.4713, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.888889, -574.881, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.888889, -575.3, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.888889, -575.7286, 0.7568628, 0, 0.2392157, 1,
8, 5.888889, -576.1666, 0.7568628, 0, 0.2392157, 1,
4, 5.939394, -579.5106, 0.654902, 0, 0.3411765, 1,
4.040404, 5.939394, -579.0341, 0.654902, 0, 0.3411765, 1,
4.080808, 5.939394, -578.5668, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.939394, -578.1088, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.939394, -577.6601, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.939394, -577.2206, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.939394, -576.7904, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.939394, -576.3694, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.939394, -575.9577, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.939394, -575.5552, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.939394, -575.162, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.939394, -574.7781, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.939394, -574.4034, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.939394, -574.038, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.939394, -573.6818, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.939394, -573.3349, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.939394, -572.9972, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.939394, -572.6688, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.939394, -572.3496, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.939394, -572.0397, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.939394, -571.7391, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.939394, -571.4477, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.939394, -571.1656, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.939394, -570.8927, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.939394, -570.629, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.939394, -570.3747, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.939394, -570.1296, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.939394, -569.8937, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.939394, -569.6671, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.939394, -569.4498, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.939394, -569.2416, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.939394, -569.0428, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.939394, -568.8532, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.939394, -568.6729, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.939394, -568.5018, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.939394, -568.34, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.939394, -568.1874, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.939394, -568.0441, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.939394, -567.9101, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.939394, -567.7853, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.939394, -567.6697, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.939394, -567.5635, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.939394, -567.4664, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.939394, -567.3787, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.939394, -567.3002, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.939394, -567.2309, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.939394, -567.1709, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.939394, -567.1201, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.939394, -567.0786, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.939394, -567.0464, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.939394, -567.0234, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.939394, -567.0097, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.939394, -567.0052, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.939394, -567.01, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.939394, -567.024, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.939394, -567.0473, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.939394, -567.0798, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.939394, -567.1216, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.939394, -567.1727, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.939394, -567.233, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.939394, -567.3026, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.939394, -567.3814, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.939394, -567.4695, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.939394, -567.5668, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.939394, -567.6734, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.939394, -567.7892, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.939394, -567.9144, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.939394, -568.0487, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.939394, -568.1923, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.939394, -568.3452, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.939394, -568.5073, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.939394, -568.6787, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.939394, -568.8593, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.939394, -569.0493, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.939394, -569.2484, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.939394, -569.4568, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.939394, -569.6744, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.939394, -569.9013, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.939394, -570.1375, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.939394, -570.3829, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.939394, -570.6376, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.939394, -570.9016, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.939394, -571.1747, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.939394, -571.4572, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.939394, -571.7488, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.939394, -572.0498, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.939394, -572.36, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.939394, -572.6795, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.939394, -573.0082, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.939394, -573.3462, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.939394, -573.6934, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.939394, -574.0499, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.939394, -574.4156, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.939394, -574.7906, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.939394, -575.1749, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.939394, -575.5684, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.939394, -575.9711, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.939394, -576.3832, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.939394, -576.8044, 0.7568628, 0, 0.2392157, 1,
8, 5.939394, -577.235, 0.7568628, 0, 0.2392157, 1,
4, 5.989899, -580.5424, 0.654902, 0, 0.3411765, 1,
4.040404, 5.989899, -580.0739, 0.654902, 0, 0.3411765, 1,
4.080808, 5.989899, -579.6145, 0.654902, 0, 0.3411765, 1,
4.121212, 5.989899, -579.1642, 0.654902, 0, 0.3411765, 1,
4.161616, 5.989899, -578.723, 0.654902, 0, 0.3411765, 1,
4.20202, 5.989899, -578.291, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.989899, -577.8679, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.989899, -577.454, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.989899, -577.0493, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.989899, -576.6535, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.989899, -576.2669, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.989899, -575.8894, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.989899, -575.5211, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.989899, -575.1617, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.989899, -574.8115, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.989899, -574.4705, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.989899, -574.1384, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.989899, -573.8156, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.989899, -573.5017, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.989899, -573.197, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.989899, -572.9014, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.989899, -572.6149, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.989899, -572.3375, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.989899, -572.0692, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.989899, -571.8101, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.989899, -571.5599, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.989899, -571.319, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.989899, -571.087, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.989899, -570.8643, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.989899, -570.6505, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.989899, -570.4459, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.989899, -570.2504, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.989899, -570.064, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.989899, -569.8867, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.989899, -569.7186, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.989899, -569.5594, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.989899, -569.4094, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.989899, -569.2686, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.989899, -569.1367, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.989899, -569.014, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.989899, -568.9005, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.989899, -568.796, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.989899, -568.7006, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.989899, -568.6143, 0.8588235, 0, 0.1372549, 1,
5.777778, 5.989899, -568.537, 0.8588235, 0, 0.1372549, 1,
5.818182, 5.989899, -568.4689, 0.8588235, 0, 0.1372549, 1,
5.858586, 5.989899, -568.41, 0.8588235, 0, 0.1372549, 1,
5.89899, 5.989899, -568.36, 0.8588235, 0, 0.1372549, 1,
5.939394, 5.989899, -568.3193, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.989899, -568.2875, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.989899, -568.265, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.989899, -568.2515, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.989899, -568.2471, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.989899, -568.2518, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.989899, -568.2656, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.989899, -568.2885, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.989899, -568.3204, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.989899, -568.3616, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.989899, -568.4117, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.989899, -568.4711, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.989899, -568.5394, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.989899, -568.6169, 0.8588235, 0, 0.1372549, 1,
6.505051, 5.989899, -568.7036, 0.8588235, 0, 0.1372549, 1,
6.545455, 5.989899, -568.7993, 0.8588235, 0, 0.1372549, 1,
6.585859, 5.989899, -568.9041, 0.8588235, 0, 0.1372549, 1,
6.626263, 5.989899, -569.0179, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.989899, -569.1409, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.989899, -569.2731, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.989899, -569.4142, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.989899, -569.5646, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.989899, -569.7239, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.989899, -569.8925, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.989899, -570.0701, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.989899, -570.2567, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.989899, -570.4525, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.989899, -570.6575, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.989899, -570.8715, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.989899, -571.0945, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.989899, -571.3267, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.989899, -571.5681, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.989899, -571.8184, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.989899, -572.0779, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.989899, -572.3466, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.989899, -572.6243, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.989899, -572.911, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.989899, -573.2069, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.989899, -573.512, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.989899, -573.826, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.989899, -574.1492, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.989899, -574.4815, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.989899, -574.8229, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.989899, -575.1735, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.989899, -575.533, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.989899, -575.9017, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.989899, -576.2795, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.989899, -576.6664, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.989899, -577.0624, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.989899, -577.4675, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.989899, -577.8817, 0.7568628, 0, 0.2392157, 1,
8, 5.989899, -578.3051, 0.7568628, 0, 0.2392157, 1,
4, 6.040404, -581.5766, 0.654902, 0, 0.3411765, 1,
4.040404, 6.040404, -581.1159, 0.654902, 0, 0.3411765, 1,
4.080808, 6.040404, -580.6641, 0.654902, 0, 0.3411765, 1,
4.121212, 6.040404, -580.2213, 0.654902, 0, 0.3411765, 1,
4.161616, 6.040404, -579.7875, 0.654902, 0, 0.3411765, 1,
4.20202, 6.040404, -579.3625, 0.654902, 0, 0.3411765, 1,
4.242424, 6.040404, -578.9466, 0.654902, 0, 0.3411765, 1,
4.282828, 6.040404, -578.5396, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.040404, -578.1415, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.040404, -577.7524, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.040404, -577.3723, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.040404, -577.0011, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.040404, -576.6388, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.040404, -576.2855, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.040404, -575.9411, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.040404, -575.6057, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.040404, -575.2792, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.040404, -574.9617, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.040404, -574.6531, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.040404, -574.3535, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.040404, -574.0629, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.040404, -573.7811, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.040404, -573.5083, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.040404, -573.2445, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.040404, -572.9896, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.040404, -572.7437, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.040404, -572.5067, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.040404, -572.2786, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.040404, -572.0596, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.040404, -571.8494, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.040404, -571.6483, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.040404, -571.456, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.040404, -571.2727, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.040404, -571.0983, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.040404, -570.9329, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.040404, -570.7765, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.040404, -570.629, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.040404, -570.4904, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.040404, -570.3608, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.040404, -570.2402, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.040404, -570.1285, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.040404, -570.0257, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.040404, -569.9319, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.040404, -569.847, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.040404, -569.7711, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.040404, -569.7042, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.040404, -569.6461, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.040404, -569.597, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.040404, -569.5569, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.040404, -569.5258, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.040404, -569.5035, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.040404, -569.4903, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.040404, -569.486, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.040404, -569.4906, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.040404, -569.5042, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.040404, -569.5267, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.040404, -569.5581, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.040404, -569.5986, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.040404, -569.6479, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.040404, -569.7062, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.040404, -569.7735, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.040404, -569.8497, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.040404, -569.9349, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.040404, -570.0289, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.040404, -570.132, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.040404, -570.244, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.040404, -570.365, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.040404, -570.4949, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.040404, -570.6337, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.040404, -570.7815, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.040404, -570.9382, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.040404, -571.1039, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.040404, -571.2786, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.040404, -571.4622, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.040404, -571.6547, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.040404, -571.8562, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.040404, -572.0667, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.040404, -572.286, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.040404, -572.5143, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.040404, -572.7516, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.040404, -572.9979, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.040404, -573.2531, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.040404, -573.5172, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.040404, -573.7903, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.040404, -574.0723, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.040404, -574.3632, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.040404, -574.6631, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.040404, -574.972, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.040404, -575.2899, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.040404, -575.6166, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.040404, -575.9523, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.040404, -576.297, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.040404, -576.6506, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.040404, -577.0132, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.040404, -577.3847, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.040404, -577.7651, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.040404, -578.1545, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.040404, -578.5529, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.040404, -578.9602, 0.654902, 0, 0.3411765, 1,
8, 6.040404, -579.3765, 0.654902, 0, 0.3411765, 1,
4, 6.090909, -582.6127, 0.654902, 0, 0.3411765, 1,
4.040404, 6.090909, -582.1597, 0.654902, 0, 0.3411765, 1,
4.080808, 6.090909, -581.7153, 0.654902, 0, 0.3411765, 1,
4.121212, 6.090909, -581.2798, 0.654902, 0, 0.3411765, 1,
4.161616, 6.090909, -580.8532, 0.654902, 0, 0.3411765, 1,
4.20202, 6.090909, -580.4353, 0.654902, 0, 0.3411765, 1,
4.242424, 6.090909, -580.0262, 0.654902, 0, 0.3411765, 1,
4.282828, 6.090909, -579.6259, 0.654902, 0, 0.3411765, 1,
4.323232, 6.090909, -579.2344, 0.654902, 0, 0.3411765, 1,
4.363636, 6.090909, -578.8517, 0.654902, 0, 0.3411765, 1,
4.40404, 6.090909, -578.4778, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.090909, -578.1128, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.090909, -577.7565, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.090909, -577.4091, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.090909, -577.0704, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.090909, -576.7405, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.090909, -576.4194, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.090909, -576.1071, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.090909, -575.8036, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.090909, -575.509, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.090909, -575.2231, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.090909, -574.946, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.090909, -574.6777, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.090909, -574.4183, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.090909, -574.1676, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.090909, -573.9257, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.090909, -573.6926, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.090909, -573.4684, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.090909, -573.2529, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.090909, -573.0462, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.090909, -572.8483, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.090909, -572.6593, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.090909, -572.479, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.090909, -572.3076, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.090909, -572.1449, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.090909, -571.991, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.090909, -571.8459, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.090909, -571.7097, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.090909, -571.5822, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.090909, -571.4636, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.090909, -571.3537, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.090909, -571.2526, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.090909, -571.1603, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.090909, -571.0769, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.090909, -571.0023, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.090909, -570.9364, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.090909, -570.8793, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.090909, -570.8311, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.090909, -570.7916, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.090909, -570.7609, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.090909, -570.7391, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.090909, -570.726, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.090909, -570.7218, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.090909, -570.7263, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.090909, -570.7397, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.090909, -570.7618, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.090909, -570.7928, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.090909, -570.8325, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.090909, -570.881, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.090909, -570.9384, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.090909, -571.0046, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.090909, -571.0795, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.090909, -571.1633, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.090909, -571.2558, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.090909, -571.3572, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.090909, -571.4673, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.090909, -571.5863, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.090909, -571.7141, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.090909, -571.8506, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.090909, -571.996, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.090909, -572.1501, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.090909, -572.313, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.090909, -572.4848, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.090909, -572.6654, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.090909, -572.8547, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.090909, -573.0529, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.090909, -573.2598, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.090909, -573.4756, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.090909, -573.7002, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.090909, -573.9335, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.090909, -574.1757, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.090909, -574.4267, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.090909, -574.6865, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.090909, -574.955, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.090909, -575.2324, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.090909, -575.5186, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.090909, -575.8135, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.090909, -576.1172, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.090909, -576.4298, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.090909, -576.7512, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.090909, -577.0814, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.090909, -577.4203, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.090909, -577.7681, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.090909, -578.1247, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.090909, -578.4901, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.090909, -578.8643, 0.654902, 0, 0.3411765, 1,
7.878788, 6.090909, -579.2472, 0.654902, 0, 0.3411765, 1,
7.919192, 6.090909, -579.639, 0.654902, 0, 0.3411765, 1,
7.959596, 6.090909, -580.0396, 0.654902, 0, 0.3411765, 1,
8, 6.090909, -580.4489, 0.654902, 0, 0.3411765, 1,
4, 6.141414, -583.6506, 0.654902, 0, 0.3411765, 1,
4.040404, 6.141414, -583.205, 0.654902, 0, 0.3411765, 1,
4.080808, 6.141414, -582.7679, 0.654902, 0, 0.3411765, 1,
4.121212, 6.141414, -582.3395, 0.654902, 0, 0.3411765, 1,
4.161616, 6.141414, -581.9199, 0.654902, 0, 0.3411765, 1,
4.20202, 6.141414, -581.5089, 0.654902, 0, 0.3411765, 1,
4.242424, 6.141414, -581.1064, 0.654902, 0, 0.3411765, 1,
4.282828, 6.141414, -580.7127, 0.654902, 0, 0.3411765, 1,
4.323232, 6.141414, -580.3276, 0.654902, 0, 0.3411765, 1,
4.363636, 6.141414, -579.9512, 0.654902, 0, 0.3411765, 1,
4.40404, 6.141414, -579.5834, 0.654902, 0, 0.3411765, 1,
4.444445, 6.141414, -579.2244, 0.654902, 0, 0.3411765, 1,
4.484848, 6.141414, -578.8739, 0.654902, 0, 0.3411765, 1,
4.525252, 6.141414, -578.5321, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.141414, -578.199, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.141414, -577.8745, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.141414, -577.5587, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.141414, -577.2515, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.141414, -576.953, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.141414, -576.6631, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.141414, -576.382, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.141414, -576.1094, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.141414, -575.8456, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.141414, -575.5903, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.141414, -575.3438, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.141414, -575.1058, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.141414, -574.8766, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.141414, -574.656, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.141414, -574.4441, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.141414, -574.2408, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.141414, -574.0461, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.141414, -573.8602, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.141414, -573.6829, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.141414, -573.5142, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.141414, -573.3542, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.141414, -573.2029, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.141414, -573.0602, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.141414, -572.9261, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.141414, -572.8008, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.141414, -572.684, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.141414, -572.576, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.141414, -572.4766, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.141414, -572.3858, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.141414, -572.3037, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.141414, -572.2303, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.141414, -572.1655, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.141414, -572.1094, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.141414, -572.0619, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.141414, -572.0231, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.141414, -571.9929, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.141414, -571.9714, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.141414, -571.9586, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.141414, -571.9544, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.141414, -571.9589, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.141414, -571.972, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.141414, -571.9938, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.141414, -572.0242, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.141414, -572.0634, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.141414, -572.1111, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.141414, -572.1675, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.141414, -572.2325, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.141414, -572.3063, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.141414, -572.3887, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.141414, -572.4797, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.141414, -572.5794, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.141414, -572.6877, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.141414, -572.8047, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.141414, -572.9304, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.141414, -573.0648, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.141414, -573.2077, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.141414, -573.3593, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.141414, -573.5197, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.141414, -573.6886, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.141414, -573.8661, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.141414, -574.0524, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.141414, -574.2473, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.141414, -574.4509, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.141414, -574.6631, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.141414, -574.884, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.141414, -575.1136, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.141414, -575.3517, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.141414, -575.5986, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.141414, -575.8541, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.141414, -576.1183, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.141414, -576.3911, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.141414, -576.6726, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.141414, -576.9627, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.141414, -577.2615, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.141414, -577.569, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.141414, -577.8851, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.141414, -578.2098, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.141414, -578.5433, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.141414, -578.8853, 0.654902, 0, 0.3411765, 1,
7.757576, 6.141414, -579.2361, 0.654902, 0, 0.3411765, 1,
7.79798, 6.141414, -579.5955, 0.654902, 0, 0.3411765, 1,
7.838384, 6.141414, -579.9635, 0.654902, 0, 0.3411765, 1,
7.878788, 6.141414, -580.3402, 0.654902, 0, 0.3411765, 1,
7.919192, 6.141414, -580.7255, 0.654902, 0, 0.3411765, 1,
7.959596, 6.141414, -581.1196, 0.654902, 0, 0.3411765, 1,
8, 6.141414, -581.5222, 0.654902, 0, 0.3411765, 1,
4, 6.191919, -584.6899, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.191919, -584.2515, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.191919, -583.8215, 0.654902, 0, 0.3411765, 1,
4.121212, 6.191919, -583.4001, 0.654902, 0, 0.3411765, 1,
4.161616, 6.191919, -582.9872, 0.654902, 0, 0.3411765, 1,
4.20202, 6.191919, -582.5829, 0.654902, 0, 0.3411765, 1,
4.242424, 6.191919, -582.1871, 0.654902, 0, 0.3411765, 1,
4.282828, 6.191919, -581.7997, 0.654902, 0, 0.3411765, 1,
4.323232, 6.191919, -581.4209, 0.654902, 0, 0.3411765, 1,
4.363636, 6.191919, -581.0506, 0.654902, 0, 0.3411765, 1,
4.40404, 6.191919, -580.6888, 0.654902, 0, 0.3411765, 1,
4.444445, 6.191919, -580.3356, 0.654902, 0, 0.3411765, 1,
4.484848, 6.191919, -579.9908, 0.654902, 0, 0.3411765, 1,
4.525252, 6.191919, -579.6546, 0.654902, 0, 0.3411765, 1,
4.565657, 6.191919, -579.3268, 0.654902, 0, 0.3411765, 1,
4.606061, 6.191919, -579.0076, 0.654902, 0, 0.3411765, 1,
4.646465, 6.191919, -578.697, 0.654902, 0, 0.3411765, 1,
4.686869, 6.191919, -578.3948, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.191919, -578.1011, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.191919, -577.816, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.191919, -577.5394, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.191919, -577.2712, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.191919, -577.0117, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.191919, -576.7606, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.191919, -576.518, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.191919, -576.284, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.191919, -576.0585, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.191919, -575.8414, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.191919, -575.6329, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.191919, -575.4329, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.191919, -575.2415, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.191919, -575.0585, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.191919, -574.8841, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.191919, -574.7182, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.191919, -574.5608, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.191919, -574.4119, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.191919, -574.2715, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.191919, -574.1396, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.191919, -574.0163, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.191919, -573.9015, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.191919, -573.7952, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.191919, -573.6974, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.191919, -573.6081, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.191919, -573.5273, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.191919, -573.4551, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.191919, -573.3914, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.191919, -573.3362, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.191919, -573.2895, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.191919, -573.2513, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.191919, -573.2216, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.191919, -573.2005, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.191919, -573.1879, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.191919, -573.1837, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.191919, -573.1881, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.191919, -573.201, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.191919, -573.2225, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.191919, -573.2524, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.191919, -573.2909, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.191919, -573.3378, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.191919, -573.3934, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.191919, -573.4573, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.191919, -573.5299, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.191919, -573.6109, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.191919, -573.7005, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.191919, -573.7986, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.191919, -573.9052, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.191919, -574.0203, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.191919, -574.1439, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.191919, -574.276, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.191919, -574.4167, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.191919, -574.5659, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.191919, -574.7235, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.191919, -574.8897, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.191919, -575.0645, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.191919, -575.2477, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.191919, -575.4394, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.191919, -575.6396, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.191919, -575.8484, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.191919, -576.0657, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.191919, -576.2916, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.191919, -576.5259, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.191919, -576.7687, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.191919, -577.0201, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.191919, -577.28, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.191919, -577.5483, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.191919, -577.8253, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.191919, -578.1107, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.191919, -578.4046, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.191919, -578.7071, 0.654902, 0, 0.3411765, 1,
7.59596, 6.191919, -579.0181, 0.654902, 0, 0.3411765, 1,
7.636364, 6.191919, -579.3375, 0.654902, 0, 0.3411765, 1,
7.676768, 6.191919, -579.6655, 0.654902, 0, 0.3411765, 1,
7.717172, 6.191919, -580.002, 0.654902, 0, 0.3411765, 1,
7.757576, 6.191919, -580.347, 0.654902, 0, 0.3411765, 1,
7.79798, 6.191919, -580.7006, 0.654902, 0, 0.3411765, 1,
7.838384, 6.191919, -581.0627, 0.654902, 0, 0.3411765, 1,
7.878788, 6.191919, -581.4333, 0.654902, 0, 0.3411765, 1,
7.919192, 6.191919, -581.8124, 0.654902, 0, 0.3411765, 1,
7.959596, 6.191919, -582.2, 0.654902, 0, 0.3411765, 1,
8, 6.191919, -582.5961, 0.654902, 0, 0.3411765, 1,
4, 6.242424, -585.7303, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.242424, -585.299, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.242424, -584.876, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.242424, -584.4614, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.242424, -584.0552, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.242424, -583.6573, 0.654902, 0, 0.3411765, 1,
4.242424, 6.242424, -583.2679, 0.654902, 0, 0.3411765, 1,
4.282828, 6.242424, -582.8868, 0.654902, 0, 0.3411765, 1,
4.323232, 6.242424, -582.514, 0.654902, 0, 0.3411765, 1,
4.363636, 6.242424, -582.1497, 0.654902, 0, 0.3411765, 1,
4.40404, 6.242424, -581.7938, 0.654902, 0, 0.3411765, 1,
4.444445, 6.242424, -581.4462, 0.654902, 0, 0.3411765, 1,
4.484848, 6.242424, -581.107, 0.654902, 0, 0.3411765, 1,
4.525252, 6.242424, -580.7762, 0.654902, 0, 0.3411765, 1,
4.565657, 6.242424, -580.4537, 0.654902, 0, 0.3411765, 1,
4.606061, 6.242424, -580.1396, 0.654902, 0, 0.3411765, 1,
4.646465, 6.242424, -579.834, 0.654902, 0, 0.3411765, 1,
4.686869, 6.242424, -579.5367, 0.654902, 0, 0.3411765, 1,
4.727273, 6.242424, -579.2477, 0.654902, 0, 0.3411765, 1,
4.767677, 6.242424, -578.9672, 0.654902, 0, 0.3411765, 1,
4.808081, 6.242424, -578.6951, 0.654902, 0, 0.3411765, 1,
4.848485, 6.242424, -578.4313, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.242424, -578.1758, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.242424, -577.9288, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.242424, -577.6902, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.242424, -577.4599, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.242424, -577.238, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.242424, -577.0245, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.242424, -576.8193, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.242424, -576.6226, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.242424, -576.4342, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.242424, -576.2542, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.242424, -576.0826, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.242424, -575.9193, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.242424, -575.7645, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.242424, -575.618, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.242424, -575.4799, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.242424, -575.3502, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.242424, -575.2288, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.242424, -575.1158, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.242424, -575.0112, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.242424, -574.915, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.242424, -574.8271, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.242424, -574.7477, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.242424, -574.6766, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.242424, -574.614, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.242424, -574.5596, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.242424, -574.5137, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.242424, -574.4761, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.242424, -574.4469, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.242424, -574.4261, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.242424, -574.4137, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.242424, -574.4096, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.242424, -574.4139, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.242424, -574.4266, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.242424, -574.4478, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.242424, -574.4772, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.242424, -574.515, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.242424, -574.5613, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.242424, -574.6158, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.242424, -574.6788, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.242424, -574.7502, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.242424, -574.83, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.242424, -574.918, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.242424, -575.0145, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.242424, -575.1194, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.242424, -575.2327, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.242424, -575.3543, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.242424, -575.4843, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.242424, -575.6227, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.242424, -575.7695, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.242424, -575.9246, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.242424, -576.0881, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.242424, -576.26, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.242424, -576.4403, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.242424, -576.629, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.242424, -576.826, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.242424, -577.0314, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.242424, -577.2452, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.242424, -577.4673, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.242424, -577.6979, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.242424, -577.9368, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.242424, -578.1841, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.242424, -578.4398, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.242424, -578.7039, 0.654902, 0, 0.3411765, 1,
7.434343, 6.242424, -578.9763, 0.654902, 0, 0.3411765, 1,
7.474748, 6.242424, -579.2571, 0.654902, 0, 0.3411765, 1,
7.515152, 6.242424, -579.5464, 0.654902, 0, 0.3411765, 1,
7.555555, 6.242424, -579.8439, 0.654902, 0, 0.3411765, 1,
7.59596, 6.242424, -580.1499, 0.654902, 0, 0.3411765, 1,
7.636364, 6.242424, -580.4642, 0.654902, 0, 0.3411765, 1,
7.676768, 6.242424, -580.7869, 0.654902, 0, 0.3411765, 1,
7.717172, 6.242424, -581.118, 0.654902, 0, 0.3411765, 1,
7.757576, 6.242424, -581.4575, 0.654902, 0, 0.3411765, 1,
7.79798, 6.242424, -581.8054, 0.654902, 0, 0.3411765, 1,
7.838384, 6.242424, -582.1616, 0.654902, 0, 0.3411765, 1,
7.878788, 6.242424, -582.5262, 0.654902, 0, 0.3411765, 1,
7.919192, 6.242424, -582.8992, 0.654902, 0, 0.3411765, 1,
7.959596, 6.242424, -583.2806, 0.654902, 0, 0.3411765, 1,
8, 6.242424, -583.6703, 0.654902, 0, 0.3411765, 1,
4, 6.292929, -586.7717, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.292929, -586.3472, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.292929, -585.931, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.292929, -585.5231, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.292929, -585.1234, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.292929, -584.7318, 0.5490196, 0, 0.4470588, 1,
4.242424, 6.292929, -584.3486, 0.5490196, 0, 0.4470588, 1,
4.282828, 6.292929, -583.9736, 0.654902, 0, 0.3411765, 1,
4.323232, 6.292929, -583.6069, 0.654902, 0, 0.3411765, 1,
4.363636, 6.292929, -583.2484, 0.654902, 0, 0.3411765, 1,
4.40404, 6.292929, -582.8981, 0.654902, 0, 0.3411765, 1,
4.444445, 6.292929, -582.556, 0.654902, 0, 0.3411765, 1,
4.484848, 6.292929, -582.2223, 0.654902, 0, 0.3411765, 1,
4.525252, 6.292929, -581.8967, 0.654902, 0, 0.3411765, 1,
4.565657, 6.292929, -581.5795, 0.654902, 0, 0.3411765, 1,
4.606061, 6.292929, -581.2704, 0.654902, 0, 0.3411765, 1,
4.646465, 6.292929, -580.9697, 0.654902, 0, 0.3411765, 1,
4.686869, 6.292929, -580.6771, 0.654902, 0, 0.3411765, 1,
4.727273, 6.292929, -580.3928, 0.654902, 0, 0.3411765, 1,
4.767677, 6.292929, -580.1167, 0.654902, 0, 0.3411765, 1,
4.808081, 6.292929, -579.8489, 0.654902, 0, 0.3411765, 1,
4.848485, 6.292929, -579.5894, 0.654902, 0, 0.3411765, 1,
4.888889, 6.292929, -579.338, 0.654902, 0, 0.3411765, 1,
4.929293, 6.292929, -579.0949, 0.654902, 0, 0.3411765, 1,
4.969697, 6.292929, -578.8601, 0.654902, 0, 0.3411765, 1,
5.010101, 6.292929, -578.6335, 0.654902, 0, 0.3411765, 1,
5.050505, 6.292929, -578.4152, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.292929, -578.205, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.292929, -578.0032, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.292929, -577.8096, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.292929, -577.6242, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.292929, -577.4471, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.292929, -577.2782, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.292929, -577.1176, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.292929, -576.9651, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.292929, -576.821, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.292929, -576.6851, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.292929, -576.5574, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.292929, -576.438, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.292929, -576.3269, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.292929, -576.2239, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.292929, -576.1293, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.292929, -576.0428, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.292929, -575.9647, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.292929, -575.8947, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.292929, -575.833, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.292929, -575.7795, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.292929, -575.7344, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.292929, -575.6974, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.292929, -575.6686, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.292929, -575.6482, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.292929, -575.636, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.292929, -575.632, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.292929, -575.6362, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.292929, -575.6487, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.292929, -575.6695, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.292929, -575.6985, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.292929, -575.7357, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.292929, -575.7812, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.292929, -575.8349, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.292929, -575.8969, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.292929, -575.9671, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.292929, -576.0456, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.292929, -576.1323, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.292929, -576.2272, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.292929, -576.3304, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.292929, -576.4418, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.292929, -576.5615, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.292929, -576.6895, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.292929, -576.8256, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.292929, -576.97, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.292929, -577.1227, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.292929, -577.2836, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.292929, -577.4528, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.292929, -577.6302, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.292929, -577.8158, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.292929, -578.0097, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.292929, -578.2119, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.292929, -578.4222, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.292929, -578.6408, 0.654902, 0, 0.3411765, 1,
7.232323, 6.292929, -578.8677, 0.654902, 0, 0.3411765, 1,
7.272727, 6.292929, -579.1028, 0.654902, 0, 0.3411765, 1,
7.313131, 6.292929, -579.3461, 0.654902, 0, 0.3411765, 1,
7.353535, 6.292929, -579.5978, 0.654902, 0, 0.3411765, 1,
7.393939, 6.292929, -579.8576, 0.654902, 0, 0.3411765, 1,
7.434343, 6.292929, -580.1257, 0.654902, 0, 0.3411765, 1,
7.474748, 6.292929, -580.402, 0.654902, 0, 0.3411765, 1,
7.515152, 6.292929, -580.6866, 0.654902, 0, 0.3411765, 1,
7.555555, 6.292929, -580.9794, 0.654902, 0, 0.3411765, 1,
7.59596, 6.292929, -581.2805, 0.654902, 0, 0.3411765, 1,
7.636364, 6.292929, -581.5898, 0.654902, 0, 0.3411765, 1,
7.676768, 6.292929, -581.9073, 0.654902, 0, 0.3411765, 1,
7.717172, 6.292929, -582.2332, 0.654902, 0, 0.3411765, 1,
7.757576, 6.292929, -582.5672, 0.654902, 0, 0.3411765, 1,
7.79798, 6.292929, -582.9095, 0.654902, 0, 0.3411765, 1,
7.838384, 6.292929, -583.26, 0.654902, 0, 0.3411765, 1,
7.878788, 6.292929, -583.6188, 0.654902, 0, 0.3411765, 1,
7.919192, 6.292929, -583.9858, 0.5490196, 0, 0.4470588, 1,
7.959596, 6.292929, -584.3611, 0.5490196, 0, 0.4470588, 1,
8, 6.292929, -584.7446, 0.5490196, 0, 0.4470588, 1,
4, 6.343434, -587.8138, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.343434, -587.3961, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.343434, -586.9865, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.343434, -586.5849, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.343434, -586.1915, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.343434, -585.8063, 0.5490196, 0, 0.4470588, 1,
4.242424, 6.343434, -585.4291, 0.5490196, 0, 0.4470588, 1,
4.282828, 6.343434, -585.0601, 0.5490196, 0, 0.4470588, 1,
4.323232, 6.343434, -584.6991, 0.5490196, 0, 0.4470588, 1,
4.363636, 6.343434, -584.3463, 0.5490196, 0, 0.4470588, 1,
4.40404, 6.343434, -584.0016, 0.5490196, 0, 0.4470588, 1,
4.444445, 6.343434, -583.665, 0.654902, 0, 0.3411765, 1,
4.484848, 6.343434, -583.3365, 0.654902, 0, 0.3411765, 1,
4.525252, 6.343434, -583.0161, 0.654902, 0, 0.3411765, 1,
4.565657, 6.343434, -582.7039, 0.654902, 0, 0.3411765, 1,
4.606061, 6.343434, -582.3997, 0.654902, 0, 0.3411765, 1,
4.646465, 6.343434, -582.1037, 0.654902, 0, 0.3411765, 1,
4.686869, 6.343434, -581.8158, 0.654902, 0, 0.3411765, 1,
4.727273, 6.343434, -581.536, 0.654902, 0, 0.3411765, 1,
4.767677, 6.343434, -581.2643, 0.654902, 0, 0.3411765, 1,
4.808081, 6.343434, -581.0007, 0.654902, 0, 0.3411765, 1,
4.848485, 6.343434, -580.7453, 0.654902, 0, 0.3411765, 1,
4.888889, 6.343434, -580.498, 0.654902, 0, 0.3411765, 1,
4.929293, 6.343434, -580.2587, 0.654902, 0, 0.3411765, 1,
4.969697, 6.343434, -580.0276, 0.654902, 0, 0.3411765, 1,
5.010101, 6.343434, -579.8046, 0.654902, 0, 0.3411765, 1,
5.050505, 6.343434, -579.5897, 0.654902, 0, 0.3411765, 1,
5.090909, 6.343434, -579.383, 0.654902, 0, 0.3411765, 1,
5.131313, 6.343434, -579.1843, 0.654902, 0, 0.3411765, 1,
5.171717, 6.343434, -578.9938, 0.654902, 0, 0.3411765, 1,
5.212121, 6.343434, -578.8113, 0.654902, 0, 0.3411765, 1,
5.252525, 6.343434, -578.637, 0.654902, 0, 0.3411765, 1,
5.292929, 6.343434, -578.4708, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.343434, -578.3127, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.343434, -578.1628, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.343434, -578.0209, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.343434, -577.8871, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.343434, -577.7615, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.343434, -577.644, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.343434, -577.5346, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.343434, -577.4333, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.343434, -577.3401, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.343434, -577.2551, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.343434, -577.1781, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.343434, -577.1093, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.343434, -577.0486, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.343434, -576.996, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.343434, -576.9515, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.343434, -576.9151, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.343434, -576.8868, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.343434, -576.8667, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.343434, -576.8546, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.343434, -576.8507, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.343434, -576.8549, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.343434, -576.8672, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.343434, -576.8876, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.343434, -576.9161, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.343434, -576.9528, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.343434, -576.9976, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.343434, -577.0504, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.343434, -577.1115, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.343434, -577.1805, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.343434, -577.2578, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.343434, -577.3431, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.343434, -577.4365, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.343434, -577.5381, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.343434, -577.6478, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.343434, -577.7656, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.343434, -577.8914, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.343434, -578.0255, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.343434, -578.1676, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.343434, -578.3178, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.343434, -578.4762, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.343434, -578.6426, 0.654902, 0, 0.3411765, 1,
6.989899, 6.343434, -578.8172, 0.654902, 0, 0.3411765, 1,
7.030303, 6.343434, -578.9999, 0.654902, 0, 0.3411765, 1,
7.070707, 6.343434, -579.1907, 0.654902, 0, 0.3411765, 1,
7.111111, 6.343434, -579.3896, 0.654902, 0, 0.3411765, 1,
7.151515, 6.343434, -579.5967, 0.654902, 0, 0.3411765, 1,
7.191919, 6.343434, -579.8118, 0.654902, 0, 0.3411765, 1,
7.232323, 6.343434, -580.0351, 0.654902, 0, 0.3411765, 1,
7.272727, 6.343434, -580.2665, 0.654902, 0, 0.3411765, 1,
7.313131, 6.343434, -580.506, 0.654902, 0, 0.3411765, 1,
7.353535, 6.343434, -580.7536, 0.654902, 0, 0.3411765, 1,
7.393939, 6.343434, -581.0093, 0.654902, 0, 0.3411765, 1,
7.434343, 6.343434, -581.2731, 0.654902, 0, 0.3411765, 1,
7.474748, 6.343434, -581.5451, 0.654902, 0, 0.3411765, 1,
7.515152, 6.343434, -581.8252, 0.654902, 0, 0.3411765, 1,
7.555555, 6.343434, -582.1133, 0.654902, 0, 0.3411765, 1,
7.59596, 6.343434, -582.4097, 0.654902, 0, 0.3411765, 1,
7.636364, 6.343434, -582.7141, 0.654902, 0, 0.3411765, 1,
7.676768, 6.343434, -583.0266, 0.654902, 0, 0.3411765, 1,
7.717172, 6.343434, -583.3472, 0.654902, 0, 0.3411765, 1,
7.757576, 6.343434, -583.676, 0.654902, 0, 0.3411765, 1,
7.79798, 6.343434, -584.0128, 0.5490196, 0, 0.4470588, 1,
7.838384, 6.343434, -584.3578, 0.5490196, 0, 0.4470588, 1,
7.878788, 6.343434, -584.7109, 0.5490196, 0, 0.4470588, 1,
7.919192, 6.343434, -585.0721, 0.5490196, 0, 0.4470588, 1,
7.959596, 6.343434, -585.4414, 0.5490196, 0, 0.4470588, 1,
8, 6.343434, -585.8188, 0.5490196, 0, 0.4470588, 1,
4, 6.393939, -588.8563, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.393939, -588.4451, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.393939, -588.0419, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.393939, -587.6468, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.393939, -587.2596, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.393939, -586.8804, 0.5490196, 0, 0.4470588, 1,
4.242424, 6.393939, -586.5092, 0.5490196, 0, 0.4470588, 1,
4.282828, 6.393939, -586.1459, 0.5490196, 0, 0.4470588, 1,
4.323232, 6.393939, -585.7906, 0.5490196, 0, 0.4470588, 1,
4.363636, 6.393939, -585.4434, 0.5490196, 0, 0.4470588, 1,
4.40404, 6.393939, -585.1041, 0.5490196, 0, 0.4470588, 1,
4.444445, 6.393939, -584.7728, 0.5490196, 0, 0.4470588, 1,
4.484848, 6.393939, -584.4495, 0.5490196, 0, 0.4470588, 1,
4.525252, 6.393939, -584.1342, 0.5490196, 0, 0.4470588, 1,
4.565657, 6.393939, -583.8268, 0.654902, 0, 0.3411765, 1,
4.606061, 6.393939, -583.5275, 0.654902, 0, 0.3411765, 1,
4.646465, 6.393939, -583.2361, 0.654902, 0, 0.3411765, 1,
4.686869, 6.393939, -582.9527, 0.654902, 0, 0.3411765, 1,
4.727273, 6.393939, -582.6773, 0.654902, 0, 0.3411765, 1,
4.767677, 6.393939, -582.4099, 0.654902, 0, 0.3411765, 1,
4.808081, 6.393939, -582.1505, 0.654902, 0, 0.3411765, 1,
4.848485, 6.393939, -581.899, 0.654902, 0, 0.3411765, 1,
4.888889, 6.393939, -581.6556, 0.654902, 0, 0.3411765, 1,
4.929293, 6.393939, -581.4202, 0.654902, 0, 0.3411765, 1,
4.969697, 6.393939, -581.1927, 0.654902, 0, 0.3411765, 1,
5.010101, 6.393939, -580.9731, 0.654902, 0, 0.3411765, 1,
5.050505, 6.393939, -580.7617, 0.654902, 0, 0.3411765, 1,
5.090909, 6.393939, -580.5582, 0.654902, 0, 0.3411765, 1,
5.131313, 6.393939, -580.3626, 0.654902, 0, 0.3411765, 1,
5.171717, 6.393939, -580.175, 0.654902, 0, 0.3411765, 1,
5.212121, 6.393939, -579.9955, 0.654902, 0, 0.3411765, 1,
5.252525, 6.393939, -579.8239, 0.654902, 0, 0.3411765, 1,
5.292929, 6.393939, -579.6603, 0.654902, 0, 0.3411765, 1,
5.333333, 6.393939, -579.5048, 0.654902, 0, 0.3411765, 1,
5.373737, 6.393939, -579.3571, 0.654902, 0, 0.3411765, 1,
5.414141, 6.393939, -579.2175, 0.654902, 0, 0.3411765, 1,
5.454545, 6.393939, -579.0859, 0.654902, 0, 0.3411765, 1,
5.494949, 6.393939, -578.9622, 0.654902, 0, 0.3411765, 1,
5.535354, 6.393939, -578.8466, 0.654902, 0, 0.3411765, 1,
5.575758, 6.393939, -578.7389, 0.654902, 0, 0.3411765, 1,
5.616162, 6.393939, -578.6392, 0.654902, 0, 0.3411765, 1,
5.656566, 6.393939, -578.5475, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.393939, -578.4637, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.393939, -578.388, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.393939, -578.3203, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.393939, -578.2605, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.393939, -578.2087, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.393939, -578.1649, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.393939, -578.1291, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.393939, -578.1013, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.393939, -578.0814, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.393939, -578.0696, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.393939, -578.0657, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.393939, -578.0699, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.393939, -578.082, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.393939, -578.1021, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.393939, -578.1301, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.393939, -578.1662, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.393939, -578.2103, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.393939, -578.2623, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.393939, -578.3223, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.393939, -578.3904, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.393939, -578.4664, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.393939, -578.5504, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.393939, -578.6423, 0.654902, 0, 0.3411765, 1,
6.626263, 6.393939, -578.7423, 0.654902, 0, 0.3411765, 1,
6.666667, 6.393939, -578.8502, 0.654902, 0, 0.3411765, 1,
6.707071, 6.393939, -578.9662, 0.654902, 0, 0.3411765, 1,
6.747475, 6.393939, -579.0901, 0.654902, 0, 0.3411765, 1,
6.787879, 6.393939, -579.222, 0.654902, 0, 0.3411765, 1,
6.828283, 6.393939, -579.3619, 0.654902, 0, 0.3411765, 1,
6.868687, 6.393939, -579.5098, 0.654902, 0, 0.3411765, 1,
6.909091, 6.393939, -579.6656, 0.654902, 0, 0.3411765, 1,
6.949495, 6.393939, -579.8295, 0.654902, 0, 0.3411765, 1,
6.989899, 6.393939, -580.0013, 0.654902, 0, 0.3411765, 1,
7.030303, 6.393939, -580.1812, 0.654902, 0, 0.3411765, 1,
7.070707, 6.393939, -580.369, 0.654902, 0, 0.3411765, 1,
7.111111, 6.393939, -580.5648, 0.654902, 0, 0.3411765, 1,
7.151515, 6.393939, -580.7685, 0.654902, 0, 0.3411765, 1,
7.191919, 6.393939, -580.9803, 0.654902, 0, 0.3411765, 1,
7.232323, 6.393939, -581.2, 0.654902, 0, 0.3411765, 1,
7.272727, 6.393939, -581.4278, 0.654902, 0, 0.3411765, 1,
7.313131, 6.393939, -581.6635, 0.654902, 0, 0.3411765, 1,
7.353535, 6.393939, -581.9072, 0.654902, 0, 0.3411765, 1,
7.393939, 6.393939, -582.1589, 0.654902, 0, 0.3411765, 1,
7.434343, 6.393939, -582.4186, 0.654902, 0, 0.3411765, 1,
7.474748, 6.393939, -582.6863, 0.654902, 0, 0.3411765, 1,
7.515152, 6.393939, -582.9619, 0.654902, 0, 0.3411765, 1,
7.555555, 6.393939, -583.2455, 0.654902, 0, 0.3411765, 1,
7.59596, 6.393939, -583.5372, 0.654902, 0, 0.3411765, 1,
7.636364, 6.393939, -583.8368, 0.654902, 0, 0.3411765, 1,
7.676768, 6.393939, -584.1444, 0.5490196, 0, 0.4470588, 1,
7.717172, 6.393939, -584.46, 0.5490196, 0, 0.4470588, 1,
7.757576, 6.393939, -584.7836, 0.5490196, 0, 0.4470588, 1,
7.79798, 6.393939, -585.1151, 0.5490196, 0, 0.4470588, 1,
7.838384, 6.393939, -585.4547, 0.5490196, 0, 0.4470588, 1,
7.878788, 6.393939, -585.8022, 0.5490196, 0, 0.4470588, 1,
7.919192, 6.393939, -586.1577, 0.5490196, 0, 0.4470588, 1,
7.959596, 6.393939, -586.5212, 0.5490196, 0, 0.4470588, 1,
8, 6.393939, -586.8928, 0.5490196, 0, 0.4470588, 1,
4, 6.444445, -589.899, 0.4470588, 0, 0.5490196, 1,
4.040404, 6.444445, -589.4943, 0.4470588, 0, 0.5490196, 1,
4.080808, 6.444445, -589.0975, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.444445, -588.7084, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.444445, -588.3273, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.444445, -587.954, 0.5490196, 0, 0.4470588, 1,
4.242424, 6.444445, -587.5886, 0.5490196, 0, 0.4470588, 1,
4.282828, 6.444445, -587.231, 0.5490196, 0, 0.4470588, 1,
4.323232, 6.444445, -586.8813, 0.5490196, 0, 0.4470588, 1,
4.363636, 6.444445, -586.5394, 0.5490196, 0, 0.4470588, 1,
4.40404, 6.444445, -586.2054, 0.5490196, 0, 0.4470588, 1,
4.444445, 6.444445, -585.8793, 0.5490196, 0, 0.4470588, 1,
4.484848, 6.444445, -585.561, 0.5490196, 0, 0.4470588, 1,
4.525252, 6.444445, -585.2506, 0.5490196, 0, 0.4470588, 1,
4.565657, 6.444445, -584.9481, 0.5490196, 0, 0.4470588, 1,
4.606061, 6.444445, -584.6534, 0.5490196, 0, 0.4470588, 1,
4.646465, 6.444445, -584.3666, 0.5490196, 0, 0.4470588, 1,
4.686869, 6.444445, -584.0876, 0.5490196, 0, 0.4470588, 1,
4.727273, 6.444445, -583.8165, 0.654902, 0, 0.3411765, 1,
4.767677, 6.444445, -583.5533, 0.654902, 0, 0.3411765, 1,
4.808081, 6.444445, -583.298, 0.654902, 0, 0.3411765, 1,
4.848485, 6.444445, -583.0504, 0.654902, 0, 0.3411765, 1,
4.888889, 6.444445, -582.8108, 0.654902, 0, 0.3411765, 1,
4.929293, 6.444445, -582.579, 0.654902, 0, 0.3411765, 1,
4.969697, 6.444445, -582.3551, 0.654902, 0, 0.3411765, 1,
5.010101, 6.444445, -582.139, 0.654902, 0, 0.3411765, 1,
5.050505, 6.444445, -581.9308, 0.654902, 0, 0.3411765, 1,
5.090909, 6.444445, -581.7305, 0.654902, 0, 0.3411765, 1,
5.131313, 6.444445, -581.538, 0.654902, 0, 0.3411765, 1,
5.171717, 6.444445, -581.3534, 0.654902, 0, 0.3411765, 1,
5.212121, 6.444445, -581.1766, 0.654902, 0, 0.3411765, 1,
5.252525, 6.444445, -581.0078, 0.654902, 0, 0.3411765, 1,
5.292929, 6.444445, -580.8467, 0.654902, 0, 0.3411765, 1,
5.333333, 6.444445, -580.6935, 0.654902, 0, 0.3411765, 1,
5.373737, 6.444445, -580.5482, 0.654902, 0, 0.3411765, 1,
5.414141, 6.444445, -580.4108, 0.654902, 0, 0.3411765, 1,
5.454545, 6.444445, -580.2812, 0.654902, 0, 0.3411765, 1,
5.494949, 6.444445, -580.1595, 0.654902, 0, 0.3411765, 1,
5.535354, 6.444445, -580.0456, 0.654902, 0, 0.3411765, 1,
5.575758, 6.444445, -579.9396, 0.654902, 0, 0.3411765, 1,
5.616162, 6.444445, -579.8414, 0.654902, 0, 0.3411765, 1,
5.656566, 6.444445, -579.7512, 0.654902, 0, 0.3411765, 1,
5.69697, 6.444445, -579.6688, 0.654902, 0, 0.3411765, 1,
5.737374, 6.444445, -579.5942, 0.654902, 0, 0.3411765, 1,
5.777778, 6.444445, -579.5275, 0.654902, 0, 0.3411765, 1,
5.818182, 6.444445, -579.4687, 0.654902, 0, 0.3411765, 1,
5.858586, 6.444445, -579.4177, 0.654902, 0, 0.3411765, 1,
5.89899, 6.444445, -579.3746, 0.654902, 0, 0.3411765, 1,
5.939394, 6.444445, -579.3394, 0.654902, 0, 0.3411765, 1,
5.979798, 6.444445, -579.312, 0.654902, 0, 0.3411765, 1,
6.020202, 6.444445, -579.2924, 0.654902, 0, 0.3411765, 1,
6.060606, 6.444445, -579.2808, 0.654902, 0, 0.3411765, 1,
6.10101, 6.444445, -579.277, 0.654902, 0, 0.3411765, 1,
6.141414, 6.444445, -579.2811, 0.654902, 0, 0.3411765, 1,
6.181818, 6.444445, -579.293, 0.654902, 0, 0.3411765, 1,
6.222222, 6.444445, -579.3127, 0.654902, 0, 0.3411765, 1,
6.262626, 6.444445, -579.3404, 0.654902, 0, 0.3411765, 1,
6.30303, 6.444445, -579.3759, 0.654902, 0, 0.3411765, 1,
6.343434, 6.444445, -579.4193, 0.654902, 0, 0.3411765, 1,
6.383838, 6.444445, -579.4705, 0.654902, 0, 0.3411765, 1,
6.424242, 6.444445, -579.5296, 0.654902, 0, 0.3411765, 1,
6.464646, 6.444445, -579.5966, 0.654902, 0, 0.3411765, 1,
6.505051, 6.444445, -579.6713, 0.654902, 0, 0.3411765, 1,
6.545455, 6.444445, -579.754, 0.654902, 0, 0.3411765, 1,
6.585859, 6.444445, -579.8445, 0.654902, 0, 0.3411765, 1,
6.626263, 6.444445, -579.943, 0.654902, 0, 0.3411765, 1,
6.666667, 6.444445, -580.0492, 0.654902, 0, 0.3411765, 1,
6.707071, 6.444445, -580.1633, 0.654902, 0, 0.3411765, 1,
6.747475, 6.444445, -580.2853, 0.654902, 0, 0.3411765, 1,
6.787879, 6.444445, -580.4152, 0.654902, 0, 0.3411765, 1,
6.828283, 6.444445, -580.5529, 0.654902, 0, 0.3411765, 1,
6.868687, 6.444445, -580.6984, 0.654902, 0, 0.3411765, 1,
6.909091, 6.444445, -580.8519, 0.654902, 0, 0.3411765, 1,
6.949495, 6.444445, -581.0132, 0.654902, 0, 0.3411765, 1,
6.989899, 6.444445, -581.1823, 0.654902, 0, 0.3411765, 1,
7.030303, 6.444445, -581.3593, 0.654902, 0, 0.3411765, 1,
7.070707, 6.444445, -581.5442, 0.654902, 0, 0.3411765, 1,
7.111111, 6.444445, -581.7369, 0.654902, 0, 0.3411765, 1,
7.151515, 6.444445, -581.9376, 0.654902, 0, 0.3411765, 1,
7.191919, 6.444445, -582.146, 0.654902, 0, 0.3411765, 1,
7.232323, 6.444445, -582.3624, 0.654902, 0, 0.3411765, 1,
7.272727, 6.444445, -582.5865, 0.654902, 0, 0.3411765, 1,
7.313131, 6.444445, -582.8185, 0.654902, 0, 0.3411765, 1,
7.353535, 6.444445, -583.0585, 0.654902, 0, 0.3411765, 1,
7.393939, 6.444445, -583.3062, 0.654902, 0, 0.3411765, 1,
7.434343, 6.444445, -583.5619, 0.654902, 0, 0.3411765, 1,
7.474748, 6.444445, -583.8254, 0.654902, 0, 0.3411765, 1,
7.515152, 6.444445, -584.0967, 0.5490196, 0, 0.4470588, 1,
7.555555, 6.444445, -584.3759, 0.5490196, 0, 0.4470588, 1,
7.59596, 6.444445, -584.663, 0.5490196, 0, 0.4470588, 1,
7.636364, 6.444445, -584.9579, 0.5490196, 0, 0.4470588, 1,
7.676768, 6.444445, -585.2607, 0.5490196, 0, 0.4470588, 1,
7.717172, 6.444445, -585.5714, 0.5490196, 0, 0.4470588, 1,
7.757576, 6.444445, -585.89, 0.5490196, 0, 0.4470588, 1,
7.79798, 6.444445, -586.2163, 0.5490196, 0, 0.4470588, 1,
7.838384, 6.444445, -586.5505, 0.5490196, 0, 0.4470588, 1,
7.878788, 6.444445, -586.8926, 0.5490196, 0, 0.4470588, 1,
7.919192, 6.444445, -587.2426, 0.5490196, 0, 0.4470588, 1,
7.959596, 6.444445, -587.6005, 0.5490196, 0, 0.4470588, 1,
8, 6.444445, -587.9662, 0.5490196, 0, 0.4470588, 1,
4, 6.494949, -590.9419, 0.4470588, 0, 0.5490196, 1,
4.040404, 6.494949, -590.5434, 0.4470588, 0, 0.5490196, 1,
4.080808, 6.494949, -590.1527, 0.4470588, 0, 0.5490196, 1,
4.121212, 6.494949, -589.7697, 0.4470588, 0, 0.5490196, 1,
4.161616, 6.494949, -589.3944, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.494949, -589.0269, 0.5490196, 0, 0.4470588, 1,
4.242424, 6.494949, -588.6672, 0.5490196, 0, 0.4470588, 1,
4.282828, 6.494949, -588.3151, 0.5490196, 0, 0.4470588, 1,
4.323232, 6.494949, -587.9708, 0.5490196, 0, 0.4470588, 1,
4.363636, 6.494949, -587.6343, 0.5490196, 0, 0.4470588, 1,
4.40404, 6.494949, -587.3055, 0.5490196, 0, 0.4470588, 1,
4.444445, 6.494949, -586.9844, 0.5490196, 0, 0.4470588, 1,
4.484848, 6.494949, -586.6711, 0.5490196, 0, 0.4470588, 1,
4.525252, 6.494949, -586.3655, 0.5490196, 0, 0.4470588, 1,
4.565657, 6.494949, -586.0676, 0.5490196, 0, 0.4470588, 1,
4.606061, 6.494949, -585.7775, 0.5490196, 0, 0.4470588, 1,
4.646465, 6.494949, -585.4951, 0.5490196, 0, 0.4470588, 1,
4.686869, 6.494949, -585.2205, 0.5490196, 0, 0.4470588, 1,
4.727273, 6.494949, -584.9536, 0.5490196, 0, 0.4470588, 1,
4.767677, 6.494949, -584.6945, 0.5490196, 0, 0.4470588, 1,
4.808081, 6.494949, -584.4431, 0.5490196, 0, 0.4470588, 1,
4.848485, 6.494949, -584.1993, 0.5490196, 0, 0.4470588, 1,
4.888889, 6.494949, -583.9634, 0.654902, 0, 0.3411765, 1,
4.929293, 6.494949, -583.7352, 0.654902, 0, 0.3411765, 1,
4.969697, 6.494949, -583.5148, 0.654902, 0, 0.3411765, 1,
5.010101, 6.494949, -583.3021, 0.654902, 0, 0.3411765, 1,
5.050505, 6.494949, -583.097, 0.654902, 0, 0.3411765, 1,
5.090909, 6.494949, -582.8998, 0.654902, 0, 0.3411765, 1,
5.131313, 6.494949, -582.7103, 0.654902, 0, 0.3411765, 1,
5.171717, 6.494949, -582.5286, 0.654902, 0, 0.3411765, 1,
5.212121, 6.494949, -582.3546, 0.654902, 0, 0.3411765, 1,
5.252525, 6.494949, -582.1883, 0.654902, 0, 0.3411765, 1,
5.292929, 6.494949, -582.0297, 0.654902, 0, 0.3411765, 1,
5.333333, 6.494949, -581.879, 0.654902, 0, 0.3411765, 1,
5.373737, 6.494949, -581.7359, 0.654902, 0, 0.3411765, 1,
5.414141, 6.494949, -581.6006, 0.654902, 0, 0.3411765, 1,
5.454545, 6.494949, -581.473, 0.654902, 0, 0.3411765, 1,
5.494949, 6.494949, -581.3531, 0.654902, 0, 0.3411765, 1,
5.535354, 6.494949, -581.2411, 0.654902, 0, 0.3411765, 1,
5.575758, 6.494949, -581.1367, 0.654902, 0, 0.3411765, 1,
5.616162, 6.494949, -581.0401, 0.654902, 0, 0.3411765, 1,
5.656566, 6.494949, -580.9512, 0.654902, 0, 0.3411765, 1,
5.69697, 6.494949, -580.8701, 0.654902, 0, 0.3411765, 1,
5.737374, 6.494949, -580.7967, 0.654902, 0, 0.3411765, 1,
5.777778, 6.494949, -580.731, 0.654902, 0, 0.3411765, 1,
5.818182, 6.494949, -580.6731, 0.654902, 0, 0.3411765, 1,
5.858586, 6.494949, -580.6229, 0.654902, 0, 0.3411765, 1,
5.89899, 6.494949, -580.5804, 0.654902, 0, 0.3411765, 1,
5.939394, 6.494949, -580.5458, 0.654902, 0, 0.3411765, 1,
5.979798, 6.494949, -580.5188, 0.654902, 0, 0.3411765, 1,
6.020202, 6.494949, -580.4996, 0.654902, 0, 0.3411765, 1,
6.060606, 6.494949, -580.4881, 0.654902, 0, 0.3411765, 1,
6.10101, 6.494949, -580.4844, 0.654902, 0, 0.3411765, 1,
6.141414, 6.494949, -580.4883, 0.654902, 0, 0.3411765, 1,
6.181818, 6.494949, -580.5001, 0.654902, 0, 0.3411765, 1,
6.222222, 6.494949, -580.5195, 0.654902, 0, 0.3411765, 1,
6.262626, 6.494949, -580.5468, 0.654902, 0, 0.3411765, 1,
6.30303, 6.494949, -580.5817, 0.654902, 0, 0.3411765, 1,
6.343434, 6.494949, -580.6245, 0.654902, 0, 0.3411765, 1,
6.383838, 6.494949, -580.6749, 0.654902, 0, 0.3411765, 1,
6.424242, 6.494949, -580.733, 0.654902, 0, 0.3411765, 1,
6.464646, 6.494949, -580.799, 0.654902, 0, 0.3411765, 1,
6.505051, 6.494949, -580.8726, 0.654902, 0, 0.3411765, 1,
6.545455, 6.494949, -580.954, 0.654902, 0, 0.3411765, 1,
6.585859, 6.494949, -581.0432, 0.654902, 0, 0.3411765, 1,
6.626263, 6.494949, -581.14, 0.654902, 0, 0.3411765, 1,
6.666667, 6.494949, -581.2446, 0.654902, 0, 0.3411765, 1,
6.707071, 6.494949, -581.357, 0.654902, 0, 0.3411765, 1,
6.747475, 6.494949, -581.4771, 0.654902, 0, 0.3411765, 1,
6.787879, 6.494949, -581.6049, 0.654902, 0, 0.3411765, 1,
6.828283, 6.494949, -581.7405, 0.654902, 0, 0.3411765, 1,
6.868687, 6.494949, -581.8838, 0.654902, 0, 0.3411765, 1,
6.909091, 6.494949, -582.0349, 0.654902, 0, 0.3411765, 1,
6.949495, 6.494949, -582.1937, 0.654902, 0, 0.3411765, 1,
6.989899, 6.494949, -582.3602, 0.654902, 0, 0.3411765, 1,
7.030303, 6.494949, -582.5344, 0.654902, 0, 0.3411765, 1,
7.070707, 6.494949, -582.7165, 0.654902, 0, 0.3411765, 1,
7.111111, 6.494949, -582.9062, 0.654902, 0, 0.3411765, 1,
7.151515, 6.494949, -583.1037, 0.654902, 0, 0.3411765, 1,
7.191919, 6.494949, -583.309, 0.654902, 0, 0.3411765, 1,
7.232323, 6.494949, -583.5219, 0.654902, 0, 0.3411765, 1,
7.272727, 6.494949, -583.7426, 0.654902, 0, 0.3411765, 1,
7.313131, 6.494949, -583.9711, 0.654902, 0, 0.3411765, 1,
7.353535, 6.494949, -584.2073, 0.5490196, 0, 0.4470588, 1,
7.393939, 6.494949, -584.4512, 0.5490196, 0, 0.4470588, 1,
7.434343, 6.494949, -584.7029, 0.5490196, 0, 0.4470588, 1,
7.474748, 6.494949, -584.9623, 0.5490196, 0, 0.4470588, 1,
7.515152, 6.494949, -585.2294, 0.5490196, 0, 0.4470588, 1,
7.555555, 6.494949, -585.5043, 0.5490196, 0, 0.4470588, 1,
7.59596, 6.494949, -585.7869, 0.5490196, 0, 0.4470588, 1,
7.636364, 6.494949, -586.0773, 0.5490196, 0, 0.4470588, 1,
7.676768, 6.494949, -586.3754, 0.5490196, 0, 0.4470588, 1,
7.717172, 6.494949, -586.6813, 0.5490196, 0, 0.4470588, 1,
7.757576, 6.494949, -586.9949, 0.5490196, 0, 0.4470588, 1,
7.79798, 6.494949, -587.3162, 0.5490196, 0, 0.4470588, 1,
7.838384, 6.494949, -587.6453, 0.5490196, 0, 0.4470588, 1,
7.878788, 6.494949, -587.9821, 0.5490196, 0, 0.4470588, 1,
7.919192, 6.494949, -588.3266, 0.5490196, 0, 0.4470588, 1,
7.959596, 6.494949, -588.6789, 0.5490196, 0, 0.4470588, 1,
8, 6.494949, -589.0389, 0.5490196, 0, 0.4470588, 1,
4, 6.545455, -591.9846, 0.4470588, 0, 0.5490196, 1,
4.040404, 6.545455, -591.5922, 0.4470588, 0, 0.5490196, 1,
4.080808, 6.545455, -591.2075, 0.4470588, 0, 0.5490196, 1,
4.121212, 6.545455, -590.8304, 0.4470588, 0, 0.5490196, 1,
4.161616, 6.545455, -590.4609, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.545455, -590.0991, 0.4470588, 0, 0.5490196, 1,
4.242424, 6.545455, -589.7448, 0.4470588, 0, 0.5490196, 1,
4.282828, 6.545455, -589.3982, 0.4470588, 0, 0.5490196, 1,
4.323232, 6.545455, -589.0592, 0.5490196, 0, 0.4470588, 1,
4.363636, 6.545455, -588.7278, 0.5490196, 0, 0.4470588, 1,
4.40404, 6.545455, -588.4041, 0.5490196, 0, 0.4470588, 1,
4.444445, 6.545455, -588.0879, 0.5490196, 0, 0.4470588, 1,
4.484848, 6.545455, -587.7794, 0.5490196, 0, 0.4470588, 1,
4.525252, 6.545455, -587.4785, 0.5490196, 0, 0.4470588, 1,
4.565657, 6.545455, -587.1852, 0.5490196, 0, 0.4470588, 1,
4.606061, 6.545455, -586.8996, 0.5490196, 0, 0.4470588, 1,
4.646465, 6.545455, -586.6215, 0.5490196, 0, 0.4470588, 1,
4.686869, 6.545455, -586.3511, 0.5490196, 0, 0.4470588, 1,
4.727273, 6.545455, -586.0883, 0.5490196, 0, 0.4470588, 1,
4.767677, 6.545455, -585.8332, 0.5490196, 0, 0.4470588, 1,
4.808081, 6.545455, -585.5856, 0.5490196, 0, 0.4470588, 1,
4.848485, 6.545455, -585.3457, 0.5490196, 0, 0.4470588, 1,
4.888889, 6.545455, -585.1134, 0.5490196, 0, 0.4470588, 1,
4.929293, 6.545455, -584.8887, 0.5490196, 0, 0.4470588, 1,
4.969697, 6.545455, -584.6716, 0.5490196, 0, 0.4470588, 1,
5.010101, 6.545455, -584.4622, 0.5490196, 0, 0.4470588, 1,
5.050505, 6.545455, -584.2604, 0.5490196, 0, 0.4470588, 1,
5.090909, 6.545455, -584.0662, 0.5490196, 0, 0.4470588, 1,
5.131313, 6.545455, -583.8796, 0.654902, 0, 0.3411765, 1,
5.171717, 6.545455, -583.7006, 0.654902, 0, 0.3411765, 1,
5.212121, 6.545455, -583.5292, 0.654902, 0, 0.3411765, 1,
5.252525, 6.545455, -583.3655, 0.654902, 0, 0.3411765, 1,
5.292929, 6.545455, -583.2094, 0.654902, 0, 0.3411765, 1,
5.333333, 6.545455, -583.061, 0.654902, 0, 0.3411765, 1,
5.373737, 6.545455, -582.9201, 0.654902, 0, 0.3411765, 1,
5.414141, 6.545455, -582.7869, 0.654902, 0, 0.3411765, 1,
5.454545, 6.545455, -582.6613, 0.654902, 0, 0.3411765, 1,
5.494949, 6.545455, -582.5433, 0.654902, 0, 0.3411765, 1,
5.535354, 6.545455, -582.4329, 0.654902, 0, 0.3411765, 1,
5.575758, 6.545455, -582.3301, 0.654902, 0, 0.3411765, 1,
5.616162, 6.545455, -582.235, 0.654902, 0, 0.3411765, 1,
5.656566, 6.545455, -582.1475, 0.654902, 0, 0.3411765, 1,
5.69697, 6.545455, -582.0676, 0.654902, 0, 0.3411765, 1,
5.737374, 6.545455, -581.9953, 0.654902, 0, 0.3411765, 1,
5.777778, 6.545455, -581.9307, 0.654902, 0, 0.3411765, 1,
5.818182, 6.545455, -581.8736, 0.654902, 0, 0.3411765, 1,
5.858586, 6.545455, -581.8242, 0.654902, 0, 0.3411765, 1,
5.89899, 6.545455, -581.7824, 0.654902, 0, 0.3411765, 1,
5.939394, 6.545455, -581.7482, 0.654902, 0, 0.3411765, 1,
5.979798, 6.545455, -581.7217, 0.654902, 0, 0.3411765, 1,
6.020202, 6.545455, -581.7028, 0.654902, 0, 0.3411765, 1,
6.060606, 6.545455, -581.6915, 0.654902, 0, 0.3411765, 1,
6.10101, 6.545455, -581.6878, 0.654902, 0, 0.3411765, 1,
6.141414, 6.545455, -581.6917, 0.654902, 0, 0.3411765, 1,
6.181818, 6.545455, -581.7033, 0.654902, 0, 0.3411765, 1,
6.222222, 6.545455, -581.7225, 0.654902, 0, 0.3411765, 1,
6.262626, 6.545455, -581.7493, 0.654902, 0, 0.3411765, 1,
6.30303, 6.545455, -581.7837, 0.654902, 0, 0.3411765, 1,
6.343434, 6.545455, -581.8257, 0.654902, 0, 0.3411765, 1,
6.383838, 6.545455, -581.8754, 0.654902, 0, 0.3411765, 1,
6.424242, 6.545455, -581.9327, 0.654902, 0, 0.3411765, 1,
6.464646, 6.545455, -581.9976, 0.654902, 0, 0.3411765, 1,
6.505051, 6.545455, -582.0701, 0.654902, 0, 0.3411765, 1,
6.545455, 6.545455, -582.1502, 0.654902, 0, 0.3411765, 1,
6.585859, 6.545455, -582.238, 0.654902, 0, 0.3411765, 1,
6.626263, 6.545455, -582.3334, 0.654902, 0, 0.3411765, 1,
6.666667, 6.545455, -582.4364, 0.654902, 0, 0.3411765, 1,
6.707071, 6.545455, -582.547, 0.654902, 0, 0.3411765, 1,
6.747475, 6.545455, -582.6653, 0.654902, 0, 0.3411765, 1,
6.787879, 6.545455, -582.7911, 0.654902, 0, 0.3411765, 1,
6.828283, 6.545455, -582.9246, 0.654902, 0, 0.3411765, 1,
6.868687, 6.545455, -583.0657, 0.654902, 0, 0.3411765, 1,
6.909091, 6.545455, -583.2145, 0.654902, 0, 0.3411765, 1,
6.949495, 6.545455, -583.3708, 0.654902, 0, 0.3411765, 1,
6.989899, 6.545455, -583.5348, 0.654902, 0, 0.3411765, 1,
7.030303, 6.545455, -583.7064, 0.654902, 0, 0.3411765, 1,
7.070707, 6.545455, -583.8856, 0.654902, 0, 0.3411765, 1,
7.111111, 6.545455, -584.0724, 0.5490196, 0, 0.4470588, 1,
7.151515, 6.545455, -584.2669, 0.5490196, 0, 0.4470588, 1,
7.191919, 6.545455, -584.4689, 0.5490196, 0, 0.4470588, 1,
7.232323, 6.545455, -584.6786, 0.5490196, 0, 0.4470588, 1,
7.272727, 6.545455, -584.896, 0.5490196, 0, 0.4470588, 1,
7.313131, 6.545455, -585.1209, 0.5490196, 0, 0.4470588, 1,
7.353535, 6.545455, -585.3535, 0.5490196, 0, 0.4470588, 1,
7.393939, 6.545455, -585.5936, 0.5490196, 0, 0.4470588, 1,
7.434343, 6.545455, -585.8414, 0.5490196, 0, 0.4470588, 1,
7.474748, 6.545455, -586.0969, 0.5490196, 0, 0.4470588, 1,
7.515152, 6.545455, -586.3599, 0.5490196, 0, 0.4470588, 1,
7.555555, 6.545455, -586.6306, 0.5490196, 0, 0.4470588, 1,
7.59596, 6.545455, -586.9089, 0.5490196, 0, 0.4470588, 1,
7.636364, 6.545455, -587.1948, 0.5490196, 0, 0.4470588, 1,
7.676768, 6.545455, -587.4883, 0.5490196, 0, 0.4470588, 1,
7.717172, 6.545455, -587.7894, 0.5490196, 0, 0.4470588, 1,
7.757576, 6.545455, -588.0982, 0.5490196, 0, 0.4470588, 1,
7.79798, 6.545455, -588.4146, 0.5490196, 0, 0.4470588, 1,
7.838384, 6.545455, -588.7386, 0.5490196, 0, 0.4470588, 1,
7.878788, 6.545455, -589.0703, 0.5490196, 0, 0.4470588, 1,
7.919192, 6.545455, -589.4095, 0.4470588, 0, 0.5490196, 1,
7.959596, 6.545455, -589.7563, 0.4470588, 0, 0.5490196, 1,
8, 6.545455, -590.1108, 0.4470588, 0, 0.5490196, 1,
4, 6.59596, -593.0269, 0.4470588, 0, 0.5490196, 1,
4.040404, 6.59596, -592.6406, 0.4470588, 0, 0.5490196, 1,
4.080808, 6.59596, -592.2617, 0.4470588, 0, 0.5490196, 1,
4.121212, 6.59596, -591.8904, 0.4470588, 0, 0.5490196, 1,
4.161616, 6.59596, -591.5265, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.59596, -591.1702, 0.4470588, 0, 0.5490196, 1,
4.242424, 6.59596, -590.8214, 0.4470588, 0, 0.5490196, 1,
4.282828, 6.59596, -590.48, 0.4470588, 0, 0.5490196, 1,
4.323232, 6.59596, -590.1462, 0.4470588, 0, 0.5490196, 1,
4.363636, 6.59596, -589.8198, 0.4470588, 0, 0.5490196, 1,
4.40404, 6.59596, -589.501, 0.4470588, 0, 0.5490196, 1,
4.444445, 6.59596, -589.1897, 0.5490196, 0, 0.4470588, 1,
4.484848, 6.59596, -588.8859, 0.5490196, 0, 0.4470588, 1,
4.525252, 6.59596, -588.5896, 0.5490196, 0, 0.4470588, 1,
4.565657, 6.59596, -588.3008, 0.5490196, 0, 0.4470588, 1,
4.606061, 6.59596, -588.0195, 0.5490196, 0, 0.4470588, 1,
4.646465, 6.59596, -587.7457, 0.5490196, 0, 0.4470588, 1,
4.686869, 6.59596, -587.4794, 0.5490196, 0, 0.4470588, 1,
4.727273, 6.59596, -587.2206, 0.5490196, 0, 0.4470588, 1,
4.767677, 6.59596, -586.9694, 0.5490196, 0, 0.4470588, 1,
4.808081, 6.59596, -586.7256, 0.5490196, 0, 0.4470588, 1,
4.848485, 6.59596, -586.4893, 0.5490196, 0, 0.4470588, 1,
4.888889, 6.59596, -586.2606, 0.5490196, 0, 0.4470588, 1,
4.929293, 6.59596, -586.0393, 0.5490196, 0, 0.4470588, 1,
4.969697, 6.59596, -585.8256, 0.5490196, 0, 0.4470588, 1,
5.010101, 6.59596, -585.6193, 0.5490196, 0, 0.4470588, 1,
5.050505, 6.59596, -585.4205, 0.5490196, 0, 0.4470588, 1,
5.090909, 6.59596, -585.2293, 0.5490196, 0, 0.4470588, 1,
5.131313, 6.59596, -585.0456, 0.5490196, 0, 0.4470588, 1,
5.171717, 6.59596, -584.8693, 0.5490196, 0, 0.4470588, 1,
5.212121, 6.59596, -584.7006, 0.5490196, 0, 0.4470588, 1,
5.252525, 6.59596, -584.5394, 0.5490196, 0, 0.4470588, 1,
5.292929, 6.59596, -584.3857, 0.5490196, 0, 0.4470588, 1,
5.333333, 6.59596, -584.2394, 0.5490196, 0, 0.4470588, 1,
5.373737, 6.59596, -584.1008, 0.5490196, 0, 0.4470588, 1,
5.414141, 6.59596, -583.9695, 0.654902, 0, 0.3411765, 1,
5.454545, 6.59596, -583.8458, 0.654902, 0, 0.3411765, 1,
5.494949, 6.59596, -583.7296, 0.654902, 0, 0.3411765, 1,
5.535354, 6.59596, -583.621, 0.654902, 0, 0.3411765, 1,
5.575758, 6.59596, -583.5198, 0.654902, 0, 0.3411765, 1,
5.616162, 6.59596, -583.4261, 0.654902, 0, 0.3411765, 1,
5.656566, 6.59596, -583.3399, 0.654902, 0, 0.3411765, 1,
5.69697, 6.59596, -583.2612, 0.654902, 0, 0.3411765, 1,
5.737374, 6.59596, -583.1901, 0.654902, 0, 0.3411765, 1,
5.777778, 6.59596, -583.1264, 0.654902, 0, 0.3411765, 1,
5.818182, 6.59596, -583.0703, 0.654902, 0, 0.3411765, 1,
5.858586, 6.59596, -583.0215, 0.654902, 0, 0.3411765, 1,
5.89899, 6.59596, -582.9804, 0.654902, 0, 0.3411765, 1,
5.939394, 6.59596, -582.9468, 0.654902, 0, 0.3411765, 1,
5.979798, 6.59596, -582.9207, 0.654902, 0, 0.3411765, 1,
6.020202, 6.59596, -582.902, 0.654902, 0, 0.3411765, 1,
6.060606, 6.59596, -582.8909, 0.654902, 0, 0.3411765, 1,
6.10101, 6.59596, -582.8872, 0.654902, 0, 0.3411765, 1,
6.141414, 6.59596, -582.8911, 0.654902, 0, 0.3411765, 1,
6.181818, 6.59596, -582.9025, 0.654902, 0, 0.3411765, 1,
6.222222, 6.59596, -582.9214, 0.654902, 0, 0.3411765, 1,
6.262626, 6.59596, -582.9478, 0.654902, 0, 0.3411765, 1,
6.30303, 6.59596, -582.9816, 0.654902, 0, 0.3411765, 1,
6.343434, 6.59596, -583.0231, 0.654902, 0, 0.3411765, 1,
6.383838, 6.59596, -583.072, 0.654902, 0, 0.3411765, 1,
6.424242, 6.59596, -583.1284, 0.654902, 0, 0.3411765, 1,
6.464646, 6.59596, -583.1923, 0.654902, 0, 0.3411765, 1,
6.505051, 6.59596, -583.2637, 0.654902, 0, 0.3411765, 1,
6.545455, 6.59596, -583.3426, 0.654902, 0, 0.3411765, 1,
6.585859, 6.59596, -583.429, 0.654902, 0, 0.3411765, 1,
6.626263, 6.59596, -583.5229, 0.654902, 0, 0.3411765, 1,
6.666667, 6.59596, -583.6244, 0.654902, 0, 0.3411765, 1,
6.707071, 6.59596, -583.7333, 0.654902, 0, 0.3411765, 1,
6.747475, 6.59596, -583.8498, 0.654902, 0, 0.3411765, 1,
6.787879, 6.59596, -583.9738, 0.654902, 0, 0.3411765, 1,
6.828283, 6.59596, -584.1052, 0.5490196, 0, 0.4470588, 1,
6.868687, 6.59596, -584.2441, 0.5490196, 0, 0.4470588, 1,
6.909091, 6.59596, -584.3906, 0.5490196, 0, 0.4470588, 1,
6.949495, 6.59596, -584.5446, 0.5490196, 0, 0.4470588, 1,
6.989899, 6.59596, -584.7061, 0.5490196, 0, 0.4470588, 1,
7.030303, 6.59596, -584.875, 0.5490196, 0, 0.4470588, 1,
7.070707, 6.59596, -585.0515, 0.5490196, 0, 0.4470588, 1,
7.111111, 6.59596, -585.2355, 0.5490196, 0, 0.4470588, 1,
7.151515, 6.59596, -585.427, 0.5490196, 0, 0.4470588, 1,
7.191919, 6.59596, -585.626, 0.5490196, 0, 0.4470588, 1,
7.232323, 6.59596, -585.8325, 0.5490196, 0, 0.4470588, 1,
7.272727, 6.59596, -586.0464, 0.5490196, 0, 0.4470588, 1,
7.313131, 6.59596, -586.268, 0.5490196, 0, 0.4470588, 1,
7.353535, 6.59596, -586.497, 0.5490196, 0, 0.4470588, 1,
7.393939, 6.59596, -586.7335, 0.5490196, 0, 0.4470588, 1,
7.434343, 6.59596, -586.9775, 0.5490196, 0, 0.4470588, 1,
7.474748, 6.59596, -587.2291, 0.5490196, 0, 0.4470588, 1,
7.515152, 6.59596, -587.4881, 0.5490196, 0, 0.4470588, 1,
7.555555, 6.59596, -587.7546, 0.5490196, 0, 0.4470588, 1,
7.59596, 6.59596, -588.0287, 0.5490196, 0, 0.4470588, 1,
7.636364, 6.59596, -588.3102, 0.5490196, 0, 0.4470588, 1,
7.676768, 6.59596, -588.5992, 0.5490196, 0, 0.4470588, 1,
7.717172, 6.59596, -588.8958, 0.5490196, 0, 0.4470588, 1,
7.757576, 6.59596, -589.1999, 0.5490196, 0, 0.4470588, 1,
7.79798, 6.59596, -589.5114, 0.4470588, 0, 0.5490196, 1,
7.838384, 6.59596, -589.8305, 0.4470588, 0, 0.5490196, 1,
7.878788, 6.59596, -590.157, 0.4470588, 0, 0.5490196, 1,
7.919192, 6.59596, -590.4911, 0.4470588, 0, 0.5490196, 1,
7.959596, 6.59596, -590.8327, 0.4470588, 0, 0.5490196, 1,
8, 6.59596, -591.1818, 0.4470588, 0, 0.5490196, 1,
4, 6.646465, -594.0688, 0.4470588, 0, 0.5490196, 1,
4.040404, 6.646465, -593.6883, 0.4470588, 0, 0.5490196, 1,
4.080808, 6.646465, -593.3152, 0.4470588, 0, 0.5490196, 1,
4.121212, 6.646465, -592.9495, 0.4470588, 0, 0.5490196, 1,
4.161616, 6.646465, -592.5911, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.646465, -592.2402, 0.4470588, 0, 0.5490196, 1,
4.242424, 6.646465, -591.8966, 0.4470588, 0, 0.5490196, 1,
4.282828, 6.646465, -591.5604, 0.4470588, 0, 0.5490196, 1,
4.323232, 6.646465, -591.2317, 0.4470588, 0, 0.5490196, 1,
4.363636, 6.646465, -590.9103, 0.4470588, 0, 0.5490196, 1,
4.40404, 6.646465, -590.5963, 0.4470588, 0, 0.5490196, 1,
4.444445, 6.646465, -590.2897, 0.4470588, 0, 0.5490196, 1,
4.484848, 6.646465, -589.9905, 0.4470588, 0, 0.5490196, 1,
4.525252, 6.646465, -589.6987, 0.4470588, 0, 0.5490196, 1,
4.565657, 6.646465, -589.4142, 0.4470588, 0, 0.5490196, 1,
4.606061, 6.646465, -589.1372, 0.5490196, 0, 0.4470588, 1,
4.646465, 6.646465, -588.8676, 0.5490196, 0, 0.4470588, 1,
4.686869, 6.646465, -588.6053, 0.5490196, 0, 0.4470588, 1,
4.727273, 6.646465, -588.3504, 0.5490196, 0, 0.4470588, 1,
4.767677, 6.646465, -588.103, 0.5490196, 0, 0.4470588, 1,
4.808081, 6.646465, -587.8629, 0.5490196, 0, 0.4470588, 1,
4.848485, 6.646465, -587.6302, 0.5490196, 0, 0.4470588, 1,
4.888889, 6.646465, -587.4049, 0.5490196, 0, 0.4470588, 1,
4.929293, 6.646465, -587.187, 0.5490196, 0, 0.4470588, 1,
4.969697, 6.646465, -586.9764, 0.5490196, 0, 0.4470588, 1,
5.010101, 6.646465, -586.7733, 0.5490196, 0, 0.4470588, 1,
5.050505, 6.646465, -586.5776, 0.5490196, 0, 0.4470588, 1,
5.090909, 6.646465, -586.3892, 0.5490196, 0, 0.4470588, 1,
5.131313, 6.646465, -586.2083, 0.5490196, 0, 0.4470588, 1,
5.171717, 6.646465, -586.0347, 0.5490196, 0, 0.4470588, 1,
5.212121, 6.646465, -585.8685, 0.5490196, 0, 0.4470588, 1,
5.252525, 6.646465, -585.7098, 0.5490196, 0, 0.4470588, 1,
5.292929, 6.646465, -585.5583, 0.5490196, 0, 0.4470588, 1,
5.333333, 6.646465, -585.4144, 0.5490196, 0, 0.4470588, 1,
5.373737, 6.646465, -585.2778, 0.5490196, 0, 0.4470588, 1,
5.414141, 6.646465, -585.1486, 0.5490196, 0, 0.4470588, 1,
5.454545, 6.646465, -585.0267, 0.5490196, 0, 0.4470588, 1,
5.494949, 6.646465, -584.9123, 0.5490196, 0, 0.4470588, 1,
5.535354, 6.646465, -584.8052, 0.5490196, 0, 0.4470588, 1,
5.575758, 6.646465, -584.7056, 0.5490196, 0, 0.4470588, 1,
5.616162, 6.646465, -584.6133, 0.5490196, 0, 0.4470588, 1,
5.656566, 6.646465, -584.5284, 0.5490196, 0, 0.4470588, 1,
5.69697, 6.646465, -584.4509, 0.5490196, 0, 0.4470588, 1,
5.737374, 6.646465, -584.3809, 0.5490196, 0, 0.4470588, 1,
5.777778, 6.646465, -584.3182, 0.5490196, 0, 0.4470588, 1,
5.818182, 6.646465, -584.2628, 0.5490196, 0, 0.4470588, 1,
5.858586, 6.646465, -584.2149, 0.5490196, 0, 0.4470588, 1,
5.89899, 6.646465, -584.1744, 0.5490196, 0, 0.4470588, 1,
5.939394, 6.646465, -584.1412, 0.5490196, 0, 0.4470588, 1,
5.979798, 6.646465, -584.1155, 0.5490196, 0, 0.4470588, 1,
6.020202, 6.646465, -584.0972, 0.5490196, 0, 0.4470588, 1,
6.060606, 6.646465, -584.0862, 0.5490196, 0, 0.4470588, 1,
6.10101, 6.646465, -584.0826, 0.5490196, 0, 0.4470588, 1,
6.141414, 6.646465, -584.0864, 0.5490196, 0, 0.4470588, 1,
6.181818, 6.646465, -584.0977, 0.5490196, 0, 0.4470588, 1,
6.222222, 6.646465, -584.1163, 0.5490196, 0, 0.4470588, 1,
6.262626, 6.646465, -584.1422, 0.5490196, 0, 0.4470588, 1,
6.30303, 6.646465, -584.1756, 0.5490196, 0, 0.4470588, 1,
6.343434, 6.646465, -584.2164, 0.5490196, 0, 0.4470588, 1,
6.383838, 6.646465, -584.2645, 0.5490196, 0, 0.4470588, 1,
6.424242, 6.646465, -584.3201, 0.5490196, 0, 0.4470588, 1,
6.464646, 6.646465, -584.3831, 0.5490196, 0, 0.4470588, 1,
6.505051, 6.646465, -584.4534, 0.5490196, 0, 0.4470588, 1,
6.545455, 6.646465, -584.5311, 0.5490196, 0, 0.4470588, 1,
6.585859, 6.646465, -584.6162, 0.5490196, 0, 0.4470588, 1,
6.626263, 6.646465, -584.7087, 0.5490196, 0, 0.4470588, 1,
6.666667, 6.646465, -584.8087, 0.5490196, 0, 0.4470588, 1,
6.707071, 6.646465, -584.916, 0.5490196, 0, 0.4470588, 1,
6.747475, 6.646465, -585.0306, 0.5490196, 0, 0.4470588, 1,
6.787879, 6.646465, -585.1527, 0.5490196, 0, 0.4470588, 1,
6.828283, 6.646465, -585.2822, 0.5490196, 0, 0.4470588, 1,
6.868687, 6.646465, -585.419, 0.5490196, 0, 0.4470588, 1,
6.909091, 6.646465, -585.5632, 0.5490196, 0, 0.4470588, 1,
6.949495, 6.646465, -585.7149, 0.5490196, 0, 0.4470588, 1,
6.989899, 6.646465, -585.8739, 0.5490196, 0, 0.4470588, 1,
7.030303, 6.646465, -586.0403, 0.5490196, 0, 0.4470588, 1,
7.070707, 6.646465, -586.2141, 0.5490196, 0, 0.4470588, 1,
7.111111, 6.646465, -586.3953, 0.5490196, 0, 0.4470588, 1,
7.151515, 6.646465, -586.5839, 0.5490196, 0, 0.4470588, 1,
7.191919, 6.646465, -586.7799, 0.5490196, 0, 0.4470588, 1,
7.232323, 6.646465, -586.9833, 0.5490196, 0, 0.4470588, 1,
7.272727, 6.646465, -587.194, 0.5490196, 0, 0.4470588, 1,
7.313131, 6.646465, -587.4122, 0.5490196, 0, 0.4470588, 1,
7.353535, 6.646465, -587.6378, 0.5490196, 0, 0.4470588, 1,
7.393939, 6.646465, -587.8707, 0.5490196, 0, 0.4470588, 1,
7.434343, 6.646465, -588.111, 0.5490196, 0, 0.4470588, 1,
7.474748, 6.646465, -588.3587, 0.5490196, 0, 0.4470588, 1,
7.515152, 6.646465, -588.6138, 0.5490196, 0, 0.4470588, 1,
7.555555, 6.646465, -588.8763, 0.5490196, 0, 0.4470588, 1,
7.59596, 6.646465, -589.1462, 0.5490196, 0, 0.4470588, 1,
7.636364, 6.646465, -589.4235, 0.4470588, 0, 0.5490196, 1,
7.676768, 6.646465, -589.7082, 0.4470588, 0, 0.5490196, 1,
7.717172, 6.646465, -590.0002, 0.4470588, 0, 0.5490196, 1,
7.757576, 6.646465, -590.2997, 0.4470588, 0, 0.5490196, 1,
7.79798, 6.646465, -590.6065, 0.4470588, 0, 0.5490196, 1,
7.838384, 6.646465, -590.9208, 0.4470588, 0, 0.5490196, 1,
7.878788, 6.646465, -591.2424, 0.4470588, 0, 0.5490196, 1,
7.919192, 6.646465, -591.5714, 0.4470588, 0, 0.5490196, 1,
7.959596, 6.646465, -591.9078, 0.4470588, 0, 0.5490196, 1,
8, 6.646465, -592.2516, 0.4470588, 0, 0.5490196, 1,
4, 6.69697, -595.11, 0.3411765, 0, 0.654902, 1,
4.040404, 6.69697, -594.7352, 0.3411765, 0, 0.654902, 1,
4.080808, 6.69697, -594.3677, 0.4470588, 0, 0.5490196, 1,
4.121212, 6.69697, -594.0075, 0.4470588, 0, 0.5490196, 1,
4.161616, 6.69697, -593.6545, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.69697, -593.3089, 0.4470588, 0, 0.5490196, 1,
4.242424, 6.69697, -592.9705, 0.4470588, 0, 0.5490196, 1,
4.282828, 6.69697, -592.6393, 0.4470588, 0, 0.5490196, 1,
4.323232, 6.69697, -592.3156, 0.4470588, 0, 0.5490196, 1,
4.363636, 6.69697, -591.999, 0.4470588, 0, 0.5490196, 1,
4.40404, 6.69697, -591.6897, 0.4470588, 0, 0.5490196, 1,
4.444445, 6.69697, -591.3877, 0.4470588, 0, 0.5490196, 1,
4.484848, 6.69697, -591.093, 0.4470588, 0, 0.5490196, 1,
4.525252, 6.69697, -590.8055, 0.4470588, 0, 0.5490196, 1,
4.565657, 6.69697, -590.5254, 0.4470588, 0, 0.5490196, 1,
4.606061, 6.69697, -590.2525, 0.4470588, 0, 0.5490196, 1,
4.646465, 6.69697, -589.9869, 0.4470588, 0, 0.5490196, 1,
4.686869, 6.69697, -589.7286, 0.4470588, 0, 0.5490196, 1,
4.727273, 6.69697, -589.4776, 0.4470588, 0, 0.5490196, 1,
4.767677, 6.69697, -589.2338, 0.5490196, 0, 0.4470588, 1,
4.808081, 6.69697, -588.9974, 0.5490196, 0, 0.4470588, 1,
4.848485, 6.69697, -588.7681, 0.5490196, 0, 0.4470588, 1,
4.888889, 6.69697, -588.5463, 0.5490196, 0, 0.4470588, 1,
4.929293, 6.69697, -588.3316, 0.5490196, 0, 0.4470588, 1,
4.969697, 6.69697, -588.1243, 0.5490196, 0, 0.4470588, 1,
5.010101, 6.69697, -587.9242, 0.5490196, 0, 0.4470588, 1,
5.050505, 6.69697, -587.7314, 0.5490196, 0, 0.4470588, 1,
5.090909, 6.69697, -587.5458, 0.5490196, 0, 0.4470588, 1,
5.131313, 6.69697, -587.3676, 0.5490196, 0, 0.4470588, 1,
5.171717, 6.69697, -587.1967, 0.5490196, 0, 0.4470588, 1,
5.212121, 6.69697, -587.033, 0.5490196, 0, 0.4470588, 1,
5.252525, 6.69697, -586.8766, 0.5490196, 0, 0.4470588, 1,
5.292929, 6.69697, -586.7275, 0.5490196, 0, 0.4470588, 1,
5.333333, 6.69697, -586.5856, 0.5490196, 0, 0.4470588, 1,
5.373737, 6.69697, -586.4511, 0.5490196, 0, 0.4470588, 1,
5.414141, 6.69697, -586.3238, 0.5490196, 0, 0.4470588, 1,
5.454545, 6.69697, -586.2038, 0.5490196, 0, 0.4470588, 1,
5.494949, 6.69697, -586.0911, 0.5490196, 0, 0.4470588, 1,
5.535354, 6.69697, -585.9857, 0.5490196, 0, 0.4470588, 1,
5.575758, 6.69697, -585.8875, 0.5490196, 0, 0.4470588, 1,
5.616162, 6.69697, -585.7966, 0.5490196, 0, 0.4470588, 1,
5.656566, 6.69697, -585.713, 0.5490196, 0, 0.4470588, 1,
5.69697, 6.69697, -585.6367, 0.5490196, 0, 0.4470588, 1,
5.737374, 6.69697, -585.5677, 0.5490196, 0, 0.4470588, 1,
5.777778, 6.69697, -585.5059, 0.5490196, 0, 0.4470588, 1,
5.818182, 6.69697, -585.4514, 0.5490196, 0, 0.4470588, 1,
5.858586, 6.69697, -585.4042, 0.5490196, 0, 0.4470588, 1,
5.89899, 6.69697, -585.3643, 0.5490196, 0, 0.4470588, 1,
5.939394, 6.69697, -585.3317, 0.5490196, 0, 0.4470588, 1,
5.979798, 6.69697, -585.3063, 0.5490196, 0, 0.4470588, 1,
6.020202, 6.69697, -585.2882, 0.5490196, 0, 0.4470588, 1,
6.060606, 6.69697, -585.2774, 0.5490196, 0, 0.4470588, 1,
6.10101, 6.69697, -585.2739, 0.5490196, 0, 0.4470588, 1,
6.141414, 6.69697, -585.2776, 0.5490196, 0, 0.4470588, 1,
6.181818, 6.69697, -585.2887, 0.5490196, 0, 0.4470588, 1,
6.222222, 6.69697, -585.307, 0.5490196, 0, 0.4470588, 1,
6.262626, 6.69697, -585.3326, 0.5490196, 0, 0.4470588, 1,
6.30303, 6.69697, -585.3655, 0.5490196, 0, 0.4470588, 1,
6.343434, 6.69697, -585.4056, 0.5490196, 0, 0.4470588, 1,
6.383838, 6.69697, -585.4531, 0.5490196, 0, 0.4470588, 1,
6.424242, 6.69697, -585.5078, 0.5490196, 0, 0.4470588, 1,
6.464646, 6.69697, -585.5698, 0.5490196, 0, 0.4470588, 1,
6.505051, 6.69697, -585.6391, 0.5490196, 0, 0.4470588, 1,
6.545455, 6.69697, -585.7156, 0.5490196, 0, 0.4470588, 1,
6.585859, 6.69697, -585.7995, 0.5490196, 0, 0.4470588, 1,
6.626263, 6.69697, -585.8906, 0.5490196, 0, 0.4470588, 1,
6.666667, 6.69697, -585.989, 0.5490196, 0, 0.4470588, 1,
6.707071, 6.69697, -586.0947, 0.5490196, 0, 0.4470588, 1,
6.747475, 6.69697, -586.2076, 0.5490196, 0, 0.4470588, 1,
6.787879, 6.69697, -586.3279, 0.5490196, 0, 0.4470588, 1,
6.828283, 6.69697, -586.4554, 0.5490196, 0, 0.4470588, 1,
6.868687, 6.69697, -586.5902, 0.5490196, 0, 0.4470588, 1,
6.909091, 6.69697, -586.7323, 0.5490196, 0, 0.4470588, 1,
6.949495, 6.69697, -586.8817, 0.5490196, 0, 0.4470588, 1,
6.989899, 6.69697, -587.0383, 0.5490196, 0, 0.4470588, 1,
7.030303, 6.69697, -587.2022, 0.5490196, 0, 0.4470588, 1,
7.070707, 6.69697, -587.3734, 0.5490196, 0, 0.4470588, 1,
7.111111, 6.69697, -587.5519, 0.5490196, 0, 0.4470588, 1,
7.151515, 6.69697, -587.7376, 0.5490196, 0, 0.4470588, 1,
7.191919, 6.69697, -587.9307, 0.5490196, 0, 0.4470588, 1,
7.232323, 6.69697, -588.131, 0.5490196, 0, 0.4470588, 1,
7.272727, 6.69697, -588.3386, 0.5490196, 0, 0.4470588, 1,
7.313131, 6.69697, -588.5535, 0.5490196, 0, 0.4470588, 1,
7.353535, 6.69697, -588.7756, 0.5490196, 0, 0.4470588, 1,
7.393939, 6.69697, -589.005, 0.5490196, 0, 0.4470588, 1,
7.434343, 6.69697, -589.2418, 0.5490196, 0, 0.4470588, 1,
7.474748, 6.69697, -589.4857, 0.4470588, 0, 0.5490196, 1,
7.515152, 6.69697, -589.737, 0.4470588, 0, 0.5490196, 1,
7.555555, 6.69697, -589.9955, 0.4470588, 0, 0.5490196, 1,
7.59596, 6.69697, -590.2614, 0.4470588, 0, 0.5490196, 1,
7.636364, 6.69697, -590.5345, 0.4470588, 0, 0.5490196, 1,
7.676768, 6.69697, -590.8149, 0.4470588, 0, 0.5490196, 1,
7.717172, 6.69697, -591.1026, 0.4470588, 0, 0.5490196, 1,
7.757576, 6.69697, -591.3975, 0.4470588, 0, 0.5490196, 1,
7.79798, 6.69697, -591.6998, 0.4470588, 0, 0.5490196, 1,
7.838384, 6.69697, -592.0093, 0.4470588, 0, 0.5490196, 1,
7.878788, 6.69697, -592.3261, 0.4470588, 0, 0.5490196, 1,
7.919192, 6.69697, -592.6501, 0.4470588, 0, 0.5490196, 1,
7.959596, 6.69697, -592.9815, 0.4470588, 0, 0.5490196, 1,
8, 6.69697, -593.3201, 0.4470588, 0, 0.5490196, 1,
4, 6.747475, -596.1505, 0.3411765, 0, 0.654902, 1,
4.040404, 6.747475, -595.7812, 0.3411765, 0, 0.654902, 1,
4.080808, 6.747475, -595.4193, 0.3411765, 0, 0.654902, 1,
4.121212, 6.747475, -595.0644, 0.3411765, 0, 0.654902, 1,
4.161616, 6.747475, -594.7167, 0.4470588, 0, 0.5490196, 1,
4.20202, 6.747475, -594.3762, 0.4470588, 0, 0.5490196, 1,
4.242424, 6.747475, -594.0428, 0.4470588, 0, 0.5490196, 1,
4.282828, 6.747475, -593.7166, 0.4470588, 0, 0.5490196, 1,
4.323232, 6.747475, -593.3976, 0.4470588, 0, 0.5490196, 1,
4.363636, 6.747475, -593.0858, 0.4470588, 0, 0.5490196, 1,
4.40404, 6.747475, -592.7811, 0.4470588, 0, 0.5490196, 1,
4.444445, 6.747475, -592.4836, 0.4470588, 0, 0.5490196, 1,
4.484848, 6.747475, -592.1934, 0.4470588, 0, 0.5490196, 1,
4.525252, 6.747475, -591.9102, 0.4470588, 0, 0.5490196, 1,
4.565657, 6.747475, -591.6342, 0.4470588, 0, 0.5490196, 1,
4.606061, 6.747475, -591.3654, 0.4470588, 0, 0.5490196, 1,
4.646465, 6.747475, -591.1038, 0.4470588, 0, 0.5490196, 1,
4.686869, 6.747475, -590.8493, 0.4470588, 0, 0.5490196, 1,
4.727273, 6.747475, -590.602, 0.4470588, 0, 0.5490196, 1,
4.767677, 6.747475, -590.3619, 0.4470588, 0, 0.5490196, 1,
4.808081, 6.747475, -590.129, 0.4470588, 0, 0.5490196, 1,
4.848485, 6.747475, -589.9032, 0.4470588, 0, 0.5490196, 1,
4.888889, 6.747475, -589.6846, 0.4470588, 0, 0.5490196, 1,
4.929293, 6.747475, -589.4731, 0.4470588, 0, 0.5490196, 1,
4.969697, 6.747475, -589.2689, 0.5490196, 0, 0.4470588, 1,
5.010101, 6.747475, -589.0718, 0.5490196, 0, 0.4470588, 1,
5.050505, 6.747475, -588.8818, 0.5490196, 0, 0.4470588, 1,
5.090909, 6.747475, -588.6991, 0.5490196, 0, 0.4470588, 1,
5.131313, 6.747475, -588.5235, 0.5490196, 0, 0.4470588, 1,
5.171717, 6.747475, -588.3551, 0.5490196, 0, 0.4470588, 1,
5.212121, 6.747475, -588.1939, 0.5490196, 0, 0.4470588, 1,
5.252525, 6.747475, -588.0398, 0.5490196, 0, 0.4470588, 1,
5.292929, 6.747475, -587.8929, 0.5490196, 0, 0.4470588, 1,
5.333333, 6.747475, -587.7532, 0.5490196, 0, 0.4470588, 1,
5.373737, 6.747475, -587.6207, 0.5490196, 0, 0.4470588, 1,
5.414141, 6.747475, -587.4953, 0.5490196, 0, 0.4470588, 1,
5.454545, 6.747475, -587.3771, 0.5490196, 0, 0.4470588, 1,
5.494949, 6.747475, -587.2661, 0.5490196, 0, 0.4470588, 1,
5.535354, 6.747475, -587.1622, 0.5490196, 0, 0.4470588, 1,
5.575758, 6.747475, -587.0655, 0.5490196, 0, 0.4470588, 1,
5.616162, 6.747475, -586.976, 0.5490196, 0, 0.4470588, 1,
5.656566, 6.747475, -586.8936, 0.5490196, 0, 0.4470588, 1,
5.69697, 6.747475, -586.8184, 0.5490196, 0, 0.4470588, 1,
5.737374, 6.747475, -586.7504, 0.5490196, 0, 0.4470588, 1,
5.777778, 6.747475, -586.6896, 0.5490196, 0, 0.4470588, 1,
5.818182, 6.747475, -586.6359, 0.5490196, 0, 0.4470588, 1,
5.858586, 6.747475, -586.5894, 0.5490196, 0, 0.4470588, 1,
5.89899, 6.747475, -586.5501, 0.5490196, 0, 0.4470588, 1,
5.939394, 6.747475, -586.5179, 0.5490196, 0, 0.4470588, 1,
5.979798, 6.747475, -586.4929, 0.5490196, 0, 0.4470588, 1,
6.020202, 6.747475, -586.4752, 0.5490196, 0, 0.4470588, 1,
6.060606, 6.747475, -586.4645, 0.5490196, 0, 0.4470588, 1,
6.10101, 6.747475, -586.4611, 0.5490196, 0, 0.4470588, 1,
6.141414, 6.747475, -586.4647, 0.5490196, 0, 0.4470588, 1,
6.181818, 6.747475, -586.4756, 0.5490196, 0, 0.4470588, 1,
6.222222, 6.747475, -586.4937, 0.5490196, 0, 0.4470588, 1,
6.262626, 6.747475, -586.5189, 0.5490196, 0, 0.4470588, 1,
6.30303, 6.747475, -586.5513, 0.5490196, 0, 0.4470588, 1,
6.343434, 6.747475, -586.5908, 0.5490196, 0, 0.4470588, 1,
6.383838, 6.747475, -586.6376, 0.5490196, 0, 0.4470588, 1,
6.424242, 6.747475, -586.6915, 0.5490196, 0, 0.4470588, 1,
6.464646, 6.747475, -586.7526, 0.5490196, 0, 0.4470588, 1,
6.505051, 6.747475, -586.8208, 0.5490196, 0, 0.4470588, 1,
6.545455, 6.747475, -586.8962, 0.5490196, 0, 0.4470588, 1,
6.585859, 6.747475, -586.9788, 0.5490196, 0, 0.4470588, 1,
6.626263, 6.747475, -587.0685, 0.5490196, 0, 0.4470588, 1,
6.666667, 6.747475, -587.1655, 0.5490196, 0, 0.4470588, 1,
6.707071, 6.747475, -587.2696, 0.5490196, 0, 0.4470588, 1,
6.747475, 6.747475, -587.3809, 0.5490196, 0, 0.4470588, 1,
6.787879, 6.747475, -587.4993, 0.5490196, 0, 0.4470588, 1,
6.828283, 6.747475, -587.6249, 0.5490196, 0, 0.4470588, 1,
6.868687, 6.747475, -587.7577, 0.5490196, 0, 0.4470588, 1,
6.909091, 6.747475, -587.8976, 0.5490196, 0, 0.4470588, 1,
6.949495, 6.747475, -588.0448, 0.5490196, 0, 0.4470588, 1,
6.989899, 6.747475, -588.1991, 0.5490196, 0, 0.4470588, 1,
7.030303, 6.747475, -588.3605, 0.5490196, 0, 0.4470588, 1,
7.070707, 6.747475, -588.5292, 0.5490196, 0, 0.4470588, 1,
7.111111, 6.747475, -588.705, 0.5490196, 0, 0.4470588, 1,
7.151515, 6.747475, -588.888, 0.5490196, 0, 0.4470588, 1,
7.191919, 6.747475, -589.0782, 0.5490196, 0, 0.4470588, 1,
7.232323, 6.747475, -589.2755, 0.5490196, 0, 0.4470588, 1,
7.272727, 6.747475, -589.48, 0.4470588, 0, 0.5490196, 1,
7.313131, 6.747475, -589.6917, 0.4470588, 0, 0.5490196, 1,
7.353535, 6.747475, -589.9105, 0.4470588, 0, 0.5490196, 1,
7.393939, 6.747475, -590.1365, 0.4470588, 0, 0.5490196, 1,
7.434343, 6.747475, -590.3697, 0.4470588, 0, 0.5490196, 1,
7.474748, 6.747475, -590.61, 0.4470588, 0, 0.5490196, 1,
7.515152, 6.747475, -590.8576, 0.4470588, 0, 0.5490196, 1,
7.555555, 6.747475, -591.1123, 0.4470588, 0, 0.5490196, 1,
7.59596, 6.747475, -591.3741, 0.4470588, 0, 0.5490196, 1,
7.636364, 6.747475, -591.6432, 0.4470588, 0, 0.5490196, 1,
7.676768, 6.747475, -591.9194, 0.4470588, 0, 0.5490196, 1,
7.717172, 6.747475, -592.2028, 0.4470588, 0, 0.5490196, 1,
7.757576, 6.747475, -592.4933, 0.4470588, 0, 0.5490196, 1,
7.79798, 6.747475, -592.7911, 0.4470588, 0, 0.5490196, 1,
7.838384, 6.747475, -593.0959, 0.4470588, 0, 0.5490196, 1,
7.878788, 6.747475, -593.408, 0.4470588, 0, 0.5490196, 1,
7.919192, 6.747475, -593.7273, 0.4470588, 0, 0.5490196, 1,
7.959596, 6.747475, -594.0537, 0.4470588, 0, 0.5490196, 1,
8, 6.747475, -594.3873, 0.4470588, 0, 0.5490196, 1,
4, 6.79798, -597.1899, 0.3411765, 0, 0.654902, 1,
4.040404, 6.79798, -596.8262, 0.3411765, 0, 0.654902, 1,
4.080808, 6.79798, -596.4695, 0.3411765, 0, 0.654902, 1,
4.121212, 6.79798, -596.1199, 0.3411765, 0, 0.654902, 1,
4.161616, 6.79798, -595.7774, 0.3411765, 0, 0.654902, 1,
4.20202, 6.79798, -595.4419, 0.3411765, 0, 0.654902, 1,
4.242424, 6.79798, -595.1135, 0.3411765, 0, 0.654902, 1,
4.282828, 6.79798, -594.7922, 0.3411765, 0, 0.654902, 1,
4.323232, 6.79798, -594.4779, 0.4470588, 0, 0.5490196, 1,
4.363636, 6.79798, -594.1707, 0.4470588, 0, 0.5490196, 1,
4.40404, 6.79798, -593.8705, 0.4470588, 0, 0.5490196, 1,
4.444445, 6.79798, -593.5775, 0.4470588, 0, 0.5490196, 1,
4.484848, 6.79798, -593.2914, 0.4470588, 0, 0.5490196, 1,
4.525252, 6.79798, -593.0125, 0.4470588, 0, 0.5490196, 1,
4.565657, 6.79798, -592.7405, 0.4470588, 0, 0.5490196, 1,
4.606061, 6.79798, -592.4757, 0.4470588, 0, 0.5490196, 1,
4.646465, 6.79798, -592.218, 0.4470588, 0, 0.5490196, 1,
4.686869, 6.79798, -591.9673, 0.4470588, 0, 0.5490196, 1,
4.727273, 6.79798, -591.7236, 0.4470588, 0, 0.5490196, 1,
4.767677, 6.79798, -591.4871, 0.4470588, 0, 0.5490196, 1,
4.808081, 6.79798, -591.2576, 0.4470588, 0, 0.5490196, 1,
4.848485, 6.79798, -591.0352, 0.4470588, 0, 0.5490196, 1,
4.888889, 6.79798, -590.8198, 0.4470588, 0, 0.5490196, 1,
4.929293, 6.79798, -590.6115, 0.4470588, 0, 0.5490196, 1,
4.969697, 6.79798, -590.4102, 0.4470588, 0, 0.5490196, 1,
5.010101, 6.79798, -590.2161, 0.4470588, 0, 0.5490196, 1,
5.050505, 6.79798, -590.0289, 0.4470588, 0, 0.5490196, 1,
5.090909, 6.79798, -589.8489, 0.4470588, 0, 0.5490196, 1,
5.131313, 6.79798, -589.6759, 0.4470588, 0, 0.5490196, 1,
5.171717, 6.79798, -589.51, 0.4470588, 0, 0.5490196, 1,
5.212121, 6.79798, -589.3511, 0.4470588, 0, 0.5490196, 1,
5.252525, 6.79798, -589.1994, 0.5490196, 0, 0.4470588, 1,
5.292929, 6.79798, -589.0547, 0.5490196, 0, 0.4470588, 1,
5.333333, 6.79798, -588.917, 0.5490196, 0, 0.4470588, 1,
5.373737, 6.79798, -588.7864, 0.5490196, 0, 0.4470588, 1,
5.414141, 6.79798, -588.6629, 0.5490196, 0, 0.4470588, 1,
5.454545, 6.79798, -588.5464, 0.5490196, 0, 0.4470588, 1,
5.494949, 6.79798, -588.437, 0.5490196, 0, 0.4470588, 1,
5.535354, 6.79798, -588.3347, 0.5490196, 0, 0.4470588, 1,
5.575758, 6.79798, -588.2394, 0.5490196, 0, 0.4470588, 1,
5.616162, 6.79798, -588.1512, 0.5490196, 0, 0.4470588, 1,
5.656566, 6.79798, -588.0701, 0.5490196, 0, 0.4470588, 1,
5.69697, 6.79798, -587.996, 0.5490196, 0, 0.4470588, 1,
5.737374, 6.79798, -587.9291, 0.5490196, 0, 0.4470588, 1,
5.777778, 6.79798, -587.8691, 0.5490196, 0, 0.4470588, 1,
5.818182, 6.79798, -587.8162, 0.5490196, 0, 0.4470588, 1,
5.858586, 6.79798, -587.7704, 0.5490196, 0, 0.4470588, 1,
5.89899, 6.79798, -587.7317, 0.5490196, 0, 0.4470588, 1,
5.939394, 6.79798, -587.7, 0.5490196, 0, 0.4470588, 1,
5.979798, 6.79798, -587.6754, 0.5490196, 0, 0.4470588, 1,
6.020202, 6.79798, -587.6578, 0.5490196, 0, 0.4470588, 1,
6.060606, 6.79798, -587.6474, 0.5490196, 0, 0.4470588, 1,
6.10101, 6.79798, -587.644, 0.5490196, 0, 0.4470588, 1,
6.141414, 6.79798, -587.6476, 0.5490196, 0, 0.4470588, 1,
6.181818, 6.79798, -587.6583, 0.5490196, 0, 0.4470588, 1,
6.222222, 6.79798, -587.6761, 0.5490196, 0, 0.4470588, 1,
6.262626, 6.79798, -587.7009, 0.5490196, 0, 0.4470588, 1,
6.30303, 6.79798, -587.7328, 0.5490196, 0, 0.4470588, 1,
6.343434, 6.79798, -587.7719, 0.5490196, 0, 0.4470588, 1,
6.383838, 6.79798, -587.8179, 0.5490196, 0, 0.4470588, 1,
6.424242, 6.79798, -587.871, 0.5490196, 0, 0.4470588, 1,
6.464646, 6.79798, -587.9312, 0.5490196, 0, 0.4470588, 1,
6.505051, 6.79798, -587.9984, 0.5490196, 0, 0.4470588, 1,
6.545455, 6.79798, -588.0727, 0.5490196, 0, 0.4470588, 1,
6.585859, 6.79798, -588.1541, 0.5490196, 0, 0.4470588, 1,
6.626263, 6.79798, -588.2425, 0.5490196, 0, 0.4470588, 1,
6.666667, 6.79798, -588.338, 0.5490196, 0, 0.4470588, 1,
6.707071, 6.79798, -588.4406, 0.5490196, 0, 0.4470588, 1,
6.747475, 6.79798, -588.5502, 0.5490196, 0, 0.4470588, 1,
6.787879, 6.79798, -588.6669, 0.5490196, 0, 0.4470588, 1,
6.828283, 6.79798, -588.7906, 0.5490196, 0, 0.4470588, 1,
6.868687, 6.79798, -588.9214, 0.5490196, 0, 0.4470588, 1,
6.909091, 6.79798, -589.0593, 0.5490196, 0, 0.4470588, 1,
6.949495, 6.79798, -589.2043, 0.5490196, 0, 0.4470588, 1,
6.989899, 6.79798, -589.3563, 0.4470588, 0, 0.5490196, 1,
7.030303, 6.79798, -589.5154, 0.4470588, 0, 0.5490196, 1,
7.070707, 6.79798, -589.6815, 0.4470588, 0, 0.5490196, 1,
7.111111, 6.79798, -589.8547, 0.4470588, 0, 0.5490196, 1,
7.151515, 6.79798, -590.035, 0.4470588, 0, 0.5490196, 1,
7.191919, 6.79798, -590.2224, 0.4470588, 0, 0.5490196, 1,
7.232323, 6.79798, -590.4167, 0.4470588, 0, 0.5490196, 1,
7.272727, 6.79798, -590.6182, 0.4470588, 0, 0.5490196, 1,
7.313131, 6.79798, -590.8268, 0.4470588, 0, 0.5490196, 1,
7.353535, 6.79798, -591.0424, 0.4470588, 0, 0.5490196, 1,
7.393939, 6.79798, -591.265, 0.4470588, 0, 0.5490196, 1,
7.434343, 6.79798, -591.4948, 0.4470588, 0, 0.5490196, 1,
7.474748, 6.79798, -591.7316, 0.4470588, 0, 0.5490196, 1,
7.515152, 6.79798, -591.9754, 0.4470588, 0, 0.5490196, 1,
7.555555, 6.79798, -592.2264, 0.4470588, 0, 0.5490196, 1,
7.59596, 6.79798, -592.4844, 0.4470588, 0, 0.5490196, 1,
7.636364, 6.79798, -592.7494, 0.4470588, 0, 0.5490196, 1,
7.676768, 6.79798, -593.0215, 0.4470588, 0, 0.5490196, 1,
7.717172, 6.79798, -593.3007, 0.4470588, 0, 0.5490196, 1,
7.757576, 6.79798, -593.587, 0.4470588, 0, 0.5490196, 1,
7.79798, 6.79798, -593.8803, 0.4470588, 0, 0.5490196, 1,
7.838384, 6.79798, -594.1807, 0.4470588, 0, 0.5490196, 1,
7.878788, 6.79798, -594.4881, 0.4470588, 0, 0.5490196, 1,
7.919192, 6.79798, -594.8026, 0.3411765, 0, 0.654902, 1,
7.959596, 6.79798, -595.1242, 0.3411765, 0, 0.654902, 1,
8, 6.79798, -595.4529, 0.3411765, 0, 0.654902, 1,
4, 6.848485, -598.2284, 0.3411765, 0, 0.654902, 1,
4.040404, 6.848485, -597.87, 0.3411765, 0, 0.654902, 1,
4.080808, 6.848485, -597.5186, 0.3411765, 0, 0.654902, 1,
4.121212, 6.848485, -597.1741, 0.3411765, 0, 0.654902, 1,
4.161616, 6.848485, -596.8365, 0.3411765, 0, 0.654902, 1,
4.20202, 6.848485, -596.506, 0.3411765, 0, 0.654902, 1,
4.242424, 6.848485, -596.1824, 0.3411765, 0, 0.654902, 1,
4.282828, 6.848485, -595.8658, 0.3411765, 0, 0.654902, 1,
4.323232, 6.848485, -595.5562, 0.3411765, 0, 0.654902, 1,
4.363636, 6.848485, -595.2534, 0.3411765, 0, 0.654902, 1,
4.40404, 6.848485, -594.9577, 0.3411765, 0, 0.654902, 1,
4.444445, 6.848485, -594.6689, 0.4470588, 0, 0.5490196, 1,
4.484848, 6.848485, -594.3871, 0.4470588, 0, 0.5490196, 1,
4.525252, 6.848485, -594.1122, 0.4470588, 0, 0.5490196, 1,
4.565657, 6.848485, -593.8444, 0.4470588, 0, 0.5490196, 1,
4.606061, 6.848485, -593.5834, 0.4470588, 0, 0.5490196, 1,
4.646465, 6.848485, -593.3295, 0.4470588, 0, 0.5490196, 1,
4.686869, 6.848485, -593.0825, 0.4470588, 0, 0.5490196, 1,
4.727273, 6.848485, -592.8424, 0.4470588, 0, 0.5490196, 1,
4.767677, 6.848485, -592.6093, 0.4470588, 0, 0.5490196, 1,
4.808081, 6.848485, -592.3832, 0.4470588, 0, 0.5490196, 1,
4.848485, 6.848485, -592.164, 0.4470588, 0, 0.5490196, 1,
4.888889, 6.848485, -591.9518, 0.4470588, 0, 0.5490196, 1,
4.929293, 6.848485, -591.7465, 0.4470588, 0, 0.5490196, 1,
4.969697, 6.848485, -591.5483, 0.4470588, 0, 0.5490196, 1,
5.010101, 6.848485, -591.3569, 0.4470588, 0, 0.5490196, 1,
5.050505, 6.848485, -591.1726, 0.4470588, 0, 0.5490196, 1,
5.090909, 6.848485, -590.9952, 0.4470588, 0, 0.5490196, 1,
5.131313, 6.848485, -590.8248, 0.4470588, 0, 0.5490196, 1,
5.171717, 6.848485, -590.6613, 0.4470588, 0, 0.5490196, 1,
5.212121, 6.848485, -590.5048, 0.4470588, 0, 0.5490196, 1,
5.252525, 6.848485, -590.3552, 0.4470588, 0, 0.5490196, 1,
5.292929, 6.848485, -590.2126, 0.4470588, 0, 0.5490196, 1,
5.333333, 6.848485, -590.077, 0.4470588, 0, 0.5490196, 1,
5.373737, 6.848485, -589.9483, 0.4470588, 0, 0.5490196, 1,
5.414141, 6.848485, -589.8266, 0.4470588, 0, 0.5490196, 1,
5.454545, 6.848485, -589.7119, 0.4470588, 0, 0.5490196, 1,
5.494949, 6.848485, -589.6041, 0.4470588, 0, 0.5490196, 1,
5.535354, 6.848485, -589.5032, 0.4470588, 0, 0.5490196, 1,
5.575758, 6.848485, -589.4094, 0.4470588, 0, 0.5490196, 1,
5.616162, 6.848485, -589.3225, 0.5490196, 0, 0.4470588, 1,
5.656566, 6.848485, -589.2426, 0.5490196, 0, 0.4470588, 1,
5.69697, 6.848485, -589.1696, 0.5490196, 0, 0.4470588, 1,
5.737374, 6.848485, -589.1036, 0.5490196, 0, 0.4470588, 1,
5.777778, 6.848485, -589.0445, 0.5490196, 0, 0.4470588, 1,
5.818182, 6.848485, -588.9924, 0.5490196, 0, 0.4470588, 1,
5.858586, 6.848485, -588.9473, 0.5490196, 0, 0.4470588, 1,
5.89899, 6.848485, -588.9091, 0.5490196, 0, 0.4470588, 1,
5.939394, 6.848485, -588.8779, 0.5490196, 0, 0.4470588, 1,
5.979798, 6.848485, -588.8536, 0.5490196, 0, 0.4470588, 1,
6.020202, 6.848485, -588.8364, 0.5490196, 0, 0.4470588, 1,
6.060606, 6.848485, -588.826, 0.5490196, 0, 0.4470588, 1,
6.10101, 6.848485, -588.8226, 0.5490196, 0, 0.4470588, 1,
6.141414, 6.848485, -588.8262, 0.5490196, 0, 0.4470588, 1,
6.181818, 6.848485, -588.8368, 0.5490196, 0, 0.4470588, 1,
6.222222, 6.848485, -588.8543, 0.5490196, 0, 0.4470588, 1,
6.262626, 6.848485, -588.8788, 0.5490196, 0, 0.4470588, 1,
6.30303, 6.848485, -588.9102, 0.5490196, 0, 0.4470588, 1,
6.343434, 6.848485, -588.9487, 0.5490196, 0, 0.4470588, 1,
6.383838, 6.848485, -588.994, 0.5490196, 0, 0.4470588, 1,
6.424242, 6.848485, -589.0463, 0.5490196, 0, 0.4470588, 1,
6.464646, 6.848485, -589.1056, 0.5490196, 0, 0.4470588, 1,
6.505051, 6.848485, -589.1719, 0.5490196, 0, 0.4470588, 1,
6.545455, 6.848485, -589.2451, 0.5490196, 0, 0.4470588, 1,
6.585859, 6.848485, -589.3253, 0.5490196, 0, 0.4470588, 1,
6.626263, 6.848485, -589.4124, 0.4470588, 0, 0.5490196, 1,
6.666667, 6.848485, -589.5065, 0.4470588, 0, 0.5490196, 1,
6.707071, 6.848485, -589.6075, 0.4470588, 0, 0.5490196, 1,
6.747475, 6.848485, -589.7155, 0.4470588, 0, 0.5490196, 1,
6.787879, 6.848485, -589.8305, 0.4470588, 0, 0.5490196, 1,
6.828283, 6.848485, -589.9525, 0.4470588, 0, 0.5490196, 1,
6.868687, 6.848485, -590.0814, 0.4470588, 0, 0.5490196, 1,
6.909091, 6.848485, -590.2172, 0.4470588, 0, 0.5490196, 1,
6.949495, 6.848485, -590.36, 0.4470588, 0, 0.5490196, 1,
6.989899, 6.848485, -590.5098, 0.4470588, 0, 0.5490196, 1,
7.030303, 6.848485, -590.6666, 0.4470588, 0, 0.5490196, 1,
7.070707, 6.848485, -590.8303, 0.4470588, 0, 0.5490196, 1,
7.111111, 6.848485, -591.0009, 0.4470588, 0, 0.5490196, 1,
7.151515, 6.848485, -591.1785, 0.4470588, 0, 0.5490196, 1,
7.191919, 6.848485, -591.3632, 0.4470588, 0, 0.5490196, 1,
7.232323, 6.848485, -591.5547, 0.4470588, 0, 0.5490196, 1,
7.272727, 6.848485, -591.7532, 0.4470588, 0, 0.5490196, 1,
7.313131, 6.848485, -591.9587, 0.4470588, 0, 0.5490196, 1,
7.353535, 6.848485, -592.1711, 0.4470588, 0, 0.5490196, 1,
7.393939, 6.848485, -592.3905, 0.4470588, 0, 0.5490196, 1,
7.434343, 6.848485, -592.6169, 0.4470588, 0, 0.5490196, 1,
7.474748, 6.848485, -592.8502, 0.4470588, 0, 0.5490196, 1,
7.515152, 6.848485, -593.0905, 0.4470588, 0, 0.5490196, 1,
7.555555, 6.848485, -593.3377, 0.4470588, 0, 0.5490196, 1,
7.59596, 6.848485, -593.5919, 0.4470588, 0, 0.5490196, 1,
7.636364, 6.848485, -593.8531, 0.4470588, 0, 0.5490196, 1,
7.676768, 6.848485, -594.1212, 0.4470588, 0, 0.5490196, 1,
7.717172, 6.848485, -594.3963, 0.4470588, 0, 0.5490196, 1,
7.757576, 6.848485, -594.6783, 0.4470588, 0, 0.5490196, 1,
7.79798, 6.848485, -594.9673, 0.3411765, 0, 0.654902, 1,
7.838384, 6.848485, -595.2633, 0.3411765, 0, 0.654902, 1,
7.878788, 6.848485, -595.5662, 0.3411765, 0, 0.654902, 1,
7.919192, 6.848485, -595.8762, 0.3411765, 0, 0.654902, 1,
7.959596, 6.848485, -596.193, 0.3411765, 0, 0.654902, 1,
8, 6.848485, -596.5168, 0.3411765, 0, 0.654902, 1,
4, 6.89899, -599.2656, 0.3411765, 0, 0.654902, 1,
4.040404, 6.89899, -598.9124, 0.3411765, 0, 0.654902, 1,
4.080808, 6.89899, -598.5661, 0.3411765, 0, 0.654902, 1,
4.121212, 6.89899, -598.2267, 0.3411765, 0, 0.654902, 1,
4.161616, 6.89899, -597.894, 0.3411765, 0, 0.654902, 1,
4.20202, 6.89899, -597.5684, 0.3411765, 0, 0.654902, 1,
4.242424, 6.89899, -597.2495, 0.3411765, 0, 0.654902, 1,
4.282828, 6.89899, -596.9375, 0.3411765, 0, 0.654902, 1,
4.323232, 6.89899, -596.6323, 0.3411765, 0, 0.654902, 1,
4.363636, 6.89899, -596.334, 0.3411765, 0, 0.654902, 1,
4.40404, 6.89899, -596.0426, 0.3411765, 0, 0.654902, 1,
4.444445, 6.89899, -595.7581, 0.3411765, 0, 0.654902, 1,
4.484848, 6.89899, -595.4803, 0.3411765, 0, 0.654902, 1,
4.525252, 6.89899, -595.2095, 0.3411765, 0, 0.654902, 1,
4.565657, 6.89899, -594.9455, 0.3411765, 0, 0.654902, 1,
4.606061, 6.89899, -594.6884, 0.4470588, 0, 0.5490196, 1,
4.646465, 6.89899, -594.4381, 0.4470588, 0, 0.5490196, 1,
4.686869, 6.89899, -594.1947, 0.4470588, 0, 0.5490196, 1,
4.727273, 6.89899, -593.9581, 0.4470588, 0, 0.5490196, 1,
4.767677, 6.89899, -593.7285, 0.4470588, 0, 0.5490196, 1,
4.808081, 6.89899, -593.5056, 0.4470588, 0, 0.5490196, 1,
4.848485, 6.89899, -593.2897, 0.4470588, 0, 0.5490196, 1,
4.888889, 6.89899, -593.0806, 0.4470588, 0, 0.5490196, 1,
4.929293, 6.89899, -592.8783, 0.4470588, 0, 0.5490196, 1,
4.969697, 6.89899, -592.6829, 0.4470588, 0, 0.5490196, 1,
5.010101, 6.89899, -592.4944, 0.4470588, 0, 0.5490196, 1,
5.050505, 6.89899, -592.3127, 0.4470588, 0, 0.5490196, 1,
5.090909, 6.89899, -592.1379, 0.4470588, 0, 0.5490196, 1,
5.131313, 6.89899, -591.97, 0.4470588, 0, 0.5490196, 1,
5.171717, 6.89899, -591.8088, 0.4470588, 0, 0.5490196, 1,
5.212121, 6.89899, -591.6547, 0.4470588, 0, 0.5490196, 1,
5.252525, 6.89899, -591.5073, 0.4470588, 0, 0.5490196, 1,
5.292929, 6.89899, -591.3668, 0.4470588, 0, 0.5490196, 1,
5.333333, 6.89899, -591.2331, 0.4470588, 0, 0.5490196, 1,
5.373737, 6.89899, -591.1063, 0.4470588, 0, 0.5490196, 1,
5.414141, 6.89899, -590.9864, 0.4470588, 0, 0.5490196, 1,
5.454545, 6.89899, -590.8733, 0.4470588, 0, 0.5490196, 1,
5.494949, 6.89899, -590.7671, 0.4470588, 0, 0.5490196, 1,
5.535354, 6.89899, -590.6677, 0.4470588, 0, 0.5490196, 1,
5.575758, 6.89899, -590.5753, 0.4470588, 0, 0.5490196, 1,
5.616162, 6.89899, -590.4896, 0.4470588, 0, 0.5490196, 1,
5.656566, 6.89899, -590.4108, 0.4470588, 0, 0.5490196, 1,
5.69697, 6.89899, -590.3389, 0.4470588, 0, 0.5490196, 1,
5.737374, 6.89899, -590.2739, 0.4470588, 0, 0.5490196, 1,
5.777778, 6.89899, -590.2157, 0.4470588, 0, 0.5490196, 1,
5.818182, 6.89899, -590.1643, 0.4470588, 0, 0.5490196, 1,
5.858586, 6.89899, -590.1199, 0.4470588, 0, 0.5490196, 1,
5.89899, 6.89899, -590.0823, 0.4470588, 0, 0.5490196, 1,
5.939394, 6.89899, -590.0515, 0.4470588, 0, 0.5490196, 1,
5.979798, 6.89899, -590.0276, 0.4470588, 0, 0.5490196, 1,
6.020202, 6.89899, -590.0106, 0.4470588, 0, 0.5490196, 1,
6.060606, 6.89899, -590.0004, 0.4470588, 0, 0.5490196, 1,
6.10101, 6.89899, -589.9971, 0.4470588, 0, 0.5490196, 1,
6.141414, 6.89899, -590.0006, 0.4470588, 0, 0.5490196, 1,
6.181818, 6.89899, -590.011, 0.4470588, 0, 0.5490196, 1,
6.222222, 6.89899, -590.0283, 0.4470588, 0, 0.5490196, 1,
6.262626, 6.89899, -590.0524, 0.4470588, 0, 0.5490196, 1,
6.30303, 6.89899, -590.0834, 0.4470588, 0, 0.5490196, 1,
6.343434, 6.89899, -590.1212, 0.4470588, 0, 0.5490196, 1,
6.383838, 6.89899, -590.1659, 0.4470588, 0, 0.5490196, 1,
6.424242, 6.89899, -590.2175, 0.4470588, 0, 0.5490196, 1,
6.464646, 6.89899, -590.2759, 0.4470588, 0, 0.5490196, 1,
6.505051, 6.89899, -590.3412, 0.4470588, 0, 0.5490196, 1,
6.545455, 6.89899, -590.4133, 0.4470588, 0, 0.5490196, 1,
6.585859, 6.89899, -590.4923, 0.4470588, 0, 0.5490196, 1,
6.626263, 6.89899, -590.5782, 0.4470588, 0, 0.5490196, 1,
6.666667, 6.89899, -590.6709, 0.4470588, 0, 0.5490196, 1,
6.707071, 6.89899, -590.7705, 0.4470588, 0, 0.5490196, 1,
6.747475, 6.89899, -590.8769, 0.4470588, 0, 0.5490196, 1,
6.787879, 6.89899, -590.9902, 0.4470588, 0, 0.5490196, 1,
6.828283, 6.89899, -591.1104, 0.4470588, 0, 0.5490196, 1,
6.868687, 6.89899, -591.2374, 0.4470588, 0, 0.5490196, 1,
6.909091, 6.89899, -591.3713, 0.4470588, 0, 0.5490196, 1,
6.949495, 6.89899, -591.512, 0.4470588, 0, 0.5490196, 1,
6.989899, 6.89899, -591.6596, 0.4470588, 0, 0.5490196, 1,
7.030303, 6.89899, -591.8141, 0.4470588, 0, 0.5490196, 1,
7.070707, 6.89899, -591.9754, 0.4470588, 0, 0.5490196, 1,
7.111111, 6.89899, -592.1436, 0.4470588, 0, 0.5490196, 1,
7.151515, 6.89899, -592.3186, 0.4470588, 0, 0.5490196, 1,
7.191919, 6.89899, -592.5005, 0.4470588, 0, 0.5490196, 1,
7.232323, 6.89899, -592.6893, 0.4470588, 0, 0.5490196, 1,
7.272727, 6.89899, -592.8849, 0.4470588, 0, 0.5490196, 1,
7.313131, 6.89899, -593.0873, 0.4470588, 0, 0.5490196, 1,
7.353535, 6.89899, -593.2967, 0.4470588, 0, 0.5490196, 1,
7.393939, 6.89899, -593.5129, 0.4470588, 0, 0.5490196, 1,
7.434343, 6.89899, -593.736, 0.4470588, 0, 0.5490196, 1,
7.474748, 6.89899, -593.9658, 0.4470588, 0, 0.5490196, 1,
7.515152, 6.89899, -594.2026, 0.4470588, 0, 0.5490196, 1,
7.555555, 6.89899, -594.4462, 0.4470588, 0, 0.5490196, 1,
7.59596, 6.89899, -594.6968, 0.4470588, 0, 0.5490196, 1,
7.636364, 6.89899, -594.9541, 0.3411765, 0, 0.654902, 1,
7.676768, 6.89899, -595.2183, 0.3411765, 0, 0.654902, 1,
7.717172, 6.89899, -595.4894, 0.3411765, 0, 0.654902, 1,
7.757576, 6.89899, -595.7673, 0.3411765, 0, 0.654902, 1,
7.79798, 6.89899, -596.0521, 0.3411765, 0, 0.654902, 1,
7.838384, 6.89899, -596.3438, 0.3411765, 0, 0.654902, 1,
7.878788, 6.89899, -596.6423, 0.3411765, 0, 0.654902, 1,
7.919192, 6.89899, -596.9476, 0.3411765, 0, 0.654902, 1,
7.959596, 6.89899, -597.2599, 0.3411765, 0, 0.654902, 1,
8, 6.89899, -597.579, 0.3411765, 0, 0.654902, 1,
4, 6.949495, -600.3015, 0.2392157, 0, 0.7568628, 1,
4.040404, 6.949495, -599.9534, 0.3411765, 0, 0.654902, 1,
4.080808, 6.949495, -599.6121, 0.3411765, 0, 0.654902, 1,
4.121212, 6.949495, -599.2776, 0.3411765, 0, 0.654902, 1,
4.161616, 6.949495, -598.9498, 0.3411765, 0, 0.654902, 1,
4.20202, 6.949495, -598.6288, 0.3411765, 0, 0.654902, 1,
4.242424, 6.949495, -598.3146, 0.3411765, 0, 0.654902, 1,
4.282828, 6.949495, -598.0071, 0.3411765, 0, 0.654902, 1,
4.323232, 6.949495, -597.7064, 0.3411765, 0, 0.654902, 1,
4.363636, 6.949495, -597.4124, 0.3411765, 0, 0.654902, 1,
4.40404, 6.949495, -597.1252, 0.3411765, 0, 0.654902, 1,
4.444445, 6.949495, -596.8447, 0.3411765, 0, 0.654902, 1,
4.484848, 6.949495, -596.571, 0.3411765, 0, 0.654902, 1,
4.525252, 6.949495, -596.3041, 0.3411765, 0, 0.654902, 1,
4.565657, 6.949495, -596.0439, 0.3411765, 0, 0.654902, 1,
4.606061, 6.949495, -595.7905, 0.3411765, 0, 0.654902, 1,
4.646465, 6.949495, -595.5439, 0.3411765, 0, 0.654902, 1,
4.686869, 6.949495, -595.304, 0.3411765, 0, 0.654902, 1,
4.727273, 6.949495, -595.0709, 0.3411765, 0, 0.654902, 1,
4.767677, 6.949495, -594.8445, 0.3411765, 0, 0.654902, 1,
4.808081, 6.949495, -594.6249, 0.4470588, 0, 0.5490196, 1,
4.848485, 6.949495, -594.4121, 0.4470588, 0, 0.5490196, 1,
4.888889, 6.949495, -594.206, 0.4470588, 0, 0.5490196, 1,
4.929293, 6.949495, -594.0067, 0.4470588, 0, 0.5490196, 1,
4.969697, 6.949495, -593.8141, 0.4470588, 0, 0.5490196, 1,
5.010101, 6.949495, -593.6284, 0.4470588, 0, 0.5490196, 1,
5.050505, 6.949495, -593.4493, 0.4470588, 0, 0.5490196, 1,
5.090909, 6.949495, -593.277, 0.4470588, 0, 0.5490196, 1,
5.131313, 6.949495, -593.1115, 0.4470588, 0, 0.5490196, 1,
5.171717, 6.949495, -592.9528, 0.4470588, 0, 0.5490196, 1,
5.212121, 6.949495, -592.8007, 0.4470588, 0, 0.5490196, 1,
5.252525, 6.949495, -592.6555, 0.4470588, 0, 0.5490196, 1,
5.292929, 6.949495, -592.517, 0.4470588, 0, 0.5490196, 1,
5.333333, 6.949495, -592.3853, 0.4470588, 0, 0.5490196, 1,
5.373737, 6.949495, -592.2604, 0.4470588, 0, 0.5490196, 1,
5.414141, 6.949495, -592.1422, 0.4470588, 0, 0.5490196, 1,
5.454545, 6.949495, -592.0307, 0.4470588, 0, 0.5490196, 1,
5.494949, 6.949495, -591.926, 0.4470588, 0, 0.5490196, 1,
5.535354, 6.949495, -591.8281, 0.4470588, 0, 0.5490196, 1,
5.575758, 6.949495, -591.737, 0.4470588, 0, 0.5490196, 1,
5.616162, 6.949495, -591.6526, 0.4470588, 0, 0.5490196, 1,
5.656566, 6.949495, -591.575, 0.4470588, 0, 0.5490196, 1,
5.69697, 6.949495, -591.5041, 0.4470588, 0, 0.5490196, 1,
5.737374, 6.949495, -591.4399, 0.4470588, 0, 0.5490196, 1,
5.777778, 6.949495, -591.3826, 0.4470588, 0, 0.5490196, 1,
5.818182, 6.949495, -591.332, 0.4470588, 0, 0.5490196, 1,
5.858586, 6.949495, -591.2882, 0.4470588, 0, 0.5490196, 1,
5.89899, 6.949495, -591.2511, 0.4470588, 0, 0.5490196, 1,
5.939394, 6.949495, -591.2208, 0.4470588, 0, 0.5490196, 1,
5.979798, 6.949495, -591.1973, 0.4470588, 0, 0.5490196, 1,
6.020202, 6.949495, -591.1805, 0.4470588, 0, 0.5490196, 1,
6.060606, 6.949495, -591.1704, 0.4470588, 0, 0.5490196, 1,
6.10101, 6.949495, -591.1672, 0.4470588, 0, 0.5490196, 1,
6.141414, 6.949495, -591.1707, 0.4470588, 0, 0.5490196, 1,
6.181818, 6.949495, -591.1809, 0.4470588, 0, 0.5490196, 1,
6.222222, 6.949495, -591.1979, 0.4470588, 0, 0.5490196, 1,
6.262626, 6.949495, -591.2217, 0.4470588, 0, 0.5490196, 1,
6.30303, 6.949495, -591.2523, 0.4470588, 0, 0.5490196, 1,
6.343434, 6.949495, -591.2896, 0.4470588, 0, 0.5490196, 1,
6.383838, 6.949495, -591.3336, 0.4470588, 0, 0.5490196, 1,
6.424242, 6.949495, -591.3844, 0.4470588, 0, 0.5490196, 1,
6.464646, 6.949495, -591.442, 0.4470588, 0, 0.5490196, 1,
6.505051, 6.949495, -591.5063, 0.4470588, 0, 0.5490196, 1,
6.545455, 6.949495, -591.5774, 0.4470588, 0, 0.5490196, 1,
6.585859, 6.949495, -591.6553, 0.4470588, 0, 0.5490196, 1,
6.626263, 6.949495, -591.7399, 0.4470588, 0, 0.5490196, 1,
6.666667, 6.949495, -591.8312, 0.4470588, 0, 0.5490196, 1,
6.707071, 6.949495, -591.9294, 0.4470588, 0, 0.5490196, 1,
6.747475, 6.949495, -592.0343, 0.4470588, 0, 0.5490196, 1,
6.787879, 6.949495, -592.1459, 0.4470588, 0, 0.5490196, 1,
6.828283, 6.949495, -592.2643, 0.4470588, 0, 0.5490196, 1,
6.868687, 6.949495, -592.3895, 0.4470588, 0, 0.5490196, 1,
6.909091, 6.949495, -592.5215, 0.4470588, 0, 0.5490196, 1,
6.949495, 6.949495, -592.6602, 0.4470588, 0, 0.5490196, 1,
6.989899, 6.949495, -592.8057, 0.4470588, 0, 0.5490196, 1,
7.030303, 6.949495, -592.9579, 0.4470588, 0, 0.5490196, 1,
7.070707, 6.949495, -593.1168, 0.4470588, 0, 0.5490196, 1,
7.111111, 6.949495, -593.2826, 0.4470588, 0, 0.5490196, 1,
7.151515, 6.949495, -593.4551, 0.4470588, 0, 0.5490196, 1,
7.191919, 6.949495, -593.6343, 0.4470588, 0, 0.5490196, 1,
7.232323, 6.949495, -593.8204, 0.4470588, 0, 0.5490196, 1,
7.272727, 6.949495, -594.0132, 0.4470588, 0, 0.5490196, 1,
7.313131, 6.949495, -594.2127, 0.4470588, 0, 0.5490196, 1,
7.353535, 6.949495, -594.419, 0.4470588, 0, 0.5490196, 1,
7.393939, 6.949495, -594.6321, 0.4470588, 0, 0.5490196, 1,
7.434343, 6.949495, -594.8519, 0.3411765, 0, 0.654902, 1,
7.474748, 6.949495, -595.0785, 0.3411765, 0, 0.654902, 1,
7.515152, 6.949495, -595.3118, 0.3411765, 0, 0.654902, 1,
7.555555, 6.949495, -595.5519, 0.3411765, 0, 0.654902, 1,
7.59596, 6.949495, -595.7988, 0.3411765, 0, 0.654902, 1,
7.636364, 6.949495, -596.0524, 0.3411765, 0, 0.654902, 1,
7.676768, 6.949495, -596.3128, 0.3411765, 0, 0.654902, 1,
7.717172, 6.949495, -596.58, 0.3411765, 0, 0.654902, 1,
7.757576, 6.949495, -596.8539, 0.3411765, 0, 0.654902, 1,
7.79798, 6.949495, -597.1345, 0.3411765, 0, 0.654902, 1,
7.838384, 6.949495, -597.4219, 0.3411765, 0, 0.654902, 1,
7.878788, 6.949495, -597.7161, 0.3411765, 0, 0.654902, 1,
7.919192, 6.949495, -598.0171, 0.3411765, 0, 0.654902, 1,
7.959596, 6.949495, -598.3248, 0.3411765, 0, 0.654902, 1,
8, 6.949495, -598.6393, 0.3411765, 0, 0.654902, 1,
4, 7, -601.3359, 0.2392157, 0, 0.7568628, 1,
4.040404, 7, -600.9929, 0.2392157, 0, 0.7568628, 1,
4.080808, 7, -600.6564, 0.2392157, 0, 0.7568628, 1,
4.121212, 7, -600.3267, 0.2392157, 0, 0.7568628, 1,
4.161616, 7, -600.0037, 0.3411765, 0, 0.654902, 1,
4.20202, 7, -599.6873, 0.3411765, 0, 0.654902, 1,
4.242424, 7, -599.3776, 0.3411765, 0, 0.654902, 1,
4.282828, 7, -599.0745, 0.3411765, 0, 0.654902, 1,
4.323232, 7, -598.7781, 0.3411765, 0, 0.654902, 1,
4.363636, 7, -598.4883, 0.3411765, 0, 0.654902, 1,
4.40404, 7, -598.2053, 0.3411765, 0, 0.654902, 1,
4.444445, 7, -597.9288, 0.3411765, 0, 0.654902, 1,
4.484848, 7, -597.6591, 0.3411765, 0, 0.654902, 1,
4.525252, 7, -597.396, 0.3411765, 0, 0.654902, 1,
4.565657, 7, -597.1396, 0.3411765, 0, 0.654902, 1,
4.606061, 7, -596.8898, 0.3411765, 0, 0.654902, 1,
4.646465, 7, -596.6467, 0.3411765, 0, 0.654902, 1,
4.686869, 7, -596.4103, 0.3411765, 0, 0.654902, 1,
4.727273, 7, -596.1805, 0.3411765, 0, 0.654902, 1,
4.767677, 7, -595.9575, 0.3411765, 0, 0.654902, 1,
4.808081, 7, -595.741, 0.3411765, 0, 0.654902, 1,
4.848485, 7, -595.5312, 0.3411765, 0, 0.654902, 1,
4.888889, 7, -595.3281, 0.3411765, 0, 0.654902, 1,
4.929293, 7, -595.1317, 0.3411765, 0, 0.654902, 1,
4.969697, 7, -594.9418, 0.3411765, 0, 0.654902, 1,
5.010101, 7, -594.7587, 0.3411765, 0, 0.654902, 1,
5.050505, 7, -594.5823, 0.4470588, 0, 0.5490196, 1,
5.090909, 7, -594.4124, 0.4470588, 0, 0.5490196, 1,
5.131313, 7, -594.2493, 0.4470588, 0, 0.5490196, 1,
5.171717, 7, -594.0928, 0.4470588, 0, 0.5490196, 1,
5.212121, 7, -593.943, 0.4470588, 0, 0.5490196, 1,
5.252525, 7, -593.7999, 0.4470588, 0, 0.5490196, 1,
5.292929, 7, -593.6634, 0.4470588, 0, 0.5490196, 1,
5.333333, 7, -593.5336, 0.4470588, 0, 0.5490196, 1,
5.373737, 7, -593.4104, 0.4470588, 0, 0.5490196, 1,
5.414141, 7, -593.2939, 0.4470588, 0, 0.5490196, 1,
5.454545, 7, -593.1841, 0.4470588, 0, 0.5490196, 1,
5.494949, 7, -593.0809, 0.4470588, 0, 0.5490196, 1,
5.535354, 7, -592.9844, 0.4470588, 0, 0.5490196, 1,
5.575758, 7, -592.8945, 0.4470588, 0, 0.5490196, 1,
5.616162, 7, -592.8113, 0.4470588, 0, 0.5490196, 1,
5.656566, 7, -592.7349, 0.4470588, 0, 0.5490196, 1,
5.69697, 7, -592.665, 0.4470588, 0, 0.5490196, 1,
5.737374, 7, -592.6018, 0.4470588, 0, 0.5490196, 1,
5.777778, 7, -592.5453, 0.4470588, 0, 0.5490196, 1,
5.818182, 7, -592.4954, 0.4470588, 0, 0.5490196, 1,
5.858586, 7, -592.4522, 0.4470588, 0, 0.5490196, 1,
5.89899, 7, -592.4156, 0.4470588, 0, 0.5490196, 1,
5.939394, 7, -592.3858, 0.4470588, 0, 0.5490196, 1,
5.979798, 7, -592.3626, 0.4470588, 0, 0.5490196, 1,
6.020202, 7, -592.3461, 0.4470588, 0, 0.5490196, 1,
6.060606, 7, -592.3362, 0.4470588, 0, 0.5490196, 1,
6.10101, 7, -592.3329, 0.4470588, 0, 0.5490196, 1,
6.141414, 7, -592.3364, 0.4470588, 0, 0.5490196, 1,
6.181818, 7, -592.3465, 0.4470588, 0, 0.5490196, 1,
6.222222, 7, -592.3633, 0.4470588, 0, 0.5490196, 1,
6.262626, 7, -592.3867, 0.4470588, 0, 0.5490196, 1,
6.30303, 7, -592.4167, 0.4470588, 0, 0.5490196, 1,
6.343434, 7, -592.4536, 0.4470588, 0, 0.5490196, 1,
6.383838, 7, -592.4969, 0.4470588, 0, 0.5490196, 1,
6.424242, 7, -592.5471, 0.4470588, 0, 0.5490196, 1,
6.464646, 7, -592.6038, 0.4470588, 0, 0.5490196, 1,
6.505051, 7, -592.6672, 0.4470588, 0, 0.5490196, 1,
6.545455, 7, -592.7372, 0.4470588, 0, 0.5490196, 1,
6.585859, 7, -592.814, 0.4470588, 0, 0.5490196, 1,
6.626263, 7, -592.8974, 0.4470588, 0, 0.5490196, 1,
6.666667, 7, -592.9875, 0.4470588, 0, 0.5490196, 1,
6.707071, 7, -593.0842, 0.4470588, 0, 0.5490196, 1,
6.747475, 7, -593.1876, 0.4470588, 0, 0.5490196, 1,
6.787879, 7, -593.2977, 0.4470588, 0, 0.5490196, 1,
6.828283, 7, -593.4144, 0.4470588, 0, 0.5490196, 1,
6.868687, 7, -593.5377, 0.4470588, 0, 0.5490196, 1,
6.909091, 7, -593.6678, 0.4470588, 0, 0.5490196, 1,
6.949495, 7, -593.8045, 0.4470588, 0, 0.5490196, 1,
6.989899, 7, -593.9479, 0.4470588, 0, 0.5490196, 1,
7.030303, 7, -594.0979, 0.4470588, 0, 0.5490196, 1,
7.070707, 7, -594.2546, 0.4470588, 0, 0.5490196, 1,
7.111111, 7, -594.4179, 0.4470588, 0, 0.5490196, 1,
7.151515, 7, -594.588, 0.4470588, 0, 0.5490196, 1,
7.191919, 7, -594.7646, 0.3411765, 0, 0.654902, 1,
7.232323, 7, -594.948, 0.3411765, 0, 0.654902, 1,
7.272727, 7, -595.138, 0.3411765, 0, 0.654902, 1,
7.313131, 7, -595.3347, 0.3411765, 0, 0.654902, 1,
7.353535, 7, -595.538, 0.3411765, 0, 0.654902, 1,
7.393939, 7, -595.748, 0.3411765, 0, 0.654902, 1,
7.434343, 7, -595.9647, 0.3411765, 0, 0.654902, 1,
7.474748, 7, -596.188, 0.3411765, 0, 0.654902, 1,
7.515152, 7, -596.418, 0.3411765, 0, 0.654902, 1,
7.555555, 7, -596.6547, 0.3411765, 0, 0.654902, 1,
7.59596, 7, -596.8979, 0.3411765, 0, 0.654902, 1,
7.636364, 7, -597.1479, 0.3411765, 0, 0.654902, 1,
7.676768, 7, -597.4046, 0.3411765, 0, 0.654902, 1,
7.717172, 7, -597.6679, 0.3411765, 0, 0.654902, 1,
7.757576, 7, -597.9379, 0.3411765, 0, 0.654902, 1,
7.79798, 7, -598.2145, 0.3411765, 0, 0.654902, 1,
7.838384, 7, -598.4978, 0.3411765, 0, 0.654902, 1,
7.878788, 7, -598.7878, 0.3411765, 0, 0.654902, 1,
7.919192, 7, -599.0844, 0.3411765, 0, 0.654902, 1,
7.959596, 7, -599.3877, 0.3411765, 0, 0.654902, 1,
8, 7, -599.6976, 0.3411765, 0, 0.654902, 1
]);
this.values[7] = v;
this.buf[7] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[7]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[7], gl.STATIC_DRAW);
this.mvMatLoc[7] = gl.getUniformLocation(this.prog[7],"mvMatrix");
this.prMatLoc[7] = gl.getUniformLocation(this.prog[7],"prMatrix");
// ****** text object 9 ******
this.flags[9] = 40;
this.vshaders[9] = "	/* ****** text object 9 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[9] = "	/* ****** text object 9 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[9]  = gl.createProgram();
gl.attachShader(this.prog[9], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[9] ));
gl.attachShader(this.prog[9], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[9] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[9], 0, "aPos");
gl.bindAttribLocation(this.prog[9], 1, "aCol");
gl.linkProgram(this.prog[9]);
texts = [
"mu"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[9] = gl.getAttribLocation(this.prog[9], "aOfs");
this.texture[9] = gl.createTexture();
this.texLoc[9] = gl.getAttribLocation(this.prog[9], "aTexcoord");
this.sampler[9] = gl.getUniformLocation(this.prog[9],"uSampler");
this.handleLoadedTexture(9);
this.offsets[9]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
6, 1.1525, -697.2096, 0, -0.5, 0.5, 0.5,
6, 1.1525, -697.2096, 1, -0.5, 0.5, 0.5,
6, 1.1525, -697.2096, 1, 1.5, 0.5, 0.5,
6, 1.1525, -697.2096, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<1; i++)
for (j=0; j<4; j++) {
ind = this.offsets[9].stride*(4*i + j) + this.offsets[9].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[9] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3
]);
this.buf[9] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[9]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[9], gl.STATIC_DRAW);
this.ibuf[9] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[9]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[9] = gl.getUniformLocation(this.prog[9],"mvMatrix");
this.prMatLoc[9] = gl.getUniformLocation(this.prog[9],"prMatrix");
this.textScaleLoc[9] = gl.getUniformLocation(this.prog[9],"textScale");
// ****** text object 10 ******
this.flags[10] = 40;
this.vshaders[10] = "	/* ****** text object 10 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[10] = "	/* ****** text object 10 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[10]  = gl.createProgram();
gl.attachShader(this.prog[10], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[10] ));
gl.attachShader(this.prog[10], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[10] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[10], 0, "aPos");
gl.bindAttribLocation(this.prog[10], 1, "aCol");
gl.linkProgram(this.prog[10]);
texts = [
"sigma"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[10] = gl.getAttribLocation(this.prog[10], "aOfs");
this.texture[10] = gl.createTexture();
this.texLoc[10] = gl.getAttribLocation(this.prog[10], "aTexcoord");
this.sampler[10] = gl.getUniformLocation(this.prog[10],"uSampler");
this.handleLoadedTexture(10);
this.offsets[10]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.322, 4.5, -697.2096, 0, -0.5, 0.5, 0.5,
3.322, 4.5, -697.2096, 1, -0.5, 0.5, 0.5,
3.322, 4.5, -697.2096, 1, 1.5, 0.5, 0.5,
3.322, 4.5, -697.2096, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<1; i++)
for (j=0; j<4; j++) {
ind = this.offsets[10].stride*(4*i + j) + this.offsets[10].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[10] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3
]);
this.buf[10] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[10]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[10], gl.STATIC_DRAW);
this.ibuf[10] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[10]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[10] = gl.getUniformLocation(this.prog[10],"mvMatrix");
this.prMatLoc[10] = gl.getUniformLocation(this.prog[10],"prMatrix");
this.textScaleLoc[10] = gl.getUniformLocation(this.prog[10],"textScale");
// ****** text object 11 ******
this.flags[11] = 40;
this.vshaders[11] = "	/* ****** text object 11 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[11] = "	/* ****** text object 11 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[11]  = gl.createProgram();
gl.attachShader(this.prog[11], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[11] ));
gl.attachShader(this.prog[11], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[11] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[11], 0, "aPos");
gl.bindAttribLocation(this.prog[11], 1, "aCol");
gl.linkProgram(this.prog[11]);
texts = [
"loglik"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[11] = gl.getAttribLocation(this.prog[11], "aOfs");
this.texture[11] = gl.createTexture();
this.texLoc[11] = gl.getAttribLocation(this.prog[11], "aTexcoord");
this.sampler[11] = gl.getUniformLocation(this.prog[11],"uSampler");
this.handleLoadedTexture(11);
this.offsets[11]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.322, 1.1525, -589.3504, 0, -0.5, 0.5, 0.5,
3.322, 1.1525, -589.3504, 1, -0.5, 0.5, 0.5,
3.322, 1.1525, -589.3504, 1, 1.5, 0.5, 0.5,
3.322, 1.1525, -589.3504, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<1; i++)
for (j=0; j<4; j++) {
ind = this.offsets[11].stride*(4*i + j) + this.offsets[11].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[11] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3
]);
this.buf[11] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[11]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[11], gl.STATIC_DRAW);
this.ibuf[11] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[11]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[11] = gl.getUniformLocation(this.prog[11],"mvMatrix");
this.prMatLoc[11] = gl.getUniformLocation(this.prog[11],"prMatrix");
this.textScaleLoc[11] = gl.getUniformLocation(this.prog[11],"textScale");
// ****** lines object 12 ******
this.flags[12] = 128;
this.vshaders[12] = "	/* ****** lines object 12 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	void main(void) {\n	  vPosition = mvMatrix * vec4(aPos, 1.);\n	  gl_Position = prMatrix * vPosition;\n	  vCol = aCol;\n	}";
this.fshaders[12] = "	/* ****** lines object 12 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  gl_FragColor = lighteffect;\n	}";
this.prog[12]  = gl.createProgram();
gl.attachShader(this.prog[12], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[12] ));
gl.attachShader(this.prog[12], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[12] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[12], 0, "aPos");
gl.bindAttribLocation(this.prog[12], 1, "aCol");
gl.linkProgram(this.prog[12]);
this.offsets[12]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:-1, tofs:-1, stride:3};
v=new Float32Array([
4, 1.925, -672.319,
8, 1.925, -672.319,
4, 1.925, -672.319,
4, 1.79625, -676.4675,
5, 1.925, -672.319,
5, 1.79625, -676.4675,
6, 1.925, -672.319,
6, 1.79625, -676.4675,
7, 1.925, -672.319,
7, 1.79625, -676.4675,
8, 1.925, -672.319,
8, 1.79625, -676.4675
]);
this.values[12] = v;
this.buf[12] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[12]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[12], gl.STATIC_DRAW);
this.mvMatLoc[12] = gl.getUniformLocation(this.prog[12],"mvMatrix");
this.prMatLoc[12] = gl.getUniformLocation(this.prog[12],"prMatrix");
// ****** text object 13 ******
this.flags[13] = 40;
this.vshaders[13] = "	/* ****** text object 13 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[13] = "	/* ****** text object 13 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[13]  = gl.createProgram();
gl.attachShader(this.prog[13], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[13] ));
gl.attachShader(this.prog[13], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[13] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[13], 0, "aPos");
gl.bindAttribLocation(this.prog[13], 1, "aCol");
gl.linkProgram(this.prog[13]);
texts = [
"4",
"5",
"6",
"7",
"8"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[13] = gl.getAttribLocation(this.prog[13], "aOfs");
this.texture[13] = gl.createTexture();
this.texLoc[13] = gl.getAttribLocation(this.prog[13], "aTexcoord");
this.sampler[13] = gl.getUniformLocation(this.prog[13],"uSampler");
this.handleLoadedTexture(13);
this.offsets[13]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
4, 1.53875, -684.7643, 0, -0.5, 0.5, 0.5,
4, 1.53875, -684.7643, 1, -0.5, 0.5, 0.5,
4, 1.53875, -684.7643, 1, 1.5, 0.5, 0.5,
4, 1.53875, -684.7643, 0, 1.5, 0.5, 0.5,
5, 1.53875, -684.7643, 0, -0.5, 0.5, 0.5,
5, 1.53875, -684.7643, 1, -0.5, 0.5, 0.5,
5, 1.53875, -684.7643, 1, 1.5, 0.5, 0.5,
5, 1.53875, -684.7643, 0, 1.5, 0.5, 0.5,
6, 1.53875, -684.7643, 0, -0.5, 0.5, 0.5,
6, 1.53875, -684.7643, 1, -0.5, 0.5, 0.5,
6, 1.53875, -684.7643, 1, 1.5, 0.5, 0.5,
6, 1.53875, -684.7643, 0, 1.5, 0.5, 0.5,
7, 1.53875, -684.7643, 0, -0.5, 0.5, 0.5,
7, 1.53875, -684.7643, 1, -0.5, 0.5, 0.5,
7, 1.53875, -684.7643, 1, 1.5, 0.5, 0.5,
7, 1.53875, -684.7643, 0, 1.5, 0.5, 0.5,
8, 1.53875, -684.7643, 0, -0.5, 0.5, 0.5,
8, 1.53875, -684.7643, 1, -0.5, 0.5, 0.5,
8, 1.53875, -684.7643, 1, 1.5, 0.5, 0.5,
8, 1.53875, -684.7643, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<5; i++)
for (j=0; j<4; j++) {
ind = this.offsets[13].stride*(4*i + j) + this.offsets[13].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[13] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3,
4, 5, 6, 4, 6, 7,
8, 9, 10, 8, 10, 11,
12, 13, 14, 12, 14, 15,
16, 17, 18, 16, 18, 19
]);
this.buf[13] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[13]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[13], gl.STATIC_DRAW);
this.ibuf[13] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[13]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[13] = gl.getUniformLocation(this.prog[13],"mvMatrix");
this.prMatLoc[13] = gl.getUniformLocation(this.prog[13],"prMatrix");
this.textScaleLoc[13] = gl.getUniformLocation(this.prog[13],"textScale");
// ****** lines object 14 ******
this.flags[14] = 128;
this.vshaders[14] = "	/* ****** lines object 14 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	void main(void) {\n	  vPosition = mvMatrix * vec4(aPos, 1.);\n	  gl_Position = prMatrix * vPosition;\n	  vCol = aCol;\n	}";
this.fshaders[14] = "	/* ****** lines object 14 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  gl_FragColor = lighteffect;\n	}";
this.prog[14]  = gl.createProgram();
gl.attachShader(this.prog[14], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[14] ));
gl.attachShader(this.prog[14], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[14] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[14], 0, "aPos");
gl.bindAttribLocation(this.prog[14], 1, "aCol");
gl.linkProgram(this.prog[14]);
this.offsets[14]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:-1, tofs:-1, stride:3};
v=new Float32Array([
3.94, 2, -672.319,
3.94, 7, -672.319,
3.94, 2, -672.319,
3.837, 2, -676.4675,
3.94, 3, -672.319,
3.837, 3, -676.4675,
3.94, 4, -672.319,
3.837, 4, -676.4675,
3.94, 5, -672.319,
3.837, 5, -676.4675,
3.94, 6, -672.319,
3.837, 6, -676.4675,
3.94, 7, -672.319,
3.837, 7, -676.4675
]);
this.values[14] = v;
this.buf[14] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[14]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[14], gl.STATIC_DRAW);
this.mvMatLoc[14] = gl.getUniformLocation(this.prog[14],"mvMatrix");
this.prMatLoc[14] = gl.getUniformLocation(this.prog[14],"prMatrix");
// ****** text object 15 ******
this.flags[15] = 40;
this.vshaders[15] = "	/* ****** text object 15 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[15] = "	/* ****** text object 15 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[15]  = gl.createProgram();
gl.attachShader(this.prog[15], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[15] ));
gl.attachShader(this.prog[15], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[15] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[15], 0, "aPos");
gl.bindAttribLocation(this.prog[15], 1, "aCol");
gl.linkProgram(this.prog[15]);
texts = [
"2",
"3",
"4",
"5",
"6",
"7"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[15] = gl.getAttribLocation(this.prog[15], "aOfs");
this.texture[15] = gl.createTexture();
this.texLoc[15] = gl.getAttribLocation(this.prog[15], "aTexcoord");
this.sampler[15] = gl.getUniformLocation(this.prog[15],"uSampler");
this.handleLoadedTexture(15);
this.offsets[15]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.631, 2, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 2, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 2, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 2, -684.7643, 0, 1.5, 0.5, 0.5,
3.631, 3, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 3, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 3, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 3, -684.7643, 0, 1.5, 0.5, 0.5,
3.631, 4, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 4, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 4, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 4, -684.7643, 0, 1.5, 0.5, 0.5,
3.631, 5, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 5, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 5, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 5, -684.7643, 0, 1.5, 0.5, 0.5,
3.631, 6, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 6, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 6, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 6, -684.7643, 0, 1.5, 0.5, 0.5,
3.631, 7, -684.7643, 0, -0.5, 0.5, 0.5,
3.631, 7, -684.7643, 1, -0.5, 0.5, 0.5,
3.631, 7, -684.7643, 1, 1.5, 0.5, 0.5,
3.631, 7, -684.7643, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<6; i++)
for (j=0; j<4; j++) {
ind = this.offsets[15].stride*(4*i + j) + this.offsets[15].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[15] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3,
4, 5, 6, 4, 6, 7,
8, 9, 10, 8, 10, 11,
12, 13, 14, 12, 14, 15,
16, 17, 18, 16, 18, 19,
20, 21, 22, 20, 22, 23
]);
this.buf[15] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[15]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[15], gl.STATIC_DRAW);
this.ibuf[15] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[15]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[15] = gl.getUniformLocation(this.prog[15],"mvMatrix");
this.prMatLoc[15] = gl.getUniformLocation(this.prog[15],"prMatrix");
this.textScaleLoc[15] = gl.getUniformLocation(this.prog[15],"textScale");
// ****** lines object 16 ******
this.flags[16] = 128;
this.vshaders[16] = "	/* ****** lines object 16 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	void main(void) {\n	  vPosition = mvMatrix * vec4(aPos, 1.);\n	  gl_Position = prMatrix * vPosition;\n	  vCol = aCol;\n	}";
this.fshaders[16] = "	/* ****** lines object 16 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  gl_FragColor = lighteffect;\n	}";
this.prog[16]  = gl.createProgram();
gl.attachShader(this.prog[16], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[16] ));
gl.attachShader(this.prog[16], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[16] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[16], 0, "aPos");
gl.bindAttribLocation(this.prog[16], 1, "aCol");
gl.linkProgram(this.prog[16]);
this.offsets[16]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:-1, tofs:-1, stride:3};
v=new Float32Array([
3.94, 1.925, -650,
3.94, 1.925, -550,
3.94, 1.925, -650,
3.837, 1.79625, -650,
3.94, 1.925, -600,
3.837, 1.79625, -600,
3.94, 1.925, -550,
3.837, 1.79625, -550
]);
this.values[16] = v;
this.buf[16] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[16]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[16], gl.STATIC_DRAW);
this.mvMatLoc[16] = gl.getUniformLocation(this.prog[16],"mvMatrix");
this.prMatLoc[16] = gl.getUniformLocation(this.prog[16],"prMatrix");
// ****** text object 17 ******
this.flags[17] = 40;
this.vshaders[17] = "	/* ****** text object 17 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	attribute vec2 aTexcoord;\n	varying vec2 vTexcoord;\n	uniform vec2 textScale;\n	attribute vec2 aOfs;\n	void main(void) {\n	  vCol = aCol;\n	  vTexcoord = aTexcoord;\n	  vec4 pos = prMatrix * mvMatrix * vec4(aPos, 1.);\n	  pos = pos/pos.w;\n	  gl_Position = pos + vec4(aOfs*textScale, 0.,0.);\n	}";
this.fshaders[17] = "	/* ****** text object 17 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	varying vec2 vTexcoord;\n	uniform sampler2D uSampler;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  vec4 textureColor = lighteffect*texture2D(uSampler, vTexcoord);\n	  if (textureColor.a < 0.1)\n	    discard;\n	  else\n	    gl_FragColor = textureColor;\n	}";
this.prog[17]  = gl.createProgram();
gl.attachShader(this.prog[17], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[17] ));
gl.attachShader(this.prog[17], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[17] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[17], 0, "aPos");
gl.bindAttribLocation(this.prog[17], 1, "aCol");
gl.linkProgram(this.prog[17]);
texts = [
"-650",
"-600",
"-550"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[17] = gl.getAttribLocation(this.prog[17], "aOfs");
this.texture[17] = gl.createTexture();
this.texLoc[17] = gl.getAttribLocation(this.prog[17], "aTexcoord");
this.sampler[17] = gl.getUniformLocation(this.prog[17],"uSampler");
this.handleLoadedTexture(17);
this.offsets[17]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.631, 1.53875, -650, 0, -0.5, 0.5, 0.5,
3.631, 1.53875, -650, 1, -0.5, 0.5, 0.5,
3.631, 1.53875, -650, 1, 1.5, 0.5, 0.5,
3.631, 1.53875, -650, 0, 1.5, 0.5, 0.5,
3.631, 1.53875, -600, 0, -0.5, 0.5, 0.5,
3.631, 1.53875, -600, 1, -0.5, 0.5, 0.5,
3.631, 1.53875, -600, 1, 1.5, 0.5, 0.5,
3.631, 1.53875, -600, 0, 1.5, 0.5, 0.5,
3.631, 1.53875, -550, 0, -0.5, 0.5, 0.5,
3.631, 1.53875, -550, 1, -0.5, 0.5, 0.5,
3.631, 1.53875, -550, 1, 1.5, 0.5, 0.5,
3.631, 1.53875, -550, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<3; i++)
for (j=0; j<4; j++) {
ind = this.offsets[17].stride*(4*i + j) + this.offsets[17].tofs;
v[ind+2] = 2*(v[ind]-v[ind+2])*texinfo.widths[i];
v[ind+3] = 2*(v[ind+1]-v[ind+3])*texinfo.textHeight;
v[ind] *= texinfo.widths[i]/texinfo.canvasX;
v[ind+1] = 1.0-(texinfo.offset + i*texinfo.skip -
v[ind+1]*texinfo.textHeight)/texinfo.canvasY;
}
this.values[17] = v;
f=new Uint16Array([
0, 1, 2, 0, 2, 3,
4, 5, 6, 4, 6, 7,
8, 9, 10, 8, 10, 11
]);
this.buf[17] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[17]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[17], gl.STATIC_DRAW);
this.ibuf[17] = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[17]);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
this.mvMatLoc[17] = gl.getUniformLocation(this.prog[17],"mvMatrix");
this.prMatLoc[17] = gl.getUniformLocation(this.prog[17],"prMatrix");
this.textScaleLoc[17] = gl.getUniformLocation(this.prog[17],"textScale");
// ****** lines object 18 ******
this.flags[18] = 128;
this.vshaders[18] = "	/* ****** lines object 18 vertex shader ****** */\n	attribute vec3 aPos;\n	attribute vec4 aCol;\n	uniform mat4 mvMatrix;\n	uniform mat4 prMatrix;\n	varying vec4 vCol;\n	varying vec4 vPosition;\n	void main(void) {\n	  vPosition = mvMatrix * vec4(aPos, 1.);\n	  gl_Position = prMatrix * vPosition;\n	  vCol = aCol;\n	}";
this.fshaders[18] = "	/* ****** lines object 18 fragment shader ****** */\n	#ifdef GL_ES\n	precision highp float;\n	#endif\n	varying vec4 vCol; // carries alpha\n	varying vec4 vPosition;\n	void main(void) {\n      vec4 colDiff = vCol;\n	  vec4 lighteffect = colDiff;\n	  gl_FragColor = lighteffect;\n	}";
this.prog[18]  = gl.createProgram();
gl.attachShader(this.prog[18], this.getShader( gl, gl.VERTEX_SHADER, this.vshaders[18] ));
gl.attachShader(this.prog[18], this.getShader( gl, gl.FRAGMENT_SHADER, this.fshaders[18] ));
//  Force aPos to location 0, aCol to location 1
gl.bindAttribLocation(this.prog[18], 0, "aPos");
gl.bindAttribLocation(this.prog[18], 1, "aCol");
gl.linkProgram(this.prog[18]);
this.offsets[18]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:-1, tofs:-1, stride:3};
v=new Float32Array([
3.94, 1.925, -672.319,
3.94, 7.075, -672.319,
3.94, 1.925, -506.3818,
3.94, 7.075, -506.3818,
3.94, 1.925, -672.319,
3.94, 1.925, -506.3818,
3.94, 7.075, -672.319,
3.94, 7.075, -506.3818,
3.94, 1.925, -672.319,
8.06, 1.925, -672.319,
3.94, 1.925, -506.3818,
8.06, 1.925, -506.3818,
3.94, 7.075, -672.319,
8.06, 7.075, -672.319,
3.94, 7.075, -506.3818,
8.06, 7.075, -506.3818,
8.06, 1.925, -672.319,
8.06, 7.075, -672.319,
8.06, 1.925, -506.3818,
8.06, 7.075, -506.3818,
8.06, 1.925, -672.319,
8.06, 1.925, -506.3818,
8.06, 7.075, -672.319,
8.06, 7.075, -506.3818
]);
this.values[18] = v;
this.buf[18] = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[18]);
gl.bufferData(gl.ARRAY_BUFFER, this.values[18], gl.STATIC_DRAW);
this.mvMatLoc[18] = gl.getUniformLocation(this.prog[18],"mvMatrix");
this.prMatLoc[18] = gl.getUniformLocation(this.prog[18],"prMatrix");
gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);
gl.clearDepth(1.0);
gl.clearColor(1,1,1,1);
var drag  = 0;
this.drawScene = function() {
gl.depthMask(true);
gl.disable(gl.BLEND);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
this.drawFns[1].call(this, 1);
gl.flush();
};
// ****** points object 7 *******
this.drawFns[7] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.enableVertexAttribArray( posLoc );
gl.enableVertexAttribArray( colLoc );
gl.vertexAttribPointer(colLoc, 4, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].cofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawArrays(gl.POINTS, 0, 10000);
};
// ****** text object 9 *******
this.drawFns[9] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
};
// ****** text object 10 *******
this.drawFns[10] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
};
// ****** text object 11 *******
this.drawFns[11] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
};
// ****** lines object 12 *******
this.drawFns[12] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.lineWidth( 1 );
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawArrays(gl.LINES, 0, 12);
};
// ****** text object 13 *******
this.drawFns[13] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 30, gl.UNSIGNED_SHORT, 0);
};
// ****** lines object 14 *******
this.drawFns[14] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.lineWidth( 1 );
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawArrays(gl.LINES, 0, 14);
};
// ****** text object 15 *******
this.drawFns[15] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
};
// ****** lines object 16 *******
this.drawFns[16] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.lineWidth( 1 );
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawArrays(gl.LINES, 0, 8);
};
// ****** text object 17 *******
this.drawFns[17] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibuf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.uniform2f( this.textScaleLoc[id], 0.75/this.vp[2], 0.75/this.vp[3]);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.enableVertexAttribArray( this.texLoc[id] );
gl.vertexAttribPointer(this.texLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].tofs);
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, this.texture[id]);
gl.uniform1i( this.sampler[id], 0);
gl.enableVertexAttribArray( this.ofsLoc[id] );
gl.vertexAttribPointer(this.ofsLoc[id], 2, gl.FLOAT, false, 4*this.offsets[id].stride, 4*this.offsets[id].oofs);
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawElements(gl.TRIANGLES, 18, gl.UNSIGNED_SHORT, 0);
};
// ****** lines object 18 *******
this.drawFns[18] = function(id, clipplanes) {
var i;
gl.useProgram(this.prog[id]);
gl.bindBuffer(gl.ARRAY_BUFFER, this.buf[id]);
gl.uniformMatrix4fv( this.prMatLoc[id], false, new Float32Array(this.prMatrix.getAsArray()) );
gl.uniformMatrix4fv( this.mvMatLoc[id], false, new Float32Array(this.mvMatrix.getAsArray()) );
var clipcheck = 0;
for (i=0; i < clipplanes.length; i++)
clipcheck = this.clipFns[clipplanes[i]].call(this, clipplanes[i], id, clipcheck);
gl.enableVertexAttribArray( posLoc );
gl.disableVertexAttribArray( colLoc );
gl.vertexAttrib4f( colLoc, 0, 0, 0, 1 );
gl.lineWidth( 1 );
gl.vertexAttribPointer(posLoc,  3, gl.FLOAT, false, 4*this.offsets[id].stride,  4*this.offsets[id].vofs);
gl.drawArrays(gl.LINES, 0, 24);
};
// ***** subscene 1 ****
this.drawFns[1] = function(id) {
var i;
this.vp = this.viewport[id];
gl.viewport(this.vp[0], this.vp[1], this.vp[2], this.vp[3]);
gl.scissor(this.vp[0], this.vp[1], this.vp[2], this.vp[3]);
gl.clearColor(1, 1, 1, 1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
this.prMatrix.makeIdentity();
var radius = 88.67721,
distance = 394.5349,
t = tan(this.FOV[1]*PI/360),
near = distance - radius,
far = distance + radius,
hlen = t*near,
aspect = this.vp[2]/this.vp[3],
z = this.zoom[1];
if (aspect > 1)
this.prMatrix.frustum(-hlen*aspect*z, hlen*aspect*z,
-hlen*z, hlen*z, near, far);
else
this.prMatrix.frustum(-hlen*z, hlen*z,
-hlen*z/aspect, hlen*z/aspect,
near, far);
this.mvMatrix.makeIdentity();
this.mvMatrix.translate( -6, -4.5, 589.3504 );
this.mvMatrix.scale( 23.27173, 18.61739, 0.5778061 );
this.mvMatrix.multRight( unnamed_chunk_41rgl.userMatrix[1] );
this.mvMatrix.translate(-0, -0, -394.5349);
var clipids = this.clipplanes[id];
if (clipids.length > 0) {
this.invMatrix = new CanvasMatrix4(this.mvMatrix);
this.invMatrix.invert();
for (i = 0; i < this.clipplanes[id].length; i++)
this.drawFns[clipids[i]].call(this, clipids[i]);
}
var subids = this.opaque[id];
for (i = 0; i < subids.length; i++)
this.drawFns[subids[i]].call(this, subids[i], clipids);
subids = this.transparent[id];
if (subids.length > 0) {
gl.depthMask(false);
gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA,
gl.ONE, gl.ONE);
gl.enable(gl.BLEND);
for (i = 0; i < subids.length; i++)
this.drawFns[subids[i]].call(this, subids[i], clipids);
}
subids = this.subscenes[id];
for (i = 0; i < subids.length; i++)
this.drawFns[subids[i]].call(this, subids[i]);
};
this.drawScene();
var vpx0 = {
1: 0
};
var vpy0 = {
1: 0
};
var vpWidths = {
1: 672
};
var vpHeights = {
1: 480
};
var activeModel = {
1: 1
};
var activeProjection = {
1: 1
};
unnamed_chunk_41rgl.listeners = {
1: [ 1 ]
};
var whichSubscene = function(coords){
if (0 <= coords.x && coords.x <= 672 && 0 <= coords.y && coords.y <= 480) return(1);
return(1);
};
var translateCoords = function(subsceneid, coords){
return {x:coords.x - vpx0[subsceneid], y:coords.y - vpy0[subsceneid]};
};
var vlen = function(v) {
return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
};
var xprod = function(a, b) {
return [a[1]*b[2] - a[2]*b[1],
a[2]*b[0] - a[0]*b[2],
a[0]*b[1] - a[1]*b[0]];
};
var screenToVector = function(x, y) {
var width = vpWidths[activeSubscene],
height = vpHeights[activeSubscene],
radius = max(width, height)/2.0,
cx = width/2.0,
cy = height/2.0,
px = (x-cx)/radius,
py = (y-cy)/radius,
plen = sqrt(px*px+py*py);
if (plen > 1.e-6) {
px = px/plen;
py = py/plen;
}
var angle = (SQRT2 - plen)/SQRT2*PI/2,
z = sin(angle),
zlen = sqrt(1.0 - z*z);
px = px * zlen;
py = py * zlen;
return [px, py, z];
};
var rotBase;
var trackballdown = function(x,y) {
rotBase = screenToVector(x, y);
var l = unnamed_chunk_41rgl.listeners[activeModel[activeSubscene]];
saveMat = {};
for (var i = 0; i < l.length; i++)
saveMat[l[i]] = new CanvasMatrix4(unnamed_chunk_41rgl.userMatrix[l[i]]);
};
var trackballmove = function(x,y) {
var rotCurrent = screenToVector(x,y),
dot = rotBase[0]*rotCurrent[0] +
rotBase[1]*rotCurrent[1] +
rotBase[2]*rotCurrent[2],
angle = acos( dot/vlen(rotBase)/vlen(rotCurrent) )*180.0/PI,
axis = xprod(rotBase, rotCurrent),
l = unnamed_chunk_41rgl.listeners[activeModel[activeSubscene]],
i;
for (i = 0; i < l.length; i++) {
unnamed_chunk_41rgl.userMatrix[l[i]].load(saveMat[l[i]]);
unnamed_chunk_41rgl.userMatrix[l[i]].rotate(angle, axis[0], axis[1], axis[2]);
}
unnamed_chunk_41rgl.drawScene();
};
var trackballend = 0;
var y0zoom = 0;
var zoom0 = 0;
var zoomdown = function(x, y) {
y0zoom = y;
zoom0 = {};
l = unnamed_chunk_41rgl.listeners[activeProjection[activeSubscene]];
for (var i = 0; i < l.length; i++)
zoom0[l[i]] = log(unnamed_chunk_41rgl.zoom[l[i]]);
};
var zoommove = function(x, y) {
l = unnamed_chunk_41rgl.listeners[activeProjection[activeSubscene]];
for (var i = 0; i < l.length; i++)
unnamed_chunk_41rgl.zoom[l[i]] = exp(zoom0[l[i]] + (y-y0zoom)/height);
unnamed_chunk_41rgl.drawScene();
};
var zoomend = 0;
var y0fov = 0;
var fov0 = 0;
var fovdown = function(x, y) {
y0fov = y;
fov0 = {};
l = unnamed_chunk_41rgl.listeners[activeProjection[activeSubscene]];
for (i = 0; i < l.length; i++)
fov0[l[i]] = unnamed_chunk_41rgl.FOV[l[i]];
};
var fovmove = function(x, y) {
l = unnamed_chunk_41rgl.listeners[activeProjection[activeSubscene]];
for (i = 0; i < l.length; i++)
unnamed_chunk_41rgl.FOV[l[i]] = max(1, min(179, fov0[l[i]] + 180*(y-y0fov)/height));
unnamed_chunk_41rgl.drawScene();
};
var fovend = 0;
var mousedown = [trackballdown, zoomdown, fovdown];
var mousemove = [trackballmove, zoommove, fovmove];
var mouseend = [trackballend, zoomend, fovend];
function relMouseCoords(event){
var totalOffsetX = 0,
totalOffsetY = 0,
currentElement = canvas;
do{
totalOffsetX += currentElement.offsetLeft;
totalOffsetY += currentElement.offsetTop;
currentElement = currentElement.offsetParent;
}
while(currentElement);
var canvasX = event.pageX - totalOffsetX,
canvasY = event.pageY - totalOffsetY;
return {x:canvasX, y:canvasY};
}
canvas.onmousedown = function ( ev ){
if (!ev.which) // Use w3c defns in preference to MS
switch (ev.button) {
case 0: ev.which = 1; break;
case 1:
case 4: ev.which = 2; break;
case 2: ev.which = 3;
}
drag = ev.which;
var f = mousedown[drag-1];
if (f) {
var coords = relMouseCoords(ev);
coords.y = height-coords.y;
activeSubscene = whichSubscene(coords);
coords = translateCoords(activeSubscene, coords);
f(coords.x, coords.y);
ev.preventDefault();
}
};
canvas.onmouseup = function ( ev ){
if ( drag === 0 ) return;
var f = mouseend[drag-1];
if (f)
f();
drag = 0;
};
canvas.onmouseout = canvas.onmouseup;
canvas.onmousemove = function ( ev ){
if ( drag === 0 ) return;
var f = mousemove[drag-1];
if (f) {
var coords = relMouseCoords(ev);
coords.y = height - coords.y;
coords = translateCoords(activeSubscene, coords);
f(coords.x, coords.y);
}
};
var wheelHandler = function(ev) {
var del = 1.1;
if (ev.shiftKey) del = 1.01;
var ds = ((ev.detail || ev.wheelDelta) > 0) ? del : (1 / del);
l = unnamed_chunk_41rgl.listeners[activeProjection[activeSubscene]];
for (var i = 0; i < l.length; i++)
unnamed_chunk_41rgl.zoom[l[i]] *= ds;
unnamed_chunk_41rgl.drawScene();
ev.preventDefault();
};
canvas.addEventListener("DOMMouseScroll", wheelHandler, false);
canvas.addEventListener("mousewheel", wheelHandler, false);
};
</script>
<canvas id="unnamed_chunk_41canvas" class="rglWebGL" width="1" height="1"></canvas>
<p id="unnamed_chunk_41debug">
<img src="unnamed_chunk_41snapshot.png" alt="unnamed_chunk_41snapshot" width=673/><br>
You must enable Javascript to view this page properly.</p>
<script>unnamed_chunk_41rgl.start();</script>

Here our best estimate for our parameters $\theta = (\mu, \sigma^2)$ will be the pair of paramters that has the greatest log-likelihood:


```r
# find the approximate MLE
MLE_dsearch <- g[which.max(g$loglik), ]
MLE_dsearch
```

```
##           mu    sigma    loglik   color
## 2153 6.10101 3.060606 -508.7984 #FFFF00
```

### Optimization

Finding maxima and minima of functions is a common operation, and there are many algorithms that have been developed to accomplish these tasks. 
Some of these algorithms are included in the base R function `optim()`. 

Optimization routines have an easier time optimizing in unconstrained space, where parameters can be anywhere between $-\infty$ and $\infty$. 
However, we are trying to optimize a parameter that must be positive, $\sigma$. 
We can transform $\sigma$ so that we can optimize over unconstrained space: `log` maps sigma from its constrained space $(0, \infty)$ to unconstrained space $(-\infty, \infty)$, and the `exp` function maps from the unconstrained space back to the constrained space (and the scale of the parameter).
This trick shows up later in the context of link-functions for generalized linear models, where we transform a constrained linear predictor to unconstrained space while estimating parameters. 

We need to provide some initial values for the parameters and a function to minimize. 
If we find the minimum of the negative log-likelihood, then we have found the MLE. 


```r
# writing a negative log-likelihood function
nll <- function(theta, y){
  mu <- theta[1]
  sigma <- exp(theta[2]) 
  # theta[2] is the log of sigma, which is unconstrained
  # this simplifies optimization because we have no boundaries
  -sum(dnorm(y, mu, sigma, log=TRUE))
}

# initial guesses
theta_init <- c(mu = 4, log_sigma = 1)

# optimize
res <- optim(theta_init, nll, y=y)
res
```

```
## $par
##        mu log_sigma 
##  6.100774  1.125011 
## 
## $value
## [1] 508.7902
## 
## $counts
## function gradient 
##       57       NA 
## 
## $convergence
## [1] 0
## 
## $message
## NULL
```

If the algorithm has converged (check to see if `res$convergence` is zero), and if there is only one minimum in the negative log-likelihood surface (we know this is true based on analytical results in this case), then we have identified the MLEs of $\mu$ and $ln(\sigma)$.
How do these estimates compare to our first estimates found via direct search? 


```r
MLE_optim <- c(res$par[1], exp(res$par[2]))
rbind(unlist(MLE_dsearch[c('mu', 'sigma')]), 
      unlist(MLE_optim))
```

```
##            mu    sigma
## [1,] 6.101010 3.060606
## [2,] 6.100774 3.080252
```

This approach is quite general, and can be modified to be used for instance in a linear regression context: 


```r
n <- 20
x <- runif(n)
y <- rnorm(n, 3 + 10 * x, sd = 1)

nll <- function(theta, y, x){
  alpha <- theta[1]
  beta <- theta[2]
  sigma <- exp(theta[3]) 
  mu <- alpha + beta * x
  -sum(dnorm(y, mu, sigma, log=TRUE))
}

# initial guesses
theta_init <- c(alpha = 4, beta = 1, log_sigma = 1)

# optimize
res <- optim(theta_init, nll, y = y, x = x)
res
```

```
## $par
##     alpha      beta log_sigma 
## 3.3852794 9.4546543 0.1145456 
## 
## $value
## [1] 30.67179
## 
## $counts
## function gradient 
##      170       NA 
## 
## $convergence
## [1] 0
## 
## $message
## NULL
```

```r
plot(x, y)
abline(a = res$par['alpha'], b = res$par['beta'])
```

![](main_files/figure-html/unnamed-chunk-45-1.png) 

Direct search and optimization approaches for maximum likelihood estimation can be useful, but in this class we will rarely make use of direct search and optimization. 
However, the likelihood function and maximum likelihood estimation will continue to play a central role. 

## Further reading

Casella and Berger. 2002. *Statistical Inference*, Chapter 7.

Scholz FW. 2004. Maximum likelihood estimation, in *Encyclopedia of Statistical Sciences*.

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 18.


Chapter 3: Bayesian inference
==============================

## Big picture

In recent decades Bayesian inference has increasingly been used in ecology. 
A key difference between Bayesian and maximum likelihood approaches lies in which quantities are treated as random variables. 
For any likelihood function, the parameters $\theta$ are not random variables because there are no probability distributions associated with $\theta$. 
In contrast, Bayesian approaches treat parameters and future data (actually all unobserved quantities) as random variables, meaning that each unknown is associated with a probability distribution $p(\theta)$.
In both Bayesian and maximum likelihood approaches, the observations $y$ are treated as a random variable, and the probability distribution for this random variable $p(y \mid \theta)$ plays a central role in both approaches. 
See Hobbs and Hooten (Ch. 5) for a more detailed treatment on differences between Bayesian and maximum likelihood based inferential approaches.
Some authors also point out differences between Bayesian and frequentist definitions of probability. 
In a frequentist framework, probabilities are defined in term of long run frequencies of events, often relying on hypothetical realizations. 
Bayesian probability definitions do not rely on the long-run frequency of events. 

Philosophy aside, Bayesian approaches have become more popular because of intuitive appeal and practical advantages. 
Intuitively, it can seem backwards to focus on the probability of the data given the parameters $p(y \mid \theta)$. 
What we really want to know is the probability of the parameters, having observed some data $p(\theta \mid y)$. 
As we will see, Bayes' theorem allows us to calculate this probability. 
Second, Bayesian approaches are often easier to implement than maximum likelihood or frequentist approaches, particularly for complex models. 
Finally, we find that in many applications, Bayesian approaches facilitate a better understanding of model structure and assumptions.

#### Learning goals

- Bayes' theorem and Bayesian probability
- relationship between likelihood and Bayesian inference
- priors (generally, informative vs. non-informative)
- proper vs. improper priors 
- intro to Bayesian computation and MCMC
- posterior summaries and comparisons
- single parameter models: MLE vs. Bayesian treatments
- Bayesian linear regression: intro to Stan

## Bayes' theorem

Bayes' theorem is an incredibly powerful theorem that follows from the rules of probability. 
To prove the theorem, we need only a few ingredients: 1) the definition of joint probabilities $p(A, B) = p(A \mid B) p(B)$ or $p(A, B) = p(B \mid A)p(A)$ (both are valid) and 2) a bit of algebra.

$$p(A, B) = p(A \mid B) p(B)$$

$$p(B \mid A)p(A) = p(A \mid B) p(B)$$

$$p(B \mid A) = \dfrac{p(A \mid B) p(B)}{p(A)}$$

This is Bayes' theorem. 
In modern applications, we typically substitute unknown parameters $\theta$ for $B$, and data $y$ for $A$:

$$p(\theta \mid y) = \dfrac{p(y \mid \theta) p(\theta)}{p(y)}$$

The terms can verbally be described as follows: 

- $p(\theta \mid y)$: the *posterior* distribution of the parameters. This tells us what the parameters probably are (and are not), conditioned on having observed some data $y$. 
- $p(y \mid \theta)$: the likelihood of the data $y$.
- $p(\theta)$: the *prior* distribution of the parameters. This should represent our prior knowledge about the values of the parameters. Prior knowledge comes from similar studies and/or first principles. 
- $p(y)$: the marginal distribution of the data. This quantity can be difficult or even impossible to compute, will always be a constant after the data have been observed, and is often ignored. 

Because $p(y)$ is a constant, it is valid and common to consider the posterior distribution up to this proportionality constant: 

$$p(\theta \mid y) \propto p(y \mid \theta) p(\theta)$$

## Prior distributions

We have already learned about likelihood, but the introduction of a prior distribution for the parameters requires some attention. 
The inclusion of prior information is not unique to Bayesian inference. 
When selecting study systems, designing experiments, cleaning or subsetting data, and choosing a likelihood function, we inevitably draw upon our previous knowledge of a system. 

From a Bayesian perspective, prior distributions $p(\theta)$ represent our knowledge/beliefs about parameters before having observed the data $y$, and the posterior distribution represents our updated knowledge/beliefs about parameters after having observed our data. 
This is similar to the way many scientists operate: we think we know something about a system, and then we do experiments and conduct surveys to update our knowledge about the system. 
But, the observations generated from the experiments and surveys are not considered in isolation. 
They are considered in the context of our previous knowledge.
In this way, the posterior distribution represents a compromise between our prior beliefs and the likelihood. 

### Analytical posterior with conjugate priors: Bernoulli case*

\* *this section is a bit math-heavy for illustration, but most of the time we won't find the posterior analytically*

The Bernoulli distribution is a probability distribution for binary random variables (e.g., those that take one of two values: dead or alive, male or female, heads or tails, 0 or 1, success or failure, and so on). 
The Bernoulli distribution has one parameter, $p:$ the probability of "success" (or more generally, the probability of one of the two outcomes) in one particular event. 
A Bernoulli random variable takes one of these two values.
The choice of which of the two possible outcomes is considered "success" is often arbitrary - we could consider either "heads" or "tails" to be a success if we wanted to. 
If $p$ is the probability of success in one particular trial, then the probability of failure is just the complement of $p:$ $1 - p$, sometimes referred to as $q$, such that $p + q = 1$.
For those familiar with the Binomial distribution, the Bernoulli distribution is a special case of the Binomial, with one trial $k=1$. 
We can use the Bernoulli distribution as a likelihood function, where $y$ is either zero or one: $y \in \{0, 1\}$, and $p$ is our only parameter. 
Because p is a probability, we know that $0 \leq p \leq 1$.

We can express the likelihood for a Bernoulli random variable as follows. 

$$p(y \mid p) = \begin{cases} p &\mbox{if } y = 1 \\ 
1-p & \mbox{if } y = 0 \end{cases}$$

Equivalently and more generally:

$$[y \mid p] = p^y (1-p)^{1-y}$$

If we have $n$ independent Bernoulli random variables, $y_1, ..., y_n$, each with probability $p$, then the joint likelihood can be written as the product of the point-wise likelihoods:

$$[y_1, ..., y_n \mid p] = p^{y_1} (1-p)^{1 - y_1} ... p^{y_n} (1 - p)^{1 - y_n}$$

$$[\pmb{y} \mid p] = \prod_{i = 1}^{n} p^{y_i} (1 - p)^{1 - y_1}$$

Recalling from algebra that $x^a x^b = x^{a + b}$, this implies:

$$[\pmb{y} \mid p] = p^{\sum_i y_i} (1 - p)^{n - \sum_i y_i}$$

Having obtained the likelihood, we now must specify a prior to complete our specification of the joint distribution of data and parameters $[y \mid p][p]$, the numerator in Bayes' theorem. 
A natural choice is the Beta distribution, which has two parameters $\alpha$ and $\beta$, with support on the interval $(0, 1)$. 
This is a good choice because its bounds are similar to those for probabilities, and the posterior induced by the prior is also a Beta distribution.
When a prior distribution for a parameter induces a posterior distribution that is of the same form (same probability distribution) as the prior, the prior is said to be a "conjugate prior" for the likelihood. 
The density of the beta distribution for parameter $p$ has two parameters $\alpha$ and $\beta$ and is defined as: 

$$[p] = c p^{\alpha - 1} (1 - p)^{\beta - 1}$$

Where $c$ is a constant that ensures that $[p \mid \alpha, \beta]$ integrates to one over the interval $(0, 1)$ (i.e., it is a true probability distribution), with $c=\dfrac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}$, and $\Gamma(x) = (x - 1)!$
That's a factorial symbol ($!$), not a punctuation mark!
To give a bit of intuition for what the beta distribution looks like, here are some plots of the beta density with different values of $\alpha$ and $\beta$:


```r
alpha <- rep(c(1, 5, 10))
beta <- rep(c(1, 5, 10))
g <- expand.grid(alpha=alpha, beta=beta)
x <- seq(0, 1, .005)

par(mfrow=c(3, 3))
for (i in 1:nrow(g)){
  plot(x, dbeta(x, g$alpha[i], g$beta[i]), 
       type='l', ylim=c(0, 10), 
       ylab="[x]", lwd=3)
  title(bquote(alpha == .(g$alpha[i]) ~ ", " ~ beta == .(g$beta[i])))
}
```

![](main_files/figure-html/unnamed-chunk-46-1.png) 

One commonly used prior is the beta(1, 1), because it corresponds to a uniform prior for $p$ (shown in the top left corner).
Now we have all of the ingredients to embark on our first Bayesian analysis for a Bernoulli random variable. 
We'll proceed by finding the posterior distribution up to some proportionality constant, then we'll use our knowledge of the beta distribution to recover the correct proportionality constant:

$$[p | y] = \dfrac{[y \mid p] [p]}{[y]}$$

$$[p | y] \propto [y \mid p] [p]$$

Plugging in the likelihood and prior that we described above:

$$[p | y] \propto p^{\sum_i y_i} (1 - p)^{n - \sum_i y_i} [p]$$

$$[p | y] \propto p^{\sum_i y_i} (1 - p)^{n - \sum_i y_i} c p^{\alpha - 1} (1 - p)^{\beta - 1}$$

Dropping $c$, because we're only working up to some proportionality constant:

$$[p | y] \propto p^{\sum_i y_i} (1 - p)^{n - \sum_i y_i} p^{\alpha - 1} (1 - p)^{\beta - 1}$$

Then, again recalling that $x^a x^b = x^{a + b}$, we find that

$$[p | y] \propto p^{\alpha -1 + \sum_i y_i} (1 - p)^{\beta - 1 +n - \sum_i y_i}$$

Notice that this is of the same form as the beta prior for $p$, with updated parameters: $\alpha_{post} = \alpha + \sum_i y_i$ and $\beta_{post} = \beta + n - \sum_i y_i$. 
In this sense, the parameters of the beta prior $\alpha$ and $\beta$ can be interpreted as the previous number of successes and failures, respectively. 
Future studies can simply use the updated values $\alpha_{post}$ and $\beta_{post}$ as priors.
We have found a quantity that is proportional to the posterior distribution, which often is enough, but here we can easily derive the proportionality constant that will ensure that the posterior integrates to one (i.e., it is a true probability distribution).

Recall the proportionality constant $c$ from the beta distribution prior that we used, which is $c=\dfrac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}$. 
Updating this proportionality constant gives us the correct value for our posterior distribution, ensuring that it integrates to one, so that we now can write down the posterior distribution in closed form:

$$[p | y] = \frac{\Gamma(\alpha_{post} + \beta_{post})}{\Gamma(\alpha_{post})\Gamma(\beta_{post})} p^{\alpha_{post} -1} (1 - p)^{\beta_{post} - 1}$$

Now we can explore the effect of our prior distributions on the posterior. 
Suppose that we have observed $n=8$ data points, $y_1, y_2, ..., y_{10}$, with $y_i \in \{0, 1\}$ and 2 successes, $\sum_i y_i = 2$. 
Let's graph the posterior distributions that are implied by the priors plotted above and the likelihood resulting from these observations.


```r
g$alpha_post <- g$alpha + 2
g$beta_post <- g$beta + 6
  
par(mfrow=c(3, 3))
for (i in 1:nrow(g)){
  plot(x, dbeta(x, g$alpha[i], g$beta[i]), 
       type='l', ylim=c(0, 10), 
       xlab="p", ylab="[p | y]", lwd=3, col='grey')
  lines(x, dbeta(x, g$alpha_post[i], g$beta_post[i]), 
        lwd=3, col='blue')
  lines(x, 8 * dbinom(x=2, size=8, prob=x), col='red')
  title(bquote(alpha == .(g$alpha[i]) ~ ", " ~ beta == .(g$beta[i])))
}
```

![](main_files/figure-html/unnamed-chunk-47-1.png) 

Here the prior is shown in grey, the likelihood is shown in red, and the posterior distribution is shown in blue.
Notice how strong priors pull the posterior distribution toward them, and weak priors result in posteriors that are mostly affected by the data. 
The majority of Bayesian statisticians nowadays advocate for the inclusion of reasonable prior distributions rather than prior distributions that feign ignorance. 
One nice feature of Bayesian approaches is that, given enough data, the prior tends to have less of an influence on the posterior. 
This is consistent with the notion that when reasonable people are presented with strong evidence, they tend to more or less agree even if they may have disagreed ahead of time (though at times the empirical evidence for this phenomenon may seem somewhat equivocal).
To show this, let's increase the amount of information in our data, so that we have $n=800$ and $\sum y = 200$ successes. 


```r
g$alpha_post <- g$alpha + 200
g$beta_post <- g$beta + 600
  
par(mfrow=c(3, 3))
for (i in 1:nrow(g)){
  plot(x, dbeta(x, g$alpha[i], g$beta[i]), 
       type='l', ylim=c(0, 32), 
       xlab="p", ylab="[p | y]", col='grey', lwd=2)
  lines(x, dbeta(x, g$alpha_post[i], g$beta_post[i]), 
        col='blue', lwd=2)
    lines(x, 800 * dbinom(x=200, size=800, prob=x), col='red')
  title(bquote(alpha == .(g$alpha[i]) ~ ", " ~ beta == .(g$beta[i])))
}
```

![](main_files/figure-html/unnamed-chunk-48-1.png) 

Here again the prior is shown in grey, the likelihood is shown in red, and the posterior distribution is shown in blue.
The posterior distribution has essentially the same shape as the likelihood because we have much more information in this larger data set relative to our priors. 

### Improper priors

Improper priors do not integrate to one (they do not define proper probability distributions). 
For instance, a normal distribution with infinite variance will not integrate to one. 
Sometimes improper priors can still lead to proper posteriors, but this must be checked analytically. 
Unless you're willing to prove that an improper prior leads to a proper posterior distribution, we recommend using proper priors. 

## Posterior computation the easy way

In reality, most of the time we don't analytically derive posterior distributions. 
Mostly, we can express our models in specialized programming languages that are designed to make Bayesian inference easier. 
Here is a Stan model statement from the above example. 

```
data {
  int n;
  int<lower=0, upper=1> y[n];
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
}
```

The above model statement has three "blocks". 
The data block specifies that we have two fixed inputs: the sample size $n$ and a vector of integer values with $n$ elements, and these have to be either zero or one. 
The parameter block specifies that we have one parameter $p$, which is a real number between 0 and 1. 
Last, the model block contains our beta prior for $p$ and our Bernoulli likelihood for the data $y$. 

Stan does not find the analytic expression for the posterior.
Rather, Stan translates the above program into a Markov chain Monte Carlo (MCMC) algorithm to simulate samples from the posterior distribution. 
In practice, this is sufficient to learn about the model parameters. 

### What is MCMC? 

MCMC is used to sample from probability distributions generally, and from posterior probability distributions in the context of Bayesian inference. 
This is a huge topic, and involves a fair bit of theory that we will not dwell on here, but it is worth reading Chapter 18 in Gelman and Hill for a taste of some of the algorithms that are used. 
In this class, we will rely on specialized software to conduct MCMC simulations, so that many of the details are left "under the hood". 
However, it is still important to know what MCMC is (supposed to be) doing, and how to identify when MCMC algorithms fail. 

In a Bayesian context, we often run multiple Markov chain simulations, where we iteratively update our parameter vector $\theta$. 
If all goes well, then eventually each Markov chain converges to the posterior distribution of the parameters, so that every draw can be considered a simulated sample from the posterior distribution. 
Typically, we initialize the chains at different (often random) points in parameter space. 
If all goes well, then after some number of iterations, every chain has converged to the posterior distribution, and we run the chains a bit longer to generate a representative sample from the posterior, then perform inference on our sample. 
Here is what this looks like in a 2-d (two parameter) model of the mean situation.

![](ch3/metrop.gif)

Notice that the chains start dispersed in parameter space and eventually all converge to the same region and stay there. 
After some large number of iterations, the estimated posterior density stabilizes. 
At this point, usually the early draws from the Markov chains are discarded as "warmup" or "burnin", as these do not represent draws from the posterior, because the chains had not converged.

It is always necessary to run diagnostics on MCMC output to ensure convergence. 
Some of these are graphical. 
Traceplots show the parameter values that a Markov chain has taken on the y-axis, and iteration number on the x-axis. 
For instance, if we run our Stan model from before:


```r
library(rstan)
m <- '
data {
  int n;
  int<lower=0, upper=1> y[n];
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
}
'

y <- c(1, 1, 0, 0, 0, 0, 0, 0)
n <- length(y)

out <- stan(model_code=m)
```

```
## 
## SAMPLING FOR MODEL '8dbabac1e0f3c5dfe4a2c8e37a6f8ed1' NOW (CHAIN 1).
## 
## Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
## Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
## Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
## Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
## Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
## Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.005603 seconds (Warm-up)
## #                0.005236 seconds (Sampling)
## #                0.010839 seconds (Total)
## 
## 
## SAMPLING FOR MODEL '8dbabac1e0f3c5dfe4a2c8e37a6f8ed1' NOW (CHAIN 2).
## 
## Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
## Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
## Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
## Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
## Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
## Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.005659 seconds (Warm-up)
## #                0.005267 seconds (Sampling)
## #                0.010926 seconds (Total)
## 
## 
## SAMPLING FOR MODEL '8dbabac1e0f3c5dfe4a2c8e37a6f8ed1' NOW (CHAIN 3).
## 
## Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
## Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
## Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
## Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
## Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
## Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.005656 seconds (Warm-up)
## #                0.005929 seconds (Sampling)
## #                0.011585 seconds (Total)
## 
## 
## SAMPLING FOR MODEL '8dbabac1e0f3c5dfe4a2c8e37a6f8ed1' NOW (CHAIN 4).
## 
## Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
## Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
## Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
## Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
## Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
## Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
## Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
## Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
## Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
## Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
## Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
## Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
## #  Elapsed Time: 0.005659 seconds (Warm-up)
## #                0.005253 seconds (Sampling)
## #                0.010912 seconds (Total)
```

```r
traceplot(out, inc_warmup=TRUE)
```

![](main_files/figure-html/unnamed-chunk-49-1.png) 

This traceplot is useful for verifying convergence: all of the chains appear to be sampling from the same region. 
We can also inspect some numerical summaries that are used to detect non-convergence. 
Specifically, we can look at the $\hat{R}$ statistic.
If $\hat{R} > 1.1$, then we ought to be worried about convergence of our chains. 
Printing our model output returns this statistic as well as some other summary statistics for the posterior draws.


```r
out
```

```
## Inference for Stan model: 8dbabac1e0f3c5dfe4a2c8e37a6f8ed1.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##       mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
## p     0.30    0.00 0.14  0.08  0.20  0.29  0.39  0.61  1293    1
## lp__ -6.63    0.02 0.76 -8.78 -6.78 -6.34 -6.16 -6.11  1017    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:13:11 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

This output also tells us that we ran four MCMC chains, each for 2000 iterations, and the first 1000 iterations were discarded as warmup (the shaded region in the traceplot). 
For each parameter (and for the log probability up to a proportionality constant `lp__`), we get the posterior mean, the MCMC standard error of the mean, the posterior standard deviation, some quantiles, and an estimate for the number of effective samples from the posterior `n_eff`, which should typically be at least in the hundreds.

### Example: normal linear models

Recall that all normal linear models can be expressed as: 

$$y \sim N(X \beta, \sigma^2)$$

To complete a Bayesian analysis, we need to select prior distributions for the unknown parameters $\beta$ and $\sigma^2$. 
For instance:

$$\beta \sim N(0, 5)$$

$$\sigma \sim halfNormal(0, 5)$$,

where the half-Normal with mean zero is a folded Gaussian probability density function with only positive mass:

![](main_files/figure-html/unnamed-chunk-51-1.png) 

Let's do a quick linear regression model and summarize/plot the results.


```r
n <- 20
x <- runif(n, 0, 3)
y <- rnorm(n, -3 + .75 * x, 1)
plot(x, y)
```

![](main_files/figure-html/unnamed-chunk-52-1.png) 

We'll use Stan again, but this time instead of specifying an object (`m` above) that is a character string containing our model statement, we'll save the model file somewhere else with a `.stan` file extension.
For instance, maybe we have a general purpose Stan model that can be used for linear models called `lm.stan`:


```c
data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  vector[n] y;
}

parameters {
  vector[p] beta;
  real<lower=0> sigma;
}

model {
  beta ~ normal(0, 5);
  sigma ~ normal(0, 5);
  y ~ normal(X * beta, sigma);
}
```

So we have this file saved somewhere as `lm.stan`, and it can fit any of the linear models that we covered in Chapter 1 by changing the design matrix, but now we can include information to improve our estimates. 
Because this is a simulated example, we'll use somewhat vague priors. 
Fitting the model:


```r
library(rstan)
X <- matrix(c(rep(1, n), x), ncol = 2)
stan_d <- list(n = nrow(X), p = ncol(X), X = X, y = y)
out <- stan('lm.stan', data = stan_d)
```

There are also some other default plots which are nice:


```r
traceplot(out)
```

![](main_files/figure-html/unnamed-chunk-55-1.png) 

```r
plot(out)
```

```
## Showing 80% intervals
```

![](main_files/figure-html/unnamed-chunk-55-2.png) 

```r
pairs(out)
```

![](main_files/figure-html/unnamed-chunk-55-3.png) 

Notice that the slopes and intercepts are correlated in the posterior (do you recall why?). 
Also, `lp__` is tracked automatically, and this is proportional to the log probability of the posterior distribution. 

Let's inspect the output in table form:


```r
out
```

```
## Inference for Stan model: lm.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##          mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
## beta[1] -3.04    0.01 0.28 -3.60 -3.21 -3.04 -2.86 -2.49  1046 1.01
## beta[2]  0.92    0.01 0.18  0.56  0.80  0.92  1.04  1.28  1088 1.01
## sigma    0.74    0.00 0.13  0.54  0.65  0.72  0.81  1.05  1304 1.00
## lp__    -3.68    0.04 1.27 -7.01 -4.24 -3.32 -2.77 -2.27  1215 1.00
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:13:38 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

And finally, let's extract our samples from the posterior and plot our estimated line of best fit. 


```r
library(scales)
post <- extract(out)

# draw points
plot(x, y)

# add a line for each draw from the posterior
n_iter <- length(post$lp__)
for (i in 1:n_iter){
  abline(post$beta[i, 1], post$beta[i, 2], col=alpha('dodgerblue', .05))
}

# add points again so that they are visible over the line
points(x, y, pch=19)
```

![](main_files/figure-html/unnamed-chunk-57-1.png) 

We might be particularly interested in the effect of x on y. 
If we have abandoned frequentism, then how might we think about this? 
One way **not** to think of it is asking what the probability is that the slope is exactly equal to zero. 
If we think of a posterior probability density of our slope, then the true probability of any one particular value is zero because this is a probability density function, not a probability mass function (this is true for $\beta=0$ and for $\beta = .0112351351$, for instance). 


```r
plot(density(post$beta[, 2]))
```

![](main_files/figure-html/unnamed-chunk-58-1.png) 

A better question (answerable with a probability density function) might be: given the data, what is the probability that $x$ has a positive effect on $y$?
This is equivalent to asking about the area to the right of zero in the posterior distribution of $\beta$, and the number is approximated simply the proportion of posterior draws greater than zero. 


```r
mean(post$beta[, 2] > 0)
```

```
## [1] 1
```

```r
ord <- order(post$beta[, 2])
plot(post$beta[ord, 2], 
     col=ifelse(post$beta[ord, 2] > 0, 'red', 'black'), 
     ylab='Sorted posterior draws for the slope')
```

![](main_files/figure-html/unnamed-chunk-59-1.png) 

We might also construct a 95% credible interval for our estimate to communicate uncertainty in our estimate of $\beta$.


```r
quantile(post$beta[, 2], probs=c(0.025, 0.975))
```

```
##      2.5%     97.5% 
## 0.5551515 1.2763808
```

Conditional on the data, there is a 95% probability that the true parameter value is in the credible interval. 
This is different from the interpretation of frequentist confidence intervals, which relies on long-run frequencies for imaginary realizations of the data collection and interval estimation procedure. 
Our confidence in confidence intervals is in the procedure of creating confidence intervals - in the hypothetical long run, 95% of the intervals that we construct will contain the true value. 
As pointed out [here](http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/), this is the right answer to the wrong question. 

The credible interval constructed above has equal probability density on either side of the interval because we based our interval end-points on quantiles. 
Sometimes, we may wish to instead construct an interval that is as narrow as possible while encapsulating some proportion of the probability mass. 
These intervals are called highest density posterior intervals, and can be constructed using the following function:


```r
HDI <- function(values, percent=0.95){
  sorted <- sort(values)
  index <- floor(percent * length(sorted))
  nCI <- length(sorted) - index
  
  width <- rep(0, nCI)
  for (i in 1:nCI){
    width[i] <- sorted[i + index] - sorted[i]
  }
  
  HDImin <- sorted[which.min(width)]
  HDImax <- sorted[which.min(width) + index]
  HDIlim <- c(HDImin, HDImax)
  return(HDIlim)
}

HDI(post$beta[, 2])
```

```
## [1] 0.5638519 1.2790790
```

In this instance, our posterior is fairly symmetric and the two types of credible intervals will not be much different. 

## Further reading

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 18.

Hobbs and Hooten. 2015. *Bayesian models: a statistical primer for ecologists*. Chapter 7. 

Gelman et al. 2014. *Bayesian data analysis. Third edition*. Chapter 1-3.

Ellison AM. 2004. Bayesian inference in Ecology. Ecology Letters 7: 509-520.


Chapter 4: Poisson models
==============

## Big picture

The Poisson distribution is often used for discrete counts, as it has integer support. 
In this chapter, we show how to extend the methods of Chapter 3 to models that use Poisson likelihoods, and we point out connections to Poisson-family generalized linear models. 
Also, we make use of link functions as maps between constrained and unconstrained spaces, and show how to include overdispersion in Poisson models, which is often necessary in real-world applications.

#### Learning goals

- parameters and behavior of the Poisson distribution
- log link as a map
- effects sizes in Poisson models
- dependence of mean and variance in non-Gaussian models 
- overdispersion: Poisson and negative-binomial
- implementation with `glm` and Stan
- graphical displays 
- model checking
- simulation of data & parameter recovery

## Poisson generalized linear models

We have covered general linear models, which by definition assume normally distributed response variables, and in this chapter we cover our first generalized linear model. 
Generalized linear models (glms) allow for other response distributions, including the Poisson distribution. 
This is admittedly a subtle distinction. 

Poisson glms are defined as follows:

$$y \sim Poisson(\lambda)$$

$$log(\lambda) = X \beta$$

Where the log-link is used to map the Poisson distribution's only parameter $\lambda$ from it's constrained space $[0, \infty)$ to the unconstrained space $(-\infty, \infty)$, and $X$ is a design matrix as before. 
Recall that $\lambda$ is the mean and variance of the Poisson distribution, in contrast for example to the Gaussian distribution which has separate parameters for the first and second moments.

Sometimes, it's desirable to include an offset. 
For instance, perhaps $y$ is the number of events per unit time, or per square-meter, and each observation has different time intervals or areas sampled. 
Then, we might be interested in modeling the rate of events per unit (time or meters square), e.g., $\dfrac{\lambda}{t}$, which is in units events per time.

$$log(\lambda / t) = X \beta$$

$$log(\lambda) - log(t) = X \beta$$

$$log(\lambda) = X \beta + log(t)$$

So that now, we can still rely on $y \sim Poisson(\lambda)$, even with varying $t$.

The class of models that utilize the Poisson distribution is far broader than what we can do with glms, but glms provide a common and simple starting point. 

## Simulation and estimation

We will demonstrate with a Poisson regression, but maintain the generality of our notation by continuing to make use of the design matrix $X$, so that the applicability to ANOVA, ANCOVA, etc. types of applications remains apparent. 


```r
n <- 50
x <- runif(n, -1, 1)
X <- matrix(c(rep(1, n), x), ncol=2)
beta <- c(.5, -1)
lambda <- exp(X %*% beta)
y <- rpois(n, lambda)
```

### Estimation with `glm`

We can obtain maximum likelihood estimates for Poisson glm coefficients, along with frequentist p-values and confidence intervals with the `glm` function. 


```r
m <- glm(y ~ x, family=poisson)
summary(m)
```

```
## 
## Call:
## glm(formula = y ~ x, family = poisson)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.2214  -0.9030  -0.2783   0.1801   3.1431  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)   0.5187     0.1123   4.621 3.82e-06 ***
## x            -0.7238     0.2125  -3.406 0.000659 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 81.821  on 49  degrees of freedom
## Residual deviance: 69.714  on 48  degrees of freedom
## AIC: 174.53
## 
## Number of Fisher Scoring iterations: 5
```

```r
confint(m)
```

```
## Waiting for profiling to be done...
```

```
##                  2.5 %     97.5 %
## (Intercept)  0.2893691  0.7302752
## x           -1.1475243 -0.3128909
```

Plotting the line of best fit with the data: 


```r
plot(x, y)

xnew <- seq(min(x), max(x), length.out = 100)
lines(xnew, exp(coef(m)[1] + coef(m)[2] * xnew))
```

![](main_files/figure-html/unnamed-chunk-64-1.png) 

### Estimation with Stan

Even though Poisson glms are often taught after Gaussian glms, they are in a way simpler, requiring one less parameter. 
Here is a Stan file `poisson_glm.stan` that can be used generally to fit any Poisson glm without an offset. 

```
data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
}

parameters {
  vector[p] beta;
}

model {
  beta ~ normal(0, 3);
  y ~ poisson_log(X * beta);
}
```

Fitting the model is the same as before:


```r
library(rstan)
X <- matrix(c(rep(1, n), x), ncol = 2)
stan_d <- list(n = nrow(X), p = ncol(X), X = X, y = y)
out <- stan('poisson_glm.stan', data = stan_d)
```

As always, it behooves us evaluate convergence and see whether there may be issues with the MCMC algorithm. 


```r
out
```

```
## Inference for Stan model: poisson_glm.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##           mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## beta[1]   0.51    0.00 0.11   0.28   0.44   0.51   0.59   0.72  1653    1
## beta[2]  -0.73    0.00 0.21  -1.14  -0.88  -0.73  -0.59  -0.33  2061    1
## lp__    -32.64    0.03 1.00 -35.34 -32.98 -32.32 -31.96 -31.70  1406    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:14:11 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(out)
```

![](main_files/figure-html/unnamed-chunk-66-1.png) 

```r
plot(out)
```

```
## Showing 80% intervals
```

![](main_files/figure-html/unnamed-chunk-66-2.png) 

```r
pairs(out)
```

![](main_files/figure-html/unnamed-chunk-66-3.png) 

Let's plot the lines of best fit from the posterior distribution:


```r
library(scales)
post <- extract(out)
plot(x, y)
n_iter <- length(post$lp__)
for (i in 1:n_iter){
  lines(xnew, exp(post$beta[i, 1] + post$beta[i, 2] * xnew), 
        col = alpha('purple3', .01))
}
points(x, y, pch=19)
```

![](main_files/figure-html/unnamed-chunk-67-1.png) 

At times, people get confused about why the fits from Poisson glms are not linear, even though we didn't include any polynomial terms. 
The model is linear on the link-scale only, so that if we were to plot x vs. $log(\lambda)$, we would see a straight line. 
There is a lot of good material in Gelman and Hill, Chapter 6 to help with the interpretation of glm parameter estimates. 
For many applications, visualization is an excellent step towards interpreting model output. 

## Overdispersion

For various reasons, the variance of counts in nature tends to be greater than the mean. 
This can be considered to be extra-Poisson variance. 
Various strategies exist to account for overdispersion, and here we cover the inclusion of lognormal random effects and also derive the negative binomial distribution as a Poisson distribution with a gamma-distributed $\lambda$. 

### Checking for overdispersion

How do you know whether the variance in your data exceeds the variance you would expect in the absence of overdispersion? 
One general strategy that falls under the class of model diagnostics known as **posterior predictive checks** does the following:

1. For each posterior draw, simulate a new vector of observations. 
2. For each simulated vector of observations, calculate some test statistic of interest. 
3. Compare the distribution of simulated test statistics to the empirical test statistic.

In this case, it is reasonable to choose the variance of $y$ as the test statistic. 


```r
# simulate new observations and store variances
y_new <- array(dim=c(n_iter, n))
var_new <- rep(NA, n_iter)
for (i in 1:n_iter){
  y_new[i, ] <- rpois(n, exp(X %*% post$beta[i, ]))
  var_new[i] <- var(y_new[i, ])
}

# compare distribution of simulated values to the empirical values
hist(var_new, breaks=40, xlab='Var(y)',
     main='Posterior predictive check \n for overdispersion')
abline(v = var(y), col=2, lwd=3)
```

![](main_files/figure-html/unnamed-chunk-68-1.png) 

In this case, we can say that the variance in y shown by the red line is perfectly compatible with the model, which makes sense because we generated the data from a Poisson distribution. 

Here's a real world application with overdispersion.
The `vegan` package has a dataset of the number of trees in a bunch of 1-hectare plots on Barro Colorado Island. 
Let's look at the mean and variance of each species to get a sense for how often Poisson distribution might work:


```r
library(vegan)
library(dplyr)
data(BCI)

# coerce into long form
d <- stack(BCI)
str(d)
```

```
## 'data.frame':	11250 obs. of  2 variables:
##  $ values: int  0 0 0 0 0 0 0 0 0 1 ...
##  $ ind   : Factor w/ 225 levels "Abarema.macradenia",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
# calculate means and variances
summ_d <- d %>%
  group_by(ind) %>%
  summarize(lmean=log(mean(values)), 
            lvar=log(var(values)))
ggplot(summ_d, aes(x=lmean, y=lvar)) + 
  geom_point() + 
  geom_abline(intecept=0, slope=1, linetype='dashed') + 
  xlab("log mean") + 
  ylab("log variance")
```

![](main_files/figure-html/unnamed-chunk-69-1.png) 

Darn.
Looks like the variance exceeds the mean for most of these species. 
Let's put together a simple model for the abundance of the species *Trichilia pallida*, where we seek to estimate the mean density for the island based on the sampled plots. 


```r
# subset data
species_d <- subset(d, ind == 'Trichilia.pallida')

# visualize abundance data
hist(species_d$values)
```

![](main_files/figure-html/unnamed-chunk-70-1.png) 

```r
plot(sort(species_d$values))
```

![](main_files/figure-html/unnamed-chunk-70-2.png) 

```r
# collect data for estimation
stan_d <- list(n = nrow(species_d), p = 1, 
               X = matrix(1, nrow = nrow(species_d)), 
               y = species_d$values)
out <- stan('poisson_glm.stan', data = stan_d)
```

Assessing convergence: 


```r
out
```

```
## Inference for Stan model: poisson_glm.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##           mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## beta[1]   0.49    0.00 0.11   0.27   0.41   0.49   0.56   0.70  1210    1
## lp__    -41.95    0.02 0.71 -44.01 -42.08 -41.68 -41.50 -41.45  1762    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:14:15 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(out)
```

![](main_files/figure-html/unnamed-chunk-71-1.png) 

```r
plot(out)
```

```
## Showing 80% intervals
```

![](main_files/figure-html/unnamed-chunk-71-2.png) 

Using a posterior predictive check for the variance of $y$:


```r
post <- extract(out)

# simulate new observations and store variances
y_new <- array(dim=c(n_iter, n))
var_new <- rep(NA, n_iter)
for (i in 1:n_iter){
  y_new[i, ] <- rpois(n, exp(post$beta[i, 1]))
  var_new[i] <- var(y_new[i, ])
}

# compare distribution of simulated values to the empirical values
hist(var_new, breaks=40, xlab='Var(y)', xlim=c(0, 7),
     main='Posterior predictive check \n for overdispersion')
abline(v = var(stan_d$y), col=2, lwd=3)
```

![](main_files/figure-html/unnamed-chunk-72-1.png) 

As we can see, the observed variance is more than twice the expected variance under a Poisson model. 

### Lognormal overdispersion

We will expand our model so that we can include some overdispersion. 
First, we will allow for additional plot-level variance by adding a term to our linear predictor:

$$y_i \sim Poisson(\lambda_i)$$

$$log(\lambda_i) = X_i' \beta + \epsilon_i$$

$$\epsilon_i \sim Normal(0, \sigma)$$

$$\sigma \sim halfCauchy(0, 2)$$

Here, each observation $y_i$ has an associated parameter $\epsilon_i$ that has a normal prior, with a variance hyperparameter that determines how much extra-Poisson variance we have. 
If there is no overdispersion, then the posterior probability mass for $\sigma$ should be near zero. 
Our Stan model `poisson_od.stan` might be:

```
data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
}

parameters {
  vector[p] beta;
  vector[n] epsilon;
  real<lower=0> sigma;
}

model {
  // priors
  beta ~ normal(0, 3);
  sigma ~ cauchy(0, 2);
  epsilon ~ normal(0, sigma);
  
  // likelihood
  y ~ poisson_log(X * beta);
}
```

Our data haven't changed, but we do need to specify the path to this updated model:


```r
od_out <- stan('poisson_od.stan', data = stan_d)
```


```r
print(od_out, pars=c('sigma', 'beta', 'lp__'))
```

```
## Inference for Stan model: poisson_od.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##           mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## sigma     1.23    0.01 0.23   0.83   1.06   1.21   1.37   1.74   712 1.02
## beta[1]  -0.18    0.01 0.27  -0.76  -0.35  -0.16   0.01   0.29   842 1.01
## lp__    -30.24    0.31 7.38 -45.25 -35.20 -29.94 -25.06 -16.73   572 1.02
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:14:44 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(od_out, pars=c('sigma', 'beta'))
```

![](main_files/figure-html/unnamed-chunk-74-1.png) 

Using an updated posterior predictive check for the variance of $y$:


```r
post <- extract(od_out)

# simulate new observations and store variances
y_new <- array(dim=c(n_iter, n))
var_new <- rep(NA, n_iter)
for (i in 1:n_iter){
  y_new[i, ] <- rpois(n, exp(post$beta[i, 1] + post$epsilon[i, ]))
  var_new[i] <- var(y_new[i, ])
}

# compare distribution of simulated values to the empirical values
hist(var_new, breaks=40, xlab='Var(y)', 
     main='Posterior predictive check \n for overdispersion')
abline(v = var(stan_d$y), col=2, lwd=3)
```

![](main_files/figure-html/unnamed-chunk-75-1.png) 

That looks much better. 
The variance that we get from our replicated simulations is entirely consistent with the variance that we actually observed. 

### Poisson-gamma overdispersion and the negative binomial

The negative binomial distribution can be thought of as a Poisson distribution with a gamma prior on $\lambda$. 

$$y \sim Poisson(\lambda)$$

$$\lambda \sim Gamma(\alpha, \beta)$$

To give a sense of what this looks like, here are some gamma distributions with varying parameters:


```r
alpha <- c(.1, 1, 10)
beta <- c(.1, 1, 10)
g <- expand.grid(alpha = alpha, beta = beta)
x <- seq(0, 20, .1)
par(mfrow=c(3, 3))
for (i in 1:9){
  plot(x, dgamma(x, g$alpha[i], g$beta[i]), type='l', 
       ylab='[x]')
}
```

![](main_files/figure-html/unnamed-chunk-76-1.png) 

We could implement this directly in Stan, but it would be easier to use the built-in and optimized negative binomial distribution function, which is parameterized in terms of its mean $\mu$ and precision $\phi$:

$$y \sim NegBinom(\mu, \phi)$$

$$log(\mu) = X \beta$$

$$\phi \sim halfCauchy(0, 2)$$

```
data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
}

parameters {
  vector[p] beta;
  real<lower=0> phi;
}

model {
  beta ~ normal(0, 3);
  phi ~ cauchy(0, 5);
  y ~ neg_binomial_2_log(X * beta, phi);
}
```

With this `nb_glm.stan` file in hand, we can fit the model as before:


```r
nb_out <- stan('nb_glm.stan', data = stan_d)
```


```r
nb_out
```

```
## Inference for Stan model: nb_glm.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##           mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## beta[1]   0.49    0.00 0.19   0.12   0.37   0.49   0.62   0.86  2304    1
## phi       0.84    0.01 0.32   0.41   0.62   0.78   1.00   1.61  2198    1
## lp__    -20.74    0.03 0.99 -23.52 -21.10 -20.44 -20.04 -19.80  1224    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:15:13 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(nb_out)
```

![](main_files/figure-html/unnamed-chunk-78-1.png) 

Conducting our posterior predictive check:


```r
post <- extract(nb_out)

# simulate new observations and store variances
y_new <- array(dim=c(n_iter, n))
var_new <- rep(NA, n_iter)
for (i in 1:n_iter){
  y_new[i, ] <- rnbinom(n, mu = exp(post$beta[i, 1]), size = post$phi[i])
  var_new[i] <- var(y_new[i, ])
}

# compare distribution of simulated values to the empirical values
par(mfrow=c(1, 1))
hist(var_new, breaks=40, xlab='Var(y)', 
     main='Posterior predictive check \n for overdispersion')
abline(v = var(stan_d$y), col=2, lwd=3)
```

![](main_files/figure-html/unnamed-chunk-79-1.png) 

Again, the replicated datasets that are simulated from the posterior predictive distribution now have variance consistent with the observed data. 

## Further reading

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 6.

Gelman et al. 2014. *Bayesian data analysis. Third edition*. Chapter 16.


Chapter 5: Binomial models
==========================

## Big picture

The binomial distribution is used all over the place in ecology, so it's important to familiarize yourself with the properties/behavior of the bionomial distribution. 
In this section we will cover the binomial in the context of glms, but we will also point out some useful hierarchical models that include the binomial distribution, including occupancy models, N-mixture models, and binomial-Poisson hierarchical models. 

#### Learning goals

- properties of the binomial distribution
- link functions for the binomial
- implementation with `glm`
- binomial overdispersion
- implementation with Stan
- occupancy models
- graphical displays
- model checking
- simulation of data & parameter recovery

## Binomial generalized linear models

The binomial distribution is used for integer valued random variables that represent the number of successes in $k$ independent trials, each with probability $p$. 
For instance, the number of chicks surviving would be a binomial random variable, with the number of eggs laid $k$ and a survival probability $p$. 
Usually, $k$ is known and $p$ is estimated, but there are some useful hierarchical models that treat $k$ as a parameter that we will cover later (e.g., N-mixture models, and binomial-Poisson hierarchy models). 
When $k=1$, the binomial distribution is also called the Bernoulli distribution.
In binomial glms, a link function is needed to map $p$ from the constrained probability space to an unconstrained space. 
Usually either the logit or probit link functions are used, and both are inverse cumulative distribution functions for other distributions (the logistic and the Gaussian, respectively). 
We can write a binomial glm as follows:

$$y \sim Binomial(p, k)$$

$$logit(p) = X \beta$$

Where $logit(p) = log(p / (1-p))$, and $\beta$ is a parameter vector to be estimated. 

## Simulation and estimation

Imagine that we're interested in whether different egg companies are more or less likely to have broken eggs in our neighborhood grocery store, and we peek into 20 cartons from each of 4 companies, each with 12 eggs per carton.
We know $k=12$ for each carton, and we want to know $p_1, ..., p_4$, the probability of broken eggs for the 4 companies.
This is analogous to ANOVA, but with a binomial response. 


```r
m <- 4
n_each <- 20
company <- rep(LETTERS[1:m], each = n_each)
p <- rep(c(.02, .01, .05, .1), each = n_each)
n <- length(p)
k <- rep(12, n)
broken <- rbinom(n, size = k, prob = p)
not_broken <- 12 - broken
boxplot(broken ~ company)
```

![](main_files/figure-html/unnamed-chunk-80-1.png) 

### Estimation with glm

We can estimate the probabilities for each company as follows: 


```r
m <- glm(cbind(broken, not_broken) ~ 0 + company, family = binomial)
summary(m)
```

```
## 
## Call:
## glm(formula = cbind(broken, not_broken) ~ 0 + company, family = binomial)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.4824  -0.9710  -0.7795   0.7236   2.1091  
## 
## Coefficients:
##          Estimate Std. Error z value Pr(>|z|)    
## companyA  -3.2452     0.3398  -9.551   <2e-16 ***
## companyB  -3.6636     0.4134  -8.861   <2e-16 ***
## companyC  -3.1355     0.3230  -9.707   <2e-16 ***
## companyD  -2.3445     0.2284 -10.263   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1053.833  on 80  degrees of freedom
## Residual deviance:   81.427  on 76  degrees of freedom
## AIC: 159.11
## 
## Number of Fisher Scoring iterations: 5
```

By default the coefficients are returned on the logit scale. 
We can back-transform with the cumulative distribution function for the logistic distribution, `plogis()`:


```r
plogis(coef(m))
```

```
##   companyA   companyB   companyC   companyD 
## 0.03750000 0.02500000 0.04166667 0.08750000
```

```r
plogis(confint(m))
```

```
## Waiting for profiling to be done...
```

```
##               2.5 %     97.5 %
## companyA 0.01819661 0.06657748
## companyB 0.01001114 0.05000587
## companyC 0.02109765 0.07192593
## companyD 0.05613067 0.12761339
```

### Estimation with Stan

To do a Bayesian analysis we need priors for $\beta$, for instance so that our model is:

$$y \sim Binomial(p, k)$$

$$logit(p) = X \beta$$

$$\beta ~ Normal(0, 2)$$

Where the normal(0, 2) prior for beta is fairly uninformative on the logit-scale. 
Note that in this example, this basically communicates the idea that we have never bought eggs in a carton before. 
Realistically, we would select priors that are concentrated toward small probabilities!

Our Stan model for a binomial glm (`binomial_glm.stan`) might look like this: 

```
data {
  int n; // sample size
  int p; // number of coefficients
  matrix[n, p] X;
  int y[n];
  int k[n];
}

parameters {
  vector[p] beta;
}

model {
  beta ~ normal(0, 2);
  y ~ binomial_logit(k, X * beta);
}
```

We can recycle the design matrix made with glm:


```r
library(rstan)
X <- model.matrix(m)
stan_d <- list(n = nrow(X), 
               p = ncol(X), 
               X = X, 
               y = broken, 
               k = k)
out <- stan('binomial_glm.stan', data = stan_d)
```



```r
out
```

```
## Inference for Stan model: binomial_glm.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##            mean se_mean   sd    2.5%     25%     50%     75%   97.5% n_eff
## beta[1]   -3.20    0.01 0.33   -3.91   -3.41   -3.17   -2.97   -2.61  2455
## beta[2]   -3.59    0.01 0.40   -4.45   -3.85   -3.56   -3.31   -2.89  2338
## beta[3]   -3.11    0.01 0.32   -3.80   -3.31   -3.09   -2.88   -2.53  2681
## beta[4]   -2.33    0.00 0.22   -2.79   -2.48   -2.32   -2.18   -1.91  3073
## lp__    -186.04    0.04 1.43 -189.65 -186.77 -185.74 -184.99 -184.23  1491
##         Rhat
## beta[1]    1
## beta[2]    1
## beta[3]    1
## beta[4]    1
## lp__       1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:15:42 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(out)
```

![](main_files/figure-html/unnamed-chunk-84-1.png) 

```r
plot(out)
```

```
## Showing 80% intervals
```

![](main_files/figure-html/unnamed-chunk-84-2.png) 

```r
pairs(out)
```

![](main_files/figure-html/unnamed-chunk-84-3.png) 

How might we graph the results? 
Ideally our plot should show the data and our model output (in this case, the expected number of broken eggs). 
In this case, we might try something like this:


```r
# put data into a data frame
d <- data.frame(broken, company)

# create data frame for posterior estimates
library(dplyr)
library(reshape2)
library(ggplot2)
library(grid)

# extract posterior samples and get summaries
post <- out %>%
  extract() %>%
  melt() %>%
  subset(L1 != 'lp__') %>%
  group_by(Var2) %>%
  summarize(median = 12 * plogis(median(value)), 
            lo = 12 * plogis(quantile(value, .025)), 
            hi = 12 * plogis(quantile(value, .975)))
post$company <- unique(company)

# plot results
ggplot(d, aes(x=company, y=broken)) + 
  geom_segment(aes(xend=company, y=lo, yend=hi), data=post, 
               size=3, col='blue', alpha=.3) + 
  geom_point(aes(x=company, y=median), data=post, 
             col='blue', size=3) + 
  geom_jitter(position=position_jitter(width=.1, height=.1), 
              shape=1) + 
  ylab("Number of broken eggs")
```

![](main_files/figure-html/unnamed-chunk-85-1.png) 

## Overdispersion

The binomial distribution has mean $np$ and variance $np(1-p)$, but sometimes there is more variation than we would expect from the binomial distribution. 
For instance, in the previous example, we should expect sources of variation other than the companies that produce and distribute the eggs. 
Any particular carton may have a different history than other cartons due to chance - maybe the stocker set one carton down hard, or an unattended child opened a carton and broke an egg then closed the carton, etc. 
Though binomial overdispersion receives little attention in ecology compared to Poisson overdispersion, it is a common problem in real-world data. 
As with the Poisson example, you can use posterior predictive checks to evaluate whether your model is underrepresenting the variance in binomial observations. 

### Binomial-normal model

One common solution to overdispersion is the addition of a normally distributed random effect to the linear predictor. 
This represents variation at the level of individual observations. 

$$y_i \sim Binomial(p_i, k_i)$$

$$logit(p_i) = X_i' \beta + \epsilon_i$$

$$\beta \sim Normal(0, 2)$$

$$\epsilon_i \sim Normal(0, \sigma)$$

$$\sigma \sim halfCauchy(0, 2)$$

This is very similar to the lognormal overdispersion strategy for Poisson models covered in the previous chapter. 

### Beta-binomial model

Another option is to use a beta distribution as a prior for the probability of success parameter $p$:

$$y \sim Binomial(p, k)$$

$$ p \sim beta(\alpha, \beta)$$

This strategy tends to be used infrequently in ecology, and we do not cover it in depth here, but there are good resources on the web for implementing beta-binomial models in Stan [here](http://stats.stackexchange.com/questions/96481/how-to-specify-a-bayesian-binomial-model-with-shrinkage-to-the-population), and [here](http://wiekvoet.blogspot.com/2014/08/beta-binomial-revisited.html).

## Occupancy models

Occupancy models represent the presence or absence of a species from a location as a Bernoulli random variable $Z$, with $z=0$ corresponding to absence and $z=1$ corresponding to presence. 
If the species is present $z=1$, it will be detected on a survey with probability $p$.
So, species may be present but undetected. 
If a species is absent, it will not be detected (we assume no false detections), but species may be present and undetected. 

We can write this as a hierarchical model with occurrence state $z$ treated as a binary latent (hidden) variable: 

$$[z \mid \psi] \sim Bernoulli(\psi)$$

$$[y \mid z, p] \sim Bernoulli(z p)$$

If we wish to avoid the use of a discrete parameter, we can marginalize $z$ out of the posterior:

$$[\psi, p \mid y] \propto \sum_z [y \mid z, p] [z \mid \psi] [p, \psi]$$

$$[\psi, p \mid y] \propto  [p, \psi] \sum_z [y \mid z, p] [z \mid \psi]$$

$$[\psi, p \mid y] \propto  [p, \psi] \big( [y \mid z=1, p] [z=1 \mid \psi] + [y \mid z=0, p] [z=0 \mid \psi] \big)$$

$$[\psi, p \mid y] \propto  [p, \psi] \big( \psi Bernoulli(p) + (1-\psi) I(y = 0)  \big)$$

where $I(y = 0)$ is an identity function that sets $y$ to zero because we assumed that there are no false detections when the species is absent. 
Multiple surveys are necessary to identify $\psi$ and $p$, so that we can expand the likelihood to account of binomial observation histories, with $k$ surveys conducted per site, and observation history vectors $y_i$ for the $i^{th}$ site:

$$[y_i \mid \psi_i, p_i] \begin{cases} \psi_i Binom(y_i, k) &\mbox{if } \sum y_i > 0 \\ 
\psi_i Binom(0, k) + (1 - \psi) & otherwise \end{cases}$$

If the organism was observed, then we know that any non-detections represent false absences. 
If the organism was not observed, then it was either there and not observed (with probability $\psi_i Binom(0, k)$) or it was not there with probability $1 - \psi$. 
This if-else structure can be exploited in Stan to implement this likelihood:

```
data { 
   int<lower=0> nsite; 
   int<lower=0> nsurvey; 
   int<lower=0,upper=1> y[nsite,nsurvey]; 
} 
parameters { 
   real<lower=0,upper=1> psi; 
   real<lower=0,upper=1> p; 
} 
model { 
   for (i in 1:nsite) { 
     if (sum(y[i]) > 0)
       // species was observed: it is there
       increment_log_prob(log(psi) + bernoulli_log(y[i],p)); 
     else 
       // it may or may not be there
       increment_log_prob(log_sum_exp(log(psi) + bernoulli_log(y[i],p), 
                                      log1m(psi))); 
   } 
} 
```

Now let's simulate some occupancy data and fit the model:


```r
nsite <- 50
nsurvey <- 3
psi <- .4
p <- .8
z <- rbinom(nsite, 1, psi)
y <- matrix(rbinom(nsite * nsurvey, 1, z * p), 
            nrow=nsite)

stan_d <- list(nsite = nsite, 
               nsurvey = nsurvey, 
               y = y)
out <- stan('occupancy.stan', data = stan_d)
```

How did we do? 


```r
out
```

```
## Inference for Stan model: occupancy.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##        mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## psi    0.37    0.00 0.07   0.25   0.33   0.37   0.42   0.51  2243    1
## p      0.75    0.00 0.06   0.62   0.71   0.76   0.80   0.87  2255    1
## lp__ -65.28    0.03 1.08 -68.10 -65.68 -64.95 -64.52 -64.26   978    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:16:12 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(out)
```

![](main_files/figure-html/unnamed-chunk-87-1.png) 

```r
par(mfrow=c(1, 2))
post <- extract(out)
plot(density(post$psi), main=expression(psi))
abline(v=psi, col='red', lwd=2)
plot(density(post$p), main='p')
abline(v=p, col='red', lwd=2)
```

![](main_files/figure-html/unnamed-chunk-87-2.png) 

This model can be extended to include covariates for $\psi$ and $p$ by making use of a link function and design matrices.

## Further reading

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 5, 6.

Gelman et al. 2014. *Bayesian data analysis. Third edition*. Chapter 16.


Chapter 6: Partial pooling and likelihood
========================================

Partial pooling is one of the primary motivations behind conventional hierarchical models. 
Also termed "borrowing information", partial pooling improves estimates of group-level parameters, particularly when we have $>3$ groups and/or varying sample sizes among groups. 
In this chapter, we illustrate partial pooling and link it to prior distributions in a maximum-likelihood context.

#### Learning goals

- motivation for and definition of partial pooling
- simple hierarchical models with likelihood
- hyperparameters
- varying intercepts (NBA freethrow example) with `lme4`
- partial pooling
- clearing up confusion about nestedness
- predictors for multiple levels
- plotting estimates for different levels from lme4 models

## Partial pooling: free throw example

Often, data are structured hierarchically. 
For instance, maybe we sample individual animals within sites, with multiple sites. 
Or, perhaps we sequence multiple genes across multiple individuals. 
There may be more than two levels, for instance if we sample parasites of different species within individuals of different species of hosts across multiple sites in a landscape. 
Commonly, sample sizes are not equal across units at various levels. 
The following example demonstrates how to use partial pooling to generate reliable estimates in the context of wildly varying sample sizes. 

Suppose we are interested in knowing who the best free throw shooter was in the 2014-2015 NBA season. 
We can pull the data from the [web](http://www.basketball-reference.com/leagues/NBA_2015_totals.html), and plot the proportion of free-throws made by player.


```r
rawd <- read.csv("leagues_NBA_2015_totals_totals.csv")
str(rawd)
```

```
## 'data.frame':	651 obs. of  30 variables:
##  $ Rk    : int  1 2 3 4 5 5 5 6 7 8 ...
##  $ Player: Factor w/ 492 levels "Aaron Brooks",..: 384 249 438 212 36 36 36 8 152 85 ...
##  $ Pos   : Factor w/ 11 levels "C","PF","PF-SF",..: 2 9 1 2 9 9 9 1 2 1 ...
##  $ Age   : int  24 20 21 28 29 29 29 26 23 26 ...
##  $ Tm    : Factor w/ 31 levels "ATL","BOS","BRK",..: 20 15 21 18 29 8 25 19 23 20 ...
##  $ G     : int  68 30 70 17 78 53 25 68 41 61 ...
##  $ GS    : int  22 0 67 0 72 53 19 8 9 16 ...
##  $ MP    : int  1287 248 1771 215 2502 1750 752 957 540 976 ...
##  $ FG    : int  152 35 217 19 375 281 94 181 40 144 ...
##  $ FGA   : int  331 86 399 44 884 657 227 329 78 301 ...
##  $ FG.   : num  0.459 0.407 0.544 0.432 0.424 0.428 0.414 0.55 0.513 0.478 ...
##  $ X3P   : int  18 10 0 0 118 82 36 0 0 0 ...
##  $ X3PA  : int  60 25 2 0 333 243 90 0 5 0 ...
##  $ X3P.1 : num  0.3 0.4 0 NA 0.354 0.337 0.4 NA 0 NA ...
##  $ X2P   : int  134 25 217 19 257 199 58 181 40 144 ...
##  $ X2PA  : int  271 61 397 44 551 414 137 329 73 301 ...
##  $ X2P.1 : num  0.494 0.41 0.547 0.432 0.466 0.481 0.423 0.55 0.548 0.478 ...
##  $ eFG.  : num  0.486 0.465 0.544 0.432 0.491 0.49 0.493 0.55 0.513 0.478 ...
##  $ FT    : int  76 14 103 22 167 127 40 81 13 50 ...
##  $ FTA   : int  97 23 205 38 198 151 47 99 27 64 ...
##  $ FT.   : num  0.784 0.609 0.502 0.579 0.843 0.841 0.851 0.818 0.481 0.781 ...
##  $ ORB   : int  79 9 199 23 27 21 6 104 78 101 ...
##  $ DRB   : int  222 19 324 54 220 159 61 211 98 237 ...
##  $ TRB   : int  301 28 523 77 247 180 67 315 176 338 ...
##  $ AST   : int  68 16 66 15 129 101 28 47 28 75 ...
##  $ STL   : int  27 16 38 4 41 32 9 21 17 37 ...
##  $ BLK   : int  22 7 86 9 7 5 2 51 16 65 ...
##  $ TOV   : int  60 14 99 9 116 83 33 69 17 59 ...
##  $ PF    : int  147 24 222 30 167 108 59 151 96 122 ...
##  $ PTS   : int  398 94 537 60 1035 771 264 443 93 338 ...
```

Some players switched teams mid-season, and they appear on separate rows, one for each team they played on. 
We need to aggregate across the teams, so that we end up with only one row per player:


```r
library(dplyr)
library(ggplot2)

# clean the data
d <- rawd %>%
  group_by(Player) %>%
  summarize(ft_made = sum(FT), 
            ft_miss = sum(FTA) - sum(FT),
            ft_shot = sum(FTA), 
            ft_pct = sum(FT) / sum(FTA)) %>%
  subset(ft_shot != 0) %>%
  arrange(-ft_pct) %>%
  droplevels()
d
```

```
## Source: local data frame [475 x 5]
## 
##                   Player ft_made ft_miss ft_shot ft_pct
##                   (fctr)   (int)   (int)   (int)  (dbl)
## 1              Alex Kirk       2       0       2      1
## 2  Chris Douglas-Roberts       8       0       8      1
## 3            C.J. Wilcox       2       0       2      1
## 4          Grant Jerrett       2       0       2      1
## 5              Ian Clark      18       0      18      1
## 6          Jannero Pargo       2       0       2      1
## 7           Jerel McNeal       2       0       2      1
## 8         John Lucas III       5       0       5      1
## 9          Kenyon Martin       2       0       2      1
## 10          Luigi Datome       2       0       2      1
## ..                   ...     ...     ...     ...    ...
```

```r
# order factor levels by ft_pct and plot
levels(d$Player) <- levels(d$Player)[order(-d$ft_pct)]
d$shooter <- factor(d$Player, levels = d$Player[order(d$ft_pct)])
ggplot(d, aes(x=ft_pct, y=shooter)) + 
  geom_point(stat='identity') + 
  theme(text = element_text(size=8))
```

![](main_files/figure-html/unnamed-chunk-89-1.png) 

Wow! Looks like we have some really good (100% accuracy) and really bad (0% accuracy) free throw shooters in the NBA. 
We can verify this by calculating maximum likelihood estimates for the probability of making a free throw for each player.
We'll assume that the number of free throws made is a binomial random variable, with $p_i$ to be estimated for the $i^{th}$ player, and $k$ equal to the number of free throw attempts.

$$y_i \sim Binom(p_i, k_i)$$


```r
# fit binomial glm
m <- glm(cbind(ft_made, ft_miss) ~ 0 + Player, 
         family=binomial, data=d)

# print estimated probabilities
probs <- m %>%
  coef() %>%
  plogis() %>%
  round(digits=4) %>% 
  sort(decreasing=TRUE)
```

It turns out that the maximum likelihood estimates are equal to the proportion of free throws made.


```r
plot(d$ft_pct, probs, 
     xlab="Empirial proportion FT made", 
     ylab="MLE: Pr(make FT)")
```

![](main_files/figure-html/unnamed-chunk-91-1.png) 

But, can we really trust these estimates? 
What if we plot the maximum likelihood estimates along with the number of free throw attempts?


```r
ggplot(d, aes(x=ft_shot, y=ft_pct)) + 
  geom_point(alpha=.6) + 
  xlab('Free throw attempts') + 
  ylab('Proportion of free throws made')
```

![](main_files/figure-html/unnamed-chunk-92-1.png) 

Well that's not good. 
It looks like the players with the highest and lowest shooting percentages took the fewest shots. 
We should be skeptical of these estimates. 
One solution is to select some minimum number of attempts, and only believe estimates for players who have taken at least that number. 
This is what the NBA does, and the limit is 125 shots. 


```r
d %>%
  filter(ft_shot >= 125) %>%
  ggplot(aes(x=ft_shot, y=ft_pct)) + 
  geom_point(alpha=.6) + 
  xlab('Free throw attempts') + 
  ylab('Proportion of free throws made')
```

![](main_files/figure-html/unnamed-chunk-93-1.png) 

This seems somewhat arbitrary - what's special about the number 125? 
Is there a better way to decide which estimates to trust? 
What if instead of tossing out the data from players that haven't taken at least 125 shots, we tried to somehow improve those estimates? 
For instance, we might instead pull these estimates towards the average, and place increasingly more trust in proportions from players with more information (shot attempts). 
This is the intuition behind partial pooling, and the secret to implementation lies in placing a prior distribution on $p_i$, the probability that player $i$ makes a free throw, so that:

$$y_i \sim Binom(p_i, k_i)$$

$$logit(p_i) \sim N(\mu_p, \sigma_p)$$

such that the likelihood to maximize is: 

$$[y_i \mid p_i] [p_i \mid \mu_p, \sigma_p]$$

where $\mu_p$ represents the overall league (among player) mean on a logit scale for the probability of making a free thrower, and $\sigma_p$ represents the variability among players in the logit probability of making a free throw. 
This type of model is sometimes called a varying-intercept or random-intercept model. 
Because $\mu_p$ and $\sigma_p$ determine the distribution of the parameter $p$, they are known as **hyperparameters**. 
This model approaches the previous model with no hyperparameters when $\sigma_p$ approaches $\infty$.
A similar strategy would be to place a beta prior directly on $p_i$, rather than placing a prior on $logit(p_i)$.

This model is hierarchical in the sense that we have a within player-level model (each player gets their own $p_i$) and an among-player model (with $\sigma_p$ controlling the among-player variaiton).
We will implement the logit-normal model in a maximum likelihood context to begin with, where we find maximum likelihood estimates for the hyperparameter $\sigma_p$.


```r
library(lme4)
m2 <- glmer(cbind(ft_made, ft_miss) ~ (1|Player), 
         family=binomial, data=d)
summary(m2)
```

```
## Generalized linear mixed model fit by maximum likelihood (Laplace
##   Approximation) [glmerMod]
##  Family: binomial  ( logit )
## Formula: cbind(ft_made, ft_miss) ~ (1 | Player)
##    Data: d
## 
##      AIC      BIC   logLik deviance df.resid 
##   3394.0   3402.3  -1695.0   3390.0      473 
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -2.41341 -0.28396  0.01969  0.28122  1.70985 
## 
## Random effects:
##  Groups Name        Variance Std.Dev.
##  Player (Intercept) 0.2951   0.5432  
## Number of obs: 475, groups:  Player, 475
## 
## Fixed effects:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.07548    0.02891    37.2   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We used the `glmer` function in the `lme4` package to implement the above model. 
We obtain an estimate for $logit(\mu_p)$ with the "(Intercept)" parameter, and it is 0.7456375 on the probability scale.
We also see a maximum likelihood estimate for $\sigma_p$ under the random effects section: 0.5431875.

Let's visualize the new estimates: 


```r
# get estimated probabilities for each player from m2
shrunken_probs <- plogis(fixef(m2) + unlist(ranef(m2)))

# match these to the player names, 
# from the row names of the ranef design matrix
shrunken_names <- m2@pp$Zt@Dimnames[[1]]

ranef_preds <- data.frame(Player = shrunken_names, 
                          p_shrink = shrunken_probs)

# join the raw data with the model output
joined <- full_join(d, ranef_preds)
```

```
## Joining by: "Player"
```

```r
# difference between naive & shrunken MLEs
joined$diff <- joined$ft_pct - joined$p_shrink

# plot naive MLE vs. shrunken MLE
ggplot(joined, aes(x=ft_pct, y=p_shrink, color=ft_shot)) + 
  geom_point(shape=1) + 
  scale_color_gradientn(colours=rainbow(4)) +
  geom_abline(yintercept=0, slope=1, linetype='dashed') +
  xlab('Naive MLE') + 
  ylab('Shrunken MLE')
```

![](main_files/figure-html/unnamed-chunk-95-1.png) 

```r
# using facets instead of colors
joined$ft_group <- cut(joined$ft_shot, 9)
ggplot(joined, aes(x=ft_pct, y=p_shrink)) + 
  geom_abline(yintercept=0, slope=1, linetype='dashed', alpha=.7) +
  geom_point(shape=1, alpha=.5, size=1) + 
  facet_wrap(~ ft_group) +
  xlab('Naive MLE') + 
  ylab('Shrunken MLE')
```

![](main_files/figure-html/unnamed-chunk-95-2.png) 

```r
# view difference in estimates as a function of freethrows shot
ggplot(joined, aes(x=ft_shot, y=diff)) + 
  geom_point(shape=1) + 
  xlab("Free throw attempts") +
  ylab("Naive MLE - Shrunken MLE")
```

![](main_files/figure-html/unnamed-chunk-95-3.png) 

The estimates from the hierarchical model differ from the MLE estimates obtained in our first model. 
In particular, the estimates that are imprecise (e.g., players with few attempted shots) are shrunken towards the grand mean. 
This **shrinkage** is highly desirable, and is consistent with the idea that we have increasing trust in estimates that are informed by more data.

What about the NBA's magic number of 125? 
Do we find that estimates are still undergoing shrinkage beyond this range, or are the empirical free throw percentages reliable if players have taken 125 or more shots? 


```r
joined %>%
  filter(ft_shot >= 125) %>%
  ggplot(aes(x=ft_shot, y=diff)) + 
  geom_point(shape=1) + 
  xlab("Free throw attempts") +
  ylab("Naive MLE - Shrunken MLE")
```

![](main_files/figure-html/unnamed-chunk-96-1.png) 

It looks like there are slight differences between the naive estimates (empirical proportions) and the probabilities that we estimate with partial pooling. 
We might conclude that the 125 shot threshold could give estimates that are reliable to plus or minus 2 or 3 percentage points based on the above graph. 

So, which player do we think is the best and the worst? 


```r
joined[which.max(joined$p_shrink), ]
```

```
## Source: local data frame [1 x 9]
## 
##          Player ft_made ft_miss ft_shot    ft_pct       shooter  p_shrink
##          (fctr)   (int)   (int)   (int)     (dbl)        (fctr)     (dbl)
## 1 Stephen Curry     308      29     337 0.9139466 Stephen Curry 0.9023947
## Variables not shown: diff (dbl), ft_group (fctr)
```

```r
joined[which.min(joined$p_shrink), ]
```

```
## Source: local data frame [1 x 9]
## 
##        Player ft_made ft_miss ft_shot    ft_pct     shooter  p_shrink
##        (fctr)   (int)   (int)   (int)     (dbl)      (fctr)     (dbl)
## 1 Joey Dorsey      24      59      83 0.2891566 Joey Dorsey 0.3570848
## Variables not shown: diff (dbl), ft_group (fctr)
```

Congrats Steph Curry (for this accomplishment and winning the NBA title), and our condolences to Joey Dorsey, who as of 2015 is playing in the Turkish Basketball League. 

## Partial, complete, and no pooling

Partial pooling is often presented as a compromise between complete pooling (in the above example, combining data from all players and estimating one NBA-wide $p$), and no pooling (using the observed proportion of free-throws made). 
This can be formalized by considering what happens to the parameter-level model in these three cases. 
With no pooling, the among-group (e.g., player) variance parameter approaches infinity, such that $p_i \sim N(\mu_p, \sigma_p): \sigma_p \rightarrow \infty$.
With complete pooling, the among-group variance parameter approaches zero, so that all groups recieve the group-level mean. 
With partial pooling, the estimation of $\sigma_p$ leads to a data-informed amount of shrinkage. 

## Multiple levels, nestedness, and crossedness

In the previous example we had two levels: within and among players. 
Many hierarchical models have more than two levels. 
For instance, at the among-player level, we know that players played on different teams. 
Perhaps there are systematic differences among teams in free-throw shooting ability, for instance because they have a coaching staff that consistently emphasize free-throw shooting, or scouts that recruit players who are particularly good at shooting free-throws. 
We can expand the previous model to include team effects as follows: 

$$y_i \sim Binom(p_i, k_i)$$

$$logit(p_i) = p_0 + \pi_i + \tau_{j[i]}$$

$$\pi_i \sim N(0, \sigma_\pi)$$

$$\tau_j \sim N(0, \sigma_\tau)$$

so that the likelihood to maximize is: 

$$[y\mid p] [p \mid p_0, \pi, \tau] [\pi \mid \sigma_\pi] [\tau \mid \sigma_\tau]$$

Here, $p_0$ is an intercept parameter that represents the mean logit probability of making a free throw across all players and teams. 
Player effects are represented by $\pi_i$, and team effects are represented by $\tau_{j[i]}$, with subscript indexing to represent that player $i$ belongs to team $j$.

Note that not all players play for all teams, that is, players are not "crossed" with teams. 
Most players, specifically those that only played for one team, are nested within teams. 
However, because some players switched teams partway through the season, these players will show up on different teams. 
There is often a lot of confusion around nestedness in these types of models.
We point out here that **nestedness is a feature of the data**, not a modeling decision. 
There are cases when the data are nested but structured poorly so that extra work is required to adequately represent the nestedness of the data. 
For instance, if we had data with measurements from 5 regions, each with 3 sites, and the sites were always labeled 1, 2, or 3, then our data might look like this:


```
##    region site
## 1       1    1
## 2       1    2
## 3       1    3
## 4       2    1
## 5       2    2
## 6       2    3
## 7       3    1
## 8       3    2
## 9       3    3
## 10      4    1
## 11      4    2
## 12      4    3
## 13      5    1
## 14      5    2
## 15      5    3
```

This would indicate that sites are crossed with region, with each site occurring in each region.
But, this is misleading. 
The observations corresponding to site 1 are actually 5 different sites, occuring in 5 different regions. 
A more accurate data structure would be: 


```
##    region site
## 1       1    1
## 2       1    2
## 3       1    3
## 4       2    4
## 5       2    5
## 6       2    6
## 7       3    7
## 8       3    8
## 9       3    9
## 10      4   10
## 11      4   11
## 12      4   12
## 13      5   13
## 14      5   14
## 15      5   15
```

This numbering scheme accurately captures the notion that each site occurs in only one region. 

### Fitting a 3 level model with `lme4`

Returning to the example with player and team effects, both drawn from a prior distribution with hyperparameters $\sigma_\pi$ and $\sigma_\tau$ to be estimated:


```r
d <- rawd %>%
  group_by(Player, Tm, Age, Pos) %>%
  summarize(ft_made = sum(FT), 
            ft_miss = sum(FTA) - sum(FT),
            ft_shot = sum(FTA), 
            ft_pct = sum(FT) / sum(FTA)) %>%
  subset(ft_shot != 0) %>%
  arrange(-ft_pct) %>%
  droplevels()
d
```

```
## Source: local data frame [626 x 8]
## Groups: Player, Tm, Age [626]
## 
##           Player     Tm   Age    Pos ft_made ft_miss ft_shot    ft_pct
##           (fctr) (fctr) (int) (fctr)   (int)   (int)   (int)     (dbl)
## 1   Aaron Brooks    CHI    30     PG     145      29     174 0.8333333
## 2   Aaron Gordon    ORL    19     PF      44      17      61 0.7213115
## 3  Adreian Payne    ATL    23     PF       1       1       2 0.5000000
## 4  Adreian Payne    MIN    23     PF      29      15      44 0.6590909
## 5  Adreian Payne    TOT    23     PF      30      16      46 0.6521739
## 6     A.J. Price    CLE    28     PG       4       2       6 0.6666667
## 7     A.J. Price    IND    28     PG      12       6      18 0.6666667
## 8     A.J. Price    TOT    28     PG      16       8      24 0.6666667
## 9  Alan Anderson    BRK    32     SG      82      19     101 0.8118812
## 10    Alec Burks    UTA    23     SG     106      23     129 0.8217054
## ..           ...    ...   ...    ...     ...     ...     ...       ...
```

```r
m <- glmer(cbind(ft_made, ft_miss) ~ (1|Player) + (1|Tm), 
         family=binomial, data=d)
summary(m)
```

```
## Generalized linear mixed model fit by maximum likelihood (Laplace
##   Approximation) [glmerMod]
##  Family: binomial  ( logit )
## Formula: cbind(ft_made, ft_miss) ~ (1 | Player) + (1 | Tm)
##    Data: d
## 
##      AIC      BIC   logLik deviance df.resid 
##   3899.1   3912.5  -1946.6   3893.1      623 
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -2.17015 -0.30554  0.01832  0.30808  1.91060 
## 
## Random effects:
##  Groups Name        Variance Std.Dev.
##  Player (Intercept) 0.2951   0.5432  
##  Tm     (Intercept) 0.0000   0.0000  
## Number of obs: 626, groups:  Player, 475; Tm, 31
## 
## Fixed effects:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.07548    0.02891    37.2   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
pr <- profile(m)
confint(pr)
```

```
##                 2.5 %     97.5 %
## .sig01      0.5002122 0.59097299
## .sig02      0.0000000 0.06070247
## (Intercept) 1.0187033 1.13228412
```

So, it appears that the MLE for the variance attributable to teams is zero. 
Profiling the likelihood, we see that there may be a little bit of variance attributable to teams, but we may be better off discarding the team information and instead including information about the player's positions. 
The logic here is that we expect players to vary in their shooting percentages based on whether they are guards, forwards, centers, etc. 


```r
m2 <- glmer(cbind(ft_made, ft_miss) ~ (1|Player) + (1|Pos), 
            family=binomial, data=d)
summary(m2)
```

```
## Generalized linear mixed model fit by maximum likelihood (Laplace
##   Approximation) [glmerMod]
##  Family: binomial  ( logit )
## Formula: cbind(ft_made, ft_miss) ~ (1 | Player) + (1 | Pos)
##    Data: d
## 
##      AIC      BIC   logLik deviance df.resid 
##   3851.9   3865.2  -1923.0   3845.9      623 
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -2.16535 -0.32463  0.02557  0.32095  1.87141 
## 
## Random effects:
##  Groups Name        Variance Std.Dev.
##  Player (Intercept) 0.23943  0.4893  
##  Pos    (Intercept) 0.03869  0.1967  
## Number of obs: 626, groups:  Player, 475; Pos, 11
## 
## Fixed effects:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.09854    0.08041   13.66   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
AIC(m, m2)
```

```
##    df      AIC
## m   3 3899.147
## m2  3 3851.925
```

So, `m2` seems to be better in terms of AIC, but one could argue that `m2` will also be more useful for predictive applications. 
For instance, if we wanted to predict the free throw shooting ability of a new player, we could get a more precise estimate with `m2` because we could use information about their position (if it was known). 
In contrast, in model `m`, we would predict the same $p$ regardless of position, because position is not in the model. 

## Level-specific covariates

In hierarchical models, covariates can be included at specific levels. 
For instance, at the player level, we might expect that age has an impact on free throw shooting ability. 
Possibly, players peak at some age, and then start to go downhill as they approach retirement. 
We can represent this with a second degree polynomial effect of age. 
Trying this:


```r
m3 <- glmer(cbind(ft_made, ft_miss) ~ Age + I(Age^2) + (1|Player) + (1|Pos), 
            family=binomial, data=d)
```

```
## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control
## $checkConv, : Model failed to converge with max|grad| = 0.0280989 (tol =
## 0.001, component 1)
```

```
## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model is nearly unidentifiable: very large eigenvalue
##  - Rescale variables?;Model is nearly unidentifiable: large eigenvalue ratio
##  - Rescale variables?
```

```r
summary(m3)
```

```
## Generalized linear mixed model fit by maximum likelihood (Laplace
##   Approximation) [glmerMod]
##  Family: binomial  ( logit )
## Formula: cbind(ft_made, ft_miss) ~ Age + I(Age^2) + (1 | Player) + (1 |  
##     Pos)
##    Data: d
## 
##      AIC      BIC   logLik deviance df.resid 
##   3838.1   3860.3  -1914.0   3828.1      621 
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -1.99299 -0.33689  0.02006  0.33940  1.88029 
## 
## Random effects:
##  Groups Name        Variance Std.Dev.
##  Player (Intercept) 0.22661  0.4760  
##  Pos    (Intercept) 0.04262  0.2065  
## Number of obs: 626, groups:  Player, 475; Pos, 11
## 
## Fixed effects:
##              Estimate Std. Error z value Pr(>|z|)   
## (Intercept) -1.956747   0.952742  -2.054  0.03999 * 
## Age          0.206198   0.069517   2.966  0.00302 **
## I(Age^2)    -0.003347   0.001252  -2.674  0.00749 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##          (Intr) Age   
## Age      -0.993       
## I(Age^2)  0.981 -0.996
## convergence code: 0
## Model failed to converge with max|grad| = 0.0280989 (tol = 0.001, component 1)
## Model is nearly unidentifiable: very large eigenvalue
##  - Rescale variables?
## Model is nearly unidentifiable: large eigenvalue ratio
##  - Rescale variables?
```

We get a warning about convergence resulting from numeric issues, with the recommendation to rescale our variables. 
Recall that unscaled continuous covariates tend to cause correlations between the estimates of slopes and intercepts, which is apparent in the Correlation of Fixed Effects section. 
Rescaling age does the trick in this example:


```r
d$age <- (d$Age - mean(d$Age)) / sd(d$Age)
m3 <- glmer(cbind(ft_made, ft_miss) ~ age + I(age^2) + (1|Player) + (1|Pos), 
            family=binomial, data=d)
summary(m3)
```

```
## Generalized linear mixed model fit by maximum likelihood (Laplace
##   Approximation) [glmerMod]
##  Family: binomial  ( logit )
## Formula: cbind(ft_made, ft_miss) ~ age + I(age^2) + (1 | Player) + (1 |  
##     Pos)
##    Data: d
## 
##      AIC      BIC   logLik deviance df.resid 
##   3838.1   3860.3  -1914.0   3828.1      621 
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -1.99300 -0.33690  0.02006  0.33940  1.88029 
## 
## Random effects:
##  Groups Name        Variance Std.Dev.
##  Player (Intercept) 0.22661  0.4760  
##  Pos    (Intercept) 0.04263  0.2065  
## Number of obs: 626, groups:  Player, 475; Pos, 11
## 
## Fixed effects:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.15916    0.08597  13.484  < 2e-16 ***
## age          0.11720    0.02843   4.122 3.76e-05 ***
## I(age^2)    -0.05722    0.02143  -2.671  0.00757 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##          (Intr) age   
## age       0.109       
## I(age^2) -0.251 -0.396
```

```r
AIC(m, m2, m3)
```

```
##    df      AIC
## m   3 3899.147
## m2  3 3851.925
## m3  5 3838.078
```

This model does seem to receive more support than the other two models. 
We can visualise the age result as follows: 


```r
lo <- 100
new_age <- seq(min(d$age), max(d$age), length.out=lo)
X <- matrix(c(rep(1, 100), new_age, new_age^2), nrow=lo)
logit_p <- X %*% fixef(m3)
scaled_age <- new_age * sd(d$Age) + mean(d$Age) 
plot(scaled_age, plogis(logit_p), type='l', 
     xlab="Age", ylab="p")
```

![](main_files/figure-html/unnamed-chunk-104-1.png) 

There are also packages that are designed to aid in visualization of the output of lmer and glmer objects. 
Handy plots here include a sorted caterpillar plot of the random effects:


```r
library(sjPlot)
library(arm)
sjp.glmer(m3, ri.nr = 1, sort = "(Intercept)")
```

![](main_files/figure-html/unnamed-chunk-105-1.png) 


```r
sjp.glmer(m3, ri.nr=2, sort = "(Intercept)")
```

```
## Plotting random effects...
```

![](main_files/figure-html/unnamed-chunk-106-1.png) 

These confidence intervals should be taken with a grain of salt, as they are calculated based on a normal approximation with Wald standard errors, which assume a quadratic log-likelihood profile. 
Later, we will get a more reliable estimate of the error for random effects using Bayesian methods. 

We assumed that the random effects are normally distributed on a logit scale. 
This assumption can be checked with a q-q plot: 


```r
sjp.glmer(m3, type = "re.qq", facet.grid=F)
```

```
## Testing for normal distribution. Dots should be plotted along the line.
```

![](main_files/figure-html/unnamed-chunk-107-1.png) 

Last, we should do a sanity check for our estimated probabilities.
One approach is to visually check the estimated probabilities vs. the naive empirical proportions.


```r
d$estimated_p <- fitted(m3)
d$diff <- d$ft_pct - d$estimated_p
ggplot(d, aes(x=ft_shot, y=diff)) + 
  geom_point(shape=1) + 
  xlab("Free throw attempts") +
  ylab("Naive MLE - Shrunken MLE")
```

![](main_files/figure-html/unnamed-chunk-108-1.png) 

```r
ggplot(d) + 
  geom_segment(aes(x="Naive MLE", xend="Shrunken MLE", 
                   y=ft_pct, yend=estimated_p, group=Player), 
               alpha=.3) + 
  xlab('Estimate') + 
  ylab('Pr(make freethrow)')
```

![](main_files/figure-html/unnamed-chunk-108-2.png) 

```r
ggplot(d, aes(x=ft_shot, y=estimated_p)) + 
  geom_point(shape=1) + 
  xlab("Free throw attempts") +
  ylab("Shrunken MLE")
```

![](main_files/figure-html/unnamed-chunk-108-3.png) 

Who does this model identify as the worst?


```r
d[which.min(d$estimated_p), 
  c("Player", "Age", "Pos", "ft_shot", "ft_pct", "estimated_p")]
```

```
## Source: local data frame [1 x 6]
## 
##        Player   Age    Pos ft_shot    ft_pct estimated_p
##        (fctr) (int) (fctr)   (int)     (dbl)       (dbl)
## 1 Ian Mahinmi    28      C     102 0.3039216   0.3652981
```

Which player might be best?


```r
d[which.max(d$estimated_p), 
  c("Player", "Age", "Pos", "ft_shot","ft_pct", "estimated_p")]
```

```
## Source: local data frame [1 x 6]
## 
##          Player   Age    Pos ft_shot    ft_pct estimated_p
##          (fctr) (int) (fctr)   (int)     (dbl)       (dbl)
## 1 Stephen Curry    26     PG     337 0.9139466   0.9023958
```



## Further reading

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 11, 12.

Gelman et al. 2014. *Bayesian data analysis, 3rd edition*. Chapter 5. 

Efron, Bradley, and Carl N. Morris. Stein's paradox in statistics. WH Freeman, 1977.

Gelman, Andrew, Jennifer Hill, and Masanao Yajima. "Why we (usually) don't have to worry about multiple comparisons." Journal of Research on Educational Effectiveness 5.2 (2012): 189-211.


Chapter 7: Bayesian hierarchical models
============================

## Big picture

Everything that we've done so far in this course has laid a foundation to understand the main course: Bayesian hierarchical models. 

#### Learning goals

- varying intercepts (NBA freethrow example) with `Stan`
- Bayesian vs. MLE approaches
- binomial-Poisson hierarchy (e.g. # eggs laid & survival)
- non-centered parameterizations
- multivariate normal distribution
- priors for hierarchical variance parameters
- prediction (new vs. observed groups)
- connections to random, fixed, & mixed effects

## Bayesian hierarchical models

The main difference between the hierarchical models of the previous chapter and Bayesian hierarchical models is the inclusion of a prior distribution on the hyperparameters (leading to each parameter being represented as a random variable with a probability distribution, consistent with the Bayesian philosophy). 
The non-Bayesian hierarchical models of the previous chapter are semi-Bayesian in that they induce a prior distribution on some but not all parameters. 
Fully Bayesian hierarchical models incorporate uncertainty in the hyperparameters, which is typically considerable. 
Bayesian approaches tend to be much easier than frequentist or MLE approaches in terms of estimation, interval construction, and prediction for all but the simplest models. 

Bayesian hierarchical models are an incredibly powerful and flexible tool for learning about the world. 
Their usefulness is derived from the ability to combine simple model components to make models that are sufficient to represent even complex processes. 
For instance, consider the following example: 

## Binomial-Poisson hierarchy

Suppose you study birds, and you'd like to estimate fitness as measured by egg output and egg survival to fledging. 
There are two response variables and they are connected. 
On the one hand, you might be interested in a model that uses the number of eggs laid by individual $i$ as a response: 

$$y_i \sim Poisson(\lambda)$$

where $\lambda$ is the expected number of eggs laid. 

Further, the number of surviving fledglings is a function of $Y_i$. 
If each egg survives indepdently (or with the addtion of covariates, conditionally independently) with probability $p$, then the number of fledglings $Z_i$ can be considered a binomial random variable: 

$$z_i \sim Binomial(y_i, p)$$

The posterior distribution of the parameters can be written as:

$$[ p, \lambda \mid y, z] = [z \mid y, p] [y \mid \lambda] [\lambda, p]$$

This is a complicated model that arises from fairly simple parts. 
We haven't yet specified the form of our prior distribution for $\lambda$ and $p$. 
The simplest prior might be something that assumes that all individuals $i=1, ..., n$ have the same values. 
For instance, suppose that we expected ahead of time that each bird would lay up to 7 eggs, with an expected value of about 2.5. 
We might then choose a lognormal prior for $\lambda$, with mean 1 and standard deviation .5:


```r
x <- seq(0, 20, .01)
dx <- dgamma(x, 2.5, scale=1.1)
plot(x, dx, type='l', 
     xlab=expression(lambda), 
     ylab=expression(paste("[", lambda, "]")))
```

![](main_files/figure-html/unnamed-chunk-112-1.png) 

```r
# What is the 97.5% quantile?
qgamma(.975, 2.5, scale=1.1)
```

```
## [1] 7.057876
```

If we thought that mean survival probability was about .375, but as low as zero and as high as .8, we could choose a beta prior for $p$


```r
x <- seq(0, 1, .01)
dx <- dbeta(x, 3, 5)
plot(x, dx, type='l', xlab='p', ylab='[p]')
```

![](main_files/figure-html/unnamed-chunk-113-1.png) 

```r
pbeta(.9, 3, 5)
```

```
## [1] 0.9998235
```

These priors would complete our specification of our Bayesian hierarchical model. 

$$y_i \sim Poisson(\lambda)$$

$$z_i \sim Binomial(y_i, p)$$

$$\lambda \sim gamma(2.5, 1.1)$$

$$p \sim beta(3, 5)$$

Suppose we have data from 100 females, and we'd like to update our priors given this new information. 
First, visualizing the data a bit:


```r
library(dplyr)
library(ggplot2)
d <- read.csv('eggs.csv')

# calculate proportion of eggs surviving
d <- d %>%
  mutate(p_survive = fledglings / eggs)
tbl_df(d)
```

```
## Source: local data frame [200 x 5]
## 
##     eggs fledglings bird_mass population p_survive
##    (int)      (int)     (dbl)     (fctr)     (dbl)
## 1      1          0 0.2278972          A      0.00
## 2      2          1 0.7182951          A      0.50
## 3      2          2 1.4752230          A      1.00
## 4      3          3 0.7623340          A      1.00
## 5      4          3 2.7034917          A      0.75
## 6      4          3 3.7814117          A      0.75
## 7      2          1 8.4822438          A      0.50
## 8      0          0 0.1786772          A       NaN
## 9      3          0 4.6517436          A      0.00
## 10     0          0 4.9095815          A       NaN
## ..   ...        ...       ...        ...       ...
```

```r
ggplot(d, aes(x=eggs)) + 
  geom_histogram()
```

![](main_files/figure-html/unnamed-chunk-114-1.png) 

```r
ggplot(d, aes(x=p_survive)) + 
  geom_histogram()
```

![](main_files/figure-html/unnamed-chunk-114-2.png) 

Notice that for birds that did not lay eggs, we have NA values in the proportion of eggs surviving. 
As a result, these individuals will not provide information on $p$ in the above model, because they cannot contribute to a binomial likelihood with $k=0$.

We can translate the model outlined above to Stan as follows to produce the file `eggs.stan`:

```
data {
  // poisson data
  int n;
  int y[n];
  
  // binomial data
  int n_b;
  int k[n_b];
  int z[n_b];
}

parameters {
  real<lower=0> lambda;
  real<lower=0, upper=1> p;
}

model {
  // priors
  lambda ~ gamma(2.5, 1.1);
  p ~ beta(3, 5);
  
  // likelihood
  y ~ poisson(lambda);
  z ~ binomial(k, p);
}
```

Now we can bundle our data to work with the model and estimate the parameters:


```r
stan_d <- list(
  n = nrow(d), 
  y = d$eggs, 
  n_b = sum(d$eggs > 0), 
  k = d$eggs[d$eggs > 0], 
  z = d$fledglings[d$eggs > 0]
)

library(rstan)
m <- stan('eggs.stan', data=stan_d)
```


```r
m
```

```
## Inference for Stan model: eggs.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##           mean se_mean   sd    2.5%     25%     50%     75%   97.5% n_eff
## lambda    2.42    0.00 0.11    2.21    2.34    2.42    2.50    2.65  2245
## p         0.62    0.00 0.02    0.57    0.60    0.62    0.63    0.66  2642
## lp__   -384.53    0.03 1.00 -387.26 -384.91 -384.23 -383.81 -383.55  1338
##        Rhat
## lambda    1
## p         1
## lp__      1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:16:54 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(m)
```

![](main_files/figure-html/unnamed-chunk-116-1.png) 

## Non-centered parameterizations for random effects

The previous model is fairly unsatisfactory in that it assumes that all individuals have the same fecundity and egg to fledgling survival. 
Expecting that this assumption is probably false, we may wish to allow individuals to vary in these quantities. 
One way to do this would be with a normal (on the link scale) random effect, exactly like log-normal overdispersion and logit-normal overdispersion for the Poisson and binomial examples covered earlier. 
For instance, we might write such a model as:

$$y_i \sim Poisson(\lambda)$$

$$z_i \sim Binomial(y_i, p)$$

$$log(\lambda) \sim N(\mu_\lambda, \sigma_\lambda)$$

$$logit(p) \sim N(\mu_p, \sigma_p)$$

$$\mu_\lambda \sim N(0, 1)$$

$$\sigma_\lambda \sim halfNormal(0, 2)$$

$$\mu_p \sim N(0, 2)$$

$$\sigma_p \sim halfNormal(0, 1.5)$$

For the purposes of illustration, we've provided somewhat vague priors, but one could adapt these to reflect the priors that we expressed in the simpler model. 
Much has been written on how to choose priors for hierarchical variance parameters.
The main take home is to avoid hard upper limits and instead to use priors that reflect your previous beliefs with soft constraints, such as half-Cauchy, half-t, or half-normal.
See the further reading section for a good reference on selecting good priors for these hyperparameters. 
In this model, the log fecundity and logit survival probabilities are drawn from independent normal distributions, allowing for individual variation around the population means. 

Updating our model and calling it `egg_ranef.stan`, we might get:

```
data {
  // poisson data
  int n;
  int y[n];
  
  // binomial data
  int n_b;
  int k[n_b];
  int z[n_b];
}

parameters {
  vector[n] log_lambda;
  vector[n_b] logit_p;
  real mu_lambda;
  real mu_p;
  real<lower=0> sigma_lambda;
  real<lower=0> sigma_p;
}

model {
  // priors
  mu_lambda ~ normal(0, 1);
  mu_p ~ normal(0, 2);
  sigma_lambda ~ normal(0, 2);
  sigma_p ~ normal(0, 1.5);
  
  log_lambda ~ normal(mu_lambda, sigma_lambda);
  logit_p ~ normal(mu_p, sigma_p);
  
  // likelihood
  y ~ poisson_log(log_lambda);
  z ~ binomial_logit(k, logit_p);
}
```

Fitting the new model:


```r
m <- stan('egg_ranef.stan', data=stan_d, 
          pars=c('mu_lambda', 'mu_p', 'sigma_lambda', 'sigma_p', 'lp__'))
```

```
## The following numerical problems occured the indicated number of times on chain 1
## If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## The following numerical problems occured the indicated number of times on chain 2
## If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## The following numerical problems occured the indicated number of times on chain 3
## If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
## The following numerical problems occured the indicated number of times on chain 4
## If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
```


```r
traceplot(m)
```

![](main_files/figure-html/unnamed-chunk-118-1.png) 

```r
m
```

```
## Inference for Stan model: egg_ranef.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##                 mean se_mean    sd    2.5%     25%     50%     75%   97.5%
## mu_lambda       0.56    0.00  0.09    0.38    0.51    0.56    0.62    0.72
## mu_p            0.15    0.00  0.24   -0.33   -0.02    0.15    0.31    0.61
## sigma_lambda    0.80    0.00  0.08    0.66    0.75    0.80    0.85    0.96
## sigma_p         2.24    0.02  0.33    1.65    2.01    2.22    2.44    2.94
## lp__         -347.11    1.00 21.03 -389.44 -360.96 -346.17 -333.00 -308.85
##              n_eff Rhat
## mu_lambda     1517 1.00
## mu_p          4000 1.00
## sigma_lambda  1019 1.00
## sigma_p        425 1.02
## lp__           440 1.02
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:17:30 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

It turns out that this parameterization is not optimal. 
We can greatly increase the efficiency of our model by using a "non-centered" parameterization for the normal random effects. 
The basis for this lies in the fact that we can recover a random vector $y \sim N(\mu, \sigma)$ by first generating a vector of standard normal variates: $y_{raw} \sim N(0, 1)$, and then translating the sample to the mean and rescaling all values by $\sigma$: 

$$ y = \mu + y_{raw} \sigma $$

This is more efficient because the MCMC algorithm used with Stan is highly optimized to sample from posteriors with geometry corresponding to N(0, 1) distributions. 
This trick is incredibly useful for nearly all hierarchical models in Stan that use normal random effects.
We can do this translation and scaling in the transformed parameters block, generating the following file called `egg_ncp.Stan`:

```
data {
  // poisson data
  int n;
  int y[n];

  // binomial data
  int n_b;
  int k[n_b];
  int z[n_b];
}

parameters {
  vector[n] log_lambdaR;
  vector[n_b] logit_pR;
  real mu_lambda;
  real mu_p;
  real<lower=0> sigma_lambda;
  real<lower=0> sigma_p;
}

transformed parameters {
  vector[n] log_lambda;
  vector[n_b] logit_p;
  
  log_lambda <- mu_lambda + log_lambdaR * sigma_lambda;
  logit_p <- mu_p + logit_pR * sigma_p;
}

model {
  // priors
  mu_lambda ~ normal(0, 1);
  mu_p ~ normal(0, 2);
  sigma_lambda ~ normal(0, 2);
  sigma_p ~ normal(0, 1.5);

  log_lambdaR ~ normal(0, 1);
  logit_pR ~ normal(0, 1);

  // likelihood
  y ~ poisson_log(log_lambda);
  z ~ binomial_logit(k, logit_p);
}
```

Fitting our new model:


```r
m <- stan('egg_ncp.stan', data=stan_d)
```


```r
traceplot(m, pars=c('mu_lambda', 'mu_p', 'sigma_lambda', 'sigma_p', 'lp__'))
```

![](main_files/figure-html/unnamed-chunk-120-1.png) 

```r
print(m, pars=c('mu_lambda', 'mu_p', 'sigma_lambda', 'sigma_p', 'lp__'))
```

```
## Inference for Stan model: egg_ncp.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##                 mean se_mean    sd    2.5%     25%     50%     75%   97.5%
## mu_lambda       0.56    0.00  0.09    0.38    0.50    0.56    0.62    0.72
## mu_p            0.16    0.01  0.23   -0.31    0.01    0.16    0.32    0.61
## sigma_lambda    0.81    0.00  0.08    0.67    0.75    0.80    0.85    0.97
## sigma_p         2.24    0.01  0.33    1.65    2.02    2.22    2.45    2.95
## lp__         -270.42    0.73 18.79 -307.52 -283.31 -269.98 -257.34 -234.43
##              n_eff Rhat
## mu_lambda     2430    1
## mu_p          1698    1
## sigma_lambda  1174    1
## sigma_p       1249    1
## lp__           654    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:18:08 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

We might be interested in whether there is any correlation between egg output and egg survival. 
To explore this, we can plot the random effects on the link scale:


```r
post <- extract(m)
ll_meds <- apply(post$log_lambda, 2, median)
lp_meds <- apply(post$logit_p, 2, median)
plot(exp(ll_meds[d$eggs > 0]), plogis(lp_meds))
```

![](main_files/figure-html/unnamed-chunk-121-1.png) 

That plot is not so informative, but if we are interested in the correlation, we can simply calculate the correlation for each draw to get the posterior distribution for the correlation. 
This is one of the huge advantages of Bayesian inference: we can calculate the posterior distribution for any derived parameters using posterior draws. 


```r
n_iter <- length(post$lp__)
cor_post <- rep(NA, n_iter)
for (i in 1:n_iter){
  cor_post[i] <- cor(post$log_lambda[i, d$eggs > 0], 
                     post$logit_p[i, ])
}
hist(cor_post, breaks=seq(-1, 1, .02))
```

![](main_files/figure-html/unnamed-chunk-122-1.png) 

It appears that birds that produce many eggs tend to have higher per-egg survival. 
But, we haven't included this correlation in the model explicitly. 
Generally speaking correlations between two random variables A and B can result from three causal scenarios: 

- A or B have a causal effect on eachother, directly or indirectly
- A and B are both affected by some other quantity or quantities
- we have conditioned on a variable that is influence by A and B (also known as Berkson's paradox)

In this case, we can model correlation between these two latent quantities by way of multivariate normal random effects rather than two independent univariate normal random effects. 

## Multivariate normal random effects

In many cases, we have multiple random effects which may be correlated. 
In these instances, many turn to the multivariate normal distribution, which generalizes the univariate normal distribution to N dimensions. 
The multivariate normal distribution has two parameters: $\mu$, which is a vector with $N$ elements, each describing the mean of the distribution in each dimension, and $\Sigma$, an $N$ by $N$ covariance matrix that encodes the variance in each dimension and correlation among dimensions. 
Any multivariate normal random vector will be a point in $N$ dimensional space. 

Consider the bivariate normal distribution, a multivariate normal with $N=2$ dimensions. 
The mean vector will have two elements $\mu_1$ and $\mu_2$, that provide the center of mass in the first and second dimension. 
The covariance matrix will have two rows and columns. 
We might write these parameters as follows: 

$\boldsymbol{\mu} = \begin{bmatrix}
\mu_1 \\
\mu_2
\end{bmatrix},$
$\boldsymbol{\Sigma} = \begin{bmatrix}
Cov[X_1, X_1] & Cov[X_1, X_2] \\
Cov[X_2, X_1] & Cov[X_2, X_2]
\end{bmatrix}$

The element of $Sigma$ in the $i^{th}$ row and $j^{th}$ column describes the covariance between the $i^{th}$ and $j^{th}$ dimension. 
By definition, the covariance between one random variable and itself (e.g., $Cov[X_1, X_1]$ and $Cov[X_2, X_2]$) is the variance of the random variable, $\sigma^2_{X_1}$ and $\sigma^2_{X_2}$.

For concreteness, suppose that we're considering the following multivariate normal distribution: 

$\boldsymbol{\mu} = \begin{bmatrix}
0 \\
0
\end{bmatrix},$
$\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0.5 \\
0.5 & 1
\end{bmatrix}$

We can visualize the density of this bivariate normal as with a heatmap: 

![](main_files/figure-html/unnamed-chunk-123-1.png) 

### Non-centered parameterization: multivariate normal

As with univariate normal random effects, a noncentered parameterization can greatly improve MCMC convergence and efficiency. 
To acheive this, we first define the Cholesky factor of a matrix $L$, which is lower triangular, and which equals sigma when multiplied by it's own transpose: $\Sigma = L L^{T}$. 
Given $L$, which is a lower triangular $d$ by $d$ matrix, $\mu$, which is the mean vector with length $d$, and $z$, which is a vector of $d$ standard normal N(0, 1) deviates, we can generate a draw from $d$ dimensional multivariate normal distribution $MvN(\mu, \Sigma)$ as follows: 

$$y = \mu + L z$$

Sometimes, it is convenient to parameterize the multivariate normal in terms of a Cholesky decomposed correlation matrix $L_R$ such that $L_R L_R^T = R$ and a vector of standard deviations $\sigma$, which can be coerced into a diagonal matrix that has the same dimensions as the desired covariance matrix. 
If we have these, then we can adapt the above equation to obtain:

$$y = \mu + diag(\sigma) L_R z$$

This parameterization is most useful for hierarchical models, because we can place separate priors on correlation matrices and on the standard deviations. 
For correlation matrices, it is currently recommended to use LKJ priors, which can be specified on the cholesky decomposed matrix (obviating the need for Cholesky decompositions at each MCMC iteration). 
The LKJ correlation distribution has one parameter that specifies how concentrated the correlations are around a uniform distribution $\eta = 1$, or the identity matrix with all correlations (non-diagonal elements) equal to zero when $\eta$ is very large. 
An LKJ correlation with $\eta=2$ implies a prior in which correlations are somewhat concentrated on zero. 
Below are the LKJ prior correlations implied by different values of $\eta$.

![](main_files/figure-html/unnamed-chunk-124-1.png) 

Let's expand the above model to explicitly allow for correlation between egg survival and egg output. 
This tends to be useful computationally when parameters are correlated, but it also may be of practical use if egg output or survival are incompletely observed and we'd like to predict the missing data using information on correlated quantities. 
The main difference will be that instead of two separate univariate normal random effects, we instead have one bivariate normal distribution, and we're modeling correlation between the two dimensions. 

$$y_i \sim Poisson(\lambda)$$

$$z_i \sim Binomial(y_i, p)$$

$$log(\lambda_i) = \alpha_{1i}$$

$$logit(p) = \alpha_{2i}$$

$$\alpha_i \sim N(\mu, \Sigma)$$

$$\mu \sim N(0, 2)$$

$$\Sigma = (diag(\sigma) L_R) (diag(\sigma) L_R)^T$$

$$\sigma \sim halfNormal(0, 2)$$

$$L_R \sim LKJcorr(2)$$

With this new parameterization, we can estimate a random effect vector $\alpha_i$ of length 2 for each individual $i=1, ..., N$, with elements corresponding to the log expected number of eggs and logit probability of survival for each egg. 
However, recall that not all individual contribute to the likelihood for egg survival. 
In particular, we have no survival data from birds that laid zero eggs. 
This bivariate approach allows us to combine information so that the number of eggs laid informs our estimates of survival probabilities. 
In this way, we will be able to predict the survival probability of eggs from individuals that did not lay eggs. 
Here is a Stan model statement, saved in the file `egg_lkj.stan`:

```
data {
  // poisson data
  int n;
  int y[n];

  // binomial data
  int n_b;
  int p_index[n_b];
  int k[n_b];
  int x[n_b];
}

parameters {
  matrix[2, n] z;
  vector[2] mu;
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] sigma;
}

transformed parameters {
  matrix[n, 2] alpha;
  vector[n] log_lambda;
  vector[n_b] logit_p;
  alpha <- (diag_pre_multiply(sigma, L) * z)';
  
  for (i in 1:n) log_lambda[i] <- alpha[i, 1];
  log_lambda <- log_lambda + mu[1];
  
  for (i in 1:n_b){
    logit_p[i] <- alpha[p_index[i], 2];
  }
  logit_p <- logit_p + mu[2];
}

model {
  // priors
  mu ~ normal(0, 2);
  sigma ~ normal(0, 2);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ normal(0, 1);

  // likelihood
  y ~ poisson_log(log_lambda);
  x ~ binomial_logit(k, logit_p);
}

generated quantities {
  // recover the correlation matrix
  matrix[2, 2] Rho;
  
  Rho <- multiply_lower_tri_self_transpose(L);
}
```

Again, here the idea is to not directly sample from the multivariate normal distribution, but instead to sample from a simpler distribution (univariate standard normal), and transform these values using the cholesky factor of the correlation matrix, vector of standard deviations, and vector of means to generate multivariate normal parameters. 
It is possible to sample directly from the multivariate normal distribution, but this approach is much more computationally efficient.
We have to generate the indexes for the survival observations, bundle the data, and then we can fit the model:


```r
p_ind <- which(d$eggs > 0)
stan_d <- list(
  n = nrow(d), 
  y = d$eggs, 
  n_b = sum(d$eggs > 0), 
  p_index = p_ind,
  k = d$eggs[d$eggs > 0], 
  x = d$fledglings[d$eggs > 0]
)
m <- stan('egg_lkj.stan', data=stan_d, 
          pars=c('Rho', 'alpha', 'sigma', 'mu'))
```

Let's check convergence for the hyperparameters, which usually implies convergence of the child parameters:


```r
print(m, pars=c('Rho', 'sigma', 'mu', 'lp__'))
```

```
## Inference for Stan model: egg_lkj.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##             mean se_mean    sd    2.5%     25%     50%     75%   97.5%
## Rho[1,1]    1.00    0.00  0.00    1.00    1.00    1.00    1.00    1.00
## Rho[1,2]    0.85    0.00  0.08    0.66    0.80    0.86    0.91    0.97
## Rho[2,1]    0.85    0.00  0.08    0.66    0.80    0.86    0.91    0.97
## Rho[2,2]    1.00    0.00  0.00    1.00    1.00    1.00    1.00    1.00
## sigma[1]    0.78    0.00  0.07    0.65    0.73    0.78    0.83    0.93
## sigma[2]    2.35    0.01  0.34    1.76    2.11    2.32    2.56    3.09
## mu[1]       0.57    0.00  0.08    0.41    0.52    0.58    0.63    0.72
## mu[2]      -0.63    0.01  0.27   -1.17   -0.81   -0.62   -0.45   -0.13
## lp__     -296.32    0.76 18.84 -332.81 -308.63 -296.60 -283.84 -259.58
##          n_eff Rhat
## Rho[1,1]  4000  NaN
## Rho[1,2]   504 1.01
## Rho[2,1]   504 1.01
## Rho[2,2]  3927 1.00
## sigma[1]  1191 1.00
## sigma[2]  1062 1.00
## mu[1]     1394 1.00
## mu[2]     1208 1.00
## lp__       621 1.00
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:19:35 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

```r
traceplot(m, pars=c('Rho', 'sigma', 'mu', 'lp__'))
```

![](main_files/figure-html/unnamed-chunk-126-1.png) 

In this example, we might proceed by adding "fixed" effects of body mass, so that we can evaluate how much of the correlation between clutch size and survival may be related to body size. 
We leave this as an exercise, but point out that this can be accomplished using design matrices or for-loops in the Stan file. 

## Varying intercepts and slopes

Some of the most common applications of Bayesian hierarchical models involve intercept and slope parameters that vary among groups. 
In these cases, it is often wise to allow the intercepts and slopes to correlate, and this is mostly accomplished via multivariate normal random effects, where one dimension corresponds to intercepts, and the other to slopes. 
To demonstrate this, we will use a classic example from a sleep study in which the reaction times of 18 subjects was measured daily with increasing levels of sleep deprivation. 


```r
library(lme4)
ggplot(sleepstudy, aes(x=Days, y=Reaction)) + 
  geom_point() + 
  stat_smooth(method='lm') +
  facet_wrap(~ Subject)
```

![](main_files/figure-html/unnamed-chunk-127-1.png) 

We might envision the following model that allows the intercepts (reaction on day 0) and slope (daily change in expected reaction time) to vary among subjects, with normally distributed error, indexing subjects by $i$ and days by $t$:

$$y_{it} \sim N(\mu_{it}, \sigma_y)$$

$$\mu_{it} = \alpha_i + \beta_i t$$

$$\begin{bmatrix} 
\alpha_i \\
\beta_i
\end{bmatrix} \sim N \bigg(
\begin{bmatrix} 
\mu_\alpha \\
\mu_\beta
\end{bmatrix}, 
\Sigma \bigg)$$

We can implement this model in `lme4` if we want:


```r
mle <- lmer(Reaction ~ Days + (Days | Subject), data=sleepstudy)
```

Or, we could translate the model to Stan:

```
data {
  int n;
  vector[n] y;
  int n_subject;
  int n_t;
  
  // indices
  int<lower=1, upper=n_subject> subject[n];
  int<lower=1, upper=n_t> t[n];
}

parameters {
  matrix[2, n_subject] z;
  vector[2] mu;
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] sigma;
  real<lower=0> sigma_y;
}

model {
  to_vector(z) ~ normal(0, 1);
}
```


```r
stan_d <- list(n = nrow(sleepstudy), 
               y = sleepstudy$Reaction, 
               tmax = max(sleepstudy$Days), 
               t = sleepstudy$Days, 
               n_subject = max(as.numeric(sleepstudy$Subject)), 
               subject = as.numeric(sleepstudy$Subject))
m <- stan('sleep.stan', data=stan_d, 
          pars = c('mu', 'sigma', 'sigma_y', 'alpha', 'Rho'))
```

Checking convergence:


```r
traceplot(m, pars = c('mu', 'sigma', 'sigma_y', 'Rho'))
```

![](main_files/figure-html/unnamed-chunk-130-1.png) 

```r
print(m, pars = c('mu', 'sigma', 'sigma_y', 'Rho'))
```

```
## Inference for Stan model: sleep.
## 4 chains, each with iter=2000; warmup=1000; thin=1; 
## post-warmup draws per chain=1000, total post-warmup draws=4000.
## 
##            mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## mu[1]    250.52    0.16 6.73 237.26 246.02 250.51 254.87 263.97  1861    1
## mu[2]     10.67    0.04 1.67   7.52   9.58  10.62  11.69  14.13  1670    1
## sigma[1]  24.38    0.14 6.19  13.87  20.01  23.82  28.08  37.77  1866    1
## sigma[2]   6.26    0.04 1.41   4.06   5.29   6.08   7.00   9.68  1566    1
## sigma_y   25.79    0.02 1.54  23.01  24.69  25.70  26.77  29.11  4000    1
## Rho[1,1]   1.00    0.00 0.00   1.00   1.00   1.00   1.00   1.00  4000  NaN
## Rho[1,2]   0.10    0.01 0.27  -0.41  -0.09   0.10   0.28   0.61  1191    1
## Rho[2,1]   0.10    0.01 0.27  -0.41  -0.09   0.10   0.28   0.61  1191    1
## Rho[2,2]   1.00    0.00 0.00   1.00   1.00   1.00   1.00   1.00  4000    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 29 23:20:17 2015.
## For each parameter, n_eff is a crude measure of effective sample size,
## and Rhat is the potential scale reduction factor on split chains (at 
## convergence, Rhat=1).
```

These results are not very different than those of the `lmer` implementation, but with a Bayesian implementation we immediately have the full posterior distribution for every parameter, which is a huge advantage. 

## Further reading

Hobbs and Hooten. 2015. *Bayesian models: a statistical primer for ecologists*. Chapter 6. 

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 13.

Gelman, A., J. Hill, and M. Yajima. 2012. Why We (Usually) Don’t Have to Worry About Multiple Comparisons. Journal of Research on Educational Effectiveness 5:189–211.  

Gelman, Andrew. Prior distributions for variance parameters in hierarchical models (comment on article by Browne and Draper). Bayesian Anal. 1 (2006), no. 3, 515--534.


Chapter 8: Hierarchical model construction
===============================

## Big picture
Translating problems to models is a key skill, and it may take a fair bit of practice. 
In this chapter we introduce tools to help with this step: parameter, process, and observation models, and a few examples that make use of familiar probability distributions.

#### Learning goals
- building models from simple submodels
- parameter, process, and observation models
- examples:  
-- occupancy model  
-- N-mixture model  
-- error in variables models

## Parameter, process, and observation models

One incredibly powerful perspective in model construction is the distinction between observation, process, and parameter models. 
The observation model includes the likelihood, and relates the latent process of interest to the quantities that we have actually observed. 
The process model is typically what we're most interested in scientifically, and typically is not fully observed. 
The parameter model includes any hierarchical structures on parameters and priors for hyperparameters. 

### Example: occupancy model

In occupancy models, we make binary observations (detected, not detected) on latent binary quantities (present, not present), and we must have priors for the detection and occurrence parameters to complete a Bayesian formulation (the parameter model).
If we visit sites $i=1, ..., n$, and conduct $j=1, ..., J$ surveys at each then we can write out these components as follows:

**Observation model**

Here our detection and nondetection data $y$ arise as a function of the latent occurrence state $z$ and probability of detection $p$. 
Note that the indexing indicates that the occurrence state $z$ does not vary among surveys.

$$y_{ij} \sim Bern(J, z_i p_{ij})$$

**Process model**

Here, the species is present $z=1$ or absent $z=0$ with some probability $\psi$. 

$$z_i \sim Bern(\psi_i)$$

**Parameter model**

Finally, priors are specified for the occurrence and detection probabilities.

$$\psi_i \sim ...$$

$$p_{ij} \sim ...$$

### Example: N-mixture model

We are often interested in estimating and explaining population sizes in wild organisms, and mark-recapture methods are not always feasible. 
Suppose we visit site $i$ multiple times $j=1, ..., J$, and on each visit we conduct a survey, recording the number of individuals observed.
If we can assume that across visits, the true abundance $N_i$ is unchanged, and that the detection of each individual is imperfect but independent, then we might construct an N-mixture model to estimate abundance. 

**Observation model**

We observe a subset of the individuals at the site, and each individual is detected with probability $p_i$:

$$y_{ij} \sim Binom(N_i, p_i)$$

Notice that both parameters of the binomial observation model are unknown. 
True abundance $N_i$ is a latent quantity.

**Process model**

We need some discrete probability distribution for $N$, such as the Poisson or negative binomial. 

$$N_i \sim Pois(\lambda_i)$$

**Parameter model**

Finally, we need priors for $p$ and $\lambda$.

$$p \sim ...$$

$$\lambda \sim ...$$

We note that this model has integer parameters, and as such cannot be fitted (presently) with Stan, but this is easy to implement in JAGS. 

### Example: error in variables models

Most of the time in ecology, covariates are observed with error and subsequently assumed to be known exactly. 
This causes bias in slope estimates, and despite the fact that this has been known for over a century, most people carry on assuming that the covariate is fixed. 

![](ch8/m2.gif)

For what follows, we'll assume a simple linear regression, in which continuous covariates are measured with error. 
But, this approach can be applied to any of the models that we've covered in this class.


True covariate values are considered latent variables, with repeated measurements of covariate values $x_{i=1}, ..., x_{i=n}$ arising from a normal distribution with a mean equal to the true value, and some measurement error $\sigma_x$.
Again, this is a special case, but in principle the covariate could have other distributions (e.g., you're interested in the effect of sex (M or F), but this is measured imperfectly).

Borrowing from some graphical tools common in structural equation modeling, we can represent the latent variables in the model as circles, and observables as boxes:

![](ch8/lv.svg)

with $\epsilon_x \sim Normal(0, \sigma_x)$ and $\epsilon_y \sim Normal(0, \sigma_y)$.

**Observation model**

We assume that for sample unit $i$ and repeat measurement $j$:

$$ x^{obs}_{ij} \sim Normal(x_i, \sigma_x) $$

$$ y \sim Normal(\mu_y, \sigma_y) $$

The trick here is to use repeated measurements of the covariates to estimate and correct for measurement error.
In order for this to be valid, the true covariate values cannot vary across repeat measurements.
If the covariate was individual weight, you would have to ensure that the true weight did not vary across repeat measurements (for me, frogs urinating during handling would violate this assumption).

**Process model**

This is what we're typically interested in: how the expected value of $y$ varies with $x$. 
In this case, $x$ is a parameter (it will require a prior).

$$\mu_y = \alpha + \beta x_i$$

**Parameter model**

In the parameter model, we would specify priors for $x$, $\alpha$, $\beta$, and the variance parameters. 
More on this example can be found [here](http://mbjoseph.github.io/2013/11/27/measure.html). 


Chapter 9: Model comparison
================

I envisions this as occuring a bit more ad hoc during the second half as students start to build their own models

- start with simpler models and build (may be counterintuitive for those used to step-down procedures)
- posterior prediction
- DIC, wAIC
- cross-checking
- simulated data

#### Reading

Hooten, M. B., and N. T. Hobbs. 2015. A guide to Bayesian model selection for ecologists. Ecological Monographs 85:3–28.  
