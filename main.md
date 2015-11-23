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
## -1.4116 -0.4933 -0.1108  0.3549  2.3596 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)  
## (Intercept)  -0.5171     0.2330  -2.219   0.0388 *
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.042 on 19 degrees of freedom
```

The summary of our model object `m` provides a lot of information. 
For reasons that will become clear shortly, the estimated population mean is referred to as the "Intercept". 
Here, we get a point estimate for the population mean $\mu$: -0.517 and an estimate of the residual standard deviation $\sigma$: 1.042, which we can square to get an estimate of the residual variance $\sigma^2$: 1.085.

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
## -1.02824 -0.22148 -0.03891  0.24722  1.35852 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -2.0258     0.1293  -15.67   <2e-16 ***
## x             2.9804     0.2302   12.95   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4602 on 48 degrees of freedom
## Multiple R-squared:  0.7774,	Adjusted R-squared:  0.7728 
## F-statistic: 167.6 on 1 and 48 DF,  p-value: < 2.2e-16
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
## (Intercept) -4.1419070 -2.628167
## x            0.9002774  1.325891
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
##     Min      1Q  Median      3Q     Max 
## -1.6912 -0.3703 -0.0450  0.5621  1.5876 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   3.3634     0.1737  19.359  < 2e-16 ***
## xstrawberry  -2.2742     0.2457  -9.256 5.91e-13 ***
## xvanilla     -1.9530     0.2457  -7.949 8.36e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.777 on 57 degrees of freedom
## Multiple R-squared:  0.6378,	Adjusted R-squared:  0.6251 
## F-statistic: 50.19 on 2 and 57 DF,  p-value: 2.689e-13
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
##     Min      1Q  Median      3Q     Max 
## -1.6912 -0.3703 -0.0450  0.5621  1.5876 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## xchocolate    3.3634     0.1737  19.359  < 2e-16 ***
## xstrawberry   1.0892     0.1737   6.269 5.19e-08 ***
## xvanilla      1.4104     0.1737   8.118 4.38e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.777 on 57 degrees of freedom
## Multiple R-squared:  0.8939,	Adjusted R-squared:  0.8883 
## F-statistic:   160 on 3 and 57 DF,  p-value: < 2.2e-16
```

Arguably, this approach is more useful because it simplifies the construction of confidence intervals for the group means:


```r
confint(m)
```

```
##                2.5 %   97.5 %
## xchocolate  3.015475 3.711273
## xstrawberry 0.741300 1.437098
## xvanilla    1.062467 1.758265
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
##      Min       1Q   Median       3Q      Max 
## -2.16850 -0.67930  0.07175  0.49410  2.35840 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.4512     0.1546   2.918  0.00543 ** 
## x1            1.2794     0.1627   7.864 4.70e-10 ***
## x2           -0.9921     0.1391  -7.131 5.79e-09 ***
## x1:x2         2.0736     0.1434  14.463  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.066 on 46 degrees of freedom
## Multiple R-squared:  0.8496,	Adjusted R-squared:  0.8398 
## F-statistic: 86.59 on 3 and 46 DF,  p-value: < 2.2e-16
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
##      Min       1Q   Median       3Q      Max 
## -0.84109 -0.17126 -0.00348  0.21522  0.67901 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.89467    0.07135  12.539  < 2e-16 ***
## x1           1.06452    0.06911  15.403  < 2e-16 ***
## x2B         -0.70326    0.09845  -7.143 5.55e-09 ***
## x1:x2B      -0.87058    0.09842  -8.846 1.73e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3431 on 46 degrees of freedom
## Multiple R-squared:  0.8773,	Adjusted R-squared:  0.8693 
## F-statistic: 109.7 on 3 and 46 DF,  p-value: < 2.2e-16
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
n <- 50
y <- rnorm(n, mu, sigma)

# generate a grid of parameter values to search over
g <- expand.grid(mu = seq(4, 8, length.out=100), 
                 sigma=seq(2, 6, length.out=100))

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
4, 2, -160.3539, 0.627451, 0.1254902, 0.9411765, 1,
4.040404, 2, -159.2681, 0.627451, 0.1254902, 0.9411765, 1,
4.080808, 2, -158.2027, 0.5607843, 0.1098039, 0.945098, 1,
4.121212, 2, -157.1578, 0.4941176, 0.09803922, 0.9529412, 1,
4.161616, 2, -156.1332, 0.4313726, 0.08627451, 0.9568627, 1,
4.20202, 2, -155.129, 0.3647059, 0.07058824, 0.9647059, 1,
4.242424, 2, -154.1453, 0.3647059, 0.07058824, 0.9647059, 1,
4.282828, 2, -153.1819, 0.3019608, 0.05882353, 0.9686275, 1,
4.323232, 2, -152.239, 0.2352941, 0.04705882, 0.9764706, 1,
4.363636, 2, -151.3165, 0.172549, 0.03137255, 0.9803922, 1,
4.40404, 2, -150.4143, 0.1058824, 0.01960784, 0.9882353, 1,
4.444445, 2, -149.5326, 0.1058824, 0.01960784, 0.9882353, 1,
4.484848, 2, -148.6713, 0.04313726, 0.007843138, 0.9921569, 1,
4.525252, 2, -147.8304, 0.03137255, 0, 0.9647059, 1,
4.565657, 2, -147.0099, 0.03137255, 0, 0.9647059, 1,
4.606061, 2, -146.2098, 0.1372549, 0, 0.8588235, 1,
4.646465, 2, -145.4301, 0.2392157, 0, 0.7568628, 1,
4.686869, 2, -144.6708, 0.2392157, 0, 0.7568628, 1,
4.727273, 2, -143.9319, 0.3411765, 0, 0.654902, 1,
4.767677, 2, -143.2134, 0.3411765, 0, 0.654902, 1,
4.808081, 2, -142.5154, 0.4470588, 0, 0.5490196, 1,
4.848485, 2, -141.8377, 0.4470588, 0, 0.5490196, 1,
4.888889, 2, -141.1804, 0.5490196, 0, 0.4470588, 1,
4.929293, 2, -140.5436, 0.5490196, 0, 0.4470588, 1,
4.969697, 2, -139.9271, 0.654902, 0, 0.3411765, 1,
5.010101, 2, -139.3311, 0.654902, 0, 0.3411765, 1,
5.050505, 2, -138.7555, 0.7568628, 0, 0.2392157, 1,
5.090909, 2, -138.2002, 0.7568628, 0, 0.2392157, 1,
5.131313, 2, -137.6654, 0.8588235, 0, 0.1372549, 1,
5.171717, 2, -137.151, 0.8588235, 0, 0.1372549, 1,
5.212121, 2, -136.657, 0.9647059, 0, 0.03137255, 1,
5.252525, 2, -136.1834, 0.9647059, 0, 0.03137255, 1,
5.292929, 2, -135.7302, 0.9647059, 0, 0.03137255, 1,
5.333333, 2, -135.2974, 1, 0.06666667, 0, 1,
5.373737, 2, -134.885, 1, 0.06666667, 0, 1,
5.414141, 2, -134.493, 1, 0.06666667, 0, 1,
5.454545, 2, -134.1215, 1, 0.1686275, 0, 1,
5.494949, 2, -133.7703, 1, 0.1686275, 0, 1,
5.535354, 2, -133.4395, 1, 0.1686275, 0, 1,
5.575758, 2, -133.1292, 1, 0.1686275, 0, 1,
5.616162, 2, -132.8392, 1, 0.2745098, 0, 1,
5.656566, 2, -132.5697, 1, 0.2745098, 0, 1,
5.69697, 2, -132.3206, 1, 0.2745098, 0, 1,
5.737374, 2, -132.0918, 1, 0.2745098, 0, 1,
5.777778, 2, -131.8835, 1, 0.2745098, 0, 1,
5.818182, 2, -131.6956, 1, 0.3764706, 0, 1,
5.858586, 2, -131.5281, 1, 0.3764706, 0, 1,
5.89899, 2, -131.381, 1, 0.3764706, 0, 1,
5.939394, 2, -131.2543, 1, 0.3764706, 0, 1,
5.979798, 2, -131.148, 1, 0.3764706, 0, 1,
6.020202, 2, -131.0621, 1, 0.3764706, 0, 1,
6.060606, 2, -130.9966, 1, 0.3764706, 0, 1,
6.10101, 2, -130.9515, 1, 0.3764706, 0, 1,
6.141414, 2, -130.9268, 1, 0.3764706, 0, 1,
6.181818, 2, -130.9226, 1, 0.3764706, 0, 1,
6.222222, 2, -130.9387, 1, 0.3764706, 0, 1,
6.262626, 2, -130.9753, 1, 0.3764706, 0, 1,
6.30303, 2, -131.0322, 1, 0.3764706, 0, 1,
6.343434, 2, -131.1096, 1, 0.3764706, 0, 1,
6.383838, 2, -131.2074, 1, 0.3764706, 0, 1,
6.424242, 2, -131.3255, 1, 0.3764706, 0, 1,
6.464646, 2, -131.4641, 1, 0.3764706, 0, 1,
6.505051, 2, -131.6231, 1, 0.3764706, 0, 1,
6.545455, 2, -131.8025, 1, 0.2745098, 0, 1,
6.585859, 2, -132.0023, 1, 0.2745098, 0, 1,
6.626263, 2, -132.2225, 1, 0.2745098, 0, 1,
6.666667, 2, -132.4631, 1, 0.2745098, 0, 1,
6.707071, 2, -132.7241, 1, 0.2745098, 0, 1,
6.747475, 2, -133.0055, 1, 0.2745098, 0, 1,
6.787879, 2, -133.3074, 1, 0.1686275, 0, 1,
6.828283, 2, -133.6296, 1, 0.1686275, 0, 1,
6.868687, 2, -133.9722, 1, 0.1686275, 0, 1,
6.909091, 2, -134.3353, 1, 0.06666667, 0, 1,
6.949495, 2, -134.7188, 1, 0.06666667, 0, 1,
6.989899, 2, -135.1226, 1, 0.06666667, 0, 1,
7.030303, 2, -135.5469, 0.9647059, 0, 0.03137255, 1,
7.070707, 2, -135.9915, 0.9647059, 0, 0.03137255, 1,
7.111111, 2, -136.4566, 0.9647059, 0, 0.03137255, 1,
7.151515, 2, -136.9421, 0.8588235, 0, 0.1372549, 1,
7.191919, 2, -137.448, 0.8588235, 0, 0.1372549, 1,
7.232323, 2, -137.9743, 0.8588235, 0, 0.1372549, 1,
7.272727, 2, -138.521, 0.7568628, 0, 0.2392157, 1,
7.313131, 2, -139.0881, 0.7568628, 0, 0.2392157, 1,
7.353535, 2, -139.6756, 0.654902, 0, 0.3411765, 1,
7.393939, 2, -140.2835, 0.654902, 0, 0.3411765, 1,
7.434343, 2, -140.9118, 0.5490196, 0, 0.4470588, 1,
7.474748, 2, -141.5606, 0.5490196, 0, 0.4470588, 1,
7.515152, 2, -142.2297, 0.4470588, 0, 0.5490196, 1,
7.555555, 2, -142.9193, 0.4470588, 0, 0.5490196, 1,
7.59596, 2, -143.6292, 0.3411765, 0, 0.654902, 1,
7.636364, 2, -144.3596, 0.2392157, 0, 0.7568628, 1,
7.676768, 2, -145.1103, 0.2392157, 0, 0.7568628, 1,
7.717172, 2, -145.8815, 0.1372549, 0, 0.8588235, 1,
7.757576, 2, -146.6731, 0.1372549, 0, 0.8588235, 1,
7.79798, 2, -147.485, 0.03137255, 0, 0.9647059, 1,
7.838384, 2, -148.3174, 0.04313726, 0.007843138, 0.9921569, 1,
7.878788, 2, -149.1702, 0.04313726, 0.007843138, 0.9921569, 1,
7.919192, 2, -150.0434, 0.1058824, 0.01960784, 0.9882353, 1,
7.959596, 2, -150.937, 0.172549, 0.03137255, 0.9803922, 1,
8, 2, -151.851, 0.2352941, 0.04705882, 0.9764706, 1,
4, 2.040404, -158.2268, 0.5607843, 0.1098039, 0.945098, 1,
4.040404, 2.040404, -157.1836, 0.4941176, 0.09803922, 0.9529412, 1,
4.080808, 2.040404, -156.16, 0.4313726, 0.08627451, 0.9568627, 1,
4.121212, 2.040404, -155.156, 0.3647059, 0.07058824, 0.9647059, 1,
4.161616, 2.040404, -154.1716, 0.3647059, 0.07058824, 0.9647059, 1,
4.20202, 2.040404, -153.2068, 0.3019608, 0.05882353, 0.9686275, 1,
4.242424, 2.040404, -152.2616, 0.2352941, 0.04705882, 0.9764706, 1,
4.282828, 2.040404, -151.3361, 0.172549, 0.03137255, 0.9803922, 1,
4.323232, 2.040404, -150.4301, 0.172549, 0.03137255, 0.9803922, 1,
4.363636, 2.040404, -149.5437, 0.1058824, 0.01960784, 0.9882353, 1,
4.40404, 2.040404, -148.677, 0.04313726, 0.007843138, 0.9921569, 1,
4.444445, 2.040404, -147.8298, 0.03137255, 0, 0.9647059, 1,
4.484848, 2.040404, -147.0023, 0.03137255, 0, 0.9647059, 1,
4.525252, 2.040404, -146.1943, 0.1372549, 0, 0.8588235, 1,
4.565657, 2.040404, -145.406, 0.2392157, 0, 0.7568628, 1,
4.606061, 2.040404, -144.6373, 0.2392157, 0, 0.7568628, 1,
4.646465, 2.040404, -143.8882, 0.3411765, 0, 0.654902, 1,
4.686869, 2.040404, -143.1586, 0.3411765, 0, 0.654902, 1,
4.727273, 2.040404, -142.4487, 0.4470588, 0, 0.5490196, 1,
4.767677, 2.040404, -141.7584, 0.4470588, 0, 0.5490196, 1,
4.808081, 2.040404, -141.0877, 0.5490196, 0, 0.4470588, 1,
4.848485, 2.040404, -140.4366, 0.654902, 0, 0.3411765, 1,
4.888889, 2.040404, -139.8052, 0.654902, 0, 0.3411765, 1,
4.929293, 2.040404, -139.1933, 0.7568628, 0, 0.2392157, 1,
4.969697, 2.040404, -138.601, 0.7568628, 0, 0.2392157, 1,
5.010101, 2.040404, -138.0283, 0.7568628, 0, 0.2392157, 1,
5.050505, 2.040404, -137.4753, 0.8588235, 0, 0.1372549, 1,
5.090909, 2.040404, -136.9418, 0.8588235, 0, 0.1372549, 1,
5.131313, 2.040404, -136.428, 0.9647059, 0, 0.03137255, 1,
5.171717, 2.040404, -135.9337, 0.9647059, 0, 0.03137255, 1,
5.212121, 2.040404, -135.4591, 1, 0.06666667, 0, 1,
5.252525, 2.040404, -135.0041, 1, 0.06666667, 0, 1,
5.292929, 2.040404, -134.5686, 1, 0.06666667, 0, 1,
5.333333, 2.040404, -134.1528, 1, 0.1686275, 0, 1,
5.373737, 2.040404, -133.7566, 1, 0.1686275, 0, 1,
5.414141, 2.040404, -133.38, 1, 0.1686275, 0, 1,
5.454545, 2.040404, -133.023, 1, 0.2745098, 0, 1,
5.494949, 2.040404, -132.6856, 1, 0.2745098, 0, 1,
5.535354, 2.040404, -132.3678, 1, 0.2745098, 0, 1,
5.575758, 2.040404, -132.0696, 1, 0.2745098, 0, 1,
5.616162, 2.040404, -131.791, 1, 0.2745098, 0, 1,
5.656566, 2.040404, -131.5321, 1, 0.3764706, 0, 1,
5.69697, 2.040404, -131.2927, 1, 0.3764706, 0, 1,
5.737374, 2.040404, -131.0729, 1, 0.3764706, 0, 1,
5.777778, 2.040404, -130.8728, 1, 0.3764706, 0, 1,
5.818182, 2.040404, -130.6922, 1, 0.3764706, 0, 1,
5.858586, 2.040404, -130.5313, 1, 0.4823529, 0, 1,
5.89899, 2.040404, -130.39, 1, 0.4823529, 0, 1,
5.939394, 2.040404, -130.2682, 1, 0.4823529, 0, 1,
5.979798, 2.040404, -130.1661, 1, 0.4823529, 0, 1,
6.020202, 2.040404, -130.0836, 1, 0.4823529, 0, 1,
6.060606, 2.040404, -130.0207, 1, 0.4823529, 0, 1,
6.10101, 2.040404, -129.9774, 1, 0.4823529, 0, 1,
6.141414, 2.040404, -129.9536, 1, 0.4823529, 0, 1,
6.181818, 2.040404, -129.9496, 1, 0.4823529, 0, 1,
6.222222, 2.040404, -129.9651, 1, 0.4823529, 0, 1,
6.262626, 2.040404, -130.0002, 1, 0.4823529, 0, 1,
6.30303, 2.040404, -130.0549, 1, 0.4823529, 0, 1,
6.343434, 2.040404, -130.1292, 1, 0.4823529, 0, 1,
6.383838, 2.040404, -130.2232, 1, 0.4823529, 0, 1,
6.424242, 2.040404, -130.3367, 1, 0.4823529, 0, 1,
6.464646, 2.040404, -130.4698, 1, 0.4823529, 0, 1,
6.505051, 2.040404, -130.6226, 1, 0.3764706, 0, 1,
6.545455, 2.040404, -130.795, 1, 0.3764706, 0, 1,
6.585859, 2.040404, -130.9869, 1, 0.3764706, 0, 1,
6.626263, 2.040404, -131.1985, 1, 0.3764706, 0, 1,
6.666667, 2.040404, -131.4297, 1, 0.3764706, 0, 1,
6.707071, 2.040404, -131.6804, 1, 0.3764706, 0, 1,
6.747475, 2.040404, -131.9508, 1, 0.2745098, 0, 1,
6.787879, 2.040404, -132.2408, 1, 0.2745098, 0, 1,
6.828283, 2.040404, -132.5504, 1, 0.2745098, 0, 1,
6.868687, 2.040404, -132.8796, 1, 0.2745098, 0, 1,
6.909091, 2.040404, -133.2284, 1, 0.1686275, 0, 1,
6.949495, 2.040404, -133.5968, 1, 0.1686275, 0, 1,
6.989899, 2.040404, -133.9849, 1, 0.1686275, 0, 1,
7.030303, 2.040404, -134.3925, 1, 0.06666667, 0, 1,
7.070707, 2.040404, -134.8197, 1, 0.06666667, 0, 1,
7.111111, 2.040404, -135.2666, 1, 0.06666667, 0, 1,
7.151515, 2.040404, -135.733, 0.9647059, 0, 0.03137255, 1,
7.191919, 2.040404, -136.2191, 0.9647059, 0, 0.03137255, 1,
7.232323, 2.040404, -136.7247, 0.9647059, 0, 0.03137255, 1,
7.272727, 2.040404, -137.25, 0.8588235, 0, 0.1372549, 1,
7.313131, 2.040404, -137.7949, 0.8588235, 0, 0.1372549, 1,
7.353535, 2.040404, -138.3593, 0.7568628, 0, 0.2392157, 1,
7.393939, 2.040404, -138.9434, 0.7568628, 0, 0.2392157, 1,
7.434343, 2.040404, -139.5471, 0.654902, 0, 0.3411765, 1,
7.474748, 2.040404, -140.1704, 0.654902, 0, 0.3411765, 1,
7.515152, 2.040404, -140.8133, 0.5490196, 0, 0.4470588, 1,
7.555555, 2.040404, -141.4758, 0.5490196, 0, 0.4470588, 1,
7.59596, 2.040404, -142.1579, 0.4470588, 0, 0.5490196, 1,
7.636364, 2.040404, -142.8596, 0.4470588, 0, 0.5490196, 1,
7.676768, 2.040404, -143.581, 0.3411765, 0, 0.654902, 1,
7.717172, 2.040404, -144.3219, 0.2392157, 0, 0.7568628, 1,
7.757576, 2.040404, -145.0824, 0.2392157, 0, 0.7568628, 1,
7.79798, 2.040404, -145.8626, 0.1372549, 0, 0.8588235, 1,
7.838384, 2.040404, -146.6623, 0.1372549, 0, 0.8588235, 1,
7.878788, 2.040404, -147.4817, 0.03137255, 0, 0.9647059, 1,
7.919192, 2.040404, -148.3206, 0.04313726, 0.007843138, 0.9921569, 1,
7.959596, 2.040404, -149.1792, 0.1058824, 0.01960784, 0.9882353, 1,
8, 2.040404, -150.0574, 0.1058824, 0.01960784, 0.9882353, 1,
4, 2.080808, -156.2605, 0.4313726, 0.08627451, 0.9568627, 1,
4.040404, 2.080808, -155.2574, 0.3647059, 0.07058824, 0.9647059, 1,
4.080808, 2.080808, -154.2732, 0.3647059, 0.07058824, 0.9647059, 1,
4.121212, 2.080808, -153.3078, 0.3019608, 0.05882353, 0.9686275, 1,
4.161616, 2.080808, -152.3612, 0.2352941, 0.04705882, 0.9764706, 1,
4.20202, 2.080808, -151.4336, 0.172549, 0.03137255, 0.9803922, 1,
4.242424, 2.080808, -150.5247, 0.172549, 0.03137255, 0.9803922, 1,
4.282828, 2.080808, -149.6348, 0.1058824, 0.01960784, 0.9882353, 1,
4.323232, 2.080808, -148.7636, 0.04313726, 0.007843138, 0.9921569, 1,
4.363636, 2.080808, -147.9113, 0.03137255, 0, 0.9647059, 1,
4.40404, 2.080808, -147.0779, 0.03137255, 0, 0.9647059, 1,
4.444445, 2.080808, -146.2634, 0.1372549, 0, 0.8588235, 1,
4.484848, 2.080808, -145.4676, 0.1372549, 0, 0.8588235, 1,
4.525252, 2.080808, -144.6908, 0.2392157, 0, 0.7568628, 1,
4.565657, 2.080808, -143.9328, 0.3411765, 0, 0.654902, 1,
4.606061, 2.080808, -143.1936, 0.3411765, 0, 0.654902, 1,
4.646465, 2.080808, -142.4733, 0.4470588, 0, 0.5490196, 1,
4.686869, 2.080808, -141.7718, 0.4470588, 0, 0.5490196, 1,
4.727273, 2.080808, -141.0892, 0.5490196, 0, 0.4470588, 1,
4.767677, 2.080808, -140.4255, 0.654902, 0, 0.3411765, 1,
4.808081, 2.080808, -139.7805, 0.654902, 0, 0.3411765, 1,
4.848485, 2.080808, -139.1545, 0.7568628, 0, 0.2392157, 1,
4.888889, 2.080808, -138.5473, 0.7568628, 0, 0.2392157, 1,
4.929293, 2.080808, -137.959, 0.8588235, 0, 0.1372549, 1,
4.969697, 2.080808, -137.3895, 0.8588235, 0, 0.1372549, 1,
5.010101, 2.080808, -136.8388, 0.8588235, 0, 0.1372549, 1,
5.050505, 2.080808, -136.307, 0.9647059, 0, 0.03137255, 1,
5.090909, 2.080808, -135.7941, 0.9647059, 0, 0.03137255, 1,
5.131313, 2.080808, -135.3, 1, 0.06666667, 0, 1,
5.171717, 2.080808, -134.8248, 1, 0.06666667, 0, 1,
5.212121, 2.080808, -134.3684, 1, 0.06666667, 0, 1,
5.252525, 2.080808, -133.9308, 1, 0.1686275, 0, 1,
5.292929, 2.080808, -133.5122, 1, 0.1686275, 0, 1,
5.333333, 2.080808, -133.1123, 1, 0.1686275, 0, 1,
5.373737, 2.080808, -132.7314, 1, 0.2745098, 0, 1,
5.414141, 2.080808, -132.3692, 1, 0.2745098, 0, 1,
5.454545, 2.080808, -132.026, 1, 0.2745098, 0, 1,
5.494949, 2.080808, -131.7015, 1, 0.3764706, 0, 1,
5.535354, 2.080808, -131.396, 1, 0.3764706, 0, 1,
5.575758, 2.080808, -131.1093, 1, 0.3764706, 0, 1,
5.616162, 2.080808, -130.8414, 1, 0.3764706, 0, 1,
5.656566, 2.080808, -130.5924, 1, 0.3764706, 0, 1,
5.69697, 2.080808, -130.3622, 1, 0.4823529, 0, 1,
5.737374, 2.080808, -130.1509, 1, 0.4823529, 0, 1,
5.777778, 2.080808, -129.9584, 1, 0.4823529, 0, 1,
5.818182, 2.080808, -129.7848, 1, 0.4823529, 0, 1,
5.858586, 2.080808, -129.6301, 1, 0.4823529, 0, 1,
5.89899, 2.080808, -129.4942, 1, 0.4823529, 0, 1,
5.939394, 2.080808, -129.3771, 1, 0.4823529, 0, 1,
5.979798, 2.080808, -129.2789, 1, 0.5843138, 0, 1,
6.020202, 2.080808, -129.1996, 1, 0.5843138, 0, 1,
6.060606, 2.080808, -129.1391, 1, 0.5843138, 0, 1,
6.10101, 2.080808, -129.0974, 1, 0.5843138, 0, 1,
6.141414, 2.080808, -129.0747, 1, 0.5843138, 0, 1,
6.181818, 2.080808, -129.0707, 1, 0.5843138, 0, 1,
6.222222, 2.080808, -129.0856, 1, 0.5843138, 0, 1,
6.262626, 2.080808, -129.1194, 1, 0.5843138, 0, 1,
6.30303, 2.080808, -129.172, 1, 0.5843138, 0, 1,
6.343434, 2.080808, -129.2435, 1, 0.5843138, 0, 1,
6.383838, 2.080808, -129.3338, 1, 0.4823529, 0, 1,
6.424242, 2.080808, -129.443, 1, 0.4823529, 0, 1,
6.464646, 2.080808, -129.571, 1, 0.4823529, 0, 1,
6.505051, 2.080808, -129.7179, 1, 0.4823529, 0, 1,
6.545455, 2.080808, -129.8836, 1, 0.4823529, 0, 1,
6.585859, 2.080808, -130.0682, 1, 0.4823529, 0, 1,
6.626263, 2.080808, -130.2716, 1, 0.4823529, 0, 1,
6.666667, 2.080808, -130.4939, 1, 0.4823529, 0, 1,
6.707071, 2.080808, -130.735, 1, 0.3764706, 0, 1,
6.747475, 2.080808, -130.995, 1, 0.3764706, 0, 1,
6.787879, 2.080808, -131.2739, 1, 0.3764706, 0, 1,
6.828283, 2.080808, -131.5716, 1, 0.3764706, 0, 1,
6.868687, 2.080808, -131.8881, 1, 0.2745098, 0, 1,
6.909091, 2.080808, -132.2235, 1, 0.2745098, 0, 1,
6.949495, 2.080808, -132.5778, 1, 0.2745098, 0, 1,
6.989899, 2.080808, -132.9509, 1, 0.2745098, 0, 1,
7.030303, 2.080808, -133.3428, 1, 0.1686275, 0, 1,
7.070707, 2.080808, -133.7536, 1, 0.1686275, 0, 1,
7.111111, 2.080808, -134.1833, 1, 0.1686275, 0, 1,
7.151515, 2.080808, -134.6318, 1, 0.06666667, 0, 1,
7.191919, 2.080808, -135.0991, 1, 0.06666667, 0, 1,
7.232323, 2.080808, -135.5853, 0.9647059, 0, 0.03137255, 1,
7.272727, 2.080808, -136.0904, 0.9647059, 0, 0.03137255, 1,
7.313131, 2.080808, -136.6143, 0.9647059, 0, 0.03137255, 1,
7.353535, 2.080808, -137.1571, 0.8588235, 0, 0.1372549, 1,
7.393939, 2.080808, -137.7187, 0.8588235, 0, 0.1372549, 1,
7.434343, 2.080808, -138.2992, 0.7568628, 0, 0.2392157, 1,
7.474748, 2.080808, -138.8985, 0.7568628, 0, 0.2392157, 1,
7.515152, 2.080808, -139.5167, 0.654902, 0, 0.3411765, 1,
7.555555, 2.080808, -140.1537, 0.654902, 0, 0.3411765, 1,
7.59596, 2.080808, -140.8096, 0.5490196, 0, 0.4470588, 1,
7.636364, 2.080808, -141.4843, 0.5490196, 0, 0.4470588, 1,
7.676768, 2.080808, -142.1779, 0.4470588, 0, 0.5490196, 1,
7.717172, 2.080808, -142.8903, 0.4470588, 0, 0.5490196, 1,
7.757576, 2.080808, -143.6216, 0.3411765, 0, 0.654902, 1,
7.79798, 2.080808, -144.3717, 0.2392157, 0, 0.7568628, 1,
7.838384, 2.080808, -145.1407, 0.2392157, 0, 0.7568628, 1,
7.878788, 2.080808, -145.9286, 0.1372549, 0, 0.8588235, 1,
7.919192, 2.080808, -146.7353, 0.03137255, 0, 0.9647059, 1,
7.959596, 2.080808, -147.5608, 0.03137255, 0, 0.9647059, 1,
8, 2.080808, -148.4052, 0.04313726, 0.007843138, 0.9921569, 1,
4, 2.121212, -154.4421, 0.3647059, 0.07058824, 0.9647059, 1,
4.040404, 2.121212, -153.4769, 0.3019608, 0.05882353, 0.9686275, 1,
4.080808, 2.121212, -152.5298, 0.2352941, 0.04705882, 0.9764706, 1,
4.121212, 2.121212, -151.6008, 0.172549, 0.03137255, 0.9803922, 1,
4.161616, 2.121212, -150.69, 0.172549, 0.03137255, 0.9803922, 1,
4.20202, 2.121212, -149.7973, 0.1058824, 0.01960784, 0.9882353, 1,
4.242424, 2.121212, -148.9228, 0.04313726, 0.007843138, 0.9921569, 1,
4.282828, 2.121212, -148.0664, 0.04313726, 0.007843138, 0.9921569, 1,
4.323232, 2.121212, -147.2281, 0.03137255, 0, 0.9647059, 1,
4.363636, 2.121212, -146.408, 0.1372549, 0, 0.8588235, 1,
4.40404, 2.121212, -145.606, 0.1372549, 0, 0.8588235, 1,
4.444445, 2.121212, -144.8222, 0.2392157, 0, 0.7568628, 1,
4.484848, 2.121212, -144.0565, 0.3411765, 0, 0.654902, 1,
4.525252, 2.121212, -143.3089, 0.3411765, 0, 0.654902, 1,
4.565657, 2.121212, -142.5795, 0.4470588, 0, 0.5490196, 1,
4.606061, 2.121212, -141.8682, 0.4470588, 0, 0.5490196, 1,
4.646465, 2.121212, -141.1751, 0.5490196, 0, 0.4470588, 1,
4.686869, 2.121212, -140.5001, 0.5490196, 0, 0.4470588, 1,
4.727273, 2.121212, -139.8433, 0.654902, 0, 0.3411765, 1,
4.767677, 2.121212, -139.2046, 0.7568628, 0, 0.2392157, 1,
4.808081, 2.121212, -138.584, 0.7568628, 0, 0.2392157, 1,
4.848485, 2.121212, -137.9816, 0.8588235, 0, 0.1372549, 1,
4.888889, 2.121212, -137.3973, 0.8588235, 0, 0.1372549, 1,
4.929293, 2.121212, -136.8311, 0.8588235, 0, 0.1372549, 1,
4.969697, 2.121212, -136.2831, 0.9647059, 0, 0.03137255, 1,
5.010101, 2.121212, -135.7532, 0.9647059, 0, 0.03137255, 1,
5.050505, 2.121212, -135.2415, 1, 0.06666667, 0, 1,
5.090909, 2.121212, -134.7479, 1, 0.06666667, 0, 1,
5.131313, 2.121212, -134.2725, 1, 0.06666667, 0, 1,
5.171717, 2.121212, -133.8152, 1, 0.1686275, 0, 1,
5.212121, 2.121212, -133.376, 1, 0.1686275, 0, 1,
5.252525, 2.121212, -132.955, 1, 0.2745098, 0, 1,
5.292929, 2.121212, -132.5521, 1, 0.2745098, 0, 1,
5.333333, 2.121212, -132.1674, 1, 0.2745098, 0, 1,
5.373737, 2.121212, -131.8008, 1, 0.2745098, 0, 1,
5.414141, 2.121212, -131.4523, 1, 0.3764706, 0, 1,
5.454545, 2.121212, -131.122, 1, 0.3764706, 0, 1,
5.494949, 2.121212, -130.8098, 1, 0.3764706, 0, 1,
5.535354, 2.121212, -130.5158, 1, 0.4823529, 0, 1,
5.575758, 2.121212, -130.2399, 1, 0.4823529, 0, 1,
5.616162, 2.121212, -129.9821, 1, 0.4823529, 0, 1,
5.656566, 2.121212, -129.7425, 1, 0.4823529, 0, 1,
5.69697, 2.121212, -129.521, 1, 0.4823529, 0, 1,
5.737374, 2.121212, -129.3177, 1, 0.4823529, 0, 1,
5.777778, 2.121212, -129.1325, 1, 0.5843138, 0, 1,
5.818182, 2.121212, -128.9654, 1, 0.5843138, 0, 1,
5.858586, 2.121212, -128.8165, 1, 0.5843138, 0, 1,
5.89899, 2.121212, -128.6857, 1, 0.5843138, 0, 1,
5.939394, 2.121212, -128.5731, 1, 0.5843138, 0, 1,
5.979798, 2.121212, -128.4786, 1, 0.5843138, 0, 1,
6.020202, 2.121212, -128.4023, 1, 0.5843138, 0, 1,
6.060606, 2.121212, -128.3441, 1, 0.5843138, 0, 1,
6.10101, 2.121212, -128.304, 1, 0.5843138, 0, 1,
6.141414, 2.121212, -128.282, 1, 0.5843138, 0, 1,
6.181818, 2.121212, -128.2783, 1, 0.5843138, 0, 1,
6.222222, 2.121212, -128.2926, 1, 0.5843138, 0, 1,
6.262626, 2.121212, -128.3251, 1, 0.5843138, 0, 1,
6.30303, 2.121212, -128.3757, 1, 0.5843138, 0, 1,
6.343434, 2.121212, -128.4445, 1, 0.5843138, 0, 1,
6.383838, 2.121212, -128.5314, 1, 0.5843138, 0, 1,
6.424242, 2.121212, -128.6365, 1, 0.5843138, 0, 1,
6.464646, 2.121212, -128.7597, 1, 0.5843138, 0, 1,
6.505051, 2.121212, -128.901, 1, 0.5843138, 0, 1,
6.545455, 2.121212, -129.0605, 1, 0.5843138, 0, 1,
6.585859, 2.121212, -129.2381, 1, 0.5843138, 0, 1,
6.626263, 2.121212, -129.4339, 1, 0.4823529, 0, 1,
6.666667, 2.121212, -129.6478, 1, 0.4823529, 0, 1,
6.707071, 2.121212, -129.8798, 1, 0.4823529, 0, 1,
6.747475, 2.121212, -130.13, 1, 0.4823529, 0, 1,
6.787879, 2.121212, -130.3983, 1, 0.4823529, 0, 1,
6.828283, 2.121212, -130.6847, 1, 0.3764706, 0, 1,
6.868687, 2.121212, -130.9893, 1, 0.3764706, 0, 1,
6.909091, 2.121212, -131.3121, 1, 0.3764706, 0, 1,
6.949495, 2.121212, -131.653, 1, 0.3764706, 0, 1,
6.989899, 2.121212, -132.012, 1, 0.2745098, 0, 1,
7.030303, 2.121212, -132.3891, 1, 0.2745098, 0, 1,
7.070707, 2.121212, -132.7845, 1, 0.2745098, 0, 1,
7.111111, 2.121212, -133.1979, 1, 0.1686275, 0, 1,
7.151515, 2.121212, -133.6295, 1, 0.1686275, 0, 1,
7.191919, 2.121212, -134.0792, 1, 0.1686275, 0, 1,
7.232323, 2.121212, -134.5471, 1, 0.06666667, 0, 1,
7.272727, 2.121212, -135.0331, 1, 0.06666667, 0, 1,
7.313131, 2.121212, -135.5372, 0.9647059, 0, 0.03137255, 1,
7.353535, 2.121212, -136.0595, 0.9647059, 0, 0.03137255, 1,
7.393939, 2.121212, -136.5999, 0.9647059, 0, 0.03137255, 1,
7.434343, 2.121212, -137.1585, 0.8588235, 0, 0.1372549, 1,
7.474748, 2.121212, -137.7352, 0.8588235, 0, 0.1372549, 1,
7.515152, 2.121212, -138.3301, 0.7568628, 0, 0.2392157, 1,
7.555555, 2.121212, -138.9431, 0.7568628, 0, 0.2392157, 1,
7.59596, 2.121212, -139.5742, 0.654902, 0, 0.3411765, 1,
7.636364, 2.121212, -140.2234, 0.654902, 0, 0.3411765, 1,
7.676768, 2.121212, -140.8909, 0.5490196, 0, 0.4470588, 1,
7.717172, 2.121212, -141.5764, 0.5490196, 0, 0.4470588, 1,
7.757576, 2.121212, -142.2801, 0.4470588, 0, 0.5490196, 1,
7.79798, 2.121212, -143.002, 0.3411765, 0, 0.654902, 1,
7.838384, 2.121212, -143.7419, 0.3411765, 0, 0.654902, 1,
7.878788, 2.121212, -144.5, 0.2392157, 0, 0.7568628, 1,
7.919192, 2.121212, -145.2763, 0.2392157, 0, 0.7568628, 1,
7.959596, 2.121212, -146.0707, 0.1372549, 0, 0.8588235, 1,
8, 2.121212, -146.8832, 0.03137255, 0, 0.9647059, 1,
4, 2.161616, -152.76, 0.2352941, 0.04705882, 0.9764706, 1,
4.040404, 2.161616, -151.8305, 0.2352941, 0.04705882, 0.9764706, 1,
4.080808, 2.161616, -150.9185, 0.172549, 0.03137255, 0.9803922, 1,
4.121212, 2.161616, -150.0239, 0.1058824, 0.01960784, 0.9882353, 1,
4.161616, 2.161616, -149.1468, 0.04313726, 0.007843138, 0.9921569, 1,
4.20202, 2.161616, -148.2872, 0.04313726, 0.007843138, 0.9921569, 1,
4.242424, 2.161616, -147.445, 0.03137255, 0, 0.9647059, 1,
4.282828, 2.161616, -146.6204, 0.1372549, 0, 0.8588235, 1,
4.323232, 2.161616, -145.8131, 0.1372549, 0, 0.8588235, 1,
4.363636, 2.161616, -145.0234, 0.2392157, 0, 0.7568628, 1,
4.40404, 2.161616, -144.2511, 0.2392157, 0, 0.7568628, 1,
4.444445, 2.161616, -143.4963, 0.3411765, 0, 0.654902, 1,
4.484848, 2.161616, -142.759, 0.4470588, 0, 0.5490196, 1,
4.525252, 2.161616, -142.0391, 0.4470588, 0, 0.5490196, 1,
4.565657, 2.161616, -141.3367, 0.5490196, 0, 0.4470588, 1,
4.606061, 2.161616, -140.6518, 0.5490196, 0, 0.4470588, 1,
4.646465, 2.161616, -139.9843, 0.654902, 0, 0.3411765, 1,
4.686869, 2.161616, -139.3343, 0.654902, 0, 0.3411765, 1,
4.727273, 2.161616, -138.7018, 0.7568628, 0, 0.2392157, 1,
4.767677, 2.161616, -138.0867, 0.7568628, 0, 0.2392157, 1,
4.808081, 2.161616, -137.4892, 0.8588235, 0, 0.1372549, 1,
4.848485, 2.161616, -136.909, 0.8588235, 0, 0.1372549, 1,
4.888889, 2.161616, -136.3464, 0.9647059, 0, 0.03137255, 1,
4.929293, 2.161616, -135.8012, 0.9647059, 0, 0.03137255, 1,
4.969697, 2.161616, -135.2735, 1, 0.06666667, 0, 1,
5.010101, 2.161616, -134.7632, 1, 0.06666667, 0, 1,
5.050505, 2.161616, -134.2705, 1, 0.1686275, 0, 1,
5.090909, 2.161616, -133.7952, 1, 0.1686275, 0, 1,
5.131313, 2.161616, -133.3373, 1, 0.1686275, 0, 1,
5.171717, 2.161616, -132.897, 1, 0.2745098, 0, 1,
5.212121, 2.161616, -132.4741, 1, 0.2745098, 0, 1,
5.252525, 2.161616, -132.0686, 1, 0.2745098, 0, 1,
5.292929, 2.161616, -131.6807, 1, 0.3764706, 0, 1,
5.333333, 2.161616, -131.3102, 1, 0.3764706, 0, 1,
5.373737, 2.161616, -130.9571, 1, 0.3764706, 0, 1,
5.414141, 2.161616, -130.6216, 1, 0.3764706, 0, 1,
5.454545, 2.161616, -130.3035, 1, 0.4823529, 0, 1,
5.494949, 2.161616, -130.0029, 1, 0.4823529, 0, 1,
5.535354, 2.161616, -129.7197, 1, 0.4823529, 0, 1,
5.575758, 2.161616, -129.4541, 1, 0.4823529, 0, 1,
5.616162, 2.161616, -129.2058, 1, 0.5843138, 0, 1,
5.656566, 2.161616, -128.9751, 1, 0.5843138, 0, 1,
5.69697, 2.161616, -128.7618, 1, 0.5843138, 0, 1,
5.737374, 2.161616, -128.566, 1, 0.5843138, 0, 1,
5.777778, 2.161616, -128.3877, 1, 0.5843138, 0, 1,
5.818182, 2.161616, -128.2268, 1, 0.5843138, 0, 1,
5.858586, 2.161616, -128.0834, 1, 0.5843138, 0, 1,
5.89899, 2.161616, -127.9575, 1, 0.6862745, 0, 1,
5.939394, 2.161616, -127.849, 1, 0.6862745, 0, 1,
5.979798, 2.161616, -127.758, 1, 0.6862745, 0, 1,
6.020202, 2.161616, -127.6845, 1, 0.6862745, 0, 1,
6.060606, 2.161616, -127.6284, 1, 0.6862745, 0, 1,
6.10101, 2.161616, -127.5899, 1, 0.6862745, 0, 1,
6.141414, 2.161616, -127.5687, 1, 0.6862745, 0, 1,
6.181818, 2.161616, -127.5651, 1, 0.6862745, 0, 1,
6.222222, 2.161616, -127.5789, 1, 0.6862745, 0, 1,
6.262626, 2.161616, -127.6102, 1, 0.6862745, 0, 1,
6.30303, 2.161616, -127.659, 1, 0.6862745, 0, 1,
6.343434, 2.161616, -127.7252, 1, 0.6862745, 0, 1,
6.383838, 2.161616, -127.8089, 1, 0.6862745, 0, 1,
6.424242, 2.161616, -127.91, 1, 0.6862745, 0, 1,
6.464646, 2.161616, -128.0287, 1, 0.6862745, 0, 1,
6.505051, 2.161616, -128.1648, 1, 0.5843138, 0, 1,
6.545455, 2.161616, -128.3183, 1, 0.5843138, 0, 1,
6.585859, 2.161616, -128.4894, 1, 0.5843138, 0, 1,
6.626263, 2.161616, -128.6779, 1, 0.5843138, 0, 1,
6.666667, 2.161616, -128.8839, 1, 0.5843138, 0, 1,
6.707071, 2.161616, -129.1073, 1, 0.5843138, 0, 1,
6.747475, 2.161616, -129.3482, 1, 0.4823529, 0, 1,
6.787879, 2.161616, -129.6066, 1, 0.4823529, 0, 1,
6.828283, 2.161616, -129.8824, 1, 0.4823529, 0, 1,
6.868687, 2.161616, -130.1758, 1, 0.4823529, 0, 1,
6.909091, 2.161616, -130.4866, 1, 0.4823529, 0, 1,
6.949495, 2.161616, -130.8148, 1, 0.3764706, 0, 1,
6.989899, 2.161616, -131.1605, 1, 0.3764706, 0, 1,
7.030303, 2.161616, -131.5237, 1, 0.3764706, 0, 1,
7.070707, 2.161616, -131.9044, 1, 0.2745098, 0, 1,
7.111111, 2.161616, -132.3025, 1, 0.2745098, 0, 1,
7.151515, 2.161616, -132.7181, 1, 0.2745098, 0, 1,
7.191919, 2.161616, -133.1512, 1, 0.1686275, 0, 1,
7.232323, 2.161616, -133.6017, 1, 0.1686275, 0, 1,
7.272727, 2.161616, -134.0697, 1, 0.1686275, 0, 1,
7.313131, 2.161616, -134.5552, 1, 0.06666667, 0, 1,
7.353535, 2.161616, -135.0582, 1, 0.06666667, 0, 1,
7.393939, 2.161616, -135.5786, 0.9647059, 0, 0.03137255, 1,
7.434343, 2.161616, -136.1165, 0.9647059, 0, 0.03137255, 1,
7.474748, 2.161616, -136.6718, 0.9647059, 0, 0.03137255, 1,
7.515152, 2.161616, -137.2446, 0.8588235, 0, 0.1372549, 1,
7.555555, 2.161616, -137.8349, 0.8588235, 0, 0.1372549, 1,
7.59596, 2.161616, -138.4427, 0.7568628, 0, 0.2392157, 1,
7.636364, 2.161616, -139.0679, 0.7568628, 0, 0.2392157, 1,
7.676768, 2.161616, -139.7106, 0.654902, 0, 0.3411765, 1,
7.717172, 2.161616, -140.3708, 0.654902, 0, 0.3411765, 1,
7.757576, 2.161616, -141.0484, 0.5490196, 0, 0.4470588, 1,
7.79798, 2.161616, -141.7435, 0.4470588, 0, 0.5490196, 1,
7.838384, 2.161616, -142.4561, 0.4470588, 0, 0.5490196, 1,
7.878788, 2.161616, -143.1861, 0.3411765, 0, 0.654902, 1,
7.919192, 2.161616, -143.9336, 0.3411765, 0, 0.654902, 1,
7.959596, 2.161616, -144.6986, 0.2392157, 0, 0.7568628, 1,
8, 2.161616, -145.481, 0.1372549, 0, 0.8588235, 1,
4, 2.20202, -151.2036, 0.172549, 0.03137255, 0.9803922, 1,
4.040404, 2.20202, -150.3079, 0.1058824, 0.01960784, 0.9882353, 1,
4.080808, 2.20202, -149.429, 0.1058824, 0.01960784, 0.9882353, 1,
4.121212, 2.20202, -148.567, 0.04313726, 0.007843138, 0.9921569, 1,
4.161616, 2.20202, -147.7218, 0.03137255, 0, 0.9647059, 1,
4.20202, 2.20202, -146.8934, 0.03137255, 0, 0.9647059, 1,
4.242424, 2.20202, -146.0819, 0.1372549, 0, 0.8588235, 1,
4.282828, 2.20202, -145.2872, 0.2392157, 0, 0.7568628, 1,
4.323232, 2.20202, -144.5093, 0.2392157, 0, 0.7568628, 1,
4.363636, 2.20202, -143.7483, 0.3411765, 0, 0.654902, 1,
4.40404, 2.20202, -143.0041, 0.3411765, 0, 0.654902, 1,
4.444445, 2.20202, -142.2768, 0.4470588, 0, 0.5490196, 1,
4.484848, 2.20202, -141.5662, 0.5490196, 0, 0.4470588, 1,
4.525252, 2.20202, -140.8725, 0.5490196, 0, 0.4470588, 1,
4.565657, 2.20202, -140.1957, 0.654902, 0, 0.3411765, 1,
4.606061, 2.20202, -139.5356, 0.654902, 0, 0.3411765, 1,
4.646465, 2.20202, -138.8925, 0.7568628, 0, 0.2392157, 1,
4.686869, 2.20202, -138.2661, 0.7568628, 0, 0.2392157, 1,
4.727273, 2.20202, -137.6566, 0.8588235, 0, 0.1372549, 1,
4.767677, 2.20202, -137.0639, 0.8588235, 0, 0.1372549, 1,
4.808081, 2.20202, -136.488, 0.9647059, 0, 0.03137255, 1,
4.848485, 2.20202, -135.929, 0.9647059, 0, 0.03137255, 1,
4.888889, 2.20202, -135.3868, 1, 0.06666667, 0, 1,
4.929293, 2.20202, -134.8614, 1, 0.06666667, 0, 1,
4.969697, 2.20202, -134.3529, 1, 0.06666667, 0, 1,
5.010101, 2.20202, -133.8612, 1, 0.1686275, 0, 1,
5.050505, 2.20202, -133.3864, 1, 0.1686275, 0, 1,
5.090909, 2.20202, -132.9283, 1, 0.2745098, 0, 1,
5.131313, 2.20202, -132.4872, 1, 0.2745098, 0, 1,
5.171717, 2.20202, -132.0628, 1, 0.2745098, 0, 1,
5.212121, 2.20202, -131.6553, 1, 0.3764706, 0, 1,
5.252525, 2.20202, -131.2646, 1, 0.3764706, 0, 1,
5.292929, 2.20202, -130.8907, 1, 0.3764706, 0, 1,
5.333333, 2.20202, -130.5337, 1, 0.4823529, 0, 1,
5.373737, 2.20202, -130.1935, 1, 0.4823529, 0, 1,
5.414141, 2.20202, -129.8702, 1, 0.4823529, 0, 1,
5.454545, 2.20202, -129.5636, 1, 0.4823529, 0, 1,
5.494949, 2.20202, -129.274, 1, 0.5843138, 0, 1,
5.535354, 2.20202, -129.0011, 1, 0.5843138, 0, 1,
5.575758, 2.20202, -128.7451, 1, 0.5843138, 0, 1,
5.616162, 2.20202, -128.5059, 1, 0.5843138, 0, 1,
5.656566, 2.20202, -128.2835, 1, 0.5843138, 0, 1,
5.69697, 2.20202, -128.078, 1, 0.5843138, 0, 1,
5.737374, 2.20202, -127.8893, 1, 0.6862745, 0, 1,
5.777778, 2.20202, -127.7175, 1, 0.6862745, 0, 1,
5.818182, 2.20202, -127.5625, 1, 0.6862745, 0, 1,
5.858586, 2.20202, -127.4243, 1, 0.6862745, 0, 1,
5.89899, 2.20202, -127.3029, 1, 0.6862745, 0, 1,
5.939394, 2.20202, -127.1984, 1, 0.6862745, 0, 1,
5.979798, 2.20202, -127.1107, 1, 0.6862745, 0, 1,
6.020202, 2.20202, -127.0399, 1, 0.6862745, 0, 1,
6.060606, 2.20202, -126.9858, 1, 0.6862745, 0, 1,
6.10101, 2.20202, -126.9487, 1, 0.6862745, 0, 1,
6.141414, 2.20202, -126.9283, 1, 0.6862745, 0, 1,
6.181818, 2.20202, -126.9248, 1, 0.6862745, 0, 1,
6.222222, 2.20202, -126.9381, 1, 0.6862745, 0, 1,
6.262626, 2.20202, -126.9683, 1, 0.6862745, 0, 1,
6.30303, 2.20202, -127.0152, 1, 0.6862745, 0, 1,
6.343434, 2.20202, -127.0791, 1, 0.6862745, 0, 1,
6.383838, 2.20202, -127.1597, 1, 0.6862745, 0, 1,
6.424242, 2.20202, -127.2572, 1, 0.6862745, 0, 1,
6.464646, 2.20202, -127.3715, 1, 0.6862745, 0, 1,
6.505051, 2.20202, -127.5027, 1, 0.6862745, 0, 1,
6.545455, 2.20202, -127.6507, 1, 0.6862745, 0, 1,
6.585859, 2.20202, -127.8155, 1, 0.6862745, 0, 1,
6.626263, 2.20202, -127.9971, 1, 0.6862745, 0, 1,
6.666667, 2.20202, -128.1956, 1, 0.5843138, 0, 1,
6.707071, 2.20202, -128.4109, 1, 0.5843138, 0, 1,
6.747475, 2.20202, -128.6431, 1, 0.5843138, 0, 1,
6.787879, 2.20202, -128.8921, 1, 0.5843138, 0, 1,
6.828283, 2.20202, -129.1579, 1, 0.5843138, 0, 1,
6.868687, 2.20202, -129.4406, 1, 0.4823529, 0, 1,
6.909091, 2.20202, -129.74, 1, 0.4823529, 0, 1,
6.949495, 2.20202, -130.0564, 1, 0.4823529, 0, 1,
6.989899, 2.20202, -130.3895, 1, 0.4823529, 0, 1,
7.030303, 2.20202, -130.7395, 1, 0.3764706, 0, 1,
7.070707, 2.20202, -131.1063, 1, 0.3764706, 0, 1,
7.111111, 2.20202, -131.49, 1, 0.3764706, 0, 1,
7.151515, 2.20202, -131.8905, 1, 0.2745098, 0, 1,
7.191919, 2.20202, -132.3078, 1, 0.2745098, 0, 1,
7.232323, 2.20202, -132.742, 1, 0.2745098, 0, 1,
7.272727, 2.20202, -133.1929, 1, 0.1686275, 0, 1,
7.313131, 2.20202, -133.6608, 1, 0.1686275, 0, 1,
7.353535, 2.20202, -134.1454, 1, 0.1686275, 0, 1,
7.393939, 2.20202, -134.6469, 1, 0.06666667, 0, 1,
7.434343, 2.20202, -135.1652, 1, 0.06666667, 0, 1,
7.474748, 2.20202, -135.7004, 0.9647059, 0, 0.03137255, 1,
7.515152, 2.20202, -136.2524, 0.9647059, 0, 0.03137255, 1,
7.555555, 2.20202, -136.8212, 0.8588235, 0, 0.1372549, 1,
7.59596, 2.20202, -137.4069, 0.8588235, 0, 0.1372549, 1,
7.636364, 2.20202, -138.0094, 0.7568628, 0, 0.2392157, 1,
7.676768, 2.20202, -138.6287, 0.7568628, 0, 0.2392157, 1,
7.717172, 2.20202, -139.2649, 0.654902, 0, 0.3411765, 1,
7.757576, 2.20202, -139.9178, 0.654902, 0, 0.3411765, 1,
7.79798, 2.20202, -140.5877, 0.5490196, 0, 0.4470588, 1,
7.838384, 2.20202, -141.2743, 0.5490196, 0, 0.4470588, 1,
7.878788, 2.20202, -141.9778, 0.4470588, 0, 0.5490196, 1,
7.919192, 2.20202, -142.6982, 0.4470588, 0, 0.5490196, 1,
7.959596, 2.20202, -143.4353, 0.3411765, 0, 0.654902, 1,
8, 2.20202, -144.1893, 0.3411765, 0, 0.654902, 1,
4, 2.242424, -149.7633, 0.1058824, 0.01960784, 0.9882353, 1,
4.040404, 2.242424, -148.8996, 0.04313726, 0.007843138, 0.9921569, 1,
4.080808, 2.242424, -148.0521, 0.04313726, 0.007843138, 0.9921569, 1,
4.121212, 2.242424, -147.2209, 0.03137255, 0, 0.9647059, 1,
4.161616, 2.242424, -146.4059, 0.1372549, 0, 0.8588235, 1,
4.20202, 2.242424, -145.6071, 0.1372549, 0, 0.8588235, 1,
4.242424, 2.242424, -144.8245, 0.2392157, 0, 0.7568628, 1,
4.282828, 2.242424, -144.0582, 0.3411765, 0, 0.654902, 1,
4.323232, 2.242424, -143.3081, 0.3411765, 0, 0.654902, 1,
4.363636, 2.242424, -142.5743, 0.4470588, 0, 0.5490196, 1,
4.40404, 2.242424, -141.8567, 0.4470588, 0, 0.5490196, 1,
4.444445, 2.242424, -141.1553, 0.5490196, 0, 0.4470588, 1,
4.484848, 2.242424, -140.4701, 0.654902, 0, 0.3411765, 1,
4.525252, 2.242424, -139.8012, 0.654902, 0, 0.3411765, 1,
4.565657, 2.242424, -139.1485, 0.7568628, 0, 0.2392157, 1,
4.606061, 2.242424, -138.5121, 0.7568628, 0, 0.2392157, 1,
4.646465, 2.242424, -137.8918, 0.8588235, 0, 0.1372549, 1,
4.686869, 2.242424, -137.2878, 0.8588235, 0, 0.1372549, 1,
4.727273, 2.242424, -136.7001, 0.9647059, 0, 0.03137255, 1,
4.767677, 2.242424, -136.1286, 0.9647059, 0, 0.03137255, 1,
4.808081, 2.242424, -135.5733, 0.9647059, 0, 0.03137255, 1,
4.848485, 2.242424, -135.0342, 1, 0.06666667, 0, 1,
4.888889, 2.242424, -134.5114, 1, 0.06666667, 0, 1,
4.929293, 2.242424, -134.0048, 1, 0.1686275, 0, 1,
4.969697, 2.242424, -133.5144, 1, 0.1686275, 0, 1,
5.010101, 2.242424, -133.0403, 1, 0.1686275, 0, 1,
5.050505, 2.242424, -132.5824, 1, 0.2745098, 0, 1,
5.090909, 2.242424, -132.1407, 1, 0.2745098, 0, 1,
5.131313, 2.242424, -131.7153, 1, 0.3764706, 0, 1,
5.171717, 2.242424, -131.3061, 1, 0.3764706, 0, 1,
5.212121, 2.242424, -130.9131, 1, 0.3764706, 0, 1,
5.252525, 2.242424, -130.5364, 1, 0.4823529, 0, 1,
5.292929, 2.242424, -130.1759, 1, 0.4823529, 0, 1,
5.333333, 2.242424, -129.8316, 1, 0.4823529, 0, 1,
5.373737, 2.242424, -129.5036, 1, 0.4823529, 0, 1,
5.414141, 2.242424, -129.1917, 1, 0.5843138, 0, 1,
5.454545, 2.242424, -128.8962, 1, 0.5843138, 0, 1,
5.494949, 2.242424, -128.6168, 1, 0.5843138, 0, 1,
5.535354, 2.242424, -128.3537, 1, 0.5843138, 0, 1,
5.575758, 2.242424, -128.1068, 1, 0.5843138, 0, 1,
5.616162, 2.242424, -127.8762, 1, 0.6862745, 0, 1,
5.656566, 2.242424, -127.6618, 1, 0.6862745, 0, 1,
5.69697, 2.242424, -127.4636, 1, 0.6862745, 0, 1,
5.737374, 2.242424, -127.2817, 1, 0.6862745, 0, 1,
5.777778, 2.242424, -127.1159, 1, 0.6862745, 0, 1,
5.818182, 2.242424, -126.9665, 1, 0.6862745, 0, 1,
5.858586, 2.242424, -126.8332, 1, 0.6862745, 0, 1,
5.89899, 2.242424, -126.7162, 1, 0.7921569, 0, 1,
5.939394, 2.242424, -126.6154, 1, 0.7921569, 0, 1,
5.979798, 2.242424, -126.5308, 1, 0.7921569, 0, 1,
6.020202, 2.242424, -126.4625, 1, 0.7921569, 0, 1,
6.060606, 2.242424, -126.4104, 1, 0.7921569, 0, 1,
6.10101, 2.242424, -126.3746, 1, 0.7921569, 0, 1,
6.141414, 2.242424, -126.355, 1, 0.7921569, 0, 1,
6.181818, 2.242424, -126.3516, 1, 0.7921569, 0, 1,
6.222222, 2.242424, -126.3644, 1, 0.7921569, 0, 1,
6.262626, 2.242424, -126.3935, 1, 0.7921569, 0, 1,
6.30303, 2.242424, -126.4388, 1, 0.7921569, 0, 1,
6.343434, 2.242424, -126.5003, 1, 0.7921569, 0, 1,
6.383838, 2.242424, -126.5781, 1, 0.7921569, 0, 1,
6.424242, 2.242424, -126.6721, 1, 0.7921569, 0, 1,
6.464646, 2.242424, -126.7823, 1, 0.7921569, 0, 1,
6.505051, 2.242424, -126.9088, 1, 0.6862745, 0, 1,
6.545455, 2.242424, -127.0515, 1, 0.6862745, 0, 1,
6.585859, 2.242424, -127.2104, 1, 0.6862745, 0, 1,
6.626263, 2.242424, -127.3856, 1, 0.6862745, 0, 1,
6.666667, 2.242424, -127.577, 1, 0.6862745, 0, 1,
6.707071, 2.242424, -127.7846, 1, 0.6862745, 0, 1,
6.747475, 2.242424, -128.0085, 1, 0.6862745, 0, 1,
6.787879, 2.242424, -128.2486, 1, 0.5843138, 0, 1,
6.828283, 2.242424, -128.5049, 1, 0.5843138, 0, 1,
6.868687, 2.242424, -128.7775, 1, 0.5843138, 0, 1,
6.909091, 2.242424, -129.0663, 1, 0.5843138, 0, 1,
6.949495, 2.242424, -129.3713, 1, 0.4823529, 0, 1,
6.989899, 2.242424, -129.6926, 1, 0.4823529, 0, 1,
7.030303, 2.242424, -130.03, 1, 0.4823529, 0, 1,
7.070707, 2.242424, -130.3838, 1, 0.4823529, 0, 1,
7.111111, 2.242424, -130.7537, 1, 0.3764706, 0, 1,
7.151515, 2.242424, -131.1399, 1, 0.3764706, 0, 1,
7.191919, 2.242424, -131.5423, 1, 0.3764706, 0, 1,
7.232323, 2.242424, -131.961, 1, 0.2745098, 0, 1,
7.272727, 2.242424, -132.3959, 1, 0.2745098, 0, 1,
7.313131, 2.242424, -132.847, 1, 0.2745098, 0, 1,
7.353535, 2.242424, -133.3143, 1, 0.1686275, 0, 1,
7.393939, 2.242424, -133.7979, 1, 0.1686275, 0, 1,
7.434343, 2.242424, -134.2977, 1, 0.06666667, 0, 1,
7.474748, 2.242424, -134.8138, 1, 0.06666667, 0, 1,
7.515152, 2.242424, -135.3461, 1, 0.06666667, 0, 1,
7.555555, 2.242424, -135.8946, 0.9647059, 0, 0.03137255, 1,
7.59596, 2.242424, -136.4593, 0.9647059, 0, 0.03137255, 1,
7.636364, 2.242424, -137.0403, 0.8588235, 0, 0.1372549, 1,
7.676768, 2.242424, -137.6375, 0.8588235, 0, 0.1372549, 1,
7.717172, 2.242424, -138.2509, 0.7568628, 0, 0.2392157, 1,
7.757576, 2.242424, -138.8806, 0.7568628, 0, 0.2392157, 1,
7.79798, 2.242424, -139.5265, 0.654902, 0, 0.3411765, 1,
7.838384, 2.242424, -140.1887, 0.654902, 0, 0.3411765, 1,
7.878788, 2.242424, -140.867, 0.5490196, 0, 0.4470588, 1,
7.919192, 2.242424, -141.5616, 0.5490196, 0, 0.4470588, 1,
7.959596, 2.242424, -142.2725, 0.4470588, 0, 0.5490196, 1,
8, 2.242424, -142.9995, 0.3411765, 0, 0.654902, 1,
4, 2.282828, -148.4305, 0.04313726, 0.007843138, 0.9921569, 1,
4.040404, 2.282828, -147.5971, 0.03137255, 0, 0.9647059, 1,
4.080808, 2.282828, -146.7793, 0.03137255, 0, 0.9647059, 1,
4.121212, 2.282828, -145.9772, 0.1372549, 0, 0.8588235, 1,
4.161616, 2.282828, -145.1908, 0.2392157, 0, 0.7568628, 1,
4.20202, 2.282828, -144.4201, 0.2392157, 0, 0.7568628, 1,
4.242424, 2.282828, -143.665, 0.3411765, 0, 0.654902, 1,
4.282828, 2.282828, -142.9255, 0.4470588, 0, 0.5490196, 1,
4.323232, 2.282828, -142.2018, 0.4470588, 0, 0.5490196, 1,
4.363636, 2.282828, -141.4937, 0.5490196, 0, 0.4470588, 1,
4.40404, 2.282828, -140.8012, 0.5490196, 0, 0.4470588, 1,
4.444445, 2.282828, -140.1244, 0.654902, 0, 0.3411765, 1,
4.484848, 2.282828, -139.4633, 0.654902, 0, 0.3411765, 1,
4.525252, 2.282828, -138.8179, 0.7568628, 0, 0.2392157, 1,
4.565657, 2.282828, -138.1881, 0.7568628, 0, 0.2392157, 1,
4.606061, 2.282828, -137.574, 0.8588235, 0, 0.1372549, 1,
4.646465, 2.282828, -136.9755, 0.8588235, 0, 0.1372549, 1,
4.686869, 2.282828, -136.3927, 0.9647059, 0, 0.03137255, 1,
4.727273, 2.282828, -135.8256, 0.9647059, 0, 0.03137255, 1,
4.767677, 2.282828, -135.2741, 1, 0.06666667, 0, 1,
4.808081, 2.282828, -134.7383, 1, 0.06666667, 0, 1,
4.848485, 2.282828, -134.2181, 1, 0.1686275, 0, 1,
4.888889, 2.282828, -133.7136, 1, 0.1686275, 0, 1,
4.929293, 2.282828, -133.2248, 1, 0.1686275, 0, 1,
4.969697, 2.282828, -132.7516, 1, 0.2745098, 0, 1,
5.010101, 2.282828, -132.2942, 1, 0.2745098, 0, 1,
5.050505, 2.282828, -131.8523, 1, 0.2745098, 0, 1,
5.090909, 2.282828, -131.4261, 1, 0.3764706, 0, 1,
5.131313, 2.282828, -131.0156, 1, 0.3764706, 0, 1,
5.171717, 2.282828, -130.6208, 1, 0.3764706, 0, 1,
5.212121, 2.282828, -130.2416, 1, 0.4823529, 0, 1,
5.252525, 2.282828, -129.8781, 1, 0.4823529, 0, 1,
5.292929, 2.282828, -129.5302, 1, 0.4823529, 0, 1,
5.333333, 2.282828, -129.198, 1, 0.5843138, 0, 1,
5.373737, 2.282828, -128.8815, 1, 0.5843138, 0, 1,
5.414141, 2.282828, -128.5806, 1, 0.5843138, 0, 1,
5.454545, 2.282828, -128.2954, 1, 0.5843138, 0, 1,
5.494949, 2.282828, -128.0259, 1, 0.6862745, 0, 1,
5.535354, 2.282828, -127.772, 1, 0.6862745, 0, 1,
5.575758, 2.282828, -127.5338, 1, 0.6862745, 0, 1,
5.616162, 2.282828, -127.3112, 1, 0.6862745, 0, 1,
5.656566, 2.282828, -127.1044, 1, 0.6862745, 0, 1,
5.69697, 2.282828, -126.9131, 1, 0.6862745, 0, 1,
5.737374, 2.282828, -126.7376, 1, 0.7921569, 0, 1,
5.777778, 2.282828, -126.5777, 1, 0.7921569, 0, 1,
5.818182, 2.282828, -126.4334, 1, 0.7921569, 0, 1,
5.858586, 2.282828, -126.3048, 1, 0.7921569, 0, 1,
5.89899, 2.282828, -126.1919, 1, 0.7921569, 0, 1,
5.939394, 2.282828, -126.0947, 1, 0.7921569, 0, 1,
5.979798, 2.282828, -126.0131, 1, 0.7921569, 0, 1,
6.020202, 2.282828, -125.9472, 1, 0.7921569, 0, 1,
6.060606, 2.282828, -125.8969, 1, 0.7921569, 0, 1,
6.10101, 2.282828, -125.8623, 1, 0.7921569, 0, 1,
6.141414, 2.282828, -125.8434, 1, 0.7921569, 0, 1,
6.181818, 2.282828, -125.8401, 1, 0.7921569, 0, 1,
6.222222, 2.282828, -125.8525, 1, 0.7921569, 0, 1,
6.262626, 2.282828, -125.8806, 1, 0.7921569, 0, 1,
6.30303, 2.282828, -125.9243, 1, 0.7921569, 0, 1,
6.343434, 2.282828, -125.9837, 1, 0.7921569, 0, 1,
6.383838, 2.282828, -126.0587, 1, 0.7921569, 0, 1,
6.424242, 2.282828, -126.1494, 1, 0.7921569, 0, 1,
6.464646, 2.282828, -126.2558, 1, 0.7921569, 0, 1,
6.505051, 2.282828, -126.3778, 1, 0.7921569, 0, 1,
6.545455, 2.282828, -126.5155, 1, 0.7921569, 0, 1,
6.585859, 2.282828, -126.6688, 1, 0.7921569, 0, 1,
6.626263, 2.282828, -126.8379, 1, 0.6862745, 0, 1,
6.666667, 2.282828, -127.0226, 1, 0.6862745, 0, 1,
6.707071, 2.282828, -127.2229, 1, 0.6862745, 0, 1,
6.747475, 2.282828, -127.4389, 1, 0.6862745, 0, 1,
6.787879, 2.282828, -127.6706, 1, 0.6862745, 0, 1,
6.828283, 2.282828, -127.9179, 1, 0.6862745, 0, 1,
6.868687, 2.282828, -128.1809, 1, 0.5843138, 0, 1,
6.909091, 2.282828, -128.4596, 1, 0.5843138, 0, 1,
6.949495, 2.282828, -128.7539, 1, 0.5843138, 0, 1,
6.989899, 2.282828, -129.0639, 1, 0.5843138, 0, 1,
7.030303, 2.282828, -129.3895, 1, 0.4823529, 0, 1,
7.070707, 2.282828, -129.7308, 1, 0.4823529, 0, 1,
7.111111, 2.282828, -130.0878, 1, 0.4823529, 0, 1,
7.151515, 2.282828, -130.4604, 1, 0.4823529, 0, 1,
7.191919, 2.282828, -130.8488, 1, 0.3764706, 0, 1,
7.232323, 2.282828, -131.2527, 1, 0.3764706, 0, 1,
7.272727, 2.282828, -131.6723, 1, 0.3764706, 0, 1,
7.313131, 2.282828, -132.1076, 1, 0.2745098, 0, 1,
7.353535, 2.282828, -132.5586, 1, 0.2745098, 0, 1,
7.393939, 2.282828, -133.0252, 1, 0.2745098, 0, 1,
7.434343, 2.282828, -133.5075, 1, 0.1686275, 0, 1,
7.474748, 2.282828, -134.0054, 1, 0.1686275, 0, 1,
7.515152, 2.282828, -134.519, 1, 0.06666667, 0, 1,
7.555555, 2.282828, -135.0483, 1, 0.06666667, 0, 1,
7.59596, 2.282828, -135.5932, 0.9647059, 0, 0.03137255, 1,
7.636364, 2.282828, -136.1538, 0.9647059, 0, 0.03137255, 1,
7.676768, 2.282828, -136.7301, 0.9647059, 0, 0.03137255, 1,
7.717172, 2.282828, -137.322, 0.8588235, 0, 0.1372549, 1,
7.757576, 2.282828, -137.9296, 0.8588235, 0, 0.1372549, 1,
7.79798, 2.282828, -138.5528, 0.7568628, 0, 0.2392157, 1,
7.838384, 2.282828, -139.1917, 0.7568628, 0, 0.2392157, 1,
7.878788, 2.282828, -139.8463, 0.654902, 0, 0.3411765, 1,
7.919192, 2.282828, -140.5165, 0.5490196, 0, 0.4470588, 1,
7.959596, 2.282828, -141.2024, 0.5490196, 0, 0.4470588, 1,
8, 2.282828, -141.904, 0.4470588, 0, 0.5490196, 1,
4, 2.323232, -147.1971, 0.03137255, 0, 0.9647059, 1,
4.040404, 2.323232, -146.3924, 0.1372549, 0, 0.8588235, 1,
4.080808, 2.323232, -145.6028, 0.1372549, 0, 0.8588235, 1,
4.121212, 2.323232, -144.8284, 0.2392157, 0, 0.7568628, 1,
4.161616, 2.323232, -144.0691, 0.3411765, 0, 0.654902, 1,
4.20202, 2.323232, -143.3249, 0.3411765, 0, 0.654902, 1,
4.242424, 2.323232, -142.5959, 0.4470588, 0, 0.5490196, 1,
4.282828, 2.323232, -141.8819, 0.4470588, 0, 0.5490196, 1,
4.323232, 2.323232, -141.1831, 0.5490196, 0, 0.4470588, 1,
4.363636, 2.323232, -140.4994, 0.5490196, 0, 0.4470588, 1,
4.40404, 2.323232, -139.8309, 0.654902, 0, 0.3411765, 1,
4.444445, 2.323232, -139.1774, 0.7568628, 0, 0.2392157, 1,
4.484848, 2.323232, -138.5391, 0.7568628, 0, 0.2392157, 1,
4.525252, 2.323232, -137.9159, 0.8588235, 0, 0.1372549, 1,
4.565657, 2.323232, -137.3078, 0.8588235, 0, 0.1372549, 1,
4.606061, 2.323232, -136.7149, 0.9647059, 0, 0.03137255, 1,
4.646465, 2.323232, -136.1371, 0.9647059, 0, 0.03137255, 1,
4.686869, 2.323232, -135.5743, 0.9647059, 0, 0.03137255, 1,
4.727273, 2.323232, -135.0268, 1, 0.06666667, 0, 1,
4.767677, 2.323232, -134.4943, 1, 0.06666667, 0, 1,
4.808081, 2.323232, -133.977, 1, 0.1686275, 0, 1,
4.848485, 2.323232, -133.4747, 1, 0.1686275, 0, 1,
4.888889, 2.323232, -132.9877, 1, 0.2745098, 0, 1,
4.929293, 2.323232, -132.5157, 1, 0.2745098, 0, 1,
4.969697, 2.323232, -132.0589, 1, 0.2745098, 0, 1,
5.010101, 2.323232, -131.6171, 1, 0.3764706, 0, 1,
5.050505, 2.323232, -131.1905, 1, 0.3764706, 0, 1,
5.090909, 2.323232, -130.7791, 1, 0.3764706, 0, 1,
5.131313, 2.323232, -130.3827, 1, 0.4823529, 0, 1,
5.171717, 2.323232, -130.0015, 1, 0.4823529, 0, 1,
5.212121, 2.323232, -129.6354, 1, 0.4823529, 0, 1,
5.252525, 2.323232, -129.2844, 1, 0.5843138, 0, 1,
5.292929, 2.323232, -128.9485, 1, 0.5843138, 0, 1,
5.333333, 2.323232, -128.6278, 1, 0.5843138, 0, 1,
5.373737, 2.323232, -128.3221, 1, 0.5843138, 0, 1,
5.414141, 2.323232, -128.0317, 1, 0.6862745, 0, 1,
5.454545, 2.323232, -127.7563, 1, 0.6862745, 0, 1,
5.494949, 2.323232, -127.496, 1, 0.6862745, 0, 1,
5.535354, 2.323232, -127.2509, 1, 0.6862745, 0, 1,
5.575758, 2.323232, -127.0209, 1, 0.6862745, 0, 1,
5.616162, 2.323232, -126.806, 1, 0.7921569, 0, 1,
5.656566, 2.323232, -126.6063, 1, 0.7921569, 0, 1,
5.69697, 2.323232, -126.4216, 1, 0.7921569, 0, 1,
5.737374, 2.323232, -126.2521, 1, 0.7921569, 0, 1,
5.777778, 2.323232, -126.0977, 1, 0.7921569, 0, 1,
5.818182, 2.323232, -125.9585, 1, 0.7921569, 0, 1,
5.858586, 2.323232, -125.8343, 1, 0.7921569, 0, 1,
5.89899, 2.323232, -125.7253, 1, 0.7921569, 0, 1,
5.939394, 2.323232, -125.6314, 1, 0.7921569, 0, 1,
5.979798, 2.323232, -125.5526, 1, 0.8941177, 0, 1,
6.020202, 2.323232, -125.489, 1, 0.8941177, 0, 1,
6.060606, 2.323232, -125.4405, 1, 0.8941177, 0, 1,
6.10101, 2.323232, -125.4071, 1, 0.8941177, 0, 1,
6.141414, 2.323232, -125.3888, 1, 0.8941177, 0, 1,
6.181818, 2.323232, -125.3856, 1, 0.8941177, 0, 1,
6.222222, 2.323232, -125.3976, 1, 0.8941177, 0, 1,
6.262626, 2.323232, -125.4247, 1, 0.8941177, 0, 1,
6.30303, 2.323232, -125.4669, 1, 0.8941177, 0, 1,
6.343434, 2.323232, -125.5242, 1, 0.8941177, 0, 1,
6.383838, 2.323232, -125.5967, 1, 0.7921569, 0, 1,
6.424242, 2.323232, -125.6842, 1, 0.7921569, 0, 1,
6.464646, 2.323232, -125.7869, 1, 0.7921569, 0, 1,
6.505051, 2.323232, -125.9048, 1, 0.7921569, 0, 1,
6.545455, 2.323232, -126.0377, 1, 0.7921569, 0, 1,
6.585859, 2.323232, -126.1858, 1, 0.7921569, 0, 1,
6.626263, 2.323232, -126.349, 1, 0.7921569, 0, 1,
6.666667, 2.323232, -126.5273, 1, 0.7921569, 0, 1,
6.707071, 2.323232, -126.7207, 1, 0.7921569, 0, 1,
6.747475, 2.323232, -126.9293, 1, 0.6862745, 0, 1,
6.787879, 2.323232, -127.153, 1, 0.6862745, 0, 1,
6.828283, 2.323232, -127.3918, 1, 0.6862745, 0, 1,
6.868687, 2.323232, -127.6457, 1, 0.6862745, 0, 1,
6.909091, 2.323232, -127.9148, 1, 0.6862745, 0, 1,
6.949495, 2.323232, -128.1989, 1, 0.5843138, 0, 1,
6.989899, 2.323232, -128.4982, 1, 0.5843138, 0, 1,
7.030303, 2.323232, -128.8127, 1, 0.5843138, 0, 1,
7.070707, 2.323232, -129.1422, 1, 0.5843138, 0, 1,
7.111111, 2.323232, -129.4869, 1, 0.4823529, 0, 1,
7.151515, 2.323232, -129.8466, 1, 0.4823529, 0, 1,
7.191919, 2.323232, -130.2216, 1, 0.4823529, 0, 1,
7.232323, 2.323232, -130.6116, 1, 0.3764706, 0, 1,
7.272727, 2.323232, -131.0168, 1, 0.3764706, 0, 1,
7.313131, 2.323232, -131.437, 1, 0.3764706, 0, 1,
7.353535, 2.323232, -131.8724, 1, 0.2745098, 0, 1,
7.393939, 2.323232, -132.323, 1, 0.2745098, 0, 1,
7.434343, 2.323232, -132.7886, 1, 0.2745098, 0, 1,
7.474748, 2.323232, -133.2694, 1, 0.1686275, 0, 1,
7.515152, 2.323232, -133.7653, 1, 0.1686275, 0, 1,
7.555555, 2.323232, -134.2763, 1, 0.06666667, 0, 1,
7.59596, 2.323232, -134.8024, 1, 0.06666667, 0, 1,
7.636364, 2.323232, -135.3437, 1, 0.06666667, 0, 1,
7.676768, 2.323232, -135.9001, 0.9647059, 0, 0.03137255, 1,
7.717172, 2.323232, -136.4716, 0.9647059, 0, 0.03137255, 1,
7.757576, 2.323232, -137.0582, 0.8588235, 0, 0.1372549, 1,
7.79798, 2.323232, -137.66, 0.8588235, 0, 0.1372549, 1,
7.838384, 2.323232, -138.2769, 0.7568628, 0, 0.2392157, 1,
7.878788, 2.323232, -138.9089, 0.7568628, 0, 0.2392157, 1,
7.919192, 2.323232, -139.556, 0.654902, 0, 0.3411765, 1,
7.959596, 2.323232, -140.2182, 0.654902, 0, 0.3411765, 1,
8, 2.323232, -140.8956, 0.5490196, 0, 0.4470588, 1,
4, 2.363636, -146.0558, 0.1372549, 0, 0.8588235, 1,
4.040404, 2.363636, -145.2784, 0.2392157, 0, 0.7568628, 1,
4.080808, 2.363636, -144.5156, 0.2392157, 0, 0.7568628, 1,
4.121212, 2.363636, -143.7675, 0.3411765, 0, 0.654902, 1,
4.161616, 2.363636, -143.0339, 0.3411765, 0, 0.654902, 1,
4.20202, 2.363636, -142.3149, 0.4470588, 0, 0.5490196, 1,
4.242424, 2.363636, -141.6106, 0.5490196, 0, 0.4470588, 1,
4.282828, 2.363636, -140.9209, 0.5490196, 0, 0.4470588, 1,
4.323232, 2.363636, -140.2457, 0.654902, 0, 0.3411765, 1,
4.363636, 2.363636, -139.5852, 0.654902, 0, 0.3411765, 1,
4.40404, 2.363636, -138.9393, 0.7568628, 0, 0.2392157, 1,
4.444445, 2.363636, -138.308, 0.7568628, 0, 0.2392157, 1,
4.484848, 2.363636, -137.6913, 0.8588235, 0, 0.1372549, 1,
4.525252, 2.363636, -137.0893, 0.8588235, 0, 0.1372549, 1,
4.565657, 2.363636, -136.5018, 0.9647059, 0, 0.03137255, 1,
4.606061, 2.363636, -135.929, 0.9647059, 0, 0.03137255, 1,
4.646465, 2.363636, -135.3707, 1, 0.06666667, 0, 1,
4.686869, 2.363636, -134.8271, 1, 0.06666667, 0, 1,
4.727273, 2.363636, -134.2981, 1, 0.06666667, 0, 1,
4.767677, 2.363636, -133.7836, 1, 0.1686275, 0, 1,
4.808081, 2.363636, -133.2838, 1, 0.1686275, 0, 1,
4.848485, 2.363636, -132.7987, 1, 0.2745098, 0, 1,
4.888889, 2.363636, -132.3281, 1, 0.2745098, 0, 1,
4.929293, 2.363636, -131.8721, 1, 0.2745098, 0, 1,
4.969697, 2.363636, -131.4307, 1, 0.3764706, 0, 1,
5.010101, 2.363636, -131.004, 1, 0.3764706, 0, 1,
5.050505, 2.363636, -130.5919, 1, 0.3764706, 0, 1,
5.090909, 2.363636, -130.1943, 1, 0.4823529, 0, 1,
5.131313, 2.363636, -129.8114, 1, 0.4823529, 0, 1,
5.171717, 2.363636, -129.4431, 1, 0.4823529, 0, 1,
5.212121, 2.363636, -129.0894, 1, 0.5843138, 0, 1,
5.252525, 2.363636, -128.7503, 1, 0.5843138, 0, 1,
5.292929, 2.363636, -128.4258, 1, 0.5843138, 0, 1,
5.333333, 2.363636, -128.116, 1, 0.5843138, 0, 1,
5.373737, 2.363636, -127.8207, 1, 0.6862745, 0, 1,
5.414141, 2.363636, -127.5401, 1, 0.6862745, 0, 1,
5.454545, 2.363636, -127.274, 1, 0.6862745, 0, 1,
5.494949, 2.363636, -127.0226, 1, 0.6862745, 0, 1,
5.535354, 2.363636, -126.7858, 1, 0.7921569, 0, 1,
5.575758, 2.363636, -126.5636, 1, 0.7921569, 0, 1,
5.616162, 2.363636, -126.356, 1, 0.7921569, 0, 1,
5.656566, 2.363636, -126.163, 1, 0.7921569, 0, 1,
5.69697, 2.363636, -125.9846, 1, 0.7921569, 0, 1,
5.737374, 2.363636, -125.8208, 1, 0.7921569, 0, 1,
5.777778, 2.363636, -125.6717, 1, 0.7921569, 0, 1,
5.818182, 2.363636, -125.5371, 1, 0.8941177, 0, 1,
5.858586, 2.363636, -125.4172, 1, 0.8941177, 0, 1,
5.89899, 2.363636, -125.3119, 1, 0.8941177, 0, 1,
5.939394, 2.363636, -125.2212, 1, 0.8941177, 0, 1,
5.979798, 2.363636, -125.1451, 1, 0.8941177, 0, 1,
6.020202, 2.363636, -125.0836, 1, 0.8941177, 0, 1,
6.060606, 2.363636, -125.0367, 1, 0.8941177, 0, 1,
6.10101, 2.363636, -125.0044, 1, 0.8941177, 0, 1,
6.141414, 2.363636, -124.9868, 1, 0.8941177, 0, 1,
6.181818, 2.363636, -124.9837, 1, 0.8941177, 0, 1,
6.222222, 2.363636, -124.9953, 1, 0.8941177, 0, 1,
6.262626, 2.363636, -125.0214, 1, 0.8941177, 0, 1,
6.30303, 2.363636, -125.0622, 1, 0.8941177, 0, 1,
6.343434, 2.363636, -125.1176, 1, 0.8941177, 0, 1,
6.383838, 2.363636, -125.1876, 1, 0.8941177, 0, 1,
6.424242, 2.363636, -125.2722, 1, 0.8941177, 0, 1,
6.464646, 2.363636, -125.3714, 1, 0.8941177, 0, 1,
6.505051, 2.363636, -125.4853, 1, 0.8941177, 0, 1,
6.545455, 2.363636, -125.6137, 1, 0.7921569, 0, 1,
6.585859, 2.363636, -125.7567, 1, 0.7921569, 0, 1,
6.626263, 2.363636, -125.9144, 1, 0.7921569, 0, 1,
6.666667, 2.363636, -126.0867, 1, 0.7921569, 0, 1,
6.707071, 2.363636, -126.2736, 1, 0.7921569, 0, 1,
6.747475, 2.363636, -126.4751, 1, 0.7921569, 0, 1,
6.787879, 2.363636, -126.6912, 1, 0.7921569, 0, 1,
6.828283, 2.363636, -126.9219, 1, 0.6862745, 0, 1,
6.868687, 2.363636, -127.1672, 1, 0.6862745, 0, 1,
6.909091, 2.363636, -127.4271, 1, 0.6862745, 0, 1,
6.949495, 2.363636, -127.7017, 1, 0.6862745, 0, 1,
6.989899, 2.363636, -127.9908, 1, 0.6862745, 0, 1,
7.030303, 2.363636, -128.2946, 1, 0.5843138, 0, 1,
7.070707, 2.363636, -128.613, 1, 0.5843138, 0, 1,
7.111111, 2.363636, -128.9459, 1, 0.5843138, 0, 1,
7.151515, 2.363636, -129.2935, 1, 0.5843138, 0, 1,
7.191919, 2.363636, -129.6557, 1, 0.4823529, 0, 1,
7.232323, 2.363636, -130.0325, 1, 0.4823529, 0, 1,
7.272727, 2.363636, -130.424, 1, 0.4823529, 0, 1,
7.313131, 2.363636, -130.83, 1, 0.3764706, 0, 1,
7.353535, 2.363636, -131.2507, 1, 0.3764706, 0, 1,
7.393939, 2.363636, -131.6859, 1, 0.3764706, 0, 1,
7.434343, 2.363636, -132.1358, 1, 0.2745098, 0, 1,
7.474748, 2.363636, -132.6003, 1, 0.2745098, 0, 1,
7.515152, 2.363636, -133.0793, 1, 0.1686275, 0, 1,
7.555555, 2.363636, -133.573, 1, 0.1686275, 0, 1,
7.59596, 2.363636, -134.0813, 1, 0.1686275, 0, 1,
7.636364, 2.363636, -134.6043, 1, 0.06666667, 0, 1,
7.676768, 2.363636, -135.1418, 1, 0.06666667, 0, 1,
7.717172, 2.363636, -135.6939, 0.9647059, 0, 0.03137255, 1,
7.757576, 2.363636, -136.2607, 0.9647059, 0, 0.03137255, 1,
7.79798, 2.363636, -136.842, 0.8588235, 0, 0.1372549, 1,
7.838384, 2.363636, -137.438, 0.8588235, 0, 0.1372549, 1,
7.878788, 2.363636, -138.0486, 0.7568628, 0, 0.2392157, 1,
7.919192, 2.363636, -138.6738, 0.7568628, 0, 0.2392157, 1,
7.959596, 2.363636, -139.3136, 0.654902, 0, 0.3411765, 1,
8, 2.363636, -139.968, 0.654902, 0, 0.3411765, 1,
4, 2.40404, -145.0002, 0.2392157, 0, 0.7568628, 1,
4.040404, 2.40404, -144.2487, 0.2392157, 0, 0.7568628, 1,
4.080808, 2.40404, -143.5113, 0.3411765, 0, 0.654902, 1,
4.121212, 2.40404, -142.7881, 0.4470588, 0, 0.5490196, 1,
4.161616, 2.40404, -142.0789, 0.4470588, 0, 0.5490196, 1,
4.20202, 2.40404, -141.384, 0.5490196, 0, 0.4470588, 1,
4.242424, 2.40404, -140.7031, 0.5490196, 0, 0.4470588, 1,
4.282828, 2.40404, -140.0363, 0.654902, 0, 0.3411765, 1,
4.323232, 2.40404, -139.3837, 0.654902, 0, 0.3411765, 1,
4.363636, 2.40404, -138.7452, 0.7568628, 0, 0.2392157, 1,
4.40404, 2.40404, -138.1208, 0.7568628, 0, 0.2392157, 1,
4.444445, 2.40404, -137.5106, 0.8588235, 0, 0.1372549, 1,
4.484848, 2.40404, -136.9145, 0.8588235, 0, 0.1372549, 1,
4.525252, 2.40404, -136.3325, 0.9647059, 0, 0.03137255, 1,
4.565657, 2.40404, -135.7646, 0.9647059, 0, 0.03137255, 1,
4.606061, 2.40404, -135.2108, 1, 0.06666667, 0, 1,
4.646465, 2.40404, -134.6712, 1, 0.06666667, 0, 1,
4.686869, 2.40404, -134.1457, 1, 0.1686275, 0, 1,
4.727273, 2.40404, -133.6343, 1, 0.1686275, 0, 1,
4.767677, 2.40404, -133.137, 1, 0.1686275, 0, 1,
4.808081, 2.40404, -132.6539, 1, 0.2745098, 0, 1,
4.848485, 2.40404, -132.1848, 1, 0.2745098, 0, 1,
4.888889, 2.40404, -131.7299, 1, 0.3764706, 0, 1,
4.929293, 2.40404, -131.2892, 1, 0.3764706, 0, 1,
4.969697, 2.40404, -130.8625, 1, 0.3764706, 0, 1,
5.010101, 2.40404, -130.45, 1, 0.4823529, 0, 1,
5.050505, 2.40404, -130.0516, 1, 0.4823529, 0, 1,
5.090909, 2.40404, -129.6673, 1, 0.4823529, 0, 1,
5.131313, 2.40404, -129.2972, 1, 0.5843138, 0, 1,
5.171717, 2.40404, -128.9411, 1, 0.5843138, 0, 1,
5.212121, 2.40404, -128.5992, 1, 0.5843138, 0, 1,
5.252525, 2.40404, -128.2714, 1, 0.5843138, 0, 1,
5.292929, 2.40404, -127.9578, 1, 0.6862745, 0, 1,
5.333333, 2.40404, -127.6582, 1, 0.6862745, 0, 1,
5.373737, 2.40404, -127.3728, 1, 0.6862745, 0, 1,
5.414141, 2.40404, -127.1015, 1, 0.6862745, 0, 1,
5.454545, 2.40404, -126.8444, 1, 0.6862745, 0, 1,
5.494949, 2.40404, -126.6013, 1, 0.7921569, 0, 1,
5.535354, 2.40404, -126.3724, 1, 0.7921569, 0, 1,
5.575758, 2.40404, -126.1576, 1, 0.7921569, 0, 1,
5.616162, 2.40404, -125.9569, 1, 0.7921569, 0, 1,
5.656566, 2.40404, -125.7704, 1, 0.7921569, 0, 1,
5.69697, 2.40404, -125.5979, 1, 0.7921569, 0, 1,
5.737374, 2.40404, -125.4396, 1, 0.8941177, 0, 1,
5.777778, 2.40404, -125.2954, 1, 0.8941177, 0, 1,
5.818182, 2.40404, -125.1654, 1, 0.8941177, 0, 1,
5.858586, 2.40404, -125.0494, 1, 0.8941177, 0, 1,
5.89899, 2.40404, -124.9476, 1, 0.8941177, 0, 1,
5.939394, 2.40404, -124.8599, 1, 0.8941177, 0, 1,
5.979798, 2.40404, -124.7864, 1, 0.8941177, 0, 1,
6.020202, 2.40404, -124.7269, 1, 0.8941177, 0, 1,
6.060606, 2.40404, -124.6816, 1, 0.8941177, 0, 1,
6.10101, 2.40404, -124.6504, 1, 0.8941177, 0, 1,
6.141414, 2.40404, -124.6333, 1, 0.8941177, 0, 1,
6.181818, 2.40404, -124.6304, 1, 0.8941177, 0, 1,
6.222222, 2.40404, -124.6415, 1, 0.8941177, 0, 1,
6.262626, 2.40404, -124.6668, 1, 0.8941177, 0, 1,
6.30303, 2.40404, -124.7063, 1, 0.8941177, 0, 1,
6.343434, 2.40404, -124.7598, 1, 0.8941177, 0, 1,
6.383838, 2.40404, -124.8275, 1, 0.8941177, 0, 1,
6.424242, 2.40404, -124.9093, 1, 0.8941177, 0, 1,
6.464646, 2.40404, -125.0052, 1, 0.8941177, 0, 1,
6.505051, 2.40404, -125.1152, 1, 0.8941177, 0, 1,
6.545455, 2.40404, -125.2394, 1, 0.8941177, 0, 1,
6.585859, 2.40404, -125.3777, 1, 0.8941177, 0, 1,
6.626263, 2.40404, -125.5301, 1, 0.8941177, 0, 1,
6.666667, 2.40404, -125.6966, 1, 0.7921569, 0, 1,
6.707071, 2.40404, -125.8772, 1, 0.7921569, 0, 1,
6.747475, 2.40404, -126.072, 1, 0.7921569, 0, 1,
6.787879, 2.40404, -126.2809, 1, 0.7921569, 0, 1,
6.828283, 2.40404, -126.5039, 1, 0.7921569, 0, 1,
6.868687, 2.40404, -126.7411, 1, 0.7921569, 0, 1,
6.909091, 2.40404, -126.9924, 1, 0.6862745, 0, 1,
6.949495, 2.40404, -127.2577, 1, 0.6862745, 0, 1,
6.989899, 2.40404, -127.5373, 1, 0.6862745, 0, 1,
7.030303, 2.40404, -127.8309, 1, 0.6862745, 0, 1,
7.070707, 2.40404, -128.1387, 1, 0.5843138, 0, 1,
7.111111, 2.40404, -128.4605, 1, 0.5843138, 0, 1,
7.151515, 2.40404, -128.7966, 1, 0.5843138, 0, 1,
7.191919, 2.40404, -129.1467, 1, 0.5843138, 0, 1,
7.232323, 2.40404, -129.5109, 1, 0.4823529, 0, 1,
7.272727, 2.40404, -129.8893, 1, 0.4823529, 0, 1,
7.313131, 2.40404, -130.2818, 1, 0.4823529, 0, 1,
7.353535, 2.40404, -130.6884, 1, 0.3764706, 0, 1,
7.393939, 2.40404, -131.1092, 1, 0.3764706, 0, 1,
7.434343, 2.40404, -131.5441, 1, 0.3764706, 0, 1,
7.474748, 2.40404, -131.9931, 1, 0.2745098, 0, 1,
7.515152, 2.40404, -132.4562, 1, 0.2745098, 0, 1,
7.555555, 2.40404, -132.9334, 1, 0.2745098, 0, 1,
7.59596, 2.40404, -133.4248, 1, 0.1686275, 0, 1,
7.636364, 2.40404, -133.9303, 1, 0.1686275, 0, 1,
7.676768, 2.40404, -134.4499, 1, 0.06666667, 0, 1,
7.717172, 2.40404, -134.9836, 1, 0.06666667, 0, 1,
7.757576, 2.40404, -135.5315, 0.9647059, 0, 0.03137255, 1,
7.79798, 2.40404, -136.0935, 0.9647059, 0, 0.03137255, 1,
7.838384, 2.40404, -136.6696, 0.9647059, 0, 0.03137255, 1,
7.878788, 2.40404, -137.2598, 0.8588235, 0, 0.1372549, 1,
7.919192, 2.40404, -137.8641, 0.8588235, 0, 0.1372549, 1,
7.959596, 2.40404, -138.4826, 0.7568628, 0, 0.2392157, 1,
8, 2.40404, -139.1152, 0.7568628, 0, 0.2392157, 1,
4, 2.444444, -144.0239, 0.3411765, 0, 0.654902, 1,
4.040404, 2.444444, -143.2971, 0.3411765, 0, 0.654902, 1,
4.080808, 2.444444, -142.5839, 0.4470588, 0, 0.5490196, 1,
4.121212, 2.444444, -141.8844, 0.4470588, 0, 0.5490196, 1,
4.161616, 2.444444, -141.1985, 0.5490196, 0, 0.4470588, 1,
4.20202, 2.444444, -140.5263, 0.5490196, 0, 0.4470588, 1,
4.242424, 2.444444, -139.8677, 0.654902, 0, 0.3411765, 1,
4.282828, 2.444444, -139.2229, 0.7568628, 0, 0.2392157, 1,
4.323232, 2.444444, -138.5916, 0.7568628, 0, 0.2392157, 1,
4.363636, 2.444444, -137.9741, 0.8588235, 0, 0.1372549, 1,
4.40404, 2.444444, -137.3702, 0.8588235, 0, 0.1372549, 1,
4.444445, 2.444444, -136.7799, 0.8588235, 0, 0.1372549, 1,
4.484848, 2.444444, -136.2033, 0.9647059, 0, 0.03137255, 1,
4.525252, 2.444444, -135.6404, 0.9647059, 0, 0.03137255, 1,
4.565657, 2.444444, -135.0911, 1, 0.06666667, 0, 1,
4.606061, 2.444444, -134.5555, 1, 0.06666667, 0, 1,
4.646465, 2.444444, -134.0336, 1, 0.1686275, 0, 1,
4.686869, 2.444444, -133.5253, 1, 0.1686275, 0, 1,
4.727273, 2.444444, -133.0307, 1, 0.1686275, 0, 1,
4.767677, 2.444444, -132.5497, 1, 0.2745098, 0, 1,
4.808081, 2.444444, -132.0824, 1, 0.2745098, 0, 1,
4.848485, 2.444444, -131.6288, 1, 0.3764706, 0, 1,
4.888889, 2.444444, -131.1888, 1, 0.3764706, 0, 1,
4.929293, 2.444444, -130.7625, 1, 0.3764706, 0, 1,
4.969697, 2.444444, -130.3498, 1, 0.4823529, 0, 1,
5.010101, 2.444444, -129.9508, 1, 0.4823529, 0, 1,
5.050505, 2.444444, -129.5655, 1, 0.4823529, 0, 1,
5.090909, 2.444444, -129.1938, 1, 0.5843138, 0, 1,
5.131313, 2.444444, -128.8358, 1, 0.5843138, 0, 1,
5.171717, 2.444444, -128.4914, 1, 0.5843138, 0, 1,
5.212121, 2.444444, -128.1607, 1, 0.5843138, 0, 1,
5.252525, 2.444444, -127.8437, 1, 0.6862745, 0, 1,
5.292929, 2.444444, -127.5403, 1, 0.6862745, 0, 1,
5.333333, 2.444444, -127.2506, 1, 0.6862745, 0, 1,
5.373737, 2.444444, -126.9745, 1, 0.6862745, 0, 1,
5.414141, 2.444444, -126.7121, 1, 0.7921569, 0, 1,
5.454545, 2.444444, -126.4634, 1, 0.7921569, 0, 1,
5.494949, 2.444444, -126.2283, 1, 0.7921569, 0, 1,
5.535354, 2.444444, -126.0069, 1, 0.7921569, 0, 1,
5.575758, 2.444444, -125.7991, 1, 0.7921569, 0, 1,
5.616162, 2.444444, -125.605, 1, 0.7921569, 0, 1,
5.656566, 2.444444, -125.4246, 1, 0.8941177, 0, 1,
5.69697, 2.444444, -125.2578, 1, 0.8941177, 0, 1,
5.737374, 2.444444, -125.1047, 1, 0.8941177, 0, 1,
5.777778, 2.444444, -124.9652, 1, 0.8941177, 0, 1,
5.818182, 2.444444, -124.8394, 1, 0.8941177, 0, 1,
5.858586, 2.444444, -124.7273, 1, 0.8941177, 0, 1,
5.89899, 2.444444, -124.6288, 1, 0.8941177, 0, 1,
5.939394, 2.444444, -124.544, 1, 0.8941177, 0, 1,
5.979798, 2.444444, -124.4728, 1, 0.8941177, 0, 1,
6.020202, 2.444444, -124.4154, 1, 0.8941177, 0, 1,
6.060606, 2.444444, -124.3715, 1, 0.8941177, 0, 1,
6.10101, 2.444444, -124.3413, 1, 0.8941177, 0, 1,
6.141414, 2.444444, -124.3248, 1, 1, 0, 1,
6.181818, 2.444444, -124.322, 1, 1, 0, 1,
6.222222, 2.444444, -124.3328, 1, 1, 0, 1,
6.262626, 2.444444, -124.3572, 1, 0.8941177, 0, 1,
6.30303, 2.444444, -124.3954, 1, 0.8941177, 0, 1,
6.343434, 2.444444, -124.4472, 1, 0.8941177, 0, 1,
6.383838, 2.444444, -124.5126, 1, 0.8941177, 0, 1,
6.424242, 2.444444, -124.5917, 1, 0.8941177, 0, 1,
6.464646, 2.444444, -124.6845, 1, 0.8941177, 0, 1,
6.505051, 2.444444, -124.7909, 1, 0.8941177, 0, 1,
6.545455, 2.444444, -124.911, 1, 0.8941177, 0, 1,
6.585859, 2.444444, -125.0448, 1, 0.8941177, 0, 1,
6.626263, 2.444444, -125.1922, 1, 0.8941177, 0, 1,
6.666667, 2.444444, -125.3532, 1, 0.8941177, 0, 1,
6.707071, 2.444444, -125.528, 1, 0.8941177, 0, 1,
6.747475, 2.444444, -125.7163, 1, 0.7921569, 0, 1,
6.787879, 2.444444, -125.9184, 1, 0.7921569, 0, 1,
6.828283, 2.444444, -126.1341, 1, 0.7921569, 0, 1,
6.868687, 2.444444, -126.3635, 1, 0.7921569, 0, 1,
6.909091, 2.444444, -126.6065, 1, 0.7921569, 0, 1,
6.949495, 2.444444, -126.8632, 1, 0.6862745, 0, 1,
6.989899, 2.444444, -127.1336, 1, 0.6862745, 0, 1,
7.030303, 2.444444, -127.4176, 1, 0.6862745, 0, 1,
7.070707, 2.444444, -127.7152, 1, 0.6862745, 0, 1,
7.111111, 2.444444, -128.0266, 1, 0.6862745, 0, 1,
7.151515, 2.444444, -128.3516, 1, 0.5843138, 0, 1,
7.191919, 2.444444, -128.6902, 1, 0.5843138, 0, 1,
7.232323, 2.444444, -129.0425, 1, 0.5843138, 0, 1,
7.272727, 2.444444, -129.4085, 1, 0.4823529, 0, 1,
7.313131, 2.444444, -129.7881, 1, 0.4823529, 0, 1,
7.353535, 2.444444, -130.1814, 1, 0.4823529, 0, 1,
7.393939, 2.444444, -130.5884, 1, 0.3764706, 0, 1,
7.434343, 2.444444, -131.009, 1, 0.3764706, 0, 1,
7.474748, 2.444444, -131.4433, 1, 0.3764706, 0, 1,
7.515152, 2.444444, -131.8912, 1, 0.2745098, 0, 1,
7.555555, 2.444444, -132.3528, 1, 0.2745098, 0, 1,
7.59596, 2.444444, -132.8281, 1, 0.2745098, 0, 1,
7.636364, 2.444444, -133.317, 1, 0.1686275, 0, 1,
7.676768, 2.444444, -133.8195, 1, 0.1686275, 0, 1,
7.717172, 2.444444, -134.3358, 1, 0.06666667, 0, 1,
7.757576, 2.444444, -134.8657, 1, 0.06666667, 0, 1,
7.79798, 2.444444, -135.4092, 1, 0.06666667, 0, 1,
7.838384, 2.444444, -135.9665, 0.9647059, 0, 0.03137255, 1,
7.878788, 2.444444, -136.5373, 0.9647059, 0, 0.03137255, 1,
7.919192, 2.444444, -137.1219, 0.8588235, 0, 0.1372549, 1,
7.959596, 2.444444, -137.7201, 0.8588235, 0, 0.1372549, 1,
8, 2.444444, -138.3319, 0.7568628, 0, 0.2392157, 1,
4, 2.484848, -143.1216, 0.3411765, 0, 0.654902, 1,
4.040404, 2.484848, -142.4182, 0.4470588, 0, 0.5490196, 1,
4.080808, 2.484848, -141.728, 0.4470588, 0, 0.5490196, 1,
4.121212, 2.484848, -141.0511, 0.5490196, 0, 0.4470588, 1,
4.161616, 2.484848, -140.3873, 0.654902, 0, 0.3411765, 1,
4.20202, 2.484848, -139.7368, 0.654902, 0, 0.3411765, 1,
4.242424, 2.484848, -139.0995, 0.7568628, 0, 0.2392157, 1,
4.282828, 2.484848, -138.4754, 0.7568628, 0, 0.2392157, 1,
4.323232, 2.484848, -137.8645, 0.8588235, 0, 0.1372549, 1,
4.363636, 2.484848, -137.2669, 0.8588235, 0, 0.1372549, 1,
4.40404, 2.484848, -136.6824, 0.9647059, 0, 0.03137255, 1,
4.444445, 2.484848, -136.1113, 0.9647059, 0, 0.03137255, 1,
4.484848, 2.484848, -135.5533, 0.9647059, 0, 0.03137255, 1,
4.525252, 2.484848, -135.0085, 1, 0.06666667, 0, 1,
4.565657, 2.484848, -134.4769, 1, 0.06666667, 0, 1,
4.606061, 2.484848, -133.9586, 1, 0.1686275, 0, 1,
4.646465, 2.484848, -133.4535, 1, 0.1686275, 0, 1,
4.686869, 2.484848, -132.9616, 1, 0.2745098, 0, 1,
4.727273, 2.484848, -132.483, 1, 0.2745098, 0, 1,
4.767677, 2.484848, -132.0175, 1, 0.2745098, 0, 1,
4.808081, 2.484848, -131.5653, 1, 0.3764706, 0, 1,
4.848485, 2.484848, -131.1263, 1, 0.3764706, 0, 1,
4.888889, 2.484848, -130.7005, 1, 0.3764706, 0, 1,
4.929293, 2.484848, -130.2879, 1, 0.4823529, 0, 1,
4.969697, 2.484848, -129.8885, 1, 0.4823529, 0, 1,
5.010101, 2.484848, -129.5024, 1, 0.4823529, 0, 1,
5.050505, 2.484848, -129.1295, 1, 0.5843138, 0, 1,
5.090909, 2.484848, -128.7698, 1, 0.5843138, 0, 1,
5.131313, 2.484848, -128.4233, 1, 0.5843138, 0, 1,
5.171717, 2.484848, -128.0901, 1, 0.5843138, 0, 1,
5.212121, 2.484848, -127.7701, 1, 0.6862745, 0, 1,
5.252525, 2.484848, -127.4632, 1, 0.6862745, 0, 1,
5.292929, 2.484848, -127.1696, 1, 0.6862745, 0, 1,
5.333333, 2.484848, -126.8893, 1, 0.6862745, 0, 1,
5.373737, 2.484848, -126.6221, 1, 0.7921569, 0, 1,
5.414141, 2.484848, -126.3682, 1, 0.7921569, 0, 1,
5.454545, 2.484848, -126.1275, 1, 0.7921569, 0, 1,
5.494949, 2.484848, -125.9, 1, 0.7921569, 0, 1,
5.535354, 2.484848, -125.6857, 1, 0.7921569, 0, 1,
5.575758, 2.484848, -125.4846, 1, 0.8941177, 0, 1,
5.616162, 2.484848, -125.2968, 1, 0.8941177, 0, 1,
5.656566, 2.484848, -125.1222, 1, 0.8941177, 0, 1,
5.69697, 2.484848, -124.9608, 1, 0.8941177, 0, 1,
5.737374, 2.484848, -124.8126, 1, 0.8941177, 0, 1,
5.777778, 2.484848, -124.6777, 1, 0.8941177, 0, 1,
5.818182, 2.484848, -124.5559, 1, 0.8941177, 0, 1,
5.858586, 2.484848, -124.4474, 1, 0.8941177, 0, 1,
5.89899, 2.484848, -124.3521, 1, 0.8941177, 0, 1,
5.939394, 2.484848, -124.27, 1, 1, 0, 1,
5.979798, 2.484848, -124.2012, 1, 1, 0, 1,
6.020202, 2.484848, -124.1455, 1, 1, 0, 1,
6.060606, 2.484848, -124.1031, 1, 1, 0, 1,
6.10101, 2.484848, -124.0739, 1, 1, 0, 1,
6.141414, 2.484848, -124.0579, 1, 1, 0, 1,
6.181818, 2.484848, -124.0552, 1, 1, 0, 1,
6.222222, 2.484848, -124.0656, 1, 1, 0, 1,
6.262626, 2.484848, -124.0893, 1, 1, 0, 1,
6.30303, 2.484848, -124.1262, 1, 1, 0, 1,
6.343434, 2.484848, -124.1763, 1, 1, 0, 1,
6.383838, 2.484848, -124.2396, 1, 1, 0, 1,
6.424242, 2.484848, -124.3162, 1, 1, 0, 1,
6.464646, 2.484848, -124.406, 1, 0.8941177, 0, 1,
6.505051, 2.484848, -124.509, 1, 0.8941177, 0, 1,
6.545455, 2.484848, -124.6252, 1, 0.8941177, 0, 1,
6.585859, 2.484848, -124.7546, 1, 0.8941177, 0, 1,
6.626263, 2.484848, -124.8973, 1, 0.8941177, 0, 1,
6.666667, 2.484848, -125.0531, 1, 0.8941177, 0, 1,
6.707071, 2.484848, -125.2222, 1, 0.8941177, 0, 1,
6.747475, 2.484848, -125.4045, 1, 0.8941177, 0, 1,
6.787879, 2.484848, -125.6001, 1, 0.7921569, 0, 1,
6.828283, 2.484848, -125.8088, 1, 0.7921569, 0, 1,
6.868687, 2.484848, -126.0308, 1, 0.7921569, 0, 1,
6.909091, 2.484848, -126.266, 1, 0.7921569, 0, 1,
6.949495, 2.484848, -126.5144, 1, 0.7921569, 0, 1,
6.989899, 2.484848, -126.776, 1, 0.7921569, 0, 1,
7.030303, 2.484848, -127.0509, 1, 0.6862745, 0, 1,
7.070707, 2.484848, -127.339, 1, 0.6862745, 0, 1,
7.111111, 2.484848, -127.6403, 1, 0.6862745, 0, 1,
7.151515, 2.484848, -127.9548, 1, 0.6862745, 0, 1,
7.191919, 2.484848, -128.2825, 1, 0.5843138, 0, 1,
7.232323, 2.484848, -128.6234, 1, 0.5843138, 0, 1,
7.272727, 2.484848, -128.9776, 1, 0.5843138, 0, 1,
7.313131, 2.484848, -129.345, 1, 0.4823529, 0, 1,
7.353535, 2.484848, -129.7256, 1, 0.4823529, 0, 1,
7.393939, 2.484848, -130.1194, 1, 0.4823529, 0, 1,
7.434343, 2.484848, -130.5265, 1, 0.4823529, 0, 1,
7.474748, 2.484848, -130.9467, 1, 0.3764706, 0, 1,
7.515152, 2.484848, -131.3802, 1, 0.3764706, 0, 1,
7.555555, 2.484848, -131.8269, 1, 0.2745098, 0, 1,
7.59596, 2.484848, -132.2869, 1, 0.2745098, 0, 1,
7.636364, 2.484848, -132.76, 1, 0.2745098, 0, 1,
7.676768, 2.484848, -133.2464, 1, 0.1686275, 0, 1,
7.717172, 2.484848, -133.746, 1, 0.1686275, 0, 1,
7.757576, 2.484848, -134.2588, 1, 0.1686275, 0, 1,
7.79798, 2.484848, -134.7848, 1, 0.06666667, 0, 1,
7.838384, 2.484848, -135.324, 1, 0.06666667, 0, 1,
7.878788, 2.484848, -135.8765, 0.9647059, 0, 0.03137255, 1,
7.919192, 2.484848, -136.4422, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.484848, -137.0211, 0.8588235, 0, 0.1372549, 1,
8, 2.484848, -137.6132, 0.8588235, 0, 0.1372549, 1,
4, 2.525253, -142.288, 0.4470588, 0, 0.5490196, 1,
4.040404, 2.525253, -141.607, 0.5490196, 0, 0.4470588, 1,
4.080808, 2.525253, -140.9387, 0.5490196, 0, 0.4470588, 1,
4.121212, 2.525253, -140.2832, 0.654902, 0, 0.3411765, 1,
4.161616, 2.525253, -139.6405, 0.654902, 0, 0.3411765, 1,
4.20202, 2.525253, -139.0107, 0.7568628, 0, 0.2392157, 1,
4.242424, 2.525253, -138.3936, 0.7568628, 0, 0.2392157, 1,
4.282828, 2.525253, -137.7893, 0.8588235, 0, 0.1372549, 1,
4.323232, 2.525253, -137.1978, 0.8588235, 0, 0.1372549, 1,
4.363636, 2.525253, -136.6192, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.525253, -136.0533, 0.9647059, 0, 0.03137255, 1,
4.444445, 2.525253, -135.5002, 1, 0.06666667, 0, 1,
4.484848, 2.525253, -134.96, 1, 0.06666667, 0, 1,
4.525252, 2.525253, -134.4325, 1, 0.06666667, 0, 1,
4.565657, 2.525253, -133.9178, 1, 0.1686275, 0, 1,
4.606061, 2.525253, -133.4159, 1, 0.1686275, 0, 1,
4.646465, 2.525253, -132.9269, 1, 0.2745098, 0, 1,
4.686869, 2.525253, -132.4506, 1, 0.2745098, 0, 1,
4.727273, 2.525253, -131.9871, 1, 0.2745098, 0, 1,
4.767677, 2.525253, -131.5364, 1, 0.3764706, 0, 1,
4.808081, 2.525253, -131.0986, 1, 0.3764706, 0, 1,
4.848485, 2.525253, -130.6735, 1, 0.3764706, 0, 1,
4.888889, 2.525253, -130.2612, 1, 0.4823529, 0, 1,
4.929293, 2.525253, -129.8617, 1, 0.4823529, 0, 1,
4.969697, 2.525253, -129.4751, 1, 0.4823529, 0, 1,
5.010101, 2.525253, -129.1012, 1, 0.5843138, 0, 1,
5.050505, 2.525253, -128.7401, 1, 0.5843138, 0, 1,
5.090909, 2.525253, -128.3918, 1, 0.5843138, 0, 1,
5.131313, 2.525253, -128.0564, 1, 0.6862745, 0, 1,
5.171717, 2.525253, -127.7337, 1, 0.6862745, 0, 1,
5.212121, 2.525253, -127.4238, 1, 0.6862745, 0, 1,
5.252525, 2.525253, -127.1267, 1, 0.6862745, 0, 1,
5.292929, 2.525253, -126.8425, 1, 0.6862745, 0, 1,
5.333333, 2.525253, -126.571, 1, 0.7921569, 0, 1,
5.373737, 2.525253, -126.3123, 1, 0.7921569, 0, 1,
5.414141, 2.525253, -126.0665, 1, 0.7921569, 0, 1,
5.454545, 2.525253, -125.8334, 1, 0.7921569, 0, 1,
5.494949, 2.525253, -125.6131, 1, 0.7921569, 0, 1,
5.535354, 2.525253, -125.4056, 1, 0.8941177, 0, 1,
5.575758, 2.525253, -125.211, 1, 0.8941177, 0, 1,
5.616162, 2.525253, -125.0291, 1, 0.8941177, 0, 1,
5.656566, 2.525253, -124.86, 1, 0.8941177, 0, 1,
5.69697, 2.525253, -124.7037, 1, 0.8941177, 0, 1,
5.737374, 2.525253, -124.5603, 1, 0.8941177, 0, 1,
5.777778, 2.525253, -124.4296, 1, 0.8941177, 0, 1,
5.818182, 2.525253, -124.3117, 1, 1, 0, 1,
5.858586, 2.525253, -124.2066, 1, 1, 0, 1,
5.89899, 2.525253, -124.1144, 1, 1, 0, 1,
5.939394, 2.525253, -124.0349, 1, 1, 0, 1,
5.979798, 2.525253, -123.9682, 1, 1, 0, 1,
6.020202, 2.525253, -123.9143, 1, 1, 0, 1,
6.060606, 2.525253, -123.8733, 1, 1, 0, 1,
6.10101, 2.525253, -123.845, 1, 1, 0, 1,
6.141414, 2.525253, -123.8295, 1, 1, 0, 1,
6.181818, 2.525253, -123.8268, 1, 1, 0, 1,
6.222222, 2.525253, -123.837, 1, 1, 0, 1,
6.262626, 2.525253, -123.8599, 1, 1, 0, 1,
6.30303, 2.525253, -123.8956, 1, 1, 0, 1,
6.343434, 2.525253, -123.9441, 1, 1, 0, 1,
6.383838, 2.525253, -124.0055, 1, 1, 0, 1,
6.424242, 2.525253, -124.0796, 1, 1, 0, 1,
6.464646, 2.525253, -124.1665, 1, 1, 0, 1,
6.505051, 2.525253, -124.2663, 1, 1, 0, 1,
6.545455, 2.525253, -124.3788, 1, 0.8941177, 0, 1,
6.585859, 2.525253, -124.5041, 1, 0.8941177, 0, 1,
6.626263, 2.525253, -124.6422, 1, 0.8941177, 0, 1,
6.666667, 2.525253, -124.7932, 1, 0.8941177, 0, 1,
6.707071, 2.525253, -124.9569, 1, 0.8941177, 0, 1,
6.747475, 2.525253, -125.1334, 1, 0.8941177, 0, 1,
6.787879, 2.525253, -125.3227, 1, 0.8941177, 0, 1,
6.828283, 2.525253, -125.5249, 1, 0.8941177, 0, 1,
6.868687, 2.525253, -125.7398, 1, 0.7921569, 0, 1,
6.909091, 2.525253, -125.9675, 1, 0.7921569, 0, 1,
6.949495, 2.525253, -126.208, 1, 0.7921569, 0, 1,
6.989899, 2.525253, -126.4614, 1, 0.7921569, 0, 1,
7.030303, 2.525253, -126.7275, 1, 0.7921569, 0, 1,
7.070707, 2.525253, -127.0064, 1, 0.6862745, 0, 1,
7.111111, 2.525253, -127.2981, 1, 0.6862745, 0, 1,
7.151515, 2.525253, -127.6027, 1, 0.6862745, 0, 1,
7.191919, 2.525253, -127.92, 1, 0.6862745, 0, 1,
7.232323, 2.525253, -128.2501, 1, 0.5843138, 0, 1,
7.272727, 2.525253, -128.593, 1, 0.5843138, 0, 1,
7.313131, 2.525253, -128.9488, 1, 0.5843138, 0, 1,
7.353535, 2.525253, -129.3173, 1, 0.4823529, 0, 1,
7.393939, 2.525253, -129.6986, 1, 0.4823529, 0, 1,
7.434343, 2.525253, -130.0927, 1, 0.4823529, 0, 1,
7.474748, 2.525253, -130.4997, 1, 0.4823529, 0, 1,
7.515152, 2.525253, -130.9194, 1, 0.3764706, 0, 1,
7.555555, 2.525253, -131.3519, 1, 0.3764706, 0, 1,
7.59596, 2.525253, -131.7972, 1, 0.2745098, 0, 1,
7.636364, 2.525253, -132.2554, 1, 0.2745098, 0, 1,
7.676768, 2.525253, -132.7263, 1, 0.2745098, 0, 1,
7.717172, 2.525253, -133.21, 1, 0.1686275, 0, 1,
7.757576, 2.525253, -133.7066, 1, 0.1686275, 0, 1,
7.79798, 2.525253, -134.2159, 1, 0.1686275, 0, 1,
7.838384, 2.525253, -134.738, 1, 0.06666667, 0, 1,
7.878788, 2.525253, -135.2729, 1, 0.06666667, 0, 1,
7.919192, 2.525253, -135.8206, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.525253, -136.3812, 0.9647059, 0, 0.03137255, 1,
8, 2.525253, -136.9545, 0.8588235, 0, 0.1372549, 1,
4, 2.565657, -141.5186, 0.5490196, 0, 0.4470588, 1,
4.040404, 2.565657, -140.8588, 0.5490196, 0, 0.4470588, 1,
4.080808, 2.565657, -140.2114, 0.654902, 0, 0.3411765, 1,
4.121212, 2.565657, -139.5764, 0.654902, 0, 0.3411765, 1,
4.161616, 2.565657, -138.9538, 0.7568628, 0, 0.2392157, 1,
4.20202, 2.565657, -138.3436, 0.7568628, 0, 0.2392157, 1,
4.242424, 2.565657, -137.7458, 0.8588235, 0, 0.1372549, 1,
4.282828, 2.565657, -137.1604, 0.8588235, 0, 0.1372549, 1,
4.323232, 2.565657, -136.5874, 0.9647059, 0, 0.03137255, 1,
4.363636, 2.565657, -136.0268, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.565657, -135.4786, 1, 0.06666667, 0, 1,
4.444445, 2.565657, -134.9428, 1, 0.06666667, 0, 1,
4.484848, 2.565657, -134.4194, 1, 0.06666667, 0, 1,
4.525252, 2.565657, -133.9085, 1, 0.1686275, 0, 1,
4.565657, 2.565657, -133.4099, 1, 0.1686275, 0, 1,
4.606061, 2.565657, -132.9237, 1, 0.2745098, 0, 1,
4.646465, 2.565657, -132.4499, 1, 0.2745098, 0, 1,
4.686869, 2.565657, -131.9885, 1, 0.2745098, 0, 1,
4.727273, 2.565657, -131.5395, 1, 0.3764706, 0, 1,
4.767677, 2.565657, -131.1029, 1, 0.3764706, 0, 1,
4.808081, 2.565657, -130.6787, 1, 0.3764706, 0, 1,
4.848485, 2.565657, -130.2669, 1, 0.4823529, 0, 1,
4.888889, 2.565657, -129.8675, 1, 0.4823529, 0, 1,
4.929293, 2.565657, -129.4805, 1, 0.4823529, 0, 1,
4.969697, 2.565657, -129.106, 1, 0.5843138, 0, 1,
5.010101, 2.565657, -128.7438, 1, 0.5843138, 0, 1,
5.050505, 2.565657, -128.394, 1, 0.5843138, 0, 1,
5.090909, 2.565657, -128.0566, 1, 0.6862745, 0, 1,
5.131313, 2.565657, -127.7316, 1, 0.6862745, 0, 1,
5.171717, 2.565657, -127.419, 1, 0.6862745, 0, 1,
5.212121, 2.565657, -127.1188, 1, 0.6862745, 0, 1,
5.252525, 2.565657, -126.831, 1, 0.6862745, 0, 1,
5.292929, 2.565657, -126.5556, 1, 0.7921569, 0, 1,
5.333333, 2.565657, -126.2926, 1, 0.7921569, 0, 1,
5.373737, 2.565657, -126.042, 1, 0.7921569, 0, 1,
5.414141, 2.565657, -125.8039, 1, 0.7921569, 0, 1,
5.454545, 2.565657, -125.5781, 1, 0.7921569, 0, 1,
5.494949, 2.565657, -125.3647, 1, 0.8941177, 0, 1,
5.535354, 2.565657, -125.1637, 1, 0.8941177, 0, 1,
5.575758, 2.565657, -124.9751, 1, 0.8941177, 0, 1,
5.616162, 2.565657, -124.7989, 1, 0.8941177, 0, 1,
5.656566, 2.565657, -124.6351, 1, 0.8941177, 0, 1,
5.69697, 2.565657, -124.4837, 1, 0.8941177, 0, 1,
5.737374, 2.565657, -124.3447, 1, 0.8941177, 0, 1,
5.777778, 2.565657, -124.2181, 1, 1, 0, 1,
5.818182, 2.565657, -124.104, 1, 1, 0, 1,
5.858586, 2.565657, -124.0022, 1, 1, 0, 1,
5.89899, 2.565657, -123.9128, 1, 1, 0, 1,
5.939394, 2.565657, -123.8358, 1, 1, 0, 1,
5.979798, 2.565657, -123.7712, 1, 1, 0, 1,
6.020202, 2.565657, -123.719, 1, 1, 0, 1,
6.060606, 2.565657, -123.6792, 1, 1, 0, 1,
6.10101, 2.565657, -123.6518, 1, 1, 0, 1,
6.141414, 2.565657, -123.6368, 1, 1, 0, 1,
6.181818, 2.565657, -123.6342, 1, 1, 0, 1,
6.222222, 2.565657, -123.644, 1, 1, 0, 1,
6.262626, 2.565657, -123.6663, 1, 1, 0, 1,
6.30303, 2.565657, -123.7009, 1, 1, 0, 1,
6.343434, 2.565657, -123.7479, 1, 1, 0, 1,
6.383838, 2.565657, -123.8073, 1, 1, 0, 1,
6.424242, 2.565657, -123.8791, 1, 1, 0, 1,
6.464646, 2.565657, -123.9633, 1, 1, 0, 1,
6.505051, 2.565657, -124.0599, 1, 1, 0, 1,
6.545455, 2.565657, -124.1689, 1, 1, 0, 1,
6.585859, 2.565657, -124.2903, 1, 1, 0, 1,
6.626263, 2.565657, -124.4241, 1, 0.8941177, 0, 1,
6.666667, 2.565657, -124.5703, 1, 0.8941177, 0, 1,
6.707071, 2.565657, -124.729, 1, 0.8941177, 0, 1,
6.747475, 2.565657, -124.9, 1, 0.8941177, 0, 1,
6.787879, 2.565657, -125.0834, 1, 0.8941177, 0, 1,
6.828283, 2.565657, -125.2792, 1, 0.8941177, 0, 1,
6.868687, 2.565657, -125.4874, 1, 0.8941177, 0, 1,
6.909091, 2.565657, -125.708, 1, 0.7921569, 0, 1,
6.949495, 2.565657, -125.941, 1, 0.7921569, 0, 1,
6.989899, 2.565657, -126.1864, 1, 0.7921569, 0, 1,
7.030303, 2.565657, -126.4442, 1, 0.7921569, 0, 1,
7.070707, 2.565657, -126.7144, 1, 0.7921569, 0, 1,
7.111111, 2.565657, -126.9971, 1, 0.6862745, 0, 1,
7.151515, 2.565657, -127.2921, 1, 0.6862745, 0, 1,
7.191919, 2.565657, -127.5995, 1, 0.6862745, 0, 1,
7.232323, 2.565657, -127.9193, 1, 0.6862745, 0, 1,
7.272727, 2.565657, -128.2515, 1, 0.5843138, 0, 1,
7.313131, 2.565657, -128.5961, 1, 0.5843138, 0, 1,
7.353535, 2.565657, -128.9531, 1, 0.5843138, 0, 1,
7.393939, 2.565657, -129.3225, 1, 0.4823529, 0, 1,
7.434343, 2.565657, -129.7043, 1, 0.4823529, 0, 1,
7.474748, 2.565657, -130.0985, 1, 0.4823529, 0, 1,
7.515152, 2.565657, -130.5052, 1, 0.4823529, 0, 1,
7.555555, 2.565657, -130.9242, 1, 0.3764706, 0, 1,
7.59596, 2.565657, -131.3556, 1, 0.3764706, 0, 1,
7.636364, 2.565657, -131.7994, 1, 0.2745098, 0, 1,
7.676768, 2.565657, -132.2556, 1, 0.2745098, 0, 1,
7.717172, 2.565657, -132.7242, 1, 0.2745098, 0, 1,
7.757576, 2.565657, -133.2052, 1, 0.1686275, 0, 1,
7.79798, 2.565657, -133.6986, 1, 0.1686275, 0, 1,
7.838384, 2.565657, -134.2044, 1, 0.1686275, 0, 1,
7.878788, 2.565657, -134.7227, 1, 0.06666667, 0, 1,
7.919192, 2.565657, -135.2533, 1, 0.06666667, 0, 1,
7.959596, 2.565657, -135.7963, 0.9647059, 0, 0.03137255, 1,
8, 2.565657, -136.3517, 0.9647059, 0, 0.03137255, 1,
4, 2.606061, -140.8088, 0.5490196, 0, 0.4470588, 1,
4.040404, 2.606061, -140.1693, 0.654902, 0, 0.3411765, 1,
4.080808, 2.606061, -139.5418, 0.654902, 0, 0.3411765, 1,
4.121212, 2.606061, -138.9264, 0.7568628, 0, 0.2392157, 1,
4.161616, 2.606061, -138.323, 0.7568628, 0, 0.2392157, 1,
4.20202, 2.606061, -137.7315, 0.8588235, 0, 0.1372549, 1,
4.242424, 2.606061, -137.1521, 0.8588235, 0, 0.1372549, 1,
4.282828, 2.606061, -136.5847, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.606061, -136.0294, 0.9647059, 0, 0.03137255, 1,
4.363636, 2.606061, -135.4861, 1, 0.06666667, 0, 1,
4.40404, 2.606061, -134.9547, 1, 0.06666667, 0, 1,
4.444445, 2.606061, -134.4354, 1, 0.06666667, 0, 1,
4.484848, 2.606061, -133.9281, 1, 0.1686275, 0, 1,
4.525252, 2.606061, -133.4329, 1, 0.1686275, 0, 1,
4.565657, 2.606061, -132.9496, 1, 0.2745098, 0, 1,
4.606061, 2.606061, -132.4784, 1, 0.2745098, 0, 1,
4.646465, 2.606061, -132.0192, 1, 0.2745098, 0, 1,
4.686869, 2.606061, -131.572, 1, 0.3764706, 0, 1,
4.727273, 2.606061, -131.1368, 1, 0.3764706, 0, 1,
4.767677, 2.606061, -130.7136, 1, 0.3764706, 0, 1,
4.808081, 2.606061, -130.3025, 1, 0.4823529, 0, 1,
4.848485, 2.606061, -129.9034, 1, 0.4823529, 0, 1,
4.888889, 2.606061, -129.5163, 1, 0.4823529, 0, 1,
4.929293, 2.606061, -129.1412, 1, 0.5843138, 0, 1,
4.969697, 2.606061, -128.7781, 1, 0.5843138, 0, 1,
5.010101, 2.606061, -128.4271, 1, 0.5843138, 0, 1,
5.050505, 2.606061, -128.088, 1, 0.5843138, 0, 1,
5.090909, 2.606061, -127.761, 1, 0.6862745, 0, 1,
5.131313, 2.606061, -127.446, 1, 0.6862745, 0, 1,
5.171717, 2.606061, -127.1431, 1, 0.6862745, 0, 1,
5.212121, 2.606061, -126.8521, 1, 0.6862745, 0, 1,
5.252525, 2.606061, -126.5732, 1, 0.7921569, 0, 1,
5.292929, 2.606061, -126.3063, 1, 0.7921569, 0, 1,
5.333333, 2.606061, -126.0514, 1, 0.7921569, 0, 1,
5.373737, 2.606061, -125.8085, 1, 0.7921569, 0, 1,
5.414141, 2.606061, -125.5776, 1, 0.7921569, 0, 1,
5.454545, 2.606061, -125.3588, 1, 0.8941177, 0, 1,
5.494949, 2.606061, -125.1519, 1, 0.8941177, 0, 1,
5.535354, 2.606061, -124.9571, 1, 0.8941177, 0, 1,
5.575758, 2.606061, -124.7743, 1, 0.8941177, 0, 1,
5.616162, 2.606061, -124.6036, 1, 0.8941177, 0, 1,
5.656566, 2.606061, -124.4448, 1, 0.8941177, 0, 1,
5.69697, 2.606061, -124.2981, 1, 1, 0, 1,
5.737374, 2.606061, -124.1634, 1, 1, 0, 1,
5.777778, 2.606061, -124.0407, 1, 1, 0, 1,
5.818182, 2.606061, -123.93, 1, 1, 0, 1,
5.858586, 2.606061, -123.8313, 1, 1, 0, 1,
5.89899, 2.606061, -123.7447, 1, 1, 0, 1,
5.939394, 2.606061, -123.6701, 1, 1, 0, 1,
5.979798, 2.606061, -123.6075, 1, 1, 0, 1,
6.020202, 2.606061, -123.5569, 1, 1, 0, 1,
6.060606, 2.606061, -123.5183, 1, 1, 0, 1,
6.10101, 2.606061, -123.4918, 1, 1, 0, 1,
6.141414, 2.606061, -123.4772, 1, 1, 0, 1,
6.181818, 2.606061, -123.4747, 1, 1, 0, 1,
6.222222, 2.606061, -123.4842, 1, 1, 0, 1,
6.262626, 2.606061, -123.5058, 1, 1, 0, 1,
6.30303, 2.606061, -123.5393, 1, 1, 0, 1,
6.343434, 2.606061, -123.5849, 1, 1, 0, 1,
6.383838, 2.606061, -123.6425, 1, 1, 0, 1,
6.424242, 2.606061, -123.7121, 1, 1, 0, 1,
6.464646, 2.606061, -123.7937, 1, 1, 0, 1,
6.505051, 2.606061, -123.8873, 1, 1, 0, 1,
6.545455, 2.606061, -123.993, 1, 1, 0, 1,
6.585859, 2.606061, -124.1106, 1, 1, 0, 1,
6.626263, 2.606061, -124.2403, 1, 1, 0, 1,
6.666667, 2.606061, -124.382, 1, 0.8941177, 0, 1,
6.707071, 2.606061, -124.5358, 1, 0.8941177, 0, 1,
6.747475, 2.606061, -124.7015, 1, 0.8941177, 0, 1,
6.787879, 2.606061, -124.8793, 1, 0.8941177, 0, 1,
6.828283, 2.606061, -125.0691, 1, 0.8941177, 0, 1,
6.868687, 2.606061, -125.2709, 1, 0.8941177, 0, 1,
6.909091, 2.606061, -125.4847, 1, 0.8941177, 0, 1,
6.949495, 2.606061, -125.7105, 1, 0.7921569, 0, 1,
6.989899, 2.606061, -125.9484, 1, 0.7921569, 0, 1,
7.030303, 2.606061, -126.1983, 1, 0.7921569, 0, 1,
7.070707, 2.606061, -126.4602, 1, 0.7921569, 0, 1,
7.111111, 2.606061, -126.7341, 1, 0.7921569, 0, 1,
7.151515, 2.606061, -127.02, 1, 0.6862745, 0, 1,
7.191919, 2.606061, -127.318, 1, 0.6862745, 0, 1,
7.232323, 2.606061, -127.628, 1, 0.6862745, 0, 1,
7.272727, 2.606061, -127.9499, 1, 0.6862745, 0, 1,
7.313131, 2.606061, -128.284, 1, 0.5843138, 0, 1,
7.353535, 2.606061, -128.63, 1, 0.5843138, 0, 1,
7.393939, 2.606061, -128.988, 1, 0.5843138, 0, 1,
7.434343, 2.606061, -129.3581, 1, 0.4823529, 0, 1,
7.474748, 2.606061, -129.7402, 1, 0.4823529, 0, 1,
7.515152, 2.606061, -130.1343, 1, 0.4823529, 0, 1,
7.555555, 2.606061, -130.5404, 1, 0.4823529, 0, 1,
7.59596, 2.606061, -130.9585, 1, 0.3764706, 0, 1,
7.636364, 2.606061, -131.3887, 1, 0.3764706, 0, 1,
7.676768, 2.606061, -131.8308, 1, 0.2745098, 0, 1,
7.717172, 2.606061, -132.285, 1, 0.2745098, 0, 1,
7.757576, 2.606061, -132.7513, 1, 0.2745098, 0, 1,
7.79798, 2.606061, -133.2295, 1, 0.1686275, 0, 1,
7.838384, 2.606061, -133.7197, 1, 0.1686275, 0, 1,
7.878788, 2.606061, -134.222, 1, 0.1686275, 0, 1,
7.919192, 2.606061, -134.7363, 1, 0.06666667, 0, 1,
7.959596, 2.606061, -135.2626, 1, 0.06666667, 0, 1,
8, 2.606061, -135.8009, 0.9647059, 0, 0.03137255, 1,
4, 2.646465, -140.1548, 0.654902, 0, 0.3411765, 1,
4.040404, 2.646465, -139.5347, 0.654902, 0, 0.3411765, 1,
4.080808, 2.646465, -138.9262, 0.7568628, 0, 0.2392157, 1,
4.121212, 2.646465, -138.3294, 0.7568628, 0, 0.2392157, 1,
4.161616, 2.646465, -137.7443, 0.8588235, 0, 0.1372549, 1,
4.20202, 2.646465, -137.1708, 0.8588235, 0, 0.1372549, 1,
4.242424, 2.646465, -136.6089, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.646465, -136.0587, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.646465, -135.5202, 0.9647059, 0, 0.03137255, 1,
4.363636, 2.646465, -134.9933, 1, 0.06666667, 0, 1,
4.40404, 2.646465, -134.4781, 1, 0.06666667, 0, 1,
4.444445, 2.646465, -133.9745, 1, 0.1686275, 0, 1,
4.484848, 2.646465, -133.4826, 1, 0.1686275, 0, 1,
4.525252, 2.646465, -133.0023, 1, 0.2745098, 0, 1,
4.565657, 2.646465, -132.5338, 1, 0.2745098, 0, 1,
4.606061, 2.646465, -132.0768, 1, 0.2745098, 0, 1,
4.646465, 2.646465, -131.6315, 1, 0.3764706, 0, 1,
4.686869, 2.646465, -131.1978, 1, 0.3764706, 0, 1,
4.727273, 2.646465, -130.7759, 1, 0.3764706, 0, 1,
4.767677, 2.646465, -130.3655, 1, 0.4823529, 0, 1,
4.808081, 2.646465, -129.9668, 1, 0.4823529, 0, 1,
4.848485, 2.646465, -129.5798, 1, 0.4823529, 0, 1,
4.888889, 2.646465, -129.2044, 1, 0.5843138, 0, 1,
4.929293, 2.646465, -128.8407, 1, 0.5843138, 0, 1,
4.969697, 2.646465, -128.4887, 1, 0.5843138, 0, 1,
5.010101, 2.646465, -128.1482, 1, 0.5843138, 0, 1,
5.050505, 2.646465, -127.8195, 1, 0.6862745, 0, 1,
5.090909, 2.646465, -127.5024, 1, 0.6862745, 0, 1,
5.131313, 2.646465, -127.1969, 1, 0.6862745, 0, 1,
5.171717, 2.646465, -126.9031, 1, 0.6862745, 0, 1,
5.212121, 2.646465, -126.621, 1, 0.7921569, 0, 1,
5.252525, 2.646465, -126.3505, 1, 0.7921569, 0, 1,
5.292929, 2.646465, -126.0917, 1, 0.7921569, 0, 1,
5.333333, 2.646465, -125.8445, 1, 0.7921569, 0, 1,
5.373737, 2.646465, -125.609, 1, 0.7921569, 0, 1,
5.414141, 2.646465, -125.3851, 1, 0.8941177, 0, 1,
5.454545, 2.646465, -125.1729, 1, 0.8941177, 0, 1,
5.494949, 2.646465, -124.9724, 1, 0.8941177, 0, 1,
5.535354, 2.646465, -124.7835, 1, 0.8941177, 0, 1,
5.575758, 2.646465, -124.6062, 1, 0.8941177, 0, 1,
5.616162, 2.646465, -124.4406, 1, 0.8941177, 0, 1,
5.656566, 2.646465, -124.2867, 1, 1, 0, 1,
5.69697, 2.646465, -124.1444, 1, 1, 0, 1,
5.737374, 2.646465, -124.0137, 1, 1, 0, 1,
5.777778, 2.646465, -123.8948, 1, 1, 0, 1,
5.818182, 2.646465, -123.7874, 1, 1, 0, 1,
5.858586, 2.646465, -123.6918, 1, 1, 0, 1,
5.89899, 2.646465, -123.6078, 1, 1, 0, 1,
5.939394, 2.646465, -123.5354, 1, 1, 0, 1,
5.979798, 2.646465, -123.4747, 1, 1, 0, 1,
6.020202, 2.646465, -123.4256, 1, 1, 0, 1,
6.060606, 2.646465, -123.3882, 1, 1, 0, 1,
6.10101, 2.646465, -123.3625, 1, 1, 0, 1,
6.141414, 2.646465, -123.3484, 1, 1, 0, 1,
6.181818, 2.646465, -123.346, 1, 1, 0, 1,
6.222222, 2.646465, -123.3552, 1, 1, 0, 1,
6.262626, 2.646465, -123.3761, 1, 1, 0, 1,
6.30303, 2.646465, -123.4086, 1, 1, 0, 1,
6.343434, 2.646465, -123.4528, 1, 1, 0, 1,
6.383838, 2.646465, -123.5086, 1, 1, 0, 1,
6.424242, 2.646465, -123.5761, 1, 1, 0, 1,
6.464646, 2.646465, -123.6553, 1, 1, 0, 1,
6.505051, 2.646465, -123.7461, 1, 1, 0, 1,
6.545455, 2.646465, -123.8485, 1, 1, 0, 1,
6.585859, 2.646465, -123.9626, 1, 1, 0, 1,
6.626263, 2.646465, -124.0884, 1, 1, 0, 1,
6.666667, 2.646465, -124.2258, 1, 1, 0, 1,
6.707071, 2.646465, -124.3749, 1, 0.8941177, 0, 1,
6.747475, 2.646465, -124.5356, 1, 0.8941177, 0, 1,
6.787879, 2.646465, -124.708, 1, 0.8941177, 0, 1,
6.828283, 2.646465, -124.892, 1, 0.8941177, 0, 1,
6.868687, 2.646465, -125.0877, 1, 0.8941177, 0, 1,
6.909091, 2.646465, -125.295, 1, 0.8941177, 0, 1,
6.949495, 2.646465, -125.514, 1, 0.8941177, 0, 1,
6.989899, 2.646465, -125.7447, 1, 0.7921569, 0, 1,
7.030303, 2.646465, -125.987, 1, 0.7921569, 0, 1,
7.070707, 2.646465, -126.241, 1, 0.7921569, 0, 1,
7.111111, 2.646465, -126.5066, 1, 0.7921569, 0, 1,
7.151515, 2.646465, -126.7838, 1, 0.7921569, 0, 1,
7.191919, 2.646465, -127.0728, 1, 0.6862745, 0, 1,
7.232323, 2.646465, -127.3733, 1, 0.6862745, 0, 1,
7.272727, 2.646465, -127.6856, 1, 0.6862745, 0, 1,
7.313131, 2.646465, -128.0095, 1, 0.6862745, 0, 1,
7.353535, 2.646465, -128.345, 1, 0.5843138, 0, 1,
7.393939, 2.646465, -128.6922, 1, 0.5843138, 0, 1,
7.434343, 2.646465, -129.0511, 1, 0.5843138, 0, 1,
7.474748, 2.646465, -129.4216, 1, 0.4823529, 0, 1,
7.515152, 2.646465, -129.8037, 1, 0.4823529, 0, 1,
7.555555, 2.646465, -130.1975, 1, 0.4823529, 0, 1,
7.59596, 2.646465, -130.603, 1, 0.3764706, 0, 1,
7.636364, 2.646465, -131.0201, 1, 0.3764706, 0, 1,
7.676768, 2.646465, -131.4489, 1, 0.3764706, 0, 1,
7.717172, 2.646465, -131.8893, 1, 0.2745098, 0, 1,
7.757576, 2.646465, -132.3414, 1, 0.2745098, 0, 1,
7.79798, 2.646465, -132.8051, 1, 0.2745098, 0, 1,
7.838384, 2.646465, -133.2805, 1, 0.1686275, 0, 1,
7.878788, 2.646465, -133.7676, 1, 0.1686275, 0, 1,
7.919192, 2.646465, -134.2663, 1, 0.1686275, 0, 1,
7.959596, 2.646465, -134.7766, 1, 0.06666667, 0, 1,
8, 2.646465, -135.2986, 1, 0.06666667, 0, 1,
4, 2.686869, -139.5529, 0.654902, 0, 0.3411765, 1,
4.040404, 2.686869, -138.9513, 0.7568628, 0, 0.2392157, 1,
4.080808, 2.686869, -138.361, 0.7568628, 0, 0.2392157, 1,
4.121212, 2.686869, -137.782, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.686869, -137.2143, 0.8588235, 0, 0.1372549, 1,
4.20202, 2.686869, -136.6579, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.686869, -136.1128, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.686869, -135.5791, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.686869, -135.0566, 1, 0.06666667, 0, 1,
4.363636, 2.686869, -134.5455, 1, 0.06666667, 0, 1,
4.40404, 2.686869, -134.0456, 1, 0.1686275, 0, 1,
4.444445, 2.686869, -133.5571, 1, 0.1686275, 0, 1,
4.484848, 2.686869, -133.0798, 1, 0.1686275, 0, 1,
4.525252, 2.686869, -132.6139, 1, 0.2745098, 0, 1,
4.565657, 2.686869, -132.1593, 1, 0.2745098, 0, 1,
4.606061, 2.686869, -131.716, 1, 0.3764706, 0, 1,
4.646465, 2.686869, -131.284, 1, 0.3764706, 0, 1,
4.686869, 2.686869, -130.8633, 1, 0.3764706, 0, 1,
4.727273, 2.686869, -130.4539, 1, 0.4823529, 0, 1,
4.767677, 2.686869, -130.0558, 1, 0.4823529, 0, 1,
4.808081, 2.686869, -129.669, 1, 0.4823529, 0, 1,
4.848485, 2.686869, -129.2935, 1, 0.5843138, 0, 1,
4.888889, 2.686869, -128.9294, 1, 0.5843138, 0, 1,
4.929293, 2.686869, -128.5765, 1, 0.5843138, 0, 1,
4.969697, 2.686869, -128.2349, 1, 0.5843138, 0, 1,
5.010101, 2.686869, -127.9047, 1, 0.6862745, 0, 1,
5.050505, 2.686869, -127.5858, 1, 0.6862745, 0, 1,
5.090909, 2.686869, -127.2781, 1, 0.6862745, 0, 1,
5.131313, 2.686869, -126.9818, 1, 0.6862745, 0, 1,
5.171717, 2.686869, -126.6968, 1, 0.7921569, 0, 1,
5.212121, 2.686869, -126.423, 1, 0.7921569, 0, 1,
5.252525, 2.686869, -126.1606, 1, 0.7921569, 0, 1,
5.292929, 2.686869, -125.9095, 1, 0.7921569, 0, 1,
5.333333, 2.686869, -125.6697, 1, 0.7921569, 0, 1,
5.373737, 2.686869, -125.4412, 1, 0.8941177, 0, 1,
5.414141, 2.686869, -125.2241, 1, 0.8941177, 0, 1,
5.454545, 2.686869, -125.0182, 1, 0.8941177, 0, 1,
5.494949, 2.686869, -124.8236, 1, 0.8941177, 0, 1,
5.535354, 2.686869, -124.6403, 1, 0.8941177, 0, 1,
5.575758, 2.686869, -124.4684, 1, 0.8941177, 0, 1,
5.616162, 2.686869, -124.3077, 1, 1, 0, 1,
5.656566, 2.686869, -124.1584, 1, 1, 0, 1,
5.69697, 2.686869, -124.0203, 1, 1, 0, 1,
5.737374, 2.686869, -123.8936, 1, 1, 0, 1,
5.777778, 2.686869, -123.7782, 1, 1, 0, 1,
5.818182, 2.686869, -123.6741, 1, 1, 0, 1,
5.858586, 2.686869, -123.5813, 1, 1, 0, 1,
5.89899, 2.686869, -123.4997, 1, 1, 0, 1,
5.939394, 2.686869, -123.4295, 1, 1, 0, 1,
5.979798, 2.686869, -123.3706, 1, 1, 0, 1,
6.020202, 2.686869, -123.3231, 1, 1, 0, 1,
6.060606, 2.686869, -123.2868, 1, 1, 0, 1,
6.10101, 2.686869, -123.2618, 1, 1, 0, 1,
6.141414, 2.686869, -123.2481, 1, 1, 0, 1,
6.181818, 2.686869, -123.2458, 1, 1, 0, 1,
6.222222, 2.686869, -123.2547, 1, 1, 0, 1,
6.262626, 2.686869, -123.275, 1, 1, 0, 1,
6.30303, 2.686869, -123.3065, 1, 1, 0, 1,
6.343434, 2.686869, -123.3494, 1, 1, 0, 1,
6.383838, 2.686869, -123.4036, 1, 1, 0, 1,
6.424242, 2.686869, -123.469, 1, 1, 0, 1,
6.464646, 2.686869, -123.5458, 1, 1, 0, 1,
6.505051, 2.686869, -123.6339, 1, 1, 0, 1,
6.545455, 2.686869, -123.7333, 1, 1, 0, 1,
6.585859, 2.686869, -123.844, 1, 1, 0, 1,
6.626263, 2.686869, -123.966, 1, 1, 0, 1,
6.666667, 2.686869, -124.0993, 1, 1, 0, 1,
6.707071, 2.686869, -124.2439, 1, 1, 0, 1,
6.747475, 2.686869, -124.3999, 1, 0.8941177, 0, 1,
6.787879, 2.686869, -124.5671, 1, 0.8941177, 0, 1,
6.828283, 2.686869, -124.7457, 1, 0.8941177, 0, 1,
6.868687, 2.686869, -124.9355, 1, 0.8941177, 0, 1,
6.909091, 2.686869, -125.1367, 1, 0.8941177, 0, 1,
6.949495, 2.686869, -125.3491, 1, 0.8941177, 0, 1,
6.989899, 2.686869, -125.5729, 1, 0.8941177, 0, 1,
7.030303, 2.686869, -125.808, 1, 0.7921569, 0, 1,
7.070707, 2.686869, -126.0543, 1, 0.7921569, 0, 1,
7.111111, 2.686869, -126.312, 1, 0.7921569, 0, 1,
7.151515, 2.686869, -126.581, 1, 0.7921569, 0, 1,
7.191919, 2.686869, -126.8613, 1, 0.6862745, 0, 1,
7.232323, 2.686869, -127.1529, 1, 0.6862745, 0, 1,
7.272727, 2.686869, -127.4558, 1, 0.6862745, 0, 1,
7.313131, 2.686869, -127.7701, 1, 0.6862745, 0, 1,
7.353535, 2.686869, -128.0956, 1, 0.5843138, 0, 1,
7.393939, 2.686869, -128.4324, 1, 0.5843138, 0, 1,
7.434343, 2.686869, -128.7805, 1, 0.5843138, 0, 1,
7.474748, 2.686869, -129.14, 1, 0.5843138, 0, 1,
7.515152, 2.686869, -129.5107, 1, 0.4823529, 0, 1,
7.555555, 2.686869, -129.8928, 1, 0.4823529, 0, 1,
7.59596, 2.686869, -130.2862, 1, 0.4823529, 0, 1,
7.636364, 2.686869, -130.6908, 1, 0.3764706, 0, 1,
7.676768, 2.686869, -131.1068, 1, 0.3764706, 0, 1,
7.717172, 2.686869, -131.5341, 1, 0.3764706, 0, 1,
7.757576, 2.686869, -131.9727, 1, 0.2745098, 0, 1,
7.79798, 2.686869, -132.4226, 1, 0.2745098, 0, 1,
7.838384, 2.686869, -132.8838, 1, 0.2745098, 0, 1,
7.878788, 2.686869, -133.3563, 1, 0.1686275, 0, 1,
7.919192, 2.686869, -133.8401, 1, 0.1686275, 0, 1,
7.959596, 2.686869, -134.3352, 1, 0.06666667, 0, 1,
8, 2.686869, -134.8417, 1, 0.06666667, 0, 1,
4, 2.727273, -138.9996, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.727273, -138.4157, 0.7568628, 0, 0.2392157, 1,
4.080808, 2.727273, -137.8428, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.727273, -137.2808, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.727273, -136.7298, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.727273, -136.1898, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.727273, -135.6608, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.727273, -135.1427, 1, 0.06666667, 0, 1,
4.323232, 2.727273, -134.6356, 1, 0.06666667, 0, 1,
4.363636, 2.727273, -134.1395, 1, 0.1686275, 0, 1,
4.40404, 2.727273, -133.6543, 1, 0.1686275, 0, 1,
4.444445, 2.727273, -133.1801, 1, 0.1686275, 0, 1,
4.484848, 2.727273, -132.7169, 1, 0.2745098, 0, 1,
4.525252, 2.727273, -132.2647, 1, 0.2745098, 0, 1,
4.565657, 2.727273, -131.8235, 1, 0.2745098, 0, 1,
4.606061, 2.727273, -131.3932, 1, 0.3764706, 0, 1,
4.646465, 2.727273, -130.9739, 1, 0.3764706, 0, 1,
4.686869, 2.727273, -130.5656, 1, 0.3764706, 0, 1,
4.727273, 2.727273, -130.1682, 1, 0.4823529, 0, 1,
4.767677, 2.727273, -129.7818, 1, 0.4823529, 0, 1,
4.808081, 2.727273, -129.4064, 1, 0.4823529, 0, 1,
4.848485, 2.727273, -129.042, 1, 0.5843138, 0, 1,
4.888889, 2.727273, -128.6885, 1, 0.5843138, 0, 1,
4.929293, 2.727273, -128.3461, 1, 0.5843138, 0, 1,
4.969697, 2.727273, -128.0145, 1, 0.6862745, 0, 1,
5.010101, 2.727273, -127.694, 1, 0.6862745, 0, 1,
5.050505, 2.727273, -127.3844, 1, 0.6862745, 0, 1,
5.090909, 2.727273, -127.0859, 1, 0.6862745, 0, 1,
5.131313, 2.727273, -126.7982, 1, 0.7921569, 0, 1,
5.171717, 2.727273, -126.5216, 1, 0.7921569, 0, 1,
5.212121, 2.727273, -126.2559, 1, 0.7921569, 0, 1,
5.252525, 2.727273, -126.0012, 1, 0.7921569, 0, 1,
5.292929, 2.727273, -125.7575, 1, 0.7921569, 0, 1,
5.333333, 2.727273, -125.5248, 1, 0.8941177, 0, 1,
5.373737, 2.727273, -125.303, 1, 0.8941177, 0, 1,
5.414141, 2.727273, -125.0922, 1, 0.8941177, 0, 1,
5.454545, 2.727273, -124.8924, 1, 0.8941177, 0, 1,
5.494949, 2.727273, -124.7035, 1, 0.8941177, 0, 1,
5.535354, 2.727273, -124.5257, 1, 0.8941177, 0, 1,
5.575758, 2.727273, -124.3587, 1, 0.8941177, 0, 1,
5.616162, 2.727273, -124.2028, 1, 1, 0, 1,
5.656566, 2.727273, -124.0579, 1, 1, 0, 1,
5.69697, 2.727273, -123.9239, 1, 1, 0, 1,
5.737374, 2.727273, -123.8009, 1, 1, 0, 1,
5.777778, 2.727273, -123.6889, 1, 1, 0, 1,
5.818182, 2.727273, -123.5878, 1, 1, 0, 1,
5.858586, 2.727273, -123.4977, 1, 1, 0, 1,
5.89899, 2.727273, -123.4186, 1, 1, 0, 1,
5.939394, 2.727273, -123.3505, 1, 1, 0, 1,
5.979798, 2.727273, -123.2933, 1, 1, 0, 1,
6.020202, 2.727273, -123.2471, 1, 1, 0, 1,
6.060606, 2.727273, -123.2119, 1, 1, 0, 1,
6.10101, 2.727273, -123.1877, 1, 1, 0, 1,
6.141414, 2.727273, -123.1744, 1, 1, 0, 1,
6.181818, 2.727273, -123.1721, 1, 1, 0, 1,
6.222222, 2.727273, -123.1808, 1, 1, 0, 1,
6.262626, 2.727273, -123.2004, 1, 1, 0, 1,
6.30303, 2.727273, -123.2311, 1, 1, 0, 1,
6.343434, 2.727273, -123.2727, 1, 1, 0, 1,
6.383838, 2.727273, -123.3252, 1, 1, 0, 1,
6.424242, 2.727273, -123.3888, 1, 1, 0, 1,
6.464646, 2.727273, -123.4633, 1, 1, 0, 1,
6.505051, 2.727273, -123.5488, 1, 1, 0, 1,
6.545455, 2.727273, -123.6453, 1, 1, 0, 1,
6.585859, 2.727273, -123.7527, 1, 1, 0, 1,
6.626263, 2.727273, -123.8712, 1, 1, 0, 1,
6.666667, 2.727273, -124.0006, 1, 1, 0, 1,
6.707071, 2.727273, -124.1409, 1, 1, 0, 1,
6.747475, 2.727273, -124.2923, 1, 1, 0, 1,
6.787879, 2.727273, -124.4546, 1, 0.8941177, 0, 1,
6.828283, 2.727273, -124.6279, 1, 0.8941177, 0, 1,
6.868687, 2.727273, -124.8121, 1, 0.8941177, 0, 1,
6.909091, 2.727273, -125.0074, 1, 0.8941177, 0, 1,
6.949495, 2.727273, -125.2136, 1, 0.8941177, 0, 1,
6.989899, 2.727273, -125.4308, 1, 0.8941177, 0, 1,
7.030303, 2.727273, -125.6589, 1, 0.7921569, 0, 1,
7.070707, 2.727273, -125.8981, 1, 0.7921569, 0, 1,
7.111111, 2.727273, -126.1482, 1, 0.7921569, 0, 1,
7.151515, 2.727273, -126.4093, 1, 0.7921569, 0, 1,
7.191919, 2.727273, -126.6813, 1, 0.7921569, 0, 1,
7.232323, 2.727273, -126.9643, 1, 0.6862745, 0, 1,
7.272727, 2.727273, -127.2583, 1, 0.6862745, 0, 1,
7.313131, 2.727273, -127.5633, 1, 0.6862745, 0, 1,
7.353535, 2.727273, -127.8793, 1, 0.6862745, 0, 1,
7.393939, 2.727273, -128.2062, 1, 0.5843138, 0, 1,
7.434343, 2.727273, -128.5441, 1, 0.5843138, 0, 1,
7.474748, 2.727273, -128.893, 1, 0.5843138, 0, 1,
7.515152, 2.727273, -129.2528, 1, 0.5843138, 0, 1,
7.555555, 2.727273, -129.6236, 1, 0.4823529, 0, 1,
7.59596, 2.727273, -130.0054, 1, 0.4823529, 0, 1,
7.636364, 2.727273, -130.3982, 1, 0.4823529, 0, 1,
7.676768, 2.727273, -130.802, 1, 0.3764706, 0, 1,
7.717172, 2.727273, -131.2167, 1, 0.3764706, 0, 1,
7.757576, 2.727273, -131.6424, 1, 0.3764706, 0, 1,
7.79798, 2.727273, -132.079, 1, 0.2745098, 0, 1,
7.838384, 2.727273, -132.5267, 1, 0.2745098, 0, 1,
7.878788, 2.727273, -132.9853, 1, 0.2745098, 0, 1,
7.919192, 2.727273, -133.4549, 1, 0.1686275, 0, 1,
7.959596, 2.727273, -133.9354, 1, 0.1686275, 0, 1,
8, 2.727273, -134.427, 1, 0.06666667, 0, 1,
4, 2.767677, -138.4919, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.767677, -137.9249, 0.8588235, 0, 0.1372549, 1,
4.080808, 2.767677, -137.3685, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.767677, -136.8229, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.767677, -136.2878, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.767677, -135.7635, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.767677, -135.2498, 1, 0.06666667, 0, 1,
4.282828, 2.767677, -134.7467, 1, 0.06666667, 0, 1,
4.323232, 2.767677, -134.2543, 1, 0.1686275, 0, 1,
4.363636, 2.767677, -133.7726, 1, 0.1686275, 0, 1,
4.40404, 2.767677, -133.3015, 1, 0.1686275, 0, 1,
4.444445, 2.767677, -132.8411, 1, 0.2745098, 0, 1,
4.484848, 2.767677, -132.3913, 1, 0.2745098, 0, 1,
4.525252, 2.767677, -131.9522, 1, 0.2745098, 0, 1,
4.565657, 2.767677, -131.5237, 1, 0.3764706, 0, 1,
4.606061, 2.767677, -131.1059, 1, 0.3764706, 0, 1,
4.646465, 2.767677, -130.6988, 1, 0.3764706, 0, 1,
4.686869, 2.767677, -130.3023, 1, 0.4823529, 0, 1,
4.727273, 2.767677, -129.9164, 1, 0.4823529, 0, 1,
4.767677, 2.767677, -129.5413, 1, 0.4823529, 0, 1,
4.808081, 2.767677, -129.1767, 1, 0.5843138, 0, 1,
4.848485, 2.767677, -128.8229, 1, 0.5843138, 0, 1,
4.888889, 2.767677, -128.4796, 1, 0.5843138, 0, 1,
4.929293, 2.767677, -128.1471, 1, 0.5843138, 0, 1,
4.969697, 2.767677, -127.8252, 1, 0.6862745, 0, 1,
5.010101, 2.767677, -127.5139, 1, 0.6862745, 0, 1,
5.050505, 2.767677, -127.2133, 1, 0.6862745, 0, 1,
5.090909, 2.767677, -126.9234, 1, 0.6862745, 0, 1,
5.131313, 2.767677, -126.6441, 1, 0.7921569, 0, 1,
5.171717, 2.767677, -126.3755, 1, 0.7921569, 0, 1,
5.212121, 2.767677, -126.1175, 1, 0.7921569, 0, 1,
5.252525, 2.767677, -125.8702, 1, 0.7921569, 0, 1,
5.292929, 2.767677, -125.6336, 1, 0.7921569, 0, 1,
5.333333, 2.767677, -125.4076, 1, 0.8941177, 0, 1,
5.373737, 2.767677, -125.1922, 1, 0.8941177, 0, 1,
5.414141, 2.767677, -124.9875, 1, 0.8941177, 0, 1,
5.454545, 2.767677, -124.7935, 1, 0.8941177, 0, 1,
5.494949, 2.767677, -124.6101, 1, 0.8941177, 0, 1,
5.535354, 2.767677, -124.4374, 1, 0.8941177, 0, 1,
5.575758, 2.767677, -124.2754, 1, 1, 0, 1,
5.616162, 2.767677, -124.1239, 1, 1, 0, 1,
5.656566, 2.767677, -123.9832, 1, 1, 0, 1,
5.69697, 2.767677, -123.8531, 1, 1, 0, 1,
5.737374, 2.767677, -123.7337, 1, 1, 0, 1,
5.777778, 2.767677, -123.6249, 1, 1, 0, 1,
5.818182, 2.767677, -123.5267, 1, 1, 0, 1,
5.858586, 2.767677, -123.4393, 1, 1, 0, 1,
5.89899, 2.767677, -123.3625, 1, 1, 0, 1,
5.939394, 2.767677, -123.2963, 1, 1, 0, 1,
5.979798, 2.767677, -123.2408, 1, 1, 0, 1,
6.020202, 2.767677, -123.1959, 1, 1, 0, 1,
6.060606, 2.767677, -123.1617, 1, 1, 0, 1,
6.10101, 2.767677, -123.1382, 1, 1, 0, 1,
6.141414, 2.767677, -123.1253, 1, 1, 0, 1,
6.181818, 2.767677, -123.1231, 1, 1, 0, 1,
6.222222, 2.767677, -123.1315, 1, 1, 0, 1,
6.262626, 2.767677, -123.1506, 1, 1, 0, 1,
6.30303, 2.767677, -123.1804, 1, 1, 0, 1,
6.343434, 2.767677, -123.2207, 1, 1, 0, 1,
6.383838, 2.767677, -123.2718, 1, 1, 0, 1,
6.424242, 2.767677, -123.3335, 1, 1, 0, 1,
6.464646, 2.767677, -123.4059, 1, 1, 0, 1,
6.505051, 2.767677, -123.4889, 1, 1, 0, 1,
6.545455, 2.767677, -123.5826, 1, 1, 0, 1,
6.585859, 2.767677, -123.6869, 1, 1, 0, 1,
6.626263, 2.767677, -123.8019, 1, 1, 0, 1,
6.666667, 2.767677, -123.9275, 1, 1, 0, 1,
6.707071, 2.767677, -124.0638, 1, 1, 0, 1,
6.747475, 2.767677, -124.2108, 1, 1, 0, 1,
6.787879, 2.767677, -124.3684, 1, 0.8941177, 0, 1,
6.828283, 2.767677, -124.5367, 1, 0.8941177, 0, 1,
6.868687, 2.767677, -124.7156, 1, 0.8941177, 0, 1,
6.909091, 2.767677, -124.9052, 1, 0.8941177, 0, 1,
6.949495, 2.767677, -125.1054, 1, 0.8941177, 0, 1,
6.989899, 2.767677, -125.3163, 1, 0.8941177, 0, 1,
7.030303, 2.767677, -125.5379, 1, 0.8941177, 0, 1,
7.070707, 2.767677, -125.7701, 1, 0.7921569, 0, 1,
7.111111, 2.767677, -126.0129, 1, 0.7921569, 0, 1,
7.151515, 2.767677, -126.2664, 1, 0.7921569, 0, 1,
7.191919, 2.767677, -126.5306, 1, 0.7921569, 0, 1,
7.232323, 2.767677, -126.8054, 1, 0.7921569, 0, 1,
7.272727, 2.767677, -127.0909, 1, 0.6862745, 0, 1,
7.313131, 2.767677, -127.387, 1, 0.6862745, 0, 1,
7.353535, 2.767677, -127.6938, 1, 0.6862745, 0, 1,
7.393939, 2.767677, -128.0113, 1, 0.6862745, 0, 1,
7.434343, 2.767677, -128.3394, 1, 0.5843138, 0, 1,
7.474748, 2.767677, -128.6782, 1, 0.5843138, 0, 1,
7.515152, 2.767677, -129.0276, 1, 0.5843138, 0, 1,
7.555555, 2.767677, -129.3876, 1, 0.4823529, 0, 1,
7.59596, 2.767677, -129.7584, 1, 0.4823529, 0, 1,
7.636364, 2.767677, -130.1398, 1, 0.4823529, 0, 1,
7.676768, 2.767677, -130.5318, 1, 0.4823529, 0, 1,
7.717172, 2.767677, -130.9345, 1, 0.3764706, 0, 1,
7.757576, 2.767677, -131.3479, 1, 0.3764706, 0, 1,
7.79798, 2.767677, -131.7719, 1, 0.3764706, 0, 1,
7.838384, 2.767677, -132.2065, 1, 0.2745098, 0, 1,
7.878788, 2.767677, -132.6519, 1, 0.2745098, 0, 1,
7.919192, 2.767677, -133.1078, 1, 0.1686275, 0, 1,
7.959596, 2.767677, -133.5745, 1, 0.1686275, 0, 1,
8, 2.767677, -134.0517, 1, 0.1686275, 0, 1,
4, 2.808081, -138.0267, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.808081, -137.4759, 0.8588235, 0, 0.1372549, 1,
4.080808, 2.808081, -136.9355, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.808081, -136.4054, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.808081, -135.8857, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.808081, -135.3763, 1, 0.06666667, 0, 1,
4.242424, 2.808081, -134.8773, 1, 0.06666667, 0, 1,
4.282828, 2.808081, -134.3886, 1, 0.06666667, 0, 1,
4.323232, 2.808081, -133.9103, 1, 0.1686275, 0, 1,
4.363636, 2.808081, -133.4423, 1, 0.1686275, 0, 1,
4.40404, 2.808081, -132.9847, 1, 0.2745098, 0, 1,
4.444445, 2.808081, -132.5374, 1, 0.2745098, 0, 1,
4.484848, 2.808081, -132.1005, 1, 0.2745098, 0, 1,
4.525252, 2.808081, -131.6739, 1, 0.3764706, 0, 1,
4.565657, 2.808081, -131.2577, 1, 0.3764706, 0, 1,
4.606061, 2.808081, -130.8518, 1, 0.3764706, 0, 1,
4.646465, 2.808081, -130.4563, 1, 0.4823529, 0, 1,
4.686869, 2.808081, -130.0711, 1, 0.4823529, 0, 1,
4.727273, 2.808081, -129.6963, 1, 0.4823529, 0, 1,
4.767677, 2.808081, -129.3318, 1, 0.4823529, 0, 1,
4.808081, 2.808081, -128.9777, 1, 0.5843138, 0, 1,
4.848485, 2.808081, -128.634, 1, 0.5843138, 0, 1,
4.888889, 2.808081, -128.3006, 1, 0.5843138, 0, 1,
4.929293, 2.808081, -127.9775, 1, 0.6862745, 0, 1,
4.969697, 2.808081, -127.6648, 1, 0.6862745, 0, 1,
5.010101, 2.808081, -127.3624, 1, 0.6862745, 0, 1,
5.050505, 2.808081, -127.0704, 1, 0.6862745, 0, 1,
5.090909, 2.808081, -126.7888, 1, 0.7921569, 0, 1,
5.131313, 2.808081, -126.5175, 1, 0.7921569, 0, 1,
5.171717, 2.808081, -126.2565, 1, 0.7921569, 0, 1,
5.212121, 2.808081, -126.006, 1, 0.7921569, 0, 1,
5.252525, 2.808081, -125.7657, 1, 0.7921569, 0, 1,
5.292929, 2.808081, -125.5358, 1, 0.8941177, 0, 1,
5.333333, 2.808081, -125.3163, 1, 0.8941177, 0, 1,
5.373737, 2.808081, -125.1071, 1, 0.8941177, 0, 1,
5.414141, 2.808081, -124.9082, 1, 0.8941177, 0, 1,
5.454545, 2.808081, -124.7197, 1, 0.8941177, 0, 1,
5.494949, 2.808081, -124.5416, 1, 0.8941177, 0, 1,
5.535354, 2.808081, -124.3738, 1, 0.8941177, 0, 1,
5.575758, 2.808081, -124.2164, 1, 1, 0, 1,
5.616162, 2.808081, -124.0693, 1, 1, 0, 1,
5.656566, 2.808081, -123.9326, 1, 1, 0, 1,
5.69697, 2.808081, -123.8062, 1, 1, 0, 1,
5.737374, 2.808081, -123.6902, 1, 1, 0, 1,
5.777778, 2.808081, -123.5845, 1, 1, 0, 1,
5.818182, 2.808081, -123.4892, 1, 1, 0, 1,
5.858586, 2.808081, -123.4042, 1, 1, 0, 1,
5.89899, 2.808081, -123.3296, 1, 1, 0, 1,
5.939394, 2.808081, -123.2653, 1, 1, 0, 1,
5.979798, 2.808081, -123.2114, 1, 1, 0, 1,
6.020202, 2.808081, -123.1678, 1, 1, 0, 1,
6.060606, 2.808081, -123.1346, 1, 1, 0, 1,
6.10101, 2.808081, -123.1117, 1, 1, 0, 1,
6.141414, 2.808081, -123.0992, 1, 1, 0, 1,
6.181818, 2.808081, -123.0971, 1, 1, 0, 1,
6.222222, 2.808081, -123.1052, 1, 1, 0, 1,
6.262626, 2.808081, -123.1238, 1, 1, 0, 1,
6.30303, 2.808081, -123.1527, 1, 1, 0, 1,
6.343434, 2.808081, -123.1919, 1, 1, 0, 1,
6.383838, 2.808081, -123.2415, 1, 1, 0, 1,
6.424242, 2.808081, -123.3015, 1, 1, 0, 1,
6.464646, 2.808081, -123.3717, 1, 1, 0, 1,
6.505051, 2.808081, -123.4524, 1, 1, 0, 1,
6.545455, 2.808081, -123.5434, 1, 1, 0, 1,
6.585859, 2.808081, -123.6448, 1, 1, 0, 1,
6.626263, 2.808081, -123.7565, 1, 1, 0, 1,
6.666667, 2.808081, -123.8785, 1, 1, 0, 1,
6.707071, 2.808081, -124.0109, 1, 1, 0, 1,
6.747475, 2.808081, -124.1537, 1, 1, 0, 1,
6.787879, 2.808081, -124.3068, 1, 1, 0, 1,
6.828283, 2.808081, -124.4702, 1, 0.8941177, 0, 1,
6.868687, 2.808081, -124.6441, 1, 0.8941177, 0, 1,
6.909091, 2.808081, -124.8282, 1, 0.8941177, 0, 1,
6.949495, 2.808081, -125.0227, 1, 0.8941177, 0, 1,
6.989899, 2.808081, -125.2276, 1, 0.8941177, 0, 1,
7.030303, 2.808081, -125.4428, 1, 0.8941177, 0, 1,
7.070707, 2.808081, -125.6684, 1, 0.7921569, 0, 1,
7.111111, 2.808081, -125.9043, 1, 0.7921569, 0, 1,
7.151515, 2.808081, -126.1506, 1, 0.7921569, 0, 1,
7.191919, 2.808081, -126.4072, 1, 0.7921569, 0, 1,
7.232323, 2.808081, -126.6742, 1, 0.7921569, 0, 1,
7.272727, 2.808081, -126.9515, 1, 0.6862745, 0, 1,
7.313131, 2.808081, -127.2392, 1, 0.6862745, 0, 1,
7.353535, 2.808081, -127.5372, 1, 0.6862745, 0, 1,
7.393939, 2.808081, -127.8456, 1, 0.6862745, 0, 1,
7.434343, 2.808081, -128.1643, 1, 0.5843138, 0, 1,
7.474748, 2.808081, -128.4934, 1, 0.5843138, 0, 1,
7.515152, 2.808081, -128.8328, 1, 0.5843138, 0, 1,
7.555555, 2.808081, -129.1826, 1, 0.5843138, 0, 1,
7.59596, 2.808081, -129.5428, 1, 0.4823529, 0, 1,
7.636364, 2.808081, -129.9133, 1, 0.4823529, 0, 1,
7.676768, 2.808081, -130.2941, 1, 0.4823529, 0, 1,
7.717172, 2.808081, -130.6853, 1, 0.3764706, 0, 1,
7.757576, 2.808081, -131.0868, 1, 0.3764706, 0, 1,
7.79798, 2.808081, -131.4987, 1, 0.3764706, 0, 1,
7.838384, 2.808081, -131.921, 1, 0.2745098, 0, 1,
7.878788, 2.808081, -132.3536, 1, 0.2745098, 0, 1,
7.919192, 2.808081, -132.7965, 1, 0.2745098, 0, 1,
7.959596, 2.808081, -133.2498, 1, 0.1686275, 0, 1,
8, 2.808081, -133.7135, 1, 0.1686275, 0, 1,
4, 2.848485, -137.6015, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.848485, -137.0662, 0.8588235, 0, 0.1372549, 1,
4.080808, 2.848485, -136.541, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.848485, -136.0259, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.848485, -135.5208, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.848485, -135.0257, 1, 0.06666667, 0, 1,
4.242424, 2.848485, -134.5408, 1, 0.06666667, 0, 1,
4.282828, 2.848485, -134.0659, 1, 0.1686275, 0, 1,
4.323232, 2.848485, -133.601, 1, 0.1686275, 0, 1,
4.363636, 2.848485, -133.1462, 1, 0.1686275, 0, 1,
4.40404, 2.848485, -132.7015, 1, 0.2745098, 0, 1,
4.444445, 2.848485, -132.2668, 1, 0.2745098, 0, 1,
4.484848, 2.848485, -131.8422, 1, 0.2745098, 0, 1,
4.525252, 2.848485, -131.4276, 1, 0.3764706, 0, 1,
4.565657, 2.848485, -131.0231, 1, 0.3764706, 0, 1,
4.606061, 2.848485, -130.6287, 1, 0.3764706, 0, 1,
4.646465, 2.848485, -130.2443, 1, 0.4823529, 0, 1,
4.686869, 2.848485, -129.87, 1, 0.4823529, 0, 1,
4.727273, 2.848485, -129.5057, 1, 0.4823529, 0, 1,
4.767677, 2.848485, -129.1515, 1, 0.5843138, 0, 1,
4.808081, 2.848485, -128.8074, 1, 0.5843138, 0, 1,
4.848485, 2.848485, -128.4733, 1, 0.5843138, 0, 1,
4.888889, 2.848485, -128.1493, 1, 0.5843138, 0, 1,
4.929293, 2.848485, -127.8354, 1, 0.6862745, 0, 1,
4.969697, 2.848485, -127.5315, 1, 0.6862745, 0, 1,
5.010101, 2.848485, -127.2376, 1, 0.6862745, 0, 1,
5.050505, 2.848485, -126.9538, 1, 0.6862745, 0, 1,
5.090909, 2.848485, -126.6801, 1, 0.7921569, 0, 1,
5.131313, 2.848485, -126.4165, 1, 0.7921569, 0, 1,
5.171717, 2.848485, -126.1629, 1, 0.7921569, 0, 1,
5.212121, 2.848485, -125.9193, 1, 0.7921569, 0, 1,
5.252525, 2.848485, -125.6859, 1, 0.7921569, 0, 1,
5.292929, 2.848485, -125.4624, 1, 0.8941177, 0, 1,
5.333333, 2.848485, -125.2491, 1, 0.8941177, 0, 1,
5.373737, 2.848485, -125.0458, 1, 0.8941177, 0, 1,
5.414141, 2.848485, -124.8525, 1, 0.8941177, 0, 1,
5.454545, 2.848485, -124.6694, 1, 0.8941177, 0, 1,
5.494949, 2.848485, -124.4962, 1, 0.8941177, 0, 1,
5.535354, 2.848485, -124.3332, 1, 1, 0, 1,
5.575758, 2.848485, -124.1802, 1, 1, 0, 1,
5.616162, 2.848485, -124.0372, 1, 1, 0, 1,
5.656566, 2.848485, -123.9044, 1, 1, 0, 1,
5.69697, 2.848485, -123.7815, 1, 1, 0, 1,
5.737374, 2.848485, -123.6688, 1, 1, 0, 1,
5.777778, 2.848485, -123.5661, 1, 1, 0, 1,
5.818182, 2.848485, -123.4734, 1, 1, 0, 1,
5.858586, 2.848485, -123.3909, 1, 1, 0, 1,
5.89899, 2.848485, -123.3183, 1, 1, 0, 1,
5.939394, 2.848485, -123.2559, 1, 1, 0, 1,
5.979798, 2.848485, -123.2035, 1, 1, 0, 1,
6.020202, 2.848485, -123.1611, 1, 1, 0, 1,
6.060606, 2.848485, -123.1289, 1, 1, 0, 1,
6.10101, 2.848485, -123.1066, 1, 1, 0, 1,
6.141414, 2.848485, -123.0945, 1, 1, 0, 1,
6.181818, 2.848485, -123.0924, 1, 1, 0, 1,
6.222222, 2.848485, -123.1003, 1, 1, 0, 1,
6.262626, 2.848485, -123.1183, 1, 1, 0, 1,
6.30303, 2.848485, -123.1464, 1, 1, 0, 1,
6.343434, 2.848485, -123.1846, 1, 1, 0, 1,
6.383838, 2.848485, -123.2328, 1, 1, 0, 1,
6.424242, 2.848485, -123.291, 1, 1, 0, 1,
6.464646, 2.848485, -123.3593, 1, 1, 0, 1,
6.505051, 2.848485, -123.4377, 1, 1, 0, 1,
6.545455, 2.848485, -123.5261, 1, 1, 0, 1,
6.585859, 2.848485, -123.6246, 1, 1, 0, 1,
6.626263, 2.848485, -123.7332, 1, 1, 0, 1,
6.666667, 2.848485, -123.8518, 1, 1, 0, 1,
6.707071, 2.848485, -123.9805, 1, 1, 0, 1,
6.747475, 2.848485, -124.1192, 1, 1, 0, 1,
6.787879, 2.848485, -124.268, 1, 1, 0, 1,
6.828283, 2.848485, -124.4269, 1, 0.8941177, 0, 1,
6.868687, 2.848485, -124.5958, 1, 0.8941177, 0, 1,
6.909091, 2.848485, -124.7748, 1, 0.8941177, 0, 1,
6.949495, 2.848485, -124.9638, 1, 0.8941177, 0, 1,
6.989899, 2.848485, -125.1629, 1, 0.8941177, 0, 1,
7.030303, 2.848485, -125.3721, 1, 0.8941177, 0, 1,
7.070707, 2.848485, -125.5913, 1, 0.7921569, 0, 1,
7.111111, 2.848485, -125.8205, 1, 0.7921569, 0, 1,
7.151515, 2.848485, -126.0599, 1, 0.7921569, 0, 1,
7.191919, 2.848485, -126.3093, 1, 0.7921569, 0, 1,
7.232323, 2.848485, -126.5687, 1, 0.7921569, 0, 1,
7.272727, 2.848485, -126.8382, 1, 0.6862745, 0, 1,
7.313131, 2.848485, -127.1178, 1, 0.6862745, 0, 1,
7.353535, 2.848485, -127.4075, 1, 0.6862745, 0, 1,
7.393939, 2.848485, -127.7072, 1, 0.6862745, 0, 1,
7.434343, 2.848485, -128.0169, 1, 0.6862745, 0, 1,
7.474748, 2.848485, -128.3367, 1, 0.5843138, 0, 1,
7.515152, 2.848485, -128.6666, 1, 0.5843138, 0, 1,
7.555555, 2.848485, -129.0065, 1, 0.5843138, 0, 1,
7.59596, 2.848485, -129.3565, 1, 0.4823529, 0, 1,
7.636364, 2.848485, -129.7166, 1, 0.4823529, 0, 1,
7.676768, 2.848485, -130.0867, 1, 0.4823529, 0, 1,
7.717172, 2.848485, -130.4669, 1, 0.4823529, 0, 1,
7.757576, 2.848485, -130.8571, 1, 0.3764706, 0, 1,
7.79798, 2.848485, -131.2574, 1, 0.3764706, 0, 1,
7.838384, 2.848485, -131.6677, 1, 0.3764706, 0, 1,
7.878788, 2.848485, -132.0882, 1, 0.2745098, 0, 1,
7.919192, 2.848485, -132.5186, 1, 0.2745098, 0, 1,
7.959596, 2.848485, -132.9592, 1, 0.2745098, 0, 1,
8, 2.848485, -133.4097, 1, 0.1686275, 0, 1,
4, 2.888889, -137.2137, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.888889, -136.6933, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.888889, -136.1827, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.888889, -135.6818, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.888889, -135.1908, 1, 0.06666667, 0, 1,
4.20202, 2.888889, -134.7095, 1, 0.06666667, 0, 1,
4.242424, 2.888889, -134.238, 1, 0.1686275, 0, 1,
4.282828, 2.888889, -133.7763, 1, 0.1686275, 0, 1,
4.323232, 2.888889, -133.3243, 1, 0.1686275, 0, 1,
4.363636, 2.888889, -132.8822, 1, 0.2745098, 0, 1,
4.40404, 2.888889, -132.4498, 1, 0.2745098, 0, 1,
4.444445, 2.888889, -132.0272, 1, 0.2745098, 0, 1,
4.484848, 2.888889, -131.6143, 1, 0.3764706, 0, 1,
4.525252, 2.888889, -131.2113, 1, 0.3764706, 0, 1,
4.565657, 2.888889, -130.8181, 1, 0.3764706, 0, 1,
4.606061, 2.888889, -130.4346, 1, 0.4823529, 0, 1,
4.646465, 2.888889, -130.0609, 1, 0.4823529, 0, 1,
4.686869, 2.888889, -129.6969, 1, 0.4823529, 0, 1,
4.727273, 2.888889, -129.3428, 1, 0.4823529, 0, 1,
4.767677, 2.888889, -128.9985, 1, 0.5843138, 0, 1,
4.808081, 2.888889, -128.6639, 1, 0.5843138, 0, 1,
4.848485, 2.888889, -128.3391, 1, 0.5843138, 0, 1,
4.888889, 2.888889, -128.0241, 1, 0.6862745, 0, 1,
4.929293, 2.888889, -127.7188, 1, 0.6862745, 0, 1,
4.969697, 2.888889, -127.4234, 1, 0.6862745, 0, 1,
5.010101, 2.888889, -127.1377, 1, 0.6862745, 0, 1,
5.050505, 2.888889, -126.8618, 1, 0.6862745, 0, 1,
5.090909, 2.888889, -126.5957, 1, 0.7921569, 0, 1,
5.131313, 2.888889, -126.3393, 1, 0.7921569, 0, 1,
5.171717, 2.888889, -126.0928, 1, 0.7921569, 0, 1,
5.212121, 2.888889, -125.856, 1, 0.7921569, 0, 1,
5.252525, 2.888889, -125.629, 1, 0.7921569, 0, 1,
5.292929, 2.888889, -125.4118, 1, 0.8941177, 0, 1,
5.333333, 2.888889, -125.2044, 1, 0.8941177, 0, 1,
5.373737, 2.888889, -125.0067, 1, 0.8941177, 0, 1,
5.414141, 2.888889, -124.8189, 1, 0.8941177, 0, 1,
5.454545, 2.888889, -124.6408, 1, 0.8941177, 0, 1,
5.494949, 2.888889, -124.4725, 1, 0.8941177, 0, 1,
5.535354, 2.888889, -124.3139, 1, 1, 0, 1,
5.575758, 2.888889, -124.1652, 1, 1, 0, 1,
5.616162, 2.888889, -124.0262, 1, 1, 0, 1,
5.656566, 2.888889, -123.897, 1, 1, 0, 1,
5.69697, 2.888889, -123.7776, 1, 1, 0, 1,
5.737374, 2.888889, -123.668, 1, 1, 0, 1,
5.777778, 2.888889, -123.5681, 1, 1, 0, 1,
5.818182, 2.888889, -123.4781, 1, 1, 0, 1,
5.858586, 2.888889, -123.3978, 1, 1, 0, 1,
5.89899, 2.888889, -123.3273, 1, 1, 0, 1,
5.939394, 2.888889, -123.2665, 1, 1, 0, 1,
5.979798, 2.888889, -123.2156, 1, 1, 0, 1,
6.020202, 2.888889, -123.1744, 1, 1, 0, 1,
6.060606, 2.888889, -123.1431, 1, 1, 0, 1,
6.10101, 2.888889, -123.1215, 1, 1, 0, 1,
6.141414, 2.888889, -123.1096, 1, 1, 0, 1,
6.181818, 2.888889, -123.1076, 1, 1, 0, 1,
6.222222, 2.888889, -123.1153, 1, 1, 0, 1,
6.262626, 2.888889, -123.1328, 1, 1, 0, 1,
6.30303, 2.888889, -123.1601, 1, 1, 0, 1,
6.343434, 2.888889, -123.1972, 1, 1, 0, 1,
6.383838, 2.888889, -123.2441, 1, 1, 0, 1,
6.424242, 2.888889, -123.3007, 1, 1, 0, 1,
6.464646, 2.888889, -123.3671, 1, 1, 0, 1,
6.505051, 2.888889, -123.4433, 1, 1, 0, 1,
6.545455, 2.888889, -123.5293, 1, 1, 0, 1,
6.585859, 2.888889, -123.6251, 1, 1, 0, 1,
6.626263, 2.888889, -123.7306, 1, 1, 0, 1,
6.666667, 2.888889, -123.8459, 1, 1, 0, 1,
6.707071, 2.888889, -123.971, 1, 1, 0, 1,
6.747475, 2.888889, -124.1059, 1, 1, 0, 1,
6.787879, 2.888889, -124.2506, 1, 1, 0, 1,
6.828283, 2.888889, -124.405, 1, 0.8941177, 0, 1,
6.868687, 2.888889, -124.5693, 1, 0.8941177, 0, 1,
6.909091, 2.888889, -124.7433, 1, 0.8941177, 0, 1,
6.949495, 2.888889, -124.927, 1, 0.8941177, 0, 1,
6.989899, 2.888889, -125.1206, 1, 0.8941177, 0, 1,
7.030303, 2.888889, -125.324, 1, 0.8941177, 0, 1,
7.070707, 2.888889, -125.5371, 1, 0.8941177, 0, 1,
7.111111, 2.888889, -125.76, 1, 0.7921569, 0, 1,
7.151515, 2.888889, -125.9927, 1, 0.7921569, 0, 1,
7.191919, 2.888889, -126.2351, 1, 0.7921569, 0, 1,
7.232323, 2.888889, -126.4874, 1, 0.7921569, 0, 1,
7.272727, 2.888889, -126.7494, 1, 0.7921569, 0, 1,
7.313131, 2.888889, -127.0212, 1, 0.6862745, 0, 1,
7.353535, 2.888889, -127.3028, 1, 0.6862745, 0, 1,
7.393939, 2.888889, -127.5942, 1, 0.6862745, 0, 1,
7.434343, 2.888889, -127.8953, 1, 0.6862745, 0, 1,
7.474748, 2.888889, -128.2063, 1, 0.5843138, 0, 1,
7.515152, 2.888889, -128.527, 1, 0.5843138, 0, 1,
7.555555, 2.888889, -128.8575, 1, 0.5843138, 0, 1,
7.59596, 2.888889, -129.1977, 1, 0.5843138, 0, 1,
7.636364, 2.888889, -129.5478, 1, 0.4823529, 0, 1,
7.676768, 2.888889, -129.9076, 1, 0.4823529, 0, 1,
7.717172, 2.888889, -130.2772, 1, 0.4823529, 0, 1,
7.757576, 2.888889, -130.6566, 1, 0.3764706, 0, 1,
7.79798, 2.888889, -131.0458, 1, 0.3764706, 0, 1,
7.838384, 2.888889, -131.4448, 1, 0.3764706, 0, 1,
7.878788, 2.888889, -131.8535, 1, 0.2745098, 0, 1,
7.919192, 2.888889, -132.272, 1, 0.2745098, 0, 1,
7.959596, 2.888889, -132.7003, 1, 0.2745098, 0, 1,
8, 2.888889, -133.1384, 1, 0.1686275, 0, 1,
4, 2.929293, -136.861, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.929293, -136.3549, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.929293, -135.8582, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.929293, -135.3711, 1, 0.06666667, 0, 1,
4.161616, 2.929293, -134.8935, 1, 0.06666667, 0, 1,
4.20202, 2.929293, -134.4254, 1, 0.06666667, 0, 1,
4.242424, 2.929293, -133.9668, 1, 0.1686275, 0, 1,
4.282828, 2.929293, -133.5177, 1, 0.1686275, 0, 1,
4.323232, 2.929293, -133.0782, 1, 0.1686275, 0, 1,
4.363636, 2.929293, -132.6481, 1, 0.2745098, 0, 1,
4.40404, 2.929293, -132.2276, 1, 0.2745098, 0, 1,
4.444445, 2.929293, -131.8166, 1, 0.2745098, 0, 1,
4.484848, 2.929293, -131.415, 1, 0.3764706, 0, 1,
4.525252, 2.929293, -131.023, 1, 0.3764706, 0, 1,
4.565657, 2.929293, -130.6406, 1, 0.3764706, 0, 1,
4.606061, 2.929293, -130.2676, 1, 0.4823529, 0, 1,
4.646465, 2.929293, -129.9041, 1, 0.4823529, 0, 1,
4.686869, 2.929293, -129.5502, 1, 0.4823529, 0, 1,
4.727273, 2.929293, -129.2057, 1, 0.5843138, 0, 1,
4.767677, 2.929293, -128.8708, 1, 0.5843138, 0, 1,
4.808081, 2.929293, -128.5454, 1, 0.5843138, 0, 1,
4.848485, 2.929293, -128.2295, 1, 0.5843138, 0, 1,
4.888889, 2.929293, -127.9231, 1, 0.6862745, 0, 1,
4.929293, 2.929293, -127.6262, 1, 0.6862745, 0, 1,
4.969697, 2.929293, -127.3389, 1, 0.6862745, 0, 1,
5.010101, 2.929293, -127.061, 1, 0.6862745, 0, 1,
5.050505, 2.929293, -126.7927, 1, 0.7921569, 0, 1,
5.090909, 2.929293, -126.5339, 1, 0.7921569, 0, 1,
5.131313, 2.929293, -126.2846, 1, 0.7921569, 0, 1,
5.171717, 2.929293, -126.0448, 1, 0.7921569, 0, 1,
5.212121, 2.929293, -125.8145, 1, 0.7921569, 0, 1,
5.252525, 2.929293, -125.5937, 1, 0.7921569, 0, 1,
5.292929, 2.929293, -125.3824, 1, 0.8941177, 0, 1,
5.333333, 2.929293, -125.1807, 1, 0.8941177, 0, 1,
5.373737, 2.929293, -124.9884, 1, 0.8941177, 0, 1,
5.414141, 2.929293, -124.8057, 1, 0.8941177, 0, 1,
5.454545, 2.929293, -124.6325, 1, 0.8941177, 0, 1,
5.494949, 2.929293, -124.4688, 1, 0.8941177, 0, 1,
5.535354, 2.929293, -124.3146, 1, 1, 0, 1,
5.575758, 2.929293, -124.17, 1, 1, 0, 1,
5.616162, 2.929293, -124.0348, 1, 1, 0, 1,
5.656566, 2.929293, -123.9091, 1, 1, 0, 1,
5.69697, 2.929293, -123.793, 1, 1, 0, 1,
5.737374, 2.929293, -123.6864, 1, 1, 0, 1,
5.777778, 2.929293, -123.5893, 1, 1, 0, 1,
5.818182, 2.929293, -123.5017, 1, 1, 0, 1,
5.858586, 2.929293, -123.4236, 1, 1, 0, 1,
5.89899, 2.929293, -123.355, 1, 1, 0, 1,
5.939394, 2.929293, -123.2959, 1, 1, 0, 1,
5.979798, 2.929293, -123.2464, 1, 1, 0, 1,
6.020202, 2.929293, -123.2064, 1, 1, 0, 1,
6.060606, 2.929293, -123.1758, 1, 1, 0, 1,
6.10101, 2.929293, -123.1548, 1, 1, 0, 1,
6.141414, 2.929293, -123.1433, 1, 1, 0, 1,
6.181818, 2.929293, -123.1413, 1, 1, 0, 1,
6.222222, 2.929293, -123.1489, 1, 1, 0, 1,
6.262626, 2.929293, -123.1659, 1, 1, 0, 1,
6.30303, 2.929293, -123.1924, 1, 1, 0, 1,
6.343434, 2.929293, -123.2285, 1, 1, 0, 1,
6.383838, 2.929293, -123.2741, 1, 1, 0, 1,
6.424242, 2.929293, -123.3292, 1, 1, 0, 1,
6.464646, 2.929293, -123.3938, 1, 1, 0, 1,
6.505051, 2.929293, -123.4679, 1, 1, 0, 1,
6.545455, 2.929293, -123.5515, 1, 1, 0, 1,
6.585859, 2.929293, -123.6446, 1, 1, 0, 1,
6.626263, 2.929293, -123.7473, 1, 1, 0, 1,
6.666667, 2.929293, -123.8595, 1, 1, 0, 1,
6.707071, 2.929293, -123.9811, 1, 1, 0, 1,
6.747475, 2.929293, -124.1123, 1, 1, 0, 1,
6.787879, 2.929293, -124.253, 1, 1, 0, 1,
6.828283, 2.929293, -124.4032, 1, 0.8941177, 0, 1,
6.868687, 2.929293, -124.563, 1, 0.8941177, 0, 1,
6.909091, 2.929293, -124.7322, 1, 0.8941177, 0, 1,
6.949495, 2.929293, -124.9109, 1, 0.8941177, 0, 1,
6.989899, 2.929293, -125.0992, 1, 0.8941177, 0, 1,
7.030303, 2.929293, -125.297, 1, 0.8941177, 0, 1,
7.070707, 2.929293, -125.5043, 1, 0.8941177, 0, 1,
7.111111, 2.929293, -125.7211, 1, 0.7921569, 0, 1,
7.151515, 2.929293, -125.9474, 1, 0.7921569, 0, 1,
7.191919, 2.929293, -126.1832, 1, 0.7921569, 0, 1,
7.232323, 2.929293, -126.4285, 1, 0.7921569, 0, 1,
7.272727, 2.929293, -126.6834, 1, 0.7921569, 0, 1,
7.313131, 2.929293, -126.9478, 1, 0.6862745, 0, 1,
7.353535, 2.929293, -127.2216, 1, 0.6862745, 0, 1,
7.393939, 2.929293, -127.505, 1, 0.6862745, 0, 1,
7.434343, 2.929293, -127.7979, 1, 0.6862745, 0, 1,
7.474748, 2.929293, -128.1003, 1, 0.5843138, 0, 1,
7.515152, 2.929293, -128.4122, 1, 0.5843138, 0, 1,
7.555555, 2.929293, -128.7337, 1, 0.5843138, 0, 1,
7.59596, 2.929293, -129.0646, 1, 0.5843138, 0, 1,
7.636364, 2.929293, -129.4051, 1, 0.4823529, 0, 1,
7.676768, 2.929293, -129.7551, 1, 0.4823529, 0, 1,
7.717172, 2.929293, -130.1146, 1, 0.4823529, 0, 1,
7.757576, 2.929293, -130.4836, 1, 0.4823529, 0, 1,
7.79798, 2.929293, -130.8621, 1, 0.3764706, 0, 1,
7.838384, 2.929293, -131.2501, 1, 0.3764706, 0, 1,
7.878788, 2.929293, -131.6476, 1, 0.3764706, 0, 1,
7.919192, 2.929293, -132.0547, 1, 0.2745098, 0, 1,
7.959596, 2.929293, -132.4713, 1, 0.2745098, 0, 1,
8, 2.929293, -132.8973, 1, 0.2745098, 0, 1,
4, 2.969697, -136.5412, 0.9647059, 0, 0.03137255, 1,
4.040404, 2.969697, -136.0488, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.969697, -135.5656, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.969697, -135.0916, 1, 0.06666667, 0, 1,
4.161616, 2.969697, -134.6269, 1, 0.06666667, 0, 1,
4.20202, 2.969697, -134.1714, 1, 0.1686275, 0, 1,
4.242424, 2.969697, -133.7253, 1, 0.1686275, 0, 1,
4.282828, 2.969697, -133.2883, 1, 0.1686275, 0, 1,
4.323232, 2.969697, -132.8606, 1, 0.2745098, 0, 1,
4.363636, 2.969697, -132.4422, 1, 0.2745098, 0, 1,
4.40404, 2.969697, -132.033, 1, 0.2745098, 0, 1,
4.444445, 2.969697, -131.6331, 1, 0.3764706, 0, 1,
4.484848, 2.969697, -131.2425, 1, 0.3764706, 0, 1,
4.525252, 2.969697, -130.8611, 1, 0.3764706, 0, 1,
4.565657, 2.969697, -130.4889, 1, 0.4823529, 0, 1,
4.606061, 2.969697, -130.126, 1, 0.4823529, 0, 1,
4.646465, 2.969697, -129.7724, 1, 0.4823529, 0, 1,
4.686869, 2.969697, -129.428, 1, 0.4823529, 0, 1,
4.727273, 2.969697, -129.0929, 1, 0.5843138, 0, 1,
4.767677, 2.969697, -128.767, 1, 0.5843138, 0, 1,
4.808081, 2.969697, -128.4504, 1, 0.5843138, 0, 1,
4.848485, 2.969697, -128.143, 1, 0.5843138, 0, 1,
4.888889, 2.969697, -127.8449, 1, 0.6862745, 0, 1,
4.929293, 2.969697, -127.556, 1, 0.6862745, 0, 1,
4.969697, 2.969697, -127.2765, 1, 0.6862745, 0, 1,
5.010101, 2.969697, -127.0061, 1, 0.6862745, 0, 1,
5.050505, 2.969697, -126.745, 1, 0.7921569, 0, 1,
5.090909, 2.969697, -126.4932, 1, 0.7921569, 0, 1,
5.131313, 2.969697, -126.2506, 1, 0.7921569, 0, 1,
5.171717, 2.969697, -126.0173, 1, 0.7921569, 0, 1,
5.212121, 2.969697, -125.7932, 1, 0.7921569, 0, 1,
5.252525, 2.969697, -125.5784, 1, 0.7921569, 0, 1,
5.292929, 2.969697, -125.3729, 1, 0.8941177, 0, 1,
5.333333, 2.969697, -125.1766, 1, 0.8941177, 0, 1,
5.373737, 2.969697, -124.9895, 1, 0.8941177, 0, 1,
5.414141, 2.969697, -124.8118, 1, 0.8941177, 0, 1,
5.454545, 2.969697, -124.6432, 1, 0.8941177, 0, 1,
5.494949, 2.969697, -124.4839, 1, 0.8941177, 0, 1,
5.535354, 2.969697, -124.3339, 1, 1, 0, 1,
5.575758, 2.969697, -124.1932, 1, 1, 0, 1,
5.616162, 2.969697, -124.0617, 1, 1, 0, 1,
5.656566, 2.969697, -123.9394, 1, 1, 0, 1,
5.69697, 2.969697, -123.8264, 1, 1, 0, 1,
5.737374, 2.969697, -123.7227, 1, 1, 0, 1,
5.777778, 2.969697, -123.6282, 1, 1, 0, 1,
5.818182, 2.969697, -123.5429, 1, 1, 0, 1,
5.858586, 2.969697, -123.467, 1, 1, 0, 1,
5.89899, 2.969697, -123.4002, 1, 1, 0, 1,
5.939394, 2.969697, -123.3428, 1, 1, 0, 1,
5.979798, 2.969697, -123.2946, 1, 1, 0, 1,
6.020202, 2.969697, -123.2556, 1, 1, 0, 1,
6.060606, 2.969697, -123.2259, 1, 1, 0, 1,
6.10101, 2.969697, -123.2055, 1, 1, 0, 1,
6.141414, 2.969697, -123.1943, 1, 1, 0, 1,
6.181818, 2.969697, -123.1923, 1, 1, 0, 1,
6.222222, 2.969697, -123.1997, 1, 1, 0, 1,
6.262626, 2.969697, -123.2162, 1, 1, 0, 1,
6.30303, 2.969697, -123.2421, 1, 1, 0, 1,
6.343434, 2.969697, -123.2772, 1, 1, 0, 1,
6.383838, 2.969697, -123.3215, 1, 1, 0, 1,
6.424242, 2.969697, -123.3751, 1, 1, 0, 1,
6.464646, 2.969697, -123.438, 1, 1, 0, 1,
6.505051, 2.969697, -123.5101, 1, 1, 0, 1,
6.545455, 2.969697, -123.5914, 1, 1, 0, 1,
6.585859, 2.969697, -123.6821, 1, 1, 0, 1,
6.626263, 2.969697, -123.7819, 1, 1, 0, 1,
6.666667, 2.969697, -123.8911, 1, 1, 0, 1,
6.707071, 2.969697, -124.0095, 1, 1, 0, 1,
6.747475, 2.969697, -124.1371, 1, 1, 0, 1,
6.787879, 2.969697, -124.274, 1, 1, 0, 1,
6.828283, 2.969697, -124.4201, 1, 0.8941177, 0, 1,
6.868687, 2.969697, -124.5756, 1, 0.8941177, 0, 1,
6.909091, 2.969697, -124.7402, 1, 0.8941177, 0, 1,
6.949495, 2.969697, -124.9141, 1, 0.8941177, 0, 1,
6.989899, 2.969697, -125.0973, 1, 0.8941177, 0, 1,
7.030303, 2.969697, -125.2897, 1, 0.8941177, 0, 1,
7.070707, 2.969697, -125.4914, 1, 0.8941177, 0, 1,
7.111111, 2.969697, -125.7024, 1, 0.7921569, 0, 1,
7.151515, 2.969697, -125.9226, 1, 0.7921569, 0, 1,
7.191919, 2.969697, -126.152, 1, 0.7921569, 0, 1,
7.232323, 2.969697, -126.3907, 1, 0.7921569, 0, 1,
7.272727, 2.969697, -126.6387, 1, 0.7921569, 0, 1,
7.313131, 2.969697, -126.8959, 1, 0.6862745, 0, 1,
7.353535, 2.969697, -127.1624, 1, 0.6862745, 0, 1,
7.393939, 2.969697, -127.4381, 1, 0.6862745, 0, 1,
7.434343, 2.969697, -127.7231, 1, 0.6862745, 0, 1,
7.474748, 2.969697, -128.0173, 1, 0.6862745, 0, 1,
7.515152, 2.969697, -128.3208, 1, 0.5843138, 0, 1,
7.555555, 2.969697, -128.6336, 1, 0.5843138, 0, 1,
7.59596, 2.969697, -128.9556, 1, 0.5843138, 0, 1,
7.636364, 2.969697, -129.2868, 1, 0.5843138, 0, 1,
7.676768, 2.969697, -129.6273, 1, 0.4823529, 0, 1,
7.717172, 2.969697, -129.9771, 1, 0.4823529, 0, 1,
7.757576, 2.969697, -130.3362, 1, 0.4823529, 0, 1,
7.79798, 2.969697, -130.7044, 1, 0.3764706, 0, 1,
7.838384, 2.969697, -131.082, 1, 0.3764706, 0, 1,
7.878788, 2.969697, -131.4688, 1, 0.3764706, 0, 1,
7.919192, 2.969697, -131.8648, 1, 0.2745098, 0, 1,
7.959596, 2.969697, -132.2701, 1, 0.2745098, 0, 1,
8, 2.969697, -132.6847, 1, 0.2745098, 0, 1,
4, 3.010101, -136.2524, 0.9647059, 0, 0.03137255, 1,
4.040404, 3.010101, -135.7731, 0.9647059, 0, 0.03137255, 1,
4.080808, 3.010101, -135.3027, 1, 0.06666667, 0, 1,
4.121212, 3.010101, -134.8414, 1, 0.06666667, 0, 1,
4.161616, 3.010101, -134.3891, 1, 0.06666667, 0, 1,
4.20202, 3.010101, -133.9458, 1, 0.1686275, 0, 1,
4.242424, 3.010101, -133.5115, 1, 0.1686275, 0, 1,
4.282828, 3.010101, -133.0862, 1, 0.1686275, 0, 1,
4.323232, 3.010101, -132.6699, 1, 0.2745098, 0, 1,
4.363636, 3.010101, -132.2627, 1, 0.2745098, 0, 1,
4.40404, 3.010101, -131.8644, 1, 0.2745098, 0, 1,
4.444445, 3.010101, -131.4752, 1, 0.3764706, 0, 1,
4.484848, 3.010101, -131.0949, 1, 0.3764706, 0, 1,
4.525252, 3.010101, -130.7237, 1, 0.3764706, 0, 1,
4.565657, 3.010101, -130.3615, 1, 0.4823529, 0, 1,
4.606061, 3.010101, -130.0082, 1, 0.4823529, 0, 1,
4.646465, 3.010101, -129.664, 1, 0.4823529, 0, 1,
4.686869, 3.010101, -129.3288, 1, 0.4823529, 0, 1,
4.727273, 3.010101, -129.0026, 1, 0.5843138, 0, 1,
4.767677, 3.010101, -128.6855, 1, 0.5843138, 0, 1,
4.808081, 3.010101, -128.3773, 1, 0.5843138, 0, 1,
4.848485, 3.010101, -128.0781, 1, 0.5843138, 0, 1,
4.888889, 3.010101, -127.7879, 1, 0.6862745, 0, 1,
4.929293, 3.010101, -127.5068, 1, 0.6862745, 0, 1,
4.969697, 3.010101, -127.2347, 1, 0.6862745, 0, 1,
5.010101, 3.010101, -126.9715, 1, 0.6862745, 0, 1,
5.050505, 3.010101, -126.7174, 1, 0.7921569, 0, 1,
5.090909, 3.010101, -126.4723, 1, 0.7921569, 0, 1,
5.131313, 3.010101, -126.2362, 1, 0.7921569, 0, 1,
5.171717, 3.010101, -126.0091, 1, 0.7921569, 0, 1,
5.212121, 3.010101, -125.791, 1, 0.7921569, 0, 1,
5.252525, 3.010101, -125.5819, 1, 0.7921569, 0, 1,
5.292929, 3.010101, -125.3819, 1, 0.8941177, 0, 1,
5.333333, 3.010101, -125.1908, 1, 0.8941177, 0, 1,
5.373737, 3.010101, -125.0087, 1, 0.8941177, 0, 1,
5.414141, 3.010101, -124.8357, 1, 0.8941177, 0, 1,
5.454545, 3.010101, -124.6717, 1, 0.8941177, 0, 1,
5.494949, 3.010101, -124.5166, 1, 0.8941177, 0, 1,
5.535354, 3.010101, -124.3706, 1, 0.8941177, 0, 1,
5.575758, 3.010101, -124.2336, 1, 1, 0, 1,
5.616162, 3.010101, -124.1056, 1, 1, 0, 1,
5.656566, 3.010101, -123.9866, 1, 1, 0, 1,
5.69697, 3.010101, -123.8766, 1, 1, 0, 1,
5.737374, 3.010101, -123.7756, 1, 1, 0, 1,
5.777778, 3.010101, -123.6837, 1, 1, 0, 1,
5.818182, 3.010101, -123.6007, 1, 1, 0, 1,
5.858586, 3.010101, -123.5268, 1, 1, 0, 1,
5.89899, 3.010101, -123.4618, 1, 1, 0, 1,
5.939394, 3.010101, -123.4059, 1, 1, 0, 1,
5.979798, 3.010101, -123.359, 1, 1, 0, 1,
6.020202, 3.010101, -123.321, 1, 1, 0, 1,
6.060606, 3.010101, -123.2921, 1, 1, 0, 1,
6.10101, 3.010101, -123.2722, 1, 1, 0, 1,
6.141414, 3.010101, -123.2613, 1, 1, 0, 1,
6.181818, 3.010101, -123.2595, 1, 1, 0, 1,
6.222222, 3.010101, -123.2666, 1, 1, 0, 1,
6.262626, 3.010101, -123.2827, 1, 1, 0, 1,
6.30303, 3.010101, -123.3079, 1, 1, 0, 1,
6.343434, 3.010101, -123.342, 1, 1, 0, 1,
6.383838, 3.010101, -123.3852, 1, 1, 0, 1,
6.424242, 3.010101, -123.4373, 1, 1, 0, 1,
6.464646, 3.010101, -123.4985, 1, 1, 0, 1,
6.505051, 3.010101, -123.5687, 1, 1, 0, 1,
6.545455, 3.010101, -123.6479, 1, 1, 0, 1,
6.585859, 3.010101, -123.7361, 1, 1, 0, 1,
6.626263, 3.010101, -123.8333, 1, 1, 0, 1,
6.666667, 3.010101, -123.9395, 1, 1, 0, 1,
6.707071, 3.010101, -124.0548, 1, 1, 0, 1,
6.747475, 3.010101, -124.179, 1, 1, 0, 1,
6.787879, 3.010101, -124.3123, 1, 1, 0, 1,
6.828283, 3.010101, -124.4545, 1, 0.8941177, 0, 1,
6.868687, 3.010101, -124.6058, 1, 0.8941177, 0, 1,
6.909091, 3.010101, -124.7661, 1, 0.8941177, 0, 1,
6.949495, 3.010101, -124.9353, 1, 0.8941177, 0, 1,
6.989899, 3.010101, -125.1136, 1, 0.8941177, 0, 1,
7.030303, 3.010101, -125.3009, 1, 0.8941177, 0, 1,
7.070707, 3.010101, -125.4972, 1, 0.8941177, 0, 1,
7.111111, 3.010101, -125.7025, 1, 0.7921569, 0, 1,
7.151515, 3.010101, -125.9169, 1, 0.7921569, 0, 1,
7.191919, 3.010101, -126.1402, 1, 0.7921569, 0, 1,
7.232323, 3.010101, -126.3725, 1, 0.7921569, 0, 1,
7.272727, 3.010101, -126.6139, 1, 0.7921569, 0, 1,
7.313131, 3.010101, -126.8643, 1, 0.6862745, 0, 1,
7.353535, 3.010101, -127.1236, 1, 0.6862745, 0, 1,
7.393939, 3.010101, -127.392, 1, 0.6862745, 0, 1,
7.434343, 3.010101, -127.6694, 1, 0.6862745, 0, 1,
7.474748, 3.010101, -127.9558, 1, 0.6862745, 0, 1,
7.515152, 3.010101, -128.2512, 1, 0.5843138, 0, 1,
7.555555, 3.010101, -128.5556, 1, 0.5843138, 0, 1,
7.59596, 3.010101, -128.869, 1, 0.5843138, 0, 1,
7.636364, 3.010101, -129.1914, 1, 0.5843138, 0, 1,
7.676768, 3.010101, -129.5229, 1, 0.4823529, 0, 1,
7.717172, 3.010101, -129.8633, 1, 0.4823529, 0, 1,
7.757576, 3.010101, -130.2128, 1, 0.4823529, 0, 1,
7.79798, 3.010101, -130.5712, 1, 0.3764706, 0, 1,
7.838384, 3.010101, -130.9387, 1, 0.3764706, 0, 1,
7.878788, 3.010101, -131.3152, 1, 0.3764706, 0, 1,
7.919192, 3.010101, -131.7007, 1, 0.3764706, 0, 1,
7.959596, 3.010101, -132.0952, 1, 0.2745098, 0, 1,
8, 3.010101, -132.4987, 1, 0.2745098, 0, 1,
4, 3.050505, -135.9926, 0.9647059, 0, 0.03137255, 1,
4.040404, 3.050505, -135.5259, 0.9647059, 0, 0.03137255, 1,
4.080808, 3.050505, -135.0679, 1, 0.06666667, 0, 1,
4.121212, 3.050505, -134.6188, 1, 0.06666667, 0, 1,
4.161616, 3.050505, -134.1784, 1, 0.1686275, 0, 1,
4.20202, 3.050505, -133.7467, 1, 0.1686275, 0, 1,
4.242424, 3.050505, -133.3239, 1, 0.1686275, 0, 1,
4.282828, 3.050505, -132.9098, 1, 0.2745098, 0, 1,
4.323232, 3.050505, -132.5044, 1, 0.2745098, 0, 1,
4.363636, 3.050505, -132.1079, 1, 0.2745098, 0, 1,
4.40404, 3.050505, -131.7201, 1, 0.3764706, 0, 1,
4.444445, 3.050505, -131.3411, 1, 0.3764706, 0, 1,
4.484848, 3.050505, -130.9709, 1, 0.3764706, 0, 1,
4.525252, 3.050505, -130.6094, 1, 0.3764706, 0, 1,
4.565657, 3.050505, -130.2567, 1, 0.4823529, 0, 1,
4.606061, 3.050505, -129.9128, 1, 0.4823529, 0, 1,
4.646465, 3.050505, -129.5776, 1, 0.4823529, 0, 1,
4.686869, 3.050505, -129.2512, 1, 0.5843138, 0, 1,
4.727273, 3.050505, -128.9336, 1, 0.5843138, 0, 1,
4.767677, 3.050505, -128.6248, 1, 0.5843138, 0, 1,
4.808081, 3.050505, -128.3247, 1, 0.5843138, 0, 1,
4.848485, 3.050505, -128.0334, 1, 0.6862745, 0, 1,
4.888889, 3.050505, -127.7509, 1, 0.6862745, 0, 1,
4.929293, 3.050505, -127.4772, 1, 0.6862745, 0, 1,
4.969697, 3.050505, -127.2122, 1, 0.6862745, 0, 1,
5.010101, 3.050505, -126.956, 1, 0.6862745, 0, 1,
5.050505, 3.050505, -126.7085, 1, 0.7921569, 0, 1,
5.090909, 3.050505, -126.4699, 1, 0.7921569, 0, 1,
5.131313, 3.050505, -126.24, 1, 0.7921569, 0, 1,
5.171717, 3.050505, -126.0189, 1, 0.7921569, 0, 1,
5.212121, 3.050505, -125.8065, 1, 0.7921569, 0, 1,
5.252525, 3.050505, -125.6029, 1, 0.7921569, 0, 1,
5.292929, 3.050505, -125.4081, 1, 0.8941177, 0, 1,
5.333333, 3.050505, -125.2221, 1, 0.8941177, 0, 1,
5.373737, 3.050505, -125.0448, 1, 0.8941177, 0, 1,
5.414141, 3.050505, -124.8763, 1, 0.8941177, 0, 1,
5.454545, 3.050505, -124.7166, 1, 0.8941177, 0, 1,
5.494949, 3.050505, -124.5657, 1, 0.8941177, 0, 1,
5.535354, 3.050505, -124.4235, 1, 0.8941177, 0, 1,
5.575758, 3.050505, -124.2901, 1, 1, 0, 1,
5.616162, 3.050505, -124.1655, 1, 1, 0, 1,
5.656566, 3.050505, -124.0496, 1, 1, 0, 1,
5.69697, 3.050505, -123.9425, 1, 1, 0, 1,
5.737374, 3.050505, -123.8442, 1, 1, 0, 1,
5.777778, 3.050505, -123.7546, 1, 1, 0, 1,
5.818182, 3.050505, -123.6739, 1, 1, 0, 1,
5.858586, 3.050505, -123.6019, 1, 1, 0, 1,
5.89899, 3.050505, -123.5386, 1, 1, 0, 1,
5.939394, 3.050505, -123.4842, 1, 1, 0, 1,
5.979798, 3.050505, -123.4385, 1, 1, 0, 1,
6.020202, 3.050505, -123.4015, 1, 1, 0, 1,
6.060606, 3.050505, -123.3734, 1, 1, 0, 1,
6.10101, 3.050505, -123.354, 1, 1, 0, 1,
6.141414, 3.050505, -123.3434, 1, 1, 0, 1,
6.181818, 3.050505, -123.3416, 1, 1, 0, 1,
6.222222, 3.050505, -123.3485, 1, 1, 0, 1,
6.262626, 3.050505, -123.3642, 1, 1, 0, 1,
6.30303, 3.050505, -123.3887, 1, 1, 0, 1,
6.343434, 3.050505, -123.422, 1, 1, 0, 1,
6.383838, 3.050505, -123.464, 1, 1, 0, 1,
6.424242, 3.050505, -123.5148, 1, 1, 0, 1,
6.464646, 3.050505, -123.5744, 1, 1, 0, 1,
6.505051, 3.050505, -123.6427, 1, 1, 0, 1,
6.545455, 3.050505, -123.7198, 1, 1, 0, 1,
6.585859, 3.050505, -123.8057, 1, 1, 0, 1,
6.626263, 3.050505, -123.9004, 1, 1, 0, 1,
6.666667, 3.050505, -124.0038, 1, 1, 0, 1,
6.707071, 3.050505, -124.116, 1, 1, 0, 1,
6.747475, 3.050505, -124.2369, 1, 1, 0, 1,
6.787879, 3.050505, -124.3667, 1, 0.8941177, 0, 1,
6.828283, 3.050505, -124.5052, 1, 0.8941177, 0, 1,
6.868687, 3.050505, -124.6525, 1, 0.8941177, 0, 1,
6.909091, 3.050505, -124.8085, 1, 0.8941177, 0, 1,
6.949495, 3.050505, -124.9734, 1, 0.8941177, 0, 1,
6.989899, 3.050505, -125.147, 1, 0.8941177, 0, 1,
7.030303, 3.050505, -125.3293, 1, 0.8941177, 0, 1,
7.070707, 3.050505, -125.5205, 1, 0.8941177, 0, 1,
7.111111, 3.050505, -125.7204, 1, 0.7921569, 0, 1,
7.151515, 3.050505, -125.9291, 1, 0.7921569, 0, 1,
7.191919, 3.050505, -126.1465, 1, 0.7921569, 0, 1,
7.232323, 3.050505, -126.3728, 1, 0.7921569, 0, 1,
7.272727, 3.050505, -126.6077, 1, 0.7921569, 0, 1,
7.313131, 3.050505, -126.8515, 1, 0.6862745, 0, 1,
7.353535, 3.050505, -127.1041, 1, 0.6862745, 0, 1,
7.393939, 3.050505, -127.3654, 1, 0.6862745, 0, 1,
7.434343, 3.050505, -127.6355, 1, 0.6862745, 0, 1,
7.474748, 3.050505, -127.9143, 1, 0.6862745, 0, 1,
7.515152, 3.050505, -128.202, 1, 0.5843138, 0, 1,
7.555555, 3.050505, -128.4984, 1, 0.5843138, 0, 1,
7.59596, 3.050505, -128.8035, 1, 0.5843138, 0, 1,
7.636364, 3.050505, -129.1175, 1, 0.5843138, 0, 1,
7.676768, 3.050505, -129.4402, 1, 0.4823529, 0, 1,
7.717172, 3.050505, -129.7717, 1, 0.4823529, 0, 1,
7.757576, 3.050505, -130.1119, 1, 0.4823529, 0, 1,
7.79798, 3.050505, -130.461, 1, 0.4823529, 0, 1,
7.838384, 3.050505, -130.8188, 1, 0.3764706, 0, 1,
7.878788, 3.050505, -131.1853, 1, 0.3764706, 0, 1,
7.919192, 3.050505, -131.5607, 1, 0.3764706, 0, 1,
7.959596, 3.050505, -131.9448, 1, 0.2745098, 0, 1,
8, 3.050505, -132.3377, 1, 0.2745098, 0, 1,
4, 3.090909, -135.7602, 0.9647059, 0, 0.03137255, 1,
4.040404, 3.090909, -135.3056, 1, 0.06666667, 0, 1,
4.080808, 3.090909, -134.8595, 1, 0.06666667, 0, 1,
4.121212, 3.090909, -134.422, 1, 0.06666667, 0, 1,
4.161616, 3.090909, -133.993, 1, 0.1686275, 0, 1,
4.20202, 3.090909, -133.5726, 1, 0.1686275, 0, 1,
4.242424, 3.090909, -133.1607, 1, 0.1686275, 0, 1,
4.282828, 3.090909, -132.7574, 1, 0.2745098, 0, 1,
4.323232, 3.090909, -132.3626, 1, 0.2745098, 0, 1,
4.363636, 3.090909, -131.9763, 1, 0.2745098, 0, 1,
4.40404, 3.090909, -131.5986, 1, 0.3764706, 0, 1,
4.444445, 3.090909, -131.2294, 1, 0.3764706, 0, 1,
4.484848, 3.090909, -130.8688, 1, 0.3764706, 0, 1,
4.525252, 3.090909, -130.5168, 1, 0.4823529, 0, 1,
4.565657, 3.090909, -130.1732, 1, 0.4823529, 0, 1,
4.606061, 3.090909, -129.8382, 1, 0.4823529, 0, 1,
4.646465, 3.090909, -129.5118, 1, 0.4823529, 0, 1,
4.686869, 3.090909, -129.1939, 1, 0.5843138, 0, 1,
4.727273, 3.090909, -128.8845, 1, 0.5843138, 0, 1,
4.767677, 3.090909, -128.5837, 1, 0.5843138, 0, 1,
4.808081, 3.090909, -128.2914, 1, 0.5843138, 0, 1,
4.848485, 3.090909, -128.0077, 1, 0.6862745, 0, 1,
4.888889, 3.090909, -127.7325, 1, 0.6862745, 0, 1,
4.929293, 3.090909, -127.4659, 1, 0.6862745, 0, 1,
4.969697, 3.090909, -127.2078, 1, 0.6862745, 0, 1,
5.010101, 3.090909, -126.9582, 1, 0.6862745, 0, 1,
5.050505, 3.090909, -126.7172, 1, 0.7921569, 0, 1,
5.090909, 3.090909, -126.4848, 1, 0.7921569, 0, 1,
5.131313, 3.090909, -126.2608, 1, 0.7921569, 0, 1,
5.171717, 3.090909, -126.0455, 1, 0.7921569, 0, 1,
5.212121, 3.090909, -125.8386, 1, 0.7921569, 0, 1,
5.252525, 3.090909, -125.6403, 1, 0.7921569, 0, 1,
5.292929, 3.090909, -125.4506, 1, 0.8941177, 0, 1,
5.333333, 3.090909, -125.2694, 1, 0.8941177, 0, 1,
5.373737, 3.090909, -125.0967, 1, 0.8941177, 0, 1,
5.414141, 3.090909, -124.9326, 1, 0.8941177, 0, 1,
5.454545, 3.090909, -124.777, 1, 0.8941177, 0, 1,
5.494949, 3.090909, -124.63, 1, 0.8941177, 0, 1,
5.535354, 3.090909, -124.4915, 1, 0.8941177, 0, 1,
5.575758, 3.090909, -124.3616, 1, 0.8941177, 0, 1,
5.616162, 3.090909, -124.2402, 1, 1, 0, 1,
5.656566, 3.090909, -124.1273, 1, 1, 0, 1,
5.69697, 3.090909, -124.023, 1, 1, 0, 1,
5.737374, 3.090909, -123.9273, 1, 1, 0, 1,
5.777778, 3.090909, -123.84, 1, 1, 0, 1,
5.818182, 3.090909, -123.7614, 1, 1, 0, 1,
5.858586, 3.090909, -123.6912, 1, 1, 0, 1,
5.89899, 3.090909, -123.6296, 1, 1, 0, 1,
5.939394, 3.090909, -123.5766, 1, 1, 0, 1,
5.979798, 3.090909, -123.5321, 1, 1, 0, 1,
6.020202, 3.090909, -123.4961, 1, 1, 0, 1,
6.060606, 3.090909, -123.4687, 1, 1, 0, 1,
6.10101, 3.090909, -123.4498, 1, 1, 0, 1,
6.141414, 3.090909, -123.4395, 1, 1, 0, 1,
6.181818, 3.090909, -123.4377, 1, 1, 0, 1,
6.222222, 3.090909, -123.4445, 1, 1, 0, 1,
6.262626, 3.090909, -123.4598, 1, 1, 0, 1,
6.30303, 3.090909, -123.4836, 1, 1, 0, 1,
6.343434, 3.090909, -123.516, 1, 1, 0, 1,
6.383838, 3.090909, -123.5569, 1, 1, 0, 1,
6.424242, 3.090909, -123.6064, 1, 1, 0, 1,
6.464646, 3.090909, -123.6644, 1, 1, 0, 1,
6.505051, 3.090909, -123.731, 1, 1, 0, 1,
6.545455, 3.090909, -123.8061, 1, 1, 0, 1,
6.585859, 3.090909, -123.8898, 1, 1, 0, 1,
6.626263, 3.090909, -123.982, 1, 1, 0, 1,
6.666667, 3.090909, -124.0827, 1, 1, 0, 1,
6.707071, 3.090909, -124.192, 1, 1, 0, 1,
6.747475, 3.090909, -124.3098, 1, 1, 0, 1,
6.787879, 3.090909, -124.4362, 1, 0.8941177, 0, 1,
6.828283, 3.090909, -124.5711, 1, 0.8941177, 0, 1,
6.868687, 3.090909, -124.7146, 1, 0.8941177, 0, 1,
6.909091, 3.090909, -124.8666, 1, 0.8941177, 0, 1,
6.949495, 3.090909, -125.0271, 1, 0.8941177, 0, 1,
6.989899, 3.090909, -125.1962, 1, 0.8941177, 0, 1,
7.030303, 3.090909, -125.3738, 1, 0.8941177, 0, 1,
7.070707, 3.090909, -125.56, 1, 0.8941177, 0, 1,
7.111111, 3.090909, -125.7547, 1, 0.7921569, 0, 1,
7.151515, 3.090909, -125.958, 1, 0.7921569, 0, 1,
7.191919, 3.090909, -126.1698, 1, 0.7921569, 0, 1,
7.232323, 3.090909, -126.3902, 1, 0.7921569, 0, 1,
7.272727, 3.090909, -126.619, 1, 0.7921569, 0, 1,
7.313131, 3.090909, -126.8565, 1, 0.6862745, 0, 1,
7.353535, 3.090909, -127.1025, 1, 0.6862745, 0, 1,
7.393939, 3.090909, -127.357, 1, 0.6862745, 0, 1,
7.434343, 3.090909, -127.6201, 1, 0.6862745, 0, 1,
7.474748, 3.090909, -127.8917, 1, 0.6862745, 0, 1,
7.515152, 3.090909, -128.1718, 1, 0.5843138, 0, 1,
7.555555, 3.090909, -128.4605, 1, 0.5843138, 0, 1,
7.59596, 3.090909, -128.7578, 1, 0.5843138, 0, 1,
7.636364, 3.090909, -129.0636, 1, 0.5843138, 0, 1,
7.676768, 3.090909, -129.3779, 1, 0.4823529, 0, 1,
7.717172, 3.090909, -129.7008, 1, 0.4823529, 0, 1,
7.757576, 3.090909, -130.0322, 1, 0.4823529, 0, 1,
7.79798, 3.090909, -130.3722, 1, 0.4823529, 0, 1,
7.838384, 3.090909, -130.7207, 1, 0.3764706, 0, 1,
7.878788, 3.090909, -131.0777, 1, 0.3764706, 0, 1,
7.919192, 3.090909, -131.4433, 1, 0.3764706, 0, 1,
7.959596, 3.090909, -131.8175, 1, 0.2745098, 0, 1,
8, 3.090909, -132.2001, 1, 0.2745098, 0, 1,
4, 3.131313, -135.5534, 0.9647059, 0, 0.03137255, 1,
4.040404, 3.131313, -135.1105, 1, 0.06666667, 0, 1,
4.080808, 3.131313, -134.6758, 1, 0.06666667, 0, 1,
4.121212, 3.131313, -134.2495, 1, 0.1686275, 0, 1,
4.161616, 3.131313, -133.8316, 1, 0.1686275, 0, 1,
4.20202, 3.131313, -133.4219, 1, 0.1686275, 0, 1,
4.242424, 3.131313, -133.0206, 1, 0.2745098, 0, 1,
4.282828, 3.131313, -132.6276, 1, 0.2745098, 0, 1,
4.323232, 3.131313, -132.2429, 1, 0.2745098, 0, 1,
4.363636, 3.131313, -131.8666, 1, 0.2745098, 0, 1,
4.40404, 3.131313, -131.4986, 1, 0.3764706, 0, 1,
4.444445, 3.131313, -131.1389, 1, 0.3764706, 0, 1,
4.484848, 3.131313, -130.7875, 1, 0.3764706, 0, 1,
4.525252, 3.131313, -130.4444, 1, 0.4823529, 0, 1,
4.565657, 3.131313, -130.1097, 1, 0.4823529, 0, 1,
4.606061, 3.131313, -129.7833, 1, 0.4823529, 0, 1,
4.646465, 3.131313, -129.4652, 1, 0.4823529, 0, 1,
4.686869, 3.131313, -129.1555, 1, 0.5843138, 0, 1,
4.727273, 3.131313, -128.854, 1, 0.5843138, 0, 1,
4.767677, 3.131313, -128.5609, 1, 0.5843138, 0, 1,
4.808081, 3.131313, -128.2762, 1, 0.5843138, 0, 1,
4.848485, 3.131313, -127.9997, 1, 0.6862745, 0, 1,
4.888889, 3.131313, -127.7316, 1, 0.6862745, 0, 1,
4.929293, 3.131313, -127.4718, 1, 0.6862745, 0, 1,
4.969697, 3.131313, -127.2203, 1, 0.6862745, 0, 1,
5.010101, 3.131313, -126.9771, 1, 0.6862745, 0, 1,
5.050505, 3.131313, -126.7423, 1, 0.7921569, 0, 1,
5.090909, 3.131313, -126.5158, 1, 0.7921569, 0, 1,
5.131313, 3.131313, -126.2976, 1, 0.7921569, 0, 1,
5.171717, 3.131313, -126.0878, 1, 0.7921569, 0, 1,
5.212121, 3.131313, -125.8863, 1, 0.7921569, 0, 1,
5.252525, 3.131313, -125.693, 1, 0.7921569, 0, 1,
5.292929, 3.131313, -125.5082, 1, 0.8941177, 0, 1,
5.333333, 3.131313, -125.3316, 1, 0.8941177, 0, 1,
5.373737, 3.131313, -125.1634, 1, 0.8941177, 0, 1,
5.414141, 3.131313, -125.0035, 1, 0.8941177, 0, 1,
5.454545, 3.131313, -124.8519, 1, 0.8941177, 0, 1,
5.494949, 3.131313, -124.7086, 1, 0.8941177, 0, 1,
5.535354, 3.131313, -124.5737, 1, 0.8941177, 0, 1,
5.575758, 3.131313, -124.4471, 1, 0.8941177, 0, 1,
5.616162, 3.131313, -124.3288, 1, 1, 0, 1,
5.656566, 3.131313, -124.2188, 1, 1, 0, 1,
5.69697, 3.131313, -124.1172, 1, 1, 0, 1,
5.737374, 3.131313, -124.0239, 1, 1, 0, 1,
5.777778, 3.131313, -123.9389, 1, 1, 0, 1,
5.818182, 3.131313, -123.8622, 1, 1, 0, 1,
5.858586, 3.131313, -123.7939, 1, 1, 0, 1,
5.89899, 3.131313, -123.7339, 1, 1, 0, 1,
5.939394, 3.131313, -123.6822, 1, 1, 0, 1,
5.979798, 3.131313, -123.6388, 1, 1, 0, 1,
6.020202, 3.131313, -123.6038, 1, 1, 0, 1,
6.060606, 3.131313, -123.5771, 1, 1, 0, 1,
6.10101, 3.131313, -123.5587, 1, 1, 0, 1,
6.141414, 3.131313, -123.5486, 1, 1, 0, 1,
6.181818, 3.131313, -123.5469, 1, 1, 0, 1,
6.222222, 3.131313, -123.5535, 1, 1, 0, 1,
6.262626, 3.131313, -123.5684, 1, 1, 0, 1,
6.30303, 3.131313, -123.5916, 1, 1, 0, 1,
6.343434, 3.131313, -123.6232, 1, 1, 0, 1,
6.383838, 3.131313, -123.6631, 1, 1, 0, 1,
6.424242, 3.131313, -123.7113, 1, 1, 0, 1,
6.464646, 3.131313, -123.7678, 1, 1, 0, 1,
6.505051, 3.131313, -123.8327, 1, 1, 0, 1,
6.545455, 3.131313, -123.9059, 1, 1, 0, 1,
6.585859, 3.131313, -123.9874, 1, 1, 0, 1,
6.626263, 3.131313, -124.0772, 1, 1, 0, 1,
6.666667, 3.131313, -124.1754, 1, 1, 0, 1,
6.707071, 3.131313, -124.2818, 1, 1, 0, 1,
6.747475, 3.131313, -124.3966, 1, 0.8941177, 0, 1,
6.787879, 3.131313, -124.5198, 1, 0.8941177, 0, 1,
6.828283, 3.131313, -124.6512, 1, 0.8941177, 0, 1,
6.868687, 3.131313, -124.791, 1, 0.8941177, 0, 1,
6.909091, 3.131313, -124.9391, 1, 0.8941177, 0, 1,
6.949495, 3.131313, -125.0955, 1, 0.8941177, 0, 1,
6.989899, 3.131313, -125.2603, 1, 0.8941177, 0, 1,
7.030303, 3.131313, -125.4334, 1, 0.8941177, 0, 1,
7.070707, 3.131313, -125.6148, 1, 0.7921569, 0, 1,
7.111111, 3.131313, -125.8045, 1, 0.7921569, 0, 1,
7.151515, 3.131313, -126.0026, 1, 0.7921569, 0, 1,
7.191919, 3.131313, -126.2089, 1, 0.7921569, 0, 1,
7.232323, 3.131313, -126.4236, 1, 0.7921569, 0, 1,
7.272727, 3.131313, -126.6467, 1, 0.7921569, 0, 1,
7.313131, 3.131313, -126.878, 1, 0.6862745, 0, 1,
7.353535, 3.131313, -127.1177, 1, 0.6862745, 0, 1,
7.393939, 3.131313, -127.3657, 1, 0.6862745, 0, 1,
7.434343, 3.131313, -127.622, 1, 0.6862745, 0, 1,
7.474748, 3.131313, -127.8867, 1, 0.6862745, 0, 1,
7.515152, 3.131313, -128.1597, 1, 0.5843138, 0, 1,
7.555555, 3.131313, -128.4409, 1, 0.5843138, 0, 1,
7.59596, 3.131313, -128.7306, 1, 0.5843138, 0, 1,
7.636364, 3.131313, -129.0285, 1, 0.5843138, 0, 1,
7.676768, 3.131313, -129.3348, 1, 0.4823529, 0, 1,
7.717172, 3.131313, -129.6494, 1, 0.4823529, 0, 1,
7.757576, 3.131313, -129.9723, 1, 0.4823529, 0, 1,
7.79798, 3.131313, -130.3036, 1, 0.4823529, 0, 1,
7.838384, 3.131313, -130.6431, 1, 0.3764706, 0, 1,
7.878788, 3.131313, -130.991, 1, 0.3764706, 0, 1,
7.919192, 3.131313, -131.3472, 1, 0.3764706, 0, 1,
7.959596, 3.131313, -131.7118, 1, 0.3764706, 0, 1,
8, 3.131313, -132.0847, 1, 0.2745098, 0, 1,
4, 3.171717, -135.3708, 1, 0.06666667, 0, 1,
4.040404, 3.171717, -134.9391, 1, 0.06666667, 0, 1,
4.080808, 3.171717, -134.5155, 1, 0.06666667, 0, 1,
4.121212, 3.171717, -134.1, 1, 0.1686275, 0, 1,
4.161616, 3.171717, -133.6926, 1, 0.1686275, 0, 1,
4.20202, 3.171717, -133.2933, 1, 0.1686275, 0, 1,
4.242424, 3.171717, -132.9021, 1, 0.2745098, 0, 1,
4.282828, 3.171717, -132.5191, 1, 0.2745098, 0, 1,
4.323232, 3.171717, -132.1442, 1, 0.2745098, 0, 1,
4.363636, 3.171717, -131.7773, 1, 0.3764706, 0, 1,
4.40404, 3.171717, -131.4186, 1, 0.3764706, 0, 1,
4.444445, 3.171717, -131.068, 1, 0.3764706, 0, 1,
4.484848, 3.171717, -130.7256, 1, 0.3764706, 0, 1,
4.525252, 3.171717, -130.3912, 1, 0.4823529, 0, 1,
4.565657, 3.171717, -130.0649, 1, 0.4823529, 0, 1,
4.606061, 3.171717, -129.7468, 1, 0.4823529, 0, 1,
4.646465, 3.171717, -129.4368, 1, 0.4823529, 0, 1,
4.686869, 3.171717, -129.1349, 1, 0.5843138, 0, 1,
4.727273, 3.171717, -128.8411, 1, 0.5843138, 0, 1,
4.767677, 3.171717, -128.5554, 1, 0.5843138, 0, 1,
4.808081, 3.171717, -128.2778, 1, 0.5843138, 0, 1,
4.848485, 3.171717, -128.0084, 1, 0.6862745, 0, 1,
4.888889, 3.171717, -127.747, 1, 0.6862745, 0, 1,
4.929293, 3.171717, -127.4938, 1, 0.6862745, 0, 1,
4.969697, 3.171717, -127.2487, 1, 0.6862745, 0, 1,
5.010101, 3.171717, -127.0117, 1, 0.6862745, 0, 1,
5.050505, 3.171717, -126.7828, 1, 0.7921569, 0, 1,
5.090909, 3.171717, -126.562, 1, 0.7921569, 0, 1,
5.131313, 3.171717, -126.3494, 1, 0.7921569, 0, 1,
5.171717, 3.171717, -126.1448, 1, 0.7921569, 0, 1,
5.212121, 3.171717, -125.9484, 1, 0.7921569, 0, 1,
5.252525, 3.171717, -125.7601, 1, 0.7921569, 0, 1,
5.292929, 3.171717, -125.5799, 1, 0.7921569, 0, 1,
5.333333, 3.171717, -125.4078, 1, 0.8941177, 0, 1,
5.373737, 3.171717, -125.2438, 1, 0.8941177, 0, 1,
5.414141, 3.171717, -125.088, 1, 0.8941177, 0, 1,
5.454545, 3.171717, -124.9402, 1, 0.8941177, 0, 1,
5.494949, 3.171717, -124.8006, 1, 0.8941177, 0, 1,
5.535354, 3.171717, -124.6691, 1, 0.8941177, 0, 1,
5.575758, 3.171717, -124.5457, 1, 0.8941177, 0, 1,
5.616162, 3.171717, -124.4304, 1, 0.8941177, 0, 1,
5.656566, 3.171717, -124.3232, 1, 1, 0, 1,
5.69697, 3.171717, -124.2241, 1, 1, 0, 1,
5.737374, 3.171717, -124.1332, 1, 1, 0, 1,
5.777778, 3.171717, -124.0504, 1, 1, 0, 1,
5.818182, 3.171717, -123.9756, 1, 1, 0, 1,
5.858586, 3.171717, -123.909, 1, 1, 0, 1,
5.89899, 3.171717, -123.8505, 1, 1, 0, 1,
5.939394, 3.171717, -123.8002, 1, 1, 0, 1,
5.979798, 3.171717, -123.7579, 1, 1, 0, 1,
6.020202, 3.171717, -123.7237, 1, 1, 0, 1,
6.060606, 3.171717, -123.6977, 1, 1, 0, 1,
6.10101, 3.171717, -123.6798, 1, 1, 0, 1,
6.141414, 3.171717, -123.67, 1, 1, 0, 1,
6.181818, 3.171717, -123.6683, 1, 1, 0, 1,
6.222222, 3.171717, -123.6747, 1, 1, 0, 1,
6.262626, 3.171717, -123.6892, 1, 1, 0, 1,
6.30303, 3.171717, -123.7119, 1, 1, 0, 1,
6.343434, 3.171717, -123.7426, 1, 1, 0, 1,
6.383838, 3.171717, -123.7815, 1, 1, 0, 1,
6.424242, 3.171717, -123.8285, 1, 1, 0, 1,
6.464646, 3.171717, -123.8836, 1, 1, 0, 1,
6.505051, 3.171717, -123.9468, 1, 1, 0, 1,
6.545455, 3.171717, -124.0182, 1, 1, 0, 1,
6.585859, 3.171717, -124.0976, 1, 1, 0, 1,
6.626263, 3.171717, -124.1852, 1, 1, 0, 1,
6.666667, 3.171717, -124.2808, 1, 1, 0, 1,
6.707071, 3.171717, -124.3846, 1, 0.8941177, 0, 1,
6.747475, 3.171717, -124.4965, 1, 0.8941177, 0, 1,
6.787879, 3.171717, -124.6165, 1, 0.8941177, 0, 1,
6.828283, 3.171717, -124.7447, 1, 0.8941177, 0, 1,
6.868687, 3.171717, -124.8809, 1, 0.8941177, 0, 1,
6.909091, 3.171717, -125.0252, 1, 0.8941177, 0, 1,
6.949495, 3.171717, -125.1777, 1, 0.8941177, 0, 1,
6.989899, 3.171717, -125.3383, 1, 0.8941177, 0, 1,
7.030303, 3.171717, -125.507, 1, 0.8941177, 0, 1,
7.070707, 3.171717, -125.6838, 1, 0.7921569, 0, 1,
7.111111, 3.171717, -125.8687, 1, 0.7921569, 0, 1,
7.151515, 3.171717, -126.0618, 1, 0.7921569, 0, 1,
7.191919, 3.171717, -126.2629, 1, 0.7921569, 0, 1,
7.232323, 3.171717, -126.4722, 1, 0.7921569, 0, 1,
7.272727, 3.171717, -126.6896, 1, 0.7921569, 0, 1,
7.313131, 3.171717, -126.9151, 1, 0.6862745, 0, 1,
7.353535, 3.171717, -127.1487, 1, 0.6862745, 0, 1,
7.393939, 3.171717, -127.3904, 1, 0.6862745, 0, 1,
7.434343, 3.171717, -127.6402, 1, 0.6862745, 0, 1,
7.474748, 3.171717, -127.8982, 1, 0.6862745, 0, 1,
7.515152, 3.171717, -128.1642, 1, 0.5843138, 0, 1,
7.555555, 3.171717, -128.4384, 1, 0.5843138, 0, 1,
7.59596, 3.171717, -128.7207, 1, 0.5843138, 0, 1,
7.636364, 3.171717, -129.0111, 1, 0.5843138, 0, 1,
7.676768, 3.171717, -129.3096, 1, 0.4823529, 0, 1,
7.717172, 3.171717, -129.6163, 1, 0.4823529, 0, 1,
7.757576, 3.171717, -129.931, 1, 0.4823529, 0, 1,
7.79798, 3.171717, -130.2539, 1, 0.4823529, 0, 1,
7.838384, 3.171717, -130.5849, 1, 0.3764706, 0, 1,
7.878788, 3.171717, -130.924, 1, 0.3764706, 0, 1,
7.919192, 3.171717, -131.2712, 1, 0.3764706, 0, 1,
7.959596, 3.171717, -131.6265, 1, 0.3764706, 0, 1,
8, 3.171717, -131.9899, 1, 0.2745098, 0, 1,
4, 3.212121, -135.211, 1, 0.06666667, 0, 1,
4.040404, 3.212121, -134.7901, 1, 0.06666667, 0, 1,
4.080808, 3.212121, -134.3771, 1, 0.06666667, 0, 1,
4.121212, 3.212121, -133.9719, 1, 0.1686275, 0, 1,
4.161616, 3.212121, -133.5747, 1, 0.1686275, 0, 1,
4.20202, 3.212121, -133.1854, 1, 0.1686275, 0, 1,
4.242424, 3.212121, -132.8041, 1, 0.2745098, 0, 1,
4.282828, 3.212121, -132.4306, 1, 0.2745098, 0, 1,
4.323232, 3.212121, -132.065, 1, 0.2745098, 0, 1,
4.363636, 3.212121, -131.7074, 1, 0.3764706, 0, 1,
4.40404, 3.212121, -131.3576, 1, 0.3764706, 0, 1,
4.444445, 3.212121, -131.0158, 1, 0.3764706, 0, 1,
4.484848, 3.212121, -130.6819, 1, 0.3764706, 0, 1,
4.525252, 3.212121, -130.3559, 1, 0.4823529, 0, 1,
4.565657, 3.212121, -130.0378, 1, 0.4823529, 0, 1,
4.606061, 3.212121, -129.7276, 1, 0.4823529, 0, 1,
4.646465, 3.212121, -129.4253, 1, 0.4823529, 0, 1,
4.686869, 3.212121, -129.131, 1, 0.5843138, 0, 1,
4.727273, 3.212121, -128.8445, 1, 0.5843138, 0, 1,
4.767677, 3.212121, -128.566, 1, 0.5843138, 0, 1,
4.808081, 3.212121, -128.2953, 1, 0.5843138, 0, 1,
4.848485, 3.212121, -128.0326, 1, 0.6862745, 0, 1,
4.888889, 3.212121, -127.7778, 1, 0.6862745, 0, 1,
4.929293, 3.212121, -127.5309, 1, 0.6862745, 0, 1,
4.969697, 3.212121, -127.2919, 1, 0.6862745, 0, 1,
5.010101, 3.212121, -127.0609, 1, 0.6862745, 0, 1,
5.050505, 3.212121, -126.8377, 1, 0.6862745, 0, 1,
5.090909, 3.212121, -126.6224, 1, 0.7921569, 0, 1,
5.131313, 3.212121, -126.4151, 1, 0.7921569, 0, 1,
5.171717, 3.212121, -126.2157, 1, 0.7921569, 0, 1,
5.212121, 3.212121, -126.0242, 1, 0.7921569, 0, 1,
5.252525, 3.212121, -125.8405, 1, 0.7921569, 0, 1,
5.292929, 3.212121, -125.6648, 1, 0.7921569, 0, 1,
5.333333, 3.212121, -125.4971, 1, 0.8941177, 0, 1,
5.373737, 3.212121, -125.3372, 1, 0.8941177, 0, 1,
5.414141, 3.212121, -125.1852, 1, 0.8941177, 0, 1,
5.454545, 3.212121, -125.0412, 1, 0.8941177, 0, 1,
5.494949, 3.212121, -124.905, 1, 0.8941177, 0, 1,
5.535354, 3.212121, -124.7768, 1, 0.8941177, 0, 1,
5.575758, 3.212121, -124.6565, 1, 0.8941177, 0, 1,
5.616162, 3.212121, -124.5441, 1, 0.8941177, 0, 1,
5.656566, 3.212121, -124.4396, 1, 0.8941177, 0, 1,
5.69697, 3.212121, -124.343, 1, 0.8941177, 0, 1,
5.737374, 3.212121, -124.2543, 1, 1, 0, 1,
5.777778, 3.212121, -124.1736, 1, 1, 0, 1,
5.818182, 3.212121, -124.1007, 1, 1, 0, 1,
5.858586, 3.212121, -124.0358, 1, 1, 0, 1,
5.89899, 3.212121, -123.9787, 1, 1, 0, 1,
5.939394, 3.212121, -123.9296, 1, 1, 0, 1,
5.979798, 3.212121, -123.8884, 1, 1, 0, 1,
6.020202, 3.212121, -123.8551, 1, 1, 0, 1,
6.060606, 3.212121, -123.8297, 1, 1, 0, 1,
6.10101, 3.212121, -123.8122, 1, 1, 0, 1,
6.141414, 3.212121, -123.8027, 1, 1, 0, 1,
6.181818, 3.212121, -123.801, 1, 1, 0, 1,
6.222222, 3.212121, -123.8073, 1, 1, 0, 1,
6.262626, 3.212121, -123.8215, 1, 1, 0, 1,
6.30303, 3.212121, -123.8435, 1, 1, 0, 1,
6.343434, 3.212121, -123.8735, 1, 1, 0, 1,
6.383838, 3.212121, -123.9114, 1, 1, 0, 1,
6.424242, 3.212121, -123.9572, 1, 1, 0, 1,
6.464646, 3.212121, -124.011, 1, 1, 0, 1,
6.505051, 3.212121, -124.0726, 1, 1, 0, 1,
6.545455, 3.212121, -124.1422, 1, 1, 0, 1,
6.585859, 3.212121, -124.2196, 1, 1, 0, 1,
6.626263, 3.212121, -124.305, 1, 1, 0, 1,
6.666667, 3.212121, -124.3983, 1, 0.8941177, 0, 1,
6.707071, 3.212121, -124.4995, 1, 0.8941177, 0, 1,
6.747475, 3.212121, -124.6086, 1, 0.8941177, 0, 1,
6.787879, 3.212121, -124.7256, 1, 0.8941177, 0, 1,
6.828283, 3.212121, -124.8505, 1, 0.8941177, 0, 1,
6.868687, 3.212121, -124.9833, 1, 0.8941177, 0, 1,
6.909091, 3.212121, -125.1241, 1, 0.8941177, 0, 1,
6.949495, 3.212121, -125.2727, 1, 0.8941177, 0, 1,
6.989899, 3.212121, -125.4293, 1, 0.8941177, 0, 1,
7.030303, 3.212121, -125.5938, 1, 0.7921569, 0, 1,
7.070707, 3.212121, -125.7662, 1, 0.7921569, 0, 1,
7.111111, 3.212121, -125.9465, 1, 0.7921569, 0, 1,
7.151515, 3.212121, -126.1347, 1, 0.7921569, 0, 1,
7.191919, 3.212121, -126.3308, 1, 0.7921569, 0, 1,
7.232323, 3.212121, -126.5348, 1, 0.7921569, 0, 1,
7.272727, 3.212121, -126.7468, 1, 0.7921569, 0, 1,
7.313131, 3.212121, -126.9667, 1, 0.6862745, 0, 1,
7.353535, 3.212121, -127.1944, 1, 0.6862745, 0, 1,
7.393939, 3.212121, -127.4301, 1, 0.6862745, 0, 1,
7.434343, 3.212121, -127.6737, 1, 0.6862745, 0, 1,
7.474748, 3.212121, -127.9252, 1, 0.6862745, 0, 1,
7.515152, 3.212121, -128.1846, 1, 0.5843138, 0, 1,
7.555555, 3.212121, -128.4519, 1, 0.5843138, 0, 1,
7.59596, 3.212121, -128.7272, 1, 0.5843138, 0, 1,
7.636364, 3.212121, -129.0103, 1, 0.5843138, 0, 1,
7.676768, 3.212121, -129.3014, 1, 0.5843138, 0, 1,
7.717172, 3.212121, -129.6003, 1, 0.4823529, 0, 1,
7.757576, 3.212121, -129.9072, 1, 0.4823529, 0, 1,
7.79798, 3.212121, -130.222, 1, 0.4823529, 0, 1,
7.838384, 3.212121, -130.5447, 1, 0.3764706, 0, 1,
7.878788, 3.212121, -130.8753, 1, 0.3764706, 0, 1,
7.919192, 3.212121, -131.2138, 1, 0.3764706, 0, 1,
7.959596, 3.212121, -131.5603, 1, 0.3764706, 0, 1,
8, 3.212121, -131.9146, 1, 0.2745098, 0, 1,
4, 3.252525, -135.0727, 1, 0.06666667, 0, 1,
4.040404, 3.252525, -134.6621, 1, 0.06666667, 0, 1,
4.080808, 3.252525, -134.2593, 1, 0.1686275, 0, 1,
4.121212, 3.252525, -133.8642, 1, 0.1686275, 0, 1,
4.161616, 3.252525, -133.4768, 1, 0.1686275, 0, 1,
4.20202, 3.252525, -133.0971, 1, 0.1686275, 0, 1,
4.242424, 3.252525, -132.7251, 1, 0.2745098, 0, 1,
4.282828, 3.252525, -132.3609, 1, 0.2745098, 0, 1,
4.323232, 3.252525, -132.0043, 1, 0.2745098, 0, 1,
4.363636, 3.252525, -131.6555, 1, 0.3764706, 0, 1,
4.40404, 3.252525, -131.3144, 1, 0.3764706, 0, 1,
4.444445, 3.252525, -130.981, 1, 0.3764706, 0, 1,
4.484848, 3.252525, -130.6553, 1, 0.3764706, 0, 1,
4.525252, 3.252525, -130.3374, 1, 0.4823529, 0, 1,
4.565657, 3.252525, -130.0271, 1, 0.4823529, 0, 1,
4.606061, 3.252525, -129.7246, 1, 0.4823529, 0, 1,
4.646465, 3.252525, -129.4298, 1, 0.4823529, 0, 1,
4.686869, 3.252525, -129.1427, 1, 0.5843138, 0, 1,
4.727273, 3.252525, -128.8633, 1, 0.5843138, 0, 1,
4.767677, 3.252525, -128.5917, 1, 0.5843138, 0, 1,
4.808081, 3.252525, -128.3277, 1, 0.5843138, 0, 1,
4.848485, 3.252525, -128.0715, 1, 0.5843138, 0, 1,
4.888889, 3.252525, -127.823, 1, 0.6862745, 0, 1,
4.929293, 3.252525, -127.5822, 1, 0.6862745, 0, 1,
4.969697, 3.252525, -127.3491, 1, 0.6862745, 0, 1,
5.010101, 3.252525, -127.1237, 1, 0.6862745, 0, 1,
5.050505, 3.252525, -126.9061, 1, 0.6862745, 0, 1,
5.090909, 3.252525, -126.6961, 1, 0.7921569, 0, 1,
5.131313, 3.252525, -126.4939, 1, 0.7921569, 0, 1,
5.171717, 3.252525, -126.2994, 1, 0.7921569, 0, 1,
5.212121, 3.252525, -126.1126, 1, 0.7921569, 0, 1,
5.252525, 3.252525, -125.9335, 1, 0.7921569, 0, 1,
5.292929, 3.252525, -125.7622, 1, 0.7921569, 0, 1,
5.333333, 3.252525, -125.5985, 1, 0.7921569, 0, 1,
5.373737, 3.252525, -125.4426, 1, 0.8941177, 0, 1,
5.414141, 3.252525, -125.2944, 1, 0.8941177, 0, 1,
5.454545, 3.252525, -125.1539, 1, 0.8941177, 0, 1,
5.494949, 3.252525, -125.0211, 1, 0.8941177, 0, 1,
5.535354, 3.252525, -124.8961, 1, 0.8941177, 0, 1,
5.575758, 3.252525, -124.7787, 1, 0.8941177, 0, 1,
5.616162, 3.252525, -124.6691, 1, 0.8941177, 0, 1,
5.656566, 3.252525, -124.5672, 1, 0.8941177, 0, 1,
5.69697, 3.252525, -124.473, 1, 0.8941177, 0, 1,
5.737374, 3.252525, -124.3865, 1, 0.8941177, 0, 1,
5.777778, 3.252525, -124.3077, 1, 1, 0, 1,
5.818182, 3.252525, -124.2367, 1, 1, 0, 1,
5.858586, 3.252525, -124.1733, 1, 1, 0, 1,
5.89899, 3.252525, -124.1177, 1, 1, 0, 1,
5.939394, 3.252525, -124.0698, 1, 1, 0, 1,
5.979798, 3.252525, -124.0296, 1, 1, 0, 1,
6.020202, 3.252525, -123.9971, 1, 1, 0, 1,
6.060606, 3.252525, -123.9724, 1, 1, 0, 1,
6.10101, 3.252525, -123.9553, 1, 1, 0, 1,
6.141414, 3.252525, -123.946, 1, 1, 0, 1,
6.181818, 3.252525, -123.9444, 1, 1, 0, 1,
6.222222, 3.252525, -123.9505, 1, 1, 0, 1,
6.262626, 3.252525, -123.9643, 1, 1, 0, 1,
6.30303, 3.252525, -123.9858, 1, 1, 0, 1,
6.343434, 3.252525, -124.0151, 1, 1, 0, 1,
6.383838, 3.252525, -124.0521, 1, 1, 0, 1,
6.424242, 3.252525, -124.0967, 1, 1, 0, 1,
6.464646, 3.252525, -124.1491, 1, 1, 0, 1,
6.505051, 3.252525, -124.2093, 1, 1, 0, 1,
6.545455, 3.252525, -124.2771, 1, 1, 0, 1,
6.585859, 3.252525, -124.3526, 1, 0.8941177, 0, 1,
6.626263, 3.252525, -124.4359, 1, 0.8941177, 0, 1,
6.666667, 3.252525, -124.5269, 1, 0.8941177, 0, 1,
6.707071, 3.252525, -124.6256, 1, 0.8941177, 0, 1,
6.747475, 3.252525, -124.732, 1, 0.8941177, 0, 1,
6.787879, 3.252525, -124.8461, 1, 0.8941177, 0, 1,
6.828283, 3.252525, -124.9679, 1, 0.8941177, 0, 1,
6.868687, 3.252525, -125.0975, 1, 0.8941177, 0, 1,
6.909091, 3.252525, -125.2348, 1, 0.8941177, 0, 1,
6.949495, 3.252525, -125.3798, 1, 0.8941177, 0, 1,
6.989899, 3.252525, -125.5325, 1, 0.8941177, 0, 1,
7.030303, 3.252525, -125.6929, 1, 0.7921569, 0, 1,
7.070707, 3.252525, -125.861, 1, 0.7921569, 0, 1,
7.111111, 3.252525, -126.0369, 1, 0.7921569, 0, 1,
7.151515, 3.252525, -126.2204, 1, 0.7921569, 0, 1,
7.191919, 3.252525, -126.4117, 1, 0.7921569, 0, 1,
7.232323, 3.252525, -126.6107, 1, 0.7921569, 0, 1,
7.272727, 3.252525, -126.8174, 1, 0.7921569, 0, 1,
7.313131, 3.252525, -127.0318, 1, 0.6862745, 0, 1,
7.353535, 3.252525, -127.254, 1, 0.6862745, 0, 1,
7.393939, 3.252525, -127.4839, 1, 0.6862745, 0, 1,
7.434343, 3.252525, -127.7214, 1, 0.6862745, 0, 1,
7.474748, 3.252525, -127.9667, 1, 0.6862745, 0, 1,
7.515152, 3.252525, -128.2197, 1, 0.5843138, 0, 1,
7.555555, 3.252525, -128.4805, 1, 0.5843138, 0, 1,
7.59596, 3.252525, -128.7489, 1, 0.5843138, 0, 1,
7.636364, 3.252525, -129.0251, 1, 0.5843138, 0, 1,
7.676768, 3.252525, -129.3089, 1, 0.4823529, 0, 1,
7.717172, 3.252525, -129.6005, 1, 0.4823529, 0, 1,
7.757576, 3.252525, -129.8998, 1, 0.4823529, 0, 1,
7.79798, 3.252525, -130.2068, 1, 0.4823529, 0, 1,
7.838384, 3.252525, -130.5216, 1, 0.4823529, 0, 1,
7.878788, 3.252525, -130.844, 1, 0.3764706, 0, 1,
7.919192, 3.252525, -131.1742, 1, 0.3764706, 0, 1,
7.959596, 3.252525, -131.5121, 1, 0.3764706, 0, 1,
8, 3.252525, -131.8577, 1, 0.2745098, 0, 1,
4, 3.292929, -134.9545, 1, 0.06666667, 0, 1,
4.040404, 3.292929, -134.554, 1, 0.06666667, 0, 1,
4.080808, 3.292929, -134.161, 1, 0.1686275, 0, 1,
4.121212, 3.292929, -133.7755, 1, 0.1686275, 0, 1,
4.161616, 3.292929, -133.3976, 1, 0.1686275, 0, 1,
4.20202, 3.292929, -133.0271, 1, 0.2745098, 0, 1,
4.242424, 3.292929, -132.6642, 1, 0.2745098, 0, 1,
4.282828, 3.292929, -132.3089, 1, 0.2745098, 0, 1,
4.323232, 3.292929, -131.961, 1, 0.2745098, 0, 1,
4.363636, 3.292929, -131.6207, 1, 0.3764706, 0, 1,
4.40404, 3.292929, -131.2879, 1, 0.3764706, 0, 1,
4.444445, 3.292929, -130.9627, 1, 0.3764706, 0, 1,
4.484848, 3.292929, -130.6449, 1, 0.3764706, 0, 1,
4.525252, 3.292929, -130.3347, 1, 0.4823529, 0, 1,
4.565657, 3.292929, -130.0321, 1, 0.4823529, 0, 1,
4.606061, 3.292929, -129.7369, 1, 0.4823529, 0, 1,
4.646465, 3.292929, -129.4493, 1, 0.4823529, 0, 1,
4.686869, 3.292929, -129.1692, 1, 0.5843138, 0, 1,
4.727273, 3.292929, -128.8966, 1, 0.5843138, 0, 1,
4.767677, 3.292929, -128.6316, 1, 0.5843138, 0, 1,
4.808081, 3.292929, -128.3741, 1, 0.5843138, 0, 1,
4.848485, 3.292929, -128.1241, 1, 0.5843138, 0, 1,
4.888889, 3.292929, -127.8816, 1, 0.6862745, 0, 1,
4.929293, 3.292929, -127.6467, 1, 0.6862745, 0, 1,
4.969697, 3.292929, -127.4193, 1, 0.6862745, 0, 1,
5.010101, 3.292929, -127.1994, 1, 0.6862745, 0, 1,
5.050505, 3.292929, -126.9871, 1, 0.6862745, 0, 1,
5.090909, 3.292929, -126.7823, 1, 0.7921569, 0, 1,
5.131313, 3.292929, -126.585, 1, 0.7921569, 0, 1,
5.171717, 3.292929, -126.3952, 1, 0.7921569, 0, 1,
5.212121, 3.292929, -126.213, 1, 0.7921569, 0, 1,
5.252525, 3.292929, -126.0383, 1, 0.7921569, 0, 1,
5.292929, 3.292929, -125.8711, 1, 0.7921569, 0, 1,
5.333333, 3.292929, -125.7115, 1, 0.7921569, 0, 1,
5.373737, 3.292929, -125.5593, 1, 0.8941177, 0, 1,
5.414141, 3.292929, -125.4147, 1, 0.8941177, 0, 1,
5.454545, 3.292929, -125.2777, 1, 0.8941177, 0, 1,
5.494949, 3.292929, -125.1481, 1, 0.8941177, 0, 1,
5.535354, 3.292929, -125.0261, 1, 0.8941177, 0, 1,
5.575758, 3.292929, -124.9116, 1, 0.8941177, 0, 1,
5.616162, 3.292929, -124.8047, 1, 0.8941177, 0, 1,
5.656566, 3.292929, -124.7052, 1, 0.8941177, 0, 1,
5.69697, 3.292929, -124.6133, 1, 0.8941177, 0, 1,
5.737374, 3.292929, -124.529, 1, 0.8941177, 0, 1,
5.777778, 3.292929, -124.4521, 1, 0.8941177, 0, 1,
5.818182, 3.292929, -124.3828, 1, 0.8941177, 0, 1,
5.858586, 3.292929, -124.321, 1, 1, 0, 1,
5.89899, 3.292929, -124.2667, 1, 1, 0, 1,
5.939394, 3.292929, -124.22, 1, 1, 0, 1,
5.979798, 3.292929, -124.1808, 1, 1, 0, 1,
6.020202, 3.292929, -124.1491, 1, 1, 0, 1,
6.060606, 3.292929, -124.125, 1, 1, 0, 1,
6.10101, 3.292929, -124.1083, 1, 1, 0, 1,
6.141414, 3.292929, -124.0992, 1, 1, 0, 1,
6.181818, 3.292929, -124.0976, 1, 1, 0, 1,
6.222222, 3.292929, -124.1036, 1, 1, 0, 1,
6.262626, 3.292929, -124.1171, 1, 1, 0, 1,
6.30303, 3.292929, -124.1381, 1, 1, 0, 1,
6.343434, 3.292929, -124.1666, 1, 1, 0, 1,
6.383838, 3.292929, -124.2027, 1, 1, 0, 1,
6.424242, 3.292929, -124.2463, 1, 1, 0, 1,
6.464646, 3.292929, -124.2974, 1, 1, 0, 1,
6.505051, 3.292929, -124.3561, 1, 0.8941177, 0, 1,
6.545455, 3.292929, -124.4222, 1, 0.8941177, 0, 1,
6.585859, 3.292929, -124.4959, 1, 0.8941177, 0, 1,
6.626263, 3.292929, -124.5772, 1, 0.8941177, 0, 1,
6.666667, 3.292929, -124.6659, 1, 0.8941177, 0, 1,
6.707071, 3.292929, -124.7622, 1, 0.8941177, 0, 1,
6.747475, 3.292929, -124.866, 1, 0.8941177, 0, 1,
6.787879, 3.292929, -124.9774, 1, 0.8941177, 0, 1,
6.828283, 3.292929, -125.0962, 1, 0.8941177, 0, 1,
6.868687, 3.292929, -125.2226, 1, 0.8941177, 0, 1,
6.909091, 3.292929, -125.3566, 1, 0.8941177, 0, 1,
6.949495, 3.292929, -125.498, 1, 0.8941177, 0, 1,
6.989899, 3.292929, -125.647, 1, 0.7921569, 0, 1,
7.030303, 3.292929, -125.8035, 1, 0.7921569, 0, 1,
7.070707, 3.292929, -125.9675, 1, 0.7921569, 0, 1,
7.111111, 3.292929, -126.1391, 1, 0.7921569, 0, 1,
7.151515, 3.292929, -126.3182, 1, 0.7921569, 0, 1,
7.191919, 3.292929, -126.5048, 1, 0.7921569, 0, 1,
7.232323, 3.292929, -126.6989, 1, 0.7921569, 0, 1,
7.272727, 3.292929, -126.9006, 1, 0.6862745, 0, 1,
7.313131, 3.292929, -127.1098, 1, 0.6862745, 0, 1,
7.353535, 3.292929, -127.3265, 1, 0.6862745, 0, 1,
7.393939, 3.292929, -127.5508, 1, 0.6862745, 0, 1,
7.434343, 3.292929, -127.7826, 1, 0.6862745, 0, 1,
7.474748, 3.292929, -128.0219, 1, 0.6862745, 0, 1,
7.515152, 3.292929, -128.2687, 1, 0.5843138, 0, 1,
7.555555, 3.292929, -128.5231, 1, 0.5843138, 0, 1,
7.59596, 3.292929, -128.785, 1, 0.5843138, 0, 1,
7.636364, 3.292929, -129.0544, 1, 0.5843138, 0, 1,
7.676768, 3.292929, -129.3313, 1, 0.4823529, 0, 1,
7.717172, 3.292929, -129.6158, 1, 0.4823529, 0, 1,
7.757576, 3.292929, -129.9078, 1, 0.4823529, 0, 1,
7.79798, 3.292929, -130.2074, 1, 0.4823529, 0, 1,
7.838384, 3.292929, -130.5144, 1, 0.4823529, 0, 1,
7.878788, 3.292929, -130.829, 1, 0.3764706, 0, 1,
7.919192, 3.292929, -131.1511, 1, 0.3764706, 0, 1,
7.959596, 3.292929, -131.4807, 1, 0.3764706, 0, 1,
8, 3.292929, -131.8179, 1, 0.2745098, 0, 1,
4, 3.333333, -134.8554, 1, 0.06666667, 0, 1,
4.040404, 3.333333, -134.4645, 1, 0.06666667, 0, 1,
4.080808, 3.333333, -134.081, 1, 0.1686275, 0, 1,
4.121212, 3.333333, -133.7048, 1, 0.1686275, 0, 1,
4.161616, 3.333333, -133.336, 1, 0.1686275, 0, 1,
4.20202, 3.333333, -132.9745, 1, 0.2745098, 0, 1,
4.242424, 3.333333, -132.6203, 1, 0.2745098, 0, 1,
4.282828, 3.333333, -132.2735, 1, 0.2745098, 0, 1,
4.323232, 3.333333, -131.9341, 1, 0.2745098, 0, 1,
4.363636, 3.333333, -131.6019, 1, 0.3764706, 0, 1,
4.40404, 3.333333, -131.2772, 1, 0.3764706, 0, 1,
4.444445, 3.333333, -130.9598, 1, 0.3764706, 0, 1,
4.484848, 3.333333, -130.6497, 1, 0.3764706, 0, 1,
4.525252, 3.333333, -130.347, 1, 0.4823529, 0, 1,
4.565657, 3.333333, -130.0516, 1, 0.4823529, 0, 1,
4.606061, 3.333333, -129.7635, 1, 0.4823529, 0, 1,
4.646465, 3.333333, -129.4828, 1, 0.4823529, 0, 1,
4.686869, 3.333333, -129.2095, 1, 0.5843138, 0, 1,
4.727273, 3.333333, -128.9435, 1, 0.5843138, 0, 1,
4.767677, 3.333333, -128.6849, 1, 0.5843138, 0, 1,
4.808081, 3.333333, -128.4335, 1, 0.5843138, 0, 1,
4.848485, 3.333333, -128.1896, 1, 0.5843138, 0, 1,
4.888889, 3.333333, -127.953, 1, 0.6862745, 0, 1,
4.929293, 3.333333, -127.7237, 1, 0.6862745, 0, 1,
4.969697, 3.333333, -127.5018, 1, 0.6862745, 0, 1,
5.010101, 3.333333, -127.2872, 1, 0.6862745, 0, 1,
5.050505, 3.333333, -127.08, 1, 0.6862745, 0, 1,
5.090909, 3.333333, -126.8801, 1, 0.6862745, 0, 1,
5.131313, 3.333333, -126.6876, 1, 0.7921569, 0, 1,
5.171717, 3.333333, -126.5024, 1, 0.7921569, 0, 1,
5.212121, 3.333333, -126.3245, 1, 0.7921569, 0, 1,
5.252525, 3.333333, -126.154, 1, 0.7921569, 0, 1,
5.292929, 3.333333, -125.9909, 1, 0.7921569, 0, 1,
5.333333, 3.333333, -125.8351, 1, 0.7921569, 0, 1,
5.373737, 3.333333, -125.6866, 1, 0.7921569, 0, 1,
5.414141, 3.333333, -125.5455, 1, 0.8941177, 0, 1,
5.454545, 3.333333, -125.4118, 1, 0.8941177, 0, 1,
5.494949, 3.333333, -125.2853, 1, 0.8941177, 0, 1,
5.535354, 3.333333, -125.1663, 1, 0.8941177, 0, 1,
5.575758, 3.333333, -125.0545, 1, 0.8941177, 0, 1,
5.616162, 3.333333, -124.9501, 1, 0.8941177, 0, 1,
5.656566, 3.333333, -124.8531, 1, 0.8941177, 0, 1,
5.69697, 3.333333, -124.7634, 1, 0.8941177, 0, 1,
5.737374, 3.333333, -124.6811, 1, 0.8941177, 0, 1,
5.777778, 3.333333, -124.6061, 1, 0.8941177, 0, 1,
5.818182, 3.333333, -124.5384, 1, 0.8941177, 0, 1,
5.858586, 3.333333, -124.4781, 1, 0.8941177, 0, 1,
5.89899, 3.333333, -124.4252, 1, 0.8941177, 0, 1,
5.939394, 3.333333, -124.3796, 1, 0.8941177, 0, 1,
5.979798, 3.333333, -124.3413, 1, 0.8941177, 0, 1,
6.020202, 3.333333, -124.3104, 1, 1, 0, 1,
6.060606, 3.333333, -124.2868, 1, 1, 0, 1,
6.10101, 3.333333, -124.2706, 1, 1, 0, 1,
6.141414, 3.333333, -124.2617, 1, 1, 0, 1,
6.181818, 3.333333, -124.2602, 1, 1, 0, 1,
6.222222, 3.333333, -124.266, 1, 1, 0, 1,
6.262626, 3.333333, -124.2791, 1, 1, 0, 1,
6.30303, 3.333333, -124.2996, 1, 1, 0, 1,
6.343434, 3.333333, -124.3275, 1, 1, 0, 1,
6.383838, 3.333333, -124.3627, 1, 0.8941177, 0, 1,
6.424242, 3.333333, -124.4052, 1, 0.8941177, 0, 1,
6.464646, 3.333333, -124.4551, 1, 0.8941177, 0, 1,
6.505051, 3.333333, -124.5123, 1, 0.8941177, 0, 1,
6.545455, 3.333333, -124.5769, 1, 0.8941177, 0, 1,
6.585859, 3.333333, -124.6488, 1, 0.8941177, 0, 1,
6.626263, 3.333333, -124.7281, 1, 0.8941177, 0, 1,
6.666667, 3.333333, -124.8147, 1, 0.8941177, 0, 1,
6.707071, 3.333333, -124.9087, 1, 0.8941177, 0, 1,
6.747475, 3.333333, -125.01, 1, 0.8941177, 0, 1,
6.787879, 3.333333, -125.1187, 1, 0.8941177, 0, 1,
6.828283, 3.333333, -125.2347, 1, 0.8941177, 0, 1,
6.868687, 3.333333, -125.358, 1, 0.8941177, 0, 1,
6.909091, 3.333333, -125.4887, 1, 0.8941177, 0, 1,
6.949495, 3.333333, -125.6268, 1, 0.7921569, 0, 1,
6.989899, 3.333333, -125.7722, 1, 0.7921569, 0, 1,
7.030303, 3.333333, -125.9249, 1, 0.7921569, 0, 1,
7.070707, 3.333333, -126.085, 1, 0.7921569, 0, 1,
7.111111, 3.333333, -126.2524, 1, 0.7921569, 0, 1,
7.151515, 3.333333, -126.4272, 1, 0.7921569, 0, 1,
7.191919, 3.333333, -126.6093, 1, 0.7921569, 0, 1,
7.232323, 3.333333, -126.7988, 1, 0.7921569, 0, 1,
7.272727, 3.333333, -126.9956, 1, 0.6862745, 0, 1,
7.313131, 3.333333, -127.1997, 1, 0.6862745, 0, 1,
7.353535, 3.333333, -127.4112, 1, 0.6862745, 0, 1,
7.393939, 3.333333, -127.6301, 1, 0.6862745, 0, 1,
7.434343, 3.333333, -127.8563, 1, 0.6862745, 0, 1,
7.474748, 3.333333, -128.0898, 1, 0.5843138, 0, 1,
7.515152, 3.333333, -128.3307, 1, 0.5843138, 0, 1,
7.555555, 3.333333, -128.579, 1, 0.5843138, 0, 1,
7.59596, 3.333333, -128.8345, 1, 0.5843138, 0, 1,
7.636364, 3.333333, -129.0975, 1, 0.5843138, 0, 1,
7.676768, 3.333333, -129.3677, 1, 0.4823529, 0, 1,
7.717172, 3.333333, -129.6454, 1, 0.4823529, 0, 1,
7.757576, 3.333333, -129.9303, 1, 0.4823529, 0, 1,
7.79798, 3.333333, -130.2226, 1, 0.4823529, 0, 1,
7.838384, 3.333333, -130.5223, 1, 0.4823529, 0, 1,
7.878788, 3.333333, -130.8293, 1, 0.3764706, 0, 1,
7.919192, 3.333333, -131.1437, 1, 0.3764706, 0, 1,
7.959596, 3.333333, -131.4654, 1, 0.3764706, 0, 1,
8, 3.333333, -131.7944, 1, 0.2745098, 0, 1,
4, 3.373737, -134.7743, 1, 0.06666667, 0, 1,
4.040404, 3.373737, -134.3927, 1, 0.06666667, 0, 1,
4.080808, 3.373737, -134.0183, 1, 0.1686275, 0, 1,
4.121212, 3.373737, -133.6511, 1, 0.1686275, 0, 1,
4.161616, 3.373737, -133.291, 1, 0.1686275, 0, 1,
4.20202, 3.373737, -132.9381, 1, 0.2745098, 0, 1,
4.242424, 3.373737, -132.5924, 1, 0.2745098, 0, 1,
4.282828, 3.373737, -132.2539, 1, 0.2745098, 0, 1,
4.323232, 3.373737, -131.9225, 1, 0.2745098, 0, 1,
4.363636, 3.373737, -131.5983, 1, 0.3764706, 0, 1,
4.40404, 3.373737, -131.2812, 1, 0.3764706, 0, 1,
4.444445, 3.373737, -130.9714, 1, 0.3764706, 0, 1,
4.484848, 3.373737, -130.6687, 1, 0.3764706, 0, 1,
4.525252, 3.373737, -130.3732, 1, 0.4823529, 0, 1,
4.565657, 3.373737, -130.0848, 1, 0.4823529, 0, 1,
4.606061, 3.373737, -129.8036, 1, 0.4823529, 0, 1,
4.646465, 3.373737, -129.5296, 1, 0.4823529, 0, 1,
4.686869, 3.373737, -129.2628, 1, 0.5843138, 0, 1,
4.727273, 3.373737, -129.0031, 1, 0.5843138, 0, 1,
4.767677, 3.373737, -128.7506, 1, 0.5843138, 0, 1,
4.808081, 3.373737, -128.5053, 1, 0.5843138, 0, 1,
4.848485, 3.373737, -128.2672, 1, 0.5843138, 0, 1,
4.888889, 3.373737, -128.0362, 1, 0.6862745, 0, 1,
4.929293, 3.373737, -127.8124, 1, 0.6862745, 0, 1,
4.969697, 3.373737, -127.5957, 1, 0.6862745, 0, 1,
5.010101, 3.373737, -127.3863, 1, 0.6862745, 0, 1,
5.050505, 3.373737, -127.184, 1, 0.6862745, 0, 1,
5.090909, 3.373737, -126.9889, 1, 0.6862745, 0, 1,
5.131313, 3.373737, -126.8009, 1, 0.7921569, 0, 1,
5.171717, 3.373737, -126.6201, 1, 0.7921569, 0, 1,
5.212121, 3.373737, -126.4465, 1, 0.7921569, 0, 1,
5.252525, 3.373737, -126.2801, 1, 0.7921569, 0, 1,
5.292929, 3.373737, -126.1208, 1, 0.7921569, 0, 1,
5.333333, 3.373737, -125.9687, 1, 0.7921569, 0, 1,
5.373737, 3.373737, -125.8238, 1, 0.7921569, 0, 1,
5.414141, 3.373737, -125.686, 1, 0.7921569, 0, 1,
5.454545, 3.373737, -125.5555, 1, 0.8941177, 0, 1,
5.494949, 3.373737, -125.4321, 1, 0.8941177, 0, 1,
5.535354, 3.373737, -125.3158, 1, 0.8941177, 0, 1,
5.575758, 3.373737, -125.2067, 1, 0.8941177, 0, 1,
5.616162, 3.373737, -125.1049, 1, 0.8941177, 0, 1,
5.656566, 3.373737, -125.0101, 1, 0.8941177, 0, 1,
5.69697, 3.373737, -124.9226, 1, 0.8941177, 0, 1,
5.737374, 3.373737, -124.8422, 1, 0.8941177, 0, 1,
5.777778, 3.373737, -124.769, 1, 0.8941177, 0, 1,
5.818182, 3.373737, -124.7029, 1, 0.8941177, 0, 1,
5.858586, 3.373737, -124.6441, 1, 0.8941177, 0, 1,
5.89899, 3.373737, -124.5924, 1, 0.8941177, 0, 1,
5.939394, 3.373737, -124.5479, 1, 0.8941177, 0, 1,
5.979798, 3.373737, -124.5105, 1, 0.8941177, 0, 1,
6.020202, 3.373737, -124.4803, 1, 0.8941177, 0, 1,
6.060606, 3.373737, -124.4573, 1, 0.8941177, 0, 1,
6.10101, 3.373737, -124.4415, 1, 0.8941177, 0, 1,
6.141414, 3.373737, -124.4328, 1, 0.8941177, 0, 1,
6.181818, 3.373737, -124.4313, 1, 0.8941177, 0, 1,
6.222222, 3.373737, -124.437, 1, 0.8941177, 0, 1,
6.262626, 3.373737, -124.4498, 1, 0.8941177, 0, 1,
6.30303, 3.373737, -124.4698, 1, 0.8941177, 0, 1,
6.343434, 3.373737, -124.497, 1, 0.8941177, 0, 1,
6.383838, 3.373737, -124.5314, 1, 0.8941177, 0, 1,
6.424242, 3.373737, -124.5729, 1, 0.8941177, 0, 1,
6.464646, 3.373737, -124.6216, 1, 0.8941177, 0, 1,
6.505051, 3.373737, -124.6775, 1, 0.8941177, 0, 1,
6.545455, 3.373737, -124.7405, 1, 0.8941177, 0, 1,
6.585859, 3.373737, -124.8107, 1, 0.8941177, 0, 1,
6.626263, 3.373737, -124.8881, 1, 0.8941177, 0, 1,
6.666667, 3.373737, -124.9727, 1, 0.8941177, 0, 1,
6.707071, 3.373737, -125.0644, 1, 0.8941177, 0, 1,
6.747475, 3.373737, -125.1633, 1, 0.8941177, 0, 1,
6.787879, 3.373737, -125.2694, 1, 0.8941177, 0, 1,
6.828283, 3.373737, -125.3826, 1, 0.8941177, 0, 1,
6.868687, 3.373737, -125.503, 1, 0.8941177, 0, 1,
6.909091, 3.373737, -125.6306, 1, 0.7921569, 0, 1,
6.949495, 3.373737, -125.7654, 1, 0.7921569, 0, 1,
6.989899, 3.373737, -125.9073, 1, 0.7921569, 0, 1,
7.030303, 3.373737, -126.0564, 1, 0.7921569, 0, 1,
7.070707, 3.373737, -126.2127, 1, 0.7921569, 0, 1,
7.111111, 3.373737, -126.3761, 1, 0.7921569, 0, 1,
7.151515, 3.373737, -126.5467, 1, 0.7921569, 0, 1,
7.191919, 3.373737, -126.7245, 1, 0.7921569, 0, 1,
7.232323, 3.373737, -126.9095, 1, 0.6862745, 0, 1,
7.272727, 3.373737, -127.1016, 1, 0.6862745, 0, 1,
7.313131, 3.373737, -127.3009, 1, 0.6862745, 0, 1,
7.353535, 3.373737, -127.5074, 1, 0.6862745, 0, 1,
7.393939, 3.373737, -127.721, 1, 0.6862745, 0, 1,
7.434343, 3.373737, -127.9418, 1, 0.6862745, 0, 1,
7.474748, 3.373737, -128.1698, 1, 0.5843138, 0, 1,
7.515152, 3.373737, -128.4049, 1, 0.5843138, 0, 1,
7.555555, 3.373737, -128.6473, 1, 0.5843138, 0, 1,
7.59596, 3.373737, -128.8968, 1, 0.5843138, 0, 1,
7.636364, 3.373737, -129.1534, 1, 0.5843138, 0, 1,
7.676768, 3.373737, -129.4173, 1, 0.4823529, 0, 1,
7.717172, 3.373737, -129.6883, 1, 0.4823529, 0, 1,
7.757576, 3.373737, -129.9665, 1, 0.4823529, 0, 1,
7.79798, 3.373737, -130.2518, 1, 0.4823529, 0, 1,
7.838384, 3.373737, -130.5443, 1, 0.4823529, 0, 1,
7.878788, 3.373737, -130.844, 1, 0.3764706, 0, 1,
7.919192, 3.373737, -131.1509, 1, 0.3764706, 0, 1,
7.959596, 3.373737, -131.4649, 1, 0.3764706, 0, 1,
8, 3.373737, -131.7861, 1, 0.3764706, 0, 1,
4, 3.414141, -134.7101, 1, 0.06666667, 0, 1,
4.040404, 3.414141, -134.3375, 1, 0.06666667, 0, 1,
4.080808, 3.414141, -133.9719, 1, 0.1686275, 0, 1,
4.121212, 3.414141, -133.6133, 1, 0.1686275, 0, 1,
4.161616, 3.414141, -133.2617, 1, 0.1686275, 0, 1,
4.20202, 3.414141, -132.9172, 1, 0.2745098, 0, 1,
4.242424, 3.414141, -132.5796, 1, 0.2745098, 0, 1,
4.282828, 3.414141, -132.249, 1, 0.2745098, 0, 1,
4.323232, 3.414141, -131.9254, 1, 0.2745098, 0, 1,
4.363636, 3.414141, -131.6088, 1, 0.3764706, 0, 1,
4.40404, 3.414141, -131.2993, 1, 0.3764706, 0, 1,
4.444445, 3.414141, -130.9967, 1, 0.3764706, 0, 1,
4.484848, 3.414141, -130.7011, 1, 0.3764706, 0, 1,
4.525252, 3.414141, -130.4126, 1, 0.4823529, 0, 1,
4.565657, 3.414141, -130.131, 1, 0.4823529, 0, 1,
4.606061, 3.414141, -129.8564, 1, 0.4823529, 0, 1,
4.646465, 3.414141, -129.5889, 1, 0.4823529, 0, 1,
4.686869, 3.414141, -129.3283, 1, 0.4823529, 0, 1,
4.727273, 3.414141, -129.0748, 1, 0.5843138, 0, 1,
4.767677, 3.414141, -128.8282, 1, 0.5843138, 0, 1,
4.808081, 3.414141, -128.5886, 1, 0.5843138, 0, 1,
4.848485, 3.414141, -128.3561, 1, 0.5843138, 0, 1,
4.888889, 3.414141, -128.1306, 1, 0.5843138, 0, 1,
4.929293, 3.414141, -127.912, 1, 0.6862745, 0, 1,
4.969697, 3.414141, -127.7005, 1, 0.6862745, 0, 1,
5.010101, 3.414141, -127.4959, 1, 0.6862745, 0, 1,
5.050505, 3.414141, -127.2984, 1, 0.6862745, 0, 1,
5.090909, 3.414141, -127.1079, 1, 0.6862745, 0, 1,
5.131313, 3.414141, -126.9243, 1, 0.6862745, 0, 1,
5.171717, 3.414141, -126.7478, 1, 0.7921569, 0, 1,
5.212121, 3.414141, -126.5783, 1, 0.7921569, 0, 1,
5.252525, 3.414141, -126.4158, 1, 0.7921569, 0, 1,
5.292929, 3.414141, -126.2602, 1, 0.7921569, 0, 1,
5.333333, 3.414141, -126.1117, 1, 0.7921569, 0, 1,
5.373737, 3.414141, -125.9702, 1, 0.7921569, 0, 1,
5.414141, 3.414141, -125.8357, 1, 0.7921569, 0, 1,
5.454545, 3.414141, -125.7082, 1, 0.7921569, 0, 1,
5.494949, 3.414141, -125.5877, 1, 0.7921569, 0, 1,
5.535354, 3.414141, -125.4742, 1, 0.8941177, 0, 1,
5.575758, 3.414141, -125.3677, 1, 0.8941177, 0, 1,
5.616162, 3.414141, -125.2682, 1, 0.8941177, 0, 1,
5.656566, 3.414141, -125.1757, 1, 0.8941177, 0, 1,
5.69697, 3.414141, -125.0902, 1, 0.8941177, 0, 1,
5.737374, 3.414141, -125.0117, 1, 0.8941177, 0, 1,
5.777778, 3.414141, -124.9402, 1, 0.8941177, 0, 1,
5.818182, 3.414141, -124.8757, 1, 0.8941177, 0, 1,
5.858586, 3.414141, -124.8182, 1, 0.8941177, 0, 1,
5.89899, 3.414141, -124.7678, 1, 0.8941177, 0, 1,
5.939394, 3.414141, -124.7243, 1, 0.8941177, 0, 1,
5.979798, 3.414141, -124.6878, 1, 0.8941177, 0, 1,
6.020202, 3.414141, -124.6583, 1, 0.8941177, 0, 1,
6.060606, 3.414141, -124.6359, 1, 0.8941177, 0, 1,
6.10101, 3.414141, -124.6204, 1, 0.8941177, 0, 1,
6.141414, 3.414141, -124.6119, 1, 0.8941177, 0, 1,
6.181818, 3.414141, -124.6105, 1, 0.8941177, 0, 1,
6.222222, 3.414141, -124.616, 1, 0.8941177, 0, 1,
6.262626, 3.414141, -124.6286, 1, 0.8941177, 0, 1,
6.30303, 3.414141, -124.6481, 1, 0.8941177, 0, 1,
6.343434, 3.414141, -124.6746, 1, 0.8941177, 0, 1,
6.383838, 3.414141, -124.7082, 1, 0.8941177, 0, 1,
6.424242, 3.414141, -124.7487, 1, 0.8941177, 0, 1,
6.464646, 3.414141, -124.7963, 1, 0.8941177, 0, 1,
6.505051, 3.414141, -124.8509, 1, 0.8941177, 0, 1,
6.545455, 3.414141, -124.9124, 1, 0.8941177, 0, 1,
6.585859, 3.414141, -124.981, 1, 0.8941177, 0, 1,
6.626263, 3.414141, -125.0565, 1, 0.8941177, 0, 1,
6.666667, 3.414141, -125.1391, 1, 0.8941177, 0, 1,
6.707071, 3.414141, -125.2287, 1, 0.8941177, 0, 1,
6.747475, 3.414141, -125.3253, 1, 0.8941177, 0, 1,
6.787879, 3.414141, -125.4288, 1, 0.8941177, 0, 1,
6.828283, 3.414141, -125.5394, 1, 0.8941177, 0, 1,
6.868687, 3.414141, -125.657, 1, 0.7921569, 0, 1,
6.909091, 3.414141, -125.7816, 1, 0.7921569, 0, 1,
6.949495, 3.414141, -125.9132, 1, 0.7921569, 0, 1,
6.989899, 3.414141, -126.0518, 1, 0.7921569, 0, 1,
7.030303, 3.414141, -126.1973, 1, 0.7921569, 0, 1,
7.070707, 3.414141, -126.3499, 1, 0.7921569, 0, 1,
7.111111, 3.414141, -126.5095, 1, 0.7921569, 0, 1,
7.151515, 3.414141, -126.6761, 1, 0.7921569, 0, 1,
7.191919, 3.414141, -126.8497, 1, 0.6862745, 0, 1,
7.232323, 3.414141, -127.0303, 1, 0.6862745, 0, 1,
7.272727, 3.414141, -127.2179, 1, 0.6862745, 0, 1,
7.313131, 3.414141, -127.4125, 1, 0.6862745, 0, 1,
7.353535, 3.414141, -127.6142, 1, 0.6862745, 0, 1,
7.393939, 3.414141, -127.8228, 1, 0.6862745, 0, 1,
7.434343, 3.414141, -128.0384, 1, 0.6862745, 0, 1,
7.474748, 3.414141, -128.261, 1, 0.5843138, 0, 1,
7.515152, 3.414141, -128.4906, 1, 0.5843138, 0, 1,
7.555555, 3.414141, -128.7272, 1, 0.5843138, 0, 1,
7.59596, 3.414141, -128.9709, 1, 0.5843138, 0, 1,
7.636364, 3.414141, -129.2215, 1, 0.5843138, 0, 1,
7.676768, 3.414141, -129.4791, 1, 0.4823529, 0, 1,
7.717172, 3.414141, -129.7438, 1, 0.4823529, 0, 1,
7.757576, 3.414141, -130.0154, 1, 0.4823529, 0, 1,
7.79798, 3.414141, -130.2941, 1, 0.4823529, 0, 1,
7.838384, 3.414141, -130.5797, 1, 0.3764706, 0, 1,
7.878788, 3.414141, -130.8723, 1, 0.3764706, 0, 1,
7.919192, 3.414141, -131.172, 1, 0.3764706, 0, 1,
7.959596, 3.414141, -131.4786, 1, 0.3764706, 0, 1,
8, 3.414141, -131.7923, 1, 0.2745098, 0, 1,
4, 3.454545, -134.662, 1, 0.06666667, 0, 1,
4.040404, 3.454545, -134.298, 1, 0.06666667, 0, 1,
4.080808, 3.454545, -133.9409, 1, 0.1686275, 0, 1,
4.121212, 3.454545, -133.5907, 1, 0.1686275, 0, 1,
4.161616, 3.454545, -133.2473, 1, 0.1686275, 0, 1,
4.20202, 3.454545, -132.9107, 1, 0.2745098, 0, 1,
4.242424, 3.454545, -132.5809, 1, 0.2745098, 0, 1,
4.282828, 3.454545, -132.2581, 1, 0.2745098, 0, 1,
4.323232, 3.454545, -131.942, 1, 0.2745098, 0, 1,
4.363636, 3.454545, -131.6328, 1, 0.3764706, 0, 1,
4.40404, 3.454545, -131.3304, 1, 0.3764706, 0, 1,
4.444445, 3.454545, -131.0349, 1, 0.3764706, 0, 1,
4.484848, 3.454545, -130.7462, 1, 0.3764706, 0, 1,
4.525252, 3.454545, -130.4643, 1, 0.4823529, 0, 1,
4.565657, 3.454545, -130.1893, 1, 0.4823529, 0, 1,
4.606061, 3.454545, -129.9211, 1, 0.4823529, 0, 1,
4.646465, 3.454545, -129.6598, 1, 0.4823529, 0, 1,
4.686869, 3.454545, -129.4053, 1, 0.4823529, 0, 1,
4.727273, 3.454545, -129.1576, 1, 0.5843138, 0, 1,
4.767677, 3.454545, -128.9168, 1, 0.5843138, 0, 1,
4.808081, 3.454545, -128.6828, 1, 0.5843138, 0, 1,
4.848485, 3.454545, -128.4557, 1, 0.5843138, 0, 1,
4.888889, 3.454545, -128.2354, 1, 0.5843138, 0, 1,
4.929293, 3.454545, -128.0219, 1, 0.6862745, 0, 1,
4.969697, 3.454545, -127.8153, 1, 0.6862745, 0, 1,
5.010101, 3.454545, -127.6155, 1, 0.6862745, 0, 1,
5.050505, 3.454545, -127.4226, 1, 0.6862745, 0, 1,
5.090909, 3.454545, -127.2365, 1, 0.6862745, 0, 1,
5.131313, 3.454545, -127.0572, 1, 0.6862745, 0, 1,
5.171717, 3.454545, -126.8848, 1, 0.6862745, 0, 1,
5.212121, 3.454545, -126.7192, 1, 0.7921569, 0, 1,
5.252525, 3.454545, -126.5605, 1, 0.7921569, 0, 1,
5.292929, 3.454545, -126.4086, 1, 0.7921569, 0, 1,
5.333333, 3.454545, -126.2635, 1, 0.7921569, 0, 1,
5.373737, 3.454545, -126.1253, 1, 0.7921569, 0, 1,
5.414141, 3.454545, -125.9939, 1, 0.7921569, 0, 1,
5.454545, 3.454545, -125.8694, 1, 0.7921569, 0, 1,
5.494949, 3.454545, -125.7517, 1, 0.7921569, 0, 1,
5.535354, 3.454545, -125.6408, 1, 0.7921569, 0, 1,
5.575758, 3.454545, -125.5368, 1, 0.8941177, 0, 1,
5.616162, 3.454545, -125.4396, 1, 0.8941177, 0, 1,
5.656566, 3.454545, -125.3492, 1, 0.8941177, 0, 1,
5.69697, 3.454545, -125.2657, 1, 0.8941177, 0, 1,
5.737374, 3.454545, -125.1891, 1, 0.8941177, 0, 1,
5.777778, 3.454545, -125.1192, 1, 0.8941177, 0, 1,
5.818182, 3.454545, -125.0563, 1, 0.8941177, 0, 1,
5.858586, 3.454545, -125.0001, 1, 0.8941177, 0, 1,
5.89899, 3.454545, -124.9508, 1, 0.8941177, 0, 1,
5.939394, 3.454545, -124.9083, 1, 0.8941177, 0, 1,
5.979798, 3.454545, -124.8727, 1, 0.8941177, 0, 1,
6.020202, 3.454545, -124.8439, 1, 0.8941177, 0, 1,
6.060606, 3.454545, -124.822, 1, 0.8941177, 0, 1,
6.10101, 3.454545, -124.8069, 1, 0.8941177, 0, 1,
6.141414, 3.454545, -124.7986, 1, 0.8941177, 0, 1,
6.181818, 3.454545, -124.7972, 1, 0.8941177, 0, 1,
6.222222, 3.454545, -124.8026, 1, 0.8941177, 0, 1,
6.262626, 3.454545, -124.8148, 1, 0.8941177, 0, 1,
6.30303, 3.454545, -124.8339, 1, 0.8941177, 0, 1,
6.343434, 3.454545, -124.8598, 1, 0.8941177, 0, 1,
6.383838, 3.454545, -124.8926, 1, 0.8941177, 0, 1,
6.424242, 3.454545, -124.9322, 1, 0.8941177, 0, 1,
6.464646, 3.454545, -124.9787, 1, 0.8941177, 0, 1,
6.505051, 3.454545, -125.032, 1, 0.8941177, 0, 1,
6.545455, 3.454545, -125.0921, 1, 0.8941177, 0, 1,
6.585859, 3.454545, -125.1591, 1, 0.8941177, 0, 1,
6.626263, 3.454545, -125.2329, 1, 0.8941177, 0, 1,
6.666667, 3.454545, -125.3135, 1, 0.8941177, 0, 1,
6.707071, 3.454545, -125.401, 1, 0.8941177, 0, 1,
6.747475, 3.454545, -125.4953, 1, 0.8941177, 0, 1,
6.787879, 3.454545, -125.5965, 1, 0.7921569, 0, 1,
6.828283, 3.454545, -125.7045, 1, 0.7921569, 0, 1,
6.868687, 3.454545, -125.8194, 1, 0.7921569, 0, 1,
6.909091, 3.454545, -125.941, 1, 0.7921569, 0, 1,
6.949495, 3.454545, -126.0696, 1, 0.7921569, 0, 1,
6.989899, 3.454545, -126.2049, 1, 0.7921569, 0, 1,
7.030303, 3.454545, -126.3471, 1, 0.7921569, 0, 1,
7.070707, 3.454545, -126.4962, 1, 0.7921569, 0, 1,
7.111111, 3.454545, -126.6521, 1, 0.7921569, 0, 1,
7.151515, 3.454545, -126.8148, 1, 0.7921569, 0, 1,
7.191919, 3.454545, -126.9843, 1, 0.6862745, 0, 1,
7.232323, 3.454545, -127.1608, 1, 0.6862745, 0, 1,
7.272727, 3.454545, -127.344, 1, 0.6862745, 0, 1,
7.313131, 3.454545, -127.5341, 1, 0.6862745, 0, 1,
7.353535, 3.454545, -127.731, 1, 0.6862745, 0, 1,
7.393939, 3.454545, -127.9348, 1, 0.6862745, 0, 1,
7.434343, 3.454545, -128.1454, 1, 0.5843138, 0, 1,
7.474748, 3.454545, -128.3628, 1, 0.5843138, 0, 1,
7.515152, 3.454545, -128.5871, 1, 0.5843138, 0, 1,
7.555555, 3.454545, -128.8182, 1, 0.5843138, 0, 1,
7.59596, 3.454545, -129.0562, 1, 0.5843138, 0, 1,
7.636364, 3.454545, -129.301, 1, 0.5843138, 0, 1,
7.676768, 3.454545, -129.5526, 1, 0.4823529, 0, 1,
7.717172, 3.454545, -129.8111, 1, 0.4823529, 0, 1,
7.757576, 3.454545, -130.0764, 1, 0.4823529, 0, 1,
7.79798, 3.454545, -130.3486, 1, 0.4823529, 0, 1,
7.838384, 3.454545, -130.6276, 1, 0.3764706, 0, 1,
7.878788, 3.454545, -130.9134, 1, 0.3764706, 0, 1,
7.919192, 3.454545, -131.2061, 1, 0.3764706, 0, 1,
7.959596, 3.454545, -131.5056, 1, 0.3764706, 0, 1,
8, 3.454545, -131.812, 1, 0.2745098, 0, 1,
4, 3.49495, -134.6289, 1, 0.06666667, 0, 1,
4.040404, 3.49495, -134.2733, 1, 0.06666667, 0, 1,
4.080808, 3.49495, -133.9244, 1, 0.1686275, 0, 1,
4.121212, 3.49495, -133.5822, 1, 0.1686275, 0, 1,
4.161616, 3.49495, -133.2467, 1, 0.1686275, 0, 1,
4.20202, 3.49495, -132.9179, 1, 0.2745098, 0, 1,
4.242424, 3.49495, -132.5957, 1, 0.2745098, 0, 1,
4.282828, 3.49495, -132.2802, 1, 0.2745098, 0, 1,
4.323232, 3.49495, -131.9715, 1, 0.2745098, 0, 1,
4.363636, 3.49495, -131.6693, 1, 0.3764706, 0, 1,
4.40404, 3.49495, -131.3739, 1, 0.3764706, 0, 1,
4.444445, 3.49495, -131.0852, 1, 0.3764706, 0, 1,
4.484848, 3.49495, -130.8031, 1, 0.3764706, 0, 1,
4.525252, 3.49495, -130.5277, 1, 0.4823529, 0, 1,
4.565657, 3.49495, -130.259, 1, 0.4823529, 0, 1,
4.606061, 3.49495, -129.997, 1, 0.4823529, 0, 1,
4.646465, 3.49495, -129.7417, 1, 0.4823529, 0, 1,
4.686869, 3.49495, -129.493, 1, 0.4823529, 0, 1,
4.727273, 3.49495, -129.2511, 1, 0.5843138, 0, 1,
4.767677, 3.49495, -129.0158, 1, 0.5843138, 0, 1,
4.808081, 3.49495, -128.7872, 1, 0.5843138, 0, 1,
4.848485, 3.49495, -128.5653, 1, 0.5843138, 0, 1,
4.888889, 3.49495, -128.3501, 1, 0.5843138, 0, 1,
4.929293, 3.49495, -128.1415, 1, 0.5843138, 0, 1,
4.969697, 3.49495, -127.9396, 1, 0.6862745, 0, 1,
5.010101, 3.49495, -127.7444, 1, 0.6862745, 0, 1,
5.050505, 3.49495, -127.5559, 1, 0.6862745, 0, 1,
5.090909, 3.49495, -127.3741, 1, 0.6862745, 0, 1,
5.131313, 3.49495, -127.199, 1, 0.6862745, 0, 1,
5.171717, 3.49495, -127.0305, 1, 0.6862745, 0, 1,
5.212121, 3.49495, -126.8687, 1, 0.6862745, 0, 1,
5.252525, 3.49495, -126.7136, 1, 0.7921569, 0, 1,
5.292929, 3.49495, -126.5652, 1, 0.7921569, 0, 1,
5.333333, 3.49495, -126.4235, 1, 0.7921569, 0, 1,
5.373737, 3.49495, -126.2885, 1, 0.7921569, 0, 1,
5.414141, 3.49495, -126.1601, 1, 0.7921569, 0, 1,
5.454545, 3.49495, -126.0384, 1, 0.7921569, 0, 1,
5.494949, 3.49495, -125.9234, 1, 0.7921569, 0, 1,
5.535354, 3.49495, -125.8151, 1, 0.7921569, 0, 1,
5.575758, 3.49495, -125.7135, 1, 0.7921569, 0, 1,
5.616162, 3.49495, -125.6185, 1, 0.7921569, 0, 1,
5.656566, 3.49495, -125.5302, 1, 0.8941177, 0, 1,
5.69697, 3.49495, -125.4487, 1, 0.8941177, 0, 1,
5.737374, 3.49495, -125.3738, 1, 0.8941177, 0, 1,
5.777778, 3.49495, -125.3055, 1, 0.8941177, 0, 1,
5.818182, 3.49495, -125.244, 1, 0.8941177, 0, 1,
5.858586, 3.49495, -125.1891, 1, 0.8941177, 0, 1,
5.89899, 3.49495, -125.141, 1, 0.8941177, 0, 1,
5.939394, 3.49495, -125.0995, 1, 0.8941177, 0, 1,
5.979798, 3.49495, -125.0647, 1, 0.8941177, 0, 1,
6.020202, 3.49495, -125.0365, 1, 0.8941177, 0, 1,
6.060606, 3.49495, -125.0151, 1, 0.8941177, 0, 1,
6.10101, 3.49495, -125.0003, 1, 0.8941177, 0, 1,
6.141414, 3.49495, -124.9923, 1, 0.8941177, 0, 1,
6.181818, 3.49495, -124.9909, 1, 0.8941177, 0, 1,
6.222222, 3.49495, -124.9961, 1, 0.8941177, 0, 1,
6.262626, 3.49495, -125.0081, 1, 0.8941177, 0, 1,
6.30303, 3.49495, -125.0268, 1, 0.8941177, 0, 1,
6.343434, 3.49495, -125.0521, 1, 0.8941177, 0, 1,
6.383838, 3.49495, -125.0841, 1, 0.8941177, 0, 1,
6.424242, 3.49495, -125.1228, 1, 0.8941177, 0, 1,
6.464646, 3.49495, -125.1682, 1, 0.8941177, 0, 1,
6.505051, 3.49495, -125.2203, 1, 0.8941177, 0, 1,
6.545455, 3.49495, -125.279, 1, 0.8941177, 0, 1,
6.585859, 3.49495, -125.3444, 1, 0.8941177, 0, 1,
6.626263, 3.49495, -125.4165, 1, 0.8941177, 0, 1,
6.666667, 3.49495, -125.4953, 1, 0.8941177, 0, 1,
6.707071, 3.49495, -125.5808, 1, 0.7921569, 0, 1,
6.747475, 3.49495, -125.673, 1, 0.7921569, 0, 1,
6.787879, 3.49495, -125.7718, 1, 0.7921569, 0, 1,
6.828283, 3.49495, -125.8773, 1, 0.7921569, 0, 1,
6.868687, 3.49495, -125.9895, 1, 0.7921569, 0, 1,
6.909091, 3.49495, -126.1084, 1, 0.7921569, 0, 1,
6.949495, 3.49495, -126.234, 1, 0.7921569, 0, 1,
6.989899, 3.49495, -126.3663, 1, 0.7921569, 0, 1,
7.030303, 3.49495, -126.5052, 1, 0.7921569, 0, 1,
7.070707, 3.49495, -126.6508, 1, 0.7921569, 0, 1,
7.111111, 3.49495, -126.8031, 1, 0.7921569, 0, 1,
7.151515, 3.49495, -126.9621, 1, 0.6862745, 0, 1,
7.191919, 3.49495, -127.1278, 1, 0.6862745, 0, 1,
7.232323, 3.49495, -127.3001, 1, 0.6862745, 0, 1,
7.272727, 3.49495, -127.4791, 1, 0.6862745, 0, 1,
7.313131, 3.49495, -127.6649, 1, 0.6862745, 0, 1,
7.353535, 3.49495, -127.8573, 1, 0.6862745, 0, 1,
7.393939, 3.49495, -128.0563, 1, 0.6862745, 0, 1,
7.434343, 3.49495, -128.2621, 1, 0.5843138, 0, 1,
7.474748, 3.49495, -128.4745, 1, 0.5843138, 0, 1,
7.515152, 3.49495, -128.6937, 1, 0.5843138, 0, 1,
7.555555, 3.49495, -128.9195, 1, 0.5843138, 0, 1,
7.59596, 3.49495, -129.152, 1, 0.5843138, 0, 1,
7.636364, 3.49495, -129.3911, 1, 0.4823529, 0, 1,
7.676768, 3.49495, -129.637, 1, 0.4823529, 0, 1,
7.717172, 3.49495, -129.8895, 1, 0.4823529, 0, 1,
7.757576, 3.49495, -130.1488, 1, 0.4823529, 0, 1,
7.79798, 3.49495, -130.4147, 1, 0.4823529, 0, 1,
7.838384, 3.49495, -130.6872, 1, 0.3764706, 0, 1,
7.878788, 3.49495, -130.9665, 1, 0.3764706, 0, 1,
7.919192, 3.49495, -131.2525, 1, 0.3764706, 0, 1,
7.959596, 3.49495, -131.5451, 1, 0.3764706, 0, 1,
8, 3.49495, -131.8444, 1, 0.2745098, 0, 1,
4, 3.535353, -134.6101, 1, 0.06666667, 0, 1,
4.040404, 3.535353, -134.2626, 1, 0.1686275, 0, 1,
4.080808, 3.535353, -133.9216, 1, 0.1686275, 0, 1,
4.121212, 3.535353, -133.5872, 1, 0.1686275, 0, 1,
4.161616, 3.535353, -133.2593, 1, 0.1686275, 0, 1,
4.20202, 3.535353, -132.9379, 1, 0.2745098, 0, 1,
4.242424, 3.535353, -132.6231, 1, 0.2745098, 0, 1,
4.282828, 3.535353, -132.3148, 1, 0.2745098, 0, 1,
4.323232, 3.535353, -132.013, 1, 0.2745098, 0, 1,
4.363636, 3.535353, -131.7178, 1, 0.3764706, 0, 1,
4.40404, 3.535353, -131.4291, 1, 0.3764706, 0, 1,
4.444445, 3.535353, -131.1469, 1, 0.3764706, 0, 1,
4.484848, 3.535353, -130.8712, 1, 0.3764706, 0, 1,
4.525252, 3.535353, -130.6021, 1, 0.3764706, 0, 1,
4.565657, 3.535353, -130.3395, 1, 0.4823529, 0, 1,
4.606061, 3.535353, -130.0835, 1, 0.4823529, 0, 1,
4.646465, 3.535353, -129.834, 1, 0.4823529, 0, 1,
4.686869, 3.535353, -129.591, 1, 0.4823529, 0, 1,
4.727273, 3.535353, -129.3545, 1, 0.4823529, 0, 1,
4.767677, 3.535353, -129.1246, 1, 0.5843138, 0, 1,
4.808081, 3.535353, -128.9012, 1, 0.5843138, 0, 1,
4.848485, 3.535353, -128.6843, 1, 0.5843138, 0, 1,
4.888889, 3.535353, -128.4739, 1, 0.5843138, 0, 1,
4.929293, 3.535353, -128.2701, 1, 0.5843138, 0, 1,
4.969697, 3.535353, -128.0728, 1, 0.5843138, 0, 1,
5.010101, 3.535353, -127.8821, 1, 0.6862745, 0, 1,
5.050505, 3.535353, -127.6979, 1, 0.6862745, 0, 1,
5.090909, 3.535353, -127.5202, 1, 0.6862745, 0, 1,
5.131313, 3.535353, -127.349, 1, 0.6862745, 0, 1,
5.171717, 3.535353, -127.1844, 1, 0.6862745, 0, 1,
5.212121, 3.535353, -127.0263, 1, 0.6862745, 0, 1,
5.252525, 3.535353, -126.8747, 1, 0.6862745, 0, 1,
5.292929, 3.535353, -126.7297, 1, 0.7921569, 0, 1,
5.333333, 3.535353, -126.5912, 1, 0.7921569, 0, 1,
5.373737, 3.535353, -126.4592, 1, 0.7921569, 0, 1,
5.414141, 3.535353, -126.3337, 1, 0.7921569, 0, 1,
5.454545, 3.535353, -126.2148, 1, 0.7921569, 0, 1,
5.494949, 3.535353, -126.1024, 1, 0.7921569, 0, 1,
5.535354, 3.535353, -125.9966, 1, 0.7921569, 0, 1,
5.575758, 3.535353, -125.8973, 1, 0.7921569, 0, 1,
5.616162, 3.535353, -125.8045, 1, 0.7921569, 0, 1,
5.656566, 3.535353, -125.7182, 1, 0.7921569, 0, 1,
5.69697, 3.535353, -125.6385, 1, 0.7921569, 0, 1,
5.737374, 3.535353, -125.5653, 1, 0.8941177, 0, 1,
5.777778, 3.535353, -125.4986, 1, 0.8941177, 0, 1,
5.818182, 3.535353, -125.4385, 1, 0.8941177, 0, 1,
5.858586, 3.535353, -125.3849, 1, 0.8941177, 0, 1,
5.89899, 3.535353, -125.3378, 1, 0.8941177, 0, 1,
5.939394, 3.535353, -125.2972, 1, 0.8941177, 0, 1,
5.979798, 3.535353, -125.2632, 1, 0.8941177, 0, 1,
6.020202, 3.535353, -125.2357, 1, 0.8941177, 0, 1,
6.060606, 3.535353, -125.2148, 1, 0.8941177, 0, 1,
6.10101, 3.535353, -125.2004, 1, 0.8941177, 0, 1,
6.141414, 3.535353, -125.1925, 1, 0.8941177, 0, 1,
6.181818, 3.535353, -125.1911, 1, 0.8941177, 0, 1,
6.222222, 3.535353, -125.1963, 1, 0.8941177, 0, 1,
6.262626, 3.535353, -125.208, 1, 0.8941177, 0, 1,
6.30303, 3.535353, -125.2262, 1, 0.8941177, 0, 1,
6.343434, 3.535353, -125.2509, 1, 0.8941177, 0, 1,
6.383838, 3.535353, -125.2822, 1, 0.8941177, 0, 1,
6.424242, 3.535353, -125.3201, 1, 0.8941177, 0, 1,
6.464646, 3.535353, -125.3644, 1, 0.8941177, 0, 1,
6.505051, 3.535353, -125.4153, 1, 0.8941177, 0, 1,
6.545455, 3.535353, -125.4727, 1, 0.8941177, 0, 1,
6.585859, 3.535353, -125.5366, 1, 0.8941177, 0, 1,
6.626263, 3.535353, -125.6071, 1, 0.7921569, 0, 1,
6.666667, 3.535353, -125.6841, 1, 0.7921569, 0, 1,
6.707071, 3.535353, -125.7676, 1, 0.7921569, 0, 1,
6.747475, 3.535353, -125.8577, 1, 0.7921569, 0, 1,
6.787879, 3.535353, -125.9543, 1, 0.7921569, 0, 1,
6.828283, 3.535353, -126.0574, 1, 0.7921569, 0, 1,
6.868687, 3.535353, -126.1671, 1, 0.7921569, 0, 1,
6.909091, 3.535353, -126.2833, 1, 0.7921569, 0, 1,
6.949495, 3.535353, -126.406, 1, 0.7921569, 0, 1,
6.989899, 3.535353, -126.5352, 1, 0.7921569, 0, 1,
7.030303, 3.535353, -126.671, 1, 0.7921569, 0, 1,
7.070707, 3.535353, -126.8133, 1, 0.7921569, 0, 1,
7.111111, 3.535353, -126.9622, 1, 0.6862745, 0, 1,
7.151515, 3.535353, -127.1175, 1, 0.6862745, 0, 1,
7.191919, 3.535353, -127.2794, 1, 0.6862745, 0, 1,
7.232323, 3.535353, -127.4479, 1, 0.6862745, 0, 1,
7.272727, 3.535353, -127.6228, 1, 0.6862745, 0, 1,
7.313131, 3.535353, -127.8043, 1, 0.6862745, 0, 1,
7.353535, 3.535353, -127.9923, 1, 0.6862745, 0, 1,
7.393939, 3.535353, -128.1869, 1, 0.5843138, 0, 1,
7.434343, 3.535353, -128.388, 1, 0.5843138, 0, 1,
7.474748, 3.535353, -128.5956, 1, 0.5843138, 0, 1,
7.515152, 3.535353, -128.8097, 1, 0.5843138, 0, 1,
7.555555, 3.535353, -129.0304, 1, 0.5843138, 0, 1,
7.59596, 3.535353, -129.2576, 1, 0.5843138, 0, 1,
7.636364, 3.535353, -129.4914, 1, 0.4823529, 0, 1,
7.676768, 3.535353, -129.7316, 1, 0.4823529, 0, 1,
7.717172, 3.535353, -129.9784, 1, 0.4823529, 0, 1,
7.757576, 3.535353, -130.2318, 1, 0.4823529, 0, 1,
7.79798, 3.535353, -130.4916, 1, 0.4823529, 0, 1,
7.838384, 3.535353, -130.758, 1, 0.3764706, 0, 1,
7.878788, 3.535353, -131.0309, 1, 0.3764706, 0, 1,
7.919192, 3.535353, -131.3104, 1, 0.3764706, 0, 1,
7.959596, 3.535353, -131.5964, 1, 0.3764706, 0, 1,
8, 3.535353, -131.8889, 1, 0.2745098, 0, 1,
4, 3.575758, -134.6047, 1, 0.06666667, 0, 1,
4.040404, 3.575758, -134.2651, 1, 0.1686275, 0, 1,
4.080808, 3.575758, -133.9318, 1, 0.1686275, 0, 1,
4.121212, 3.575758, -133.6049, 1, 0.1686275, 0, 1,
4.161616, 3.575758, -133.2843, 1, 0.1686275, 0, 1,
4.20202, 3.575758, -132.9702, 1, 0.2745098, 0, 1,
4.242424, 3.575758, -132.6624, 1, 0.2745098, 0, 1,
4.282828, 3.575758, -132.3611, 1, 0.2745098, 0, 1,
4.323232, 3.575758, -132.0661, 1, 0.2745098, 0, 1,
4.363636, 3.575758, -131.7775, 1, 0.3764706, 0, 1,
4.40404, 3.575758, -131.4952, 1, 0.3764706, 0, 1,
4.444445, 3.575758, -131.2194, 1, 0.3764706, 0, 1,
4.484848, 3.575758, -130.9499, 1, 0.3764706, 0, 1,
4.525252, 3.575758, -130.6869, 1, 0.3764706, 0, 1,
4.565657, 3.575758, -130.4302, 1, 0.4823529, 0, 1,
4.606061, 3.575758, -130.1799, 1, 0.4823529, 0, 1,
4.646465, 3.575758, -129.9359, 1, 0.4823529, 0, 1,
4.686869, 3.575758, -129.6984, 1, 0.4823529, 0, 1,
4.727273, 3.575758, -129.4673, 1, 0.4823529, 0, 1,
4.767677, 3.575758, -129.2425, 1, 0.5843138, 0, 1,
4.808081, 3.575758, -129.0241, 1, 0.5843138, 0, 1,
4.848485, 3.575758, -128.8121, 1, 0.5843138, 0, 1,
4.888889, 3.575758, -128.6065, 1, 0.5843138, 0, 1,
4.929293, 3.575758, -128.4073, 1, 0.5843138, 0, 1,
4.969697, 3.575758, -128.2144, 1, 0.5843138, 0, 1,
5.010101, 3.575758, -128.0279, 1, 0.6862745, 0, 1,
5.050505, 3.575758, -127.8479, 1, 0.6862745, 0, 1,
5.090909, 3.575758, -127.6742, 1, 0.6862745, 0, 1,
5.131313, 3.575758, -127.5068, 1, 0.6862745, 0, 1,
5.171717, 3.575758, -127.3459, 1, 0.6862745, 0, 1,
5.212121, 3.575758, -127.1914, 1, 0.6862745, 0, 1,
5.252525, 3.575758, -127.0432, 1, 0.6862745, 0, 1,
5.292929, 3.575758, -126.9014, 1, 0.6862745, 0, 1,
5.333333, 3.575758, -126.766, 1, 0.7921569, 0, 1,
5.373737, 3.575758, -126.637, 1, 0.7921569, 0, 1,
5.414141, 3.575758, -126.5144, 1, 0.7921569, 0, 1,
5.454545, 3.575758, -126.3982, 1, 0.7921569, 0, 1,
5.494949, 3.575758, -126.2883, 1, 0.7921569, 0, 1,
5.535354, 3.575758, -126.1848, 1, 0.7921569, 0, 1,
5.575758, 3.575758, -126.0877, 1, 0.7921569, 0, 1,
5.616162, 3.575758, -125.997, 1, 0.7921569, 0, 1,
5.656566, 3.575758, -125.9127, 1, 0.7921569, 0, 1,
5.69697, 3.575758, -125.8348, 1, 0.7921569, 0, 1,
5.737374, 3.575758, -125.7632, 1, 0.7921569, 0, 1,
5.777778, 3.575758, -125.698, 1, 0.7921569, 0, 1,
5.818182, 3.575758, -125.6392, 1, 0.7921569, 0, 1,
5.858586, 3.575758, -125.5868, 1, 0.7921569, 0, 1,
5.89899, 3.575758, -125.5408, 1, 0.8941177, 0, 1,
5.939394, 3.575758, -125.5012, 1, 0.8941177, 0, 1,
5.979798, 3.575758, -125.4679, 1, 0.8941177, 0, 1,
6.020202, 3.575758, -125.4411, 1, 0.8941177, 0, 1,
6.060606, 3.575758, -125.4206, 1, 0.8941177, 0, 1,
6.10101, 3.575758, -125.4065, 1, 0.8941177, 0, 1,
6.141414, 3.575758, -125.3988, 1, 0.8941177, 0, 1,
6.181818, 3.575758, -125.3974, 1, 0.8941177, 0, 1,
6.222222, 3.575758, -125.4025, 1, 0.8941177, 0, 1,
6.262626, 3.575758, -125.4139, 1, 0.8941177, 0, 1,
6.30303, 3.575758, -125.4317, 1, 0.8941177, 0, 1,
6.343434, 3.575758, -125.4559, 1, 0.8941177, 0, 1,
6.383838, 3.575758, -125.4865, 1, 0.8941177, 0, 1,
6.424242, 3.575758, -125.5235, 1, 0.8941177, 0, 1,
6.464646, 3.575758, -125.5668, 1, 0.8941177, 0, 1,
6.505051, 3.575758, -125.6166, 1, 0.7921569, 0, 1,
6.545455, 3.575758, -125.6727, 1, 0.7921569, 0, 1,
6.585859, 3.575758, -125.7352, 1, 0.7921569, 0, 1,
6.626263, 3.575758, -125.8041, 1, 0.7921569, 0, 1,
6.666667, 3.575758, -125.8794, 1, 0.7921569, 0, 1,
6.707071, 3.575758, -125.961, 1, 0.7921569, 0, 1,
6.747475, 3.575758, -126.049, 1, 0.7921569, 0, 1,
6.787879, 3.575758, -126.1435, 1, 0.7921569, 0, 1,
6.828283, 3.575758, -126.2443, 1, 0.7921569, 0, 1,
6.868687, 3.575758, -126.3515, 1, 0.7921569, 0, 1,
6.909091, 3.575758, -126.465, 1, 0.7921569, 0, 1,
6.949495, 3.575758, -126.585, 1, 0.7921569, 0, 1,
6.989899, 3.575758, -126.7113, 1, 0.7921569, 0, 1,
7.030303, 3.575758, -126.8441, 1, 0.6862745, 0, 1,
7.070707, 3.575758, -126.9832, 1, 0.6862745, 0, 1,
7.111111, 3.575758, -127.1287, 1, 0.6862745, 0, 1,
7.151515, 3.575758, -127.2806, 1, 0.6862745, 0, 1,
7.191919, 3.575758, -127.4388, 1, 0.6862745, 0, 1,
7.232323, 3.575758, -127.6035, 1, 0.6862745, 0, 1,
7.272727, 3.575758, -127.7745, 1, 0.6862745, 0, 1,
7.313131, 3.575758, -127.9519, 1, 0.6862745, 0, 1,
7.353535, 3.575758, -128.1357, 1, 0.5843138, 0, 1,
7.393939, 3.575758, -128.3259, 1, 0.5843138, 0, 1,
7.434343, 3.575758, -128.5225, 1, 0.5843138, 0, 1,
7.474748, 3.575758, -128.7254, 1, 0.5843138, 0, 1,
7.515152, 3.575758, -128.9348, 1, 0.5843138, 0, 1,
7.555555, 3.575758, -129.1505, 1, 0.5843138, 0, 1,
7.59596, 3.575758, -129.3726, 1, 0.4823529, 0, 1,
7.636364, 3.575758, -129.6011, 1, 0.4823529, 0, 1,
7.676768, 3.575758, -129.8359, 1, 0.4823529, 0, 1,
7.717172, 3.575758, -130.0772, 1, 0.4823529, 0, 1,
7.757576, 3.575758, -130.3248, 1, 0.4823529, 0, 1,
7.79798, 3.575758, -130.5788, 1, 0.3764706, 0, 1,
7.838384, 3.575758, -130.8392, 1, 0.3764706, 0, 1,
7.878788, 3.575758, -131.106, 1, 0.3764706, 0, 1,
7.919192, 3.575758, -131.3792, 1, 0.3764706, 0, 1,
7.959596, 3.575758, -131.6588, 1, 0.3764706, 0, 1,
8, 3.575758, -131.9447, 1, 0.2745098, 0, 1,
4, 3.616162, -134.6121, 1, 0.06666667, 0, 1,
4.040404, 3.616162, -134.28, 1, 0.06666667, 0, 1,
4.080808, 3.616162, -133.9541, 1, 0.1686275, 0, 1,
4.121212, 3.616162, -133.6345, 1, 0.1686275, 0, 1,
4.161616, 3.616162, -133.3211, 1, 0.1686275, 0, 1,
4.20202, 3.616162, -133.0139, 1, 0.2745098, 0, 1,
4.242424, 3.616162, -132.713, 1, 0.2745098, 0, 1,
4.282828, 3.616162, -132.4183, 1, 0.2745098, 0, 1,
4.323232, 3.616162, -132.1299, 1, 0.2745098, 0, 1,
4.363636, 3.616162, -131.8477, 1, 0.2745098, 0, 1,
4.40404, 3.616162, -131.5717, 1, 0.3764706, 0, 1,
4.444445, 3.616162, -131.302, 1, 0.3764706, 0, 1,
4.484848, 3.616162, -131.0385, 1, 0.3764706, 0, 1,
4.525252, 3.616162, -130.7813, 1, 0.3764706, 0, 1,
4.565657, 3.616162, -130.5303, 1, 0.4823529, 0, 1,
4.606061, 3.616162, -130.2856, 1, 0.4823529, 0, 1,
4.646465, 3.616162, -130.0471, 1, 0.4823529, 0, 1,
4.686869, 3.616162, -129.8148, 1, 0.4823529, 0, 1,
4.727273, 3.616162, -129.5888, 1, 0.4823529, 0, 1,
4.767677, 3.616162, -129.369, 1, 0.4823529, 0, 1,
4.808081, 3.616162, -129.1555, 1, 0.5843138, 0, 1,
4.848485, 3.616162, -128.9482, 1, 0.5843138, 0, 1,
4.888889, 3.616162, -128.7472, 1, 0.5843138, 0, 1,
4.929293, 3.616162, -128.5524, 1, 0.5843138, 0, 1,
4.969697, 3.616162, -128.3638, 1, 0.5843138, 0, 1,
5.010101, 3.616162, -128.1815, 1, 0.5843138, 0, 1,
5.050505, 3.616162, -128.0054, 1, 0.6862745, 0, 1,
5.090909, 3.616162, -127.8356, 1, 0.6862745, 0, 1,
5.131313, 3.616162, -127.672, 1, 0.6862745, 0, 1,
5.171717, 3.616162, -127.5146, 1, 0.6862745, 0, 1,
5.212121, 3.616162, -127.3635, 1, 0.6862745, 0, 1,
5.252525, 3.616162, -127.2186, 1, 0.6862745, 0, 1,
5.292929, 3.616162, -127.08, 1, 0.6862745, 0, 1,
5.333333, 3.616162, -126.9476, 1, 0.6862745, 0, 1,
5.373737, 3.616162, -126.8215, 1, 0.6862745, 0, 1,
5.414141, 3.616162, -126.7016, 1, 0.7921569, 0, 1,
5.454545, 3.616162, -126.5879, 1, 0.7921569, 0, 1,
5.494949, 3.616162, -126.4805, 1, 0.7921569, 0, 1,
5.535354, 3.616162, -126.3793, 1, 0.7921569, 0, 1,
5.575758, 3.616162, -126.2844, 1, 0.7921569, 0, 1,
5.616162, 3.616162, -126.1957, 1, 0.7921569, 0, 1,
5.656566, 3.616162, -126.1132, 1, 0.7921569, 0, 1,
5.69697, 3.616162, -126.037, 1, 0.7921569, 0, 1,
5.737374, 3.616162, -125.9671, 1, 0.7921569, 0, 1,
5.777778, 3.616162, -125.9033, 1, 0.7921569, 0, 1,
5.818182, 3.616162, -125.8459, 1, 0.7921569, 0, 1,
5.858586, 3.616162, -125.7946, 1, 0.7921569, 0, 1,
5.89899, 3.616162, -125.7496, 1, 0.7921569, 0, 1,
5.939394, 3.616162, -125.7109, 1, 0.7921569, 0, 1,
5.979798, 3.616162, -125.6784, 1, 0.7921569, 0, 1,
6.020202, 3.616162, -125.6521, 1, 0.7921569, 0, 1,
6.060606, 3.616162, -125.6321, 1, 0.7921569, 0, 1,
6.10101, 3.616162, -125.6183, 1, 0.7921569, 0, 1,
6.141414, 3.616162, -125.6107, 1, 0.7921569, 0, 1,
6.181818, 3.616162, -125.6094, 1, 0.7921569, 0, 1,
6.222222, 3.616162, -125.6144, 1, 0.7921569, 0, 1,
6.262626, 3.616162, -125.6255, 1, 0.7921569, 0, 1,
6.30303, 3.616162, -125.643, 1, 0.7921569, 0, 1,
6.343434, 3.616162, -125.6666, 1, 0.7921569, 0, 1,
6.383838, 3.616162, -125.6965, 1, 0.7921569, 0, 1,
6.424242, 3.616162, -125.7327, 1, 0.7921569, 0, 1,
6.464646, 3.616162, -125.7751, 1, 0.7921569, 0, 1,
6.505051, 3.616162, -125.8237, 1, 0.7921569, 0, 1,
6.545455, 3.616162, -125.8786, 1, 0.7921569, 0, 1,
6.585859, 3.616162, -125.9397, 1, 0.7921569, 0, 1,
6.626263, 3.616162, -126.007, 1, 0.7921569, 0, 1,
6.666667, 3.616162, -126.0806, 1, 0.7921569, 0, 1,
6.707071, 3.616162, -126.1605, 1, 0.7921569, 0, 1,
6.747475, 3.616162, -126.2466, 1, 0.7921569, 0, 1,
6.787879, 3.616162, -126.3389, 1, 0.7921569, 0, 1,
6.828283, 3.616162, -126.4375, 1, 0.7921569, 0, 1,
6.868687, 3.616162, -126.5423, 1, 0.7921569, 0, 1,
6.909091, 3.616162, -126.6533, 1, 0.7921569, 0, 1,
6.949495, 3.616162, -126.7706, 1, 0.7921569, 0, 1,
6.989899, 3.616162, -126.8942, 1, 0.6862745, 0, 1,
7.030303, 3.616162, -127.0239, 1, 0.6862745, 0, 1,
7.070707, 3.616162, -127.16, 1, 0.6862745, 0, 1,
7.111111, 3.616162, -127.3022, 1, 0.6862745, 0, 1,
7.151515, 3.616162, -127.4507, 1, 0.6862745, 0, 1,
7.191919, 3.616162, -127.6055, 1, 0.6862745, 0, 1,
7.232323, 3.616162, -127.7665, 1, 0.6862745, 0, 1,
7.272727, 3.616162, -127.9337, 1, 0.6862745, 0, 1,
7.313131, 3.616162, -128.1072, 1, 0.5843138, 0, 1,
7.353535, 3.616162, -128.2869, 1, 0.5843138, 0, 1,
7.393939, 3.616162, -128.4728, 1, 0.5843138, 0, 1,
7.434343, 3.616162, -128.665, 1, 0.5843138, 0, 1,
7.474748, 3.616162, -128.8635, 1, 0.5843138, 0, 1,
7.515152, 3.616162, -129.0681, 1, 0.5843138, 0, 1,
7.555555, 3.616162, -129.2791, 1, 0.5843138, 0, 1,
7.59596, 3.616162, -129.4962, 1, 0.4823529, 0, 1,
7.636364, 3.616162, -129.7197, 1, 0.4823529, 0, 1,
7.676768, 3.616162, -129.9493, 1, 0.4823529, 0, 1,
7.717172, 3.616162, -130.1852, 1, 0.4823529, 0, 1,
7.757576, 3.616162, -130.4273, 1, 0.4823529, 0, 1,
7.79798, 3.616162, -130.6757, 1, 0.3764706, 0, 1,
7.838384, 3.616162, -130.9303, 1, 0.3764706, 0, 1,
7.878788, 3.616162, -131.1912, 1, 0.3764706, 0, 1,
7.919192, 3.616162, -131.4583, 1, 0.3764706, 0, 1,
7.959596, 3.616162, -131.7316, 1, 0.3764706, 0, 1,
8, 3.616162, -132.0112, 1, 0.2745098, 0, 1,
4, 3.656566, -134.6316, 1, 0.06666667, 0, 1,
4.040404, 3.656566, -134.3067, 1, 0.06666667, 0, 1,
4.080808, 3.656566, -133.988, 1, 0.1686275, 0, 1,
4.121212, 3.656566, -133.6754, 1, 0.1686275, 0, 1,
4.161616, 3.656566, -133.3689, 1, 0.1686275, 0, 1,
4.20202, 3.656566, -133.0685, 1, 0.1686275, 0, 1,
4.242424, 3.656566, -132.7742, 1, 0.2745098, 0, 1,
4.282828, 3.656566, -132.486, 1, 0.2745098, 0, 1,
4.323232, 3.656566, -132.2039, 1, 0.2745098, 0, 1,
4.363636, 3.656566, -131.9279, 1, 0.2745098, 0, 1,
4.40404, 3.656566, -131.658, 1, 0.3764706, 0, 1,
4.444445, 3.656566, -131.3942, 1, 0.3764706, 0, 1,
4.484848, 3.656566, -131.1365, 1, 0.3764706, 0, 1,
4.525252, 3.656566, -130.8849, 1, 0.3764706, 0, 1,
4.565657, 3.656566, -130.6395, 1, 0.3764706, 0, 1,
4.606061, 3.656566, -130.4001, 1, 0.4823529, 0, 1,
4.646465, 3.656566, -130.1669, 1, 0.4823529, 0, 1,
4.686869, 3.656566, -129.9397, 1, 0.4823529, 0, 1,
4.727273, 3.656566, -129.7187, 1, 0.4823529, 0, 1,
4.767677, 3.656566, -129.5037, 1, 0.4823529, 0, 1,
4.808081, 3.656566, -129.2949, 1, 0.5843138, 0, 1,
4.848485, 3.656566, -129.0921, 1, 0.5843138, 0, 1,
4.888889, 3.656566, -128.8955, 1, 0.5843138, 0, 1,
4.929293, 3.656566, -128.705, 1, 0.5843138, 0, 1,
4.969697, 3.656566, -128.5206, 1, 0.5843138, 0, 1,
5.010101, 3.656566, -128.3423, 1, 0.5843138, 0, 1,
5.050505, 3.656566, -128.17, 1, 0.5843138, 0, 1,
5.090909, 3.656566, -128.0039, 1, 0.6862745, 0, 1,
5.131313, 3.656566, -127.8439, 1, 0.6862745, 0, 1,
5.171717, 3.656566, -127.69, 1, 0.6862745, 0, 1,
5.212121, 3.656566, -127.5422, 1, 0.6862745, 0, 1,
5.252525, 3.656566, -127.4006, 1, 0.6862745, 0, 1,
5.292929, 3.656566, -127.265, 1, 0.6862745, 0, 1,
5.333333, 3.656566, -127.1355, 1, 0.6862745, 0, 1,
5.373737, 3.656566, -127.0121, 1, 0.6862745, 0, 1,
5.414141, 3.656566, -126.8949, 1, 0.6862745, 0, 1,
5.454545, 3.656566, -126.7837, 1, 0.7921569, 0, 1,
5.494949, 3.656566, -126.6786, 1, 0.7921569, 0, 1,
5.535354, 3.656566, -126.5797, 1, 0.7921569, 0, 1,
5.575758, 3.656566, -126.4868, 1, 0.7921569, 0, 1,
5.616162, 3.656566, -126.4001, 1, 0.7921569, 0, 1,
5.656566, 3.656566, -126.3195, 1, 0.7921569, 0, 1,
5.69697, 3.656566, -126.2449, 1, 0.7921569, 0, 1,
5.737374, 3.656566, -126.1765, 1, 0.7921569, 0, 1,
5.777778, 3.656566, -126.1142, 1, 0.7921569, 0, 1,
5.818182, 3.656566, -126.058, 1, 0.7921569, 0, 1,
5.858586, 3.656566, -126.0078, 1, 0.7921569, 0, 1,
5.89899, 3.656566, -125.9638, 1, 0.7921569, 0, 1,
5.939394, 3.656566, -125.9259, 1, 0.7921569, 0, 1,
5.979798, 3.656566, -125.8941, 1, 0.7921569, 0, 1,
6.020202, 3.656566, -125.8684, 1, 0.7921569, 0, 1,
6.060606, 3.656566, -125.8488, 1, 0.7921569, 0, 1,
6.10101, 3.656566, -125.8354, 1, 0.7921569, 0, 1,
6.141414, 3.656566, -125.828, 1, 0.7921569, 0, 1,
6.181818, 3.656566, -125.8267, 1, 0.7921569, 0, 1,
6.222222, 3.656566, -125.8315, 1, 0.7921569, 0, 1,
6.262626, 3.656566, -125.8425, 1, 0.7921569, 0, 1,
6.30303, 3.656566, -125.8595, 1, 0.7921569, 0, 1,
6.343434, 3.656566, -125.8827, 1, 0.7921569, 0, 1,
6.383838, 3.656566, -125.9119, 1, 0.7921569, 0, 1,
6.424242, 3.656566, -125.9473, 1, 0.7921569, 0, 1,
6.464646, 3.656566, -125.9887, 1, 0.7921569, 0, 1,
6.505051, 3.656566, -126.0363, 1, 0.7921569, 0, 1,
6.545455, 3.656566, -126.0899, 1, 0.7921569, 0, 1,
6.585859, 3.656566, -126.1497, 1, 0.7921569, 0, 1,
6.626263, 3.656566, -126.2156, 1, 0.7921569, 0, 1,
6.666667, 3.656566, -126.2876, 1, 0.7921569, 0, 1,
6.707071, 3.656566, -126.3657, 1, 0.7921569, 0, 1,
6.747475, 3.656566, -126.4499, 1, 0.7921569, 0, 1,
6.787879, 3.656566, -126.5402, 1, 0.7921569, 0, 1,
6.828283, 3.656566, -126.6366, 1, 0.7921569, 0, 1,
6.868687, 3.656566, -126.7391, 1, 0.7921569, 0, 1,
6.909091, 3.656566, -126.8477, 1, 0.6862745, 0, 1,
6.949495, 3.656566, -126.9624, 1, 0.6862745, 0, 1,
6.989899, 3.656566, -127.0832, 1, 0.6862745, 0, 1,
7.030303, 3.656566, -127.2101, 1, 0.6862745, 0, 1,
7.070707, 3.656566, -127.3432, 1, 0.6862745, 0, 1,
7.111111, 3.656566, -127.4823, 1, 0.6862745, 0, 1,
7.151515, 3.656566, -127.6275, 1, 0.6862745, 0, 1,
7.191919, 3.656566, -127.7789, 1, 0.6862745, 0, 1,
7.232323, 3.656566, -127.9363, 1, 0.6862745, 0, 1,
7.272727, 3.656566, -128.0999, 1, 0.5843138, 0, 1,
7.313131, 3.656566, -128.2695, 1, 0.5843138, 0, 1,
7.353535, 3.656566, -128.4453, 1, 0.5843138, 0, 1,
7.393939, 3.656566, -128.6272, 1, 0.5843138, 0, 1,
7.434343, 3.656566, -128.8152, 1, 0.5843138, 0, 1,
7.474748, 3.656566, -129.0092, 1, 0.5843138, 0, 1,
7.515152, 3.656566, -129.2094, 1, 0.5843138, 0, 1,
7.555555, 3.656566, -129.4157, 1, 0.4823529, 0, 1,
7.59596, 3.656566, -129.6281, 1, 0.4823529, 0, 1,
7.636364, 3.656566, -129.8466, 1, 0.4823529, 0, 1,
7.676768, 3.656566, -130.0712, 1, 0.4823529, 0, 1,
7.717172, 3.656566, -130.3019, 1, 0.4823529, 0, 1,
7.757576, 3.656566, -130.5387, 1, 0.4823529, 0, 1,
7.79798, 3.656566, -130.7816, 1, 0.3764706, 0, 1,
7.838384, 3.656566, -131.0307, 1, 0.3764706, 0, 1,
7.878788, 3.656566, -131.2858, 1, 0.3764706, 0, 1,
7.919192, 3.656566, -131.547, 1, 0.3764706, 0, 1,
7.959596, 3.656566, -131.8144, 1, 0.2745098, 0, 1,
8, 3.656566, -132.0878, 1, 0.2745098, 0, 1,
4, 3.69697, -134.6624, 1, 0.06666667, 0, 1,
4.040404, 3.69697, -134.3446, 1, 0.06666667, 0, 1,
4.080808, 3.69697, -134.0328, 1, 0.1686275, 0, 1,
4.121212, 3.69697, -133.727, 1, 0.1686275, 0, 1,
4.161616, 3.69697, -133.4271, 1, 0.1686275, 0, 1,
4.20202, 3.69697, -133.1333, 1, 0.1686275, 0, 1,
4.242424, 3.69697, -132.8454, 1, 0.2745098, 0, 1,
4.282828, 3.69697, -132.5634, 1, 0.2745098, 0, 1,
4.323232, 3.69697, -132.2874, 1, 0.2745098, 0, 1,
4.363636, 3.69697, -132.0175, 1, 0.2745098, 0, 1,
4.40404, 3.69697, -131.7534, 1, 0.3764706, 0, 1,
4.444445, 3.69697, -131.4954, 1, 0.3764706, 0, 1,
4.484848, 3.69697, -131.2433, 1, 0.3764706, 0, 1,
4.525252, 3.69697, -130.9972, 1, 0.3764706, 0, 1,
4.565657, 3.69697, -130.7571, 1, 0.3764706, 0, 1,
4.606061, 3.69697, -130.5229, 1, 0.4823529, 0, 1,
4.646465, 3.69697, -130.2947, 1, 0.4823529, 0, 1,
4.686869, 3.69697, -130.0725, 1, 0.4823529, 0, 1,
4.727273, 3.69697, -129.8563, 1, 0.4823529, 0, 1,
4.767677, 3.69697, -129.646, 1, 0.4823529, 0, 1,
4.808081, 3.69697, -129.4417, 1, 0.4823529, 0, 1,
4.848485, 3.69697, -129.2434, 1, 0.5843138, 0, 1,
4.888889, 3.69697, -129.051, 1, 0.5843138, 0, 1,
4.929293, 3.69697, -128.8646, 1, 0.5843138, 0, 1,
4.969697, 3.69697, -128.6842, 1, 0.5843138, 0, 1,
5.010101, 3.69697, -128.5098, 1, 0.5843138, 0, 1,
5.050505, 3.69697, -128.3413, 1, 0.5843138, 0, 1,
5.090909, 3.69697, -128.1788, 1, 0.5843138, 0, 1,
5.131313, 3.69697, -128.0223, 1, 0.6862745, 0, 1,
5.171717, 3.69697, -127.8717, 1, 0.6862745, 0, 1,
5.212121, 3.69697, -127.7272, 1, 0.6862745, 0, 1,
5.252525, 3.69697, -127.5886, 1, 0.6862745, 0, 1,
5.292929, 3.69697, -127.4559, 1, 0.6862745, 0, 1,
5.333333, 3.69697, -127.3293, 1, 0.6862745, 0, 1,
5.373737, 3.69697, -127.2086, 1, 0.6862745, 0, 1,
5.414141, 3.69697, -127.0939, 1, 0.6862745, 0, 1,
5.454545, 3.69697, -126.9851, 1, 0.6862745, 0, 1,
5.494949, 3.69697, -126.8823, 1, 0.6862745, 0, 1,
5.535354, 3.69697, -126.7855, 1, 0.7921569, 0, 1,
5.575758, 3.69697, -126.6947, 1, 0.7921569, 0, 1,
5.616162, 3.69697, -126.6098, 1, 0.7921569, 0, 1,
5.656566, 3.69697, -126.531, 1, 0.7921569, 0, 1,
5.69697, 3.69697, -126.4581, 1, 0.7921569, 0, 1,
5.737374, 3.69697, -126.3911, 1, 0.7921569, 0, 1,
5.777778, 3.69697, -126.3301, 1, 0.7921569, 0, 1,
5.818182, 3.69697, -126.2751, 1, 0.7921569, 0, 1,
5.858586, 3.69697, -126.2261, 1, 0.7921569, 0, 1,
5.89899, 3.69697, -126.1831, 1, 0.7921569, 0, 1,
5.939394, 3.69697, -126.146, 1, 0.7921569, 0, 1,
5.979798, 3.69697, -126.1149, 1, 0.7921569, 0, 1,
6.020202, 3.69697, -126.0897, 1, 0.7921569, 0, 1,
6.060606, 3.69697, -126.0706, 1, 0.7921569, 0, 1,
6.10101, 3.69697, -126.0574, 1, 0.7921569, 0, 1,
6.141414, 3.69697, -126.0502, 1, 0.7921569, 0, 1,
6.181818, 3.69697, -126.0489, 1, 0.7921569, 0, 1,
6.222222, 3.69697, -126.0536, 1, 0.7921569, 0, 1,
6.262626, 3.69697, -126.0643, 1, 0.7921569, 0, 1,
6.30303, 3.69697, -126.081, 1, 0.7921569, 0, 1,
6.343434, 3.69697, -126.1037, 1, 0.7921569, 0, 1,
6.383838, 3.69697, -126.1323, 1, 0.7921569, 0, 1,
6.424242, 3.69697, -126.1668, 1, 0.7921569, 0, 1,
6.464646, 3.69697, -126.2074, 1, 0.7921569, 0, 1,
6.505051, 3.69697, -126.2539, 1, 0.7921569, 0, 1,
6.545455, 3.69697, -126.3064, 1, 0.7921569, 0, 1,
6.585859, 3.69697, -126.3649, 1, 0.7921569, 0, 1,
6.626263, 3.69697, -126.4294, 1, 0.7921569, 0, 1,
6.666667, 3.69697, -126.4998, 1, 0.7921569, 0, 1,
6.707071, 3.69697, -126.5762, 1, 0.7921569, 0, 1,
6.747475, 3.69697, -126.6585, 1, 0.7921569, 0, 1,
6.787879, 3.69697, -126.7469, 1, 0.7921569, 0, 1,
6.828283, 3.69697, -126.8412, 1, 0.6862745, 0, 1,
6.868687, 3.69697, -126.9414, 1, 0.6862745, 0, 1,
6.909091, 3.69697, -127.0477, 1, 0.6862745, 0, 1,
6.949495, 3.69697, -127.1599, 1, 0.6862745, 0, 1,
6.989899, 3.69697, -127.2781, 1, 0.6862745, 0, 1,
7.030303, 3.69697, -127.4023, 1, 0.6862745, 0, 1,
7.070707, 3.69697, -127.5324, 1, 0.6862745, 0, 1,
7.111111, 3.69697, -127.6685, 1, 0.6862745, 0, 1,
7.151515, 3.69697, -127.8106, 1, 0.6862745, 0, 1,
7.191919, 3.69697, -127.9587, 1, 0.6862745, 0, 1,
7.232323, 3.69697, -128.1127, 1, 0.5843138, 0, 1,
7.272727, 3.69697, -128.2727, 1, 0.5843138, 0, 1,
7.313131, 3.69697, -128.4387, 1, 0.5843138, 0, 1,
7.353535, 3.69697, -128.6106, 1, 0.5843138, 0, 1,
7.393939, 3.69697, -128.7885, 1, 0.5843138, 0, 1,
7.434343, 3.69697, -128.9724, 1, 0.5843138, 0, 1,
7.474748, 3.69697, -129.1623, 1, 0.5843138, 0, 1,
7.515152, 3.69697, -129.3581, 1, 0.4823529, 0, 1,
7.555555, 3.69697, -129.5599, 1, 0.4823529, 0, 1,
7.59596, 3.69697, -129.7677, 1, 0.4823529, 0, 1,
7.636364, 3.69697, -129.9814, 1, 0.4823529, 0, 1,
7.676768, 3.69697, -130.2012, 1, 0.4823529, 0, 1,
7.717172, 3.69697, -130.4268, 1, 0.4823529, 0, 1,
7.757576, 3.69697, -130.6585, 1, 0.3764706, 0, 1,
7.79798, 3.69697, -130.8961, 1, 0.3764706, 0, 1,
7.838384, 3.69697, -131.1398, 1, 0.3764706, 0, 1,
7.878788, 3.69697, -131.3893, 1, 0.3764706, 0, 1,
7.919192, 3.69697, -131.6449, 1, 0.3764706, 0, 1,
7.959596, 3.69697, -131.9064, 1, 0.2745098, 0, 1,
8, 3.69697, -132.1739, 1, 0.2745098, 0, 1,
4, 3.737374, -134.7039, 1, 0.06666667, 0, 1,
4.040404, 3.737374, -134.393, 1, 0.06666667, 0, 1,
4.080808, 3.737374, -134.0879, 1, 0.1686275, 0, 1,
4.121212, 3.737374, -133.7887, 1, 0.1686275, 0, 1,
4.161616, 3.737374, -133.4953, 1, 0.1686275, 0, 1,
4.20202, 3.737374, -133.2077, 1, 0.1686275, 0, 1,
4.242424, 3.737374, -132.926, 1, 0.2745098, 0, 1,
4.282828, 3.737374, -132.6501, 1, 0.2745098, 0, 1,
4.323232, 3.737374, -132.3801, 1, 0.2745098, 0, 1,
4.363636, 3.737374, -132.1159, 1, 0.2745098, 0, 1,
4.40404, 3.737374, -131.8576, 1, 0.2745098, 0, 1,
4.444445, 3.737374, -131.6051, 1, 0.3764706, 0, 1,
4.484848, 3.737374, -131.3584, 1, 0.3764706, 0, 1,
4.525252, 3.737374, -131.1176, 1, 0.3764706, 0, 1,
4.565657, 3.737374, -130.8826, 1, 0.3764706, 0, 1,
4.606061, 3.737374, -130.6535, 1, 0.3764706, 0, 1,
4.646465, 3.737374, -130.4302, 1, 0.4823529, 0, 1,
4.686869, 3.737374, -130.2128, 1, 0.4823529, 0, 1,
4.727273, 3.737374, -130.0012, 1, 0.4823529, 0, 1,
4.767677, 3.737374, -129.7954, 1, 0.4823529, 0, 1,
4.808081, 3.737374, -129.5955, 1, 0.4823529, 0, 1,
4.848485, 3.737374, -129.4015, 1, 0.4823529, 0, 1,
4.888889, 3.737374, -129.2133, 1, 0.5843138, 0, 1,
4.929293, 3.737374, -129.0309, 1, 0.5843138, 0, 1,
4.969697, 3.737374, -128.8543, 1, 0.5843138, 0, 1,
5.010101, 3.737374, -128.6837, 1, 0.5843138, 0, 1,
5.050505, 3.737374, -128.5188, 1, 0.5843138, 0, 1,
5.090909, 3.737374, -128.3598, 1, 0.5843138, 0, 1,
5.131313, 3.737374, -128.2066, 1, 0.5843138, 0, 1,
5.171717, 3.737374, -128.0593, 1, 0.6862745, 0, 1,
5.212121, 3.737374, -127.9179, 1, 0.6862745, 0, 1,
5.252525, 3.737374, -127.7822, 1, 0.6862745, 0, 1,
5.292929, 3.737374, -127.6525, 1, 0.6862745, 0, 1,
5.333333, 3.737374, -127.5285, 1, 0.6862745, 0, 1,
5.373737, 3.737374, -127.4104, 1, 0.6862745, 0, 1,
5.414141, 3.737374, -127.2982, 1, 0.6862745, 0, 1,
5.454545, 3.737374, -127.1918, 1, 0.6862745, 0, 1,
5.494949, 3.737374, -127.0912, 1, 0.6862745, 0, 1,
5.535354, 3.737374, -126.9965, 1, 0.6862745, 0, 1,
5.575758, 3.737374, -126.9076, 1, 0.6862745, 0, 1,
5.616162, 3.737374, -126.8246, 1, 0.6862745, 0, 1,
5.656566, 3.737374, -126.7474, 1, 0.7921569, 0, 1,
5.69697, 3.737374, -126.676, 1, 0.7921569, 0, 1,
5.737374, 3.737374, -126.6105, 1, 0.7921569, 0, 1,
5.777778, 3.737374, -126.5509, 1, 0.7921569, 0, 1,
5.818182, 3.737374, -126.4971, 1, 0.7921569, 0, 1,
5.858586, 3.737374, -126.4491, 1, 0.7921569, 0, 1,
5.89899, 3.737374, -126.407, 1, 0.7921569, 0, 1,
5.939394, 3.737374, -126.3707, 1, 0.7921569, 0, 1,
5.979798, 3.737374, -126.3403, 1, 0.7921569, 0, 1,
6.020202, 3.737374, -126.3157, 1, 0.7921569, 0, 1,
6.060606, 3.737374, -126.2969, 1, 0.7921569, 0, 1,
6.10101, 3.737374, -126.284, 1, 0.7921569, 0, 1,
6.141414, 3.737374, -126.2769, 1, 0.7921569, 0, 1,
6.181818, 3.737374, -126.2757, 1, 0.7921569, 0, 1,
6.222222, 3.737374, -126.2803, 1, 0.7921569, 0, 1,
6.262626, 3.737374, -126.2908, 1, 0.7921569, 0, 1,
6.30303, 3.737374, -126.3071, 1, 0.7921569, 0, 1,
6.343434, 3.737374, -126.3293, 1, 0.7921569, 0, 1,
6.383838, 3.737374, -126.3573, 1, 0.7921569, 0, 1,
6.424242, 3.737374, -126.3911, 1, 0.7921569, 0, 1,
6.464646, 3.737374, -126.4308, 1, 0.7921569, 0, 1,
6.505051, 3.737374, -126.4763, 1, 0.7921569, 0, 1,
6.545455, 3.737374, -126.5277, 1, 0.7921569, 0, 1,
6.585859, 3.737374, -126.5849, 1, 0.7921569, 0, 1,
6.626263, 3.737374, -126.648, 1, 0.7921569, 0, 1,
6.666667, 3.737374, -126.7169, 1, 0.7921569, 0, 1,
6.707071, 3.737374, -126.7916, 1, 0.7921569, 0, 1,
6.747475, 3.737374, -126.8722, 1, 0.6862745, 0, 1,
6.787879, 3.737374, -126.9586, 1, 0.6862745, 0, 1,
6.828283, 3.737374, -127.0509, 1, 0.6862745, 0, 1,
6.868687, 3.737374, -127.149, 1, 0.6862745, 0, 1,
6.909091, 3.737374, -127.253, 1, 0.6862745, 0, 1,
6.949495, 3.737374, -127.3628, 1, 0.6862745, 0, 1,
6.989899, 3.737374, -127.4785, 1, 0.6862745, 0, 1,
7.030303, 3.737374, -127.6, 1, 0.6862745, 0, 1,
7.070707, 3.737374, -127.7273, 1, 0.6862745, 0, 1,
7.111111, 3.737374, -127.8605, 1, 0.6862745, 0, 1,
7.151515, 3.737374, -127.9995, 1, 0.6862745, 0, 1,
7.191919, 3.737374, -128.1444, 1, 0.5843138, 0, 1,
7.232323, 3.737374, -128.2951, 1, 0.5843138, 0, 1,
7.272727, 3.737374, -128.4517, 1, 0.5843138, 0, 1,
7.313131, 3.737374, -128.6141, 1, 0.5843138, 0, 1,
7.353535, 3.737374, -128.7823, 1, 0.5843138, 0, 1,
7.393939, 3.737374, -128.9564, 1, 0.5843138, 0, 1,
7.434343, 3.737374, -129.1363, 1, 0.5843138, 0, 1,
7.474748, 3.737374, -129.3221, 1, 0.4823529, 0, 1,
7.515152, 3.737374, -129.5137, 1, 0.4823529, 0, 1,
7.555555, 3.737374, -129.7112, 1, 0.4823529, 0, 1,
7.59596, 3.737374, -129.9145, 1, 0.4823529, 0, 1,
7.636364, 3.737374, -130.1237, 1, 0.4823529, 0, 1,
7.676768, 3.737374, -130.3387, 1, 0.4823529, 0, 1,
7.717172, 3.737374, -130.5595, 1, 0.3764706, 0, 1,
7.757576, 3.737374, -130.7862, 1, 0.3764706, 0, 1,
7.79798, 3.737374, -131.0187, 1, 0.3764706, 0, 1,
7.838384, 3.737374, -131.2571, 1, 0.3764706, 0, 1,
7.878788, 3.737374, -131.5013, 1, 0.3764706, 0, 1,
7.919192, 3.737374, -131.7513, 1, 0.3764706, 0, 1,
7.959596, 3.737374, -132.0072, 1, 0.2745098, 0, 1,
8, 3.737374, -132.269, 1, 0.2745098, 0, 1,
4, 3.777778, -134.7557, 1, 0.06666667, 0, 1,
4.040404, 3.777778, -134.4514, 1, 0.06666667, 0, 1,
4.080808, 3.777778, -134.1528, 1, 0.1686275, 0, 1,
4.121212, 3.777778, -133.8599, 1, 0.1686275, 0, 1,
4.161616, 3.777778, -133.5727, 1, 0.1686275, 0, 1,
4.20202, 3.777778, -133.2913, 1, 0.1686275, 0, 1,
4.242424, 3.777778, -133.0156, 1, 0.2745098, 0, 1,
4.282828, 3.777778, -132.7456, 1, 0.2745098, 0, 1,
4.323232, 3.777778, -132.4813, 1, 0.2745098, 0, 1,
4.363636, 3.777778, -132.2227, 1, 0.2745098, 0, 1,
4.40404, 3.777778, -131.9699, 1, 0.2745098, 0, 1,
4.444445, 3.777778, -131.7227, 1, 0.3764706, 0, 1,
4.484848, 3.777778, -131.4813, 1, 0.3764706, 0, 1,
4.525252, 3.777778, -131.2456, 1, 0.3764706, 0, 1,
4.565657, 3.777778, -131.0157, 1, 0.3764706, 0, 1,
4.606061, 3.777778, -130.7914, 1, 0.3764706, 0, 1,
4.646465, 3.777778, -130.5729, 1, 0.3764706, 0, 1,
4.686869, 3.777778, -130.3601, 1, 0.4823529, 0, 1,
4.727273, 3.777778, -130.153, 1, 0.4823529, 0, 1,
4.767677, 3.777778, -129.9516, 1, 0.4823529, 0, 1,
4.808081, 3.777778, -129.756, 1, 0.4823529, 0, 1,
4.848485, 3.777778, -129.566, 1, 0.4823529, 0, 1,
4.888889, 3.777778, -129.3818, 1, 0.4823529, 0, 1,
4.929293, 3.777778, -129.2033, 1, 0.5843138, 0, 1,
4.969697, 3.777778, -129.0305, 1, 0.5843138, 0, 1,
5.010101, 3.777778, -128.8635, 1, 0.5843138, 0, 1,
5.050505, 3.777778, -128.7021, 1, 0.5843138, 0, 1,
5.090909, 3.777778, -128.5465, 1, 0.5843138, 0, 1,
5.131313, 3.777778, -128.3966, 1, 0.5843138, 0, 1,
5.171717, 3.777778, -128.2525, 1, 0.5843138, 0, 1,
5.212121, 3.777778, -128.114, 1, 0.5843138, 0, 1,
5.252525, 3.777778, -127.9813, 1, 0.6862745, 0, 1,
5.292929, 3.777778, -127.8542, 1, 0.6862745, 0, 1,
5.333333, 3.777778, -127.7329, 1, 0.6862745, 0, 1,
5.373737, 3.777778, -127.6174, 1, 0.6862745, 0, 1,
5.414141, 3.777778, -127.5075, 1, 0.6862745, 0, 1,
5.454545, 3.777778, -127.4033, 1, 0.6862745, 0, 1,
5.494949, 3.777778, -127.3049, 1, 0.6862745, 0, 1,
5.535354, 3.777778, -127.2122, 1, 0.6862745, 0, 1,
5.575758, 3.777778, -127.1252, 1, 0.6862745, 0, 1,
5.616162, 3.777778, -127.044, 1, 0.6862745, 0, 1,
5.656566, 3.777778, -126.9684, 1, 0.6862745, 0, 1,
5.69697, 3.777778, -126.8986, 1, 0.6862745, 0, 1,
5.737374, 3.777778, -126.8345, 1, 0.6862745, 0, 1,
5.777778, 3.777778, -126.7761, 1, 0.7921569, 0, 1,
5.818182, 3.777778, -126.7234, 1, 0.7921569, 0, 1,
5.858586, 3.777778, -126.6765, 1, 0.7921569, 0, 1,
5.89899, 3.777778, -126.6352, 1, 0.7921569, 0, 1,
5.939394, 3.777778, -126.5997, 1, 0.7921569, 0, 1,
5.979798, 3.777778, -126.5699, 1, 0.7921569, 0, 1,
6.020202, 3.777778, -126.5459, 1, 0.7921569, 0, 1,
6.060606, 3.777778, -126.5275, 1, 0.7921569, 0, 1,
6.10101, 3.777778, -126.5149, 1, 0.7921569, 0, 1,
6.141414, 3.777778, -126.508, 1, 0.7921569, 0, 1,
6.181818, 3.777778, -126.5068, 1, 0.7921569, 0, 1,
6.222222, 3.777778, -126.5113, 1, 0.7921569, 0, 1,
6.262626, 3.777778, -126.5215, 1, 0.7921569, 0, 1,
6.30303, 3.777778, -126.5375, 1, 0.7921569, 0, 1,
6.343434, 3.777778, -126.5592, 1, 0.7921569, 0, 1,
6.383838, 3.777778, -126.5866, 1, 0.7921569, 0, 1,
6.424242, 3.777778, -126.6197, 1, 0.7921569, 0, 1,
6.464646, 3.777778, -126.6586, 1, 0.7921569, 0, 1,
6.505051, 3.777778, -126.7031, 1, 0.7921569, 0, 1,
6.545455, 3.777778, -126.7534, 1, 0.7921569, 0, 1,
6.585859, 3.777778, -126.8094, 1, 0.7921569, 0, 1,
6.626263, 3.777778, -126.8711, 1, 0.6862745, 0, 1,
6.666667, 3.777778, -126.9386, 1, 0.6862745, 0, 1,
6.707071, 3.777778, -127.0117, 1, 0.6862745, 0, 1,
6.747475, 3.777778, -127.0906, 1, 0.6862745, 0, 1,
6.787879, 3.777778, -127.1752, 1, 0.6862745, 0, 1,
6.828283, 3.777778, -127.2655, 1, 0.6862745, 0, 1,
6.868687, 3.777778, -127.3615, 1, 0.6862745, 0, 1,
6.909091, 3.777778, -127.4633, 1, 0.6862745, 0, 1,
6.949495, 3.777778, -127.5708, 1, 0.6862745, 0, 1,
6.989899, 3.777778, -127.6839, 1, 0.6862745, 0, 1,
7.030303, 3.777778, -127.8029, 1, 0.6862745, 0, 1,
7.070707, 3.777778, -127.9275, 1, 0.6862745, 0, 1,
7.111111, 3.777778, -128.0578, 1, 0.6862745, 0, 1,
7.151515, 3.777778, -128.1939, 1, 0.5843138, 0, 1,
7.191919, 3.777778, -128.3357, 1, 0.5843138, 0, 1,
7.232323, 3.777778, -128.4832, 1, 0.5843138, 0, 1,
7.272727, 3.777778, -128.6364, 1, 0.5843138, 0, 1,
7.313131, 3.777778, -128.7954, 1, 0.5843138, 0, 1,
7.353535, 3.777778, -128.9601, 1, 0.5843138, 0, 1,
7.393939, 3.777778, -129.1304, 1, 0.5843138, 0, 1,
7.434343, 3.777778, -129.3065, 1, 0.4823529, 0, 1,
7.474748, 3.777778, -129.4884, 1, 0.4823529, 0, 1,
7.515152, 3.777778, -129.6759, 1, 0.4823529, 0, 1,
7.555555, 3.777778, -129.8692, 1, 0.4823529, 0, 1,
7.59596, 3.777778, -130.0681, 1, 0.4823529, 0, 1,
7.636364, 3.777778, -130.2729, 1, 0.4823529, 0, 1,
7.676768, 3.777778, -130.4833, 1, 0.4823529, 0, 1,
7.717172, 3.777778, -130.6994, 1, 0.3764706, 0, 1,
7.757576, 3.777778, -130.9213, 1, 0.3764706, 0, 1,
7.79798, 3.777778, -131.1488, 1, 0.3764706, 0, 1,
7.838384, 3.777778, -131.3822, 1, 0.3764706, 0, 1,
7.878788, 3.777778, -131.6212, 1, 0.3764706, 0, 1,
7.919192, 3.777778, -131.8659, 1, 0.2745098, 0, 1,
7.959596, 3.777778, -132.1164, 1, 0.2745098, 0, 1,
8, 3.777778, -132.3725, 1, 0.2745098, 0, 1,
4, 3.818182, -134.8171, 1, 0.06666667, 0, 1,
4.040404, 3.818182, -134.5191, 1, 0.06666667, 0, 1,
4.080808, 3.818182, -134.2268, 1, 0.1686275, 0, 1,
4.121212, 3.818182, -133.9401, 1, 0.1686275, 0, 1,
4.161616, 3.818182, -133.659, 1, 0.1686275, 0, 1,
4.20202, 3.818182, -133.3835, 1, 0.1686275, 0, 1,
4.242424, 3.818182, -133.1136, 1, 0.1686275, 0, 1,
4.282828, 3.818182, -132.8492, 1, 0.2745098, 0, 1,
4.323232, 3.818182, -132.5905, 1, 0.2745098, 0, 1,
4.363636, 3.818182, -132.3374, 1, 0.2745098, 0, 1,
4.40404, 3.818182, -132.0899, 1, 0.2745098, 0, 1,
4.444445, 3.818182, -131.8479, 1, 0.2745098, 0, 1,
4.484848, 3.818182, -131.6116, 1, 0.3764706, 0, 1,
4.525252, 3.818182, -131.3809, 1, 0.3764706, 0, 1,
4.565657, 3.818182, -131.1558, 1, 0.3764706, 0, 1,
4.606061, 3.818182, -130.9362, 1, 0.3764706, 0, 1,
4.646465, 3.818182, -130.7223, 1, 0.3764706, 0, 1,
4.686869, 3.818182, -130.514, 1, 0.4823529, 0, 1,
4.727273, 3.818182, -130.3112, 1, 0.4823529, 0, 1,
4.767677, 3.818182, -130.1141, 1, 0.4823529, 0, 1,
4.808081, 3.818182, -129.9226, 1, 0.4823529, 0, 1,
4.848485, 3.818182, -129.7366, 1, 0.4823529, 0, 1,
4.888889, 3.818182, -129.5563, 1, 0.4823529, 0, 1,
4.929293, 3.818182, -129.3816, 1, 0.4823529, 0, 1,
4.969697, 3.818182, -129.2124, 1, 0.5843138, 0, 1,
5.010101, 3.818182, -129.0489, 1, 0.5843138, 0, 1,
5.050505, 3.818182, -128.891, 1, 0.5843138, 0, 1,
5.090909, 3.818182, -128.7386, 1, 0.5843138, 0, 1,
5.131313, 3.818182, -128.5919, 1, 0.5843138, 0, 1,
5.171717, 3.818182, -128.4507, 1, 0.5843138, 0, 1,
5.212121, 3.818182, -128.3152, 1, 0.5843138, 0, 1,
5.252525, 3.818182, -128.1852, 1, 0.5843138, 0, 1,
5.292929, 3.818182, -128.0609, 1, 0.5843138, 0, 1,
5.333333, 3.818182, -127.9421, 1, 0.6862745, 0, 1,
5.373737, 3.818182, -127.829, 1, 0.6862745, 0, 1,
5.414141, 3.818182, -127.7214, 1, 0.6862745, 0, 1,
5.454545, 3.818182, -127.6195, 1, 0.6862745, 0, 1,
5.494949, 3.818182, -127.5231, 1, 0.6862745, 0, 1,
5.535354, 3.818182, -127.4324, 1, 0.6862745, 0, 1,
5.575758, 3.818182, -127.3472, 1, 0.6862745, 0, 1,
5.616162, 3.818182, -127.2677, 1, 0.6862745, 0, 1,
5.656566, 3.818182, -127.1937, 1, 0.6862745, 0, 1,
5.69697, 3.818182, -127.1254, 1, 0.6862745, 0, 1,
5.737374, 3.818182, -127.0626, 1, 0.6862745, 0, 1,
5.777778, 3.818182, -127.0054, 1, 0.6862745, 0, 1,
5.818182, 3.818182, -126.9539, 1, 0.6862745, 0, 1,
5.858586, 3.818182, -126.9079, 1, 0.6862745, 0, 1,
5.89899, 3.818182, -126.8676, 1, 0.6862745, 0, 1,
5.939394, 3.818182, -126.8328, 1, 0.6862745, 0, 1,
5.979798, 3.818182, -126.8036, 1, 0.7921569, 0, 1,
6.020202, 3.818182, -126.7801, 1, 0.7921569, 0, 1,
6.060606, 3.818182, -126.7621, 1, 0.7921569, 0, 1,
6.10101, 3.818182, -126.7497, 1, 0.7921569, 0, 1,
6.141414, 3.818182, -126.743, 1, 0.7921569, 0, 1,
6.181818, 3.818182, -126.7418, 1, 0.7921569, 0, 1,
6.222222, 3.818182, -126.7462, 1, 0.7921569, 0, 1,
6.262626, 3.818182, -126.7563, 1, 0.7921569, 0, 1,
6.30303, 3.818182, -126.7719, 1, 0.7921569, 0, 1,
6.343434, 3.818182, -126.7931, 1, 0.7921569, 0, 1,
6.383838, 3.818182, -126.8199, 1, 0.6862745, 0, 1,
6.424242, 3.818182, -126.8524, 1, 0.6862745, 0, 1,
6.464646, 3.818182, -126.8904, 1, 0.6862745, 0, 1,
6.505051, 3.818182, -126.934, 1, 0.6862745, 0, 1,
6.545455, 3.818182, -126.9832, 1, 0.6862745, 0, 1,
6.585859, 3.818182, -127.038, 1, 0.6862745, 0, 1,
6.626263, 3.818182, -127.0985, 1, 0.6862745, 0, 1,
6.666667, 3.818182, -127.1645, 1, 0.6862745, 0, 1,
6.707071, 3.818182, -127.2361, 1, 0.6862745, 0, 1,
6.747475, 3.818182, -127.3133, 1, 0.6862745, 0, 1,
6.787879, 3.818182, -127.3961, 1, 0.6862745, 0, 1,
6.828283, 3.818182, -127.4845, 1, 0.6862745, 0, 1,
6.868687, 3.818182, -127.5786, 1, 0.6862745, 0, 1,
6.909091, 3.818182, -127.6782, 1, 0.6862745, 0, 1,
6.949495, 3.818182, -127.7834, 1, 0.6862745, 0, 1,
6.989899, 3.818182, -127.8942, 1, 0.6862745, 0, 1,
7.030303, 3.818182, -128.0106, 1, 0.6862745, 0, 1,
7.070707, 3.818182, -128.1326, 1, 0.5843138, 0, 1,
7.111111, 3.818182, -128.2602, 1, 0.5843138, 0, 1,
7.151515, 3.818182, -128.3934, 1, 0.5843138, 0, 1,
7.191919, 3.818182, -128.5322, 1, 0.5843138, 0, 1,
7.232323, 3.818182, -128.6766, 1, 0.5843138, 0, 1,
7.272727, 3.818182, -128.8266, 1, 0.5843138, 0, 1,
7.313131, 3.818182, -128.9822, 1, 0.5843138, 0, 1,
7.353535, 3.818182, -129.1434, 1, 0.5843138, 0, 1,
7.393939, 3.818182, -129.3102, 1, 0.4823529, 0, 1,
7.434343, 3.818182, -129.4826, 1, 0.4823529, 0, 1,
7.474748, 3.818182, -129.6606, 1, 0.4823529, 0, 1,
7.515152, 3.818182, -129.8442, 1, 0.4823529, 0, 1,
7.555555, 3.818182, -130.0334, 1, 0.4823529, 0, 1,
7.59596, 3.818182, -130.2282, 1, 0.4823529, 0, 1,
7.636364, 3.818182, -130.4286, 1, 0.4823529, 0, 1,
7.676768, 3.818182, -130.6346, 1, 0.3764706, 0, 1,
7.717172, 3.818182, -130.8462, 1, 0.3764706, 0, 1,
7.757576, 3.818182, -131.0634, 1, 0.3764706, 0, 1,
7.79798, 3.818182, -131.2861, 1, 0.3764706, 0, 1,
7.838384, 3.818182, -131.5145, 1, 0.3764706, 0, 1,
7.878788, 3.818182, -131.7485, 1, 0.3764706, 0, 1,
7.919192, 3.818182, -131.9881, 1, 0.2745098, 0, 1,
7.959596, 3.818182, -132.2333, 1, 0.2745098, 0, 1,
8, 3.818182, -132.4841, 1, 0.2745098, 0, 1,
4, 3.858586, -134.8875, 1, 0.06666667, 0, 1,
4.040404, 3.858586, -134.5958, 1, 0.06666667, 0, 1,
4.080808, 3.858586, -134.3096, 1, 0.06666667, 0, 1,
4.121212, 3.858586, -134.0289, 1, 0.1686275, 0, 1,
4.161616, 3.858586, -133.7536, 1, 0.1686275, 0, 1,
4.20202, 3.858586, -133.4838, 1, 0.1686275, 0, 1,
4.242424, 3.858586, -133.2195, 1, 0.1686275, 0, 1,
4.282828, 3.858586, -132.9607, 1, 0.2745098, 0, 1,
4.323232, 3.858586, -132.7074, 1, 0.2745098, 0, 1,
4.363636, 3.858586, -132.4595, 1, 0.2745098, 0, 1,
4.40404, 3.858586, -132.2172, 1, 0.2745098, 0, 1,
4.444445, 3.858586, -131.9803, 1, 0.2745098, 0, 1,
4.484848, 3.858586, -131.7489, 1, 0.3764706, 0, 1,
4.525252, 3.858586, -131.5229, 1, 0.3764706, 0, 1,
4.565657, 3.858586, -131.3025, 1, 0.3764706, 0, 1,
4.606061, 3.858586, -131.0876, 1, 0.3764706, 0, 1,
4.646465, 3.858586, -130.8781, 1, 0.3764706, 0, 1,
4.686869, 3.858586, -130.6741, 1, 0.3764706, 0, 1,
4.727273, 3.858586, -130.4756, 1, 0.4823529, 0, 1,
4.767677, 3.858586, -130.2826, 1, 0.4823529, 0, 1,
4.808081, 3.858586, -130.095, 1, 0.4823529, 0, 1,
4.848485, 3.858586, -129.9129, 1, 0.4823529, 0, 1,
4.888889, 3.858586, -129.7364, 1, 0.4823529, 0, 1,
4.929293, 3.858586, -129.5653, 1, 0.4823529, 0, 1,
4.969697, 3.858586, -129.3997, 1, 0.4823529, 0, 1,
5.010101, 3.858586, -129.2395, 1, 0.5843138, 0, 1,
5.050505, 3.858586, -129.0849, 1, 0.5843138, 0, 1,
5.090909, 3.858586, -128.9357, 1, 0.5843138, 0, 1,
5.131313, 3.858586, -128.792, 1, 0.5843138, 0, 1,
5.171717, 3.858586, -128.6538, 1, 0.5843138, 0, 1,
5.212121, 3.858586, -128.5211, 1, 0.5843138, 0, 1,
5.252525, 3.858586, -128.3939, 1, 0.5843138, 0, 1,
5.292929, 3.858586, -128.2721, 1, 0.5843138, 0, 1,
5.333333, 3.858586, -128.1558, 1, 0.5843138, 0, 1,
5.373737, 3.858586, -128.045, 1, 0.6862745, 0, 1,
5.414141, 3.858586, -127.9397, 1, 0.6862745, 0, 1,
5.454545, 3.858586, -127.8399, 1, 0.6862745, 0, 1,
5.494949, 3.858586, -127.7456, 1, 0.6862745, 0, 1,
5.535354, 3.858586, -127.6567, 1, 0.6862745, 0, 1,
5.575758, 3.858586, -127.5733, 1, 0.6862745, 0, 1,
5.616162, 3.858586, -127.4954, 1, 0.6862745, 0, 1,
5.656566, 3.858586, -127.423, 1, 0.6862745, 0, 1,
5.69697, 3.858586, -127.3561, 1, 0.6862745, 0, 1,
5.737374, 3.858586, -127.2946, 1, 0.6862745, 0, 1,
5.777778, 3.858586, -127.2387, 1, 0.6862745, 0, 1,
5.818182, 3.858586, -127.1882, 1, 0.6862745, 0, 1,
5.858586, 3.858586, -127.1432, 1, 0.6862745, 0, 1,
5.89899, 3.858586, -127.1036, 1, 0.6862745, 0, 1,
5.939394, 3.858586, -127.0696, 1, 0.6862745, 0, 1,
5.979798, 3.858586, -127.0411, 1, 0.6862745, 0, 1,
6.020202, 3.858586, -127.018, 1, 0.6862745, 0, 1,
6.060606, 3.858586, -127.0004, 1, 0.6862745, 0, 1,
6.10101, 3.858586, -126.9883, 1, 0.6862745, 0, 1,
6.141414, 3.858586, -126.9816, 1, 0.6862745, 0, 1,
6.181818, 3.858586, -126.9805, 1, 0.6862745, 0, 1,
6.222222, 3.858586, -126.9848, 1, 0.6862745, 0, 1,
6.262626, 3.858586, -126.9947, 1, 0.6862745, 0, 1,
6.30303, 3.858586, -127.01, 1, 0.6862745, 0, 1,
6.343434, 3.858586, -127.0307, 1, 0.6862745, 0, 1,
6.383838, 3.858586, -127.057, 1, 0.6862745, 0, 1,
6.424242, 3.858586, -127.0888, 1, 0.6862745, 0, 1,
6.464646, 3.858586, -127.126, 1, 0.6862745, 0, 1,
6.505051, 3.858586, -127.1687, 1, 0.6862745, 0, 1,
6.545455, 3.858586, -127.2169, 1, 0.6862745, 0, 1,
6.585859, 3.858586, -127.2706, 1, 0.6862745, 0, 1,
6.626263, 3.858586, -127.3297, 1, 0.6862745, 0, 1,
6.666667, 3.858586, -127.3944, 1, 0.6862745, 0, 1,
6.707071, 3.858586, -127.4645, 1, 0.6862745, 0, 1,
6.747475, 3.858586, -127.5401, 1, 0.6862745, 0, 1,
6.787879, 3.858586, -127.6212, 1, 0.6862745, 0, 1,
6.828283, 3.858586, -127.7078, 1, 0.6862745, 0, 1,
6.868687, 3.858586, -127.7998, 1, 0.6862745, 0, 1,
6.909091, 3.858586, -127.8974, 1, 0.6862745, 0, 1,
6.949495, 3.858586, -128.0004, 1, 0.6862745, 0, 1,
6.989899, 3.858586, -128.1089, 1, 0.5843138, 0, 1,
7.030303, 3.858586, -128.2229, 1, 0.5843138, 0, 1,
7.070707, 3.858586, -128.3423, 1, 0.5843138, 0, 1,
7.111111, 3.858586, -128.4673, 1, 0.5843138, 0, 1,
7.151515, 3.858586, -128.5977, 1, 0.5843138, 0, 1,
7.191919, 3.858586, -128.7336, 1, 0.5843138, 0, 1,
7.232323, 3.858586, -128.875, 1, 0.5843138, 0, 1,
7.272727, 3.858586, -129.0219, 1, 0.5843138, 0, 1,
7.313131, 3.858586, -129.1742, 1, 0.5843138, 0, 1,
7.353535, 3.858586, -129.3321, 1, 0.4823529, 0, 1,
7.393939, 3.858586, -129.4954, 1, 0.4823529, 0, 1,
7.434343, 3.858586, -129.6642, 1, 0.4823529, 0, 1,
7.474748, 3.858586, -129.8385, 1, 0.4823529, 0, 1,
7.515152, 3.858586, -130.0183, 1, 0.4823529, 0, 1,
7.555555, 3.858586, -130.2035, 1, 0.4823529, 0, 1,
7.59596, 3.858586, -130.3943, 1, 0.4823529, 0, 1,
7.636364, 3.858586, -130.5905, 1, 0.3764706, 0, 1,
7.676768, 3.858586, -130.7922, 1, 0.3764706, 0, 1,
7.717172, 3.858586, -130.9994, 1, 0.3764706, 0, 1,
7.757576, 3.858586, -131.212, 1, 0.3764706, 0, 1,
7.79798, 3.858586, -131.4302, 1, 0.3764706, 0, 1,
7.838384, 3.858586, -131.6538, 1, 0.3764706, 0, 1,
7.878788, 3.858586, -131.8829, 1, 0.2745098, 0, 1,
7.919192, 3.858586, -132.1175, 1, 0.2745098, 0, 1,
7.959596, 3.858586, -132.3576, 1, 0.2745098, 0, 1,
8, 3.858586, -132.6031, 1, 0.2745098, 0, 1,
4, 3.89899, -134.9666, 1, 0.06666667, 0, 1,
4.040404, 3.89899, -134.6809, 1, 0.06666667, 0, 1,
4.080808, 3.89899, -134.4006, 1, 0.06666667, 0, 1,
4.121212, 3.89899, -134.1256, 1, 0.1686275, 0, 1,
4.161616, 3.89899, -133.8561, 1, 0.1686275, 0, 1,
4.20202, 3.89899, -133.5918, 1, 0.1686275, 0, 1,
4.242424, 3.89899, -133.333, 1, 0.1686275, 0, 1,
4.282828, 3.89899, -133.0795, 1, 0.1686275, 0, 1,
4.323232, 3.89899, -132.8314, 1, 0.2745098, 0, 1,
4.363636, 3.89899, -132.5887, 1, 0.2745098, 0, 1,
4.40404, 3.89899, -132.3513, 1, 0.2745098, 0, 1,
4.444445, 3.89899, -132.1193, 1, 0.2745098, 0, 1,
4.484848, 3.89899, -131.8927, 1, 0.2745098, 0, 1,
4.525252, 3.89899, -131.6714, 1, 0.3764706, 0, 1,
4.565657, 3.89899, -131.4555, 1, 0.3764706, 0, 1,
4.606061, 3.89899, -131.245, 1, 0.3764706, 0, 1,
4.646465, 3.89899, -131.0398, 1, 0.3764706, 0, 1,
4.686869, 3.89899, -130.8401, 1, 0.3764706, 0, 1,
4.727273, 3.89899, -130.6456, 1, 0.3764706, 0, 1,
4.767677, 3.89899, -130.4566, 1, 0.4823529, 0, 1,
4.808081, 3.89899, -130.2729, 1, 0.4823529, 0, 1,
4.848485, 3.89899, -130.0946, 1, 0.4823529, 0, 1,
4.888889, 3.89899, -129.9217, 1, 0.4823529, 0, 1,
4.929293, 3.89899, -129.7541, 1, 0.4823529, 0, 1,
4.969697, 3.89899, -129.5919, 1, 0.4823529, 0, 1,
5.010101, 3.89899, -129.4351, 1, 0.4823529, 0, 1,
5.050505, 3.89899, -129.2836, 1, 0.5843138, 0, 1,
5.090909, 3.89899, -129.1375, 1, 0.5843138, 0, 1,
5.131313, 3.89899, -128.9968, 1, 0.5843138, 0, 1,
5.171717, 3.89899, -128.8614, 1, 0.5843138, 0, 1,
5.212121, 3.89899, -128.7315, 1, 0.5843138, 0, 1,
5.252525, 3.89899, -128.6068, 1, 0.5843138, 0, 1,
5.292929, 3.89899, -128.4876, 1, 0.5843138, 0, 1,
5.333333, 3.89899, -128.3737, 1, 0.5843138, 0, 1,
5.373737, 3.89899, -128.2652, 1, 0.5843138, 0, 1,
5.414141, 3.89899, -128.1621, 1, 0.5843138, 0, 1,
5.454545, 3.89899, -128.0643, 1, 0.5843138, 0, 1,
5.494949, 3.89899, -127.9719, 1, 0.6862745, 0, 1,
5.535354, 3.89899, -127.8849, 1, 0.6862745, 0, 1,
5.575758, 3.89899, -127.8032, 1, 0.6862745, 0, 1,
5.616162, 3.89899, -127.7269, 1, 0.6862745, 0, 1,
5.656566, 3.89899, -127.656, 1, 0.6862745, 0, 1,
5.69697, 3.89899, -127.5904, 1, 0.6862745, 0, 1,
5.737374, 3.89899, -127.5303, 1, 0.6862745, 0, 1,
5.777778, 3.89899, -127.4754, 1, 0.6862745, 0, 1,
5.818182, 3.89899, -127.426, 1, 0.6862745, 0, 1,
5.858586, 3.89899, -127.3819, 1, 0.6862745, 0, 1,
5.89899, 3.89899, -127.3432, 1, 0.6862745, 0, 1,
5.939394, 3.89899, -127.3099, 1, 0.6862745, 0, 1,
5.979798, 3.89899, -127.2819, 1, 0.6862745, 0, 1,
6.020202, 3.89899, -127.2593, 1, 0.6862745, 0, 1,
6.060606, 3.89899, -127.2421, 1, 0.6862745, 0, 1,
6.10101, 3.89899, -127.2302, 1, 0.6862745, 0, 1,
6.141414, 3.89899, -127.2237, 1, 0.6862745, 0, 1,
6.181818, 3.89899, -127.2226, 1, 0.6862745, 0, 1,
6.222222, 3.89899, -127.2269, 1, 0.6862745, 0, 1,
6.262626, 3.89899, -127.2365, 1, 0.6862745, 0, 1,
6.30303, 3.89899, -127.2515, 1, 0.6862745, 0, 1,
6.343434, 3.89899, -127.2718, 1, 0.6862745, 0, 1,
6.383838, 3.89899, -127.2975, 1, 0.6862745, 0, 1,
6.424242, 3.89899, -127.3286, 1, 0.6862745, 0, 1,
6.464646, 3.89899, -127.3651, 1, 0.6862745, 0, 1,
6.505051, 3.89899, -127.4069, 1, 0.6862745, 0, 1,
6.545455, 3.89899, -127.4541, 1, 0.6862745, 0, 1,
6.585859, 3.89899, -127.5067, 1, 0.6862745, 0, 1,
6.626263, 3.89899, -127.5646, 1, 0.6862745, 0, 1,
6.666667, 3.89899, -127.628, 1, 0.6862745, 0, 1,
6.707071, 3.89899, -127.6966, 1, 0.6862745, 0, 1,
6.747475, 3.89899, -127.7707, 1, 0.6862745, 0, 1,
6.787879, 3.89899, -127.8501, 1, 0.6862745, 0, 1,
6.828283, 3.89899, -127.9349, 1, 0.6862745, 0, 1,
6.868687, 3.89899, -128.025, 1, 0.6862745, 0, 1,
6.909091, 3.89899, -128.1206, 1, 0.5843138, 0, 1,
6.949495, 3.89899, -128.2215, 1, 0.5843138, 0, 1,
6.989899, 3.89899, -128.3277, 1, 0.5843138, 0, 1,
7.030303, 3.89899, -128.4394, 1, 0.5843138, 0, 1,
7.070707, 3.89899, -128.5564, 1, 0.5843138, 0, 1,
7.111111, 3.89899, -128.6787, 1, 0.5843138, 0, 1,
7.151515, 3.89899, -128.8065, 1, 0.5843138, 0, 1,
7.191919, 3.89899, -128.9396, 1, 0.5843138, 0, 1,
7.232323, 3.89899, -129.0781, 1, 0.5843138, 0, 1,
7.272727, 3.89899, -129.2219, 1, 0.5843138, 0, 1,
7.313131, 3.89899, -129.3711, 1, 0.4823529, 0, 1,
7.353535, 3.89899, -129.5257, 1, 0.4823529, 0, 1,
7.393939, 3.89899, -129.6857, 1, 0.4823529, 0, 1,
7.434343, 3.89899, -129.851, 1, 0.4823529, 0, 1,
7.474748, 3.89899, -130.0217, 1, 0.4823529, 0, 1,
7.515152, 3.89899, -130.1978, 1, 0.4823529, 0, 1,
7.555555, 3.89899, -130.3792, 1, 0.4823529, 0, 1,
7.59596, 3.89899, -130.566, 1, 0.3764706, 0, 1,
7.636364, 3.89899, -130.7582, 1, 0.3764706, 0, 1,
7.676768, 3.89899, -130.9557, 1, 0.3764706, 0, 1,
7.717172, 3.89899, -131.1586, 1, 0.3764706, 0, 1,
7.757576, 3.89899, -131.3669, 1, 0.3764706, 0, 1,
7.79798, 3.89899, -131.5806, 1, 0.3764706, 0, 1,
7.838384, 3.89899, -131.7996, 1, 0.2745098, 0, 1,
7.878788, 3.89899, -132.024, 1, 0.2745098, 0, 1,
7.919192, 3.89899, -132.2537, 1, 0.2745098, 0, 1,
7.959596, 3.89899, -132.4888, 1, 0.2745098, 0, 1,
8, 3.89899, -132.7293, 1, 0.2745098, 0, 1,
4, 3.939394, -135.0538, 1, 0.06666667, 0, 1,
4.040404, 3.939394, -134.774, 1, 0.06666667, 0, 1,
4.080808, 3.939394, -134.4994, 1, 0.06666667, 0, 1,
4.121212, 3.939394, -134.23, 1, 0.1686275, 0, 1,
4.161616, 3.939394, -133.966, 1, 0.1686275, 0, 1,
4.20202, 3.939394, -133.7071, 1, 0.1686275, 0, 1,
4.242424, 3.939394, -133.4536, 1, 0.1686275, 0, 1,
4.282828, 3.939394, -133.2053, 1, 0.1686275, 0, 1,
4.323232, 3.939394, -132.9622, 1, 0.2745098, 0, 1,
4.363636, 3.939394, -132.7244, 1, 0.2745098, 0, 1,
4.40404, 3.939394, -132.4919, 1, 0.2745098, 0, 1,
4.444445, 3.939394, -132.2646, 1, 0.2745098, 0, 1,
4.484848, 3.939394, -132.0426, 1, 0.2745098, 0, 1,
4.525252, 3.939394, -131.8259, 1, 0.2745098, 0, 1,
4.565657, 3.939394, -131.6144, 1, 0.3764706, 0, 1,
4.606061, 3.939394, -131.4082, 1, 0.3764706, 0, 1,
4.646465, 3.939394, -131.2072, 1, 0.3764706, 0, 1,
4.686869, 3.939394, -131.0115, 1, 0.3764706, 0, 1,
4.727273, 3.939394, -130.8211, 1, 0.3764706, 0, 1,
4.767677, 3.939394, -130.6359, 1, 0.3764706, 0, 1,
4.808081, 3.939394, -130.4559, 1, 0.4823529, 0, 1,
4.848485, 3.939394, -130.2813, 1, 0.4823529, 0, 1,
4.888889, 3.939394, -130.1119, 1, 0.4823529, 0, 1,
4.929293, 3.939394, -129.9477, 1, 0.4823529, 0, 1,
4.969697, 3.939394, -129.7888, 1, 0.4823529, 0, 1,
5.010101, 3.939394, -129.6352, 1, 0.4823529, 0, 1,
5.050505, 3.939394, -129.4868, 1, 0.4823529, 0, 1,
5.090909, 3.939394, -129.3437, 1, 0.4823529, 0, 1,
5.131313, 3.939394, -129.2059, 1, 0.5843138, 0, 1,
5.171717, 3.939394, -129.0733, 1, 0.5843138, 0, 1,
5.212121, 3.939394, -128.9459, 1, 0.5843138, 0, 1,
5.252525, 3.939394, -128.8239, 1, 0.5843138, 0, 1,
5.292929, 3.939394, -128.707, 1, 0.5843138, 0, 1,
5.333333, 3.939394, -128.5955, 1, 0.5843138, 0, 1,
5.373737, 3.939394, -128.4892, 1, 0.5843138, 0, 1,
5.414141, 3.939394, -128.3882, 1, 0.5843138, 0, 1,
5.454545, 3.939394, -128.2924, 1, 0.5843138, 0, 1,
5.494949, 3.939394, -128.2019, 1, 0.5843138, 0, 1,
5.535354, 3.939394, -128.1166, 1, 0.5843138, 0, 1,
5.575758, 3.939394, -128.0366, 1, 0.6862745, 0, 1,
5.616162, 3.939394, -127.9619, 1, 0.6862745, 0, 1,
5.656566, 3.939394, -127.8924, 1, 0.6862745, 0, 1,
5.69697, 3.939394, -127.8282, 1, 0.6862745, 0, 1,
5.737374, 3.939394, -127.7693, 1, 0.6862745, 0, 1,
5.777778, 3.939394, -127.7156, 1, 0.6862745, 0, 1,
5.818182, 3.939394, -127.6671, 1, 0.6862745, 0, 1,
5.858586, 3.939394, -127.624, 1, 0.6862745, 0, 1,
5.89899, 3.939394, -127.586, 1, 0.6862745, 0, 1,
5.939394, 3.939394, -127.5534, 1, 0.6862745, 0, 1,
5.979798, 3.939394, -127.526, 1, 0.6862745, 0, 1,
6.020202, 3.939394, -127.5038, 1, 0.6862745, 0, 1,
6.060606, 3.939394, -127.487, 1, 0.6862745, 0, 1,
6.10101, 3.939394, -127.4753, 1, 0.6862745, 0, 1,
6.141414, 3.939394, -127.469, 1, 0.6862745, 0, 1,
6.181818, 3.939394, -127.4679, 1, 0.6862745, 0, 1,
6.222222, 3.939394, -127.4721, 1, 0.6862745, 0, 1,
6.262626, 3.939394, -127.4815, 1, 0.6862745, 0, 1,
6.30303, 3.939394, -127.4961, 1, 0.6862745, 0, 1,
6.343434, 3.939394, -127.5161, 1, 0.6862745, 0, 1,
6.383838, 3.939394, -127.5413, 1, 0.6862745, 0, 1,
6.424242, 3.939394, -127.5717, 1, 0.6862745, 0, 1,
6.464646, 3.939394, -127.6075, 1, 0.6862745, 0, 1,
6.505051, 3.939394, -127.6484, 1, 0.6862745, 0, 1,
6.545455, 3.939394, -127.6947, 1, 0.6862745, 0, 1,
6.585859, 3.939394, -127.7462, 1, 0.6862745, 0, 1,
6.626263, 3.939394, -127.8029, 1, 0.6862745, 0, 1,
6.666667, 3.939394, -127.865, 1, 0.6862745, 0, 1,
6.707071, 3.939394, -127.9322, 1, 0.6862745, 0, 1,
6.747475, 3.939394, -128.0048, 1, 0.6862745, 0, 1,
6.787879, 3.939394, -128.0826, 1, 0.5843138, 0, 1,
6.828283, 3.939394, -128.1656, 1, 0.5843138, 0, 1,
6.868687, 3.939394, -128.2539, 1, 0.5843138, 0, 1,
6.909091, 3.939394, -128.3475, 1, 0.5843138, 0, 1,
6.949495, 3.939394, -128.4464, 1, 0.5843138, 0, 1,
6.989899, 3.939394, -128.5504, 1, 0.5843138, 0, 1,
7.030303, 3.939394, -128.6598, 1, 0.5843138, 0, 1,
7.070707, 3.939394, -128.7744, 1, 0.5843138, 0, 1,
7.111111, 3.939394, -128.8943, 1, 0.5843138, 0, 1,
7.151515, 3.939394, -129.0194, 1, 0.5843138, 0, 1,
7.191919, 3.939394, -129.1498, 1, 0.5843138, 0, 1,
7.232323, 3.939394, -129.2855, 1, 0.5843138, 0, 1,
7.272727, 3.939394, -129.4264, 1, 0.4823529, 0, 1,
7.313131, 3.939394, -129.5726, 1, 0.4823529, 0, 1,
7.353535, 3.939394, -129.724, 1, 0.4823529, 0, 1,
7.393939, 3.939394, -129.8807, 1, 0.4823529, 0, 1,
7.434343, 3.939394, -130.0426, 1, 0.4823529, 0, 1,
7.474748, 3.939394, -130.2099, 1, 0.4823529, 0, 1,
7.515152, 3.939394, -130.3823, 1, 0.4823529, 0, 1,
7.555555, 3.939394, -130.56, 1, 0.3764706, 0, 1,
7.59596, 3.939394, -130.743, 1, 0.3764706, 0, 1,
7.636364, 3.939394, -130.9313, 1, 0.3764706, 0, 1,
7.676768, 3.939394, -131.1248, 1, 0.3764706, 0, 1,
7.717172, 3.939394, -131.3236, 1, 0.3764706, 0, 1,
7.757576, 3.939394, -131.5276, 1, 0.3764706, 0, 1,
7.79798, 3.939394, -131.7369, 1, 0.3764706, 0, 1,
7.838384, 3.939394, -131.9514, 1, 0.2745098, 0, 1,
7.878788, 3.939394, -132.1712, 1, 0.2745098, 0, 1,
7.919192, 3.939394, -132.3963, 1, 0.2745098, 0, 1,
7.959596, 3.939394, -132.6266, 1, 0.2745098, 0, 1,
8, 3.939394, -132.8622, 1, 0.2745098, 0, 1,
4, 3.979798, -135.1488, 1, 0.06666667, 0, 1,
4.040404, 3.979798, -134.8746, 1, 0.06666667, 0, 1,
4.080808, 3.979798, -134.6055, 1, 0.06666667, 0, 1,
4.121212, 3.979798, -134.3416, 1, 0.06666667, 0, 1,
4.161616, 3.979798, -134.0829, 1, 0.1686275, 0, 1,
4.20202, 3.979798, -133.8293, 1, 0.1686275, 0, 1,
4.242424, 3.979798, -133.5809, 1, 0.1686275, 0, 1,
4.282828, 3.979798, -133.3376, 1, 0.1686275, 0, 1,
4.323232, 3.979798, -133.0994, 1, 0.1686275, 0, 1,
4.363636, 3.979798, -132.8665, 1, 0.2745098, 0, 1,
4.40404, 3.979798, -132.6386, 1, 0.2745098, 0, 1,
4.444445, 3.979798, -132.4159, 1, 0.2745098, 0, 1,
4.484848, 3.979798, -132.1984, 1, 0.2745098, 0, 1,
4.525252, 3.979798, -131.9861, 1, 0.2745098, 0, 1,
4.565657, 3.979798, -131.7788, 1, 0.3764706, 0, 1,
4.606061, 3.979798, -131.5768, 1, 0.3764706, 0, 1,
4.646465, 3.979798, -131.3799, 1, 0.3764706, 0, 1,
4.686869, 3.979798, -131.1881, 1, 0.3764706, 0, 1,
4.727273, 3.979798, -131.0015, 1, 0.3764706, 0, 1,
4.767677, 3.979798, -130.8201, 1, 0.3764706, 0, 1,
4.808081, 3.979798, -130.6438, 1, 0.3764706, 0, 1,
4.848485, 3.979798, -130.4726, 1, 0.4823529, 0, 1,
4.888889, 3.979798, -130.3067, 1, 0.4823529, 0, 1,
4.929293, 3.979798, -130.1458, 1, 0.4823529, 0, 1,
4.969697, 3.979798, -129.9901, 1, 0.4823529, 0, 1,
5.010101, 3.979798, -129.8396, 1, 0.4823529, 0, 1,
5.050505, 3.979798, -129.6942, 1, 0.4823529, 0, 1,
5.090909, 3.979798, -129.554, 1, 0.4823529, 0, 1,
5.131313, 3.979798, -129.419, 1, 0.4823529, 0, 1,
5.171717, 3.979798, -129.289, 1, 0.5843138, 0, 1,
5.212121, 3.979798, -129.1643, 1, 0.5843138, 0, 1,
5.252525, 3.979798, -129.0447, 1, 0.5843138, 0, 1,
5.292929, 3.979798, -128.9302, 1, 0.5843138, 0, 1,
5.333333, 3.979798, -128.8209, 1, 0.5843138, 0, 1,
5.373737, 3.979798, -128.7168, 1, 0.5843138, 0, 1,
5.414141, 3.979798, -128.6178, 1, 0.5843138, 0, 1,
5.454545, 3.979798, -128.5239, 1, 0.5843138, 0, 1,
5.494949, 3.979798, -128.4353, 1, 0.5843138, 0, 1,
5.535354, 3.979798, -128.3517, 1, 0.5843138, 0, 1,
5.575758, 3.979798, -128.2733, 1, 0.5843138, 0, 1,
5.616162, 3.979798, -128.2001, 1, 0.5843138, 0, 1,
5.656566, 3.979798, -128.132, 1, 0.5843138, 0, 1,
5.69697, 3.979798, -128.0691, 1, 0.5843138, 0, 1,
5.737374, 3.979798, -128.0114, 1, 0.6862745, 0, 1,
5.777778, 3.979798, -127.9588, 1, 0.6862745, 0, 1,
5.818182, 3.979798, -127.9113, 1, 0.6862745, 0, 1,
5.858586, 3.979798, -127.869, 1, 0.6862745, 0, 1,
5.89899, 3.979798, -127.8318, 1, 0.6862745, 0, 1,
5.939394, 3.979798, -127.7999, 1, 0.6862745, 0, 1,
5.979798, 3.979798, -127.773, 1, 0.6862745, 0, 1,
6.020202, 3.979798, -127.7513, 1, 0.6862745, 0, 1,
6.060606, 3.979798, -127.7348, 1, 0.6862745, 0, 1,
6.10101, 3.979798, -127.7234, 1, 0.6862745, 0, 1,
6.141414, 3.979798, -127.7172, 1, 0.6862745, 0, 1,
6.181818, 3.979798, -127.7161, 1, 0.6862745, 0, 1,
6.222222, 3.979798, -127.7202, 1, 0.6862745, 0, 1,
6.262626, 3.979798, -127.7294, 1, 0.6862745, 0, 1,
6.30303, 3.979798, -127.7438, 1, 0.6862745, 0, 1,
6.343434, 3.979798, -127.7633, 1, 0.6862745, 0, 1,
6.383838, 3.979798, -127.788, 1, 0.6862745, 0, 1,
6.424242, 3.979798, -127.8179, 1, 0.6862745, 0, 1,
6.464646, 3.979798, -127.8529, 1, 0.6862745, 0, 1,
6.505051, 3.979798, -127.893, 1, 0.6862745, 0, 1,
6.545455, 3.979798, -127.9383, 1, 0.6862745, 0, 1,
6.585859, 3.979798, -127.9888, 1, 0.6862745, 0, 1,
6.626263, 3.979798, -128.0444, 1, 0.6862745, 0, 1,
6.666667, 3.979798, -128.1051, 1, 0.5843138, 0, 1,
6.707071, 3.979798, -128.1711, 1, 0.5843138, 0, 1,
6.747475, 3.979798, -128.2421, 1, 0.5843138, 0, 1,
6.787879, 3.979798, -128.3184, 1, 0.5843138, 0, 1,
6.828283, 3.979798, -128.3997, 1, 0.5843138, 0, 1,
6.868687, 3.979798, -128.4863, 1, 0.5843138, 0, 1,
6.909091, 3.979798, -128.578, 1, 0.5843138, 0, 1,
6.949495, 3.979798, -128.6748, 1, 0.5843138, 0, 1,
6.989899, 3.979798, -128.7768, 1, 0.5843138, 0, 1,
7.030303, 3.979798, -128.8839, 1, 0.5843138, 0, 1,
7.070707, 3.979798, -128.9962, 1, 0.5843138, 0, 1,
7.111111, 3.979798, -129.1137, 1, 0.5843138, 0, 1,
7.151515, 3.979798, -129.2363, 1, 0.5843138, 0, 1,
7.191919, 3.979798, -129.364, 1, 0.4823529, 0, 1,
7.232323, 3.979798, -129.497, 1, 0.4823529, 0, 1,
7.272727, 3.979798, -129.635, 1, 0.4823529, 0, 1,
7.313131, 3.979798, -129.7782, 1, 0.4823529, 0, 1,
7.353535, 3.979798, -129.9266, 1, 0.4823529, 0, 1,
7.393939, 3.979798, -130.0801, 1, 0.4823529, 0, 1,
7.434343, 3.979798, -130.2388, 1, 0.4823529, 0, 1,
7.474748, 3.979798, -130.4027, 1, 0.4823529, 0, 1,
7.515152, 3.979798, -130.5716, 1, 0.3764706, 0, 1,
7.555555, 3.979798, -130.7458, 1, 0.3764706, 0, 1,
7.59596, 3.979798, -130.9251, 1, 0.3764706, 0, 1,
7.636364, 3.979798, -131.1095, 1, 0.3764706, 0, 1,
7.676768, 3.979798, -131.2991, 1, 0.3764706, 0, 1,
7.717172, 3.979798, -131.4939, 1, 0.3764706, 0, 1,
7.757576, 3.979798, -131.6938, 1, 0.3764706, 0, 1,
7.79798, 3.979798, -131.8988, 1, 0.2745098, 0, 1,
7.838384, 3.979798, -132.1091, 1, 0.2745098, 0, 1,
7.878788, 3.979798, -132.3244, 1, 0.2745098, 0, 1,
7.919192, 3.979798, -132.545, 1, 0.2745098, 0, 1,
7.959596, 3.979798, -132.7706, 1, 0.2745098, 0, 1,
8, 3.979798, -133.0015, 1, 0.2745098, 0, 1,
4, 4.020202, -135.2511, 1, 0.06666667, 0, 1,
4.040404, 4.020202, -134.9823, 1, 0.06666667, 0, 1,
4.080808, 4.020202, -134.7187, 1, 0.06666667, 0, 1,
4.121212, 4.020202, -134.46, 1, 0.06666667, 0, 1,
4.161616, 4.020202, -134.2065, 1, 0.1686275, 0, 1,
4.20202, 4.020202, -133.9579, 1, 0.1686275, 0, 1,
4.242424, 4.020202, -133.7145, 1, 0.1686275, 0, 1,
4.282828, 4.020202, -133.476, 1, 0.1686275, 0, 1,
4.323232, 4.020202, -133.2427, 1, 0.1686275, 0, 1,
4.363636, 4.020202, -133.0144, 1, 0.2745098, 0, 1,
4.40404, 4.020202, -132.7911, 1, 0.2745098, 0, 1,
4.444445, 4.020202, -132.5729, 1, 0.2745098, 0, 1,
4.484848, 4.020202, -132.3597, 1, 0.2745098, 0, 1,
4.525252, 4.020202, -132.1516, 1, 0.2745098, 0, 1,
4.565657, 4.020202, -131.9485, 1, 0.2745098, 0, 1,
4.606061, 4.020202, -131.7505, 1, 0.3764706, 0, 1,
4.646465, 4.020202, -131.5575, 1, 0.3764706, 0, 1,
4.686869, 4.020202, -131.3696, 1, 0.3764706, 0, 1,
4.727273, 4.020202, -131.1867, 1, 0.3764706, 0, 1,
4.767677, 4.020202, -131.0089, 1, 0.3764706, 0, 1,
4.808081, 4.020202, -130.8361, 1, 0.3764706, 0, 1,
4.848485, 4.020202, -130.6684, 1, 0.3764706, 0, 1,
4.888889, 4.020202, -130.5058, 1, 0.4823529, 0, 1,
4.929293, 4.020202, -130.3481, 1, 0.4823529, 0, 1,
4.969697, 4.020202, -130.1956, 1, 0.4823529, 0, 1,
5.010101, 4.020202, -130.048, 1, 0.4823529, 0, 1,
5.050505, 4.020202, -129.9056, 1, 0.4823529, 0, 1,
5.090909, 4.020202, -129.7682, 1, 0.4823529, 0, 1,
5.131313, 4.020202, -129.6358, 1, 0.4823529, 0, 1,
5.171717, 4.020202, -129.5085, 1, 0.4823529, 0, 1,
5.212121, 4.020202, -129.3862, 1, 0.4823529, 0, 1,
5.252525, 4.020202, -129.269, 1, 0.5843138, 0, 1,
5.292929, 4.020202, -129.1568, 1, 0.5843138, 0, 1,
5.333333, 4.020202, -129.0497, 1, 0.5843138, 0, 1,
5.373737, 4.020202, -128.9477, 1, 0.5843138, 0, 1,
5.414141, 4.020202, -128.8507, 1, 0.5843138, 0, 1,
5.454545, 4.020202, -128.7587, 1, 0.5843138, 0, 1,
5.494949, 4.020202, -128.6718, 1, 0.5843138, 0, 1,
5.535354, 4.020202, -128.5899, 1, 0.5843138, 0, 1,
5.575758, 4.020202, -128.5131, 1, 0.5843138, 0, 1,
5.616162, 4.020202, -128.4414, 1, 0.5843138, 0, 1,
5.656566, 4.020202, -128.3746, 1, 0.5843138, 0, 1,
5.69697, 4.020202, -128.313, 1, 0.5843138, 0, 1,
5.737374, 4.020202, -128.2564, 1, 0.5843138, 0, 1,
5.777778, 4.020202, -128.2048, 1, 0.5843138, 0, 1,
5.818182, 4.020202, -128.1583, 1, 0.5843138, 0, 1,
5.858586, 4.020202, -128.1169, 1, 0.5843138, 0, 1,
5.89899, 4.020202, -128.0804, 1, 0.5843138, 0, 1,
5.939394, 4.020202, -128.0491, 1, 0.6862745, 0, 1,
5.979798, 4.020202, -128.0228, 1, 0.6862745, 0, 1,
6.020202, 4.020202, -128.0015, 1, 0.6862745, 0, 1,
6.060606, 4.020202, -127.9853, 1, 0.6862745, 0, 1,
6.10101, 4.020202, -127.9742, 1, 0.6862745, 0, 1,
6.141414, 4.020202, -127.9681, 1, 0.6862745, 0, 1,
6.181818, 4.020202, -127.967, 1, 0.6862745, 0, 1,
6.222222, 4.020202, -127.971, 1, 0.6862745, 0, 1,
6.262626, 4.020202, -127.98, 1, 0.6862745, 0, 1,
6.30303, 4.020202, -127.9941, 1, 0.6862745, 0, 1,
6.343434, 4.020202, -128.0133, 1, 0.6862745, 0, 1,
6.383838, 4.020202, -128.0375, 1, 0.6862745, 0, 1,
6.424242, 4.020202, -128.0667, 1, 0.5843138, 0, 1,
6.464646, 4.020202, -128.101, 1, 0.5843138, 0, 1,
6.505051, 4.020202, -128.1404, 1, 0.5843138, 0, 1,
6.545455, 4.020202, -128.1848, 1, 0.5843138, 0, 1,
6.585859, 4.020202, -128.2342, 1, 0.5843138, 0, 1,
6.626263, 4.020202, -128.2887, 1, 0.5843138, 0, 1,
6.666667, 4.020202, -128.3483, 1, 0.5843138, 0, 1,
6.707071, 4.020202, -128.4129, 1, 0.5843138, 0, 1,
6.747475, 4.020202, -128.4825, 1, 0.5843138, 0, 1,
6.787879, 4.020202, -128.5572, 1, 0.5843138, 0, 1,
6.828283, 4.020202, -128.637, 1, 0.5843138, 0, 1,
6.868687, 4.020202, -128.7218, 1, 0.5843138, 0, 1,
6.909091, 4.020202, -128.8116, 1, 0.5843138, 0, 1,
6.949495, 4.020202, -128.9065, 1, 0.5843138, 0, 1,
6.989899, 4.020202, -129.0065, 1, 0.5843138, 0, 1,
7.030303, 4.020202, -129.1115, 1, 0.5843138, 0, 1,
7.070707, 4.020202, -129.2215, 1, 0.5843138, 0, 1,
7.111111, 4.020202, -129.3366, 1, 0.4823529, 0, 1,
7.151515, 4.020202, -129.4568, 1, 0.4823529, 0, 1,
7.191919, 4.020202, -129.582, 1, 0.4823529, 0, 1,
7.232323, 4.020202, -129.7122, 1, 0.4823529, 0, 1,
7.272727, 4.020202, -129.8475, 1, 0.4823529, 0, 1,
7.313131, 4.020202, -129.9879, 1, 0.4823529, 0, 1,
7.353535, 4.020202, -130.1333, 1, 0.4823529, 0, 1,
7.393939, 4.020202, -130.2838, 1, 0.4823529, 0, 1,
7.434343, 4.020202, -130.4393, 1, 0.4823529, 0, 1,
7.474748, 4.020202, -130.5998, 1, 0.3764706, 0, 1,
7.515152, 4.020202, -130.7654, 1, 0.3764706, 0, 1,
7.555555, 4.020202, -130.9361, 1, 0.3764706, 0, 1,
7.59596, 4.020202, -131.1118, 1, 0.3764706, 0, 1,
7.636364, 4.020202, -131.2926, 1, 0.3764706, 0, 1,
7.676768, 4.020202, -131.4784, 1, 0.3764706, 0, 1,
7.717172, 4.020202, -131.6692, 1, 0.3764706, 0, 1,
7.757576, 4.020202, -131.8651, 1, 0.2745098, 0, 1,
7.79798, 4.020202, -132.0661, 1, 0.2745098, 0, 1,
7.838384, 4.020202, -132.2721, 1, 0.2745098, 0, 1,
7.878788, 4.020202, -132.4832, 1, 0.2745098, 0, 1,
7.919192, 4.020202, -132.6993, 1, 0.2745098, 0, 1,
7.959596, 4.020202, -132.9205, 1, 0.2745098, 0, 1,
8, 4.020202, -133.1467, 1, 0.1686275, 0, 1,
4, 4.060606, -135.3602, 1, 0.06666667, 0, 1,
4.040404, 4.060606, -135.0968, 1, 0.06666667, 0, 1,
4.080808, 4.060606, -134.8384, 1, 0.06666667, 0, 1,
4.121212, 4.060606, -134.5849, 1, 0.06666667, 0, 1,
4.161616, 4.060606, -134.3363, 1, 0.06666667, 0, 1,
4.20202, 4.060606, -134.0927, 1, 0.1686275, 0, 1,
4.242424, 4.060606, -133.8541, 1, 0.1686275, 0, 1,
4.282828, 4.060606, -133.6204, 1, 0.1686275, 0, 1,
4.323232, 4.060606, -133.3916, 1, 0.1686275, 0, 1,
4.363636, 4.060606, -133.1678, 1, 0.1686275, 0, 1,
4.40404, 4.060606, -132.949, 1, 0.2745098, 0, 1,
4.444445, 4.060606, -132.7351, 1, 0.2745098, 0, 1,
4.484848, 4.060606, -132.5261, 1, 0.2745098, 0, 1,
4.525252, 4.060606, -132.3221, 1, 0.2745098, 0, 1,
4.565657, 4.060606, -132.1231, 1, 0.2745098, 0, 1,
4.606061, 4.060606, -131.929, 1, 0.2745098, 0, 1,
4.646465, 4.060606, -131.7398, 1, 0.3764706, 0, 1,
4.686869, 4.060606, -131.5556, 1, 0.3764706, 0, 1,
4.727273, 4.060606, -131.3764, 1, 0.3764706, 0, 1,
4.767677, 4.060606, -131.2021, 1, 0.3764706, 0, 1,
4.808081, 4.060606, -131.0327, 1, 0.3764706, 0, 1,
4.848485, 4.060606, -130.8683, 1, 0.3764706, 0, 1,
4.888889, 4.060606, -130.7089, 1, 0.3764706, 0, 1,
4.929293, 4.060606, -130.5544, 1, 0.3764706, 0, 1,
4.969697, 4.060606, -130.4048, 1, 0.4823529, 0, 1,
5.010101, 4.060606, -130.2603, 1, 0.4823529, 0, 1,
5.050505, 4.060606, -130.1206, 1, 0.4823529, 0, 1,
5.090909, 4.060606, -129.9859, 1, 0.4823529, 0, 1,
5.131313, 4.060606, -129.8562, 1, 0.4823529, 0, 1,
5.171717, 4.060606, -129.7314, 1, 0.4823529, 0, 1,
5.212121, 4.060606, -129.6115, 1, 0.4823529, 0, 1,
5.252525, 4.060606, -129.4966, 1, 0.4823529, 0, 1,
5.292929, 4.060606, -129.3867, 1, 0.4823529, 0, 1,
5.333333, 4.060606, -129.2817, 1, 0.5843138, 0, 1,
5.373737, 4.060606, -129.1817, 1, 0.5843138, 0, 1,
5.414141, 4.060606, -129.0866, 1, 0.5843138, 0, 1,
5.454545, 4.060606, -128.9964, 1, 0.5843138, 0, 1,
5.494949, 4.060606, -128.9112, 1, 0.5843138, 0, 1,
5.535354, 4.060606, -128.831, 1, 0.5843138, 0, 1,
5.575758, 4.060606, -128.7557, 1, 0.5843138, 0, 1,
5.616162, 4.060606, -128.6854, 1, 0.5843138, 0, 1,
5.656566, 4.060606, -128.62, 1, 0.5843138, 0, 1,
5.69697, 4.060606, -128.5595, 1, 0.5843138, 0, 1,
5.737374, 4.060606, -128.5041, 1, 0.5843138, 0, 1,
5.777778, 4.060606, -128.4535, 1, 0.5843138, 0, 1,
5.818182, 4.060606, -128.4079, 1, 0.5843138, 0, 1,
5.858586, 4.060606, -128.3673, 1, 0.5843138, 0, 1,
5.89899, 4.060606, -128.3316, 1, 0.5843138, 0, 1,
5.939394, 4.060606, -128.3009, 1, 0.5843138, 0, 1,
5.979798, 4.060606, -128.2751, 1, 0.5843138, 0, 1,
6.020202, 4.060606, -128.2542, 1, 0.5843138, 0, 1,
6.060606, 4.060606, -128.2384, 1, 0.5843138, 0, 1,
6.10101, 4.060606, -128.2274, 1, 0.5843138, 0, 1,
6.141414, 4.060606, -128.2214, 1, 0.5843138, 0, 1,
6.181818, 4.060606, -128.2204, 1, 0.5843138, 0, 1,
6.222222, 4.060606, -128.2243, 1, 0.5843138, 0, 1,
6.262626, 4.060606, -128.2332, 1, 0.5843138, 0, 1,
6.30303, 4.060606, -128.247, 1, 0.5843138, 0, 1,
6.343434, 4.060606, -128.2658, 1, 0.5843138, 0, 1,
6.383838, 4.060606, -128.2895, 1, 0.5843138, 0, 1,
6.424242, 4.060606, -128.3182, 1, 0.5843138, 0, 1,
6.464646, 4.060606, -128.3518, 1, 0.5843138, 0, 1,
6.505051, 4.060606, -128.3904, 1, 0.5843138, 0, 1,
6.545455, 4.060606, -128.4339, 1, 0.5843138, 0, 1,
6.585859, 4.060606, -128.4823, 1, 0.5843138, 0, 1,
6.626263, 4.060606, -128.5358, 1, 0.5843138, 0, 1,
6.666667, 4.060606, -128.5941, 1, 0.5843138, 0, 1,
6.707071, 4.060606, -128.6574, 1, 0.5843138, 0, 1,
6.747475, 4.060606, -128.7257, 1, 0.5843138, 0, 1,
6.787879, 4.060606, -128.7989, 1, 0.5843138, 0, 1,
6.828283, 4.060606, -128.8771, 1, 0.5843138, 0, 1,
6.868687, 4.060606, -128.9602, 1, 0.5843138, 0, 1,
6.909091, 4.060606, -129.0483, 1, 0.5843138, 0, 1,
6.949495, 4.060606, -129.1413, 1, 0.5843138, 0, 1,
6.989899, 4.060606, -129.2393, 1, 0.5843138, 0, 1,
7.030303, 4.060606, -129.3422, 1, 0.4823529, 0, 1,
7.070707, 4.060606, -129.4501, 1, 0.4823529, 0, 1,
7.111111, 4.060606, -129.5629, 1, 0.4823529, 0, 1,
7.151515, 4.060606, -129.6807, 1, 0.4823529, 0, 1,
7.191919, 4.060606, -129.8034, 1, 0.4823529, 0, 1,
7.232323, 4.060606, -129.9311, 1, 0.4823529, 0, 1,
7.272727, 4.060606, -130.0637, 1, 0.4823529, 0, 1,
7.313131, 4.060606, -130.2013, 1, 0.4823529, 0, 1,
7.353535, 4.060606, -130.3438, 1, 0.4823529, 0, 1,
7.393939, 4.060606, -130.4913, 1, 0.4823529, 0, 1,
7.434343, 4.060606, -130.6437, 1, 0.3764706, 0, 1,
7.474748, 4.060606, -130.8011, 1, 0.3764706, 0, 1,
7.515152, 4.060606, -130.9634, 1, 0.3764706, 0, 1,
7.555555, 4.060606, -131.1307, 1, 0.3764706, 0, 1,
7.59596, 4.060606, -131.3029, 1, 0.3764706, 0, 1,
7.636364, 4.060606, -131.4801, 1, 0.3764706, 0, 1,
7.676768, 4.060606, -131.6622, 1, 0.3764706, 0, 1,
7.717172, 4.060606, -131.8493, 1, 0.2745098, 0, 1,
7.757576, 4.060606, -132.0414, 1, 0.2745098, 0, 1,
7.79798, 4.060606, -132.2383, 1, 0.2745098, 0, 1,
7.838384, 4.060606, -132.4403, 1, 0.2745098, 0, 1,
7.878788, 4.060606, -132.6472, 1, 0.2745098, 0, 1,
7.919192, 4.060606, -132.859, 1, 0.2745098, 0, 1,
7.959596, 4.060606, -133.0758, 1, 0.1686275, 0, 1,
8, 4.060606, -133.2975, 1, 0.1686275, 0, 1,
4, 4.10101, -135.476, 1, 0.06666667, 0, 1,
4.040404, 4.10101, -135.2177, 1, 0.06666667, 0, 1,
4.080808, 4.10101, -134.9643, 1, 0.06666667, 0, 1,
4.121212, 4.10101, -134.7158, 1, 0.06666667, 0, 1,
4.161616, 4.10101, -134.4721, 1, 0.06666667, 0, 1,
4.20202, 4.10101, -134.2333, 1, 0.1686275, 0, 1,
4.242424, 4.10101, -133.9993, 1, 0.1686275, 0, 1,
4.282828, 4.10101, -133.7702, 1, 0.1686275, 0, 1,
4.323232, 4.10101, -133.5459, 1, 0.1686275, 0, 1,
4.363636, 4.10101, -133.3265, 1, 0.1686275, 0, 1,
4.40404, 4.10101, -133.112, 1, 0.1686275, 0, 1,
4.444445, 4.10101, -132.9023, 1, 0.2745098, 0, 1,
4.484848, 4.10101, -132.6974, 1, 0.2745098, 0, 1,
4.525252, 4.10101, -132.4974, 1, 0.2745098, 0, 1,
4.565657, 4.10101, -132.3023, 1, 0.2745098, 0, 1,
4.606061, 4.10101, -132.112, 1, 0.2745098, 0, 1,
4.646465, 4.10101, -131.9265, 1, 0.2745098, 0, 1,
4.686869, 4.10101, -131.7459, 1, 0.3764706, 0, 1,
4.727273, 4.10101, -131.5702, 1, 0.3764706, 0, 1,
4.767677, 4.10101, -131.3993, 1, 0.3764706, 0, 1,
4.808081, 4.10101, -131.2333, 1, 0.3764706, 0, 1,
4.848485, 4.10101, -131.0721, 1, 0.3764706, 0, 1,
4.888889, 4.10101, -130.9158, 1, 0.3764706, 0, 1,
4.929293, 4.10101, -130.7643, 1, 0.3764706, 0, 1,
4.969697, 4.10101, -130.6177, 1, 0.3764706, 0, 1,
5.010101, 4.10101, -130.476, 1, 0.4823529, 0, 1,
5.050505, 4.10101, -130.3391, 1, 0.4823529, 0, 1,
5.090909, 4.10101, -130.207, 1, 0.4823529, 0, 1,
5.131313, 4.10101, -130.0798, 1, 0.4823529, 0, 1,
5.171717, 4.10101, -129.9575, 1, 0.4823529, 0, 1,
5.212121, 4.10101, -129.84, 1, 0.4823529, 0, 1,
5.252525, 4.10101, -129.7273, 1, 0.4823529, 0, 1,
5.292929, 4.10101, -129.6195, 1, 0.4823529, 0, 1,
5.333333, 4.10101, -129.5166, 1, 0.4823529, 0, 1,
5.373737, 4.10101, -129.4185, 1, 0.4823529, 0, 1,
5.414141, 4.10101, -129.3253, 1, 0.4823529, 0, 1,
5.454545, 4.10101, -129.2369, 1, 0.5843138, 0, 1,
5.494949, 4.10101, -129.1534, 1, 0.5843138, 0, 1,
5.535354, 4.10101, -129.0747, 1, 0.5843138, 0, 1,
5.575758, 4.10101, -129.0009, 1, 0.5843138, 0, 1,
5.616162, 4.10101, -128.932, 1, 0.5843138, 0, 1,
5.656566, 4.10101, -128.8679, 1, 0.5843138, 0, 1,
5.69697, 4.10101, -128.8086, 1, 0.5843138, 0, 1,
5.737374, 4.10101, -128.7542, 1, 0.5843138, 0, 1,
5.777778, 4.10101, -128.7047, 1, 0.5843138, 0, 1,
5.818182, 4.10101, -128.66, 1, 0.5843138, 0, 1,
5.858586, 4.10101, -128.6201, 1, 0.5843138, 0, 1,
5.89899, 4.10101, -128.5851, 1, 0.5843138, 0, 1,
5.939394, 4.10101, -128.555, 1, 0.5843138, 0, 1,
5.979798, 4.10101, -128.5297, 1, 0.5843138, 0, 1,
6.020202, 4.10101, -128.5093, 1, 0.5843138, 0, 1,
6.060606, 4.10101, -128.4937, 1, 0.5843138, 0, 1,
6.10101, 4.10101, -128.483, 1, 0.5843138, 0, 1,
6.141414, 4.10101, -128.4771, 1, 0.5843138, 0, 1,
6.181818, 4.10101, -128.4761, 1, 0.5843138, 0, 1,
6.222222, 4.10101, -128.4799, 1, 0.5843138, 0, 1,
6.262626, 4.10101, -128.4886, 1, 0.5843138, 0, 1,
6.30303, 4.10101, -128.5022, 1, 0.5843138, 0, 1,
6.343434, 4.10101, -128.5206, 1, 0.5843138, 0, 1,
6.383838, 4.10101, -128.5439, 1, 0.5843138, 0, 1,
6.424242, 4.10101, -128.5719, 1, 0.5843138, 0, 1,
6.464646, 4.10101, -128.6049, 1, 0.5843138, 0, 1,
6.505051, 4.10101, -128.6427, 1, 0.5843138, 0, 1,
6.545455, 4.10101, -128.6854, 1, 0.5843138, 0, 1,
6.585859, 4.10101, -128.7329, 1, 0.5843138, 0, 1,
6.626263, 4.10101, -128.7853, 1, 0.5843138, 0, 1,
6.666667, 4.10101, -128.8425, 1, 0.5843138, 0, 1,
6.707071, 4.10101, -128.9046, 1, 0.5843138, 0, 1,
6.747475, 4.10101, -128.9715, 1, 0.5843138, 0, 1,
6.787879, 4.10101, -129.0433, 1, 0.5843138, 0, 1,
6.828283, 4.10101, -129.1199, 1, 0.5843138, 0, 1,
6.868687, 4.10101, -129.2014, 1, 0.5843138, 0, 1,
6.909091, 4.10101, -129.2878, 1, 0.5843138, 0, 1,
6.949495, 4.10101, -129.379, 1, 0.4823529, 0, 1,
6.989899, 4.10101, -129.475, 1, 0.4823529, 0, 1,
7.030303, 4.10101, -129.5759, 1, 0.4823529, 0, 1,
7.070707, 4.10101, -129.6817, 1, 0.4823529, 0, 1,
7.111111, 4.10101, -129.7923, 1, 0.4823529, 0, 1,
7.151515, 4.10101, -129.9078, 1, 0.4823529, 0, 1,
7.191919, 4.10101, -130.0281, 1, 0.4823529, 0, 1,
7.232323, 4.10101, -130.1533, 1, 0.4823529, 0, 1,
7.272727, 4.10101, -130.2833, 1, 0.4823529, 0, 1,
7.313131, 4.10101, -130.4182, 1, 0.4823529, 0, 1,
7.353535, 4.10101, -130.5579, 1, 0.3764706, 0, 1,
7.393939, 4.10101, -130.7025, 1, 0.3764706, 0, 1,
7.434343, 4.10101, -130.8519, 1, 0.3764706, 0, 1,
7.474748, 4.10101, -131.0062, 1, 0.3764706, 0, 1,
7.515152, 4.10101, -131.1654, 1, 0.3764706, 0, 1,
7.555555, 4.10101, -131.3294, 1, 0.3764706, 0, 1,
7.59596, 4.10101, -131.4982, 1, 0.3764706, 0, 1,
7.636364, 4.10101, -131.6719, 1, 0.3764706, 0, 1,
7.676768, 4.10101, -131.8505, 1, 0.2745098, 0, 1,
7.717172, 4.10101, -132.0339, 1, 0.2745098, 0, 1,
7.757576, 4.10101, -132.2222, 1, 0.2745098, 0, 1,
7.79798, 4.10101, -132.4153, 1, 0.2745098, 0, 1,
7.838384, 4.10101, -132.6133, 1, 0.2745098, 0, 1,
7.878788, 4.10101, -132.8161, 1, 0.2745098, 0, 1,
7.919192, 4.10101, -133.0237, 1, 0.2745098, 0, 1,
7.959596, 4.10101, -133.2363, 1, 0.1686275, 0, 1,
8, 4.10101, -133.4537, 1, 0.1686275, 0, 1,
4, 4.141414, -135.5979, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.141414, -135.3446, 1, 0.06666667, 0, 1,
4.080808, 4.141414, -135.0962, 1, 0.06666667, 0, 1,
4.121212, 4.141414, -134.8525, 1, 0.06666667, 0, 1,
4.161616, 4.141414, -134.6135, 1, 0.06666667, 0, 1,
4.20202, 4.141414, -134.3793, 1, 0.06666667, 0, 1,
4.242424, 4.141414, -134.1499, 1, 0.1686275, 0, 1,
4.282828, 4.141414, -133.9252, 1, 0.1686275, 0, 1,
4.323232, 4.141414, -133.7053, 1, 0.1686275, 0, 1,
4.363636, 4.141414, -133.4902, 1, 0.1686275, 0, 1,
4.40404, 4.141414, -133.2798, 1, 0.1686275, 0, 1,
4.444445, 4.141414, -133.0741, 1, 0.1686275, 0, 1,
4.484848, 4.141414, -132.8733, 1, 0.2745098, 0, 1,
4.525252, 4.141414, -132.6771, 1, 0.2745098, 0, 1,
4.565657, 4.141414, -132.4858, 1, 0.2745098, 0, 1,
4.606061, 4.141414, -132.2992, 1, 0.2745098, 0, 1,
4.646465, 4.141414, -132.1174, 1, 0.2745098, 0, 1,
4.686869, 4.141414, -131.9403, 1, 0.2745098, 0, 1,
4.727273, 4.141414, -131.7679, 1, 0.3764706, 0, 1,
4.767677, 4.141414, -131.6004, 1, 0.3764706, 0, 1,
4.808081, 4.141414, -131.4376, 1, 0.3764706, 0, 1,
4.848485, 4.141414, -131.2795, 1, 0.3764706, 0, 1,
4.888889, 4.141414, -131.1263, 1, 0.3764706, 0, 1,
4.929293, 4.141414, -130.9777, 1, 0.3764706, 0, 1,
4.969697, 4.141414, -130.834, 1, 0.3764706, 0, 1,
5.010101, 4.141414, -130.695, 1, 0.3764706, 0, 1,
5.050505, 4.141414, -130.5607, 1, 0.3764706, 0, 1,
5.090909, 4.141414, -130.4312, 1, 0.4823529, 0, 1,
5.131313, 4.141414, -130.3065, 1, 0.4823529, 0, 1,
5.171717, 4.141414, -130.1865, 1, 0.4823529, 0, 1,
5.212121, 4.141414, -130.0713, 1, 0.4823529, 0, 1,
5.252525, 4.141414, -129.9609, 1, 0.4823529, 0, 1,
5.292929, 4.141414, -129.8552, 1, 0.4823529, 0, 1,
5.333333, 4.141414, -129.7542, 1, 0.4823529, 0, 1,
5.373737, 4.141414, -129.6581, 1, 0.4823529, 0, 1,
5.414141, 4.141414, -129.5666, 1, 0.4823529, 0, 1,
5.454545, 4.141414, -129.48, 1, 0.4823529, 0, 1,
5.494949, 4.141414, -129.3981, 1, 0.4823529, 0, 1,
5.535354, 4.141414, -129.3209, 1, 0.4823529, 0, 1,
5.575758, 4.141414, -129.2486, 1, 0.5843138, 0, 1,
5.616162, 4.141414, -129.1809, 1, 0.5843138, 0, 1,
5.656566, 4.141414, -129.1181, 1, 0.5843138, 0, 1,
5.69697, 4.141414, -129.06, 1, 0.5843138, 0, 1,
5.737374, 4.141414, -129.0066, 1, 0.5843138, 0, 1,
5.777778, 4.141414, -128.958, 1, 0.5843138, 0, 1,
5.818182, 4.141414, -128.9142, 1, 0.5843138, 0, 1,
5.858586, 4.141414, -128.8752, 1, 0.5843138, 0, 1,
5.89899, 4.141414, -128.8409, 1, 0.5843138, 0, 1,
5.939394, 4.141414, -128.8113, 1, 0.5843138, 0, 1,
5.979798, 4.141414, -128.7865, 1, 0.5843138, 0, 1,
6.020202, 4.141414, -128.7665, 1, 0.5843138, 0, 1,
6.060606, 4.141414, -128.7512, 1, 0.5843138, 0, 1,
6.10101, 4.141414, -128.7407, 1, 0.5843138, 0, 1,
6.141414, 4.141414, -128.7349, 1, 0.5843138, 0, 1,
6.181818, 4.141414, -128.7339, 1, 0.5843138, 0, 1,
6.222222, 4.141414, -128.7377, 1, 0.5843138, 0, 1,
6.262626, 4.141414, -128.7462, 1, 0.5843138, 0, 1,
6.30303, 4.141414, -128.7595, 1, 0.5843138, 0, 1,
6.343434, 4.141414, -128.7776, 1, 0.5843138, 0, 1,
6.383838, 4.141414, -128.8004, 1, 0.5843138, 0, 1,
6.424242, 4.141414, -128.8279, 1, 0.5843138, 0, 1,
6.464646, 4.141414, -128.8602, 1, 0.5843138, 0, 1,
6.505051, 4.141414, -128.8973, 1, 0.5843138, 0, 1,
6.545455, 4.141414, -128.9391, 1, 0.5843138, 0, 1,
6.585859, 4.141414, -128.9857, 1, 0.5843138, 0, 1,
6.626263, 4.141414, -129.0371, 1, 0.5843138, 0, 1,
6.666667, 4.141414, -129.0932, 1, 0.5843138, 0, 1,
6.707071, 4.141414, -129.1541, 1, 0.5843138, 0, 1,
6.747475, 4.141414, -129.2197, 1, 0.5843138, 0, 1,
6.787879, 4.141414, -129.2901, 1, 0.5843138, 0, 1,
6.828283, 4.141414, -129.3653, 1, 0.4823529, 0, 1,
6.868687, 4.141414, -129.4452, 1, 0.4823529, 0, 1,
6.909091, 4.141414, -129.5298, 1, 0.4823529, 0, 1,
6.949495, 4.141414, -129.6193, 1, 0.4823529, 0, 1,
6.989899, 4.141414, -129.7135, 1, 0.4823529, 0, 1,
7.030303, 4.141414, -129.8124, 1, 0.4823529, 0, 1,
7.070707, 4.141414, -129.9161, 1, 0.4823529, 0, 1,
7.111111, 4.141414, -130.0246, 1, 0.4823529, 0, 1,
7.151515, 4.141414, -130.1378, 1, 0.4823529, 0, 1,
7.191919, 4.141414, -130.2558, 1, 0.4823529, 0, 1,
7.232323, 4.141414, -130.3785, 1, 0.4823529, 0, 1,
7.272727, 4.141414, -130.506, 1, 0.4823529, 0, 1,
7.313131, 4.141414, -130.6383, 1, 0.3764706, 0, 1,
7.353535, 4.141414, -130.7753, 1, 0.3764706, 0, 1,
7.393939, 4.141414, -130.9171, 1, 0.3764706, 0, 1,
7.434343, 4.141414, -131.0636, 1, 0.3764706, 0, 1,
7.474748, 4.141414, -131.2149, 1, 0.3764706, 0, 1,
7.515152, 4.141414, -131.371, 1, 0.3764706, 0, 1,
7.555555, 4.141414, -131.5318, 1, 0.3764706, 0, 1,
7.59596, 4.141414, -131.6974, 1, 0.3764706, 0, 1,
7.636364, 4.141414, -131.8677, 1, 0.2745098, 0, 1,
7.676768, 4.141414, -132.0428, 1, 0.2745098, 0, 1,
7.717172, 4.141414, -132.2226, 1, 0.2745098, 0, 1,
7.757576, 4.141414, -132.4072, 1, 0.2745098, 0, 1,
7.79798, 4.141414, -132.5966, 1, 0.2745098, 0, 1,
7.838384, 4.141414, -132.7907, 1, 0.2745098, 0, 1,
7.878788, 4.141414, -132.9896, 1, 0.2745098, 0, 1,
7.919192, 4.141414, -133.1933, 1, 0.1686275, 0, 1,
7.959596, 4.141414, -133.4017, 1, 0.1686275, 0, 1,
8, 4.141414, -133.6148, 1, 0.1686275, 0, 1,
4, 4.181818, -135.7256, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.181818, -135.4773, 1, 0.06666667, 0, 1,
4.080808, 4.181818, -135.2336, 1, 0.06666667, 0, 1,
4.121212, 4.181818, -134.9946, 1, 0.06666667, 0, 1,
4.161616, 4.181818, -134.7602, 1, 0.06666667, 0, 1,
4.20202, 4.181818, -134.5305, 1, 0.06666667, 0, 1,
4.242424, 4.181818, -134.3055, 1, 0.06666667, 0, 1,
4.282828, 4.181818, -134.0852, 1, 0.1686275, 0, 1,
4.323232, 4.181818, -133.8695, 1, 0.1686275, 0, 1,
4.363636, 4.181818, -133.6585, 1, 0.1686275, 0, 1,
4.40404, 4.181818, -133.4521, 1, 0.1686275, 0, 1,
4.444445, 4.181818, -133.2504, 1, 0.1686275, 0, 1,
4.484848, 4.181818, -133.0534, 1, 0.1686275, 0, 1,
4.525252, 4.181818, -132.8611, 1, 0.2745098, 0, 1,
4.565657, 4.181818, -132.6734, 1, 0.2745098, 0, 1,
4.606061, 4.181818, -132.4904, 1, 0.2745098, 0, 1,
4.646465, 4.181818, -132.3121, 1, 0.2745098, 0, 1,
4.686869, 4.181818, -132.1384, 1, 0.2745098, 0, 1,
4.727273, 4.181818, -131.9694, 1, 0.2745098, 0, 1,
4.767677, 4.181818, -131.805, 1, 0.2745098, 0, 1,
4.808081, 4.181818, -131.6454, 1, 0.3764706, 0, 1,
4.848485, 4.181818, -131.4904, 1, 0.3764706, 0, 1,
4.888889, 4.181818, -131.34, 1, 0.3764706, 0, 1,
4.929293, 4.181818, -131.1944, 1, 0.3764706, 0, 1,
4.969697, 4.181818, -131.0534, 1, 0.3764706, 0, 1,
5.010101, 4.181818, -130.917, 1, 0.3764706, 0, 1,
5.050505, 4.181818, -130.7854, 1, 0.3764706, 0, 1,
5.090909, 4.181818, -130.6584, 1, 0.3764706, 0, 1,
5.131313, 4.181818, -130.536, 1, 0.4823529, 0, 1,
5.171717, 4.181818, -130.4184, 1, 0.4823529, 0, 1,
5.212121, 4.181818, -130.3054, 1, 0.4823529, 0, 1,
5.252525, 4.181818, -130.197, 1, 0.4823529, 0, 1,
5.292929, 4.181818, -130.0934, 1, 0.4823529, 0, 1,
5.333333, 4.181818, -129.9944, 1, 0.4823529, 0, 1,
5.373737, 4.181818, -129.9001, 1, 0.4823529, 0, 1,
5.414141, 4.181818, -129.8104, 1, 0.4823529, 0, 1,
5.454545, 4.181818, -129.7254, 1, 0.4823529, 0, 1,
5.494949, 4.181818, -129.6451, 1, 0.4823529, 0, 1,
5.535354, 4.181818, -129.5694, 1, 0.4823529, 0, 1,
5.575758, 4.181818, -129.4984, 1, 0.4823529, 0, 1,
5.616162, 4.181818, -129.4321, 1, 0.4823529, 0, 1,
5.656566, 4.181818, -129.3705, 1, 0.4823529, 0, 1,
5.69697, 4.181818, -129.3135, 1, 0.4823529, 0, 1,
5.737374, 4.181818, -129.2612, 1, 0.5843138, 0, 1,
5.777778, 4.181818, -129.2135, 1, 0.5843138, 0, 1,
5.818182, 4.181818, -129.1705, 1, 0.5843138, 0, 1,
5.858586, 4.181818, -129.1322, 1, 0.5843138, 0, 1,
5.89899, 4.181818, -129.0986, 1, 0.5843138, 0, 1,
5.939394, 4.181818, -129.0696, 1, 0.5843138, 0, 1,
5.979798, 4.181818, -129.0453, 1, 0.5843138, 0, 1,
6.020202, 4.181818, -129.0256, 1, 0.5843138, 0, 1,
6.060606, 4.181818, -129.0106, 1, 0.5843138, 0, 1,
6.10101, 4.181818, -129.0003, 1, 0.5843138, 0, 1,
6.141414, 4.181818, -128.9947, 1, 0.5843138, 0, 1,
6.181818, 4.181818, -128.9937, 1, 0.5843138, 0, 1,
6.222222, 4.181818, -128.9974, 1, 0.5843138, 0, 1,
6.262626, 4.181818, -129.0058, 1, 0.5843138, 0, 1,
6.30303, 4.181818, -129.0188, 1, 0.5843138, 0, 1,
6.343434, 4.181818, -129.0365, 1, 0.5843138, 0, 1,
6.383838, 4.181818, -129.0589, 1, 0.5843138, 0, 1,
6.424242, 4.181818, -129.0859, 1, 0.5843138, 0, 1,
6.464646, 4.181818, -129.1176, 1, 0.5843138, 0, 1,
6.505051, 4.181818, -129.1539, 1, 0.5843138, 0, 1,
6.545455, 4.181818, -129.195, 1, 0.5843138, 0, 1,
6.585859, 4.181818, -129.2407, 1, 0.5843138, 0, 1,
6.626263, 4.181818, -129.291, 1, 0.5843138, 0, 1,
6.666667, 4.181818, -129.3461, 1, 0.4823529, 0, 1,
6.707071, 4.181818, -129.4058, 1, 0.4823529, 0, 1,
6.747475, 4.181818, -129.4702, 1, 0.4823529, 0, 1,
6.787879, 4.181818, -129.5392, 1, 0.4823529, 0, 1,
6.828283, 4.181818, -129.6129, 1, 0.4823529, 0, 1,
6.868687, 4.181818, -129.6913, 1, 0.4823529, 0, 1,
6.909091, 4.181818, -129.7743, 1, 0.4823529, 0, 1,
6.949495, 4.181818, -129.862, 1, 0.4823529, 0, 1,
6.989899, 4.181818, -129.9544, 1, 0.4823529, 0, 1,
7.030303, 4.181818, -130.0514, 1, 0.4823529, 0, 1,
7.070707, 4.181818, -130.1532, 1, 0.4823529, 0, 1,
7.111111, 4.181818, -130.2595, 1, 0.4823529, 0, 1,
7.151515, 4.181818, -130.3706, 1, 0.4823529, 0, 1,
7.191919, 4.181818, -130.4863, 1, 0.4823529, 0, 1,
7.232323, 4.181818, -130.6067, 1, 0.3764706, 0, 1,
7.272727, 4.181818, -130.7317, 1, 0.3764706, 0, 1,
7.313131, 4.181818, -130.8614, 1, 0.3764706, 0, 1,
7.353535, 4.181818, -130.9958, 1, 0.3764706, 0, 1,
7.393939, 4.181818, -131.1349, 1, 0.3764706, 0, 1,
7.434343, 4.181818, -131.2786, 1, 0.3764706, 0, 1,
7.474748, 4.181818, -131.427, 1, 0.3764706, 0, 1,
7.515152, 4.181818, -131.58, 1, 0.3764706, 0, 1,
7.555555, 4.181818, -131.7377, 1, 0.3764706, 0, 1,
7.59596, 4.181818, -131.9001, 1, 0.2745098, 0, 1,
7.636364, 4.181818, -132.0672, 1, 0.2745098, 0, 1,
7.676768, 4.181818, -132.2389, 1, 0.2745098, 0, 1,
7.717172, 4.181818, -132.4153, 1, 0.2745098, 0, 1,
7.757576, 4.181818, -132.5964, 1, 0.2745098, 0, 1,
7.79798, 4.181818, -132.7821, 1, 0.2745098, 0, 1,
7.838384, 4.181818, -132.9725, 1, 0.2745098, 0, 1,
7.878788, 4.181818, -133.1676, 1, 0.1686275, 0, 1,
7.919192, 4.181818, -133.3673, 1, 0.1686275, 0, 1,
7.959596, 4.181818, -133.5717, 1, 0.1686275, 0, 1,
8, 4.181818, -133.7807, 1, 0.1686275, 0, 1,
4, 4.222222, -135.859, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.222222, -135.6153, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.222222, -135.3763, 1, 0.06666667, 0, 1,
4.121212, 4.222222, -135.1418, 1, 0.06666667, 0, 1,
4.161616, 4.222222, -134.9119, 1, 0.06666667, 0, 1,
4.20202, 4.222222, -134.6866, 1, 0.06666667, 0, 1,
4.242424, 4.222222, -134.4659, 1, 0.06666667, 0, 1,
4.282828, 4.222222, -134.2497, 1, 0.1686275, 0, 1,
4.323232, 4.222222, -134.0382, 1, 0.1686275, 0, 1,
4.363636, 4.222222, -133.8312, 1, 0.1686275, 0, 1,
4.40404, 4.222222, -133.6288, 1, 0.1686275, 0, 1,
4.444445, 4.222222, -133.4309, 1, 0.1686275, 0, 1,
4.484848, 4.222222, -133.2377, 1, 0.1686275, 0, 1,
4.525252, 4.222222, -133.049, 1, 0.1686275, 0, 1,
4.565657, 4.222222, -132.8649, 1, 0.2745098, 0, 1,
4.606061, 4.222222, -132.6853, 1, 0.2745098, 0, 1,
4.646465, 4.222222, -132.5104, 1, 0.2745098, 0, 1,
4.686869, 4.222222, -132.34, 1, 0.2745098, 0, 1,
4.727273, 4.222222, -132.1743, 1, 0.2745098, 0, 1,
4.767677, 4.222222, -132.013, 1, 0.2745098, 0, 1,
4.808081, 4.222222, -131.8564, 1, 0.2745098, 0, 1,
4.848485, 4.222222, -131.7044, 1, 0.3764706, 0, 1,
4.888889, 4.222222, -131.5569, 1, 0.3764706, 0, 1,
4.929293, 4.222222, -131.414, 1, 0.3764706, 0, 1,
4.969697, 4.222222, -131.2757, 1, 0.3764706, 0, 1,
5.010101, 4.222222, -131.1419, 1, 0.3764706, 0, 1,
5.050505, 4.222222, -131.0128, 1, 0.3764706, 0, 1,
5.090909, 4.222222, -130.8882, 1, 0.3764706, 0, 1,
5.131313, 4.222222, -130.7682, 1, 0.3764706, 0, 1,
5.171717, 4.222222, -130.6528, 1, 0.3764706, 0, 1,
5.212121, 4.222222, -130.5419, 1, 0.4823529, 0, 1,
5.252525, 4.222222, -130.4357, 1, 0.4823529, 0, 1,
5.292929, 4.222222, -130.334, 1, 0.4823529, 0, 1,
5.333333, 4.222222, -130.2369, 1, 0.4823529, 0, 1,
5.373737, 4.222222, -130.1443, 1, 0.4823529, 0, 1,
5.414141, 4.222222, -130.0564, 1, 0.4823529, 0, 1,
5.454545, 4.222222, -129.973, 1, 0.4823529, 0, 1,
5.494949, 4.222222, -129.8942, 1, 0.4823529, 0, 1,
5.535354, 4.222222, -129.82, 1, 0.4823529, 0, 1,
5.575758, 4.222222, -129.7504, 1, 0.4823529, 0, 1,
5.616162, 4.222222, -129.6853, 1, 0.4823529, 0, 1,
5.656566, 4.222222, -129.6248, 1, 0.4823529, 0, 1,
5.69697, 4.222222, -129.5689, 1, 0.4823529, 0, 1,
5.737374, 4.222222, -129.5176, 1, 0.4823529, 0, 1,
5.777778, 4.222222, -129.4709, 1, 0.4823529, 0, 1,
5.818182, 4.222222, -129.4287, 1, 0.4823529, 0, 1,
5.858586, 4.222222, -129.3911, 1, 0.4823529, 0, 1,
5.89899, 4.222222, -129.3581, 1, 0.4823529, 0, 1,
5.939394, 4.222222, -129.3297, 1, 0.4823529, 0, 1,
5.979798, 4.222222, -129.3058, 1, 0.4823529, 0, 1,
6.020202, 4.222222, -129.2866, 1, 0.5843138, 0, 1,
6.060606, 4.222222, -129.2719, 1, 0.5843138, 0, 1,
6.10101, 4.222222, -129.2617, 1, 0.5843138, 0, 1,
6.141414, 4.222222, -129.2562, 1, 0.5843138, 0, 1,
6.181818, 4.222222, -129.2553, 1, 0.5843138, 0, 1,
6.222222, 4.222222, -129.2589, 1, 0.5843138, 0, 1,
6.262626, 4.222222, -129.2671, 1, 0.5843138, 0, 1,
6.30303, 4.222222, -129.2799, 1, 0.5843138, 0, 1,
6.343434, 4.222222, -129.2972, 1, 0.5843138, 0, 1,
6.383838, 4.222222, -129.3192, 1, 0.4823529, 0, 1,
6.424242, 4.222222, -129.3457, 1, 0.4823529, 0, 1,
6.464646, 4.222222, -129.3768, 1, 0.4823529, 0, 1,
6.505051, 4.222222, -129.4124, 1, 0.4823529, 0, 1,
6.545455, 4.222222, -129.4527, 1, 0.4823529, 0, 1,
6.585859, 4.222222, -129.4975, 1, 0.4823529, 0, 1,
6.626263, 4.222222, -129.5469, 1, 0.4823529, 0, 1,
6.666667, 4.222222, -129.6009, 1, 0.4823529, 0, 1,
6.707071, 4.222222, -129.6595, 1, 0.4823529, 0, 1,
6.747475, 4.222222, -129.7226, 1, 0.4823529, 0, 1,
6.787879, 4.222222, -129.7904, 1, 0.4823529, 0, 1,
6.828283, 4.222222, -129.8627, 1, 0.4823529, 0, 1,
6.868687, 4.222222, -129.9395, 1, 0.4823529, 0, 1,
6.909091, 4.222222, -130.021, 1, 0.4823529, 0, 1,
6.949495, 4.222222, -130.107, 1, 0.4823529, 0, 1,
6.989899, 4.222222, -130.1976, 1, 0.4823529, 0, 1,
7.030303, 4.222222, -130.2928, 1, 0.4823529, 0, 1,
7.070707, 4.222222, -130.3926, 1, 0.4823529, 0, 1,
7.111111, 4.222222, -130.497, 1, 0.4823529, 0, 1,
7.151515, 4.222222, -130.6059, 1, 0.3764706, 0, 1,
7.191919, 4.222222, -130.7194, 1, 0.3764706, 0, 1,
7.232323, 4.222222, -130.8375, 1, 0.3764706, 0, 1,
7.272727, 4.222222, -130.9602, 1, 0.3764706, 0, 1,
7.313131, 4.222222, -131.0874, 1, 0.3764706, 0, 1,
7.353535, 4.222222, -131.2192, 1, 0.3764706, 0, 1,
7.393939, 4.222222, -131.3556, 1, 0.3764706, 0, 1,
7.434343, 4.222222, -131.4966, 1, 0.3764706, 0, 1,
7.474748, 4.222222, -131.6422, 1, 0.3764706, 0, 1,
7.515152, 4.222222, -131.7923, 1, 0.2745098, 0, 1,
7.555555, 4.222222, -131.947, 1, 0.2745098, 0, 1,
7.59596, 4.222222, -132.1063, 1, 0.2745098, 0, 1,
7.636364, 4.222222, -132.2702, 1, 0.2745098, 0, 1,
7.676768, 4.222222, -132.4387, 1, 0.2745098, 0, 1,
7.717172, 4.222222, -132.6117, 1, 0.2745098, 0, 1,
7.757576, 4.222222, -132.7893, 1, 0.2745098, 0, 1,
7.79798, 4.222222, -132.9715, 1, 0.2745098, 0, 1,
7.838384, 4.222222, -133.1583, 1, 0.1686275, 0, 1,
7.878788, 4.222222, -133.3496, 1, 0.1686275, 0, 1,
7.919192, 4.222222, -133.5455, 1, 0.1686275, 0, 1,
7.959596, 4.222222, -133.746, 1, 0.1686275, 0, 1,
8, 4.222222, -133.9511, 1, 0.1686275, 0, 1,
4, 4.262626, -135.9975, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.262626, -135.7585, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.262626, -135.524, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.262626, -135.2939, 1, 0.06666667, 0, 1,
4.161616, 4.262626, -135.0684, 1, 0.06666667, 0, 1,
4.20202, 4.262626, -134.8473, 1, 0.06666667, 0, 1,
4.242424, 4.262626, -134.6308, 1, 0.06666667, 0, 1,
4.282828, 4.262626, -134.4187, 1, 0.06666667, 0, 1,
4.323232, 4.262626, -134.2111, 1, 0.1686275, 0, 1,
4.363636, 4.262626, -134.008, 1, 0.1686275, 0, 1,
4.40404, 4.262626, -133.8094, 1, 0.1686275, 0, 1,
4.444445, 4.262626, -133.6153, 1, 0.1686275, 0, 1,
4.484848, 4.262626, -133.4257, 1, 0.1686275, 0, 1,
4.525252, 4.262626, -133.2406, 1, 0.1686275, 0, 1,
4.565657, 4.262626, -133.0599, 1, 0.1686275, 0, 1,
4.606061, 4.262626, -132.8838, 1, 0.2745098, 0, 1,
4.646465, 4.262626, -132.7122, 1, 0.2745098, 0, 1,
4.686869, 4.262626, -132.545, 1, 0.2745098, 0, 1,
4.727273, 4.262626, -132.3824, 1, 0.2745098, 0, 1,
4.767677, 4.262626, -132.2242, 1, 0.2745098, 0, 1,
4.808081, 4.262626, -132.0705, 1, 0.2745098, 0, 1,
4.848485, 4.262626, -131.9213, 1, 0.2745098, 0, 1,
4.888889, 4.262626, -131.7766, 1, 0.3764706, 0, 1,
4.929293, 4.262626, -131.6364, 1, 0.3764706, 0, 1,
4.969697, 4.262626, -131.5007, 1, 0.3764706, 0, 1,
5.010101, 4.262626, -131.3695, 1, 0.3764706, 0, 1,
5.050505, 4.262626, -131.2428, 1, 0.3764706, 0, 1,
5.090909, 4.262626, -131.1206, 1, 0.3764706, 0, 1,
5.131313, 4.262626, -131.0028, 1, 0.3764706, 0, 1,
5.171717, 4.262626, -130.8896, 1, 0.3764706, 0, 1,
5.212121, 4.262626, -130.7808, 1, 0.3764706, 0, 1,
5.252525, 4.262626, -130.6766, 1, 0.3764706, 0, 1,
5.292929, 4.262626, -130.5768, 1, 0.3764706, 0, 1,
5.333333, 4.262626, -130.4815, 1, 0.4823529, 0, 1,
5.373737, 4.262626, -130.3907, 1, 0.4823529, 0, 1,
5.414141, 4.262626, -130.3044, 1, 0.4823529, 0, 1,
5.454545, 4.262626, -130.2226, 1, 0.4823529, 0, 1,
5.494949, 4.262626, -130.1453, 1, 0.4823529, 0, 1,
5.535354, 4.262626, -130.0725, 1, 0.4823529, 0, 1,
5.575758, 4.262626, -130.0042, 1, 0.4823529, 0, 1,
5.616162, 4.262626, -129.9404, 1, 0.4823529, 0, 1,
5.656566, 4.262626, -129.881, 1, 0.4823529, 0, 1,
5.69697, 4.262626, -129.8262, 1, 0.4823529, 0, 1,
5.737374, 4.262626, -129.7758, 1, 0.4823529, 0, 1,
5.777778, 4.262626, -129.73, 1, 0.4823529, 0, 1,
5.818182, 4.262626, -129.6886, 1, 0.4823529, 0, 1,
5.858586, 4.262626, -129.6517, 1, 0.4823529, 0, 1,
5.89899, 4.262626, -129.6193, 1, 0.4823529, 0, 1,
5.939394, 4.262626, -129.5914, 1, 0.4823529, 0, 1,
5.979798, 4.262626, -129.5681, 1, 0.4823529, 0, 1,
6.020202, 4.262626, -129.5491, 1, 0.4823529, 0, 1,
6.060606, 4.262626, -129.5347, 1, 0.4823529, 0, 1,
6.10101, 4.262626, -129.5248, 1, 0.4823529, 0, 1,
6.141414, 4.262626, -129.5194, 1, 0.4823529, 0, 1,
6.181818, 4.262626, -129.5184, 1, 0.4823529, 0, 1,
6.222222, 4.262626, -129.522, 1, 0.4823529, 0, 1,
6.262626, 4.262626, -129.53, 1, 0.4823529, 0, 1,
6.30303, 4.262626, -129.5426, 1, 0.4823529, 0, 1,
6.343434, 4.262626, -129.5596, 1, 0.4823529, 0, 1,
6.383838, 4.262626, -129.5811, 1, 0.4823529, 0, 1,
6.424242, 4.262626, -129.6071, 1, 0.4823529, 0, 1,
6.464646, 4.262626, -129.6376, 1, 0.4823529, 0, 1,
6.505051, 4.262626, -129.6727, 1, 0.4823529, 0, 1,
6.545455, 4.262626, -129.7121, 1, 0.4823529, 0, 1,
6.585859, 4.262626, -129.7561, 1, 0.4823529, 0, 1,
6.626263, 4.262626, -129.8046, 1, 0.4823529, 0, 1,
6.666667, 4.262626, -129.8576, 1, 0.4823529, 0, 1,
6.707071, 4.262626, -129.915, 1, 0.4823529, 0, 1,
6.747475, 4.262626, -129.977, 1, 0.4823529, 0, 1,
6.787879, 4.262626, -130.0434, 1, 0.4823529, 0, 1,
6.828283, 4.262626, -130.1144, 1, 0.4823529, 0, 1,
6.868687, 4.262626, -130.1898, 1, 0.4823529, 0, 1,
6.909091, 4.262626, -130.2697, 1, 0.4823529, 0, 1,
6.949495, 4.262626, -130.3541, 1, 0.4823529, 0, 1,
6.989899, 4.262626, -130.443, 1, 0.4823529, 0, 1,
7.030303, 4.262626, -130.5364, 1, 0.4823529, 0, 1,
7.070707, 4.262626, -130.6343, 1, 0.3764706, 0, 1,
7.111111, 4.262626, -130.7367, 1, 0.3764706, 0, 1,
7.151515, 4.262626, -130.8436, 1, 0.3764706, 0, 1,
7.191919, 4.262626, -130.955, 1, 0.3764706, 0, 1,
7.232323, 4.262626, -131.0708, 1, 0.3764706, 0, 1,
7.272727, 4.262626, -131.1912, 1, 0.3764706, 0, 1,
7.313131, 4.262626, -131.316, 1, 0.3764706, 0, 1,
7.353535, 4.262626, -131.4454, 1, 0.3764706, 0, 1,
7.393939, 4.262626, -131.5792, 1, 0.3764706, 0, 1,
7.434343, 4.262626, -131.7175, 1, 0.3764706, 0, 1,
7.474748, 4.262626, -131.8603, 1, 0.2745098, 0, 1,
7.515152, 4.262626, -132.0076, 1, 0.2745098, 0, 1,
7.555555, 4.262626, -132.1594, 1, 0.2745098, 0, 1,
7.59596, 4.262626, -132.3157, 1, 0.2745098, 0, 1,
7.636364, 4.262626, -132.4765, 1, 0.2745098, 0, 1,
7.676768, 4.262626, -132.6418, 1, 0.2745098, 0, 1,
7.717172, 4.262626, -132.8115, 1, 0.2745098, 0, 1,
7.757576, 4.262626, -132.9858, 1, 0.2745098, 0, 1,
7.79798, 4.262626, -133.1646, 1, 0.1686275, 0, 1,
7.838384, 4.262626, -133.3478, 1, 0.1686275, 0, 1,
7.878788, 4.262626, -133.5355, 1, 0.1686275, 0, 1,
7.919192, 4.262626, -133.7278, 1, 0.1686275, 0, 1,
7.959596, 4.262626, -133.9245, 1, 0.1686275, 0, 1,
8, 4.262626, -134.1257, 1, 0.1686275, 0, 1,
4, 4.30303, -136.1411, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.30303, -135.9065, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.30303, -135.6764, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.30303, -135.4506, 1, 0.06666667, 0, 1,
4.161616, 4.30303, -135.2293, 1, 0.06666667, 0, 1,
4.20202, 4.30303, -135.0124, 1, 0.06666667, 0, 1,
4.242424, 4.30303, -134.7999, 1, 0.06666667, 0, 1,
4.282828, 4.30303, -134.5918, 1, 0.06666667, 0, 1,
4.323232, 4.30303, -134.388, 1, 0.06666667, 0, 1,
4.363636, 4.30303, -134.1888, 1, 0.1686275, 0, 1,
4.40404, 4.30303, -133.9939, 1, 0.1686275, 0, 1,
4.444445, 4.30303, -133.8034, 1, 0.1686275, 0, 1,
4.484848, 4.30303, -133.6173, 1, 0.1686275, 0, 1,
4.525252, 4.30303, -133.4357, 1, 0.1686275, 0, 1,
4.565657, 4.30303, -133.2584, 1, 0.1686275, 0, 1,
4.606061, 4.30303, -133.0856, 1, 0.1686275, 0, 1,
4.646465, 4.30303, -132.9171, 1, 0.2745098, 0, 1,
4.686869, 4.30303, -132.7531, 1, 0.2745098, 0, 1,
4.727273, 4.30303, -132.5935, 1, 0.2745098, 0, 1,
4.767677, 4.30303, -132.4383, 1, 0.2745098, 0, 1,
4.808081, 4.30303, -132.2875, 1, 0.2745098, 0, 1,
4.848485, 4.30303, -132.1411, 1, 0.2745098, 0, 1,
4.888889, 4.30303, -131.9991, 1, 0.2745098, 0, 1,
4.929293, 4.30303, -131.8615, 1, 0.2745098, 0, 1,
4.969697, 4.30303, -131.7283, 1, 0.3764706, 0, 1,
5.010101, 4.30303, -131.5996, 1, 0.3764706, 0, 1,
5.050505, 4.30303, -131.4752, 1, 0.3764706, 0, 1,
5.090909, 4.30303, -131.3553, 1, 0.3764706, 0, 1,
5.131313, 4.30303, -131.2397, 1, 0.3764706, 0, 1,
5.171717, 4.30303, -131.1286, 1, 0.3764706, 0, 1,
5.212121, 4.30303, -131.0219, 1, 0.3764706, 0, 1,
5.252525, 4.30303, -130.9196, 1, 0.3764706, 0, 1,
5.292929, 4.30303, -130.8217, 1, 0.3764706, 0, 1,
5.333333, 4.30303, -130.7282, 1, 0.3764706, 0, 1,
5.373737, 4.30303, -130.6391, 1, 0.3764706, 0, 1,
5.414141, 4.30303, -130.5544, 1, 0.3764706, 0, 1,
5.454545, 4.30303, -130.4741, 1, 0.4823529, 0, 1,
5.494949, 4.30303, -130.3983, 1, 0.4823529, 0, 1,
5.535354, 4.30303, -130.3268, 1, 0.4823529, 0, 1,
5.575758, 4.30303, -130.2598, 1, 0.4823529, 0, 1,
5.616162, 4.30303, -130.1971, 1, 0.4823529, 0, 1,
5.656566, 4.30303, -130.1389, 1, 0.4823529, 0, 1,
5.69697, 4.30303, -130.0851, 1, 0.4823529, 0, 1,
5.737374, 4.30303, -130.0357, 1, 0.4823529, 0, 1,
5.777778, 4.30303, -129.9907, 1, 0.4823529, 0, 1,
5.818182, 4.30303, -129.9501, 1, 0.4823529, 0, 1,
5.858586, 4.30303, -129.9139, 1, 0.4823529, 0, 1,
5.89899, 4.30303, -129.8821, 1, 0.4823529, 0, 1,
5.939394, 4.30303, -129.8547, 1, 0.4823529, 0, 1,
5.979798, 4.30303, -129.8318, 1, 0.4823529, 0, 1,
6.020202, 4.30303, -129.8132, 1, 0.4823529, 0, 1,
6.060606, 4.30303, -129.7991, 1, 0.4823529, 0, 1,
6.10101, 4.30303, -129.7893, 1, 0.4823529, 0, 1,
6.141414, 4.30303, -129.784, 1, 0.4823529, 0, 1,
6.181818, 4.30303, -129.7831, 1, 0.4823529, 0, 1,
6.222222, 4.30303, -129.7866, 1, 0.4823529, 0, 1,
6.262626, 4.30303, -129.7945, 1, 0.4823529, 0, 1,
6.30303, 4.30303, -129.8068, 1, 0.4823529, 0, 1,
6.343434, 4.30303, -129.8235, 1, 0.4823529, 0, 1,
6.383838, 4.30303, -129.8446, 1, 0.4823529, 0, 1,
6.424242, 4.30303, -129.8701, 1, 0.4823529, 0, 1,
6.464646, 4.30303, -129.9001, 1, 0.4823529, 0, 1,
6.505051, 4.30303, -129.9344, 1, 0.4823529, 0, 1,
6.545455, 4.30303, -129.9732, 1, 0.4823529, 0, 1,
6.585859, 4.30303, -130.0163, 1, 0.4823529, 0, 1,
6.626263, 4.30303, -130.0639, 1, 0.4823529, 0, 1,
6.666667, 4.30303, -130.1159, 1, 0.4823529, 0, 1,
6.707071, 4.30303, -130.1723, 1, 0.4823529, 0, 1,
6.747475, 4.30303, -130.2331, 1, 0.4823529, 0, 1,
6.787879, 4.30303, -130.2983, 1, 0.4823529, 0, 1,
6.828283, 4.30303, -130.3679, 1, 0.4823529, 0, 1,
6.868687, 4.30303, -130.4419, 1, 0.4823529, 0, 1,
6.909091, 4.30303, -130.5203, 1, 0.4823529, 0, 1,
6.949495, 4.30303, -130.6032, 1, 0.3764706, 0, 1,
6.989899, 4.30303, -130.6904, 1, 0.3764706, 0, 1,
7.030303, 4.30303, -130.7821, 1, 0.3764706, 0, 1,
7.070707, 4.30303, -130.8781, 1, 0.3764706, 0, 1,
7.111111, 4.30303, -130.9786, 1, 0.3764706, 0, 1,
7.151515, 4.30303, -131.0835, 1, 0.3764706, 0, 1,
7.191919, 4.30303, -131.1928, 1, 0.3764706, 0, 1,
7.232323, 4.30303, -131.3065, 1, 0.3764706, 0, 1,
7.272727, 4.30303, -131.4246, 1, 0.3764706, 0, 1,
7.313131, 4.30303, -131.5471, 1, 0.3764706, 0, 1,
7.353535, 4.30303, -131.674, 1, 0.3764706, 0, 1,
7.393939, 4.30303, -131.8053, 1, 0.2745098, 0, 1,
7.434343, 4.30303, -131.9411, 1, 0.2745098, 0, 1,
7.474748, 4.30303, -132.0812, 1, 0.2745098, 0, 1,
7.515152, 4.30303, -132.2258, 1, 0.2745098, 0, 1,
7.555555, 4.30303, -132.3747, 1, 0.2745098, 0, 1,
7.59596, 4.30303, -132.5281, 1, 0.2745098, 0, 1,
7.636364, 4.30303, -132.6859, 1, 0.2745098, 0, 1,
7.676768, 4.30303, -132.8481, 1, 0.2745098, 0, 1,
7.717172, 4.30303, -133.0146, 1, 0.2745098, 0, 1,
7.757576, 4.30303, -133.1857, 1, 0.1686275, 0, 1,
7.79798, 4.30303, -133.3611, 1, 0.1686275, 0, 1,
7.838384, 4.30303, -133.5409, 1, 0.1686275, 0, 1,
7.878788, 4.30303, -133.7251, 1, 0.1686275, 0, 1,
7.919192, 4.30303, -133.9137, 1, 0.1686275, 0, 1,
7.959596, 4.30303, -134.1068, 1, 0.1686275, 0, 1,
8, 4.30303, -134.3042, 1, 0.06666667, 0, 1,
4, 4.343434, -136.2894, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.343434, -136.0591, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.343434, -135.8333, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.343434, -135.6117, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.343434, -135.3945, 1, 0.06666667, 0, 1,
4.20202, 4.343434, -135.1815, 1, 0.06666667, 0, 1,
4.242424, 4.343434, -134.973, 1, 0.06666667, 0, 1,
4.282828, 4.343434, -134.7687, 1, 0.06666667, 0, 1,
4.323232, 4.343434, -134.5688, 1, 0.06666667, 0, 1,
4.363636, 4.343434, -134.3732, 1, 0.06666667, 0, 1,
4.40404, 4.343434, -134.1819, 1, 0.1686275, 0, 1,
4.444445, 4.343434, -133.9949, 1, 0.1686275, 0, 1,
4.484848, 4.343434, -133.8123, 1, 0.1686275, 0, 1,
4.525252, 4.343434, -133.634, 1, 0.1686275, 0, 1,
4.565657, 4.343434, -133.46, 1, 0.1686275, 0, 1,
4.606061, 4.343434, -133.2904, 1, 0.1686275, 0, 1,
4.646465, 4.343434, -133.1251, 1, 0.1686275, 0, 1,
4.686869, 4.343434, -132.9641, 1, 0.2745098, 0, 1,
4.727273, 4.343434, -132.8074, 1, 0.2745098, 0, 1,
4.767677, 4.343434, -132.6551, 1, 0.2745098, 0, 1,
4.808081, 4.343434, -132.5071, 1, 0.2745098, 0, 1,
4.848485, 4.343434, -132.3634, 1, 0.2745098, 0, 1,
4.888889, 4.343434, -132.224, 1, 0.2745098, 0, 1,
4.929293, 4.343434, -132.089, 1, 0.2745098, 0, 1,
4.969697, 4.343434, -131.9583, 1, 0.2745098, 0, 1,
5.010101, 4.343434, -131.8319, 1, 0.2745098, 0, 1,
5.050505, 4.343434, -131.7099, 1, 0.3764706, 0, 1,
5.090909, 4.343434, -131.5921, 1, 0.3764706, 0, 1,
5.131313, 4.343434, -131.4788, 1, 0.3764706, 0, 1,
5.171717, 4.343434, -131.3697, 1, 0.3764706, 0, 1,
5.212121, 4.343434, -131.2649, 1, 0.3764706, 0, 1,
5.252525, 4.343434, -131.1645, 1, 0.3764706, 0, 1,
5.292929, 4.343434, -131.0684, 1, 0.3764706, 0, 1,
5.333333, 4.343434, -130.9767, 1, 0.3764706, 0, 1,
5.373737, 4.343434, -130.8892, 1, 0.3764706, 0, 1,
5.414141, 4.343434, -130.8061, 1, 0.3764706, 0, 1,
5.454545, 4.343434, -130.7273, 1, 0.3764706, 0, 1,
5.494949, 4.343434, -130.6529, 1, 0.3764706, 0, 1,
5.535354, 4.343434, -130.5827, 1, 0.3764706, 0, 1,
5.575758, 4.343434, -130.517, 1, 0.4823529, 0, 1,
5.616162, 4.343434, -130.4555, 1, 0.4823529, 0, 1,
5.656566, 4.343434, -130.3983, 1, 0.4823529, 0, 1,
5.69697, 4.343434, -130.3455, 1, 0.4823529, 0, 1,
5.737374, 4.343434, -130.297, 1, 0.4823529, 0, 1,
5.777778, 4.343434, -130.2528, 1, 0.4823529, 0, 1,
5.818182, 4.343434, -130.213, 1, 0.4823529, 0, 1,
5.858586, 4.343434, -130.1775, 1, 0.4823529, 0, 1,
5.89899, 4.343434, -130.1463, 1, 0.4823529, 0, 1,
5.939394, 4.343434, -130.1194, 1, 0.4823529, 0, 1,
5.979798, 4.343434, -130.0969, 1, 0.4823529, 0, 1,
6.020202, 4.343434, -130.0787, 1, 0.4823529, 0, 1,
6.060606, 4.343434, -130.0648, 1, 0.4823529, 0, 1,
6.10101, 4.343434, -130.0552, 1, 0.4823529, 0, 1,
6.141414, 4.343434, -130.05, 1, 0.4823529, 0, 1,
6.181818, 4.343434, -130.0491, 1, 0.4823529, 0, 1,
6.222222, 4.343434, -130.0525, 1, 0.4823529, 0, 1,
6.262626, 4.343434, -130.0603, 1, 0.4823529, 0, 1,
6.30303, 4.343434, -130.0723, 1, 0.4823529, 0, 1,
6.343434, 4.343434, -130.0887, 1, 0.4823529, 0, 1,
6.383838, 4.343434, -130.1095, 1, 0.4823529, 0, 1,
6.424242, 4.343434, -130.1345, 1, 0.4823529, 0, 1,
6.464646, 4.343434, -130.1639, 1, 0.4823529, 0, 1,
6.505051, 4.343434, -130.1976, 1, 0.4823529, 0, 1,
6.545455, 4.343434, -130.2357, 1, 0.4823529, 0, 1,
6.585859, 4.343434, -130.278, 1, 0.4823529, 0, 1,
6.626263, 4.343434, -130.3247, 1, 0.4823529, 0, 1,
6.666667, 4.343434, -130.3757, 1, 0.4823529, 0, 1,
6.707071, 4.343434, -130.4311, 1, 0.4823529, 0, 1,
6.747475, 4.343434, -130.4907, 1, 0.4823529, 0, 1,
6.787879, 4.343434, -130.5547, 1, 0.3764706, 0, 1,
6.828283, 4.343434, -130.623, 1, 0.3764706, 0, 1,
6.868687, 4.343434, -130.6957, 1, 0.3764706, 0, 1,
6.909091, 4.343434, -130.7727, 1, 0.3764706, 0, 1,
6.949495, 4.343434, -130.854, 1, 0.3764706, 0, 1,
6.989899, 4.343434, -130.9396, 1, 0.3764706, 0, 1,
7.030303, 4.343434, -131.0296, 1, 0.3764706, 0, 1,
7.070707, 4.343434, -131.1239, 1, 0.3764706, 0, 1,
7.111111, 4.343434, -131.2225, 1, 0.3764706, 0, 1,
7.151515, 4.343434, -131.3254, 1, 0.3764706, 0, 1,
7.191919, 4.343434, -131.4327, 1, 0.3764706, 0, 1,
7.232323, 4.343434, -131.5443, 1, 0.3764706, 0, 1,
7.272727, 4.343434, -131.6602, 1, 0.3764706, 0, 1,
7.313131, 4.343434, -131.7804, 1, 0.3764706, 0, 1,
7.353535, 4.343434, -131.905, 1, 0.2745098, 0, 1,
7.393939, 4.343434, -132.0339, 1, 0.2745098, 0, 1,
7.434343, 4.343434, -132.1671, 1, 0.2745098, 0, 1,
7.474748, 4.343434, -132.3046, 1, 0.2745098, 0, 1,
7.515152, 4.343434, -132.4465, 1, 0.2745098, 0, 1,
7.555555, 4.343434, -132.5927, 1, 0.2745098, 0, 1,
7.59596, 4.343434, -132.7433, 1, 0.2745098, 0, 1,
7.636364, 4.343434, -132.8981, 1, 0.2745098, 0, 1,
7.676768, 4.343434, -133.0573, 1, 0.1686275, 0, 1,
7.717172, 4.343434, -133.2208, 1, 0.1686275, 0, 1,
7.757576, 4.343434, -133.3886, 1, 0.1686275, 0, 1,
7.79798, 4.343434, -133.5608, 1, 0.1686275, 0, 1,
7.838384, 4.343434, -133.7373, 1, 0.1686275, 0, 1,
7.878788, 4.343434, -133.9181, 1, 0.1686275, 0, 1,
7.919192, 4.343434, -134.1032, 1, 0.1686275, 0, 1,
7.959596, 4.343434, -134.2927, 1, 0.06666667, 0, 1,
8, 4.343434, -134.4865, 1, 0.06666667, 0, 1,
4, 4.383838, -136.4421, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.383838, -136.2161, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.383838, -135.9943, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.383838, -135.7768, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.383838, -135.5636, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.383838, -135.3546, 1, 0.06666667, 0, 1,
4.242424, 4.383838, -135.1498, 1, 0.06666667, 0, 1,
4.282828, 4.383838, -134.9493, 1, 0.06666667, 0, 1,
4.323232, 4.383838, -134.7531, 1, 0.06666667, 0, 1,
4.363636, 4.383838, -134.561, 1, 0.06666667, 0, 1,
4.40404, 4.383838, -134.3733, 1, 0.06666667, 0, 1,
4.444445, 4.383838, -134.1897, 1, 0.1686275, 0, 1,
4.484848, 4.383838, -134.0105, 1, 0.1686275, 0, 1,
4.525252, 4.383838, -133.8354, 1, 0.1686275, 0, 1,
4.565657, 4.383838, -133.6647, 1, 0.1686275, 0, 1,
4.606061, 4.383838, -133.4981, 1, 0.1686275, 0, 1,
4.646465, 4.383838, -133.3359, 1, 0.1686275, 0, 1,
4.686869, 4.383838, -133.1778, 1, 0.1686275, 0, 1,
4.727273, 4.383838, -133.024, 1, 0.2745098, 0, 1,
4.767677, 4.383838, -132.8745, 1, 0.2745098, 0, 1,
4.808081, 4.383838, -132.7292, 1, 0.2745098, 0, 1,
4.848485, 4.383838, -132.5882, 1, 0.2745098, 0, 1,
4.888889, 4.383838, -132.4513, 1, 0.2745098, 0, 1,
4.929293, 4.383838, -132.3188, 1, 0.2745098, 0, 1,
4.969697, 4.383838, -132.1905, 1, 0.2745098, 0, 1,
5.010101, 4.383838, -132.0664, 1, 0.2745098, 0, 1,
5.050505, 4.383838, -131.9466, 1, 0.2745098, 0, 1,
5.090909, 4.383838, -131.8311, 1, 0.2745098, 0, 1,
5.131313, 4.383838, -131.7197, 1, 0.3764706, 0, 1,
5.171717, 4.383838, -131.6127, 1, 0.3764706, 0, 1,
5.212121, 4.383838, -131.5098, 1, 0.3764706, 0, 1,
5.252525, 4.383838, -131.4113, 1, 0.3764706, 0, 1,
5.292929, 4.383838, -131.3169, 1, 0.3764706, 0, 1,
5.333333, 4.383838, -131.2269, 1, 0.3764706, 0, 1,
5.373737, 4.383838, -131.141, 1, 0.3764706, 0, 1,
5.414141, 4.383838, -131.0594, 1, 0.3764706, 0, 1,
5.454545, 4.383838, -130.9821, 1, 0.3764706, 0, 1,
5.494949, 4.383838, -130.909, 1, 0.3764706, 0, 1,
5.535354, 4.383838, -130.8402, 1, 0.3764706, 0, 1,
5.575758, 4.383838, -130.7756, 1, 0.3764706, 0, 1,
5.616162, 4.383838, -130.7152, 1, 0.3764706, 0, 1,
5.656566, 4.383838, -130.6591, 1, 0.3764706, 0, 1,
5.69697, 4.383838, -130.6073, 1, 0.3764706, 0, 1,
5.737374, 4.383838, -130.5597, 1, 0.3764706, 0, 1,
5.777778, 4.383838, -130.5163, 1, 0.4823529, 0, 1,
5.818182, 4.383838, -130.4772, 1, 0.4823529, 0, 1,
5.858586, 4.383838, -130.4423, 1, 0.4823529, 0, 1,
5.89899, 4.383838, -130.4117, 1, 0.4823529, 0, 1,
5.939394, 4.383838, -130.3853, 1, 0.4823529, 0, 1,
5.979798, 4.383838, -130.3632, 1, 0.4823529, 0, 1,
6.020202, 4.383838, -130.3453, 1, 0.4823529, 0, 1,
6.060606, 4.383838, -130.3317, 1, 0.4823529, 0, 1,
6.10101, 4.383838, -130.3223, 1, 0.4823529, 0, 1,
6.141414, 4.383838, -130.3172, 1, 0.4823529, 0, 1,
6.181818, 4.383838, -130.3163, 1, 0.4823529, 0, 1,
6.222222, 4.383838, -130.3197, 1, 0.4823529, 0, 1,
6.262626, 4.383838, -130.3273, 1, 0.4823529, 0, 1,
6.30303, 4.383838, -130.3391, 1, 0.4823529, 0, 1,
6.343434, 4.383838, -130.3552, 1, 0.4823529, 0, 1,
6.383838, 4.383838, -130.3756, 1, 0.4823529, 0, 1,
6.424242, 4.383838, -130.4002, 1, 0.4823529, 0, 1,
6.464646, 4.383838, -130.429, 1, 0.4823529, 0, 1,
6.505051, 4.383838, -130.4621, 1, 0.4823529, 0, 1,
6.545455, 4.383838, -130.4994, 1, 0.4823529, 0, 1,
6.585859, 4.383838, -130.541, 1, 0.4823529, 0, 1,
6.626263, 4.383838, -130.5869, 1, 0.3764706, 0, 1,
6.666667, 4.383838, -130.6369, 1, 0.3764706, 0, 1,
6.707071, 4.383838, -130.6913, 1, 0.3764706, 0, 1,
6.747475, 4.383838, -130.7498, 1, 0.3764706, 0, 1,
6.787879, 4.383838, -130.8127, 1, 0.3764706, 0, 1,
6.828283, 4.383838, -130.8797, 1, 0.3764706, 0, 1,
6.868687, 4.383838, -130.951, 1, 0.3764706, 0, 1,
6.909091, 4.383838, -131.0266, 1, 0.3764706, 0, 1,
6.949495, 4.383838, -131.1064, 1, 0.3764706, 0, 1,
6.989899, 4.383838, -131.1905, 1, 0.3764706, 0, 1,
7.030303, 4.383838, -131.2788, 1, 0.3764706, 0, 1,
7.070707, 4.383838, -131.3713, 1, 0.3764706, 0, 1,
7.111111, 4.383838, -131.4681, 1, 0.3764706, 0, 1,
7.151515, 4.383838, -131.5692, 1, 0.3764706, 0, 1,
7.191919, 4.383838, -131.6745, 1, 0.3764706, 0, 1,
7.232323, 4.383838, -131.784, 1, 0.3764706, 0, 1,
7.272727, 4.383838, -131.8978, 1, 0.2745098, 0, 1,
7.313131, 4.383838, -132.0159, 1, 0.2745098, 0, 1,
7.353535, 4.383838, -132.1381, 1, 0.2745098, 0, 1,
7.393939, 4.383838, -132.2647, 1, 0.2745098, 0, 1,
7.434343, 4.383838, -132.3954, 1, 0.2745098, 0, 1,
7.474748, 4.383838, -132.5305, 1, 0.2745098, 0, 1,
7.515152, 4.383838, -132.6697, 1, 0.2745098, 0, 1,
7.555555, 4.383838, -132.8133, 1, 0.2745098, 0, 1,
7.59596, 4.383838, -132.961, 1, 0.2745098, 0, 1,
7.636364, 4.383838, -133.113, 1, 0.1686275, 0, 1,
7.676768, 4.383838, -133.2693, 1, 0.1686275, 0, 1,
7.717172, 4.383838, -133.4298, 1, 0.1686275, 0, 1,
7.757576, 4.383838, -133.5946, 1, 0.1686275, 0, 1,
7.79798, 4.383838, -133.7636, 1, 0.1686275, 0, 1,
7.838384, 4.383838, -133.9368, 1, 0.1686275, 0, 1,
7.878788, 4.383838, -134.1143, 1, 0.1686275, 0, 1,
7.919192, 4.383838, -134.2961, 1, 0.06666667, 0, 1,
7.959596, 4.383838, -134.4821, 1, 0.06666667, 0, 1,
8, 4.383838, -134.6723, 1, 0.06666667, 0, 1,
4, 4.424242, -136.599, 0.9647059, 0, 0.03137255, 1,
4.040404, 4.424242, -136.3771, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.424242, -136.1594, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.424242, -135.9458, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.424242, -135.7365, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.424242, -135.5313, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.424242, -135.3302, 1, 0.06666667, 0, 1,
4.282828, 4.424242, -135.1334, 1, 0.06666667, 0, 1,
4.323232, 4.424242, -134.9407, 1, 0.06666667, 0, 1,
4.363636, 4.424242, -134.7522, 1, 0.06666667, 0, 1,
4.40404, 4.424242, -134.5678, 1, 0.06666667, 0, 1,
4.444445, 4.424242, -134.3876, 1, 0.06666667, 0, 1,
4.484848, 4.424242, -134.2116, 1, 0.1686275, 0, 1,
4.525252, 4.424242, -134.0398, 1, 0.1686275, 0, 1,
4.565657, 4.424242, -133.8721, 1, 0.1686275, 0, 1,
4.606061, 4.424242, -133.7086, 1, 0.1686275, 0, 1,
4.646465, 4.424242, -133.5493, 1, 0.1686275, 0, 1,
4.686869, 4.424242, -133.3941, 1, 0.1686275, 0, 1,
4.727273, 4.424242, -133.2431, 1, 0.1686275, 0, 1,
4.767677, 4.424242, -133.0963, 1, 0.1686275, 0, 1,
4.808081, 4.424242, -132.9536, 1, 0.2745098, 0, 1,
4.848485, 4.424242, -132.8151, 1, 0.2745098, 0, 1,
4.888889, 4.424242, -132.6808, 1, 0.2745098, 0, 1,
4.929293, 4.424242, -132.5507, 1, 0.2745098, 0, 1,
4.969697, 4.424242, -132.4247, 1, 0.2745098, 0, 1,
5.010101, 4.424242, -132.3029, 1, 0.2745098, 0, 1,
5.050505, 4.424242, -132.1853, 1, 0.2745098, 0, 1,
5.090909, 4.424242, -132.0718, 1, 0.2745098, 0, 1,
5.131313, 4.424242, -131.9625, 1, 0.2745098, 0, 1,
5.171717, 4.424242, -131.8574, 1, 0.2745098, 0, 1,
5.212121, 4.424242, -131.7565, 1, 0.3764706, 0, 1,
5.252525, 4.424242, -131.6597, 1, 0.3764706, 0, 1,
5.292929, 4.424242, -131.567, 1, 0.3764706, 0, 1,
5.333333, 4.424242, -131.4786, 1, 0.3764706, 0, 1,
5.373737, 4.424242, -131.3943, 1, 0.3764706, 0, 1,
5.414141, 4.424242, -131.3142, 1, 0.3764706, 0, 1,
5.454545, 4.424242, -131.2383, 1, 0.3764706, 0, 1,
5.494949, 4.424242, -131.1665, 1, 0.3764706, 0, 1,
5.535354, 4.424242, -131.099, 1, 0.3764706, 0, 1,
5.575758, 4.424242, -131.0355, 1, 0.3764706, 0, 1,
5.616162, 4.424242, -130.9763, 1, 0.3764706, 0, 1,
5.656566, 4.424242, -130.9212, 1, 0.3764706, 0, 1,
5.69697, 4.424242, -130.8703, 1, 0.3764706, 0, 1,
5.737374, 4.424242, -130.8235, 1, 0.3764706, 0, 1,
5.777778, 4.424242, -130.781, 1, 0.3764706, 0, 1,
5.818182, 4.424242, -130.7426, 1, 0.3764706, 0, 1,
5.858586, 4.424242, -130.7083, 1, 0.3764706, 0, 1,
5.89899, 4.424242, -130.6783, 1, 0.3764706, 0, 1,
5.939394, 4.424242, -130.6524, 1, 0.3764706, 0, 1,
5.979798, 4.424242, -130.6307, 1, 0.3764706, 0, 1,
6.020202, 4.424242, -130.6131, 1, 0.3764706, 0, 1,
6.060606, 4.424242, -130.5997, 1, 0.3764706, 0, 1,
6.10101, 4.424242, -130.5905, 1, 0.3764706, 0, 1,
6.141414, 4.424242, -130.5855, 1, 0.3764706, 0, 1,
6.181818, 4.424242, -130.5846, 1, 0.3764706, 0, 1,
6.222222, 4.424242, -130.5879, 1, 0.3764706, 0, 1,
6.262626, 4.424242, -130.5954, 1, 0.3764706, 0, 1,
6.30303, 4.424242, -130.607, 1, 0.3764706, 0, 1,
6.343434, 4.424242, -130.6228, 1, 0.3764706, 0, 1,
6.383838, 4.424242, -130.6428, 1, 0.3764706, 0, 1,
6.424242, 4.424242, -130.6669, 1, 0.3764706, 0, 1,
6.464646, 4.424242, -130.6953, 1, 0.3764706, 0, 1,
6.505051, 4.424242, -130.7278, 1, 0.3764706, 0, 1,
6.545455, 4.424242, -130.7644, 1, 0.3764706, 0, 1,
6.585859, 4.424242, -130.8053, 1, 0.3764706, 0, 1,
6.626263, 4.424242, -130.8503, 1, 0.3764706, 0, 1,
6.666667, 4.424242, -130.8994, 1, 0.3764706, 0, 1,
6.707071, 4.424242, -130.9528, 1, 0.3764706, 0, 1,
6.747475, 4.424242, -131.0103, 1, 0.3764706, 0, 1,
6.787879, 4.424242, -131.0719, 1, 0.3764706, 0, 1,
6.828283, 4.424242, -131.1378, 1, 0.3764706, 0, 1,
6.868687, 4.424242, -131.2078, 1, 0.3764706, 0, 1,
6.909091, 4.424242, -131.282, 1, 0.3764706, 0, 1,
6.949495, 4.424242, -131.3604, 1, 0.3764706, 0, 1,
6.989899, 4.424242, -131.4429, 1, 0.3764706, 0, 1,
7.030303, 4.424242, -131.5296, 1, 0.3764706, 0, 1,
7.070707, 4.424242, -131.6205, 1, 0.3764706, 0, 1,
7.111111, 4.424242, -131.7155, 1, 0.3764706, 0, 1,
7.151515, 4.424242, -131.8147, 1, 0.2745098, 0, 1,
7.191919, 4.424242, -131.9181, 1, 0.2745098, 0, 1,
7.232323, 4.424242, -132.0256, 1, 0.2745098, 0, 1,
7.272727, 4.424242, -132.1374, 1, 0.2745098, 0, 1,
7.313131, 4.424242, -132.2533, 1, 0.2745098, 0, 1,
7.353535, 4.424242, -132.3733, 1, 0.2745098, 0, 1,
7.393939, 4.424242, -132.4975, 1, 0.2745098, 0, 1,
7.434343, 4.424242, -132.6259, 1, 0.2745098, 0, 1,
7.474748, 4.424242, -132.7585, 1, 0.2745098, 0, 1,
7.515152, 4.424242, -132.8952, 1, 0.2745098, 0, 1,
7.555555, 4.424242, -133.0362, 1, 0.1686275, 0, 1,
7.59596, 4.424242, -133.1812, 1, 0.1686275, 0, 1,
7.636364, 4.424242, -133.3305, 1, 0.1686275, 0, 1,
7.676768, 4.424242, -133.4839, 1, 0.1686275, 0, 1,
7.717172, 4.424242, -133.6415, 1, 0.1686275, 0, 1,
7.757576, 4.424242, -133.8033, 1, 0.1686275, 0, 1,
7.79798, 4.424242, -133.9692, 1, 0.1686275, 0, 1,
7.838384, 4.424242, -134.1393, 1, 0.1686275, 0, 1,
7.878788, 4.424242, -134.3136, 1, 0.06666667, 0, 1,
7.919192, 4.424242, -134.492, 1, 0.06666667, 0, 1,
7.959596, 4.424242, -134.6746, 1, 0.06666667, 0, 1,
8, 4.424242, -134.8614, 1, 0.06666667, 0, 1,
4, 4.464646, -136.7599, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.464646, -136.542, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.464646, -136.3282, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.464646, -136.1185, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.464646, -135.9129, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.464646, -135.7114, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.464646, -135.514, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.464646, -135.3207, 1, 0.06666667, 0, 1,
4.323232, 4.464646, -135.1315, 1, 0.06666667, 0, 1,
4.363636, 4.464646, -134.9464, 1, 0.06666667, 0, 1,
4.40404, 4.464646, -134.7653, 1, 0.06666667, 0, 1,
4.444445, 4.464646, -134.5884, 1, 0.06666667, 0, 1,
4.484848, 4.464646, -134.4155, 1, 0.06666667, 0, 1,
4.525252, 4.464646, -134.2468, 1, 0.1686275, 0, 1,
4.565657, 4.464646, -134.0821, 1, 0.1686275, 0, 1,
4.606061, 4.464646, -133.9216, 1, 0.1686275, 0, 1,
4.646465, 4.464646, -133.7651, 1, 0.1686275, 0, 1,
4.686869, 4.464646, -133.6127, 1, 0.1686275, 0, 1,
4.727273, 4.464646, -133.4645, 1, 0.1686275, 0, 1,
4.767677, 4.464646, -133.3203, 1, 0.1686275, 0, 1,
4.808081, 4.464646, -133.1802, 1, 0.1686275, 0, 1,
4.848485, 4.464646, -133.0442, 1, 0.1686275, 0, 1,
4.888889, 4.464646, -132.9123, 1, 0.2745098, 0, 1,
4.929293, 4.464646, -132.7845, 1, 0.2745098, 0, 1,
4.969697, 4.464646, -132.6608, 1, 0.2745098, 0, 1,
5.010101, 4.464646, -132.5412, 1, 0.2745098, 0, 1,
5.050505, 4.464646, -132.4257, 1, 0.2745098, 0, 1,
5.090909, 4.464646, -132.3143, 1, 0.2745098, 0, 1,
5.131313, 4.464646, -132.207, 1, 0.2745098, 0, 1,
5.171717, 4.464646, -132.1037, 1, 0.2745098, 0, 1,
5.212121, 4.464646, -132.0046, 1, 0.2745098, 0, 1,
5.252525, 4.464646, -131.9096, 1, 0.2745098, 0, 1,
5.292929, 4.464646, -131.8186, 1, 0.2745098, 0, 1,
5.333333, 4.464646, -131.7318, 1, 0.3764706, 0, 1,
5.373737, 4.464646, -131.649, 1, 0.3764706, 0, 1,
5.414141, 4.464646, -131.5704, 1, 0.3764706, 0, 1,
5.454545, 4.464646, -131.4958, 1, 0.3764706, 0, 1,
5.494949, 4.464646, -131.4253, 1, 0.3764706, 0, 1,
5.535354, 4.464646, -131.359, 1, 0.3764706, 0, 1,
5.575758, 4.464646, -131.2967, 1, 0.3764706, 0, 1,
5.616162, 4.464646, -131.2385, 1, 0.3764706, 0, 1,
5.656566, 4.464646, -131.1844, 1, 0.3764706, 0, 1,
5.69697, 4.464646, -131.1344, 1, 0.3764706, 0, 1,
5.737374, 4.464646, -131.0885, 1, 0.3764706, 0, 1,
5.777778, 4.464646, -131.0467, 1, 0.3764706, 0, 1,
5.818182, 4.464646, -131.009, 1, 0.3764706, 0, 1,
5.858586, 4.464646, -130.9754, 1, 0.3764706, 0, 1,
5.89899, 4.464646, -130.9459, 1, 0.3764706, 0, 1,
5.939394, 4.464646, -130.9204, 1, 0.3764706, 0, 1,
5.979798, 4.464646, -130.8991, 1, 0.3764706, 0, 1,
6.020202, 4.464646, -130.8819, 1, 0.3764706, 0, 1,
6.060606, 4.464646, -130.8687, 1, 0.3764706, 0, 1,
6.10101, 4.464646, -130.8597, 1, 0.3764706, 0, 1,
6.141414, 4.464646, -130.8547, 1, 0.3764706, 0, 1,
6.181818, 4.464646, -130.8539, 1, 0.3764706, 0, 1,
6.222222, 4.464646, -130.8571, 1, 0.3764706, 0, 1,
6.262626, 4.464646, -130.8645, 1, 0.3764706, 0, 1,
6.30303, 4.464646, -130.8759, 1, 0.3764706, 0, 1,
6.343434, 4.464646, -130.8914, 1, 0.3764706, 0, 1,
6.383838, 4.464646, -130.911, 1, 0.3764706, 0, 1,
6.424242, 4.464646, -130.9347, 1, 0.3764706, 0, 1,
6.464646, 4.464646, -130.9626, 1, 0.3764706, 0, 1,
6.505051, 4.464646, -130.9945, 1, 0.3764706, 0, 1,
6.545455, 4.464646, -131.0305, 1, 0.3764706, 0, 1,
6.585859, 4.464646, -131.0705, 1, 0.3764706, 0, 1,
6.626263, 4.464646, -131.1147, 1, 0.3764706, 0, 1,
6.666667, 4.464646, -131.163, 1, 0.3764706, 0, 1,
6.707071, 4.464646, -131.2154, 1, 0.3764706, 0, 1,
6.747475, 4.464646, -131.2719, 1, 0.3764706, 0, 1,
6.787879, 4.464646, -131.3324, 1, 0.3764706, 0, 1,
6.828283, 4.464646, -131.3971, 1, 0.3764706, 0, 1,
6.868687, 4.464646, -131.4659, 1, 0.3764706, 0, 1,
6.909091, 4.464646, -131.5387, 1, 0.3764706, 0, 1,
6.949495, 4.464646, -131.6157, 1, 0.3764706, 0, 1,
6.989899, 4.464646, -131.6967, 1, 0.3764706, 0, 1,
7.030303, 4.464646, -131.7818, 1, 0.3764706, 0, 1,
7.070707, 4.464646, -131.8711, 1, 0.2745098, 0, 1,
7.111111, 4.464646, -131.9644, 1, 0.2745098, 0, 1,
7.151515, 4.464646, -132.0618, 1, 0.2745098, 0, 1,
7.191919, 4.464646, -132.1633, 1, 0.2745098, 0, 1,
7.232323, 4.464646, -132.269, 1, 0.2745098, 0, 1,
7.272727, 4.464646, -132.3787, 1, 0.2745098, 0, 1,
7.313131, 4.464646, -132.4925, 1, 0.2745098, 0, 1,
7.353535, 4.464646, -132.6104, 1, 0.2745098, 0, 1,
7.393939, 4.464646, -132.7324, 1, 0.2745098, 0, 1,
7.434343, 4.464646, -132.8584, 1, 0.2745098, 0, 1,
7.474748, 4.464646, -132.9886, 1, 0.2745098, 0, 1,
7.515152, 4.464646, -133.1229, 1, 0.1686275, 0, 1,
7.555555, 4.464646, -133.2613, 1, 0.1686275, 0, 1,
7.59596, 4.464646, -133.4037, 1, 0.1686275, 0, 1,
7.636364, 4.464646, -133.5503, 1, 0.1686275, 0, 1,
7.676768, 4.464646, -133.701, 1, 0.1686275, 0, 1,
7.717172, 4.464646, -133.8557, 1, 0.1686275, 0, 1,
7.757576, 4.464646, -134.0146, 1, 0.1686275, 0, 1,
7.79798, 4.464646, -134.1775, 1, 0.1686275, 0, 1,
7.838384, 4.464646, -134.3445, 1, 0.06666667, 0, 1,
7.878788, 4.464646, -134.5157, 1, 0.06666667, 0, 1,
7.919192, 4.464646, -134.6909, 1, 0.06666667, 0, 1,
7.959596, 4.464646, -134.8702, 1, 0.06666667, 0, 1,
8, 4.464646, -135.0536, 1, 0.06666667, 0, 1,
4, 4.505051, -136.9246, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.505051, -136.7106, 0.9647059, 0, 0.03137255, 1,
4.080808, 4.505051, -136.5006, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.505051, -136.2947, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.505051, -136.0927, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.505051, -135.8948, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.505051, -135.7009, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.505051, -135.5111, 1, 0.06666667, 0, 1,
4.323232, 4.505051, -135.3252, 1, 0.06666667, 0, 1,
4.363636, 4.505051, -135.1434, 1, 0.06666667, 0, 1,
4.40404, 4.505051, -134.9656, 1, 0.06666667, 0, 1,
4.444445, 4.505051, -134.7918, 1, 0.06666667, 0, 1,
4.484848, 4.505051, -134.6221, 1, 0.06666667, 0, 1,
4.525252, 4.505051, -134.4563, 1, 0.06666667, 0, 1,
4.565657, 4.505051, -134.2946, 1, 0.06666667, 0, 1,
4.606061, 4.505051, -134.1369, 1, 0.1686275, 0, 1,
4.646465, 4.505051, -133.9833, 1, 0.1686275, 0, 1,
4.686869, 4.505051, -133.8336, 1, 0.1686275, 0, 1,
4.727273, 4.505051, -133.688, 1, 0.1686275, 0, 1,
4.767677, 4.505051, -133.5464, 1, 0.1686275, 0, 1,
4.808081, 4.505051, -133.4088, 1, 0.1686275, 0, 1,
4.848485, 4.505051, -133.2753, 1, 0.1686275, 0, 1,
4.888889, 4.505051, -133.1457, 1, 0.1686275, 0, 1,
4.929293, 4.505051, -133.0202, 1, 0.2745098, 0, 1,
4.969697, 4.505051, -132.8987, 1, 0.2745098, 0, 1,
5.010101, 4.505051, -132.7812, 1, 0.2745098, 0, 1,
5.050505, 4.505051, -132.6678, 1, 0.2745098, 0, 1,
5.090909, 4.505051, -132.5584, 1, 0.2745098, 0, 1,
5.131313, 4.505051, -132.453, 1, 0.2745098, 0, 1,
5.171717, 4.505051, -132.3516, 1, 0.2745098, 0, 1,
5.212121, 4.505051, -132.2542, 1, 0.2745098, 0, 1,
5.252525, 4.505051, -132.1609, 1, 0.2745098, 0, 1,
5.292929, 4.505051, -132.0715, 1, 0.2745098, 0, 1,
5.333333, 4.505051, -131.9863, 1, 0.2745098, 0, 1,
5.373737, 4.505051, -131.905, 1, 0.2745098, 0, 1,
5.414141, 4.505051, -131.8277, 1, 0.2745098, 0, 1,
5.454545, 4.505051, -131.7545, 1, 0.3764706, 0, 1,
5.494949, 4.505051, -131.6853, 1, 0.3764706, 0, 1,
5.535354, 4.505051, -131.6201, 1, 0.3764706, 0, 1,
5.575758, 4.505051, -131.5589, 1, 0.3764706, 0, 1,
5.616162, 4.505051, -131.5018, 1, 0.3764706, 0, 1,
5.656566, 4.505051, -131.4487, 1, 0.3764706, 0, 1,
5.69697, 4.505051, -131.3996, 1, 0.3764706, 0, 1,
5.737374, 4.505051, -131.3545, 1, 0.3764706, 0, 1,
5.777778, 4.505051, -131.3134, 1, 0.3764706, 0, 1,
5.818182, 4.505051, -131.2764, 1, 0.3764706, 0, 1,
5.858586, 4.505051, -131.2434, 1, 0.3764706, 0, 1,
5.89899, 4.505051, -131.2144, 1, 0.3764706, 0, 1,
5.939394, 4.505051, -131.1894, 1, 0.3764706, 0, 1,
5.979798, 4.505051, -131.1684, 1, 0.3764706, 0, 1,
6.020202, 4.505051, -131.1515, 1, 0.3764706, 0, 1,
6.060606, 4.505051, -131.1386, 1, 0.3764706, 0, 1,
6.10101, 4.505051, -131.1297, 1, 0.3764706, 0, 1,
6.141414, 4.505051, -131.1249, 1, 0.3764706, 0, 1,
6.181818, 4.505051, -131.124, 1, 0.3764706, 0, 1,
6.222222, 4.505051, -131.1272, 1, 0.3764706, 0, 1,
6.262626, 4.505051, -131.1344, 1, 0.3764706, 0, 1,
6.30303, 4.505051, -131.1456, 1, 0.3764706, 0, 1,
6.343434, 4.505051, -131.1609, 1, 0.3764706, 0, 1,
6.383838, 4.505051, -131.1802, 1, 0.3764706, 0, 1,
6.424242, 4.505051, -131.2034, 1, 0.3764706, 0, 1,
6.464646, 4.505051, -131.2308, 1, 0.3764706, 0, 1,
6.505051, 4.505051, -131.2621, 1, 0.3764706, 0, 1,
6.545455, 4.505051, -131.2974, 1, 0.3764706, 0, 1,
6.585859, 4.505051, -131.3368, 1, 0.3764706, 0, 1,
6.626263, 4.505051, -131.3802, 1, 0.3764706, 0, 1,
6.666667, 4.505051, -131.4276, 1, 0.3764706, 0, 1,
6.707071, 4.505051, -131.4791, 1, 0.3764706, 0, 1,
6.747475, 4.505051, -131.5345, 1, 0.3764706, 0, 1,
6.787879, 4.505051, -131.594, 1, 0.3764706, 0, 1,
6.828283, 4.505051, -131.6575, 1, 0.3764706, 0, 1,
6.868687, 4.505051, -131.7251, 1, 0.3764706, 0, 1,
6.909091, 4.505051, -131.7966, 1, 0.2745098, 0, 1,
6.949495, 4.505051, -131.8722, 1, 0.2745098, 0, 1,
6.989899, 4.505051, -131.9518, 1, 0.2745098, 0, 1,
7.030303, 4.505051, -132.0354, 1, 0.2745098, 0, 1,
7.070707, 4.505051, -132.1231, 1, 0.2745098, 0, 1,
7.111111, 4.505051, -132.2147, 1, 0.2745098, 0, 1,
7.151515, 4.505051, -132.3104, 1, 0.2745098, 0, 1,
7.191919, 4.505051, -132.4101, 1, 0.2745098, 0, 1,
7.232323, 4.505051, -132.5138, 1, 0.2745098, 0, 1,
7.272727, 4.505051, -132.6216, 1, 0.2745098, 0, 1,
7.313131, 4.505051, -132.7334, 1, 0.2745098, 0, 1,
7.353535, 4.505051, -132.8491, 1, 0.2745098, 0, 1,
7.393939, 4.505051, -132.969, 1, 0.2745098, 0, 1,
7.434343, 4.505051, -133.0928, 1, 0.1686275, 0, 1,
7.474748, 4.505051, -133.2206, 1, 0.1686275, 0, 1,
7.515152, 4.505051, -133.3525, 1, 0.1686275, 0, 1,
7.555555, 4.505051, -133.4884, 1, 0.1686275, 0, 1,
7.59596, 4.505051, -133.6284, 1, 0.1686275, 0, 1,
7.636364, 4.505051, -133.7723, 1, 0.1686275, 0, 1,
7.676768, 4.505051, -133.9203, 1, 0.1686275, 0, 1,
7.717172, 4.505051, -134.0723, 1, 0.1686275, 0, 1,
7.757576, 4.505051, -134.2283, 1, 0.1686275, 0, 1,
7.79798, 4.505051, -134.3883, 1, 0.06666667, 0, 1,
7.838384, 4.505051, -134.5524, 1, 0.06666667, 0, 1,
7.878788, 4.505051, -134.7204, 1, 0.06666667, 0, 1,
7.919192, 4.505051, -134.8925, 1, 0.06666667, 0, 1,
7.959596, 4.505051, -135.0686, 1, 0.06666667, 0, 1,
8, 4.505051, -135.2488, 1, 0.06666667, 0, 1,
4, 4.545455, -137.0928, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.545455, -136.8826, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.545455, -136.6764, 0.9647059, 0, 0.03137255, 1,
4.121212, 4.545455, -136.4741, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.545455, -136.2757, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.545455, -136.0813, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.545455, -135.8909, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.545455, -135.7043, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.545455, -135.5218, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.545455, -135.3432, 1, 0.06666667, 0, 1,
4.40404, 4.545455, -135.1685, 1, 0.06666667, 0, 1,
4.444445, 4.545455, -134.9978, 1, 0.06666667, 0, 1,
4.484848, 4.545455, -134.8311, 1, 0.06666667, 0, 1,
4.525252, 4.545455, -134.6683, 1, 0.06666667, 0, 1,
4.565657, 4.545455, -134.5094, 1, 0.06666667, 0, 1,
4.606061, 4.545455, -134.3545, 1, 0.06666667, 0, 1,
4.646465, 4.545455, -134.2036, 1, 0.1686275, 0, 1,
4.686869, 4.545455, -134.0566, 1, 0.1686275, 0, 1,
4.727273, 4.545455, -133.9135, 1, 0.1686275, 0, 1,
4.767677, 4.545455, -133.7744, 1, 0.1686275, 0, 1,
4.808081, 4.545455, -133.6393, 1, 0.1686275, 0, 1,
4.848485, 4.545455, -133.5081, 1, 0.1686275, 0, 1,
4.888889, 4.545455, -133.3809, 1, 0.1686275, 0, 1,
4.929293, 4.545455, -133.2576, 1, 0.1686275, 0, 1,
4.969697, 4.545455, -133.1382, 1, 0.1686275, 0, 1,
5.010101, 4.545455, -133.0228, 1, 0.2745098, 0, 1,
5.050505, 4.545455, -132.9114, 1, 0.2745098, 0, 1,
5.090909, 4.545455, -132.8039, 1, 0.2745098, 0, 1,
5.131313, 4.545455, -132.7003, 1, 0.2745098, 0, 1,
5.171717, 4.545455, -132.6008, 1, 0.2745098, 0, 1,
5.212121, 4.545455, -132.5051, 1, 0.2745098, 0, 1,
5.252525, 4.545455, -132.4134, 1, 0.2745098, 0, 1,
5.292929, 4.545455, -132.3257, 1, 0.2745098, 0, 1,
5.333333, 4.545455, -132.2419, 1, 0.2745098, 0, 1,
5.373737, 4.545455, -132.1621, 1, 0.2745098, 0, 1,
5.414141, 4.545455, -132.0862, 1, 0.2745098, 0, 1,
5.454545, 4.545455, -132.0142, 1, 0.2745098, 0, 1,
5.494949, 4.545455, -131.9463, 1, 0.2745098, 0, 1,
5.535354, 4.545455, -131.8822, 1, 0.2745098, 0, 1,
5.575758, 4.545455, -131.8221, 1, 0.2745098, 0, 1,
5.616162, 4.545455, -131.766, 1, 0.3764706, 0, 1,
5.656566, 4.545455, -131.7138, 1, 0.3764706, 0, 1,
5.69697, 4.545455, -131.6656, 1, 0.3764706, 0, 1,
5.737374, 4.545455, -131.6213, 1, 0.3764706, 0, 1,
5.777778, 4.545455, -131.581, 1, 0.3764706, 0, 1,
5.818182, 4.545455, -131.5446, 1, 0.3764706, 0, 1,
5.858586, 4.545455, -131.5122, 1, 0.3764706, 0, 1,
5.89899, 4.545455, -131.4837, 1, 0.3764706, 0, 1,
5.939394, 4.545455, -131.4592, 1, 0.3764706, 0, 1,
5.979798, 4.545455, -131.4386, 1, 0.3764706, 0, 1,
6.020202, 4.545455, -131.4219, 1, 0.3764706, 0, 1,
6.060606, 4.545455, -131.4093, 1, 0.3764706, 0, 1,
6.10101, 4.545455, -131.4005, 1, 0.3764706, 0, 1,
6.141414, 4.545455, -131.3958, 1, 0.3764706, 0, 1,
6.181818, 4.545455, -131.3949, 1, 0.3764706, 0, 1,
6.222222, 4.545455, -131.3981, 1, 0.3764706, 0, 1,
6.262626, 4.545455, -131.4051, 1, 0.3764706, 0, 1,
6.30303, 4.545455, -131.4162, 1, 0.3764706, 0, 1,
6.343434, 4.545455, -131.4311, 1, 0.3764706, 0, 1,
6.383838, 4.545455, -131.4501, 1, 0.3764706, 0, 1,
6.424242, 4.545455, -131.4729, 1, 0.3764706, 0, 1,
6.464646, 4.545455, -131.4998, 1, 0.3764706, 0, 1,
6.505051, 4.545455, -131.5306, 1, 0.3764706, 0, 1,
6.545455, 4.545455, -131.5653, 1, 0.3764706, 0, 1,
6.585859, 4.545455, -131.604, 1, 0.3764706, 0, 1,
6.626263, 4.545455, -131.6466, 1, 0.3764706, 0, 1,
6.666667, 4.545455, -131.6932, 1, 0.3764706, 0, 1,
6.707071, 4.545455, -131.7437, 1, 0.3764706, 0, 1,
6.747475, 4.545455, -131.7982, 1, 0.2745098, 0, 1,
6.787879, 4.545455, -131.8566, 1, 0.2745098, 0, 1,
6.828283, 4.545455, -131.919, 1, 0.2745098, 0, 1,
6.868687, 4.545455, -131.9854, 1, 0.2745098, 0, 1,
6.909091, 4.545455, -132.0556, 1, 0.2745098, 0, 1,
6.949495, 4.545455, -132.1299, 1, 0.2745098, 0, 1,
6.989899, 4.545455, -132.2081, 1, 0.2745098, 0, 1,
7.030303, 4.545455, -132.2902, 1, 0.2745098, 0, 1,
7.070707, 4.545455, -132.3763, 1, 0.2745098, 0, 1,
7.111111, 4.545455, -132.4663, 1, 0.2745098, 0, 1,
7.151515, 4.545455, -132.5603, 1, 0.2745098, 0, 1,
7.191919, 4.545455, -132.6582, 1, 0.2745098, 0, 1,
7.232323, 4.545455, -132.7601, 1, 0.2745098, 0, 1,
7.272727, 4.545455, -132.866, 1, 0.2745098, 0, 1,
7.313131, 4.545455, -132.9758, 1, 0.2745098, 0, 1,
7.353535, 4.545455, -133.0895, 1, 0.1686275, 0, 1,
7.393939, 4.545455, -133.2072, 1, 0.1686275, 0, 1,
7.434343, 4.545455, -133.3289, 1, 0.1686275, 0, 1,
7.474748, 4.545455, -133.4545, 1, 0.1686275, 0, 1,
7.515152, 4.545455, -133.584, 1, 0.1686275, 0, 1,
7.555555, 4.545455, -133.7175, 1, 0.1686275, 0, 1,
7.59596, 4.545455, -133.8549, 1, 0.1686275, 0, 1,
7.636364, 4.545455, -133.9963, 1, 0.1686275, 0, 1,
7.676768, 4.545455, -134.1417, 1, 0.1686275, 0, 1,
7.717172, 4.545455, -134.291, 1, 0.06666667, 0, 1,
7.757576, 4.545455, -134.4442, 1, 0.06666667, 0, 1,
7.79798, 4.545455, -134.6014, 1, 0.06666667, 0, 1,
7.838384, 4.545455, -134.7626, 1, 0.06666667, 0, 1,
7.878788, 4.545455, -134.9277, 1, 0.06666667, 0, 1,
7.919192, 4.545455, -135.0967, 1, 0.06666667, 0, 1,
7.959596, 4.545455, -135.2697, 1, 0.06666667, 0, 1,
8, 4.545455, -135.4467, 1, 0.06666667, 0, 1,
4, 4.585859, -137.2645, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.585859, -137.0579, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.585859, -136.8553, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.585859, -136.6565, 0.9647059, 0, 0.03137255, 1,
4.161616, 4.585859, -136.4617, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.585859, -136.2707, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.585859, -136.0836, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.585859, -135.9003, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.585859, -135.721, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.585859, -135.5455, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.585859, -135.3739, 1, 0.06666667, 0, 1,
4.444445, 4.585859, -135.2062, 1, 0.06666667, 0, 1,
4.484848, 4.585859, -135.0424, 1, 0.06666667, 0, 1,
4.525252, 4.585859, -134.8824, 1, 0.06666667, 0, 1,
4.565657, 4.585859, -134.7264, 1, 0.06666667, 0, 1,
4.606061, 4.585859, -134.5742, 1, 0.06666667, 0, 1,
4.646465, 4.585859, -134.4259, 1, 0.06666667, 0, 1,
4.686869, 4.585859, -134.2815, 1, 0.06666667, 0, 1,
4.727273, 4.585859, -134.1409, 1, 0.1686275, 0, 1,
4.767677, 4.585859, -134.0043, 1, 0.1686275, 0, 1,
4.808081, 4.585859, -133.8715, 1, 0.1686275, 0, 1,
4.848485, 4.585859, -133.7426, 1, 0.1686275, 0, 1,
4.888889, 4.585859, -133.6176, 1, 0.1686275, 0, 1,
4.929293, 4.585859, -133.4965, 1, 0.1686275, 0, 1,
4.969697, 4.585859, -133.3792, 1, 0.1686275, 0, 1,
5.010101, 4.585859, -133.2658, 1, 0.1686275, 0, 1,
5.050505, 4.585859, -133.1564, 1, 0.1686275, 0, 1,
5.090909, 4.585859, -133.0508, 1, 0.1686275, 0, 1,
5.131313, 4.585859, -132.949, 1, 0.2745098, 0, 1,
5.171717, 4.585859, -132.8512, 1, 0.2745098, 0, 1,
5.212121, 4.585859, -132.7572, 1, 0.2745098, 0, 1,
5.252525, 4.585859, -132.6671, 1, 0.2745098, 0, 1,
5.292929, 4.585859, -132.5809, 1, 0.2745098, 0, 1,
5.333333, 4.585859, -132.4986, 1, 0.2745098, 0, 1,
5.373737, 4.585859, -132.4202, 1, 0.2745098, 0, 1,
5.414141, 4.585859, -132.3456, 1, 0.2745098, 0, 1,
5.454545, 4.585859, -132.2749, 1, 0.2745098, 0, 1,
5.494949, 4.585859, -132.2082, 1, 0.2745098, 0, 1,
5.535354, 4.585859, -132.1452, 1, 0.2745098, 0, 1,
5.575758, 4.585859, -132.0862, 1, 0.2745098, 0, 1,
5.616162, 4.585859, -132.0311, 1, 0.2745098, 0, 1,
5.656566, 4.585859, -131.9798, 1, 0.2745098, 0, 1,
5.69697, 4.585859, -131.9324, 1, 0.2745098, 0, 1,
5.737374, 4.585859, -131.8889, 1, 0.2745098, 0, 1,
5.777778, 4.585859, -131.8493, 1, 0.2745098, 0, 1,
5.818182, 4.585859, -131.8135, 1, 0.2745098, 0, 1,
5.858586, 4.585859, -131.7817, 1, 0.3764706, 0, 1,
5.89899, 4.585859, -131.7537, 1, 0.3764706, 0, 1,
5.939394, 4.585859, -131.7296, 1, 0.3764706, 0, 1,
5.979798, 4.585859, -131.7094, 1, 0.3764706, 0, 1,
6.020202, 4.585859, -131.6931, 1, 0.3764706, 0, 1,
6.060606, 4.585859, -131.6806, 1, 0.3764706, 0, 1,
6.10101, 4.585859, -131.672, 1, 0.3764706, 0, 1,
6.141414, 4.585859, -131.6673, 1, 0.3764706, 0, 1,
6.181818, 4.585859, -131.6665, 1, 0.3764706, 0, 1,
6.222222, 4.585859, -131.6696, 1, 0.3764706, 0, 1,
6.262626, 4.585859, -131.6765, 1, 0.3764706, 0, 1,
6.30303, 4.585859, -131.6874, 1, 0.3764706, 0, 1,
6.343434, 4.585859, -131.7021, 1, 0.3764706, 0, 1,
6.383838, 4.585859, -131.7207, 1, 0.3764706, 0, 1,
6.424242, 4.585859, -131.7432, 1, 0.3764706, 0, 1,
6.464646, 4.585859, -131.7695, 1, 0.3764706, 0, 1,
6.505051, 4.585859, -131.7998, 1, 0.2745098, 0, 1,
6.545455, 4.585859, -131.8339, 1, 0.2745098, 0, 1,
6.585859, 4.585859, -131.8719, 1, 0.2745098, 0, 1,
6.626263, 4.585859, -131.9138, 1, 0.2745098, 0, 1,
6.666667, 4.585859, -131.9595, 1, 0.2745098, 0, 1,
6.707071, 4.585859, -132.0092, 1, 0.2745098, 0, 1,
6.747475, 4.585859, -132.0627, 1, 0.2745098, 0, 1,
6.787879, 4.585859, -132.1201, 1, 0.2745098, 0, 1,
6.828283, 4.585859, -132.1814, 1, 0.2745098, 0, 1,
6.868687, 4.585859, -132.2466, 1, 0.2745098, 0, 1,
6.909091, 4.585859, -132.3156, 1, 0.2745098, 0, 1,
6.949495, 4.585859, -132.3886, 1, 0.2745098, 0, 1,
6.989899, 4.585859, -132.4654, 1, 0.2745098, 0, 1,
7.030303, 4.585859, -132.5461, 1, 0.2745098, 0, 1,
7.070707, 4.585859, -132.6306, 1, 0.2745098, 0, 1,
7.111111, 4.585859, -132.7191, 1, 0.2745098, 0, 1,
7.151515, 4.585859, -132.8114, 1, 0.2745098, 0, 1,
7.191919, 4.585859, -132.9077, 1, 0.2745098, 0, 1,
7.232323, 4.585859, -133.0078, 1, 0.2745098, 0, 1,
7.272727, 4.585859, -133.1118, 1, 0.1686275, 0, 1,
7.313131, 4.585859, -133.2196, 1, 0.1686275, 0, 1,
7.353535, 4.585859, -133.3314, 1, 0.1686275, 0, 1,
7.393939, 4.585859, -133.447, 1, 0.1686275, 0, 1,
7.434343, 4.585859, -133.5665, 1, 0.1686275, 0, 1,
7.474748, 4.585859, -133.6899, 1, 0.1686275, 0, 1,
7.515152, 4.585859, -133.8172, 1, 0.1686275, 0, 1,
7.555555, 4.585859, -133.9483, 1, 0.1686275, 0, 1,
7.59596, 4.585859, -134.0834, 1, 0.1686275, 0, 1,
7.636364, 4.585859, -134.2223, 1, 0.1686275, 0, 1,
7.676768, 4.585859, -134.3651, 1, 0.06666667, 0, 1,
7.717172, 4.585859, -134.5117, 1, 0.06666667, 0, 1,
7.757576, 4.585859, -134.6623, 1, 0.06666667, 0, 1,
7.79798, 4.585859, -134.8168, 1, 0.06666667, 0, 1,
7.838384, 4.585859, -134.9751, 1, 0.06666667, 0, 1,
7.878788, 4.585859, -135.1373, 1, 0.06666667, 0, 1,
7.919192, 4.585859, -135.3034, 1, 0.06666667, 0, 1,
7.959596, 4.585859, -135.4733, 1, 0.06666667, 0, 1,
8, 4.585859, -135.6472, 0.9647059, 0, 0.03137255, 1,
4, 4.626263, -137.4393, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.626263, -137.2363, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.626263, -137.0372, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.626263, -136.8419, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.626263, -136.6504, 0.9647059, 0, 0.03137255, 1,
4.20202, 4.626263, -136.4628, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.626263, -136.2789, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.626263, -136.0988, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.626263, -135.9226, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.626263, -135.7502, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.626263, -135.5816, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.626263, -135.4168, 1, 0.06666667, 0, 1,
4.484848, 4.626263, -135.2558, 1, 0.06666667, 0, 1,
4.525252, 4.626263, -135.0987, 1, 0.06666667, 0, 1,
4.565657, 4.626263, -134.9453, 1, 0.06666667, 0, 1,
4.606061, 4.626263, -134.7958, 1, 0.06666667, 0, 1,
4.646465, 4.626263, -134.6501, 1, 0.06666667, 0, 1,
4.686869, 4.626263, -134.5081, 1, 0.06666667, 0, 1,
4.727273, 4.626263, -134.3701, 1, 0.06666667, 0, 1,
4.767677, 4.626263, -134.2358, 1, 0.1686275, 0, 1,
4.808081, 4.626263, -134.1053, 1, 0.1686275, 0, 1,
4.848485, 4.626263, -133.9787, 1, 0.1686275, 0, 1,
4.888889, 4.626263, -133.8558, 1, 0.1686275, 0, 1,
4.929293, 4.626263, -133.7368, 1, 0.1686275, 0, 1,
4.969697, 4.626263, -133.6216, 1, 0.1686275, 0, 1,
5.010101, 4.626263, -133.5102, 1, 0.1686275, 0, 1,
5.050505, 4.626263, -133.4026, 1, 0.1686275, 0, 1,
5.090909, 4.626263, -133.2988, 1, 0.1686275, 0, 1,
5.131313, 4.626263, -133.1989, 1, 0.1686275, 0, 1,
5.171717, 4.626263, -133.1027, 1, 0.1686275, 0, 1,
5.212121, 4.626263, -133.0104, 1, 0.2745098, 0, 1,
5.252525, 4.626263, -132.9219, 1, 0.2745098, 0, 1,
5.292929, 4.626263, -132.8372, 1, 0.2745098, 0, 1,
5.333333, 4.626263, -132.7563, 1, 0.2745098, 0, 1,
5.373737, 4.626263, -132.6792, 1, 0.2745098, 0, 1,
5.414141, 4.626263, -132.606, 1, 0.2745098, 0, 1,
5.454545, 4.626263, -132.5365, 1, 0.2745098, 0, 1,
5.494949, 4.626263, -132.4709, 1, 0.2745098, 0, 1,
5.535354, 4.626263, -132.4091, 1, 0.2745098, 0, 1,
5.575758, 4.626263, -132.3511, 1, 0.2745098, 0, 1,
5.616162, 4.626263, -132.2969, 1, 0.2745098, 0, 1,
5.656566, 4.626263, -132.2465, 1, 0.2745098, 0, 1,
5.69697, 4.626263, -132.1999, 1, 0.2745098, 0, 1,
5.737374, 4.626263, -132.1572, 1, 0.2745098, 0, 1,
5.777778, 4.626263, -132.1183, 1, 0.2745098, 0, 1,
5.818182, 4.626263, -132.0831, 1, 0.2745098, 0, 1,
5.858586, 4.626263, -132.0518, 1, 0.2745098, 0, 1,
5.89899, 4.626263, -132.0243, 1, 0.2745098, 0, 1,
5.939394, 4.626263, -132.0007, 1, 0.2745098, 0, 1,
5.979798, 4.626263, -131.9808, 1, 0.2745098, 0, 1,
6.020202, 4.626263, -131.9647, 1, 0.2745098, 0, 1,
6.060606, 4.626263, -131.9525, 1, 0.2745098, 0, 1,
6.10101, 4.626263, -131.9441, 1, 0.2745098, 0, 1,
6.141414, 4.626263, -131.9395, 1, 0.2745098, 0, 1,
6.181818, 4.626263, -131.9387, 1, 0.2745098, 0, 1,
6.222222, 4.626263, -131.9417, 1, 0.2745098, 0, 1,
6.262626, 4.626263, -131.9485, 1, 0.2745098, 0, 1,
6.30303, 4.626263, -131.9592, 1, 0.2745098, 0, 1,
6.343434, 4.626263, -131.9736, 1, 0.2745098, 0, 1,
6.383838, 4.626263, -131.9919, 1, 0.2745098, 0, 1,
6.424242, 4.626263, -132.014, 1, 0.2745098, 0, 1,
6.464646, 4.626263, -132.0399, 1, 0.2745098, 0, 1,
6.505051, 4.626263, -132.0696, 1, 0.2745098, 0, 1,
6.545455, 4.626263, -132.1031, 1, 0.2745098, 0, 1,
6.585859, 4.626263, -132.1405, 1, 0.2745098, 0, 1,
6.626263, 4.626263, -132.1816, 1, 0.2745098, 0, 1,
6.666667, 4.626263, -132.2266, 1, 0.2745098, 0, 1,
6.707071, 4.626263, -132.2754, 1, 0.2745098, 0, 1,
6.747475, 4.626263, -132.328, 1, 0.2745098, 0, 1,
6.787879, 4.626263, -132.3844, 1, 0.2745098, 0, 1,
6.828283, 4.626263, -132.4446, 1, 0.2745098, 0, 1,
6.868687, 4.626263, -132.5086, 1, 0.2745098, 0, 1,
6.909091, 4.626263, -132.5765, 1, 0.2745098, 0, 1,
6.949495, 4.626263, -132.6482, 1, 0.2745098, 0, 1,
6.989899, 4.626263, -132.7236, 1, 0.2745098, 0, 1,
7.030303, 4.626263, -132.8029, 1, 0.2745098, 0, 1,
7.070707, 4.626263, -132.886, 1, 0.2745098, 0, 1,
7.111111, 4.626263, -132.973, 1, 0.2745098, 0, 1,
7.151515, 4.626263, -133.0637, 1, 0.1686275, 0, 1,
7.191919, 4.626263, -133.1582, 1, 0.1686275, 0, 1,
7.232323, 4.626263, -133.2566, 1, 0.1686275, 0, 1,
7.272727, 4.626263, -133.3588, 1, 0.1686275, 0, 1,
7.313131, 4.626263, -133.4648, 1, 0.1686275, 0, 1,
7.353535, 4.626263, -133.5746, 1, 0.1686275, 0, 1,
7.393939, 4.626263, -133.6882, 1, 0.1686275, 0, 1,
7.434343, 4.626263, -133.8056, 1, 0.1686275, 0, 1,
7.474748, 4.626263, -133.9269, 1, 0.1686275, 0, 1,
7.515152, 4.626263, -134.0519, 1, 0.1686275, 0, 1,
7.555555, 4.626263, -134.1808, 1, 0.1686275, 0, 1,
7.59596, 4.626263, -134.3135, 1, 0.06666667, 0, 1,
7.636364, 4.626263, -134.45, 1, 0.06666667, 0, 1,
7.676768, 4.626263, -134.5903, 1, 0.06666667, 0, 1,
7.717172, 4.626263, -134.7344, 1, 0.06666667, 0, 1,
7.757576, 4.626263, -134.8824, 1, 0.06666667, 0, 1,
7.79798, 4.626263, -135.0341, 1, 0.06666667, 0, 1,
7.838384, 4.626263, -135.1897, 1, 0.06666667, 0, 1,
7.878788, 4.626263, -135.3491, 1, 0.06666667, 0, 1,
7.919192, 4.626263, -135.5123, 1, 0.06666667, 0, 1,
7.959596, 4.626263, -135.6793, 0.9647059, 0, 0.03137255, 1,
8, 4.626263, -135.8501, 0.9647059, 0, 0.03137255, 1,
4, 4.666667, -137.6171, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.666667, -137.4176, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.666667, -137.222, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.666667, -137.03, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.666667, -136.8418, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.666667, -136.6574, 0.9647059, 0, 0.03137255, 1,
4.242424, 4.666667, -136.4767, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.666667, -136.2998, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.666667, -136.1266, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.666667, -135.9571, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.666667, -135.7914, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.666667, -135.6295, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.666667, -135.4713, 1, 0.06666667, 0, 1,
4.525252, 4.666667, -135.3168, 1, 0.06666667, 0, 1,
4.565657, 4.666667, -135.1661, 1, 0.06666667, 0, 1,
4.606061, 4.666667, -135.0192, 1, 0.06666667, 0, 1,
4.646465, 4.666667, -134.876, 1, 0.06666667, 0, 1,
4.686869, 4.666667, -134.7365, 1, 0.06666667, 0, 1,
4.727273, 4.666667, -134.6008, 1, 0.06666667, 0, 1,
4.767677, 4.666667, -134.4688, 1, 0.06666667, 0, 1,
4.808081, 4.666667, -134.3406, 1, 0.06666667, 0, 1,
4.848485, 4.666667, -134.2161, 1, 0.1686275, 0, 1,
4.888889, 4.666667, -134.0954, 1, 0.1686275, 0, 1,
4.929293, 4.666667, -133.9784, 1, 0.1686275, 0, 1,
4.969697, 4.666667, -133.8652, 1, 0.1686275, 0, 1,
5.010101, 4.666667, -133.7557, 1, 0.1686275, 0, 1,
5.050505, 4.666667, -133.65, 1, 0.1686275, 0, 1,
5.090909, 4.666667, -133.548, 1, 0.1686275, 0, 1,
5.131313, 4.666667, -133.4498, 1, 0.1686275, 0, 1,
5.171717, 4.666667, -133.3553, 1, 0.1686275, 0, 1,
5.212121, 4.666667, -133.2646, 1, 0.1686275, 0, 1,
5.252525, 4.666667, -133.1776, 1, 0.1686275, 0, 1,
5.292929, 4.666667, -133.0943, 1, 0.1686275, 0, 1,
5.333333, 4.666667, -133.0148, 1, 0.2745098, 0, 1,
5.373737, 4.666667, -132.9391, 1, 0.2745098, 0, 1,
5.414141, 4.666667, -132.8671, 1, 0.2745098, 0, 1,
5.454545, 4.666667, -132.7989, 1, 0.2745098, 0, 1,
5.494949, 4.666667, -132.7344, 1, 0.2745098, 0, 1,
5.535354, 4.666667, -132.6736, 1, 0.2745098, 0, 1,
5.575758, 4.666667, -132.6166, 1, 0.2745098, 0, 1,
5.616162, 4.666667, -132.5634, 1, 0.2745098, 0, 1,
5.656566, 4.666667, -132.5138, 1, 0.2745098, 0, 1,
5.69697, 4.666667, -132.4681, 1, 0.2745098, 0, 1,
5.737374, 4.666667, -132.4261, 1, 0.2745098, 0, 1,
5.777778, 4.666667, -132.3878, 1, 0.2745098, 0, 1,
5.818182, 4.666667, -132.3533, 1, 0.2745098, 0, 1,
5.858586, 4.666667, -132.3225, 1, 0.2745098, 0, 1,
5.89899, 4.666667, -132.2955, 1, 0.2745098, 0, 1,
5.939394, 4.666667, -132.2722, 1, 0.2745098, 0, 1,
5.979798, 4.666667, -132.2527, 1, 0.2745098, 0, 1,
6.020202, 4.666667, -132.2369, 1, 0.2745098, 0, 1,
6.060606, 4.666667, -132.2249, 1, 0.2745098, 0, 1,
6.10101, 4.666667, -132.2166, 1, 0.2745098, 0, 1,
6.141414, 4.666667, -132.2121, 1, 0.2745098, 0, 1,
6.181818, 4.666667, -132.2113, 1, 0.2745098, 0, 1,
6.222222, 4.666667, -132.2143, 1, 0.2745098, 0, 1,
6.262626, 4.666667, -132.221, 1, 0.2745098, 0, 1,
6.30303, 4.666667, -132.2315, 1, 0.2745098, 0, 1,
6.343434, 4.666667, -132.2457, 1, 0.2745098, 0, 1,
6.383838, 4.666667, -132.2636, 1, 0.2745098, 0, 1,
6.424242, 4.666667, -132.2853, 1, 0.2745098, 0, 1,
6.464646, 4.666667, -132.3108, 1, 0.2745098, 0, 1,
6.505051, 4.666667, -132.34, 1, 0.2745098, 0, 1,
6.545455, 4.666667, -132.3729, 1, 0.2745098, 0, 1,
6.585859, 4.666667, -132.4096, 1, 0.2745098, 0, 1,
6.626263, 4.666667, -132.4501, 1, 0.2745098, 0, 1,
6.666667, 4.666667, -132.4943, 1, 0.2745098, 0, 1,
6.707071, 4.666667, -132.5422, 1, 0.2745098, 0, 1,
6.747475, 4.666667, -132.5939, 1, 0.2745098, 0, 1,
6.787879, 4.666667, -132.6493, 1, 0.2745098, 0, 1,
6.828283, 4.666667, -132.7085, 1, 0.2745098, 0, 1,
6.868687, 4.666667, -132.7715, 1, 0.2745098, 0, 1,
6.909091, 4.666667, -132.8381, 1, 0.2745098, 0, 1,
6.949495, 4.666667, -132.9086, 1, 0.2745098, 0, 1,
6.989899, 4.666667, -132.9827, 1, 0.2745098, 0, 1,
7.030303, 4.666667, -133.0607, 1, 0.1686275, 0, 1,
7.070707, 4.666667, -133.1423, 1, 0.1686275, 0, 1,
7.111111, 4.666667, -133.2278, 1, 0.1686275, 0, 1,
7.151515, 4.666667, -133.3169, 1, 0.1686275, 0, 1,
7.191919, 4.666667, -133.4099, 1, 0.1686275, 0, 1,
7.232323, 4.666667, -133.5065, 1, 0.1686275, 0, 1,
7.272727, 4.666667, -133.6069, 1, 0.1686275, 0, 1,
7.313131, 4.666667, -133.7111, 1, 0.1686275, 0, 1,
7.353535, 4.666667, -133.819, 1, 0.1686275, 0, 1,
7.393939, 4.666667, -133.9307, 1, 0.1686275, 0, 1,
7.434343, 4.666667, -134.0461, 1, 0.1686275, 0, 1,
7.474748, 4.666667, -134.1652, 1, 0.1686275, 0, 1,
7.515152, 4.666667, -134.2881, 1, 0.06666667, 0, 1,
7.555555, 4.666667, -134.4148, 1, 0.06666667, 0, 1,
7.59596, 4.666667, -134.5452, 1, 0.06666667, 0, 1,
7.636364, 4.666667, -134.6793, 1, 0.06666667, 0, 1,
7.676768, 4.666667, -134.8172, 1, 0.06666667, 0, 1,
7.717172, 4.666667, -134.9589, 1, 0.06666667, 0, 1,
7.757576, 4.666667, -135.1043, 1, 0.06666667, 0, 1,
7.79798, 4.666667, -135.2534, 1, 0.06666667, 0, 1,
7.838384, 4.666667, -135.4063, 1, 0.06666667, 0, 1,
7.878788, 4.666667, -135.5629, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.666667, -135.7233, 0.9647059, 0, 0.03137255, 1,
7.959596, 4.666667, -135.8874, 0.9647059, 0, 0.03137255, 1,
8, 4.666667, -136.0553, 0.9647059, 0, 0.03137255, 1,
4, 4.707071, -137.7977, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.707071, -137.6017, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.707071, -137.4094, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.707071, -137.2207, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.707071, -137.0357, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.707071, -136.8544, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.707071, -136.6768, 0.9647059, 0, 0.03137255, 1,
4.282828, 4.707071, -136.5029, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.707071, -136.3327, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.707071, -136.1662, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.707071, -136.0033, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.707071, -135.8441, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.707071, -135.6886, 0.9647059, 0, 0.03137255, 1,
4.525252, 4.707071, -135.5368, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.707071, -135.3887, 1, 0.06666667, 0, 1,
4.606061, 4.707071, -135.2442, 1, 0.06666667, 0, 1,
4.646465, 4.707071, -135.1035, 1, 0.06666667, 0, 1,
4.686869, 4.707071, -134.9664, 1, 0.06666667, 0, 1,
4.727273, 4.707071, -134.833, 1, 0.06666667, 0, 1,
4.767677, 4.707071, -134.7033, 1, 0.06666667, 0, 1,
4.808081, 4.707071, -134.5773, 1, 0.06666667, 0, 1,
4.848485, 4.707071, -134.4549, 1, 0.06666667, 0, 1,
4.888889, 4.707071, -134.3363, 1, 0.06666667, 0, 1,
4.929293, 4.707071, -134.2213, 1, 0.1686275, 0, 1,
4.969697, 4.707071, -134.11, 1, 0.1686275, 0, 1,
5.010101, 4.707071, -134.0024, 1, 0.1686275, 0, 1,
5.050505, 4.707071, -133.8985, 1, 0.1686275, 0, 1,
5.090909, 4.707071, -133.7982, 1, 0.1686275, 0, 1,
5.131313, 4.707071, -133.7017, 1, 0.1686275, 0, 1,
5.171717, 4.707071, -133.6088, 1, 0.1686275, 0, 1,
5.212121, 4.707071, -133.5196, 1, 0.1686275, 0, 1,
5.252525, 4.707071, -133.4341, 1, 0.1686275, 0, 1,
5.292929, 4.707071, -133.3523, 1, 0.1686275, 0, 1,
5.333333, 4.707071, -133.2742, 1, 0.1686275, 0, 1,
5.373737, 4.707071, -133.1997, 1, 0.1686275, 0, 1,
5.414141, 4.707071, -133.129, 1, 0.1686275, 0, 1,
5.454545, 4.707071, -133.0619, 1, 0.1686275, 0, 1,
5.494949, 4.707071, -132.9985, 1, 0.2745098, 0, 1,
5.535354, 4.707071, -132.9388, 1, 0.2745098, 0, 1,
5.575758, 4.707071, -132.8827, 1, 0.2745098, 0, 1,
5.616162, 4.707071, -132.8304, 1, 0.2745098, 0, 1,
5.656566, 4.707071, -132.7817, 1, 0.2745098, 0, 1,
5.69697, 4.707071, -132.7368, 1, 0.2745098, 0, 1,
5.737374, 4.707071, -132.6954, 1, 0.2745098, 0, 1,
5.777778, 4.707071, -132.6579, 1, 0.2745098, 0, 1,
5.818182, 4.707071, -132.6239, 1, 0.2745098, 0, 1,
5.858586, 4.707071, -132.5937, 1, 0.2745098, 0, 1,
5.89899, 4.707071, -132.5671, 1, 0.2745098, 0, 1,
5.939394, 4.707071, -132.5443, 1, 0.2745098, 0, 1,
5.979798, 4.707071, -132.5251, 1, 0.2745098, 0, 1,
6.020202, 4.707071, -132.5096, 1, 0.2745098, 0, 1,
6.060606, 4.707071, -132.4977, 1, 0.2745098, 0, 1,
6.10101, 4.707071, -132.4896, 1, 0.2745098, 0, 1,
6.141414, 4.707071, -132.4851, 1, 0.2745098, 0, 1,
6.181818, 4.707071, -132.4844, 1, 0.2745098, 0, 1,
6.222222, 4.707071, -132.4873, 1, 0.2745098, 0, 1,
6.262626, 4.707071, -132.4939, 1, 0.2745098, 0, 1,
6.30303, 4.707071, -132.5042, 1, 0.2745098, 0, 1,
6.343434, 4.707071, -132.5181, 1, 0.2745098, 0, 1,
6.383838, 4.707071, -132.5358, 1, 0.2745098, 0, 1,
6.424242, 4.707071, -132.5571, 1, 0.2745098, 0, 1,
6.464646, 4.707071, -132.5821, 1, 0.2745098, 0, 1,
6.505051, 4.707071, -132.6108, 1, 0.2745098, 0, 1,
6.545455, 4.707071, -132.6432, 1, 0.2745098, 0, 1,
6.585859, 4.707071, -132.6793, 1, 0.2745098, 0, 1,
6.626263, 4.707071, -132.7191, 1, 0.2745098, 0, 1,
6.666667, 4.707071, -132.7625, 1, 0.2745098, 0, 1,
6.707071, 4.707071, -132.8096, 1, 0.2745098, 0, 1,
6.747475, 4.707071, -132.8604, 1, 0.2745098, 0, 1,
6.787879, 4.707071, -132.9149, 1, 0.2745098, 0, 1,
6.828283, 4.707071, -132.9731, 1, 0.2745098, 0, 1,
6.868687, 4.707071, -133.0349, 1, 0.1686275, 0, 1,
6.909091, 4.707071, -133.1005, 1, 0.1686275, 0, 1,
6.949495, 4.707071, -133.1697, 1, 0.1686275, 0, 1,
6.989899, 4.707071, -133.2426, 1, 0.1686275, 0, 1,
7.030303, 4.707071, -133.3192, 1, 0.1686275, 0, 1,
7.070707, 4.707071, -133.3995, 1, 0.1686275, 0, 1,
7.111111, 4.707071, -133.4834, 1, 0.1686275, 0, 1,
7.151515, 4.707071, -133.5711, 1, 0.1686275, 0, 1,
7.191919, 4.707071, -133.6624, 1, 0.1686275, 0, 1,
7.232323, 4.707071, -133.7574, 1, 0.1686275, 0, 1,
7.272727, 4.707071, -133.8561, 1, 0.1686275, 0, 1,
7.313131, 4.707071, -133.9585, 1, 0.1686275, 0, 1,
7.353535, 4.707071, -134.0646, 1, 0.1686275, 0, 1,
7.393939, 4.707071, -134.1743, 1, 0.1686275, 0, 1,
7.434343, 4.707071, -134.2878, 1, 0.06666667, 0, 1,
7.474748, 4.707071, -134.4049, 1, 0.06666667, 0, 1,
7.515152, 4.707071, -134.5257, 1, 0.06666667, 0, 1,
7.555555, 4.707071, -134.6502, 1, 0.06666667, 0, 1,
7.59596, 4.707071, -134.7784, 1, 0.06666667, 0, 1,
7.636364, 4.707071, -134.9102, 1, 0.06666667, 0, 1,
7.676768, 4.707071, -135.0457, 1, 0.06666667, 0, 1,
7.717172, 4.707071, -135.185, 1, 0.06666667, 0, 1,
7.757576, 4.707071, -135.3279, 1, 0.06666667, 0, 1,
7.79798, 4.707071, -135.4745, 1, 0.06666667, 0, 1,
7.838384, 4.707071, -135.6247, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.707071, -135.7787, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.707071, -135.9363, 0.9647059, 0, 0.03137255, 1,
7.959596, 4.707071, -136.0977, 0.9647059, 0, 0.03137255, 1,
8, 4.707071, -136.2627, 0.9647059, 0, 0.03137255, 1,
4, 4.747475, -137.981, 0.8588235, 0, 0.1372549, 1,
4.040404, 4.747475, -137.7883, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.747475, -137.5993, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.747475, -137.4138, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.747475, -137.232, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.747475, -137.0538, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.747475, -136.8792, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.747475, -136.7082, 0.9647059, 0, 0.03137255, 1,
4.323232, 4.747475, -136.5409, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.747475, -136.3771, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.747475, -136.217, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.747475, -136.0605, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.747475, -135.9077, 0.9647059, 0, 0.03137255, 1,
4.525252, 4.747475, -135.7585, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.747475, -135.6128, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.747475, -135.4708, 1, 0.06666667, 0, 1,
4.646465, 4.747475, -135.3325, 1, 0.06666667, 0, 1,
4.686869, 4.747475, -135.1977, 1, 0.06666667, 0, 1,
4.727273, 4.747475, -135.0666, 1, 0.06666667, 0, 1,
4.767677, 4.747475, -134.9391, 1, 0.06666667, 0, 1,
4.808081, 4.747475, -134.8152, 1, 0.06666667, 0, 1,
4.848485, 4.747475, -134.6949, 1, 0.06666667, 0, 1,
4.888889, 4.747475, -134.5783, 1, 0.06666667, 0, 1,
4.929293, 4.747475, -134.4652, 1, 0.06666667, 0, 1,
4.969697, 4.747475, -134.3558, 1, 0.06666667, 0, 1,
5.010101, 4.747475, -134.25, 1, 0.1686275, 0, 1,
5.050505, 4.747475, -134.1479, 1, 0.1686275, 0, 1,
5.090909, 4.747475, -134.0493, 1, 0.1686275, 0, 1,
5.131313, 4.747475, -133.9544, 1, 0.1686275, 0, 1,
5.171717, 4.747475, -133.8631, 1, 0.1686275, 0, 1,
5.212121, 4.747475, -133.7755, 1, 0.1686275, 0, 1,
5.252525, 4.747475, -133.6914, 1, 0.1686275, 0, 1,
5.292929, 4.747475, -133.611, 1, 0.1686275, 0, 1,
5.333333, 4.747475, -133.5342, 1, 0.1686275, 0, 1,
5.373737, 4.747475, -133.461, 1, 0.1686275, 0, 1,
5.414141, 4.747475, -133.3914, 1, 0.1686275, 0, 1,
5.454545, 4.747475, -133.3255, 1, 0.1686275, 0, 1,
5.494949, 4.747475, -133.2632, 1, 0.1686275, 0, 1,
5.535354, 4.747475, -133.2045, 1, 0.1686275, 0, 1,
5.575758, 4.747475, -133.1494, 1, 0.1686275, 0, 1,
5.616162, 4.747475, -133.0979, 1, 0.1686275, 0, 1,
5.656566, 4.747475, -133.0501, 1, 0.1686275, 0, 1,
5.69697, 4.747475, -133.0059, 1, 0.2745098, 0, 1,
5.737374, 4.747475, -132.9653, 1, 0.2745098, 0, 1,
5.777778, 4.747475, -132.9283, 1, 0.2745098, 0, 1,
5.818182, 4.747475, -132.8949, 1, 0.2745098, 0, 1,
5.858586, 4.747475, -132.8652, 1, 0.2745098, 0, 1,
5.89899, 4.747475, -132.8391, 1, 0.2745098, 0, 1,
5.939394, 4.747475, -132.8166, 1, 0.2745098, 0, 1,
5.979798, 4.747475, -132.7978, 1, 0.2745098, 0, 1,
6.020202, 4.747475, -132.7825, 1, 0.2745098, 0, 1,
6.060606, 4.747475, -132.7709, 1, 0.2745098, 0, 1,
6.10101, 4.747475, -132.7629, 1, 0.2745098, 0, 1,
6.141414, 4.747475, -132.7585, 1, 0.2745098, 0, 1,
6.181818, 4.747475, -132.7578, 1, 0.2745098, 0, 1,
6.222222, 4.747475, -132.7606, 1, 0.2745098, 0, 1,
6.262626, 4.747475, -132.7671, 1, 0.2745098, 0, 1,
6.30303, 4.747475, -132.7772, 1, 0.2745098, 0, 1,
6.343434, 4.747475, -132.791, 1, 0.2745098, 0, 1,
6.383838, 4.747475, -132.8083, 1, 0.2745098, 0, 1,
6.424242, 4.747475, -132.8293, 1, 0.2745098, 0, 1,
6.464646, 4.747475, -132.8539, 1, 0.2745098, 0, 1,
6.505051, 4.747475, -132.8821, 1, 0.2745098, 0, 1,
6.545455, 4.747475, -132.9139, 1, 0.2745098, 0, 1,
6.585859, 4.747475, -132.9494, 1, 0.2745098, 0, 1,
6.626263, 4.747475, -132.9885, 1, 0.2745098, 0, 1,
6.666667, 4.747475, -133.0312, 1, 0.1686275, 0, 1,
6.707071, 4.747475, -133.0775, 1, 0.1686275, 0, 1,
6.747475, 4.747475, -133.1274, 1, 0.1686275, 0, 1,
6.787879, 4.747475, -133.181, 1, 0.1686275, 0, 1,
6.828283, 4.747475, -133.2382, 1, 0.1686275, 0, 1,
6.868687, 4.747475, -133.299, 1, 0.1686275, 0, 1,
6.909091, 4.747475, -133.3634, 1, 0.1686275, 0, 1,
6.949495, 4.747475, -133.4315, 1, 0.1686275, 0, 1,
6.989899, 4.747475, -133.5032, 1, 0.1686275, 0, 1,
7.030303, 4.747475, -133.5784, 1, 0.1686275, 0, 1,
7.070707, 4.747475, -133.6574, 1, 0.1686275, 0, 1,
7.111111, 4.747475, -133.7399, 1, 0.1686275, 0, 1,
7.151515, 4.747475, -133.8261, 1, 0.1686275, 0, 1,
7.191919, 4.747475, -133.9158, 1, 0.1686275, 0, 1,
7.232323, 4.747475, -134.0092, 1, 0.1686275, 0, 1,
7.272727, 4.747475, -134.1063, 1, 0.1686275, 0, 1,
7.313131, 4.747475, -134.2069, 1, 0.1686275, 0, 1,
7.353535, 4.747475, -134.3112, 1, 0.06666667, 0, 1,
7.393939, 4.747475, -134.4191, 1, 0.06666667, 0, 1,
7.434343, 4.747475, -134.5306, 1, 0.06666667, 0, 1,
7.474748, 4.747475, -134.6457, 1, 0.06666667, 0, 1,
7.515152, 4.747475, -134.7645, 1, 0.06666667, 0, 1,
7.555555, 4.747475, -134.8869, 1, 0.06666667, 0, 1,
7.59596, 4.747475, -135.0128, 1, 0.06666667, 0, 1,
7.636364, 4.747475, -135.1425, 1, 0.06666667, 0, 1,
7.676768, 4.747475, -135.2757, 1, 0.06666667, 0, 1,
7.717172, 4.747475, -135.4126, 1, 0.06666667, 0, 1,
7.757576, 4.747475, -135.5531, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.747475, -135.6972, 0.9647059, 0, 0.03137255, 1,
7.838384, 4.747475, -135.8449, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.747475, -135.9962, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.747475, -136.1512, 0.9647059, 0, 0.03137255, 1,
7.959596, 4.747475, -136.3098, 0.9647059, 0, 0.03137255, 1,
8, 4.747475, -136.472, 0.9647059, 0, 0.03137255, 1,
4, 4.787879, -138.1669, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.787879, -137.9774, 0.8588235, 0, 0.1372549, 1,
4.080808, 4.787879, -137.7915, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.787879, -137.6092, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.787879, -137.4304, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.787879, -137.2552, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.787879, -137.0836, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.787879, -136.9155, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.787879, -136.7509, 0.9647059, 0, 0.03137255, 1,
4.363636, 4.787879, -136.59, 0.9647059, 0, 0.03137255, 1,
4.40404, 4.787879, -136.4325, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.787879, -136.2787, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.787879, -136.1284, 0.9647059, 0, 0.03137255, 1,
4.525252, 4.787879, -135.9817, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.787879, -135.8385, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.787879, -135.6989, 0.9647059, 0, 0.03137255, 1,
4.646465, 4.787879, -135.5628, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.787879, -135.4303, 1, 0.06666667, 0, 1,
4.727273, 4.787879, -135.3014, 1, 0.06666667, 0, 1,
4.767677, 4.787879, -135.176, 1, 0.06666667, 0, 1,
4.808081, 4.787879, -135.0542, 1, 0.06666667, 0, 1,
4.848485, 4.787879, -134.936, 1, 0.06666667, 0, 1,
4.888889, 4.787879, -134.8213, 1, 0.06666667, 0, 1,
4.929293, 4.787879, -134.7102, 1, 0.06666667, 0, 1,
4.969697, 4.787879, -134.6026, 1, 0.06666667, 0, 1,
5.010101, 4.787879, -134.4986, 1, 0.06666667, 0, 1,
5.050505, 4.787879, -134.3982, 1, 0.06666667, 0, 1,
5.090909, 4.787879, -134.3013, 1, 0.06666667, 0, 1,
5.131313, 4.787879, -134.208, 1, 0.1686275, 0, 1,
5.171717, 4.787879, -134.1182, 1, 0.1686275, 0, 1,
5.212121, 4.787879, -134.032, 1, 0.1686275, 0, 1,
5.252525, 4.787879, -133.9494, 1, 0.1686275, 0, 1,
5.292929, 4.787879, -133.8703, 1, 0.1686275, 0, 1,
5.333333, 4.787879, -133.7948, 1, 0.1686275, 0, 1,
5.373737, 4.787879, -133.7228, 1, 0.1686275, 0, 1,
5.414141, 4.787879, -133.6544, 1, 0.1686275, 0, 1,
5.454545, 4.787879, -133.5896, 1, 0.1686275, 0, 1,
5.494949, 4.787879, -133.5283, 1, 0.1686275, 0, 1,
5.535354, 4.787879, -133.4706, 1, 0.1686275, 0, 1,
5.575758, 4.787879, -133.4164, 1, 0.1686275, 0, 1,
5.616162, 4.787879, -133.3658, 1, 0.1686275, 0, 1,
5.656566, 4.787879, -133.3188, 1, 0.1686275, 0, 1,
5.69697, 4.787879, -133.2753, 1, 0.1686275, 0, 1,
5.737374, 4.787879, -133.2354, 1, 0.1686275, 0, 1,
5.777778, 4.787879, -133.1991, 1, 0.1686275, 0, 1,
5.818182, 4.787879, -133.1663, 1, 0.1686275, 0, 1,
5.858586, 4.787879, -133.1371, 1, 0.1686275, 0, 1,
5.89899, 4.787879, -133.1114, 1, 0.1686275, 0, 1,
5.939394, 4.787879, -133.0893, 1, 0.1686275, 0, 1,
5.979798, 4.787879, -133.0707, 1, 0.1686275, 0, 1,
6.020202, 4.787879, -133.0557, 1, 0.1686275, 0, 1,
6.060606, 4.787879, -133.0443, 1, 0.1686275, 0, 1,
6.10101, 4.787879, -133.0365, 1, 0.1686275, 0, 1,
6.141414, 4.787879, -133.0322, 1, 0.1686275, 0, 1,
6.181818, 4.787879, -133.0314, 1, 0.1686275, 0, 1,
6.222222, 4.787879, -133.0342, 1, 0.1686275, 0, 1,
6.262626, 4.787879, -133.0406, 1, 0.1686275, 0, 1,
6.30303, 4.787879, -133.0505, 1, 0.1686275, 0, 1,
6.343434, 4.787879, -133.064, 1, 0.1686275, 0, 1,
6.383838, 4.787879, -133.0811, 1, 0.1686275, 0, 1,
6.424242, 4.787879, -133.1017, 1, 0.1686275, 0, 1,
6.464646, 4.787879, -133.1259, 1, 0.1686275, 0, 1,
6.505051, 4.787879, -133.1536, 1, 0.1686275, 0, 1,
6.545455, 4.787879, -133.1849, 1, 0.1686275, 0, 1,
6.585859, 4.787879, -133.2198, 1, 0.1686275, 0, 1,
6.626263, 4.787879, -133.2582, 1, 0.1686275, 0, 1,
6.666667, 4.787879, -133.3002, 1, 0.1686275, 0, 1,
6.707071, 4.787879, -133.3458, 1, 0.1686275, 0, 1,
6.747475, 4.787879, -133.3949, 1, 0.1686275, 0, 1,
6.787879, 4.787879, -133.4475, 1, 0.1686275, 0, 1,
6.828283, 4.787879, -133.5038, 1, 0.1686275, 0, 1,
6.868687, 4.787879, -133.5635, 1, 0.1686275, 0, 1,
6.909091, 4.787879, -133.6269, 1, 0.1686275, 0, 1,
6.949495, 4.787879, -133.6938, 1, 0.1686275, 0, 1,
6.989899, 4.787879, -133.7643, 1, 0.1686275, 0, 1,
7.030303, 4.787879, -133.8383, 1, 0.1686275, 0, 1,
7.070707, 4.787879, -133.9159, 1, 0.1686275, 0, 1,
7.111111, 4.787879, -133.997, 1, 0.1686275, 0, 1,
7.151515, 4.787879, -134.0818, 1, 0.1686275, 0, 1,
7.191919, 4.787879, -134.17, 1, 0.1686275, 0, 1,
7.232323, 4.787879, -134.2619, 1, 0.1686275, 0, 1,
7.272727, 4.787879, -134.3573, 1, 0.06666667, 0, 1,
7.313131, 4.787879, -134.4562, 1, 0.06666667, 0, 1,
7.353535, 4.787879, -134.5587, 1, 0.06666667, 0, 1,
7.393939, 4.787879, -134.6648, 1, 0.06666667, 0, 1,
7.434343, 4.787879, -134.7744, 1, 0.06666667, 0, 1,
7.474748, 4.787879, -134.8876, 1, 0.06666667, 0, 1,
7.515152, 4.787879, -135.0044, 1, 0.06666667, 0, 1,
7.555555, 4.787879, -135.1247, 1, 0.06666667, 0, 1,
7.59596, 4.787879, -135.2486, 1, 0.06666667, 0, 1,
7.636364, 4.787879, -135.376, 1, 0.06666667, 0, 1,
7.676768, 4.787879, -135.507, 1, 0.06666667, 0, 1,
7.717172, 4.787879, -135.6416, 0.9647059, 0, 0.03137255, 1,
7.757576, 4.787879, -135.7797, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.787879, -135.9214, 0.9647059, 0, 0.03137255, 1,
7.838384, 4.787879, -136.0667, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.787879, -136.2155, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.787879, -136.3678, 0.9647059, 0, 0.03137255, 1,
7.959596, 4.787879, -136.5238, 0.9647059, 0, 0.03137255, 1,
8, 4.787879, -136.6832, 0.9647059, 0, 0.03137255, 1,
4, 4.828283, -138.3552, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.828283, -138.1689, 0.7568628, 0, 0.2392157, 1,
4.080808, 4.828283, -137.9861, 0.8588235, 0, 0.1372549, 1,
4.121212, 4.828283, -137.8068, 0.8588235, 0, 0.1372549, 1,
4.161616, 4.828283, -137.631, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.828283, -137.4587, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.828283, -137.2899, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.828283, -137.1246, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.828283, -136.9628, 0.8588235, 0, 0.1372549, 1,
4.363636, 4.828283, -136.8045, 0.8588235, 0, 0.1372549, 1,
4.40404, 4.828283, -136.6497, 0.9647059, 0, 0.03137255, 1,
4.444445, 4.828283, -136.4984, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.828283, -136.3506, 0.9647059, 0, 0.03137255, 1,
4.525252, 4.828283, -136.2063, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.828283, -136.0656, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.828283, -135.9283, 0.9647059, 0, 0.03137255, 1,
4.646465, 4.828283, -135.7945, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.828283, -135.6642, 0.9647059, 0, 0.03137255, 1,
4.727273, 4.828283, -135.5374, 0.9647059, 0, 0.03137255, 1,
4.767677, 4.828283, -135.4141, 1, 0.06666667, 0, 1,
4.808081, 4.828283, -135.2944, 1, 0.06666667, 0, 1,
4.848485, 4.828283, -135.1781, 1, 0.06666667, 0, 1,
4.888889, 4.828283, -135.0653, 1, 0.06666667, 0, 1,
4.929293, 4.828283, -134.956, 1, 0.06666667, 0, 1,
4.969697, 4.828283, -134.8503, 1, 0.06666667, 0, 1,
5.010101, 4.828283, -134.748, 1, 0.06666667, 0, 1,
5.050505, 4.828283, -134.6492, 1, 0.06666667, 0, 1,
5.090909, 4.828283, -134.554, 1, 0.06666667, 0, 1,
5.131313, 4.828283, -134.4622, 1, 0.06666667, 0, 1,
5.171717, 4.828283, -134.3739, 1, 0.06666667, 0, 1,
5.212121, 4.828283, -134.2892, 1, 0.06666667, 0, 1,
5.252525, 4.828283, -134.2079, 1, 0.1686275, 0, 1,
5.292929, 4.828283, -134.1301, 1, 0.1686275, 0, 1,
5.333333, 4.828283, -134.0559, 1, 0.1686275, 0, 1,
5.373737, 4.828283, -133.9851, 1, 0.1686275, 0, 1,
5.414141, 4.828283, -133.9179, 1, 0.1686275, 0, 1,
5.454545, 4.828283, -133.8541, 1, 0.1686275, 0, 1,
5.494949, 4.828283, -133.7939, 1, 0.1686275, 0, 1,
5.535354, 4.828283, -133.7371, 1, 0.1686275, 0, 1,
5.575758, 4.828283, -133.6839, 1, 0.1686275, 0, 1,
5.616162, 4.828283, -133.6341, 1, 0.1686275, 0, 1,
5.656566, 4.828283, -133.5879, 1, 0.1686275, 0, 1,
5.69697, 4.828283, -133.5451, 1, 0.1686275, 0, 1,
5.737374, 4.828283, -133.5059, 1, 0.1686275, 0, 1,
5.777778, 4.828283, -133.4701, 1, 0.1686275, 0, 1,
5.818182, 4.828283, -133.4379, 1, 0.1686275, 0, 1,
5.858586, 4.828283, -133.4091, 1, 0.1686275, 0, 1,
5.89899, 4.828283, -133.3839, 1, 0.1686275, 0, 1,
5.939394, 4.828283, -133.3622, 1, 0.1686275, 0, 1,
5.979798, 4.828283, -133.3439, 1, 0.1686275, 0, 1,
6.020202, 4.828283, -133.3292, 1, 0.1686275, 0, 1,
6.060606, 4.828283, -133.3179, 1, 0.1686275, 0, 1,
6.10101, 4.828283, -133.3102, 1, 0.1686275, 0, 1,
6.141414, 4.828283, -133.306, 1, 0.1686275, 0, 1,
6.181818, 4.828283, -133.3052, 1, 0.1686275, 0, 1,
6.222222, 4.828283, -133.308, 1, 0.1686275, 0, 1,
6.262626, 4.828283, -133.3143, 1, 0.1686275, 0, 1,
6.30303, 4.828283, -133.3241, 1, 0.1686275, 0, 1,
6.343434, 4.828283, -133.3373, 1, 0.1686275, 0, 1,
6.383838, 4.828283, -133.3541, 1, 0.1686275, 0, 1,
6.424242, 4.828283, -133.3744, 1, 0.1686275, 0, 1,
6.464646, 4.828283, -133.3982, 1, 0.1686275, 0, 1,
6.505051, 4.828283, -133.4254, 1, 0.1686275, 0, 1,
6.545455, 4.828283, -133.4562, 1, 0.1686275, 0, 1,
6.585859, 4.828283, -133.4905, 1, 0.1686275, 0, 1,
6.626263, 4.828283, -133.5283, 1, 0.1686275, 0, 1,
6.666667, 4.828283, -133.5696, 1, 0.1686275, 0, 1,
6.707071, 4.828283, -133.6143, 1, 0.1686275, 0, 1,
6.747475, 4.828283, -133.6626, 1, 0.1686275, 0, 1,
6.787879, 4.828283, -133.7144, 1, 0.1686275, 0, 1,
6.828283, 4.828283, -133.7697, 1, 0.1686275, 0, 1,
6.868687, 4.828283, -133.8285, 1, 0.1686275, 0, 1,
6.909091, 4.828283, -133.8908, 1, 0.1686275, 0, 1,
6.949495, 4.828283, -133.9566, 1, 0.1686275, 0, 1,
6.989899, 4.828283, -134.0259, 1, 0.1686275, 0, 1,
7.030303, 4.828283, -134.0987, 1, 0.1686275, 0, 1,
7.070707, 4.828283, -134.175, 1, 0.1686275, 0, 1,
7.111111, 4.828283, -134.2548, 1, 0.1686275, 0, 1,
7.151515, 4.828283, -134.3381, 1, 0.06666667, 0, 1,
7.191919, 4.828283, -134.4249, 1, 0.06666667, 0, 1,
7.232323, 4.828283, -134.5152, 1, 0.06666667, 0, 1,
7.272727, 4.828283, -134.609, 1, 0.06666667, 0, 1,
7.313131, 4.828283, -134.7063, 1, 0.06666667, 0, 1,
7.353535, 4.828283, -134.8071, 1, 0.06666667, 0, 1,
7.393939, 4.828283, -134.9114, 1, 0.06666667, 0, 1,
7.434343, 4.828283, -135.0192, 1, 0.06666667, 0, 1,
7.474748, 4.828283, -135.1305, 1, 0.06666667, 0, 1,
7.515152, 4.828283, -135.2454, 1, 0.06666667, 0, 1,
7.555555, 4.828283, -135.3637, 1, 0.06666667, 0, 1,
7.59596, 4.828283, -135.4855, 1, 0.06666667, 0, 1,
7.636364, 4.828283, -135.6108, 0.9647059, 0, 0.03137255, 1,
7.676768, 4.828283, -135.7396, 0.9647059, 0, 0.03137255, 1,
7.717172, 4.828283, -135.8719, 0.9647059, 0, 0.03137255, 1,
7.757576, 4.828283, -136.0078, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.828283, -136.1471, 0.9647059, 0, 0.03137255, 1,
7.838384, 4.828283, -136.2899, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.828283, -136.4362, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.828283, -136.5861, 0.9647059, 0, 0.03137255, 1,
7.959596, 4.828283, -136.7394, 0.9647059, 0, 0.03137255, 1,
8, 4.828283, -136.8962, 0.8588235, 0, 0.1372549, 1,
4, 4.868687, -138.5457, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.868687, -138.3624, 0.7568628, 0, 0.2392157, 1,
4.080808, 4.868687, -138.1826, 0.7568628, 0, 0.2392157, 1,
4.121212, 4.868687, -138.0063, 0.7568628, 0, 0.2392157, 1,
4.161616, 4.868687, -137.8334, 0.8588235, 0, 0.1372549, 1,
4.20202, 4.868687, -137.664, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.868687, -137.498, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.868687, -137.3354, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.868687, -137.1763, 0.8588235, 0, 0.1372549, 1,
4.363636, 4.868687, -137.0206, 0.8588235, 0, 0.1372549, 1,
4.40404, 4.868687, -136.8684, 0.8588235, 0, 0.1372549, 1,
4.444445, 4.868687, -136.7196, 0.9647059, 0, 0.03137255, 1,
4.484848, 4.868687, -136.5742, 0.9647059, 0, 0.03137255, 1,
4.525252, 4.868687, -136.4323, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.868687, -136.2939, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.868687, -136.1589, 0.9647059, 0, 0.03137255, 1,
4.646465, 4.868687, -136.0273, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.868687, -135.8992, 0.9647059, 0, 0.03137255, 1,
4.727273, 4.868687, -135.7745, 0.9647059, 0, 0.03137255, 1,
4.767677, 4.868687, -135.6532, 0.9647059, 0, 0.03137255, 1,
4.808081, 4.868687, -135.5354, 0.9647059, 0, 0.03137255, 1,
4.848485, 4.868687, -135.4211, 1, 0.06666667, 0, 1,
4.888889, 4.868687, -135.3102, 1, 0.06666667, 0, 1,
4.929293, 4.868687, -135.2027, 1, 0.06666667, 0, 1,
4.969697, 4.868687, -135.0987, 1, 0.06666667, 0, 1,
5.010101, 4.868687, -134.9981, 1, 0.06666667, 0, 1,
5.050505, 4.868687, -134.901, 1, 0.06666667, 0, 1,
5.090909, 4.868687, -134.8073, 1, 0.06666667, 0, 1,
5.131313, 4.868687, -134.717, 1, 0.06666667, 0, 1,
5.171717, 4.868687, -134.6302, 1, 0.06666667, 0, 1,
5.212121, 4.868687, -134.5469, 1, 0.06666667, 0, 1,
5.252525, 4.868687, -134.4669, 1, 0.06666667, 0, 1,
5.292929, 4.868687, -134.3905, 1, 0.06666667, 0, 1,
5.333333, 4.868687, -134.3174, 1, 0.06666667, 0, 1,
5.373737, 4.868687, -134.2478, 1, 0.1686275, 0, 1,
5.414141, 4.868687, -134.1817, 1, 0.1686275, 0, 1,
5.454545, 4.868687, -134.119, 1, 0.1686275, 0, 1,
5.494949, 4.868687, -134.0598, 1, 0.1686275, 0, 1,
5.535354, 4.868687, -134.0039, 1, 0.1686275, 0, 1,
5.575758, 4.868687, -133.9516, 1, 0.1686275, 0, 1,
5.616162, 4.868687, -133.9026, 1, 0.1686275, 0, 1,
5.656566, 4.868687, -133.8571, 1, 0.1686275, 0, 1,
5.69697, 4.868687, -133.8151, 1, 0.1686275, 0, 1,
5.737374, 4.868687, -133.7765, 1, 0.1686275, 0, 1,
5.777778, 4.868687, -133.7414, 1, 0.1686275, 0, 1,
5.818182, 4.868687, -133.7096, 1, 0.1686275, 0, 1,
5.858586, 4.868687, -133.6814, 1, 0.1686275, 0, 1,
5.89899, 4.868687, -133.6566, 1, 0.1686275, 0, 1,
5.939394, 4.868687, -133.6352, 1, 0.1686275, 0, 1,
5.979798, 4.868687, -133.6172, 1, 0.1686275, 0, 1,
6.020202, 4.868687, -133.6028, 1, 0.1686275, 0, 1,
6.060606, 4.868687, -133.5917, 1, 0.1686275, 0, 1,
6.10101, 4.868687, -133.5841, 1, 0.1686275, 0, 1,
6.141414, 4.868687, -133.5799, 1, 0.1686275, 0, 1,
6.181818, 4.868687, -133.5792, 1, 0.1686275, 0, 1,
6.222222, 4.868687, -133.5819, 1, 0.1686275, 0, 1,
6.262626, 4.868687, -133.5881, 1, 0.1686275, 0, 1,
6.30303, 4.868687, -133.5977, 1, 0.1686275, 0, 1,
6.343434, 4.868687, -133.6108, 1, 0.1686275, 0, 1,
6.383838, 4.868687, -133.6273, 1, 0.1686275, 0, 1,
6.424242, 4.868687, -133.6472, 1, 0.1686275, 0, 1,
6.464646, 4.868687, -133.6706, 1, 0.1686275, 0, 1,
6.505051, 4.868687, -133.6974, 1, 0.1686275, 0, 1,
6.545455, 4.868687, -133.7277, 1, 0.1686275, 0, 1,
6.585859, 4.868687, -133.7614, 1, 0.1686275, 0, 1,
6.626263, 4.868687, -133.7986, 1, 0.1686275, 0, 1,
6.666667, 4.868687, -133.8392, 1, 0.1686275, 0, 1,
6.707071, 4.868687, -133.8832, 1, 0.1686275, 0, 1,
6.747475, 4.868687, -133.9307, 1, 0.1686275, 0, 1,
6.787879, 4.868687, -133.9816, 1, 0.1686275, 0, 1,
6.828283, 4.868687, -134.036, 1, 0.1686275, 0, 1,
6.868687, 4.868687, -134.0938, 1, 0.1686275, 0, 1,
6.909091, 4.868687, -134.1551, 1, 0.1686275, 0, 1,
6.949495, 4.868687, -134.2198, 1, 0.1686275, 0, 1,
6.989899, 4.868687, -134.2879, 1, 0.06666667, 0, 1,
7.030303, 4.868687, -134.3595, 1, 0.06666667, 0, 1,
7.070707, 4.868687, -134.4346, 1, 0.06666667, 0, 1,
7.111111, 4.868687, -134.5131, 1, 0.06666667, 0, 1,
7.151515, 4.868687, -134.595, 1, 0.06666667, 0, 1,
7.191919, 4.868687, -134.6803, 1, 0.06666667, 0, 1,
7.232323, 4.868687, -134.7692, 1, 0.06666667, 0, 1,
7.272727, 4.868687, -134.8614, 1, 0.06666667, 0, 1,
7.313131, 4.868687, -134.9571, 1, 0.06666667, 0, 1,
7.353535, 4.868687, -135.0563, 1, 0.06666667, 0, 1,
7.393939, 4.868687, -135.1588, 1, 0.06666667, 0, 1,
7.434343, 4.868687, -135.2649, 1, 0.06666667, 0, 1,
7.474748, 4.868687, -135.3743, 1, 0.06666667, 0, 1,
7.515152, 4.868687, -135.4873, 1, 0.06666667, 0, 1,
7.555555, 4.868687, -135.6036, 0.9647059, 0, 0.03137255, 1,
7.59596, 4.868687, -135.7234, 0.9647059, 0, 0.03137255, 1,
7.636364, 4.868687, -135.8467, 0.9647059, 0, 0.03137255, 1,
7.676768, 4.868687, -135.9733, 0.9647059, 0, 0.03137255, 1,
7.717172, 4.868687, -136.1035, 0.9647059, 0, 0.03137255, 1,
7.757576, 4.868687, -136.2371, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.868687, -136.3741, 0.9647059, 0, 0.03137255, 1,
7.838384, 4.868687, -136.5145, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.868687, -136.6584, 0.9647059, 0, 0.03137255, 1,
7.919192, 4.868687, -136.8058, 0.8588235, 0, 0.1372549, 1,
7.959596, 4.868687, -136.9566, 0.8588235, 0, 0.1372549, 1,
8, 4.868687, -137.1108, 0.8588235, 0, 0.1372549, 1,
4, 4.909091, -138.7383, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.909091, -138.558, 0.7568628, 0, 0.2392157, 1,
4.080808, 4.909091, -138.3812, 0.7568628, 0, 0.2392157, 1,
4.121212, 4.909091, -138.2078, 0.7568628, 0, 0.2392157, 1,
4.161616, 4.909091, -138.0377, 0.7568628, 0, 0.2392157, 1,
4.20202, 4.909091, -137.871, 0.8588235, 0, 0.1372549, 1,
4.242424, 4.909091, -137.7078, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.909091, -137.5479, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.909091, -137.3914, 0.8588235, 0, 0.1372549, 1,
4.363636, 4.909091, -137.2382, 0.8588235, 0, 0.1372549, 1,
4.40404, 4.909091, -137.0885, 0.8588235, 0, 0.1372549, 1,
4.444445, 4.909091, -136.9421, 0.8588235, 0, 0.1372549, 1,
4.484848, 4.909091, -136.7992, 0.8588235, 0, 0.1372549, 1,
4.525252, 4.909091, -136.6596, 0.9647059, 0, 0.03137255, 1,
4.565657, 4.909091, -136.5234, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.909091, -136.3906, 0.9647059, 0, 0.03137255, 1,
4.646465, 4.909091, -136.2612, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.909091, -136.1352, 0.9647059, 0, 0.03137255, 1,
4.727273, 4.909091, -136.0125, 0.9647059, 0, 0.03137255, 1,
4.767677, 4.909091, -135.8933, 0.9647059, 0, 0.03137255, 1,
4.808081, 4.909091, -135.7774, 0.9647059, 0, 0.03137255, 1,
4.848485, 4.909091, -135.6649, 0.9647059, 0, 0.03137255, 1,
4.888889, 4.909091, -135.5558, 0.9647059, 0, 0.03137255, 1,
4.929293, 4.909091, -135.4501, 1, 0.06666667, 0, 1,
4.969697, 4.909091, -135.3478, 1, 0.06666667, 0, 1,
5.010101, 4.909091, -135.2489, 1, 0.06666667, 0, 1,
5.050505, 4.909091, -135.1533, 1, 0.06666667, 0, 1,
5.090909, 4.909091, -135.0612, 1, 0.06666667, 0, 1,
5.131313, 4.909091, -134.9724, 1, 0.06666667, 0, 1,
5.171717, 4.909091, -134.887, 1, 0.06666667, 0, 1,
5.212121, 4.909091, -134.805, 1, 0.06666667, 0, 1,
5.252525, 4.909091, -134.7264, 1, 0.06666667, 0, 1,
5.292929, 4.909091, -134.6512, 1, 0.06666667, 0, 1,
5.333333, 4.909091, -134.5794, 1, 0.06666667, 0, 1,
5.373737, 4.909091, -134.5109, 1, 0.06666667, 0, 1,
5.414141, 4.909091, -134.4459, 1, 0.06666667, 0, 1,
5.454545, 4.909091, -134.3842, 1, 0.06666667, 0, 1,
5.494949, 4.909091, -134.3259, 1, 0.06666667, 0, 1,
5.535354, 4.909091, -134.271, 1, 0.06666667, 0, 1,
5.575758, 4.909091, -134.2195, 1, 0.1686275, 0, 1,
5.616162, 4.909091, -134.1714, 1, 0.1686275, 0, 1,
5.656566, 4.909091, -134.1266, 1, 0.1686275, 0, 1,
5.69697, 4.909091, -134.0853, 1, 0.1686275, 0, 1,
5.737374, 4.909091, -134.0473, 1, 0.1686275, 0, 1,
5.777778, 4.909091, -134.0127, 1, 0.1686275, 0, 1,
5.818182, 4.909091, -133.9815, 1, 0.1686275, 0, 1,
5.858586, 4.909091, -133.9537, 1, 0.1686275, 0, 1,
5.89899, 4.909091, -133.9293, 1, 0.1686275, 0, 1,
5.939394, 4.909091, -133.9083, 1, 0.1686275, 0, 1,
5.979798, 4.909091, -133.8906, 1, 0.1686275, 0, 1,
6.020202, 4.909091, -133.8764, 1, 0.1686275, 0, 1,
6.060606, 4.909091, -133.8655, 1, 0.1686275, 0, 1,
6.10101, 4.909091, -133.858, 1, 0.1686275, 0, 1,
6.141414, 4.909091, -133.8539, 1, 0.1686275, 0, 1,
6.181818, 4.909091, -133.8532, 1, 0.1686275, 0, 1,
6.222222, 4.909091, -133.8559, 1, 0.1686275, 0, 1,
6.262626, 4.909091, -133.862, 1, 0.1686275, 0, 1,
6.30303, 4.909091, -133.8714, 1, 0.1686275, 0, 1,
6.343434, 4.909091, -133.8843, 1, 0.1686275, 0, 1,
6.383838, 4.909091, -133.9005, 1, 0.1686275, 0, 1,
6.424242, 4.909091, -133.9201, 1, 0.1686275, 0, 1,
6.464646, 4.909091, -133.9431, 1, 0.1686275, 0, 1,
6.505051, 4.909091, -133.9695, 1, 0.1686275, 0, 1,
6.545455, 4.909091, -133.9993, 1, 0.1686275, 0, 1,
6.585859, 4.909091, -134.0324, 1, 0.1686275, 0, 1,
6.626263, 4.909091, -134.069, 1, 0.1686275, 0, 1,
6.666667, 4.909091, -134.1089, 1, 0.1686275, 0, 1,
6.707071, 4.909091, -134.1523, 1, 0.1686275, 0, 1,
6.747475, 4.909091, -134.199, 1, 0.1686275, 0, 1,
6.787879, 4.909091, -134.2491, 1, 0.1686275, 0, 1,
6.828283, 4.909091, -134.3026, 1, 0.06666667, 0, 1,
6.868687, 4.909091, -134.3594, 1, 0.06666667, 0, 1,
6.909091, 4.909091, -134.4197, 1, 0.06666667, 0, 1,
6.949495, 4.909091, -134.4833, 1, 0.06666667, 0, 1,
6.989899, 4.909091, -134.5504, 1, 0.06666667, 0, 1,
7.030303, 4.909091, -134.6208, 1, 0.06666667, 0, 1,
7.070707, 4.909091, -134.6946, 1, 0.06666667, 0, 1,
7.111111, 4.909091, -134.7718, 1, 0.06666667, 0, 1,
7.151515, 4.909091, -134.8524, 1, 0.06666667, 0, 1,
7.191919, 4.909091, -134.9363, 1, 0.06666667, 0, 1,
7.232323, 4.909091, -135.0237, 1, 0.06666667, 0, 1,
7.272727, 4.909091, -135.1144, 1, 0.06666667, 0, 1,
7.313131, 4.909091, -135.2086, 1, 0.06666667, 0, 1,
7.353535, 4.909091, -135.3061, 1, 0.06666667, 0, 1,
7.393939, 4.909091, -135.407, 1, 0.06666667, 0, 1,
7.434343, 4.909091, -135.5113, 1, 0.06666667, 0, 1,
7.474748, 4.909091, -135.6189, 0.9647059, 0, 0.03137255, 1,
7.515152, 4.909091, -135.73, 0.9647059, 0, 0.03137255, 1,
7.555555, 4.909091, -135.8445, 0.9647059, 0, 0.03137255, 1,
7.59596, 4.909091, -135.9623, 0.9647059, 0, 0.03137255, 1,
7.636364, 4.909091, -136.0835, 0.9647059, 0, 0.03137255, 1,
7.676768, 4.909091, -136.2081, 0.9647059, 0, 0.03137255, 1,
7.717172, 4.909091, -136.3361, 0.9647059, 0, 0.03137255, 1,
7.757576, 4.909091, -136.4675, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.909091, -136.6023, 0.9647059, 0, 0.03137255, 1,
7.838384, 4.909091, -136.7404, 0.9647059, 0, 0.03137255, 1,
7.878788, 4.909091, -136.882, 0.8588235, 0, 0.1372549, 1,
7.919192, 4.909091, -137.0269, 0.8588235, 0, 0.1372549, 1,
7.959596, 4.909091, -137.1752, 0.8588235, 0, 0.1372549, 1,
8, 4.909091, -137.327, 0.8588235, 0, 0.1372549, 1,
4, 4.949495, -138.9329, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.949495, -138.7556, 0.7568628, 0, 0.2392157, 1,
4.080808, 4.949495, -138.5816, 0.7568628, 0, 0.2392157, 1,
4.121212, 4.949495, -138.411, 0.7568628, 0, 0.2392157, 1,
4.161616, 4.949495, -138.2437, 0.7568628, 0, 0.2392157, 1,
4.20202, 4.949495, -138.0798, 0.7568628, 0, 0.2392157, 1,
4.242424, 4.949495, -137.9191, 0.8588235, 0, 0.1372549, 1,
4.282828, 4.949495, -137.7618, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.949495, -137.6079, 0.8588235, 0, 0.1372549, 1,
4.363636, 4.949495, -137.4572, 0.8588235, 0, 0.1372549, 1,
4.40404, 4.949495, -137.3099, 0.8588235, 0, 0.1372549, 1,
4.444445, 4.949495, -137.166, 0.8588235, 0, 0.1372549, 1,
4.484848, 4.949495, -137.0253, 0.8588235, 0, 0.1372549, 1,
4.525252, 4.949495, -136.888, 0.8588235, 0, 0.1372549, 1,
4.565657, 4.949495, -136.754, 0.9647059, 0, 0.03137255, 1,
4.606061, 4.949495, -136.6234, 0.9647059, 0, 0.03137255, 1,
4.646465, 4.949495, -136.4961, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.949495, -136.3721, 0.9647059, 0, 0.03137255, 1,
4.727273, 4.949495, -136.2515, 0.9647059, 0, 0.03137255, 1,
4.767677, 4.949495, -136.1341, 0.9647059, 0, 0.03137255, 1,
4.808081, 4.949495, -136.0202, 0.9647059, 0, 0.03137255, 1,
4.848485, 4.949495, -135.9095, 0.9647059, 0, 0.03137255, 1,
4.888889, 4.949495, -135.8022, 0.9647059, 0, 0.03137255, 1,
4.929293, 4.949495, -135.6982, 0.9647059, 0, 0.03137255, 1,
4.969697, 4.949495, -135.5975, 0.9647059, 0, 0.03137255, 1,
5.010101, 4.949495, -135.5002, 1, 0.06666667, 0, 1,
5.050505, 4.949495, -135.4062, 1, 0.06666667, 0, 1,
5.090909, 4.949495, -135.3156, 1, 0.06666667, 0, 1,
5.131313, 4.949495, -135.2283, 1, 0.06666667, 0, 1,
5.171717, 4.949495, -135.1443, 1, 0.06666667, 0, 1,
5.212121, 4.949495, -135.0636, 1, 0.06666667, 0, 1,
5.252525, 4.949495, -134.9863, 1, 0.06666667, 0, 1,
5.292929, 4.949495, -134.9123, 1, 0.06666667, 0, 1,
5.333333, 4.949495, -134.8416, 1, 0.06666667, 0, 1,
5.373737, 4.949495, -134.7743, 1, 0.06666667, 0, 1,
5.414141, 4.949495, -134.7103, 1, 0.06666667, 0, 1,
5.454545, 4.949495, -134.6496, 1, 0.06666667, 0, 1,
5.494949, 4.949495, -134.5923, 1, 0.06666667, 0, 1,
5.535354, 4.949495, -134.5383, 1, 0.06666667, 0, 1,
5.575758, 4.949495, -134.4876, 1, 0.06666667, 0, 1,
5.616162, 4.949495, -134.4402, 1, 0.06666667, 0, 1,
5.656566, 4.949495, -134.3962, 1, 0.06666667, 0, 1,
5.69697, 4.949495, -134.3555, 1, 0.06666667, 0, 1,
5.737374, 4.949495, -134.3182, 1, 0.06666667, 0, 1,
5.777778, 4.949495, -134.2842, 1, 0.06666667, 0, 1,
5.818182, 4.949495, -134.2535, 1, 0.1686275, 0, 1,
5.858586, 4.949495, -134.2261, 1, 0.1686275, 0, 1,
5.89899, 4.949495, -134.2021, 1, 0.1686275, 0, 1,
5.939394, 4.949495, -134.1814, 1, 0.1686275, 0, 1,
5.979798, 4.949495, -134.1641, 1, 0.1686275, 0, 1,
6.020202, 4.949495, -134.1501, 1, 0.1686275, 0, 1,
6.060606, 4.949495, -134.1394, 1, 0.1686275, 0, 1,
6.10101, 4.949495, -134.132, 1, 0.1686275, 0, 1,
6.141414, 4.949495, -134.128, 1, 0.1686275, 0, 1,
6.181818, 4.949495, -134.1273, 1, 0.1686275, 0, 1,
6.222222, 4.949495, -134.1299, 1, 0.1686275, 0, 1,
6.262626, 4.949495, -134.1359, 1, 0.1686275, 0, 1,
6.30303, 4.949495, -134.1452, 1, 0.1686275, 0, 1,
6.343434, 4.949495, -134.1578, 1, 0.1686275, 0, 1,
6.383838, 4.949495, -134.1738, 1, 0.1686275, 0, 1,
6.424242, 4.949495, -134.1931, 1, 0.1686275, 0, 1,
6.464646, 4.949495, -134.2157, 1, 0.1686275, 0, 1,
6.505051, 4.949495, -134.2417, 1, 0.1686275, 0, 1,
6.545455, 4.949495, -134.271, 1, 0.06666667, 0, 1,
6.585859, 4.949495, -134.3036, 1, 0.06666667, 0, 1,
6.626263, 4.949495, -134.3395, 1, 0.06666667, 0, 1,
6.666667, 4.949495, -134.3788, 1, 0.06666667, 0, 1,
6.707071, 4.949495, -134.4214, 1, 0.06666667, 0, 1,
6.747475, 4.949495, -134.4674, 1, 0.06666667, 0, 1,
6.787879, 4.949495, -134.5167, 1, 0.06666667, 0, 1,
6.828283, 4.949495, -134.5693, 1, 0.06666667, 0, 1,
6.868687, 4.949495, -134.6252, 1, 0.06666667, 0, 1,
6.909091, 4.949495, -134.6845, 1, 0.06666667, 0, 1,
6.949495, 4.949495, -134.7471, 1, 0.06666667, 0, 1,
6.989899, 4.949495, -134.8131, 1, 0.06666667, 0, 1,
7.030303, 4.949495, -134.8823, 1, 0.06666667, 0, 1,
7.070707, 4.949495, -134.9549, 1, 0.06666667, 0, 1,
7.111111, 4.949495, -135.0309, 1, 0.06666667, 0, 1,
7.151515, 4.949495, -135.1102, 1, 0.06666667, 0, 1,
7.191919, 4.949495, -135.1927, 1, 0.06666667, 0, 1,
7.232323, 4.949495, -135.2787, 1, 0.06666667, 0, 1,
7.272727, 4.949495, -135.368, 1, 0.06666667, 0, 1,
7.313131, 4.949495, -135.4606, 1, 0.06666667, 0, 1,
7.353535, 4.949495, -135.5565, 0.9647059, 0, 0.03137255, 1,
7.393939, 4.949495, -135.6557, 0.9647059, 0, 0.03137255, 1,
7.434343, 4.949495, -135.7583, 0.9647059, 0, 0.03137255, 1,
7.474748, 4.949495, -135.8643, 0.9647059, 0, 0.03137255, 1,
7.515152, 4.949495, -135.9735, 0.9647059, 0, 0.03137255, 1,
7.555555, 4.949495, -136.0861, 0.9647059, 0, 0.03137255, 1,
7.59596, 4.949495, -136.202, 0.9647059, 0, 0.03137255, 1,
7.636364, 4.949495, -136.3213, 0.9647059, 0, 0.03137255, 1,
7.676768, 4.949495, -136.4439, 0.9647059, 0, 0.03137255, 1,
7.717172, 4.949495, -136.5698, 0.9647059, 0, 0.03137255, 1,
7.757576, 4.949495, -136.6991, 0.9647059, 0, 0.03137255, 1,
7.79798, 4.949495, -136.8316, 0.8588235, 0, 0.1372549, 1,
7.838384, 4.949495, -136.9675, 0.8588235, 0, 0.1372549, 1,
7.878788, 4.949495, -137.1068, 0.8588235, 0, 0.1372549, 1,
7.919192, 4.949495, -137.2494, 0.8588235, 0, 0.1372549, 1,
7.959596, 4.949495, -137.3953, 0.8588235, 0, 0.1372549, 1,
8, 4.949495, -137.5445, 0.8588235, 0, 0.1372549, 1,
4, 4.989899, -139.1294, 0.7568628, 0, 0.2392157, 1,
4.040404, 4.989899, -138.9549, 0.7568628, 0, 0.2392157, 1,
4.080808, 4.989899, -138.7838, 0.7568628, 0, 0.2392157, 1,
4.121212, 4.989899, -138.6159, 0.7568628, 0, 0.2392157, 1,
4.161616, 4.989899, -138.4513, 0.7568628, 0, 0.2392157, 1,
4.20202, 4.989899, -138.29, 0.7568628, 0, 0.2392157, 1,
4.242424, 4.989899, -138.132, 0.7568628, 0, 0.2392157, 1,
4.282828, 4.989899, -137.9772, 0.8588235, 0, 0.1372549, 1,
4.323232, 4.989899, -137.8257, 0.8588235, 0, 0.1372549, 1,
4.363636, 4.989899, -137.6775, 0.8588235, 0, 0.1372549, 1,
4.40404, 4.989899, -137.5326, 0.8588235, 0, 0.1372549, 1,
4.444445, 4.989899, -137.3909, 0.8588235, 0, 0.1372549, 1,
4.484848, 4.989899, -137.2526, 0.8588235, 0, 0.1372549, 1,
4.525252, 4.989899, -137.1175, 0.8588235, 0, 0.1372549, 1,
4.565657, 4.989899, -136.9857, 0.8588235, 0, 0.1372549, 1,
4.606061, 4.989899, -136.8571, 0.8588235, 0, 0.1372549, 1,
4.646465, 4.989899, -136.7319, 0.9647059, 0, 0.03137255, 1,
4.686869, 4.989899, -136.6099, 0.9647059, 0, 0.03137255, 1,
4.727273, 4.989899, -136.4912, 0.9647059, 0, 0.03137255, 1,
4.767677, 4.989899, -136.3758, 0.9647059, 0, 0.03137255, 1,
4.808081, 4.989899, -136.2636, 0.9647059, 0, 0.03137255, 1,
4.848485, 4.989899, -136.1548, 0.9647059, 0, 0.03137255, 1,
4.888889, 4.989899, -136.0492, 0.9647059, 0, 0.03137255, 1,
4.929293, 4.989899, -135.9469, 0.9647059, 0, 0.03137255, 1,
4.969697, 4.989899, -135.8478, 0.9647059, 0, 0.03137255, 1,
5.010101, 4.989899, -135.7521, 0.9647059, 0, 0.03137255, 1,
5.050505, 4.989899, -135.6596, 0.9647059, 0, 0.03137255, 1,
5.090909, 4.989899, -135.5704, 0.9647059, 0, 0.03137255, 1,
5.131313, 4.989899, -135.4845, 1, 0.06666667, 0, 1,
5.171717, 4.989899, -135.4019, 1, 0.06666667, 0, 1,
5.212121, 4.989899, -135.3225, 1, 0.06666667, 0, 1,
5.252525, 4.989899, -135.2464, 1, 0.06666667, 0, 1,
5.292929, 4.989899, -135.1736, 1, 0.06666667, 0, 1,
5.333333, 4.989899, -135.1041, 1, 0.06666667, 0, 1,
5.373737, 4.989899, -135.0378, 1, 0.06666667, 0, 1,
5.414141, 4.989899, -134.9749, 1, 0.06666667, 0, 1,
5.454545, 4.989899, -134.9152, 1, 0.06666667, 0, 1,
5.494949, 4.989899, -134.8587, 1, 0.06666667, 0, 1,
5.535354, 4.989899, -134.8056, 1, 0.06666667, 0, 1,
5.575758, 4.989899, -134.7558, 1, 0.06666667, 0, 1,
5.616162, 4.989899, -134.7092, 1, 0.06666667, 0, 1,
5.656566, 4.989899, -134.6659, 1, 0.06666667, 0, 1,
5.69697, 4.989899, -134.6258, 1, 0.06666667, 0, 1,
5.737374, 4.989899, -134.5891, 1, 0.06666667, 0, 1,
5.777778, 4.989899, -134.5556, 1, 0.06666667, 0, 1,
5.818182, 4.989899, -134.5255, 1, 0.06666667, 0, 1,
5.858586, 4.989899, -134.4985, 1, 0.06666667, 0, 1,
5.89899, 4.989899, -134.4749, 1, 0.06666667, 0, 1,
5.939394, 4.989899, -134.4545, 1, 0.06666667, 0, 1,
5.979798, 4.989899, -134.4375, 1, 0.06666667, 0, 1,
6.020202, 4.989899, -134.4237, 1, 0.06666667, 0, 1,
6.060606, 4.989899, -134.4132, 1, 0.06666667, 0, 1,
6.10101, 4.989899, -134.4059, 1, 0.06666667, 0, 1,
6.141414, 4.989899, -134.4019, 1, 0.06666667, 0, 1,
6.181818, 4.989899, -134.4013, 1, 0.06666667, 0, 1,
6.222222, 4.989899, -134.4039, 1, 0.06666667, 0, 1,
6.262626, 4.989899, -134.4097, 1, 0.06666667, 0, 1,
6.30303, 4.989899, -134.4189, 1, 0.06666667, 0, 1,
6.343434, 4.989899, -134.4313, 1, 0.06666667, 0, 1,
6.383838, 4.989899, -134.447, 1, 0.06666667, 0, 1,
6.424242, 4.989899, -134.466, 1, 0.06666667, 0, 1,
6.464646, 4.989899, -134.4883, 1, 0.06666667, 0, 1,
6.505051, 4.989899, -134.5138, 1, 0.06666667, 0, 1,
6.545455, 4.989899, -134.5426, 1, 0.06666667, 0, 1,
6.585859, 4.989899, -134.5747, 1, 0.06666667, 0, 1,
6.626263, 4.989899, -134.6101, 1, 0.06666667, 0, 1,
6.666667, 4.989899, -134.6487, 1, 0.06666667, 0, 1,
6.707071, 4.989899, -134.6907, 1, 0.06666667, 0, 1,
6.747475, 4.989899, -134.7359, 1, 0.06666667, 0, 1,
6.787879, 4.989899, -134.7844, 1, 0.06666667, 0, 1,
6.828283, 4.989899, -134.8362, 1, 0.06666667, 0, 1,
6.868687, 4.989899, -134.8912, 1, 0.06666667, 0, 1,
6.909091, 4.989899, -134.9495, 1, 0.06666667, 0, 1,
6.949495, 4.989899, -135.0111, 1, 0.06666667, 0, 1,
6.989899, 4.989899, -135.076, 1, 0.06666667, 0, 1,
7.030303, 4.989899, -135.1441, 1, 0.06666667, 0, 1,
7.070707, 4.989899, -135.2156, 1, 0.06666667, 0, 1,
7.111111, 4.989899, -135.2903, 1, 0.06666667, 0, 1,
7.151515, 4.989899, -135.3683, 1, 0.06666667, 0, 1,
7.191919, 4.989899, -135.4496, 1, 0.06666667, 0, 1,
7.232323, 4.989899, -135.5341, 0.9647059, 0, 0.03137255, 1,
7.272727, 4.989899, -135.6219, 0.9647059, 0, 0.03137255, 1,
7.313131, 4.989899, -135.713, 0.9647059, 0, 0.03137255, 1,
7.353535, 4.989899, -135.8074, 0.9647059, 0, 0.03137255, 1,
7.393939, 4.989899, -135.9051, 0.9647059, 0, 0.03137255, 1,
7.434343, 4.989899, -136.006, 0.9647059, 0, 0.03137255, 1,
7.474748, 4.989899, -136.1102, 0.9647059, 0, 0.03137255, 1,
7.515152, 4.989899, -136.2177, 0.9647059, 0, 0.03137255, 1,
7.555555, 4.989899, -136.3285, 0.9647059, 0, 0.03137255, 1,
7.59596, 4.989899, -136.4426, 0.9647059, 0, 0.03137255, 1,
7.636364, 4.989899, -136.5599, 0.9647059, 0, 0.03137255, 1,
7.676768, 4.989899, -136.6805, 0.9647059, 0, 0.03137255, 1,
7.717172, 4.989899, -136.8044, 0.8588235, 0, 0.1372549, 1,
7.757576, 4.989899, -136.9316, 0.8588235, 0, 0.1372549, 1,
7.79798, 4.989899, -137.062, 0.8588235, 0, 0.1372549, 1,
7.838384, 4.989899, -137.1957, 0.8588235, 0, 0.1372549, 1,
7.878788, 4.989899, -137.3327, 0.8588235, 0, 0.1372549, 1,
7.919192, 4.989899, -137.473, 0.8588235, 0, 0.1372549, 1,
7.959596, 4.989899, -137.6166, 0.8588235, 0, 0.1372549, 1,
8, 4.989899, -137.7634, 0.8588235, 0, 0.1372549, 1,
4, 5.030303, -139.3276, 0.654902, 0, 0.3411765, 1,
4.040404, 5.030303, -139.156, 0.7568628, 0, 0.2392157, 1,
4.080808, 5.030303, -138.9875, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.030303, -138.8224, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.030303, -138.6604, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.030303, -138.5017, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.030303, -138.3462, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.030303, -138.1939, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.030303, -138.0448, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.030303, -137.899, 0.8588235, 0, 0.1372549, 1,
4.40404, 5.030303, -137.7564, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.030303, -137.617, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.030303, -137.4808, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.030303, -137.3479, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.030303, -137.2182, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.030303, -137.0917, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.030303, -136.9685, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.030303, -136.8484, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.030303, -136.7316, 0.9647059, 0, 0.03137255, 1,
4.767677, 5.030303, -136.6181, 0.9647059, 0, 0.03137255, 1,
4.808081, 5.030303, -136.5077, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.030303, -136.4006, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.030303, -136.2967, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.030303, -136.196, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.030303, -136.0986, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.030303, -136.0044, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.030303, -135.9134, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.030303, -135.8256, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.030303, -135.7411, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.030303, -135.6597, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.030303, -135.5816, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.030303, -135.5068, 1, 0.06666667, 0, 1,
5.292929, 5.030303, -135.4351, 1, 0.06666667, 0, 1,
5.333333, 5.030303, -135.3667, 1, 0.06666667, 0, 1,
5.373737, 5.030303, -135.3015, 1, 0.06666667, 0, 1,
5.414141, 5.030303, -135.2396, 1, 0.06666667, 0, 1,
5.454545, 5.030303, -135.1808, 1, 0.06666667, 0, 1,
5.494949, 5.030303, -135.1253, 1, 0.06666667, 0, 1,
5.535354, 5.030303, -135.073, 1, 0.06666667, 0, 1,
5.575758, 5.030303, -135.024, 1, 0.06666667, 0, 1,
5.616162, 5.030303, -134.9781, 1, 0.06666667, 0, 1,
5.656566, 5.030303, -134.9355, 1, 0.06666667, 0, 1,
5.69697, 5.030303, -134.8961, 1, 0.06666667, 0, 1,
5.737374, 5.030303, -134.86, 1, 0.06666667, 0, 1,
5.777778, 5.030303, -134.8271, 1, 0.06666667, 0, 1,
5.818182, 5.030303, -134.7973, 1, 0.06666667, 0, 1,
5.858586, 5.030303, -134.7709, 1, 0.06666667, 0, 1,
5.89899, 5.030303, -134.7476, 1, 0.06666667, 0, 1,
5.939394, 5.030303, -134.7276, 1, 0.06666667, 0, 1,
5.979798, 5.030303, -134.7108, 1, 0.06666667, 0, 1,
6.020202, 5.030303, -134.6972, 1, 0.06666667, 0, 1,
6.060606, 5.030303, -134.6869, 1, 0.06666667, 0, 1,
6.10101, 5.030303, -134.6797, 1, 0.06666667, 0, 1,
6.141414, 5.030303, -134.6758, 1, 0.06666667, 0, 1,
6.181818, 5.030303, -134.6752, 1, 0.06666667, 0, 1,
6.222222, 5.030303, -134.6777, 1, 0.06666667, 0, 1,
6.262626, 5.030303, -134.6835, 1, 0.06666667, 0, 1,
6.30303, 5.030303, -134.6925, 1, 0.06666667, 0, 1,
6.343434, 5.030303, -134.7047, 1, 0.06666667, 0, 1,
6.383838, 5.030303, -134.7202, 1, 0.06666667, 0, 1,
6.424242, 5.030303, -134.7389, 1, 0.06666667, 0, 1,
6.464646, 5.030303, -134.7608, 1, 0.06666667, 0, 1,
6.505051, 5.030303, -134.7859, 1, 0.06666667, 0, 1,
6.545455, 5.030303, -134.8143, 1, 0.06666667, 0, 1,
6.585859, 5.030303, -134.8458, 1, 0.06666667, 0, 1,
6.626263, 5.030303, -134.8806, 1, 0.06666667, 0, 1,
6.666667, 5.030303, -134.9187, 1, 0.06666667, 0, 1,
6.707071, 5.030303, -134.9599, 1, 0.06666667, 0, 1,
6.747475, 5.030303, -135.0044, 1, 0.06666667, 0, 1,
6.787879, 5.030303, -135.0521, 1, 0.06666667, 0, 1,
6.828283, 5.030303, -135.1031, 1, 0.06666667, 0, 1,
6.868687, 5.030303, -135.1572, 1, 0.06666667, 0, 1,
6.909091, 5.030303, -135.2146, 1, 0.06666667, 0, 1,
6.949495, 5.030303, -135.2753, 1, 0.06666667, 0, 1,
6.989899, 5.030303, -135.3391, 1, 0.06666667, 0, 1,
7.030303, 5.030303, -135.4062, 1, 0.06666667, 0, 1,
7.070707, 5.030303, -135.4765, 1, 0.06666667, 0, 1,
7.111111, 5.030303, -135.55, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.030303, -135.6267, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.030303, -135.7067, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.030303, -135.7899, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.030303, -135.8763, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.030303, -135.9659, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.030303, -136.0588, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.030303, -136.1549, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.030303, -136.2542, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.030303, -136.3568, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.030303, -136.4626, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.030303, -136.5716, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.030303, -136.6838, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.030303, -136.7993, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.030303, -136.9179, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.030303, -137.0398, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.030303, -137.165, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.030303, -137.2933, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.030303, -137.4249, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.030303, -137.5597, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.030303, -137.6977, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.030303, -137.839, 0.8588235, 0, 0.1372549, 1,
8, 5.030303, -137.9835, 0.8588235, 0, 0.1372549, 1,
4, 5.070707, -139.5275, 0.654902, 0, 0.3411765, 1,
4.040404, 5.070707, -139.3586, 0.654902, 0, 0.3411765, 1,
4.080808, 5.070707, -139.1929, 0.7568628, 0, 0.2392157, 1,
4.121212, 5.070707, -139.0303, 0.7568628, 0, 0.2392157, 1,
4.161616, 5.070707, -138.8709, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.070707, -138.7147, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.070707, -138.5616, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.070707, -138.4118, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.070707, -138.2651, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.070707, -138.1216, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.070707, -137.9812, 0.8588235, 0, 0.1372549, 1,
4.444445, 5.070707, -137.844, 0.8588235, 0, 0.1372549, 1,
4.484848, 5.070707, -137.7101, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.070707, -137.5792, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.070707, -137.4516, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.070707, -137.3271, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.070707, -137.2058, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.070707, -137.0877, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.070707, -136.9727, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.070707, -136.861, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.070707, -136.7524, 0.9647059, 0, 0.03137255, 1,
4.848485, 5.070707, -136.647, 0.9647059, 0, 0.03137255, 1,
4.888889, 5.070707, -136.5447, 0.9647059, 0, 0.03137255, 1,
4.929293, 5.070707, -136.4456, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.070707, -136.3497, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.070707, -136.257, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.070707, -136.1674, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.070707, -136.0811, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.070707, -135.9979, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.070707, -135.9178, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.070707, -135.841, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.070707, -135.7673, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.070707, -135.6968, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.070707, -135.6295, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.070707, -135.5653, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.070707, -135.5043, 1, 0.06666667, 0, 1,
5.454545, 5.070707, -135.4465, 1, 0.06666667, 0, 1,
5.494949, 5.070707, -135.3919, 1, 0.06666667, 0, 1,
5.535354, 5.070707, -135.3405, 1, 0.06666667, 0, 1,
5.575758, 5.070707, -135.2922, 1, 0.06666667, 0, 1,
5.616162, 5.070707, -135.2471, 1, 0.06666667, 0, 1,
5.656566, 5.070707, -135.2051, 1, 0.06666667, 0, 1,
5.69697, 5.070707, -135.1664, 1, 0.06666667, 0, 1,
5.737374, 5.070707, -135.1308, 1, 0.06666667, 0, 1,
5.777778, 5.070707, -135.0984, 1, 0.06666667, 0, 1,
5.818182, 5.070707, -135.0692, 1, 0.06666667, 0, 1,
5.858586, 5.070707, -135.0431, 1, 0.06666667, 0, 1,
5.89899, 5.070707, -135.0202, 1, 0.06666667, 0, 1,
5.939394, 5.070707, -135.0005, 1, 0.06666667, 0, 1,
5.979798, 5.070707, -134.984, 1, 0.06666667, 0, 1,
6.020202, 5.070707, -134.9706, 1, 0.06666667, 0, 1,
6.060606, 5.070707, -134.9604, 1, 0.06666667, 0, 1,
6.10101, 5.070707, -134.9534, 1, 0.06666667, 0, 1,
6.141414, 5.070707, -134.9496, 1, 0.06666667, 0, 1,
6.181818, 5.070707, -134.9489, 1, 0.06666667, 0, 1,
6.222222, 5.070707, -134.9514, 1, 0.06666667, 0, 1,
6.262626, 5.070707, -134.9571, 1, 0.06666667, 0, 1,
6.30303, 5.070707, -134.966, 1, 0.06666667, 0, 1,
6.343434, 5.070707, -134.978, 1, 0.06666667, 0, 1,
6.383838, 5.070707, -134.9932, 1, 0.06666667, 0, 1,
6.424242, 5.070707, -135.0116, 1, 0.06666667, 0, 1,
6.464646, 5.070707, -135.0332, 1, 0.06666667, 0, 1,
6.505051, 5.070707, -135.0579, 1, 0.06666667, 0, 1,
6.545455, 5.070707, -135.0858, 1, 0.06666667, 0, 1,
6.585859, 5.070707, -135.1169, 1, 0.06666667, 0, 1,
6.626263, 5.070707, -135.1511, 1, 0.06666667, 0, 1,
6.666667, 5.070707, -135.1886, 1, 0.06666667, 0, 1,
6.707071, 5.070707, -135.2292, 1, 0.06666667, 0, 1,
6.747475, 5.070707, -135.2729, 1, 0.06666667, 0, 1,
6.787879, 5.070707, -135.3199, 1, 0.06666667, 0, 1,
6.828283, 5.070707, -135.37, 1, 0.06666667, 0, 1,
6.868687, 5.070707, -135.4233, 1, 0.06666667, 0, 1,
6.909091, 5.070707, -135.4798, 1, 0.06666667, 0, 1,
6.949495, 5.070707, -135.5395, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.070707, -135.6023, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.070707, -135.6683, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.070707, -135.7375, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.070707, -135.8098, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.070707, -135.8854, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.070707, -135.9641, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.070707, -136.0459, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.070707, -136.131, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.070707, -136.2192, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.070707, -136.3106, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.070707, -136.4052, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.070707, -136.5029, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.070707, -136.6039, 0.9647059, 0, 0.03137255, 1,
7.515152, 5.070707, -136.7079, 0.9647059, 0, 0.03137255, 1,
7.555555, 5.070707, -136.8152, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.070707, -136.9257, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.070707, -137.0393, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.070707, -137.1561, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.070707, -137.276, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.070707, -137.3992, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.070707, -137.5255, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.070707, -137.655, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.070707, -137.7877, 0.8588235, 0, 0.1372549, 1,
7.919192, 5.070707, -137.9235, 0.8588235, 0, 0.1372549, 1,
7.959596, 5.070707, -138.0625, 0.7568628, 0, 0.2392157, 1,
8, 5.070707, -138.2047, 0.7568628, 0, 0.2392157, 1,
4, 5.111111, -139.729, 0.654902, 0, 0.3411765, 1,
4.040404, 5.111111, -139.5627, 0.654902, 0, 0.3411765, 1,
4.080808, 5.111111, -139.3996, 0.654902, 0, 0.3411765, 1,
4.121212, 5.111111, -139.2396, 0.654902, 0, 0.3411765, 1,
4.161616, 5.111111, -139.0827, 0.7568628, 0, 0.2392157, 1,
4.20202, 5.111111, -138.9289, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.111111, -138.7783, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.111111, -138.6308, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.111111, -138.4864, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.111111, -138.3452, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.111111, -138.207, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.111111, -138.072, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.111111, -137.9401, 0.8588235, 0, 0.1372549, 1,
4.525252, 5.111111, -137.8114, 0.8588235, 0, 0.1372549, 1,
4.565657, 5.111111, -137.6857, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.111111, -137.5632, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.111111, -137.4438, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.111111, -137.3276, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.111111, -137.2144, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.111111, -137.1044, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.111111, -136.9975, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.111111, -136.8938, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.111111, -136.7931, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.111111, -136.6956, 0.9647059, 0, 0.03137255, 1,
4.969697, 5.111111, -136.6012, 0.9647059, 0, 0.03137255, 1,
5.010101, 5.111111, -136.51, 0.9647059, 0, 0.03137255, 1,
5.050505, 5.111111, -136.4218, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.111111, -136.3368, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.111111, -136.2549, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.111111, -136.1761, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.111111, -136.1005, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.111111, -136.028, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.111111, -135.9586, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.111111, -135.8923, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.111111, -135.8292, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.111111, -135.7692, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.111111, -135.7123, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.111111, -135.6585, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.111111, -135.6078, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.111111, -135.5603, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.111111, -135.5159, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.111111, -135.4747, 1, 0.06666667, 0, 1,
5.69697, 5.111111, -135.4365, 1, 0.06666667, 0, 1,
5.737374, 5.111111, -135.4015, 1, 0.06666667, 0, 1,
5.777778, 5.111111, -135.3696, 1, 0.06666667, 0, 1,
5.818182, 5.111111, -135.3408, 1, 0.06666667, 0, 1,
5.858586, 5.111111, -135.3152, 1, 0.06666667, 0, 1,
5.89899, 5.111111, -135.2926, 1, 0.06666667, 0, 1,
5.939394, 5.111111, -135.2732, 1, 0.06666667, 0, 1,
5.979798, 5.111111, -135.257, 1, 0.06666667, 0, 1,
6.020202, 5.111111, -135.2438, 1, 0.06666667, 0, 1,
6.060606, 5.111111, -135.2338, 1, 0.06666667, 0, 1,
6.10101, 5.111111, -135.2269, 1, 0.06666667, 0, 1,
6.141414, 5.111111, -135.2231, 1, 0.06666667, 0, 1,
6.181818, 5.111111, -135.2225, 1, 0.06666667, 0, 1,
6.222222, 5.111111, -135.2249, 1, 0.06666667, 0, 1,
6.262626, 5.111111, -135.2305, 1, 0.06666667, 0, 1,
6.30303, 5.111111, -135.2392, 1, 0.06666667, 0, 1,
6.343434, 5.111111, -135.2511, 1, 0.06666667, 0, 1,
6.383838, 5.111111, -135.2661, 1, 0.06666667, 0, 1,
6.424242, 5.111111, -135.2842, 1, 0.06666667, 0, 1,
6.464646, 5.111111, -135.3054, 1, 0.06666667, 0, 1,
6.505051, 5.111111, -135.3297, 1, 0.06666667, 0, 1,
6.545455, 5.111111, -135.3572, 1, 0.06666667, 0, 1,
6.585859, 5.111111, -135.3878, 1, 0.06666667, 0, 1,
6.626263, 5.111111, -135.4215, 1, 0.06666667, 0, 1,
6.666667, 5.111111, -135.4583, 1, 0.06666667, 0, 1,
6.707071, 5.111111, -135.4983, 1, 0.06666667, 0, 1,
6.747475, 5.111111, -135.5414, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.111111, -135.5876, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.111111, -135.637, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.111111, -135.6894, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.111111, -135.745, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.111111, -135.8037, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.111111, -135.8656, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.111111, -135.9305, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.111111, -135.9986, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.111111, -136.0698, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.111111, -136.1442, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.111111, -136.2216, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.111111, -136.3022, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.111111, -136.3859, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.111111, -136.4727, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.111111, -136.5627, 0.9647059, 0, 0.03137255, 1,
7.393939, 5.111111, -136.6558, 0.9647059, 0, 0.03137255, 1,
7.434343, 5.111111, -136.752, 0.9647059, 0, 0.03137255, 1,
7.474748, 5.111111, -136.8513, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.111111, -136.9538, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.111111, -137.0594, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.111111, -137.1681, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.111111, -137.2799, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.111111, -137.3949, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.111111, -137.513, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.111111, -137.6342, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.111111, -137.7585, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.111111, -137.8859, 0.8588235, 0, 0.1372549, 1,
7.878788, 5.111111, -138.0165, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.111111, -138.1502, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.111111, -138.287, 0.7568628, 0, 0.2392157, 1,
8, 5.111111, -138.427, 0.7568628, 0, 0.2392157, 1,
4, 5.151515, -139.9319, 0.654902, 0, 0.3411765, 1,
4.040404, 5.151515, -139.7682, 0.654902, 0, 0.3411765, 1,
4.080808, 5.151515, -139.6076, 0.654902, 0, 0.3411765, 1,
4.121212, 5.151515, -139.4501, 0.654902, 0, 0.3411765, 1,
4.161616, 5.151515, -139.2957, 0.654902, 0, 0.3411765, 1,
4.20202, 5.151515, -139.1443, 0.7568628, 0, 0.2392157, 1,
4.242424, 5.151515, -138.9961, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.151515, -138.8509, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.151515, -138.7087, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.151515, -138.5697, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.151515, -138.4337, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.151515, -138.3008, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.151515, -138.171, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.151515, -138.0442, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.151515, -137.9206, 0.8588235, 0, 0.1372549, 1,
4.606061, 5.151515, -137.8, 0.8588235, 0, 0.1372549, 1,
4.646465, 5.151515, -137.6824, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.151515, -137.568, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.151515, -137.4566, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.151515, -137.3483, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.151515, -137.2431, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.151515, -137.141, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.151515, -137.0419, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.151515, -136.9459, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.151515, -136.853, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.151515, -136.7632, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.151515, -136.6764, 0.9647059, 0, 0.03137255, 1,
5.090909, 5.151515, -136.5927, 0.9647059, 0, 0.03137255, 1,
5.131313, 5.151515, -136.5121, 0.9647059, 0, 0.03137255, 1,
5.171717, 5.151515, -136.4346, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.151515, -136.3601, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.151515, -136.2887, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.151515, -136.2204, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.151515, -136.1552, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.151515, -136.093, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.151515, -136.0339, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.151515, -135.9779, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.151515, -135.925, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.151515, -135.8752, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.151515, -135.8284, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.151515, -135.7847, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.151515, -135.744, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.151515, -135.7065, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.151515, -135.672, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.151515, -135.6406, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.151515, -135.6123, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.151515, -135.587, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.151515, -135.5649, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.151515, -135.5458, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.151515, -135.5298, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.151515, -135.5168, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.151515, -135.5069, 1, 0.06666667, 0, 1,
6.10101, 5.151515, -135.5001, 1, 0.06666667, 0, 1,
6.141414, 5.151515, -135.4964, 1, 0.06666667, 0, 1,
6.181818, 5.151515, -135.4958, 1, 0.06666667, 0, 1,
6.222222, 5.151515, -135.4982, 1, 0.06666667, 0, 1,
6.262626, 5.151515, -135.5037, 1, 0.06666667, 0, 1,
6.30303, 5.151515, -135.5123, 1, 0.06666667, 0, 1,
6.343434, 5.151515, -135.524, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.151515, -135.5387, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.151515, -135.5565, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.151515, -135.5774, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.151515, -135.6014, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.151515, -135.6284, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.151515, -135.6585, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.151515, -135.6917, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.151515, -135.728, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.151515, -135.7673, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.151515, -135.8097, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.151515, -135.8552, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.151515, -135.9038, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.151515, -135.9554, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.151515, -136.0102, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.151515, -136.068, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.151515, -136.1288, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.151515, -136.1928, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.151515, -136.2598, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.151515, -136.3299, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.151515, -136.4031, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.151515, -136.4793, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.151515, -136.5587, 0.9647059, 0, 0.03137255, 1,
7.272727, 5.151515, -136.6411, 0.9647059, 0, 0.03137255, 1,
7.313131, 5.151515, -136.7265, 0.9647059, 0, 0.03137255, 1,
7.353535, 5.151515, -136.8151, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.151515, -136.9067, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.151515, -137.0014, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.151515, -137.0992, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.151515, -137.2001, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.151515, -137.304, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.151515, -137.411, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.151515, -137.5211, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.151515, -137.6342, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.151515, -137.7505, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.151515, -137.8698, 0.8588235, 0, 0.1372549, 1,
7.79798, 5.151515, -137.9922, 0.8588235, 0, 0.1372549, 1,
7.838384, 5.151515, -138.1176, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.151515, -138.2462, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.151515, -138.3778, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.151515, -138.5125, 0.7568628, 0, 0.2392157, 1,
8, 5.151515, -138.6503, 0.7568628, 0, 0.2392157, 1,
4, 5.191919, -140.1361, 0.654902, 0, 0.3411765, 1,
4.040404, 5.191919, -139.975, 0.654902, 0, 0.3411765, 1,
4.080808, 5.191919, -139.8169, 0.654902, 0, 0.3411765, 1,
4.121212, 5.191919, -139.6618, 0.654902, 0, 0.3411765, 1,
4.161616, 5.191919, -139.5098, 0.654902, 0, 0.3411765, 1,
4.20202, 5.191919, -139.3608, 0.654902, 0, 0.3411765, 1,
4.242424, 5.191919, -139.2148, 0.7568628, 0, 0.2392157, 1,
4.282828, 5.191919, -139.0719, 0.7568628, 0, 0.2392157, 1,
4.323232, 5.191919, -138.932, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.191919, -138.7951, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.191919, -138.6612, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.191919, -138.5304, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.191919, -138.4025, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.191919, -138.2778, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.191919, -138.156, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.191919, -138.0373, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.191919, -137.9216, 0.8588235, 0, 0.1372549, 1,
4.686869, 5.191919, -137.8089, 0.8588235, 0, 0.1372549, 1,
4.727273, 5.191919, -137.6993, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.191919, -137.5927, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.191919, -137.4891, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.191919, -137.3885, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.191919, -137.291, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.191919, -137.1965, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.191919, -137.105, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.191919, -137.0166, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.191919, -136.9311, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.191919, -136.8488, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.191919, -136.7694, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.191919, -136.6931, 0.9647059, 0, 0.03137255, 1,
5.212121, 5.191919, -136.6198, 0.9647059, 0, 0.03137255, 1,
5.252525, 5.191919, -136.5495, 0.9647059, 0, 0.03137255, 1,
5.292929, 5.191919, -136.4822, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.191919, -136.418, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.191919, -136.3568, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.191919, -136.2986, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.191919, -136.2435, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.191919, -136.1914, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.191919, -136.1423, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.191919, -136.0963, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.191919, -136.0532, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.191919, -136.0132, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.191919, -135.9763, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.191919, -135.9423, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.191919, -135.9114, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.191919, -135.8835, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.191919, -135.8587, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.191919, -135.8368, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.191919, -135.818, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.191919, -135.8023, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.191919, -135.7895, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.191919, -135.7798, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.191919, -135.7731, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.191919, -135.7695, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.191919, -135.7688, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.191919, -135.7712, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.191919, -135.7766, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.191919, -135.7851, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.191919, -135.7966, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.191919, -135.8111, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.191919, -135.8286, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.191919, -135.8492, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.191919, -135.8728, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.191919, -135.8994, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.191919, -135.929, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.191919, -135.9617, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.191919, -135.9974, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.191919, -136.0361, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.191919, -136.0779, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.191919, -136.1227, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.191919, -136.1705, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.191919, -136.2214, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.191919, -136.2752, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.191919, -136.3321, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.191919, -136.3921, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.191919, -136.455, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.191919, -136.521, 0.9647059, 0, 0.03137255, 1,
7.111111, 5.191919, -136.59, 0.9647059, 0, 0.03137255, 1,
7.151515, 5.191919, -136.6621, 0.9647059, 0, 0.03137255, 1,
7.191919, 5.191919, -136.7371, 0.9647059, 0, 0.03137255, 1,
7.232323, 5.191919, -136.8152, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.191919, -136.8963, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.191919, -136.9805, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.191919, -137.0677, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.191919, -137.1579, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.191919, -137.2511, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.191919, -137.3474, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.191919, -137.4467, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.191919, -137.549, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.191919, -137.6544, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.191919, -137.7627, 0.8588235, 0, 0.1372549, 1,
7.676768, 5.191919, -137.8741, 0.8588235, 0, 0.1372549, 1,
7.717172, 5.191919, -137.9886, 0.8588235, 0, 0.1372549, 1,
7.757576, 5.191919, -138.106, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.191919, -138.2265, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.191919, -138.35, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.191919, -138.4766, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.191919, -138.6062, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.191919, -138.7388, 0.7568628, 0, 0.2392157, 1,
8, 5.191919, -138.8744, 0.7568628, 0, 0.2392157, 1,
4, 5.232323, -140.3417, 0.654902, 0, 0.3411765, 1,
4.040404, 5.232323, -140.183, 0.654902, 0, 0.3411765, 1,
4.080808, 5.232323, -140.0274, 0.654902, 0, 0.3411765, 1,
4.121212, 5.232323, -139.8747, 0.654902, 0, 0.3411765, 1,
4.161616, 5.232323, -139.725, 0.654902, 0, 0.3411765, 1,
4.20202, 5.232323, -139.5783, 0.654902, 0, 0.3411765, 1,
4.242424, 5.232323, -139.4345, 0.654902, 0, 0.3411765, 1,
4.282828, 5.232323, -139.2938, 0.654902, 0, 0.3411765, 1,
4.323232, 5.232323, -139.156, 0.7568628, 0, 0.2392157, 1,
4.363636, 5.232323, -139.0212, 0.7568628, 0, 0.2392157, 1,
4.40404, 5.232323, -138.8894, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.232323, -138.7606, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.232323, -138.6348, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.232323, -138.5119, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.232323, -138.392, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.232323, -138.2751, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.232323, -138.1612, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.232323, -138.0503, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.232323, -137.9423, 0.8588235, 0, 0.1372549, 1,
4.767677, 5.232323, -137.8373, 0.8588235, 0, 0.1372549, 1,
4.808081, 5.232323, -137.7353, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.232323, -137.6363, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.232323, -137.5403, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.232323, -137.4473, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.232323, -137.3572, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.232323, -137.2701, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.232323, -137.186, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.232323, -137.1049, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.232323, -137.0267, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.232323, -136.9516, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.232323, -136.8794, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.232323, -136.8102, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.232323, -136.744, 0.9647059, 0, 0.03137255, 1,
5.333333, 5.232323, -136.6807, 0.9647059, 0, 0.03137255, 1,
5.373737, 5.232323, -136.6205, 0.9647059, 0, 0.03137255, 1,
5.414141, 5.232323, -136.5632, 0.9647059, 0, 0.03137255, 1,
5.454545, 5.232323, -136.5089, 0.9647059, 0, 0.03137255, 1,
5.494949, 5.232323, -136.4576, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.232323, -136.4093, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.232323, -136.364, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.232323, -136.3216, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.232323, -136.2822, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.232323, -136.2458, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.232323, -136.2124, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.232323, -136.1819, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.232323, -136.1545, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.232323, -136.13, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.232323, -136.1085, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.232323, -136.09, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.232323, -136.0745, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.232323, -136.0619, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.232323, -136.0524, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.232323, -136.0458, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.232323, -136.0422, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.232323, -136.0415, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.232323, -136.0439, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.232323, -136.0493, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.232323, -136.0576, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.232323, -136.0689, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.232323, -136.0832, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.232323, -136.1004, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.232323, -136.1207, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.232323, -136.1439, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.232323, -136.1701, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.232323, -136.1993, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.232323, -136.2315, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.232323, -136.2666, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.232323, -136.3048, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.232323, -136.3459, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.232323, -136.39, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.232323, -136.4371, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.232323, -136.4871, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.232323, -136.5402, 0.9647059, 0, 0.03137255, 1,
6.949495, 5.232323, -136.5962, 0.9647059, 0, 0.03137255, 1,
6.989899, 5.232323, -136.6552, 0.9647059, 0, 0.03137255, 1,
7.030303, 5.232323, -136.7172, 0.9647059, 0, 0.03137255, 1,
7.070707, 5.232323, -136.7822, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.232323, -136.8501, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.232323, -136.921, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.232323, -136.9949, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.232323, -137.0719, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.232323, -137.1517, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.232323, -137.2346, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.232323, -137.3204, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.232323, -137.4092, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.232323, -137.5011, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.232323, -137.5958, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.232323, -137.6936, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.232323, -137.7943, 0.8588235, 0, 0.1372549, 1,
7.59596, 5.232323, -137.8981, 0.8588235, 0, 0.1372549, 1,
7.636364, 5.232323, -138.0048, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.232323, -138.1145, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.232323, -138.2272, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.232323, -138.3428, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.232323, -138.4614, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.232323, -138.5831, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.232323, -138.7077, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.232323, -138.8352, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.232323, -138.9658, 0.7568628, 0, 0.2392157, 1,
8, 5.232323, -139.0993, 0.7568628, 0, 0.2392157, 1,
4, 5.272727, -140.5484, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.272727, -140.3922, 0.654902, 0, 0.3411765, 1,
4.080808, 5.272727, -140.2389, 0.654902, 0, 0.3411765, 1,
4.121212, 5.272727, -140.0885, 0.654902, 0, 0.3411765, 1,
4.161616, 5.272727, -139.9411, 0.654902, 0, 0.3411765, 1,
4.20202, 5.272727, -139.7967, 0.654902, 0, 0.3411765, 1,
4.242424, 5.272727, -139.6551, 0.654902, 0, 0.3411765, 1,
4.282828, 5.272727, -139.5165, 0.654902, 0, 0.3411765, 1,
4.323232, 5.272727, -139.3809, 0.654902, 0, 0.3411765, 1,
4.363636, 5.272727, -139.2481, 0.654902, 0, 0.3411765, 1,
4.40404, 5.272727, -139.1183, 0.7568628, 0, 0.2392157, 1,
4.444445, 5.272727, -138.9915, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.272727, -138.8675, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.272727, -138.7466, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.272727, -138.6285, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.272727, -138.5134, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.272727, -138.4012, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.272727, -138.292, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.272727, -138.1857, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.272727, -138.0823, 0.7568628, 0, 0.2392157, 1,
4.808081, 5.272727, -137.9819, 0.8588235, 0, 0.1372549, 1,
4.848485, 5.272727, -137.8844, 0.8588235, 0, 0.1372549, 1,
4.888889, 5.272727, -137.7898, 0.8588235, 0, 0.1372549, 1,
4.929293, 5.272727, -137.6982, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.272727, -137.6095, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.272727, -137.5237, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.272727, -137.4409, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.272727, -137.361, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.272727, -137.2841, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.272727, -137.2101, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.272727, -137.139, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.272727, -137.0708, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.272727, -137.0056, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.272727, -136.9434, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.272727, -136.884, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.272727, -136.8276, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.272727, -136.7742, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.272727, -136.7236, 0.9647059, 0, 0.03137255, 1,
5.535354, 5.272727, -136.6761, 0.9647059, 0, 0.03137255, 1,
5.575758, 5.272727, -136.6314, 0.9647059, 0, 0.03137255, 1,
5.616162, 5.272727, -136.5897, 0.9647059, 0, 0.03137255, 1,
5.656566, 5.272727, -136.5509, 0.9647059, 0, 0.03137255, 1,
5.69697, 5.272727, -136.5151, 0.9647059, 0, 0.03137255, 1,
5.737374, 5.272727, -136.4821, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.272727, -136.4522, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.272727, -136.4251, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.272727, -136.401, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.272727, -136.3799, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.272727, -136.3616, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.272727, -136.3464, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.272727, -136.334, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.272727, -136.3246, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.272727, -136.3181, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.272727, -136.3145, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.272727, -136.3139, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.272727, -136.3163, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.272727, -136.3215, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.272727, -136.3297, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.272727, -136.3408, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.272727, -136.3549, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.272727, -136.3719, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.272727, -136.3918, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.272727, -136.4147, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.272727, -136.4405, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.272727, -136.4693, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.272727, -136.5009, 0.9647059, 0, 0.03137255, 1,
6.666667, 5.272727, -136.5356, 0.9647059, 0, 0.03137255, 1,
6.707071, 5.272727, -136.5731, 0.9647059, 0, 0.03137255, 1,
6.747475, 5.272727, -136.6136, 0.9647059, 0, 0.03137255, 1,
6.787879, 5.272727, -136.657, 0.9647059, 0, 0.03137255, 1,
6.828283, 5.272727, -136.7034, 0.9647059, 0, 0.03137255, 1,
6.868687, 5.272727, -136.7527, 0.9647059, 0, 0.03137255, 1,
6.909091, 5.272727, -136.8049, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.272727, -136.8601, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.272727, -136.9182, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.272727, -136.9792, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.272727, -137.0432, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.272727, -137.1101, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.272727, -137.18, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.272727, -137.2528, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.272727, -137.3285, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.272727, -137.4072, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.272727, -137.4888, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.272727, -137.5733, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.272727, -137.6608, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.272727, -137.7511, 0.8588235, 0, 0.1372549, 1,
7.474748, 5.272727, -137.8445, 0.8588235, 0, 0.1372549, 1,
7.515152, 5.272727, -137.9408, 0.8588235, 0, 0.1372549, 1,
7.555555, 5.272727, -138.04, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.272727, -138.1421, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.272727, -138.2472, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.272727, -138.3552, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.272727, -138.4662, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.272727, -138.58, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.272727, -138.6969, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.272727, -138.8166, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.272727, -138.9393, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.272727, -139.065, 0.7568628, 0, 0.2392157, 1,
7.959596, 5.272727, -139.1935, 0.7568628, 0, 0.2392157, 1,
8, 5.272727, -139.325, 0.654902, 0, 0.3411765, 1,
4, 5.313131, -140.7562, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.313131, -140.6024, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.313131, -140.4514, 0.654902, 0, 0.3411765, 1,
4.121212, 5.313131, -140.3033, 0.654902, 0, 0.3411765, 1,
4.161616, 5.313131, -140.1582, 0.654902, 0, 0.3411765, 1,
4.20202, 5.313131, -140.0159, 0.654902, 0, 0.3411765, 1,
4.242424, 5.313131, -139.8765, 0.654902, 0, 0.3411765, 1,
4.282828, 5.313131, -139.74, 0.654902, 0, 0.3411765, 1,
4.323232, 5.313131, -139.6064, 0.654902, 0, 0.3411765, 1,
4.363636, 5.313131, -139.4757, 0.654902, 0, 0.3411765, 1,
4.40404, 5.313131, -139.3478, 0.654902, 0, 0.3411765, 1,
4.444445, 5.313131, -139.2229, 0.7568628, 0, 0.2392157, 1,
4.484848, 5.313131, -139.1008, 0.7568628, 0, 0.2392157, 1,
4.525252, 5.313131, -138.9817, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.313131, -138.8654, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.313131, -138.7521, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.313131, -138.6416, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.313131, -138.534, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.313131, -138.4293, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.313131, -138.3275, 0.7568628, 0, 0.2392157, 1,
4.808081, 5.313131, -138.2286, 0.7568628, 0, 0.2392157, 1,
4.848485, 5.313131, -138.1326, 0.7568628, 0, 0.2392157, 1,
4.888889, 5.313131, -138.0394, 0.7568628, 0, 0.2392157, 1,
4.929293, 5.313131, -137.9492, 0.8588235, 0, 0.1372549, 1,
4.969697, 5.313131, -137.8618, 0.8588235, 0, 0.1372549, 1,
5.010101, 5.313131, -137.7774, 0.8588235, 0, 0.1372549, 1,
5.050505, 5.313131, -137.6958, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.313131, -137.6171, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.313131, -137.5414, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.313131, -137.4685, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.313131, -137.3985, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.313131, -137.3314, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.313131, -137.2671, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.313131, -137.2058, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.313131, -137.1474, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.313131, -137.0918, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.313131, -137.0392, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.313131, -136.9894, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.313131, -136.9426, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.313131, -136.8986, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.313131, -136.8575, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.313131, -136.8193, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.313131, -136.784, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.313131, -136.7516, 0.9647059, 0, 0.03137255, 1,
5.777778, 5.313131, -136.7221, 0.9647059, 0, 0.03137255, 1,
5.818182, 5.313131, -136.6954, 0.9647059, 0, 0.03137255, 1,
5.858586, 5.313131, -136.6717, 0.9647059, 0, 0.03137255, 1,
5.89899, 5.313131, -136.6509, 0.9647059, 0, 0.03137255, 1,
5.939394, 5.313131, -136.6329, 0.9647059, 0, 0.03137255, 1,
5.979798, 5.313131, -136.6179, 0.9647059, 0, 0.03137255, 1,
6.020202, 5.313131, -136.6057, 0.9647059, 0, 0.03137255, 1,
6.060606, 5.313131, -136.5964, 0.9647059, 0, 0.03137255, 1,
6.10101, 5.313131, -136.59, 0.9647059, 0, 0.03137255, 1,
6.141414, 5.313131, -136.5865, 0.9647059, 0, 0.03137255, 1,
6.181818, 5.313131, -136.5859, 0.9647059, 0, 0.03137255, 1,
6.222222, 5.313131, -136.5882, 0.9647059, 0, 0.03137255, 1,
6.262626, 5.313131, -136.5934, 0.9647059, 0, 0.03137255, 1,
6.30303, 5.313131, -136.6015, 0.9647059, 0, 0.03137255, 1,
6.343434, 5.313131, -136.6124, 0.9647059, 0, 0.03137255, 1,
6.383838, 5.313131, -136.6263, 0.9647059, 0, 0.03137255, 1,
6.424242, 5.313131, -136.643, 0.9647059, 0, 0.03137255, 1,
6.464646, 5.313131, -136.6627, 0.9647059, 0, 0.03137255, 1,
6.505051, 5.313131, -136.6852, 0.9647059, 0, 0.03137255, 1,
6.545455, 5.313131, -136.7106, 0.9647059, 0, 0.03137255, 1,
6.585859, 5.313131, -136.7389, 0.9647059, 0, 0.03137255, 1,
6.626263, 5.313131, -136.7701, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.313131, -136.8042, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.313131, -136.8412, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.313131, -136.8811, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.313131, -136.9238, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.313131, -136.9695, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.313131, -137.0181, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.313131, -137.0695, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.313131, -137.1238, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.313131, -137.181, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.313131, -137.2412, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.313131, -137.3042, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.313131, -137.3701, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.313131, -137.4389, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.313131, -137.5105, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.313131, -137.5851, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.313131, -137.6626, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.313131, -137.7429, 0.8588235, 0, 0.1372549, 1,
7.353535, 5.313131, -137.8262, 0.8588235, 0, 0.1372549, 1,
7.393939, 5.313131, -137.9123, 0.8588235, 0, 0.1372549, 1,
7.434343, 5.313131, -138.0014, 0.7568628, 0, 0.2392157, 1,
7.474748, 5.313131, -138.0933, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.313131, -138.1881, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.313131, -138.2858, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.313131, -138.3864, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.313131, -138.4899, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.313131, -138.5963, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.313131, -138.7055, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.313131, -138.8177, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.313131, -138.9328, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.313131, -139.0507, 0.7568628, 0, 0.2392157, 1,
7.878788, 5.313131, -139.1715, 0.7568628, 0, 0.2392157, 1,
7.919192, 5.313131, -139.2953, 0.654902, 0, 0.3411765, 1,
7.959596, 5.313131, -139.4219, 0.654902, 0, 0.3411765, 1,
8, 5.313131, -139.5514, 0.654902, 0, 0.3411765, 1,
4, 5.353535, -140.9651, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.353535, -140.8136, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.353535, -140.6649, 0.5490196, 0, 0.4470588, 1,
4.121212, 5.353535, -140.519, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.353535, -140.376, 0.654902, 0, 0.3411765, 1,
4.20202, 5.353535, -140.2359, 0.654902, 0, 0.3411765, 1,
4.242424, 5.353535, -140.0986, 0.654902, 0, 0.3411765, 1,
4.282828, 5.353535, -139.9641, 0.654902, 0, 0.3411765, 1,
4.323232, 5.353535, -139.8325, 0.654902, 0, 0.3411765, 1,
4.363636, 5.353535, -139.7038, 0.654902, 0, 0.3411765, 1,
4.40404, 5.353535, -139.5779, 0.654902, 0, 0.3411765, 1,
4.444445, 5.353535, -139.4548, 0.654902, 0, 0.3411765, 1,
4.484848, 5.353535, -139.3346, 0.654902, 0, 0.3411765, 1,
4.525252, 5.353535, -139.2172, 0.7568628, 0, 0.2392157, 1,
4.565657, 5.353535, -139.1027, 0.7568628, 0, 0.2392157, 1,
4.606061, 5.353535, -138.9911, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.353535, -138.8822, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.353535, -138.7763, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.353535, -138.6731, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.353535, -138.5729, 0.7568628, 0, 0.2392157, 1,
4.808081, 5.353535, -138.4754, 0.7568628, 0, 0.2392157, 1,
4.848485, 5.353535, -138.3809, 0.7568628, 0, 0.2392157, 1,
4.888889, 5.353535, -138.2891, 0.7568628, 0, 0.2392157, 1,
4.929293, 5.353535, -138.2003, 0.7568628, 0, 0.2392157, 1,
4.969697, 5.353535, -138.1142, 0.7568628, 0, 0.2392157, 1,
5.010101, 5.353535, -138.031, 0.7568628, 0, 0.2392157, 1,
5.050505, 5.353535, -137.9507, 0.8588235, 0, 0.1372549, 1,
5.090909, 5.353535, -137.8732, 0.8588235, 0, 0.1372549, 1,
5.131313, 5.353535, -137.7986, 0.8588235, 0, 0.1372549, 1,
5.171717, 5.353535, -137.7268, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.353535, -137.6578, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.353535, -137.5917, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.353535, -137.5285, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.353535, -137.4681, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.353535, -137.4105, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.353535, -137.3558, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.353535, -137.3039, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.353535, -137.2549, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.353535, -137.2088, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.353535, -137.1655, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.353535, -137.125, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.353535, -137.0874, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.353535, -137.0526, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.353535, -137.0207, 0.8588235, 0, 0.1372549, 1,
5.777778, 5.353535, -136.9916, 0.8588235, 0, 0.1372549, 1,
5.818182, 5.353535, -136.9654, 0.8588235, 0, 0.1372549, 1,
5.858586, 5.353535, -136.942, 0.8588235, 0, 0.1372549, 1,
5.89899, 5.353535, -136.9215, 0.8588235, 0, 0.1372549, 1,
5.939394, 5.353535, -136.9038, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.353535, -136.8889, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.353535, -136.877, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.353535, -136.8678, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.353535, -136.8615, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.353535, -136.8581, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.353535, -136.8575, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.353535, -136.8597, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.353535, -136.8649, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.353535, -136.8728, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.353535, -136.8836, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.353535, -136.8972, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.353535, -136.9137, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.353535, -136.9331, 0.8588235, 0, 0.1372549, 1,
6.505051, 5.353535, -136.9553, 0.8588235, 0, 0.1372549, 1,
6.545455, 5.353535, -136.9803, 0.8588235, 0, 0.1372549, 1,
6.585859, 5.353535, -137.0082, 0.8588235, 0, 0.1372549, 1,
6.626263, 5.353535, -137.0389, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.353535, -137.0725, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.353535, -137.1089, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.353535, -137.1482, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.353535, -137.1903, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.353535, -137.2353, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.353535, -137.2831, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.353535, -137.3338, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.353535, -137.3873, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.353535, -137.4437, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.353535, -137.5029, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.353535, -137.5649, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.353535, -137.6299, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.353535, -137.6976, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.353535, -137.7682, 0.8588235, 0, 0.1372549, 1,
7.232323, 5.353535, -137.8417, 0.8588235, 0, 0.1372549, 1,
7.272727, 5.353535, -137.918, 0.8588235, 0, 0.1372549, 1,
7.313131, 5.353535, -137.9971, 0.7568628, 0, 0.2392157, 1,
7.353535, 5.353535, -138.0791, 0.7568628, 0, 0.2392157, 1,
7.393939, 5.353535, -138.164, 0.7568628, 0, 0.2392157, 1,
7.434343, 5.353535, -138.2516, 0.7568628, 0, 0.2392157, 1,
7.474748, 5.353535, -138.3422, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.353535, -138.4356, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.353535, -138.5318, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.353535, -138.6309, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.353535, -138.7328, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.353535, -138.8376, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.353535, -138.9452, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.353535, -139.0557, 0.7568628, 0, 0.2392157, 1,
7.79798, 5.353535, -139.169, 0.7568628, 0, 0.2392157, 1,
7.838384, 5.353535, -139.2852, 0.654902, 0, 0.3411765, 1,
7.878788, 5.353535, -139.4042, 0.654902, 0, 0.3411765, 1,
7.919192, 5.353535, -139.5261, 0.654902, 0, 0.3411765, 1,
7.959596, 5.353535, -139.6508, 0.654902, 0, 0.3411765, 1,
8, 5.353535, -139.7784, 0.654902, 0, 0.3411765, 1,
4, 5.393939, -141.1749, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.393939, -141.0256, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.393939, -140.8792, 0.5490196, 0, 0.4470588, 1,
4.121212, 5.393939, -140.7355, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.393939, -140.5946, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.393939, -140.4566, 0.654902, 0, 0.3411765, 1,
4.242424, 5.393939, -140.3213, 0.654902, 0, 0.3411765, 1,
4.282828, 5.393939, -140.1889, 0.654902, 0, 0.3411765, 1,
4.323232, 5.393939, -140.0592, 0.654902, 0, 0.3411765, 1,
4.363636, 5.393939, -139.9324, 0.654902, 0, 0.3411765, 1,
4.40404, 5.393939, -139.8084, 0.654902, 0, 0.3411765, 1,
4.444445, 5.393939, -139.6872, 0.654902, 0, 0.3411765, 1,
4.484848, 5.393939, -139.5688, 0.654902, 0, 0.3411765, 1,
4.525252, 5.393939, -139.4531, 0.654902, 0, 0.3411765, 1,
4.565657, 5.393939, -139.3403, 0.654902, 0, 0.3411765, 1,
4.606061, 5.393939, -139.2303, 0.7568628, 0, 0.2392157, 1,
4.646465, 5.393939, -139.1231, 0.7568628, 0, 0.2392157, 1,
4.686869, 5.393939, -139.0188, 0.7568628, 0, 0.2392157, 1,
4.727273, 5.393939, -138.9172, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.393939, -138.8184, 0.7568628, 0, 0.2392157, 1,
4.808081, 5.393939, -138.7224, 0.7568628, 0, 0.2392157, 1,
4.848485, 5.393939, -138.6293, 0.7568628, 0, 0.2392157, 1,
4.888889, 5.393939, -138.5389, 0.7568628, 0, 0.2392157, 1,
4.929293, 5.393939, -138.4513, 0.7568628, 0, 0.2392157, 1,
4.969697, 5.393939, -138.3666, 0.7568628, 0, 0.2392157, 1,
5.010101, 5.393939, -138.2846, 0.7568628, 0, 0.2392157, 1,
5.050505, 5.393939, -138.2055, 0.7568628, 0, 0.2392157, 1,
5.090909, 5.393939, -138.1292, 0.7568628, 0, 0.2392157, 1,
5.131313, 5.393939, -138.0556, 0.7568628, 0, 0.2392157, 1,
5.171717, 5.393939, -137.9849, 0.8588235, 0, 0.1372549, 1,
5.212121, 5.393939, -137.917, 0.8588235, 0, 0.1372549, 1,
5.252525, 5.393939, -137.8519, 0.8588235, 0, 0.1372549, 1,
5.292929, 5.393939, -137.7896, 0.8588235, 0, 0.1372549, 1,
5.333333, 5.393939, -137.7301, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.393939, -137.6734, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.393939, -137.6195, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.393939, -137.5684, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.393939, -137.5201, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.393939, -137.4747, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.393939, -137.432, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.393939, -137.3921, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.393939, -137.3551, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.393939, -137.3208, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.393939, -137.2894, 0.8588235, 0, 0.1372549, 1,
5.777778, 5.393939, -137.2607, 0.8588235, 0, 0.1372549, 1,
5.818182, 5.393939, -137.2349, 0.8588235, 0, 0.1372549, 1,
5.858586, 5.393939, -137.2119, 0.8588235, 0, 0.1372549, 1,
5.89899, 5.393939, -137.1916, 0.8588235, 0, 0.1372549, 1,
5.939394, 5.393939, -137.1742, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.393939, -137.1596, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.393939, -137.1478, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.393939, -137.1388, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.393939, -137.1326, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.393939, -137.1292, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.393939, -137.1286, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.393939, -137.1308, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.393939, -137.1359, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.393939, -137.1437, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.393939, -137.1543, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.393939, -137.1678, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.393939, -137.184, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.393939, -137.2031, 0.8588235, 0, 0.1372549, 1,
6.505051, 5.393939, -137.2249, 0.8588235, 0, 0.1372549, 1,
6.545455, 5.393939, -137.2496, 0.8588235, 0, 0.1372549, 1,
6.585859, 5.393939, -137.2771, 0.8588235, 0, 0.1372549, 1,
6.626263, 5.393939, -137.3073, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.393939, -137.3404, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.393939, -137.3763, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.393939, -137.415, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.393939, -137.4565, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.393939, -137.5008, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.393939, -137.5479, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.393939, -137.5978, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.393939, -137.6505, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.393939, -137.7061, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.393939, -137.7644, 0.8588235, 0, 0.1372549, 1,
7.070707, 5.393939, -137.8255, 0.8588235, 0, 0.1372549, 1,
7.111111, 5.393939, -137.8895, 0.8588235, 0, 0.1372549, 1,
7.151515, 5.393939, -137.9562, 0.8588235, 0, 0.1372549, 1,
7.191919, 5.393939, -138.0257, 0.7568628, 0, 0.2392157, 1,
7.232323, 5.393939, -138.0981, 0.7568628, 0, 0.2392157, 1,
7.272727, 5.393939, -138.1733, 0.7568628, 0, 0.2392157, 1,
7.313131, 5.393939, -138.2512, 0.7568628, 0, 0.2392157, 1,
7.353535, 5.393939, -138.332, 0.7568628, 0, 0.2392157, 1,
7.393939, 5.393939, -138.4156, 0.7568628, 0, 0.2392157, 1,
7.434343, 5.393939, -138.502, 0.7568628, 0, 0.2392157, 1,
7.474748, 5.393939, -138.5912, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.393939, -138.6832, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.393939, -138.778, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.393939, -138.8756, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.393939, -138.976, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.393939, -139.0792, 0.7568628, 0, 0.2392157, 1,
7.717172, 5.393939, -139.1852, 0.7568628, 0, 0.2392157, 1,
7.757576, 5.393939, -139.294, 0.654902, 0, 0.3411765, 1,
7.79798, 5.393939, -139.4057, 0.654902, 0, 0.3411765, 1,
7.838384, 5.393939, -139.5201, 0.654902, 0, 0.3411765, 1,
7.878788, 5.393939, -139.6374, 0.654902, 0, 0.3411765, 1,
7.919192, 5.393939, -139.7574, 0.654902, 0, 0.3411765, 1,
7.959596, 5.393939, -139.8803, 0.654902, 0, 0.3411765, 1,
8, 5.393939, -140.0059, 0.654902, 0, 0.3411765, 1,
4, 5.434343, -141.3856, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.434343, -141.2386, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.434343, -141.0943, 0.5490196, 0, 0.4470588, 1,
4.121212, 5.434343, -140.9527, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.434343, -140.8139, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.434343, -140.6779, 0.5490196, 0, 0.4470588, 1,
4.242424, 5.434343, -140.5447, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.434343, -140.4142, 0.654902, 0, 0.3411765, 1,
4.323232, 5.434343, -140.2865, 0.654902, 0, 0.3411765, 1,
4.363636, 5.434343, -140.1615, 0.654902, 0, 0.3411765, 1,
4.40404, 5.434343, -140.0393, 0.654902, 0, 0.3411765, 1,
4.444445, 5.434343, -139.9199, 0.654902, 0, 0.3411765, 1,
4.484848, 5.434343, -139.8033, 0.654902, 0, 0.3411765, 1,
4.525252, 5.434343, -139.6894, 0.654902, 0, 0.3411765, 1,
4.565657, 5.434343, -139.5782, 0.654902, 0, 0.3411765, 1,
4.606061, 5.434343, -139.4698, 0.654902, 0, 0.3411765, 1,
4.646465, 5.434343, -139.3642, 0.654902, 0, 0.3411765, 1,
4.686869, 5.434343, -139.2614, 0.654902, 0, 0.3411765, 1,
4.727273, 5.434343, -139.1613, 0.7568628, 0, 0.2392157, 1,
4.767677, 5.434343, -139.064, 0.7568628, 0, 0.2392157, 1,
4.808081, 5.434343, -138.9695, 0.7568628, 0, 0.2392157, 1,
4.848485, 5.434343, -138.8777, 0.7568628, 0, 0.2392157, 1,
4.888889, 5.434343, -138.7887, 0.7568628, 0, 0.2392157, 1,
4.929293, 5.434343, -138.7024, 0.7568628, 0, 0.2392157, 1,
4.969697, 5.434343, -138.6189, 0.7568628, 0, 0.2392157, 1,
5.010101, 5.434343, -138.5382, 0.7568628, 0, 0.2392157, 1,
5.050505, 5.434343, -138.4602, 0.7568628, 0, 0.2392157, 1,
5.090909, 5.434343, -138.385, 0.7568628, 0, 0.2392157, 1,
5.131313, 5.434343, -138.3126, 0.7568628, 0, 0.2392157, 1,
5.171717, 5.434343, -138.2429, 0.7568628, 0, 0.2392157, 1,
5.212121, 5.434343, -138.176, 0.7568628, 0, 0.2392157, 1,
5.252525, 5.434343, -138.1118, 0.7568628, 0, 0.2392157, 1,
5.292929, 5.434343, -138.0504, 0.7568628, 0, 0.2392157, 1,
5.333333, 5.434343, -137.9918, 0.8588235, 0, 0.1372549, 1,
5.373737, 5.434343, -137.936, 0.8588235, 0, 0.1372549, 1,
5.414141, 5.434343, -137.8829, 0.8588235, 0, 0.1372549, 1,
5.454545, 5.434343, -137.8326, 0.8588235, 0, 0.1372549, 1,
5.494949, 5.434343, -137.785, 0.8588235, 0, 0.1372549, 1,
5.535354, 5.434343, -137.7402, 0.8588235, 0, 0.1372549, 1,
5.575758, 5.434343, -137.6981, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.434343, -137.6589, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.434343, -137.6224, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.434343, -137.5886, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.434343, -137.5576, 0.8588235, 0, 0.1372549, 1,
5.777778, 5.434343, -137.5294, 0.8588235, 0, 0.1372549, 1,
5.818182, 5.434343, -137.504, 0.8588235, 0, 0.1372549, 1,
5.858586, 5.434343, -137.4813, 0.8588235, 0, 0.1372549, 1,
5.89899, 5.434343, -137.4613, 0.8588235, 0, 0.1372549, 1,
5.939394, 5.434343, -137.4442, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.434343, -137.4298, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.434343, -137.4182, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.434343, -137.4093, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.434343, -137.4032, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.434343, -137.3998, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.434343, -137.3993, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.434343, -137.4015, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.434343, -137.4064, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.434343, -137.4141, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.434343, -137.4246, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.434343, -137.4378, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.434343, -137.4538, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.434343, -137.4726, 0.8588235, 0, 0.1372549, 1,
6.505051, 5.434343, -137.4942, 0.8588235, 0, 0.1372549, 1,
6.545455, 5.434343, -137.5184, 0.8588235, 0, 0.1372549, 1,
6.585859, 5.434343, -137.5455, 0.8588235, 0, 0.1372549, 1,
6.626263, 5.434343, -137.5753, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.434343, -137.6079, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.434343, -137.6433, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.434343, -137.6814, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.434343, -137.7223, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.434343, -137.7659, 0.8588235, 0, 0.1372549, 1,
6.868687, 5.434343, -137.8123, 0.8588235, 0, 0.1372549, 1,
6.909091, 5.434343, -137.8615, 0.8588235, 0, 0.1372549, 1,
6.949495, 5.434343, -137.9134, 0.8588235, 0, 0.1372549, 1,
6.989899, 5.434343, -137.9681, 0.8588235, 0, 0.1372549, 1,
7.030303, 5.434343, -138.0256, 0.7568628, 0, 0.2392157, 1,
7.070707, 5.434343, -138.0858, 0.7568628, 0, 0.2392157, 1,
7.111111, 5.434343, -138.1488, 0.7568628, 0, 0.2392157, 1,
7.151515, 5.434343, -138.2146, 0.7568628, 0, 0.2392157, 1,
7.191919, 5.434343, -138.2831, 0.7568628, 0, 0.2392157, 1,
7.232323, 5.434343, -138.3544, 0.7568628, 0, 0.2392157, 1,
7.272727, 5.434343, -138.4284, 0.7568628, 0, 0.2392157, 1,
7.313131, 5.434343, -138.5052, 0.7568628, 0, 0.2392157, 1,
7.353535, 5.434343, -138.5848, 0.7568628, 0, 0.2392157, 1,
7.393939, 5.434343, -138.6672, 0.7568628, 0, 0.2392157, 1,
7.434343, 5.434343, -138.7523, 0.7568628, 0, 0.2392157, 1,
7.474748, 5.434343, -138.8401, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.434343, -138.9308, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.434343, -139.0242, 0.7568628, 0, 0.2392157, 1,
7.59596, 5.434343, -139.1203, 0.7568628, 0, 0.2392157, 1,
7.636364, 5.434343, -139.2193, 0.7568628, 0, 0.2392157, 1,
7.676768, 5.434343, -139.3209, 0.654902, 0, 0.3411765, 1,
7.717172, 5.434343, -139.4254, 0.654902, 0, 0.3411765, 1,
7.757576, 5.434343, -139.5326, 0.654902, 0, 0.3411765, 1,
7.79798, 5.434343, -139.6426, 0.654902, 0, 0.3411765, 1,
7.838384, 5.434343, -139.7553, 0.654902, 0, 0.3411765, 1,
7.878788, 5.434343, -139.8708, 0.654902, 0, 0.3411765, 1,
7.919192, 5.434343, -139.9891, 0.654902, 0, 0.3411765, 1,
7.959596, 5.434343, -140.1101, 0.654902, 0, 0.3411765, 1,
8, 5.434343, -140.2339, 0.654902, 0, 0.3411765, 1,
4, 5.474748, -141.5971, 0.5490196, 0, 0.4470588, 1,
4.040404, 5.474748, -141.4522, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.474748, -141.3101, 0.5490196, 0, 0.4470588, 1,
4.121212, 5.474748, -141.1706, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.474748, -141.0339, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.474748, -140.8999, 0.5490196, 0, 0.4470588, 1,
4.242424, 5.474748, -140.7686, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.474748, -140.64, 0.5490196, 0, 0.4470588, 1,
4.323232, 5.474748, -140.5142, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.474748, -140.3911, 0.654902, 0, 0.3411765, 1,
4.40404, 5.474748, -140.2707, 0.654902, 0, 0.3411765, 1,
4.444445, 5.474748, -140.153, 0.654902, 0, 0.3411765, 1,
4.484848, 5.474748, -140.0381, 0.654902, 0, 0.3411765, 1,
4.525252, 5.474748, -139.9258, 0.654902, 0, 0.3411765, 1,
4.565657, 5.474748, -139.8163, 0.654902, 0, 0.3411765, 1,
4.606061, 5.474748, -139.7095, 0.654902, 0, 0.3411765, 1,
4.646465, 5.474748, -139.6055, 0.654902, 0, 0.3411765, 1,
4.686869, 5.474748, -139.5042, 0.654902, 0, 0.3411765, 1,
4.727273, 5.474748, -139.4056, 0.654902, 0, 0.3411765, 1,
4.767677, 5.474748, -139.3097, 0.654902, 0, 0.3411765, 1,
4.808081, 5.474748, -139.2165, 0.7568628, 0, 0.2392157, 1,
4.848485, 5.474748, -139.1261, 0.7568628, 0, 0.2392157, 1,
4.888889, 5.474748, -139.0384, 0.7568628, 0, 0.2392157, 1,
4.929293, 5.474748, -138.9534, 0.7568628, 0, 0.2392157, 1,
4.969697, 5.474748, -138.8711, 0.7568628, 0, 0.2392157, 1,
5.010101, 5.474748, -138.7916, 0.7568628, 0, 0.2392157, 1,
5.050505, 5.474748, -138.7147, 0.7568628, 0, 0.2392157, 1,
5.090909, 5.474748, -138.6406, 0.7568628, 0, 0.2392157, 1,
5.131313, 5.474748, -138.5693, 0.7568628, 0, 0.2392157, 1,
5.171717, 5.474748, -138.5006, 0.7568628, 0, 0.2392157, 1,
5.212121, 5.474748, -138.4347, 0.7568628, 0, 0.2392157, 1,
5.252525, 5.474748, -138.3715, 0.7568628, 0, 0.2392157, 1,
5.292929, 5.474748, -138.311, 0.7568628, 0, 0.2392157, 1,
5.333333, 5.474748, -138.2533, 0.7568628, 0, 0.2392157, 1,
5.373737, 5.474748, -138.1982, 0.7568628, 0, 0.2392157, 1,
5.414141, 5.474748, -138.1459, 0.7568628, 0, 0.2392157, 1,
5.454545, 5.474748, -138.0963, 0.7568628, 0, 0.2392157, 1,
5.494949, 5.474748, -138.0495, 0.7568628, 0, 0.2392157, 1,
5.535354, 5.474748, -138.0053, 0.7568628, 0, 0.2392157, 1,
5.575758, 5.474748, -137.9639, 0.8588235, 0, 0.1372549, 1,
5.616162, 5.474748, -137.9252, 0.8588235, 0, 0.1372549, 1,
5.656566, 5.474748, -137.8892, 0.8588235, 0, 0.1372549, 1,
5.69697, 5.474748, -137.856, 0.8588235, 0, 0.1372549, 1,
5.737374, 5.474748, -137.8255, 0.8588235, 0, 0.1372549, 1,
5.777778, 5.474748, -137.7977, 0.8588235, 0, 0.1372549, 1,
5.818182, 5.474748, -137.7726, 0.8588235, 0, 0.1372549, 1,
5.858586, 5.474748, -137.7502, 0.8588235, 0, 0.1372549, 1,
5.89899, 5.474748, -137.7306, 0.8588235, 0, 0.1372549, 1,
5.939394, 5.474748, -137.7137, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.474748, -137.6995, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.474748, -137.688, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.474748, -137.6793, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.474748, -137.6733, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.474748, -137.67, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.474748, -137.6694, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.474748, -137.6716, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.474748, -137.6765, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.474748, -137.6841, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.474748, -137.6944, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.474748, -137.7074, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.474748, -137.7232, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.474748, -137.7417, 0.8588235, 0, 0.1372549, 1,
6.505051, 5.474748, -137.7629, 0.8588235, 0, 0.1372549, 1,
6.545455, 5.474748, -137.7868, 0.8588235, 0, 0.1372549, 1,
6.585859, 5.474748, -137.8135, 0.8588235, 0, 0.1372549, 1,
6.626263, 5.474748, -137.8429, 0.8588235, 0, 0.1372549, 1,
6.666667, 5.474748, -137.875, 0.8588235, 0, 0.1372549, 1,
6.707071, 5.474748, -137.9098, 0.8588235, 0, 0.1372549, 1,
6.747475, 5.474748, -137.9474, 0.8588235, 0, 0.1372549, 1,
6.787879, 5.474748, -137.9877, 0.8588235, 0, 0.1372549, 1,
6.828283, 5.474748, -138.0307, 0.7568628, 0, 0.2392157, 1,
6.868687, 5.474748, -138.0764, 0.7568628, 0, 0.2392157, 1,
6.909091, 5.474748, -138.1249, 0.7568628, 0, 0.2392157, 1,
6.949495, 5.474748, -138.176, 0.7568628, 0, 0.2392157, 1,
6.989899, 5.474748, -138.2299, 0.7568628, 0, 0.2392157, 1,
7.030303, 5.474748, -138.2865, 0.7568628, 0, 0.2392157, 1,
7.070707, 5.474748, -138.3459, 0.7568628, 0, 0.2392157, 1,
7.111111, 5.474748, -138.408, 0.7568628, 0, 0.2392157, 1,
7.151515, 5.474748, -138.4727, 0.7568628, 0, 0.2392157, 1,
7.191919, 5.474748, -138.5403, 0.7568628, 0, 0.2392157, 1,
7.232323, 5.474748, -138.6105, 0.7568628, 0, 0.2392157, 1,
7.272727, 5.474748, -138.6835, 0.7568628, 0, 0.2392157, 1,
7.313131, 5.474748, -138.7591, 0.7568628, 0, 0.2392157, 1,
7.353535, 5.474748, -138.8375, 0.7568628, 0, 0.2392157, 1,
7.393939, 5.474748, -138.9187, 0.7568628, 0, 0.2392157, 1,
7.434343, 5.474748, -139.0025, 0.7568628, 0, 0.2392157, 1,
7.474748, 5.474748, -139.0891, 0.7568628, 0, 0.2392157, 1,
7.515152, 5.474748, -139.1784, 0.7568628, 0, 0.2392157, 1,
7.555555, 5.474748, -139.2704, 0.654902, 0, 0.3411765, 1,
7.59596, 5.474748, -139.3652, 0.654902, 0, 0.3411765, 1,
7.636364, 5.474748, -139.4626, 0.654902, 0, 0.3411765, 1,
7.676768, 5.474748, -139.5628, 0.654902, 0, 0.3411765, 1,
7.717172, 5.474748, -139.6657, 0.654902, 0, 0.3411765, 1,
7.757576, 5.474748, -139.7714, 0.654902, 0, 0.3411765, 1,
7.79798, 5.474748, -139.8797, 0.654902, 0, 0.3411765, 1,
7.838384, 5.474748, -139.9908, 0.654902, 0, 0.3411765, 1,
7.878788, 5.474748, -140.1046, 0.654902, 0, 0.3411765, 1,
7.919192, 5.474748, -140.2212, 0.654902, 0, 0.3411765, 1,
7.959596, 5.474748, -140.3404, 0.654902, 0, 0.3411765, 1,
8, 5.474748, -140.4624, 0.654902, 0, 0.3411765, 1,
4, 5.515152, -141.8094, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.515152, -141.6666, 0.5490196, 0, 0.4470588, 1,
4.080808, 5.515152, -141.5265, 0.5490196, 0, 0.4470588, 1,
4.121212, 5.515152, -141.3891, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.515152, -141.2544, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.515152, -141.1223, 0.5490196, 0, 0.4470588, 1,
4.242424, 5.515152, -140.993, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.515152, -140.8663, 0.5490196, 0, 0.4470588, 1,
4.323232, 5.515152, -140.7423, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.515152, -140.6209, 0.5490196, 0, 0.4470588, 1,
4.40404, 5.515152, -140.5023, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.515152, -140.3864, 0.654902, 0, 0.3411765, 1,
4.484848, 5.515152, -140.2731, 0.654902, 0, 0.3411765, 1,
4.525252, 5.515152, -140.1625, 0.654902, 0, 0.3411765, 1,
4.565657, 5.515152, -140.0546, 0.654902, 0, 0.3411765, 1,
4.606061, 5.515152, -139.9494, 0.654902, 0, 0.3411765, 1,
4.646465, 5.515152, -139.8468, 0.654902, 0, 0.3411765, 1,
4.686869, 5.515152, -139.747, 0.654902, 0, 0.3411765, 1,
4.727273, 5.515152, -139.6498, 0.654902, 0, 0.3411765, 1,
4.767677, 5.515152, -139.5553, 0.654902, 0, 0.3411765, 1,
4.808081, 5.515152, -139.4635, 0.654902, 0, 0.3411765, 1,
4.848485, 5.515152, -139.3744, 0.654902, 0, 0.3411765, 1,
4.888889, 5.515152, -139.288, 0.654902, 0, 0.3411765, 1,
4.929293, 5.515152, -139.2043, 0.7568628, 0, 0.2392157, 1,
4.969697, 5.515152, -139.1232, 0.7568628, 0, 0.2392157, 1,
5.010101, 5.515152, -139.0448, 0.7568628, 0, 0.2392157, 1,
5.050505, 5.515152, -138.9691, 0.7568628, 0, 0.2392157, 1,
5.090909, 5.515152, -138.8961, 0.7568628, 0, 0.2392157, 1,
5.131313, 5.515152, -138.8258, 0.7568628, 0, 0.2392157, 1,
5.171717, 5.515152, -138.7581, 0.7568628, 0, 0.2392157, 1,
5.212121, 5.515152, -138.6931, 0.7568628, 0, 0.2392157, 1,
5.252525, 5.515152, -138.6309, 0.7568628, 0, 0.2392157, 1,
5.292929, 5.515152, -138.5713, 0.7568628, 0, 0.2392157, 1,
5.333333, 5.515152, -138.5143, 0.7568628, 0, 0.2392157, 1,
5.373737, 5.515152, -138.4601, 0.7568628, 0, 0.2392157, 1,
5.414141, 5.515152, -138.4086, 0.7568628, 0, 0.2392157, 1,
5.454545, 5.515152, -138.3597, 0.7568628, 0, 0.2392157, 1,
5.494949, 5.515152, -138.3135, 0.7568628, 0, 0.2392157, 1,
5.535354, 5.515152, -138.27, 0.7568628, 0, 0.2392157, 1,
5.575758, 5.515152, -138.2292, 0.7568628, 0, 0.2392157, 1,
5.616162, 5.515152, -138.1911, 0.7568628, 0, 0.2392157, 1,
5.656566, 5.515152, -138.1556, 0.7568628, 0, 0.2392157, 1,
5.69697, 5.515152, -138.1229, 0.7568628, 0, 0.2392157, 1,
5.737374, 5.515152, -138.0928, 0.7568628, 0, 0.2392157, 1,
5.777778, 5.515152, -138.0654, 0.7568628, 0, 0.2392157, 1,
5.818182, 5.515152, -138.0407, 0.7568628, 0, 0.2392157, 1,
5.858586, 5.515152, -138.0187, 0.7568628, 0, 0.2392157, 1,
5.89899, 5.515152, -137.9993, 0.7568628, 0, 0.2392157, 1,
5.939394, 5.515152, -137.9827, 0.8588235, 0, 0.1372549, 1,
5.979798, 5.515152, -137.9687, 0.8588235, 0, 0.1372549, 1,
6.020202, 5.515152, -137.9574, 0.8588235, 0, 0.1372549, 1,
6.060606, 5.515152, -137.9488, 0.8588235, 0, 0.1372549, 1,
6.10101, 5.515152, -137.9428, 0.8588235, 0, 0.1372549, 1,
6.141414, 5.515152, -137.9396, 0.8588235, 0, 0.1372549, 1,
6.181818, 5.515152, -137.939, 0.8588235, 0, 0.1372549, 1,
6.222222, 5.515152, -137.9412, 0.8588235, 0, 0.1372549, 1,
6.262626, 5.515152, -137.946, 0.8588235, 0, 0.1372549, 1,
6.30303, 5.515152, -137.9535, 0.8588235, 0, 0.1372549, 1,
6.343434, 5.515152, -137.9636, 0.8588235, 0, 0.1372549, 1,
6.383838, 5.515152, -137.9765, 0.8588235, 0, 0.1372549, 1,
6.424242, 5.515152, -137.992, 0.8588235, 0, 0.1372549, 1,
6.464646, 5.515152, -138.0103, 0.7568628, 0, 0.2392157, 1,
6.505051, 5.515152, -138.0312, 0.7568628, 0, 0.2392157, 1,
6.545455, 5.515152, -138.0547, 0.7568628, 0, 0.2392157, 1,
6.585859, 5.515152, -138.081, 0.7568628, 0, 0.2392157, 1,
6.626263, 5.515152, -138.11, 0.7568628, 0, 0.2392157, 1,
6.666667, 5.515152, -138.1416, 0.7568628, 0, 0.2392157, 1,
6.707071, 5.515152, -138.1759, 0.7568628, 0, 0.2392157, 1,
6.747475, 5.515152, -138.213, 0.7568628, 0, 0.2392157, 1,
6.787879, 5.515152, -138.2527, 0.7568628, 0, 0.2392157, 1,
6.828283, 5.515152, -138.295, 0.7568628, 0, 0.2392157, 1,
6.868687, 5.515152, -138.3401, 0.7568628, 0, 0.2392157, 1,
6.909091, 5.515152, -138.3878, 0.7568628, 0, 0.2392157, 1,
6.949495, 5.515152, -138.4382, 0.7568628, 0, 0.2392157, 1,
6.989899, 5.515152, -138.4914, 0.7568628, 0, 0.2392157, 1,
7.030303, 5.515152, -138.5471, 0.7568628, 0, 0.2392157, 1,
7.070707, 5.515152, -138.6056, 0.7568628, 0, 0.2392157, 1,
7.111111, 5.515152, -138.6668, 0.7568628, 0, 0.2392157, 1,
7.151515, 5.515152, -138.7306, 0.7568628, 0, 0.2392157, 1,
7.191919, 5.515152, -138.7972, 0.7568628, 0, 0.2392157, 1,
7.232323, 5.515152, -138.8664, 0.7568628, 0, 0.2392157, 1,
7.272727, 5.515152, -138.9383, 0.7568628, 0, 0.2392157, 1,
7.313131, 5.515152, -139.0128, 0.7568628, 0, 0.2392157, 1,
7.353535, 5.515152, -139.0901, 0.7568628, 0, 0.2392157, 1,
7.393939, 5.515152, -139.1701, 0.7568628, 0, 0.2392157, 1,
7.434343, 5.515152, -139.2527, 0.654902, 0, 0.3411765, 1,
7.474748, 5.515152, -139.338, 0.654902, 0, 0.3411765, 1,
7.515152, 5.515152, -139.426, 0.654902, 0, 0.3411765, 1,
7.555555, 5.515152, -139.5167, 0.654902, 0, 0.3411765, 1,
7.59596, 5.515152, -139.61, 0.654902, 0, 0.3411765, 1,
7.636364, 5.515152, -139.7061, 0.654902, 0, 0.3411765, 1,
7.676768, 5.515152, -139.8048, 0.654902, 0, 0.3411765, 1,
7.717172, 5.515152, -139.9062, 0.654902, 0, 0.3411765, 1,
7.757576, 5.515152, -140.0103, 0.654902, 0, 0.3411765, 1,
7.79798, 5.515152, -140.1171, 0.654902, 0, 0.3411765, 1,
7.838384, 5.515152, -140.2266, 0.654902, 0, 0.3411765, 1,
7.878788, 5.515152, -140.3387, 0.654902, 0, 0.3411765, 1,
7.919192, 5.515152, -140.4535, 0.654902, 0, 0.3411765, 1,
7.959596, 5.515152, -140.571, 0.5490196, 0, 0.4470588, 1,
8, 5.515152, -140.6913, 0.5490196, 0, 0.4470588, 1,
4, 5.555555, -142.0224, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.555555, -141.8817, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.555555, -141.7436, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.555555, -141.6082, 0.5490196, 0, 0.4470588, 1,
4.161616, 5.555555, -141.4754, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.555555, -141.3453, 0.5490196, 0, 0.4470588, 1,
4.242424, 5.555555, -141.2178, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.555555, -141.0929, 0.5490196, 0, 0.4470588, 1,
4.323232, 5.555555, -140.9707, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.555555, -140.8512, 0.5490196, 0, 0.4470588, 1,
4.40404, 5.555555, -140.7342, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.555555, -140.62, 0.5490196, 0, 0.4470588, 1,
4.484848, 5.555555, -140.5083, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.555555, -140.3994, 0.654902, 0, 0.3411765, 1,
4.565657, 5.555555, -140.293, 0.654902, 0, 0.3411765, 1,
4.606061, 5.555555, -140.1893, 0.654902, 0, 0.3411765, 1,
4.646465, 5.555555, -140.0883, 0.654902, 0, 0.3411765, 1,
4.686869, 5.555555, -139.9899, 0.654902, 0, 0.3411765, 1,
4.727273, 5.555555, -139.8941, 0.654902, 0, 0.3411765, 1,
4.767677, 5.555555, -139.801, 0.654902, 0, 0.3411765, 1,
4.808081, 5.555555, -139.7105, 0.654902, 0, 0.3411765, 1,
4.848485, 5.555555, -139.6227, 0.654902, 0, 0.3411765, 1,
4.888889, 5.555555, -139.5375, 0.654902, 0, 0.3411765, 1,
4.929293, 5.555555, -139.455, 0.654902, 0, 0.3411765, 1,
4.969697, 5.555555, -139.3751, 0.654902, 0, 0.3411765, 1,
5.010101, 5.555555, -139.2978, 0.654902, 0, 0.3411765, 1,
5.050505, 5.555555, -139.2232, 0.7568628, 0, 0.2392157, 1,
5.090909, 5.555555, -139.1513, 0.7568628, 0, 0.2392157, 1,
5.131313, 5.555555, -139.082, 0.7568628, 0, 0.2392157, 1,
5.171717, 5.555555, -139.0153, 0.7568628, 0, 0.2392157, 1,
5.212121, 5.555555, -138.9513, 0.7568628, 0, 0.2392157, 1,
5.252525, 5.555555, -138.8899, 0.7568628, 0, 0.2392157, 1,
5.292929, 5.555555, -138.8312, 0.7568628, 0, 0.2392157, 1,
5.333333, 5.555555, -138.7751, 0.7568628, 0, 0.2392157, 1,
5.373737, 5.555555, -138.7216, 0.7568628, 0, 0.2392157, 1,
5.414141, 5.555555, -138.6708, 0.7568628, 0, 0.2392157, 1,
5.454545, 5.555555, -138.6227, 0.7568628, 0, 0.2392157, 1,
5.494949, 5.555555, -138.5772, 0.7568628, 0, 0.2392157, 1,
5.535354, 5.555555, -138.5343, 0.7568628, 0, 0.2392157, 1,
5.575758, 5.555555, -138.4941, 0.7568628, 0, 0.2392157, 1,
5.616162, 5.555555, -138.4565, 0.7568628, 0, 0.2392157, 1,
5.656566, 5.555555, -138.4216, 0.7568628, 0, 0.2392157, 1,
5.69697, 5.555555, -138.3893, 0.7568628, 0, 0.2392157, 1,
5.737374, 5.555555, -138.3596, 0.7568628, 0, 0.2392157, 1,
5.777778, 5.555555, -138.3326, 0.7568628, 0, 0.2392157, 1,
5.818182, 5.555555, -138.3083, 0.7568628, 0, 0.2392157, 1,
5.858586, 5.555555, -138.2866, 0.7568628, 0, 0.2392157, 1,
5.89899, 5.555555, -138.2675, 0.7568628, 0, 0.2392157, 1,
5.939394, 5.555555, -138.2511, 0.7568628, 0, 0.2392157, 1,
5.979798, 5.555555, -138.2373, 0.7568628, 0, 0.2392157, 1,
6.020202, 5.555555, -138.2262, 0.7568628, 0, 0.2392157, 1,
6.060606, 5.555555, -138.2177, 0.7568628, 0, 0.2392157, 1,
6.10101, 5.555555, -138.2119, 0.7568628, 0, 0.2392157, 1,
6.141414, 5.555555, -138.2086, 0.7568628, 0, 0.2392157, 1,
6.181818, 5.555555, -138.2081, 0.7568628, 0, 0.2392157, 1,
6.222222, 5.555555, -138.2102, 0.7568628, 0, 0.2392157, 1,
6.262626, 5.555555, -138.2149, 0.7568628, 0, 0.2392157, 1,
6.30303, 5.555555, -138.2223, 0.7568628, 0, 0.2392157, 1,
6.343434, 5.555555, -138.2323, 0.7568628, 0, 0.2392157, 1,
6.383838, 5.555555, -138.245, 0.7568628, 0, 0.2392157, 1,
6.424242, 5.555555, -138.2603, 0.7568628, 0, 0.2392157, 1,
6.464646, 5.555555, -138.2783, 0.7568628, 0, 0.2392157, 1,
6.505051, 5.555555, -138.2989, 0.7568628, 0, 0.2392157, 1,
6.545455, 5.555555, -138.3221, 0.7568628, 0, 0.2392157, 1,
6.585859, 5.555555, -138.348, 0.7568628, 0, 0.2392157, 1,
6.626263, 5.555555, -138.3766, 0.7568628, 0, 0.2392157, 1,
6.666667, 5.555555, -138.4077, 0.7568628, 0, 0.2392157, 1,
6.707071, 5.555555, -138.4416, 0.7568628, 0, 0.2392157, 1,
6.747475, 5.555555, -138.4781, 0.7568628, 0, 0.2392157, 1,
6.787879, 5.555555, -138.5172, 0.7568628, 0, 0.2392157, 1,
6.828283, 5.555555, -138.5589, 0.7568628, 0, 0.2392157, 1,
6.868687, 5.555555, -138.6033, 0.7568628, 0, 0.2392157, 1,
6.909091, 5.555555, -138.6504, 0.7568628, 0, 0.2392157, 1,
6.949495, 5.555555, -138.7001, 0.7568628, 0, 0.2392157, 1,
6.989899, 5.555555, -138.7524, 0.7568628, 0, 0.2392157, 1,
7.030303, 5.555555, -138.8074, 0.7568628, 0, 0.2392157, 1,
7.070707, 5.555555, -138.865, 0.7568628, 0, 0.2392157, 1,
7.111111, 5.555555, -138.9253, 0.7568628, 0, 0.2392157, 1,
7.151515, 5.555555, -138.9882, 0.7568628, 0, 0.2392157, 1,
7.191919, 5.555555, -139.0538, 0.7568628, 0, 0.2392157, 1,
7.232323, 5.555555, -139.122, 0.7568628, 0, 0.2392157, 1,
7.272727, 5.555555, -139.1929, 0.7568628, 0, 0.2392157, 1,
7.313131, 5.555555, -139.2663, 0.654902, 0, 0.3411765, 1,
7.353535, 5.555555, -139.3425, 0.654902, 0, 0.3411765, 1,
7.393939, 5.555555, -139.4213, 0.654902, 0, 0.3411765, 1,
7.434343, 5.555555, -139.5027, 0.654902, 0, 0.3411765, 1,
7.474748, 5.555555, -139.5868, 0.654902, 0, 0.3411765, 1,
7.515152, 5.555555, -139.6735, 0.654902, 0, 0.3411765, 1,
7.555555, 5.555555, -139.7629, 0.654902, 0, 0.3411765, 1,
7.59596, 5.555555, -139.8549, 0.654902, 0, 0.3411765, 1,
7.636364, 5.555555, -139.9495, 0.654902, 0, 0.3411765, 1,
7.676768, 5.555555, -140.0468, 0.654902, 0, 0.3411765, 1,
7.717172, 5.555555, -140.1468, 0.654902, 0, 0.3411765, 1,
7.757576, 5.555555, -140.2494, 0.654902, 0, 0.3411765, 1,
7.79798, 5.555555, -140.3546, 0.654902, 0, 0.3411765, 1,
7.838384, 5.555555, -140.4625, 0.654902, 0, 0.3411765, 1,
7.878788, 5.555555, -140.573, 0.5490196, 0, 0.4470588, 1,
7.919192, 5.555555, -140.6862, 0.5490196, 0, 0.4470588, 1,
7.959596, 5.555555, -140.802, 0.5490196, 0, 0.4470588, 1,
8, 5.555555, -140.9204, 0.5490196, 0, 0.4470588, 1,
4, 5.59596, -142.236, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.59596, -142.0973, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.59596, -141.9612, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.59596, -141.8277, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.59596, -141.6969, 0.5490196, 0, 0.4470588, 1,
4.20202, 5.59596, -141.5686, 0.5490196, 0, 0.4470588, 1,
4.242424, 5.59596, -141.4429, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.59596, -141.3199, 0.5490196, 0, 0.4470588, 1,
4.323232, 5.59596, -141.1994, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.59596, -141.0816, 0.5490196, 0, 0.4470588, 1,
4.40404, 5.59596, -140.9664, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.59596, -140.8537, 0.5490196, 0, 0.4470588, 1,
4.484848, 5.59596, -140.7437, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.59596, -140.6363, 0.5490196, 0, 0.4470588, 1,
4.565657, 5.59596, -140.5315, 0.5490196, 0, 0.4470588, 1,
4.606061, 5.59596, -140.4293, 0.654902, 0, 0.3411765, 1,
4.646465, 5.59596, -140.3297, 0.654902, 0, 0.3411765, 1,
4.686869, 5.59596, -140.2327, 0.654902, 0, 0.3411765, 1,
4.727273, 5.59596, -140.1383, 0.654902, 0, 0.3411765, 1,
4.767677, 5.59596, -140.0466, 0.654902, 0, 0.3411765, 1,
4.808081, 5.59596, -139.9574, 0.654902, 0, 0.3411765, 1,
4.848485, 5.59596, -139.8708, 0.654902, 0, 0.3411765, 1,
4.888889, 5.59596, -139.7869, 0.654902, 0, 0.3411765, 1,
4.929293, 5.59596, -139.7055, 0.654902, 0, 0.3411765, 1,
4.969697, 5.59596, -139.6268, 0.654902, 0, 0.3411765, 1,
5.010101, 5.59596, -139.5507, 0.654902, 0, 0.3411765, 1,
5.050505, 5.59596, -139.4771, 0.654902, 0, 0.3411765, 1,
5.090909, 5.59596, -139.4062, 0.654902, 0, 0.3411765, 1,
5.131313, 5.59596, -139.3379, 0.654902, 0, 0.3411765, 1,
5.171717, 5.59596, -139.2722, 0.654902, 0, 0.3411765, 1,
5.212121, 5.59596, -139.2091, 0.7568628, 0, 0.2392157, 1,
5.252525, 5.59596, -139.1486, 0.7568628, 0, 0.2392157, 1,
5.292929, 5.59596, -139.0907, 0.7568628, 0, 0.2392157, 1,
5.333333, 5.59596, -139.0354, 0.7568628, 0, 0.2392157, 1,
5.373737, 5.59596, -138.9827, 0.7568628, 0, 0.2392157, 1,
5.414141, 5.59596, -138.9327, 0.7568628, 0, 0.2392157, 1,
5.454545, 5.59596, -138.8852, 0.7568628, 0, 0.2392157, 1,
5.494949, 5.59596, -138.8403, 0.7568628, 0, 0.2392157, 1,
5.535354, 5.59596, -138.7981, 0.7568628, 0, 0.2392157, 1,
5.575758, 5.59596, -138.7585, 0.7568628, 0, 0.2392157, 1,
5.616162, 5.59596, -138.7214, 0.7568628, 0, 0.2392157, 1,
5.656566, 5.59596, -138.687, 0.7568628, 0, 0.2392157, 1,
5.69697, 5.59596, -138.6552, 0.7568628, 0, 0.2392157, 1,
5.737374, 5.59596, -138.6259, 0.7568628, 0, 0.2392157, 1,
5.777778, 5.59596, -138.5993, 0.7568628, 0, 0.2392157, 1,
5.818182, 5.59596, -138.5753, 0.7568628, 0, 0.2392157, 1,
5.858586, 5.59596, -138.5539, 0.7568628, 0, 0.2392157, 1,
5.89899, 5.59596, -138.5351, 0.7568628, 0, 0.2392157, 1,
5.939394, 5.59596, -138.519, 0.7568628, 0, 0.2392157, 1,
5.979798, 5.59596, -138.5054, 0.7568628, 0, 0.2392157, 1,
6.020202, 5.59596, -138.4944, 0.7568628, 0, 0.2392157, 1,
6.060606, 5.59596, -138.486, 0.7568628, 0, 0.2392157, 1,
6.10101, 5.59596, -138.4803, 0.7568628, 0, 0.2392157, 1,
6.141414, 5.59596, -138.4771, 0.7568628, 0, 0.2392157, 1,
6.181818, 5.59596, -138.4766, 0.7568628, 0, 0.2392157, 1,
6.222222, 5.59596, -138.4787, 0.7568628, 0, 0.2392157, 1,
6.262626, 5.59596, -138.4833, 0.7568628, 0, 0.2392157, 1,
6.30303, 5.59596, -138.4906, 0.7568628, 0, 0.2392157, 1,
6.343434, 5.59596, -138.5005, 0.7568628, 0, 0.2392157, 1,
6.383838, 5.59596, -138.513, 0.7568628, 0, 0.2392157, 1,
6.424242, 5.59596, -138.5281, 0.7568628, 0, 0.2392157, 1,
6.464646, 5.59596, -138.5458, 0.7568628, 0, 0.2392157, 1,
6.505051, 5.59596, -138.5661, 0.7568628, 0, 0.2392157, 1,
6.545455, 5.59596, -138.589, 0.7568628, 0, 0.2392157, 1,
6.585859, 5.59596, -138.6145, 0.7568628, 0, 0.2392157, 1,
6.626263, 5.59596, -138.6426, 0.7568628, 0, 0.2392157, 1,
6.666667, 5.59596, -138.6734, 0.7568628, 0, 0.2392157, 1,
6.707071, 5.59596, -138.7067, 0.7568628, 0, 0.2392157, 1,
6.747475, 5.59596, -138.7427, 0.7568628, 0, 0.2392157, 1,
6.787879, 5.59596, -138.7812, 0.7568628, 0, 0.2392157, 1,
6.828283, 5.59596, -138.8224, 0.7568628, 0, 0.2392157, 1,
6.868687, 5.59596, -138.8661, 0.7568628, 0, 0.2392157, 1,
6.909091, 5.59596, -138.9125, 0.7568628, 0, 0.2392157, 1,
6.949495, 5.59596, -138.9615, 0.7568628, 0, 0.2392157, 1,
6.989899, 5.59596, -139.0131, 0.7568628, 0, 0.2392157, 1,
7.030303, 5.59596, -139.0673, 0.7568628, 0, 0.2392157, 1,
7.070707, 5.59596, -139.1241, 0.7568628, 0, 0.2392157, 1,
7.111111, 5.59596, -139.1835, 0.7568628, 0, 0.2392157, 1,
7.151515, 5.59596, -139.2455, 0.654902, 0, 0.3411765, 1,
7.191919, 5.59596, -139.3101, 0.654902, 0, 0.3411765, 1,
7.232323, 5.59596, -139.3773, 0.654902, 0, 0.3411765, 1,
7.272727, 5.59596, -139.4472, 0.654902, 0, 0.3411765, 1,
7.313131, 5.59596, -139.5196, 0.654902, 0, 0.3411765, 1,
7.353535, 5.59596, -139.5947, 0.654902, 0, 0.3411765, 1,
7.393939, 5.59596, -139.6723, 0.654902, 0, 0.3411765, 1,
7.434343, 5.59596, -139.7526, 0.654902, 0, 0.3411765, 1,
7.474748, 5.59596, -139.8354, 0.654902, 0, 0.3411765, 1,
7.515152, 5.59596, -139.9209, 0.654902, 0, 0.3411765, 1,
7.555555, 5.59596, -140.009, 0.654902, 0, 0.3411765, 1,
7.59596, 5.59596, -140.0997, 0.654902, 0, 0.3411765, 1,
7.636364, 5.59596, -140.193, 0.654902, 0, 0.3411765, 1,
7.676768, 5.59596, -140.2889, 0.654902, 0, 0.3411765, 1,
7.717172, 5.59596, -140.3874, 0.654902, 0, 0.3411765, 1,
7.757576, 5.59596, -140.4885, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.59596, -140.5922, 0.5490196, 0, 0.4470588, 1,
7.838384, 5.59596, -140.6985, 0.5490196, 0, 0.4470588, 1,
7.878788, 5.59596, -140.8075, 0.5490196, 0, 0.4470588, 1,
7.919192, 5.59596, -140.919, 0.5490196, 0, 0.4470588, 1,
7.959596, 5.59596, -141.0331, 0.5490196, 0, 0.4470588, 1,
8, 5.59596, -141.1499, 0.5490196, 0, 0.4470588, 1,
4, 5.636364, -142.4502, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.636364, -142.3135, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.636364, -142.1793, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.636364, -142.0478, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.636364, -141.9188, 0.4470588, 0, 0.5490196, 1,
4.20202, 5.636364, -141.7923, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.636364, -141.6685, 0.5490196, 0, 0.4470588, 1,
4.282828, 5.636364, -141.5472, 0.5490196, 0, 0.4470588, 1,
4.323232, 5.636364, -141.4285, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.636364, -141.3123, 0.5490196, 0, 0.4470588, 1,
4.40404, 5.636364, -141.1987, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.636364, -141.0877, 0.5490196, 0, 0.4470588, 1,
4.484848, 5.636364, -140.9792, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.636364, -140.8734, 0.5490196, 0, 0.4470588, 1,
4.565657, 5.636364, -140.77, 0.5490196, 0, 0.4470588, 1,
4.606061, 5.636364, -140.6693, 0.5490196, 0, 0.4470588, 1,
4.646465, 5.636364, -140.5711, 0.5490196, 0, 0.4470588, 1,
4.686869, 5.636364, -140.4755, 0.654902, 0, 0.3411765, 1,
4.727273, 5.636364, -140.3825, 0.654902, 0, 0.3411765, 1,
4.767677, 5.636364, -140.292, 0.654902, 0, 0.3411765, 1,
4.808081, 5.636364, -140.2041, 0.654902, 0, 0.3411765, 1,
4.848485, 5.636364, -140.1188, 0.654902, 0, 0.3411765, 1,
4.888889, 5.636364, -140.0361, 0.654902, 0, 0.3411765, 1,
4.929293, 5.636364, -139.9559, 0.654902, 0, 0.3411765, 1,
4.969697, 5.636364, -139.8783, 0.654902, 0, 0.3411765, 1,
5.010101, 5.636364, -139.8032, 0.654902, 0, 0.3411765, 1,
5.050505, 5.636364, -139.7307, 0.654902, 0, 0.3411765, 1,
5.090909, 5.636364, -139.6608, 0.654902, 0, 0.3411765, 1,
5.131313, 5.636364, -139.5935, 0.654902, 0, 0.3411765, 1,
5.171717, 5.636364, -139.5287, 0.654902, 0, 0.3411765, 1,
5.212121, 5.636364, -139.4665, 0.654902, 0, 0.3411765, 1,
5.252525, 5.636364, -139.4069, 0.654902, 0, 0.3411765, 1,
5.292929, 5.636364, -139.3498, 0.654902, 0, 0.3411765, 1,
5.333333, 5.636364, -139.2953, 0.654902, 0, 0.3411765, 1,
5.373737, 5.636364, -139.2434, 0.654902, 0, 0.3411765, 1,
5.414141, 5.636364, -139.194, 0.7568628, 0, 0.2392157, 1,
5.454545, 5.636364, -139.1473, 0.7568628, 0, 0.2392157, 1,
5.494949, 5.636364, -139.103, 0.7568628, 0, 0.2392157, 1,
5.535354, 5.636364, -139.0614, 0.7568628, 0, 0.2392157, 1,
5.575758, 5.636364, -139.0223, 0.7568628, 0, 0.2392157, 1,
5.616162, 5.636364, -138.9858, 0.7568628, 0, 0.2392157, 1,
5.656566, 5.636364, -138.9519, 0.7568628, 0, 0.2392157, 1,
5.69697, 5.636364, -138.9205, 0.7568628, 0, 0.2392157, 1,
5.737374, 5.636364, -138.8917, 0.7568628, 0, 0.2392157, 1,
5.777778, 5.636364, -138.8655, 0.7568628, 0, 0.2392157, 1,
5.818182, 5.636364, -138.8418, 0.7568628, 0, 0.2392157, 1,
5.858586, 5.636364, -138.8207, 0.7568628, 0, 0.2392157, 1,
5.89899, 5.636364, -138.8022, 0.7568628, 0, 0.2392157, 1,
5.939394, 5.636364, -138.7862, 0.7568628, 0, 0.2392157, 1,
5.979798, 5.636364, -138.7729, 0.7568628, 0, 0.2392157, 1,
6.020202, 5.636364, -138.7621, 0.7568628, 0, 0.2392157, 1,
6.060606, 5.636364, -138.7538, 0.7568628, 0, 0.2392157, 1,
6.10101, 5.636364, -138.7481, 0.7568628, 0, 0.2392157, 1,
6.141414, 5.636364, -138.745, 0.7568628, 0, 0.2392157, 1,
6.181818, 5.636364, -138.7445, 0.7568628, 0, 0.2392157, 1,
6.222222, 5.636364, -138.7465, 0.7568628, 0, 0.2392157, 1,
6.262626, 5.636364, -138.7511, 0.7568628, 0, 0.2392157, 1,
6.30303, 5.636364, -138.7583, 0.7568628, 0, 0.2392157, 1,
6.343434, 5.636364, -138.768, 0.7568628, 0, 0.2392157, 1,
6.383838, 5.636364, -138.7803, 0.7568628, 0, 0.2392157, 1,
6.424242, 5.636364, -138.7952, 0.7568628, 0, 0.2392157, 1,
6.464646, 5.636364, -138.8127, 0.7568628, 0, 0.2392157, 1,
6.505051, 5.636364, -138.8327, 0.7568628, 0, 0.2392157, 1,
6.545455, 5.636364, -138.8553, 0.7568628, 0, 0.2392157, 1,
6.585859, 5.636364, -138.8804, 0.7568628, 0, 0.2392157, 1,
6.626263, 5.636364, -138.9082, 0.7568628, 0, 0.2392157, 1,
6.666667, 5.636364, -138.9384, 0.7568628, 0, 0.2392157, 1,
6.707071, 5.636364, -138.9713, 0.7568628, 0, 0.2392157, 1,
6.747475, 5.636364, -139.0067, 0.7568628, 0, 0.2392157, 1,
6.787879, 5.636364, -139.0448, 0.7568628, 0, 0.2392157, 1,
6.828283, 5.636364, -139.0853, 0.7568628, 0, 0.2392157, 1,
6.868687, 5.636364, -139.1285, 0.7568628, 0, 0.2392157, 1,
6.909091, 5.636364, -139.1742, 0.7568628, 0, 0.2392157, 1,
6.949495, 5.636364, -139.2225, 0.7568628, 0, 0.2392157, 1,
6.989899, 5.636364, -139.2733, 0.654902, 0, 0.3411765, 1,
7.030303, 5.636364, -139.3267, 0.654902, 0, 0.3411765, 1,
7.070707, 5.636364, -139.3827, 0.654902, 0, 0.3411765, 1,
7.111111, 5.636364, -139.4413, 0.654902, 0, 0.3411765, 1,
7.151515, 5.636364, -139.5024, 0.654902, 0, 0.3411765, 1,
7.191919, 5.636364, -139.5661, 0.654902, 0, 0.3411765, 1,
7.232323, 5.636364, -139.6324, 0.654902, 0, 0.3411765, 1,
7.272727, 5.636364, -139.7012, 0.654902, 0, 0.3411765, 1,
7.313131, 5.636364, -139.7726, 0.654902, 0, 0.3411765, 1,
7.353535, 5.636364, -139.8466, 0.654902, 0, 0.3411765, 1,
7.393939, 5.636364, -139.9231, 0.654902, 0, 0.3411765, 1,
7.434343, 5.636364, -140.0022, 0.654902, 0, 0.3411765, 1,
7.474748, 5.636364, -140.0839, 0.654902, 0, 0.3411765, 1,
7.515152, 5.636364, -140.1682, 0.654902, 0, 0.3411765, 1,
7.555555, 5.636364, -140.255, 0.654902, 0, 0.3411765, 1,
7.59596, 5.636364, -140.3444, 0.654902, 0, 0.3411765, 1,
7.636364, 5.636364, -140.4363, 0.654902, 0, 0.3411765, 1,
7.676768, 5.636364, -140.5309, 0.5490196, 0, 0.4470588, 1,
7.717172, 5.636364, -140.628, 0.5490196, 0, 0.4470588, 1,
7.757576, 5.636364, -140.7276, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.636364, -140.8299, 0.5490196, 0, 0.4470588, 1,
7.838384, 5.636364, -140.9347, 0.5490196, 0, 0.4470588, 1,
7.878788, 5.636364, -141.0421, 0.5490196, 0, 0.4470588, 1,
7.919192, 5.636364, -141.152, 0.5490196, 0, 0.4470588, 1,
7.959596, 5.636364, -141.2645, 0.5490196, 0, 0.4470588, 1,
8, 5.636364, -141.3796, 0.5490196, 0, 0.4470588, 1,
4, 5.676768, -142.6649, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.676768, -142.5301, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.676768, -142.3979, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.676768, -142.2682, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.676768, -142.141, 0.4470588, 0, 0.5490196, 1,
4.20202, 5.676768, -142.0164, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.676768, -141.8943, 0.4470588, 0, 0.5490196, 1,
4.282828, 5.676768, -141.7747, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.676768, -141.6577, 0.5490196, 0, 0.4470588, 1,
4.363636, 5.676768, -141.5431, 0.5490196, 0, 0.4470588, 1,
4.40404, 5.676768, -141.4312, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.676768, -141.3217, 0.5490196, 0, 0.4470588, 1,
4.484848, 5.676768, -141.2148, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.676768, -141.1104, 0.5490196, 0, 0.4470588, 1,
4.565657, 5.676768, -141.0086, 0.5490196, 0, 0.4470588, 1,
4.606061, 5.676768, -140.9093, 0.5490196, 0, 0.4470588, 1,
4.646465, 5.676768, -140.8125, 0.5490196, 0, 0.4470588, 1,
4.686869, 5.676768, -140.7182, 0.5490196, 0, 0.4470588, 1,
4.727273, 5.676768, -140.6265, 0.5490196, 0, 0.4470588, 1,
4.767677, 5.676768, -140.5374, 0.5490196, 0, 0.4470588, 1,
4.808081, 5.676768, -140.4507, 0.654902, 0, 0.3411765, 1,
4.848485, 5.676768, -140.3666, 0.654902, 0, 0.3411765, 1,
4.888889, 5.676768, -140.285, 0.654902, 0, 0.3411765, 1,
4.929293, 5.676768, -140.206, 0.654902, 0, 0.3411765, 1,
4.969697, 5.676768, -140.1295, 0.654902, 0, 0.3411765, 1,
5.010101, 5.676768, -140.0555, 0.654902, 0, 0.3411765, 1,
5.050505, 5.676768, -139.984, 0.654902, 0, 0.3411765, 1,
5.090909, 5.676768, -139.9151, 0.654902, 0, 0.3411765, 1,
5.131313, 5.676768, -139.8487, 0.654902, 0, 0.3411765, 1,
5.171717, 5.676768, -139.7849, 0.654902, 0, 0.3411765, 1,
5.212121, 5.676768, -139.7235, 0.654902, 0, 0.3411765, 1,
5.252525, 5.676768, -139.6648, 0.654902, 0, 0.3411765, 1,
5.292929, 5.676768, -139.6085, 0.654902, 0, 0.3411765, 1,
5.333333, 5.676768, -139.5548, 0.654902, 0, 0.3411765, 1,
5.373737, 5.676768, -139.5036, 0.654902, 0, 0.3411765, 1,
5.414141, 5.676768, -139.4549, 0.654902, 0, 0.3411765, 1,
5.454545, 5.676768, -139.4088, 0.654902, 0, 0.3411765, 1,
5.494949, 5.676768, -139.3652, 0.654902, 0, 0.3411765, 1,
5.535354, 5.676768, -139.3242, 0.654902, 0, 0.3411765, 1,
5.575758, 5.676768, -139.2857, 0.654902, 0, 0.3411765, 1,
5.616162, 5.676768, -139.2497, 0.654902, 0, 0.3411765, 1,
5.656566, 5.676768, -139.2162, 0.7568628, 0, 0.2392157, 1,
5.69697, 5.676768, -139.1853, 0.7568628, 0, 0.2392157, 1,
5.737374, 5.676768, -139.1569, 0.7568628, 0, 0.2392157, 1,
5.777778, 5.676768, -139.131, 0.7568628, 0, 0.2392157, 1,
5.818182, 5.676768, -139.1077, 0.7568628, 0, 0.2392157, 1,
5.858586, 5.676768, -139.0869, 0.7568628, 0, 0.2392157, 1,
5.89899, 5.676768, -139.0687, 0.7568628, 0, 0.2392157, 1,
5.939394, 5.676768, -139.0529, 0.7568628, 0, 0.2392157, 1,
5.979798, 5.676768, -139.0397, 0.7568628, 0, 0.2392157, 1,
6.020202, 5.676768, -139.0291, 0.7568628, 0, 0.2392157, 1,
6.060606, 5.676768, -139.021, 0.7568628, 0, 0.2392157, 1,
6.10101, 5.676768, -139.0154, 0.7568628, 0, 0.2392157, 1,
6.141414, 5.676768, -139.0123, 0.7568628, 0, 0.2392157, 1,
6.181818, 5.676768, -139.0118, 0.7568628, 0, 0.2392157, 1,
6.222222, 5.676768, -139.0138, 0.7568628, 0, 0.2392157, 1,
6.262626, 5.676768, -139.0183, 0.7568628, 0, 0.2392157, 1,
6.30303, 5.676768, -139.0254, 0.7568628, 0, 0.2392157, 1,
6.343434, 5.676768, -139.035, 0.7568628, 0, 0.2392157, 1,
6.383838, 5.676768, -139.0471, 0.7568628, 0, 0.2392157, 1,
6.424242, 5.676768, -139.0618, 0.7568628, 0, 0.2392157, 1,
6.464646, 5.676768, -139.079, 0.7568628, 0, 0.2392157, 1,
6.505051, 5.676768, -139.0987, 0.7568628, 0, 0.2392157, 1,
6.545455, 5.676768, -139.121, 0.7568628, 0, 0.2392157, 1,
6.585859, 5.676768, -139.1458, 0.7568628, 0, 0.2392157, 1,
6.626263, 5.676768, -139.1731, 0.7568628, 0, 0.2392157, 1,
6.666667, 5.676768, -139.203, 0.7568628, 0, 0.2392157, 1,
6.707071, 5.676768, -139.2354, 0.7568628, 0, 0.2392157, 1,
6.747475, 5.676768, -139.2703, 0.654902, 0, 0.3411765, 1,
6.787879, 5.676768, -139.3078, 0.654902, 0, 0.3411765, 1,
6.828283, 5.676768, -139.3478, 0.654902, 0, 0.3411765, 1,
6.868687, 5.676768, -139.3903, 0.654902, 0, 0.3411765, 1,
6.909091, 5.676768, -139.4354, 0.654902, 0, 0.3411765, 1,
6.949495, 5.676768, -139.483, 0.654902, 0, 0.3411765, 1,
6.989899, 5.676768, -139.5331, 0.654902, 0, 0.3411765, 1,
7.030303, 5.676768, -139.5858, 0.654902, 0, 0.3411765, 1,
7.070707, 5.676768, -139.6409, 0.654902, 0, 0.3411765, 1,
7.111111, 5.676768, -139.6987, 0.654902, 0, 0.3411765, 1,
7.151515, 5.676768, -139.7589, 0.654902, 0, 0.3411765, 1,
7.191919, 5.676768, -139.8217, 0.654902, 0, 0.3411765, 1,
7.232323, 5.676768, -139.8871, 0.654902, 0, 0.3411765, 1,
7.272727, 5.676768, -139.9549, 0.654902, 0, 0.3411765, 1,
7.313131, 5.676768, -140.0253, 0.654902, 0, 0.3411765, 1,
7.353535, 5.676768, -140.0982, 0.654902, 0, 0.3411765, 1,
7.393939, 5.676768, -140.1737, 0.654902, 0, 0.3411765, 1,
7.434343, 5.676768, -140.2517, 0.654902, 0, 0.3411765, 1,
7.474748, 5.676768, -140.3322, 0.654902, 0, 0.3411765, 1,
7.515152, 5.676768, -140.4153, 0.654902, 0, 0.3411765, 1,
7.555555, 5.676768, -140.5008, 0.5490196, 0, 0.4470588, 1,
7.59596, 5.676768, -140.589, 0.5490196, 0, 0.4470588, 1,
7.636364, 5.676768, -140.6796, 0.5490196, 0, 0.4470588, 1,
7.676768, 5.676768, -140.7728, 0.5490196, 0, 0.4470588, 1,
7.717172, 5.676768, -140.8685, 0.5490196, 0, 0.4470588, 1,
7.757576, 5.676768, -140.9668, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.676768, -141.0676, 0.5490196, 0, 0.4470588, 1,
7.838384, 5.676768, -141.1709, 0.5490196, 0, 0.4470588, 1,
7.878788, 5.676768, -141.2767, 0.5490196, 0, 0.4470588, 1,
7.919192, 5.676768, -141.3851, 0.5490196, 0, 0.4470588, 1,
7.959596, 5.676768, -141.496, 0.5490196, 0, 0.4470588, 1,
8, 5.676768, -141.6095, 0.5490196, 0, 0.4470588, 1,
4, 5.717172, -142.8801, 0.4470588, 0, 0.5490196, 1,
4.040404, 5.717172, -142.7472, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.717172, -142.6169, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.717172, -142.489, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.717172, -142.3636, 0.4470588, 0, 0.5490196, 1,
4.20202, 5.717172, -142.2407, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.717172, -142.1203, 0.4470588, 0, 0.5490196, 1,
4.282828, 5.717172, -142.0024, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.717172, -141.887, 0.4470588, 0, 0.5490196, 1,
4.363636, 5.717172, -141.7741, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.717172, -141.6637, 0.5490196, 0, 0.4470588, 1,
4.444445, 5.717172, -141.5558, 0.5490196, 0, 0.4470588, 1,
4.484848, 5.717172, -141.4504, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.717172, -141.3475, 0.5490196, 0, 0.4470588, 1,
4.565657, 5.717172, -141.2471, 0.5490196, 0, 0.4470588, 1,
4.606061, 5.717172, -141.1492, 0.5490196, 0, 0.4470588, 1,
4.646465, 5.717172, -141.0538, 0.5490196, 0, 0.4470588, 1,
4.686869, 5.717172, -140.9609, 0.5490196, 0, 0.4470588, 1,
4.727273, 5.717172, -140.8704, 0.5490196, 0, 0.4470588, 1,
4.767677, 5.717172, -140.7825, 0.5490196, 0, 0.4470588, 1,
4.808081, 5.717172, -140.6971, 0.5490196, 0, 0.4470588, 1,
4.848485, 5.717172, -140.6142, 0.5490196, 0, 0.4470588, 1,
4.888889, 5.717172, -140.5337, 0.5490196, 0, 0.4470588, 1,
4.929293, 5.717172, -140.4558, 0.654902, 0, 0.3411765, 1,
4.969697, 5.717172, -140.3804, 0.654902, 0, 0.3411765, 1,
5.010101, 5.717172, -140.3074, 0.654902, 0, 0.3411765, 1,
5.050505, 5.717172, -140.237, 0.654902, 0, 0.3411765, 1,
5.090909, 5.717172, -140.169, 0.654902, 0, 0.3411765, 1,
5.131313, 5.717172, -140.1036, 0.654902, 0, 0.3411765, 1,
5.171717, 5.717172, -140.0406, 0.654902, 0, 0.3411765, 1,
5.212121, 5.717172, -139.9802, 0.654902, 0, 0.3411765, 1,
5.252525, 5.717172, -139.9222, 0.654902, 0, 0.3411765, 1,
5.292929, 5.717172, -139.8667, 0.654902, 0, 0.3411765, 1,
5.333333, 5.717172, -139.8138, 0.654902, 0, 0.3411765, 1,
5.373737, 5.717172, -139.7633, 0.654902, 0, 0.3411765, 1,
5.414141, 5.717172, -139.7153, 0.654902, 0, 0.3411765, 1,
5.454545, 5.717172, -139.6699, 0.654902, 0, 0.3411765, 1,
5.494949, 5.717172, -139.6269, 0.654902, 0, 0.3411765, 1,
5.535354, 5.717172, -139.5864, 0.654902, 0, 0.3411765, 1,
5.575758, 5.717172, -139.5484, 0.654902, 0, 0.3411765, 1,
5.616162, 5.717172, -139.513, 0.654902, 0, 0.3411765, 1,
5.656566, 5.717172, -139.48, 0.654902, 0, 0.3411765, 1,
5.69697, 5.717172, -139.4495, 0.654902, 0, 0.3411765, 1,
5.737374, 5.717172, -139.4215, 0.654902, 0, 0.3411765, 1,
5.777778, 5.717172, -139.396, 0.654902, 0, 0.3411765, 1,
5.818182, 5.717172, -139.373, 0.654902, 0, 0.3411765, 1,
5.858586, 5.717172, -139.3525, 0.654902, 0, 0.3411765, 1,
5.89899, 5.717172, -139.3345, 0.654902, 0, 0.3411765, 1,
5.939394, 5.717172, -139.319, 0.654902, 0, 0.3411765, 1,
5.979798, 5.717172, -139.306, 0.654902, 0, 0.3411765, 1,
6.020202, 5.717172, -139.2955, 0.654902, 0, 0.3411765, 1,
6.060606, 5.717172, -139.2875, 0.654902, 0, 0.3411765, 1,
6.10101, 5.717172, -139.282, 0.654902, 0, 0.3411765, 1,
6.141414, 5.717172, -139.2789, 0.654902, 0, 0.3411765, 1,
6.181818, 5.717172, -139.2784, 0.654902, 0, 0.3411765, 1,
6.222222, 5.717172, -139.2804, 0.654902, 0, 0.3411765, 1,
6.262626, 5.717172, -139.2849, 0.654902, 0, 0.3411765, 1,
6.30303, 5.717172, -139.2918, 0.654902, 0, 0.3411765, 1,
6.343434, 5.717172, -139.3013, 0.654902, 0, 0.3411765, 1,
6.383838, 5.717172, -139.3133, 0.654902, 0, 0.3411765, 1,
6.424242, 5.717172, -139.3277, 0.654902, 0, 0.3411765, 1,
6.464646, 5.717172, -139.3447, 0.654902, 0, 0.3411765, 1,
6.505051, 5.717172, -139.3641, 0.654902, 0, 0.3411765, 1,
6.545455, 5.717172, -139.3861, 0.654902, 0, 0.3411765, 1,
6.585859, 5.717172, -139.4105, 0.654902, 0, 0.3411765, 1,
6.626263, 5.717172, -139.4375, 0.654902, 0, 0.3411765, 1,
6.666667, 5.717172, -139.4669, 0.654902, 0, 0.3411765, 1,
6.707071, 5.717172, -139.4989, 0.654902, 0, 0.3411765, 1,
6.747475, 5.717172, -139.5333, 0.654902, 0, 0.3411765, 1,
6.787879, 5.717172, -139.5703, 0.654902, 0, 0.3411765, 1,
6.828283, 5.717172, -139.6097, 0.654902, 0, 0.3411765, 1,
6.868687, 5.717172, -139.6516, 0.654902, 0, 0.3411765, 1,
6.909091, 5.717172, -139.696, 0.654902, 0, 0.3411765, 1,
6.949495, 5.717172, -139.743, 0.654902, 0, 0.3411765, 1,
6.989899, 5.717172, -139.7924, 0.654902, 0, 0.3411765, 1,
7.030303, 5.717172, -139.8443, 0.654902, 0, 0.3411765, 1,
7.070707, 5.717172, -139.8987, 0.654902, 0, 0.3411765, 1,
7.111111, 5.717172, -139.9556, 0.654902, 0, 0.3411765, 1,
7.151515, 5.717172, -140.0151, 0.654902, 0, 0.3411765, 1,
7.191919, 5.717172, -140.077, 0.654902, 0, 0.3411765, 1,
7.232323, 5.717172, -140.1414, 0.654902, 0, 0.3411765, 1,
7.272727, 5.717172, -140.2083, 0.654902, 0, 0.3411765, 1,
7.313131, 5.717172, -140.2777, 0.654902, 0, 0.3411765, 1,
7.353535, 5.717172, -140.3496, 0.654902, 0, 0.3411765, 1,
7.393939, 5.717172, -140.424, 0.654902, 0, 0.3411765, 1,
7.434343, 5.717172, -140.5009, 0.5490196, 0, 0.4470588, 1,
7.474748, 5.717172, -140.5802, 0.5490196, 0, 0.4470588, 1,
7.515152, 5.717172, -140.6621, 0.5490196, 0, 0.4470588, 1,
7.555555, 5.717172, -140.7465, 0.5490196, 0, 0.4470588, 1,
7.59596, 5.717172, -140.8334, 0.5490196, 0, 0.4470588, 1,
7.636364, 5.717172, -140.9228, 0.5490196, 0, 0.4470588, 1,
7.676768, 5.717172, -141.0146, 0.5490196, 0, 0.4470588, 1,
7.717172, 5.717172, -141.109, 0.5490196, 0, 0.4470588, 1,
7.757576, 5.717172, -141.2059, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.717172, -141.3053, 0.5490196, 0, 0.4470588, 1,
7.838384, 5.717172, -141.4071, 0.5490196, 0, 0.4470588, 1,
7.878788, 5.717172, -141.5115, 0.5490196, 0, 0.4470588, 1,
7.919192, 5.717172, -141.6183, 0.5490196, 0, 0.4470588, 1,
7.959596, 5.717172, -141.7277, 0.4470588, 0, 0.5490196, 1,
8, 5.717172, -141.8396, 0.4470588, 0, 0.5490196, 1,
4, 5.757576, -143.0957, 0.3411765, 0, 0.654902, 1,
4.040404, 5.757576, -142.9647, 0.4470588, 0, 0.5490196, 1,
4.080808, 5.757576, -142.8362, 0.4470588, 0, 0.5490196, 1,
4.121212, 5.757576, -142.7101, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.757576, -142.5864, 0.4470588, 0, 0.5490196, 1,
4.20202, 5.757576, -142.4653, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.757576, -142.3466, 0.4470588, 0, 0.5490196, 1,
4.282828, 5.757576, -142.2303, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.757576, -142.1165, 0.4470588, 0, 0.5490196, 1,
4.363636, 5.757576, -142.0052, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.757576, -141.8964, 0.4470588, 0, 0.5490196, 1,
4.444445, 5.757576, -141.79, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.757576, -141.6861, 0.5490196, 0, 0.4470588, 1,
4.525252, 5.757576, -141.5846, 0.5490196, 0, 0.4470588, 1,
4.565657, 5.757576, -141.4856, 0.5490196, 0, 0.4470588, 1,
4.606061, 5.757576, -141.389, 0.5490196, 0, 0.4470588, 1,
4.646465, 5.757576, -141.2949, 0.5490196, 0, 0.4470588, 1,
4.686869, 5.757576, -141.2033, 0.5490196, 0, 0.4470588, 1,
4.727273, 5.757576, -141.1142, 0.5490196, 0, 0.4470588, 1,
4.767677, 5.757576, -141.0275, 0.5490196, 0, 0.4470588, 1,
4.808081, 5.757576, -140.9432, 0.5490196, 0, 0.4470588, 1,
4.848485, 5.757576, -140.8615, 0.5490196, 0, 0.4470588, 1,
4.888889, 5.757576, -140.7822, 0.5490196, 0, 0.4470588, 1,
4.929293, 5.757576, -140.7053, 0.5490196, 0, 0.4470588, 1,
4.969697, 5.757576, -140.6309, 0.5490196, 0, 0.4470588, 1,
5.010101, 5.757576, -140.559, 0.5490196, 0, 0.4470588, 1,
5.050505, 5.757576, -140.4895, 0.5490196, 0, 0.4470588, 1,
5.090909, 5.757576, -140.4226, 0.654902, 0, 0.3411765, 1,
5.131313, 5.757576, -140.358, 0.654902, 0, 0.3411765, 1,
5.171717, 5.757576, -140.2959, 0.654902, 0, 0.3411765, 1,
5.212121, 5.757576, -140.2363, 0.654902, 0, 0.3411765, 1,
5.252525, 5.757576, -140.1792, 0.654902, 0, 0.3411765, 1,
5.292929, 5.757576, -140.1245, 0.654902, 0, 0.3411765, 1,
5.333333, 5.757576, -140.0723, 0.654902, 0, 0.3411765, 1,
5.373737, 5.757576, -140.0225, 0.654902, 0, 0.3411765, 1,
5.414141, 5.757576, -139.9752, 0.654902, 0, 0.3411765, 1,
5.454545, 5.757576, -139.9304, 0.654902, 0, 0.3411765, 1,
5.494949, 5.757576, -139.888, 0.654902, 0, 0.3411765, 1,
5.535354, 5.757576, -139.8481, 0.654902, 0, 0.3411765, 1,
5.575758, 5.757576, -139.8107, 0.654902, 0, 0.3411765, 1,
5.616162, 5.757576, -139.7757, 0.654902, 0, 0.3411765, 1,
5.656566, 5.757576, -139.7431, 0.654902, 0, 0.3411765, 1,
5.69697, 5.757576, -139.7131, 0.654902, 0, 0.3411765, 1,
5.737374, 5.757576, -139.6855, 0.654902, 0, 0.3411765, 1,
5.777778, 5.757576, -139.6604, 0.654902, 0, 0.3411765, 1,
5.818182, 5.757576, -139.6377, 0.654902, 0, 0.3411765, 1,
5.858586, 5.757576, -139.6175, 0.654902, 0, 0.3411765, 1,
5.89899, 5.757576, -139.5997, 0.654902, 0, 0.3411765, 1,
5.939394, 5.757576, -139.5844, 0.654902, 0, 0.3411765, 1,
5.979798, 5.757576, -139.5716, 0.654902, 0, 0.3411765, 1,
6.020202, 5.757576, -139.5612, 0.654902, 0, 0.3411765, 1,
6.060606, 5.757576, -139.5533, 0.654902, 0, 0.3411765, 1,
6.10101, 5.757576, -139.5479, 0.654902, 0, 0.3411765, 1,
6.141414, 5.757576, -139.5449, 0.654902, 0, 0.3411765, 1,
6.181818, 5.757576, -139.5444, 0.654902, 0, 0.3411765, 1,
6.222222, 5.757576, -139.5464, 0.654902, 0, 0.3411765, 1,
6.262626, 5.757576, -139.5508, 0.654902, 0, 0.3411765, 1,
6.30303, 5.757576, -139.5576, 0.654902, 0, 0.3411765, 1,
6.343434, 5.757576, -139.567, 0.654902, 0, 0.3411765, 1,
6.383838, 5.757576, -139.5788, 0.654902, 0, 0.3411765, 1,
6.424242, 5.757576, -139.593, 0.654902, 0, 0.3411765, 1,
6.464646, 5.757576, -139.6097, 0.654902, 0, 0.3411765, 1,
6.505051, 5.757576, -139.6289, 0.654902, 0, 0.3411765, 1,
6.545455, 5.757576, -139.6506, 0.654902, 0, 0.3411765, 1,
6.585859, 5.757576, -139.6747, 0.654902, 0, 0.3411765, 1,
6.626263, 5.757576, -139.7012, 0.654902, 0, 0.3411765, 1,
6.666667, 5.757576, -139.7303, 0.654902, 0, 0.3411765, 1,
6.707071, 5.757576, -139.7618, 0.654902, 0, 0.3411765, 1,
6.747475, 5.757576, -139.7957, 0.654902, 0, 0.3411765, 1,
6.787879, 5.757576, -139.8322, 0.654902, 0, 0.3411765, 1,
6.828283, 5.757576, -139.871, 0.654902, 0, 0.3411765, 1,
6.868687, 5.757576, -139.9124, 0.654902, 0, 0.3411765, 1,
6.909091, 5.757576, -139.9562, 0.654902, 0, 0.3411765, 1,
6.949495, 5.757576, -140.0025, 0.654902, 0, 0.3411765, 1,
6.989899, 5.757576, -140.0512, 0.654902, 0, 0.3411765, 1,
7.030303, 5.757576, -140.1024, 0.654902, 0, 0.3411765, 1,
7.070707, 5.757576, -140.1561, 0.654902, 0, 0.3411765, 1,
7.111111, 5.757576, -140.2122, 0.654902, 0, 0.3411765, 1,
7.151515, 5.757576, -140.2708, 0.654902, 0, 0.3411765, 1,
7.191919, 5.757576, -140.3318, 0.654902, 0, 0.3411765, 1,
7.232323, 5.757576, -140.3953, 0.654902, 0, 0.3411765, 1,
7.272727, 5.757576, -140.4613, 0.654902, 0, 0.3411765, 1,
7.313131, 5.757576, -140.5297, 0.5490196, 0, 0.4470588, 1,
7.353535, 5.757576, -140.6006, 0.5490196, 0, 0.4470588, 1,
7.393939, 5.757576, -140.6739, 0.5490196, 0, 0.4470588, 1,
7.434343, 5.757576, -140.7498, 0.5490196, 0, 0.4470588, 1,
7.474748, 5.757576, -140.828, 0.5490196, 0, 0.4470588, 1,
7.515152, 5.757576, -140.9088, 0.5490196, 0, 0.4470588, 1,
7.555555, 5.757576, -140.992, 0.5490196, 0, 0.4470588, 1,
7.59596, 5.757576, -141.0776, 0.5490196, 0, 0.4470588, 1,
7.636364, 5.757576, -141.1658, 0.5490196, 0, 0.4470588, 1,
7.676768, 5.757576, -141.2564, 0.5490196, 0, 0.4470588, 1,
7.717172, 5.757576, -141.3494, 0.5490196, 0, 0.4470588, 1,
7.757576, 5.757576, -141.4449, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.757576, -141.5429, 0.5490196, 0, 0.4470588, 1,
7.838384, 5.757576, -141.6433, 0.5490196, 0, 0.4470588, 1,
7.878788, 5.757576, -141.7462, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.757576, -141.8516, 0.4470588, 0, 0.5490196, 1,
7.959596, 5.757576, -141.9594, 0.4470588, 0, 0.5490196, 1,
8, 5.757576, -142.0697, 0.4470588, 0, 0.5490196, 1,
4, 5.79798, -143.3117, 0.3411765, 0, 0.654902, 1,
4.040404, 5.79798, -143.1825, 0.3411765, 0, 0.654902, 1,
4.080808, 5.79798, -143.0558, 0.3411765, 0, 0.654902, 1,
4.121212, 5.79798, -142.9314, 0.4470588, 0, 0.5490196, 1,
4.161616, 5.79798, -142.8095, 0.4470588, 0, 0.5490196, 1,
4.20202, 5.79798, -142.69, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.79798, -142.573, 0.4470588, 0, 0.5490196, 1,
4.282828, 5.79798, -142.4583, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.79798, -142.3461, 0.4470588, 0, 0.5490196, 1,
4.363636, 5.79798, -142.2364, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.79798, -142.129, 0.4470588, 0, 0.5490196, 1,
4.444445, 5.79798, -142.0241, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.79798, -141.9216, 0.4470588, 0, 0.5490196, 1,
4.525252, 5.79798, -141.8216, 0.4470588, 0, 0.5490196, 1,
4.565657, 5.79798, -141.7239, 0.4470588, 0, 0.5490196, 1,
4.606061, 5.79798, -141.6287, 0.5490196, 0, 0.4470588, 1,
4.646465, 5.79798, -141.5359, 0.5490196, 0, 0.4470588, 1,
4.686869, 5.79798, -141.4456, 0.5490196, 0, 0.4470588, 1,
4.727273, 5.79798, -141.3577, 0.5490196, 0, 0.4470588, 1,
4.767677, 5.79798, -141.2722, 0.5490196, 0, 0.4470588, 1,
4.808081, 5.79798, -141.1891, 0.5490196, 0, 0.4470588, 1,
4.848485, 5.79798, -141.1085, 0.5490196, 0, 0.4470588, 1,
4.888889, 5.79798, -141.0303, 0.5490196, 0, 0.4470588, 1,
4.929293, 5.79798, -140.9545, 0.5490196, 0, 0.4470588, 1,
4.969697, 5.79798, -140.8812, 0.5490196, 0, 0.4470588, 1,
5.010101, 5.79798, -140.8102, 0.5490196, 0, 0.4470588, 1,
5.050505, 5.79798, -140.7417, 0.5490196, 0, 0.4470588, 1,
5.090909, 5.79798, -140.6757, 0.5490196, 0, 0.4470588, 1,
5.131313, 5.79798, -140.612, 0.5490196, 0, 0.4470588, 1,
5.171717, 5.79798, -140.5508, 0.5490196, 0, 0.4470588, 1,
5.212121, 5.79798, -140.4921, 0.5490196, 0, 0.4470588, 1,
5.252525, 5.79798, -140.4357, 0.654902, 0, 0.3411765, 1,
5.292929, 5.79798, -140.3818, 0.654902, 0, 0.3411765, 1,
5.333333, 5.79798, -140.3303, 0.654902, 0, 0.3411765, 1,
5.373737, 5.79798, -140.2812, 0.654902, 0, 0.3411765, 1,
5.414141, 5.79798, -140.2346, 0.654902, 0, 0.3411765, 1,
5.454545, 5.79798, -140.1904, 0.654902, 0, 0.3411765, 1,
5.494949, 5.79798, -140.1486, 0.654902, 0, 0.3411765, 1,
5.535354, 5.79798, -140.1092, 0.654902, 0, 0.3411765, 1,
5.575758, 5.79798, -140.0723, 0.654902, 0, 0.3411765, 1,
5.616162, 5.79798, -140.0378, 0.654902, 0, 0.3411765, 1,
5.656566, 5.79798, -140.0057, 0.654902, 0, 0.3411765, 1,
5.69697, 5.79798, -139.9761, 0.654902, 0, 0.3411765, 1,
5.737374, 5.79798, -139.9489, 0.654902, 0, 0.3411765, 1,
5.777778, 5.79798, -139.9241, 0.654902, 0, 0.3411765, 1,
5.818182, 5.79798, -139.9017, 0.654902, 0, 0.3411765, 1,
5.858586, 5.79798, -139.8818, 0.654902, 0, 0.3411765, 1,
5.89899, 5.79798, -139.8643, 0.654902, 0, 0.3411765, 1,
5.939394, 5.79798, -139.8492, 0.654902, 0, 0.3411765, 1,
5.979798, 5.79798, -139.8365, 0.654902, 0, 0.3411765, 1,
6.020202, 5.79798, -139.8263, 0.654902, 0, 0.3411765, 1,
6.060606, 5.79798, -139.8185, 0.654902, 0, 0.3411765, 1,
6.10101, 5.79798, -139.8132, 0.654902, 0, 0.3411765, 1,
6.141414, 5.79798, -139.8102, 0.654902, 0, 0.3411765, 1,
6.181818, 5.79798, -139.8097, 0.654902, 0, 0.3411765, 1,
6.222222, 5.79798, -139.8116, 0.654902, 0, 0.3411765, 1,
6.262626, 5.79798, -139.816, 0.654902, 0, 0.3411765, 1,
6.30303, 5.79798, -139.8228, 0.654902, 0, 0.3411765, 1,
6.343434, 5.79798, -139.832, 0.654902, 0, 0.3411765, 1,
6.383838, 5.79798, -139.8436, 0.654902, 0, 0.3411765, 1,
6.424242, 5.79798, -139.8577, 0.654902, 0, 0.3411765, 1,
6.464646, 5.79798, -139.8742, 0.654902, 0, 0.3411765, 1,
6.505051, 5.79798, -139.8931, 0.654902, 0, 0.3411765, 1,
6.545455, 5.79798, -139.9144, 0.654902, 0, 0.3411765, 1,
6.585859, 5.79798, -139.9382, 0.654902, 0, 0.3411765, 1,
6.626263, 5.79798, -139.9644, 0.654902, 0, 0.3411765, 1,
6.666667, 5.79798, -139.993, 0.654902, 0, 0.3411765, 1,
6.707071, 5.79798, -140.0241, 0.654902, 0, 0.3411765, 1,
6.747475, 5.79798, -140.0576, 0.654902, 0, 0.3411765, 1,
6.787879, 5.79798, -140.0935, 0.654902, 0, 0.3411765, 1,
6.828283, 5.79798, -140.1318, 0.654902, 0, 0.3411765, 1,
6.868687, 5.79798, -140.1726, 0.654902, 0, 0.3411765, 1,
6.909091, 5.79798, -140.2158, 0.654902, 0, 0.3411765, 1,
6.949495, 5.79798, -140.2614, 0.654902, 0, 0.3411765, 1,
6.989899, 5.79798, -140.3095, 0.654902, 0, 0.3411765, 1,
7.030303, 5.79798, -140.36, 0.654902, 0, 0.3411765, 1,
7.070707, 5.79798, -140.4129, 0.654902, 0, 0.3411765, 1,
7.111111, 5.79798, -140.4682, 0.654902, 0, 0.3411765, 1,
7.151515, 5.79798, -140.526, 0.5490196, 0, 0.4470588, 1,
7.191919, 5.79798, -140.5862, 0.5490196, 0, 0.4470588, 1,
7.232323, 5.79798, -140.6488, 0.5490196, 0, 0.4470588, 1,
7.272727, 5.79798, -140.7139, 0.5490196, 0, 0.4470588, 1,
7.313131, 5.79798, -140.7813, 0.5490196, 0, 0.4470588, 1,
7.353535, 5.79798, -140.8512, 0.5490196, 0, 0.4470588, 1,
7.393939, 5.79798, -140.9236, 0.5490196, 0, 0.4470588, 1,
7.434343, 5.79798, -140.9983, 0.5490196, 0, 0.4470588, 1,
7.474748, 5.79798, -141.0755, 0.5490196, 0, 0.4470588, 1,
7.515152, 5.79798, -141.1552, 0.5490196, 0, 0.4470588, 1,
7.555555, 5.79798, -141.2372, 0.5490196, 0, 0.4470588, 1,
7.59596, 5.79798, -141.3217, 0.5490196, 0, 0.4470588, 1,
7.636364, 5.79798, -141.4086, 0.5490196, 0, 0.4470588, 1,
7.676768, 5.79798, -141.4979, 0.5490196, 0, 0.4470588, 1,
7.717172, 5.79798, -141.5897, 0.5490196, 0, 0.4470588, 1,
7.757576, 5.79798, -141.6839, 0.5490196, 0, 0.4470588, 1,
7.79798, 5.79798, -141.7805, 0.4470588, 0, 0.5490196, 1,
7.838384, 5.79798, -141.8795, 0.4470588, 0, 0.5490196, 1,
7.878788, 5.79798, -141.981, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.79798, -142.0849, 0.4470588, 0, 0.5490196, 1,
7.959596, 5.79798, -142.1912, 0.4470588, 0, 0.5490196, 1,
8, 5.79798, -142.3, 0.4470588, 0, 0.5490196, 1,
4, 5.838384, -143.5281, 0.3411765, 0, 0.654902, 1,
4.040404, 5.838384, -143.4007, 0.3411765, 0, 0.654902, 1,
4.080808, 5.838384, -143.2756, 0.3411765, 0, 0.654902, 1,
4.121212, 5.838384, -143.153, 0.3411765, 0, 0.654902, 1,
4.161616, 5.838384, -143.0328, 0.3411765, 0, 0.654902, 1,
4.20202, 5.838384, -142.9149, 0.4470588, 0, 0.5490196, 1,
4.242424, 5.838384, -142.7995, 0.4470588, 0, 0.5490196, 1,
4.282828, 5.838384, -142.6864, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.838384, -142.5758, 0.4470588, 0, 0.5490196, 1,
4.363636, 5.838384, -142.4675, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.838384, -142.3617, 0.4470588, 0, 0.5490196, 1,
4.444445, 5.838384, -142.2582, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.838384, -142.1571, 0.4470588, 0, 0.5490196, 1,
4.525252, 5.838384, -142.0585, 0.4470588, 0, 0.5490196, 1,
4.565657, 5.838384, -141.9622, 0.4470588, 0, 0.5490196, 1,
4.606061, 5.838384, -141.8683, 0.4470588, 0, 0.5490196, 1,
4.646465, 5.838384, -141.7768, 0.4470588, 0, 0.5490196, 1,
4.686869, 5.838384, -141.6877, 0.5490196, 0, 0.4470588, 1,
4.727273, 5.838384, -141.601, 0.5490196, 0, 0.4470588, 1,
4.767677, 5.838384, -141.5167, 0.5490196, 0, 0.4470588, 1,
4.808081, 5.838384, -141.4348, 0.5490196, 0, 0.4470588, 1,
4.848485, 5.838384, -141.3552, 0.5490196, 0, 0.4470588, 1,
4.888889, 5.838384, -141.2781, 0.5490196, 0, 0.4470588, 1,
4.929293, 5.838384, -141.2034, 0.5490196, 0, 0.4470588, 1,
4.969697, 5.838384, -141.131, 0.5490196, 0, 0.4470588, 1,
5.010101, 5.838384, -141.0611, 0.5490196, 0, 0.4470588, 1,
5.050505, 5.838384, -140.9935, 0.5490196, 0, 0.4470588, 1,
5.090909, 5.838384, -140.9284, 0.5490196, 0, 0.4470588, 1,
5.131313, 5.838384, -140.8656, 0.5490196, 0, 0.4470588, 1,
5.171717, 5.838384, -140.8053, 0.5490196, 0, 0.4470588, 1,
5.212121, 5.838384, -140.7473, 0.5490196, 0, 0.4470588, 1,
5.252525, 5.838384, -140.6917, 0.5490196, 0, 0.4470588, 1,
5.292929, 5.838384, -140.6385, 0.5490196, 0, 0.4470588, 1,
5.333333, 5.838384, -140.5877, 0.5490196, 0, 0.4470588, 1,
5.373737, 5.838384, -140.5394, 0.5490196, 0, 0.4470588, 1,
5.414141, 5.838384, -140.4933, 0.5490196, 0, 0.4470588, 1,
5.454545, 5.838384, -140.4498, 0.654902, 0, 0.3411765, 1,
5.494949, 5.838384, -140.4085, 0.654902, 0, 0.3411765, 1,
5.535354, 5.838384, -140.3697, 0.654902, 0, 0.3411765, 1,
5.575758, 5.838384, -140.3333, 0.654902, 0, 0.3411765, 1,
5.616162, 5.838384, -140.2993, 0.654902, 0, 0.3411765, 1,
5.656566, 5.838384, -140.2677, 0.654902, 0, 0.3411765, 1,
5.69697, 5.838384, -140.2384, 0.654902, 0, 0.3411765, 1,
5.737374, 5.838384, -140.2116, 0.654902, 0, 0.3411765, 1,
5.777778, 5.838384, -140.1871, 0.654902, 0, 0.3411765, 1,
5.818182, 5.838384, -140.1651, 0.654902, 0, 0.3411765, 1,
5.858586, 5.838384, -140.1454, 0.654902, 0, 0.3411765, 1,
5.89899, 5.838384, -140.1282, 0.654902, 0, 0.3411765, 1,
5.939394, 5.838384, -140.1133, 0.654902, 0, 0.3411765, 1,
5.979798, 5.838384, -140.1008, 0.654902, 0, 0.3411765, 1,
6.020202, 5.838384, -140.0907, 0.654902, 0, 0.3411765, 1,
6.060606, 5.838384, -140.0831, 0.654902, 0, 0.3411765, 1,
6.10101, 5.838384, -140.0778, 0.654902, 0, 0.3411765, 1,
6.141414, 5.838384, -140.0749, 0.654902, 0, 0.3411765, 1,
6.181818, 5.838384, -140.0744, 0.654902, 0, 0.3411765, 1,
6.222222, 5.838384, -140.0763, 0.654902, 0, 0.3411765, 1,
6.262626, 5.838384, -140.0806, 0.654902, 0, 0.3411765, 1,
6.30303, 5.838384, -140.0872, 0.654902, 0, 0.3411765, 1,
6.343434, 5.838384, -140.0963, 0.654902, 0, 0.3411765, 1,
6.383838, 5.838384, -140.1078, 0.654902, 0, 0.3411765, 1,
6.424242, 5.838384, -140.1217, 0.654902, 0, 0.3411765, 1,
6.464646, 5.838384, -140.1379, 0.654902, 0, 0.3411765, 1,
6.505051, 5.838384, -140.1566, 0.654902, 0, 0.3411765, 1,
6.545455, 5.838384, -140.1776, 0.654902, 0, 0.3411765, 1,
6.585859, 5.838384, -140.2011, 0.654902, 0, 0.3411765, 1,
6.626263, 5.838384, -140.2269, 0.654902, 0, 0.3411765, 1,
6.666667, 5.838384, -140.2551, 0.654902, 0, 0.3411765, 1,
6.707071, 5.838384, -140.2858, 0.654902, 0, 0.3411765, 1,
6.747475, 5.838384, -140.3188, 0.654902, 0, 0.3411765, 1,
6.787879, 5.838384, -140.3542, 0.654902, 0, 0.3411765, 1,
6.828283, 5.838384, -140.392, 0.654902, 0, 0.3411765, 1,
6.868687, 5.838384, -140.4322, 0.654902, 0, 0.3411765, 1,
6.909091, 5.838384, -140.4748, 0.654902, 0, 0.3411765, 1,
6.949495, 5.838384, -140.5198, 0.5490196, 0, 0.4470588, 1,
6.989899, 5.838384, -140.5672, 0.5490196, 0, 0.4470588, 1,
7.030303, 5.838384, -140.617, 0.5490196, 0, 0.4470588, 1,
7.070707, 5.838384, -140.6692, 0.5490196, 0, 0.4470588, 1,
7.111111, 5.838384, -140.7238, 0.5490196, 0, 0.4470588, 1,
7.151515, 5.838384, -140.7807, 0.5490196, 0, 0.4470588, 1,
7.191919, 5.838384, -140.8401, 0.5490196, 0, 0.4470588, 1,
7.232323, 5.838384, -140.9019, 0.5490196, 0, 0.4470588, 1,
7.272727, 5.838384, -140.966, 0.5490196, 0, 0.4470588, 1,
7.313131, 5.838384, -141.0326, 0.5490196, 0, 0.4470588, 1,
7.353535, 5.838384, -141.1015, 0.5490196, 0, 0.4470588, 1,
7.393939, 5.838384, -141.1729, 0.5490196, 0, 0.4470588, 1,
7.434343, 5.838384, -141.2466, 0.5490196, 0, 0.4470588, 1,
7.474748, 5.838384, -141.3227, 0.5490196, 0, 0.4470588, 1,
7.515152, 5.838384, -141.4012, 0.5490196, 0, 0.4470588, 1,
7.555555, 5.838384, -141.4821, 0.5490196, 0, 0.4470588, 1,
7.59596, 5.838384, -141.5655, 0.5490196, 0, 0.4470588, 1,
7.636364, 5.838384, -141.6512, 0.5490196, 0, 0.4470588, 1,
7.676768, 5.838384, -141.7393, 0.4470588, 0, 0.5490196, 1,
7.717172, 5.838384, -141.8298, 0.4470588, 0, 0.5490196, 1,
7.757576, 5.838384, -141.9227, 0.4470588, 0, 0.5490196, 1,
7.79798, 5.838384, -142.0179, 0.4470588, 0, 0.5490196, 1,
7.838384, 5.838384, -142.1156, 0.4470588, 0, 0.5490196, 1,
7.878788, 5.838384, -142.2157, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.838384, -142.3182, 0.4470588, 0, 0.5490196, 1,
7.959596, 5.838384, -142.423, 0.4470588, 0, 0.5490196, 1,
8, 5.838384, -142.5303, 0.4470588, 0, 0.5490196, 1,
4, 5.878788, -143.7447, 0.3411765, 0, 0.654902, 1,
4.040404, 5.878788, -143.619, 0.3411765, 0, 0.654902, 1,
4.080808, 5.878788, -143.4957, 0.3411765, 0, 0.654902, 1,
4.121212, 5.878788, -143.3748, 0.3411765, 0, 0.654902, 1,
4.161616, 5.878788, -143.2562, 0.3411765, 0, 0.654902, 1,
4.20202, 5.878788, -143.14, 0.3411765, 0, 0.654902, 1,
4.242424, 5.878788, -143.0261, 0.3411765, 0, 0.654902, 1,
4.282828, 5.878788, -142.9146, 0.4470588, 0, 0.5490196, 1,
4.323232, 5.878788, -142.8055, 0.4470588, 0, 0.5490196, 1,
4.363636, 5.878788, -142.6987, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.878788, -142.5943, 0.4470588, 0, 0.5490196, 1,
4.444445, 5.878788, -142.4922, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.878788, -142.3925, 0.4470588, 0, 0.5490196, 1,
4.525252, 5.878788, -142.2952, 0.4470588, 0, 0.5490196, 1,
4.565657, 5.878788, -142.2003, 0.4470588, 0, 0.5490196, 1,
4.606061, 5.878788, -142.1077, 0.4470588, 0, 0.5490196, 1,
4.646465, 5.878788, -142.0174, 0.4470588, 0, 0.5490196, 1,
4.686869, 5.878788, -141.9295, 0.4470588, 0, 0.5490196, 1,
4.727273, 5.878788, -141.844, 0.4470588, 0, 0.5490196, 1,
4.767677, 5.878788, -141.7608, 0.4470588, 0, 0.5490196, 1,
4.808081, 5.878788, -141.6801, 0.5490196, 0, 0.4470588, 1,
4.848485, 5.878788, -141.6016, 0.5490196, 0, 0.4470588, 1,
4.888889, 5.878788, -141.5256, 0.5490196, 0, 0.4470588, 1,
4.929293, 5.878788, -141.4518, 0.5490196, 0, 0.4470588, 1,
4.969697, 5.878788, -141.3805, 0.5490196, 0, 0.4470588, 1,
5.010101, 5.878788, -141.3115, 0.5490196, 0, 0.4470588, 1,
5.050505, 5.878788, -141.2449, 0.5490196, 0, 0.4470588, 1,
5.090909, 5.878788, -141.1806, 0.5490196, 0, 0.4470588, 1,
5.131313, 5.878788, -141.1187, 0.5490196, 0, 0.4470588, 1,
5.171717, 5.878788, -141.0592, 0.5490196, 0, 0.4470588, 1,
5.212121, 5.878788, -141.002, 0.5490196, 0, 0.4470588, 1,
5.252525, 5.878788, -140.9472, 0.5490196, 0, 0.4470588, 1,
5.292929, 5.878788, -140.8947, 0.5490196, 0, 0.4470588, 1,
5.333333, 5.878788, -140.8447, 0.5490196, 0, 0.4470588, 1,
5.373737, 5.878788, -140.7969, 0.5490196, 0, 0.4470588, 1,
5.414141, 5.878788, -140.7516, 0.5490196, 0, 0.4470588, 1,
5.454545, 5.878788, -140.7085, 0.5490196, 0, 0.4470588, 1,
5.494949, 5.878788, -140.6679, 0.5490196, 0, 0.4470588, 1,
5.535354, 5.878788, -140.6296, 0.5490196, 0, 0.4470588, 1,
5.575758, 5.878788, -140.5937, 0.5490196, 0, 0.4470588, 1,
5.616162, 5.878788, -140.5601, 0.5490196, 0, 0.4470588, 1,
5.656566, 5.878788, -140.5289, 0.5490196, 0, 0.4470588, 1,
5.69697, 5.878788, -140.5001, 0.5490196, 0, 0.4470588, 1,
5.737374, 5.878788, -140.4736, 0.654902, 0, 0.3411765, 1,
5.777778, 5.878788, -140.4495, 0.654902, 0, 0.3411765, 1,
5.818182, 5.878788, -140.4278, 0.654902, 0, 0.3411765, 1,
5.858586, 5.878788, -140.4084, 0.654902, 0, 0.3411765, 1,
5.89899, 5.878788, -140.3914, 0.654902, 0, 0.3411765, 1,
5.939394, 5.878788, -140.3767, 0.654902, 0, 0.3411765, 1,
5.979798, 5.878788, -140.3644, 0.654902, 0, 0.3411765, 1,
6.020202, 5.878788, -140.3545, 0.654902, 0, 0.3411765, 1,
6.060606, 5.878788, -140.3469, 0.654902, 0, 0.3411765, 1,
6.10101, 5.878788, -140.3417, 0.654902, 0, 0.3411765, 1,
6.141414, 5.878788, -140.3388, 0.654902, 0, 0.3411765, 1,
6.181818, 5.878788, -140.3383, 0.654902, 0, 0.3411765, 1,
6.222222, 5.878788, -140.3402, 0.654902, 0, 0.3411765, 1,
6.262626, 5.878788, -140.3444, 0.654902, 0, 0.3411765, 1,
6.30303, 5.878788, -140.351, 0.654902, 0, 0.3411765, 1,
6.343434, 5.878788, -140.36, 0.654902, 0, 0.3411765, 1,
6.383838, 5.878788, -140.3713, 0.654902, 0, 0.3411765, 1,
6.424242, 5.878788, -140.3849, 0.654902, 0, 0.3411765, 1,
6.464646, 5.878788, -140.401, 0.654902, 0, 0.3411765, 1,
6.505051, 5.878788, -140.4194, 0.654902, 0, 0.3411765, 1,
6.545455, 5.878788, -140.4402, 0.654902, 0, 0.3411765, 1,
6.585859, 5.878788, -140.4633, 0.654902, 0, 0.3411765, 1,
6.626263, 5.878788, -140.4888, 0.5490196, 0, 0.4470588, 1,
6.666667, 5.878788, -140.5166, 0.5490196, 0, 0.4470588, 1,
6.707071, 5.878788, -140.5468, 0.5490196, 0, 0.4470588, 1,
6.747475, 5.878788, -140.5794, 0.5490196, 0, 0.4470588, 1,
6.787879, 5.878788, -140.6143, 0.5490196, 0, 0.4470588, 1,
6.828283, 5.878788, -140.6516, 0.5490196, 0, 0.4470588, 1,
6.868687, 5.878788, -140.6913, 0.5490196, 0, 0.4470588, 1,
6.909091, 5.878788, -140.7333, 0.5490196, 0, 0.4470588, 1,
6.949495, 5.878788, -140.7777, 0.5490196, 0, 0.4470588, 1,
6.989899, 5.878788, -140.8244, 0.5490196, 0, 0.4470588, 1,
7.030303, 5.878788, -140.8735, 0.5490196, 0, 0.4470588, 1,
7.070707, 5.878788, -140.925, 0.5490196, 0, 0.4470588, 1,
7.111111, 5.878788, -140.9788, 0.5490196, 0, 0.4470588, 1,
7.151515, 5.878788, -141.035, 0.5490196, 0, 0.4470588, 1,
7.191919, 5.878788, -141.0936, 0.5490196, 0, 0.4470588, 1,
7.232323, 5.878788, -141.1545, 0.5490196, 0, 0.4470588, 1,
7.272727, 5.878788, -141.2177, 0.5490196, 0, 0.4470588, 1,
7.313131, 5.878788, -141.2834, 0.5490196, 0, 0.4470588, 1,
7.353535, 5.878788, -141.3514, 0.5490196, 0, 0.4470588, 1,
7.393939, 5.878788, -141.4218, 0.5490196, 0, 0.4470588, 1,
7.434343, 5.878788, -141.4945, 0.5490196, 0, 0.4470588, 1,
7.474748, 5.878788, -141.5695, 0.5490196, 0, 0.4470588, 1,
7.515152, 5.878788, -141.647, 0.5490196, 0, 0.4470588, 1,
7.555555, 5.878788, -141.7268, 0.4470588, 0, 0.5490196, 1,
7.59596, 5.878788, -141.809, 0.4470588, 0, 0.5490196, 1,
7.636364, 5.878788, -141.8935, 0.4470588, 0, 0.5490196, 1,
7.676768, 5.878788, -141.9804, 0.4470588, 0, 0.5490196, 1,
7.717172, 5.878788, -142.0697, 0.4470588, 0, 0.5490196, 1,
7.757576, 5.878788, -142.1613, 0.4470588, 0, 0.5490196, 1,
7.79798, 5.878788, -142.2552, 0.4470588, 0, 0.5490196, 1,
7.838384, 5.878788, -142.3516, 0.4470588, 0, 0.5490196, 1,
7.878788, 5.878788, -142.4503, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.878788, -142.5514, 0.4470588, 0, 0.5490196, 1,
7.959596, 5.878788, -142.6548, 0.4470588, 0, 0.5490196, 1,
8, 5.878788, -142.7606, 0.4470588, 0, 0.5490196, 1,
4, 5.919192, -143.9616, 0.3411765, 0, 0.654902, 1,
4.040404, 5.919192, -143.8376, 0.3411765, 0, 0.654902, 1,
4.080808, 5.919192, -143.716, 0.3411765, 0, 0.654902, 1,
4.121212, 5.919192, -143.5967, 0.3411765, 0, 0.654902, 1,
4.161616, 5.919192, -143.4797, 0.3411765, 0, 0.654902, 1,
4.20202, 5.919192, -143.3651, 0.3411765, 0, 0.654902, 1,
4.242424, 5.919192, -143.2528, 0.3411765, 0, 0.654902, 1,
4.282828, 5.919192, -143.1428, 0.3411765, 0, 0.654902, 1,
4.323232, 5.919192, -143.0351, 0.3411765, 0, 0.654902, 1,
4.363636, 5.919192, -142.9298, 0.4470588, 0, 0.5490196, 1,
4.40404, 5.919192, -142.8268, 0.4470588, 0, 0.5490196, 1,
4.444445, 5.919192, -142.7262, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.919192, -142.6278, 0.4470588, 0, 0.5490196, 1,
4.525252, 5.919192, -142.5318, 0.4470588, 0, 0.5490196, 1,
4.565657, 5.919192, -142.4382, 0.4470588, 0, 0.5490196, 1,
4.606061, 5.919192, -142.3468, 0.4470588, 0, 0.5490196, 1,
4.646465, 5.919192, -142.2578, 0.4470588, 0, 0.5490196, 1,
4.686869, 5.919192, -142.1711, 0.4470588, 0, 0.5490196, 1,
4.727273, 5.919192, -142.0868, 0.4470588, 0, 0.5490196, 1,
4.767677, 5.919192, -142.0047, 0.4470588, 0, 0.5490196, 1,
4.808081, 5.919192, -141.925, 0.4470588, 0, 0.5490196, 1,
4.848485, 5.919192, -141.8477, 0.4470588, 0, 0.5490196, 1,
4.888889, 5.919192, -141.7726, 0.4470588, 0, 0.5490196, 1,
4.929293, 5.919192, -141.6999, 0.5490196, 0, 0.4470588, 1,
4.969697, 5.919192, -141.6295, 0.5490196, 0, 0.4470588, 1,
5.010101, 5.919192, -141.5615, 0.5490196, 0, 0.4470588, 1,
5.050505, 5.919192, -141.4958, 0.5490196, 0, 0.4470588, 1,
5.090909, 5.919192, -141.4324, 0.5490196, 0, 0.4470588, 1,
5.131313, 5.919192, -141.3713, 0.5490196, 0, 0.4470588, 1,
5.171717, 5.919192, -141.3126, 0.5490196, 0, 0.4470588, 1,
5.212121, 5.919192, -141.2562, 0.5490196, 0, 0.4470588, 1,
5.252525, 5.919192, -141.2021, 0.5490196, 0, 0.4470588, 1,
5.292929, 5.919192, -141.1504, 0.5490196, 0, 0.4470588, 1,
5.333333, 5.919192, -141.101, 0.5490196, 0, 0.4470588, 1,
5.373737, 5.919192, -141.0539, 0.5490196, 0, 0.4470588, 1,
5.414141, 5.919192, -141.0092, 0.5490196, 0, 0.4470588, 1,
5.454545, 5.919192, -140.9667, 0.5490196, 0, 0.4470588, 1,
5.494949, 5.919192, -140.9267, 0.5490196, 0, 0.4470588, 1,
5.535354, 5.919192, -140.8889, 0.5490196, 0, 0.4470588, 1,
5.575758, 5.919192, -140.8535, 0.5490196, 0, 0.4470588, 1,
5.616162, 5.919192, -140.8204, 0.5490196, 0, 0.4470588, 1,
5.656566, 5.919192, -140.7896, 0.5490196, 0, 0.4470588, 1,
5.69697, 5.919192, -140.7611, 0.5490196, 0, 0.4470588, 1,
5.737374, 5.919192, -140.735, 0.5490196, 0, 0.4470588, 1,
5.777778, 5.919192, -140.7112, 0.5490196, 0, 0.4470588, 1,
5.818182, 5.919192, -140.6898, 0.5490196, 0, 0.4470588, 1,
5.858586, 5.919192, -140.6707, 0.5490196, 0, 0.4470588, 1,
5.89899, 5.919192, -140.6539, 0.5490196, 0, 0.4470588, 1,
5.939394, 5.919192, -140.6394, 0.5490196, 0, 0.4470588, 1,
5.979798, 5.919192, -140.6273, 0.5490196, 0, 0.4470588, 1,
6.020202, 5.919192, -140.6175, 0.5490196, 0, 0.4470588, 1,
6.060606, 5.919192, -140.61, 0.5490196, 0, 0.4470588, 1,
6.10101, 5.919192, -140.6048, 0.5490196, 0, 0.4470588, 1,
6.141414, 5.919192, -140.602, 0.5490196, 0, 0.4470588, 1,
6.181818, 5.919192, -140.6015, 0.5490196, 0, 0.4470588, 1,
6.222222, 5.919192, -140.6034, 0.5490196, 0, 0.4470588, 1,
6.262626, 5.919192, -140.6076, 0.5490196, 0, 0.4470588, 1,
6.30303, 5.919192, -140.6141, 0.5490196, 0, 0.4470588, 1,
6.343434, 5.919192, -140.6229, 0.5490196, 0, 0.4470588, 1,
6.383838, 5.919192, -140.634, 0.5490196, 0, 0.4470588, 1,
6.424242, 5.919192, -140.6475, 0.5490196, 0, 0.4470588, 1,
6.464646, 5.919192, -140.6634, 0.5490196, 0, 0.4470588, 1,
6.505051, 5.919192, -140.6815, 0.5490196, 0, 0.4470588, 1,
6.545455, 5.919192, -140.702, 0.5490196, 0, 0.4470588, 1,
6.585859, 5.919192, -140.7248, 0.5490196, 0, 0.4470588, 1,
6.626263, 5.919192, -140.7499, 0.5490196, 0, 0.4470588, 1,
6.666667, 5.919192, -140.7774, 0.5490196, 0, 0.4470588, 1,
6.707071, 5.919192, -140.8072, 0.5490196, 0, 0.4470588, 1,
6.747475, 5.919192, -140.8393, 0.5490196, 0, 0.4470588, 1,
6.787879, 5.919192, -140.8738, 0.5490196, 0, 0.4470588, 1,
6.828283, 5.919192, -140.9106, 0.5490196, 0, 0.4470588, 1,
6.868687, 5.919192, -140.9497, 0.5490196, 0, 0.4470588, 1,
6.909091, 5.919192, -140.9911, 0.5490196, 0, 0.4470588, 1,
6.949495, 5.919192, -141.0349, 0.5490196, 0, 0.4470588, 1,
6.989899, 5.919192, -141.081, 0.5490196, 0, 0.4470588, 1,
7.030303, 5.919192, -141.1295, 0.5490196, 0, 0.4470588, 1,
7.070707, 5.919192, -141.1802, 0.5490196, 0, 0.4470588, 1,
7.111111, 5.919192, -141.2333, 0.5490196, 0, 0.4470588, 1,
7.151515, 5.919192, -141.2888, 0.5490196, 0, 0.4470588, 1,
7.191919, 5.919192, -141.3465, 0.5490196, 0, 0.4470588, 1,
7.232323, 5.919192, -141.4066, 0.5490196, 0, 0.4470588, 1,
7.272727, 5.919192, -141.469, 0.5490196, 0, 0.4470588, 1,
7.313131, 5.919192, -141.5338, 0.5490196, 0, 0.4470588, 1,
7.353535, 5.919192, -141.6008, 0.5490196, 0, 0.4470588, 1,
7.393939, 5.919192, -141.6702, 0.5490196, 0, 0.4470588, 1,
7.434343, 5.919192, -141.742, 0.4470588, 0, 0.5490196, 1,
7.474748, 5.919192, -141.816, 0.4470588, 0, 0.5490196, 1,
7.515152, 5.919192, -141.8924, 0.4470588, 0, 0.5490196, 1,
7.555555, 5.919192, -141.9711, 0.4470588, 0, 0.5490196, 1,
7.59596, 5.919192, -142.0522, 0.4470588, 0, 0.5490196, 1,
7.636364, 5.919192, -142.1356, 0.4470588, 0, 0.5490196, 1,
7.676768, 5.919192, -142.2213, 0.4470588, 0, 0.5490196, 1,
7.717172, 5.919192, -142.3093, 0.4470588, 0, 0.5490196, 1,
7.757576, 5.919192, -142.3997, 0.4470588, 0, 0.5490196, 1,
7.79798, 5.919192, -142.4924, 0.4470588, 0, 0.5490196, 1,
7.838384, 5.919192, -142.5874, 0.4470588, 0, 0.5490196, 1,
7.878788, 5.919192, -142.6848, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.919192, -142.7845, 0.4470588, 0, 0.5490196, 1,
7.959596, 5.919192, -142.8865, 0.4470588, 0, 0.5490196, 1,
8, 5.919192, -142.9908, 0.3411765, 0, 0.654902, 1,
4, 5.959596, -144.1787, 0.3411765, 0, 0.654902, 1,
4.040404, 5.959596, -144.0564, 0.3411765, 0, 0.654902, 1,
4.080808, 5.959596, -143.9364, 0.3411765, 0, 0.654902, 1,
4.121212, 5.959596, -143.8187, 0.3411765, 0, 0.654902, 1,
4.161616, 5.959596, -143.7033, 0.3411765, 0, 0.654902, 1,
4.20202, 5.959596, -143.5902, 0.3411765, 0, 0.654902, 1,
4.242424, 5.959596, -143.4795, 0.3411765, 0, 0.654902, 1,
4.282828, 5.959596, -143.371, 0.3411765, 0, 0.654902, 1,
4.323232, 5.959596, -143.2648, 0.3411765, 0, 0.654902, 1,
4.363636, 5.959596, -143.1609, 0.3411765, 0, 0.654902, 1,
4.40404, 5.959596, -143.0593, 0.3411765, 0, 0.654902, 1,
4.444445, 5.959596, -142.96, 0.4470588, 0, 0.5490196, 1,
4.484848, 5.959596, -142.863, 0.4470588, 0, 0.5490196, 1,
4.525252, 5.959596, -142.7682, 0.4470588, 0, 0.5490196, 1,
4.565657, 5.959596, -142.6758, 0.4470588, 0, 0.5490196, 1,
4.606061, 5.959596, -142.5857, 0.4470588, 0, 0.5490196, 1,
4.646465, 5.959596, -142.4979, 0.4470588, 0, 0.5490196, 1,
4.686869, 5.959596, -142.4124, 0.4470588, 0, 0.5490196, 1,
4.727273, 5.959596, -142.3292, 0.4470588, 0, 0.5490196, 1,
4.767677, 5.959596, -142.2483, 0.4470588, 0, 0.5490196, 1,
4.808081, 5.959596, -142.1697, 0.4470588, 0, 0.5490196, 1,
4.848485, 5.959596, -142.0933, 0.4470588, 0, 0.5490196, 1,
4.888889, 5.959596, -142.0193, 0.4470588, 0, 0.5490196, 1,
4.929293, 5.959596, -141.9476, 0.4470588, 0, 0.5490196, 1,
4.969697, 5.959596, -141.8782, 0.4470588, 0, 0.5490196, 1,
5.010101, 5.959596, -141.811, 0.4470588, 0, 0.5490196, 1,
5.050505, 5.959596, -141.7462, 0.4470588, 0, 0.5490196, 1,
5.090909, 5.959596, -141.6837, 0.5490196, 0, 0.4470588, 1,
5.131313, 5.959596, -141.6234, 0.5490196, 0, 0.4470588, 1,
5.171717, 5.959596, -141.5655, 0.5490196, 0, 0.4470588, 1,
5.212121, 5.959596, -141.5099, 0.5490196, 0, 0.4470588, 1,
5.252525, 5.959596, -141.4565, 0.5490196, 0, 0.4470588, 1,
5.292929, 5.959596, -141.4055, 0.5490196, 0, 0.4470588, 1,
5.333333, 5.959596, -141.3568, 0.5490196, 0, 0.4470588, 1,
5.373737, 5.959596, -141.3103, 0.5490196, 0, 0.4470588, 1,
5.414141, 5.959596, -141.2662, 0.5490196, 0, 0.4470588, 1,
5.454545, 5.959596, -141.2243, 0.5490196, 0, 0.4470588, 1,
5.494949, 5.959596, -141.1848, 0.5490196, 0, 0.4470588, 1,
5.535354, 5.959596, -141.1475, 0.5490196, 0, 0.4470588, 1,
5.575758, 5.959596, -141.1126, 0.5490196, 0, 0.4470588, 1,
5.616162, 5.959596, -141.0799, 0.5490196, 0, 0.4470588, 1,
5.656566, 5.959596, -141.0495, 0.5490196, 0, 0.4470588, 1,
5.69697, 5.959596, -141.0215, 0.5490196, 0, 0.4470588, 1,
5.737374, 5.959596, -140.9957, 0.5490196, 0, 0.4470588, 1,
5.777778, 5.959596, -140.9723, 0.5490196, 0, 0.4470588, 1,
5.818182, 5.959596, -140.9511, 0.5490196, 0, 0.4470588, 1,
5.858586, 5.959596, -140.9322, 0.5490196, 0, 0.4470588, 1,
5.89899, 5.959596, -140.9157, 0.5490196, 0, 0.4470588, 1,
5.939394, 5.959596, -140.9014, 0.5490196, 0, 0.4470588, 1,
5.979798, 5.959596, -140.8894, 0.5490196, 0, 0.4470588, 1,
6.020202, 5.959596, -140.8798, 0.5490196, 0, 0.4470588, 1,
6.060606, 5.959596, -140.8724, 0.5490196, 0, 0.4470588, 1,
6.10101, 5.959596, -140.8673, 0.5490196, 0, 0.4470588, 1,
6.141414, 5.959596, -140.8645, 0.5490196, 0, 0.4470588, 1,
6.181818, 5.959596, -140.864, 0.5490196, 0, 0.4470588, 1,
6.222222, 5.959596, -140.8659, 0.5490196, 0, 0.4470588, 1,
6.262626, 5.959596, -140.87, 0.5490196, 0, 0.4470588, 1,
6.30303, 5.959596, -140.8764, 0.5490196, 0, 0.4470588, 1,
6.343434, 5.959596, -140.8851, 0.5490196, 0, 0.4470588, 1,
6.383838, 5.959596, -140.8961, 0.5490196, 0, 0.4470588, 1,
6.424242, 5.959596, -140.9094, 0.5490196, 0, 0.4470588, 1,
6.464646, 5.959596, -140.925, 0.5490196, 0, 0.4470588, 1,
6.505051, 5.959596, -140.9429, 0.5490196, 0, 0.4470588, 1,
6.545455, 5.959596, -140.9632, 0.5490196, 0, 0.4470588, 1,
6.585859, 5.959596, -140.9857, 0.5490196, 0, 0.4470588, 1,
6.626263, 5.959596, -141.0105, 0.5490196, 0, 0.4470588, 1,
6.666667, 5.959596, -141.0376, 0.5490196, 0, 0.4470588, 1,
6.707071, 5.959596, -141.0669, 0.5490196, 0, 0.4470588, 1,
6.747475, 5.959596, -141.0986, 0.5490196, 0, 0.4470588, 1,
6.787879, 5.959596, -141.1326, 0.5490196, 0, 0.4470588, 1,
6.828283, 5.959596, -141.1689, 0.5490196, 0, 0.4470588, 1,
6.868687, 5.959596, -141.2075, 0.5490196, 0, 0.4470588, 1,
6.909091, 5.959596, -141.2484, 0.5490196, 0, 0.4470588, 1,
6.949495, 5.959596, -141.2916, 0.5490196, 0, 0.4470588, 1,
6.989899, 5.959596, -141.3371, 0.5490196, 0, 0.4470588, 1,
7.030303, 5.959596, -141.3848, 0.5490196, 0, 0.4470588, 1,
7.070707, 5.959596, -141.4349, 0.5490196, 0, 0.4470588, 1,
7.111111, 5.959596, -141.4873, 0.5490196, 0, 0.4470588, 1,
7.151515, 5.959596, -141.542, 0.5490196, 0, 0.4470588, 1,
7.191919, 5.959596, -141.599, 0.5490196, 0, 0.4470588, 1,
7.232323, 5.959596, -141.6582, 0.5490196, 0, 0.4470588, 1,
7.272727, 5.959596, -141.7198, 0.5490196, 0, 0.4470588, 1,
7.313131, 5.959596, -141.7837, 0.4470588, 0, 0.5490196, 1,
7.353535, 5.959596, -141.8498, 0.4470588, 0, 0.5490196, 1,
7.393939, 5.959596, -141.9183, 0.4470588, 0, 0.5490196, 1,
7.434343, 5.959596, -141.9891, 0.4470588, 0, 0.5490196, 1,
7.474748, 5.959596, -142.0621, 0.4470588, 0, 0.5490196, 1,
7.515152, 5.959596, -142.1375, 0.4470588, 0, 0.5490196, 1,
7.555555, 5.959596, -142.2151, 0.4470588, 0, 0.5490196, 1,
7.59596, 5.959596, -142.2951, 0.4470588, 0, 0.5490196, 1,
7.636364, 5.959596, -142.3774, 0.4470588, 0, 0.5490196, 1,
7.676768, 5.959596, -142.4619, 0.4470588, 0, 0.5490196, 1,
7.717172, 5.959596, -142.5488, 0.4470588, 0, 0.5490196, 1,
7.757576, 5.959596, -142.6379, 0.4470588, 0, 0.5490196, 1,
7.79798, 5.959596, -142.7294, 0.4470588, 0, 0.5490196, 1,
7.838384, 5.959596, -142.8231, 0.4470588, 0, 0.5490196, 1,
7.878788, 5.959596, -142.9191, 0.4470588, 0, 0.5490196, 1,
7.919192, 5.959596, -143.0175, 0.3411765, 0, 0.654902, 1,
7.959596, 5.959596, -143.1181, 0.3411765, 0, 0.654902, 1,
8, 5.959596, -143.2211, 0.3411765, 0, 0.654902, 1,
4, 6, -144.396, 0.2392157, 0, 0.7568628, 1,
4.040404, 6, -144.2753, 0.2392157, 0, 0.7568628, 1,
4.080808, 6, -144.157, 0.3411765, 0, 0.654902, 1,
4.121212, 6, -144.0408, 0.3411765, 0, 0.654902, 1,
4.161616, 6, -143.927, 0.3411765, 0, 0.654902, 1,
4.20202, 6, -143.8154, 0.3411765, 0, 0.654902, 1,
4.242424, 6, -143.7061, 0.3411765, 0, 0.654902, 1,
4.282828, 6, -143.5991, 0.3411765, 0, 0.654902, 1,
4.323232, 6, -143.4943, 0.3411765, 0, 0.654902, 1,
4.363636, 6, -143.3918, 0.3411765, 0, 0.654902, 1,
4.40404, 6, -143.2916, 0.3411765, 0, 0.654902, 1,
4.444445, 6, -143.1936, 0.3411765, 0, 0.654902, 1,
4.484848, 6, -143.0979, 0.3411765, 0, 0.654902, 1,
4.525252, 6, -143.0045, 0.3411765, 0, 0.654902, 1,
4.565657, 6, -142.9133, 0.4470588, 0, 0.5490196, 1,
4.606061, 6, -142.8244, 0.4470588, 0, 0.5490196, 1,
4.646465, 6, -142.7378, 0.4470588, 0, 0.5490196, 1,
4.686869, 6, -142.6534, 0.4470588, 0, 0.5490196, 1,
4.727273, 6, -142.5713, 0.4470588, 0, 0.5490196, 1,
4.767677, 6, -142.4915, 0.4470588, 0, 0.5490196, 1,
4.808081, 6, -142.4139, 0.4470588, 0, 0.5490196, 1,
4.848485, 6, -142.3386, 0.4470588, 0, 0.5490196, 1,
4.888889, 6, -142.2656, 0.4470588, 0, 0.5490196, 1,
4.929293, 6, -142.1948, 0.4470588, 0, 0.5490196, 1,
4.969697, 6, -142.1263, 0.4470588, 0, 0.5490196, 1,
5.010101, 6, -142.0601, 0.4470588, 0, 0.5490196, 1,
5.050505, 6, -141.9961, 0.4470588, 0, 0.5490196, 1,
5.090909, 6, -141.9344, 0.4470588, 0, 0.5490196, 1,
5.131313, 6, -141.875, 0.4470588, 0, 0.5490196, 1,
5.171717, 6, -141.8179, 0.4470588, 0, 0.5490196, 1,
5.212121, 6, -141.763, 0.4470588, 0, 0.5490196, 1,
5.252525, 6, -141.7104, 0.5490196, 0, 0.4470588, 1,
5.292929, 6, -141.66, 0.5490196, 0, 0.4470588, 1,
5.333333, 6, -141.6119, 0.5490196, 0, 0.4470588, 1,
5.373737, 6, -141.5661, 0.5490196, 0, 0.4470588, 1,
5.414141, 6, -141.5225, 0.5490196, 0, 0.4470588, 1,
5.454545, 6, -141.4812, 0.5490196, 0, 0.4470588, 1,
5.494949, 6, -141.4422, 0.5490196, 0, 0.4470588, 1,
5.535354, 6, -141.4055, 0.5490196, 0, 0.4470588, 1,
5.575758, 6, -141.371, 0.5490196, 0, 0.4470588, 1,
5.616162, 6, -141.3388, 0.5490196, 0, 0.4470588, 1,
5.656566, 6, -141.3088, 0.5490196, 0, 0.4470588, 1,
5.69697, 6, -141.2812, 0.5490196, 0, 0.4470588, 1,
5.737374, 6, -141.2557, 0.5490196, 0, 0.4470588, 1,
5.777778, 6, -141.2326, 0.5490196, 0, 0.4470588, 1,
5.818182, 6, -141.2117, 0.5490196, 0, 0.4470588, 1,
5.858586, 6, -141.1931, 0.5490196, 0, 0.4470588, 1,
5.89899, 6, -141.1768, 0.5490196, 0, 0.4470588, 1,
5.939394, 6, -141.1627, 0.5490196, 0, 0.4470588, 1,
5.979798, 6, -141.1509, 0.5490196, 0, 0.4470588, 1,
6.020202, 6, -141.1413, 0.5490196, 0, 0.4470588, 1,
6.060606, 6, -141.134, 0.5490196, 0, 0.4470588, 1,
6.10101, 6, -141.129, 0.5490196, 0, 0.4470588, 1,
6.141414, 6, -141.1263, 0.5490196, 0, 0.4470588, 1,
6.181818, 6, -141.1258, 0.5490196, 0, 0.4470588, 1,
6.222222, 6, -141.1276, 0.5490196, 0, 0.4470588, 1,
6.262626, 6, -141.1317, 0.5490196, 0, 0.4470588, 1,
6.30303, 6, -141.138, 0.5490196, 0, 0.4470588, 1,
6.343434, 6, -141.1466, 0.5490196, 0, 0.4470588, 1,
6.383838, 6, -141.1575, 0.5490196, 0, 0.4470588, 1,
6.424242, 6, -141.1706, 0.5490196, 0, 0.4470588, 1,
6.464646, 6, -141.186, 0.5490196, 0, 0.4470588, 1,
6.505051, 6, -141.2037, 0.5490196, 0, 0.4470588, 1,
6.545455, 6, -141.2236, 0.5490196, 0, 0.4470588, 1,
6.585859, 6, -141.2458, 0.5490196, 0, 0.4470588, 1,
6.626263, 6, -141.2703, 0.5490196, 0, 0.4470588, 1,
6.666667, 6, -141.297, 0.5490196, 0, 0.4470588, 1,
6.707071, 6, -141.326, 0.5490196, 0, 0.4470588, 1,
6.747475, 6, -141.3573, 0.5490196, 0, 0.4470588, 1,
6.787879, 6, -141.3908, 0.5490196, 0, 0.4470588, 1,
6.828283, 6, -141.4266, 0.5490196, 0, 0.4470588, 1,
6.868687, 6, -141.4647, 0.5490196, 0, 0.4470588, 1,
6.909091, 6, -141.505, 0.5490196, 0, 0.4470588, 1,
6.949495, 6, -141.5476, 0.5490196, 0, 0.4470588, 1,
6.989899, 6, -141.5925, 0.5490196, 0, 0.4470588, 1,
7.030303, 6, -141.6396, 0.5490196, 0, 0.4470588, 1,
7.070707, 6, -141.689, 0.5490196, 0, 0.4470588, 1,
7.111111, 6, -141.7407, 0.4470588, 0, 0.5490196, 1,
7.151515, 6, -141.7947, 0.4470588, 0, 0.5490196, 1,
7.191919, 6, -141.8509, 0.4470588, 0, 0.5490196, 1,
7.232323, 6, -141.9093, 0.4470588, 0, 0.5490196, 1,
7.272727, 6, -141.9701, 0.4470588, 0, 0.5490196, 1,
7.313131, 6, -142.0331, 0.4470588, 0, 0.5490196, 1,
7.353535, 6, -142.0984, 0.4470588, 0, 0.5490196, 1,
7.393939, 6, -142.1659, 0.4470588, 0, 0.5490196, 1,
7.434343, 6, -142.2357, 0.4470588, 0, 0.5490196, 1,
7.474748, 6, -142.3078, 0.4470588, 0, 0.5490196, 1,
7.515152, 6, -142.3822, 0.4470588, 0, 0.5490196, 1,
7.555555, 6, -142.4588, 0.4470588, 0, 0.5490196, 1,
7.59596, 6, -142.5377, 0.4470588, 0, 0.5490196, 1,
7.636364, 6, -142.6188, 0.4470588, 0, 0.5490196, 1,
7.676768, 6, -142.7022, 0.4470588, 0, 0.5490196, 1,
7.717172, 6, -142.7879, 0.4470588, 0, 0.5490196, 1,
7.757576, 6, -142.8759, 0.4470588, 0, 0.5490196, 1,
7.79798, 6, -142.9661, 0.3411765, 0, 0.654902, 1,
7.838384, 6, -143.0586, 0.3411765, 0, 0.654902, 1,
7.878788, 6, -143.1533, 0.3411765, 0, 0.654902, 1,
7.919192, 6, -143.2504, 0.3411765, 0, 0.654902, 1,
7.959596, 6, -143.3497, 0.3411765, 0, 0.654902, 1,
8, 6, -143.4512, 0.3411765, 0, 0.654902, 1
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
6, 1.322, -166.6697, 0, -0.5, 0.5, 0.5,
6, 1.322, -166.6697, 1, -0.5, 0.5, 0.5,
6, 1.322, -166.6697, 1, 1.5, 0.5, 0.5,
6, 1.322, -166.6697, 0, 1.5, 0.5, 0.5
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
3.322, 4, -166.6697, 0, -0.5, 0.5, 0.5,
3.322, 4, -166.6697, 1, -0.5, 0.5, 0.5,
3.322, 4, -166.6697, 1, 1.5, 0.5, 0.5,
3.322, 4, -166.6697, 0, 1.5, 0.5, 0.5
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
3.322, 1.322, -141.7231, 0, -0.5, 0.5, 0.5,
3.322, 1.322, -141.7231, 1, -0.5, 0.5, 0.5,
3.322, 1.322, -141.7231, 1, 1.5, 0.5, 0.5,
3.322, 1.322, -141.7231, 0, 1.5, 0.5, 0.5
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
4, 1.94, -160.9128,
8, 1.94, -160.9128,
4, 1.94, -160.9128,
4, 1.837, -161.8723,
5, 1.94, -160.9128,
5, 1.837, -161.8723,
6, 1.94, -160.9128,
6, 1.837, -161.8723,
7, 1.94, -160.9128,
7, 1.837, -161.8723,
8, 1.94, -160.9128,
8, 1.837, -161.8723
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
4, 1.631, -163.7913, 0, -0.5, 0.5, 0.5,
4, 1.631, -163.7913, 1, -0.5, 0.5, 0.5,
4, 1.631, -163.7913, 1, 1.5, 0.5, 0.5,
4, 1.631, -163.7913, 0, 1.5, 0.5, 0.5,
5, 1.631, -163.7913, 0, -0.5, 0.5, 0.5,
5, 1.631, -163.7913, 1, -0.5, 0.5, 0.5,
5, 1.631, -163.7913, 1, 1.5, 0.5, 0.5,
5, 1.631, -163.7913, 0, 1.5, 0.5, 0.5,
6, 1.631, -163.7913, 0, -0.5, 0.5, 0.5,
6, 1.631, -163.7913, 1, -0.5, 0.5, 0.5,
6, 1.631, -163.7913, 1, 1.5, 0.5, 0.5,
6, 1.631, -163.7913, 0, 1.5, 0.5, 0.5,
7, 1.631, -163.7913, 0, -0.5, 0.5, 0.5,
7, 1.631, -163.7913, 1, -0.5, 0.5, 0.5,
7, 1.631, -163.7913, 1, 1.5, 0.5, 0.5,
7, 1.631, -163.7913, 0, 1.5, 0.5, 0.5,
8, 1.631, -163.7913, 0, -0.5, 0.5, 0.5,
8, 1.631, -163.7913, 1, -0.5, 0.5, 0.5,
8, 1.631, -163.7913, 1, 1.5, 0.5, 0.5,
8, 1.631, -163.7913, 0, 1.5, 0.5, 0.5
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
3.94, 2, -160.9128,
3.94, 6, -160.9128,
3.94, 2, -160.9128,
3.837, 2, -161.8723,
3.94, 3, -160.9128,
3.837, 3, -161.8723,
3.94, 4, -160.9128,
3.837, 4, -161.8723,
3.94, 5, -160.9128,
3.837, 5, -161.8723,
3.94, 6, -160.9128,
3.837, 6, -161.8723
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
"6"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[15] = gl.getAttribLocation(this.prog[15], "aOfs");
this.texture[15] = gl.createTexture();
this.texLoc[15] = gl.getAttribLocation(this.prog[15], "aTexcoord");
this.sampler[15] = gl.getUniformLocation(this.prog[15],"uSampler");
this.handleLoadedTexture(15);
this.offsets[15]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.631, 2, -163.7913, 0, -0.5, 0.5, 0.5,
3.631, 2, -163.7913, 1, -0.5, 0.5, 0.5,
3.631, 2, -163.7913, 1, 1.5, 0.5, 0.5,
3.631, 2, -163.7913, 0, 1.5, 0.5, 0.5,
3.631, 3, -163.7913, 0, -0.5, 0.5, 0.5,
3.631, 3, -163.7913, 1, -0.5, 0.5, 0.5,
3.631, 3, -163.7913, 1, 1.5, 0.5, 0.5,
3.631, 3, -163.7913, 0, 1.5, 0.5, 0.5,
3.631, 4, -163.7913, 0, -0.5, 0.5, 0.5,
3.631, 4, -163.7913, 1, -0.5, 0.5, 0.5,
3.631, 4, -163.7913, 1, 1.5, 0.5, 0.5,
3.631, 4, -163.7913, 0, 1.5, 0.5, 0.5,
3.631, 5, -163.7913, 0, -0.5, 0.5, 0.5,
3.631, 5, -163.7913, 1, -0.5, 0.5, 0.5,
3.631, 5, -163.7913, 1, 1.5, 0.5, 0.5,
3.631, 5, -163.7913, 0, 1.5, 0.5, 0.5,
3.631, 6, -163.7913, 0, -0.5, 0.5, 0.5,
3.631, 6, -163.7913, 1, -0.5, 0.5, 0.5,
3.631, 6, -163.7913, 1, 1.5, 0.5, 0.5,
3.631, 6, -163.7913, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<5; i++)
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
16, 17, 18, 16, 18, 19
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
3.94, 1.94, -160,
3.94, 1.94, -130,
3.94, 1.94, -160,
3.837, 1.837, -160,
3.94, 1.94, -150,
3.837, 1.837, -150,
3.94, 1.94, -140,
3.837, 1.837, -140,
3.94, 1.94, -130,
3.837, 1.837, -130
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
"-160",
"-150",
"-140",
"-130"
];
texinfo = drawTextToCanvas(texts, 1);
this.ofsLoc[17] = gl.getAttribLocation(this.prog[17], "aOfs");
this.texture[17] = gl.createTexture();
this.texLoc[17] = gl.getAttribLocation(this.prog[17], "aTexcoord");
this.sampler[17] = gl.getUniformLocation(this.prog[17],"uSampler");
this.handleLoadedTexture(17);
this.offsets[17]={vofs:0, cofs:-1, nofs:-1, radofs:-1, oofs:5, tofs:3, stride:7};
v=new Float32Array([
3.631, 1.631, -160, 0, -0.5, 0.5, 0.5,
3.631, 1.631, -160, 1, -0.5, 0.5, 0.5,
3.631, 1.631, -160, 1, 1.5, 0.5, 0.5,
3.631, 1.631, -160, 0, 1.5, 0.5, 0.5,
3.631, 1.631, -150, 0, -0.5, 0.5, 0.5,
3.631, 1.631, -150, 1, -0.5, 0.5, 0.5,
3.631, 1.631, -150, 1, 1.5, 0.5, 0.5,
3.631, 1.631, -150, 0, 1.5, 0.5, 0.5,
3.631, 1.631, -140, 0, -0.5, 0.5, 0.5,
3.631, 1.631, -140, 1, -0.5, 0.5, 0.5,
3.631, 1.631, -140, 1, 1.5, 0.5, 0.5,
3.631, 1.631, -140, 0, 1.5, 0.5, 0.5,
3.631, 1.631, -130, 0, -0.5, 0.5, 0.5,
3.631, 1.631, -130, 1, -0.5, 0.5, 0.5,
3.631, 1.631, -130, 1, 1.5, 0.5, 0.5,
3.631, 1.631, -130, 0, 1.5, 0.5, 0.5
]);
for (i=0; i<4; i++)
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
8, 9, 10, 8, 10, 11,
12, 13, 14, 12, 14, 15
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
3.94, 1.94, -160.9128,
3.94, 6.06, -160.9128,
3.94, 1.94, -122.5334,
3.94, 6.06, -122.5334,
3.94, 1.94, -160.9128,
3.94, 1.94, -122.5334,
3.94, 6.06, -160.9128,
3.94, 6.06, -122.5334,
3.94, 1.94, -160.9128,
8.06, 1.94, -160.9128,
3.94, 1.94, -122.5334,
8.06, 1.94, -122.5334,
3.94, 6.06, -160.9128,
8.06, 6.06, -160.9128,
3.94, 6.06, -122.5334,
8.06, 6.06, -122.5334,
8.06, 1.94, -160.9128,
8.06, 6.06, -160.9128,
8.06, 1.94, -122.5334,
8.06, 6.06, -122.5334,
8.06, 1.94, -160.9128,
8.06, 1.94, -122.5334,
8.06, 6.06, -160.9128,
8.06, 6.06, -122.5334
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
gl.drawArrays(gl.LINES, 0, 12);
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
gl.drawElements(gl.TRIANGLES, 30, gl.UNSIGNED_SHORT, 0);
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
gl.drawArrays(gl.LINES, 0, 10);
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
gl.drawElements(gl.TRIANGLES, 24, gl.UNSIGNED_SHORT, 0);
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
var radius = 20.72866,
distance = 92.22416,
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
this.mvMatrix.translate( -6, -4, 141.7231 );
this.mvMatrix.scale( 5.439863, 5.439863, 0.5839657 );
this.mvMatrix.multRight( unnamed_chunk_41rgl.userMatrix[1] );
this.mvMatrix.translate(-0, -0, -92.22416);
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
##            mu    sigma    loglik   color
## 2155 6.181818 2.848485 -123.0924 #FFFF00
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
##  6.170227  1.042925 
## 
## $value
## [1] 123.0912
## 
## $counts
## function gradient 
##       59       NA 
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
## [1,] 6.181818 2.848485
## [2,] 6.170227 2.837506
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
##      alpha       beta  log_sigma 
## 3.28009218 9.53705146 0.09467786 
## 
## $value
## [1] 30.27319
## 
## $counts
## function gradient 
##      142       NA 
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
## #  Elapsed Time: 0.005883 seconds (Warm-up)
## #                0.0048 seconds (Sampling)
## #                0.010683 seconds (Total)
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
## #  Elapsed Time: 0.005634 seconds (Warm-up)
## #                0.00512 seconds (Sampling)
## #                0.010754 seconds (Total)
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
## #  Elapsed Time: 0.005702 seconds (Warm-up)
## #                0.005164 seconds (Sampling)
## #                0.010866 seconds (Total)
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
## #  Elapsed Time: 0.005816 seconds (Warm-up)
## #                0.005085 seconds (Sampling)
## #                0.010901 seconds (Total)
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
## p     0.30    0.00 0.14  0.07  0.19  0.29  0.39  0.61  1462    1
## lp__ -6.66    0.02 0.81 -8.95 -6.82 -6.34 -6.16 -6.11  1409    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:47:25 2015.
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
## beta[1] -2.64    0.01 0.36 -3.31 -2.89 -2.65 -2.41 -1.89  1315    1
## beta[2]  0.80    0.01 0.21  0.38  0.67  0.80  0.93  1.19  1284    1
## sigma    0.83    0.00 0.15  0.59  0.73  0.81  0.91  1.18  1401    1
## lp__    -5.95    0.04 1.27 -9.34 -6.52 -5.59 -5.02 -4.50  1146    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:47:53 2015.
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
## [1] 0.9995
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
##     2.5%    97.5% 
## 0.377329 1.190671
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
## [1] 0.3895868 1.1985794
```

In this instance, our posterior is fairly symmetric and the two types of credible intervals will not be much different. 

## Further reading

Gelman and Hill. 2009. *Data analysis using regression and multilevel/hierarchical models*. Chapter 18.

Hobbs and Hooten. 2015. *Bayesian models: a statistical primer for ecologists*. Chapter 7. 

Gelman et al. 2014. *Bayesian data analysis. Third edition*. 

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
## -2.0848  -1.2064  -0.3074   0.6838   2.7958  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)   0.4964     0.1138   4.363 1.28e-05 ***
## x            -0.9085     0.2069  -4.392 1.12e-05 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 88.396  on 49  degrees of freedom
## Residual deviance: 68.888  on 48  degrees of freedom
## AIC: 167.31
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
## (Intercept)  0.2639737  0.7107933
## x           -1.3173875 -0.5049892
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
## beta[1]   0.48    0.00 0.11   0.26   0.40   0.49   0.56   0.71  1866    1
## beta[2]  -0.90    0.00 0.21  -1.31  -1.04  -0.91  -0.77  -0.49  1763    1
## lp__    -30.67    0.03 0.98 -33.29 -31.07 -30.37 -29.96 -29.69  1313    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:48:23 2015.
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
## beta[1]   0.49    0.00 0.11   0.26   0.41   0.49   0.57   0.70  1143    1
## lp__    -41.97    0.02 0.74 -44.08 -42.13 -41.68 -41.50 -41.45  1320    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:48:27 2015.
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

$$y \sim Poisson(\lambda)$$

$$log(\lambda) = X \beta + \epsilon$$

$$\epsilon \sim Normal(0, \sigma)$$

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
## sigma     1.22    0.01 0.23   0.82   1.05   1.19   1.36   1.73   638    1
## beta[1]  -0.15    0.01 0.26  -0.72  -0.31  -0.14   0.03   0.31   764    1
## lp__    -29.66    0.30 7.35 -45.10 -34.46 -29.39 -24.52 -16.09   593    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:48:56 2015.
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
## beta[1]   0.50    0.00 0.19   0.14   0.38   0.50   0.62   0.89  1855    1
## phi       0.85    0.01 0.31   0.41   0.64   0.80   1.01   1.60  1763    1
## lp__    -20.75    0.03 0.98 -23.41 -21.10 -20.45 -20.05 -19.80  1239    1
## 
## Samples were drawn using NUTS(diag_e) at Sun Nov 22 19:49:25 2015.
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


Chapter 5: Binomial models
===============

Here, the students will continue to use a combination of methods for implementation. Key points to take away from this section include the properties/behavior of bionomial models, ways to check binomial modles, and a hint that Bayesian approaches are going to be more flexible.  The binomial-Poisson hierarchical model is a classic that should reinforce the notion that Bayesian approaches will generally be easier for more complex examples.

#### Priorities

- binomial distribution (relationship between mean and variance)
- logit link as a map
- proportion vs. binary models (will help with understanding hierarchical models later)
- implementation with `glm`
- overdispersion in proportion models and understanding the difference between individual and group level probabilities
- implementation with  Stan, JAGS
- separation
- hierarchical model number 1: occupancy model (a classic) (maybe, or we could do it later)
  - review marginalization 
- graphical displays
- model checking

#### Optional

- simulation of data & parameter recovery


Chapter 6: Partial pooling and likelihood
==============================

The main dish. I'd like to avoid a recipe-based approach where we discuss varying intercept and varying slope models as primary objectives. Instead, I think it's important to cover these topics as special cases of the general approach of hierarchical modeling as a means to impose probabilistic structures on parameters. From that perspective, students should be able to better extend these methods for their own work.

#### Priorities

- definition
- review previous examples
- hyperparameters (they've always been there even when we don't acknowledge them)
- varying intercepts (NBA freethrow example) with `lme4`
- partial pooling
- clearing up confusion about nestedness
- simple hierarchical models with likelihood
- continous predictors for multiple levels

#### Optional

- plotting estimates for different levels from lme4 models



Chapter 7: Bayesian hierarchical models
============================

#### Priorities

- varying intercepts (NBA freethrow example) with `Stan`
- hierarchical models in Stan
- highlight Bayesian connection to priors
- classic examples: 
    - hierarchical model number 1: occupancy model (a classic)
    - hierarchical model: binomial-Poisson hierarchy (e.g. # eggs laid & survival)
- introduction to the multivariate normal distribution
- parameters for hierarchical variance parameters
- prediction (new vs. observed groups)
- priors 
- note crossing of the 'ease' threshold (?)


#### Optional

- posterior prediction
- basic Bayesian models in MCMCglmm
- random effects, fixed effects, mixed effects models as special instances of hierarchical linear models

#### Reading

Gelman, A., J. Hill, and M. Yajima. 2012. Why We (Usually) Don’t Have to Worry About Multiple Comparisons. Journal of Research on Educational Effectiveness 5:189–211.  
Gelman and Hill discussion of random effects terminology (very good)


Chapter 8: Hierarchical model construction
===============================

This is where I think we will have the greatest impact on students future work. Translating problems to models is a key skill, and it may take a fair bit of practice. Tools to implement include graphical skills (e.g. drawing DAGs), and familiarity with probability distributions.

- parameter, process, and observation models
- building complexity from simple pieces
- translating biological problems and observations to models
- example: what method works best for detecting a species?
- example: error in variables models
- more practice in developing models (don't necessarily have to implement)



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
