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
##      Min       1Q   Median       3Q      Max 
## -1.72012 -0.37640 -0.04246  0.43507  1.15490 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)
## (Intercept)   0.2365     0.1555   1.521    0.145
## 
## Residual standard error: 0.6956 on 19 degrees of freedom
```

The summary of our model object `m` provides a lot of information. 
For reasons that will become clear shortly, the estimated population mean is referred to as the "Intercept". 
Here, we get a point estimate for the population mean $\mu$: 0.237 and an estimate of the residual standard deviation $\sigma$: 0.696, which we can square to get an estimate of the residual variance $\sigma^2$: 0.484.

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
## -0.95307 -0.31503 -0.00134  0.33680  1.07898 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -2.0456     0.1415  -14.45  < 2e-16 ***
## x             3.0310     0.2478   12.23 2.32e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.469 on 48 degrees of freedom
## Multiple R-squared:  0.7571,	Adjusted R-squared:  0.7521 
## F-statistic: 149.6 on 1 and 48 DF,  p-value: 2.321e-16
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
## (Intercept) -4.2940107 -2.618840
## x            0.8916902  1.362811
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
## -2.52881 -0.69270  0.02409  0.86102  2.45571 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   3.4103     0.2472  13.794  < 2e-16 ***
## xstrawberry  -2.8431     0.3496  -8.131 4.16e-11 ***
## xvanilla     -2.0911     0.3496  -5.981 1.55e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.106 on 57 degrees of freedom
## Multiple R-squared:  0.5547,	Adjusted R-squared:  0.5391 
## F-statistic:  35.5 on 2 and 57 DF,  p-value: 9.69e-11
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
## -2.52881 -0.69270  0.02409  0.86102  2.45571 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## xchocolate    3.4103     0.2472  13.794  < 2e-16 ***
## xstrawberry   0.5673     0.2472   2.294   0.0255 *  
## xvanilla      1.3192     0.2472   5.336 1.71e-06 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.106 on 57 degrees of freedom
## Multiple R-squared:  0.7972,	Adjusted R-squared:  0.7865 
## F-statistic: 74.67 on 3 and 57 DF,  p-value: < 2.2e-16
```

Arguably, this approach is more useful because it simplifies the construction of confidence intervals for the group means:


```r
confint(m)
```

```
##                  2.5 %   97.5 %
## xchocolate  2.91524176 3.905418
## xstrawberry 0.07216815 1.062344
## xvanilla    0.82411941 1.814296
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
## -2.0625 -0.6593 -0.0798  0.8095  2.2110 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.2506     0.1497   1.674    0.101    
## x1            1.0563     0.1443   7.319 3.03e-09 ***
## x2           -0.9545     0.1376  -6.939 1.13e-08 ***
## x1:x2         2.0981     0.1151  18.235  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.046 on 46 degrees of freedom
## Multiple R-squared:  0.9137,	Adjusted R-squared:  0.9081 
## F-statistic: 162.4 on 3 and 46 DF,  p-value: < 2.2e-16
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
## -1.41994 -0.27408 -0.04474  0.24786  1.04121 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.2019     0.1150   1.755   0.0859 .  
## x1           -0.1100     0.1005  -1.095   0.2794    
## x2B          -1.2037     0.1373  -8.767 2.25e-11 ***
## x1:x2B       -0.2729     0.1270  -2.148   0.0370 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4427 on 46 degrees of freedom
## Multiple R-squared:  0.7009,	Adjusted R-squared:  0.6814 
## F-statistic: 35.93 on 3 and 46 DF,  p-value: 4.081e-12
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
4, 2, -693.1819, 0.627451, 0.1254902, 0.9411765, 1,
4.040404, 2, -689.0544, 0.627451, 0.1254902, 0.9411765, 1,
4.080808, 2, -685.0085, 0.5607843, 0.1098039, 0.945098, 1,
4.121212, 2, -681.0443, 0.4941176, 0.09803922, 0.9529412, 1,
4.161616, 2, -677.1617, 0.4941176, 0.09803922, 0.9529412, 1,
4.20202, 2, -673.3607, 0.4313726, 0.08627451, 0.9568627, 1,
4.242424, 2, -669.6414, 0.3647059, 0.07058824, 0.9647059, 1,
4.282828, 2, -666.0037, 0.3647059, 0.07058824, 0.9647059, 1,
4.323232, 2, -662.4475, 0.3019608, 0.05882353, 0.9686275, 1,
4.363636, 2, -658.973, 0.3019608, 0.05882353, 0.9686275, 1,
4.40404, 2, -655.5802, 0.2352941, 0.04705882, 0.9764706, 1,
4.444445, 2, -652.2689, 0.172549, 0.03137255, 0.9803922, 1,
4.484848, 2, -649.0393, 0.172549, 0.03137255, 0.9803922, 1,
4.525252, 2, -645.8914, 0.1058824, 0.01960784, 0.9882353, 1,
4.565657, 2, -642.825, 0.1058824, 0.01960784, 0.9882353, 1,
4.606061, 2, -639.8403, 0.04313726, 0.007843138, 0.9921569, 1,
4.646465, 2, -636.9371, 0.04313726, 0.007843138, 0.9921569, 1,
4.686869, 2, -634.1157, 0.03137255, 0, 0.9647059, 1,
4.727273, 2, -631.3758, 0.03137255, 0, 0.9647059, 1,
4.767677, 2, -628.7175, 0.1372549, 0, 0.8588235, 1,
4.808081, 2, -626.1409, 0.1372549, 0, 0.8588235, 1,
4.848485, 2, -623.6459, 0.2392157, 0, 0.7568628, 1,
4.888889, 2, -621.2325, 0.2392157, 0, 0.7568628, 1,
4.929293, 2, -618.9008, 0.2392157, 0, 0.7568628, 1,
4.969697, 2, -616.6507, 0.3411765, 0, 0.654902, 1,
5.010101, 2, -614.4822, 0.3411765, 0, 0.654902, 1,
5.050505, 2, -612.3953, 0.3411765, 0, 0.654902, 1,
5.090909, 2, -610.3901, 0.4470588, 0, 0.5490196, 1,
5.131313, 2, -608.4664, 0.4470588, 0, 0.5490196, 1,
5.171717, 2, -606.6245, 0.4470588, 0, 0.5490196, 1,
5.212121, 2, -604.8641, 0.5490196, 0, 0.4470588, 1,
5.252525, 2, -603.1853, 0.5490196, 0, 0.4470588, 1,
5.292929, 2, -601.5882, 0.5490196, 0, 0.4470588, 1,
5.333333, 2, -600.0727, 0.654902, 0, 0.3411765, 1,
5.373737, 2, -598.6389, 0.654902, 0, 0.3411765, 1,
5.414141, 2, -597.2866, 0.654902, 0, 0.3411765, 1,
5.454545, 2, -596.0159, 0.654902, 0, 0.3411765, 1,
5.494949, 2, -594.827, 0.7568628, 0, 0.2392157, 1,
5.535354, 2, -593.7195, 0.7568628, 0, 0.2392157, 1,
5.575758, 2, -592.6938, 0.7568628, 0, 0.2392157, 1,
5.616162, 2, -591.7497, 0.7568628, 0, 0.2392157, 1,
5.656566, 2, -590.8871, 0.7568628, 0, 0.2392157, 1,
5.69697, 2, -590.1063, 0.7568628, 0, 0.2392157, 1,
5.737374, 2, -589.407, 0.7568628, 0, 0.2392157, 1,
5.777778, 2, -588.7894, 0.8588235, 0, 0.1372549, 1,
5.818182, 2, -588.2534, 0.8588235, 0, 0.1372549, 1,
5.858586, 2, -587.799, 0.8588235, 0, 0.1372549, 1,
5.89899, 2, -587.4263, 0.8588235, 0, 0.1372549, 1,
5.939394, 2, -587.1351, 0.8588235, 0, 0.1372549, 1,
5.979798, 2, -586.9256, 0.8588235, 0, 0.1372549, 1,
6.020202, 2, -586.7977, 0.8588235, 0, 0.1372549, 1,
6.060606, 2, -586.7515, 0.8588235, 0, 0.1372549, 1,
6.10101, 2, -586.7868, 0.8588235, 0, 0.1372549, 1,
6.141414, 2, -586.9038, 0.8588235, 0, 0.1372549, 1,
6.181818, 2, -587.1024, 0.8588235, 0, 0.1372549, 1,
6.222222, 2, -587.3826, 0.8588235, 0, 0.1372549, 1,
6.262626, 2, -587.7445, 0.8588235, 0, 0.1372549, 1,
6.30303, 2, -588.188, 0.8588235, 0, 0.1372549, 1,
6.343434, 2, -588.7131, 0.8588235, 0, 0.1372549, 1,
6.383838, 2, -589.3198, 0.7568628, 0, 0.2392157, 1,
6.424242, 2, -590.0082, 0.7568628, 0, 0.2392157, 1,
6.464646, 2, -590.7781, 0.7568628, 0, 0.2392157, 1,
6.505051, 2, -591.6298, 0.7568628, 0, 0.2392157, 1,
6.545455, 2, -592.563, 0.7568628, 0, 0.2392157, 1,
6.585859, 2, -593.5778, 0.7568628, 0, 0.2392157, 1,
6.626263, 2, -594.6743, 0.7568628, 0, 0.2392157, 1,
6.666667, 2, -595.8524, 0.654902, 0, 0.3411765, 1,
6.707071, 2, -597.1121, 0.654902, 0, 0.3411765, 1,
6.747475, 2, -598.4535, 0.654902, 0, 0.3411765, 1,
6.787879, 2, -599.8765, 0.654902, 0, 0.3411765, 1,
6.828283, 2, -601.381, 0.5490196, 0, 0.4470588, 1,
6.868687, 2, -602.9673, 0.5490196, 0, 0.4470588, 1,
6.909091, 2, -604.6351, 0.5490196, 0, 0.4470588, 1,
6.949495, 2, -606.3846, 0.5490196, 0, 0.4470588, 1,
6.989899, 2, -608.2157, 0.4470588, 0, 0.5490196, 1,
7.030303, 2, -610.1284, 0.4470588, 0, 0.5490196, 1,
7.070707, 2, -612.1227, 0.4470588, 0, 0.5490196, 1,
7.111111, 2, -614.1987, 0.3411765, 0, 0.654902, 1,
7.151515, 2, -616.3563, 0.3411765, 0, 0.654902, 1,
7.191919, 2, -618.5955, 0.2392157, 0, 0.7568628, 1,
7.232323, 2, -620.9164, 0.2392157, 0, 0.7568628, 1,
7.272727, 2, -623.3188, 0.2392157, 0, 0.7568628, 1,
7.313131, 2, -625.8029, 0.1372549, 0, 0.8588235, 1,
7.353535, 2, -628.3687, 0.1372549, 0, 0.8588235, 1,
7.393939, 2, -631.016, 0.03137255, 0, 0.9647059, 1,
7.434343, 2, -633.7449, 0.03137255, 0, 0.9647059, 1,
7.474748, 2, -636.5555, 0.04313726, 0.007843138, 0.9921569, 1,
7.515152, 2, -639.4478, 0.04313726, 0.007843138, 0.9921569, 1,
7.555555, 2, -642.4216, 0.1058824, 0.01960784, 0.9882353, 1,
7.59596, 2, -645.4771, 0.1058824, 0.01960784, 0.9882353, 1,
7.636364, 2, -648.6141, 0.172549, 0.03137255, 0.9803922, 1,
7.676768, 2, -651.8329, 0.172549, 0.03137255, 0.9803922, 1,
7.717172, 2, -655.1332, 0.2352941, 0.04705882, 0.9764706, 1,
7.757576, 2, -658.5151, 0.3019608, 0.05882353, 0.9686275, 1,
7.79798, 2, -661.9788, 0.3019608, 0.05882353, 0.9686275, 1,
7.838384, 2, -665.5239, 0.3647059, 0.07058824, 0.9647059, 1,
7.878788, 2, -669.1508, 0.3647059, 0.07058824, 0.9647059, 1,
7.919192, 2, -672.8593, 0.4313726, 0.08627451, 0.9568627, 1,
7.959596, 2, -676.6493, 0.4941176, 0.09803922, 0.9529412, 1,
8, 2, -680.521, 0.4941176, 0.09803922, 0.9529412, 1,
4, 2.050505, -680.1303, 0.4941176, 0.09803922, 0.9529412, 1,
4.040404, 2.050505, -676.2037, 0.4941176, 0.09803922, 0.9529412, 1,
4.080808, 2.050505, -672.3547, 0.4313726, 0.08627451, 0.9568627, 1,
4.121212, 2.050505, -668.5833, 0.3647059, 0.07058824, 0.9647059, 1,
4.161616, 2.050505, -664.8896, 0.3647059, 0.07058824, 0.9647059, 1,
4.20202, 2.050505, -661.2736, 0.3019608, 0.05882353, 0.9686275, 1,
4.242424, 2.050505, -657.7352, 0.2352941, 0.04705882, 0.9764706, 1,
4.282828, 2.050505, -654.2744, 0.2352941, 0.04705882, 0.9764706, 1,
4.323232, 2.050505, -650.8914, 0.172549, 0.03137255, 0.9803922, 1,
4.363636, 2.050505, -647.5859, 0.172549, 0.03137255, 0.9803922, 1,
4.40404, 2.050505, -644.3581, 0.1058824, 0.01960784, 0.9882353, 1,
4.444445, 2.050505, -641.208, 0.1058824, 0.01960784, 0.9882353, 1,
4.484848, 2.050505, -638.1355, 0.04313726, 0.007843138, 0.9921569, 1,
4.525252, 2.050505, -635.1407, 0.03137255, 0, 0.9647059, 1,
4.565657, 2.050505, -632.2235, 0.03137255, 0, 0.9647059, 1,
4.606061, 2.050505, -629.384, 0.1372549, 0, 0.8588235, 1,
4.646465, 2.050505, -626.6221, 0.1372549, 0, 0.8588235, 1,
4.686869, 2.050505, -623.9379, 0.1372549, 0, 0.8588235, 1,
4.727273, 2.050505, -621.3314, 0.2392157, 0, 0.7568628, 1,
4.767677, 2.050505, -618.8024, 0.2392157, 0, 0.7568628, 1,
4.808081, 2.050505, -616.3512, 0.3411765, 0, 0.654902, 1,
4.848485, 2.050505, -613.9776, 0.3411765, 0, 0.654902, 1,
4.888889, 2.050505, -611.6816, 0.4470588, 0, 0.5490196, 1,
4.929293, 2.050505, -609.4634, 0.4470588, 0, 0.5490196, 1,
4.969697, 2.050505, -607.3227, 0.4470588, 0, 0.5490196, 1,
5.010101, 2.050505, -605.2597, 0.5490196, 0, 0.4470588, 1,
5.050505, 2.050505, -603.2744, 0.5490196, 0, 0.4470588, 1,
5.090909, 2.050505, -601.3667, 0.5490196, 0, 0.4470588, 1,
5.131313, 2.050505, -599.5367, 0.654902, 0, 0.3411765, 1,
5.171717, 2.050505, -597.7843, 0.654902, 0, 0.3411765, 1,
5.212121, 2.050505, -596.1096, 0.654902, 0, 0.3411765, 1,
5.252525, 2.050505, -594.5125, 0.7568628, 0, 0.2392157, 1,
5.292929, 2.050505, -592.9931, 0.7568628, 0, 0.2392157, 1,
5.333333, 2.050505, -591.5513, 0.7568628, 0, 0.2392157, 1,
5.373737, 2.050505, -590.1872, 0.7568628, 0, 0.2392157, 1,
5.414141, 2.050505, -588.9008, 0.8588235, 0, 0.1372549, 1,
5.454545, 2.050505, -587.692, 0.8588235, 0, 0.1372549, 1,
5.494949, 2.050505, -586.5608, 0.8588235, 0, 0.1372549, 1,
5.535354, 2.050505, -585.5073, 0.8588235, 0, 0.1372549, 1,
5.575758, 2.050505, -584.5314, 0.8588235, 0, 0.1372549, 1,
5.616162, 2.050505, -583.6332, 0.8588235, 0, 0.1372549, 1,
5.656566, 2.050505, -582.8127, 0.9647059, 0, 0.03137255, 1,
5.69697, 2.050505, -582.0698, 0.9647059, 0, 0.03137255, 1,
5.737374, 2.050505, -581.4046, 0.9647059, 0, 0.03137255, 1,
5.777778, 2.050505, -580.817, 0.9647059, 0, 0.03137255, 1,
5.818182, 2.050505, -580.3071, 0.9647059, 0, 0.03137255, 1,
5.858586, 2.050505, -579.8748, 0.9647059, 0, 0.03137255, 1,
5.89899, 2.050505, -579.5201, 0.9647059, 0, 0.03137255, 1,
5.939394, 2.050505, -579.2432, 0.9647059, 0, 0.03137255, 1,
5.979798, 2.050505, -579.0439, 0.9647059, 0, 0.03137255, 1,
6.020202, 2.050505, -578.9222, 0.9647059, 0, 0.03137255, 1,
6.060606, 2.050505, -578.8782, 0.9647059, 0, 0.03137255, 1,
6.10101, 2.050505, -578.9119, 0.9647059, 0, 0.03137255, 1,
6.141414, 2.050505, -579.0231, 0.9647059, 0, 0.03137255, 1,
6.181818, 2.050505, -579.2121, 0.9647059, 0, 0.03137255, 1,
6.222222, 2.050505, -579.4787, 0.9647059, 0, 0.03137255, 1,
6.262626, 2.050505, -579.8229, 0.9647059, 0, 0.03137255, 1,
6.30303, 2.050505, -580.2448, 0.9647059, 0, 0.03137255, 1,
6.343434, 2.050505, -580.7444, 0.9647059, 0, 0.03137255, 1,
6.383838, 2.050505, -581.3216, 0.9647059, 0, 0.03137255, 1,
6.424242, 2.050505, -581.9764, 0.9647059, 0, 0.03137255, 1,
6.464646, 2.050505, -582.709, 0.9647059, 0, 0.03137255, 1,
6.505051, 2.050505, -583.5192, 0.8588235, 0, 0.1372549, 1,
6.545455, 2.050505, -584.407, 0.8588235, 0, 0.1372549, 1,
6.585859, 2.050505, -585.3724, 0.8588235, 0, 0.1372549, 1,
6.626263, 2.050505, -586.4156, 0.8588235, 0, 0.1372549, 1,
6.666667, 2.050505, -587.5364, 0.8588235, 0, 0.1372549, 1,
6.707071, 2.050505, -588.7348, 0.8588235, 0, 0.1372549, 1,
6.747475, 2.050505, -590.0109, 0.7568628, 0, 0.2392157, 1,
6.787879, 2.050505, -591.3646, 0.7568628, 0, 0.2392157, 1,
6.828283, 2.050505, -592.796, 0.7568628, 0, 0.2392157, 1,
6.868687, 2.050505, -594.3051, 0.7568628, 0, 0.2392157, 1,
6.909091, 2.050505, -595.8918, 0.654902, 0, 0.3411765, 1,
6.949495, 2.050505, -597.5561, 0.654902, 0, 0.3411765, 1,
6.989899, 2.050505, -599.2982, 0.654902, 0, 0.3411765, 1,
7.030303, 2.050505, -601.1178, 0.5490196, 0, 0.4470588, 1,
7.070707, 2.050505, -603.0151, 0.5490196, 0, 0.4470588, 1,
7.111111, 2.050505, -604.9901, 0.5490196, 0, 0.4470588, 1,
7.151515, 2.050505, -607.0427, 0.4470588, 0, 0.5490196, 1,
7.191919, 2.050505, -609.173, 0.4470588, 0, 0.5490196, 1,
7.232323, 2.050505, -611.3809, 0.4470588, 0, 0.5490196, 1,
7.272727, 2.050505, -613.6664, 0.3411765, 0, 0.654902, 1,
7.313131, 2.050505, -616.0297, 0.3411765, 0, 0.654902, 1,
7.353535, 2.050505, -618.4706, 0.2392157, 0, 0.7568628, 1,
7.393939, 2.050505, -620.9891, 0.2392157, 0, 0.7568628, 1,
7.434343, 2.050505, -623.5853, 0.2392157, 0, 0.7568628, 1,
7.474748, 2.050505, -626.2591, 0.1372549, 0, 0.8588235, 1,
7.515152, 2.050505, -629.0106, 0.1372549, 0, 0.8588235, 1,
7.555555, 2.050505, -631.8398, 0.03137255, 0, 0.9647059, 1,
7.59596, 2.050505, -634.7466, 0.03137255, 0, 0.9647059, 1,
7.636364, 2.050505, -637.731, 0.04313726, 0.007843138, 0.9921569, 1,
7.676768, 2.050505, -640.7931, 0.04313726, 0.007843138, 0.9921569, 1,
7.717172, 2.050505, -643.9329, 0.1058824, 0.01960784, 0.9882353, 1,
7.757576, 2.050505, -647.1503, 0.172549, 0.03137255, 0.9803922, 1,
7.79798, 2.050505, -650.4454, 0.172549, 0.03137255, 0.9803922, 1,
7.838384, 2.050505, -653.8181, 0.2352941, 0.04705882, 0.9764706, 1,
7.878788, 2.050505, -657.2684, 0.2352941, 0.04705882, 0.9764706, 1,
7.919192, 2.050505, -660.7964, 0.3019608, 0.05882353, 0.9686275, 1,
7.959596, 2.050505, -664.4022, 0.3647059, 0.07058824, 0.9647059, 1,
8, 2.050505, -668.0854, 0.3647059, 0.07058824, 0.9647059, 1,
4, 2.10101, -668.2426, 0.3647059, 0.07058824, 0.9647059, 1,
4.040404, 2.10101, -664.5024, 0.3647059, 0.07058824, 0.9647059, 1,
4.080808, 2.10101, -660.8363, 0.3019608, 0.05882353, 0.9686275, 1,
4.121212, 2.10101, -657.2441, 0.2352941, 0.04705882, 0.9764706, 1,
4.161616, 2.10101, -653.7258, 0.2352941, 0.04705882, 0.9764706, 1,
4.20202, 2.10101, -650.2816, 0.172549, 0.03137255, 0.9803922, 1,
4.242424, 2.10101, -646.9112, 0.172549, 0.03137255, 0.9803922, 1,
4.282828, 2.10101, -643.6149, 0.1058824, 0.01960784, 0.9882353, 1,
4.323232, 2.10101, -640.3925, 0.04313726, 0.007843138, 0.9921569, 1,
4.363636, 2.10101, -637.244, 0.04313726, 0.007843138, 0.9921569, 1,
4.40404, 2.10101, -634.1696, 0.03137255, 0, 0.9647059, 1,
4.444445, 2.10101, -631.1691, 0.03137255, 0, 0.9647059, 1,
4.484848, 2.10101, -628.2425, 0.1372549, 0, 0.8588235, 1,
4.525252, 2.10101, -625.39, 0.1372549, 0, 0.8588235, 1,
4.565657, 2.10101, -622.6113, 0.2392157, 0, 0.7568628, 1,
4.606061, 2.10101, -619.9067, 0.2392157, 0, 0.7568628, 1,
4.646465, 2.10101, -617.276, 0.3411765, 0, 0.654902, 1,
4.686869, 2.10101, -614.7193, 0.3411765, 0, 0.654902, 1,
4.727273, 2.10101, -612.2366, 0.3411765, 0, 0.654902, 1,
4.767677, 2.10101, -609.8278, 0.4470588, 0, 0.5490196, 1,
4.808081, 2.10101, -607.4929, 0.4470588, 0, 0.5490196, 1,
4.848485, 2.10101, -605.2321, 0.5490196, 0, 0.4470588, 1,
4.888889, 2.10101, -603.0452, 0.5490196, 0, 0.4470588, 1,
4.929293, 2.10101, -600.9323, 0.5490196, 0, 0.4470588, 1,
4.969697, 2.10101, -598.8933, 0.654902, 0, 0.3411765, 1,
5.010101, 2.10101, -596.9283, 0.654902, 0, 0.3411765, 1,
5.050505, 2.10101, -595.0373, 0.654902, 0, 0.3411765, 1,
5.090909, 2.10101, -593.2202, 0.7568628, 0, 0.2392157, 1,
5.131313, 2.10101, -591.4771, 0.7568628, 0, 0.2392157, 1,
5.171717, 2.10101, -589.8079, 0.7568628, 0, 0.2392157, 1,
5.212121, 2.10101, -588.2128, 0.8588235, 0, 0.1372549, 1,
5.252525, 2.10101, -586.6916, 0.8588235, 0, 0.1372549, 1,
5.292929, 2.10101, -585.2443, 0.8588235, 0, 0.1372549, 1,
5.333333, 2.10101, -583.871, 0.8588235, 0, 0.1372549, 1,
5.373737, 2.10101, -582.5717, 0.9647059, 0, 0.03137255, 1,
5.414141, 2.10101, -581.3464, 0.9647059, 0, 0.03137255, 1,
5.454545, 2.10101, -580.1949, 0.9647059, 0, 0.03137255, 1,
5.494949, 2.10101, -579.1176, 0.9647059, 0, 0.03137255, 1,
5.535354, 2.10101, -578.1141, 0.9647059, 0, 0.03137255, 1,
5.575758, 2.10101, -577.1846, 1, 0.06666667, 0, 1,
5.616162, 2.10101, -576.329, 1, 0.06666667, 0, 1,
5.656566, 2.10101, -575.5475, 1, 0.06666667, 0, 1,
5.69697, 2.10101, -574.8399, 1, 0.06666667, 0, 1,
5.737374, 2.10101, -574.2062, 1, 0.06666667, 0, 1,
5.777778, 2.10101, -573.6466, 1, 0.06666667, 0, 1,
5.818182, 2.10101, -573.1609, 1, 0.06666667, 0, 1,
5.858586, 2.10101, -572.7491, 1, 0.06666667, 0, 1,
5.89899, 2.10101, -572.4113, 1, 0.06666667, 0, 1,
5.939394, 2.10101, -572.1475, 1, 0.06666667, 0, 1,
5.979798, 2.10101, -571.9577, 1, 0.06666667, 0, 1,
6.020202, 2.10101, -571.8418, 1, 0.06666667, 0, 1,
6.060606, 2.10101, -571.7999, 1, 0.06666667, 0, 1,
6.10101, 2.10101, -571.8319, 1, 0.06666667, 0, 1,
6.141414, 2.10101, -571.9379, 1, 0.06666667, 0, 1,
6.181818, 2.10101, -572.1179, 1, 0.06666667, 0, 1,
6.222222, 2.10101, -572.3718, 1, 0.06666667, 0, 1,
6.262626, 2.10101, -572.6997, 1, 0.06666667, 0, 1,
6.30303, 2.10101, -573.1016, 1, 0.06666667, 0, 1,
6.343434, 2.10101, -573.5775, 1, 0.06666667, 0, 1,
6.383838, 2.10101, -574.1272, 1, 0.06666667, 0, 1,
6.424242, 2.10101, -574.751, 1, 0.06666667, 0, 1,
6.464646, 2.10101, -575.4487, 1, 0.06666667, 0, 1,
6.505051, 2.10101, -576.2204, 1, 0.06666667, 0, 1,
6.545455, 2.10101, -577.066, 1, 0.06666667, 0, 1,
6.585859, 2.10101, -577.9857, 0.9647059, 0, 0.03137255, 1,
6.626263, 2.10101, -578.9792, 0.9647059, 0, 0.03137255, 1,
6.666667, 2.10101, -580.0468, 0.9647059, 0, 0.03137255, 1,
6.707071, 2.10101, -581.1883, 0.9647059, 0, 0.03137255, 1,
6.747475, 2.10101, -582.4037, 0.9647059, 0, 0.03137255, 1,
6.787879, 2.10101, -583.6932, 0.8588235, 0, 0.1372549, 1,
6.828283, 2.10101, -585.0566, 0.8588235, 0, 0.1372549, 1,
6.868687, 2.10101, -586.494, 0.8588235, 0, 0.1372549, 1,
6.909091, 2.10101, -588.0053, 0.8588235, 0, 0.1372549, 1,
6.949495, 2.10101, -589.5906, 0.7568628, 0, 0.2392157, 1,
6.989899, 2.10101, -591.2499, 0.7568628, 0, 0.2392157, 1,
7.030303, 2.10101, -592.9831, 0.7568628, 0, 0.2392157, 1,
7.070707, 2.10101, -594.7903, 0.7568628, 0, 0.2392157, 1,
7.111111, 2.10101, -596.6714, 0.654902, 0, 0.3411765, 1,
7.151515, 2.10101, -598.6266, 0.654902, 0, 0.3411765, 1,
7.191919, 2.10101, -600.6556, 0.5490196, 0, 0.4470588, 1,
7.232323, 2.10101, -602.7587, 0.5490196, 0, 0.4470588, 1,
7.272727, 2.10101, -604.9357, 0.5490196, 0, 0.4470588, 1,
7.313131, 2.10101, -607.1867, 0.4470588, 0, 0.5490196, 1,
7.353535, 2.10101, -609.5117, 0.4470588, 0, 0.5490196, 1,
7.393939, 2.10101, -611.9105, 0.4470588, 0, 0.5490196, 1,
7.434343, 2.10101, -614.3834, 0.3411765, 0, 0.654902, 1,
7.474748, 2.10101, -616.9302, 0.3411765, 0, 0.654902, 1,
7.515152, 2.10101, -619.551, 0.2392157, 0, 0.7568628, 1,
7.555555, 2.10101, -622.2458, 0.2392157, 0, 0.7568628, 1,
7.59596, 2.10101, -625.0145, 0.1372549, 0, 0.8588235, 1,
7.636364, 2.10101, -627.8572, 0.1372549, 0, 0.8588235, 1,
7.676768, 2.10101, -630.7739, 0.03137255, 0, 0.9647059, 1,
7.717172, 2.10101, -633.7645, 0.03137255, 0, 0.9647059, 1,
7.757576, 2.10101, -636.8291, 0.04313726, 0.007843138, 0.9921569, 1,
7.79798, 2.10101, -639.9677, 0.04313726, 0.007843138, 0.9921569, 1,
7.838384, 2.10101, -643.1802, 0.1058824, 0.01960784, 0.9882353, 1,
7.878788, 2.10101, -646.4667, 0.1058824, 0.01960784, 0.9882353, 1,
7.919192, 2.10101, -649.8271, 0.172549, 0.03137255, 0.9803922, 1,
7.959596, 2.10101, -653.2615, 0.2352941, 0.04705882, 0.9764706, 1,
8, 2.10101, -656.7699, 0.2352941, 0.04705882, 0.9764706, 1,
4, 2.151515, -657.4053, 0.2352941, 0.04705882, 0.9764706, 1,
4.040404, 2.151515, -653.8387, 0.2352941, 0.04705882, 0.9764706, 1,
4.080808, 2.151515, -650.3426, 0.172549, 0.03137255, 0.9803922, 1,
4.121212, 2.151515, -646.9171, 0.172549, 0.03137255, 0.9803922, 1,
4.161616, 2.151515, -643.562, 0.1058824, 0.01960784, 0.9882353, 1,
4.20202, 2.151515, -640.2775, 0.04313726, 0.007843138, 0.9921569, 1,
4.242424, 2.151515, -637.0636, 0.04313726, 0.007843138, 0.9921569, 1,
4.282828, 2.151515, -633.9202, 0.03137255, 0, 0.9647059, 1,
4.323232, 2.151515, -630.8473, 0.03137255, 0, 0.9647059, 1,
4.363636, 2.151515, -627.8449, 0.1372549, 0, 0.8588235, 1,
4.40404, 2.151515, -624.9131, 0.1372549, 0, 0.8588235, 1,
4.444445, 2.151515, -622.0518, 0.2392157, 0, 0.7568628, 1,
4.484848, 2.151515, -619.261, 0.2392157, 0, 0.7568628, 1,
4.525252, 2.151515, -616.5408, 0.3411765, 0, 0.654902, 1,
4.565657, 2.151515, -613.8912, 0.3411765, 0, 0.654902, 1,
4.606061, 2.151515, -611.312, 0.4470588, 0, 0.5490196, 1,
4.646465, 2.151515, -608.8034, 0.4470588, 0, 0.5490196, 1,
4.686869, 2.151515, -606.3653, 0.5490196, 0, 0.4470588, 1,
4.727273, 2.151515, -603.9977, 0.5490196, 0, 0.4470588, 1,
4.767677, 2.151515, -601.7007, 0.5490196, 0, 0.4470588, 1,
4.808081, 2.151515, -599.4742, 0.654902, 0, 0.3411765, 1,
4.848485, 2.151515, -597.3182, 0.654902, 0, 0.3411765, 1,
4.888889, 2.151515, -595.2328, 0.654902, 0, 0.3411765, 1,
4.929293, 2.151515, -593.218, 0.7568628, 0, 0.2392157, 1,
4.969697, 2.151515, -591.2736, 0.7568628, 0, 0.2392157, 1,
5.010101, 2.151515, -589.3997, 0.7568628, 0, 0.2392157, 1,
5.050505, 2.151515, -587.5964, 0.8588235, 0, 0.1372549, 1,
5.090909, 2.151515, -585.8637, 0.8588235, 0, 0.1372549, 1,
5.131313, 2.151515, -584.2015, 0.8588235, 0, 0.1372549, 1,
5.171717, 2.151515, -582.6097, 0.9647059, 0, 0.03137255, 1,
5.212121, 2.151515, -581.0886, 0.9647059, 0, 0.03137255, 1,
5.252525, 2.151515, -579.6379, 0.9647059, 0, 0.03137255, 1,
5.292929, 2.151515, -578.2579, 0.9647059, 0, 0.03137255, 1,
5.333333, 2.151515, -576.9483, 1, 0.06666667, 0, 1,
5.373737, 2.151515, -575.7093, 1, 0.06666667, 0, 1,
5.414141, 2.151515, -574.5408, 1, 0.06666667, 0, 1,
5.454545, 2.151515, -573.4428, 1, 0.06666667, 0, 1,
5.494949, 2.151515, -572.4153, 1, 0.06666667, 0, 1,
5.535354, 2.151515, -571.4584, 1, 0.1686275, 0, 1,
5.575758, 2.151515, -570.5721, 1, 0.1686275, 0, 1,
5.616162, 2.151515, -569.7562, 1, 0.1686275, 0, 1,
5.656566, 2.151515, -569.0109, 1, 0.1686275, 0, 1,
5.69697, 2.151515, -568.3362, 1, 0.1686275, 0, 1,
5.737374, 2.151515, -567.7319, 1, 0.1686275, 0, 1,
5.777778, 2.151515, -567.1982, 1, 0.1686275, 0, 1,
5.818182, 2.151515, -566.735, 1, 0.1686275, 0, 1,
5.858586, 2.151515, -566.3424, 1, 0.1686275, 0, 1,
5.89899, 2.151515, -566.0203, 1, 0.1686275, 0, 1,
5.939394, 2.151515, -565.7687, 1, 0.2745098, 0, 1,
5.979798, 2.151515, -565.5877, 1, 0.2745098, 0, 1,
6.020202, 2.151515, -565.4772, 1, 0.2745098, 0, 1,
6.060606, 2.151515, -565.4372, 1, 0.2745098, 0, 1,
6.10101, 2.151515, -565.4678, 1, 0.2745098, 0, 1,
6.141414, 2.151515, -565.5688, 1, 0.2745098, 0, 1,
6.181818, 2.151515, -565.7405, 1, 0.2745098, 0, 1,
6.222222, 2.151515, -565.9826, 1, 0.1686275, 0, 1,
6.262626, 2.151515, -566.2953, 1, 0.1686275, 0, 1,
6.30303, 2.151515, -566.6785, 1, 0.1686275, 0, 1,
6.343434, 2.151515, -567.1323, 1, 0.1686275, 0, 1,
6.383838, 2.151515, -567.6566, 1, 0.1686275, 0, 1,
6.424242, 2.151515, -568.2514, 1, 0.1686275, 0, 1,
6.464646, 2.151515, -568.9167, 1, 0.1686275, 0, 1,
6.505051, 2.151515, -569.6526, 1, 0.1686275, 0, 1,
6.545455, 2.151515, -570.459, 1, 0.1686275, 0, 1,
6.585859, 2.151515, -571.336, 1, 0.1686275, 0, 1,
6.626263, 2.151515, -572.2834, 1, 0.06666667, 0, 1,
6.666667, 2.151515, -573.3015, 1, 0.06666667, 0, 1,
6.707071, 2.151515, -574.39, 1, 0.06666667, 0, 1,
6.747475, 2.151515, -575.5491, 1, 0.06666667, 0, 1,
6.787879, 2.151515, -576.7787, 1, 0.06666667, 0, 1,
6.828283, 2.151515, -578.0789, 0.9647059, 0, 0.03137255, 1,
6.868687, 2.151515, -579.4495, 0.9647059, 0, 0.03137255, 1,
6.909091, 2.151515, -580.8907, 0.9647059, 0, 0.03137255, 1,
6.949495, 2.151515, -582.4025, 0.9647059, 0, 0.03137255, 1,
6.989899, 2.151515, -583.9848, 0.8588235, 0, 0.1372549, 1,
7.030303, 2.151515, -585.6376, 0.8588235, 0, 0.1372549, 1,
7.070707, 2.151515, -587.3609, 0.8588235, 0, 0.1372549, 1,
7.111111, 2.151515, -589.1548, 0.7568628, 0, 0.2392157, 1,
7.151515, 2.151515, -591.0192, 0.7568628, 0, 0.2392157, 1,
7.191919, 2.151515, -592.9542, 0.7568628, 0, 0.2392157, 1,
7.232323, 2.151515, -594.9597, 0.654902, 0, 0.3411765, 1,
7.272727, 2.151515, -597.0356, 0.654902, 0, 0.3411765, 1,
7.313131, 2.151515, -599.1822, 0.654902, 0, 0.3411765, 1,
7.353535, 2.151515, -601.3992, 0.5490196, 0, 0.4470588, 1,
7.393939, 2.151515, -603.6868, 0.5490196, 0, 0.4470588, 1,
7.434343, 2.151515, -606.045, 0.5490196, 0, 0.4470588, 1,
7.474748, 2.151515, -608.4736, 0.4470588, 0, 0.5490196, 1,
7.515152, 2.151515, -610.9728, 0.4470588, 0, 0.5490196, 1,
7.555555, 2.151515, -613.5426, 0.3411765, 0, 0.654902, 1,
7.59596, 2.151515, -616.1829, 0.3411765, 0, 0.654902, 1,
7.636364, 2.151515, -618.8937, 0.2392157, 0, 0.7568628, 1,
7.676768, 2.151515, -621.675, 0.2392157, 0, 0.7568628, 1,
7.717172, 2.151515, -624.5269, 0.1372549, 0, 0.8588235, 1,
7.757576, 2.151515, -627.4493, 0.1372549, 0, 0.8588235, 1,
7.79798, 2.151515, -630.4422, 0.03137255, 0, 0.9647059, 1,
7.838384, 2.151515, -633.5057, 0.03137255, 0, 0.9647059, 1,
7.878788, 2.151515, -636.6396, 0.04313726, 0.007843138, 0.9921569, 1,
7.919192, 2.151515, -639.8442, 0.04313726, 0.007843138, 0.9921569, 1,
7.959596, 2.151515, -643.1193, 0.1058824, 0.01960784, 0.9882353, 1,
8, 2.151515, -646.4648, 0.1058824, 0.01960784, 0.9882353, 1,
4, 2.20202, -647.5179, 0.172549, 0.03137255, 0.9803922, 1,
4.040404, 2.20202, -644.113, 0.1058824, 0.01960784, 0.9882353, 1,
4.080808, 2.20202, -640.7755, 0.04313726, 0.007843138, 0.9921569, 1,
4.121212, 2.20202, -637.5052, 0.04313726, 0.007843138, 0.9921569, 1,
4.161616, 2.20202, -634.3024, 0.03137255, 0, 0.9647059, 1,
4.20202, 2.20202, -631.1668, 0.03137255, 0, 0.9647059, 1,
4.242424, 2.20202, -628.0986, 0.1372549, 0, 0.8588235, 1,
4.282828, 2.20202, -625.0978, 0.1372549, 0, 0.8588235, 1,
4.323232, 2.20202, -622.1642, 0.2392157, 0, 0.7568628, 1,
4.363636, 2.20202, -619.298, 0.2392157, 0, 0.7568628, 1,
4.40404, 2.20202, -616.4991, 0.3411765, 0, 0.654902, 1,
4.444445, 2.20202, -613.7676, 0.3411765, 0, 0.654902, 1,
4.484848, 2.20202, -611.1034, 0.4470588, 0, 0.5490196, 1,
4.525252, 2.20202, -608.5065, 0.4470588, 0, 0.5490196, 1,
4.565657, 2.20202, -605.977, 0.5490196, 0, 0.4470588, 1,
4.606061, 2.20202, -603.5148, 0.5490196, 0, 0.4470588, 1,
4.646465, 2.20202, -601.1199, 0.5490196, 0, 0.4470588, 1,
4.686869, 2.20202, -598.7924, 0.654902, 0, 0.3411765, 1,
4.727273, 2.20202, -596.5322, 0.654902, 0, 0.3411765, 1,
4.767677, 2.20202, -594.3393, 0.7568628, 0, 0.2392157, 1,
4.808081, 2.20202, -592.2137, 0.7568628, 0, 0.2392157, 1,
4.848485, 2.20202, -590.1556, 0.7568628, 0, 0.2392157, 1,
4.888889, 2.20202, -588.1647, 0.8588235, 0, 0.1372549, 1,
4.929293, 2.20202, -586.2412, 0.8588235, 0, 0.1372549, 1,
4.969697, 2.20202, -584.385, 0.8588235, 0, 0.1372549, 1,
5.010101, 2.20202, -582.5961, 0.9647059, 0, 0.03137255, 1,
5.050505, 2.20202, -580.8746, 0.9647059, 0, 0.03137255, 1,
5.090909, 2.20202, -579.2204, 0.9647059, 0, 0.03137255, 1,
5.131313, 2.20202, -577.6335, 0.9647059, 0, 0.03137255, 1,
5.171717, 2.20202, -576.1141, 1, 0.06666667, 0, 1,
5.212121, 2.20202, -574.6619, 1, 0.06666667, 0, 1,
5.252525, 2.20202, -573.277, 1, 0.06666667, 0, 1,
5.292929, 2.20202, -571.9595, 1, 0.06666667, 0, 1,
5.333333, 2.20202, -570.7093, 1, 0.1686275, 0, 1,
5.373737, 2.20202, -569.5264, 1, 0.1686275, 0, 1,
5.414141, 2.20202, -568.4109, 1, 0.1686275, 0, 1,
5.454545, 2.20202, -567.3628, 1, 0.1686275, 0, 1,
5.494949, 2.20202, -566.3819, 1, 0.1686275, 0, 1,
5.535354, 2.20202, -565.4684, 1, 0.2745098, 0, 1,
5.575758, 2.20202, -564.6223, 1, 0.2745098, 0, 1,
5.616162, 2.20202, -563.8434, 1, 0.2745098, 0, 1,
5.656566, 2.20202, -563.1319, 1, 0.2745098, 0, 1,
5.69697, 2.20202, -562.4877, 1, 0.2745098, 0, 1,
5.737374, 2.20202, -561.9109, 1, 0.2745098, 0, 1,
5.777778, 2.20202, -561.4014, 1, 0.2745098, 0, 1,
5.818182, 2.20202, -560.9592, 1, 0.2745098, 0, 1,
5.858586, 2.20202, -560.5844, 1, 0.2745098, 0, 1,
5.89899, 2.20202, -560.2769, 1, 0.2745098, 0, 1,
5.939394, 2.20202, -560.0367, 1, 0.3764706, 0, 1,
5.979798, 2.20202, -559.8638, 1, 0.3764706, 0, 1,
6.020202, 2.20202, -559.7584, 1, 0.3764706, 0, 1,
6.060606, 2.20202, -559.7202, 1, 0.3764706, 0, 1,
6.10101, 2.20202, -559.7494, 1, 0.3764706, 0, 1,
6.141414, 2.20202, -559.8459, 1, 0.3764706, 0, 1,
6.181818, 2.20202, -560.0097, 1, 0.3764706, 0, 1,
6.222222, 2.20202, -560.2409, 1, 0.2745098, 0, 1,
6.262626, 2.20202, -560.5394, 1, 0.2745098, 0, 1,
6.30303, 2.20202, -560.9052, 1, 0.2745098, 0, 1,
6.343434, 2.20202, -561.3384, 1, 0.2745098, 0, 1,
6.383838, 2.20202, -561.8389, 1, 0.2745098, 0, 1,
6.424242, 2.20202, -562.4067, 1, 0.2745098, 0, 1,
6.464646, 2.20202, -563.0419, 1, 0.2745098, 0, 1,
6.505051, 2.20202, -563.7444, 1, 0.2745098, 0, 1,
6.545455, 2.20202, -564.5143, 1, 0.2745098, 0, 1,
6.585859, 2.20202, -565.3515, 1, 0.2745098, 0, 1,
6.626263, 2.20202, -566.256, 1, 0.1686275, 0, 1,
6.666667, 2.20202, -567.2278, 1, 0.1686275, 0, 1,
6.707071, 2.20202, -568.267, 1, 0.1686275, 0, 1,
6.747475, 2.20202, -569.3736, 1, 0.1686275, 0, 1,
6.787879, 2.20202, -570.5474, 1, 0.1686275, 0, 1,
6.828283, 2.20202, -571.7886, 1, 0.06666667, 0, 1,
6.868687, 2.20202, -573.0971, 1, 0.06666667, 0, 1,
6.909091, 2.20202, -574.473, 1, 0.06666667, 0, 1,
6.949495, 2.20202, -575.9162, 1, 0.06666667, 0, 1,
6.989899, 2.20202, -577.4267, 1, 0.06666667, 0, 1,
7.030303, 2.20202, -579.0046, 0.9647059, 0, 0.03137255, 1,
7.070707, 2.20202, -580.6498, 0.9647059, 0, 0.03137255, 1,
7.111111, 2.20202, -582.3623, 0.9647059, 0, 0.03137255, 1,
7.151515, 2.20202, -584.1422, 0.8588235, 0, 0.1372549, 1,
7.191919, 2.20202, -585.9894, 0.8588235, 0, 0.1372549, 1,
7.232323, 2.20202, -587.9039, 0.8588235, 0, 0.1372549, 1,
7.272727, 2.20202, -589.8857, 0.7568628, 0, 0.2392157, 1,
7.313131, 2.20202, -591.935, 0.7568628, 0, 0.2392157, 1,
7.353535, 2.20202, -594.0515, 0.7568628, 0, 0.2392157, 1,
7.393939, 2.20202, -596.2354, 0.654902, 0, 0.3411765, 1,
7.434343, 2.20202, -598.4866, 0.654902, 0, 0.3411765, 1,
7.474748, 2.20202, -600.8051, 0.5490196, 0, 0.4470588, 1,
7.515152, 2.20202, -603.191, 0.5490196, 0, 0.4470588, 1,
7.555555, 2.20202, -605.6442, 0.5490196, 0, 0.4470588, 1,
7.59596, 2.20202, -608.1647, 0.4470588, 0, 0.5490196, 1,
7.636364, 2.20202, -610.7526, 0.4470588, 0, 0.5490196, 1,
7.676768, 2.20202, -613.4078, 0.3411765, 0, 0.654902, 1,
7.717172, 2.20202, -616.1304, 0.3411765, 0, 0.654902, 1,
7.757576, 2.20202, -618.9203, 0.2392157, 0, 0.7568628, 1,
7.79798, 2.20202, -621.7775, 0.2392157, 0, 0.7568628, 1,
7.838384, 2.20202, -624.702, 0.1372549, 0, 0.8588235, 1,
7.878788, 2.20202, -627.6939, 0.1372549, 0, 0.8588235, 1,
7.919192, 2.20202, -630.7531, 0.03137255, 0, 0.9647059, 1,
7.959596, 2.20202, -633.8797, 0.03137255, 0, 0.9647059, 1,
8, 2.20202, -637.0735, 0.04313726, 0.007843138, 0.9921569, 1,
4, 2.252525, -638.4915, 0.04313726, 0.007843138, 0.9921569, 1,
4.040404, 2.252525, -635.2376, 0.03137255, 0, 0.9647059, 1,
4.080808, 2.252525, -632.0481, 0.03137255, 0, 0.9647059, 1,
4.121212, 2.252525, -628.9229, 0.1372549, 0, 0.8588235, 1,
4.161616, 2.252525, -625.862, 0.1372549, 0, 0.8588235, 1,
4.20202, 2.252525, -622.8655, 0.2392157, 0, 0.7568628, 1,
4.242424, 2.252525, -619.9333, 0.2392157, 0, 0.7568628, 1,
4.282828, 2.252525, -617.0655, 0.3411765, 0, 0.654902, 1,
4.323232, 2.252525, -614.262, 0.3411765, 0, 0.654902, 1,
4.363636, 2.252525, -611.5229, 0.4470588, 0, 0.5490196, 1,
4.40404, 2.252525, -608.8481, 0.4470588, 0, 0.5490196, 1,
4.444445, 2.252525, -606.2377, 0.5490196, 0, 0.4470588, 1,
4.484848, 2.252525, -603.6917, 0.5490196, 0, 0.4470588, 1,
4.525252, 2.252525, -601.2099, 0.5490196, 0, 0.4470588, 1,
4.565657, 2.252525, -598.7925, 0.654902, 0, 0.3411765, 1,
4.606061, 2.252525, -596.4395, 0.654902, 0, 0.3411765, 1,
4.646465, 2.252525, -594.1508, 0.7568628, 0, 0.2392157, 1,
4.686869, 2.252525, -591.9265, 0.7568628, 0, 0.2392157, 1,
4.727273, 2.252525, -589.7665, 0.7568628, 0, 0.2392157, 1,
4.767677, 2.252525, -587.6709, 0.8588235, 0, 0.1372549, 1,
4.808081, 2.252525, -585.6396, 0.8588235, 0, 0.1372549, 1,
4.848485, 2.252525, -583.6727, 0.8588235, 0, 0.1372549, 1,
4.888889, 2.252525, -581.7701, 0.9647059, 0, 0.03137255, 1,
4.929293, 2.252525, -579.9318, 0.9647059, 0, 0.03137255, 1,
4.969697, 2.252525, -578.158, 0.9647059, 0, 0.03137255, 1,
5.010101, 2.252525, -576.4484, 1, 0.06666667, 0, 1,
5.050505, 2.252525, -574.8032, 1, 0.06666667, 0, 1,
5.090909, 2.252525, -573.2224, 1, 0.06666667, 0, 1,
5.131313, 2.252525, -571.7059, 1, 0.06666667, 0, 1,
5.171717, 2.252525, -570.2537, 1, 0.1686275, 0, 1,
5.212121, 2.252525, -568.866, 1, 0.1686275, 0, 1,
5.252525, 2.252525, -567.5425, 1, 0.1686275, 0, 1,
5.292929, 2.252525, -566.2834, 1, 0.1686275, 0, 1,
5.333333, 2.252525, -565.0886, 1, 0.2745098, 0, 1,
5.373737, 2.252525, -563.9583, 1, 0.2745098, 0, 1,
5.414141, 2.252525, -562.8922, 1, 0.2745098, 0, 1,
5.454545, 2.252525, -561.8905, 1, 0.2745098, 0, 1,
5.494949, 2.252525, -560.9531, 1, 0.2745098, 0, 1,
5.535354, 2.252525, -560.0801, 1, 0.3764706, 0, 1,
5.575758, 2.252525, -559.2715, 1, 0.3764706, 0, 1,
5.616162, 2.252525, -558.5272, 1, 0.3764706, 0, 1,
5.656566, 2.252525, -557.8472, 1, 0.3764706, 0, 1,
5.69697, 2.252525, -557.2316, 1, 0.3764706, 0, 1,
5.737374, 2.252525, -556.6804, 1, 0.3764706, 0, 1,
5.777778, 2.252525, -556.1934, 1, 0.3764706, 0, 1,
5.818182, 2.252525, -555.7709, 1, 0.3764706, 0, 1,
5.858586, 2.252525, -555.4127, 1, 0.3764706, 0, 1,
5.89899, 2.252525, -555.1188, 1, 0.3764706, 0, 1,
5.939394, 2.252525, -554.8893, 1, 0.3764706, 0, 1,
5.979798, 2.252525, -554.7241, 1, 0.3764706, 0, 1,
6.020202, 2.252525, -554.6233, 1, 0.3764706, 0, 1,
6.060606, 2.252525, -554.5868, 1, 0.3764706, 0, 1,
6.10101, 2.252525, -554.6147, 1, 0.3764706, 0, 1,
6.141414, 2.252525, -554.7069, 1, 0.3764706, 0, 1,
6.181818, 2.252525, -554.8635, 1, 0.3764706, 0, 1,
6.222222, 2.252525, -555.0844, 1, 0.3764706, 0, 1,
6.262626, 2.252525, -555.3697, 1, 0.3764706, 0, 1,
6.30303, 2.252525, -555.7193, 1, 0.3764706, 0, 1,
6.343434, 2.252525, -556.1332, 1, 0.3764706, 0, 1,
6.383838, 2.252525, -556.6116, 1, 0.3764706, 0, 1,
6.424242, 2.252525, -557.1542, 1, 0.3764706, 0, 1,
6.464646, 2.252525, -557.7612, 1, 0.3764706, 0, 1,
6.505051, 2.252525, -558.4326, 1, 0.3764706, 0, 1,
6.545455, 2.252525, -559.1683, 1, 0.3764706, 0, 1,
6.585859, 2.252525, -559.9684, 1, 0.3764706, 0, 1,
6.626263, 2.252525, -560.8328, 1, 0.2745098, 0, 1,
6.666667, 2.252525, -561.7616, 1, 0.2745098, 0, 1,
6.707071, 2.252525, -562.7547, 1, 0.2745098, 0, 1,
6.747475, 2.252525, -563.8121, 1, 0.2745098, 0, 1,
6.787879, 2.252525, -564.934, 1, 0.2745098, 0, 1,
6.828283, 2.252525, -566.1201, 1, 0.1686275, 0, 1,
6.868687, 2.252525, -567.3706, 1, 0.1686275, 0, 1,
6.909091, 2.252525, -568.6854, 1, 0.1686275, 0, 1,
6.949495, 2.252525, -570.0646, 1, 0.1686275, 0, 1,
6.989899, 2.252525, -571.5082, 1, 0.1686275, 0, 1,
7.030303, 2.252525, -573.0161, 1, 0.06666667, 0, 1,
7.070707, 2.252525, -574.5884, 1, 0.06666667, 0, 1,
7.111111, 2.252525, -576.225, 1, 0.06666667, 0, 1,
7.151515, 2.252525, -577.9259, 0.9647059, 0, 0.03137255, 1,
7.191919, 2.252525, -579.6912, 0.9647059, 0, 0.03137255, 1,
7.232323, 2.252525, -581.5208, 0.9647059, 0, 0.03137255, 1,
7.272727, 2.252525, -583.4148, 0.8588235, 0, 0.1372549, 1,
7.313131, 2.252525, -585.3732, 0.8588235, 0, 0.1372549, 1,
7.353535, 2.252525, -587.3959, 0.8588235, 0, 0.1372549, 1,
7.393939, 2.252525, -589.4829, 0.7568628, 0, 0.2392157, 1,
7.434343, 2.252525, -591.6343, 0.7568628, 0, 0.2392157, 1,
7.474748, 2.252525, -593.85, 0.7568628, 0, 0.2392157, 1,
7.515152, 2.252525, -596.1301, 0.654902, 0, 0.3411765, 1,
7.555555, 2.252525, -598.4745, 0.654902, 0, 0.3411765, 1,
7.59596, 2.252525, -600.8833, 0.5490196, 0, 0.4470588, 1,
7.636364, 2.252525, -603.3564, 0.5490196, 0, 0.4470588, 1,
7.676768, 2.252525, -605.8939, 0.5490196, 0, 0.4470588, 1,
7.717172, 2.252525, -608.4957, 0.4470588, 0, 0.5490196, 1,
7.757576, 2.252525, -611.1619, 0.4470588, 0, 0.5490196, 1,
7.79798, 2.252525, -613.8925, 0.3411765, 0, 0.654902, 1,
7.838384, 2.252525, -616.6873, 0.3411765, 0, 0.654902, 1,
7.878788, 2.252525, -619.5466, 0.2392157, 0, 0.7568628, 1,
7.919192, 2.252525, -622.4701, 0.2392157, 0, 0.7568628, 1,
7.959596, 2.252525, -625.458, 0.1372549, 0, 0.8588235, 1,
8, 2.252525, -628.5103, 0.1372549, 0, 0.8588235, 1,
4, 2.30303, -630.247, 0.03137255, 0, 0.9647059, 1,
4.040404, 2.30303, -627.1342, 0.1372549, 0, 0.8588235, 1,
4.080808, 2.30303, -624.083, 0.1372549, 0, 0.8588235, 1,
4.121212, 2.30303, -621.0934, 0.2392157, 0, 0.7568628, 1,
4.161616, 2.30303, -618.1653, 0.2392157, 0, 0.7568628, 1,
4.20202, 2.30303, -615.2988, 0.3411765, 0, 0.654902, 1,
4.242424, 2.30303, -612.4938, 0.3411765, 0, 0.654902, 1,
4.282828, 2.30303, -609.7504, 0.4470588, 0, 0.5490196, 1,
4.323232, 2.30303, -607.0685, 0.4470588, 0, 0.5490196, 1,
4.363636, 2.30303, -604.4482, 0.5490196, 0, 0.4470588, 1,
4.40404, 2.30303, -601.8895, 0.5490196, 0, 0.4470588, 1,
4.444445, 2.30303, -599.3923, 0.654902, 0, 0.3411765, 1,
4.484848, 2.30303, -596.9567, 0.654902, 0, 0.3411765, 1,
4.525252, 2.30303, -594.5826, 0.7568628, 0, 0.2392157, 1,
4.565657, 2.30303, -592.2701, 0.7568628, 0, 0.2392157, 1,
4.606061, 2.30303, -590.0191, 0.7568628, 0, 0.2392157, 1,
4.646465, 2.30303, -587.8297, 0.8588235, 0, 0.1372549, 1,
4.686869, 2.30303, -585.7019, 0.8588235, 0, 0.1372549, 1,
4.727273, 2.30303, -583.6356, 0.8588235, 0, 0.1372549, 1,
4.767677, 2.30303, -581.6309, 0.9647059, 0, 0.03137255, 1,
4.808081, 2.30303, -579.6877, 0.9647059, 0, 0.03137255, 1,
4.848485, 2.30303, -577.8061, 0.9647059, 0, 0.03137255, 1,
4.888889, 2.30303, -575.986, 1, 0.06666667, 0, 1,
4.929293, 2.30303, -574.2275, 1, 0.06666667, 0, 1,
4.969697, 2.30303, -572.5306, 1, 0.06666667, 0, 1,
5.010101, 2.30303, -570.8952, 1, 0.1686275, 0, 1,
5.050505, 2.30303, -569.3214, 1, 0.1686275, 0, 1,
5.090909, 2.30303, -567.8091, 1, 0.1686275, 0, 1,
5.131313, 2.30303, -566.3584, 1, 0.1686275, 0, 1,
5.171717, 2.30303, -564.9692, 1, 0.2745098, 0, 1,
5.212121, 2.30303, -563.6417, 1, 0.2745098, 0, 1,
5.252525, 2.30303, -562.3756, 1, 0.2745098, 0, 1,
5.292929, 2.30303, -561.1711, 1, 0.2745098, 0, 1,
5.333333, 2.30303, -560.0282, 1, 0.3764706, 0, 1,
5.373737, 2.30303, -558.9468, 1, 0.3764706, 0, 1,
5.414141, 2.30303, -557.9271, 1, 0.3764706, 0, 1,
5.454545, 2.30303, -556.9688, 1, 0.3764706, 0, 1,
5.494949, 2.30303, -556.0721, 1, 0.3764706, 0, 1,
5.535354, 2.30303, -555.237, 1, 0.3764706, 0, 1,
5.575758, 2.30303, -554.4634, 1, 0.3764706, 0, 1,
5.616162, 2.30303, -553.7513, 1, 0.4823529, 0, 1,
5.656566, 2.30303, -553.1009, 1, 0.4823529, 0, 1,
5.69697, 2.30303, -552.512, 1, 0.4823529, 0, 1,
5.737374, 2.30303, -551.9847, 1, 0.4823529, 0, 1,
5.777778, 2.30303, -551.5189, 1, 0.4823529, 0, 1,
5.818182, 2.30303, -551.1146, 1, 0.4823529, 0, 1,
5.858586, 2.30303, -550.772, 1, 0.4823529, 0, 1,
5.89899, 2.30303, -550.4908, 1, 0.4823529, 0, 1,
5.939394, 2.30303, -550.2712, 1, 0.4823529, 0, 1,
5.979798, 2.30303, -550.1133, 1, 0.4823529, 0, 1,
6.020202, 2.30303, -550.0168, 1, 0.4823529, 0, 1,
6.060606, 2.30303, -549.9819, 1, 0.4823529, 0, 1,
6.10101, 2.30303, -550.0086, 1, 0.4823529, 0, 1,
6.141414, 2.30303, -550.0968, 1, 0.4823529, 0, 1,
6.181818, 2.30303, -550.2466, 1, 0.4823529, 0, 1,
6.222222, 2.30303, -550.4579, 1, 0.4823529, 0, 1,
6.262626, 2.30303, -550.7308, 1, 0.4823529, 0, 1,
6.30303, 2.30303, -551.0653, 1, 0.4823529, 0, 1,
6.343434, 2.30303, -551.4613, 1, 0.4823529, 0, 1,
6.383838, 2.30303, -551.9189, 1, 0.4823529, 0, 1,
6.424242, 2.30303, -552.438, 1, 0.4823529, 0, 1,
6.464646, 2.30303, -553.0187, 1, 0.4823529, 0, 1,
6.505051, 2.30303, -553.6609, 1, 0.4823529, 0, 1,
6.545455, 2.30303, -554.3647, 1, 0.3764706, 0, 1,
6.585859, 2.30303, -555.1301, 1, 0.3764706, 0, 1,
6.626263, 2.30303, -555.957, 1, 0.3764706, 0, 1,
6.666667, 2.30303, -556.8455, 1, 0.3764706, 0, 1,
6.707071, 2.30303, -557.7955, 1, 0.3764706, 0, 1,
6.747475, 2.30303, -558.8071, 1, 0.3764706, 0, 1,
6.787879, 2.30303, -559.8802, 1, 0.3764706, 0, 1,
6.828283, 2.30303, -561.015, 1, 0.2745098, 0, 1,
6.868687, 2.30303, -562.2112, 1, 0.2745098, 0, 1,
6.909091, 2.30303, -563.469, 1, 0.2745098, 0, 1,
6.949495, 2.30303, -564.7884, 1, 0.2745098, 0, 1,
6.989899, 2.30303, -566.1693, 1, 0.1686275, 0, 1,
7.030303, 2.30303, -567.6118, 1, 0.1686275, 0, 1,
7.070707, 2.30303, -569.1158, 1, 0.1686275, 0, 1,
7.111111, 2.30303, -570.6815, 1, 0.1686275, 0, 1,
7.151515, 2.30303, -572.3086, 1, 0.06666667, 0, 1,
7.191919, 2.30303, -573.9973, 1, 0.06666667, 0, 1,
7.232323, 2.30303, -575.7476, 1, 0.06666667, 0, 1,
7.272727, 2.30303, -577.5594, 0.9647059, 0, 0.03137255, 1,
7.313131, 2.30303, -579.4328, 0.9647059, 0, 0.03137255, 1,
7.353535, 2.30303, -581.3678, 0.9647059, 0, 0.03137255, 1,
7.393939, 2.30303, -583.3643, 0.8588235, 0, 0.1372549, 1,
7.434343, 2.30303, -585.4224, 0.8588235, 0, 0.1372549, 1,
7.474748, 2.30303, -587.5419, 0.8588235, 0, 0.1372549, 1,
7.515152, 2.30303, -589.7231, 0.7568628, 0, 0.2392157, 1,
7.555555, 2.30303, -591.9659, 0.7568628, 0, 0.2392157, 1,
7.59596, 2.30303, -594.2701, 0.7568628, 0, 0.2392157, 1,
7.636364, 2.30303, -596.636, 0.654902, 0, 0.3411765, 1,
7.676768, 2.30303, -599.0634, 0.654902, 0, 0.3411765, 1,
7.717172, 2.30303, -601.5524, 0.5490196, 0, 0.4470588, 1,
7.757576, 2.30303, -604.1029, 0.5490196, 0, 0.4470588, 1,
7.79798, 2.30303, -606.715, 0.4470588, 0, 0.5490196, 1,
7.838384, 2.30303, -609.3886, 0.4470588, 0, 0.5490196, 1,
7.878788, 2.30303, -612.1238, 0.4470588, 0, 0.5490196, 1,
7.919192, 2.30303, -614.9205, 0.3411765, 0, 0.654902, 1,
7.959596, 2.30303, -617.7789, 0.3411765, 0, 0.654902, 1,
8, 2.30303, -620.6987, 0.2392157, 0, 0.7568628, 1,
4, 2.353535, -622.7137, 0.2392157, 0, 0.7568628, 1,
4.040404, 2.353535, -619.7331, 0.2392157, 0, 0.7568628, 1,
4.080808, 2.353535, -616.8115, 0.3411765, 0, 0.654902, 1,
4.121212, 2.353535, -613.9487, 0.3411765, 0, 0.654902, 1,
4.161616, 2.353535, -611.145, 0.4470588, 0, 0.5490196, 1,
4.20202, 2.353535, -608.4001, 0.4470588, 0, 0.5490196, 1,
4.242424, 2.353535, -605.7143, 0.5490196, 0, 0.4470588, 1,
4.282828, 2.353535, -603.0873, 0.5490196, 0, 0.4470588, 1,
4.323232, 2.353535, -600.5193, 0.654902, 0, 0.3411765, 1,
4.363636, 2.353535, -598.0103, 0.654902, 0, 0.3411765, 1,
4.40404, 2.353535, -595.5602, 0.654902, 0, 0.3411765, 1,
4.444445, 2.353535, -593.1691, 0.7568628, 0, 0.2392157, 1,
4.484848, 2.353535, -590.8368, 0.7568628, 0, 0.2392157, 1,
4.525252, 2.353535, -588.5635, 0.8588235, 0, 0.1372549, 1,
4.565657, 2.353535, -586.3492, 0.8588235, 0, 0.1372549, 1,
4.606061, 2.353535, -584.1938, 0.8588235, 0, 0.1372549, 1,
4.646465, 2.353535, -582.0974, 0.9647059, 0, 0.03137255, 1,
4.686869, 2.353535, -580.0599, 0.9647059, 0, 0.03137255, 1,
4.727273, 2.353535, -578.0814, 0.9647059, 0, 0.03137255, 1,
4.767677, 2.353535, -576.1617, 1, 0.06666667, 0, 1,
4.808081, 2.353535, -574.3011, 1, 0.06666667, 0, 1,
4.848485, 2.353535, -572.4993, 1, 0.06666667, 0, 1,
4.888889, 2.353535, -570.7566, 1, 0.1686275, 0, 1,
4.929293, 2.353535, -569.0728, 1, 0.1686275, 0, 1,
4.969697, 2.353535, -567.4478, 1, 0.1686275, 0, 1,
5.010101, 2.353535, -565.8819, 1, 0.2745098, 0, 1,
5.050505, 2.353535, -564.3749, 1, 0.2745098, 0, 1,
5.090909, 2.353535, -562.9268, 1, 0.2745098, 0, 1,
5.131313, 2.353535, -561.5377, 1, 0.2745098, 0, 1,
5.171717, 2.353535, -560.2075, 1, 0.2745098, 0, 1,
5.212121, 2.353535, -558.9363, 1, 0.3764706, 0, 1,
5.252525, 2.353535, -557.724, 1, 0.3764706, 0, 1,
5.292929, 2.353535, -556.5707, 1, 0.3764706, 0, 1,
5.333333, 2.353535, -555.4763, 1, 0.3764706, 0, 1,
5.373737, 2.353535, -554.4409, 1, 0.3764706, 0, 1,
5.414141, 2.353535, -553.4644, 1, 0.4823529, 0, 1,
5.454545, 2.353535, -552.5468, 1, 0.4823529, 0, 1,
5.494949, 2.353535, -551.6882, 1, 0.4823529, 0, 1,
5.535354, 2.353535, -550.8885, 1, 0.4823529, 0, 1,
5.575758, 2.353535, -550.1477, 1, 0.4823529, 0, 1,
5.616162, 2.353535, -549.4659, 1, 0.4823529, 0, 1,
5.656566, 2.353535, -548.8431, 1, 0.4823529, 0, 1,
5.69697, 2.353535, -548.2792, 1, 0.5843138, 0, 1,
5.737374, 2.353535, -547.7742, 1, 0.5843138, 0, 1,
5.777778, 2.353535, -547.3282, 1, 0.5843138, 0, 1,
5.818182, 2.353535, -546.9412, 1, 0.5843138, 0, 1,
5.858586, 2.353535, -546.613, 1, 0.5843138, 0, 1,
5.89899, 2.353535, -546.3438, 1, 0.5843138, 0, 1,
5.939394, 2.353535, -546.1336, 1, 0.5843138, 0, 1,
5.979798, 2.353535, -545.9823, 1, 0.5843138, 0, 1,
6.020202, 2.353535, -545.89, 1, 0.5843138, 0, 1,
6.060606, 2.353535, -545.8566, 1, 0.5843138, 0, 1,
6.10101, 2.353535, -545.8821, 1, 0.5843138, 0, 1,
6.141414, 2.353535, -545.9666, 1, 0.5843138, 0, 1,
6.181818, 2.353535, -546.11, 1, 0.5843138, 0, 1,
6.222222, 2.353535, -546.3123, 1, 0.5843138, 0, 1,
6.262626, 2.353535, -546.5737, 1, 0.5843138, 0, 1,
6.30303, 2.353535, -546.8939, 1, 0.5843138, 0, 1,
6.343434, 2.353535, -547.2731, 1, 0.5843138, 0, 1,
6.383838, 2.353535, -547.7112, 1, 0.5843138, 0, 1,
6.424242, 2.353535, -548.2083, 1, 0.5843138, 0, 1,
6.464646, 2.353535, -548.7643, 1, 0.4823529, 0, 1,
6.505051, 2.353535, -549.3793, 1, 0.4823529, 0, 1,
6.545455, 2.353535, -550.0533, 1, 0.4823529, 0, 1,
6.585859, 2.353535, -550.7861, 1, 0.4823529, 0, 1,
6.626263, 2.353535, -551.5779, 1, 0.4823529, 0, 1,
6.666667, 2.353535, -552.4286, 1, 0.4823529, 0, 1,
6.707071, 2.353535, -553.3384, 1, 0.4823529, 0, 1,
6.747475, 2.353535, -554.307, 1, 0.4823529, 0, 1,
6.787879, 2.353535, -555.3346, 1, 0.3764706, 0, 1,
6.828283, 2.353535, -556.4211, 1, 0.3764706, 0, 1,
6.868687, 2.353535, -557.5666, 1, 0.3764706, 0, 1,
6.909091, 2.353535, -558.771, 1, 0.3764706, 0, 1,
6.949495, 2.353535, -560.0344, 1, 0.3764706, 0, 1,
6.989899, 2.353535, -561.3566, 1, 0.2745098, 0, 1,
7.030303, 2.353535, -562.7379, 1, 0.2745098, 0, 1,
7.070707, 2.353535, -564.178, 1, 0.2745098, 0, 1,
7.111111, 2.353535, -565.6772, 1, 0.2745098, 0, 1,
7.151515, 2.353535, -567.2353, 1, 0.1686275, 0, 1,
7.191919, 2.353535, -568.8523, 1, 0.1686275, 0, 1,
7.232323, 2.353535, -570.5283, 1, 0.1686275, 0, 1,
7.272727, 2.353535, -572.2632, 1, 0.06666667, 0, 1,
7.313131, 2.353535, -574.057, 1, 0.06666667, 0, 1,
7.353535, 2.353535, -575.9098, 1, 0.06666667, 0, 1,
7.393939, 2.353535, -577.8215, 0.9647059, 0, 0.03137255, 1,
7.434343, 2.353535, -579.7922, 0.9647059, 0, 0.03137255, 1,
7.474748, 2.353535, -581.8218, 0.9647059, 0, 0.03137255, 1,
7.515152, 2.353535, -583.9104, 0.8588235, 0, 0.1372549, 1,
7.555555, 2.353535, -586.0579, 0.8588235, 0, 0.1372549, 1,
7.59596, 2.353535, -588.2644, 0.8588235, 0, 0.1372549, 1,
7.636364, 2.353535, -590.5298, 0.7568628, 0, 0.2392157, 1,
7.676768, 2.353535, -592.8541, 0.7568628, 0, 0.2392157, 1,
7.717172, 2.353535, -595.2374, 0.654902, 0, 0.3411765, 1,
7.757576, 2.353535, -597.6796, 0.654902, 0, 0.3411765, 1,
7.79798, 2.353535, -600.1808, 0.654902, 0, 0.3411765, 1,
7.838384, 2.353535, -602.7409, 0.5490196, 0, 0.4470588, 1,
7.878788, 2.353535, -605.36, 0.5490196, 0, 0.4470588, 1,
7.919192, 2.353535, -608.038, 0.4470588, 0, 0.5490196, 1,
7.959596, 2.353535, -610.775, 0.4470588, 0, 0.5490196, 1,
8, 2.353535, -613.5709, 0.3411765, 0, 0.654902, 1,
4, 2.40404, -615.8287, 0.3411765, 0, 0.654902, 1,
4.040404, 2.40404, -612.972, 0.3411765, 0, 0.654902, 1,
4.080808, 2.40404, -610.1718, 0.4470588, 0, 0.5490196, 1,
4.121212, 2.40404, -607.4281, 0.4470588, 0, 0.5490196, 1,
4.161616, 2.40404, -604.7409, 0.5490196, 0, 0.4470588, 1,
4.20202, 2.40404, -602.1102, 0.5490196, 0, 0.4470588, 1,
4.242424, 2.40404, -599.5359, 0.654902, 0, 0.3411765, 1,
4.282828, 2.40404, -597.0182, 0.654902, 0, 0.3411765, 1,
4.323232, 2.40404, -594.557, 0.7568628, 0, 0.2392157, 1,
4.363636, 2.40404, -592.1523, 0.7568628, 0, 0.2392157, 1,
4.40404, 2.40404, -589.8041, 0.7568628, 0, 0.2392157, 1,
4.444445, 2.40404, -587.5123, 0.8588235, 0, 0.1372549, 1,
4.484848, 2.40404, -585.277, 0.8588235, 0, 0.1372549, 1,
4.525252, 2.40404, -583.0983, 0.9647059, 0, 0.03137255, 1,
4.565657, 2.40404, -580.976, 0.9647059, 0, 0.03137255, 1,
4.606061, 2.40404, -578.9103, 0.9647059, 0, 0.03137255, 1,
4.646465, 2.40404, -576.901, 1, 0.06666667, 0, 1,
4.686869, 2.40404, -574.9482, 1, 0.06666667, 0, 1,
4.727273, 2.40404, -573.0519, 1, 0.06666667, 0, 1,
4.767677, 2.40404, -571.2121, 1, 0.1686275, 0, 1,
4.808081, 2.40404, -569.4288, 1, 0.1686275, 0, 1,
4.848485, 2.40404, -567.702, 1, 0.1686275, 0, 1,
4.888889, 2.40404, -566.0316, 1, 0.1686275, 0, 1,
4.929293, 2.40404, -564.4178, 1, 0.2745098, 0, 1,
4.969697, 2.40404, -562.8605, 1, 0.2745098, 0, 1,
5.010101, 2.40404, -561.3596, 1, 0.2745098, 0, 1,
5.050505, 2.40404, -559.9153, 1, 0.3764706, 0, 1,
5.090909, 2.40404, -558.5274, 1, 0.3764706, 0, 1,
5.131313, 2.40404, -557.196, 1, 0.3764706, 0, 1,
5.171717, 2.40404, -555.9212, 1, 0.3764706, 0, 1,
5.212121, 2.40404, -554.7028, 1, 0.3764706, 0, 1,
5.252525, 2.40404, -553.541, 1, 0.4823529, 0, 1,
5.292929, 2.40404, -552.4355, 1, 0.4823529, 0, 1,
5.333333, 2.40404, -551.3867, 1, 0.4823529, 0, 1,
5.373737, 2.40404, -550.3942, 1, 0.4823529, 0, 1,
5.414141, 2.40404, -549.4583, 1, 0.4823529, 0, 1,
5.454545, 2.40404, -548.5789, 1, 0.4823529, 0, 1,
5.494949, 2.40404, -547.756, 1, 0.5843138, 0, 1,
5.535354, 2.40404, -546.9896, 1, 0.5843138, 0, 1,
5.575758, 2.40404, -546.2796, 1, 0.5843138, 0, 1,
5.616162, 2.40404, -545.6262, 1, 0.5843138, 0, 1,
5.656566, 2.40404, -545.0292, 1, 0.5843138, 0, 1,
5.69697, 2.40404, -544.4888, 1, 0.5843138, 0, 1,
5.737374, 2.40404, -544.0048, 1, 0.5843138, 0, 1,
5.777778, 2.40404, -543.5773, 1, 0.5843138, 0, 1,
5.818182, 2.40404, -543.2064, 1, 0.5843138, 0, 1,
5.858586, 2.40404, -542.8919, 1, 0.5843138, 0, 1,
5.89899, 2.40404, -542.6339, 1, 0.6862745, 0, 1,
5.939394, 2.40404, -542.4324, 1, 0.6862745, 0, 1,
5.979798, 2.40404, -542.2874, 1, 0.6862745, 0, 1,
6.020202, 2.40404, -542.1989, 1, 0.6862745, 0, 1,
6.060606, 2.40404, -542.1669, 1, 0.6862745, 0, 1,
6.10101, 2.40404, -542.1913, 1, 0.6862745, 0, 1,
6.141414, 2.40404, -542.2723, 1, 0.6862745, 0, 1,
6.181818, 2.40404, -542.4097, 1, 0.6862745, 0, 1,
6.222222, 2.40404, -542.6037, 1, 0.6862745, 0, 1,
6.262626, 2.40404, -542.8541, 1, 0.5843138, 0, 1,
6.30303, 2.40404, -543.1611, 1, 0.5843138, 0, 1,
6.343434, 2.40404, -543.5245, 1, 0.5843138, 0, 1,
6.383838, 2.40404, -543.9445, 1, 0.5843138, 0, 1,
6.424242, 2.40404, -544.4208, 1, 0.5843138, 0, 1,
6.464646, 2.40404, -544.9538, 1, 0.5843138, 0, 1,
6.505051, 2.40404, -545.5432, 1, 0.5843138, 0, 1,
6.545455, 2.40404, -546.1891, 1, 0.5843138, 0, 1,
6.585859, 2.40404, -546.8915, 1, 0.5843138, 0, 1,
6.626263, 2.40404, -547.6504, 1, 0.5843138, 0, 1,
6.666667, 2.40404, -548.4658, 1, 0.5843138, 0, 1,
6.707071, 2.40404, -549.3376, 1, 0.4823529, 0, 1,
6.747475, 2.40404, -550.266, 1, 0.4823529, 0, 1,
6.787879, 2.40404, -551.2509, 1, 0.4823529, 0, 1,
6.828283, 2.40404, -552.2922, 1, 0.4823529, 0, 1,
6.868687, 2.40404, -553.39, 1, 0.4823529, 0, 1,
6.909091, 2.40404, -554.5444, 1, 0.3764706, 0, 1,
6.949495, 2.40404, -555.7552, 1, 0.3764706, 0, 1,
6.989899, 2.40404, -557.0225, 1, 0.3764706, 0, 1,
7.030303, 2.40404, -558.3463, 1, 0.3764706, 0, 1,
7.070707, 2.40404, -559.7266, 1, 0.3764706, 0, 1,
7.111111, 2.40404, -561.1635, 1, 0.2745098, 0, 1,
7.151515, 2.40404, -562.6567, 1, 0.2745098, 0, 1,
7.191919, 2.40404, -564.2065, 1, 0.2745098, 0, 1,
7.232323, 2.40404, -565.8128, 1, 0.2745098, 0, 1,
7.272727, 2.40404, -567.4756, 1, 0.1686275, 0, 1,
7.313131, 2.40404, -569.1949, 1, 0.1686275, 0, 1,
7.353535, 2.40404, -570.9706, 1, 0.1686275, 0, 1,
7.393939, 2.40404, -572.8029, 1, 0.06666667, 0, 1,
7.434343, 2.40404, -574.6917, 1, 0.06666667, 0, 1,
7.474748, 2.40404, -576.6369, 1, 0.06666667, 0, 1,
7.515152, 2.40404, -578.6386, 0.9647059, 0, 0.03137255, 1,
7.555555, 2.40404, -580.6968, 0.9647059, 0, 0.03137255, 1,
7.59596, 2.40404, -582.8116, 0.9647059, 0, 0.03137255, 1,
7.636364, 2.40404, -584.9828, 0.8588235, 0, 0.1372549, 1,
7.676768, 2.40404, -587.2105, 0.8588235, 0, 0.1372549, 1,
7.717172, 2.40404, -589.4947, 0.7568628, 0, 0.2392157, 1,
7.757576, 2.40404, -591.8354, 0.7568628, 0, 0.2392157, 1,
7.79798, 2.40404, -594.2325, 0.7568628, 0, 0.2392157, 1,
7.838384, 2.40404, -596.6863, 0.654902, 0, 0.3411765, 1,
7.878788, 2.40404, -599.1964, 0.654902, 0, 0.3411765, 1,
7.919192, 2.40404, -601.7631, 0.5490196, 0, 0.4470588, 1,
7.959596, 2.40404, -604.3862, 0.5490196, 0, 0.4470588, 1,
8, 2.40404, -607.0659, 0.4470588, 0, 0.5490196, 1,
4, 2.454545, -609.5353, 0.4470588, 0, 0.5490196, 1,
4.040404, 2.454545, -606.795, 0.4470588, 0, 0.5490196, 1,
4.080808, 2.454545, -604.1089, 0.5490196, 0, 0.4470588, 1,
4.121212, 2.454545, -601.4769, 0.5490196, 0, 0.4470588, 1,
4.161616, 2.454545, -598.8992, 0.654902, 0, 0.3411765, 1,
4.20202, 2.454545, -596.3756, 0.654902, 0, 0.3411765, 1,
4.242424, 2.454545, -593.9062, 0.7568628, 0, 0.2392157, 1,
4.282828, 2.454545, -591.4911, 0.7568628, 0, 0.2392157, 1,
4.323232, 2.454545, -589.1301, 0.7568628, 0, 0.2392157, 1,
4.363636, 2.454545, -586.8233, 0.8588235, 0, 0.1372549, 1,
4.40404, 2.454545, -584.5707, 0.8588235, 0, 0.1372549, 1,
4.444445, 2.454545, -582.3723, 0.9647059, 0, 0.03137255, 1,
4.484848, 2.454545, -580.2281, 0.9647059, 0, 0.03137255, 1,
4.525252, 2.454545, -578.1381, 0.9647059, 0, 0.03137255, 1,
4.565657, 2.454545, -576.1022, 1, 0.06666667, 0, 1,
4.606061, 2.454545, -574.1206, 1, 0.06666667, 0, 1,
4.646465, 2.454545, -572.1931, 1, 0.06666667, 0, 1,
4.686869, 2.454545, -570.3199, 1, 0.1686275, 0, 1,
4.727273, 2.454545, -568.5008, 1, 0.1686275, 0, 1,
4.767677, 2.454545, -566.736, 1, 0.1686275, 0, 1,
4.808081, 2.454545, -565.0253, 1, 0.2745098, 0, 1,
4.848485, 2.454545, -563.3688, 1, 0.2745098, 0, 1,
4.888889, 2.454545, -561.7665, 1, 0.2745098, 0, 1,
4.929293, 2.454545, -560.2184, 1, 0.2745098, 0, 1,
4.969697, 2.454545, -558.7245, 1, 0.3764706, 0, 1,
5.010101, 2.454545, -557.2848, 1, 0.3764706, 0, 1,
5.050505, 2.454545, -555.8992, 1, 0.3764706, 0, 1,
5.090909, 2.454545, -554.5679, 1, 0.3764706, 0, 1,
5.131313, 2.454545, -553.2908, 1, 0.4823529, 0, 1,
5.171717, 2.454545, -552.0679, 1, 0.4823529, 0, 1,
5.212121, 2.454545, -550.8991, 1, 0.4823529, 0, 1,
5.252525, 2.454545, -549.7845, 1, 0.4823529, 0, 1,
5.292929, 2.454545, -548.7242, 1, 0.4823529, 0, 1,
5.333333, 2.454545, -547.718, 1, 0.5843138, 0, 1,
5.373737, 2.454545, -546.766, 1, 0.5843138, 0, 1,
5.414141, 2.454545, -545.8682, 1, 0.5843138, 0, 1,
5.454545, 2.454545, -545.0246, 1, 0.5843138, 0, 1,
5.494949, 2.454545, -544.2352, 1, 0.5843138, 0, 1,
5.535354, 2.454545, -543.5, 1, 0.5843138, 0, 1,
5.575758, 2.454545, -542.819, 1, 0.5843138, 0, 1,
5.616162, 2.454545, -542.1921, 1, 0.6862745, 0, 1,
5.656566, 2.454545, -541.6195, 1, 0.6862745, 0, 1,
5.69697, 2.454545, -541.1011, 1, 0.6862745, 0, 1,
5.737374, 2.454545, -540.6368, 1, 0.6862745, 0, 1,
5.777778, 2.454545, -540.2267, 1, 0.6862745, 0, 1,
5.818182, 2.454545, -539.8708, 1, 0.6862745, 0, 1,
5.858586, 2.454545, -539.5692, 1, 0.6862745, 0, 1,
5.89899, 2.454545, -539.3217, 1, 0.6862745, 0, 1,
5.939394, 2.454545, -539.1284, 1, 0.6862745, 0, 1,
5.979798, 2.454545, -538.9893, 1, 0.6862745, 0, 1,
6.020202, 2.454545, -538.9044, 1, 0.6862745, 0, 1,
6.060606, 2.454545, -538.8737, 1, 0.6862745, 0, 1,
6.10101, 2.454545, -538.8972, 1, 0.6862745, 0, 1,
6.141414, 2.454545, -538.9749, 1, 0.6862745, 0, 1,
6.181818, 2.454545, -539.1067, 1, 0.6862745, 0, 1,
6.222222, 2.454545, -539.2928, 1, 0.6862745, 0, 1,
6.262626, 2.454545, -539.533, 1, 0.6862745, 0, 1,
6.30303, 2.454545, -539.8275, 1, 0.6862745, 0, 1,
6.343434, 2.454545, -540.1761, 1, 0.6862745, 0, 1,
6.383838, 2.454545, -540.5789, 1, 0.6862745, 0, 1,
6.424242, 2.454545, -541.0359, 1, 0.6862745, 0, 1,
6.464646, 2.454545, -541.5471, 1, 0.6862745, 0, 1,
6.505051, 2.454545, -542.1125, 1, 0.6862745, 0, 1,
6.545455, 2.454545, -542.7321, 1, 0.6862745, 0, 1,
6.585859, 2.454545, -543.4059, 1, 0.5843138, 0, 1,
6.626263, 2.454545, -544.1339, 1, 0.5843138, 0, 1,
6.666667, 2.454545, -544.916, 1, 0.5843138, 0, 1,
6.707071, 2.454545, -545.7524, 1, 0.5843138, 0, 1,
6.747475, 2.454545, -546.6429, 1, 0.5843138, 0, 1,
6.787879, 2.454545, -547.5877, 1, 0.5843138, 0, 1,
6.828283, 2.454545, -548.5866, 1, 0.4823529, 0, 1,
6.868687, 2.454545, -549.6398, 1, 0.4823529, 0, 1,
6.909091, 2.454545, -550.7471, 1, 0.4823529, 0, 1,
6.949495, 2.454545, -551.9086, 1, 0.4823529, 0, 1,
6.989899, 2.454545, -553.1243, 1, 0.4823529, 0, 1,
7.030303, 2.454545, -554.3942, 1, 0.3764706, 0, 1,
7.070707, 2.454545, -555.7183, 1, 0.3764706, 0, 1,
7.111111, 2.454545, -557.0966, 1, 0.3764706, 0, 1,
7.151515, 2.454545, -558.5291, 1, 0.3764706, 0, 1,
7.191919, 2.454545, -560.0157, 1, 0.3764706, 0, 1,
7.232323, 2.454545, -561.5566, 1, 0.2745098, 0, 1,
7.272727, 2.454545, -563.1517, 1, 0.2745098, 0, 1,
7.313131, 2.454545, -564.8009, 1, 0.2745098, 0, 1,
7.353535, 2.454545, -566.5043, 1, 0.1686275, 0, 1,
7.393939, 2.454545, -568.262, 1, 0.1686275, 0, 1,
7.434343, 2.454545, -570.0738, 1, 0.1686275, 0, 1,
7.474748, 2.454545, -571.9398, 1, 0.06666667, 0, 1,
7.515152, 2.454545, -573.86, 1, 0.06666667, 0, 1,
7.555555, 2.454545, -575.8344, 1, 0.06666667, 0, 1,
7.59596, 2.454545, -577.863, 0.9647059, 0, 0.03137255, 1,
7.636364, 2.454545, -579.9458, 0.9647059, 0, 0.03137255, 1,
7.676768, 2.454545, -582.0828, 0.9647059, 0, 0.03137255, 1,
7.717172, 2.454545, -584.2739, 0.8588235, 0, 0.1372549, 1,
7.757576, 2.454545, -586.5193, 0.8588235, 0, 0.1372549, 1,
7.79798, 2.454545, -588.8188, 0.8588235, 0, 0.1372549, 1,
7.838384, 2.454545, -591.1726, 0.7568628, 0, 0.2392157, 1,
7.878788, 2.454545, -593.5805, 0.7568628, 0, 0.2392157, 1,
7.919192, 2.454545, -596.0427, 0.654902, 0, 0.3411765, 1,
7.959596, 2.454545, -598.559, 0.654902, 0, 0.3411765, 1,
8, 2.454545, -601.1295, 0.5490196, 0, 0.4470588, 1,
4, 2.50505, -603.7831, 0.5490196, 0, 0.4470588, 1,
4.040404, 2.50505, -601.1521, 0.5490196, 0, 0.4470588, 1,
4.080808, 2.50505, -598.5732, 0.654902, 0, 0.3411765, 1,
4.121212, 2.50505, -596.0463, 0.654902, 0, 0.3411765, 1,
4.161616, 2.50505, -593.5715, 0.7568628, 0, 0.2392157, 1,
4.20202, 2.50505, -591.1486, 0.7568628, 0, 0.2392157, 1,
4.242424, 2.50505, -588.7778, 0.8588235, 0, 0.1372549, 1,
4.282828, 2.50505, -586.459, 0.8588235, 0, 0.1372549, 1,
4.323232, 2.50505, -584.1923, 0.8588235, 0, 0.1372549, 1,
4.363636, 2.50505, -581.9776, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.50505, -579.8149, 0.9647059, 0, 0.03137255, 1,
4.444445, 2.50505, -577.7043, 0.9647059, 0, 0.03137255, 1,
4.484848, 2.50505, -575.6456, 1, 0.06666667, 0, 1,
4.525252, 2.50505, -573.639, 1, 0.06666667, 0, 1,
4.565657, 2.50505, -571.6844, 1, 0.1686275, 0, 1,
4.606061, 2.50505, -569.7819, 1, 0.1686275, 0, 1,
4.646465, 2.50505, -567.9315, 1, 0.1686275, 0, 1,
4.686869, 2.50505, -566.1329, 1, 0.1686275, 0, 1,
4.727273, 2.50505, -564.3865, 1, 0.2745098, 0, 1,
4.767677, 2.50505, -562.6921, 1, 0.2745098, 0, 1,
4.808081, 2.50505, -561.0497, 1, 0.2745098, 0, 1,
4.848485, 2.50505, -559.4594, 1, 0.3764706, 0, 1,
4.888889, 2.50505, -557.921, 1, 0.3764706, 0, 1,
4.929293, 2.50505, -556.4347, 1, 0.3764706, 0, 1,
4.969697, 2.50505, -555.0004, 1, 0.3764706, 0, 1,
5.010101, 2.50505, -553.6182, 1, 0.4823529, 0, 1,
5.050505, 2.50505, -552.288, 1, 0.4823529, 0, 1,
5.090909, 2.50505, -551.0098, 1, 0.4823529, 0, 1,
5.131313, 2.50505, -549.7836, 1, 0.4823529, 0, 1,
5.171717, 2.50505, -548.6094, 1, 0.4823529, 0, 1,
5.212121, 2.50505, -547.4874, 1, 0.5843138, 0, 1,
5.252525, 2.50505, -546.4173, 1, 0.5843138, 0, 1,
5.292929, 2.50505, -545.3992, 1, 0.5843138, 0, 1,
5.333333, 2.50505, -544.4332, 1, 0.5843138, 0, 1,
5.373737, 2.50505, -543.5192, 1, 0.5843138, 0, 1,
5.414141, 2.50505, -542.6573, 1, 0.6862745, 0, 1,
5.454545, 2.50505, -541.8474, 1, 0.6862745, 0, 1,
5.494949, 2.50505, -541.0895, 1, 0.6862745, 0, 1,
5.535354, 2.50505, -540.3836, 1, 0.6862745, 0, 1,
5.575758, 2.50505, -539.7297, 1, 0.6862745, 0, 1,
5.616162, 2.50505, -539.1279, 1, 0.6862745, 0, 1,
5.656566, 2.50505, -538.5782, 1, 0.6862745, 0, 1,
5.69697, 2.50505, -538.0804, 1, 0.6862745, 0, 1,
5.737374, 2.50505, -537.6347, 1, 0.6862745, 0, 1,
5.777778, 2.50505, -537.241, 1, 0.6862745, 0, 1,
5.818182, 2.50505, -536.8994, 1, 0.7921569, 0, 1,
5.858586, 2.50505, -536.6097, 1, 0.7921569, 0, 1,
5.89899, 2.50505, -536.3721, 1, 0.7921569, 0, 1,
5.939394, 2.50505, -536.1865, 1, 0.7921569, 0, 1,
5.979798, 2.50505, -536.053, 1, 0.7921569, 0, 1,
6.020202, 2.50505, -535.9714, 1, 0.7921569, 0, 1,
6.060606, 2.50505, -535.942, 1, 0.7921569, 0, 1,
6.10101, 2.50505, -535.9645, 1, 0.7921569, 0, 1,
6.141414, 2.50505, -536.0391, 1, 0.7921569, 0, 1,
6.181818, 2.50505, -536.1656, 1, 0.7921569, 0, 1,
6.222222, 2.50505, -536.3443, 1, 0.7921569, 0, 1,
6.262626, 2.50505, -536.575, 1, 0.7921569, 0, 1,
6.30303, 2.50505, -536.8577, 1, 0.7921569, 0, 1,
6.343434, 2.50505, -537.1924, 1, 0.6862745, 0, 1,
6.383838, 2.50505, -537.5791, 1, 0.6862745, 0, 1,
6.424242, 2.50505, -538.0179, 1, 0.6862745, 0, 1,
6.464646, 2.50505, -538.5087, 1, 0.6862745, 0, 1,
6.505051, 2.50505, -539.0515, 1, 0.6862745, 0, 1,
6.545455, 2.50505, -539.6464, 1, 0.6862745, 0, 1,
6.585859, 2.50505, -540.2933, 1, 0.6862745, 0, 1,
6.626263, 2.50505, -540.9922, 1, 0.6862745, 0, 1,
6.666667, 2.50505, -541.7431, 1, 0.6862745, 0, 1,
6.707071, 2.50505, -542.5461, 1, 0.6862745, 0, 1,
6.747475, 2.50505, -543.4011, 1, 0.5843138, 0, 1,
6.787879, 2.50505, -544.3082, 1, 0.5843138, 0, 1,
6.828283, 2.50505, -545.2672, 1, 0.5843138, 0, 1,
6.868687, 2.50505, -546.2783, 1, 0.5843138, 0, 1,
6.909091, 2.50505, -547.3414, 1, 0.5843138, 0, 1,
6.949495, 2.50505, -548.4566, 1, 0.5843138, 0, 1,
6.989899, 2.50505, -549.6238, 1, 0.4823529, 0, 1,
7.030303, 2.50505, -550.843, 1, 0.4823529, 0, 1,
7.070707, 2.50505, -552.1142, 1, 0.4823529, 0, 1,
7.111111, 2.50505, -553.4375, 1, 0.4823529, 0, 1,
7.151515, 2.50505, -554.8128, 1, 0.3764706, 0, 1,
7.191919, 2.50505, -556.2401, 1, 0.3764706, 0, 1,
7.232323, 2.50505, -557.7195, 1, 0.3764706, 0, 1,
7.272727, 2.50505, -559.2509, 1, 0.3764706, 0, 1,
7.313131, 2.50505, -560.8342, 1, 0.2745098, 0, 1,
7.353535, 2.50505, -562.4697, 1, 0.2745098, 0, 1,
7.393939, 2.50505, -564.1572, 1, 0.2745098, 0, 1,
7.434343, 2.50505, -565.8967, 1, 0.2745098, 0, 1,
7.474748, 2.50505, -567.6882, 1, 0.1686275, 0, 1,
7.515152, 2.50505, -569.5317, 1, 0.1686275, 0, 1,
7.555555, 2.50505, -571.4274, 1, 0.1686275, 0, 1,
7.59596, 2.50505, -573.375, 1, 0.06666667, 0, 1,
7.636364, 2.50505, -575.3746, 1, 0.06666667, 0, 1,
7.676768, 2.50505, -577.4263, 1, 0.06666667, 0, 1,
7.717172, 2.50505, -579.53, 0.9647059, 0, 0.03137255, 1,
7.757576, 2.50505, -581.6857, 0.9647059, 0, 0.03137255, 1,
7.79798, 2.50505, -583.8935, 0.8588235, 0, 0.1372549, 1,
7.838384, 2.50505, -586.1533, 0.8588235, 0, 0.1372549, 1,
7.878788, 2.50505, -588.4651, 0.8588235, 0, 0.1372549, 1,
7.919192, 2.50505, -590.829, 0.7568628, 0, 0.2392157, 1,
7.959596, 2.50505, -593.2449, 0.7568628, 0, 0.2392157, 1,
8, 2.50505, -595.7128, 0.654902, 0, 0.3411765, 1,
4, 2.555556, -598.5263, 0.654902, 0, 0.3411765, 1,
4.040404, 2.555556, -595.9983, 0.654902, 0, 0.3411765, 1,
4.080808, 2.555556, -593.5203, 0.7568628, 0, 0.2392157, 1,
4.121212, 2.555556, -591.0923, 0.7568628, 0, 0.2392157, 1,
4.161616, 2.555556, -588.7143, 0.8588235, 0, 0.1372549, 1,
4.20202, 2.555556, -586.3863, 0.8588235, 0, 0.1372549, 1,
4.242424, 2.555556, -584.1083, 0.8588235, 0, 0.1372549, 1,
4.282828, 2.555556, -581.8802, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.555556, -579.7022, 0.9647059, 0, 0.03137255, 1,
4.363636, 2.555556, -577.5742, 0.9647059, 0, 0.03137255, 1,
4.40404, 2.555556, -575.4961, 1, 0.06666667, 0, 1,
4.444445, 2.555556, -573.4681, 1, 0.06666667, 0, 1,
4.484848, 2.555556, -571.49, 1, 0.1686275, 0, 1,
4.525252, 2.555556, -569.562, 1, 0.1686275, 0, 1,
4.565657, 2.555556, -567.6838, 1, 0.1686275, 0, 1,
4.606061, 2.555556, -565.8558, 1, 0.2745098, 0, 1,
4.646465, 2.555556, -564.0777, 1, 0.2745098, 0, 1,
4.686869, 2.555556, -562.3496, 1, 0.2745098, 0, 1,
4.727273, 2.555556, -560.6715, 1, 0.2745098, 0, 1,
4.767677, 2.555556, -559.0434, 1, 0.3764706, 0, 1,
4.808081, 2.555556, -557.4653, 1, 0.3764706, 0, 1,
4.848485, 2.555556, -555.9371, 1, 0.3764706, 0, 1,
4.888889, 2.555556, -554.459, 1, 0.3764706, 0, 1,
4.929293, 2.555556, -553.0309, 1, 0.4823529, 0, 1,
4.969697, 2.555556, -551.6527, 1, 0.4823529, 0, 1,
5.010101, 2.555556, -550.3246, 1, 0.4823529, 0, 1,
5.050505, 2.555556, -549.0464, 1, 0.4823529, 0, 1,
5.090909, 2.555556, -547.8182, 1, 0.5843138, 0, 1,
5.131313, 2.555556, -546.6401, 1, 0.5843138, 0, 1,
5.171717, 2.555556, -545.5119, 1, 0.5843138, 0, 1,
5.212121, 2.555556, -544.4337, 1, 0.5843138, 0, 1,
5.252525, 2.555556, -543.4055, 1, 0.5843138, 0, 1,
5.292929, 2.555556, -542.4273, 1, 0.6862745, 0, 1,
5.333333, 2.555556, -541.4991, 1, 0.6862745, 0, 1,
5.373737, 2.555556, -540.6209, 1, 0.6862745, 0, 1,
5.414141, 2.555556, -539.7927, 1, 0.6862745, 0, 1,
5.454545, 2.555556, -539.0145, 1, 0.6862745, 0, 1,
5.494949, 2.555556, -538.2862, 1, 0.6862745, 0, 1,
5.535354, 2.555556, -537.608, 1, 0.6862745, 0, 1,
5.575758, 2.555556, -536.9797, 1, 0.6862745, 0, 1,
5.616162, 2.555556, -536.4014, 1, 0.7921569, 0, 1,
5.656566, 2.555556, -535.8732, 1, 0.7921569, 0, 1,
5.69697, 2.555556, -535.3949, 1, 0.7921569, 0, 1,
5.737374, 2.555556, -534.9666, 1, 0.7921569, 0, 1,
5.777778, 2.555556, -534.5883, 1, 0.7921569, 0, 1,
5.818182, 2.555556, -534.2601, 1, 0.7921569, 0, 1,
5.858586, 2.555556, -533.9818, 1, 0.7921569, 0, 1,
5.89899, 2.555556, -533.7534, 1, 0.7921569, 0, 1,
5.939394, 2.555556, -533.5751, 1, 0.7921569, 0, 1,
5.979798, 2.555556, -533.4468, 1, 0.7921569, 0, 1,
6.020202, 2.555556, -533.3685, 1, 0.7921569, 0, 1,
6.060606, 2.555556, -533.3401, 1, 0.7921569, 0, 1,
6.10101, 2.555556, -533.3618, 1, 0.7921569, 0, 1,
6.141414, 2.555556, -533.4335, 1, 0.7921569, 0, 1,
6.181818, 2.555556, -533.5551, 1, 0.7921569, 0, 1,
6.222222, 2.555556, -533.7267, 1, 0.7921569, 0, 1,
6.262626, 2.555556, -533.9484, 1, 0.7921569, 0, 1,
6.30303, 2.555556, -534.22, 1, 0.7921569, 0, 1,
6.343434, 2.555556, -534.5416, 1, 0.7921569, 0, 1,
6.383838, 2.555556, -534.9132, 1, 0.7921569, 0, 1,
6.424242, 2.555556, -535.3348, 1, 0.7921569, 0, 1,
6.464646, 2.555556, -535.8064, 1, 0.7921569, 0, 1,
6.505051, 2.555556, -536.328, 1, 0.7921569, 0, 1,
6.545455, 2.555556, -536.8996, 1, 0.7921569, 0, 1,
6.585859, 2.555556, -537.5212, 1, 0.6862745, 0, 1,
6.626263, 2.555556, -538.1927, 1, 0.6862745, 0, 1,
6.666667, 2.555556, -538.9142, 1, 0.6862745, 0, 1,
6.707071, 2.555556, -539.6859, 1, 0.6862745, 0, 1,
6.747475, 2.555556, -540.5074, 1, 0.6862745, 0, 1,
6.787879, 2.555556, -541.3789, 1, 0.6862745, 0, 1,
6.828283, 2.555556, -542.3004, 1, 0.6862745, 0, 1,
6.868687, 2.555556, -543.272, 1, 0.5843138, 0, 1,
6.909091, 2.555556, -544.2935, 1, 0.5843138, 0, 1,
6.949495, 2.555556, -545.365, 1, 0.5843138, 0, 1,
6.989899, 2.555556, -546.4865, 1, 0.5843138, 0, 1,
7.030303, 2.555556, -547.658, 1, 0.5843138, 0, 1,
7.070707, 2.555556, -548.8795, 1, 0.4823529, 0, 1,
7.111111, 2.555556, -550.1509, 1, 0.4823529, 0, 1,
7.151515, 2.555556, -551.4724, 1, 0.4823529, 0, 1,
7.191919, 2.555556, -552.8439, 1, 0.4823529, 0, 1,
7.232323, 2.555556, -554.2654, 1, 0.4823529, 0, 1,
7.272727, 2.555556, -555.7368, 1, 0.3764706, 0, 1,
7.313131, 2.555556, -557.2582, 1, 0.3764706, 0, 1,
7.353535, 2.555556, -558.8297, 1, 0.3764706, 0, 1,
7.393939, 2.555556, -560.4512, 1, 0.2745098, 0, 1,
7.434343, 2.555556, -562.1226, 1, 0.2745098, 0, 1,
7.474748, 2.555556, -563.844, 1, 0.2745098, 0, 1,
7.515152, 2.555556, -565.6154, 1, 0.2745098, 0, 1,
7.555555, 2.555556, -567.4368, 1, 0.1686275, 0, 1,
7.59596, 2.555556, -569.3082, 1, 0.1686275, 0, 1,
7.636364, 2.555556, -571.2296, 1, 0.1686275, 0, 1,
7.676768, 2.555556, -573.201, 1, 0.06666667, 0, 1,
7.717172, 2.555556, -575.2224, 1, 0.06666667, 0, 1,
7.757576, 2.555556, -577.2937, 1, 0.06666667, 0, 1,
7.79798, 2.555556, -579.4151, 0.9647059, 0, 0.03137255, 1,
7.838384, 2.555556, -581.5864, 0.9647059, 0, 0.03137255, 1,
7.878788, 2.555556, -583.8078, 0.8588235, 0, 0.1372549, 1,
7.919192, 2.555556, -586.0791, 0.8588235, 0, 0.1372549, 1,
7.959596, 2.555556, -588.4005, 0.8588235, 0, 0.1372549, 1,
8, 2.555556, -590.7718, 0.7568628, 0, 0.2392157, 1,
4, 2.606061, -593.7239, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.606061, -591.2929, 0.7568628, 0, 0.2392157, 1,
4.080808, 2.606061, -588.91, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.606061, -586.5753, 0.8588235, 0, 0.1372549, 1,
4.161616, 2.606061, -584.2885, 0.8588235, 0, 0.1372549, 1,
4.20202, 2.606061, -582.0499, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.606061, -579.8593, 0.9647059, 0, 0.03137255, 1,
4.282828, 2.606061, -577.7167, 0.9647059, 0, 0.03137255, 1,
4.323232, 2.606061, -575.6223, 1, 0.06666667, 0, 1,
4.363636, 2.606061, -573.576, 1, 0.06666667, 0, 1,
4.40404, 2.606061, -571.5777, 1, 0.1686275, 0, 1,
4.444445, 2.606061, -569.6275, 1, 0.1686275, 0, 1,
4.484848, 2.606061, -567.7253, 1, 0.1686275, 0, 1,
4.525252, 2.606061, -565.8713, 1, 0.2745098, 0, 1,
4.565657, 2.606061, -564.0653, 1, 0.2745098, 0, 1,
4.606061, 2.606061, -562.3074, 1, 0.2745098, 0, 1,
4.646465, 2.606061, -560.5975, 1, 0.2745098, 0, 1,
4.686869, 2.606061, -558.9358, 1, 0.3764706, 0, 1,
4.727273, 2.606061, -557.3221, 1, 0.3764706, 0, 1,
4.767677, 2.606061, -555.7565, 1, 0.3764706, 0, 1,
4.808081, 2.606061, -554.239, 1, 0.4823529, 0, 1,
4.848485, 2.606061, -552.7695, 1, 0.4823529, 0, 1,
4.888889, 2.606061, -551.3481, 1, 0.4823529, 0, 1,
4.929293, 2.606061, -549.9747, 1, 0.4823529, 0, 1,
4.969697, 2.606061, -548.6495, 1, 0.4823529, 0, 1,
5.010101, 2.606061, -547.3723, 1, 0.5843138, 0, 1,
5.050505, 2.606061, -546.1432, 1, 0.5843138, 0, 1,
5.090909, 2.606061, -544.9622, 1, 0.5843138, 0, 1,
5.131313, 2.606061, -543.8293, 1, 0.5843138, 0, 1,
5.171717, 2.606061, -542.7444, 1, 0.6862745, 0, 1,
5.212121, 2.606061, -541.7076, 1, 0.6862745, 0, 1,
5.252525, 2.606061, -540.7189, 1, 0.6862745, 0, 1,
5.292929, 2.606061, -539.7782, 1, 0.6862745, 0, 1,
5.333333, 2.606061, -538.8856, 1, 0.6862745, 0, 1,
5.373737, 2.606061, -538.0411, 1, 0.6862745, 0, 1,
5.414141, 2.606061, -537.2447, 1, 0.6862745, 0, 1,
5.454545, 2.606061, -536.4963, 1, 0.7921569, 0, 1,
5.494949, 2.606061, -535.796, 1, 0.7921569, 0, 1,
5.535354, 2.606061, -535.1438, 1, 0.7921569, 0, 1,
5.575758, 2.606061, -534.5397, 1, 0.7921569, 0, 1,
5.616162, 2.606061, -533.9836, 1, 0.7921569, 0, 1,
5.656566, 2.606061, -533.4756, 1, 0.7921569, 0, 1,
5.69697, 2.606061, -533.0157, 1, 0.7921569, 0, 1,
5.737374, 2.606061, -532.6039, 1, 0.7921569, 0, 1,
5.777778, 2.606061, -532.2401, 1, 0.7921569, 0, 1,
5.818182, 2.606061, -531.9244, 1, 0.7921569, 0, 1,
5.858586, 2.606061, -531.6568, 1, 0.7921569, 0, 1,
5.89899, 2.606061, -531.4373, 1, 0.7921569, 0, 1,
5.939394, 2.606061, -531.2658, 1, 0.7921569, 0, 1,
5.979798, 2.606061, -531.1424, 1, 0.8941177, 0, 1,
6.020202, 2.606061, -531.0671, 1, 0.8941177, 0, 1,
6.060606, 2.606061, -531.0399, 1, 0.8941177, 0, 1,
6.10101, 2.606061, -531.0607, 1, 0.8941177, 0, 1,
6.141414, 2.606061, -531.1296, 1, 0.8941177, 0, 1,
6.181818, 2.606061, -531.2465, 1, 0.7921569, 0, 1,
6.222222, 2.606061, -531.4116, 1, 0.7921569, 0, 1,
6.262626, 2.606061, -531.6247, 1, 0.7921569, 0, 1,
6.30303, 2.606061, -531.8859, 1, 0.7921569, 0, 1,
6.343434, 2.606061, -532.1952, 1, 0.7921569, 0, 1,
6.383838, 2.606061, -532.5525, 1, 0.7921569, 0, 1,
6.424242, 2.606061, -532.9579, 1, 0.7921569, 0, 1,
6.464646, 2.606061, -533.4114, 1, 0.7921569, 0, 1,
6.505051, 2.606061, -533.913, 1, 0.7921569, 0, 1,
6.545455, 2.606061, -534.4626, 1, 0.7921569, 0, 1,
6.585859, 2.606061, -535.0604, 1, 0.7921569, 0, 1,
6.626263, 2.606061, -535.7061, 1, 0.7921569, 0, 1,
6.666667, 2.606061, -536.4, 1, 0.7921569, 0, 1,
6.707071, 2.606061, -537.142, 1, 0.6862745, 0, 1,
6.747475, 2.606061, -537.9319, 1, 0.6862745, 0, 1,
6.787879, 2.606061, -538.77, 1, 0.6862745, 0, 1,
6.828283, 2.606061, -539.6562, 1, 0.6862745, 0, 1,
6.868687, 2.606061, -540.5905, 1, 0.6862745, 0, 1,
6.909091, 2.606061, -541.5728, 1, 0.6862745, 0, 1,
6.949495, 2.606061, -542.6031, 1, 0.6862745, 0, 1,
6.989899, 2.606061, -543.6816, 1, 0.5843138, 0, 1,
7.030303, 2.606061, -544.8081, 1, 0.5843138, 0, 1,
7.070707, 2.606061, -545.9827, 1, 0.5843138, 0, 1,
7.111111, 2.606061, -547.2054, 1, 0.5843138, 0, 1,
7.151515, 2.606061, -548.4761, 1, 0.5843138, 0, 1,
7.191919, 2.606061, -549.795, 1, 0.4823529, 0, 1,
7.232323, 2.606061, -551.1619, 1, 0.4823529, 0, 1,
7.272727, 2.606061, -552.5768, 1, 0.4823529, 0, 1,
7.313131, 2.606061, -554.0399, 1, 0.4823529, 0, 1,
7.353535, 2.606061, -555.551, 1, 0.3764706, 0, 1,
7.393939, 2.606061, -557.1102, 1, 0.3764706, 0, 1,
7.434343, 2.606061, -558.7175, 1, 0.3764706, 0, 1,
7.474748, 2.606061, -560.3728, 1, 0.2745098, 0, 1,
7.515152, 2.606061, -562.0762, 1, 0.2745098, 0, 1,
7.555555, 2.606061, -563.8277, 1, 0.2745098, 0, 1,
7.59596, 2.606061, -565.6273, 1, 0.2745098, 0, 1,
7.636364, 2.606061, -567.4749, 1, 0.1686275, 0, 1,
7.676768, 2.606061, -569.3707, 1, 0.1686275, 0, 1,
7.717172, 2.606061, -571.3145, 1, 0.1686275, 0, 1,
7.757576, 2.606061, -573.3063, 1, 0.06666667, 0, 1,
7.79798, 2.606061, -575.3463, 1, 0.06666667, 0, 1,
7.838384, 2.606061, -577.4343, 1, 0.06666667, 0, 1,
7.878788, 2.606061, -579.5703, 0.9647059, 0, 0.03137255, 1,
7.919192, 2.606061, -581.7545, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.606061, -583.9867, 0.8588235, 0, 0.1372549, 1,
8, 2.606061, -586.267, 0.8588235, 0, 0.1372549, 1,
4, 2.656566, -589.3387, 0.7568628, 0, 0.2392157, 1,
4.040404, 2.656566, -586.9993, 0.8588235, 0, 0.1372549, 1,
4.080808, 2.656566, -584.7062, 0.8588235, 0, 0.1372549, 1,
4.121212, 2.656566, -582.4593, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.656566, -580.2587, 0.9647059, 0, 0.03137255, 1,
4.20202, 2.656566, -578.1044, 0.9647059, 0, 0.03137255, 1,
4.242424, 2.656566, -575.9963, 1, 0.06666667, 0, 1,
4.282828, 2.656566, -573.9344, 1, 0.06666667, 0, 1,
4.323232, 2.656566, -571.9189, 1, 0.06666667, 0, 1,
4.363636, 2.656566, -569.9496, 1, 0.1686275, 0, 1,
4.40404, 2.656566, -568.0266, 1, 0.1686275, 0, 1,
4.444445, 2.656566, -566.1498, 1, 0.1686275, 0, 1,
4.484848, 2.656566, -564.3193, 1, 0.2745098, 0, 1,
4.525252, 2.656566, -562.5351, 1, 0.2745098, 0, 1,
4.565657, 2.656566, -560.7971, 1, 0.2745098, 0, 1,
4.606061, 2.656566, -559.1054, 1, 0.3764706, 0, 1,
4.646465, 2.656566, -557.46, 1, 0.3764706, 0, 1,
4.686869, 2.656566, -555.8608, 1, 0.3764706, 0, 1,
4.727273, 2.656566, -554.3079, 1, 0.4823529, 0, 1,
4.767677, 2.656566, -552.8012, 1, 0.4823529, 0, 1,
4.808081, 2.656566, -551.3408, 1, 0.4823529, 0, 1,
4.848485, 2.656566, -549.9267, 1, 0.4823529, 0, 1,
4.888889, 2.656566, -548.5588, 1, 0.4823529, 0, 1,
4.929293, 2.656566, -547.2372, 1, 0.5843138, 0, 1,
4.969697, 2.656566, -545.9619, 1, 0.5843138, 0, 1,
5.010101, 2.656566, -544.7328, 1, 0.5843138, 0, 1,
5.050505, 2.656566, -543.55, 1, 0.5843138, 0, 1,
5.090909, 2.656566, -542.4135, 1, 0.6862745, 0, 1,
5.131313, 2.656566, -541.3232, 1, 0.6862745, 0, 1,
5.171717, 2.656566, -540.2792, 1, 0.6862745, 0, 1,
5.212121, 2.656566, -539.2814, 1, 0.6862745, 0, 1,
5.252525, 2.656566, -538.33, 1, 0.6862745, 0, 1,
5.292929, 2.656566, -537.4247, 1, 0.6862745, 0, 1,
5.333333, 2.656566, -536.5657, 1, 0.7921569, 0, 1,
5.373737, 2.656566, -535.7531, 1, 0.7921569, 0, 1,
5.414141, 2.656566, -534.9866, 1, 0.7921569, 0, 1,
5.454545, 2.656566, -534.2664, 1, 0.7921569, 0, 1,
5.494949, 2.656566, -533.5925, 1, 0.7921569, 0, 1,
5.535354, 2.656566, -532.9648, 1, 0.7921569, 0, 1,
5.575758, 2.656566, -532.3835, 1, 0.7921569, 0, 1,
5.616162, 2.656566, -531.8484, 1, 0.7921569, 0, 1,
5.656566, 2.656566, -531.3595, 1, 0.7921569, 0, 1,
5.69697, 2.656566, -530.9169, 1, 0.8941177, 0, 1,
5.737374, 2.656566, -530.5206, 1, 0.8941177, 0, 1,
5.777778, 2.656566, -530.1705, 1, 0.8941177, 0, 1,
5.818182, 2.656566, -529.8667, 1, 0.8941177, 0, 1,
5.858586, 2.656566, -529.6092, 1, 0.8941177, 0, 1,
5.89899, 2.656566, -529.3979, 1, 0.8941177, 0, 1,
5.939394, 2.656566, -529.2329, 1, 0.8941177, 0, 1,
5.979798, 2.656566, -529.1141, 1, 0.8941177, 0, 1,
6.020202, 2.656566, -529.0417, 1, 0.8941177, 0, 1,
6.060606, 2.656566, -529.0154, 1, 0.8941177, 0, 1,
6.10101, 2.656566, -529.0355, 1, 0.8941177, 0, 1,
6.141414, 2.656566, -529.1018, 1, 0.8941177, 0, 1,
6.181818, 2.656566, -529.2144, 1, 0.8941177, 0, 1,
6.222222, 2.656566, -529.3732, 1, 0.8941177, 0, 1,
6.262626, 2.656566, -529.5783, 1, 0.8941177, 0, 1,
6.30303, 2.656566, -529.8297, 1, 0.8941177, 0, 1,
6.343434, 2.656566, -530.1273, 1, 0.8941177, 0, 1,
6.383838, 2.656566, -530.4711, 1, 0.8941177, 0, 1,
6.424242, 2.656566, -530.8613, 1, 0.8941177, 0, 1,
6.464646, 2.656566, -531.2977, 1, 0.7921569, 0, 1,
6.505051, 2.656566, -531.7804, 1, 0.7921569, 0, 1,
6.545455, 2.656566, -532.3093, 1, 0.7921569, 0, 1,
6.585859, 2.656566, -532.8845, 1, 0.7921569, 0, 1,
6.626263, 2.656566, -533.506, 1, 0.7921569, 0, 1,
6.666667, 2.656566, -534.1738, 1, 0.7921569, 0, 1,
6.707071, 2.656566, -534.8878, 1, 0.7921569, 0, 1,
6.747475, 2.656566, -535.648, 1, 0.7921569, 0, 1,
6.787879, 2.656566, -536.4545, 1, 0.7921569, 0, 1,
6.828283, 2.656566, -537.3073, 1, 0.6862745, 0, 1,
6.868687, 2.656566, -538.2064, 1, 0.6862745, 0, 1,
6.909091, 2.656566, -539.1517, 1, 0.6862745, 0, 1,
6.949495, 2.656566, -540.1432, 1, 0.6862745, 0, 1,
6.989899, 2.656566, -541.1811, 1, 0.6862745, 0, 1,
7.030303, 2.656566, -542.2652, 1, 0.6862745, 0, 1,
7.070707, 2.656566, -543.3956, 1, 0.5843138, 0, 1,
7.111111, 2.656566, -544.5722, 1, 0.5843138, 0, 1,
7.151515, 2.656566, -545.795, 1, 0.5843138, 0, 1,
7.191919, 2.656566, -547.0642, 1, 0.5843138, 0, 1,
7.232323, 2.656566, -548.3796, 1, 0.5843138, 0, 1,
7.272727, 2.656566, -549.7413, 1, 0.4823529, 0, 1,
7.313131, 2.656566, -551.1493, 1, 0.4823529, 0, 1,
7.353535, 2.656566, -552.6035, 1, 0.4823529, 0, 1,
7.393939, 2.656566, -554.1039, 1, 0.4823529, 0, 1,
7.434343, 2.656566, -555.6507, 1, 0.3764706, 0, 1,
7.474748, 2.656566, -557.2437, 1, 0.3764706, 0, 1,
7.515152, 2.656566, -558.883, 1, 0.3764706, 0, 1,
7.555555, 2.656566, -560.5685, 1, 0.2745098, 0, 1,
7.59596, 2.656566, -562.3003, 1, 0.2745098, 0, 1,
7.636364, 2.656566, -564.0784, 1, 0.2745098, 0, 1,
7.676768, 2.656566, -565.9026, 1, 0.2745098, 0, 1,
7.717172, 2.656566, -567.7733, 1, 0.1686275, 0, 1,
7.757576, 2.656566, -569.6901, 1, 0.1686275, 0, 1,
7.79798, 2.656566, -571.6532, 1, 0.1686275, 0, 1,
7.838384, 2.656566, -573.6626, 1, 0.06666667, 0, 1,
7.878788, 2.656566, -575.7182, 1, 0.06666667, 0, 1,
7.919192, 2.656566, -577.8201, 0.9647059, 0, 0.03137255, 1,
7.959596, 2.656566, -579.9683, 0.9647059, 0, 0.03137255, 1,
8, 2.656566, -582.1627, 0.9647059, 0, 0.03137255, 1,
4, 2.707071, -585.3372, 0.8588235, 0, 0.1372549, 1,
4.040404, 2.707071, -583.0843, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.707071, -580.8759, 0.9647059, 0, 0.03137255, 1,
4.121212, 2.707071, -578.7122, 0.9647059, 0, 0.03137255, 1,
4.161616, 2.707071, -576.5929, 1, 0.06666667, 0, 1,
4.20202, 2.707071, -574.5182, 1, 0.06666667, 0, 1,
4.242424, 2.707071, -572.488, 1, 0.06666667, 0, 1,
4.282828, 2.707071, -570.5024, 1, 0.1686275, 0, 1,
4.323232, 2.707071, -568.5613, 1, 0.1686275, 0, 1,
4.363636, 2.707071, -566.6649, 1, 0.1686275, 0, 1,
4.40404, 2.707071, -564.8129, 1, 0.2745098, 0, 1,
4.444445, 2.707071, -563.0056, 1, 0.2745098, 0, 1,
4.484848, 2.707071, -561.2427, 1, 0.2745098, 0, 1,
4.525252, 2.707071, -559.5244, 1, 0.3764706, 0, 1,
4.565657, 2.707071, -557.8507, 1, 0.3764706, 0, 1,
4.606061, 2.707071, -556.2216, 1, 0.3764706, 0, 1,
4.646465, 2.707071, -554.6369, 1, 0.3764706, 0, 1,
4.686869, 2.707071, -553.0969, 1, 0.4823529, 0, 1,
4.727273, 2.707071, -551.6013, 1, 0.4823529, 0, 1,
4.767677, 2.707071, -550.1504, 1, 0.4823529, 0, 1,
4.808081, 2.707071, -548.744, 1, 0.4823529, 0, 1,
4.848485, 2.707071, -547.3821, 1, 0.5843138, 0, 1,
4.888889, 2.707071, -546.0648, 1, 0.5843138, 0, 1,
4.929293, 2.707071, -544.7921, 1, 0.5843138, 0, 1,
4.969697, 2.707071, -543.5639, 1, 0.5843138, 0, 1,
5.010101, 2.707071, -542.3802, 1, 0.6862745, 0, 1,
5.050505, 2.707071, -541.2411, 1, 0.6862745, 0, 1,
5.090909, 2.707071, -540.1466, 1, 0.6862745, 0, 1,
5.131313, 2.707071, -539.0966, 1, 0.6862745, 0, 1,
5.171717, 2.707071, -538.0912, 1, 0.6862745, 0, 1,
5.212121, 2.707071, -537.1304, 1, 0.6862745, 0, 1,
5.252525, 2.707071, -536.2141, 1, 0.7921569, 0, 1,
5.292929, 2.707071, -535.3423, 1, 0.7921569, 0, 1,
5.333333, 2.707071, -534.5151, 1, 0.7921569, 0, 1,
5.373737, 2.707071, -533.7324, 1, 0.7921569, 0, 1,
5.414141, 2.707071, -532.9943, 1, 0.7921569, 0, 1,
5.454545, 2.707071, -532.3007, 1, 0.7921569, 0, 1,
5.494949, 2.707071, -531.6517, 1, 0.7921569, 0, 1,
5.535354, 2.707071, -531.0473, 1, 0.8941177, 0, 1,
5.575758, 2.707071, -530.4874, 1, 0.8941177, 0, 1,
5.616162, 2.707071, -529.972, 1, 0.8941177, 0, 1,
5.656566, 2.707071, -529.5013, 1, 0.8941177, 0, 1,
5.69697, 2.707071, -529.075, 1, 0.8941177, 0, 1,
5.737374, 2.707071, -528.6934, 1, 0.8941177, 0, 1,
5.777778, 2.707071, -528.3563, 1, 0.8941177, 0, 1,
5.818182, 2.707071, -528.0637, 1, 0.8941177, 0, 1,
5.858586, 2.707071, -527.8157, 1, 0.8941177, 0, 1,
5.89899, 2.707071, -527.6122, 1, 0.8941177, 0, 1,
5.939394, 2.707071, -527.4532, 1, 0.8941177, 0, 1,
5.979798, 2.707071, -527.3389, 1, 0.8941177, 0, 1,
6.020202, 2.707071, -527.2691, 1, 0.8941177, 0, 1,
6.060606, 2.707071, -527.2438, 1, 0.8941177, 0, 1,
6.10101, 2.707071, -527.2631, 1, 0.8941177, 0, 1,
6.141414, 2.707071, -527.327, 1, 0.8941177, 0, 1,
6.181818, 2.707071, -527.4354, 1, 0.8941177, 0, 1,
6.222222, 2.707071, -527.5884, 1, 0.8941177, 0, 1,
6.262626, 2.707071, -527.7859, 1, 0.8941177, 0, 1,
6.30303, 2.707071, -528.028, 1, 0.8941177, 0, 1,
6.343434, 2.707071, -528.3146, 1, 0.8941177, 0, 1,
6.383838, 2.707071, -528.6458, 1, 0.8941177, 0, 1,
6.424242, 2.707071, -529.0215, 1, 0.8941177, 0, 1,
6.464646, 2.707071, -529.4418, 1, 0.8941177, 0, 1,
6.505051, 2.707071, -529.9066, 1, 0.8941177, 0, 1,
6.545455, 2.707071, -530.416, 1, 0.8941177, 0, 1,
6.585859, 2.707071, -530.9699, 1, 0.8941177, 0, 1,
6.626263, 2.707071, -531.5684, 1, 0.7921569, 0, 1,
6.666667, 2.707071, -532.2115, 1, 0.7921569, 0, 1,
6.707071, 2.707071, -532.899, 1, 0.7921569, 0, 1,
6.747475, 2.707071, -533.6312, 1, 0.7921569, 0, 1,
6.787879, 2.707071, -534.408, 1, 0.7921569, 0, 1,
6.828283, 2.707071, -535.2292, 1, 0.7921569, 0, 1,
6.868687, 2.707071, -536.095, 1, 0.7921569, 0, 1,
6.909091, 2.707071, -537.0054, 1, 0.6862745, 0, 1,
6.949495, 2.707071, -537.9603, 1, 0.6862745, 0, 1,
6.989899, 2.707071, -538.9598, 1, 0.6862745, 0, 1,
7.030303, 2.707071, -540.0038, 1, 0.6862745, 0, 1,
7.070707, 2.707071, -541.0924, 1, 0.6862745, 0, 1,
7.111111, 2.707071, -542.2255, 1, 0.6862745, 0, 1,
7.151515, 2.707071, -543.4032, 1, 0.5843138, 0, 1,
7.191919, 2.707071, -544.6254, 1, 0.5843138, 0, 1,
7.232323, 2.707071, -545.8923, 1, 0.5843138, 0, 1,
7.272727, 2.707071, -547.2036, 1, 0.5843138, 0, 1,
7.313131, 2.707071, -548.5595, 1, 0.4823529, 0, 1,
7.353535, 2.707071, -549.96, 1, 0.4823529, 0, 1,
7.393939, 2.707071, -551.405, 1, 0.4823529, 0, 1,
7.434343, 2.707071, -552.8945, 1, 0.4823529, 0, 1,
7.474748, 2.707071, -554.4286, 1, 0.3764706, 0, 1,
7.515152, 2.707071, -556.0073, 1, 0.3764706, 0, 1,
7.555555, 2.707071, -557.6306, 1, 0.3764706, 0, 1,
7.59596, 2.707071, -559.2983, 1, 0.3764706, 0, 1,
7.636364, 2.707071, -561.0106, 1, 0.2745098, 0, 1,
7.676768, 2.707071, -562.7675, 1, 0.2745098, 0, 1,
7.717172, 2.707071, -564.569, 1, 0.2745098, 0, 1,
7.757576, 2.707071, -566.4149, 1, 0.1686275, 0, 1,
7.79798, 2.707071, -568.3055, 1, 0.1686275, 0, 1,
7.838384, 2.707071, -570.2406, 1, 0.1686275, 0, 1,
7.878788, 2.707071, -572.2202, 1, 0.06666667, 0, 1,
7.919192, 2.707071, -574.2444, 1, 0.06666667, 0, 1,
7.959596, 2.707071, -576.3132, 1, 0.06666667, 0, 1,
8, 2.707071, -578.4265, 0.9647059, 0, 0.03137255, 1,
4, 2.757576, -581.689, 0.9647059, 0, 0.03137255, 1,
4.040404, 2.757576, -579.5179, 0.9647059, 0, 0.03137255, 1,
4.080808, 2.757576, -577.3896, 1, 0.06666667, 0, 1,
4.121212, 2.757576, -575.3044, 1, 0.06666667, 0, 1,
4.161616, 2.757576, -573.262, 1, 0.06666667, 0, 1,
4.20202, 2.757576, -571.2626, 1, 0.1686275, 0, 1,
4.242424, 2.757576, -569.3062, 1, 0.1686275, 0, 1,
4.282828, 2.757576, -567.3926, 1, 0.1686275, 0, 1,
4.323232, 2.757576, -565.522, 1, 0.2745098, 0, 1,
4.363636, 2.757576, -563.6944, 1, 0.2745098, 0, 1,
4.40404, 2.757576, -561.9097, 1, 0.2745098, 0, 1,
4.444445, 2.757576, -560.1678, 1, 0.2745098, 0, 1,
4.484848, 2.757576, -558.469, 1, 0.3764706, 0, 1,
4.525252, 2.757576, -556.8131, 1, 0.3764706, 0, 1,
4.565657, 2.757576, -555.2001, 1, 0.3764706, 0, 1,
4.606061, 2.757576, -553.6301, 1, 0.4823529, 0, 1,
4.646465, 2.757576, -552.103, 1, 0.4823529, 0, 1,
4.686869, 2.757576, -550.6188, 1, 0.4823529, 0, 1,
4.727273, 2.757576, -549.1776, 1, 0.4823529, 0, 1,
4.767677, 2.757576, -547.7793, 1, 0.5843138, 0, 1,
4.808081, 2.757576, -546.4239, 1, 0.5843138, 0, 1,
4.848485, 2.757576, -545.1115, 1, 0.5843138, 0, 1,
4.888889, 2.757576, -543.842, 1, 0.5843138, 0, 1,
4.929293, 2.757576, -542.6155, 1, 0.6862745, 0, 1,
4.969697, 2.757576, -541.4318, 1, 0.6862745, 0, 1,
5.010101, 2.757576, -540.2911, 1, 0.6862745, 0, 1,
5.050505, 2.757576, -539.1934, 1, 0.6862745, 0, 1,
5.090909, 2.757576, -538.1386, 1, 0.6862745, 0, 1,
5.131313, 2.757576, -537.1267, 1, 0.6862745, 0, 1,
5.171717, 2.757576, -536.1578, 1, 0.7921569, 0, 1,
5.212121, 2.757576, -535.2318, 1, 0.7921569, 0, 1,
5.252525, 2.757576, -534.3488, 1, 0.7921569, 0, 1,
5.292929, 2.757576, -533.5086, 1, 0.7921569, 0, 1,
5.333333, 2.757576, -532.7114, 1, 0.7921569, 0, 1,
5.373737, 2.757576, -531.9572, 1, 0.7921569, 0, 1,
5.414141, 2.757576, -531.2458, 1, 0.7921569, 0, 1,
5.454545, 2.757576, -530.5775, 1, 0.8941177, 0, 1,
5.494949, 2.757576, -529.952, 1, 0.8941177, 0, 1,
5.535354, 2.757576, -529.3695, 1, 0.8941177, 0, 1,
5.575758, 2.757576, -528.83, 1, 0.8941177, 0, 1,
5.616162, 2.757576, -528.3333, 1, 0.8941177, 0, 1,
5.656566, 2.757576, -527.8796, 1, 0.8941177, 0, 1,
5.69697, 2.757576, -527.4689, 1, 0.8941177, 0, 1,
5.737374, 2.757576, -527.101, 1, 0.8941177, 0, 1,
5.777778, 2.757576, -526.7761, 1, 0.8941177, 0, 1,
5.818182, 2.757576, -526.4942, 1, 0.8941177, 0, 1,
5.858586, 2.757576, -526.2552, 1, 0.8941177, 0, 1,
5.89899, 2.757576, -526.0591, 1, 0.8941177, 0, 1,
5.939394, 2.757576, -525.9059, 1, 0.8941177, 0, 1,
5.979798, 2.757576, -525.7957, 1, 0.8941177, 0, 1,
6.020202, 2.757576, -525.7285, 1, 0.8941177, 0, 1,
6.060606, 2.757576, -525.7041, 1, 0.8941177, 0, 1,
6.10101, 2.757576, -525.7227, 1, 0.8941177, 0, 1,
6.141414, 2.757576, -525.7842, 1, 0.8941177, 0, 1,
6.181818, 2.757576, -525.8887, 1, 0.8941177, 0, 1,
6.222222, 2.757576, -526.0361, 1, 0.8941177, 0, 1,
6.262626, 2.757576, -526.2265, 1, 0.8941177, 0, 1,
6.30303, 2.757576, -526.4598, 1, 0.8941177, 0, 1,
6.343434, 2.757576, -526.736, 1, 0.8941177, 0, 1,
6.383838, 2.757576, -527.0552, 1, 0.8941177, 0, 1,
6.424242, 2.757576, -527.4172, 1, 0.8941177, 0, 1,
6.464646, 2.757576, -527.8223, 1, 0.8941177, 0, 1,
6.505051, 2.757576, -528.2703, 1, 0.8941177, 0, 1,
6.545455, 2.757576, -528.7611, 1, 0.8941177, 0, 1,
6.585859, 2.757576, -529.295, 1, 0.8941177, 0, 1,
6.626263, 2.757576, -529.8718, 1, 0.8941177, 0, 1,
6.666667, 2.757576, -530.4915, 1, 0.8941177, 0, 1,
6.707071, 2.757576, -531.1541, 1, 0.8941177, 0, 1,
6.747475, 2.757576, -531.8597, 1, 0.7921569, 0, 1,
6.787879, 2.757576, -532.6082, 1, 0.7921569, 0, 1,
6.828283, 2.757576, -533.3997, 1, 0.7921569, 0, 1,
6.868687, 2.757576, -534.2341, 1, 0.7921569, 0, 1,
6.909091, 2.757576, -535.1114, 1, 0.7921569, 0, 1,
6.949495, 2.757576, -536.0316, 1, 0.7921569, 0, 1,
6.989899, 2.757576, -536.9948, 1, 0.6862745, 0, 1,
7.030303, 2.757576, -538.001, 1, 0.6862745, 0, 1,
7.070707, 2.757576, -539.05, 1, 0.6862745, 0, 1,
7.111111, 2.757576, -540.142, 1, 0.6862745, 0, 1,
7.151515, 2.757576, -541.277, 1, 0.6862745, 0, 1,
7.191919, 2.757576, -542.4549, 1, 0.6862745, 0, 1,
7.232323, 2.757576, -543.6757, 1, 0.5843138, 0, 1,
7.272727, 2.757576, -544.9395, 1, 0.5843138, 0, 1,
7.313131, 2.757576, -546.2462, 1, 0.5843138, 0, 1,
7.353535, 2.757576, -547.5958, 1, 0.5843138, 0, 1,
7.393939, 2.757576, -548.9883, 1, 0.4823529, 0, 1,
7.434343, 2.757576, -550.4238, 1, 0.4823529, 0, 1,
7.474748, 2.757576, -551.9023, 1, 0.4823529, 0, 1,
7.515152, 2.757576, -553.4236, 1, 0.4823529, 0, 1,
7.555555, 2.757576, -554.9879, 1, 0.3764706, 0, 1,
7.59596, 2.757576, -556.5952, 1, 0.3764706, 0, 1,
7.636364, 2.757576, -558.2454, 1, 0.3764706, 0, 1,
7.676768, 2.757576, -559.9385, 1, 0.3764706, 0, 1,
7.717172, 2.757576, -561.6746, 1, 0.2745098, 0, 1,
7.757576, 2.757576, -563.4535, 1, 0.2745098, 0, 1,
7.79798, 2.757576, -565.2755, 1, 0.2745098, 0, 1,
7.838384, 2.757576, -567.1403, 1, 0.1686275, 0, 1,
7.878788, 2.757576, -569.0481, 1, 0.1686275, 0, 1,
7.919192, 2.757576, -570.9988, 1, 0.1686275, 0, 1,
7.959596, 2.757576, -572.9925, 1, 0.06666667, 0, 1,
8, 2.757576, -575.0291, 1, 0.06666667, 0, 1,
4, 2.808081, -578.3665, 0.9647059, 0, 0.03137255, 1,
4.040404, 2.808081, -576.2727, 1, 0.06666667, 0, 1,
4.080808, 2.808081, -574.2204, 1, 0.06666667, 0, 1,
4.121212, 2.808081, -572.2094, 1, 0.06666667, 0, 1,
4.161616, 2.808081, -570.2399, 1, 0.1686275, 0, 1,
4.20202, 2.808081, -568.3118, 1, 0.1686275, 0, 1,
4.242424, 2.808081, -566.425, 1, 0.1686275, 0, 1,
4.282828, 2.808081, -564.5797, 1, 0.2745098, 0, 1,
4.323232, 2.808081, -562.7758, 1, 0.2745098, 0, 1,
4.363636, 2.808081, -561.0133, 1, 0.2745098, 0, 1,
4.40404, 2.808081, -559.2922, 1, 0.3764706, 0, 1,
4.444445, 2.808081, -557.6125, 1, 0.3764706, 0, 1,
4.484848, 2.808081, -555.9742, 1, 0.3764706, 0, 1,
4.525252, 2.808081, -554.3773, 1, 0.3764706, 0, 1,
4.565657, 2.808081, -552.8218, 1, 0.4823529, 0, 1,
4.606061, 2.808081, -551.3078, 1, 0.4823529, 0, 1,
4.646465, 2.808081, -549.8351, 1, 0.4823529, 0, 1,
4.686869, 2.808081, -548.4039, 1, 0.5843138, 0, 1,
4.727273, 2.808081, -547.014, 1, 0.5843138, 0, 1,
4.767677, 2.808081, -545.6655, 1, 0.5843138, 0, 1,
4.808081, 2.808081, -544.3585, 1, 0.5843138, 0, 1,
4.848485, 2.808081, -543.0928, 1, 0.5843138, 0, 1,
4.888889, 2.808081, -541.8686, 1, 0.6862745, 0, 1,
4.929293, 2.808081, -540.6858, 1, 0.6862745, 0, 1,
4.969697, 2.808081, -539.5444, 1, 0.6862745, 0, 1,
5.010101, 2.808081, -538.4443, 1, 0.6862745, 0, 1,
5.050505, 2.808081, -537.3857, 1, 0.6862745, 0, 1,
5.090909, 2.808081, -536.3685, 1, 0.7921569, 0, 1,
5.131313, 2.808081, -535.3927, 1, 0.7921569, 0, 1,
5.171717, 2.808081, -534.4583, 1, 0.7921569, 0, 1,
5.212121, 2.808081, -533.5654, 1, 0.7921569, 0, 1,
5.252525, 2.808081, -532.7137, 1, 0.7921569, 0, 1,
5.292929, 2.808081, -531.9036, 1, 0.7921569, 0, 1,
5.333333, 2.808081, -531.1348, 1, 0.8941177, 0, 1,
5.373737, 2.808081, -530.4075, 1, 0.8941177, 0, 1,
5.414141, 2.808081, -529.7215, 1, 0.8941177, 0, 1,
5.454545, 2.808081, -529.077, 1, 0.8941177, 0, 1,
5.494949, 2.808081, -528.4738, 1, 0.8941177, 0, 1,
5.535354, 2.808081, -527.912, 1, 0.8941177, 0, 1,
5.575758, 2.808081, -527.3917, 1, 0.8941177, 0, 1,
5.616162, 2.808081, -526.9128, 1, 0.8941177, 0, 1,
5.656566, 2.808081, -526.4753, 1, 0.8941177, 0, 1,
5.69697, 2.808081, -526.0792, 1, 0.8941177, 0, 1,
5.737374, 2.808081, -525.7244, 1, 0.8941177, 0, 1,
5.777778, 2.808081, -525.4111, 1, 0.8941177, 0, 1,
5.818182, 2.808081, -525.1392, 1, 1, 0, 1,
5.858586, 2.808081, -524.9087, 1, 1, 0, 1,
5.89899, 2.808081, -524.7196, 1, 1, 0, 1,
5.939394, 2.808081, -524.572, 1, 1, 0, 1,
5.979798, 2.808081, -524.4656, 1, 1, 0, 1,
6.020202, 2.808081, -524.4008, 1, 1, 0, 1,
6.060606, 2.808081, -524.3773, 1, 1, 0, 1,
6.10101, 2.808081, -524.3953, 1, 1, 0, 1,
6.141414, 2.808081, -524.4546, 1, 1, 0, 1,
6.181818, 2.808081, -524.5554, 1, 1, 0, 1,
6.222222, 2.808081, -524.6975, 1, 1, 0, 1,
6.262626, 2.808081, -524.881, 1, 1, 0, 1,
6.30303, 2.808081, -525.106, 1, 1, 0, 1,
6.343434, 2.808081, -525.3724, 1, 1, 0, 1,
6.383838, 2.808081, -525.6802, 1, 0.8941177, 0, 1,
6.424242, 2.808081, -526.0294, 1, 0.8941177, 0, 1,
6.464646, 2.808081, -526.4199, 1, 0.8941177, 0, 1,
6.505051, 2.808081, -526.8519, 1, 0.8941177, 0, 1,
6.545455, 2.808081, -527.3253, 1, 0.8941177, 0, 1,
6.585859, 2.808081, -527.8401, 1, 0.8941177, 0, 1,
6.626263, 2.808081, -528.3964, 1, 0.8941177, 0, 1,
6.666667, 2.808081, -528.994, 1, 0.8941177, 0, 1,
6.707071, 2.808081, -529.633, 1, 0.8941177, 0, 1,
6.747475, 2.808081, -530.3134, 1, 0.8941177, 0, 1,
6.787879, 2.808081, -531.0353, 1, 0.8941177, 0, 1,
6.828283, 2.808081, -531.7985, 1, 0.7921569, 0, 1,
6.868687, 2.808081, -532.6031, 1, 0.7921569, 0, 1,
6.909091, 2.808081, -533.4492, 1, 0.7921569, 0, 1,
6.949495, 2.808081, -534.3367, 1, 0.7921569, 0, 1,
6.989899, 2.808081, -535.2655, 1, 0.7921569, 0, 1,
7.030303, 2.808081, -536.2358, 1, 0.7921569, 0, 1,
7.070707, 2.808081, -537.2475, 1, 0.6862745, 0, 1,
7.111111, 2.808081, -538.3005, 1, 0.6862745, 0, 1,
7.151515, 2.808081, -539.395, 1, 0.6862745, 0, 1,
7.191919, 2.808081, -540.5309, 1, 0.6862745, 0, 1,
7.232323, 2.808081, -541.7083, 1, 0.6862745, 0, 1,
7.272727, 2.808081, -542.9269, 1, 0.5843138, 0, 1,
7.313131, 2.808081, -544.1871, 1, 0.5843138, 0, 1,
7.353535, 2.808081, -545.4886, 1, 0.5843138, 0, 1,
7.393939, 2.808081, -546.8315, 1, 0.5843138, 0, 1,
7.434343, 2.808081, -548.2158, 1, 0.5843138, 0, 1,
7.474748, 2.808081, -549.6415, 1, 0.4823529, 0, 1,
7.515152, 2.808081, -551.1087, 1, 0.4823529, 0, 1,
7.555555, 2.808081, -552.6172, 1, 0.4823529, 0, 1,
7.59596, 2.808081, -554.1672, 1, 0.4823529, 0, 1,
7.636364, 2.808081, -555.7585, 1, 0.3764706, 0, 1,
7.676768, 2.808081, -557.3913, 1, 0.3764706, 0, 1,
7.717172, 2.808081, -559.0654, 1, 0.3764706, 0, 1,
7.757576, 2.808081, -560.781, 1, 0.2745098, 0, 1,
7.79798, 2.808081, -562.538, 1, 0.2745098, 0, 1,
7.838384, 2.808081, -564.3364, 1, 0.2745098, 0, 1,
7.878788, 2.808081, -566.1762, 1, 0.1686275, 0, 1,
7.919192, 2.808081, -568.0574, 1, 0.1686275, 0, 1,
7.959596, 2.808081, -569.98, 1, 0.1686275, 0, 1,
8, 2.808081, -571.944, 1, 0.06666667, 0, 1,
4, 2.858586, -575.3445, 1, 0.06666667, 0, 1,
4.040404, 2.858586, -573.324, 1, 0.06666667, 0, 1,
4.080808, 2.858586, -571.3436, 1, 0.1686275, 0, 1,
4.121212, 2.858586, -569.4031, 1, 0.1686275, 0, 1,
4.161616, 2.858586, -567.5025, 1, 0.1686275, 0, 1,
4.20202, 2.858586, -565.6419, 1, 0.2745098, 0, 1,
4.242424, 2.858586, -563.8213, 1, 0.2745098, 0, 1,
4.282828, 2.858586, -562.0406, 1, 0.2745098, 0, 1,
4.323232, 2.858586, -560.2998, 1, 0.2745098, 0, 1,
4.363636, 2.858586, -558.5991, 1, 0.3764706, 0, 1,
4.40404, 2.858586, -556.9382, 1, 0.3764706, 0, 1,
4.444445, 2.858586, -555.3174, 1, 0.3764706, 0, 1,
4.484848, 2.858586, -553.7365, 1, 0.4823529, 0, 1,
4.525252, 2.858586, -552.1955, 1, 0.4823529, 0, 1,
4.565657, 2.858586, -550.6945, 1, 0.4823529, 0, 1,
4.606061, 2.858586, -549.2335, 1, 0.4823529, 0, 1,
4.646465, 2.858586, -547.8124, 1, 0.5843138, 0, 1,
4.686869, 2.858586, -546.4312, 1, 0.5843138, 0, 1,
4.727273, 2.858586, -545.09, 1, 0.5843138, 0, 1,
4.767677, 2.858586, -543.7888, 1, 0.5843138, 0, 1,
4.808081, 2.858586, -542.5275, 1, 0.6862745, 0, 1,
4.848485, 2.858586, -541.3062, 1, 0.6862745, 0, 1,
4.888889, 2.858586, -540.1249, 1, 0.6862745, 0, 1,
4.929293, 2.858586, -538.9835, 1, 0.6862745, 0, 1,
4.969697, 2.858586, -537.882, 1, 0.6862745, 0, 1,
5.010101, 2.858586, -536.8206, 1, 0.7921569, 0, 1,
5.050505, 2.858586, -535.799, 1, 0.7921569, 0, 1,
5.090909, 2.858586, -534.8174, 1, 0.7921569, 0, 1,
5.131313, 2.858586, -533.8758, 1, 0.7921569, 0, 1,
5.171717, 2.858586, -532.9741, 1, 0.7921569, 0, 1,
5.212121, 2.858586, -532.1124, 1, 0.7921569, 0, 1,
5.252525, 2.858586, -531.2906, 1, 0.7921569, 0, 1,
5.292929, 2.858586, -530.5089, 1, 0.8941177, 0, 1,
5.333333, 2.858586, -529.767, 1, 0.8941177, 0, 1,
5.373737, 2.858586, -529.0651, 1, 0.8941177, 0, 1,
5.414141, 2.858586, -528.4032, 1, 0.8941177, 0, 1,
5.454545, 2.858586, -527.7812, 1, 0.8941177, 0, 1,
5.494949, 2.858586, -527.1992, 1, 0.8941177, 0, 1,
5.535354, 2.858586, -526.6571, 1, 0.8941177, 0, 1,
5.575758, 2.858586, -526.155, 1, 0.8941177, 0, 1,
5.616162, 2.858586, -525.6929, 1, 0.8941177, 0, 1,
5.656566, 2.858586, -525.2706, 1, 1, 0, 1,
5.69697, 2.858586, -524.8884, 1, 1, 0, 1,
5.737374, 2.858586, -524.5461, 1, 1, 0, 1,
5.777778, 2.858586, -524.2438, 1, 1, 0, 1,
5.818182, 2.858586, -523.9814, 1, 1, 0, 1,
5.858586, 2.858586, -523.759, 1, 1, 0, 1,
5.89899, 2.858586, -523.5765, 1, 1, 0, 1,
5.939394, 2.858586, -523.434, 1, 1, 0, 1,
5.979798, 2.858586, -523.3314, 1, 1, 0, 1,
6.020202, 2.858586, -523.2689, 1, 1, 0, 1,
6.060606, 2.858586, -523.2462, 1, 1, 0, 1,
6.10101, 2.858586, -523.2635, 1, 1, 0, 1,
6.141414, 2.858586, -523.3207, 1, 1, 0, 1,
6.181818, 2.858586, -523.418, 1, 1, 0, 1,
6.222222, 2.858586, -523.5552, 1, 1, 0, 1,
6.262626, 2.858586, -523.7323, 1, 1, 0, 1,
6.30303, 2.858586, -523.9494, 1, 1, 0, 1,
6.343434, 2.858586, -524.2064, 1, 1, 0, 1,
6.383838, 2.858586, -524.5034, 1, 1, 0, 1,
6.424242, 2.858586, -524.8404, 1, 1, 0, 1,
6.464646, 2.858586, -525.2173, 1, 1, 0, 1,
6.505051, 2.858586, -525.6342, 1, 0.8941177, 0, 1,
6.545455, 2.858586, -526.0909, 1, 0.8941177, 0, 1,
6.585859, 2.858586, -526.5878, 1, 0.8941177, 0, 1,
6.626263, 2.858586, -527.1245, 1, 0.8941177, 0, 1,
6.666667, 2.858586, -527.7012, 1, 0.8941177, 0, 1,
6.707071, 2.858586, -528.3178, 1, 0.8941177, 0, 1,
6.747475, 2.858586, -528.9744, 1, 0.8941177, 0, 1,
6.787879, 2.858586, -529.671, 1, 0.8941177, 0, 1,
6.828283, 2.858586, -530.4075, 1, 0.8941177, 0, 1,
6.868687, 2.858586, -531.184, 1, 0.8941177, 0, 1,
6.909091, 2.858586, -532.0004, 1, 0.7921569, 0, 1,
6.949495, 2.858586, -532.8568, 1, 0.7921569, 0, 1,
6.989899, 2.858586, -533.7531, 1, 0.7921569, 0, 1,
7.030303, 2.858586, -534.6893, 1, 0.7921569, 0, 1,
7.070707, 2.858586, -535.6656, 1, 0.7921569, 0, 1,
7.111111, 2.858586, -536.6818, 1, 0.7921569, 0, 1,
7.151515, 2.858586, -537.7379, 1, 0.6862745, 0, 1,
7.191919, 2.858586, -538.834, 1, 0.6862745, 0, 1,
7.232323, 2.858586, -539.9701, 1, 0.6862745, 0, 1,
7.272727, 2.858586, -541.1461, 1, 0.6862745, 0, 1,
7.313131, 2.858586, -542.3621, 1, 0.6862745, 0, 1,
7.353535, 2.858586, -543.618, 1, 0.5843138, 0, 1,
7.393939, 2.858586, -544.9139, 1, 0.5843138, 0, 1,
7.434343, 2.858586, -546.2498, 1, 0.5843138, 0, 1,
7.474748, 2.858586, -547.6255, 1, 0.5843138, 0, 1,
7.515152, 2.858586, -549.0413, 1, 0.4823529, 0, 1,
7.555555, 2.858586, -550.497, 1, 0.4823529, 0, 1,
7.59596, 2.858586, -551.9927, 1, 0.4823529, 0, 1,
7.636364, 2.858586, -553.5283, 1, 0.4823529, 0, 1,
7.676768, 2.858586, -555.1039, 1, 0.3764706, 0, 1,
7.717172, 2.858586, -556.7194, 1, 0.3764706, 0, 1,
7.757576, 2.858586, -558.3749, 1, 0.3764706, 0, 1,
7.79798, 2.858586, -560.0704, 1, 0.3764706, 0, 1,
7.838384, 2.858586, -561.8057, 1, 0.2745098, 0, 1,
7.878788, 2.858586, -563.5811, 1, 0.2745098, 0, 1,
7.919192, 2.858586, -565.3964, 1, 0.2745098, 0, 1,
7.959596, 2.858586, -567.2517, 1, 0.1686275, 0, 1,
8, 2.858586, -569.1469, 1, 0.1686275, 0, 1,
4, 2.909091, -572.6001, 1, 0.06666667, 0, 1,
4.040404, 2.909091, -570.6492, 1, 0.1686275, 0, 1,
4.080808, 2.909091, -568.7369, 1, 0.1686275, 0, 1,
4.121212, 2.909091, -566.8632, 1, 0.1686275, 0, 1,
4.161616, 2.909091, -565.0281, 1, 0.2745098, 0, 1,
4.20202, 2.909091, -563.2315, 1, 0.2745098, 0, 1,
4.242424, 2.909091, -561.4735, 1, 0.2745098, 0, 1,
4.282828, 2.909091, -559.7542, 1, 0.3764706, 0, 1,
4.323232, 2.909091, -558.0733, 1, 0.3764706, 0, 1,
4.363636, 2.909091, -556.4311, 1, 0.3764706, 0, 1,
4.40404, 2.909091, -554.8275, 1, 0.3764706, 0, 1,
4.444445, 2.909091, -553.2623, 1, 0.4823529, 0, 1,
4.484848, 2.909091, -551.7358, 1, 0.4823529, 0, 1,
4.525252, 2.909091, -550.2479, 1, 0.4823529, 0, 1,
4.565657, 2.909091, -548.7986, 1, 0.4823529, 0, 1,
4.606061, 2.909091, -547.3878, 1, 0.5843138, 0, 1,
4.646465, 2.909091, -546.0157, 1, 0.5843138, 0, 1,
4.686869, 2.909091, -544.6821, 1, 0.5843138, 0, 1,
4.727273, 2.909091, -543.3871, 1, 0.5843138, 0, 1,
4.767677, 2.909091, -542.1306, 1, 0.6862745, 0, 1,
4.808081, 2.909091, -540.9128, 1, 0.6862745, 0, 1,
4.848485, 2.909091, -539.7335, 1, 0.6862745, 0, 1,
4.888889, 2.909091, -538.5928, 1, 0.6862745, 0, 1,
4.929293, 2.909091, -537.4907, 1, 0.6862745, 0, 1,
4.969697, 2.909091, -536.4272, 1, 0.7921569, 0, 1,
5.010101, 2.909091, -535.4022, 1, 0.7921569, 0, 1,
5.050505, 2.909091, -534.4158, 1, 0.7921569, 0, 1,
5.090909, 2.909091, -533.468, 1, 0.7921569, 0, 1,
5.131313, 2.909091, -532.5588, 1, 0.7921569, 0, 1,
5.171717, 2.909091, -531.6882, 1, 0.7921569, 0, 1,
5.212121, 2.909091, -530.8561, 1, 0.8941177, 0, 1,
5.252525, 2.909091, -530.0627, 1, 0.8941177, 0, 1,
5.292929, 2.909091, -529.3078, 1, 0.8941177, 0, 1,
5.333333, 2.909091, -528.5915, 1, 0.8941177, 0, 1,
5.373737, 2.909091, -527.9138, 1, 0.8941177, 0, 1,
5.414141, 2.909091, -527.2746, 1, 0.8941177, 0, 1,
5.454545, 2.909091, -526.674, 1, 0.8941177, 0, 1,
5.494949, 2.909091, -526.112, 1, 0.8941177, 0, 1,
5.535354, 2.909091, -525.5886, 1, 0.8941177, 0, 1,
5.575758, 2.909091, -525.1038, 1, 1, 0, 1,
5.616162, 2.909091, -524.6575, 1, 1, 0, 1,
5.656566, 2.909091, -524.2499, 1, 1, 0, 1,
5.69697, 2.909091, -523.8808, 1, 1, 0, 1,
5.737374, 2.909091, -523.5503, 1, 1, 0, 1,
5.777778, 2.909091, -523.2584, 1, 1, 0, 1,
5.818182, 2.909091, -523.005, 1, 1, 0, 1,
5.858586, 2.909091, -522.7902, 1, 1, 0, 1,
5.89899, 2.909091, -522.614, 1, 1, 0, 1,
5.939394, 2.909091, -522.4764, 1, 1, 0, 1,
5.979798, 2.909091, -522.3774, 1, 1, 0, 1,
6.020202, 2.909091, -522.317, 1, 1, 0, 1,
6.060606, 2.909091, -522.2951, 1, 1, 0, 1,
6.10101, 2.909091, -522.3118, 1, 1, 0, 1,
6.141414, 2.909091, -522.3671, 1, 1, 0, 1,
6.181818, 2.909091, -522.461, 1, 1, 0, 1,
6.222222, 2.909091, -522.5934, 1, 1, 0, 1,
6.262626, 2.909091, -522.7645, 1, 1, 0, 1,
6.30303, 2.909091, -522.9741, 1, 1, 0, 1,
6.343434, 2.909091, -523.2223, 1, 1, 0, 1,
6.383838, 2.909091, -523.509, 1, 1, 0, 1,
6.424242, 2.909091, -523.8344, 1, 1, 0, 1,
6.464646, 2.909091, -524.1984, 1, 1, 0, 1,
6.505051, 2.909091, -524.6008, 1, 1, 0, 1,
6.545455, 2.909091, -525.0419, 1, 1, 0, 1,
6.585859, 2.909091, -525.5216, 1, 0.8941177, 0, 1,
6.626263, 2.909091, -526.0399, 1, 0.8941177, 0, 1,
6.666667, 2.909091, -526.5967, 1, 0.8941177, 0, 1,
6.707071, 2.909091, -527.1921, 1, 0.8941177, 0, 1,
6.747475, 2.909091, -527.8261, 1, 0.8941177, 0, 1,
6.787879, 2.909091, -528.4987, 1, 0.8941177, 0, 1,
6.828283, 2.909091, -529.2099, 1, 0.8941177, 0, 1,
6.868687, 2.909091, -529.9596, 1, 0.8941177, 0, 1,
6.909091, 2.909091, -530.7479, 1, 0.8941177, 0, 1,
6.949495, 2.909091, -531.5748, 1, 0.7921569, 0, 1,
6.989899, 2.909091, -532.4403, 1, 0.7921569, 0, 1,
7.030303, 2.909091, -533.3444, 1, 0.7921569, 0, 1,
7.070707, 2.909091, -534.287, 1, 0.7921569, 0, 1,
7.111111, 2.909091, -535.2682, 1, 0.7921569, 0, 1,
7.151515, 2.909091, -536.288, 1, 0.7921569, 0, 1,
7.191919, 2.909091, -537.3464, 1, 0.6862745, 0, 1,
7.232323, 2.909091, -538.4434, 1, 0.6862745, 0, 1,
7.272727, 2.909091, -539.5789, 1, 0.6862745, 0, 1,
7.313131, 2.909091, -540.7531, 1, 0.6862745, 0, 1,
7.353535, 2.909091, -541.9658, 1, 0.6862745, 0, 1,
7.393939, 2.909091, -543.217, 1, 0.5843138, 0, 1,
7.434343, 2.909091, -544.5069, 1, 0.5843138, 0, 1,
7.474748, 2.909091, -545.8353, 1, 0.5843138, 0, 1,
7.515152, 2.909091, -547.2023, 1, 0.5843138, 0, 1,
7.555555, 2.909091, -548.6079, 1, 0.4823529, 0, 1,
7.59596, 2.909091, -550.0521, 1, 0.4823529, 0, 1,
7.636364, 2.909091, -551.5349, 1, 0.4823529, 0, 1,
7.676768, 2.909091, -553.0562, 1, 0.4823529, 0, 1,
7.717172, 2.909091, -554.6161, 1, 0.3764706, 0, 1,
7.757576, 2.909091, -556.2147, 1, 0.3764706, 0, 1,
7.79798, 2.909091, -557.8517, 1, 0.3764706, 0, 1,
7.838384, 2.909091, -559.5274, 1, 0.3764706, 0, 1,
7.878788, 2.909091, -561.2416, 1, 0.2745098, 0, 1,
7.919192, 2.909091, -562.9944, 1, 0.2745098, 0, 1,
7.959596, 2.909091, -564.7859, 1, 0.2745098, 0, 1,
8, 2.909091, -566.6158, 1, 0.1686275, 0, 1,
4, 2.959596, -570.1125, 1, 0.1686275, 0, 1,
4.040404, 2.959596, -568.2277, 1, 0.1686275, 0, 1,
4.080808, 2.959596, -566.3801, 1, 0.1686275, 0, 1,
4.121212, 2.959596, -564.5698, 1, 0.2745098, 0, 1,
4.161616, 2.959596, -562.7967, 1, 0.2745098, 0, 1,
4.20202, 2.959596, -561.061, 1, 0.2745098, 0, 1,
4.242424, 2.959596, -559.3625, 1, 0.3764706, 0, 1,
4.282828, 2.959596, -557.7012, 1, 0.3764706, 0, 1,
4.323232, 2.959596, -556.0773, 1, 0.3764706, 0, 1,
4.363636, 2.959596, -554.4907, 1, 0.3764706, 0, 1,
4.40404, 2.959596, -552.9412, 1, 0.4823529, 0, 1,
4.444445, 2.959596, -551.4291, 1, 0.4823529, 0, 1,
4.484848, 2.959596, -549.9543, 1, 0.4823529, 0, 1,
4.525252, 2.959596, -548.5167, 1, 0.5843138, 0, 1,
4.565657, 2.959596, -547.1165, 1, 0.5843138, 0, 1,
4.606061, 2.959596, -545.7534, 1, 0.5843138, 0, 1,
4.646465, 2.959596, -544.4277, 1, 0.5843138, 0, 1,
4.686869, 2.959596, -543.1392, 1, 0.5843138, 0, 1,
4.727273, 2.959596, -541.888, 1, 0.6862745, 0, 1,
4.767677, 2.959596, -540.6741, 1, 0.6862745, 0, 1,
4.808081, 2.959596, -539.4974, 1, 0.6862745, 0, 1,
4.848485, 2.959596, -538.3581, 1, 0.6862745, 0, 1,
4.888889, 2.959596, -537.256, 1, 0.6862745, 0, 1,
4.929293, 2.959596, -536.1912, 1, 0.7921569, 0, 1,
4.969697, 2.959596, -535.1636, 1, 0.7921569, 0, 1,
5.010101, 2.959596, -534.1733, 1, 0.7921569, 0, 1,
5.050505, 2.959596, -533.2203, 1, 0.7921569, 0, 1,
5.090909, 2.959596, -532.3046, 1, 0.7921569, 0, 1,
5.131313, 2.959596, -531.4262, 1, 0.7921569, 0, 1,
5.171717, 2.959596, -530.585, 1, 0.8941177, 0, 1,
5.212121, 2.959596, -529.7811, 1, 0.8941177, 0, 1,
5.252525, 2.959596, -529.0145, 1, 0.8941177, 0, 1,
5.292929, 2.959596, -528.2852, 1, 0.8941177, 0, 1,
5.333333, 2.959596, -527.5931, 1, 0.8941177, 0, 1,
5.373737, 2.959596, -526.9383, 1, 0.8941177, 0, 1,
5.414141, 2.959596, -526.3207, 1, 0.8941177, 0, 1,
5.454545, 2.959596, -525.7405, 1, 0.8941177, 0, 1,
5.494949, 2.959596, -525.1975, 1, 1, 0, 1,
5.535354, 2.959596, -524.6918, 1, 1, 0, 1,
5.575758, 2.959596, -524.2234, 1, 1, 0, 1,
5.616162, 2.959596, -523.7922, 1, 1, 0, 1,
5.656566, 2.959596, -523.3984, 1, 1, 0, 1,
5.69697, 2.959596, -523.0418, 1, 1, 0, 1,
5.737374, 2.959596, -522.7225, 1, 1, 0, 1,
5.777778, 2.959596, -522.4404, 1, 1, 0, 1,
5.818182, 2.959596, -522.1956, 1, 1, 0, 1,
5.858586, 2.959596, -521.9882, 1, 1, 0, 1,
5.89899, 2.959596, -521.8179, 1, 1, 0, 1,
5.939394, 2.959596, -521.6849, 1, 1, 0, 1,
5.979798, 2.959596, -521.5893, 1, 1, 0, 1,
6.020202, 2.959596, -521.5309, 1, 1, 0, 1,
6.060606, 2.959596, -521.5098, 1, 1, 0, 1,
6.10101, 2.959596, -521.5259, 1, 1, 0, 1,
6.141414, 2.959596, -521.5793, 1, 1, 0, 1,
6.181818, 2.959596, -521.67, 1, 1, 0, 1,
6.222222, 2.959596, -521.798, 1, 1, 0, 1,
6.262626, 2.959596, -521.9633, 1, 1, 0, 1,
6.30303, 2.959596, -522.1658, 1, 1, 0, 1,
6.343434, 2.959596, -522.4056, 1, 1, 0, 1,
6.383838, 2.959596, -522.6826, 1, 1, 0, 1,
6.424242, 2.959596, -522.997, 1, 1, 0, 1,
6.464646, 2.959596, -523.3486, 1, 1, 0, 1,
6.505051, 2.959596, -523.7375, 1, 1, 0, 1,
6.545455, 2.959596, -524.1637, 1, 1, 0, 1,
6.585859, 2.959596, -524.6271, 1, 1, 0, 1,
6.626263, 2.959596, -525.1278, 1, 1, 0, 1,
6.666667, 2.959596, -525.6658, 1, 0.8941177, 0, 1,
6.707071, 2.959596, -526.2411, 1, 0.8941177, 0, 1,
6.747475, 2.959596, -526.8536, 1, 0.8941177, 0, 1,
6.787879, 2.959596, -527.5035, 1, 0.8941177, 0, 1,
6.828283, 2.959596, -528.1906, 1, 0.8941177, 0, 1,
6.868687, 2.959596, -528.9149, 1, 0.8941177, 0, 1,
6.909091, 2.959596, -529.6766, 1, 0.8941177, 0, 1,
6.949495, 2.959596, -530.4755, 1, 0.8941177, 0, 1,
6.989899, 2.959596, -531.3117, 1, 0.7921569, 0, 1,
7.030303, 2.959596, -532.1851, 1, 0.7921569, 0, 1,
7.070707, 2.959596, -533.0959, 1, 0.7921569, 0, 1,
7.111111, 2.959596, -534.0439, 1, 0.7921569, 0, 1,
7.151515, 2.959596, -535.0292, 1, 0.7921569, 0, 1,
7.191919, 2.959596, -536.0518, 1, 0.7921569, 0, 1,
7.232323, 2.959596, -537.1116, 1, 0.6862745, 0, 1,
7.272727, 2.959596, -538.2087, 1, 0.6862745, 0, 1,
7.313131, 2.959596, -539.3431, 1, 0.6862745, 0, 1,
7.353535, 2.959596, -540.5148, 1, 0.6862745, 0, 1,
7.393939, 2.959596, -541.7237, 1, 0.6862745, 0, 1,
7.434343, 2.959596, -542.9699, 1, 0.5843138, 0, 1,
7.474748, 2.959596, -544.2534, 1, 0.5843138, 0, 1,
7.515152, 2.959596, -545.5742, 1, 0.5843138, 0, 1,
7.555555, 2.959596, -546.9323, 1, 0.5843138, 0, 1,
7.59596, 2.959596, -548.3275, 1, 0.5843138, 0, 1,
7.636364, 2.959596, -549.7601, 1, 0.4823529, 0, 1,
7.676768, 2.959596, -551.23, 1, 0.4823529, 0, 1,
7.717172, 2.959596, -552.7371, 1, 0.4823529, 0, 1,
7.757576, 2.959596, -554.2816, 1, 0.4823529, 0, 1,
7.79798, 2.959596, -555.8632, 1, 0.3764706, 0, 1,
7.838384, 2.959596, -557.4822, 1, 0.3764706, 0, 1,
7.878788, 2.959596, -559.1384, 1, 0.3764706, 0, 1,
7.919192, 2.959596, -560.8319, 1, 0.2745098, 0, 1,
7.959596, 2.959596, -562.5627, 1, 0.2745098, 0, 1,
8, 2.959596, -564.3308, 1, 0.2745098, 0, 1,
4, 3.010101, -567.8627, 1, 0.1686275, 0, 1,
4.040404, 3.010101, -566.0405, 1, 0.1686275, 0, 1,
4.080808, 3.010101, -564.2545, 1, 0.2745098, 0, 1,
4.121212, 3.010101, -562.5043, 1, 0.2745098, 0, 1,
4.161616, 3.010101, -560.7903, 1, 0.2745098, 0, 1,
4.20202, 3.010101, -559.1123, 1, 0.3764706, 0, 1,
4.242424, 3.010101, -557.4703, 1, 0.3764706, 0, 1,
4.282828, 3.010101, -555.8644, 1, 0.3764706, 0, 1,
4.323232, 3.010101, -554.2945, 1, 0.4823529, 0, 1,
4.363636, 3.010101, -552.7606, 1, 0.4823529, 0, 1,
4.40404, 3.010101, -551.2628, 1, 0.4823529, 0, 1,
4.444445, 3.010101, -549.801, 1, 0.4823529, 0, 1,
4.484848, 3.010101, -548.3752, 1, 0.5843138, 0, 1,
4.525252, 3.010101, -546.9855, 1, 0.5843138, 0, 1,
4.565657, 3.010101, -545.6318, 1, 0.5843138, 0, 1,
4.606061, 3.010101, -544.3141, 1, 0.5843138, 0, 1,
4.646465, 3.010101, -543.0325, 1, 0.5843138, 0, 1,
4.686869, 3.010101, -541.7869, 1, 0.6862745, 0, 1,
4.727273, 3.010101, -540.5773, 1, 0.6862745, 0, 1,
4.767677, 3.010101, -539.4038, 1, 0.6862745, 0, 1,
4.808081, 3.010101, -538.2664, 1, 0.6862745, 0, 1,
4.848485, 3.010101, -537.1649, 1, 0.6862745, 0, 1,
4.888889, 3.010101, -536.0995, 1, 0.7921569, 0, 1,
4.929293, 3.010101, -535.0701, 1, 0.7921569, 0, 1,
4.969697, 3.010101, -534.0767, 1, 0.7921569, 0, 1,
5.010101, 3.010101, -533.1194, 1, 0.7921569, 0, 1,
5.050505, 3.010101, -532.1981, 1, 0.7921569, 0, 1,
5.090909, 3.010101, -531.3129, 1, 0.7921569, 0, 1,
5.131313, 3.010101, -530.4636, 1, 0.8941177, 0, 1,
5.171717, 3.010101, -529.6505, 1, 0.8941177, 0, 1,
5.212121, 3.010101, -528.8734, 1, 0.8941177, 0, 1,
5.252525, 3.010101, -528.1322, 1, 0.8941177, 0, 1,
5.292929, 3.010101, -527.4271, 1, 0.8941177, 0, 1,
5.333333, 3.010101, -526.7581, 1, 0.8941177, 0, 1,
5.373737, 3.010101, -526.1251, 1, 0.8941177, 0, 1,
5.414141, 3.010101, -525.5281, 1, 0.8941177, 0, 1,
5.454545, 3.010101, -524.9672, 1, 1, 0, 1,
5.494949, 3.010101, -524.4423, 1, 1, 0, 1,
5.535354, 3.010101, -523.9534, 1, 1, 0, 1,
5.575758, 3.010101, -523.5005, 1, 1, 0, 1,
5.616162, 3.010101, -523.0837, 1, 1, 0, 1,
5.656566, 3.010101, -522.703, 1, 1, 0, 1,
5.69697, 3.010101, -522.3583, 1, 1, 0, 1,
5.737374, 3.010101, -522.0496, 1, 1, 0, 1,
5.777778, 3.010101, -521.7769, 1, 1, 0, 1,
5.818182, 3.010101, -521.5403, 1, 1, 0, 1,
5.858586, 3.010101, -521.3397, 1, 1, 0, 1,
5.89899, 3.010101, -521.1751, 1, 1, 0, 1,
5.939394, 3.010101, -521.0466, 1, 1, 0, 1,
5.979798, 3.010101, -520.9541, 1, 1, 0, 1,
6.020202, 3.010101, -520.8976, 1, 1, 0, 1,
6.060606, 3.010101, -520.8772, 1, 1, 0, 1,
6.10101, 3.010101, -520.8928, 1, 1, 0, 1,
6.141414, 3.010101, -520.9445, 1, 1, 0, 1,
6.181818, 3.010101, -521.0321, 1, 1, 0, 1,
6.222222, 3.010101, -521.1558, 1, 1, 0, 1,
6.262626, 3.010101, -521.3156, 1, 1, 0, 1,
6.30303, 3.010101, -521.5114, 1, 1, 0, 1,
6.343434, 3.010101, -521.7432, 1, 1, 0, 1,
6.383838, 3.010101, -522.011, 1, 1, 0, 1,
6.424242, 3.010101, -522.3149, 1, 1, 0, 1,
6.464646, 3.010101, -522.6548, 1, 1, 0, 1,
6.505051, 3.010101, -523.0308, 1, 1, 0, 1,
6.545455, 3.010101, -523.4428, 1, 1, 0, 1,
6.585859, 3.010101, -523.8908, 1, 1, 0, 1,
6.626263, 3.010101, -524.3749, 1, 1, 0, 1,
6.666667, 3.010101, -524.895, 1, 1, 0, 1,
6.707071, 3.010101, -525.4511, 1, 0.8941177, 0, 1,
6.747475, 3.010101, -526.0433, 1, 0.8941177, 0, 1,
6.787879, 3.010101, -526.6714, 1, 0.8941177, 0, 1,
6.828283, 3.010101, -527.3357, 1, 0.8941177, 0, 1,
6.868687, 3.010101, -528.0359, 1, 0.8941177, 0, 1,
6.909091, 3.010101, -528.7723, 1, 0.8941177, 0, 1,
6.949495, 3.010101, -529.5446, 1, 0.8941177, 0, 1,
6.989899, 3.010101, -530.353, 1, 0.8941177, 0, 1,
7.030303, 3.010101, -531.1973, 1, 0.7921569, 0, 1,
7.070707, 3.010101, -532.0778, 1, 0.7921569, 0, 1,
7.111111, 3.010101, -532.9943, 1, 0.7921569, 0, 1,
7.151515, 3.010101, -533.9468, 1, 0.7921569, 0, 1,
7.191919, 3.010101, -534.9353, 1, 0.7921569, 0, 1,
7.232323, 3.010101, -535.9599, 1, 0.7921569, 0, 1,
7.272727, 3.010101, -537.0205, 1, 0.6862745, 0, 1,
7.313131, 3.010101, -538.1171, 1, 0.6862745, 0, 1,
7.353535, 3.010101, -539.2498, 1, 0.6862745, 0, 1,
7.393939, 3.010101, -540.4185, 1, 0.6862745, 0, 1,
7.434343, 3.010101, -541.6233, 1, 0.6862745, 0, 1,
7.474748, 3.010101, -542.8641, 1, 0.5843138, 0, 1,
7.515152, 3.010101, -544.1409, 1, 0.5843138, 0, 1,
7.555555, 3.010101, -545.4537, 1, 0.5843138, 0, 1,
7.59596, 3.010101, -546.8026, 1, 0.5843138, 0, 1,
7.636364, 3.010101, -548.1875, 1, 0.5843138, 0, 1,
7.676768, 3.010101, -549.6085, 1, 0.4823529, 0, 1,
7.717172, 3.010101, -551.0654, 1, 0.4823529, 0, 1,
7.757576, 3.010101, -552.5585, 1, 0.4823529, 0, 1,
7.79798, 3.010101, -554.0875, 1, 0.4823529, 0, 1,
7.838384, 3.010101, -555.6526, 1, 0.3764706, 0, 1,
7.878788, 3.010101, -557.2537, 1, 0.3764706, 0, 1,
7.919192, 3.010101, -558.8909, 1, 0.3764706, 0, 1,
7.959596, 3.010101, -560.5641, 1, 0.2745098, 0, 1,
8, 3.010101, -562.2733, 1, 0.2745098, 0, 1,
4, 3.060606, -565.8331, 1, 0.2745098, 0, 1,
4.040404, 3.060606, -564.0706, 1, 0.2745098, 0, 1,
4.080808, 3.060606, -562.343, 1, 0.2745098, 0, 1,
4.121212, 3.060606, -560.6502, 1, 0.2745098, 0, 1,
4.161616, 3.060606, -558.9922, 1, 0.3764706, 0, 1,
4.20202, 3.060606, -557.3691, 1, 0.3764706, 0, 1,
4.242424, 3.060606, -555.7809, 1, 0.3764706, 0, 1,
4.282828, 3.060606, -554.2275, 1, 0.4823529, 0, 1,
4.323232, 3.060606, -552.709, 1, 0.4823529, 0, 1,
4.363636, 3.060606, -551.2254, 1, 0.4823529, 0, 1,
4.40404, 3.060606, -549.7766, 1, 0.4823529, 0, 1,
4.444445, 3.060606, -548.3626, 1, 0.5843138, 0, 1,
4.484848, 3.060606, -546.9835, 1, 0.5843138, 0, 1,
4.525252, 3.060606, -545.6393, 1, 0.5843138, 0, 1,
4.565657, 3.060606, -544.3299, 1, 0.5843138, 0, 1,
4.606061, 3.060606, -543.0554, 1, 0.5843138, 0, 1,
4.646465, 3.060606, -541.8157, 1, 0.6862745, 0, 1,
4.686869, 3.060606, -540.6108, 1, 0.6862745, 0, 1,
4.727273, 3.060606, -539.4409, 1, 0.6862745, 0, 1,
4.767677, 3.060606, -538.3058, 1, 0.6862745, 0, 1,
4.808081, 3.060606, -537.2055, 1, 0.6862745, 0, 1,
4.848485, 3.060606, -536.1401, 1, 0.7921569, 0, 1,
4.888889, 3.060606, -535.1096, 1, 0.7921569, 0, 1,
4.929293, 3.060606, -534.1138, 1, 0.7921569, 0, 1,
4.969697, 3.060606, -533.153, 1, 0.7921569, 0, 1,
5.010101, 3.060606, -532.2271, 1, 0.7921569, 0, 1,
5.050505, 3.060606, -531.3359, 1, 0.7921569, 0, 1,
5.090909, 3.060606, -530.4796, 1, 0.8941177, 0, 1,
5.131313, 3.060606, -529.6582, 1, 0.8941177, 0, 1,
5.171717, 3.060606, -528.8716, 1, 0.8941177, 0, 1,
5.212121, 3.060606, -528.1199, 1, 0.8941177, 0, 1,
5.252525, 3.060606, -527.4031, 1, 0.8941177, 0, 1,
5.292929, 3.060606, -526.7211, 1, 0.8941177, 0, 1,
5.333333, 3.060606, -526.0739, 1, 0.8941177, 0, 1,
5.373737, 3.060606, -525.4617, 1, 0.8941177, 0, 1,
5.414141, 3.060606, -524.8842, 1, 1, 0, 1,
5.454545, 3.060606, -524.3416, 1, 1, 0, 1,
5.494949, 3.060606, -523.8339, 1, 1, 0, 1,
5.535354, 3.060606, -523.361, 1, 1, 0, 1,
5.575758, 3.060606, -522.923, 1, 1, 0, 1,
5.616162, 3.060606, -522.5198, 1, 1, 0, 1,
5.656566, 3.060606, -522.1516, 1, 1, 0, 1,
5.69697, 3.060606, -521.8181, 1, 1, 0, 1,
5.737374, 3.060606, -521.5195, 1, 1, 0, 1,
5.777778, 3.060606, -521.2557, 1, 1, 0, 1,
5.818182, 3.060606, -521.0269, 1, 1, 0, 1,
5.858586, 3.060606, -520.8328, 1, 1, 0, 1,
5.89899, 3.060606, -520.6737, 1, 1, 0, 1,
5.939394, 3.060606, -520.5494, 1, 1, 0, 1,
5.979798, 3.060606, -520.4599, 1, 1, 0, 1,
6.020202, 3.060606, -520.4053, 1, 1, 0, 1,
6.060606, 3.060606, -520.3855, 1, 1, 0, 1,
6.10101, 3.060606, -520.4006, 1, 1, 0, 1,
6.141414, 3.060606, -520.4506, 1, 1, 0, 1,
6.181818, 3.060606, -520.5354, 1, 1, 0, 1,
6.222222, 3.060606, -520.655, 1, 1, 0, 1,
6.262626, 3.060606, -520.8096, 1, 1, 0, 1,
6.30303, 3.060606, -520.999, 1, 1, 0, 1,
6.343434, 3.060606, -521.2232, 1, 1, 0, 1,
6.383838, 3.060606, -521.4822, 1, 1, 0, 1,
6.424242, 3.060606, -521.7762, 1, 1, 0, 1,
6.464646, 3.060606, -522.105, 1, 1, 0, 1,
6.505051, 3.060606, -522.4686, 1, 1, 0, 1,
6.545455, 3.060606, -522.8671, 1, 1, 0, 1,
6.585859, 3.060606, -523.3005, 1, 1, 0, 1,
6.626263, 3.060606, -523.7687, 1, 1, 0, 1,
6.666667, 3.060606, -524.2718, 1, 1, 0, 1,
6.707071, 3.060606, -524.8097, 1, 1, 0, 1,
6.747475, 3.060606, -525.3825, 1, 1, 0, 1,
6.787879, 3.060606, -525.9901, 1, 0.8941177, 0, 1,
6.828283, 3.060606, -526.6326, 1, 0.8941177, 0, 1,
6.868687, 3.060606, -527.3099, 1, 0.8941177, 0, 1,
6.909091, 3.060606, -528.0222, 1, 0.8941177, 0, 1,
6.949495, 3.060606, -528.7692, 1, 0.8941177, 0, 1,
6.989899, 3.060606, -529.5511, 1, 0.8941177, 0, 1,
7.030303, 3.060606, -530.3679, 1, 0.8941177, 0, 1,
7.070707, 3.060606, -531.2195, 1, 0.7921569, 0, 1,
7.111111, 3.060606, -532.106, 1, 0.7921569, 0, 1,
7.151515, 3.060606, -533.0273, 1, 0.7921569, 0, 1,
7.191919, 3.060606, -533.9835, 1, 0.7921569, 0, 1,
7.232323, 3.060606, -534.9745, 1, 0.7921569, 0, 1,
7.272727, 3.060606, -536.0004, 1, 0.7921569, 0, 1,
7.313131, 3.060606, -537.0612, 1, 0.6862745, 0, 1,
7.353535, 3.060606, -538.1568, 1, 0.6862745, 0, 1,
7.393939, 3.060606, -539.2872, 1, 0.6862745, 0, 1,
7.434343, 3.060606, -540.4526, 1, 0.6862745, 0, 1,
7.474748, 3.060606, -541.6527, 1, 0.6862745, 0, 1,
7.515152, 3.060606, -542.8878, 1, 0.5843138, 0, 1,
7.555555, 3.060606, -544.1577, 1, 0.5843138, 0, 1,
7.59596, 3.060606, -545.4623, 1, 0.5843138, 0, 1,
7.636364, 3.060606, -546.8019, 1, 0.5843138, 0, 1,
7.676768, 3.060606, -548.1764, 1, 0.5843138, 0, 1,
7.717172, 3.060606, -549.5857, 1, 0.4823529, 0, 1,
7.757576, 3.060606, -551.0298, 1, 0.4823529, 0, 1,
7.79798, 3.060606, -552.5089, 1, 0.4823529, 0, 1,
7.838384, 3.060606, -554.0227, 1, 0.4823529, 0, 1,
7.878788, 3.060606, -555.5714, 1, 0.3764706, 0, 1,
7.919192, 3.060606, -557.155, 1, 0.3764706, 0, 1,
7.959596, 3.060606, -558.7734, 1, 0.3764706, 0, 1,
8, 3.060606, -560.4267, 1, 0.2745098, 0, 1,
4, 3.111111, -564.0079, 1, 0.2745098, 0, 1,
4.040404, 3.111111, -562.3021, 1, 0.2745098, 0, 1,
4.080808, 3.111111, -560.6301, 1, 0.2745098, 0, 1,
4.121212, 3.111111, -558.9919, 1, 0.3764706, 0, 1,
4.161616, 3.111111, -557.3873, 1, 0.3764706, 0, 1,
4.20202, 3.111111, -555.8165, 1, 0.3764706, 0, 1,
4.242424, 3.111111, -554.2794, 1, 0.4823529, 0, 1,
4.282828, 3.111111, -552.7761, 1, 0.4823529, 0, 1,
4.323232, 3.111111, -551.3065, 1, 0.4823529, 0, 1,
4.363636, 3.111111, -549.8705, 1, 0.4823529, 0, 1,
4.40404, 3.111111, -548.4684, 1, 0.5843138, 0, 1,
4.444445, 3.111111, -547.1, 1, 0.5843138, 0, 1,
4.484848, 3.111111, -545.7653, 1, 0.5843138, 0, 1,
4.525252, 3.111111, -544.4644, 1, 0.5843138, 0, 1,
4.565657, 3.111111, -543.1971, 1, 0.5843138, 0, 1,
4.606061, 3.111111, -541.9636, 1, 0.6862745, 0, 1,
4.646465, 3.111111, -540.7639, 1, 0.6862745, 0, 1,
4.686869, 3.111111, -539.5978, 1, 0.6862745, 0, 1,
4.727273, 3.111111, -538.4656, 1, 0.6862745, 0, 1,
4.767677, 3.111111, -537.367, 1, 0.6862745, 0, 1,
4.808081, 3.111111, -536.3022, 1, 0.7921569, 0, 1,
4.848485, 3.111111, -535.2711, 1, 0.7921569, 0, 1,
4.888889, 3.111111, -534.2737, 1, 0.7921569, 0, 1,
4.929293, 3.111111, -533.3101, 1, 0.7921569, 0, 1,
4.969697, 3.111111, -532.3802, 1, 0.7921569, 0, 1,
5.010101, 3.111111, -531.4841, 1, 0.7921569, 0, 1,
5.050505, 3.111111, -530.6216, 1, 0.8941177, 0, 1,
5.090909, 3.111111, -529.7929, 1, 0.8941177, 0, 1,
5.131313, 3.111111, -528.9979, 1, 0.8941177, 0, 1,
5.171717, 3.111111, -528.2367, 1, 0.8941177, 0, 1,
5.212121, 3.111111, -527.5092, 1, 0.8941177, 0, 1,
5.252525, 3.111111, -526.8154, 1, 0.8941177, 0, 1,
5.292929, 3.111111, -526.1554, 1, 0.8941177, 0, 1,
5.333333, 3.111111, -525.5291, 1, 0.8941177, 0, 1,
5.373737, 3.111111, -524.9365, 1, 1, 0, 1,
5.414141, 3.111111, -524.3777, 1, 1, 0, 1,
5.454545, 3.111111, -523.8526, 1, 1, 0, 1,
5.494949, 3.111111, -523.3612, 1, 1, 0, 1,
5.535354, 3.111111, -522.9036, 1, 1, 0, 1,
5.575758, 3.111111, -522.4797, 1, 1, 0, 1,
5.616162, 3.111111, -522.0895, 1, 1, 0, 1,
5.656566, 3.111111, -521.733, 1, 1, 0, 1,
5.69697, 3.111111, -521.4103, 1, 1, 0, 1,
5.737374, 3.111111, -521.1213, 1, 1, 0, 1,
5.777778, 3.111111, -520.8661, 1, 1, 0, 1,
5.818182, 3.111111, -520.6446, 1, 1, 0, 1,
5.858586, 3.111111, -520.4568, 1, 1, 0, 1,
5.89899, 3.111111, -520.3027, 1, 1, 0, 1,
5.939394, 3.111111, -520.1824, 1, 1, 0, 1,
5.979798, 3.111111, -520.0959, 1, 1, 0, 1,
6.020202, 3.111111, -520.043, 1, 1, 0, 1,
6.060606, 3.111111, -520.0239, 1, 1, 0, 1,
6.10101, 3.111111, -520.0385, 1, 1, 0, 1,
6.141414, 3.111111, -520.0869, 1, 1, 0, 1,
6.181818, 3.111111, -520.1689, 1, 1, 0, 1,
6.222222, 3.111111, -520.2847, 1, 1, 0, 1,
6.262626, 3.111111, -520.4343, 1, 1, 0, 1,
6.30303, 3.111111, -520.6176, 1, 1, 0, 1,
6.343434, 3.111111, -520.8346, 1, 1, 0, 1,
6.383838, 3.111111, -521.0853, 1, 1, 0, 1,
6.424242, 3.111111, -521.3698, 1, 1, 0, 1,
6.464646, 3.111111, -521.688, 1, 1, 0, 1,
6.505051, 3.111111, -522.0399, 1, 1, 0, 1,
6.545455, 3.111111, -522.4256, 1, 1, 0, 1,
6.585859, 3.111111, -522.845, 1, 1, 0, 1,
6.626263, 3.111111, -523.2982, 1, 1, 0, 1,
6.666667, 3.111111, -523.785, 1, 1, 0, 1,
6.707071, 3.111111, -524.3056, 1, 1, 0, 1,
6.747475, 3.111111, -524.8599, 1, 1, 0, 1,
6.787879, 3.111111, -525.448, 1, 0.8941177, 0, 1,
6.828283, 3.111111, -526.0698, 1, 0.8941177, 0, 1,
6.868687, 3.111111, -526.7253, 1, 0.8941177, 0, 1,
6.909091, 3.111111, -527.4146, 1, 0.8941177, 0, 1,
6.949495, 3.111111, -528.1376, 1, 0.8941177, 0, 1,
6.989899, 3.111111, -528.8943, 1, 0.8941177, 0, 1,
7.030303, 3.111111, -529.6848, 1, 0.8941177, 0, 1,
7.070707, 3.111111, -530.509, 1, 0.8941177, 0, 1,
7.111111, 3.111111, -531.3669, 1, 0.7921569, 0, 1,
7.151515, 3.111111, -532.2585, 1, 0.7921569, 0, 1,
7.191919, 3.111111, -533.184, 1, 0.7921569, 0, 1,
7.232323, 3.111111, -534.1431, 1, 0.7921569, 0, 1,
7.272727, 3.111111, -535.1359, 1, 0.7921569, 0, 1,
7.313131, 3.111111, -536.1625, 1, 0.7921569, 0, 1,
7.353535, 3.111111, -537.2228, 1, 0.6862745, 0, 1,
7.393939, 3.111111, -538.3169, 1, 0.6862745, 0, 1,
7.434343, 3.111111, -539.4447, 1, 0.6862745, 0, 1,
7.474748, 3.111111, -540.6062, 1, 0.6862745, 0, 1,
7.515152, 3.111111, -541.8015, 1, 0.6862745, 0, 1,
7.555555, 3.111111, -543.0305, 1, 0.5843138, 0, 1,
7.59596, 3.111111, -544.2932, 1, 0.5843138, 0, 1,
7.636364, 3.111111, -545.5896, 1, 0.5843138, 0, 1,
7.676768, 3.111111, -546.9198, 1, 0.5843138, 0, 1,
7.717172, 3.111111, -548.2837, 1, 0.5843138, 0, 1,
7.757576, 3.111111, -549.6813, 1, 0.4823529, 0, 1,
7.79798, 3.111111, -551.1127, 1, 0.4823529, 0, 1,
7.838384, 3.111111, -552.5778, 1, 0.4823529, 0, 1,
7.878788, 3.111111, -554.0767, 1, 0.4823529, 0, 1,
7.919192, 3.111111, -555.6093, 1, 0.3764706, 0, 1,
7.959596, 3.111111, -557.1755, 1, 0.3764706, 0, 1,
8, 3.111111, -558.7756, 1, 0.3764706, 0, 1,
4, 3.161616, -562.3723, 1, 0.2745098, 0, 1,
4.040404, 3.161616, -560.7206, 1, 0.2745098, 0, 1,
4.080808, 3.161616, -559.1016, 1, 0.3764706, 0, 1,
4.121212, 3.161616, -557.5153, 1, 0.3764706, 0, 1,
4.161616, 3.161616, -555.9616, 1, 0.3764706, 0, 1,
4.20202, 3.161616, -554.4406, 1, 0.3764706, 0, 1,
4.242424, 3.161616, -552.9522, 1, 0.4823529, 0, 1,
4.282828, 3.161616, -551.4965, 1, 0.4823529, 0, 1,
4.323232, 3.161616, -550.0734, 1, 0.4823529, 0, 1,
4.363636, 3.161616, -548.683, 1, 0.4823529, 0, 1,
4.40404, 3.161616, -547.3254, 1, 0.5843138, 0, 1,
4.444445, 3.161616, -546.0003, 1, 0.5843138, 0, 1,
4.484848, 3.161616, -544.7079, 1, 0.5843138, 0, 1,
4.525252, 3.161616, -543.4482, 1, 0.5843138, 0, 1,
4.565657, 3.161616, -542.2211, 1, 0.6862745, 0, 1,
4.606061, 3.161616, -541.0267, 1, 0.6862745, 0, 1,
4.646465, 3.161616, -539.865, 1, 0.6862745, 0, 1,
4.686869, 3.161616, -538.736, 1, 0.6862745, 0, 1,
4.727273, 3.161616, -537.6395, 1, 0.6862745, 0, 1,
4.767677, 3.161616, -536.5758, 1, 0.7921569, 0, 1,
4.808081, 3.161616, -535.5447, 1, 0.7921569, 0, 1,
4.848485, 3.161616, -534.5463, 1, 0.7921569, 0, 1,
4.888889, 3.161616, -533.5806, 1, 0.7921569, 0, 1,
4.929293, 3.161616, -532.6475, 1, 0.7921569, 0, 1,
4.969697, 3.161616, -531.7471, 1, 0.7921569, 0, 1,
5.010101, 3.161616, -530.8793, 1, 0.8941177, 0, 1,
5.050505, 3.161616, -530.0442, 1, 0.8941177, 0, 1,
5.090909, 3.161616, -529.2418, 1, 0.8941177, 0, 1,
5.131313, 3.161616, -528.472, 1, 0.8941177, 0, 1,
5.171717, 3.161616, -527.7349, 1, 0.8941177, 0, 1,
5.212121, 3.161616, -527.0305, 1, 0.8941177, 0, 1,
5.252525, 3.161616, -526.3586, 1, 0.8941177, 0, 1,
5.292929, 3.161616, -525.7195, 1, 0.8941177, 0, 1,
5.333333, 3.161616, -525.1131, 1, 1, 0, 1,
5.373737, 3.161616, -524.5393, 1, 1, 0, 1,
5.414141, 3.161616, -523.9982, 1, 1, 0, 1,
5.454545, 3.161616, -523.4897, 1, 1, 0, 1,
5.494949, 3.161616, -523.0139, 1, 1, 0, 1,
5.535354, 3.161616, -522.5707, 1, 1, 0, 1,
5.575758, 3.161616, -522.1603, 1, 1, 0, 1,
5.616162, 3.161616, -521.7825, 1, 1, 0, 1,
5.656566, 3.161616, -521.4373, 1, 1, 0, 1,
5.69697, 3.161616, -521.1248, 1, 1, 0, 1,
5.737374, 3.161616, -520.845, 1, 1, 0, 1,
5.777778, 3.161616, -520.5978, 1, 1, 0, 1,
5.818182, 3.161616, -520.3834, 1, 1, 0, 1,
5.858586, 3.161616, -520.2015, 1, 1, 0, 1,
5.89899, 3.161616, -520.0524, 1, 1, 0, 1,
5.939394, 3.161616, -519.9359, 1, 1, 0, 1,
5.979798, 3.161616, -519.8521, 1, 1, 0, 1,
6.020202, 3.161616, -519.8008, 1, 1, 0, 1,
6.060606, 3.161616, -519.7823, 1, 1, 0, 1,
6.10101, 3.161616, -519.7965, 1, 1, 0, 1,
6.141414, 3.161616, -519.8433, 1, 1, 0, 1,
6.181818, 3.161616, -519.9228, 1, 1, 0, 1,
6.222222, 3.161616, -520.0349, 1, 1, 0, 1,
6.262626, 3.161616, -520.1797, 1, 1, 0, 1,
6.30303, 3.161616, -520.3572, 1, 1, 0, 1,
6.343434, 3.161616, -520.5673, 1, 1, 0, 1,
6.383838, 3.161616, -520.8101, 1, 1, 0, 1,
6.424242, 3.161616, -521.0856, 1, 1, 0, 1,
6.464646, 3.161616, -521.3937, 1, 1, 0, 1,
6.505051, 3.161616, -521.7345, 1, 1, 0, 1,
6.545455, 3.161616, -522.1079, 1, 1, 0, 1,
6.585859, 3.161616, -522.514, 1, 1, 0, 1,
6.626263, 3.161616, -522.9528, 1, 1, 0, 1,
6.666667, 3.161616, -523.4243, 1, 1, 0, 1,
6.707071, 3.161616, -523.9283, 1, 1, 0, 1,
6.747475, 3.161616, -524.4651, 1, 1, 0, 1,
6.787879, 3.161616, -525.0345, 1, 1, 0, 1,
6.828283, 3.161616, -525.6367, 1, 0.8941177, 0, 1,
6.868687, 3.161616, -526.2714, 1, 0.8941177, 0, 1,
6.909091, 3.161616, -526.9388, 1, 0.8941177, 0, 1,
6.949495, 3.161616, -527.6389, 1, 0.8941177, 0, 1,
6.989899, 3.161616, -528.3716, 1, 0.8941177, 0, 1,
7.030303, 3.161616, -529.137, 1, 0.8941177, 0, 1,
7.070707, 3.161616, -529.9351, 1, 0.8941177, 0, 1,
7.111111, 3.161616, -530.7659, 1, 0.8941177, 0, 1,
7.151515, 3.161616, -531.6293, 1, 0.7921569, 0, 1,
7.191919, 3.161616, -532.5253, 1, 0.7921569, 0, 1,
7.232323, 3.161616, -533.454, 1, 0.7921569, 0, 1,
7.272727, 3.161616, -534.4154, 1, 0.7921569, 0, 1,
7.313131, 3.161616, -535.4095, 1, 0.7921569, 0, 1,
7.353535, 3.161616, -536.4362, 1, 0.7921569, 0, 1,
7.393939, 3.161616, -537.4956, 1, 0.6862745, 0, 1,
7.434343, 3.161616, -538.5876, 1, 0.6862745, 0, 1,
7.474748, 3.161616, -539.7123, 1, 0.6862745, 0, 1,
7.515152, 3.161616, -540.8697, 1, 0.6862745, 0, 1,
7.555555, 3.161616, -542.0598, 1, 0.6862745, 0, 1,
7.59596, 3.161616, -543.2824, 1, 0.5843138, 0, 1,
7.636364, 3.161616, -544.5378, 1, 0.5843138, 0, 1,
7.676768, 3.161616, -545.8258, 1, 0.5843138, 0, 1,
7.717172, 3.161616, -547.1465, 1, 0.5843138, 0, 1,
7.757576, 3.161616, -548.4998, 1, 0.5843138, 0, 1,
7.79798, 3.161616, -549.8859, 1, 0.4823529, 0, 1,
7.838384, 3.161616, -551.3045, 1, 0.4823529, 0, 1,
7.878788, 3.161616, -552.7559, 1, 0.4823529, 0, 1,
7.919192, 3.161616, -554.2399, 1, 0.4823529, 0, 1,
7.959596, 3.161616, -555.7565, 1, 0.3764706, 0, 1,
8, 3.161616, -557.3058, 1, 0.3764706, 0, 1,
4, 3.212121, -560.913, 1, 0.2745098, 0, 1,
4.040404, 3.212121, -559.3129, 1, 0.3764706, 0, 1,
4.080808, 3.212121, -557.7443, 1, 0.3764706, 0, 1,
4.121212, 3.212121, -556.2075, 1, 0.3764706, 0, 1,
4.161616, 3.212121, -554.7023, 1, 0.3764706, 0, 1,
4.20202, 3.212121, -553.2287, 1, 0.4823529, 0, 1,
4.242424, 3.212121, -551.7867, 1, 0.4823529, 0, 1,
4.282828, 3.212121, -550.3765, 1, 0.4823529, 0, 1,
4.323232, 3.212121, -548.9978, 1, 0.4823529, 0, 1,
4.363636, 3.212121, -547.6508, 1, 0.5843138, 0, 1,
4.40404, 3.212121, -546.3354, 1, 0.5843138, 0, 1,
4.444445, 3.212121, -545.0518, 1, 0.5843138, 0, 1,
4.484848, 3.212121, -543.7997, 1, 0.5843138, 0, 1,
4.525252, 3.212121, -542.5793, 1, 0.6862745, 0, 1,
4.565657, 3.212121, -541.3905, 1, 0.6862745, 0, 1,
4.606061, 3.212121, -540.2334, 1, 0.6862745, 0, 1,
4.646465, 3.212121, -539.1079, 1, 0.6862745, 0, 1,
4.686869, 3.212121, -538.014, 1, 0.6862745, 0, 1,
4.727273, 3.212121, -536.9518, 1, 0.7921569, 0, 1,
4.767677, 3.212121, -535.9213, 1, 0.7921569, 0, 1,
4.808081, 3.212121, -534.9224, 1, 0.7921569, 0, 1,
4.848485, 3.212121, -533.9551, 1, 0.7921569, 0, 1,
4.888889, 3.212121, -533.0195, 1, 0.7921569, 0, 1,
4.929293, 3.212121, -532.1155, 1, 0.7921569, 0, 1,
4.969697, 3.212121, -531.2432, 1, 0.7921569, 0, 1,
5.010101, 3.212121, -530.4025, 1, 0.8941177, 0, 1,
5.050505, 3.212121, -529.5934, 1, 0.8941177, 0, 1,
5.090909, 3.212121, -528.816, 1, 0.8941177, 0, 1,
5.131313, 3.212121, -528.0703, 1, 0.8941177, 0, 1,
5.171717, 3.212121, -527.3562, 1, 0.8941177, 0, 1,
5.212121, 3.212121, -526.6737, 1, 0.8941177, 0, 1,
5.252525, 3.212121, -526.0229, 1, 0.8941177, 0, 1,
5.292929, 3.212121, -525.4037, 1, 1, 0, 1,
5.333333, 3.212121, -524.8162, 1, 1, 0, 1,
5.373737, 3.212121, -524.2603, 1, 1, 0, 1,
5.414141, 3.212121, -523.7361, 1, 1, 0, 1,
5.454545, 3.212121, -523.2435, 1, 1, 0, 1,
5.494949, 3.212121, -522.7825, 1, 1, 0, 1,
5.535354, 3.212121, -522.3532, 1, 1, 0, 1,
5.575758, 3.212121, -521.9555, 1, 1, 0, 1,
5.616162, 3.212121, -521.5895, 1, 1, 0, 1,
5.656566, 3.212121, -521.2551, 1, 1, 0, 1,
5.69697, 3.212121, -520.9524, 1, 1, 0, 1,
5.737374, 3.212121, -520.6813, 1, 1, 0, 1,
5.777778, 3.212121, -520.4418, 1, 1, 0, 1,
5.818182, 3.212121, -520.2341, 1, 1, 0, 1,
5.858586, 3.212121, -520.0579, 1, 1, 0, 1,
5.89899, 3.212121, -519.9134, 1, 1, 0, 1,
5.939394, 3.212121, -519.8005, 1, 1, 0, 1,
5.979798, 3.212121, -519.7193, 1, 1, 0, 1,
6.020202, 3.212121, -519.6697, 1, 1, 0, 1,
6.060606, 3.212121, -519.6518, 1, 1, 0, 1,
6.10101, 3.212121, -519.6655, 1, 1, 0, 1,
6.141414, 3.212121, -519.7108, 1, 1, 0, 1,
6.181818, 3.212121, -519.7878, 1, 1, 0, 1,
6.222222, 3.212121, -519.8965, 1, 1, 0, 1,
6.262626, 3.212121, -520.0367, 1, 1, 0, 1,
6.30303, 3.212121, -520.2087, 1, 1, 0, 1,
6.343434, 3.212121, -520.4123, 1, 1, 0, 1,
6.383838, 3.212121, -520.6475, 1, 1, 0, 1,
6.424242, 3.212121, -520.9144, 1, 1, 0, 1,
6.464646, 3.212121, -521.2129, 1, 1, 0, 1,
6.505051, 3.212121, -521.543, 1, 1, 0, 1,
6.545455, 3.212121, -521.9048, 1, 1, 0, 1,
6.585859, 3.212121, -522.2983, 1, 1, 0, 1,
6.626263, 3.212121, -522.7233, 1, 1, 0, 1,
6.666667, 3.212121, -523.1801, 1, 1, 0, 1,
6.707071, 3.212121, -523.6685, 1, 1, 0, 1,
6.747475, 3.212121, -524.1885, 1, 1, 0, 1,
6.787879, 3.212121, -524.7401, 1, 1, 0, 1,
6.828283, 3.212121, -525.3234, 1, 1, 0, 1,
6.868687, 3.212121, -525.9384, 1, 0.8941177, 0, 1,
6.909091, 3.212121, -526.585, 1, 0.8941177, 0, 1,
6.949495, 3.212121, -527.2632, 1, 0.8941177, 0, 1,
6.989899, 3.212121, -527.9731, 1, 0.8941177, 0, 1,
7.030303, 3.212121, -528.7146, 1, 0.8941177, 0, 1,
7.070707, 3.212121, -529.4878, 1, 0.8941177, 0, 1,
7.111111, 3.212121, -530.2926, 1, 0.8941177, 0, 1,
7.151515, 3.212121, -531.1291, 1, 0.8941177, 0, 1,
7.191919, 3.212121, -531.9972, 1, 0.7921569, 0, 1,
7.232323, 3.212121, -532.8969, 1, 0.7921569, 0, 1,
7.272727, 3.212121, -533.8283, 1, 0.7921569, 0, 1,
7.313131, 3.212121, -534.7914, 1, 0.7921569, 0, 1,
7.353535, 3.212121, -535.7861, 1, 0.7921569, 0, 1,
7.393939, 3.212121, -536.8124, 1, 0.7921569, 0, 1,
7.434343, 3.212121, -537.8704, 1, 0.6862745, 0, 1,
7.474748, 3.212121, -538.96, 1, 0.6862745, 0, 1,
7.515152, 3.212121, -540.0812, 1, 0.6862745, 0, 1,
7.555555, 3.212121, -541.2341, 1, 0.6862745, 0, 1,
7.59596, 3.212121, -542.4187, 1, 0.6862745, 0, 1,
7.636364, 3.212121, -543.6349, 1, 0.5843138, 0, 1,
7.676768, 3.212121, -544.8827, 1, 0.5843138, 0, 1,
7.717172, 3.212121, -546.1622, 1, 0.5843138, 0, 1,
7.757576, 3.212121, -547.4733, 1, 0.5843138, 0, 1,
7.79798, 3.212121, -548.8161, 1, 0.4823529, 0, 1,
7.838384, 3.212121, -550.1905, 1, 0.4823529, 0, 1,
7.878788, 3.212121, -551.5966, 1, 0.4823529, 0, 1,
7.919192, 3.212121, -553.0342, 1, 0.4823529, 0, 1,
7.959596, 3.212121, -554.5036, 1, 0.3764706, 0, 1,
8, 3.212121, -556.0046, 1, 0.3764706, 0, 1,
4, 3.262626, -559.6175, 1, 0.3764706, 0, 1,
4.040404, 3.262626, -558.0665, 1, 0.3764706, 0, 1,
4.080808, 3.262626, -556.5462, 1, 0.3764706, 0, 1,
4.121212, 3.262626, -555.0565, 1, 0.3764706, 0, 1,
4.161616, 3.262626, -553.5975, 1, 0.4823529, 0, 1,
4.20202, 3.262626, -552.1693, 1, 0.4823529, 0, 1,
4.242424, 3.262626, -550.7716, 1, 0.4823529, 0, 1,
4.282828, 3.262626, -549.4047, 1, 0.4823529, 0, 1,
4.323232, 3.262626, -548.0684, 1, 0.5843138, 0, 1,
4.363636, 3.262626, -546.7628, 1, 0.5843138, 0, 1,
4.40404, 3.262626, -545.4878, 1, 0.5843138, 0, 1,
4.444445, 3.262626, -544.2435, 1, 0.5843138, 0, 1,
4.484848, 3.262626, -543.0299, 1, 0.5843138, 0, 1,
4.525252, 3.262626, -541.847, 1, 0.6862745, 0, 1,
4.565657, 3.262626, -540.6948, 1, 0.6862745, 0, 1,
4.606061, 3.262626, -539.5732, 1, 0.6862745, 0, 1,
4.646465, 3.262626, -538.4822, 1, 0.6862745, 0, 1,
4.686869, 3.262626, -537.422, 1, 0.6862745, 0, 1,
4.727273, 3.262626, -536.3925, 1, 0.7921569, 0, 1,
4.767677, 3.262626, -535.3936, 1, 0.7921569, 0, 1,
4.808081, 3.262626, -534.4254, 1, 0.7921569, 0, 1,
4.848485, 3.262626, -533.4878, 1, 0.7921569, 0, 1,
4.888889, 3.262626, -532.5809, 1, 0.7921569, 0, 1,
4.929293, 3.262626, -531.7047, 1, 0.7921569, 0, 1,
4.969697, 3.262626, -530.8592, 1, 0.8941177, 0, 1,
5.010101, 3.262626, -530.0443, 1, 0.8941177, 0, 1,
5.050505, 3.262626, -529.2601, 1, 0.8941177, 0, 1,
5.090909, 3.262626, -528.5066, 1, 0.8941177, 0, 1,
5.131313, 3.262626, -527.7838, 1, 0.8941177, 0, 1,
5.171717, 3.262626, -527.0916, 1, 0.8941177, 0, 1,
5.212121, 3.262626, -526.4301, 1, 0.8941177, 0, 1,
5.252525, 3.262626, -525.7993, 1, 0.8941177, 0, 1,
5.292929, 3.262626, -525.1991, 1, 1, 0, 1,
5.333333, 3.262626, -524.6296, 1, 1, 0, 1,
5.373737, 3.262626, -524.0908, 1, 1, 0, 1,
5.414141, 3.262626, -523.5826, 1, 1, 0, 1,
5.454545, 3.262626, -523.1052, 1, 1, 0, 1,
5.494949, 3.262626, -522.6584, 1, 1, 0, 1,
5.535354, 3.262626, -522.2422, 1, 1, 0, 1,
5.575758, 3.262626, -521.8568, 1, 1, 0, 1,
5.616162, 3.262626, -521.502, 1, 1, 0, 1,
5.656566, 3.262626, -521.1779, 1, 1, 0, 1,
5.69697, 3.262626, -520.8845, 1, 1, 0, 1,
5.737374, 3.262626, -520.6218, 1, 1, 0, 1,
5.777778, 3.262626, -520.3896, 1, 1, 0, 1,
5.818182, 3.262626, -520.1882, 1, 1, 0, 1,
5.858586, 3.262626, -520.0175, 1, 1, 0, 1,
5.89899, 3.262626, -519.8774, 1, 1, 0, 1,
5.939394, 3.262626, -519.768, 1, 1, 0, 1,
5.979798, 3.262626, -519.6893, 1, 1, 0, 1,
6.020202, 3.262626, -519.6412, 1, 1, 0, 1,
6.060606, 3.262626, -519.6238, 1, 1, 0, 1,
6.10101, 3.262626, -519.6371, 1, 1, 0, 1,
6.141414, 3.262626, -519.6811, 1, 1, 0, 1,
6.181818, 3.262626, -519.7557, 1, 1, 0, 1,
6.222222, 3.262626, -519.861, 1, 1, 0, 1,
6.262626, 3.262626, -519.997, 1, 1, 0, 1,
6.30303, 3.262626, -520.1636, 1, 1, 0, 1,
6.343434, 3.262626, -520.361, 1, 1, 0, 1,
6.383838, 3.262626, -520.589, 1, 1, 0, 1,
6.424242, 3.262626, -520.8477, 1, 1, 0, 1,
6.464646, 3.262626, -521.137, 1, 1, 0, 1,
6.505051, 3.262626, -521.457, 1, 1, 0, 1,
6.545455, 3.262626, -521.8077, 1, 1, 0, 1,
6.585859, 3.262626, -522.189, 1, 1, 0, 1,
6.626263, 3.262626, -522.601, 1, 1, 0, 1,
6.666667, 3.262626, -523.0438, 1, 1, 0, 1,
6.707071, 3.262626, -523.5171, 1, 1, 0, 1,
6.747475, 3.262626, -524.0212, 1, 1, 0, 1,
6.787879, 3.262626, -524.5558, 1, 1, 0, 1,
6.828283, 3.262626, -525.1213, 1, 1, 0, 1,
6.868687, 3.262626, -525.7173, 1, 0.8941177, 0, 1,
6.909091, 3.262626, -526.3441, 1, 0.8941177, 0, 1,
6.949495, 3.262626, -527.0015, 1, 0.8941177, 0, 1,
6.989899, 3.262626, -527.6895, 1, 0.8941177, 0, 1,
7.030303, 3.262626, -528.4083, 1, 0.8941177, 0, 1,
7.070707, 3.262626, -529.1577, 1, 0.8941177, 0, 1,
7.111111, 3.262626, -529.9378, 1, 0.8941177, 0, 1,
7.151515, 3.262626, -530.7485, 1, 0.8941177, 0, 1,
7.191919, 3.262626, -531.59, 1, 0.7921569, 0, 1,
7.232323, 3.262626, -532.4621, 1, 0.7921569, 0, 1,
7.272727, 3.262626, -533.3649, 1, 0.7921569, 0, 1,
7.313131, 3.262626, -534.2983, 1, 0.7921569, 0, 1,
7.353535, 3.262626, -535.2625, 1, 0.7921569, 0, 1,
7.393939, 3.262626, -536.2573, 1, 0.7921569, 0, 1,
7.434343, 3.262626, -537.2827, 1, 0.6862745, 0, 1,
7.474748, 3.262626, -538.3389, 1, 0.6862745, 0, 1,
7.515152, 3.262626, -539.4257, 1, 0.6862745, 0, 1,
7.555555, 3.262626, -540.5432, 1, 0.6862745, 0, 1,
7.59596, 3.262626, -541.6913, 1, 0.6862745, 0, 1,
7.636364, 3.262626, -542.8702, 1, 0.5843138, 0, 1,
7.676768, 3.262626, -544.0797, 1, 0.5843138, 0, 1,
7.717172, 3.262626, -545.3198, 1, 0.5843138, 0, 1,
7.757576, 3.262626, -546.5907, 1, 0.5843138, 0, 1,
7.79798, 3.262626, -547.8922, 1, 0.5843138, 0, 1,
7.838384, 3.262626, -549.2244, 1, 0.4823529, 0, 1,
7.878788, 3.262626, -550.5873, 1, 0.4823529, 0, 1,
7.919192, 3.262626, -551.9808, 1, 0.4823529, 0, 1,
7.959596, 3.262626, -553.405, 1, 0.4823529, 0, 1,
8, 3.262626, -554.8599, 1, 0.3764706, 0, 1,
4, 3.313131, -558.4745, 1, 0.3764706, 0, 1,
4.040404, 3.313131, -556.9704, 1, 0.3764706, 0, 1,
4.080808, 3.313131, -555.4961, 1, 0.3764706, 0, 1,
4.121212, 3.313131, -554.0515, 1, 0.4823529, 0, 1,
4.161616, 3.313131, -552.6367, 1, 0.4823529, 0, 1,
4.20202, 3.313131, -551.2516, 1, 0.4823529, 0, 1,
4.242424, 3.313131, -549.8962, 1, 0.4823529, 0, 1,
4.282828, 3.313131, -548.5706, 1, 0.4823529, 0, 1,
4.323232, 3.313131, -547.2748, 1, 0.5843138, 0, 1,
4.363636, 3.313131, -546.0087, 1, 0.5843138, 0, 1,
4.40404, 3.313131, -544.7723, 1, 0.5843138, 0, 1,
4.444445, 3.313131, -543.5657, 1, 0.5843138, 0, 1,
4.484848, 3.313131, -542.3888, 1, 0.6862745, 0, 1,
4.525252, 3.313131, -541.2416, 1, 0.6862745, 0, 1,
4.565657, 3.313131, -540.1243, 1, 0.6862745, 0, 1,
4.606061, 3.313131, -539.0366, 1, 0.6862745, 0, 1,
4.646465, 3.313131, -537.9787, 1, 0.6862745, 0, 1,
4.686869, 3.313131, -536.9505, 1, 0.7921569, 0, 1,
4.727273, 3.313131, -535.9521, 1, 0.7921569, 0, 1,
4.767677, 3.313131, -534.9835, 1, 0.7921569, 0, 1,
4.808081, 3.313131, -534.0445, 1, 0.7921569, 0, 1,
4.848485, 3.313131, -533.1353, 1, 0.7921569, 0, 1,
4.888889, 3.313131, -532.2559, 1, 0.7921569, 0, 1,
4.929293, 3.313131, -531.4062, 1, 0.7921569, 0, 1,
4.969697, 3.313131, -530.5862, 1, 0.8941177, 0, 1,
5.010101, 3.313131, -529.796, 1, 0.8941177, 0, 1,
5.050505, 3.313131, -529.0356, 1, 0.8941177, 0, 1,
5.090909, 3.313131, -528.3048, 1, 0.8941177, 0, 1,
5.131313, 3.313131, -527.6039, 1, 0.8941177, 0, 1,
5.171717, 3.313131, -526.9326, 1, 0.8941177, 0, 1,
5.212121, 3.313131, -526.2911, 1, 0.8941177, 0, 1,
5.252525, 3.313131, -525.6794, 1, 0.8941177, 0, 1,
5.292929, 3.313131, -525.0974, 1, 1, 0, 1,
5.333333, 3.313131, -524.5452, 1, 1, 0, 1,
5.373737, 3.313131, -524.0226, 1, 1, 0, 1,
5.414141, 3.313131, -523.5298, 1, 1, 0, 1,
5.454545, 3.313131, -523.0668, 1, 1, 0, 1,
5.494949, 3.313131, -522.6336, 1, 1, 0, 1,
5.535354, 3.313131, -522.23, 1, 1, 0, 1,
5.575758, 3.313131, -521.8563, 1, 1, 0, 1,
5.616162, 3.313131, -521.5122, 1, 1, 0, 1,
5.656566, 3.313131, -521.1979, 1, 1, 0, 1,
5.69697, 3.313131, -520.9133, 1, 1, 0, 1,
5.737374, 3.313131, -520.6586, 1, 1, 0, 1,
5.777778, 3.313131, -520.4335, 1, 1, 0, 1,
5.818182, 3.313131, -520.2382, 1, 1, 0, 1,
5.858586, 3.313131, -520.0726, 1, 1, 0, 1,
5.89899, 3.313131, -519.9367, 1, 1, 0, 1,
5.939394, 3.313131, -519.8306, 1, 1, 0, 1,
5.979798, 3.313131, -519.7543, 1, 1, 0, 1,
6.020202, 3.313131, -519.7077, 1, 1, 0, 1,
6.060606, 3.313131, -519.6909, 1, 1, 0, 1,
6.10101, 3.313131, -519.7037, 1, 1, 0, 1,
6.141414, 3.313131, -519.7463, 1, 1, 0, 1,
6.181818, 3.313131, -519.8187, 1, 1, 0, 1,
6.222222, 3.313131, -519.9208, 1, 1, 0, 1,
6.262626, 3.313131, -520.0527, 1, 1, 0, 1,
6.30303, 3.313131, -520.2143, 1, 1, 0, 1,
6.343434, 3.313131, -520.4056, 1, 1, 0, 1,
6.383838, 3.313131, -520.6268, 1, 1, 0, 1,
6.424242, 3.313131, -520.8776, 1, 1, 0, 1,
6.464646, 3.313131, -521.1582, 1, 1, 0, 1,
6.505051, 3.313131, -521.4685, 1, 1, 0, 1,
6.545455, 3.313131, -521.8086, 1, 1, 0, 1,
6.585859, 3.313131, -522.1784, 1, 1, 0, 1,
6.626263, 3.313131, -522.5779, 1, 1, 0, 1,
6.666667, 3.313131, -523.0073, 1, 1, 0, 1,
6.707071, 3.313131, -523.4663, 1, 1, 0, 1,
6.747475, 3.313131, -523.9551, 1, 1, 0, 1,
6.787879, 3.313131, -524.4736, 1, 1, 0, 1,
6.828283, 3.313131, -525.0219, 1, 1, 0, 1,
6.868687, 3.313131, -525.6, 1, 0.8941177, 0, 1,
6.909091, 3.313131, -526.2077, 1, 0.8941177, 0, 1,
6.949495, 3.313131, -526.8452, 1, 0.8941177, 0, 1,
6.989899, 3.313131, -527.5125, 1, 0.8941177, 0, 1,
7.030303, 3.313131, -528.2095, 1, 0.8941177, 0, 1,
7.070707, 3.313131, -528.9362, 1, 0.8941177, 0, 1,
7.111111, 3.313131, -529.6927, 1, 0.8941177, 0, 1,
7.151515, 3.313131, -530.4789, 1, 0.8941177, 0, 1,
7.191919, 3.313131, -531.2949, 1, 0.7921569, 0, 1,
7.232323, 3.313131, -532.1407, 1, 0.7921569, 0, 1,
7.272727, 3.313131, -533.0161, 1, 0.7921569, 0, 1,
7.313131, 3.313131, -533.9213, 1, 0.7921569, 0, 1,
7.353535, 3.313131, -534.8563, 1, 0.7921569, 0, 1,
7.393939, 3.313131, -535.821, 1, 0.7921569, 0, 1,
7.434343, 3.313131, -536.8154, 1, 0.7921569, 0, 1,
7.474748, 3.313131, -537.8397, 1, 0.6862745, 0, 1,
7.515152, 3.313131, -538.8936, 1, 0.6862745, 0, 1,
7.555555, 3.313131, -539.9772, 1, 0.6862745, 0, 1,
7.59596, 3.313131, -541.0906, 1, 0.6862745, 0, 1,
7.636364, 3.313131, -542.2338, 1, 0.6862745, 0, 1,
7.676768, 3.313131, -543.4067, 1, 0.5843138, 0, 1,
7.717172, 3.313131, -544.6094, 1, 0.5843138, 0, 1,
7.757576, 3.313131, -545.8418, 1, 0.5843138, 0, 1,
7.79798, 3.313131, -547.1039, 1, 0.5843138, 0, 1,
7.838384, 3.313131, -548.3958, 1, 0.5843138, 0, 1,
7.878788, 3.313131, -549.7175, 1, 0.4823529, 0, 1,
7.919192, 3.313131, -551.0688, 1, 0.4823529, 0, 1,
7.959596, 3.313131, -552.45, 1, 0.4823529, 0, 1,
8, 3.313131, -553.8608, 1, 0.4823529, 0, 1,
4, 3.363636, -557.4734, 1, 0.3764706, 0, 1,
4.040404, 3.363636, -556.0142, 1, 0.3764706, 0, 1,
4.080808, 3.363636, -554.5838, 1, 0.3764706, 0, 1,
4.121212, 3.363636, -553.1823, 1, 0.4823529, 0, 1,
4.161616, 3.363636, -551.8096, 1, 0.4823529, 0, 1,
4.20202, 3.363636, -550.4658, 1, 0.4823529, 0, 1,
4.242424, 3.363636, -549.1508, 1, 0.4823529, 0, 1,
4.282828, 3.363636, -547.8647, 1, 0.5843138, 0, 1,
4.323232, 3.363636, -546.6075, 1, 0.5843138, 0, 1,
4.363636, 3.363636, -545.3792, 1, 0.5843138, 0, 1,
4.40404, 3.363636, -544.1796, 1, 0.5843138, 0, 1,
4.444445, 3.363636, -543.009, 1, 0.5843138, 0, 1,
4.484848, 3.363636, -541.8671, 1, 0.6862745, 0, 1,
4.525252, 3.363636, -540.7542, 1, 0.6862745, 0, 1,
4.565657, 3.363636, -539.6701, 1, 0.6862745, 0, 1,
4.606061, 3.363636, -538.6149, 1, 0.6862745, 0, 1,
4.646465, 3.363636, -537.5885, 1, 0.6862745, 0, 1,
4.686869, 3.363636, -536.5909, 1, 0.7921569, 0, 1,
4.727273, 3.363636, -535.6223, 1, 0.7921569, 0, 1,
4.767677, 3.363636, -534.6825, 1, 0.7921569, 0, 1,
4.808081, 3.363636, -533.7715, 1, 0.7921569, 0, 1,
4.848485, 3.363636, -532.8895, 1, 0.7921569, 0, 1,
4.888889, 3.363636, -532.0363, 1, 0.7921569, 0, 1,
4.929293, 3.363636, -531.2119, 1, 0.7921569, 0, 1,
4.969697, 3.363636, -530.4164, 1, 0.8941177, 0, 1,
5.010101, 3.363636, -529.6497, 1, 0.8941177, 0, 1,
5.050505, 3.363636, -528.9119, 1, 0.8941177, 0, 1,
5.090909, 3.363636, -528.2029, 1, 0.8941177, 0, 1,
5.131313, 3.363636, -527.5229, 1, 0.8941177, 0, 1,
5.171717, 3.363636, -526.8716, 1, 0.8941177, 0, 1,
5.212121, 3.363636, -526.2493, 1, 0.8941177, 0, 1,
5.252525, 3.363636, -525.6558, 1, 0.8941177, 0, 1,
5.292929, 3.363636, -525.0911, 1, 1, 0, 1,
5.333333, 3.363636, -524.5554, 1, 1, 0, 1,
5.373737, 3.363636, -524.0484, 1, 1, 0, 1,
5.414141, 3.363636, -523.5703, 1, 1, 0, 1,
5.454545, 3.363636, -523.1211, 1, 1, 0, 1,
5.494949, 3.363636, -522.7007, 1, 1, 0, 1,
5.535354, 3.363636, -522.3092, 1, 1, 0, 1,
5.575758, 3.363636, -521.9466, 1, 1, 0, 1,
5.616162, 3.363636, -521.6128, 1, 1, 0, 1,
5.656566, 3.363636, -521.3079, 1, 1, 0, 1,
5.69697, 3.363636, -521.0318, 1, 1, 0, 1,
5.737374, 3.363636, -520.7845, 1, 1, 0, 1,
5.777778, 3.363636, -520.5662, 1, 1, 0, 1,
5.818182, 3.363636, -520.3767, 1, 1, 0, 1,
5.858586, 3.363636, -520.2161, 1, 1, 0, 1,
5.89899, 3.363636, -520.0843, 1, 1, 0, 1,
5.939394, 3.363636, -519.9813, 1, 1, 0, 1,
5.979798, 3.363636, -519.9073, 1, 1, 0, 1,
6.020202, 3.363636, -519.8621, 1, 1, 0, 1,
6.060606, 3.363636, -519.8457, 1, 1, 0, 1,
6.10101, 3.363636, -519.8582, 1, 1, 0, 1,
6.141414, 3.363636, -519.8995, 1, 1, 0, 1,
6.181818, 3.363636, -519.9698, 1, 1, 0, 1,
6.222222, 3.363636, -520.0688, 1, 1, 0, 1,
6.262626, 3.363636, -520.1968, 1, 1, 0, 1,
6.30303, 3.363636, -520.3536, 1, 1, 0, 1,
6.343434, 3.363636, -520.5392, 1, 1, 0, 1,
6.383838, 3.363636, -520.7537, 1, 1, 0, 1,
6.424242, 3.363636, -520.9971, 1, 1, 0, 1,
6.464646, 3.363636, -521.2693, 1, 1, 0, 1,
6.505051, 3.363636, -521.5704, 1, 1, 0, 1,
6.545455, 3.363636, -521.9003, 1, 1, 0, 1,
6.585859, 3.363636, -522.2592, 1, 1, 0, 1,
6.626263, 3.363636, -522.6468, 1, 1, 0, 1,
6.666667, 3.363636, -523.0633, 1, 1, 0, 1,
6.707071, 3.363636, -523.5087, 1, 1, 0, 1,
6.747475, 3.363636, -523.9828, 1, 1, 0, 1,
6.787879, 3.363636, -524.486, 1, 1, 0, 1,
6.828283, 3.363636, -525.0179, 1, 1, 0, 1,
6.868687, 3.363636, -525.5787, 1, 0.8941177, 0, 1,
6.909091, 3.363636, -526.1683, 1, 0.8941177, 0, 1,
6.949495, 3.363636, -526.7869, 1, 0.8941177, 0, 1,
6.989899, 3.363636, -527.4343, 1, 0.8941177, 0, 1,
7.030303, 3.363636, -528.1105, 1, 0.8941177, 0, 1,
7.070707, 3.363636, -528.8156, 1, 0.8941177, 0, 1,
7.111111, 3.363636, -529.5495, 1, 0.8941177, 0, 1,
7.151515, 3.363636, -530.3123, 1, 0.8941177, 0, 1,
7.191919, 3.363636, -531.1039, 1, 0.8941177, 0, 1,
7.232323, 3.363636, -531.9245, 1, 0.7921569, 0, 1,
7.272727, 3.363636, -532.7739, 1, 0.7921569, 0, 1,
7.313131, 3.363636, -533.6521, 1, 0.7921569, 0, 1,
7.353535, 3.363636, -534.5591, 1, 0.7921569, 0, 1,
7.393939, 3.363636, -535.4951, 1, 0.7921569, 0, 1,
7.434343, 3.363636, -536.4599, 1, 0.7921569, 0, 1,
7.474748, 3.363636, -537.4536, 1, 0.6862745, 0, 1,
7.515152, 3.363636, -538.4761, 1, 0.6862745, 0, 1,
7.555555, 3.363636, -539.5275, 1, 0.6862745, 0, 1,
7.59596, 3.363636, -540.6077, 1, 0.6862745, 0, 1,
7.636364, 3.363636, -541.7168, 1, 0.6862745, 0, 1,
7.676768, 3.363636, -542.8547, 1, 0.5843138, 0, 1,
7.717172, 3.363636, -544.0215, 1, 0.5843138, 0, 1,
7.757576, 3.363636, -545.2172, 1, 0.5843138, 0, 1,
7.79798, 3.363636, -546.4418, 1, 0.5843138, 0, 1,
7.838384, 3.363636, -547.6951, 1, 0.5843138, 0, 1,
7.878788, 3.363636, -548.9774, 1, 0.4823529, 0, 1,
7.919192, 3.363636, -550.2885, 1, 0.4823529, 0, 1,
7.959596, 3.363636, -551.6284, 1, 0.4823529, 0, 1,
8, 3.363636, -552.9973, 1, 0.4823529, 0, 1,
4, 3.414141, -556.6046, 1, 0.3764706, 0, 1,
4.040404, 3.414141, -555.1882, 1, 0.3764706, 0, 1,
4.080808, 3.414141, -553.7999, 1, 0.4823529, 0, 1,
4.121212, 3.414141, -552.4395, 1, 0.4823529, 0, 1,
4.161616, 3.414141, -551.1072, 1, 0.4823529, 0, 1,
4.20202, 3.414141, -549.8028, 1, 0.4823529, 0, 1,
4.242424, 3.414141, -548.5265, 1, 0.5843138, 0, 1,
4.282828, 3.414141, -547.2781, 1, 0.5843138, 0, 1,
4.323232, 3.414141, -546.0578, 1, 0.5843138, 0, 1,
4.363636, 3.414141, -544.8655, 1, 0.5843138, 0, 1,
4.40404, 3.414141, -543.7012, 1, 0.5843138, 0, 1,
4.444445, 3.414141, -542.5649, 1, 0.6862745, 0, 1,
4.484848, 3.414141, -541.4567, 1, 0.6862745, 0, 1,
4.525252, 3.414141, -540.3764, 1, 0.6862745, 0, 1,
4.565657, 3.414141, -539.3242, 1, 0.6862745, 0, 1,
4.606061, 3.414141, -538.2999, 1, 0.6862745, 0, 1,
4.646465, 3.414141, -537.3036, 1, 0.6862745, 0, 1,
4.686869, 3.414141, -536.3354, 1, 0.7921569, 0, 1,
4.727273, 3.414141, -535.3953, 1, 0.7921569, 0, 1,
4.767677, 3.414141, -534.483, 1, 0.7921569, 0, 1,
4.808081, 3.414141, -533.5988, 1, 0.7921569, 0, 1,
4.848485, 3.414141, -532.7427, 1, 0.7921569, 0, 1,
4.888889, 3.414141, -531.9145, 1, 0.7921569, 0, 1,
4.929293, 3.414141, -531.1143, 1, 0.8941177, 0, 1,
4.969697, 3.414141, -530.3422, 1, 0.8941177, 0, 1,
5.010101, 3.414141, -529.598, 1, 0.8941177, 0, 1,
5.050505, 3.414141, -528.8819, 1, 0.8941177, 0, 1,
5.090909, 3.414141, -528.1938, 1, 0.8941177, 0, 1,
5.131313, 3.414141, -527.5337, 1, 0.8941177, 0, 1,
5.171717, 3.414141, -526.9016, 1, 0.8941177, 0, 1,
5.212121, 3.414141, -526.2975, 1, 0.8941177, 0, 1,
5.252525, 3.414141, -525.7214, 1, 0.8941177, 0, 1,
5.292929, 3.414141, -525.1733, 1, 1, 0, 1,
5.333333, 3.414141, -524.6533, 1, 1, 0, 1,
5.373737, 3.414141, -524.1612, 1, 1, 0, 1,
5.414141, 3.414141, -523.6972, 1, 1, 0, 1,
5.454545, 3.414141, -523.2612, 1, 1, 0, 1,
5.494949, 3.414141, -522.8531, 1, 1, 0, 1,
5.535354, 3.414141, -522.4731, 1, 1, 0, 1,
5.575758, 3.414141, -522.1211, 1, 1, 0, 1,
5.616162, 3.414141, -521.7971, 1, 1, 0, 1,
5.656566, 3.414141, -521.5012, 1, 1, 0, 1,
5.69697, 3.414141, -521.2332, 1, 1, 0, 1,
5.737374, 3.414141, -520.9932, 1, 1, 0, 1,
5.777778, 3.414141, -520.7813, 1, 1, 0, 1,
5.818182, 3.414141, -520.5974, 1, 1, 0, 1,
5.858586, 3.414141, -520.4414, 1, 1, 0, 1,
5.89899, 3.414141, -520.3135, 1, 1, 0, 1,
5.939394, 3.414141, -520.2136, 1, 1, 0, 1,
5.979798, 3.414141, -520.1417, 1, 1, 0, 1,
6.020202, 3.414141, -520.0978, 1, 1, 0, 1,
6.060606, 3.414141, -520.082, 1, 1, 0, 1,
6.10101, 3.414141, -520.0941, 1, 1, 0, 1,
6.141414, 3.414141, -520.1342, 1, 1, 0, 1,
6.181818, 3.414141, -520.2024, 1, 1, 0, 1,
6.222222, 3.414141, -520.2985, 1, 1, 0, 1,
6.262626, 3.414141, -520.4227, 1, 1, 0, 1,
6.30303, 3.414141, -520.5749, 1, 1, 0, 1,
6.343434, 3.414141, -520.7551, 1, 1, 0, 1,
6.383838, 3.414141, -520.9633, 1, 1, 0, 1,
6.424242, 3.414141, -521.1995, 1, 1, 0, 1,
6.464646, 3.414141, -521.4637, 1, 1, 0, 1,
6.505051, 3.414141, -521.756, 1, 1, 0, 1,
6.545455, 3.414141, -522.0762, 1, 1, 0, 1,
6.585859, 3.414141, -522.4245, 1, 1, 0, 1,
6.626263, 3.414141, -522.8008, 1, 1, 0, 1,
6.666667, 3.414141, -523.205, 1, 1, 0, 1,
6.707071, 3.414141, -523.6373, 1, 1, 0, 1,
6.747475, 3.414141, -524.0976, 1, 1, 0, 1,
6.787879, 3.414141, -524.5859, 1, 1, 0, 1,
6.828283, 3.414141, -525.1022, 1, 1, 0, 1,
6.868687, 3.414141, -525.6465, 1, 0.8941177, 0, 1,
6.909091, 3.414141, -526.2189, 1, 0.8941177, 0, 1,
6.949495, 3.414141, -526.8193, 1, 0.8941177, 0, 1,
6.989899, 3.414141, -527.4476, 1, 0.8941177, 0, 1,
7.030303, 3.414141, -528.104, 1, 0.8941177, 0, 1,
7.070707, 3.414141, -528.7884, 1, 0.8941177, 0, 1,
7.111111, 3.414141, -529.5007, 1, 0.8941177, 0, 1,
7.151515, 3.414141, -530.2411, 1, 0.8941177, 0, 1,
7.191919, 3.414141, -531.0096, 1, 0.8941177, 0, 1,
7.232323, 3.414141, -531.806, 1, 0.7921569, 0, 1,
7.272727, 3.414141, -532.6304, 1, 0.7921569, 0, 1,
7.313131, 3.414141, -533.4828, 1, 0.7921569, 0, 1,
7.353535, 3.414141, -534.3633, 1, 0.7921569, 0, 1,
7.393939, 3.414141, -535.2718, 1, 0.7921569, 0, 1,
7.434343, 3.414141, -536.2083, 1, 0.7921569, 0, 1,
7.474748, 3.414141, -537.1727, 1, 0.6862745, 0, 1,
7.515152, 3.414141, -538.1652, 1, 0.6862745, 0, 1,
7.555555, 3.414141, -539.1857, 1, 0.6862745, 0, 1,
7.59596, 3.414141, -540.2343, 1, 0.6862745, 0, 1,
7.636364, 3.414141, -541.3108, 1, 0.6862745, 0, 1,
7.676768, 3.414141, -542.4153, 1, 0.6862745, 0, 1,
7.717172, 3.414141, -543.5479, 1, 0.5843138, 0, 1,
7.757576, 3.414141, -544.7084, 1, 0.5843138, 0, 1,
7.79798, 3.414141, -545.897, 1, 0.5843138, 0, 1,
7.838384, 3.414141, -547.1135, 1, 0.5843138, 0, 1,
7.878788, 3.414141, -548.3581, 1, 0.5843138, 0, 1,
7.919192, 3.414141, -549.6307, 1, 0.4823529, 0, 1,
7.959596, 3.414141, -550.9313, 1, 0.4823529, 0, 1,
8, 3.414141, -552.2599, 1, 0.4823529, 0, 1,
4, 3.464647, -555.8592, 1, 0.3764706, 0, 1,
4.040404, 3.464647, -554.4838, 1, 0.3764706, 0, 1,
4.080808, 3.464647, -553.1356, 1, 0.4823529, 0, 1,
4.121212, 3.464647, -551.8146, 1, 0.4823529, 0, 1,
4.161616, 3.464647, -550.5208, 1, 0.4823529, 0, 1,
4.20202, 3.464647, -549.2542, 1, 0.4823529, 0, 1,
4.242424, 3.464647, -548.0148, 1, 0.5843138, 0, 1,
4.282828, 3.464647, -546.8026, 1, 0.5843138, 0, 1,
4.323232, 3.464647, -545.6176, 1, 0.5843138, 0, 1,
4.363636, 3.464647, -544.4598, 1, 0.5843138, 0, 1,
4.40404, 3.464647, -543.3292, 1, 0.5843138, 0, 1,
4.444445, 3.464647, -542.2258, 1, 0.6862745, 0, 1,
4.484848, 3.464647, -541.1497, 1, 0.6862745, 0, 1,
4.525252, 3.464647, -540.1006, 1, 0.6862745, 0, 1,
4.565657, 3.464647, -539.0789, 1, 0.6862745, 0, 1,
4.606061, 3.464647, -538.0842, 1, 0.6862745, 0, 1,
4.646465, 3.464647, -537.1168, 1, 0.6862745, 0, 1,
4.686869, 3.464647, -536.1766, 1, 0.7921569, 0, 1,
4.727273, 3.464647, -535.2637, 1, 0.7921569, 0, 1,
4.767677, 3.464647, -534.3779, 1, 0.7921569, 0, 1,
4.808081, 3.464647, -533.5192, 1, 0.7921569, 0, 1,
4.848485, 3.464647, -532.6879, 1, 0.7921569, 0, 1,
4.888889, 3.464647, -531.8837, 1, 0.7921569, 0, 1,
4.929293, 3.464647, -531.1066, 1, 0.8941177, 0, 1,
4.969697, 3.464647, -530.3568, 1, 0.8941177, 0, 1,
5.010101, 3.464647, -529.6342, 1, 0.8941177, 0, 1,
5.050505, 3.464647, -528.9388, 1, 0.8941177, 0, 1,
5.090909, 3.464647, -528.2706, 1, 0.8941177, 0, 1,
5.131313, 3.464647, -527.6296, 1, 0.8941177, 0, 1,
5.171717, 3.464647, -527.0158, 1, 0.8941177, 0, 1,
5.212121, 3.464647, -526.4292, 1, 0.8941177, 0, 1,
5.252525, 3.464647, -525.8698, 1, 0.8941177, 0, 1,
5.292929, 3.464647, -525.3376, 1, 1, 0, 1,
5.333333, 3.464647, -524.8326, 1, 1, 0, 1,
5.373737, 3.464647, -524.3547, 1, 1, 0, 1,
5.414141, 3.464647, -523.9042, 1, 1, 0, 1,
5.454545, 3.464647, -523.4808, 1, 1, 0, 1,
5.494949, 3.464647, -523.0845, 1, 1, 0, 1,
5.535354, 3.464647, -522.7155, 1, 1, 0, 1,
5.575758, 3.464647, -522.3737, 1, 1, 0, 1,
5.616162, 3.464647, -522.0591, 1, 1, 0, 1,
5.656566, 3.464647, -521.7717, 1, 1, 0, 1,
5.69697, 3.464647, -521.5115, 1, 1, 0, 1,
5.737374, 3.464647, -521.2784, 1, 1, 0, 1,
5.777778, 3.464647, -521.0726, 1, 1, 0, 1,
5.818182, 3.464647, -520.894, 1, 1, 0, 1,
5.858586, 3.464647, -520.7426, 1, 1, 0, 1,
5.89899, 3.464647, -520.6184, 1, 1, 0, 1,
5.939394, 3.464647, -520.5214, 1, 1, 0, 1,
5.979798, 3.464647, -520.4516, 1, 1, 0, 1,
6.020202, 3.464647, -520.4089, 1, 1, 0, 1,
6.060606, 3.464647, -520.3936, 1, 1, 0, 1,
6.10101, 3.464647, -520.4053, 1, 1, 0, 1,
6.141414, 3.464647, -520.4443, 1, 1, 0, 1,
6.181818, 3.464647, -520.5105, 1, 1, 0, 1,
6.222222, 3.464647, -520.6039, 1, 1, 0, 1,
6.262626, 3.464647, -520.7245, 1, 1, 0, 1,
6.30303, 3.464647, -520.8723, 1, 1, 0, 1,
6.343434, 3.464647, -521.0472, 1, 1, 0, 1,
6.383838, 3.464647, -521.2494, 1, 1, 0, 1,
6.424242, 3.464647, -521.4788, 1, 1, 0, 1,
6.464646, 3.464647, -521.7354, 1, 1, 0, 1,
6.505051, 3.464647, -522.0192, 1, 1, 0, 1,
6.545455, 3.464647, -522.3301, 1, 1, 0, 1,
6.585859, 3.464647, -522.6683, 1, 1, 0, 1,
6.626263, 3.464647, -523.0337, 1, 1, 0, 1,
6.666667, 3.464647, -523.4263, 1, 1, 0, 1,
6.707071, 3.464647, -523.846, 1, 1, 0, 1,
6.747475, 3.464647, -524.293, 1, 1, 0, 1,
6.787879, 3.464647, -524.7672, 1, 1, 0, 1,
6.828283, 3.464647, -525.2686, 1, 1, 0, 1,
6.868687, 3.464647, -525.7971, 1, 0.8941177, 0, 1,
6.909091, 3.464647, -526.3529, 1, 0.8941177, 0, 1,
6.949495, 3.464647, -526.9359, 1, 0.8941177, 0, 1,
6.989899, 3.464647, -527.5461, 1, 0.8941177, 0, 1,
7.030303, 3.464647, -528.1834, 1, 0.8941177, 0, 1,
7.070707, 3.464647, -528.848, 1, 0.8941177, 0, 1,
7.111111, 3.464647, -529.5398, 1, 0.8941177, 0, 1,
7.151515, 3.464647, -530.2587, 1, 0.8941177, 0, 1,
7.191919, 3.464647, -531.0049, 1, 0.8941177, 0, 1,
7.232323, 3.464647, -531.7783, 1, 0.7921569, 0, 1,
7.272727, 3.464647, -532.5789, 1, 0.7921569, 0, 1,
7.313131, 3.464647, -533.4066, 1, 0.7921569, 0, 1,
7.353535, 3.464647, -534.2616, 1, 0.7921569, 0, 1,
7.393939, 3.464647, -535.1437, 1, 0.7921569, 0, 1,
7.434343, 3.464647, -536.0531, 1, 0.7921569, 0, 1,
7.474748, 3.464647, -536.9897, 1, 0.6862745, 0, 1,
7.515152, 3.464647, -537.9535, 1, 0.6862745, 0, 1,
7.555555, 3.464647, -538.9445, 1, 0.6862745, 0, 1,
7.59596, 3.464647, -539.9626, 1, 0.6862745, 0, 1,
7.636364, 3.464647, -541.008, 1, 0.6862745, 0, 1,
7.676768, 3.464647, -542.0805, 1, 0.6862745, 0, 1,
7.717172, 3.464647, -543.1803, 1, 0.5843138, 0, 1,
7.757576, 3.464647, -544.3073, 1, 0.5843138, 0, 1,
7.79798, 3.464647, -545.4614, 1, 0.5843138, 0, 1,
7.838384, 3.464647, -546.6428, 1, 0.5843138, 0, 1,
7.878788, 3.464647, -547.8514, 1, 0.5843138, 0, 1,
7.919192, 3.464647, -549.0871, 1, 0.4823529, 0, 1,
7.959596, 3.464647, -550.3501, 1, 0.4823529, 0, 1,
8, 3.464647, -551.6403, 1, 0.4823529, 0, 1,
4, 3.515152, -555.2288, 1, 0.3764706, 0, 1,
4.040404, 3.515152, -553.8927, 1, 0.4823529, 0, 1,
4.080808, 3.515152, -552.5829, 1, 0.4823529, 0, 1,
4.121212, 3.515152, -551.2997, 1, 0.4823529, 0, 1,
4.161616, 3.515152, -550.0428, 1, 0.4823529, 0, 1,
4.20202, 3.515152, -548.8123, 1, 0.4823529, 0, 1,
4.242424, 3.515152, -547.6083, 1, 0.5843138, 0, 1,
4.282828, 3.515152, -546.4307, 1, 0.5843138, 0, 1,
4.323232, 3.515152, -545.2795, 1, 0.5843138, 0, 1,
4.363636, 3.515152, -544.1547, 1, 0.5843138, 0, 1,
4.40404, 3.515152, -543.0563, 1, 0.5843138, 0, 1,
4.444445, 3.515152, -541.9844, 1, 0.6862745, 0, 1,
4.484848, 3.515152, -540.939, 1, 0.6862745, 0, 1,
4.525252, 3.515152, -539.9199, 1, 0.6862745, 0, 1,
4.565657, 3.515152, -538.9272, 1, 0.6862745, 0, 1,
4.606061, 3.515152, -537.961, 1, 0.6862745, 0, 1,
4.646465, 3.515152, -537.0212, 1, 0.6862745, 0, 1,
4.686869, 3.515152, -536.1078, 1, 0.7921569, 0, 1,
4.727273, 3.515152, -535.2209, 1, 0.7921569, 0, 1,
4.767677, 3.515152, -534.3604, 1, 0.7921569, 0, 1,
4.808081, 3.515152, -533.5262, 1, 0.7921569, 0, 1,
4.848485, 3.515152, -532.7186, 1, 0.7921569, 0, 1,
4.888889, 3.515152, -531.9373, 1, 0.7921569, 0, 1,
4.929293, 3.515152, -531.1824, 1, 0.8941177, 0, 1,
4.969697, 3.515152, -530.454, 1, 0.8941177, 0, 1,
5.010101, 3.515152, -529.752, 1, 0.8941177, 0, 1,
5.050505, 3.515152, -529.0765, 1, 0.8941177, 0, 1,
5.090909, 3.515152, -528.4273, 1, 0.8941177, 0, 1,
5.131313, 3.515152, -527.8046, 1, 0.8941177, 0, 1,
5.171717, 3.515152, -527.2083, 1, 0.8941177, 0, 1,
5.212121, 3.515152, -526.6384, 1, 0.8941177, 0, 1,
5.252525, 3.515152, -526.095, 1, 0.8941177, 0, 1,
5.292929, 3.515152, -525.578, 1, 0.8941177, 0, 1,
5.333333, 3.515152, -525.0874, 1, 1, 0, 1,
5.373737, 3.515152, -524.6232, 1, 1, 0, 1,
5.414141, 3.515152, -524.1854, 1, 1, 0, 1,
5.454545, 3.515152, -523.7741, 1, 1, 0, 1,
5.494949, 3.515152, -523.3892, 1, 1, 0, 1,
5.535354, 3.515152, -523.0307, 1, 1, 0, 1,
5.575758, 3.515152, -522.6987, 1, 1, 0, 1,
5.616162, 3.515152, -522.393, 1, 1, 0, 1,
5.656566, 3.515152, -522.1138, 1, 1, 0, 1,
5.69697, 3.515152, -521.861, 1, 1, 0, 1,
5.737374, 3.515152, -521.6346, 1, 1, 0, 1,
5.777778, 3.515152, -521.4348, 1, 1, 0, 1,
5.818182, 3.515152, -521.2612, 1, 1, 0, 1,
5.858586, 3.515152, -521.1141, 1, 1, 0, 1,
5.89899, 3.515152, -520.9935, 1, 1, 0, 1,
5.939394, 3.515152, -520.8992, 1, 1, 0, 1,
5.979798, 3.515152, -520.8314, 1, 1, 0, 1,
6.020202, 3.515152, -520.79, 1, 1, 0, 1,
6.060606, 3.515152, -520.775, 1, 1, 0, 1,
6.10101, 3.515152, -520.7864, 1, 1, 0, 1,
6.141414, 3.515152, -520.8243, 1, 1, 0, 1,
6.181818, 3.515152, -520.8886, 1, 1, 0, 1,
6.222222, 3.515152, -520.9793, 1, 1, 0, 1,
6.262626, 3.515152, -521.0965, 1, 1, 0, 1,
6.30303, 3.515152, -521.2401, 1, 1, 0, 1,
6.343434, 3.515152, -521.41, 1, 1, 0, 1,
6.383838, 3.515152, -521.6064, 1, 1, 0, 1,
6.424242, 3.515152, -521.8293, 1, 1, 0, 1,
6.464646, 3.515152, -522.0786, 1, 1, 0, 1,
6.505051, 3.515152, -522.3542, 1, 1, 0, 1,
6.545455, 3.515152, -522.6563, 1, 1, 0, 1,
6.585859, 3.515152, -522.9849, 1, 1, 0, 1,
6.626263, 3.515152, -523.3398, 1, 1, 0, 1,
6.666667, 3.515152, -523.7212, 1, 1, 0, 1,
6.707071, 3.515152, -524.129, 1, 1, 0, 1,
6.747475, 3.515152, -524.5632, 1, 1, 0, 1,
6.787879, 3.515152, -525.0239, 1, 1, 0, 1,
6.828283, 3.515152, -525.5109, 1, 0.8941177, 0, 1,
6.868687, 3.515152, -526.0244, 1, 0.8941177, 0, 1,
6.909091, 3.515152, -526.5643, 1, 0.8941177, 0, 1,
6.949495, 3.515152, -527.1307, 1, 0.8941177, 0, 1,
6.989899, 3.515152, -527.7234, 1, 0.8941177, 0, 1,
7.030303, 3.515152, -528.3427, 1, 0.8941177, 0, 1,
7.070707, 3.515152, -528.9882, 1, 0.8941177, 0, 1,
7.111111, 3.515152, -529.6603, 1, 0.8941177, 0, 1,
7.151515, 3.515152, -530.3588, 1, 0.8941177, 0, 1,
7.191919, 3.515152, -531.0836, 1, 0.8941177, 0, 1,
7.232323, 3.515152, -531.835, 1, 0.7921569, 0, 1,
7.272727, 3.515152, -532.6127, 1, 0.7921569, 0, 1,
7.313131, 3.515152, -533.4168, 1, 0.7921569, 0, 1,
7.353535, 3.515152, -534.2474, 1, 0.7921569, 0, 1,
7.393939, 3.515152, -535.1044, 1, 0.7921569, 0, 1,
7.434343, 3.515152, -535.9878, 1, 0.7921569, 0, 1,
7.474748, 3.515152, -536.8976, 1, 0.7921569, 0, 1,
7.515152, 3.515152, -537.8339, 1, 0.6862745, 0, 1,
7.555555, 3.515152, -538.7966, 1, 0.6862745, 0, 1,
7.59596, 3.515152, -539.7858, 1, 0.6862745, 0, 1,
7.636364, 3.515152, -540.8013, 1, 0.6862745, 0, 1,
7.676768, 3.515152, -541.8433, 1, 0.6862745, 0, 1,
7.717172, 3.515152, -542.9117, 1, 0.5843138, 0, 1,
7.757576, 3.515152, -544.0065, 1, 0.5843138, 0, 1,
7.79798, 3.515152, -545.1277, 1, 0.5843138, 0, 1,
7.838384, 3.515152, -546.2754, 1, 0.5843138, 0, 1,
7.878788, 3.515152, -547.4495, 1, 0.5843138, 0, 1,
7.919192, 3.515152, -548.65, 1, 0.4823529, 0, 1,
7.959596, 3.515152, -549.8769, 1, 0.4823529, 0, 1,
8, 3.515152, -551.1302, 1, 0.4823529, 0, 1,
4, 3.565657, -554.7059, 1, 0.3764706, 0, 1,
4.040404, 3.565657, -553.4073, 1, 0.4823529, 0, 1,
4.080808, 3.565657, -552.1345, 1, 0.4823529, 0, 1,
4.121212, 3.565657, -550.8872, 1, 0.4823529, 0, 1,
4.161616, 3.565657, -549.6657, 1, 0.4823529, 0, 1,
4.20202, 3.565657, -548.4698, 1, 0.5843138, 0, 1,
4.242424, 3.565657, -547.2997, 1, 0.5843138, 0, 1,
4.282828, 3.565657, -546.1552, 1, 0.5843138, 0, 1,
4.323232, 3.565657, -545.0364, 1, 0.5843138, 0, 1,
4.363636, 3.565657, -543.9432, 1, 0.5843138, 0, 1,
4.40404, 3.565657, -542.8758, 1, 0.5843138, 0, 1,
4.444445, 3.565657, -541.834, 1, 0.6862745, 0, 1,
4.484848, 3.565657, -540.8179, 1, 0.6862745, 0, 1,
4.525252, 3.565657, -539.8276, 1, 0.6862745, 0, 1,
4.565657, 3.565657, -538.8628, 1, 0.6862745, 0, 1,
4.606061, 3.565657, -537.9238, 1, 0.6862745, 0, 1,
4.646465, 3.565657, -537.0104, 1, 0.6862745, 0, 1,
4.686869, 3.565657, -536.1227, 1, 0.7921569, 0, 1,
4.727273, 3.565657, -535.2607, 1, 0.7921569, 0, 1,
4.767677, 3.565657, -534.4244, 1, 0.7921569, 0, 1,
4.808081, 3.565657, -533.6138, 1, 0.7921569, 0, 1,
4.848485, 3.565657, -532.8288, 1, 0.7921569, 0, 1,
4.888889, 3.565657, -532.0695, 1, 0.7921569, 0, 1,
4.929293, 3.565657, -531.3359, 1, 0.7921569, 0, 1,
4.969697, 3.565657, -530.628, 1, 0.8941177, 0, 1,
5.010101, 3.565657, -529.9457, 1, 0.8941177, 0, 1,
5.050505, 3.565657, -529.2892, 1, 0.8941177, 0, 1,
5.090909, 3.565657, -528.6583, 1, 0.8941177, 0, 1,
5.131313, 3.565657, -528.0531, 1, 0.8941177, 0, 1,
5.171717, 3.565657, -527.4736, 1, 0.8941177, 0, 1,
5.212121, 3.565657, -526.9197, 1, 0.8941177, 0, 1,
5.252525, 3.565657, -526.3915, 1, 0.8941177, 0, 1,
5.292929, 3.565657, -525.889, 1, 0.8941177, 0, 1,
5.333333, 3.565657, -525.4123, 1, 0.8941177, 0, 1,
5.373737, 3.565657, -524.9611, 1, 1, 0, 1,
5.414141, 3.565657, -524.5357, 1, 1, 0, 1,
5.454545, 3.565657, -524.1359, 1, 1, 0, 1,
5.494949, 3.565657, -523.7618, 1, 1, 0, 1,
5.535354, 3.565657, -523.4135, 1, 1, 0, 1,
5.575758, 3.565657, -523.0908, 1, 1, 0, 1,
5.616162, 3.565657, -522.7937, 1, 1, 0, 1,
5.656566, 3.565657, -522.5223, 1, 1, 0, 1,
5.69697, 3.565657, -522.2767, 1, 1, 0, 1,
5.737374, 3.565657, -522.0566, 1, 1, 0, 1,
5.777778, 3.565657, -521.8624, 1, 1, 0, 1,
5.818182, 3.565657, -521.6937, 1, 1, 0, 1,
5.858586, 3.565657, -521.5508, 1, 1, 0, 1,
5.89899, 3.565657, -521.4335, 1, 1, 0, 1,
5.939394, 3.565657, -521.3419, 1, 1, 0, 1,
5.979798, 3.565657, -521.2759, 1, 1, 0, 1,
6.020202, 3.565657, -521.2357, 1, 1, 0, 1,
6.060606, 3.565657, -521.2212, 1, 1, 0, 1,
6.10101, 3.565657, -521.2323, 1, 1, 0, 1,
6.141414, 3.565657, -521.2691, 1, 1, 0, 1,
6.181818, 3.565657, -521.3316, 1, 1, 0, 1,
6.222222, 3.565657, -521.4197, 1, 1, 0, 1,
6.262626, 3.565657, -521.5336, 1, 1, 0, 1,
6.30303, 3.565657, -521.6732, 1, 1, 0, 1,
6.343434, 3.565657, -521.8383, 1, 1, 0, 1,
6.383838, 3.565657, -522.0292, 1, 1, 0, 1,
6.424242, 3.565657, -522.2458, 1, 1, 0, 1,
6.464646, 3.565657, -522.488, 1, 1, 0, 1,
6.505051, 3.565657, -522.756, 1, 1, 0, 1,
6.545455, 3.565657, -523.0496, 1, 1, 0, 1,
6.585859, 3.565657, -523.3689, 1, 1, 0, 1,
6.626263, 3.565657, -523.7139, 1, 1, 0, 1,
6.666667, 3.565657, -524.0845, 1, 1, 0, 1,
6.707071, 3.565657, -524.4808, 1, 1, 0, 1,
6.747475, 3.565657, -524.9028, 1, 1, 0, 1,
6.787879, 3.565657, -525.3505, 1, 1, 0, 1,
6.828283, 3.565657, -525.8239, 1, 0.8941177, 0, 1,
6.868687, 3.565657, -526.3229, 1, 0.8941177, 0, 1,
6.909091, 3.565657, -526.8477, 1, 0.8941177, 0, 1,
6.949495, 3.565657, -527.3981, 1, 0.8941177, 0, 1,
6.989899, 3.565657, -527.9742, 1, 0.8941177, 0, 1,
7.030303, 3.565657, -528.5759, 1, 0.8941177, 0, 1,
7.070707, 3.565657, -529.2034, 1, 0.8941177, 0, 1,
7.111111, 3.565657, -529.8566, 1, 0.8941177, 0, 1,
7.151515, 3.565657, -530.5353, 1, 0.8941177, 0, 1,
7.191919, 3.565657, -531.2399, 1, 0.7921569, 0, 1,
7.232323, 3.565657, -531.97, 1, 0.7921569, 0, 1,
7.272727, 3.565657, -532.7259, 1, 0.7921569, 0, 1,
7.313131, 3.565657, -533.5074, 1, 0.7921569, 0, 1,
7.353535, 3.565657, -534.3146, 1, 0.7921569, 0, 1,
7.393939, 3.565657, -535.1475, 1, 0.7921569, 0, 1,
7.434343, 3.565657, -536.0061, 1, 0.7921569, 0, 1,
7.474748, 3.565657, -536.8904, 1, 0.7921569, 0, 1,
7.515152, 3.565657, -537.8003, 1, 0.6862745, 0, 1,
7.555555, 3.565657, -538.7359, 1, 0.6862745, 0, 1,
7.59596, 3.565657, -539.6972, 1, 0.6862745, 0, 1,
7.636364, 3.565657, -540.6842, 1, 0.6862745, 0, 1,
7.676768, 3.565657, -541.6968, 1, 0.6862745, 0, 1,
7.717172, 3.565657, -542.7352, 1, 0.6862745, 0, 1,
7.757576, 3.565657, -543.7992, 1, 0.5843138, 0, 1,
7.79798, 3.565657, -544.8889, 1, 0.5843138, 0, 1,
7.838384, 3.565657, -546.0043, 1, 0.5843138, 0, 1,
7.878788, 3.565657, -547.1453, 1, 0.5843138, 0, 1,
7.919192, 3.565657, -548.3121, 1, 0.5843138, 0, 1,
7.959596, 3.565657, -549.5045, 1, 0.4823529, 0, 1,
8, 3.565657, -550.7226, 1, 0.4823529, 0, 1,
4, 3.616162, -554.2833, 1, 0.4823529, 0, 1,
4.040404, 3.616162, -553.0208, 1, 0.4823529, 0, 1,
4.080808, 3.616162, -551.7832, 1, 0.4823529, 0, 1,
4.121212, 3.616162, -550.5706, 1, 0.4823529, 0, 1,
4.161616, 3.616162, -549.3829, 1, 0.4823529, 0, 1,
4.20202, 3.616162, -548.2202, 1, 0.5843138, 0, 1,
4.242424, 3.616162, -547.0825, 1, 0.5843138, 0, 1,
4.282828, 3.616162, -545.9698, 1, 0.5843138, 0, 1,
4.323232, 3.616162, -544.882, 1, 0.5843138, 0, 1,
4.363636, 3.616162, -543.8192, 1, 0.5843138, 0, 1,
4.40404, 3.616162, -542.7814, 1, 0.5843138, 0, 1,
4.444445, 3.616162, -541.7685, 1, 0.6862745, 0, 1,
4.484848, 3.616162, -540.7806, 1, 0.6862745, 0, 1,
4.525252, 3.616162, -539.8176, 1, 0.6862745, 0, 1,
4.565657, 3.616162, -538.8797, 1, 0.6862745, 0, 1,
4.606061, 3.616162, -537.9667, 1, 0.6862745, 0, 1,
4.646465, 3.616162, -537.0786, 1, 0.6862745, 0, 1,
4.686869, 3.616162, -536.2156, 1, 0.7921569, 0, 1,
4.727273, 3.616162, -535.3775, 1, 0.7921569, 0, 1,
4.767677, 3.616162, -534.5643, 1, 0.7921569, 0, 1,
4.808081, 3.616162, -533.7762, 1, 0.7921569, 0, 1,
4.848485, 3.616162, -533.013, 1, 0.7921569, 0, 1,
4.888889, 3.616162, -532.2748, 1, 0.7921569, 0, 1,
4.929293, 3.616162, -531.5615, 1, 0.7921569, 0, 1,
4.969697, 3.616162, -530.8732, 1, 0.8941177, 0, 1,
5.010101, 3.616162, -530.2099, 1, 0.8941177, 0, 1,
5.050505, 3.616162, -529.5716, 1, 0.8941177, 0, 1,
5.090909, 3.616162, -528.9582, 1, 0.8941177, 0, 1,
5.131313, 3.616162, -528.3698, 1, 0.8941177, 0, 1,
5.171717, 3.616162, -527.8063, 1, 0.8941177, 0, 1,
5.212121, 3.616162, -527.2678, 1, 0.8941177, 0, 1,
5.252525, 3.616162, -526.7543, 1, 0.8941177, 0, 1,
5.292929, 3.616162, -526.2658, 1, 0.8941177, 0, 1,
5.333333, 3.616162, -525.8022, 1, 0.8941177, 0, 1,
5.373737, 3.616162, -525.3636, 1, 1, 0, 1,
5.414141, 3.616162, -524.95, 1, 1, 0, 1,
5.454545, 3.616162, -524.5613, 1, 1, 0, 1,
5.494949, 3.616162, -524.1976, 1, 1, 0, 1,
5.535354, 3.616162, -523.8588, 1, 1, 0, 1,
5.575758, 3.616162, -523.5451, 1, 1, 0, 1,
5.616162, 3.616162, -523.2563, 1, 1, 0, 1,
5.656566, 3.616162, -522.9924, 1, 1, 0, 1,
5.69697, 3.616162, -522.7536, 1, 1, 0, 1,
5.737374, 3.616162, -522.5397, 1, 1, 0, 1,
5.777778, 3.616162, -522.3508, 1, 1, 0, 1,
5.818182, 3.616162, -522.1868, 1, 1, 0, 1,
5.858586, 3.616162, -522.0478, 1, 1, 0, 1,
5.89899, 3.616162, -521.9338, 1, 1, 0, 1,
5.939394, 3.616162, -521.8447, 1, 1, 0, 1,
5.979798, 3.616162, -521.7806, 1, 1, 0, 1,
6.020202, 3.616162, -521.7415, 1, 1, 0, 1,
6.060606, 3.616162, -521.7274, 1, 1, 0, 1,
6.10101, 3.616162, -521.7382, 1, 1, 0, 1,
6.141414, 3.616162, -521.774, 1, 1, 0, 1,
6.181818, 3.616162, -521.8347, 1, 1, 0, 1,
6.222222, 3.616162, -521.9205, 1, 1, 0, 1,
6.262626, 3.616162, -522.0311, 1, 1, 0, 1,
6.30303, 3.616162, -522.1668, 1, 1, 0, 1,
6.343434, 3.616162, -522.3274, 1, 1, 0, 1,
6.383838, 3.616162, -522.513, 1, 1, 0, 1,
6.424242, 3.616162, -522.7236, 1, 1, 0, 1,
6.464646, 3.616162, -522.9591, 1, 1, 0, 1,
6.505051, 3.616162, -523.2196, 1, 1, 0, 1,
6.545455, 3.616162, -523.5051, 1, 1, 0, 1,
6.585859, 3.616162, -523.8155, 1, 1, 0, 1,
6.626263, 3.616162, -524.1509, 1, 1, 0, 1,
6.666667, 3.616162, -524.5113, 1, 1, 0, 1,
6.707071, 3.616162, -524.8966, 1, 1, 0, 1,
6.747475, 3.616162, -525.3069, 1, 1, 0, 1,
6.787879, 3.616162, -525.7422, 1, 0.8941177, 0, 1,
6.828283, 3.616162, -526.2024, 1, 0.8941177, 0, 1,
6.868687, 3.616162, -526.6876, 1, 0.8941177, 0, 1,
6.909091, 3.616162, -527.1978, 1, 0.8941177, 0, 1,
6.949495, 3.616162, -527.733, 1, 0.8941177, 0, 1,
6.989899, 3.616162, -528.2931, 1, 0.8941177, 0, 1,
7.030303, 3.616162, -528.8782, 1, 0.8941177, 0, 1,
7.070707, 3.616162, -529.4882, 1, 0.8941177, 0, 1,
7.111111, 3.616162, -530.1232, 1, 0.8941177, 0, 1,
7.151515, 3.616162, -530.7832, 1, 0.8941177, 0, 1,
7.191919, 3.616162, -531.4681, 1, 0.7921569, 0, 1,
7.232323, 3.616162, -532.178, 1, 0.7921569, 0, 1,
7.272727, 3.616162, -532.913, 1, 0.7921569, 0, 1,
7.313131, 3.616162, -533.6728, 1, 0.7921569, 0, 1,
7.353535, 3.616162, -534.4576, 1, 0.7921569, 0, 1,
7.393939, 3.616162, -535.2675, 1, 0.7921569, 0, 1,
7.434343, 3.616162, -536.1022, 1, 0.7921569, 0, 1,
7.474748, 3.616162, -536.9619, 1, 0.7921569, 0, 1,
7.515152, 3.616162, -537.8466, 1, 0.6862745, 0, 1,
7.555555, 3.616162, -538.7563, 1, 0.6862745, 0, 1,
7.59596, 3.616162, -539.6909, 1, 0.6862745, 0, 1,
7.636364, 3.616162, -540.6505, 1, 0.6862745, 0, 1,
7.676768, 3.616162, -541.6351, 1, 0.6862745, 0, 1,
7.717172, 3.616162, -542.6447, 1, 0.6862745, 0, 1,
7.757576, 3.616162, -543.6791, 1, 0.5843138, 0, 1,
7.79798, 3.616162, -544.7386, 1, 0.5843138, 0, 1,
7.838384, 3.616162, -545.8231, 1, 0.5843138, 0, 1,
7.878788, 3.616162, -546.9324, 1, 0.5843138, 0, 1,
7.919192, 3.616162, -548.0668, 1, 0.5843138, 0, 1,
7.959596, 3.616162, -549.2262, 1, 0.4823529, 0, 1,
8, 3.616162, -550.4105, 1, 0.4823529, 0, 1,
4, 3.666667, -553.9545, 1, 0.4823529, 0, 1,
4.040404, 3.666667, -552.7264, 1, 0.4823529, 0, 1,
4.080808, 3.666667, -551.5228, 1, 0.4823529, 0, 1,
4.121212, 3.666667, -550.3433, 1, 0.4823529, 0, 1,
4.161616, 3.666667, -549.1882, 1, 0.4823529, 0, 1,
4.20202, 3.666667, -548.0573, 1, 0.5843138, 0, 1,
4.242424, 3.666667, -546.9507, 1, 0.5843138, 0, 1,
4.282828, 3.666667, -545.8684, 1, 0.5843138, 0, 1,
4.323232, 3.666667, -544.8104, 1, 0.5843138, 0, 1,
4.363636, 3.666667, -543.7766, 1, 0.5843138, 0, 1,
4.40404, 3.666667, -542.7672, 1, 0.5843138, 0, 1,
4.444445, 3.666667, -541.782, 1, 0.6862745, 0, 1,
4.484848, 3.666667, -540.8212, 1, 0.6862745, 0, 1,
4.525252, 3.666667, -539.8846, 1, 0.6862745, 0, 1,
4.565657, 3.666667, -538.9722, 1, 0.6862745, 0, 1,
4.606061, 3.666667, -538.0842, 1, 0.6862745, 0, 1,
4.646465, 3.666667, -537.2205, 1, 0.6862745, 0, 1,
4.686869, 3.666667, -536.381, 1, 0.7921569, 0, 1,
4.727273, 3.666667, -535.5659, 1, 0.7921569, 0, 1,
4.767677, 3.666667, -534.775, 1, 0.7921569, 0, 1,
4.808081, 3.666667, -534.0084, 1, 0.7921569, 0, 1,
4.848485, 3.666667, -533.2661, 1, 0.7921569, 0, 1,
4.888889, 3.666667, -532.548, 1, 0.7921569, 0, 1,
4.929293, 3.666667, -531.8543, 1, 0.7921569, 0, 1,
4.969697, 3.666667, -531.1849, 1, 0.8941177, 0, 1,
5.010101, 3.666667, -530.5397, 1, 0.8941177, 0, 1,
5.050505, 3.666667, -529.9188, 1, 0.8941177, 0, 1,
5.090909, 3.666667, -529.3222, 1, 0.8941177, 0, 1,
5.131313, 3.666667, -528.7499, 1, 0.8941177, 0, 1,
5.171717, 3.666667, -528.2018, 1, 0.8941177, 0, 1,
5.212121, 3.666667, -527.6781, 1, 0.8941177, 0, 1,
5.252525, 3.666667, -527.1786, 1, 0.8941177, 0, 1,
5.292929, 3.666667, -526.7035, 1, 0.8941177, 0, 1,
5.333333, 3.666667, -526.2526, 1, 0.8941177, 0, 1,
5.373737, 3.666667, -525.826, 1, 0.8941177, 0, 1,
5.414141, 3.666667, -525.4236, 1, 0.8941177, 0, 1,
5.454545, 3.666667, -525.0456, 1, 1, 0, 1,
5.494949, 3.666667, -524.6918, 1, 1, 0, 1,
5.535354, 3.666667, -524.3624, 1, 1, 0, 1,
5.575758, 3.666667, -524.0572, 1, 1, 0, 1,
5.616162, 3.666667, -523.7763, 1, 1, 0, 1,
5.656566, 3.666667, -523.5197, 1, 1, 0, 1,
5.69697, 3.666667, -523.2874, 1, 1, 0, 1,
5.737374, 3.666667, -523.0793, 1, 1, 0, 1,
5.777778, 3.666667, -522.8956, 1, 1, 0, 1,
5.818182, 3.666667, -522.7361, 1, 1, 0, 1,
5.858586, 3.666667, -522.6009, 1, 1, 0, 1,
5.89899, 3.666667, -522.49, 1, 1, 0, 1,
5.939394, 3.666667, -522.4034, 1, 1, 0, 1,
5.979798, 3.666667, -522.341, 1, 1, 0, 1,
6.020202, 3.666667, -522.303, 1, 1, 0, 1,
6.060606, 3.666667, -522.2892, 1, 1, 0, 1,
6.10101, 3.666667, -522.2997, 1, 1, 0, 1,
6.141414, 3.666667, -522.3345, 1, 1, 0, 1,
6.181818, 3.666667, -522.3936, 1, 1, 0, 1,
6.222222, 3.666667, -522.477, 1, 1, 0, 1,
6.262626, 3.666667, -522.5847, 1, 1, 0, 1,
6.30303, 3.666667, -522.7166, 1, 1, 0, 1,
6.343434, 3.666667, -522.8729, 1, 1, 0, 1,
6.383838, 3.666667, -523.0533, 1, 1, 0, 1,
6.424242, 3.666667, -523.2582, 1, 1, 0, 1,
6.464646, 3.666667, -523.4872, 1, 1, 0, 1,
6.505051, 3.666667, -523.7406, 1, 1, 0, 1,
6.545455, 3.666667, -524.0182, 1, 1, 0, 1,
6.585859, 3.666667, -524.3202, 1, 1, 0, 1,
6.626263, 3.666667, -524.6464, 1, 1, 0, 1,
6.666667, 3.666667, -524.9969, 1, 1, 0, 1,
6.707071, 3.666667, -525.3718, 1, 1, 0, 1,
6.747475, 3.666667, -525.7708, 1, 0.8941177, 0, 1,
6.787879, 3.666667, -526.1942, 1, 0.8941177, 0, 1,
6.828283, 3.666667, -526.6418, 1, 0.8941177, 0, 1,
6.868687, 3.666667, -527.1138, 1, 0.8941177, 0, 1,
6.909091, 3.666667, -527.61, 1, 0.8941177, 0, 1,
6.949495, 3.666667, -528.1305, 1, 0.8941177, 0, 1,
6.989899, 3.666667, -528.6753, 1, 0.8941177, 0, 1,
7.030303, 3.666667, -529.2443, 1, 0.8941177, 0, 1,
7.070707, 3.666667, -529.8377, 1, 0.8941177, 0, 1,
7.111111, 3.666667, -530.4554, 1, 0.8941177, 0, 1,
7.151515, 3.666667, -531.0973, 1, 0.8941177, 0, 1,
7.191919, 3.666667, -531.7635, 1, 0.7921569, 0, 1,
7.232323, 3.666667, -532.454, 1, 0.7921569, 0, 1,
7.272727, 3.666667, -533.1688, 1, 0.7921569, 0, 1,
7.313131, 3.666667, -533.9078, 1, 0.7921569, 0, 1,
7.353535, 3.666667, -534.6712, 1, 0.7921569, 0, 1,
7.393939, 3.666667, -535.4589, 1, 0.7921569, 0, 1,
7.434343, 3.666667, -536.2708, 1, 0.7921569, 0, 1,
7.474748, 3.666667, -537.107, 1, 0.6862745, 0, 1,
7.515152, 3.666667, -537.9675, 1, 0.6862745, 0, 1,
7.555555, 3.666667, -538.8522, 1, 0.6862745, 0, 1,
7.59596, 3.666667, -539.7613, 1, 0.6862745, 0, 1,
7.636364, 3.666667, -540.6946, 1, 0.6862745, 0, 1,
7.676768, 3.666667, -541.6523, 1, 0.6862745, 0, 1,
7.717172, 3.666667, -542.6342, 1, 0.6862745, 0, 1,
7.757576, 3.666667, -543.6404, 1, 0.5843138, 0, 1,
7.79798, 3.666667, -544.6709, 1, 0.5843138, 0, 1,
7.838384, 3.666667, -545.7256, 1, 0.5843138, 0, 1,
7.878788, 3.666667, -546.8047, 1, 0.5843138, 0, 1,
7.919192, 3.666667, -547.9081, 1, 0.5843138, 0, 1,
7.959596, 3.666667, -549.0357, 1, 0.4823529, 0, 1,
8, 3.666667, -550.1876, 1, 0.4823529, 0, 1,
4, 3.717172, -553.7133, 1, 0.4823529, 0, 1,
4.040404, 3.717172, -552.5184, 1, 0.4823529, 0, 1,
4.080808, 3.717172, -551.3472, 1, 0.4823529, 0, 1,
4.121212, 3.717172, -550.1996, 1, 0.4823529, 0, 1,
4.161616, 3.717172, -549.0756, 1, 0.4823529, 0, 1,
4.20202, 3.717172, -547.9753, 1, 0.5843138, 0, 1,
4.242424, 3.717172, -546.8986, 1, 0.5843138, 0, 1,
4.282828, 3.717172, -545.8455, 1, 0.5843138, 0, 1,
4.323232, 3.717172, -544.816, 1, 0.5843138, 0, 1,
4.363636, 3.717172, -543.8102, 1, 0.5843138, 0, 1,
4.40404, 3.717172, -542.8279, 1, 0.5843138, 0, 1,
4.444445, 3.717172, -541.8694, 1, 0.6862745, 0, 1,
4.484848, 3.717172, -540.9344, 1, 0.6862745, 0, 1,
4.525252, 3.717172, -540.0231, 1, 0.6862745, 0, 1,
4.565657, 3.717172, -539.1354, 1, 0.6862745, 0, 1,
4.606061, 3.717172, -538.2714, 1, 0.6862745, 0, 1,
4.646465, 3.717172, -537.431, 1, 0.6862745, 0, 1,
4.686869, 3.717172, -536.6142, 1, 0.7921569, 0, 1,
4.727273, 3.717172, -535.821, 1, 0.7921569, 0, 1,
4.767677, 3.717172, -535.0515, 1, 0.7921569, 0, 1,
4.808081, 3.717172, -534.3055, 1, 0.7921569, 0, 1,
4.848485, 3.717172, -533.5833, 1, 0.7921569, 0, 1,
4.888889, 3.717172, -532.8846, 1, 0.7921569, 0, 1,
4.929293, 3.717172, -532.2096, 1, 0.7921569, 0, 1,
4.969697, 3.717172, -531.5582, 1, 0.7921569, 0, 1,
5.010101, 3.717172, -530.9305, 1, 0.8941177, 0, 1,
5.050505, 3.717172, -530.3264, 1, 0.8941177, 0, 1,
5.090909, 3.717172, -529.7458, 1, 0.8941177, 0, 1,
5.131313, 3.717172, -529.189, 1, 0.8941177, 0, 1,
5.171717, 3.717172, -528.6557, 1, 0.8941177, 0, 1,
5.212121, 3.717172, -528.1461, 1, 0.8941177, 0, 1,
5.252525, 3.717172, -527.6602, 1, 0.8941177, 0, 1,
5.292929, 3.717172, -527.1978, 1, 0.8941177, 0, 1,
5.333333, 3.717172, -526.759, 1, 0.8941177, 0, 1,
5.373737, 3.717172, -526.3439, 1, 0.8941177, 0, 1,
5.414141, 3.717172, -525.9525, 1, 0.8941177, 0, 1,
5.454545, 3.717172, -525.5847, 1, 0.8941177, 0, 1,
5.494949, 3.717172, -525.2405, 1, 1, 0, 1,
5.535354, 3.717172, -524.9199, 1, 1, 0, 1,
5.575758, 3.717172, -524.6229, 1, 1, 0, 1,
5.616162, 3.717172, -524.3496, 1, 1, 0, 1,
5.656566, 3.717172, -524.0999, 1, 1, 0, 1,
5.69697, 3.717172, -523.8738, 1, 1, 0, 1,
5.737374, 3.717172, -523.6714, 1, 1, 0, 1,
5.777778, 3.717172, -523.4926, 1, 1, 0, 1,
5.818182, 3.717172, -523.3375, 1, 1, 0, 1,
5.858586, 3.717172, -523.2059, 1, 1, 0, 1,
5.89899, 3.717172, -523.098, 1, 1, 0, 1,
5.939394, 3.717172, -523.0137, 1, 1, 0, 1,
5.979798, 3.717172, -522.9531, 1, 1, 0, 1,
6.020202, 3.717172, -522.9161, 1, 1, 0, 1,
6.060606, 3.717172, -522.9026, 1, 1, 0, 1,
6.10101, 3.717172, -522.9129, 1, 1, 0, 1,
6.141414, 3.717172, -522.9468, 1, 1, 0, 1,
6.181818, 3.717172, -523.0043, 1, 1, 0, 1,
6.222222, 3.717172, -523.0854, 1, 1, 0, 1,
6.262626, 3.717172, -523.1901, 1, 1, 0, 1,
6.30303, 3.717172, -523.3185, 1, 1, 0, 1,
6.343434, 3.717172, -523.4705, 1, 1, 0, 1,
6.383838, 3.717172, -523.6462, 1, 1, 0, 1,
6.424242, 3.717172, -523.8455, 1, 1, 0, 1,
6.464646, 3.717172, -524.0684, 1, 1, 0, 1,
6.505051, 3.717172, -524.3149, 1, 1, 0, 1,
6.545455, 3.717172, -524.5851, 1, 1, 0, 1,
6.585859, 3.717172, -524.8788, 1, 1, 0, 1,
6.626263, 3.717172, -525.1963, 1, 1, 0, 1,
6.666667, 3.717172, -525.5373, 1, 0.8941177, 0, 1,
6.707071, 3.717172, -525.902, 1, 0.8941177, 0, 1,
6.747475, 3.717172, -526.2903, 1, 0.8941177, 0, 1,
6.787879, 3.717172, -526.7023, 1, 0.8941177, 0, 1,
6.828283, 3.717172, -527.1378, 1, 0.8941177, 0, 1,
6.868687, 3.717172, -527.597, 1, 0.8941177, 0, 1,
6.909091, 3.717172, -528.0798, 1, 0.8941177, 0, 1,
6.949495, 3.717172, -528.5863, 1, 0.8941177, 0, 1,
6.989899, 3.717172, -529.1164, 1, 0.8941177, 0, 1,
7.030303, 3.717172, -529.6701, 1, 0.8941177, 0, 1,
7.070707, 3.717172, -530.2474, 1, 0.8941177, 0, 1,
7.111111, 3.717172, -530.8484, 1, 0.8941177, 0, 1,
7.151515, 3.717172, -531.473, 1, 0.7921569, 0, 1,
7.191919, 3.717172, -532.1212, 1, 0.7921569, 0, 1,
7.232323, 3.717172, -532.7931, 1, 0.7921569, 0, 1,
7.272727, 3.717172, -533.4886, 1, 0.7921569, 0, 1,
7.313131, 3.717172, -534.2077, 1, 0.7921569, 0, 1,
7.353535, 3.717172, -534.9504, 1, 0.7921569, 0, 1,
7.393939, 3.717172, -535.7169, 1, 0.7921569, 0, 1,
7.434343, 3.717172, -536.5068, 1, 0.7921569, 0, 1,
7.474748, 3.717172, -537.3205, 1, 0.6862745, 0, 1,
7.515152, 3.717172, -538.1578, 1, 0.6862745, 0, 1,
7.555555, 3.717172, -539.0187, 1, 0.6862745, 0, 1,
7.59596, 3.717172, -539.9032, 1, 0.6862745, 0, 1,
7.636364, 3.717172, -540.8113, 1, 0.6862745, 0, 1,
7.676768, 3.717172, -541.7432, 1, 0.6862745, 0, 1,
7.717172, 3.717172, -542.6985, 1, 0.6862745, 0, 1,
7.757576, 3.717172, -543.6776, 1, 0.5843138, 0, 1,
7.79798, 3.717172, -544.6803, 1, 0.5843138, 0, 1,
7.838384, 3.717172, -545.7066, 1, 0.5843138, 0, 1,
7.878788, 3.717172, -546.7565, 1, 0.5843138, 0, 1,
7.919192, 3.717172, -547.8301, 1, 0.5843138, 0, 1,
7.959596, 3.717172, -548.9272, 1, 0.4823529, 0, 1,
8, 3.717172, -550.0481, 1, 0.4823529, 0, 1,
4, 3.767677, -553.5541, 1, 0.4823529, 0, 1,
4.040404, 3.767677, -552.3911, 1, 0.4823529, 0, 1,
4.080808, 3.767677, -551.251, 1, 0.4823529, 0, 1,
4.121212, 3.767677, -550.134, 1, 0.4823529, 0, 1,
4.161616, 3.767677, -549.0399, 1, 0.4823529, 0, 1,
4.20202, 3.767677, -547.9689, 1, 0.5843138, 0, 1,
4.242424, 3.767677, -546.9208, 1, 0.5843138, 0, 1,
4.282828, 3.767677, -545.8958, 1, 0.5843138, 0, 1,
4.323232, 3.767677, -544.8937, 1, 0.5843138, 0, 1,
4.363636, 3.767677, -543.9147, 1, 0.5843138, 0, 1,
4.40404, 3.767677, -542.9587, 1, 0.5843138, 0, 1,
4.444445, 3.767677, -542.0256, 1, 0.6862745, 0, 1,
4.484848, 3.767677, -541.1155, 1, 0.6862745, 0, 1,
4.525252, 3.767677, -540.2285, 1, 0.6862745, 0, 1,
4.565657, 3.767677, -539.3645, 1, 0.6862745, 0, 1,
4.606061, 3.767677, -538.5234, 1, 0.6862745, 0, 1,
4.646465, 3.767677, -537.7054, 1, 0.6862745, 0, 1,
4.686869, 3.767677, -536.9103, 1, 0.7921569, 0, 1,
4.727273, 3.767677, -536.1383, 1, 0.7921569, 0, 1,
4.767677, 3.767677, -535.3892, 1, 0.7921569, 0, 1,
4.808081, 3.767677, -534.6632, 1, 0.7921569, 0, 1,
4.848485, 3.767677, -533.9601, 1, 0.7921569, 0, 1,
4.888889, 3.767677, -533.2802, 1, 0.7921569, 0, 1,
4.929293, 3.767677, -532.6231, 1, 0.7921569, 0, 1,
4.969697, 3.767677, -531.989, 1, 0.7921569, 0, 1,
5.010101, 3.767677, -531.378, 1, 0.7921569, 0, 1,
5.050505, 3.767677, -530.79, 1, 0.8941177, 0, 1,
5.090909, 3.767677, -530.2249, 1, 0.8941177, 0, 1,
5.131313, 3.767677, -529.6829, 1, 0.8941177, 0, 1,
5.171717, 3.767677, -529.1638, 1, 0.8941177, 0, 1,
5.212121, 3.767677, -528.6678, 1, 0.8941177, 0, 1,
5.252525, 3.767677, -528.1948, 1, 0.8941177, 0, 1,
5.292929, 3.767677, -527.7447, 1, 0.8941177, 0, 1,
5.333333, 3.767677, -527.3177, 1, 0.8941177, 0, 1,
5.373737, 3.767677, -526.9136, 1, 0.8941177, 0, 1,
5.414141, 3.767677, -526.5326, 1, 0.8941177, 0, 1,
5.454545, 3.767677, -526.1746, 1, 0.8941177, 0, 1,
5.494949, 3.767677, -525.8395, 1, 0.8941177, 0, 1,
5.535354, 3.767677, -525.5275, 1, 0.8941177, 0, 1,
5.575758, 3.767677, -525.2384, 1, 1, 0, 1,
5.616162, 3.767677, -524.9724, 1, 1, 0, 1,
5.656566, 3.767677, -524.7294, 1, 1, 0, 1,
5.69697, 3.767677, -524.5093, 1, 1, 0, 1,
5.737374, 3.767677, -524.3123, 1, 1, 0, 1,
5.777778, 3.767677, -524.1382, 1, 1, 0, 1,
5.818182, 3.767677, -523.9872, 1, 1, 0, 1,
5.858586, 3.767677, -523.8591, 1, 1, 0, 1,
5.89899, 3.767677, -523.7542, 1, 1, 0, 1,
5.939394, 3.767677, -523.6721, 1, 1, 0, 1,
5.979798, 3.767677, -523.613, 1, 1, 0, 1,
6.020202, 3.767677, -523.577, 1, 1, 0, 1,
6.060606, 3.767677, -523.564, 1, 1, 0, 1,
6.10101, 3.767677, -523.5739, 1, 1, 0, 1,
6.141414, 3.767677, -523.6069, 1, 1, 0, 1,
6.181818, 3.767677, -523.6628, 1, 1, 0, 1,
6.222222, 3.767677, -523.7418, 1, 1, 0, 1,
6.262626, 3.767677, -523.8438, 1, 1, 0, 1,
6.30303, 3.767677, -523.9688, 1, 1, 0, 1,
6.343434, 3.767677, -524.1168, 1, 1, 0, 1,
6.383838, 3.767677, -524.2877, 1, 1, 0, 1,
6.424242, 3.767677, -524.4817, 1, 1, 0, 1,
6.464646, 3.767677, -524.6986, 1, 1, 0, 1,
6.505051, 3.767677, -524.9386, 1, 1, 0, 1,
6.545455, 3.767677, -525.2015, 1, 1, 0, 1,
6.585859, 3.767677, -525.4875, 1, 0.8941177, 0, 1,
6.626263, 3.767677, -525.7965, 1, 0.8941177, 0, 1,
6.666667, 3.767677, -526.1285, 1, 0.8941177, 0, 1,
6.707071, 3.767677, -526.4835, 1, 0.8941177, 0, 1,
6.747475, 3.767677, -526.8614, 1, 0.8941177, 0, 1,
6.787879, 3.767677, -527.2624, 1, 0.8941177, 0, 1,
6.828283, 3.767677, -527.6863, 1, 0.8941177, 0, 1,
6.868687, 3.767677, -528.1333, 1, 0.8941177, 0, 1,
6.909091, 3.767677, -528.6033, 1, 0.8941177, 0, 1,
6.949495, 3.767677, -529.0963, 1, 0.8941177, 0, 1,
6.989899, 3.767677, -529.6122, 1, 0.8941177, 0, 1,
7.030303, 3.767677, -530.1512, 1, 0.8941177, 0, 1,
7.070707, 3.767677, -530.7131, 1, 0.8941177, 0, 1,
7.111111, 3.767677, -531.2981, 1, 0.7921569, 0, 1,
7.151515, 3.767677, -531.9061, 1, 0.7921569, 0, 1,
7.191919, 3.767677, -532.537, 1, 0.7921569, 0, 1,
7.232323, 3.767677, -533.191, 1, 0.7921569, 0, 1,
7.272727, 3.767677, -533.868, 1, 0.7921569, 0, 1,
7.313131, 3.767677, -534.568, 1, 0.7921569, 0, 1,
7.353535, 3.767677, -535.291, 1, 0.7921569, 0, 1,
7.393939, 3.767677, -536.0369, 1, 0.7921569, 0, 1,
7.434343, 3.767677, -536.8059, 1, 0.7921569, 0, 1,
7.474748, 3.767677, -537.5978, 1, 0.6862745, 0, 1,
7.515152, 3.767677, -538.4128, 1, 0.6862745, 0, 1,
7.555555, 3.767677, -539.2508, 1, 0.6862745, 0, 1,
7.59596, 3.767677, -540.1118, 1, 0.6862745, 0, 1,
7.636364, 3.767677, -540.9957, 1, 0.6862745, 0, 1,
7.676768, 3.767677, -541.9027, 1, 0.6862745, 0, 1,
7.717172, 3.767677, -542.8327, 1, 0.5843138, 0, 1,
7.757576, 3.767677, -543.7856, 1, 0.5843138, 0, 1,
7.79798, 3.767677, -544.7617, 1, 0.5843138, 0, 1,
7.838384, 3.767677, -545.7606, 1, 0.5843138, 0, 1,
7.878788, 3.767677, -546.7826, 1, 0.5843138, 0, 1,
7.919192, 3.767677, -547.8276, 1, 0.5843138, 0, 1,
7.959596, 3.767677, -548.8956, 1, 0.4823529, 0, 1,
8, 3.767677, -549.9865, 1, 0.4823529, 0, 1,
4, 3.818182, -553.4717, 1, 0.4823529, 0, 1,
4.040404, 3.818182, -552.3392, 1, 0.4823529, 0, 1,
4.080808, 3.818182, -551.2291, 1, 0.4823529, 0, 1,
4.121212, 3.818182, -550.1414, 1, 0.4823529, 0, 1,
4.161616, 3.818182, -549.0761, 1, 0.4823529, 0, 1,
4.20202, 3.818182, -548.0332, 1, 0.5843138, 0, 1,
4.242424, 3.818182, -547.0127, 1, 0.5843138, 0, 1,
4.282828, 3.818182, -546.0146, 1, 0.5843138, 0, 1,
4.323232, 3.818182, -545.0389, 1, 0.5843138, 0, 1,
4.363636, 3.818182, -544.0856, 1, 0.5843138, 0, 1,
4.40404, 3.818182, -543.1547, 1, 0.5843138, 0, 1,
4.444445, 3.818182, -542.2462, 1, 0.6862745, 0, 1,
4.484848, 3.818182, -541.36, 1, 0.6862745, 0, 1,
4.525252, 3.818182, -540.4963, 1, 0.6862745, 0, 1,
4.565657, 3.818182, -539.6549, 1, 0.6862745, 0, 1,
4.606061, 3.818182, -538.836, 1, 0.6862745, 0, 1,
4.646465, 3.818182, -538.0394, 1, 0.6862745, 0, 1,
4.686869, 3.818182, -537.2653, 1, 0.6862745, 0, 1,
4.727273, 3.818182, -536.5135, 1, 0.7921569, 0, 1,
4.767677, 3.818182, -535.7842, 1, 0.7921569, 0, 1,
4.808081, 3.818182, -535.0772, 1, 0.7921569, 0, 1,
4.848485, 3.818182, -534.3926, 1, 0.7921569, 0, 1,
4.888889, 3.818182, -533.7305, 1, 0.7921569, 0, 1,
4.929293, 3.818182, -533.0907, 1, 0.7921569, 0, 1,
4.969697, 3.818182, -532.4733, 1, 0.7921569, 0, 1,
5.010101, 3.818182, -531.8784, 1, 0.7921569, 0, 1,
5.050505, 3.818182, -531.3057, 1, 0.7921569, 0, 1,
5.090909, 3.818182, -530.7556, 1, 0.8941177, 0, 1,
5.131313, 3.818182, -530.2278, 1, 0.8941177, 0, 1,
5.171717, 3.818182, -529.7224, 1, 0.8941177, 0, 1,
5.212121, 3.818182, -529.2394, 1, 0.8941177, 0, 1,
5.252525, 3.818182, -528.7787, 1, 0.8941177, 0, 1,
5.292929, 3.818182, -528.3405, 1, 0.8941177, 0, 1,
5.333333, 3.818182, -527.9247, 1, 0.8941177, 0, 1,
5.373737, 3.818182, -527.5313, 1, 0.8941177, 0, 1,
5.414141, 3.818182, -527.1603, 1, 0.8941177, 0, 1,
5.454545, 3.818182, -526.8116, 1, 0.8941177, 0, 1,
5.494949, 3.818182, -526.4854, 1, 0.8941177, 0, 1,
5.535354, 3.818182, -526.1816, 1, 0.8941177, 0, 1,
5.575758, 3.818182, -525.9001, 1, 0.8941177, 0, 1,
5.616162, 3.818182, -525.6411, 1, 0.8941177, 0, 1,
5.656566, 3.818182, -525.4044, 1, 1, 0, 1,
5.69697, 3.818182, -525.1902, 1, 1, 0, 1,
5.737374, 3.818182, -524.9983, 1, 1, 0, 1,
5.777778, 3.818182, -524.8289, 1, 1, 0, 1,
5.818182, 3.818182, -524.6818, 1, 1, 0, 1,
5.858586, 3.818182, -524.5571, 1, 1, 0, 1,
5.89899, 3.818182, -524.4548, 1, 1, 0, 1,
5.939394, 3.818182, -524.3749, 1, 1, 0, 1,
5.979798, 3.818182, -524.3174, 1, 1, 0, 1,
6.020202, 3.818182, -524.2823, 1, 1, 0, 1,
6.060606, 3.818182, -524.2697, 1, 1, 0, 1,
6.10101, 3.818182, -524.2794, 1, 1, 0, 1,
6.141414, 3.818182, -524.3115, 1, 1, 0, 1,
6.181818, 3.818182, -524.366, 1, 1, 0, 1,
6.222222, 3.818182, -524.4429, 1, 1, 0, 1,
6.262626, 3.818182, -524.5421, 1, 1, 0, 1,
6.30303, 3.818182, -524.6638, 1, 1, 0, 1,
6.343434, 3.818182, -524.8079, 1, 1, 0, 1,
6.383838, 3.818182, -524.9744, 1, 1, 0, 1,
6.424242, 3.818182, -525.1633, 1, 1, 0, 1,
6.464646, 3.818182, -525.3745, 1, 1, 0, 1,
6.505051, 3.818182, -525.6082, 1, 0.8941177, 0, 1,
6.545455, 3.818182, -525.8642, 1, 0.8941177, 0, 1,
6.585859, 3.818182, -526.1427, 1, 0.8941177, 0, 1,
6.626263, 3.818182, -526.4435, 1, 0.8941177, 0, 1,
6.666667, 3.818182, -526.7668, 1, 0.8941177, 0, 1,
6.707071, 3.818182, -527.1124, 1, 0.8941177, 0, 1,
6.747475, 3.818182, -527.4805, 1, 0.8941177, 0, 1,
6.787879, 3.818182, -527.8708, 1, 0.8941177, 0, 1,
6.828283, 3.818182, -528.2837, 1, 0.8941177, 0, 1,
6.868687, 3.818182, -528.7189, 1, 0.8941177, 0, 1,
6.909091, 3.818182, -529.1765, 1, 0.8941177, 0, 1,
6.949495, 3.818182, -529.6566, 1, 0.8941177, 0, 1,
6.989899, 3.818182, -530.1589, 1, 0.8941177, 0, 1,
7.030303, 3.818182, -530.6838, 1, 0.8941177, 0, 1,
7.070707, 3.818182, -531.231, 1, 0.7921569, 0, 1,
7.111111, 3.818182, -531.8005, 1, 0.7921569, 0, 1,
7.151515, 3.818182, -532.3926, 1, 0.7921569, 0, 1,
7.191919, 3.818182, -533.007, 1, 0.7921569, 0, 1,
7.232323, 3.818182, -533.6437, 1, 0.7921569, 0, 1,
7.272727, 3.818182, -534.3029, 1, 0.7921569, 0, 1,
7.313131, 3.818182, -534.9845, 1, 0.7921569, 0, 1,
7.353535, 3.818182, -535.6885, 1, 0.7921569, 0, 1,
7.393939, 3.818182, -536.4148, 1, 0.7921569, 0, 1,
7.434343, 3.818182, -537.1636, 1, 0.6862745, 0, 1,
7.474748, 3.818182, -537.9348, 1, 0.6862745, 0, 1,
7.515152, 3.818182, -538.7283, 1, 0.6862745, 0, 1,
7.555555, 3.818182, -539.5443, 1, 0.6862745, 0, 1,
7.59596, 3.818182, -540.3826, 1, 0.6862745, 0, 1,
7.636364, 3.818182, -541.2433, 1, 0.6862745, 0, 1,
7.676768, 3.818182, -542.1265, 1, 0.6862745, 0, 1,
7.717172, 3.818182, -543.032, 1, 0.5843138, 0, 1,
7.757576, 3.818182, -543.96, 1, 0.5843138, 0, 1,
7.79798, 3.818182, -544.9103, 1, 0.5843138, 0, 1,
7.838384, 3.818182, -545.883, 1, 0.5843138, 0, 1,
7.878788, 3.818182, -546.8781, 1, 0.5843138, 0, 1,
7.919192, 3.818182, -547.8956, 1, 0.5843138, 0, 1,
7.959596, 3.818182, -548.9355, 1, 0.4823529, 0, 1,
8, 3.818182, -549.9978, 1, 0.4823529, 0, 1,
4, 3.868687, -553.4611, 1, 0.4823529, 0, 1,
4.040404, 3.868687, -552.358, 1, 0.4823529, 0, 1,
4.080808, 3.868687, -551.2767, 1, 0.4823529, 0, 1,
4.121212, 3.868687, -550.2172, 1, 0.4823529, 0, 1,
4.161616, 3.868687, -549.1795, 1, 0.4823529, 0, 1,
4.20202, 3.868687, -548.1637, 1, 0.5843138, 0, 1,
4.242424, 3.868687, -547.1696, 1, 0.5843138, 0, 1,
4.282828, 3.868687, -546.1974, 1, 0.5843138, 0, 1,
4.323232, 3.868687, -545.247, 1, 0.5843138, 0, 1,
4.363636, 3.868687, -544.3184, 1, 0.5843138, 0, 1,
4.40404, 3.868687, -543.4117, 1, 0.5843138, 0, 1,
4.444445, 3.868687, -542.5267, 1, 0.6862745, 0, 1,
4.484848, 3.868687, -541.6636, 1, 0.6862745, 0, 1,
4.525252, 3.868687, -540.8222, 1, 0.6862745, 0, 1,
4.565657, 3.868687, -540.0027, 1, 0.6862745, 0, 1,
4.606061, 3.868687, -539.205, 1, 0.6862745, 0, 1,
4.646465, 3.868687, -538.4291, 1, 0.6862745, 0, 1,
4.686869, 3.868687, -537.675, 1, 0.6862745, 0, 1,
4.727273, 3.868687, -536.9428, 1, 0.7921569, 0, 1,
4.767677, 3.868687, -536.2324, 1, 0.7921569, 0, 1,
4.808081, 3.868687, -535.5438, 1, 0.7921569, 0, 1,
4.848485, 3.868687, -534.877, 1, 0.7921569, 0, 1,
4.888889, 3.868687, -534.2319, 1, 0.7921569, 0, 1,
4.929293, 3.868687, -533.6088, 1, 0.7921569, 0, 1,
4.969697, 3.868687, -533.0074, 1, 0.7921569, 0, 1,
5.010101, 3.868687, -532.4279, 1, 0.7921569, 0, 1,
5.050505, 3.868687, -531.8701, 1, 0.7921569, 0, 1,
5.090909, 3.868687, -531.3342, 1, 0.7921569, 0, 1,
5.131313, 3.868687, -530.8201, 1, 0.8941177, 0, 1,
5.171717, 3.868687, -530.3278, 1, 0.8941177, 0, 1,
5.212121, 3.868687, -529.8573, 1, 0.8941177, 0, 1,
5.252525, 3.868687, -529.4086, 1, 0.8941177, 0, 1,
5.292929, 3.868687, -528.9818, 1, 0.8941177, 0, 1,
5.333333, 3.868687, -528.5768, 1, 0.8941177, 0, 1,
5.373737, 3.868687, -528.1935, 1, 0.8941177, 0, 1,
5.414141, 3.868687, -527.8322, 1, 0.8941177, 0, 1,
5.454545, 3.868687, -527.4926, 1, 0.8941177, 0, 1,
5.494949, 3.868687, -527.1748, 1, 0.8941177, 0, 1,
5.535354, 3.868687, -526.8788, 1, 0.8941177, 0, 1,
5.575758, 3.868687, -526.6047, 1, 0.8941177, 0, 1,
5.616162, 3.868687, -526.3524, 1, 0.8941177, 0, 1,
5.656566, 3.868687, -526.1218, 1, 0.8941177, 0, 1,
5.69697, 3.868687, -525.9131, 1, 0.8941177, 0, 1,
5.737374, 3.868687, -525.7263, 1, 0.8941177, 0, 1,
5.777778, 3.868687, -525.5612, 1, 0.8941177, 0, 1,
5.818182, 3.868687, -525.418, 1, 0.8941177, 0, 1,
5.858586, 3.868687, -525.2965, 1, 1, 0, 1,
5.89899, 3.868687, -525.1969, 1, 1, 0, 1,
5.939394, 3.868687, -525.1191, 1, 1, 0, 1,
5.979798, 3.868687, -525.063, 1, 1, 0, 1,
6.020202, 3.868687, -525.0289, 1, 1, 0, 1,
6.060606, 3.868687, -525.0165, 1, 1, 0, 1,
6.10101, 3.868687, -525.026, 1, 1, 0, 1,
6.141414, 3.868687, -525.0573, 1, 1, 0, 1,
6.181818, 3.868687, -525.1104, 1, 1, 0, 1,
6.222222, 3.868687, -525.1852, 1, 1, 0, 1,
6.262626, 3.868687, -525.2819, 1, 1, 0, 1,
6.30303, 3.868687, -525.4005, 1, 1, 0, 1,
6.343434, 3.868687, -525.5408, 1, 0.8941177, 0, 1,
6.383838, 3.868687, -525.7029, 1, 0.8941177, 0, 1,
6.424242, 3.868687, -525.8869, 1, 0.8941177, 0, 1,
6.464646, 3.868687, -526.0927, 1, 0.8941177, 0, 1,
6.505051, 3.868687, -526.3203, 1, 0.8941177, 0, 1,
6.545455, 3.868687, -526.5697, 1, 0.8941177, 0, 1,
6.585859, 3.868687, -526.8409, 1, 0.8941177, 0, 1,
6.626263, 3.868687, -527.134, 1, 0.8941177, 0, 1,
6.666667, 3.868687, -527.4489, 1, 0.8941177, 0, 1,
6.707071, 3.868687, -527.7855, 1, 0.8941177, 0, 1,
6.747475, 3.868687, -528.144, 1, 0.8941177, 0, 1,
6.787879, 3.868687, -528.5243, 1, 0.8941177, 0, 1,
6.828283, 3.868687, -528.9265, 1, 0.8941177, 0, 1,
6.868687, 3.868687, -529.3503, 1, 0.8941177, 0, 1,
6.909091, 3.868687, -529.7961, 1, 0.8941177, 0, 1,
6.949495, 3.868687, -530.2637, 1, 0.8941177, 0, 1,
6.989899, 3.868687, -530.7531, 1, 0.8941177, 0, 1,
7.030303, 3.868687, -531.2642, 1, 0.7921569, 0, 1,
7.070707, 3.868687, -531.7972, 1, 0.7921569, 0, 1,
7.111111, 3.868687, -532.3521, 1, 0.7921569, 0, 1,
7.151515, 3.868687, -532.9287, 1, 0.7921569, 0, 1,
7.191919, 3.868687, -533.5272, 1, 0.7921569, 0, 1,
7.232323, 3.868687, -534.1475, 1, 0.7921569, 0, 1,
7.272727, 3.868687, -534.7895, 1, 0.7921569, 0, 1,
7.313131, 3.868687, -535.4534, 1, 0.7921569, 0, 1,
7.353535, 3.868687, -536.1391, 1, 0.7921569, 0, 1,
7.393939, 3.868687, -536.8467, 1, 0.7921569, 0, 1,
7.434343, 3.868687, -537.576, 1, 0.6862745, 0, 1,
7.474748, 3.868687, -538.3271, 1, 0.6862745, 0, 1,
7.515152, 3.868687, -539.1001, 1, 0.6862745, 0, 1,
7.555555, 3.868687, -539.8949, 1, 0.6862745, 0, 1,
7.59596, 3.868687, -540.7115, 1, 0.6862745, 0, 1,
7.636364, 3.868687, -541.5499, 1, 0.6862745, 0, 1,
7.676768, 3.868687, -542.4102, 1, 0.6862745, 0, 1,
7.717172, 3.868687, -543.2922, 1, 0.5843138, 0, 1,
7.757576, 3.868687, -544.196, 1, 0.5843138, 0, 1,
7.79798, 3.868687, -545.1217, 1, 0.5843138, 0, 1,
7.838384, 3.868687, -546.0692, 1, 0.5843138, 0, 1,
7.878788, 3.868687, -547.0385, 1, 0.5843138, 0, 1,
7.919192, 3.868687, -548.0297, 1, 0.5843138, 0, 1,
7.959596, 3.868687, -549.0426, 1, 0.4823529, 0, 1,
8, 3.868687, -550.0773, 1, 0.4823529, 0, 1,
4, 3.919192, -553.5177, 1, 0.4823529, 0, 1,
4.040404, 3.919192, -552.4429, 1, 0.4823529, 0, 1,
4.080808, 3.919192, -551.3892, 1, 0.4823529, 0, 1,
4.121212, 3.919192, -550.3569, 1, 0.4823529, 0, 1,
4.161616, 3.919192, -549.3458, 1, 0.4823529, 0, 1,
4.20202, 3.919192, -548.356, 1, 0.5843138, 0, 1,
4.242424, 3.919192, -547.3874, 1, 0.5843138, 0, 1,
4.282828, 3.919192, -546.4401, 1, 0.5843138, 0, 1,
4.323232, 3.919192, -545.514, 1, 0.5843138, 0, 1,
4.363636, 3.919192, -544.6092, 1, 0.5843138, 0, 1,
4.40404, 3.919192, -543.7256, 1, 0.5843138, 0, 1,
4.444445, 3.919192, -542.8633, 1, 0.5843138, 0, 1,
4.484848, 3.919192, -542.0223, 1, 0.6862745, 0, 1,
4.525252, 3.919192, -541.2025, 1, 0.6862745, 0, 1,
4.565657, 3.919192, -540.404, 1, 0.6862745, 0, 1,
4.606061, 3.919192, -539.6267, 1, 0.6862745, 0, 1,
4.646465, 3.919192, -538.8707, 1, 0.6862745, 0, 1,
4.686869, 3.919192, -538.1359, 1, 0.6862745, 0, 1,
4.727273, 3.919192, -537.4224, 1, 0.6862745, 0, 1,
4.767677, 3.919192, -536.7302, 1, 0.7921569, 0, 1,
4.808081, 3.919192, -536.0592, 1, 0.7921569, 0, 1,
4.848485, 3.919192, -535.4095, 1, 0.7921569, 0, 1,
4.888889, 3.919192, -534.781, 1, 0.7921569, 0, 1,
4.929293, 3.919192, -534.1738, 1, 0.7921569, 0, 1,
4.969697, 3.919192, -533.5878, 1, 0.7921569, 0, 1,
5.010101, 3.919192, -533.0231, 1, 0.7921569, 0, 1,
5.050505, 3.919192, -532.4796, 1, 0.7921569, 0, 1,
5.090909, 3.919192, -531.9575, 1, 0.7921569, 0, 1,
5.131313, 3.919192, -531.4565, 1, 0.7921569, 0, 1,
5.171717, 3.919192, -530.9768, 1, 0.8941177, 0, 1,
5.212121, 3.919192, -530.5184, 1, 0.8941177, 0, 1,
5.252525, 3.919192, -530.0812, 1, 0.8941177, 0, 1,
5.292929, 3.919192, -529.6653, 1, 0.8941177, 0, 1,
5.333333, 3.919192, -529.2706, 1, 0.8941177, 0, 1,
5.373737, 3.919192, -528.8972, 1, 0.8941177, 0, 1,
5.414141, 3.919192, -528.5451, 1, 0.8941177, 0, 1,
5.454545, 3.919192, -528.2142, 1, 0.8941177, 0, 1,
5.494949, 3.919192, -527.9045, 1, 0.8941177, 0, 1,
5.535354, 3.919192, -527.6161, 1, 0.8941177, 0, 1,
5.575758, 3.919192, -527.3491, 1, 0.8941177, 0, 1,
5.616162, 3.919192, -527.1032, 1, 0.8941177, 0, 1,
5.656566, 3.919192, -526.8786, 1, 0.8941177, 0, 1,
5.69697, 3.919192, -526.6752, 1, 0.8941177, 0, 1,
5.737374, 3.919192, -526.4931, 1, 0.8941177, 0, 1,
5.777778, 3.919192, -526.3323, 1, 0.8941177, 0, 1,
5.818182, 3.919192, -526.1927, 1, 0.8941177, 0, 1,
5.858586, 3.919192, -526.0743, 1, 0.8941177, 0, 1,
5.89899, 3.919192, -525.9773, 1, 0.8941177, 0, 1,
5.939394, 3.919192, -525.9015, 1, 0.8941177, 0, 1,
5.979798, 3.919192, -525.8469, 1, 0.8941177, 0, 1,
6.020202, 3.919192, -525.8136, 1, 0.8941177, 0, 1,
6.060606, 3.919192, -525.8016, 1, 0.8941177, 0, 1,
6.10101, 3.919192, -525.8108, 1, 0.8941177, 0, 1,
6.141414, 3.919192, -525.8412, 1, 0.8941177, 0, 1,
6.181818, 3.919192, -525.8929, 1, 0.8941177, 0, 1,
6.222222, 3.919192, -525.9659, 1, 0.8941177, 0, 1,
6.262626, 3.919192, -526.0602, 1, 0.8941177, 0, 1,
6.30303, 3.919192, -526.1757, 1, 0.8941177, 0, 1,
6.343434, 3.919192, -526.3124, 1, 0.8941177, 0, 1,
6.383838, 3.919192, -526.4704, 1, 0.8941177, 0, 1,
6.424242, 3.919192, -526.6497, 1, 0.8941177, 0, 1,
6.464646, 3.919192, -526.8502, 1, 0.8941177, 0, 1,
6.505051, 3.919192, -527.072, 1, 0.8941177, 0, 1,
6.545455, 3.919192, -527.315, 1, 0.8941177, 0, 1,
6.585859, 3.919192, -527.5793, 1, 0.8941177, 0, 1,
6.626263, 3.919192, -527.8648, 1, 0.8941177, 0, 1,
6.666667, 3.919192, -528.1716, 1, 0.8941177, 0, 1,
6.707071, 3.919192, -528.4996, 1, 0.8941177, 0, 1,
6.747475, 3.919192, -528.8489, 1, 0.8941177, 0, 1,
6.787879, 3.919192, -529.2195, 1, 0.8941177, 0, 1,
6.828283, 3.919192, -529.6113, 1, 0.8941177, 0, 1,
6.868687, 3.919192, -530.0244, 1, 0.8941177, 0, 1,
6.909091, 3.919192, -530.4587, 1, 0.8941177, 0, 1,
6.949495, 3.919192, -530.9144, 1, 0.8941177, 0, 1,
6.989899, 3.919192, -531.3912, 1, 0.7921569, 0, 1,
7.030303, 3.919192, -531.8893, 1, 0.7921569, 0, 1,
7.070707, 3.919192, -532.4086, 1, 0.7921569, 0, 1,
7.111111, 3.919192, -532.9493, 1, 0.7921569, 0, 1,
7.151515, 3.919192, -533.5112, 1, 0.7921569, 0, 1,
7.191919, 3.919192, -534.0943, 1, 0.7921569, 0, 1,
7.232323, 3.919192, -534.6987, 1, 0.7921569, 0, 1,
7.272727, 3.919192, -535.3243, 1, 0.7921569, 0, 1,
7.313131, 3.919192, -535.9712, 1, 0.7921569, 0, 1,
7.353535, 3.919192, -536.6393, 1, 0.7921569, 0, 1,
7.393939, 3.919192, -537.3287, 1, 0.6862745, 0, 1,
7.434343, 3.919192, -538.0394, 1, 0.6862745, 0, 1,
7.474748, 3.919192, -538.7714, 1, 0.6862745, 0, 1,
7.515152, 3.919192, -539.5245, 1, 0.6862745, 0, 1,
7.555555, 3.919192, -540.299, 1, 0.6862745, 0, 1,
7.59596, 3.919192, -541.0946, 1, 0.6862745, 0, 1,
7.636364, 3.919192, -541.9116, 1, 0.6862745, 0, 1,
7.676768, 3.919192, -542.7498, 1, 0.6862745, 0, 1,
7.717172, 3.919192, -543.6093, 1, 0.5843138, 0, 1,
7.757576, 3.919192, -544.4899, 1, 0.5843138, 0, 1,
7.79798, 3.919192, -545.3919, 1, 0.5843138, 0, 1,
7.838384, 3.919192, -546.3151, 1, 0.5843138, 0, 1,
7.878788, 3.919192, -547.2596, 1, 0.5843138, 0, 1,
7.919192, 3.919192, -548.2254, 1, 0.5843138, 0, 1,
7.959596, 3.919192, -549.2123, 1, 0.4823529, 0, 1,
8, 3.919192, -550.2206, 1, 0.4823529, 0, 1,
4, 3.969697, -553.6374, 1, 0.4823529, 0, 1,
4.040404, 3.969697, -552.5897, 1, 0.4823529, 0, 1,
4.080808, 3.969697, -551.5627, 1, 0.4823529, 0, 1,
4.121212, 3.969697, -550.5565, 1, 0.4823529, 0, 1,
4.161616, 3.969697, -549.571, 1, 0.4823529, 0, 1,
4.20202, 3.969697, -548.6061, 1, 0.4823529, 0, 1,
4.242424, 3.969697, -547.662, 1, 0.5843138, 0, 1,
4.282828, 3.969697, -546.7387, 1, 0.5843138, 0, 1,
4.323232, 3.969697, -545.8361, 1, 0.5843138, 0, 1,
4.363636, 3.969697, -544.9541, 1, 0.5843138, 0, 1,
4.40404, 3.969697, -544.0929, 1, 0.5843138, 0, 1,
4.444445, 3.969697, -543.2524, 1, 0.5843138, 0, 1,
4.484848, 3.969697, -542.4326, 1, 0.6862745, 0, 1,
4.525252, 3.969697, -541.6335, 1, 0.6862745, 0, 1,
4.565657, 3.969697, -540.8552, 1, 0.6862745, 0, 1,
4.606061, 3.969697, -540.0976, 1, 0.6862745, 0, 1,
4.646465, 3.969697, -539.3607, 1, 0.6862745, 0, 1,
4.686869, 3.969697, -538.6445, 1, 0.6862745, 0, 1,
4.727273, 3.969697, -537.949, 1, 0.6862745, 0, 1,
4.767677, 3.969697, -537.2743, 1, 0.6862745, 0, 1,
4.808081, 3.969697, -536.6203, 1, 0.7921569, 0, 1,
4.848485, 3.969697, -535.987, 1, 0.7921569, 0, 1,
4.888889, 3.969697, -535.3744, 1, 0.7921569, 0, 1,
4.929293, 3.969697, -534.7825, 1, 0.7921569, 0, 1,
4.969697, 3.969697, -534.2114, 1, 0.7921569, 0, 1,
5.010101, 3.969697, -533.6609, 1, 0.7921569, 0, 1,
5.050505, 3.969697, -533.1312, 1, 0.7921569, 0, 1,
5.090909, 3.969697, -532.6222, 1, 0.7921569, 0, 1,
5.131313, 3.969697, -532.134, 1, 0.7921569, 0, 1,
5.171717, 3.969697, -531.6664, 1, 0.7921569, 0, 1,
5.212121, 3.969697, -531.2195, 1, 0.7921569, 0, 1,
5.252525, 3.969697, -530.7935, 1, 0.8941177, 0, 1,
5.292929, 3.969697, -530.388, 1, 0.8941177, 0, 1,
5.333333, 3.969697, -530.0034, 1, 0.8941177, 0, 1,
5.373737, 3.969697, -529.6394, 1, 0.8941177, 0, 1,
5.414141, 3.969697, -529.2961, 1, 0.8941177, 0, 1,
5.454545, 3.969697, -528.9736, 1, 0.8941177, 0, 1,
5.494949, 3.969697, -528.6718, 1, 0.8941177, 0, 1,
5.535354, 3.969697, -528.3907, 1, 0.8941177, 0, 1,
5.575758, 3.969697, -528.1304, 1, 0.8941177, 0, 1,
5.616162, 3.969697, -527.8907, 1, 0.8941177, 0, 1,
5.656566, 3.969697, -527.6718, 1, 0.8941177, 0, 1,
5.69697, 3.969697, -527.4736, 1, 0.8941177, 0, 1,
5.737374, 3.969697, -527.2961, 1, 0.8941177, 0, 1,
5.777778, 3.969697, -527.1393, 1, 0.8941177, 0, 1,
5.818182, 3.969697, -527.0032, 1, 0.8941177, 0, 1,
5.858586, 3.969697, -526.8879, 1, 0.8941177, 0, 1,
5.89899, 3.969697, -526.7933, 1, 0.8941177, 0, 1,
5.939394, 3.969697, -526.7194, 1, 0.8941177, 0, 1,
5.979798, 3.969697, -526.6662, 1, 0.8941177, 0, 1,
6.020202, 3.969697, -526.6337, 1, 0.8941177, 0, 1,
6.060606, 3.969697, -526.622, 1, 0.8941177, 0, 1,
6.10101, 3.969697, -526.631, 1, 0.8941177, 0, 1,
6.141414, 3.969697, -526.6606, 1, 0.8941177, 0, 1,
6.181818, 3.969697, -526.7111, 1, 0.8941177, 0, 1,
6.222222, 3.969697, -526.7822, 1, 0.8941177, 0, 1,
6.262626, 3.969697, -526.8741, 1, 0.8941177, 0, 1,
6.30303, 3.969697, -526.9866, 1, 0.8941177, 0, 1,
6.343434, 3.969697, -527.1199, 1, 0.8941177, 0, 1,
6.383838, 3.969697, -527.2739, 1, 0.8941177, 0, 1,
6.424242, 3.969697, -527.4487, 1, 0.8941177, 0, 1,
6.464646, 3.969697, -527.6441, 1, 0.8941177, 0, 1,
6.505051, 3.969697, -527.8603, 1, 0.8941177, 0, 1,
6.545455, 3.969697, -528.0972, 1, 0.8941177, 0, 1,
6.585859, 3.969697, -528.3547, 1, 0.8941177, 0, 1,
6.626263, 3.969697, -528.6331, 1, 0.8941177, 0, 1,
6.666667, 3.969697, -528.9321, 1, 0.8941177, 0, 1,
6.707071, 3.969697, -529.2519, 1, 0.8941177, 0, 1,
6.747475, 3.969697, -529.5923, 1, 0.8941177, 0, 1,
6.787879, 3.969697, -529.9536, 1, 0.8941177, 0, 1,
6.828283, 3.969697, -530.3354, 1, 0.8941177, 0, 1,
6.868687, 3.969697, -530.7381, 1, 0.8941177, 0, 1,
6.909091, 3.969697, -531.1614, 1, 0.8941177, 0, 1,
6.949495, 3.969697, -531.6055, 1, 0.7921569, 0, 1,
6.989899, 3.969697, -532.0703, 1, 0.7921569, 0, 1,
7.030303, 3.969697, -532.5558, 1, 0.7921569, 0, 1,
7.070707, 3.969697, -533.062, 1, 0.7921569, 0, 1,
7.111111, 3.969697, -533.589, 1, 0.7921569, 0, 1,
7.151515, 3.969697, -534.1367, 1, 0.7921569, 0, 1,
7.191919, 3.969697, -534.705, 1, 0.7921569, 0, 1,
7.232323, 3.969697, -535.2941, 1, 0.7921569, 0, 1,
7.272727, 3.969697, -535.9039, 1, 0.7921569, 0, 1,
7.313131, 3.969697, -536.5345, 1, 0.7921569, 0, 1,
7.353535, 3.969697, -537.1857, 1, 0.6862745, 0, 1,
7.393939, 3.969697, -537.8577, 1, 0.6862745, 0, 1,
7.434343, 3.969697, -538.5504, 1, 0.6862745, 0, 1,
7.474748, 3.969697, -539.2639, 1, 0.6862745, 0, 1,
7.515152, 3.969697, -539.998, 1, 0.6862745, 0, 1,
7.555555, 3.969697, -540.7528, 1, 0.6862745, 0, 1,
7.59596, 3.969697, -541.5284, 1, 0.6862745, 0, 1,
7.636364, 3.969697, -542.3247, 1, 0.6862745, 0, 1,
7.676768, 3.969697, -543.1417, 1, 0.5843138, 0, 1,
7.717172, 3.969697, -543.9794, 1, 0.5843138, 0, 1,
7.757576, 3.969697, -544.8379, 1, 0.5843138, 0, 1,
7.79798, 3.969697, -545.717, 1, 0.5843138, 0, 1,
7.838384, 3.969697, -546.6169, 1, 0.5843138, 0, 1,
7.878788, 3.969697, -547.5375, 1, 0.5843138, 0, 1,
7.919192, 3.969697, -548.4788, 1, 0.5843138, 0, 1,
7.959596, 3.969697, -549.4409, 1, 0.4823529, 0, 1,
8, 3.969697, -550.4236, 1, 0.4823529, 0, 1,
4, 4.020202, -553.8161, 1, 0.4823529, 0, 1,
4.040404, 4.020202, -552.7946, 1, 0.4823529, 0, 1,
4.080808, 4.020202, -551.7933, 1, 0.4823529, 0, 1,
4.121212, 4.020202, -550.8121, 1, 0.4823529, 0, 1,
4.161616, 4.020202, -549.8512, 1, 0.4823529, 0, 1,
4.20202, 4.020202, -548.9105, 1, 0.4823529, 0, 1,
4.242424, 4.020202, -547.99, 1, 0.5843138, 0, 1,
4.282828, 4.020202, -547.0897, 1, 0.5843138, 0, 1,
4.323232, 4.020202, -546.2095, 1, 0.5843138, 0, 1,
4.363636, 4.020202, -545.3496, 1, 0.5843138, 0, 1,
4.40404, 4.020202, -544.5099, 1, 0.5843138, 0, 1,
4.444445, 4.020202, -543.6904, 1, 0.5843138, 0, 1,
4.484848, 4.020202, -542.8911, 1, 0.5843138, 0, 1,
4.525252, 4.020202, -542.112, 1, 0.6862745, 0, 1,
4.565657, 4.020202, -541.3531, 1, 0.6862745, 0, 1,
4.606061, 4.020202, -540.6144, 1, 0.6862745, 0, 1,
4.646465, 4.020202, -539.8959, 1, 0.6862745, 0, 1,
4.686869, 4.020202, -539.1976, 1, 0.6862745, 0, 1,
4.727273, 4.020202, -538.5195, 1, 0.6862745, 0, 1,
4.767677, 4.020202, -537.8616, 1, 0.6862745, 0, 1,
4.808081, 4.020202, -537.2239, 1, 0.6862745, 0, 1,
4.848485, 4.020202, -536.6064, 1, 0.7921569, 0, 1,
4.888889, 4.020202, -536.0091, 1, 0.7921569, 0, 1,
4.929293, 4.020202, -535.432, 1, 0.7921569, 0, 1,
4.969697, 4.020202, -534.8751, 1, 0.7921569, 0, 1,
5.010101, 4.020202, -534.3384, 1, 0.7921569, 0, 1,
5.050505, 4.020202, -533.8219, 1, 0.7921569, 0, 1,
5.090909, 4.020202, -533.3256, 1, 0.7921569, 0, 1,
5.131313, 4.020202, -532.8495, 1, 0.7921569, 0, 1,
5.171717, 4.020202, -532.3937, 1, 0.7921569, 0, 1,
5.212121, 4.020202, -531.958, 1, 0.7921569, 0, 1,
5.252525, 4.020202, -531.5425, 1, 0.7921569, 0, 1,
5.292929, 4.020202, -531.1472, 1, 0.8941177, 0, 1,
5.333333, 4.020202, -530.7722, 1, 0.8941177, 0, 1,
5.373737, 4.020202, -530.4173, 1, 0.8941177, 0, 1,
5.414141, 4.020202, -530.0826, 1, 0.8941177, 0, 1,
5.454545, 4.020202, -529.7681, 1, 0.8941177, 0, 1,
5.494949, 4.020202, -529.4739, 1, 0.8941177, 0, 1,
5.535354, 4.020202, -529.1998, 1, 0.8941177, 0, 1,
5.575758, 4.020202, -528.9459, 1, 0.8941177, 0, 1,
5.616162, 4.020202, -528.7123, 1, 0.8941177, 0, 1,
5.656566, 4.020202, -528.4988, 1, 0.8941177, 0, 1,
5.69697, 4.020202, -528.3055, 1, 0.8941177, 0, 1,
5.737374, 4.020202, -528.1324, 1, 0.8941177, 0, 1,
5.777778, 4.020202, -527.9796, 1, 0.8941177, 0, 1,
5.818182, 4.020202, -527.8469, 1, 0.8941177, 0, 1,
5.858586, 4.020202, -527.7345, 1, 0.8941177, 0, 1,
5.89899, 4.020202, -527.6422, 1, 0.8941177, 0, 1,
5.939394, 4.020202, -527.5702, 1, 0.8941177, 0, 1,
5.979798, 4.020202, -527.5183, 1, 0.8941177, 0, 1,
6.020202, 4.020202, -527.4867, 1, 0.8941177, 0, 1,
6.060606, 4.020202, -527.4752, 1, 0.8941177, 0, 1,
6.10101, 4.020202, -527.4839, 1, 0.8941177, 0, 1,
6.141414, 4.020202, -527.5129, 1, 0.8941177, 0, 1,
6.181818, 4.020202, -527.5621, 1, 0.8941177, 0, 1,
6.222222, 4.020202, -527.6315, 1, 0.8941177, 0, 1,
6.262626, 4.020202, -527.721, 1, 0.8941177, 0, 1,
6.30303, 4.020202, -527.8307, 1, 0.8941177, 0, 1,
6.343434, 4.020202, -527.9607, 1, 0.8941177, 0, 1,
6.383838, 4.020202, -528.1109, 1, 0.8941177, 0, 1,
6.424242, 4.020202, -528.2812, 1, 0.8941177, 0, 1,
6.464646, 4.020202, -528.4718, 1, 0.8941177, 0, 1,
6.505051, 4.020202, -528.6826, 1, 0.8941177, 0, 1,
6.545455, 4.020202, -528.9136, 1, 0.8941177, 0, 1,
6.585859, 4.020202, -529.1647, 1, 0.8941177, 0, 1,
6.626263, 4.020202, -529.4361, 1, 0.8941177, 0, 1,
6.666667, 4.020202, -529.7277, 1, 0.8941177, 0, 1,
6.707071, 4.020202, -530.0394, 1, 0.8941177, 0, 1,
6.747475, 4.020202, -530.3714, 1, 0.8941177, 0, 1,
6.787879, 4.020202, -530.7236, 1, 0.8941177, 0, 1,
6.828283, 4.020202, -531.0959, 1, 0.8941177, 0, 1,
6.868687, 4.020202, -531.4885, 1, 0.7921569, 0, 1,
6.909091, 4.020202, -531.9013, 1, 0.7921569, 0, 1,
6.949495, 4.020202, -532.3343, 1, 0.7921569, 0, 1,
6.989899, 4.020202, -532.7875, 1, 0.7921569, 0, 1,
7.030303, 4.020202, -533.2609, 1, 0.7921569, 0, 1,
7.070707, 4.020202, -533.7545, 1, 0.7921569, 0, 1,
7.111111, 4.020202, -534.2682, 1, 0.7921569, 0, 1,
7.151515, 4.020202, -534.8022, 1, 0.7921569, 0, 1,
7.191919, 4.020202, -535.3564, 1, 0.7921569, 0, 1,
7.232323, 4.020202, -535.9308, 1, 0.7921569, 0, 1,
7.272727, 4.020202, -536.5255, 1, 0.7921569, 0, 1,
7.313131, 4.020202, -537.1402, 1, 0.6862745, 0, 1,
7.353535, 4.020202, -537.7752, 1, 0.6862745, 0, 1,
7.393939, 4.020202, -538.4304, 1, 0.6862745, 0, 1,
7.434343, 4.020202, -539.1058, 1, 0.6862745, 0, 1,
7.474748, 4.020202, -539.8015, 1, 0.6862745, 0, 1,
7.515152, 4.020202, -540.5172, 1, 0.6862745, 0, 1,
7.555555, 4.020202, -541.2532, 1, 0.6862745, 0, 1,
7.59596, 4.020202, -542.0095, 1, 0.6862745, 0, 1,
7.636364, 4.020202, -542.7858, 1, 0.5843138, 0, 1,
7.676768, 4.020202, -543.5825, 1, 0.5843138, 0, 1,
7.717172, 4.020202, -544.3993, 1, 0.5843138, 0, 1,
7.757576, 4.020202, -545.2363, 1, 0.5843138, 0, 1,
7.79798, 4.020202, -546.0935, 1, 0.5843138, 0, 1,
7.838384, 4.020202, -546.9709, 1, 0.5843138, 0, 1,
7.878788, 4.020202, -547.8685, 1, 0.5843138, 0, 1,
7.919192, 4.020202, -548.7864, 1, 0.4823529, 0, 1,
7.959596, 4.020202, -549.7244, 1, 0.4823529, 0, 1,
8, 4.020202, -550.6826, 1, 0.4823529, 0, 1,
4, 4.070707, -554.0502, 1, 0.4823529, 0, 1,
4.040404, 4.070707, -553.0538, 1, 0.4823529, 0, 1,
4.080808, 4.070707, -552.0772, 1, 0.4823529, 0, 1,
4.121212, 4.070707, -551.1202, 1, 0.4823529, 0, 1,
4.161616, 4.070707, -550.183, 1, 0.4823529, 0, 1,
4.20202, 4.070707, -549.2655, 1, 0.4823529, 0, 1,
4.242424, 4.070707, -548.3677, 1, 0.5843138, 0, 1,
4.282828, 4.070707, -547.4896, 1, 0.5843138, 0, 1,
4.323232, 4.070707, -546.6312, 1, 0.5843138, 0, 1,
4.363636, 4.070707, -545.7925, 1, 0.5843138, 0, 1,
4.40404, 4.070707, -544.9734, 1, 0.5843138, 0, 1,
4.444445, 4.070707, -544.1741, 1, 0.5843138, 0, 1,
4.484848, 4.070707, -543.3946, 1, 0.5843138, 0, 1,
4.525252, 4.070707, -542.6346, 1, 0.6862745, 0, 1,
4.565657, 4.070707, -541.8945, 1, 0.6862745, 0, 1,
4.606061, 4.070707, -541.174, 1, 0.6862745, 0, 1,
4.646465, 4.070707, -540.4732, 1, 0.6862745, 0, 1,
4.686869, 4.070707, -539.7921, 1, 0.6862745, 0, 1,
4.727273, 4.070707, -539.1307, 1, 0.6862745, 0, 1,
4.767677, 4.070707, -538.4891, 1, 0.6862745, 0, 1,
4.808081, 4.070707, -537.8671, 1, 0.6862745, 0, 1,
4.848485, 4.070707, -537.2648, 1, 0.6862745, 0, 1,
4.888889, 4.070707, -536.6823, 1, 0.7921569, 0, 1,
4.929293, 4.070707, -536.1194, 1, 0.7921569, 0, 1,
4.969697, 4.070707, -535.5762, 1, 0.7921569, 0, 1,
5.010101, 4.070707, -535.0528, 1, 0.7921569, 0, 1,
5.050505, 4.070707, -534.5491, 1, 0.7921569, 0, 1,
5.090909, 4.070707, -534.065, 1, 0.7921569, 0, 1,
5.131313, 4.070707, -533.6006, 1, 0.7921569, 0, 1,
5.171717, 4.070707, -533.156, 1, 0.7921569, 0, 1,
5.212121, 4.070707, -532.7311, 1, 0.7921569, 0, 1,
5.252525, 4.070707, -532.3259, 1, 0.7921569, 0, 1,
5.292929, 4.070707, -531.9403, 1, 0.7921569, 0, 1,
5.333333, 4.070707, -531.5745, 1, 0.7921569, 0, 1,
5.373737, 4.070707, -531.2283, 1, 0.7921569, 0, 1,
5.414141, 4.070707, -530.9019, 1, 0.8941177, 0, 1,
5.454545, 4.070707, -530.5952, 1, 0.8941177, 0, 1,
5.494949, 4.070707, -530.3082, 1, 0.8941177, 0, 1,
5.535354, 4.070707, -530.0409, 1, 0.8941177, 0, 1,
5.575758, 4.070707, -529.7933, 1, 0.8941177, 0, 1,
5.616162, 4.070707, -529.5654, 1, 0.8941177, 0, 1,
5.656566, 4.070707, -529.3572, 1, 0.8941177, 0, 1,
5.69697, 4.070707, -529.1687, 1, 0.8941177, 0, 1,
5.737374, 4.070707, -528.9999, 1, 0.8941177, 0, 1,
5.777778, 4.070707, -528.8508, 1, 0.8941177, 0, 1,
5.818182, 4.070707, -528.7214, 1, 0.8941177, 0, 1,
5.858586, 4.070707, -528.6118, 1, 0.8941177, 0, 1,
5.89899, 4.070707, -528.5217, 1, 0.8941177, 0, 1,
5.939394, 4.070707, -528.4515, 1, 0.8941177, 0, 1,
5.979798, 4.070707, -528.4009, 1, 0.8941177, 0, 1,
6.020202, 4.070707, -528.37, 1, 0.8941177, 0, 1,
6.060606, 4.070707, -528.3588, 1, 0.8941177, 0, 1,
6.10101, 4.070707, -528.3674, 1, 0.8941177, 0, 1,
6.141414, 4.070707, -528.3956, 1, 0.8941177, 0, 1,
6.181818, 4.070707, -528.4435, 1, 0.8941177, 0, 1,
6.222222, 4.070707, -528.5112, 1, 0.8941177, 0, 1,
6.262626, 4.070707, -528.5986, 1, 0.8941177, 0, 1,
6.30303, 4.070707, -528.7056, 1, 0.8941177, 0, 1,
6.343434, 4.070707, -528.8324, 1, 0.8941177, 0, 1,
6.383838, 4.070707, -528.9788, 1, 0.8941177, 0, 1,
6.424242, 4.070707, -529.145, 1, 0.8941177, 0, 1,
6.464646, 4.070707, -529.3309, 1, 0.8941177, 0, 1,
6.505051, 4.070707, -529.5364, 1, 0.8941177, 0, 1,
6.545455, 4.070707, -529.7617, 1, 0.8941177, 0, 1,
6.585859, 4.070707, -530.0067, 1, 0.8941177, 0, 1,
6.626263, 4.070707, -530.2714, 1, 0.8941177, 0, 1,
6.666667, 4.070707, -530.5557, 1, 0.8941177, 0, 1,
6.707071, 4.070707, -530.8598, 1, 0.8941177, 0, 1,
6.747475, 4.070707, -531.1836, 1, 0.8941177, 0, 1,
6.787879, 4.070707, -531.5271, 1, 0.7921569, 0, 1,
6.828283, 4.070707, -531.8903, 1, 0.7921569, 0, 1,
6.868687, 4.070707, -532.2732, 1, 0.7921569, 0, 1,
6.909091, 4.070707, -532.6758, 1, 0.7921569, 0, 1,
6.949495, 4.070707, -533.0981, 1, 0.7921569, 0, 1,
6.989899, 4.070707, -533.5401, 1, 0.7921569, 0, 1,
7.030303, 4.070707, -534.0018, 1, 0.7921569, 0, 1,
7.070707, 4.070707, -534.4833, 1, 0.7921569, 0, 1,
7.111111, 4.070707, -534.9844, 1, 0.7921569, 0, 1,
7.151515, 4.070707, -535.5052, 1, 0.7921569, 0, 1,
7.191919, 4.070707, -536.0457, 1, 0.7921569, 0, 1,
7.232323, 4.070707, -536.606, 1, 0.7921569, 0, 1,
7.272727, 4.070707, -537.1859, 1, 0.6862745, 0, 1,
7.313131, 4.070707, -537.7855, 1, 0.6862745, 0, 1,
7.353535, 4.070707, -538.4048, 1, 0.6862745, 0, 1,
7.393939, 4.070707, -539.0439, 1, 0.6862745, 0, 1,
7.434343, 4.070707, -539.7026, 1, 0.6862745, 0, 1,
7.474748, 4.070707, -540.3811, 1, 0.6862745, 0, 1,
7.515152, 4.070707, -541.0792, 1, 0.6862745, 0, 1,
7.555555, 4.070707, -541.7971, 1, 0.6862745, 0, 1,
7.59596, 4.070707, -542.5347, 1, 0.6862745, 0, 1,
7.636364, 4.070707, -543.2919, 1, 0.5843138, 0, 1,
7.676768, 4.070707, -544.0689, 1, 0.5843138, 0, 1,
7.717172, 4.070707, -544.8655, 1, 0.5843138, 0, 1,
7.757576, 4.070707, -545.6819, 1, 0.5843138, 0, 1,
7.79798, 4.070707, -546.518, 1, 0.5843138, 0, 1,
7.838384, 4.070707, -547.3738, 1, 0.5843138, 0, 1,
7.878788, 4.070707, -548.2493, 1, 0.5843138, 0, 1,
7.919192, 4.070707, -549.1445, 1, 0.4823529, 0, 1,
7.959596, 4.070707, -550.0593, 1, 0.4823529, 0, 1,
8, 4.070707, -550.994, 1, 0.4823529, 0, 1,
4, 4.121212, -554.3361, 1, 0.3764706, 0, 1,
4.040404, 4.121212, -553.3641, 1, 0.4823529, 0, 1,
4.080808, 4.121212, -552.4112, 1, 0.4823529, 0, 1,
4.121212, 4.121212, -551.4776, 1, 0.4823529, 0, 1,
4.161616, 4.121212, -550.5632, 1, 0.4823529, 0, 1,
4.20202, 4.121212, -549.668, 1, 0.4823529, 0, 1,
4.242424, 4.121212, -548.7921, 1, 0.4823529, 0, 1,
4.282828, 4.121212, -547.9354, 1, 0.5843138, 0, 1,
4.323232, 4.121212, -547.0978, 1, 0.5843138, 0, 1,
4.363636, 4.121212, -546.2796, 1, 0.5843138, 0, 1,
4.40404, 4.121212, -545.4805, 1, 0.5843138, 0, 1,
4.444445, 4.121212, -544.7007, 1, 0.5843138, 0, 1,
4.484848, 4.121212, -543.9401, 1, 0.5843138, 0, 1,
4.525252, 4.121212, -543.1987, 1, 0.5843138, 0, 1,
4.565657, 4.121212, -542.4766, 1, 0.6862745, 0, 1,
4.606061, 4.121212, -541.7736, 1, 0.6862745, 0, 1,
4.646465, 4.121212, -541.0899, 1, 0.6862745, 0, 1,
4.686869, 4.121212, -540.4254, 1, 0.6862745, 0, 1,
4.727273, 4.121212, -539.7802, 1, 0.6862745, 0, 1,
4.767677, 4.121212, -539.1541, 1, 0.6862745, 0, 1,
4.808081, 4.121212, -538.5473, 1, 0.6862745, 0, 1,
4.848485, 4.121212, -537.9597, 1, 0.6862745, 0, 1,
4.888889, 4.121212, -537.3913, 1, 0.6862745, 0, 1,
4.929293, 4.121212, -536.8422, 1, 0.7921569, 0, 1,
4.969697, 4.121212, -536.3123, 1, 0.7921569, 0, 1,
5.010101, 4.121212, -535.8015, 1, 0.7921569, 0, 1,
5.050505, 4.121212, -535.3101, 1, 0.7921569, 0, 1,
5.090909, 4.121212, -534.8378, 1, 0.7921569, 0, 1,
5.131313, 4.121212, -534.3848, 1, 0.7921569, 0, 1,
5.171717, 4.121212, -533.9509, 1, 0.7921569, 0, 1,
5.212121, 4.121212, -533.5364, 1, 0.7921569, 0, 1,
5.252525, 4.121212, -533.141, 1, 0.7921569, 0, 1,
5.292929, 4.121212, -532.7648, 1, 0.7921569, 0, 1,
5.333333, 4.121212, -532.408, 1, 0.7921569, 0, 1,
5.373737, 4.121212, -532.0703, 1, 0.7921569, 0, 1,
5.414141, 4.121212, -531.7518, 1, 0.7921569, 0, 1,
5.454545, 4.121212, -531.4525, 1, 0.7921569, 0, 1,
5.494949, 4.121212, -531.1725, 1, 0.8941177, 0, 1,
5.535354, 4.121212, -530.9117, 1, 0.8941177, 0, 1,
5.575758, 4.121212, -530.6701, 1, 0.8941177, 0, 1,
5.616162, 4.121212, -530.4478, 1, 0.8941177, 0, 1,
5.656566, 4.121212, -530.2446, 1, 0.8941177, 0, 1,
5.69697, 4.121212, -530.0607, 1, 0.8941177, 0, 1,
5.737374, 4.121212, -529.8961, 1, 0.8941177, 0, 1,
5.777778, 4.121212, -529.7506, 1, 0.8941177, 0, 1,
5.818182, 4.121212, -529.6244, 1, 0.8941177, 0, 1,
5.858586, 4.121212, -529.5173, 1, 0.8941177, 0, 1,
5.89899, 4.121212, -529.4296, 1, 0.8941177, 0, 1,
5.939394, 4.121212, -529.361, 1, 0.8941177, 0, 1,
5.979798, 4.121212, -529.3116, 1, 0.8941177, 0, 1,
6.020202, 4.121212, -529.2816, 1, 0.8941177, 0, 1,
6.060606, 4.121212, -529.2706, 1, 0.8941177, 0, 1,
6.10101, 4.121212, -529.279, 1, 0.8941177, 0, 1,
6.141414, 4.121212, -529.3065, 1, 0.8941177, 0, 1,
6.181818, 4.121212, -529.3533, 1, 0.8941177, 0, 1,
6.222222, 4.121212, -529.4193, 1, 0.8941177, 0, 1,
6.262626, 4.121212, -529.5045, 1, 0.8941177, 0, 1,
6.30303, 4.121212, -529.6089, 1, 0.8941177, 0, 1,
6.343434, 4.121212, -529.7326, 1, 0.8941177, 0, 1,
6.383838, 4.121212, -529.8755, 1, 0.8941177, 0, 1,
6.424242, 4.121212, -530.0377, 1, 0.8941177, 0, 1,
6.464646, 4.121212, -530.219, 1, 0.8941177, 0, 1,
6.505051, 4.121212, -530.4196, 1, 0.8941177, 0, 1,
6.545455, 4.121212, -530.6393, 1, 0.8941177, 0, 1,
6.585859, 4.121212, -530.8784, 1, 0.8941177, 0, 1,
6.626263, 4.121212, -531.1365, 1, 0.8941177, 0, 1,
6.666667, 4.121212, -531.414, 1, 0.7921569, 0, 1,
6.707071, 4.121212, -531.7107, 1, 0.7921569, 0, 1,
6.747475, 4.121212, -532.0266, 1, 0.7921569, 0, 1,
6.787879, 4.121212, -532.3617, 1, 0.7921569, 0, 1,
6.828283, 4.121212, -532.7161, 1, 0.7921569, 0, 1,
6.868687, 4.121212, -533.0897, 1, 0.7921569, 0, 1,
6.909091, 4.121212, -533.4824, 1, 0.7921569, 0, 1,
6.949495, 4.121212, -533.8945, 1, 0.7921569, 0, 1,
6.989899, 4.121212, -534.3257, 1, 0.7921569, 0, 1,
7.030303, 4.121212, -534.7762, 1, 0.7921569, 0, 1,
7.070707, 4.121212, -535.2458, 1, 0.7921569, 0, 1,
7.111111, 4.121212, -535.7347, 1, 0.7921569, 0, 1,
7.151515, 4.121212, -536.2429, 1, 0.7921569, 0, 1,
7.191919, 4.121212, -536.7703, 1, 0.7921569, 0, 1,
7.232323, 4.121212, -537.3168, 1, 0.6862745, 0, 1,
7.272727, 4.121212, -537.8826, 1, 0.6862745, 0, 1,
7.313131, 4.121212, -538.4677, 1, 0.6862745, 0, 1,
7.353535, 4.121212, -539.072, 1, 0.6862745, 0, 1,
7.393939, 4.121212, -539.6954, 1, 0.6862745, 0, 1,
7.434343, 4.121212, -540.3381, 1, 0.6862745, 0, 1,
7.474748, 4.121212, -541, 1, 0.6862745, 0, 1,
7.515152, 4.121212, -541.6812, 1, 0.6862745, 0, 1,
7.555555, 4.121212, -542.3815, 1, 0.6862745, 0, 1,
7.59596, 4.121212, -543.1011, 1, 0.5843138, 0, 1,
7.636364, 4.121212, -543.84, 1, 0.5843138, 0, 1,
7.676768, 4.121212, -544.598, 1, 0.5843138, 0, 1,
7.717172, 4.121212, -545.3752, 1, 0.5843138, 0, 1,
7.757576, 4.121212, -546.1718, 1, 0.5843138, 0, 1,
7.79798, 4.121212, -546.9874, 1, 0.5843138, 0, 1,
7.838384, 4.121212, -547.8224, 1, 0.5843138, 0, 1,
7.878788, 4.121212, -548.6765, 1, 0.4823529, 0, 1,
7.919192, 4.121212, -549.5499, 1, 0.4823529, 0, 1,
7.959596, 4.121212, -550.4425, 1, 0.4823529, 0, 1,
8, 4.121212, -551.3544, 1, 0.4823529, 0, 1,
4, 4.171717, -554.6707, 1, 0.3764706, 0, 1,
4.040404, 4.171717, -553.722, 1, 0.4823529, 0, 1,
4.080808, 4.171717, -552.7922, 1, 0.4823529, 0, 1,
4.121212, 4.171717, -551.881, 1, 0.4823529, 0, 1,
4.161616, 4.171717, -550.9886, 1, 0.4823529, 0, 1,
4.20202, 4.171717, -550.115, 1, 0.4823529, 0, 1,
4.242424, 4.171717, -549.2601, 1, 0.4823529, 0, 1,
4.282828, 4.171717, -548.424, 1, 0.5843138, 0, 1,
4.323232, 4.171717, -547.6067, 1, 0.5843138, 0, 1,
4.363636, 4.171717, -546.8081, 1, 0.5843138, 0, 1,
4.40404, 4.171717, -546.0283, 1, 0.5843138, 0, 1,
4.444445, 4.171717, -545.2672, 1, 0.5843138, 0, 1,
4.484848, 4.171717, -544.5249, 1, 0.5843138, 0, 1,
4.525252, 4.171717, -543.8014, 1, 0.5843138, 0, 1,
4.565657, 4.171717, -543.0966, 1, 0.5843138, 0, 1,
4.606061, 4.171717, -542.4106, 1, 0.6862745, 0, 1,
4.646465, 4.171717, -541.7433, 1, 0.6862745, 0, 1,
4.686869, 4.171717, -541.0948, 1, 0.6862745, 0, 1,
4.727273, 4.171717, -540.4651, 1, 0.6862745, 0, 1,
4.767677, 4.171717, -539.8541, 1, 0.6862745, 0, 1,
4.808081, 4.171717, -539.2619, 1, 0.6862745, 0, 1,
4.848485, 4.171717, -538.6884, 1, 0.6862745, 0, 1,
4.888889, 4.171717, -538.1337, 1, 0.6862745, 0, 1,
4.929293, 4.171717, -537.5978, 1, 0.6862745, 0, 1,
4.969697, 4.171717, -537.0806, 1, 0.6862745, 0, 1,
5.010101, 4.171717, -536.5822, 1, 0.7921569, 0, 1,
5.050505, 4.171717, -536.1025, 1, 0.7921569, 0, 1,
5.090909, 4.171717, -535.6417, 1, 0.7921569, 0, 1,
5.131313, 4.171717, -535.1995, 1, 0.7921569, 0, 1,
5.171717, 4.171717, -534.7762, 1, 0.7921569, 0, 1,
5.212121, 4.171717, -534.3716, 1, 0.7921569, 0, 1,
5.252525, 4.171717, -533.9857, 1, 0.7921569, 0, 1,
5.292929, 4.171717, -533.6186, 1, 0.7921569, 0, 1,
5.333333, 4.171717, -533.2703, 1, 0.7921569, 0, 1,
5.373737, 4.171717, -532.9407, 1, 0.7921569, 0, 1,
5.414141, 4.171717, -532.6299, 1, 0.7921569, 0, 1,
5.454545, 4.171717, -532.3379, 1, 0.7921569, 0, 1,
5.494949, 4.171717, -532.0646, 1, 0.7921569, 0, 1,
5.535354, 4.171717, -531.8101, 1, 0.7921569, 0, 1,
5.575758, 4.171717, -531.5743, 1, 0.7921569, 0, 1,
5.616162, 4.171717, -531.3573, 1, 0.7921569, 0, 1,
5.656566, 4.171717, -531.1591, 1, 0.8941177, 0, 1,
5.69697, 4.171717, -530.9796, 1, 0.8941177, 0, 1,
5.737374, 4.171717, -530.8188, 1, 0.8941177, 0, 1,
5.777778, 4.171717, -530.6769, 1, 0.8941177, 0, 1,
5.818182, 4.171717, -530.5537, 1, 0.8941177, 0, 1,
5.858586, 4.171717, -530.4493, 1, 0.8941177, 0, 1,
5.89899, 4.171717, -530.3636, 1, 0.8941177, 0, 1,
5.939394, 4.171717, -530.2967, 1, 0.8941177, 0, 1,
5.979798, 4.171717, -530.2485, 1, 0.8941177, 0, 1,
6.020202, 4.171717, -530.2191, 1, 0.8941177, 0, 1,
6.060606, 4.171717, -530.2085, 1, 0.8941177, 0, 1,
6.10101, 4.171717, -530.2166, 1, 0.8941177, 0, 1,
6.141414, 4.171717, -530.2435, 1, 0.8941177, 0, 1,
6.181818, 4.171717, -530.2892, 1, 0.8941177, 0, 1,
6.222222, 4.171717, -530.3536, 1, 0.8941177, 0, 1,
6.262626, 4.171717, -530.4368, 1, 0.8941177, 0, 1,
6.30303, 4.171717, -530.5387, 1, 0.8941177, 0, 1,
6.343434, 4.171717, -530.6594, 1, 0.8941177, 0, 1,
6.383838, 4.171717, -530.7988, 1, 0.8941177, 0, 1,
6.424242, 4.171717, -530.957, 1, 0.8941177, 0, 1,
6.464646, 4.171717, -531.134, 1, 0.8941177, 0, 1,
6.505051, 4.171717, -531.3298, 1, 0.7921569, 0, 1,
6.545455, 4.171717, -531.5443, 1, 0.7921569, 0, 1,
6.585859, 4.171717, -531.7775, 1, 0.7921569, 0, 1,
6.626263, 4.171717, -532.0295, 1, 0.7921569, 0, 1,
6.666667, 4.171717, -532.3003, 1, 0.7921569, 0, 1,
6.707071, 4.171717, -532.5898, 1, 0.7921569, 0, 1,
6.747475, 4.171717, -532.8981, 1, 0.7921569, 0, 1,
6.787879, 4.171717, -533.2252, 1, 0.7921569, 0, 1,
6.828283, 4.171717, -533.571, 1, 0.7921569, 0, 1,
6.868687, 4.171717, -533.9356, 1, 0.7921569, 0, 1,
6.909091, 4.171717, -534.3189, 1, 0.7921569, 0, 1,
6.949495, 4.171717, -534.721, 1, 0.7921569, 0, 1,
6.989899, 4.171717, -535.1419, 1, 0.7921569, 0, 1,
7.030303, 4.171717, -535.5815, 1, 0.7921569, 0, 1,
7.070707, 4.171717, -536.0399, 1, 0.7921569, 0, 1,
7.111111, 4.171717, -536.517, 1, 0.7921569, 0, 1,
7.151515, 4.171717, -537.0129, 1, 0.6862745, 0, 1,
7.191919, 4.171717, -537.5276, 1, 0.6862745, 0, 1,
7.232323, 4.171717, -538.061, 1, 0.6862745, 0, 1,
7.272727, 4.171717, -538.6132, 1, 0.6862745, 0, 1,
7.313131, 4.171717, -539.1842, 1, 0.6862745, 0, 1,
7.353535, 4.171717, -539.7739, 1, 0.6862745, 0, 1,
7.393939, 4.171717, -540.3824, 1, 0.6862745, 0, 1,
7.434343, 4.171717, -541.0096, 1, 0.6862745, 0, 1,
7.474748, 4.171717, -541.6556, 1, 0.6862745, 0, 1,
7.515152, 4.171717, -542.3204, 1, 0.6862745, 0, 1,
7.555555, 4.171717, -543.0038, 1, 0.5843138, 0, 1,
7.59596, 4.171717, -543.7061, 1, 0.5843138, 0, 1,
7.636364, 4.171717, -544.4272, 1, 0.5843138, 0, 1,
7.676768, 4.171717, -545.167, 1, 0.5843138, 0, 1,
7.717172, 4.171717, -545.9255, 1, 0.5843138, 0, 1,
7.757576, 4.171717, -546.7029, 1, 0.5843138, 0, 1,
7.79798, 4.171717, -547.4989, 1, 0.5843138, 0, 1,
7.838384, 4.171717, -548.3138, 1, 0.5843138, 0, 1,
7.878788, 4.171717, -549.1473, 1, 0.4823529, 0, 1,
7.919192, 4.171717, -549.9998, 1, 0.4823529, 0, 1,
7.959596, 4.171717, -550.8708, 1, 0.4823529, 0, 1,
8, 4.171717, -551.7607, 1, 0.4823529, 0, 1,
4, 4.222222, -555.051, 1, 0.3764706, 0, 1,
4.040404, 4.222222, -554.1249, 1, 0.4823529, 0, 1,
4.080808, 4.222222, -553.2171, 1, 0.4823529, 0, 1,
4.121212, 4.222222, -552.3276, 1, 0.4823529, 0, 1,
4.161616, 4.222222, -551.4564, 1, 0.4823529, 0, 1,
4.20202, 4.222222, -550.6036, 1, 0.4823529, 0, 1,
4.242424, 4.222222, -549.769, 1, 0.4823529, 0, 1,
4.282828, 4.222222, -548.9528, 1, 0.4823529, 0, 1,
4.323232, 4.222222, -548.1549, 1, 0.5843138, 0, 1,
4.363636, 4.222222, -547.3753, 1, 0.5843138, 0, 1,
4.40404, 4.222222, -546.6141, 1, 0.5843138, 0, 1,
4.444445, 4.222222, -545.8711, 1, 0.5843138, 0, 1,
4.484848, 4.222222, -545.1464, 1, 0.5843138, 0, 1,
4.525252, 4.222222, -544.4401, 1, 0.5843138, 0, 1,
4.565657, 4.222222, -543.7521, 1, 0.5843138, 0, 1,
4.606061, 4.222222, -543.0824, 1, 0.5843138, 0, 1,
4.646465, 4.222222, -542.431, 1, 0.6862745, 0, 1,
4.686869, 4.222222, -541.7979, 1, 0.6862745, 0, 1,
4.727273, 4.222222, -541.1832, 1, 0.6862745, 0, 1,
4.767677, 4.222222, -540.5867, 1, 0.6862745, 0, 1,
4.808081, 4.222222, -540.0085, 1, 0.6862745, 0, 1,
4.848485, 4.222222, -539.4487, 1, 0.6862745, 0, 1,
4.888889, 4.222222, -538.9072, 1, 0.6862745, 0, 1,
4.929293, 4.222222, -538.384, 1, 0.6862745, 0, 1,
4.969697, 4.222222, -537.8792, 1, 0.6862745, 0, 1,
5.010101, 4.222222, -537.3926, 1, 0.6862745, 0, 1,
5.050505, 4.222222, -536.9244, 1, 0.7921569, 0, 1,
5.090909, 4.222222, -536.4744, 1, 0.7921569, 0, 1,
5.131313, 4.222222, -536.0428, 1, 0.7921569, 0, 1,
5.171717, 4.222222, -535.6295, 1, 0.7921569, 0, 1,
5.212121, 4.222222, -535.2346, 1, 0.7921569, 0, 1,
5.252525, 4.222222, -534.8578, 1, 0.7921569, 0, 1,
5.292929, 4.222222, -534.4995, 1, 0.7921569, 0, 1,
5.333333, 4.222222, -534.1595, 1, 0.7921569, 0, 1,
5.373737, 4.222222, -533.8378, 1, 0.7921569, 0, 1,
5.414141, 4.222222, -533.5343, 1, 0.7921569, 0, 1,
5.454545, 4.222222, -533.2492, 1, 0.7921569, 0, 1,
5.494949, 4.222222, -532.9824, 1, 0.7921569, 0, 1,
5.535354, 4.222222, -532.7339, 1, 0.7921569, 0, 1,
5.575758, 4.222222, -532.5038, 1, 0.7921569, 0, 1,
5.616162, 4.222222, -532.292, 1, 0.7921569, 0, 1,
5.656566, 4.222222, -532.0984, 1, 0.7921569, 0, 1,
5.69697, 4.222222, -531.9232, 1, 0.7921569, 0, 1,
5.737374, 4.222222, -531.7664, 1, 0.7921569, 0, 1,
5.777778, 4.222222, -531.6277, 1, 0.7921569, 0, 1,
5.818182, 4.222222, -531.5075, 1, 0.7921569, 0, 1,
5.858586, 4.222222, -531.4055, 1, 0.7921569, 0, 1,
5.89899, 4.222222, -531.3219, 1, 0.7921569, 0, 1,
5.939394, 4.222222, -531.2566, 1, 0.7921569, 0, 1,
5.979798, 4.222222, -531.2096, 1, 0.7921569, 0, 1,
6.020202, 4.222222, -531.1808, 1, 0.8941177, 0, 1,
6.060606, 4.222222, -531.1705, 1, 0.8941177, 0, 1,
6.10101, 4.222222, -531.1784, 1, 0.8941177, 0, 1,
6.141414, 4.222222, -531.2047, 1, 0.7921569, 0, 1,
6.181818, 4.222222, -531.2492, 1, 0.7921569, 0, 1,
6.222222, 4.222222, -531.3121, 1, 0.7921569, 0, 1,
6.262626, 4.222222, -531.3933, 1, 0.7921569, 0, 1,
6.30303, 4.222222, -531.4928, 1, 0.7921569, 0, 1,
6.343434, 4.222222, -531.6107, 1, 0.7921569, 0, 1,
6.383838, 4.222222, -531.7468, 1, 0.7921569, 0, 1,
6.424242, 4.222222, -531.9012, 1, 0.7921569, 0, 1,
6.464646, 4.222222, -532.074, 1, 0.7921569, 0, 1,
6.505051, 4.222222, -532.2651, 1, 0.7921569, 0, 1,
6.545455, 4.222222, -532.4745, 1, 0.7921569, 0, 1,
6.585859, 4.222222, -532.7021, 1, 0.7921569, 0, 1,
6.626263, 4.222222, -532.9482, 1, 0.7921569, 0, 1,
6.666667, 4.222222, -533.2125, 1, 0.7921569, 0, 1,
6.707071, 4.222222, -533.4952, 1, 0.7921569, 0, 1,
6.747475, 4.222222, -533.7961, 1, 0.7921569, 0, 1,
6.787879, 4.222222, -534.1154, 1, 0.7921569, 0, 1,
6.828283, 4.222222, -534.4531, 1, 0.7921569, 0, 1,
6.868687, 4.222222, -534.809, 1, 0.7921569, 0, 1,
6.909091, 4.222222, -535.1832, 1, 0.7921569, 0, 1,
6.949495, 4.222222, -535.5757, 1, 0.7921569, 0, 1,
6.989899, 4.222222, -535.9866, 1, 0.7921569, 0, 1,
7.030303, 4.222222, -536.4157, 1, 0.7921569, 0, 1,
7.070707, 4.222222, -536.8632, 1, 0.7921569, 0, 1,
7.111111, 4.222222, -537.329, 1, 0.6862745, 0, 1,
7.151515, 4.222222, -537.8131, 1, 0.6862745, 0, 1,
7.191919, 4.222222, -538.3156, 1, 0.6862745, 0, 1,
7.232323, 4.222222, -538.8363, 1, 0.6862745, 0, 1,
7.272727, 4.222222, -539.3754, 1, 0.6862745, 0, 1,
7.313131, 4.222222, -539.9327, 1, 0.6862745, 0, 1,
7.353535, 4.222222, -540.5084, 1, 0.6862745, 0, 1,
7.393939, 4.222222, -541.1024, 1, 0.6862745, 0, 1,
7.434343, 4.222222, -541.7147, 1, 0.6862745, 0, 1,
7.474748, 4.222222, -542.3454, 1, 0.6862745, 0, 1,
7.515152, 4.222222, -542.9943, 1, 0.5843138, 0, 1,
7.555555, 4.222222, -543.6616, 1, 0.5843138, 0, 1,
7.59596, 4.222222, -544.3472, 1, 0.5843138, 0, 1,
7.636364, 4.222222, -545.051, 1, 0.5843138, 0, 1,
7.676768, 4.222222, -545.7733, 1, 0.5843138, 0, 1,
7.717172, 4.222222, -546.5137, 1, 0.5843138, 0, 1,
7.757576, 4.222222, -547.2726, 1, 0.5843138, 0, 1,
7.79798, 4.222222, -548.0497, 1, 0.5843138, 0, 1,
7.838384, 4.222222, -548.8452, 1, 0.4823529, 0, 1,
7.878788, 4.222222, -549.659, 1, 0.4823529, 0, 1,
7.919192, 4.222222, -550.4911, 1, 0.4823529, 0, 1,
7.959596, 4.222222, -551.3415, 1, 0.4823529, 0, 1,
8, 4.222222, -552.2102, 1, 0.4823529, 0, 1,
4, 4.272727, -555.4741, 1, 0.3764706, 0, 1,
4.040404, 4.272727, -554.5698, 1, 0.3764706, 0, 1,
4.080808, 4.272727, -553.6833, 1, 0.4823529, 0, 1,
4.121212, 4.272727, -552.8147, 1, 0.4823529, 0, 1,
4.161616, 4.272727, -551.964, 1, 0.4823529, 0, 1,
4.20202, 4.272727, -551.1312, 1, 0.4823529, 0, 1,
4.242424, 4.272727, -550.3163, 1, 0.4823529, 0, 1,
4.282828, 4.272727, -549.5192, 1, 0.4823529, 0, 1,
4.323232, 4.272727, -548.7401, 1, 0.4823529, 0, 1,
4.363636, 4.272727, -547.9788, 1, 0.5843138, 0, 1,
4.40404, 4.272727, -547.2354, 1, 0.5843138, 0, 1,
4.444445, 4.272727, -546.5099, 1, 0.5843138, 0, 1,
4.484848, 4.272727, -545.8023, 1, 0.5843138, 0, 1,
4.525252, 4.272727, -545.1125, 1, 0.5843138, 0, 1,
4.565657, 4.272727, -544.4407, 1, 0.5843138, 0, 1,
4.606061, 4.272727, -543.7867, 1, 0.5843138, 0, 1,
4.646465, 4.272727, -543.1506, 1, 0.5843138, 0, 1,
4.686869, 4.272727, -542.5325, 1, 0.6862745, 0, 1,
4.727273, 4.272727, -541.9321, 1, 0.6862745, 0, 1,
4.767677, 4.272727, -541.3497, 1, 0.6862745, 0, 1,
4.808081, 4.272727, -540.7852, 1, 0.6862745, 0, 1,
4.848485, 4.272727, -540.2385, 1, 0.6862745, 0, 1,
4.888889, 4.272727, -539.7097, 1, 0.6862745, 0, 1,
4.929293, 4.272727, -539.1989, 1, 0.6862745, 0, 1,
4.969697, 4.272727, -538.7058, 1, 0.6862745, 0, 1,
5.010101, 4.272727, -538.2307, 1, 0.6862745, 0, 1,
5.050505, 4.272727, -537.7734, 1, 0.6862745, 0, 1,
5.090909, 4.272727, -537.3341, 1, 0.6862745, 0, 1,
5.131313, 4.272727, -536.9126, 1, 0.7921569, 0, 1,
5.171717, 4.272727, -536.509, 1, 0.7921569, 0, 1,
5.212121, 4.272727, -536.1234, 1, 0.7921569, 0, 1,
5.252525, 4.272727, -535.7555, 1, 0.7921569, 0, 1,
5.292929, 4.272727, -535.4056, 1, 0.7921569, 0, 1,
5.333333, 4.272727, -535.0735, 1, 0.7921569, 0, 1,
5.373737, 4.272727, -534.7593, 1, 0.7921569, 0, 1,
5.414141, 4.272727, -534.4631, 1, 0.7921569, 0, 1,
5.454545, 4.272727, -534.1847, 1, 0.7921569, 0, 1,
5.494949, 4.272727, -533.9241, 1, 0.7921569, 0, 1,
5.535354, 4.272727, -533.6815, 1, 0.7921569, 0, 1,
5.575758, 4.272727, -533.4568, 1, 0.7921569, 0, 1,
5.616162, 4.272727, -533.2499, 1, 0.7921569, 0, 1,
5.656566, 4.272727, -533.0609, 1, 0.7921569, 0, 1,
5.69697, 4.272727, -532.8898, 1, 0.7921569, 0, 1,
5.737374, 4.272727, -532.7366, 1, 0.7921569, 0, 1,
5.777778, 4.272727, -532.6013, 1, 0.7921569, 0, 1,
5.818182, 4.272727, -532.4839, 1, 0.7921569, 0, 1,
5.858586, 4.272727, -532.3843, 1, 0.7921569, 0, 1,
5.89899, 4.272727, -532.3026, 1, 0.7921569, 0, 1,
5.939394, 4.272727, -532.2388, 1, 0.7921569, 0, 1,
5.979798, 4.272727, -532.1929, 1, 0.7921569, 0, 1,
6.020202, 4.272727, -532.1649, 1, 0.7921569, 0, 1,
6.060606, 4.272727, -532.1548, 1, 0.7921569, 0, 1,
6.10101, 4.272727, -532.1625, 1, 0.7921569, 0, 1,
6.141414, 4.272727, -532.1882, 1, 0.7921569, 0, 1,
6.181818, 4.272727, -532.2317, 1, 0.7921569, 0, 1,
6.222222, 4.272727, -532.2931, 1, 0.7921569, 0, 1,
6.262626, 4.272727, -532.3724, 1, 0.7921569, 0, 1,
6.30303, 4.272727, -532.4695, 1, 0.7921569, 0, 1,
6.343434, 4.272727, -532.5846, 1, 0.7921569, 0, 1,
6.383838, 4.272727, -532.7175, 1, 0.7921569, 0, 1,
6.424242, 4.272727, -532.8683, 1, 0.7921569, 0, 1,
6.464646, 4.272727, -533.037, 1, 0.7921569, 0, 1,
6.505051, 4.272727, -533.2236, 1, 0.7921569, 0, 1,
6.545455, 4.272727, -533.4281, 1, 0.7921569, 0, 1,
6.585859, 4.272727, -533.6505, 1, 0.7921569, 0, 1,
6.626263, 4.272727, -533.8907, 1, 0.7921569, 0, 1,
6.666667, 4.272727, -534.1489, 1, 0.7921569, 0, 1,
6.707071, 4.272727, -534.4249, 1, 0.7921569, 0, 1,
6.747475, 4.272727, -534.7188, 1, 0.7921569, 0, 1,
6.787879, 4.272727, -535.0305, 1, 0.7921569, 0, 1,
6.828283, 4.272727, -535.3602, 1, 0.7921569, 0, 1,
6.868687, 4.272727, -535.7078, 1, 0.7921569, 0, 1,
6.909091, 4.272727, -536.0732, 1, 0.7921569, 0, 1,
6.949495, 4.272727, -536.4565, 1, 0.7921569, 0, 1,
6.989899, 4.272727, -536.8577, 1, 0.7921569, 0, 1,
7.030303, 4.272727, -537.2768, 1, 0.6862745, 0, 1,
7.070707, 4.272727, -537.7137, 1, 0.6862745, 0, 1,
7.111111, 4.272727, -538.1686, 1, 0.6862745, 0, 1,
7.151515, 4.272727, -538.6413, 1, 0.6862745, 0, 1,
7.191919, 4.272727, -539.132, 1, 0.6862745, 0, 1,
7.232323, 4.272727, -539.6404, 1, 0.6862745, 0, 1,
7.272727, 4.272727, -540.1669, 1, 0.6862745, 0, 1,
7.313131, 4.272727, -540.7111, 1, 0.6862745, 0, 1,
7.353535, 4.272727, -541.2733, 1, 0.6862745, 0, 1,
7.393939, 4.272727, -541.8533, 1, 0.6862745, 0, 1,
7.434343, 4.272727, -542.4512, 1, 0.6862745, 0, 1,
7.474748, 4.272727, -543.0671, 1, 0.5843138, 0, 1,
7.515152, 4.272727, -543.7007, 1, 0.5843138, 0, 1,
7.555555, 4.272727, -544.3523, 1, 0.5843138, 0, 1,
7.59596, 4.272727, -545.0218, 1, 0.5843138, 0, 1,
7.636364, 4.272727, -545.7091, 1, 0.5843138, 0, 1,
7.676768, 4.272727, -546.4144, 1, 0.5843138, 0, 1,
7.717172, 4.272727, -547.1375, 1, 0.5843138, 0, 1,
7.757576, 4.272727, -547.8785, 1, 0.5843138, 0, 1,
7.79798, 4.272727, -548.6374, 1, 0.4823529, 0, 1,
7.838384, 4.272727, -549.4141, 1, 0.4823529, 0, 1,
7.878788, 4.272727, -550.2088, 1, 0.4823529, 0, 1,
7.919192, 4.272727, -551.0213, 1, 0.4823529, 0, 1,
7.959596, 4.272727, -551.8517, 1, 0.4823529, 0, 1,
8, 4.272727, -552.7001, 1, 0.4823529, 0, 1,
4, 4.323232, -555.9373, 1, 0.3764706, 0, 1,
4.040404, 4.323232, -555.054, 1, 0.3764706, 0, 1,
4.080808, 4.323232, -554.1881, 1, 0.4823529, 0, 1,
4.121212, 4.323232, -553.3397, 1, 0.4823529, 0, 1,
4.161616, 4.323232, -552.5088, 1, 0.4823529, 0, 1,
4.20202, 4.323232, -551.6953, 1, 0.4823529, 0, 1,
4.242424, 4.323232, -550.8994, 1, 0.4823529, 0, 1,
4.282828, 4.323232, -550.1208, 1, 0.4823529, 0, 1,
4.323232, 4.323232, -549.3597, 1, 0.4823529, 0, 1,
4.363636, 4.323232, -548.6161, 1, 0.4823529, 0, 1,
4.40404, 4.323232, -547.89, 1, 0.5843138, 0, 1,
4.444445, 4.323232, -547.1814, 1, 0.5843138, 0, 1,
4.484848, 4.323232, -546.4902, 1, 0.5843138, 0, 1,
4.525252, 4.323232, -545.8165, 1, 0.5843138, 0, 1,
4.565657, 4.323232, -545.1602, 1, 0.5843138, 0, 1,
4.606061, 4.323232, -544.5215, 1, 0.5843138, 0, 1,
4.646465, 4.323232, -543.9001, 1, 0.5843138, 0, 1,
4.686869, 4.323232, -543.2963, 1, 0.5843138, 0, 1,
4.727273, 4.323232, -542.71, 1, 0.6862745, 0, 1,
4.767677, 4.323232, -542.1411, 1, 0.6862745, 0, 1,
4.808081, 4.323232, -541.5896, 1, 0.6862745, 0, 1,
4.848485, 4.323232, -541.0557, 1, 0.6862745, 0, 1,
4.888889, 4.323232, -540.5391, 1, 0.6862745, 0, 1,
4.929293, 4.323232, -540.0401, 1, 0.6862745, 0, 1,
4.969697, 4.323232, -539.5586, 1, 0.6862745, 0, 1,
5.010101, 4.323232, -539.0945, 1, 0.6862745, 0, 1,
5.050505, 4.323232, -538.6479, 1, 0.6862745, 0, 1,
5.090909, 4.323232, -538.2187, 1, 0.6862745, 0, 1,
5.131313, 4.323232, -537.807, 1, 0.6862745, 0, 1,
5.171717, 4.323232, -537.4128, 1, 0.6862745, 0, 1,
5.212121, 4.323232, -537.0361, 1, 0.6862745, 0, 1,
5.252525, 4.323232, -536.6768, 1, 0.7921569, 0, 1,
5.292929, 4.323232, -536.335, 1, 0.7921569, 0, 1,
5.333333, 4.323232, -536.0106, 1, 0.7921569, 0, 1,
5.373737, 4.323232, -535.7038, 1, 0.7921569, 0, 1,
5.414141, 4.323232, -535.4144, 1, 0.7921569, 0, 1,
5.454545, 4.323232, -535.1425, 1, 0.7921569, 0, 1,
5.494949, 4.323232, -534.888, 1, 0.7921569, 0, 1,
5.535354, 4.323232, -534.651, 1, 0.7921569, 0, 1,
5.575758, 4.323232, -534.4315, 1, 0.7921569, 0, 1,
5.616162, 4.323232, -534.2294, 1, 0.7921569, 0, 1,
5.656566, 4.323232, -534.0448, 1, 0.7921569, 0, 1,
5.69697, 4.323232, -533.8777, 1, 0.7921569, 0, 1,
5.737374, 4.323232, -533.728, 1, 0.7921569, 0, 1,
5.777778, 4.323232, -533.5958, 1, 0.7921569, 0, 1,
5.818182, 4.323232, -533.4811, 1, 0.7921569, 0, 1,
5.858586, 4.323232, -533.3839, 1, 0.7921569, 0, 1,
5.89899, 4.323232, -533.3041, 1, 0.7921569, 0, 1,
5.939394, 4.323232, -533.2418, 1, 0.7921569, 0, 1,
5.979798, 4.323232, -533.197, 1, 0.7921569, 0, 1,
6.020202, 4.323232, -533.1696, 1, 0.7921569, 0, 1,
6.060606, 4.323232, -533.1597, 1, 0.7921569, 0, 1,
6.10101, 4.323232, -533.1673, 1, 0.7921569, 0, 1,
6.141414, 4.323232, -533.1923, 1, 0.7921569, 0, 1,
6.181818, 4.323232, -533.2348, 1, 0.7921569, 0, 1,
6.222222, 4.323232, -533.2948, 1, 0.7921569, 0, 1,
6.262626, 4.323232, -533.3723, 1, 0.7921569, 0, 1,
6.30303, 4.323232, -533.4672, 1, 0.7921569, 0, 1,
6.343434, 4.323232, -533.5795, 1, 0.7921569, 0, 1,
6.383838, 4.323232, -533.7094, 1, 0.7921569, 0, 1,
6.424242, 4.323232, -533.8567, 1, 0.7921569, 0, 1,
6.464646, 4.323232, -534.0215, 1, 0.7921569, 0, 1,
6.505051, 4.323232, -534.2037, 1, 0.7921569, 0, 1,
6.545455, 4.323232, -534.4034, 1, 0.7921569, 0, 1,
6.585859, 4.323232, -534.6207, 1, 0.7921569, 0, 1,
6.626263, 4.323232, -534.8553, 1, 0.7921569, 0, 1,
6.666667, 4.323232, -535.1074, 1, 0.7921569, 0, 1,
6.707071, 4.323232, -535.377, 1, 0.7921569, 0, 1,
6.747475, 4.323232, -535.6641, 1, 0.7921569, 0, 1,
6.787879, 4.323232, -535.9686, 1, 0.7921569, 0, 1,
6.828283, 4.323232, -536.2906, 1, 0.7921569, 0, 1,
6.868687, 4.323232, -536.6301, 1, 0.7921569, 0, 1,
6.909091, 4.323232, -536.9871, 1, 0.6862745, 0, 1,
6.949495, 4.323232, -537.3615, 1, 0.6862745, 0, 1,
6.989899, 4.323232, -537.7534, 1, 0.6862745, 0, 1,
7.030303, 4.323232, -538.1627, 1, 0.6862745, 0, 1,
7.070707, 4.323232, -538.5895, 1, 0.6862745, 0, 1,
7.111111, 4.323232, -539.0338, 1, 0.6862745, 0, 1,
7.151515, 4.323232, -539.4955, 1, 0.6862745, 0, 1,
7.191919, 4.323232, -539.9748, 1, 0.6862745, 0, 1,
7.232323, 4.323232, -540.4715, 1, 0.6862745, 0, 1,
7.272727, 4.323232, -540.9857, 1, 0.6862745, 0, 1,
7.313131, 4.323232, -541.5173, 1, 0.6862745, 0, 1,
7.353535, 4.323232, -542.0664, 1, 0.6862745, 0, 1,
7.393939, 4.323232, -542.6329, 1, 0.6862745, 0, 1,
7.434343, 4.323232, -543.217, 1, 0.5843138, 0, 1,
7.474748, 4.323232, -543.8185, 1, 0.5843138, 0, 1,
7.515152, 4.323232, -544.4375, 1, 0.5843138, 0, 1,
7.555555, 4.323232, -545.0739, 1, 0.5843138, 0, 1,
7.59596, 4.323232, -545.7278, 1, 0.5843138, 0, 1,
7.636364, 4.323232, -546.3992, 1, 0.5843138, 0, 1,
7.676768, 4.323232, -547.0881, 1, 0.5843138, 0, 1,
7.717172, 4.323232, -547.7944, 1, 0.5843138, 0, 1,
7.757576, 4.323232, -548.5182, 1, 0.5843138, 0, 1,
7.79798, 4.323232, -549.2594, 1, 0.4823529, 0, 1,
7.838384, 4.323232, -550.0181, 1, 0.4823529, 0, 1,
7.878788, 4.323232, -550.7943, 1, 0.4823529, 0, 1,
7.919192, 4.323232, -551.588, 1, 0.4823529, 0, 1,
7.959596, 4.323232, -552.3991, 1, 0.4823529, 0, 1,
8, 4.323232, -553.2277, 1, 0.4823529, 0, 1,
4, 4.373737, -556.4383, 1, 0.3764706, 0, 1,
4.040404, 4.373737, -555.5753, 1, 0.3764706, 0, 1,
4.080808, 4.373737, -554.7292, 1, 0.3764706, 0, 1,
4.121212, 4.373737, -553.9003, 1, 0.4823529, 0, 1,
4.161616, 4.373737, -553.0884, 1, 0.4823529, 0, 1,
4.20202, 4.373737, -552.2937, 1, 0.4823529, 0, 1,
4.242424, 4.373737, -551.516, 1, 0.4823529, 0, 1,
4.282828, 4.373737, -550.7553, 1, 0.4823529, 0, 1,
4.323232, 4.373737, -550.0117, 1, 0.4823529, 0, 1,
4.363636, 4.373737, -549.2852, 1, 0.4823529, 0, 1,
4.40404, 4.373737, -548.5757, 1, 0.4823529, 0, 1,
4.444445, 4.373737, -547.8834, 1, 0.5843138, 0, 1,
4.484848, 4.373737, -547.2081, 1, 0.5843138, 0, 1,
4.525252, 4.373737, -546.5498, 1, 0.5843138, 0, 1,
4.565657, 4.373737, -545.9086, 1, 0.5843138, 0, 1,
4.606061, 4.373737, -545.2845, 1, 0.5843138, 0, 1,
4.646465, 4.373737, -544.6775, 1, 0.5843138, 0, 1,
4.686869, 4.373737, -544.0875, 1, 0.5843138, 0, 1,
4.727273, 4.373737, -543.5146, 1, 0.5843138, 0, 1,
4.767677, 4.373737, -542.9588, 1, 0.5843138, 0, 1,
4.808081, 4.373737, -542.42, 1, 0.6862745, 0, 1,
4.848485, 4.373737, -541.8983, 1, 0.6862745, 0, 1,
4.888889, 4.373737, -541.3937, 1, 0.6862745, 0, 1,
4.929293, 4.373737, -540.9061, 1, 0.6862745, 0, 1,
4.969697, 4.373737, -540.4356, 1, 0.6862745, 0, 1,
5.010101, 4.373737, -539.9822, 1, 0.6862745, 0, 1,
5.050505, 4.373737, -539.5458, 1, 0.6862745, 0, 1,
5.090909, 4.373737, -539.1265, 1, 0.6862745, 0, 1,
5.131313, 4.373737, -538.7242, 1, 0.6862745, 0, 1,
5.171717, 4.373737, -538.3391, 1, 0.6862745, 0, 1,
5.212121, 4.373737, -537.971, 1, 0.6862745, 0, 1,
5.252525, 4.373737, -537.62, 1, 0.6862745, 0, 1,
5.292929, 4.373737, -537.286, 1, 0.6862745, 0, 1,
5.333333, 4.373737, -536.9691, 1, 0.7921569, 0, 1,
5.373737, 4.373737, -536.6693, 1, 0.7921569, 0, 1,
5.414141, 4.373737, -536.3865, 1, 0.7921569, 0, 1,
5.454545, 4.373737, -536.1208, 1, 0.7921569, 0, 1,
5.494949, 4.373737, -535.8723, 1, 0.7921569, 0, 1,
5.535354, 4.373737, -535.6407, 1, 0.7921569, 0, 1,
5.575758, 4.373737, -535.4262, 1, 0.7921569, 0, 1,
5.616162, 4.373737, -535.2288, 1, 0.7921569, 0, 1,
5.656566, 4.373737, -535.0485, 1, 0.7921569, 0, 1,
5.69697, 4.373737, -534.8851, 1, 0.7921569, 0, 1,
5.737374, 4.373737, -534.739, 1, 0.7921569, 0, 1,
5.777778, 4.373737, -534.6098, 1, 0.7921569, 0, 1,
5.818182, 4.373737, -534.4977, 1, 0.7921569, 0, 1,
5.858586, 4.373737, -534.4027, 1, 0.7921569, 0, 1,
5.89899, 4.373737, -534.3248, 1, 0.7921569, 0, 1,
5.939394, 4.373737, -534.2639, 1, 0.7921569, 0, 1,
5.979798, 4.373737, -534.2201, 1, 0.7921569, 0, 1,
6.020202, 4.373737, -534.1933, 1, 0.7921569, 0, 1,
6.060606, 4.373737, -534.1837, 1, 0.7921569, 0, 1,
6.10101, 4.373737, -534.191, 1, 0.7921569, 0, 1,
6.141414, 4.373737, -534.2155, 1, 0.7921569, 0, 1,
6.181818, 4.373737, -534.257, 1, 0.7921569, 0, 1,
6.222222, 4.373737, -534.3156, 1, 0.7921569, 0, 1,
6.262626, 4.373737, -534.3913, 1, 0.7921569, 0, 1,
6.30303, 4.373737, -534.484, 1, 0.7921569, 0, 1,
6.343434, 4.373737, -534.5938, 1, 0.7921569, 0, 1,
6.383838, 4.373737, -534.7207, 1, 0.7921569, 0, 1,
6.424242, 4.373737, -534.8646, 1, 0.7921569, 0, 1,
6.464646, 4.373737, -535.0256, 1, 0.7921569, 0, 1,
6.505051, 4.373737, -535.2037, 1, 0.7921569, 0, 1,
6.545455, 4.373737, -535.3989, 1, 0.7921569, 0, 1,
6.585859, 4.373737, -535.6111, 1, 0.7921569, 0, 1,
6.626263, 4.373737, -535.8403, 1, 0.7921569, 0, 1,
6.666667, 4.373737, -536.0867, 1, 0.7921569, 0, 1,
6.707071, 4.373737, -536.3501, 1, 0.7921569, 0, 1,
6.747475, 4.373737, -536.6306, 1, 0.7921569, 0, 1,
6.787879, 4.373737, -536.9281, 1, 0.7921569, 0, 1,
6.828283, 4.373737, -537.2427, 1, 0.6862745, 0, 1,
6.868687, 4.373737, -537.5744, 1, 0.6862745, 0, 1,
6.909091, 4.373737, -537.9232, 1, 0.6862745, 0, 1,
6.949495, 4.373737, -538.2889, 1, 0.6862745, 0, 1,
6.989899, 4.373737, -538.6718, 1, 0.6862745, 0, 1,
7.030303, 4.373737, -539.0718, 1, 0.6862745, 0, 1,
7.070707, 4.373737, -539.4888, 1, 0.6862745, 0, 1,
7.111111, 4.373737, -539.9229, 1, 0.6862745, 0, 1,
7.151515, 4.373737, -540.374, 1, 0.6862745, 0, 1,
7.191919, 4.373737, -540.8423, 1, 0.6862745, 0, 1,
7.232323, 4.373737, -541.3276, 1, 0.6862745, 0, 1,
7.272727, 4.373737, -541.8299, 1, 0.6862745, 0, 1,
7.313131, 4.373737, -542.3493, 1, 0.6862745, 0, 1,
7.353535, 4.373737, -542.8858, 1, 0.5843138, 0, 1,
7.393939, 4.373737, -543.4394, 1, 0.5843138, 0, 1,
7.434343, 4.373737, -544.01, 1, 0.5843138, 0, 1,
7.474748, 4.373737, -544.5977, 1, 0.5843138, 0, 1,
7.515152, 4.373737, -545.2025, 1, 0.5843138, 0, 1,
7.555555, 4.373737, -545.8243, 1, 0.5843138, 0, 1,
7.59596, 4.373737, -546.4632, 1, 0.5843138, 0, 1,
7.636364, 4.373737, -547.1191, 1, 0.5843138, 0, 1,
7.676768, 4.373737, -547.7922, 1, 0.5843138, 0, 1,
7.717172, 4.373737, -548.4823, 1, 0.5843138, 0, 1,
7.757576, 4.373737, -549.1895, 1, 0.4823529, 0, 1,
7.79798, 4.373737, -549.9137, 1, 0.4823529, 0, 1,
7.838384, 4.373737, -550.655, 1, 0.4823529, 0, 1,
7.878788, 4.373737, -551.4134, 1, 0.4823529, 0, 1,
7.919192, 4.373737, -552.1888, 1, 0.4823529, 0, 1,
7.959596, 4.373737, -552.9813, 1, 0.4823529, 0, 1,
8, 4.373737, -553.7909, 1, 0.4823529, 0, 1,
4, 4.424242, -556.9746, 1, 0.3764706, 0, 1,
4.040404, 4.424242, -556.1312, 1, 0.3764706, 0, 1,
4.080808, 4.424242, -555.3044, 1, 0.3764706, 0, 1,
4.121212, 4.424242, -554.4943, 1, 0.3764706, 0, 1,
4.161616, 4.424242, -553.7008, 1, 0.4823529, 0, 1,
4.20202, 4.424242, -552.9241, 1, 0.4823529, 0, 1,
4.242424, 4.424242, -552.164, 1, 0.4823529, 0, 1,
4.282828, 4.424242, -551.4207, 1, 0.4823529, 0, 1,
4.323232, 4.424242, -550.6939, 1, 0.4823529, 0, 1,
4.363636, 4.424242, -549.9839, 1, 0.4823529, 0, 1,
4.40404, 4.424242, -549.2906, 1, 0.4823529, 0, 1,
4.444445, 4.424242, -548.6139, 1, 0.4823529, 0, 1,
4.484848, 4.424242, -547.9539, 1, 0.5843138, 0, 1,
4.525252, 4.424242, -547.3106, 1, 0.5843138, 0, 1,
4.565657, 4.424242, -546.684, 1, 0.5843138, 0, 1,
4.606061, 4.424242, -546.074, 1, 0.5843138, 0, 1,
4.646465, 4.424242, -545.4808, 1, 0.5843138, 0, 1,
4.686869, 4.424242, -544.9042, 1, 0.5843138, 0, 1,
4.727273, 4.424242, -544.3443, 1, 0.5843138, 0, 1,
4.767677, 4.424242, -543.8011, 1, 0.5843138, 0, 1,
4.808081, 4.424242, -543.2745, 1, 0.5843138, 0, 1,
4.848485, 4.424242, -542.7647, 1, 0.6862745, 0, 1,
4.888889, 4.424242, -542.2715, 1, 0.6862745, 0, 1,
4.929293, 4.424242, -541.795, 1, 0.6862745, 0, 1,
4.969697, 4.424242, -541.3352, 1, 0.6862745, 0, 1,
5.010101, 4.424242, -540.892, 1, 0.6862745, 0, 1,
5.050505, 4.424242, -540.4656, 1, 0.6862745, 0, 1,
5.090909, 4.424242, -540.0558, 1, 0.6862745, 0, 1,
5.131313, 4.424242, -539.6627, 1, 0.6862745, 0, 1,
5.171717, 4.424242, -539.2863, 1, 0.6862745, 0, 1,
5.212121, 4.424242, -538.9266, 1, 0.6862745, 0, 1,
5.252525, 4.424242, -538.5835, 1, 0.6862745, 0, 1,
5.292929, 4.424242, -538.2571, 1, 0.6862745, 0, 1,
5.333333, 4.424242, -537.9474, 1, 0.6862745, 0, 1,
5.373737, 4.424242, -537.6544, 1, 0.6862745, 0, 1,
5.414141, 4.424242, -537.3781, 1, 0.6862745, 0, 1,
5.454545, 4.424242, -537.1184, 1, 0.6862745, 0, 1,
5.494949, 4.424242, -536.8754, 1, 0.7921569, 0, 1,
5.535354, 4.424242, -536.6491, 1, 0.7921569, 0, 1,
5.575758, 4.424242, -536.4395, 1, 0.7921569, 0, 1,
5.616162, 4.424242, -536.2466, 1, 0.7921569, 0, 1,
5.656566, 4.424242, -536.0703, 1, 0.7921569, 0, 1,
5.69697, 4.424242, -535.9108, 1, 0.7921569, 0, 1,
5.737374, 4.424242, -535.7678, 1, 0.7921569, 0, 1,
5.777778, 4.424242, -535.6416, 1, 0.7921569, 0, 1,
5.818182, 4.424242, -535.5321, 1, 0.7921569, 0, 1,
5.858586, 4.424242, -535.4393, 1, 0.7921569, 0, 1,
5.89899, 4.424242, -535.363, 1, 0.7921569, 0, 1,
5.939394, 4.424242, -535.3036, 1, 0.7921569, 0, 1,
5.979798, 4.424242, -535.2607, 1, 0.7921569, 0, 1,
6.020202, 4.424242, -535.2346, 1, 0.7921569, 0, 1,
6.060606, 4.424242, -535.2252, 1, 0.7921569, 0, 1,
6.10101, 4.424242, -535.2324, 1, 0.7921569, 0, 1,
6.141414, 4.424242, -535.2563, 1, 0.7921569, 0, 1,
6.181818, 4.424242, -535.2969, 1, 0.7921569, 0, 1,
6.222222, 4.424242, -535.3541, 1, 0.7921569, 0, 1,
6.262626, 4.424242, -535.4281, 1, 0.7921569, 0, 1,
6.30303, 4.424242, -535.5187, 1, 0.7921569, 0, 1,
6.343434, 4.424242, -535.626, 1, 0.7921569, 0, 1,
6.383838, 4.424242, -535.75, 1, 0.7921569, 0, 1,
6.424242, 4.424242, -535.8907, 1, 0.7921569, 0, 1,
6.464646, 4.424242, -536.048, 1, 0.7921569, 0, 1,
6.505051, 4.424242, -536.222, 1, 0.7921569, 0, 1,
6.545455, 4.424242, -536.4128, 1, 0.7921569, 0, 1,
6.585859, 4.424242, -536.6202, 1, 0.7921569, 0, 1,
6.626263, 4.424242, -536.8442, 1, 0.7921569, 0, 1,
6.666667, 4.424242, -537.085, 1, 0.6862745, 0, 1,
6.707071, 4.424242, -537.3424, 1, 0.6862745, 0, 1,
6.747475, 4.424242, -537.6165, 1, 0.6862745, 0, 1,
6.787879, 4.424242, -537.9073, 1, 0.6862745, 0, 1,
6.828283, 4.424242, -538.2148, 1, 0.6862745, 0, 1,
6.868687, 4.424242, -538.5389, 1, 0.6862745, 0, 1,
6.909091, 4.424242, -538.8798, 1, 0.6862745, 0, 1,
6.949495, 4.424242, -539.2372, 1, 0.6862745, 0, 1,
6.989899, 4.424242, -539.6115, 1, 0.6862745, 0, 1,
7.030303, 4.424242, -540.0023, 1, 0.6862745, 0, 1,
7.070707, 4.424242, -540.4099, 1, 0.6862745, 0, 1,
7.111111, 4.424242, -540.8341, 1, 0.6862745, 0, 1,
7.151515, 4.424242, -541.275, 1, 0.6862745, 0, 1,
7.191919, 4.424242, -541.7326, 1, 0.6862745, 0, 1,
7.232323, 4.424242, -542.2069, 1, 0.6862745, 0, 1,
7.272727, 4.424242, -542.6979, 1, 0.6862745, 0, 1,
7.313131, 4.424242, -543.2055, 1, 0.5843138, 0, 1,
7.353535, 4.424242, -543.7298, 1, 0.5843138, 0, 1,
7.393939, 4.424242, -544.2708, 1, 0.5843138, 0, 1,
7.434343, 4.424242, -544.8284, 1, 0.5843138, 0, 1,
7.474748, 4.424242, -545.4028, 1, 0.5843138, 0, 1,
7.515152, 4.424242, -545.9938, 1, 0.5843138, 0, 1,
7.555555, 4.424242, -546.6016, 1, 0.5843138, 0, 1,
7.59596, 4.424242, -547.226, 1, 0.5843138, 0, 1,
7.636364, 4.424242, -547.867, 1, 0.5843138, 0, 1,
7.676768, 4.424242, -548.5248, 1, 0.5843138, 0, 1,
7.717172, 4.424242, -549.1992, 1, 0.4823529, 0, 1,
7.757576, 4.424242, -549.8903, 1, 0.4823529, 0, 1,
7.79798, 4.424242, -550.5981, 1, 0.4823529, 0, 1,
7.838384, 4.424242, -551.3226, 1, 0.4823529, 0, 1,
7.878788, 4.424242, -552.0638, 1, 0.4823529, 0, 1,
7.919192, 4.424242, -552.8216, 1, 0.4823529, 0, 1,
7.959596, 4.424242, -553.5961, 1, 0.4823529, 0, 1,
8, 4.424242, -554.3873, 1, 0.3764706, 0, 1,
4, 4.474748, -557.5441, 1, 0.3764706, 0, 1,
4.040404, 4.474748, -556.7196, 1, 0.3764706, 0, 1,
4.080808, 4.474748, -555.9114, 1, 0.3764706, 0, 1,
4.121212, 4.474748, -555.1194, 1, 0.3764706, 0, 1,
4.161616, 4.474748, -554.3438, 1, 0.3764706, 0, 1,
4.20202, 4.474748, -553.5845, 1, 0.4823529, 0, 1,
4.242424, 4.474748, -552.8415, 1, 0.4823529, 0, 1,
4.282828, 4.474748, -552.1148, 1, 0.4823529, 0, 1,
4.323232, 4.474748, -551.4044, 1, 0.4823529, 0, 1,
4.363636, 4.474748, -550.7103, 1, 0.4823529, 0, 1,
4.40404, 4.474748, -550.0325, 1, 0.4823529, 0, 1,
4.444445, 4.474748, -549.3711, 1, 0.4823529, 0, 1,
4.484848, 4.474748, -548.7259, 1, 0.4823529, 0, 1,
4.525252, 4.474748, -548.097, 1, 0.5843138, 0, 1,
4.565657, 4.474748, -547.4845, 1, 0.5843138, 0, 1,
4.606061, 4.474748, -546.8882, 1, 0.5843138, 0, 1,
4.646465, 4.474748, -546.3083, 1, 0.5843138, 0, 1,
4.686869, 4.474748, -545.7446, 1, 0.5843138, 0, 1,
4.727273, 4.474748, -545.1973, 1, 0.5843138, 0, 1,
4.767677, 4.474748, -544.6663, 1, 0.5843138, 0, 1,
4.808081, 4.474748, -544.1516, 1, 0.5843138, 0, 1,
4.848485, 4.474748, -543.6531, 1, 0.5843138, 0, 1,
4.888889, 4.474748, -543.171, 1, 0.5843138, 0, 1,
4.929293, 4.474748, -542.7053, 1, 0.6862745, 0, 1,
4.969697, 4.474748, -542.2557, 1, 0.6862745, 0, 1,
5.010101, 4.474748, -541.8225, 1, 0.6862745, 0, 1,
5.050505, 4.474748, -541.4056, 1, 0.6862745, 0, 1,
5.090909, 4.474748, -541.0051, 1, 0.6862745, 0, 1,
5.131313, 4.474748, -540.6208, 1, 0.6862745, 0, 1,
5.171717, 4.474748, -540.2528, 1, 0.6862745, 0, 1,
5.212121, 4.474748, -539.9012, 1, 0.6862745, 0, 1,
5.252525, 4.474748, -539.5658, 1, 0.6862745, 0, 1,
5.292929, 4.474748, -539.2468, 1, 0.6862745, 0, 1,
5.333333, 4.474748, -538.944, 1, 0.6862745, 0, 1,
5.373737, 4.474748, -538.6576, 1, 0.6862745, 0, 1,
5.414141, 4.474748, -538.3875, 1, 0.6862745, 0, 1,
5.454545, 4.474748, -538.1336, 1, 0.6862745, 0, 1,
5.494949, 4.474748, -537.8961, 1, 0.6862745, 0, 1,
5.535354, 4.474748, -537.6749, 1, 0.6862745, 0, 1,
5.575758, 4.474748, -537.47, 1, 0.6862745, 0, 1,
5.616162, 4.474748, -537.2813, 1, 0.6862745, 0, 1,
5.656566, 4.474748, -537.1091, 1, 0.6862745, 0, 1,
5.69697, 4.474748, -536.9531, 1, 0.7921569, 0, 1,
5.737374, 4.474748, -536.8134, 1, 0.7921569, 0, 1,
5.777778, 4.474748, -536.69, 1, 0.7921569, 0, 1,
5.818182, 4.474748, -536.5829, 1, 0.7921569, 0, 1,
5.858586, 4.474748, -536.4921, 1, 0.7921569, 0, 1,
5.89899, 4.474748, -536.4177, 1, 0.7921569, 0, 1,
5.939394, 4.474748, -536.3595, 1, 0.7921569, 0, 1,
5.979798, 4.474748, -536.3176, 1, 0.7921569, 0, 1,
6.020202, 4.474748, -536.2921, 1, 0.7921569, 0, 1,
6.060606, 4.474748, -536.2828, 1, 0.7921569, 0, 1,
6.10101, 4.474748, -536.2899, 1, 0.7921569, 0, 1,
6.141414, 4.474748, -536.3133, 1, 0.7921569, 0, 1,
6.181818, 4.474748, -536.353, 1, 0.7921569, 0, 1,
6.222222, 4.474748, -536.4089, 1, 0.7921569, 0, 1,
6.262626, 4.474748, -536.4813, 1, 0.7921569, 0, 1,
6.30303, 4.474748, -536.5698, 1, 0.7921569, 0, 1,
6.343434, 4.474748, -536.6747, 1, 0.7921569, 0, 1,
6.383838, 4.474748, -536.796, 1, 0.7921569, 0, 1,
6.424242, 4.474748, -536.9335, 1, 0.7921569, 0, 1,
6.464646, 4.474748, -537.0873, 1, 0.6862745, 0, 1,
6.505051, 4.474748, -537.2574, 1, 0.6862745, 0, 1,
6.545455, 4.474748, -537.4438, 1, 0.6862745, 0, 1,
6.585859, 4.474748, -537.6465, 1, 0.6862745, 0, 1,
6.626263, 4.474748, -537.8656, 1, 0.6862745, 0, 1,
6.666667, 4.474748, -538.101, 1, 0.6862745, 0, 1,
6.707071, 4.474748, -538.3526, 1, 0.6862745, 0, 1,
6.747475, 4.474748, -538.6205, 1, 0.6862745, 0, 1,
6.787879, 4.474748, -538.9048, 1, 0.6862745, 0, 1,
6.828283, 4.474748, -539.2054, 1, 0.6862745, 0, 1,
6.868687, 4.474748, -539.5222, 1, 0.6862745, 0, 1,
6.909091, 4.474748, -539.8554, 1, 0.6862745, 0, 1,
6.949495, 4.474748, -540.2049, 1, 0.6862745, 0, 1,
6.989899, 4.474748, -540.5707, 1, 0.6862745, 0, 1,
7.030303, 4.474748, -540.9528, 1, 0.6862745, 0, 1,
7.070707, 4.474748, -541.3512, 1, 0.6862745, 0, 1,
7.111111, 4.474748, -541.7659, 1, 0.6862745, 0, 1,
7.151515, 4.474748, -542.1969, 1, 0.6862745, 0, 1,
7.191919, 4.474748, -542.6442, 1, 0.6862745, 0, 1,
7.232323, 4.474748, -543.1078, 1, 0.5843138, 0, 1,
7.272727, 4.474748, -543.5878, 1, 0.5843138, 0, 1,
7.313131, 4.474748, -544.084, 1, 0.5843138, 0, 1,
7.353535, 4.474748, -544.5966, 1, 0.5843138, 0, 1,
7.393939, 4.474748, -545.1254, 1, 0.5843138, 0, 1,
7.434343, 4.474748, -545.6706, 1, 0.5843138, 0, 1,
7.474748, 4.474748, -546.2321, 1, 0.5843138, 0, 1,
7.515152, 4.474748, -546.8098, 1, 0.5843138, 0, 1,
7.555555, 4.474748, -547.4039, 1, 0.5843138, 0, 1,
7.59596, 4.474748, -548.0143, 1, 0.5843138, 0, 1,
7.636364, 4.474748, -548.641, 1, 0.4823529, 0, 1,
7.676768, 4.474748, -549.2839, 1, 0.4823529, 0, 1,
7.717172, 4.474748, -549.9432, 1, 0.4823529, 0, 1,
7.757576, 4.474748, -550.6188, 1, 0.4823529, 0, 1,
7.79798, 4.474748, -551.3108, 1, 0.4823529, 0, 1,
7.838384, 4.474748, -552.019, 1, 0.4823529, 0, 1,
7.878788, 4.474748, -552.7435, 1, 0.4823529, 0, 1,
7.919192, 4.474748, -553.4843, 1, 0.4823529, 0, 1,
7.959596, 4.474748, -554.2415, 1, 0.4823529, 0, 1,
8, 4.474748, -555.0149, 1, 0.3764706, 0, 1,
4, 4.525252, -558.1448, 1, 0.3764706, 0, 1,
4.040404, 4.525252, -557.3385, 1, 0.3764706, 0, 1,
4.080808, 4.525252, -556.5482, 1, 0.3764706, 0, 1,
4.121212, 4.525252, -555.7739, 1, 0.3764706, 0, 1,
4.161616, 4.525252, -555.0155, 1, 0.3764706, 0, 1,
4.20202, 4.525252, -554.2731, 1, 0.4823529, 0, 1,
4.242424, 4.525252, -553.5465, 1, 0.4823529, 0, 1,
4.282828, 4.525252, -552.8359, 1, 0.4823529, 0, 1,
4.323232, 4.525252, -552.1414, 1, 0.4823529, 0, 1,
4.363636, 4.525252, -551.4626, 1, 0.4823529, 0, 1,
4.40404, 4.525252, -550.7999, 1, 0.4823529, 0, 1,
4.444445, 4.525252, -550.1531, 1, 0.4823529, 0, 1,
4.484848, 4.525252, -549.5223, 1, 0.4823529, 0, 1,
4.525252, 4.525252, -548.9074, 1, 0.4823529, 0, 1,
4.565657, 4.525252, -548.3084, 1, 0.5843138, 0, 1,
4.606061, 4.525252, -547.7254, 1, 0.5843138, 0, 1,
4.646465, 4.525252, -547.1583, 1, 0.5843138, 0, 1,
4.686869, 4.525252, -546.6072, 1, 0.5843138, 0, 1,
4.727273, 4.525252, -546.072, 1, 0.5843138, 0, 1,
4.767677, 4.525252, -545.5528, 1, 0.5843138, 0, 1,
4.808081, 4.525252, -545.0495, 1, 0.5843138, 0, 1,
4.848485, 4.525252, -544.5621, 1, 0.5843138, 0, 1,
4.888889, 4.525252, -544.0907, 1, 0.5843138, 0, 1,
4.929293, 4.525252, -543.6353, 1, 0.5843138, 0, 1,
4.969697, 4.525252, -543.1957, 1, 0.5843138, 0, 1,
5.010101, 4.525252, -542.7722, 1, 0.5843138, 0, 1,
5.050505, 4.525252, -542.3645, 1, 0.6862745, 0, 1,
5.090909, 4.525252, -541.9728, 1, 0.6862745, 0, 1,
5.131313, 4.525252, -541.5971, 1, 0.6862745, 0, 1,
5.171717, 4.525252, -541.2373, 1, 0.6862745, 0, 1,
5.212121, 4.525252, -540.8934, 1, 0.6862745, 0, 1,
5.252525, 4.525252, -540.5655, 1, 0.6862745, 0, 1,
5.292929, 4.525252, -540.2535, 1, 0.6862745, 0, 1,
5.333333, 4.525252, -539.9575, 1, 0.6862745, 0, 1,
5.373737, 4.525252, -539.6774, 1, 0.6862745, 0, 1,
5.414141, 4.525252, -539.4133, 1, 0.6862745, 0, 1,
5.454545, 4.525252, -539.1651, 1, 0.6862745, 0, 1,
5.494949, 4.525252, -538.9329, 1, 0.6862745, 0, 1,
5.535354, 4.525252, -538.7166, 1, 0.6862745, 0, 1,
5.575758, 4.525252, -538.5162, 1, 0.6862745, 0, 1,
5.616162, 4.525252, -538.3318, 1, 0.6862745, 0, 1,
5.656566, 4.525252, -538.1633, 1, 0.6862745, 0, 1,
5.69697, 4.525252, -538.0107, 1, 0.6862745, 0, 1,
5.737374, 4.525252, -537.8741, 1, 0.6862745, 0, 1,
5.777778, 4.525252, -537.7535, 1, 0.6862745, 0, 1,
5.818182, 4.525252, -537.6488, 1, 0.6862745, 0, 1,
5.858586, 4.525252, -537.5601, 1, 0.6862745, 0, 1,
5.89899, 4.525252, -537.4872, 1, 0.6862745, 0, 1,
5.939394, 4.525252, -537.4304, 1, 0.6862745, 0, 1,
5.979798, 4.525252, -537.3895, 1, 0.6862745, 0, 1,
6.020202, 4.525252, -537.3645, 1, 0.6862745, 0, 1,
6.060606, 4.525252, -537.3555, 1, 0.6862745, 0, 1,
6.10101, 4.525252, -537.3624, 1, 0.6862745, 0, 1,
6.141414, 4.525252, -537.3852, 1, 0.6862745, 0, 1,
6.181818, 4.525252, -537.424, 1, 0.6862745, 0, 1,
6.222222, 4.525252, -537.4788, 1, 0.6862745, 0, 1,
6.262626, 4.525252, -537.5494, 1, 0.6862745, 0, 1,
6.30303, 4.525252, -537.636, 1, 0.6862745, 0, 1,
6.343434, 4.525252, -537.7386, 1, 0.6862745, 0, 1,
6.383838, 4.525252, -537.8571, 1, 0.6862745, 0, 1,
6.424242, 4.525252, -537.9916, 1, 0.6862745, 0, 1,
6.464646, 4.525252, -538.142, 1, 0.6862745, 0, 1,
6.505051, 4.525252, -538.3083, 1, 0.6862745, 0, 1,
6.545455, 4.525252, -538.4906, 1, 0.6862745, 0, 1,
6.585859, 4.525252, -538.6888, 1, 0.6862745, 0, 1,
6.626263, 4.525252, -538.903, 1, 0.6862745, 0, 1,
6.666667, 4.525252, -539.1332, 1, 0.6862745, 0, 1,
6.707071, 4.525252, -539.3792, 1, 0.6862745, 0, 1,
6.747475, 4.525252, -539.6412, 1, 0.6862745, 0, 1,
6.787879, 4.525252, -539.9192, 1, 0.6862745, 0, 1,
6.828283, 4.525252, -540.2131, 1, 0.6862745, 0, 1,
6.868687, 4.525252, -540.5229, 1, 0.6862745, 0, 1,
6.909091, 4.525252, -540.8487, 1, 0.6862745, 0, 1,
6.949495, 4.525252, -541.1904, 1, 0.6862745, 0, 1,
6.989899, 4.525252, -541.5481, 1, 0.6862745, 0, 1,
7.030303, 4.525252, -541.9218, 1, 0.6862745, 0, 1,
7.070707, 4.525252, -542.3113, 1, 0.6862745, 0, 1,
7.111111, 4.525252, -542.7168, 1, 0.6862745, 0, 1,
7.151515, 4.525252, -543.1382, 1, 0.5843138, 0, 1,
7.191919, 4.525252, -543.5756, 1, 0.5843138, 0, 1,
7.232323, 4.525252, -544.029, 1, 0.5843138, 0, 1,
7.272727, 4.525252, -544.4982, 1, 0.5843138, 0, 1,
7.313131, 4.525252, -544.9835, 1, 0.5843138, 0, 1,
7.353535, 4.525252, -545.4846, 1, 0.5843138, 0, 1,
7.393939, 4.525252, -546.0018, 1, 0.5843138, 0, 1,
7.434343, 4.525252, -546.5348, 1, 0.5843138, 0, 1,
7.474748, 4.525252, -547.0838, 1, 0.5843138, 0, 1,
7.515152, 4.525252, -547.6487, 1, 0.5843138, 0, 1,
7.555555, 4.525252, -548.2296, 1, 0.5843138, 0, 1,
7.59596, 4.525252, -548.8265, 1, 0.4823529, 0, 1,
7.636364, 4.525252, -549.4392, 1, 0.4823529, 0, 1,
7.676768, 4.525252, -550.0679, 1, 0.4823529, 0, 1,
7.717172, 4.525252, -550.7126, 1, 0.4823529, 0, 1,
7.757576, 4.525252, -551.3732, 1, 0.4823529, 0, 1,
7.79798, 4.525252, -552.0497, 1, 0.4823529, 0, 1,
7.838384, 4.525252, -552.7422, 1, 0.4823529, 0, 1,
7.878788, 4.525252, -553.4507, 1, 0.4823529, 0, 1,
7.919192, 4.525252, -554.1751, 1, 0.4823529, 0, 1,
7.959596, 4.525252, -554.9154, 1, 0.3764706, 0, 1,
8, 4.525252, -555.6717, 1, 0.3764706, 0, 1,
4, 4.575758, -558.7747, 1, 0.3764706, 0, 1,
4.040404, 4.575758, -557.9861, 1, 0.3764706, 0, 1,
4.080808, 4.575758, -557.2132, 1, 0.3764706, 0, 1,
4.121212, 4.575758, -556.4558, 1, 0.3764706, 0, 1,
4.161616, 4.575758, -555.7141, 1, 0.3764706, 0, 1,
4.20202, 4.575758, -554.9879, 1, 0.3764706, 0, 1,
4.242424, 4.575758, -554.2773, 1, 0.4823529, 0, 1,
4.282828, 4.575758, -553.5824, 1, 0.4823529, 0, 1,
4.323232, 4.575758, -552.903, 1, 0.4823529, 0, 1,
4.363636, 4.575758, -552.2393, 1, 0.4823529, 0, 1,
4.40404, 4.575758, -551.5911, 1, 0.4823529, 0, 1,
4.444445, 4.575758, -550.9584, 1, 0.4823529, 0, 1,
4.484848, 4.575758, -550.3414, 1, 0.4823529, 0, 1,
4.525252, 4.575758, -549.7401, 1, 0.4823529, 0, 1,
4.565657, 4.575758, -549.1542, 1, 0.4823529, 0, 1,
4.606061, 4.575758, -548.584, 1, 0.4823529, 0, 1,
4.646465, 4.575758, -548.0294, 1, 0.5843138, 0, 1,
4.686869, 4.575758, -547.4904, 1, 0.5843138, 0, 1,
4.727273, 4.575758, -546.9669, 1, 0.5843138, 0, 1,
4.767677, 4.575758, -546.4591, 1, 0.5843138, 0, 1,
4.808081, 4.575758, -545.9669, 1, 0.5843138, 0, 1,
4.848485, 4.575758, -545.4902, 1, 0.5843138, 0, 1,
4.888889, 4.575758, -545.0291, 1, 0.5843138, 0, 1,
4.929293, 4.575758, -544.5837, 1, 0.5843138, 0, 1,
4.969697, 4.575758, -544.1538, 1, 0.5843138, 0, 1,
5.010101, 4.575758, -543.7395, 1, 0.5843138, 0, 1,
5.050505, 4.575758, -543.3408, 1, 0.5843138, 0, 1,
5.090909, 4.575758, -542.9578, 1, 0.5843138, 0, 1,
5.131313, 4.575758, -542.5902, 1, 0.6862745, 0, 1,
5.171717, 4.575758, -542.2383, 1, 0.6862745, 0, 1,
5.212121, 4.575758, -541.902, 1, 0.6862745, 0, 1,
5.252525, 4.575758, -541.5813, 1, 0.6862745, 0, 1,
5.292929, 4.575758, -541.2762, 1, 0.6862745, 0, 1,
5.333333, 4.575758, -540.9866, 1, 0.6862745, 0, 1,
5.373737, 4.575758, -540.7127, 1, 0.6862745, 0, 1,
5.414141, 4.575758, -540.4544, 1, 0.6862745, 0, 1,
5.454545, 4.575758, -540.2117, 1, 0.6862745, 0, 1,
5.494949, 4.575758, -539.9845, 1, 0.6862745, 0, 1,
5.535354, 4.575758, -539.7729, 1, 0.6862745, 0, 1,
5.575758, 4.575758, -539.577, 1, 0.6862745, 0, 1,
5.616162, 4.575758, -539.3966, 1, 0.6862745, 0, 1,
5.656566, 4.575758, -539.2318, 1, 0.6862745, 0, 1,
5.69697, 4.575758, -539.0826, 1, 0.6862745, 0, 1,
5.737374, 4.575758, -538.949, 1, 0.6862745, 0, 1,
5.777778, 4.575758, -538.8311, 1, 0.6862745, 0, 1,
5.818182, 4.575758, -538.7286, 1, 0.6862745, 0, 1,
5.858586, 4.575758, -538.6418, 1, 0.6862745, 0, 1,
5.89899, 4.575758, -538.5706, 1, 0.6862745, 0, 1,
5.939394, 4.575758, -538.515, 1, 0.6862745, 0, 1,
5.979798, 4.575758, -538.475, 1, 0.6862745, 0, 1,
6.020202, 4.575758, -538.4506, 1, 0.6862745, 0, 1,
6.060606, 4.575758, -538.4417, 1, 0.6862745, 0, 1,
6.10101, 4.575758, -538.4485, 1, 0.6862745, 0, 1,
6.141414, 4.575758, -538.4708, 1, 0.6862745, 0, 1,
6.181818, 4.575758, -538.5087, 1, 0.6862745, 0, 1,
6.222222, 4.575758, -538.5623, 1, 0.6862745, 0, 1,
6.262626, 4.575758, -538.6314, 1, 0.6862745, 0, 1,
6.30303, 4.575758, -538.7161, 1, 0.6862745, 0, 1,
6.343434, 4.575758, -538.8165, 1, 0.6862745, 0, 1,
6.383838, 4.575758, -538.9324, 1, 0.6862745, 0, 1,
6.424242, 4.575758, -539.0639, 1, 0.6862745, 0, 1,
6.464646, 4.575758, -539.211, 1, 0.6862745, 0, 1,
6.505051, 4.575758, -539.3737, 1, 0.6862745, 0, 1,
6.545455, 4.575758, -539.5519, 1, 0.6862745, 0, 1,
6.585859, 4.575758, -539.7458, 1, 0.6862745, 0, 1,
6.626263, 4.575758, -539.9553, 1, 0.6862745, 0, 1,
6.666667, 4.575758, -540.1804, 1, 0.6862745, 0, 1,
6.707071, 4.575758, -540.4211, 1, 0.6862745, 0, 1,
6.747475, 4.575758, -540.6773, 1, 0.6862745, 0, 1,
6.787879, 4.575758, -540.9492, 1, 0.6862745, 0, 1,
6.828283, 4.575758, -541.2366, 1, 0.6862745, 0, 1,
6.868687, 4.575758, -541.5397, 1, 0.6862745, 0, 1,
6.909091, 4.575758, -541.8583, 1, 0.6862745, 0, 1,
6.949495, 4.575758, -542.1925, 1, 0.6862745, 0, 1,
6.989899, 4.575758, -542.5424, 1, 0.6862745, 0, 1,
7.030303, 4.575758, -542.9078, 1, 0.5843138, 0, 1,
7.070707, 4.575758, -543.2888, 1, 0.5843138, 0, 1,
7.111111, 4.575758, -543.6854, 1, 0.5843138, 0, 1,
7.151515, 4.575758, -544.0975, 1, 0.5843138, 0, 1,
7.191919, 4.575758, -544.5253, 1, 0.5843138, 0, 1,
7.232323, 4.575758, -544.9688, 1, 0.5843138, 0, 1,
7.272727, 4.575758, -545.4277, 1, 0.5843138, 0, 1,
7.313131, 4.575758, -545.9023, 1, 0.5843138, 0, 1,
7.353535, 4.575758, -546.3925, 1, 0.5843138, 0, 1,
7.393939, 4.575758, -546.8982, 1, 0.5843138, 0, 1,
7.434343, 4.575758, -547.4196, 1, 0.5843138, 0, 1,
7.474748, 4.575758, -547.9565, 1, 0.5843138, 0, 1,
7.515152, 4.575758, -548.509, 1, 0.5843138, 0, 1,
7.555555, 4.575758, -549.0771, 1, 0.4823529, 0, 1,
7.59596, 4.575758, -549.6609, 1, 0.4823529, 0, 1,
7.636364, 4.575758, -550.2603, 1, 0.4823529, 0, 1,
7.676768, 4.575758, -550.8751, 1, 0.4823529, 0, 1,
7.717172, 4.575758, -551.5057, 1, 0.4823529, 0, 1,
7.757576, 4.575758, -552.1517, 1, 0.4823529, 0, 1,
7.79798, 4.575758, -552.8135, 1, 0.4823529, 0, 1,
7.838384, 4.575758, -553.4907, 1, 0.4823529, 0, 1,
7.878788, 4.575758, -554.1837, 1, 0.4823529, 0, 1,
7.919192, 4.575758, -554.8921, 1, 0.3764706, 0, 1,
7.959596, 4.575758, -555.6162, 1, 0.3764706, 0, 1,
8, 4.575758, -556.3558, 1, 0.3764706, 0, 1,
4, 4.626263, -559.4319, 1, 0.3764706, 0, 1,
4.040404, 4.626263, -558.6605, 1, 0.3764706, 0, 1,
4.080808, 4.626263, -557.9044, 1, 0.3764706, 0, 1,
4.121212, 4.626263, -557.1635, 1, 0.3764706, 0, 1,
4.161616, 4.626263, -556.4378, 1, 0.3764706, 0, 1,
4.20202, 4.626263, -555.7274, 1, 0.3764706, 0, 1,
4.242424, 4.626263, -555.0323, 1, 0.3764706, 0, 1,
4.282828, 4.626263, -554.3524, 1, 0.3764706, 0, 1,
4.323232, 4.626263, -553.6878, 1, 0.4823529, 0, 1,
4.363636, 4.626263, -553.0385, 1, 0.4823529, 0, 1,
4.40404, 4.626263, -552.4044, 1, 0.4823529, 0, 1,
4.444445, 4.626263, -551.7855, 1, 0.4823529, 0, 1,
4.484848, 4.626263, -551.1819, 1, 0.4823529, 0, 1,
4.525252, 4.626263, -550.5935, 1, 0.4823529, 0, 1,
4.565657, 4.626263, -550.0204, 1, 0.4823529, 0, 1,
4.606061, 4.626263, -549.4626, 1, 0.4823529, 0, 1,
4.646465, 4.626263, -548.92, 1, 0.4823529, 0, 1,
4.686869, 4.626263, -548.3927, 1, 0.5843138, 0, 1,
4.727273, 4.626263, -547.8806, 1, 0.5843138, 0, 1,
4.767677, 4.626263, -547.3838, 1, 0.5843138, 0, 1,
4.808081, 4.626263, -546.9023, 1, 0.5843138, 0, 1,
4.848485, 4.626263, -546.436, 1, 0.5843138, 0, 1,
4.888889, 4.626263, -545.9849, 1, 0.5843138, 0, 1,
4.929293, 4.626263, -545.5491, 1, 0.5843138, 0, 1,
4.969697, 4.626263, -545.1286, 1, 0.5843138, 0, 1,
5.010101, 4.626263, -544.7233, 1, 0.5843138, 0, 1,
5.050505, 4.626263, -544.3333, 1, 0.5843138, 0, 1,
5.090909, 4.626263, -543.9585, 1, 0.5843138, 0, 1,
5.131313, 4.626263, -543.599, 1, 0.5843138, 0, 1,
5.171717, 4.626263, -543.2547, 1, 0.5843138, 0, 1,
5.212121, 4.626263, -542.9257, 1, 0.5843138, 0, 1,
5.252525, 4.626263, -542.6119, 1, 0.6862745, 0, 1,
5.292929, 4.626263, -542.3135, 1, 0.6862745, 0, 1,
5.333333, 4.626263, -542.0302, 1, 0.6862745, 0, 1,
5.373737, 4.626263, -541.7622, 1, 0.6862745, 0, 1,
5.414141, 4.626263, -541.5095, 1, 0.6862745, 0, 1,
5.454545, 4.626263, -541.272, 1, 0.6862745, 0, 1,
5.494949, 4.626263, -541.0498, 1, 0.6862745, 0, 1,
5.535354, 4.626263, -540.8428, 1, 0.6862745, 0, 1,
5.575758, 4.626263, -540.6511, 1, 0.6862745, 0, 1,
5.616162, 4.626263, -540.4747, 1, 0.6862745, 0, 1,
5.656566, 4.626263, -540.3135, 1, 0.6862745, 0, 1,
5.69697, 4.626263, -540.1675, 1, 0.6862745, 0, 1,
5.737374, 4.626263, -540.0369, 1, 0.6862745, 0, 1,
5.777778, 4.626263, -539.9214, 1, 0.6862745, 0, 1,
5.818182, 4.626263, -539.8212, 1, 0.6862745, 0, 1,
5.858586, 4.626263, -539.7363, 1, 0.6862745, 0, 1,
5.89899, 4.626263, -539.6666, 1, 0.6862745, 0, 1,
5.939394, 4.626263, -539.6122, 1, 0.6862745, 0, 1,
5.979798, 4.626263, -539.5731, 1, 0.6862745, 0, 1,
6.020202, 4.626263, -539.5492, 1, 0.6862745, 0, 1,
6.060606, 4.626263, -539.5405, 1, 0.6862745, 0, 1,
6.10101, 4.626263, -539.5471, 1, 0.6862745, 0, 1,
6.141414, 4.626263, -539.569, 1, 0.6862745, 0, 1,
6.181818, 4.626263, -539.6061, 1, 0.6862745, 0, 1,
6.222222, 4.626263, -539.6585, 1, 0.6862745, 0, 1,
6.262626, 4.626263, -539.7261, 1, 0.6862745, 0, 1,
6.30303, 4.626263, -539.809, 1, 0.6862745, 0, 1,
6.343434, 4.626263, -539.9072, 1, 0.6862745, 0, 1,
6.383838, 4.626263, -540.0206, 1, 0.6862745, 0, 1,
6.424242, 4.626263, -540.1492, 1, 0.6862745, 0, 1,
6.464646, 4.626263, -540.2931, 1, 0.6862745, 0, 1,
6.505051, 4.626263, -540.4523, 1, 0.6862745, 0, 1,
6.545455, 4.626263, -540.6267, 1, 0.6862745, 0, 1,
6.585859, 4.626263, -540.8163, 1, 0.6862745, 0, 1,
6.626263, 4.626263, -541.0213, 1, 0.6862745, 0, 1,
6.666667, 4.626263, -541.2415, 1, 0.6862745, 0, 1,
6.707071, 4.626263, -541.4769, 1, 0.6862745, 0, 1,
6.747475, 4.626263, -541.7276, 1, 0.6862745, 0, 1,
6.787879, 4.626263, -541.9935, 1, 0.6862745, 0, 1,
6.828283, 4.626263, -542.2747, 1, 0.6862745, 0, 1,
6.868687, 4.626263, -542.5712, 1, 0.6862745, 0, 1,
6.909091, 4.626263, -542.8829, 1, 0.5843138, 0, 1,
6.949495, 4.626263, -543.2099, 1, 0.5843138, 0, 1,
6.989899, 4.626263, -543.5521, 1, 0.5843138, 0, 1,
7.030303, 4.626263, -543.9096, 1, 0.5843138, 0, 1,
7.070707, 4.626263, -544.2823, 1, 0.5843138, 0, 1,
7.111111, 4.626263, -544.6703, 1, 0.5843138, 0, 1,
7.151515, 4.626263, -545.0735, 1, 0.5843138, 0, 1,
7.191919, 4.626263, -545.4921, 1, 0.5843138, 0, 1,
7.232323, 4.626263, -545.9258, 1, 0.5843138, 0, 1,
7.272727, 4.626263, -546.3748, 1, 0.5843138, 0, 1,
7.313131, 4.626263, -546.8391, 1, 0.5843138, 0, 1,
7.353535, 4.626263, -547.3186, 1, 0.5843138, 0, 1,
7.393939, 4.626263, -547.8134, 1, 0.5843138, 0, 1,
7.434343, 4.626263, -548.3234, 1, 0.5843138, 0, 1,
7.474748, 4.626263, -548.8487, 1, 0.4823529, 0, 1,
7.515152, 4.626263, -549.3892, 1, 0.4823529, 0, 1,
7.555555, 4.626263, -549.9451, 1, 0.4823529, 0, 1,
7.59596, 4.626263, -550.5161, 1, 0.4823529, 0, 1,
7.636364, 4.626263, -551.1024, 1, 0.4823529, 0, 1,
7.676768, 4.626263, -551.704, 1, 0.4823529, 0, 1,
7.717172, 4.626263, -552.3208, 1, 0.4823529, 0, 1,
7.757576, 4.626263, -552.9529, 1, 0.4823529, 0, 1,
7.79798, 4.626263, -553.6002, 1, 0.4823529, 0, 1,
7.838384, 4.626263, -554.2628, 1, 0.4823529, 0, 1,
7.878788, 4.626263, -554.9406, 1, 0.3764706, 0, 1,
7.919192, 4.626263, -555.6337, 1, 0.3764706, 0, 1,
7.959596, 4.626263, -556.342, 1, 0.3764706, 0, 1,
8, 4.626263, -557.0657, 1, 0.3764706, 0, 1,
4, 4.676768, -560.1149, 1, 0.3764706, 0, 1,
4.040404, 4.676768, -559.3601, 1, 0.3764706, 0, 1,
4.080808, 4.676768, -558.6202, 1, 0.3764706, 0, 1,
4.121212, 4.676768, -557.8952, 1, 0.3764706, 0, 1,
4.161616, 4.676768, -557.1852, 1, 0.3764706, 0, 1,
4.20202, 4.676768, -556.4901, 1, 0.3764706, 0, 1,
4.242424, 4.676768, -555.8098, 1, 0.3764706, 0, 1,
4.282828, 4.676768, -555.1445, 1, 0.3764706, 0, 1,
4.323232, 4.676768, -554.4942, 1, 0.3764706, 0, 1,
4.363636, 4.676768, -553.8588, 1, 0.4823529, 0, 1,
4.40404, 4.676768, -553.2383, 1, 0.4823529, 0, 1,
4.444445, 4.676768, -552.6328, 1, 0.4823529, 0, 1,
4.484848, 4.676768, -552.0421, 1, 0.4823529, 0, 1,
4.525252, 4.676768, -551.4664, 1, 0.4823529, 0, 1,
4.565657, 4.676768, -550.9056, 1, 0.4823529, 0, 1,
4.606061, 4.676768, -550.3598, 1, 0.4823529, 0, 1,
4.646465, 4.676768, -549.8289, 1, 0.4823529, 0, 1,
4.686869, 4.676768, -549.3129, 1, 0.4823529, 0, 1,
4.727273, 4.676768, -548.8118, 1, 0.4823529, 0, 1,
4.767677, 4.676768, -548.3256, 1, 0.5843138, 0, 1,
4.808081, 4.676768, -547.8544, 1, 0.5843138, 0, 1,
4.848485, 4.676768, -547.3981, 1, 0.5843138, 0, 1,
4.888889, 4.676768, -546.9568, 1, 0.5843138, 0, 1,
4.929293, 4.676768, -546.5303, 1, 0.5843138, 0, 1,
4.969697, 4.676768, -546.1188, 1, 0.5843138, 0, 1,
5.010101, 4.676768, -545.7223, 1, 0.5843138, 0, 1,
5.050505, 4.676768, -545.3406, 1, 0.5843138, 0, 1,
5.090909, 4.676768, -544.9739, 1, 0.5843138, 0, 1,
5.131313, 4.676768, -544.6221, 1, 0.5843138, 0, 1,
5.171717, 4.676768, -544.2852, 1, 0.5843138, 0, 1,
5.212121, 4.676768, -543.9633, 1, 0.5843138, 0, 1,
5.252525, 4.676768, -543.6563, 1, 0.5843138, 0, 1,
5.292929, 4.676768, -543.3642, 1, 0.5843138, 0, 1,
5.333333, 4.676768, -543.087, 1, 0.5843138, 0, 1,
5.373737, 4.676768, -542.8248, 1, 0.5843138, 0, 1,
5.414141, 4.676768, -542.5775, 1, 0.6862745, 0, 1,
5.454545, 4.676768, -542.3452, 1, 0.6862745, 0, 1,
5.494949, 4.676768, -542.1277, 1, 0.6862745, 0, 1,
5.535354, 4.676768, -541.9252, 1, 0.6862745, 0, 1,
5.575758, 4.676768, -541.7376, 1, 0.6862745, 0, 1,
5.616162, 4.676768, -541.5649, 1, 0.6862745, 0, 1,
5.656566, 4.676768, -541.4072, 1, 0.6862745, 0, 1,
5.69697, 4.676768, -541.2644, 1, 0.6862745, 0, 1,
5.737374, 4.676768, -541.1365, 1, 0.6862745, 0, 1,
5.777778, 4.676768, -541.0236, 1, 0.6862745, 0, 1,
5.818182, 4.676768, -540.9255, 1, 0.6862745, 0, 1,
5.858586, 4.676768, -540.8424, 1, 0.6862745, 0, 1,
5.89899, 4.676768, -540.7742, 1, 0.6862745, 0, 1,
5.939394, 4.676768, -540.721, 1, 0.6862745, 0, 1,
5.979798, 4.676768, -540.6827, 1, 0.6862745, 0, 1,
6.020202, 4.676768, -540.6593, 1, 0.6862745, 0, 1,
6.060606, 4.676768, -540.6508, 1, 0.6862745, 0, 1,
6.10101, 4.676768, -540.6573, 1, 0.6862745, 0, 1,
6.141414, 4.676768, -540.6787, 1, 0.6862745, 0, 1,
6.181818, 4.676768, -540.715, 1, 0.6862745, 0, 1,
6.222222, 4.676768, -540.7663, 1, 0.6862745, 0, 1,
6.262626, 4.676768, -540.8325, 1, 0.6862745, 0, 1,
6.30303, 4.676768, -540.9136, 1, 0.6862745, 0, 1,
6.343434, 4.676768, -541.0096, 1, 0.6862745, 0, 1,
6.383838, 4.676768, -541.1205, 1, 0.6862745, 0, 1,
6.424242, 4.676768, -541.2465, 1, 0.6862745, 0, 1,
6.464646, 4.676768, -541.3873, 1, 0.6862745, 0, 1,
6.505051, 4.676768, -541.543, 1, 0.6862745, 0, 1,
6.545455, 4.676768, -541.7137, 1, 0.6862745, 0, 1,
6.585859, 4.676768, -541.8993, 1, 0.6862745, 0, 1,
6.626263, 4.676768, -542.0998, 1, 0.6862745, 0, 1,
6.666667, 4.676768, -542.3152, 1, 0.6862745, 0, 1,
6.707071, 4.676768, -542.5456, 1, 0.6862745, 0, 1,
6.747475, 4.676768, -542.791, 1, 0.5843138, 0, 1,
6.787879, 4.676768, -543.0511, 1, 0.5843138, 0, 1,
6.828283, 4.676768, -543.3264, 1, 0.5843138, 0, 1,
6.868687, 4.676768, -543.6164, 1, 0.5843138, 0, 1,
6.909091, 4.676768, -543.9214, 1, 0.5843138, 0, 1,
6.949495, 4.676768, -544.2414, 1, 0.5843138, 0, 1,
6.989899, 4.676768, -544.5762, 1, 0.5843138, 0, 1,
7.030303, 4.676768, -544.926, 1, 0.5843138, 0, 1,
7.070707, 4.676768, -545.2908, 1, 0.5843138, 0, 1,
7.111111, 4.676768, -545.6704, 1, 0.5843138, 0, 1,
7.151515, 4.676768, -546.065, 1, 0.5843138, 0, 1,
7.191919, 4.676768, -546.4745, 1, 0.5843138, 0, 1,
7.232323, 4.676768, -546.899, 1, 0.5843138, 0, 1,
7.272727, 4.676768, -547.3383, 1, 0.5843138, 0, 1,
7.313131, 4.676768, -547.7926, 1, 0.5843138, 0, 1,
7.353535, 4.676768, -548.2618, 1, 0.5843138, 0, 1,
7.393939, 4.676768, -548.746, 1, 0.4823529, 0, 1,
7.434343, 4.676768, -549.2451, 1, 0.4823529, 0, 1,
7.474748, 4.676768, -549.7591, 1, 0.4823529, 0, 1,
7.515152, 4.676768, -550.288, 1, 0.4823529, 0, 1,
7.555555, 4.676768, -550.8318, 1, 0.4823529, 0, 1,
7.59596, 4.676768, -551.3906, 1, 0.4823529, 0, 1,
7.636364, 4.676768, -551.9644, 1, 0.4823529, 0, 1,
7.676768, 4.676768, -552.553, 1, 0.4823529, 0, 1,
7.717172, 4.676768, -553.1566, 1, 0.4823529, 0, 1,
7.757576, 4.676768, -553.7751, 1, 0.4823529, 0, 1,
7.79798, 4.676768, -554.4085, 1, 0.3764706, 0, 1,
7.838384, 4.676768, -555.0568, 1, 0.3764706, 0, 1,
7.878788, 4.676768, -555.7201, 1, 0.3764706, 0, 1,
7.919192, 4.676768, -556.3983, 1, 0.3764706, 0, 1,
7.959596, 4.676768, -557.0914, 1, 0.3764706, 0, 1,
8, 4.676768, -557.7995, 1, 0.3764706, 0, 1,
4, 4.727273, -560.8221, 1, 0.2745098, 0, 1,
4.040404, 4.727273, -560.0833, 1, 0.3764706, 0, 1,
4.080808, 4.727273, -559.3591, 1, 0.3764706, 0, 1,
4.121212, 4.727273, -558.6495, 1, 0.3764706, 0, 1,
4.161616, 4.727273, -557.9546, 1, 0.3764706, 0, 1,
4.20202, 4.727273, -557.2742, 1, 0.3764706, 0, 1,
4.242424, 4.727273, -556.6085, 1, 0.3764706, 0, 1,
4.282828, 4.727273, -555.9573, 1, 0.3764706, 0, 1,
4.323232, 4.727273, -555.3208, 1, 0.3764706, 0, 1,
4.363636, 4.727273, -554.6989, 1, 0.3764706, 0, 1,
4.40404, 4.727273, -554.0916, 1, 0.4823529, 0, 1,
4.444445, 4.727273, -553.4989, 1, 0.4823529, 0, 1,
4.484848, 4.727273, -552.9208, 1, 0.4823529, 0, 1,
4.525252, 4.727273, -552.3574, 1, 0.4823529, 0, 1,
4.565657, 4.727273, -551.8085, 1, 0.4823529, 0, 1,
4.606061, 4.727273, -551.2742, 1, 0.4823529, 0, 1,
4.646465, 4.727273, -550.7546, 1, 0.4823529, 0, 1,
4.686869, 4.727273, -550.2496, 1, 0.4823529, 0, 1,
4.727273, 4.727273, -549.7592, 1, 0.4823529, 0, 1,
4.767677, 4.727273, -549.2833, 1, 0.4823529, 0, 1,
4.808081, 4.727273, -548.8221, 1, 0.4823529, 0, 1,
4.848485, 4.727273, -548.3755, 1, 0.5843138, 0, 1,
4.888889, 4.727273, -547.9436, 1, 0.5843138, 0, 1,
4.929293, 4.727273, -547.5262, 1, 0.5843138, 0, 1,
4.969697, 4.727273, -547.1235, 1, 0.5843138, 0, 1,
5.010101, 4.727273, -546.7353, 1, 0.5843138, 0, 1,
5.050505, 4.727273, -546.3618, 1, 0.5843138, 0, 1,
5.090909, 4.727273, -546.0029, 1, 0.5843138, 0, 1,
5.131313, 4.727273, -545.6585, 1, 0.5843138, 0, 1,
5.171717, 4.727273, -545.3288, 1, 0.5843138, 0, 1,
5.212121, 4.727273, -545.0137, 1, 0.5843138, 0, 1,
5.252525, 4.727273, -544.7133, 1, 0.5843138, 0, 1,
5.292929, 4.727273, -544.4274, 1, 0.5843138, 0, 1,
5.333333, 4.727273, -544.1561, 1, 0.5843138, 0, 1,
5.373737, 4.727273, -543.8994, 1, 0.5843138, 0, 1,
5.414141, 4.727273, -543.6574, 1, 0.5843138, 0, 1,
5.454545, 4.727273, -543.43, 1, 0.5843138, 0, 1,
5.494949, 4.727273, -543.2172, 1, 0.5843138, 0, 1,
5.535354, 4.727273, -543.0189, 1, 0.5843138, 0, 1,
5.575758, 4.727273, -542.8353, 1, 0.5843138, 0, 1,
5.616162, 4.727273, -542.6663, 1, 0.6862745, 0, 1,
5.656566, 4.727273, -542.512, 1, 0.6862745, 0, 1,
5.69697, 4.727273, -542.3722, 1, 0.6862745, 0, 1,
5.737374, 4.727273, -542.247, 1, 0.6862745, 0, 1,
5.777778, 4.727273, -542.1365, 1, 0.6862745, 0, 1,
5.818182, 4.727273, -542.0405, 1, 0.6862745, 0, 1,
5.858586, 4.727273, -541.9592, 1, 0.6862745, 0, 1,
5.89899, 4.727273, -541.8925, 1, 0.6862745, 0, 1,
5.939394, 4.727273, -541.8403, 1, 0.6862745, 0, 1,
5.979798, 4.727273, -541.8029, 1, 0.6862745, 0, 1,
6.020202, 4.727273, -541.78, 1, 0.6862745, 0, 1,
6.060606, 4.727273, -541.7717, 1, 0.6862745, 0, 1,
6.10101, 4.727273, -541.778, 1, 0.6862745, 0, 1,
6.141414, 4.727273, -541.799, 1, 0.6862745, 0, 1,
6.181818, 4.727273, -541.8345, 1, 0.6862745, 0, 1,
6.222222, 4.727273, -541.8846, 1, 0.6862745, 0, 1,
6.262626, 4.727273, -541.9494, 1, 0.6862745, 0, 1,
6.30303, 4.727273, -542.0288, 1, 0.6862745, 0, 1,
6.343434, 4.727273, -542.1228, 1, 0.6862745, 0, 1,
6.383838, 4.727273, -542.2314, 1, 0.6862745, 0, 1,
6.424242, 4.727273, -542.3546, 1, 0.6862745, 0, 1,
6.464646, 4.727273, -542.4924, 1, 0.6862745, 0, 1,
6.505051, 4.727273, -542.6448, 1, 0.6862745, 0, 1,
6.545455, 4.727273, -542.8119, 1, 0.5843138, 0, 1,
6.585859, 4.727273, -542.9935, 1, 0.5843138, 0, 1,
6.626263, 4.727273, -543.1898, 1, 0.5843138, 0, 1,
6.666667, 4.727273, -543.4007, 1, 0.5843138, 0, 1,
6.707071, 4.727273, -543.6262, 1, 0.5843138, 0, 1,
6.747475, 4.727273, -543.8663, 1, 0.5843138, 0, 1,
6.787879, 4.727273, -544.121, 1, 0.5843138, 0, 1,
6.828283, 4.727273, -544.3903, 1, 0.5843138, 0, 1,
6.868687, 4.727273, -544.6742, 1, 0.5843138, 0, 1,
6.909091, 4.727273, -544.9727, 1, 0.5843138, 0, 1,
6.949495, 4.727273, -545.2859, 1, 0.5843138, 0, 1,
6.989899, 4.727273, -545.6136, 1, 0.5843138, 0, 1,
7.030303, 4.727273, -545.956, 1, 0.5843138, 0, 1,
7.070707, 4.727273, -546.313, 1, 0.5843138, 0, 1,
7.111111, 4.727273, -546.6846, 1, 0.5843138, 0, 1,
7.151515, 4.727273, -547.0708, 1, 0.5843138, 0, 1,
7.191919, 4.727273, -547.4716, 1, 0.5843138, 0, 1,
7.232323, 4.727273, -547.887, 1, 0.5843138, 0, 1,
7.272727, 4.727273, -548.317, 1, 0.5843138, 0, 1,
7.313131, 4.727273, -548.7617, 1, 0.4823529, 0, 1,
7.353535, 4.727273, -549.2209, 1, 0.4823529, 0, 1,
7.393939, 4.727273, -549.6948, 1, 0.4823529, 0, 1,
7.434343, 4.727273, -550.1832, 1, 0.4823529, 0, 1,
7.474748, 4.727273, -550.6863, 1, 0.4823529, 0, 1,
7.515152, 4.727273, -551.204, 1, 0.4823529, 0, 1,
7.555555, 4.727273, -551.7363, 1, 0.4823529, 0, 1,
7.59596, 4.727273, -552.2832, 1, 0.4823529, 0, 1,
7.636364, 4.727273, -552.8447, 1, 0.4823529, 0, 1,
7.676768, 4.727273, -553.4208, 1, 0.4823529, 0, 1,
7.717172, 4.727273, -554.0116, 1, 0.4823529, 0, 1,
7.757576, 4.727273, -554.6169, 1, 0.3764706, 0, 1,
7.79798, 4.727273, -555.2369, 1, 0.3764706, 0, 1,
7.838384, 4.727273, -555.8715, 1, 0.3764706, 0, 1,
7.878788, 4.727273, -556.5207, 1, 0.3764706, 0, 1,
7.919192, 4.727273, -557.1844, 1, 0.3764706, 0, 1,
7.959596, 4.727273, -557.8629, 1, 0.3764706, 0, 1,
8, 4.727273, -558.5558, 1, 0.3764706, 0, 1,
4, 4.777778, -561.5519, 1, 0.2745098, 0, 1,
4.040404, 4.777778, -560.8286, 1, 0.2745098, 0, 1,
4.080808, 4.777778, -560.1196, 1, 0.3764706, 0, 1,
4.121212, 4.777778, -559.425, 1, 0.3764706, 0, 1,
4.161616, 4.777778, -558.7446, 1, 0.3764706, 0, 1,
4.20202, 4.777778, -558.0786, 1, 0.3764706, 0, 1,
4.242424, 4.777778, -557.4269, 1, 0.3764706, 0, 1,
4.282828, 4.777778, -556.7894, 1, 0.3764706, 0, 1,
4.323232, 4.777778, -556.1663, 1, 0.3764706, 0, 1,
4.363636, 4.777778, -555.5574, 1, 0.3764706, 0, 1,
4.40404, 4.777778, -554.963, 1, 0.3764706, 0, 1,
4.444445, 4.777778, -554.3827, 1, 0.3764706, 0, 1,
4.484848, 4.777778, -553.8168, 1, 0.4823529, 0, 1,
4.525252, 4.777778, -553.2651, 1, 0.4823529, 0, 1,
4.565657, 4.777778, -552.7278, 1, 0.4823529, 0, 1,
4.606061, 4.777778, -552.2048, 1, 0.4823529, 0, 1,
4.646465, 4.777778, -551.6961, 1, 0.4823529, 0, 1,
4.686869, 4.777778, -551.2017, 1, 0.4823529, 0, 1,
4.727273, 4.777778, -550.7216, 1, 0.4823529, 0, 1,
4.767677, 4.777778, -550.2558, 1, 0.4823529, 0, 1,
4.808081, 4.777778, -549.8043, 1, 0.4823529, 0, 1,
4.848485, 4.777778, -549.3671, 1, 0.4823529, 0, 1,
4.888889, 4.777778, -548.9442, 1, 0.4823529, 0, 1,
4.929293, 4.777778, -548.5356, 1, 0.5843138, 0, 1,
4.969697, 4.777778, -548.1413, 1, 0.5843138, 0, 1,
5.010101, 4.777778, -547.7614, 1, 0.5843138, 0, 1,
5.050505, 4.777778, -547.3956, 1, 0.5843138, 0, 1,
5.090909, 4.777778, -547.0443, 1, 0.5843138, 0, 1,
5.131313, 4.777778, -546.7072, 1, 0.5843138, 0, 1,
5.171717, 4.777778, -546.3844, 1, 0.5843138, 0, 1,
5.212121, 4.777778, -546.0759, 1, 0.5843138, 0, 1,
5.252525, 4.777778, -545.7818, 1, 0.5843138, 0, 1,
5.292929, 4.777778, -545.5019, 1, 0.5843138, 0, 1,
5.333333, 4.777778, -545.2363, 1, 0.5843138, 0, 1,
5.373737, 4.777778, -544.9851, 1, 0.5843138, 0, 1,
5.414141, 4.777778, -544.7482, 1, 0.5843138, 0, 1,
5.454545, 4.777778, -544.5255, 1, 0.5843138, 0, 1,
5.494949, 4.777778, -544.3171, 1, 0.5843138, 0, 1,
5.535354, 4.777778, -544.1231, 1, 0.5843138, 0, 1,
5.575758, 4.777778, -543.9434, 1, 0.5843138, 0, 1,
5.616162, 4.777778, -543.7779, 1, 0.5843138, 0, 1,
5.656566, 4.777778, -543.6268, 1, 0.5843138, 0, 1,
5.69697, 4.777778, -543.4899, 1, 0.5843138, 0, 1,
5.737374, 4.777778, -543.3674, 1, 0.5843138, 0, 1,
5.777778, 4.777778, -543.2592, 1, 0.5843138, 0, 1,
5.818182, 4.777778, -543.1653, 1, 0.5843138, 0, 1,
5.858586, 4.777778, -543.0856, 1, 0.5843138, 0, 1,
5.89899, 4.777778, -543.0203, 1, 0.5843138, 0, 1,
5.939394, 4.777778, -542.9693, 1, 0.5843138, 0, 1,
5.979798, 4.777778, -542.9326, 1, 0.5843138, 0, 1,
6.020202, 4.777778, -542.9102, 1, 0.5843138, 0, 1,
6.060606, 4.777778, -542.9021, 1, 0.5843138, 0, 1,
6.10101, 4.777778, -542.9083, 1, 0.5843138, 0, 1,
6.141414, 4.777778, -542.9288, 1, 0.5843138, 0, 1,
6.181818, 4.777778, -542.9636, 1, 0.5843138, 0, 1,
6.222222, 4.777778, -543.0127, 1, 0.5843138, 0, 1,
6.262626, 4.777778, -543.0761, 1, 0.5843138, 0, 1,
6.30303, 4.777778, -543.1538, 1, 0.5843138, 0, 1,
6.343434, 4.777778, -543.2458, 1, 0.5843138, 0, 1,
6.383838, 4.777778, -543.3521, 1, 0.5843138, 0, 1,
6.424242, 4.777778, -543.4728, 1, 0.5843138, 0, 1,
6.464646, 4.777778, -543.6077, 1, 0.5843138, 0, 1,
6.505051, 4.777778, -543.7569, 1, 0.5843138, 0, 1,
6.545455, 4.777778, -543.9204, 1, 0.5843138, 0, 1,
6.585859, 4.777778, -544.0983, 1, 0.5843138, 0, 1,
6.626263, 4.777778, -544.2904, 1, 0.5843138, 0, 1,
6.666667, 4.777778, -544.4968, 1, 0.5843138, 0, 1,
6.707071, 4.777778, -544.7176, 1, 0.5843138, 0, 1,
6.747475, 4.777778, -544.9526, 1, 0.5843138, 0, 1,
6.787879, 4.777778, -545.202, 1, 0.5843138, 0, 1,
6.828283, 4.777778, -545.4656, 1, 0.5843138, 0, 1,
6.868687, 4.777778, -545.7436, 1, 0.5843138, 0, 1,
6.909091, 4.777778, -546.0358, 1, 0.5843138, 0, 1,
6.949495, 4.777778, -546.3424, 1, 0.5843138, 0, 1,
6.989899, 4.777778, -546.6633, 1, 0.5843138, 0, 1,
7.030303, 4.777778, -546.9984, 1, 0.5843138, 0, 1,
7.070707, 4.777778, -547.3479, 1, 0.5843138, 0, 1,
7.111111, 4.777778, -547.7117, 1, 0.5843138, 0, 1,
7.151515, 4.777778, -548.0897, 1, 0.5843138, 0, 1,
7.191919, 4.777778, -548.4821, 1, 0.5843138, 0, 1,
7.232323, 4.777778, -548.8888, 1, 0.4823529, 0, 1,
7.272727, 4.777778, -549.3098, 1, 0.4823529, 0, 1,
7.313131, 4.777778, -549.7451, 1, 0.4823529, 0, 1,
7.353535, 4.777778, -550.1946, 1, 0.4823529, 0, 1,
7.393939, 4.777778, -550.6586, 1, 0.4823529, 0, 1,
7.434343, 4.777778, -551.1367, 1, 0.4823529, 0, 1,
7.474748, 4.777778, -551.6292, 1, 0.4823529, 0, 1,
7.515152, 4.777778, -552.136, 1, 0.4823529, 0, 1,
7.555555, 4.777778, -552.6572, 1, 0.4823529, 0, 1,
7.59596, 4.777778, -553.1926, 1, 0.4823529, 0, 1,
7.636364, 4.777778, -553.7422, 1, 0.4823529, 0, 1,
7.676768, 4.777778, -554.3063, 1, 0.4823529, 0, 1,
7.717172, 4.777778, -554.8846, 1, 0.3764706, 0, 1,
7.757576, 4.777778, -555.4772, 1, 0.3764706, 0, 1,
7.79798, 4.777778, -556.0842, 1, 0.3764706, 0, 1,
7.838384, 4.777778, -556.7054, 1, 0.3764706, 0, 1,
7.878788, 4.777778, -557.3409, 1, 0.3764706, 0, 1,
7.919192, 4.777778, -557.9907, 1, 0.3764706, 0, 1,
7.959596, 4.777778, -558.6548, 1, 0.3764706, 0, 1,
8, 4.777778, -559.3333, 1, 0.3764706, 0, 1,
4, 4.828283, -562.3029, 1, 0.2745098, 0, 1,
4.040404, 4.828283, -561.5947, 1, 0.2745098, 0, 1,
4.080808, 4.828283, -560.9005, 1, 0.2745098, 0, 1,
4.121212, 4.828283, -560.2203, 1, 0.2745098, 0, 1,
4.161616, 4.828283, -559.5541, 1, 0.3764706, 0, 1,
4.20202, 4.828283, -558.9019, 1, 0.3764706, 0, 1,
4.242424, 4.828283, -558.2637, 1, 0.3764706, 0, 1,
4.282828, 4.828283, -557.6395, 1, 0.3764706, 0, 1,
4.323232, 4.828283, -557.0294, 1, 0.3764706, 0, 1,
4.363636, 4.828283, -556.4332, 1, 0.3764706, 0, 1,
4.40404, 4.828283, -555.851, 1, 0.3764706, 0, 1,
4.444445, 4.828283, -555.2829, 1, 0.3764706, 0, 1,
4.484848, 4.828283, -554.7288, 1, 0.3764706, 0, 1,
4.525252, 4.828283, -554.1886, 1, 0.4823529, 0, 1,
4.565657, 4.828283, -553.6625, 1, 0.4823529, 0, 1,
4.606061, 4.828283, -553.1503, 1, 0.4823529, 0, 1,
4.646465, 4.828283, -552.6522, 1, 0.4823529, 0, 1,
4.686869, 4.828283, -552.1681, 1, 0.4823529, 0, 1,
4.727273, 4.828283, -551.698, 1, 0.4823529, 0, 1,
4.767677, 4.828283, -551.2419, 1, 0.4823529, 0, 1,
4.808081, 4.828283, -550.7997, 1, 0.4823529, 0, 1,
4.848485, 4.828283, -550.3716, 1, 0.4823529, 0, 1,
4.888889, 4.828283, -549.9576, 1, 0.4823529, 0, 1,
4.929293, 4.828283, -549.5575, 1, 0.4823529, 0, 1,
4.969697, 4.828283, -549.1714, 1, 0.4823529, 0, 1,
5.010101, 4.828283, -548.7993, 1, 0.4823529, 0, 1,
5.050505, 4.828283, -548.4412, 1, 0.5843138, 0, 1,
5.090909, 4.828283, -548.0972, 1, 0.5843138, 0, 1,
5.131313, 4.828283, -547.7671, 1, 0.5843138, 0, 1,
5.171717, 4.828283, -547.451, 1, 0.5843138, 0, 1,
5.212121, 4.828283, -547.149, 1, 0.5843138, 0, 1,
5.252525, 4.828283, -546.861, 1, 0.5843138, 0, 1,
5.292929, 4.828283, -546.5869, 1, 0.5843138, 0, 1,
5.333333, 4.828283, -546.3269, 1, 0.5843138, 0, 1,
5.373737, 4.828283, -546.0809, 1, 0.5843138, 0, 1,
5.414141, 4.828283, -545.8488, 1, 0.5843138, 0, 1,
5.454545, 4.828283, -545.6308, 1, 0.5843138, 0, 1,
5.494949, 4.828283, -545.4268, 1, 0.5843138, 0, 1,
5.535354, 4.828283, -545.2368, 1, 0.5843138, 0, 1,
5.575758, 4.828283, -545.0608, 1, 0.5843138, 0, 1,
5.616162, 4.828283, -544.8988, 1, 0.5843138, 0, 1,
5.656566, 4.828283, -544.7508, 1, 0.5843138, 0, 1,
5.69697, 4.828283, -544.6168, 1, 0.5843138, 0, 1,
5.737374, 4.828283, -544.4968, 1, 0.5843138, 0, 1,
5.777778, 4.828283, -544.3909, 1, 0.5843138, 0, 1,
5.818182, 4.828283, -544.2989, 1, 0.5843138, 0, 1,
5.858586, 4.828283, -544.2209, 1, 0.5843138, 0, 1,
5.89899, 4.828283, -544.157, 1, 0.5843138, 0, 1,
5.939394, 4.828283, -544.107, 1, 0.5843138, 0, 1,
5.979798, 4.828283, -544.071, 1, 0.5843138, 0, 1,
6.020202, 4.828283, -544.0491, 1, 0.5843138, 0, 1,
6.060606, 4.828283, -544.0412, 1, 0.5843138, 0, 1,
6.10101, 4.828283, -544.0472, 1, 0.5843138, 0, 1,
6.141414, 4.828283, -544.0673, 1, 0.5843138, 0, 1,
6.181818, 4.828283, -544.1014, 1, 0.5843138, 0, 1,
6.222222, 4.828283, -544.1495, 1, 0.5843138, 0, 1,
6.262626, 4.828283, -544.2116, 1, 0.5843138, 0, 1,
6.30303, 4.828283, -544.2877, 1, 0.5843138, 0, 1,
6.343434, 4.828283, -544.3777, 1, 0.5843138, 0, 1,
6.383838, 4.828283, -544.4819, 1, 0.5843138, 0, 1,
6.424242, 4.828283, -544.6, 1, 0.5843138, 0, 1,
6.464646, 4.828283, -544.7321, 1, 0.5843138, 0, 1,
6.505051, 4.828283, -544.8782, 1, 0.5843138, 0, 1,
6.545455, 4.828283, -545.0383, 1, 0.5843138, 0, 1,
6.585859, 4.828283, -545.2125, 1, 0.5843138, 0, 1,
6.626263, 4.828283, -545.4006, 1, 0.5843138, 0, 1,
6.666667, 4.828283, -545.6028, 1, 0.5843138, 0, 1,
6.707071, 4.828283, -545.8189, 1, 0.5843138, 0, 1,
6.747475, 4.828283, -546.0491, 1, 0.5843138, 0, 1,
6.787879, 4.828283, -546.2932, 1, 0.5843138, 0, 1,
6.828283, 4.828283, -546.5514, 1, 0.5843138, 0, 1,
6.868687, 4.828283, -546.8235, 1, 0.5843138, 0, 1,
6.909091, 4.828283, -547.1097, 1, 0.5843138, 0, 1,
6.949495, 4.828283, -547.4099, 1, 0.5843138, 0, 1,
6.989899, 4.828283, -547.7241, 1, 0.5843138, 0, 1,
7.030303, 4.828283, -548.0523, 1, 0.5843138, 0, 1,
7.070707, 4.828283, -548.3945, 1, 0.5843138, 0, 1,
7.111111, 4.828283, -548.7507, 1, 0.4823529, 0, 1,
7.151515, 4.828283, -549.1209, 1, 0.4823529, 0, 1,
7.191919, 4.828283, -549.5051, 1, 0.4823529, 0, 1,
7.232323, 4.828283, -549.9033, 1, 0.4823529, 0, 1,
7.272727, 4.828283, -550.3156, 1, 0.4823529, 0, 1,
7.313131, 4.828283, -550.7418, 1, 0.4823529, 0, 1,
7.353535, 4.828283, -551.182, 1, 0.4823529, 0, 1,
7.393939, 4.828283, -551.6362, 1, 0.4823529, 0, 1,
7.434343, 4.828283, -552.1045, 1, 0.4823529, 0, 1,
7.474748, 4.828283, -552.5867, 1, 0.4823529, 0, 1,
7.515152, 4.828283, -553.083, 1, 0.4823529, 0, 1,
7.555555, 4.828283, -553.5933, 1, 0.4823529, 0, 1,
7.59596, 4.828283, -554.1175, 1, 0.4823529, 0, 1,
7.636364, 4.828283, -554.6558, 1, 0.3764706, 0, 1,
7.676768, 4.828283, -555.2081, 1, 0.3764706, 0, 1,
7.717172, 4.828283, -555.7744, 1, 0.3764706, 0, 1,
7.757576, 4.828283, -556.3546, 1, 0.3764706, 0, 1,
7.79798, 4.828283, -556.9489, 1, 0.3764706, 0, 1,
7.838384, 4.828283, -557.5573, 1, 0.3764706, 0, 1,
7.878788, 4.828283, -558.1795, 1, 0.3764706, 0, 1,
7.919192, 4.828283, -558.8159, 1, 0.3764706, 0, 1,
7.959596, 4.828283, -559.4661, 1, 0.3764706, 0, 1,
8, 4.828283, -560.1305, 1, 0.2745098, 0, 1,
4, 4.878788, -563.0737, 1, 0.2745098, 0, 1,
4.040404, 4.878788, -562.3801, 1, 0.2745098, 0, 1,
4.080808, 4.878788, -561.7002, 1, 0.2745098, 0, 1,
4.121212, 4.878788, -561.0341, 1, 0.2745098, 0, 1,
4.161616, 4.878788, -560.3816, 1, 0.2745098, 0, 1,
4.20202, 4.878788, -559.7428, 1, 0.3764706, 0, 1,
4.242424, 4.878788, -559.1178, 1, 0.3764706, 0, 1,
4.282828, 4.878788, -558.5065, 1, 0.3764706, 0, 1,
4.323232, 4.878788, -557.9089, 1, 0.3764706, 0, 1,
4.363636, 4.878788, -557.325, 1, 0.3764706, 0, 1,
4.40404, 4.878788, -556.7548, 1, 0.3764706, 0, 1,
4.444445, 4.878788, -556.1984, 1, 0.3764706, 0, 1,
4.484848, 4.878788, -555.6556, 1, 0.3764706, 0, 1,
4.525252, 4.878788, -555.1266, 1, 0.3764706, 0, 1,
4.565657, 4.878788, -554.6113, 1, 0.3764706, 0, 1,
4.606061, 4.878788, -554.1097, 1, 0.4823529, 0, 1,
4.646465, 4.878788, -553.6219, 1, 0.4823529, 0, 1,
4.686869, 4.878788, -553.1477, 1, 0.4823529, 0, 1,
4.727273, 4.878788, -552.6873, 1, 0.4823529, 0, 1,
4.767677, 4.878788, -552.2405, 1, 0.4823529, 0, 1,
4.808081, 4.878788, -551.8076, 1, 0.4823529, 0, 1,
4.848485, 4.878788, -551.3883, 1, 0.4823529, 0, 1,
4.888889, 4.878788, -550.9827, 1, 0.4823529, 0, 1,
4.929293, 4.878788, -550.5909, 1, 0.4823529, 0, 1,
4.969697, 4.878788, -550.2128, 1, 0.4823529, 0, 1,
5.010101, 4.878788, -549.8483, 1, 0.4823529, 0, 1,
5.050505, 4.878788, -549.4976, 1, 0.4823529, 0, 1,
5.090909, 4.878788, -549.1606, 1, 0.4823529, 0, 1,
5.131313, 4.878788, -548.8374, 1, 0.4823529, 0, 1,
5.171717, 4.878788, -548.5278, 1, 0.5843138, 0, 1,
5.212121, 4.878788, -548.232, 1, 0.5843138, 0, 1,
5.252525, 4.878788, -547.9499, 1, 0.5843138, 0, 1,
5.292929, 4.878788, -547.6815, 1, 0.5843138, 0, 1,
5.333333, 4.878788, -547.4268, 1, 0.5843138, 0, 1,
5.373737, 4.878788, -547.1859, 1, 0.5843138, 0, 1,
5.414141, 4.878788, -546.9586, 1, 0.5843138, 0, 1,
5.454545, 4.878788, -546.7451, 1, 0.5843138, 0, 1,
5.494949, 4.878788, -546.5453, 1, 0.5843138, 0, 1,
5.535354, 4.878788, -546.3592, 1, 0.5843138, 0, 1,
5.575758, 4.878788, -546.1868, 1, 0.5843138, 0, 1,
5.616162, 4.878788, -546.0281, 1, 0.5843138, 0, 1,
5.656566, 4.878788, -545.8832, 1, 0.5843138, 0, 1,
5.69697, 4.878788, -545.752, 1, 0.5843138, 0, 1,
5.737374, 4.878788, -545.6345, 1, 0.5843138, 0, 1,
5.777778, 4.878788, -545.5307, 1, 0.5843138, 0, 1,
5.818182, 4.878788, -545.4406, 1, 0.5843138, 0, 1,
5.858586, 4.878788, -545.3643, 1, 0.5843138, 0, 1,
5.89899, 4.878788, -545.3016, 1, 0.5843138, 0, 1,
5.939394, 4.878788, -545.2527, 1, 0.5843138, 0, 1,
5.979798, 4.878788, -545.2175, 1, 0.5843138, 0, 1,
6.020202, 4.878788, -545.196, 1, 0.5843138, 0, 1,
6.060606, 4.878788, -545.1882, 1, 0.5843138, 0, 1,
6.10101, 4.878788, -545.1942, 1, 0.5843138, 0, 1,
6.141414, 4.878788, -545.2138, 1, 0.5843138, 0, 1,
6.181818, 4.878788, -545.2472, 1, 0.5843138, 0, 1,
6.222222, 4.878788, -545.2943, 1, 0.5843138, 0, 1,
6.262626, 4.878788, -545.3551, 1, 0.5843138, 0, 1,
6.30303, 4.878788, -545.4296, 1, 0.5843138, 0, 1,
6.343434, 4.878788, -545.5178, 1, 0.5843138, 0, 1,
6.383838, 4.878788, -545.6198, 1, 0.5843138, 0, 1,
6.424242, 4.878788, -545.7355, 1, 0.5843138, 0, 1,
6.464646, 4.878788, -545.8649, 1, 0.5843138, 0, 1,
6.505051, 4.878788, -546.008, 1, 0.5843138, 0, 1,
6.545455, 4.878788, -546.1648, 1, 0.5843138, 0, 1,
6.585859, 4.878788, -546.3354, 1, 0.5843138, 0, 1,
6.626263, 4.878788, -546.5197, 1, 0.5843138, 0, 1,
6.666667, 4.878788, -546.7176, 1, 0.5843138, 0, 1,
6.707071, 4.878788, -546.9293, 1, 0.5843138, 0, 1,
6.747475, 4.878788, -547.1547, 1, 0.5843138, 0, 1,
6.787879, 4.878788, -547.3939, 1, 0.5843138, 0, 1,
6.828283, 4.878788, -547.6467, 1, 0.5843138, 0, 1,
6.868687, 4.878788, -547.9133, 1, 0.5843138, 0, 1,
6.909091, 4.878788, -548.1935, 1, 0.5843138, 0, 1,
6.949495, 4.878788, -548.4875, 1, 0.5843138, 0, 1,
6.989899, 4.878788, -548.7952, 1, 0.4823529, 0, 1,
7.030303, 4.878788, -549.1167, 1, 0.4823529, 0, 1,
7.070707, 4.878788, -549.4518, 1, 0.4823529, 0, 1,
7.111111, 4.878788, -549.8007, 1, 0.4823529, 0, 1,
7.151515, 4.878788, -550.1633, 1, 0.4823529, 0, 1,
7.191919, 4.878788, -550.5396, 1, 0.4823529, 0, 1,
7.232323, 4.878788, -550.9296, 1, 0.4823529, 0, 1,
7.272727, 4.878788, -551.3333, 1, 0.4823529, 0, 1,
7.313131, 4.878788, -551.7508, 1, 0.4823529, 0, 1,
7.353535, 4.878788, -552.1819, 1, 0.4823529, 0, 1,
7.393939, 4.878788, -552.6268, 1, 0.4823529, 0, 1,
7.434343, 4.878788, -553.0854, 1, 0.4823529, 0, 1,
7.474748, 4.878788, -553.5577, 1, 0.4823529, 0, 1,
7.515152, 4.878788, -554.0438, 1, 0.4823529, 0, 1,
7.555555, 4.878788, -554.5435, 1, 0.3764706, 0, 1,
7.59596, 4.878788, -555.057, 1, 0.3764706, 0, 1,
7.636364, 4.878788, -555.5842, 1, 0.3764706, 0, 1,
7.676768, 4.878788, -556.1251, 1, 0.3764706, 0, 1,
7.717172, 4.878788, -556.6797, 1, 0.3764706, 0, 1,
7.757576, 4.878788, -557.248, 1, 0.3764706, 0, 1,
7.79798, 4.878788, -557.8301, 1, 0.3764706, 0, 1,
7.838384, 4.878788, -558.4258, 1, 0.3764706, 0, 1,
7.878788, 4.878788, -559.0353, 1, 0.3764706, 0, 1,
7.919192, 4.878788, -559.6585, 1, 0.3764706, 0, 1,
7.959596, 4.878788, -560.2955, 1, 0.2745098, 0, 1,
8, 4.878788, -560.9461, 1, 0.2745098, 0, 1,
4, 4.929293, -563.8633, 1, 0.2745098, 0, 1,
4.040404, 4.929293, -563.1838, 1, 0.2745098, 0, 1,
4.080808, 4.929293, -562.5178, 1, 0.2745098, 0, 1,
4.121212, 4.929293, -561.8651, 1, 0.2745098, 0, 1,
4.161616, 4.929293, -561.226, 1, 0.2745098, 0, 1,
4.20202, 4.929293, -560.6002, 1, 0.2745098, 0, 1,
4.242424, 4.929293, -559.9879, 1, 0.3764706, 0, 1,
4.282828, 4.929293, -559.3891, 1, 0.3764706, 0, 1,
4.323232, 4.929293, -558.8036, 1, 0.3764706, 0, 1,
4.363636, 4.929293, -558.2317, 1, 0.3764706, 0, 1,
4.40404, 4.929293, -557.6732, 1, 0.3764706, 0, 1,
4.444445, 4.929293, -557.1281, 1, 0.3764706, 0, 1,
4.484848, 4.929293, -556.5964, 1, 0.3764706, 0, 1,
4.525252, 4.929293, -556.0781, 1, 0.3764706, 0, 1,
4.565657, 4.929293, -555.5734, 1, 0.3764706, 0, 1,
4.606061, 4.929293, -555.082, 1, 0.3764706, 0, 1,
4.646465, 4.929293, -554.6041, 1, 0.3764706, 0, 1,
4.686869, 4.929293, -554.1396, 1, 0.4823529, 0, 1,
4.727273, 4.929293, -553.6885, 1, 0.4823529, 0, 1,
4.767677, 4.929293, -553.2509, 1, 0.4823529, 0, 1,
4.808081, 4.929293, -552.8268, 1, 0.4823529, 0, 1,
4.848485, 4.929293, -552.416, 1, 0.4823529, 0, 1,
4.888889, 4.929293, -552.0187, 1, 0.4823529, 0, 1,
4.929293, 4.929293, -551.6349, 1, 0.4823529, 0, 1,
4.969697, 4.929293, -551.2645, 1, 0.4823529, 0, 1,
5.010101, 4.929293, -550.9075, 1, 0.4823529, 0, 1,
5.050505, 4.929293, -550.5639, 1, 0.4823529, 0, 1,
5.090909, 4.929293, -550.2338, 1, 0.4823529, 0, 1,
5.131313, 4.929293, -549.9171, 1, 0.4823529, 0, 1,
5.171717, 4.929293, -549.6139, 1, 0.4823529, 0, 1,
5.212121, 4.929293, -549.3241, 1, 0.4823529, 0, 1,
5.252525, 4.929293, -549.0477, 1, 0.4823529, 0, 1,
5.292929, 4.929293, -548.7848, 1, 0.4823529, 0, 1,
5.333333, 4.929293, -548.5353, 1, 0.5843138, 0, 1,
5.373737, 4.929293, -548.2993, 1, 0.5843138, 0, 1,
5.414141, 4.929293, -548.0767, 1, 0.5843138, 0, 1,
5.454545, 4.929293, -547.8675, 1, 0.5843138, 0, 1,
5.494949, 4.929293, -547.6718, 1, 0.5843138, 0, 1,
5.535354, 4.929293, -547.4894, 1, 0.5843138, 0, 1,
5.575758, 4.929293, -547.3206, 1, 0.5843138, 0, 1,
5.616162, 4.929293, -547.1652, 1, 0.5843138, 0, 1,
5.656566, 4.929293, -547.0232, 1, 0.5843138, 0, 1,
5.69697, 4.929293, -546.8947, 1, 0.5843138, 0, 1,
5.737374, 4.929293, -546.7795, 1, 0.5843138, 0, 1,
5.777778, 4.929293, -546.6779, 1, 0.5843138, 0, 1,
5.818182, 4.929293, -546.5896, 1, 0.5843138, 0, 1,
5.858586, 4.929293, -546.5148, 1, 0.5843138, 0, 1,
5.89899, 4.929293, -546.4534, 1, 0.5843138, 0, 1,
5.939394, 4.929293, -546.4055, 1, 0.5843138, 0, 1,
5.979798, 4.929293, -546.371, 1, 0.5843138, 0, 1,
6.020202, 4.929293, -546.35, 1, 0.5843138, 0, 1,
6.060606, 4.929293, -546.3423, 1, 0.5843138, 0, 1,
6.10101, 4.929293, -546.3481, 1, 0.5843138, 0, 1,
6.141414, 4.929293, -546.3674, 1, 0.5843138, 0, 1,
6.181818, 4.929293, -546.4001, 1, 0.5843138, 0, 1,
6.222222, 4.929293, -546.4462, 1, 0.5843138, 0, 1,
6.262626, 4.929293, -546.5058, 1, 0.5843138, 0, 1,
6.30303, 4.929293, -546.5789, 1, 0.5843138, 0, 1,
6.343434, 4.929293, -546.6653, 1, 0.5843138, 0, 1,
6.383838, 4.929293, -546.7651, 1, 0.5843138, 0, 1,
6.424242, 4.929293, -546.8785, 1, 0.5843138, 0, 1,
6.464646, 4.929293, -547.0052, 1, 0.5843138, 0, 1,
6.505051, 4.929293, -547.1454, 1, 0.5843138, 0, 1,
6.545455, 4.929293, -547.2991, 1, 0.5843138, 0, 1,
6.585859, 4.929293, -547.4661, 1, 0.5843138, 0, 1,
6.626263, 4.929293, -547.6466, 1, 0.5843138, 0, 1,
6.666667, 4.929293, -547.8406, 1, 0.5843138, 0, 1,
6.707071, 4.929293, -548.048, 1, 0.5843138, 0, 1,
6.747475, 4.929293, -548.2688, 1, 0.5843138, 0, 1,
6.787879, 4.929293, -548.5031, 1, 0.5843138, 0, 1,
6.828283, 4.929293, -548.7507, 1, 0.4823529, 0, 1,
6.868687, 4.929293, -549.0118, 1, 0.4823529, 0, 1,
6.909091, 4.929293, -549.2864, 1, 0.4823529, 0, 1,
6.949495, 4.929293, -549.5744, 1, 0.4823529, 0, 1,
6.989899, 4.929293, -549.8759, 1, 0.4823529, 0, 1,
7.030303, 4.929293, -550.1907, 1, 0.4823529, 0, 1,
7.070707, 4.929293, -550.519, 1, 0.4823529, 0, 1,
7.111111, 4.929293, -550.8608, 1, 0.4823529, 0, 1,
7.151515, 4.929293, -551.216, 1, 0.4823529, 0, 1,
7.191919, 4.929293, -551.5846, 1, 0.4823529, 0, 1,
7.232323, 4.929293, -551.9667, 1, 0.4823529, 0, 1,
7.272727, 4.929293, -552.3622, 1, 0.4823529, 0, 1,
7.313131, 4.929293, -552.7711, 1, 0.4823529, 0, 1,
7.353535, 4.929293, -553.1935, 1, 0.4823529, 0, 1,
7.393939, 4.929293, -553.6293, 1, 0.4823529, 0, 1,
7.434343, 4.929293, -554.0786, 1, 0.4823529, 0, 1,
7.474748, 4.929293, -554.5413, 1, 0.3764706, 0, 1,
7.515152, 4.929293, -555.0174, 1, 0.3764706, 0, 1,
7.555555, 4.929293, -555.507, 1, 0.3764706, 0, 1,
7.59596, 4.929293, -556.0099, 1, 0.3764706, 0, 1,
7.636364, 4.929293, -556.5264, 1, 0.3764706, 0, 1,
7.676768, 4.929293, -557.0563, 1, 0.3764706, 0, 1,
7.717172, 4.929293, -557.5995, 1, 0.3764706, 0, 1,
7.757576, 4.929293, -558.1563, 1, 0.3764706, 0, 1,
7.79798, 4.929293, -558.7265, 1, 0.3764706, 0, 1,
7.838384, 4.929293, -559.3101, 1, 0.3764706, 0, 1,
7.878788, 4.929293, -559.9072, 1, 0.3764706, 0, 1,
7.919192, 4.929293, -560.5177, 1, 0.2745098, 0, 1,
7.959596, 4.929293, -561.1416, 1, 0.2745098, 0, 1,
8, 4.929293, -561.779, 1, 0.2745098, 0, 1,
4, 4.979798, -564.6702, 1, 0.2745098, 0, 1,
4.040404, 4.979798, -564.0045, 1, 0.2745098, 0, 1,
4.080808, 4.979798, -563.3519, 1, 0.2745098, 0, 1,
4.121212, 4.979798, -562.7124, 1, 0.2745098, 0, 1,
4.161616, 4.979798, -562.0862, 1, 0.2745098, 0, 1,
4.20202, 4.979798, -561.4731, 1, 0.2745098, 0, 1,
4.242424, 4.979798, -560.8731, 1, 0.2745098, 0, 1,
4.282828, 4.979798, -560.2864, 1, 0.2745098, 0, 1,
4.323232, 4.979798, -559.7128, 1, 0.3764706, 0, 1,
4.363636, 4.979798, -559.1523, 1, 0.3764706, 0, 1,
4.40404, 4.979798, -558.605, 1, 0.3764706, 0, 1,
4.444445, 4.979798, -558.0709, 1, 0.3764706, 0, 1,
4.484848, 4.979798, -557.55, 1, 0.3764706, 0, 1,
4.525252, 4.979798, -557.0422, 1, 0.3764706, 0, 1,
4.565657, 4.979798, -556.5476, 1, 0.3764706, 0, 1,
4.606061, 4.979798, -556.0662, 1, 0.3764706, 0, 1,
4.646465, 4.979798, -555.5979, 1, 0.3764706, 0, 1,
4.686869, 4.979798, -555.1428, 1, 0.3764706, 0, 1,
4.727273, 4.979798, -554.7009, 1, 0.3764706, 0, 1,
4.767677, 4.979798, -554.2721, 1, 0.4823529, 0, 1,
4.808081, 4.979798, -553.8564, 1, 0.4823529, 0, 1,
4.848485, 4.979798, -553.454, 1, 0.4823529, 0, 1,
4.888889, 4.979798, -553.0648, 1, 0.4823529, 0, 1,
4.929293, 4.979798, -552.6886, 1, 0.4823529, 0, 1,
4.969697, 4.979798, -552.3257, 1, 0.4823529, 0, 1,
5.010101, 4.979798, -551.9759, 1, 0.4823529, 0, 1,
5.050505, 4.979798, -551.6393, 1, 0.4823529, 0, 1,
5.090909, 4.979798, -551.3159, 1, 0.4823529, 0, 1,
5.131313, 4.979798, -551.0056, 1, 0.4823529, 0, 1,
5.171717, 4.979798, -550.7084, 1, 0.4823529, 0, 1,
5.212121, 4.979798, -550.4245, 1, 0.4823529, 0, 1,
5.252525, 4.979798, -550.1537, 1, 0.4823529, 0, 1,
5.292929, 4.979798, -549.8961, 1, 0.4823529, 0, 1,
5.333333, 4.979798, -549.6516, 1, 0.4823529, 0, 1,
5.373737, 4.979798, -549.4203, 1, 0.4823529, 0, 1,
5.414141, 4.979798, -549.2022, 1, 0.4823529, 0, 1,
5.454545, 4.979798, -548.9973, 1, 0.4823529, 0, 1,
5.494949, 4.979798, -548.8055, 1, 0.4823529, 0, 1,
5.535354, 4.979798, -548.6269, 1, 0.4823529, 0, 1,
5.575758, 4.979798, -548.4614, 1, 0.5843138, 0, 1,
5.616162, 4.979798, -548.3091, 1, 0.5843138, 0, 1,
5.656566, 4.979798, -548.17, 1, 0.5843138, 0, 1,
5.69697, 4.979798, -548.0441, 1, 0.5843138, 0, 1,
5.737374, 4.979798, -547.9313, 1, 0.5843138, 0, 1,
5.777778, 4.979798, -547.8316, 1, 0.5843138, 0, 1,
5.818182, 4.979798, -547.7452, 1, 0.5843138, 0, 1,
5.858586, 4.979798, -547.6719, 1, 0.5843138, 0, 1,
5.89899, 4.979798, -547.6118, 1, 0.5843138, 0, 1,
5.939394, 4.979798, -547.5648, 1, 0.5843138, 0, 1,
5.979798, 4.979798, -547.531, 1, 0.5843138, 0, 1,
6.020202, 4.979798, -547.5104, 1, 0.5843138, 0, 1,
6.060606, 4.979798, -547.5029, 1, 0.5843138, 0, 1,
6.10101, 4.979798, -547.5086, 1, 0.5843138, 0, 1,
6.141414, 4.979798, -547.5275, 1, 0.5843138, 0, 1,
6.181818, 4.979798, -547.5595, 1, 0.5843138, 0, 1,
6.222222, 4.979798, -547.6047, 1, 0.5843138, 0, 1,
6.262626, 4.979798, -547.6631, 1, 0.5843138, 0, 1,
6.30303, 4.979798, -547.7346, 1, 0.5843138, 0, 1,
6.343434, 4.979798, -547.8193, 1, 0.5843138, 0, 1,
6.383838, 4.979798, -547.9172, 1, 0.5843138, 0, 1,
6.424242, 4.979798, -548.0282, 1, 0.5843138, 0, 1,
6.464646, 4.979798, -548.1524, 1, 0.5843138, 0, 1,
6.505051, 4.979798, -548.2898, 1, 0.5843138, 0, 1,
6.545455, 4.979798, -548.4403, 1, 0.5843138, 0, 1,
6.585859, 4.979798, -548.604, 1, 0.4823529, 0, 1,
6.626263, 4.979798, -548.7809, 1, 0.4823529, 0, 1,
6.666667, 4.979798, -548.9709, 1, 0.4823529, 0, 1,
6.707071, 4.979798, -549.1741, 1, 0.4823529, 0, 1,
6.747475, 4.979798, -549.3904, 1, 0.4823529, 0, 1,
6.787879, 4.979798, -549.62, 1, 0.4823529, 0, 1,
6.828283, 4.979798, -549.8627, 1, 0.4823529, 0, 1,
6.868687, 4.979798, -550.1185, 1, 0.4823529, 0, 1,
6.909091, 4.979798, -550.3876, 1, 0.4823529, 0, 1,
6.949495, 4.979798, -550.6697, 1, 0.4823529, 0, 1,
6.989899, 4.979798, -550.9651, 1, 0.4823529, 0, 1,
7.030303, 4.979798, -551.2736, 1, 0.4823529, 0, 1,
7.070707, 4.979798, -551.5953, 1, 0.4823529, 0, 1,
7.111111, 4.979798, -551.9302, 1, 0.4823529, 0, 1,
7.151515, 4.979798, -552.2782, 1, 0.4823529, 0, 1,
7.191919, 4.979798, -552.6394, 1, 0.4823529, 0, 1,
7.232323, 4.979798, -553.0137, 1, 0.4823529, 0, 1,
7.272727, 4.979798, -553.4012, 1, 0.4823529, 0, 1,
7.313131, 4.979798, -553.8019, 1, 0.4823529, 0, 1,
7.353535, 4.979798, -554.2158, 1, 0.4823529, 0, 1,
7.393939, 4.979798, -554.6428, 1, 0.3764706, 0, 1,
7.434343, 4.979798, -555.083, 1, 0.3764706, 0, 1,
7.474748, 4.979798, -555.5364, 1, 0.3764706, 0, 1,
7.515152, 4.979798, -556.0029, 1, 0.3764706, 0, 1,
7.555555, 4.979798, -556.4825, 1, 0.3764706, 0, 1,
7.59596, 4.979798, -556.9754, 1, 0.3764706, 0, 1,
7.636364, 4.979798, -557.4814, 1, 0.3764706, 0, 1,
7.676768, 4.979798, -558.0006, 1, 0.3764706, 0, 1,
7.717172, 4.979798, -558.533, 1, 0.3764706, 0, 1,
7.757576, 4.979798, -559.0784, 1, 0.3764706, 0, 1,
7.79798, 4.979798, -559.6371, 1, 0.3764706, 0, 1,
7.838384, 4.979798, -560.209, 1, 0.2745098, 0, 1,
7.878788, 4.979798, -560.794, 1, 0.2745098, 0, 1,
7.919192, 4.979798, -561.3922, 1, 0.2745098, 0, 1,
7.959596, 4.979798, -562.0035, 1, 0.2745098, 0, 1,
8, 4.979798, -562.628, 1, 0.2745098, 0, 1,
4, 5.030303, -565.4935, 1, 0.2745098, 0, 1,
4.040404, 5.030303, -564.8411, 1, 0.2745098, 0, 1,
4.080808, 5.030303, -564.2015, 1, 0.2745098, 0, 1,
4.121212, 5.030303, -563.5748, 1, 0.2745098, 0, 1,
4.161616, 5.030303, -562.9611, 1, 0.2745098, 0, 1,
4.20202, 5.030303, -562.3602, 1, 0.2745098, 0, 1,
4.242424, 5.030303, -561.7723, 1, 0.2745098, 0, 1,
4.282828, 5.030303, -561.1973, 1, 0.2745098, 0, 1,
4.323232, 5.030303, -560.6351, 1, 0.2745098, 0, 1,
4.363636, 5.030303, -560.0859, 1, 0.3764706, 0, 1,
4.40404, 5.030303, -559.5496, 1, 0.3764706, 0, 1,
4.444445, 5.030303, -559.0261, 1, 0.3764706, 0, 1,
4.484848, 5.030303, -558.5156, 1, 0.3764706, 0, 1,
4.525252, 5.030303, -558.0179, 1, 0.3764706, 0, 1,
4.565657, 5.030303, -557.5332, 1, 0.3764706, 0, 1,
4.606061, 5.030303, -557.0614, 1, 0.3764706, 0, 1,
4.646465, 5.030303, -556.6025, 1, 0.3764706, 0, 1,
4.686869, 5.030303, -556.1564, 1, 0.3764706, 0, 1,
4.727273, 5.030303, -555.7233, 1, 0.3764706, 0, 1,
4.767677, 5.030303, -555.3032, 1, 0.3764706, 0, 1,
4.808081, 5.030303, -554.8958, 1, 0.3764706, 0, 1,
4.848485, 5.030303, -554.5014, 1, 0.3764706, 0, 1,
4.888889, 5.030303, -554.1199, 1, 0.4823529, 0, 1,
4.929293, 5.030303, -553.7513, 1, 0.4823529, 0, 1,
4.969697, 5.030303, -553.3956, 1, 0.4823529, 0, 1,
5.010101, 5.030303, -553.0529, 1, 0.4823529, 0, 1,
5.050505, 5.030303, -552.723, 1, 0.4823529, 0, 1,
5.090909, 5.030303, -552.4059, 1, 0.4823529, 0, 1,
5.131313, 5.030303, -552.1019, 1, 0.4823529, 0, 1,
5.171717, 5.030303, -551.8107, 1, 0.4823529, 0, 1,
5.212121, 5.030303, -551.5324, 1, 0.4823529, 0, 1,
5.252525, 5.030303, -551.267, 1, 0.4823529, 0, 1,
5.292929, 5.030303, -551.0146, 1, 0.4823529, 0, 1,
5.333333, 5.030303, -550.775, 1, 0.4823529, 0, 1,
5.373737, 5.030303, -550.5483, 1, 0.4823529, 0, 1,
5.414141, 5.030303, -550.3346, 1, 0.4823529, 0, 1,
5.454545, 5.030303, -550.1337, 1, 0.4823529, 0, 1,
5.494949, 5.030303, -549.9458, 1, 0.4823529, 0, 1,
5.535354, 5.030303, -549.7708, 1, 0.4823529, 0, 1,
5.575758, 5.030303, -549.6086, 1, 0.4823529, 0, 1,
5.616162, 5.030303, -549.4594, 1, 0.4823529, 0, 1,
5.656566, 5.030303, -549.323, 1, 0.4823529, 0, 1,
5.69697, 5.030303, -549.1995, 1, 0.4823529, 0, 1,
5.737374, 5.030303, -549.089, 1, 0.4823529, 0, 1,
5.777778, 5.030303, -548.9914, 1, 0.4823529, 0, 1,
5.818182, 5.030303, -548.9066, 1, 0.4823529, 0, 1,
5.858586, 5.030303, -548.8348, 1, 0.4823529, 0, 1,
5.89899, 5.030303, -548.7759, 1, 0.4823529, 0, 1,
5.939394, 5.030303, -548.7299, 1, 0.4823529, 0, 1,
5.979798, 5.030303, -548.6967, 1, 0.4823529, 0, 1,
6.020202, 5.030303, -548.6765, 1, 0.4823529, 0, 1,
6.060606, 5.030303, -548.6692, 1, 0.4823529, 0, 1,
6.10101, 5.030303, -548.6748, 1, 0.4823529, 0, 1,
6.141414, 5.030303, -548.6933, 1, 0.4823529, 0, 1,
6.181818, 5.030303, -548.7247, 1, 0.4823529, 0, 1,
6.222222, 5.030303, -548.769, 1, 0.4823529, 0, 1,
6.262626, 5.030303, -548.8262, 1, 0.4823529, 0, 1,
6.30303, 5.030303, -548.8963, 1, 0.4823529, 0, 1,
6.343434, 5.030303, -548.9793, 1, 0.4823529, 0, 1,
6.383838, 5.030303, -549.0752, 1, 0.4823529, 0, 1,
6.424242, 5.030303, -549.184, 1, 0.4823529, 0, 1,
6.464646, 5.030303, -549.3057, 1, 0.4823529, 0, 1,
6.505051, 5.030303, -549.4404, 1, 0.4823529, 0, 1,
6.545455, 5.030303, -549.5879, 1, 0.4823529, 0, 1,
6.585859, 5.030303, -549.7483, 1, 0.4823529, 0, 1,
6.626263, 5.030303, -549.9216, 1, 0.4823529, 0, 1,
6.666667, 5.030303, -550.1078, 1, 0.4823529, 0, 1,
6.707071, 5.030303, -550.307, 1, 0.4823529, 0, 1,
6.747475, 5.030303, -550.519, 1, 0.4823529, 0, 1,
6.787879, 5.030303, -550.744, 1, 0.4823529, 0, 1,
6.828283, 5.030303, -550.9818, 1, 0.4823529, 0, 1,
6.868687, 5.030303, -551.2326, 1, 0.4823529, 0, 1,
6.909091, 5.030303, -551.4962, 1, 0.4823529, 0, 1,
6.949495, 5.030303, -551.7728, 1, 0.4823529, 0, 1,
6.989899, 5.030303, -552.0623, 1, 0.4823529, 0, 1,
7.030303, 5.030303, -552.3646, 1, 0.4823529, 0, 1,
7.070707, 5.030303, -552.6799, 1, 0.4823529, 0, 1,
7.111111, 5.030303, -553.0081, 1, 0.4823529, 0, 1,
7.151515, 5.030303, -553.3491, 1, 0.4823529, 0, 1,
7.191919, 5.030303, -553.7031, 1, 0.4823529, 0, 1,
7.232323, 5.030303, -554.0699, 1, 0.4823529, 0, 1,
7.272727, 5.030303, -554.4497, 1, 0.3764706, 0, 1,
7.313131, 5.030303, -554.8424, 1, 0.3764706, 0, 1,
7.353535, 5.030303, -555.248, 1, 0.3764706, 0, 1,
7.393939, 5.030303, -555.6664, 1, 0.3764706, 0, 1,
7.434343, 5.030303, -556.0978, 1, 0.3764706, 0, 1,
7.474748, 5.030303, -556.5422, 1, 0.3764706, 0, 1,
7.515152, 5.030303, -556.9993, 1, 0.3764706, 0, 1,
7.555555, 5.030303, -557.4694, 1, 0.3764706, 0, 1,
7.59596, 5.030303, -557.9525, 1, 0.3764706, 0, 1,
7.636364, 5.030303, -558.4484, 1, 0.3764706, 0, 1,
7.676768, 5.030303, -558.9572, 1, 0.3764706, 0, 1,
7.717172, 5.030303, -559.4789, 1, 0.3764706, 0, 1,
7.757576, 5.030303, -560.0135, 1, 0.3764706, 0, 1,
7.79798, 5.030303, -560.561, 1, 0.2745098, 0, 1,
7.838384, 5.030303, -561.1214, 1, 0.2745098, 0, 1,
7.878788, 5.030303, -561.6948, 1, 0.2745098, 0, 1,
7.919192, 5.030303, -562.2809, 1, 0.2745098, 0, 1,
7.959596, 5.030303, -562.8801, 1, 0.2745098, 0, 1,
8, 5.030303, -563.4921, 1, 0.2745098, 0, 1,
4, 5.080808, -566.3322, 1, 0.1686275, 0, 1,
4.040404, 5.080808, -565.6926, 1, 0.2745098, 0, 1,
4.080808, 5.080808, -565.0657, 1, 0.2745098, 0, 1,
4.121212, 5.080808, -564.4514, 1, 0.2745098, 0, 1,
4.161616, 5.080808, -563.8498, 1, 0.2745098, 0, 1,
4.20202, 5.080808, -563.2608, 1, 0.2745098, 0, 1,
4.242424, 5.080808, -562.6845, 1, 0.2745098, 0, 1,
4.282828, 5.080808, -562.1208, 1, 0.2745098, 0, 1,
4.323232, 5.080808, -561.5698, 1, 0.2745098, 0, 1,
4.363636, 5.080808, -561.0314, 1, 0.2745098, 0, 1,
4.40404, 5.080808, -560.5057, 1, 0.2745098, 0, 1,
4.444445, 5.080808, -559.9926, 1, 0.3764706, 0, 1,
4.484848, 5.080808, -559.4922, 1, 0.3764706, 0, 1,
4.525252, 5.080808, -559.0044, 1, 0.3764706, 0, 1,
4.565657, 5.080808, -558.5293, 1, 0.3764706, 0, 1,
4.606061, 5.080808, -558.0668, 1, 0.3764706, 0, 1,
4.646465, 5.080808, -557.6169, 1, 0.3764706, 0, 1,
4.686869, 5.080808, -557.1797, 1, 0.3764706, 0, 1,
4.727273, 5.080808, -556.7552, 1, 0.3764706, 0, 1,
4.767677, 5.080808, -556.3433, 1, 0.3764706, 0, 1,
4.808081, 5.080808, -555.9441, 1, 0.3764706, 0, 1,
4.848485, 5.080808, -555.5575, 1, 0.3764706, 0, 1,
4.888889, 5.080808, -555.1835, 1, 0.3764706, 0, 1,
4.929293, 5.080808, -554.8222, 1, 0.3764706, 0, 1,
4.969697, 5.080808, -554.4736, 1, 0.3764706, 0, 1,
5.010101, 5.080808, -554.1376, 1, 0.4823529, 0, 1,
5.050505, 5.080808, -553.8142, 1, 0.4823529, 0, 1,
5.090909, 5.080808, -553.5035, 1, 0.4823529, 0, 1,
5.131313, 5.080808, -553.2054, 1, 0.4823529, 0, 1,
5.171717, 5.080808, -552.92, 1, 0.4823529, 0, 1,
5.212121, 5.080808, -552.6472, 1, 0.4823529, 0, 1,
5.252525, 5.080808, -552.3871, 1, 0.4823529, 0, 1,
5.292929, 5.080808, -552.1396, 1, 0.4823529, 0, 1,
5.333333, 5.080808, -551.9048, 1, 0.4823529, 0, 1,
5.373737, 5.080808, -551.6826, 1, 0.4823529, 0, 1,
5.414141, 5.080808, -551.4731, 1, 0.4823529, 0, 1,
5.454545, 5.080808, -551.2762, 1, 0.4823529, 0, 1,
5.494949, 5.080808, -551.0919, 1, 0.4823529, 0, 1,
5.535354, 5.080808, -550.9203, 1, 0.4823529, 0, 1,
5.575758, 5.080808, -550.7614, 1, 0.4823529, 0, 1,
5.616162, 5.080808, -550.6151, 1, 0.4823529, 0, 1,
5.656566, 5.080808, -550.4814, 1, 0.4823529, 0, 1,
5.69697, 5.080808, -550.3605, 1, 0.4823529, 0, 1,
5.737374, 5.080808, -550.2521, 1, 0.4823529, 0, 1,
5.777778, 5.080808, -550.1564, 1, 0.4823529, 0, 1,
5.818182, 5.080808, -550.0734, 1, 0.4823529, 0, 1,
5.858586, 5.080808, -550.0029, 1, 0.4823529, 0, 1,
5.89899, 5.080808, -549.9452, 1, 0.4823529, 0, 1,
5.939394, 5.080808, -549.9001, 1, 0.4823529, 0, 1,
5.979798, 5.080808, -549.8676, 1, 0.4823529, 0, 1,
6.020202, 5.080808, -549.8478, 1, 0.4823529, 0, 1,
6.060606, 5.080808, -549.8406, 1, 0.4823529, 0, 1,
6.10101, 5.080808, -549.8461, 1, 0.4823529, 0, 1,
6.141414, 5.080808, -549.8643, 1, 0.4823529, 0, 1,
6.181818, 5.080808, -549.895, 1, 0.4823529, 0, 1,
6.222222, 5.080808, -549.9384, 1, 0.4823529, 0, 1,
6.262626, 5.080808, -549.9945, 1, 0.4823529, 0, 1,
6.30303, 5.080808, -550.0632, 1, 0.4823529, 0, 1,
6.343434, 5.080808, -550.1446, 1, 0.4823529, 0, 1,
6.383838, 5.080808, -550.2386, 1, 0.4823529, 0, 1,
6.424242, 5.080808, -550.3453, 1, 0.4823529, 0, 1,
6.464646, 5.080808, -550.4646, 1, 0.4823529, 0, 1,
6.505051, 5.080808, -550.5966, 1, 0.4823529, 0, 1,
6.545455, 5.080808, -550.7411, 1, 0.4823529, 0, 1,
6.585859, 5.080808, -550.8984, 1, 0.4823529, 0, 1,
6.626263, 5.080808, -551.0683, 1, 0.4823529, 0, 1,
6.666667, 5.080808, -551.2509, 1, 0.4823529, 0, 1,
6.707071, 5.080808, -551.446, 1, 0.4823529, 0, 1,
6.747475, 5.080808, -551.6539, 1, 0.4823529, 0, 1,
6.787879, 5.080808, -551.8744, 1, 0.4823529, 0, 1,
6.828283, 5.080808, -552.1075, 1, 0.4823529, 0, 1,
6.868687, 5.080808, -552.3533, 1, 0.4823529, 0, 1,
6.909091, 5.080808, -552.6118, 1, 0.4823529, 0, 1,
6.949495, 5.080808, -552.8828, 1, 0.4823529, 0, 1,
6.989899, 5.080808, -553.1666, 1, 0.4823529, 0, 1,
7.030303, 5.080808, -553.463, 1, 0.4823529, 0, 1,
7.070707, 5.080808, -553.772, 1, 0.4823529, 0, 1,
7.111111, 5.080808, -554.0936, 1, 0.4823529, 0, 1,
7.151515, 5.080808, -554.4279, 1, 0.3764706, 0, 1,
7.191919, 5.080808, -554.7749, 1, 0.3764706, 0, 1,
7.232323, 5.080808, -555.1345, 1, 0.3764706, 0, 1,
7.272727, 5.080808, -555.5068, 1, 0.3764706, 0, 1,
7.313131, 5.080808, -555.8917, 1, 0.3764706, 0, 1,
7.353535, 5.080808, -556.2892, 1, 0.3764706, 0, 1,
7.393939, 5.080808, -556.6995, 1, 0.3764706, 0, 1,
7.434343, 5.080808, -557.1223, 1, 0.3764706, 0, 1,
7.474748, 5.080808, -557.5579, 1, 0.3764706, 0, 1,
7.515152, 5.080808, -558.006, 1, 0.3764706, 0, 1,
7.555555, 5.080808, -558.4668, 1, 0.3764706, 0, 1,
7.59596, 5.080808, -558.9402, 1, 0.3764706, 0, 1,
7.636364, 5.080808, -559.4263, 1, 0.3764706, 0, 1,
7.676768, 5.080808, -559.925, 1, 0.3764706, 0, 1,
7.717172, 5.080808, -560.4365, 1, 0.2745098, 0, 1,
7.757576, 5.080808, -560.9605, 1, 0.2745098, 0, 1,
7.79798, 5.080808, -561.4972, 1, 0.2745098, 0, 1,
7.838384, 5.080808, -562.0465, 1, 0.2745098, 0, 1,
7.878788, 5.080808, -562.6085, 1, 0.2745098, 0, 1,
7.919192, 5.080808, -563.1831, 1, 0.2745098, 0, 1,
7.959596, 5.080808, -563.7704, 1, 0.2745098, 0, 1,
8, 5.080808, -564.3703, 1, 0.2745098, 0, 1,
4, 5.131313, -567.1851, 1, 0.1686275, 0, 1,
4.040404, 5.131313, -566.558, 1, 0.1686275, 0, 1,
4.080808, 5.131313, -565.9434, 1, 0.1686275, 0, 1,
4.121212, 5.131313, -565.3412, 1, 0.2745098, 0, 1,
4.161616, 5.131313, -564.7513, 1, 0.2745098, 0, 1,
4.20202, 5.131313, -564.1739, 1, 0.2745098, 0, 1,
4.242424, 5.131313, -563.6089, 1, 0.2745098, 0, 1,
4.282828, 5.131313, -563.0563, 1, 0.2745098, 0, 1,
4.323232, 5.131313, -562.516, 1, 0.2745098, 0, 1,
4.363636, 5.131313, -561.9882, 1, 0.2745098, 0, 1,
4.40404, 5.131313, -561.4728, 1, 0.2745098, 0, 1,
4.444445, 5.131313, -560.9697, 1, 0.2745098, 0, 1,
4.484848, 5.131313, -560.4791, 1, 0.2745098, 0, 1,
4.525252, 5.131313, -560.0009, 1, 0.3764706, 0, 1,
4.565657, 5.131313, -559.535, 1, 0.3764706, 0, 1,
4.606061, 5.131313, -559.0816, 1, 0.3764706, 0, 1,
4.646465, 5.131313, -558.6406, 1, 0.3764706, 0, 1,
4.686869, 5.131313, -558.212, 1, 0.3764706, 0, 1,
4.727273, 5.131313, -557.7957, 1, 0.3764706, 0, 1,
4.767677, 5.131313, -557.3919, 1, 0.3764706, 0, 1,
4.808081, 5.131313, -557.0005, 1, 0.3764706, 0, 1,
4.848485, 5.131313, -556.6215, 1, 0.3764706, 0, 1,
4.888889, 5.131313, -556.2548, 1, 0.3764706, 0, 1,
4.929293, 5.131313, -555.9006, 1, 0.3764706, 0, 1,
4.969697, 5.131313, -555.5588, 1, 0.3764706, 0, 1,
5.010101, 5.131313, -555.2293, 1, 0.3764706, 0, 1,
5.050505, 5.131313, -554.9123, 1, 0.3764706, 0, 1,
5.090909, 5.131313, -554.6077, 1, 0.3764706, 0, 1,
5.131313, 5.131313, -554.3154, 1, 0.4823529, 0, 1,
5.171717, 5.131313, -554.0356, 1, 0.4823529, 0, 1,
5.212121, 5.131313, -553.7682, 1, 0.4823529, 0, 1,
5.252525, 5.131313, -553.5131, 1, 0.4823529, 0, 1,
5.292929, 5.131313, -553.2705, 1, 0.4823529, 0, 1,
5.333333, 5.131313, -553.0403, 1, 0.4823529, 0, 1,
5.373737, 5.131313, -552.8224, 1, 0.4823529, 0, 1,
5.414141, 5.131313, -552.6171, 1, 0.4823529, 0, 1,
5.454545, 5.131313, -552.424, 1, 0.4823529, 0, 1,
5.494949, 5.131313, -552.2434, 1, 0.4823529, 0, 1,
5.535354, 5.131313, -552.0751, 1, 0.4823529, 0, 1,
5.575758, 5.131313, -551.9193, 1, 0.4823529, 0, 1,
5.616162, 5.131313, -551.7759, 1, 0.4823529, 0, 1,
5.656566, 5.131313, -551.6448, 1, 0.4823529, 0, 1,
5.69697, 5.131313, -551.5262, 1, 0.4823529, 0, 1,
5.737374, 5.131313, -551.42, 1, 0.4823529, 0, 1,
5.777778, 5.131313, -551.3262, 1, 0.4823529, 0, 1,
5.818182, 5.131313, -551.2448, 1, 0.4823529, 0, 1,
5.858586, 5.131313, -551.1757, 1, 0.4823529, 0, 1,
5.89899, 5.131313, -551.1191, 1, 0.4823529, 0, 1,
5.939394, 5.131313, -551.0749, 1, 0.4823529, 0, 1,
5.979798, 5.131313, -551.043, 1, 0.4823529, 0, 1,
6.020202, 5.131313, -551.0236, 1, 0.4823529, 0, 1,
6.060606, 5.131313, -551.0166, 1, 0.4823529, 0, 1,
6.10101, 5.131313, -551.022, 1, 0.4823529, 0, 1,
6.141414, 5.131313, -551.0397, 1, 0.4823529, 0, 1,
6.181818, 5.131313, -551.0699, 1, 0.4823529, 0, 1,
6.222222, 5.131313, -551.1125, 1, 0.4823529, 0, 1,
6.262626, 5.131313, -551.1674, 1, 0.4823529, 0, 1,
6.30303, 5.131313, -551.2348, 1, 0.4823529, 0, 1,
6.343434, 5.131313, -551.3146, 1, 0.4823529, 0, 1,
6.383838, 5.131313, -551.4067, 1, 0.4823529, 0, 1,
6.424242, 5.131313, -551.5114, 1, 0.4823529, 0, 1,
6.464646, 5.131313, -551.6283, 1, 0.4823529, 0, 1,
6.505051, 5.131313, -551.7577, 1, 0.4823529, 0, 1,
6.545455, 5.131313, -551.8995, 1, 0.4823529, 0, 1,
6.585859, 5.131313, -552.0536, 1, 0.4823529, 0, 1,
6.626263, 5.131313, -552.2202, 1, 0.4823529, 0, 1,
6.666667, 5.131313, -552.3992, 1, 0.4823529, 0, 1,
6.707071, 5.131313, -552.5905, 1, 0.4823529, 0, 1,
6.747475, 5.131313, -552.7943, 1, 0.4823529, 0, 1,
6.787879, 5.131313, -553.0105, 1, 0.4823529, 0, 1,
6.828283, 5.131313, -553.2391, 1, 0.4823529, 0, 1,
6.868687, 5.131313, -553.48, 1, 0.4823529, 0, 1,
6.909091, 5.131313, -553.7334, 1, 0.4823529, 0, 1,
6.949495, 5.131313, -553.9991, 1, 0.4823529, 0, 1,
6.989899, 5.131313, -554.2773, 1, 0.4823529, 0, 1,
7.030303, 5.131313, -554.5679, 1, 0.3764706, 0, 1,
7.070707, 5.131313, -554.8709, 1, 0.3764706, 0, 1,
7.111111, 5.131313, -555.1863, 1, 0.3764706, 0, 1,
7.151515, 5.131313, -555.514, 1, 0.3764706, 0, 1,
7.191919, 5.131313, -555.8542, 1, 0.3764706, 0, 1,
7.232323, 5.131313, -556.2068, 1, 0.3764706, 0, 1,
7.272727, 5.131313, -556.5718, 1, 0.3764706, 0, 1,
7.313131, 5.131313, -556.9491, 1, 0.3764706, 0, 1,
7.353535, 5.131313, -557.3389, 1, 0.3764706, 0, 1,
7.393939, 5.131313, -557.7411, 1, 0.3764706, 0, 1,
7.434343, 5.131313, -558.1556, 1, 0.3764706, 0, 1,
7.474748, 5.131313, -558.5826, 1, 0.3764706, 0, 1,
7.515152, 5.131313, -559.022, 1, 0.3764706, 0, 1,
7.555555, 5.131313, -559.4738, 1, 0.3764706, 0, 1,
7.59596, 5.131313, -559.9379, 1, 0.3764706, 0, 1,
7.636364, 5.131313, -560.4145, 1, 0.2745098, 0, 1,
7.676768, 5.131313, -560.9035, 1, 0.2745098, 0, 1,
7.717172, 5.131313, -561.4048, 1, 0.2745098, 0, 1,
7.757576, 5.131313, -561.9186, 1, 0.2745098, 0, 1,
7.79798, 5.131313, -562.4448, 1, 0.2745098, 0, 1,
7.838384, 5.131313, -562.9834, 1, 0.2745098, 0, 1,
7.878788, 5.131313, -563.5344, 1, 0.2745098, 0, 1,
7.919192, 5.131313, -564.0977, 1, 0.2745098, 0, 1,
7.959596, 5.131313, -564.6735, 1, 0.2745098, 0, 1,
8, 5.131313, -565.2617, 1, 0.2745098, 0, 1,
4, 5.181818, -568.0513, 1, 0.1686275, 0, 1,
4.040404, 5.181818, -567.4365, 1, 0.1686275, 0, 1,
4.080808, 5.181818, -566.8337, 1, 0.1686275, 0, 1,
4.121212, 5.181818, -566.2432, 1, 0.1686275, 0, 1,
4.161616, 5.181818, -565.6648, 1, 0.2745098, 0, 1,
4.20202, 5.181818, -565.0986, 1, 0.2745098, 0, 1,
4.242424, 5.181818, -564.5445, 1, 0.2745098, 0, 1,
4.282828, 5.181818, -564.0026, 1, 0.2745098, 0, 1,
4.323232, 5.181818, -563.4728, 1, 0.2745098, 0, 1,
4.363636, 5.181818, -562.9553, 1, 0.2745098, 0, 1,
4.40404, 5.181818, -562.4498, 1, 0.2745098, 0, 1,
4.444445, 5.181818, -561.9565, 1, 0.2745098, 0, 1,
4.484848, 5.181818, -561.4755, 1, 0.2745098, 0, 1,
4.525252, 5.181818, -561.0065, 1, 0.2745098, 0, 1,
4.565657, 5.181818, -560.5497, 1, 0.2745098, 0, 1,
4.606061, 5.181818, -560.1051, 1, 0.3764706, 0, 1,
4.646465, 5.181818, -559.6726, 1, 0.3764706, 0, 1,
4.686869, 5.181818, -559.2523, 1, 0.3764706, 0, 1,
4.727273, 5.181818, -558.8441, 1, 0.3764706, 0, 1,
4.767677, 5.181818, -558.4481, 1, 0.3764706, 0, 1,
4.808081, 5.181818, -558.0643, 1, 0.3764706, 0, 1,
4.848485, 5.181818, -557.6926, 1, 0.3764706, 0, 1,
4.888889, 5.181818, -557.3331, 1, 0.3764706, 0, 1,
4.929293, 5.181818, -556.9858, 1, 0.3764706, 0, 1,
4.969697, 5.181818, -556.6506, 1, 0.3764706, 0, 1,
5.010101, 5.181818, -556.3275, 1, 0.3764706, 0, 1,
5.050505, 5.181818, -556.0167, 1, 0.3764706, 0, 1,
5.090909, 5.181818, -555.7179, 1, 0.3764706, 0, 1,
5.131313, 5.181818, -555.4313, 1, 0.3764706, 0, 1,
5.171717, 5.181818, -555.157, 1, 0.3764706, 0, 1,
5.212121, 5.181818, -554.8947, 1, 0.3764706, 0, 1,
5.252525, 5.181818, -554.6447, 1, 0.3764706, 0, 1,
5.292929, 5.181818, -554.4067, 1, 0.3764706, 0, 1,
5.333333, 5.181818, -554.181, 1, 0.4823529, 0, 1,
5.373737, 5.181818, -553.9673, 1, 0.4823529, 0, 1,
5.414141, 5.181818, -553.7659, 1, 0.4823529, 0, 1,
5.454545, 5.181818, -553.5766, 1, 0.4823529, 0, 1,
5.494949, 5.181818, -553.3995, 1, 0.4823529, 0, 1,
5.535354, 5.181818, -553.2346, 1, 0.4823529, 0, 1,
5.575758, 5.181818, -553.0817, 1, 0.4823529, 0, 1,
5.616162, 5.181818, -552.9411, 1, 0.4823529, 0, 1,
5.656566, 5.181818, -552.8126, 1, 0.4823529, 0, 1,
5.69697, 5.181818, -552.6963, 1, 0.4823529, 0, 1,
5.737374, 5.181818, -552.5921, 1, 0.4823529, 0, 1,
5.777778, 5.181818, -552.5001, 1, 0.4823529, 0, 1,
5.818182, 5.181818, -552.4202, 1, 0.4823529, 0, 1,
5.858586, 5.181818, -552.3525, 1, 0.4823529, 0, 1,
5.89899, 5.181818, -552.297, 1, 0.4823529, 0, 1,
5.939394, 5.181818, -552.2537, 1, 0.4823529, 0, 1,
5.979798, 5.181818, -552.2224, 1, 0.4823529, 0, 1,
6.020202, 5.181818, -552.2034, 1, 0.4823529, 0, 1,
6.060606, 5.181818, -552.1965, 1, 0.4823529, 0, 1,
6.10101, 5.181818, -552.2018, 1, 0.4823529, 0, 1,
6.141414, 5.181818, -552.2192, 1, 0.4823529, 0, 1,
6.181818, 5.181818, -552.2488, 1, 0.4823529, 0, 1,
6.222222, 5.181818, -552.2905, 1, 0.4823529, 0, 1,
6.262626, 5.181818, -552.3444, 1, 0.4823529, 0, 1,
6.30303, 5.181818, -552.4105, 1, 0.4823529, 0, 1,
6.343434, 5.181818, -552.4887, 1, 0.4823529, 0, 1,
6.383838, 5.181818, -552.5791, 1, 0.4823529, 0, 1,
6.424242, 5.181818, -552.6816, 1, 0.4823529, 0, 1,
6.464646, 5.181818, -552.7963, 1, 0.4823529, 0, 1,
6.505051, 5.181818, -552.9232, 1, 0.4823529, 0, 1,
6.545455, 5.181818, -553.0623, 1, 0.4823529, 0, 1,
6.585859, 5.181818, -553.2134, 1, 0.4823529, 0, 1,
6.626263, 5.181818, -553.3768, 1, 0.4823529, 0, 1,
6.666667, 5.181818, -553.5522, 1, 0.4823529, 0, 1,
6.707071, 5.181818, -553.7399, 1, 0.4823529, 0, 1,
6.747475, 5.181818, -553.9398, 1, 0.4823529, 0, 1,
6.787879, 5.181818, -554.1517, 1, 0.4823529, 0, 1,
6.828283, 5.181818, -554.3759, 1, 0.3764706, 0, 1,
6.868687, 5.181818, -554.6122, 1, 0.3764706, 0, 1,
6.909091, 5.181818, -554.8606, 1, 0.3764706, 0, 1,
6.949495, 5.181818, -555.1212, 1, 0.3764706, 0, 1,
6.989899, 5.181818, -555.394, 1, 0.3764706, 0, 1,
7.030303, 5.181818, -555.679, 1, 0.3764706, 0, 1,
7.070707, 5.181818, -555.976, 1, 0.3764706, 0, 1,
7.111111, 5.181818, -556.2853, 1, 0.3764706, 0, 1,
7.151515, 5.181818, -556.6067, 1, 0.3764706, 0, 1,
7.191919, 5.181818, -556.9403, 1, 0.3764706, 0, 1,
7.232323, 5.181818, -557.286, 1, 0.3764706, 0, 1,
7.272727, 5.181818, -557.6439, 1, 0.3764706, 0, 1,
7.313131, 5.181818, -558.014, 1, 0.3764706, 0, 1,
7.353535, 5.181818, -558.3962, 1, 0.3764706, 0, 1,
7.393939, 5.181818, -558.7905, 1, 0.3764706, 0, 1,
7.434343, 5.181818, -559.1971, 1, 0.3764706, 0, 1,
7.474748, 5.181818, -559.6158, 1, 0.3764706, 0, 1,
7.515152, 5.181818, -560.0466, 1, 0.3764706, 0, 1,
7.555555, 5.181818, -560.4896, 1, 0.2745098, 0, 1,
7.59596, 5.181818, -560.9448, 1, 0.2745098, 0, 1,
7.636364, 5.181818, -561.4121, 1, 0.2745098, 0, 1,
7.676768, 5.181818, -561.8916, 1, 0.2745098, 0, 1,
7.717172, 5.181818, -562.3832, 1, 0.2745098, 0, 1,
7.757576, 5.181818, -562.8871, 1, 0.2745098, 0, 1,
7.79798, 5.181818, -563.403, 1, 0.2745098, 0, 1,
7.838384, 5.181818, -563.9312, 1, 0.2745098, 0, 1,
7.878788, 5.181818, -564.4714, 1, 0.2745098, 0, 1,
7.919192, 5.181818, -565.0239, 1, 0.2745098, 0, 1,
7.959596, 5.181818, -565.5885, 1, 0.2745098, 0, 1,
8, 5.181818, -566.1652, 1, 0.1686275, 0, 1,
4, 5.232323, -568.9301, 1, 0.1686275, 0, 1,
4.040404, 5.232323, -568.327, 1, 0.1686275, 0, 1,
4.080808, 5.232323, -567.7359, 1, 0.1686275, 0, 1,
4.121212, 5.232323, -567.1567, 1, 0.1686275, 0, 1,
4.161616, 5.232323, -566.5894, 1, 0.1686275, 0, 1,
4.20202, 5.232323, -566.0341, 1, 0.1686275, 0, 1,
4.242424, 5.232323, -565.4907, 1, 0.2745098, 0, 1,
4.282828, 5.232323, -564.9592, 1, 0.2745098, 0, 1,
4.323232, 5.232323, -564.4396, 1, 0.2745098, 0, 1,
4.363636, 5.232323, -563.9319, 1, 0.2745098, 0, 1,
4.40404, 5.232323, -563.4362, 1, 0.2745098, 0, 1,
4.444445, 5.232323, -562.9525, 1, 0.2745098, 0, 1,
4.484848, 5.232323, -562.4805, 1, 0.2745098, 0, 1,
4.525252, 5.232323, -562.0206, 1, 0.2745098, 0, 1,
4.565657, 5.232323, -561.5726, 1, 0.2745098, 0, 1,
4.606061, 5.232323, -561.1365, 1, 0.2745098, 0, 1,
4.646465, 5.232323, -560.7123, 1, 0.2745098, 0, 1,
4.686869, 5.232323, -560.3001, 1, 0.2745098, 0, 1,
4.727273, 5.232323, -559.8998, 1, 0.3764706, 0, 1,
4.767677, 5.232323, -559.5114, 1, 0.3764706, 0, 1,
4.808081, 5.232323, -559.1349, 1, 0.3764706, 0, 1,
4.848485, 5.232323, -558.7704, 1, 0.3764706, 0, 1,
4.888889, 5.232323, -558.4178, 1, 0.3764706, 0, 1,
4.929293, 5.232323, -558.0771, 1, 0.3764706, 0, 1,
4.969697, 5.232323, -557.7484, 1, 0.3764706, 0, 1,
5.010101, 5.232323, -557.4315, 1, 0.3764706, 0, 1,
5.050505, 5.232323, -557.1266, 1, 0.3764706, 0, 1,
5.090909, 5.232323, -556.8336, 1, 0.3764706, 0, 1,
5.131313, 5.232323, -556.5526, 1, 0.3764706, 0, 1,
5.171717, 5.232323, -556.2834, 1, 0.3764706, 0, 1,
5.212121, 5.232323, -556.0262, 1, 0.3764706, 0, 1,
5.252525, 5.232323, -555.7809, 1, 0.3764706, 0, 1,
5.292929, 5.232323, -555.5476, 1, 0.3764706, 0, 1,
5.333333, 5.232323, -555.3262, 1, 0.3764706, 0, 1,
5.373737, 5.232323, -555.1167, 1, 0.3764706, 0, 1,
5.414141, 5.232323, -554.9191, 1, 0.3764706, 0, 1,
5.454545, 5.232323, -554.7335, 1, 0.3764706, 0, 1,
5.494949, 5.232323, -554.5598, 1, 0.3764706, 0, 1,
5.535354, 5.232323, -554.3979, 1, 0.3764706, 0, 1,
5.575758, 5.232323, -554.2481, 1, 0.4823529, 0, 1,
5.616162, 5.232323, -554.1102, 1, 0.4823529, 0, 1,
5.656566, 5.232323, -553.9841, 1, 0.4823529, 0, 1,
5.69697, 5.232323, -553.8701, 1, 0.4823529, 0, 1,
5.737374, 5.232323, -553.7679, 1, 0.4823529, 0, 1,
5.777778, 5.232323, -553.6776, 1, 0.4823529, 0, 1,
5.818182, 5.232323, -553.5993, 1, 0.4823529, 0, 1,
5.858586, 5.232323, -553.5329, 1, 0.4823529, 0, 1,
5.89899, 5.232323, -553.4785, 1, 0.4823529, 0, 1,
5.939394, 5.232323, -553.4359, 1, 0.4823529, 0, 1,
5.979798, 5.232323, -553.4053, 1, 0.4823529, 0, 1,
6.020202, 5.232323, -553.3866, 1, 0.4823529, 0, 1,
6.060606, 5.232323, -553.3799, 1, 0.4823529, 0, 1,
6.10101, 5.232323, -553.385, 1, 0.4823529, 0, 1,
6.141414, 5.232323, -553.4021, 1, 0.4823529, 0, 1,
6.181818, 5.232323, -553.4312, 1, 0.4823529, 0, 1,
6.222222, 5.232323, -553.4721, 1, 0.4823529, 0, 1,
6.262626, 5.232323, -553.525, 1, 0.4823529, 0, 1,
6.30303, 5.232323, -553.5897, 1, 0.4823529, 0, 1,
6.343434, 5.232323, -553.6664, 1, 0.4823529, 0, 1,
6.383838, 5.232323, -553.7551, 1, 0.4823529, 0, 1,
6.424242, 5.232323, -553.8557, 1, 0.4823529, 0, 1,
6.464646, 5.232323, -553.9682, 1, 0.4823529, 0, 1,
6.505051, 5.232323, -554.0926, 1, 0.4823529, 0, 1,
6.545455, 5.232323, -554.2289, 1, 0.4823529, 0, 1,
6.585859, 5.232323, -554.3773, 1, 0.3764706, 0, 1,
6.626263, 5.232323, -554.5375, 1, 0.3764706, 0, 1,
6.666667, 5.232323, -554.7096, 1, 0.3764706, 0, 1,
6.707071, 5.232323, -554.8936, 1, 0.3764706, 0, 1,
6.747475, 5.232323, -555.0896, 1, 0.3764706, 0, 1,
6.787879, 5.232323, -555.2975, 1, 0.3764706, 0, 1,
6.828283, 5.232323, -555.5173, 1, 0.3764706, 0, 1,
6.868687, 5.232323, -555.7491, 1, 0.3764706, 0, 1,
6.909091, 5.232323, -555.9928, 1, 0.3764706, 0, 1,
6.949495, 5.232323, -556.2484, 1, 0.3764706, 0, 1,
6.989899, 5.232323, -556.5159, 1, 0.3764706, 0, 1,
7.030303, 5.232323, -556.7954, 1, 0.3764706, 0, 1,
7.070707, 5.232323, -557.0868, 1, 0.3764706, 0, 1,
7.111111, 5.232323, -557.3901, 1, 0.3764706, 0, 1,
7.151515, 5.232323, -557.7053, 1, 0.3764706, 0, 1,
7.191919, 5.232323, -558.0325, 1, 0.3764706, 0, 1,
7.232323, 5.232323, -558.3716, 1, 0.3764706, 0, 1,
7.272727, 5.232323, -558.7226, 1, 0.3764706, 0, 1,
7.313131, 5.232323, -559.0856, 1, 0.3764706, 0, 1,
7.353535, 5.232323, -559.4604, 1, 0.3764706, 0, 1,
7.393939, 5.232323, -559.8472, 1, 0.3764706, 0, 1,
7.434343, 5.232323, -560.246, 1, 0.2745098, 0, 1,
7.474748, 5.232323, -560.6566, 1, 0.2745098, 0, 1,
7.515152, 5.232323, -561.0792, 1, 0.2745098, 0, 1,
7.555555, 5.232323, -561.5137, 1, 0.2745098, 0, 1,
7.59596, 5.232323, -561.9601, 1, 0.2745098, 0, 1,
7.636364, 5.232323, -562.4185, 1, 0.2745098, 0, 1,
7.676768, 5.232323, -562.8887, 1, 0.2745098, 0, 1,
7.717172, 5.232323, -563.3709, 1, 0.2745098, 0, 1,
7.757576, 5.232323, -563.8651, 1, 0.2745098, 0, 1,
7.79798, 5.232323, -564.3711, 1, 0.2745098, 0, 1,
7.838384, 5.232323, -564.8891, 1, 0.2745098, 0, 1,
7.878788, 5.232323, -565.419, 1, 0.2745098, 0, 1,
7.919192, 5.232323, -565.9608, 1, 0.1686275, 0, 1,
7.959596, 5.232323, -566.5146, 1, 0.1686275, 0, 1,
8, 5.232323, -567.0803, 1, 0.1686275, 0, 1,
4, 5.282828, -569.8205, 1, 0.1686275, 0, 1,
4.040404, 5.282828, -569.2289, 1, 0.1686275, 0, 1,
4.080808, 5.282828, -568.649, 1, 0.1686275, 0, 1,
4.121212, 5.282828, -568.0809, 1, 0.1686275, 0, 1,
4.161616, 5.282828, -567.5244, 1, 0.1686275, 0, 1,
4.20202, 5.282828, -566.9796, 1, 0.1686275, 0, 1,
4.242424, 5.282828, -566.4465, 1, 0.1686275, 0, 1,
4.282828, 5.282828, -565.9251, 1, 0.1686275, 0, 1,
4.323232, 5.282828, -565.4155, 1, 0.2745098, 0, 1,
4.363636, 5.282828, -564.9175, 1, 0.2745098, 0, 1,
4.40404, 5.282828, -564.4312, 1, 0.2745098, 0, 1,
4.444445, 5.282828, -563.9566, 1, 0.2745098, 0, 1,
4.484848, 5.282828, -563.4937, 1, 0.2745098, 0, 1,
4.525252, 5.282828, -563.0425, 1, 0.2745098, 0, 1,
4.565657, 5.282828, -562.603, 1, 0.2745098, 0, 1,
4.606061, 5.282828, -562.1752, 1, 0.2745098, 0, 1,
4.646465, 5.282828, -561.7592, 1, 0.2745098, 0, 1,
4.686869, 5.282828, -561.3547, 1, 0.2745098, 0, 1,
4.727273, 5.282828, -560.962, 1, 0.2745098, 0, 1,
4.767677, 5.282828, -560.5811, 1, 0.2745098, 0, 1,
4.808081, 5.282828, -560.2117, 1, 0.2745098, 0, 1,
4.848485, 5.282828, -559.8541, 1, 0.3764706, 0, 1,
4.888889, 5.282828, -559.5082, 1, 0.3764706, 0, 1,
4.929293, 5.282828, -559.1741, 1, 0.3764706, 0, 1,
4.969697, 5.282828, -558.8516, 1, 0.3764706, 0, 1,
5.010101, 5.282828, -558.5407, 1, 0.3764706, 0, 1,
5.050505, 5.282828, -558.2416, 1, 0.3764706, 0, 1,
5.090909, 5.282828, -557.9542, 1, 0.3764706, 0, 1,
5.131313, 5.282828, -557.6785, 1, 0.3764706, 0, 1,
5.171717, 5.282828, -557.4145, 1, 0.3764706, 0, 1,
5.212121, 5.282828, -557.1622, 1, 0.3764706, 0, 1,
5.252525, 5.282828, -556.9216, 1, 0.3764706, 0, 1,
5.292929, 5.282828, -556.6927, 1, 0.3764706, 0, 1,
5.333333, 5.282828, -556.4755, 1, 0.3764706, 0, 1,
5.373737, 5.282828, -556.27, 1, 0.3764706, 0, 1,
5.414141, 5.282828, -556.0762, 1, 0.3764706, 0, 1,
5.454545, 5.282828, -555.894, 1, 0.3764706, 0, 1,
5.494949, 5.282828, -555.7236, 1, 0.3764706, 0, 1,
5.535354, 5.282828, -555.5649, 1, 0.3764706, 0, 1,
5.575758, 5.282828, -555.4179, 1, 0.3764706, 0, 1,
5.616162, 5.282828, -555.2826, 1, 0.3764706, 0, 1,
5.656566, 5.282828, -555.1589, 1, 0.3764706, 0, 1,
5.69697, 5.282828, -555.047, 1, 0.3764706, 0, 1,
5.737374, 5.282828, -554.9468, 1, 0.3764706, 0, 1,
5.777778, 5.282828, -554.8583, 1, 0.3764706, 0, 1,
5.818182, 5.282828, -554.7814, 1, 0.3764706, 0, 1,
5.858586, 5.282828, -554.7163, 1, 0.3764706, 0, 1,
5.89899, 5.282828, -554.6629, 1, 0.3764706, 0, 1,
5.939394, 5.282828, -554.6212, 1, 0.3764706, 0, 1,
5.979798, 5.282828, -554.5911, 1, 0.3764706, 0, 1,
6.020202, 5.282828, -554.5728, 1, 0.3764706, 0, 1,
6.060606, 5.282828, -554.5662, 1, 0.3764706, 0, 1,
6.10101, 5.282828, -554.5712, 1, 0.3764706, 0, 1,
6.141414, 5.282828, -554.588, 1, 0.3764706, 0, 1,
6.181818, 5.282828, -554.6165, 1, 0.3764706, 0, 1,
6.222222, 5.282828, -554.6567, 1, 0.3764706, 0, 1,
6.262626, 5.282828, -554.7085, 1, 0.3764706, 0, 1,
6.30303, 5.282828, -554.7721, 1, 0.3764706, 0, 1,
6.343434, 5.282828, -554.8474, 1, 0.3764706, 0, 1,
6.383838, 5.282828, -554.9343, 1, 0.3764706, 0, 1,
6.424242, 5.282828, -555.033, 1, 0.3764706, 0, 1,
6.464646, 5.282828, -555.1433, 1, 0.3764706, 0, 1,
6.505051, 5.282828, -555.2654, 1, 0.3764706, 0, 1,
6.545455, 5.282828, -555.3991, 1, 0.3764706, 0, 1,
6.585859, 5.282828, -555.5446, 1, 0.3764706, 0, 1,
6.626263, 5.282828, -555.7017, 1, 0.3764706, 0, 1,
6.666667, 5.282828, -555.8706, 1, 0.3764706, 0, 1,
6.707071, 5.282828, -556.0511, 1, 0.3764706, 0, 1,
6.747475, 5.282828, -556.2434, 1, 0.3764706, 0, 1,
6.787879, 5.282828, -556.4473, 1, 0.3764706, 0, 1,
6.828283, 5.282828, -556.663, 1, 0.3764706, 0, 1,
6.868687, 5.282828, -556.8903, 1, 0.3764706, 0, 1,
6.909091, 5.282828, -557.1294, 1, 0.3764706, 0, 1,
6.949495, 5.282828, -557.3801, 1, 0.3764706, 0, 1,
6.989899, 5.282828, -557.6426, 1, 0.3764706, 0, 1,
7.030303, 5.282828, -557.9167, 1, 0.3764706, 0, 1,
7.070707, 5.282828, -558.2026, 1, 0.3764706, 0, 1,
7.111111, 5.282828, -558.5001, 1, 0.3764706, 0, 1,
7.151515, 5.282828, -558.8093, 1, 0.3764706, 0, 1,
7.191919, 5.282828, -559.1303, 1, 0.3764706, 0, 1,
7.232323, 5.282828, -559.463, 1, 0.3764706, 0, 1,
7.272727, 5.282828, -559.8073, 1, 0.3764706, 0, 1,
7.313131, 5.282828, -560.1633, 1, 0.2745098, 0, 1,
7.353535, 5.282828, -560.5311, 1, 0.2745098, 0, 1,
7.393939, 5.282828, -560.9105, 1, 0.2745098, 0, 1,
7.434343, 5.282828, -561.3016, 1, 0.2745098, 0, 1,
7.474748, 5.282828, -561.7045, 1, 0.2745098, 0, 1,
7.515152, 5.282828, -562.119, 1, 0.2745098, 0, 1,
7.555555, 5.282828, -562.5452, 1, 0.2745098, 0, 1,
7.59596, 5.282828, -562.9832, 1, 0.2745098, 0, 1,
7.636364, 5.282828, -563.4327, 1, 0.2745098, 0, 1,
7.676768, 5.282828, -563.8941, 1, 0.2745098, 0, 1,
7.717172, 5.282828, -564.3671, 1, 0.2745098, 0, 1,
7.757576, 5.282828, -564.8518, 1, 0.2745098, 0, 1,
7.79798, 5.282828, -565.3483, 1, 0.2745098, 0, 1,
7.838384, 5.282828, -565.8564, 1, 0.2745098, 0, 1,
7.878788, 5.282828, -566.3762, 1, 0.1686275, 0, 1,
7.919192, 5.282828, -566.9077, 1, 0.1686275, 0, 1,
7.959596, 5.282828, -567.4509, 1, 0.1686275, 0, 1,
8, 5.282828, -568.0059, 1, 0.1686275, 0, 1,
4, 5.333333, -570.7218, 1, 0.1686275, 0, 1,
4.040404, 5.333333, -570.1414, 1, 0.1686275, 0, 1,
4.080808, 5.333333, -569.5724, 1, 0.1686275, 0, 1,
4.121212, 5.333333, -569.015, 1, 0.1686275, 0, 1,
4.161616, 5.333333, -568.4689, 1, 0.1686275, 0, 1,
4.20202, 5.333333, -567.9344, 1, 0.1686275, 0, 1,
4.242424, 5.333333, -567.4114, 1, 0.1686275, 0, 1,
4.282828, 5.333333, -566.8998, 1, 0.1686275, 0, 1,
4.323232, 5.333333, -566.3998, 1, 0.1686275, 0, 1,
4.363636, 5.333333, -565.9112, 1, 0.1686275, 0, 1,
4.40404, 5.333333, -565.434, 1, 0.2745098, 0, 1,
4.444445, 5.333333, -564.9684, 1, 0.2745098, 0, 1,
4.484848, 5.333333, -564.5142, 1, 0.2745098, 0, 1,
4.525252, 5.333333, -564.0715, 1, 0.2745098, 0, 1,
4.565657, 5.333333, -563.6403, 1, 0.2745098, 0, 1,
4.606061, 5.333333, -563.2206, 1, 0.2745098, 0, 1,
4.646465, 5.333333, -562.8124, 1, 0.2745098, 0, 1,
4.686869, 5.333333, -562.4156, 1, 0.2745098, 0, 1,
4.727273, 5.333333, -562.0303, 1, 0.2745098, 0, 1,
4.767677, 5.333333, -561.6565, 1, 0.2745098, 0, 1,
4.808081, 5.333333, -561.2941, 1, 0.2745098, 0, 1,
4.848485, 5.333333, -560.9433, 1, 0.2745098, 0, 1,
4.888889, 5.333333, -560.6039, 1, 0.2745098, 0, 1,
4.929293, 5.333333, -560.276, 1, 0.2745098, 0, 1,
4.969697, 5.333333, -559.9596, 1, 0.3764706, 0, 1,
5.010101, 5.333333, -559.6547, 1, 0.3764706, 0, 1,
5.050505, 5.333333, -559.3611, 1, 0.3764706, 0, 1,
5.090909, 5.333333, -559.0792, 1, 0.3764706, 0, 1,
5.131313, 5.333333, -558.8087, 1, 0.3764706, 0, 1,
5.171717, 5.333333, -558.5496, 1, 0.3764706, 0, 1,
5.212121, 5.333333, -558.3021, 1, 0.3764706, 0, 1,
5.252525, 5.333333, -558.066, 1, 0.3764706, 0, 1,
5.292929, 5.333333, -557.8414, 1, 0.3764706, 0, 1,
5.333333, 5.333333, -557.6283, 1, 0.3764706, 0, 1,
5.373737, 5.333333, -557.4267, 1, 0.3764706, 0, 1,
5.414141, 5.333333, -557.2365, 1, 0.3764706, 0, 1,
5.454545, 5.333333, -557.0578, 1, 0.3764706, 0, 1,
5.494949, 5.333333, -556.8906, 1, 0.3764706, 0, 1,
5.535354, 5.333333, -556.7349, 1, 0.3764706, 0, 1,
5.575758, 5.333333, -556.5906, 1, 0.3764706, 0, 1,
5.616162, 5.333333, -556.4579, 1, 0.3764706, 0, 1,
5.656566, 5.333333, -556.3366, 1, 0.3764706, 0, 1,
5.69697, 5.333333, -556.2268, 1, 0.3764706, 0, 1,
5.737374, 5.333333, -556.1284, 1, 0.3764706, 0, 1,
5.777778, 5.333333, -556.0416, 1, 0.3764706, 0, 1,
5.818182, 5.333333, -555.9662, 1, 0.3764706, 0, 1,
5.858586, 5.333333, -555.9023, 1, 0.3764706, 0, 1,
5.89899, 5.333333, -555.8499, 1, 0.3764706, 0, 1,
5.939394, 5.333333, -555.809, 1, 0.3764706, 0, 1,
5.979798, 5.333333, -555.7795, 1, 0.3764706, 0, 1,
6.020202, 5.333333, -555.7615, 1, 0.3764706, 0, 1,
6.060606, 5.333333, -555.755, 1, 0.3764706, 0, 1,
6.10101, 5.333333, -555.7599, 1, 0.3764706, 0, 1,
6.141414, 5.333333, -555.7764, 1, 0.3764706, 0, 1,
6.181818, 5.333333, -555.8044, 1, 0.3764706, 0, 1,
6.222222, 5.333333, -555.8438, 1, 0.3764706, 0, 1,
6.262626, 5.333333, -555.8947, 1, 0.3764706, 0, 1,
6.30303, 5.333333, -555.957, 1, 0.3764706, 0, 1,
6.343434, 5.333333, -556.0309, 1, 0.3764706, 0, 1,
6.383838, 5.333333, -556.1162, 1, 0.3764706, 0, 1,
6.424242, 5.333333, -556.213, 1, 0.3764706, 0, 1,
6.464646, 5.333333, -556.3213, 1, 0.3764706, 0, 1,
6.505051, 5.333333, -556.441, 1, 0.3764706, 0, 1,
6.545455, 5.333333, -556.5723, 1, 0.3764706, 0, 1,
6.585859, 5.333333, -556.715, 1, 0.3764706, 0, 1,
6.626263, 5.333333, -556.8691, 1, 0.3764706, 0, 1,
6.666667, 5.333333, -557.0349, 1, 0.3764706, 0, 1,
6.707071, 5.333333, -557.212, 1, 0.3764706, 0, 1,
6.747475, 5.333333, -557.4006, 1, 0.3764706, 0, 1,
6.787879, 5.333333, -557.6007, 1, 0.3764706, 0, 1,
6.828283, 5.333333, -557.8123, 1, 0.3764706, 0, 1,
6.868687, 5.333333, -558.0353, 1, 0.3764706, 0, 1,
6.909091, 5.333333, -558.2699, 1, 0.3764706, 0, 1,
6.949495, 5.333333, -558.5159, 1, 0.3764706, 0, 1,
6.989899, 5.333333, -558.7734, 1, 0.3764706, 0, 1,
7.030303, 5.333333, -559.0424, 1, 0.3764706, 0, 1,
7.070707, 5.333333, -559.3229, 1, 0.3764706, 0, 1,
7.111111, 5.333333, -559.6148, 1, 0.3764706, 0, 1,
7.151515, 5.333333, -559.9182, 1, 0.3764706, 0, 1,
7.191919, 5.333333, -560.2331, 1, 0.2745098, 0, 1,
7.232323, 5.333333, -560.5594, 1, 0.2745098, 0, 1,
7.272727, 5.333333, -560.8973, 1, 0.2745098, 0, 1,
7.313131, 5.333333, -561.2466, 1, 0.2745098, 0, 1,
7.353535, 5.333333, -561.6074, 1, 0.2745098, 0, 1,
7.393939, 5.333333, -561.9797, 1, 0.2745098, 0, 1,
7.434343, 5.333333, -562.3635, 1, 0.2745098, 0, 1,
7.474748, 5.333333, -562.7587, 1, 0.2745098, 0, 1,
7.515152, 5.333333, -563.1654, 1, 0.2745098, 0, 1,
7.555555, 5.333333, -563.5836, 1, 0.2745098, 0, 1,
7.59596, 5.333333, -564.0133, 1, 0.2745098, 0, 1,
7.636364, 5.333333, -564.4545, 1, 0.2745098, 0, 1,
7.676768, 5.333333, -564.9071, 1, 0.2745098, 0, 1,
7.717172, 5.333333, -565.3712, 1, 0.2745098, 0, 1,
7.757576, 5.333333, -565.8468, 1, 0.2745098, 0, 1,
7.79798, 5.333333, -566.3339, 1, 0.1686275, 0, 1,
7.838384, 5.333333, -566.8324, 1, 0.1686275, 0, 1,
7.878788, 5.333333, -567.3424, 1, 0.1686275, 0, 1,
7.919192, 5.333333, -567.8639, 1, 0.1686275, 0, 1,
7.959596, 5.333333, -568.3969, 1, 0.1686275, 0, 1,
8, 5.333333, -568.9413, 1, 0.1686275, 0, 1,
4, 5.383838, -571.6332, 1, 0.1686275, 0, 1,
4.040404, 5.383838, -571.0636, 1, 0.1686275, 0, 1,
4.080808, 5.383838, -570.5052, 1, 0.1686275, 0, 1,
4.121212, 5.383838, -569.9582, 1, 0.1686275, 0, 1,
4.161616, 5.383838, -569.4224, 1, 0.1686275, 0, 1,
4.20202, 5.383838, -568.8979, 1, 0.1686275, 0, 1,
4.242424, 5.383838, -568.3846, 1, 0.1686275, 0, 1,
4.282828, 5.383838, -567.8826, 1, 0.1686275, 0, 1,
4.323232, 5.383838, -567.3919, 1, 0.1686275, 0, 1,
4.363636, 5.383838, -566.9124, 1, 0.1686275, 0, 1,
4.40404, 5.383838, -566.4442, 1, 0.1686275, 0, 1,
4.444445, 5.383838, -565.9872, 1, 0.1686275, 0, 1,
4.484848, 5.383838, -565.5416, 1, 0.2745098, 0, 1,
4.525252, 5.383838, -565.1071, 1, 0.2745098, 0, 1,
4.565657, 5.383838, -564.684, 1, 0.2745098, 0, 1,
4.606061, 5.383838, -564.2721, 1, 0.2745098, 0, 1,
4.646465, 5.383838, -563.8715, 1, 0.2745098, 0, 1,
4.686869, 5.383838, -563.4821, 1, 0.2745098, 0, 1,
4.727273, 5.383838, -563.104, 1, 0.2745098, 0, 1,
4.767677, 5.383838, -562.7372, 1, 0.2745098, 0, 1,
4.808081, 5.383838, -562.3816, 1, 0.2745098, 0, 1,
4.848485, 5.383838, -562.0373, 1, 0.2745098, 0, 1,
4.888889, 5.383838, -561.7042, 1, 0.2745098, 0, 1,
4.929293, 5.383838, -561.3824, 1, 0.2745098, 0, 1,
4.969697, 5.383838, -561.072, 1, 0.2745098, 0, 1,
5.010101, 5.383838, -560.7727, 1, 0.2745098, 0, 1,
5.050505, 5.383838, -560.4847, 1, 0.2745098, 0, 1,
5.090909, 5.383838, -560.208, 1, 0.2745098, 0, 1,
5.131313, 5.383838, -559.9426, 1, 0.3764706, 0, 1,
5.171717, 5.383838, -559.6884, 1, 0.3764706, 0, 1,
5.212121, 5.383838, -559.4454, 1, 0.3764706, 0, 1,
5.252525, 5.383838, -559.2137, 1, 0.3764706, 0, 1,
5.292929, 5.383838, -558.9933, 1, 0.3764706, 0, 1,
5.333333, 5.383838, -558.7842, 1, 0.3764706, 0, 1,
5.373737, 5.383838, -558.5864, 1, 0.3764706, 0, 1,
5.414141, 5.383838, -558.3997, 1, 0.3764706, 0, 1,
5.454545, 5.383838, -558.2244, 1, 0.3764706, 0, 1,
5.494949, 5.383838, -558.0603, 1, 0.3764706, 0, 1,
5.535354, 5.383838, -557.9075, 1, 0.3764706, 0, 1,
5.575758, 5.383838, -557.7659, 1, 0.3764706, 0, 1,
5.616162, 5.383838, -557.6356, 1, 0.3764706, 0, 1,
5.656566, 5.383838, -557.5166, 1, 0.3764706, 0, 1,
5.69697, 5.383838, -557.4089, 1, 0.3764706, 0, 1,
5.737374, 5.383838, -557.3124, 1, 0.3764706, 0, 1,
5.777778, 5.383838, -557.2271, 1, 0.3764706, 0, 1,
5.818182, 5.383838, -557.1531, 1, 0.3764706, 0, 1,
5.858586, 5.383838, -557.0905, 1, 0.3764706, 0, 1,
5.89899, 5.383838, -557.039, 1, 0.3764706, 0, 1,
5.939394, 5.383838, -556.9988, 1, 0.3764706, 0, 1,
5.979798, 5.383838, -556.9699, 1, 0.3764706, 0, 1,
6.020202, 5.383838, -556.9523, 1, 0.3764706, 0, 1,
6.060606, 5.383838, -556.9459, 1, 0.3764706, 0, 1,
6.10101, 5.383838, -556.9507, 1, 0.3764706, 0, 1,
6.141414, 5.383838, -556.9669, 1, 0.3764706, 0, 1,
6.181818, 5.383838, -556.9943, 1, 0.3764706, 0, 1,
6.222222, 5.383838, -557.033, 1, 0.3764706, 0, 1,
6.262626, 5.383838, -557.0829, 1, 0.3764706, 0, 1,
6.30303, 5.383838, -557.1441, 1, 0.3764706, 0, 1,
6.343434, 5.383838, -557.2166, 1, 0.3764706, 0, 1,
6.383838, 5.383838, -557.3004, 1, 0.3764706, 0, 1,
6.424242, 5.383838, -557.3953, 1, 0.3764706, 0, 1,
6.464646, 5.383838, -557.5016, 1, 0.3764706, 0, 1,
6.505051, 5.383838, -557.6191, 1, 0.3764706, 0, 1,
6.545455, 5.383838, -557.7479, 1, 0.3764706, 0, 1,
6.585859, 5.383838, -557.8879, 1, 0.3764706, 0, 1,
6.626263, 5.383838, -558.0392, 1, 0.3764706, 0, 1,
6.666667, 5.383838, -558.2018, 1, 0.3764706, 0, 1,
6.707071, 5.383838, -558.3757, 1, 0.3764706, 0, 1,
6.747475, 5.383838, -558.5608, 1, 0.3764706, 0, 1,
6.787879, 5.383838, -558.7571, 1, 0.3764706, 0, 1,
6.828283, 5.383838, -558.9648, 1, 0.3764706, 0, 1,
6.868687, 5.383838, -559.1837, 1, 0.3764706, 0, 1,
6.909091, 5.383838, -559.4138, 1, 0.3764706, 0, 1,
6.949495, 5.383838, -559.6553, 1, 0.3764706, 0, 1,
6.989899, 5.383838, -559.908, 1, 0.3764706, 0, 1,
7.030303, 5.383838, -560.1719, 1, 0.2745098, 0, 1,
7.070707, 5.383838, -560.4471, 1, 0.2745098, 0, 1,
7.111111, 5.383838, -560.7336, 1, 0.2745098, 0, 1,
7.151515, 5.383838, -561.0313, 1, 0.2745098, 0, 1,
7.191919, 5.383838, -561.3403, 1, 0.2745098, 0, 1,
7.232323, 5.383838, -561.6606, 1, 0.2745098, 0, 1,
7.272727, 5.383838, -561.9922, 1, 0.2745098, 0, 1,
7.313131, 5.383838, -562.335, 1, 0.2745098, 0, 1,
7.353535, 5.383838, -562.689, 1, 0.2745098, 0, 1,
7.393939, 5.383838, -563.0544, 1, 0.2745098, 0, 1,
7.434343, 5.383838, -563.431, 1, 0.2745098, 0, 1,
7.474748, 5.383838, -563.8188, 1, 0.2745098, 0, 1,
7.515152, 5.383838, -564.218, 1, 0.2745098, 0, 1,
7.555555, 5.383838, -564.6283, 1, 0.2745098, 0, 1,
7.59596, 5.383838, -565.05, 1, 0.2745098, 0, 1,
7.636364, 5.383838, -565.4829, 1, 0.2745098, 0, 1,
7.676768, 5.383838, -565.9271, 1, 0.1686275, 0, 1,
7.717172, 5.383838, -566.3825, 1, 0.1686275, 0, 1,
7.757576, 5.383838, -566.8492, 1, 0.1686275, 0, 1,
7.79798, 5.383838, -567.3272, 1, 0.1686275, 0, 1,
7.838384, 5.383838, -567.8164, 1, 0.1686275, 0, 1,
7.878788, 5.383838, -568.3169, 1, 0.1686275, 0, 1,
7.919192, 5.383838, -568.8287, 1, 0.1686275, 0, 1,
7.959596, 5.383838, -569.3517, 1, 0.1686275, 0, 1,
8, 5.383838, -569.886, 1, 0.1686275, 0, 1,
4, 5.434343, -572.554, 1, 0.06666667, 0, 1,
4.040404, 5.434343, -571.9949, 1, 0.06666667, 0, 1,
4.080808, 5.434343, -571.447, 1, 0.1686275, 0, 1,
4.121212, 5.434343, -570.91, 1, 0.1686275, 0, 1,
4.161616, 5.434343, -570.3842, 1, 0.1686275, 0, 1,
4.20202, 5.434343, -569.8693, 1, 0.1686275, 0, 1,
4.242424, 5.434343, -569.3655, 1, 0.1686275, 0, 1,
4.282828, 5.434343, -568.8728, 1, 0.1686275, 0, 1,
4.323232, 5.434343, -568.3912, 1, 0.1686275, 0, 1,
4.363636, 5.434343, -567.9205, 1, 0.1686275, 0, 1,
4.40404, 5.434343, -567.461, 1, 0.1686275, 0, 1,
4.444445, 5.434343, -567.0125, 1, 0.1686275, 0, 1,
4.484848, 5.434343, -566.5751, 1, 0.1686275, 0, 1,
4.525252, 5.434343, -566.1487, 1, 0.1686275, 0, 1,
4.565657, 5.434343, -565.7334, 1, 0.2745098, 0, 1,
4.606061, 5.434343, -565.3291, 1, 0.2745098, 0, 1,
4.646465, 5.434343, -564.9359, 1, 0.2745098, 0, 1,
4.686869, 5.434343, -564.5537, 1, 0.2745098, 0, 1,
4.727273, 5.434343, -564.1826, 1, 0.2745098, 0, 1,
4.767677, 5.434343, -563.8226, 1, 0.2745098, 0, 1,
4.808081, 5.434343, -563.4736, 1, 0.2745098, 0, 1,
4.848485, 5.434343, -563.1356, 1, 0.2745098, 0, 1,
4.888889, 5.434343, -562.8088, 1, 0.2745098, 0, 1,
4.929293, 5.434343, -562.4929, 1, 0.2745098, 0, 1,
4.969697, 5.434343, -562.1882, 1, 0.2745098, 0, 1,
5.010101, 5.434343, -561.8945, 1, 0.2745098, 0, 1,
5.050505, 5.434343, -561.6118, 1, 0.2745098, 0, 1,
5.090909, 5.434343, -561.3402, 1, 0.2745098, 0, 1,
5.131313, 5.434343, -561.0797, 1, 0.2745098, 0, 1,
5.171717, 5.434343, -560.8301, 1, 0.2745098, 0, 1,
5.212121, 5.434343, -560.5917, 1, 0.2745098, 0, 1,
5.252525, 5.434343, -560.3643, 1, 0.2745098, 0, 1,
5.292929, 5.434343, -560.148, 1, 0.2745098, 0, 1,
5.333333, 5.434343, -559.9427, 1, 0.3764706, 0, 1,
5.373737, 5.434343, -559.7485, 1, 0.3764706, 0, 1,
5.414141, 5.434343, -559.5654, 1, 0.3764706, 0, 1,
5.454545, 5.434343, -559.3933, 1, 0.3764706, 0, 1,
5.494949, 5.434343, -559.2322, 1, 0.3764706, 0, 1,
5.535354, 5.434343, -559.0823, 1, 0.3764706, 0, 1,
5.575758, 5.434343, -558.9433, 1, 0.3764706, 0, 1,
5.616162, 5.434343, -558.8154, 1, 0.3764706, 0, 1,
5.656566, 5.434343, -558.6986, 1, 0.3764706, 0, 1,
5.69697, 5.434343, -558.5928, 1, 0.3764706, 0, 1,
5.737374, 5.434343, -558.4981, 1, 0.3764706, 0, 1,
5.777778, 5.434343, -558.4145, 1, 0.3764706, 0, 1,
5.818182, 5.434343, -558.3419, 1, 0.3764706, 0, 1,
5.858586, 5.434343, -558.2803, 1, 0.3764706, 0, 1,
5.89899, 5.434343, -558.2299, 1, 0.3764706, 0, 1,
5.939394, 5.434343, -558.1904, 1, 0.3764706, 0, 1,
5.979798, 5.434343, -558.162, 1, 0.3764706, 0, 1,
6.020202, 5.434343, -558.1447, 1, 0.3764706, 0, 1,
6.060606, 5.434343, -558.1384, 1, 0.3764706, 0, 1,
6.10101, 5.434343, -558.1432, 1, 0.3764706, 0, 1,
6.141414, 5.434343, -558.1591, 1, 0.3764706, 0, 1,
6.181818, 5.434343, -558.186, 1, 0.3764706, 0, 1,
6.222222, 5.434343, -558.2239, 1, 0.3764706, 0, 1,
6.262626, 5.434343, -558.2729, 1, 0.3764706, 0, 1,
6.30303, 5.434343, -558.333, 1, 0.3764706, 0, 1,
6.343434, 5.434343, -558.4041, 1, 0.3764706, 0, 1,
6.383838, 5.434343, -558.4863, 1, 0.3764706, 0, 1,
6.424242, 5.434343, -558.5795, 1, 0.3764706, 0, 1,
6.464646, 5.434343, -558.6838, 1, 0.3764706, 0, 1,
6.505051, 5.434343, -558.7992, 1, 0.3764706, 0, 1,
6.545455, 5.434343, -558.9256, 1, 0.3764706, 0, 1,
6.585859, 5.434343, -559.063, 1, 0.3764706, 0, 1,
6.626263, 5.434343, -559.2115, 1, 0.3764706, 0, 1,
6.666667, 5.434343, -559.3712, 1, 0.3764706, 0, 1,
6.707071, 5.434343, -559.5417, 1, 0.3764706, 0, 1,
6.747475, 5.434343, -559.7234, 1, 0.3764706, 0, 1,
6.787879, 5.434343, -559.9162, 1, 0.3764706, 0, 1,
6.828283, 5.434343, -560.12, 1, 0.3764706, 0, 1,
6.868687, 5.434343, -560.3348, 1, 0.2745098, 0, 1,
6.909091, 5.434343, -560.5607, 1, 0.2745098, 0, 1,
6.949495, 5.434343, -560.7977, 1, 0.2745098, 0, 1,
6.989899, 5.434343, -561.0457, 1, 0.2745098, 0, 1,
7.030303, 5.434343, -561.3047, 1, 0.2745098, 0, 1,
7.070707, 5.434343, -561.5749, 1, 0.2745098, 0, 1,
7.111111, 5.434343, -561.8561, 1, 0.2745098, 0, 1,
7.151515, 5.434343, -562.1483, 1, 0.2745098, 0, 1,
7.191919, 5.434343, -562.4516, 1, 0.2745098, 0, 1,
7.232323, 5.434343, -562.7659, 1, 0.2745098, 0, 1,
7.272727, 5.434343, -563.0914, 1, 0.2745098, 0, 1,
7.313131, 5.434343, -563.4278, 1, 0.2745098, 0, 1,
7.353535, 5.434343, -563.7753, 1, 0.2745098, 0, 1,
7.393939, 5.434343, -564.1339, 1, 0.2745098, 0, 1,
7.434343, 5.434343, -564.5035, 1, 0.2745098, 0, 1,
7.474748, 5.434343, -564.8842, 1, 0.2745098, 0, 1,
7.515152, 5.434343, -565.2759, 1, 0.2745098, 0, 1,
7.555555, 5.434343, -565.6787, 1, 0.2745098, 0, 1,
7.59596, 5.434343, -566.0926, 1, 0.1686275, 0, 1,
7.636364, 5.434343, -566.5175, 1, 0.1686275, 0, 1,
7.676768, 5.434343, -566.9534, 1, 0.1686275, 0, 1,
7.717172, 5.434343, -567.4005, 1, 0.1686275, 0, 1,
7.757576, 5.434343, -567.8585, 1, 0.1686275, 0, 1,
7.79798, 5.434343, -568.3277, 1, 0.1686275, 0, 1,
7.838384, 5.434343, -568.8079, 1, 0.1686275, 0, 1,
7.878788, 5.434343, -569.2991, 1, 0.1686275, 0, 1,
7.919192, 5.434343, -569.8014, 1, 0.1686275, 0, 1,
7.959596, 5.434343, -570.3148, 1, 0.1686275, 0, 1,
8, 5.434343, -570.8392, 1, 0.1686275, 0, 1,
4, 5.484848, -573.4836, 1, 0.06666667, 0, 1,
4.040404, 5.484848, -572.9348, 1, 0.06666667, 0, 1,
4.080808, 5.484848, -572.3969, 1, 0.06666667, 0, 1,
4.121212, 5.484848, -571.8698, 1, 0.06666667, 0, 1,
4.161616, 5.484848, -571.3535, 1, 0.1686275, 0, 1,
4.20202, 5.484848, -570.8481, 1, 0.1686275, 0, 1,
4.242424, 5.484848, -570.3536, 1, 0.1686275, 0, 1,
4.282828, 5.484848, -569.8699, 1, 0.1686275, 0, 1,
4.323232, 5.484848, -569.397, 1, 0.1686275, 0, 1,
4.363636, 5.484848, -568.9351, 1, 0.1686275, 0, 1,
4.40404, 5.484848, -568.4839, 1, 0.1686275, 0, 1,
4.444445, 5.484848, -568.0437, 1, 0.1686275, 0, 1,
4.484848, 5.484848, -567.6143, 1, 0.1686275, 0, 1,
4.525252, 5.484848, -567.1957, 1, 0.1686275, 0, 1,
4.565657, 5.484848, -566.788, 1, 0.1686275, 0, 1,
4.606061, 5.484848, -566.3911, 1, 0.1686275, 0, 1,
4.646465, 5.484848, -566.0051, 1, 0.1686275, 0, 1,
4.686869, 5.484848, -565.6299, 1, 0.2745098, 0, 1,
4.727273, 5.484848, -565.2657, 1, 0.2745098, 0, 1,
4.767677, 5.484848, -564.9122, 1, 0.2745098, 0, 1,
4.808081, 5.484848, -564.5696, 1, 0.2745098, 0, 1,
4.848485, 5.484848, -564.2379, 1, 0.2745098, 0, 1,
4.888889, 5.484848, -563.917, 1, 0.2745098, 0, 1,
4.929293, 5.484848, -563.6069, 1, 0.2745098, 0, 1,
4.969697, 5.484848, -563.3077, 1, 0.2745098, 0, 1,
5.010101, 5.484848, -563.0194, 1, 0.2745098, 0, 1,
5.050505, 5.484848, -562.7419, 1, 0.2745098, 0, 1,
5.090909, 5.484848, -562.4753, 1, 0.2745098, 0, 1,
5.131313, 5.484848, -562.2195, 1, 0.2745098, 0, 1,
5.171717, 5.484848, -561.9747, 1, 0.2745098, 0, 1,
5.212121, 5.484848, -561.7406, 1, 0.2745098, 0, 1,
5.252525, 5.484848, -561.5174, 1, 0.2745098, 0, 1,
5.292929, 5.484848, -561.305, 1, 0.2745098, 0, 1,
5.333333, 5.484848, -561.1035, 1, 0.2745098, 0, 1,
5.373737, 5.484848, -560.9128, 1, 0.2745098, 0, 1,
5.414141, 5.484848, -560.733, 1, 0.2745098, 0, 1,
5.454545, 5.484848, -560.5641, 1, 0.2745098, 0, 1,
5.494949, 5.484848, -560.406, 1, 0.2745098, 0, 1,
5.535354, 5.484848, -560.2588, 1, 0.2745098, 0, 1,
5.575758, 5.484848, -560.1224, 1, 0.2745098, 0, 1,
5.616162, 5.484848, -559.9968, 1, 0.3764706, 0, 1,
5.656566, 5.484848, -559.8822, 1, 0.3764706, 0, 1,
5.69697, 5.484848, -559.7783, 1, 0.3764706, 0, 1,
5.737374, 5.484848, -559.6854, 1, 0.3764706, 0, 1,
5.777778, 5.484848, -559.6033, 1, 0.3764706, 0, 1,
5.818182, 5.484848, -559.532, 1, 0.3764706, 0, 1,
5.858586, 5.484848, -559.4716, 1, 0.3764706, 0, 1,
5.89899, 5.484848, -559.422, 1, 0.3764706, 0, 1,
5.939394, 5.484848, -559.3833, 1, 0.3764706, 0, 1,
5.979798, 5.484848, -559.3554, 1, 0.3764706, 0, 1,
6.020202, 5.484848, -559.3384, 1, 0.3764706, 0, 1,
6.060606, 5.484848, -559.3323, 1, 0.3764706, 0, 1,
6.10101, 5.484848, -559.337, 1, 0.3764706, 0, 1,
6.141414, 5.484848, -559.3525, 1, 0.3764706, 0, 1,
6.181818, 5.484848, -559.379, 1, 0.3764706, 0, 1,
6.222222, 5.484848, -559.4162, 1, 0.3764706, 0, 1,
6.262626, 5.484848, -559.4643, 1, 0.3764706, 0, 1,
6.30303, 5.484848, -559.5233, 1, 0.3764706, 0, 1,
6.343434, 5.484848, -559.5931, 1, 0.3764706, 0, 1,
6.383838, 5.484848, -559.6738, 1, 0.3764706, 0, 1,
6.424242, 5.484848, -559.7653, 1, 0.3764706, 0, 1,
6.464646, 5.484848, -559.8677, 1, 0.3764706, 0, 1,
6.505051, 5.484848, -559.9809, 1, 0.3764706, 0, 1,
6.545455, 5.484848, -560.105, 1, 0.3764706, 0, 1,
6.585859, 5.484848, -560.2399, 1, 0.2745098, 0, 1,
6.626263, 5.484848, -560.3857, 1, 0.2745098, 0, 1,
6.666667, 5.484848, -560.5424, 1, 0.2745098, 0, 1,
6.707071, 5.484848, -560.7098, 1, 0.2745098, 0, 1,
6.747475, 5.484848, -560.8882, 1, 0.2745098, 0, 1,
6.787879, 5.484848, -561.0774, 1, 0.2745098, 0, 1,
6.828283, 5.484848, -561.2775, 1, 0.2745098, 0, 1,
6.868687, 5.484848, -561.4884, 1, 0.2745098, 0, 1,
6.909091, 5.484848, -561.7101, 1, 0.2745098, 0, 1,
6.949495, 5.484848, -561.9427, 1, 0.2745098, 0, 1,
6.989899, 5.484848, -562.1862, 1, 0.2745098, 0, 1,
7.030303, 5.484848, -562.4406, 1, 0.2745098, 0, 1,
7.070707, 5.484848, -562.7057, 1, 0.2745098, 0, 1,
7.111111, 5.484848, -562.9818, 1, 0.2745098, 0, 1,
7.151515, 5.484848, -563.2686, 1, 0.2745098, 0, 1,
7.191919, 5.484848, -563.5663, 1, 0.2745098, 0, 1,
7.232323, 5.484848, -563.8749, 1, 0.2745098, 0, 1,
7.272727, 5.484848, -564.1944, 1, 0.2745098, 0, 1,
7.313131, 5.484848, -564.5247, 1, 0.2745098, 0, 1,
7.353535, 5.484848, -564.8658, 1, 0.2745098, 0, 1,
7.393939, 5.484848, -565.2178, 1, 0.2745098, 0, 1,
7.434343, 5.484848, -565.5807, 1, 0.2745098, 0, 1,
7.474748, 5.484848, -565.9544, 1, 0.1686275, 0, 1,
7.515152, 5.484848, -566.3389, 1, 0.1686275, 0, 1,
7.555555, 5.484848, -566.7344, 1, 0.1686275, 0, 1,
7.59596, 5.484848, -567.1406, 1, 0.1686275, 0, 1,
7.636364, 5.484848, -567.5577, 1, 0.1686275, 0, 1,
7.676768, 5.484848, -567.9857, 1, 0.1686275, 0, 1,
7.717172, 5.484848, -568.4245, 1, 0.1686275, 0, 1,
7.757576, 5.484848, -568.8742, 1, 0.1686275, 0, 1,
7.79798, 5.484848, -569.3347, 1, 0.1686275, 0, 1,
7.838384, 5.484848, -569.8061, 1, 0.1686275, 0, 1,
7.878788, 5.484848, -570.2883, 1, 0.1686275, 0, 1,
7.919192, 5.484848, -570.7814, 1, 0.1686275, 0, 1,
7.959596, 5.484848, -571.2853, 1, 0.1686275, 0, 1,
8, 5.484848, -571.8002, 1, 0.06666667, 0, 1,
4, 5.535354, -574.4213, 1, 0.06666667, 0, 1,
4.040404, 5.535354, -573.8824, 1, 0.06666667, 0, 1,
4.080808, 5.535354, -573.3542, 1, 0.06666667, 0, 1,
4.121212, 5.535354, -572.8367, 1, 0.06666667, 0, 1,
4.161616, 5.535354, -572.3299, 1, 0.06666667, 0, 1,
4.20202, 5.535354, -571.8337, 1, 0.06666667, 0, 1,
4.242424, 5.535354, -571.3481, 1, 0.1686275, 0, 1,
4.282828, 5.535354, -570.8732, 1, 0.1686275, 0, 1,
4.323232, 5.535354, -570.409, 1, 0.1686275, 0, 1,
4.363636, 5.535354, -569.9554, 1, 0.1686275, 0, 1,
4.40404, 5.535354, -569.5125, 1, 0.1686275, 0, 1,
4.444445, 5.535354, -569.0802, 1, 0.1686275, 0, 1,
4.484848, 5.535354, -568.6586, 1, 0.1686275, 0, 1,
4.525252, 5.535354, -568.2476, 1, 0.1686275, 0, 1,
4.565657, 5.535354, -567.8473, 1, 0.1686275, 0, 1,
4.606061, 5.535354, -567.4576, 1, 0.1686275, 0, 1,
4.646465, 5.535354, -567.0787, 1, 0.1686275, 0, 1,
4.686869, 5.535354, -566.7103, 1, 0.1686275, 0, 1,
4.727273, 5.535354, -566.3527, 1, 0.1686275, 0, 1,
4.767677, 5.535354, -566.0056, 1, 0.1686275, 0, 1,
4.808081, 5.535354, -565.6693, 1, 0.2745098, 0, 1,
4.848485, 5.535354, -565.3435, 1, 0.2745098, 0, 1,
4.888889, 5.535354, -565.0284, 1, 0.2745098, 0, 1,
4.929293, 5.535354, -564.7241, 1, 0.2745098, 0, 1,
4.969697, 5.535354, -564.4303, 1, 0.2745098, 0, 1,
5.010101, 5.535354, -564.1472, 1, 0.2745098, 0, 1,
5.050505, 5.535354, -563.8748, 1, 0.2745098, 0, 1,
5.090909, 5.535354, -563.613, 1, 0.2745098, 0, 1,
5.131313, 5.535354, -563.3619, 1, 0.2745098, 0, 1,
5.171717, 5.535354, -563.1214, 1, 0.2745098, 0, 1,
5.212121, 5.535354, -562.8916, 1, 0.2745098, 0, 1,
5.252525, 5.535354, -562.6724, 1, 0.2745098, 0, 1,
5.292929, 5.535354, -562.4639, 1, 0.2745098, 0, 1,
5.333333, 5.535354, -562.2661, 1, 0.2745098, 0, 1,
5.373737, 5.535354, -562.0789, 1, 0.2745098, 0, 1,
5.414141, 5.535354, -561.9023, 1, 0.2745098, 0, 1,
5.454545, 5.535354, -561.7365, 1, 0.2745098, 0, 1,
5.494949, 5.535354, -561.5813, 1, 0.2745098, 0, 1,
5.535354, 5.535354, -561.4367, 1, 0.2745098, 0, 1,
5.575758, 5.535354, -561.3028, 1, 0.2745098, 0, 1,
5.616162, 5.535354, -561.1796, 1, 0.2745098, 0, 1,
5.656566, 5.535354, -561.067, 1, 0.2745098, 0, 1,
5.69697, 5.535354, -560.965, 1, 0.2745098, 0, 1,
5.737374, 5.535354, -560.8737, 1, 0.2745098, 0, 1,
5.777778, 5.535354, -560.7931, 1, 0.2745098, 0, 1,
5.818182, 5.535354, -560.7231, 1, 0.2745098, 0, 1,
5.858586, 5.535354, -560.6638, 1, 0.2745098, 0, 1,
5.89899, 5.535354, -560.6151, 1, 0.2745098, 0, 1,
5.939394, 5.535354, -560.5771, 1, 0.2745098, 0, 1,
5.979798, 5.535354, -560.5497, 1, 0.2745098, 0, 1,
6.020202, 5.535354, -560.5331, 1, 0.2745098, 0, 1,
6.060606, 5.535354, -560.527, 1, 0.2745098, 0, 1,
6.10101, 5.535354, -560.5317, 1, 0.2745098, 0, 1,
6.141414, 5.535354, -560.5469, 1, 0.2745098, 0, 1,
6.181818, 5.535354, -560.5729, 1, 0.2745098, 0, 1,
6.222222, 5.535354, -560.6094, 1, 0.2745098, 0, 1,
6.262626, 5.535354, -560.6567, 1, 0.2745098, 0, 1,
6.30303, 5.535354, -560.7145, 1, 0.2745098, 0, 1,
6.343434, 5.535354, -560.7831, 1, 0.2745098, 0, 1,
6.383838, 5.535354, -560.8623, 1, 0.2745098, 0, 1,
6.424242, 5.535354, -560.9522, 1, 0.2745098, 0, 1,
6.464646, 5.535354, -561.0527, 1, 0.2745098, 0, 1,
6.505051, 5.535354, -561.1639, 1, 0.2745098, 0, 1,
6.545455, 5.535354, -561.2857, 1, 0.2745098, 0, 1,
6.585859, 5.535354, -561.4182, 1, 0.2745098, 0, 1,
6.626263, 5.535354, -561.5613, 1, 0.2745098, 0, 1,
6.666667, 5.535354, -561.7151, 1, 0.2745098, 0, 1,
6.707071, 5.535354, -561.8796, 1, 0.2745098, 0, 1,
6.747475, 5.535354, -562.0547, 1, 0.2745098, 0, 1,
6.787879, 5.535354, -562.2405, 1, 0.2745098, 0, 1,
6.828283, 5.535354, -562.4369, 1, 0.2745098, 0, 1,
6.868687, 5.535354, -562.644, 1, 0.2745098, 0, 1,
6.909091, 5.535354, -562.8617, 1, 0.2745098, 0, 1,
6.949495, 5.535354, -563.0901, 1, 0.2745098, 0, 1,
6.989899, 5.535354, -563.3292, 1, 0.2745098, 0, 1,
7.030303, 5.535354, -563.5789, 1, 0.2745098, 0, 1,
7.070707, 5.535354, -563.8392, 1, 0.2745098, 0, 1,
7.111111, 5.535354, -564.1102, 1, 0.2745098, 0, 1,
7.151515, 5.535354, -564.3919, 1, 0.2745098, 0, 1,
7.191919, 5.535354, -564.6842, 1, 0.2745098, 0, 1,
7.232323, 5.535354, -564.9872, 1, 0.2745098, 0, 1,
7.272727, 5.535354, -565.3008, 1, 0.2745098, 0, 1,
7.313131, 5.535354, -565.6251, 1, 0.2745098, 0, 1,
7.353535, 5.535354, -565.9601, 1, 0.1686275, 0, 1,
7.393939, 5.535354, -566.3057, 1, 0.1686275, 0, 1,
7.434343, 5.535354, -566.6619, 1, 0.1686275, 0, 1,
7.474748, 5.535354, -567.0289, 1, 0.1686275, 0, 1,
7.515152, 5.535354, -567.4064, 1, 0.1686275, 0, 1,
7.555555, 5.535354, -567.7946, 1, 0.1686275, 0, 1,
7.59596, 5.535354, -568.1935, 1, 0.1686275, 0, 1,
7.636364, 5.535354, -568.6031, 1, 0.1686275, 0, 1,
7.676768, 5.535354, -569.0233, 1, 0.1686275, 0, 1,
7.717172, 5.535354, -569.4541, 1, 0.1686275, 0, 1,
7.757576, 5.535354, -569.8956, 1, 0.1686275, 0, 1,
7.79798, 5.535354, -570.3478, 1, 0.1686275, 0, 1,
7.838384, 5.535354, -570.8106, 1, 0.1686275, 0, 1,
7.878788, 5.535354, -571.2841, 1, 0.1686275, 0, 1,
7.919192, 5.535354, -571.7682, 1, 0.06666667, 0, 1,
7.959596, 5.535354, -572.263, 1, 0.06666667, 0, 1,
8, 5.535354, -572.7684, 1, 0.06666667, 0, 1,
4, 5.585859, -575.3665, 1, 0.06666667, 0, 1,
4.040404, 5.585859, -574.8374, 1, 0.06666667, 0, 1,
4.080808, 5.585859, -574.3187, 1, 0.06666667, 0, 1,
4.121212, 5.585859, -573.8105, 1, 0.06666667, 0, 1,
4.161616, 5.585859, -573.3127, 1, 0.06666667, 0, 1,
4.20202, 5.585859, -572.8255, 1, 0.06666667, 0, 1,
4.242424, 5.585859, -572.3487, 1, 0.06666667, 0, 1,
4.282828, 5.585859, -571.8823, 1, 0.06666667, 0, 1,
4.323232, 5.585859, -571.4265, 1, 0.1686275, 0, 1,
4.363636, 5.585859, -570.981, 1, 0.1686275, 0, 1,
4.40404, 5.585859, -570.5461, 1, 0.1686275, 0, 1,
4.444445, 5.585859, -570.1216, 1, 0.1686275, 0, 1,
4.484848, 5.585859, -569.7075, 1, 0.1686275, 0, 1,
4.525252, 5.585859, -569.304, 1, 0.1686275, 0, 1,
4.565657, 5.585859, -568.9109, 1, 0.1686275, 0, 1,
4.606061, 5.585859, -568.5283, 1, 0.1686275, 0, 1,
4.646465, 5.585859, -568.1561, 1, 0.1686275, 0, 1,
4.686869, 5.585859, -567.7944, 1, 0.1686275, 0, 1,
4.727273, 5.585859, -567.4431, 1, 0.1686275, 0, 1,
4.767677, 5.585859, -567.1024, 1, 0.1686275, 0, 1,
4.808081, 5.585859, -566.772, 1, 0.1686275, 0, 1,
4.848485, 5.585859, -566.4521, 1, 0.1686275, 0, 1,
4.888889, 5.585859, -566.1428, 1, 0.1686275, 0, 1,
4.929293, 5.585859, -565.8439, 1, 0.2745098, 0, 1,
4.969697, 5.585859, -565.5554, 1, 0.2745098, 0, 1,
5.010101, 5.585859, -565.2774, 1, 0.2745098, 0, 1,
5.050505, 5.585859, -565.0099, 1, 0.2745098, 0, 1,
5.090909, 5.585859, -564.7528, 1, 0.2745098, 0, 1,
5.131313, 5.585859, -564.5062, 1, 0.2745098, 0, 1,
5.171717, 5.585859, -564.2701, 1, 0.2745098, 0, 1,
5.212121, 5.585859, -564.0444, 1, 0.2745098, 0, 1,
5.252525, 5.585859, -563.8292, 1, 0.2745098, 0, 1,
5.292929, 5.585859, -563.6244, 1, 0.2745098, 0, 1,
5.333333, 5.585859, -563.4301, 1, 0.2745098, 0, 1,
5.373737, 5.585859, -563.2463, 1, 0.2745098, 0, 1,
5.414141, 5.585859, -563.0729, 1, 0.2745098, 0, 1,
5.454545, 5.585859, -562.9101, 1, 0.2745098, 0, 1,
5.494949, 5.585859, -562.7576, 1, 0.2745098, 0, 1,
5.535354, 5.585859, -562.6157, 1, 0.2745098, 0, 1,
5.575758, 5.585859, -562.4842, 1, 0.2745098, 0, 1,
5.616162, 5.585859, -562.3632, 1, 0.2745098, 0, 1,
5.656566, 5.585859, -562.2526, 1, 0.2745098, 0, 1,
5.69697, 5.585859, -562.1525, 1, 0.2745098, 0, 1,
5.737374, 5.585859, -562.0628, 1, 0.2745098, 0, 1,
5.777778, 5.585859, -561.9836, 1, 0.2745098, 0, 1,
5.818182, 5.585859, -561.9149, 1, 0.2745098, 0, 1,
5.858586, 5.585859, -561.8567, 1, 0.2745098, 0, 1,
5.89899, 5.585859, -561.8089, 1, 0.2745098, 0, 1,
5.939394, 5.585859, -561.7715, 1, 0.2745098, 0, 1,
5.979798, 5.585859, -561.7447, 1, 0.2745098, 0, 1,
6.020202, 5.585859, -561.7283, 1, 0.2745098, 0, 1,
6.060606, 5.585859, -561.7224, 1, 0.2745098, 0, 1,
6.10101, 5.585859, -561.7269, 1, 0.2745098, 0, 1,
6.141414, 5.585859, -561.7419, 1, 0.2745098, 0, 1,
6.181818, 5.585859, -561.7674, 1, 0.2745098, 0, 1,
6.222222, 5.585859, -561.8033, 1, 0.2745098, 0, 1,
6.262626, 5.585859, -561.8497, 1, 0.2745098, 0, 1,
6.30303, 5.585859, -561.9066, 1, 0.2745098, 0, 1,
6.343434, 5.585859, -561.9739, 1, 0.2745098, 0, 1,
6.383838, 5.585859, -562.0516, 1, 0.2745098, 0, 1,
6.424242, 5.585859, -562.1399, 1, 0.2745098, 0, 1,
6.464646, 5.585859, -562.2386, 1, 0.2745098, 0, 1,
6.505051, 5.585859, -562.3478, 1, 0.2745098, 0, 1,
6.545455, 5.585859, -562.4674, 1, 0.2745098, 0, 1,
6.585859, 5.585859, -562.5975, 1, 0.2745098, 0, 1,
6.626263, 5.585859, -562.7381, 1, 0.2745098, 0, 1,
6.666667, 5.585859, -562.8891, 1, 0.2745098, 0, 1,
6.707071, 5.585859, -563.0506, 1, 0.2745098, 0, 1,
6.747475, 5.585859, -563.2225, 1, 0.2745098, 0, 1,
6.787879, 5.585859, -563.405, 1, 0.2745098, 0, 1,
6.828283, 5.585859, -563.5978, 1, 0.2745098, 0, 1,
6.868687, 5.585859, -563.8012, 1, 0.2745098, 0, 1,
6.909091, 5.585859, -564.015, 1, 0.2745098, 0, 1,
6.949495, 5.585859, -564.2393, 1, 0.2745098, 0, 1,
6.989899, 5.585859, -564.4741, 1, 0.2745098, 0, 1,
7.030303, 5.585859, -564.7192, 1, 0.2745098, 0, 1,
7.070707, 5.585859, -564.9749, 1, 0.2745098, 0, 1,
7.111111, 5.585859, -565.241, 1, 0.2745098, 0, 1,
7.151515, 5.585859, -565.5176, 1, 0.2745098, 0, 1,
7.191919, 5.585859, -565.8047, 1, 0.2745098, 0, 1,
7.232323, 5.585859, -566.1022, 1, 0.1686275, 0, 1,
7.272727, 5.585859, -566.4102, 1, 0.1686275, 0, 1,
7.313131, 5.585859, -566.7287, 1, 0.1686275, 0, 1,
7.353535, 5.585859, -567.0576, 1, 0.1686275, 0, 1,
7.393939, 5.585859, -567.397, 1, 0.1686275, 0, 1,
7.434343, 5.585859, -567.7468, 1, 0.1686275, 0, 1,
7.474748, 5.585859, -568.1072, 1, 0.1686275, 0, 1,
7.515152, 5.585859, -568.4779, 1, 0.1686275, 0, 1,
7.555555, 5.585859, -568.8592, 1, 0.1686275, 0, 1,
7.59596, 5.585859, -569.2509, 1, 0.1686275, 0, 1,
7.636364, 5.585859, -569.653, 1, 0.1686275, 0, 1,
7.676768, 5.585859, -570.0657, 1, 0.1686275, 0, 1,
7.717172, 5.585859, -570.4888, 1, 0.1686275, 0, 1,
7.757576, 5.585859, -570.9223, 1, 0.1686275, 0, 1,
7.79798, 5.585859, -571.3663, 1, 0.1686275, 0, 1,
7.838384, 5.585859, -571.8208, 1, 0.06666667, 0, 1,
7.878788, 5.585859, -572.2858, 1, 0.06666667, 0, 1,
7.919192, 5.585859, -572.7612, 1, 0.06666667, 0, 1,
7.959596, 5.585859, -573.2471, 1, 0.06666667, 0, 1,
8, 5.585859, -573.7434, 1, 0.06666667, 0, 1,
4, 5.636364, -576.3187, 1, 0.06666667, 0, 1,
4.040404, 5.636364, -575.799, 1, 0.06666667, 0, 1,
4.080808, 5.636364, -575.2896, 1, 0.06666667, 0, 1,
4.121212, 5.636364, -574.7905, 1, 0.06666667, 0, 1,
4.161616, 5.636364, -574.3016, 1, 0.06666667, 0, 1,
4.20202, 5.636364, -573.823, 1, 0.06666667, 0, 1,
4.242424, 5.636364, -573.3547, 1, 0.06666667, 0, 1,
4.282828, 5.636364, -572.8967, 1, 0.06666667, 0, 1,
4.323232, 5.636364, -572.4489, 1, 0.06666667, 0, 1,
4.363636, 5.636364, -572.0115, 1, 0.06666667, 0, 1,
4.40404, 5.636364, -571.5842, 1, 0.1686275, 0, 1,
4.444445, 5.636364, -571.1674, 1, 0.1686275, 0, 1,
4.484848, 5.636364, -570.7607, 1, 0.1686275, 0, 1,
4.525252, 5.636364, -570.3643, 1, 0.1686275, 0, 1,
4.565657, 5.636364, -569.9783, 1, 0.1686275, 0, 1,
4.606061, 5.636364, -569.6024, 1, 0.1686275, 0, 1,
4.646465, 5.636364, -569.2369, 1, 0.1686275, 0, 1,
4.686869, 5.636364, -568.8817, 1, 0.1686275, 0, 1,
4.727273, 5.636364, -568.5367, 1, 0.1686275, 0, 1,
4.767677, 5.636364, -568.202, 1, 0.1686275, 0, 1,
4.808081, 5.636364, -567.8776, 1, 0.1686275, 0, 1,
4.848485, 5.636364, -567.5634, 1, 0.1686275, 0, 1,
4.888889, 5.636364, -567.2595, 1, 0.1686275, 0, 1,
4.929293, 5.636364, -566.9659, 1, 0.1686275, 0, 1,
4.969697, 5.636364, -566.6826, 1, 0.1686275, 0, 1,
5.010101, 5.636364, -566.4096, 1, 0.1686275, 0, 1,
5.050505, 5.636364, -566.1469, 1, 0.1686275, 0, 1,
5.090909, 5.636364, -565.8943, 1, 0.2745098, 0, 1,
5.131313, 5.636364, -565.6522, 1, 0.2745098, 0, 1,
5.171717, 5.636364, -565.4202, 1, 0.2745098, 0, 1,
5.212121, 5.636364, -565.1985, 1, 0.2745098, 0, 1,
5.252525, 5.636364, -564.9872, 1, 0.2745098, 0, 1,
5.292929, 5.636364, -564.7861, 1, 0.2745098, 0, 1,
5.333333, 5.636364, -564.5953, 1, 0.2745098, 0, 1,
5.373737, 5.636364, -564.4147, 1, 0.2745098, 0, 1,
5.414141, 5.636364, -564.2445, 1, 0.2745098, 0, 1,
5.454545, 5.636364, -564.0845, 1, 0.2745098, 0, 1,
5.494949, 5.636364, -563.9348, 1, 0.2745098, 0, 1,
5.535354, 5.636364, -563.7953, 1, 0.2745098, 0, 1,
5.575758, 5.636364, -563.6662, 1, 0.2745098, 0, 1,
5.616162, 5.636364, -563.5473, 1, 0.2745098, 0, 1,
5.656566, 5.636364, -563.4387, 1, 0.2745098, 0, 1,
5.69697, 5.636364, -563.3404, 1, 0.2745098, 0, 1,
5.737374, 5.636364, -563.2524, 1, 0.2745098, 0, 1,
5.777778, 5.636364, -563.1746, 1, 0.2745098, 0, 1,
5.818182, 5.636364, -563.1071, 1, 0.2745098, 0, 1,
5.858586, 5.636364, -563.0499, 1, 0.2745098, 0, 1,
5.89899, 5.636364, -563.0029, 1, 0.2745098, 0, 1,
5.939394, 5.636364, -562.9663, 1, 0.2745098, 0, 1,
5.979798, 5.636364, -562.9399, 1, 0.2745098, 0, 1,
6.020202, 5.636364, -562.9238, 1, 0.2745098, 0, 1,
6.060606, 5.636364, -562.918, 1, 0.2745098, 0, 1,
6.10101, 5.636364, -562.9224, 1, 0.2745098, 0, 1,
6.141414, 5.636364, -562.9372, 1, 0.2745098, 0, 1,
6.181818, 5.636364, -562.9622, 1, 0.2745098, 0, 1,
6.222222, 5.636364, -562.9975, 1, 0.2745098, 0, 1,
6.262626, 5.636364, -563.043, 1, 0.2745098, 0, 1,
6.30303, 5.636364, -563.0989, 1, 0.2745098, 0, 1,
6.343434, 5.636364, -563.165, 1, 0.2745098, 0, 1,
6.383838, 5.636364, -563.2414, 1, 0.2745098, 0, 1,
6.424242, 5.636364, -563.3281, 1, 0.2745098, 0, 1,
6.464646, 5.636364, -563.425, 1, 0.2745098, 0, 1,
6.505051, 5.636364, -563.5322, 1, 0.2745098, 0, 1,
6.545455, 5.636364, -563.6497, 1, 0.2745098, 0, 1,
6.585859, 5.636364, -563.7775, 1, 0.2745098, 0, 1,
6.626263, 5.636364, -563.9156, 1, 0.2745098, 0, 1,
6.666667, 5.636364, -564.0639, 1, 0.2745098, 0, 1,
6.707071, 5.636364, -564.2225, 1, 0.2745098, 0, 1,
6.747475, 5.636364, -564.3914, 1, 0.2745098, 0, 1,
6.787879, 5.636364, -564.5706, 1, 0.2745098, 0, 1,
6.828283, 5.636364, -564.76, 1, 0.2745098, 0, 1,
6.868687, 5.636364, -564.9597, 1, 0.2745098, 0, 1,
6.909091, 5.636364, -565.1697, 1, 0.2745098, 0, 1,
6.949495, 5.636364, -565.39, 1, 0.2745098, 0, 1,
6.989899, 5.636364, -565.6205, 1, 0.2745098, 0, 1,
7.030303, 5.636364, -565.8614, 1, 0.2745098, 0, 1,
7.070707, 5.636364, -566.1125, 1, 0.1686275, 0, 1,
7.111111, 5.636364, -566.3739, 1, 0.1686275, 0, 1,
7.151515, 5.636364, -566.6456, 1, 0.1686275, 0, 1,
7.191919, 5.636364, -566.9275, 1, 0.1686275, 0, 1,
7.232323, 5.636364, -567.2197, 1, 0.1686275, 0, 1,
7.272727, 5.636364, -567.5222, 1, 0.1686275, 0, 1,
7.313131, 5.636364, -567.835, 1, 0.1686275, 0, 1,
7.353535, 5.636364, -568.158, 1, 0.1686275, 0, 1,
7.393939, 5.636364, -568.4914, 1, 0.1686275, 0, 1,
7.434343, 5.636364, -568.835, 1, 0.1686275, 0, 1,
7.474748, 5.636364, -569.1888, 1, 0.1686275, 0, 1,
7.515152, 5.636364, -569.553, 1, 0.1686275, 0, 1,
7.555555, 5.636364, -569.9274, 1, 0.1686275, 0, 1,
7.59596, 5.636364, -570.3122, 1, 0.1686275, 0, 1,
7.636364, 5.636364, -570.7072, 1, 0.1686275, 0, 1,
7.676768, 5.636364, -571.1124, 1, 0.1686275, 0, 1,
7.717172, 5.636364, -571.528, 1, 0.1686275, 0, 1,
7.757576, 5.636364, -571.9538, 1, 0.06666667, 0, 1,
7.79798, 5.636364, -572.3899, 1, 0.06666667, 0, 1,
7.838384, 5.636364, -572.8363, 1, 0.06666667, 0, 1,
7.878788, 5.636364, -573.2929, 1, 0.06666667, 0, 1,
7.919192, 5.636364, -573.7599, 1, 0.06666667, 0, 1,
7.959596, 5.636364, -574.2371, 1, 0.06666667, 0, 1,
8, 5.636364, -574.7245, 1, 0.06666667, 0, 1,
4, 5.686869, -577.2773, 1, 0.06666667, 0, 1,
4.040404, 5.686869, -576.7668, 1, 0.06666667, 0, 1,
4.080808, 5.686869, -576.2664, 1, 0.06666667, 0, 1,
4.121212, 5.686869, -575.7761, 1, 0.06666667, 0, 1,
4.161616, 5.686869, -575.2959, 1, 0.06666667, 0, 1,
4.20202, 5.686869, -574.8258, 1, 0.06666667, 0, 1,
4.242424, 5.686869, -574.3657, 1, 0.06666667, 0, 1,
4.282828, 5.686869, -573.9158, 1, 0.06666667, 0, 1,
4.323232, 5.686869, -573.476, 1, 0.06666667, 0, 1,
4.363636, 5.686869, -573.0463, 1, 0.06666667, 0, 1,
4.40404, 5.686869, -572.6266, 1, 0.06666667, 0, 1,
4.444445, 5.686869, -572.217, 1, 0.06666667, 0, 1,
4.484848, 5.686869, -571.8176, 1, 0.06666667, 0, 1,
4.525252, 5.686869, -571.4282, 1, 0.1686275, 0, 1,
4.565657, 5.686869, -571.049, 1, 0.1686275, 0, 1,
4.606061, 5.686869, -570.6798, 1, 0.1686275, 0, 1,
4.646465, 5.686869, -570.3207, 1, 0.1686275, 0, 1,
4.686869, 5.686869, -569.9718, 1, 0.1686275, 0, 1,
4.727273, 5.686869, -569.6329, 1, 0.1686275, 0, 1,
4.767677, 5.686869, -569.3041, 1, 0.1686275, 0, 1,
4.808081, 5.686869, -568.9854, 1, 0.1686275, 0, 1,
4.848485, 5.686869, -568.6768, 1, 0.1686275, 0, 1,
4.888889, 5.686869, -568.3784, 1, 0.1686275, 0, 1,
4.929293, 5.686869, -568.09, 1, 0.1686275, 0, 1,
4.969697, 5.686869, -567.8116, 1, 0.1686275, 0, 1,
5.010101, 5.686869, -567.5435, 1, 0.1686275, 0, 1,
5.050505, 5.686869, -567.2853, 1, 0.1686275, 0, 1,
5.090909, 5.686869, -567.0373, 1, 0.1686275, 0, 1,
5.131313, 5.686869, -566.7994, 1, 0.1686275, 0, 1,
5.171717, 5.686869, -566.5716, 1, 0.1686275, 0, 1,
5.212121, 5.686869, -566.3538, 1, 0.1686275, 0, 1,
5.252525, 5.686869, -566.1462, 1, 0.1686275, 0, 1,
5.292929, 5.686869, -565.9487, 1, 0.1686275, 0, 1,
5.333333, 5.686869, -565.7612, 1, 0.2745098, 0, 1,
5.373737, 5.686869, -565.5839, 1, 0.2745098, 0, 1,
5.414141, 5.686869, -565.4166, 1, 0.2745098, 0, 1,
5.454545, 5.686869, -565.2595, 1, 0.2745098, 0, 1,
5.494949, 5.686869, -565.1124, 1, 0.2745098, 0, 1,
5.535354, 5.686869, -564.9755, 1, 0.2745098, 0, 1,
5.575758, 5.686869, -564.8486, 1, 0.2745098, 0, 1,
5.616162, 5.686869, -564.7318, 1, 0.2745098, 0, 1,
5.656566, 5.686869, -564.6251, 1, 0.2745098, 0, 1,
5.69697, 5.686869, -564.5285, 1, 0.2745098, 0, 1,
5.737374, 5.686869, -564.442, 1, 0.2745098, 0, 1,
5.777778, 5.686869, -564.3657, 1, 0.2745098, 0, 1,
5.818182, 5.686869, -564.2994, 1, 0.2745098, 0, 1,
5.858586, 5.686869, -564.2432, 1, 0.2745098, 0, 1,
5.89899, 5.686869, -564.1971, 1, 0.2745098, 0, 1,
5.939394, 5.686869, -564.1611, 1, 0.2745098, 0, 1,
5.979798, 5.686869, -564.1351, 1, 0.2745098, 0, 1,
6.020202, 5.686869, -564.1193, 1, 0.2745098, 0, 1,
6.060606, 5.686869, -564.1136, 1, 0.2745098, 0, 1,
6.10101, 5.686869, -564.118, 1, 0.2745098, 0, 1,
6.141414, 5.686869, -564.1324, 1, 0.2745098, 0, 1,
6.181818, 5.686869, -564.157, 1, 0.2745098, 0, 1,
6.222222, 5.686869, -564.1917, 1, 0.2745098, 0, 1,
6.262626, 5.686869, -564.2364, 1, 0.2745098, 0, 1,
6.30303, 5.686869, -564.2913, 1, 0.2745098, 0, 1,
6.343434, 5.686869, -564.3562, 1, 0.2745098, 0, 1,
6.383838, 5.686869, -564.4313, 1, 0.2745098, 0, 1,
6.424242, 5.686869, -564.5164, 1, 0.2745098, 0, 1,
6.464646, 5.686869, -564.6116, 1, 0.2745098, 0, 1,
6.505051, 5.686869, -564.717, 1, 0.2745098, 0, 1,
6.545455, 5.686869, -564.8324, 1, 0.2745098, 0, 1,
6.585859, 5.686869, -564.9579, 1, 0.2745098, 0, 1,
6.626263, 5.686869, -565.0935, 1, 0.2745098, 0, 1,
6.666667, 5.686869, -565.2393, 1, 0.2745098, 0, 1,
6.707071, 5.686869, -565.395, 1, 0.2745098, 0, 1,
6.747475, 5.686869, -565.561, 1, 0.2745098, 0, 1,
6.787879, 5.686869, -565.7369, 1, 0.2745098, 0, 1,
6.828283, 5.686869, -565.923, 1, 0.1686275, 0, 1,
6.868687, 5.686869, -566.1192, 1, 0.1686275, 0, 1,
6.909091, 5.686869, -566.3255, 1, 0.1686275, 0, 1,
6.949495, 5.686869, -566.5419, 1, 0.1686275, 0, 1,
6.989899, 5.686869, -566.7684, 1, 0.1686275, 0, 1,
7.030303, 5.686869, -567.0049, 1, 0.1686275, 0, 1,
7.070707, 5.686869, -567.2516, 1, 0.1686275, 0, 1,
7.111111, 5.686869, -567.5084, 1, 0.1686275, 0, 1,
7.151515, 5.686869, -567.7753, 1, 0.1686275, 0, 1,
7.191919, 5.686869, -568.0522, 1, 0.1686275, 0, 1,
7.232323, 5.686869, -568.3392, 1, 0.1686275, 0, 1,
7.272727, 5.686869, -568.6364, 1, 0.1686275, 0, 1,
7.313131, 5.686869, -568.9437, 1, 0.1686275, 0, 1,
7.353535, 5.686869, -569.261, 1, 0.1686275, 0, 1,
7.393939, 5.686869, -569.5884, 1, 0.1686275, 0, 1,
7.434343, 5.686869, -569.926, 1, 0.1686275, 0, 1,
7.474748, 5.686869, -570.2736, 1, 0.1686275, 0, 1,
7.515152, 5.686869, -570.6313, 1, 0.1686275, 0, 1,
7.555555, 5.686869, -570.9991, 1, 0.1686275, 0, 1,
7.59596, 5.686869, -571.377, 1, 0.1686275, 0, 1,
7.636364, 5.686869, -571.765, 1, 0.06666667, 0, 1,
7.676768, 5.686869, -572.1631, 1, 0.06666667, 0, 1,
7.717172, 5.686869, -572.5714, 1, 0.06666667, 0, 1,
7.757576, 5.686869, -572.9896, 1, 0.06666667, 0, 1,
7.79798, 5.686869, -573.418, 1, 0.06666667, 0, 1,
7.838384, 5.686869, -573.8565, 1, 0.06666667, 0, 1,
7.878788, 5.686869, -574.3051, 1, 0.06666667, 0, 1,
7.919192, 5.686869, -574.7637, 1, 0.06666667, 0, 1,
7.959596, 5.686869, -575.2325, 1, 0.06666667, 0, 1,
8, 5.686869, -575.7114, 1, 0.06666667, 0, 1,
4, 5.737374, -578.2419, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.737374, -577.7404, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.737374, -577.2487, 1, 0.06666667, 0, 1,
4.121212, 5.737374, -576.767, 1, 0.06666667, 0, 1,
4.161616, 5.737374, -576.2952, 1, 0.06666667, 0, 1,
4.20202, 5.737374, -575.8333, 1, 0.06666667, 0, 1,
4.242424, 5.737374, -575.3813, 1, 0.06666667, 0, 1,
4.282828, 5.737374, -574.9393, 1, 0.06666667, 0, 1,
4.323232, 5.737374, -574.5072, 1, 0.06666667, 0, 1,
4.363636, 5.737374, -574.085, 1, 0.06666667, 0, 1,
4.40404, 5.737374, -573.6727, 1, 0.06666667, 0, 1,
4.444445, 5.737374, -573.2703, 1, 0.06666667, 0, 1,
4.484848, 5.737374, -572.8779, 1, 0.06666667, 0, 1,
4.525252, 5.737374, -572.4954, 1, 0.06666667, 0, 1,
4.565657, 5.737374, -572.1227, 1, 0.06666667, 0, 1,
4.606061, 5.737374, -571.76, 1, 0.06666667, 0, 1,
4.646465, 5.737374, -571.4072, 1, 0.1686275, 0, 1,
4.686869, 5.737374, -571.0644, 1, 0.1686275, 0, 1,
4.727273, 5.737374, -570.7314, 1, 0.1686275, 0, 1,
4.767677, 5.737374, -570.4084, 1, 0.1686275, 0, 1,
4.808081, 5.737374, -570.0953, 1, 0.1686275, 0, 1,
4.848485, 5.737374, -569.7922, 1, 0.1686275, 0, 1,
4.888889, 5.737374, -569.4989, 1, 0.1686275, 0, 1,
4.929293, 5.737374, -569.2156, 1, 0.1686275, 0, 1,
4.969697, 5.737374, -568.9421, 1, 0.1686275, 0, 1,
5.010101, 5.737374, -568.6786, 1, 0.1686275, 0, 1,
5.050505, 5.737374, -568.425, 1, 0.1686275, 0, 1,
5.090909, 5.737374, -568.1813, 1, 0.1686275, 0, 1,
5.131313, 5.737374, -567.9476, 1, 0.1686275, 0, 1,
5.171717, 5.737374, -567.7238, 1, 0.1686275, 0, 1,
5.212121, 5.737374, -567.5099, 1, 0.1686275, 0, 1,
5.252525, 5.737374, -567.3058, 1, 0.1686275, 0, 1,
5.292929, 5.737374, -567.1118, 1, 0.1686275, 0, 1,
5.333333, 5.737374, -566.9276, 1, 0.1686275, 0, 1,
5.373737, 5.737374, -566.7534, 1, 0.1686275, 0, 1,
5.414141, 5.737374, -566.5891, 1, 0.1686275, 0, 1,
5.454545, 5.737374, -566.4347, 1, 0.1686275, 0, 1,
5.494949, 5.737374, -566.2902, 1, 0.1686275, 0, 1,
5.535354, 5.737374, -566.1556, 1, 0.1686275, 0, 1,
5.575758, 5.737374, -566.031, 1, 0.1686275, 0, 1,
5.616162, 5.737374, -565.9163, 1, 0.1686275, 0, 1,
5.656566, 5.737374, -565.8115, 1, 0.2745098, 0, 1,
5.69697, 5.737374, -565.7166, 1, 0.2745098, 0, 1,
5.737374, 5.737374, -565.6316, 1, 0.2745098, 0, 1,
5.777778, 5.737374, -565.5565, 1, 0.2745098, 0, 1,
5.818182, 5.737374, -565.4914, 1, 0.2745098, 0, 1,
5.858586, 5.737374, -565.4362, 1, 0.2745098, 0, 1,
5.89899, 5.737374, -565.3909, 1, 0.2745098, 0, 1,
5.939394, 5.737374, -565.3555, 1, 0.2745098, 0, 1,
5.979798, 5.737374, -565.33, 1, 0.2745098, 0, 1,
6.020202, 5.737374, -565.3145, 1, 0.2745098, 0, 1,
6.060606, 5.737374, -565.3089, 1, 0.2745098, 0, 1,
6.10101, 5.737374, -565.3132, 1, 0.2745098, 0, 1,
6.141414, 5.737374, -565.3274, 1, 0.2745098, 0, 1,
6.181818, 5.737374, -565.3515, 1, 0.2745098, 0, 1,
6.222222, 5.737374, -565.3856, 1, 0.2745098, 0, 1,
6.262626, 5.737374, -565.4296, 1, 0.2745098, 0, 1,
6.30303, 5.737374, -565.4835, 1, 0.2745098, 0, 1,
6.343434, 5.737374, -565.5472, 1, 0.2745098, 0, 1,
6.383838, 5.737374, -565.621, 1, 0.2745098, 0, 1,
6.424242, 5.737374, -565.7047, 1, 0.2745098, 0, 1,
6.464646, 5.737374, -565.7982, 1, 0.2745098, 0, 1,
6.505051, 5.737374, -565.9017, 1, 0.2745098, 0, 1,
6.545455, 5.737374, -566.0151, 1, 0.1686275, 0, 1,
6.585859, 5.737374, -566.1384, 1, 0.1686275, 0, 1,
6.626263, 5.737374, -566.2717, 1, 0.1686275, 0, 1,
6.666667, 5.737374, -566.4148, 1, 0.1686275, 0, 1,
6.707071, 5.737374, -566.5679, 1, 0.1686275, 0, 1,
6.747475, 5.737374, -566.7309, 1, 0.1686275, 0, 1,
6.787879, 5.737374, -566.9038, 1, 0.1686275, 0, 1,
6.828283, 5.737374, -567.0866, 1, 0.1686275, 0, 1,
6.868687, 5.737374, -567.2794, 1, 0.1686275, 0, 1,
6.909091, 5.737374, -567.4821, 1, 0.1686275, 0, 1,
6.949495, 5.737374, -567.6946, 1, 0.1686275, 0, 1,
6.989899, 5.737374, -567.9171, 1, 0.1686275, 0, 1,
7.030303, 5.737374, -568.1495, 1, 0.1686275, 0, 1,
7.070707, 5.737374, -568.3919, 1, 0.1686275, 0, 1,
7.111111, 5.737374, -568.6442, 1, 0.1686275, 0, 1,
7.151515, 5.737374, -568.9064, 1, 0.1686275, 0, 1,
7.191919, 5.737374, -569.1785, 1, 0.1686275, 0, 1,
7.232323, 5.737374, -569.4604, 1, 0.1686275, 0, 1,
7.272727, 5.737374, -569.7524, 1, 0.1686275, 0, 1,
7.313131, 5.737374, -570.0543, 1, 0.1686275, 0, 1,
7.353535, 5.737374, -570.366, 1, 0.1686275, 0, 1,
7.393939, 5.737374, -570.6877, 1, 0.1686275, 0, 1,
7.434343, 5.737374, -571.0193, 1, 0.1686275, 0, 1,
7.474748, 5.737374, -571.3609, 1, 0.1686275, 0, 1,
7.515152, 5.737374, -571.7123, 1, 0.06666667, 0, 1,
7.555555, 5.737374, -572.0737, 1, 0.06666667, 0, 1,
7.59596, 5.737374, -572.445, 1, 0.06666667, 0, 1,
7.636364, 5.737374, -572.8262, 1, 0.06666667, 0, 1,
7.676768, 5.737374, -573.2173, 1, 0.06666667, 0, 1,
7.717172, 5.737374, -573.6183, 1, 0.06666667, 0, 1,
7.757576, 5.737374, -574.0294, 1, 0.06666667, 0, 1,
7.79798, 5.737374, -574.4502, 1, 0.06666667, 0, 1,
7.838384, 5.737374, -574.881, 1, 0.06666667, 0, 1,
7.878788, 5.737374, -575.3217, 1, 0.06666667, 0, 1,
7.919192, 5.737374, -575.7723, 1, 0.06666667, 0, 1,
7.959596, 5.737374, -576.2329, 1, 0.06666667, 0, 1,
8, 5.737374, -576.7034, 1, 0.06666667, 0, 1,
4, 5.787879, -579.2119, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.787879, -578.7191, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.787879, -578.236, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.787879, -577.7626, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.787879, -577.299, 1, 0.06666667, 0, 1,
4.20202, 5.787879, -576.8452, 1, 0.06666667, 0, 1,
4.242424, 5.787879, -576.4011, 1, 0.06666667, 0, 1,
4.282828, 5.787879, -575.9667, 1, 0.06666667, 0, 1,
4.323232, 5.787879, -575.5421, 1, 0.06666667, 0, 1,
4.363636, 5.787879, -575.1272, 1, 0.06666667, 0, 1,
4.40404, 5.787879, -574.7221, 1, 0.06666667, 0, 1,
4.444445, 5.787879, -574.3267, 1, 0.06666667, 0, 1,
4.484848, 5.787879, -573.9411, 1, 0.06666667, 0, 1,
4.525252, 5.787879, -573.5652, 1, 0.06666667, 0, 1,
4.565657, 5.787879, -573.199, 1, 0.06666667, 0, 1,
4.606061, 5.787879, -572.8427, 1, 0.06666667, 0, 1,
4.646465, 5.787879, -572.496, 1, 0.06666667, 0, 1,
4.686869, 5.787879, -572.1591, 1, 0.06666667, 0, 1,
4.727273, 5.787879, -571.832, 1, 0.06666667, 0, 1,
4.767677, 5.787879, -571.5146, 1, 0.1686275, 0, 1,
4.808081, 5.787879, -571.2069, 1, 0.1686275, 0, 1,
4.848485, 5.787879, -570.909, 1, 0.1686275, 0, 1,
4.888889, 5.787879, -570.6208, 1, 0.1686275, 0, 1,
4.929293, 5.787879, -570.3424, 1, 0.1686275, 0, 1,
4.969697, 5.787879, -570.0737, 1, 0.1686275, 0, 1,
5.010101, 5.787879, -569.8148, 1, 0.1686275, 0, 1,
5.050505, 5.787879, -569.5656, 1, 0.1686275, 0, 1,
5.090909, 5.787879, -569.3262, 1, 0.1686275, 0, 1,
5.131313, 5.787879, -569.0965, 1, 0.1686275, 0, 1,
5.171717, 5.787879, -568.8765, 1, 0.1686275, 0, 1,
5.212121, 5.787879, -568.6663, 1, 0.1686275, 0, 1,
5.252525, 5.787879, -568.4659, 1, 0.1686275, 0, 1,
5.292929, 5.787879, -568.2752, 1, 0.1686275, 0, 1,
5.333333, 5.787879, -568.0942, 1, 0.1686275, 0, 1,
5.373737, 5.787879, -567.923, 1, 0.1686275, 0, 1,
5.414141, 5.787879, -567.7615, 1, 0.1686275, 0, 1,
5.454545, 5.787879, -567.6099, 1, 0.1686275, 0, 1,
5.494949, 5.787879, -567.4679, 1, 0.1686275, 0, 1,
5.535354, 5.787879, -567.3356, 1, 0.1686275, 0, 1,
5.575758, 5.787879, -567.2131, 1, 0.1686275, 0, 1,
5.616162, 5.787879, -567.1004, 1, 0.1686275, 0, 1,
5.656566, 5.787879, -566.9974, 1, 0.1686275, 0, 1,
5.69697, 5.787879, -566.9042, 1, 0.1686275, 0, 1,
5.737374, 5.787879, -566.8207, 1, 0.1686275, 0, 1,
5.777778, 5.787879, -566.7469, 1, 0.1686275, 0, 1,
5.818182, 5.787879, -566.683, 1, 0.1686275, 0, 1,
5.858586, 5.787879, -566.6287, 1, 0.1686275, 0, 1,
5.89899, 5.787879, -566.5842, 1, 0.1686275, 0, 1,
5.939394, 5.787879, -566.5494, 1, 0.1686275, 0, 1,
5.979798, 5.787879, -566.5244, 1, 0.1686275, 0, 1,
6.020202, 5.787879, -566.5092, 1, 0.1686275, 0, 1,
6.060606, 5.787879, -566.5036, 1, 0.1686275, 0, 1,
6.10101, 5.787879, -566.5078, 1, 0.1686275, 0, 1,
6.141414, 5.787879, -566.5218, 1, 0.1686275, 0, 1,
6.181818, 5.787879, -566.5455, 1, 0.1686275, 0, 1,
6.222222, 5.787879, -566.579, 1, 0.1686275, 0, 1,
6.262626, 5.787879, -566.6222, 1, 0.1686275, 0, 1,
6.30303, 5.787879, -566.6752, 1, 0.1686275, 0, 1,
6.343434, 5.787879, -566.7379, 1, 0.1686275, 0, 1,
6.383838, 5.787879, -566.8103, 1, 0.1686275, 0, 1,
6.424242, 5.787879, -566.8925, 1, 0.1686275, 0, 1,
6.464646, 5.787879, -566.9844, 1, 0.1686275, 0, 1,
6.505051, 5.787879, -567.0861, 1, 0.1686275, 0, 1,
6.545455, 5.787879, -567.1976, 1, 0.1686275, 0, 1,
6.585859, 5.787879, -567.3187, 1, 0.1686275, 0, 1,
6.626263, 5.787879, -567.4496, 1, 0.1686275, 0, 1,
6.666667, 5.787879, -567.5903, 1, 0.1686275, 0, 1,
6.707071, 5.787879, -567.7407, 1, 0.1686275, 0, 1,
6.747475, 5.787879, -567.9009, 1, 0.1686275, 0, 1,
6.787879, 5.787879, -568.0708, 1, 0.1686275, 0, 1,
6.828283, 5.787879, -568.2505, 1, 0.1686275, 0, 1,
6.868687, 5.787879, -568.4399, 1, 0.1686275, 0, 1,
6.909091, 5.787879, -568.639, 1, 0.1686275, 0, 1,
6.949495, 5.787879, -568.8479, 1, 0.1686275, 0, 1,
6.989899, 5.787879, -569.0665, 1, 0.1686275, 0, 1,
7.030303, 5.787879, -569.2949, 1, 0.1686275, 0, 1,
7.070707, 5.787879, -569.5331, 1, 0.1686275, 0, 1,
7.111111, 5.787879, -569.7809, 1, 0.1686275, 0, 1,
7.151515, 5.787879, -570.0386, 1, 0.1686275, 0, 1,
7.191919, 5.787879, -570.306, 1, 0.1686275, 0, 1,
7.232323, 5.787879, -570.5831, 1, 0.1686275, 0, 1,
7.272727, 5.787879, -570.8699, 1, 0.1686275, 0, 1,
7.313131, 5.787879, -571.1666, 1, 0.1686275, 0, 1,
7.353535, 5.787879, -571.4729, 1, 0.1686275, 0, 1,
7.393939, 5.787879, -571.789, 1, 0.06666667, 0, 1,
7.434343, 5.787879, -572.1149, 1, 0.06666667, 0, 1,
7.474748, 5.787879, -572.4504, 1, 0.06666667, 0, 1,
7.515152, 5.787879, -572.7958, 1, 0.06666667, 0, 1,
7.555555, 5.787879, -573.1509, 1, 0.06666667, 0, 1,
7.59596, 5.787879, -573.5157, 1, 0.06666667, 0, 1,
7.636364, 5.787879, -573.8903, 1, 0.06666667, 0, 1,
7.676768, 5.787879, -574.2747, 1, 0.06666667, 0, 1,
7.717172, 5.787879, -574.6687, 1, 0.06666667, 0, 1,
7.757576, 5.787879, -575.0725, 1, 0.06666667, 0, 1,
7.79798, 5.787879, -575.4861, 1, 0.06666667, 0, 1,
7.838384, 5.787879, -575.9094, 1, 0.06666667, 0, 1,
7.878788, 5.787879, -576.3425, 1, 0.06666667, 0, 1,
7.919192, 5.787879, -576.7853, 1, 0.06666667, 0, 1,
7.959596, 5.787879, -577.2379, 1, 0.06666667, 0, 1,
8, 5.787879, -577.7001, 0.9647059, 0, 0.03137255, 1,
4, 5.838384, -580.1869, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.838384, -579.7026, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.838384, -579.2278, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.838384, -578.7626, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.838384, -578.307, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.838384, -577.861, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.838384, -577.4245, 1, 0.06666667, 0, 1,
4.282828, 5.838384, -576.9976, 1, 0.06666667, 0, 1,
4.323232, 5.838384, -576.5803, 1, 0.06666667, 0, 1,
4.363636, 5.838384, -576.1726, 1, 0.06666667, 0, 1,
4.40404, 5.838384, -575.7744, 1, 0.06666667, 0, 1,
4.444445, 5.838384, -575.3859, 1, 0.06666667, 0, 1,
4.484848, 5.838384, -575.0069, 1, 0.06666667, 0, 1,
4.525252, 5.838384, -574.6375, 1, 0.06666667, 0, 1,
4.565657, 5.838384, -574.2776, 1, 0.06666667, 0, 1,
4.606061, 5.838384, -573.9274, 1, 0.06666667, 0, 1,
4.646465, 5.838384, -573.5867, 1, 0.06666667, 0, 1,
4.686869, 5.838384, -573.2556, 1, 0.06666667, 0, 1,
4.727273, 5.838384, -572.9341, 1, 0.06666667, 0, 1,
4.767677, 5.838384, -572.6222, 1, 0.06666667, 0, 1,
4.808081, 5.838384, -572.3198, 1, 0.06666667, 0, 1,
4.848485, 5.838384, -572.027, 1, 0.06666667, 0, 1,
4.888889, 5.838384, -571.7438, 1, 0.06666667, 0, 1,
4.929293, 5.838384, -571.4702, 1, 0.1686275, 0, 1,
4.969697, 5.838384, -571.2062, 1, 0.1686275, 0, 1,
5.010101, 5.838384, -570.9517, 1, 0.1686275, 0, 1,
5.050505, 5.838384, -570.7068, 1, 0.1686275, 0, 1,
5.090909, 5.838384, -570.4715, 1, 0.1686275, 0, 1,
5.131313, 5.838384, -570.2457, 1, 0.1686275, 0, 1,
5.171717, 5.838384, -570.0296, 1, 0.1686275, 0, 1,
5.212121, 5.838384, -569.823, 1, 0.1686275, 0, 1,
5.252525, 5.838384, -569.626, 1, 0.1686275, 0, 1,
5.292929, 5.838384, -569.4386, 1, 0.1686275, 0, 1,
5.333333, 5.838384, -569.2607, 1, 0.1686275, 0, 1,
5.373737, 5.838384, -569.0925, 1, 0.1686275, 0, 1,
5.414141, 5.838384, -568.9338, 1, 0.1686275, 0, 1,
5.454545, 5.838384, -568.7847, 1, 0.1686275, 0, 1,
5.494949, 5.838384, -568.6452, 1, 0.1686275, 0, 1,
5.535354, 5.838384, -568.5153, 1, 0.1686275, 0, 1,
5.575758, 5.838384, -568.3948, 1, 0.1686275, 0, 1,
5.616162, 5.838384, -568.2841, 1, 0.1686275, 0, 1,
5.656566, 5.838384, -568.1829, 1, 0.1686275, 0, 1,
5.69697, 5.838384, -568.0912, 1, 0.1686275, 0, 1,
5.737374, 5.838384, -568.0092, 1, 0.1686275, 0, 1,
5.777778, 5.838384, -567.9367, 1, 0.1686275, 0, 1,
5.818182, 5.838384, -567.8738, 1, 0.1686275, 0, 1,
5.858586, 5.838384, -567.8205, 1, 0.1686275, 0, 1,
5.89899, 5.838384, -567.7767, 1, 0.1686275, 0, 1,
5.939394, 5.838384, -567.7426, 1, 0.1686275, 0, 1,
5.979798, 5.838384, -567.718, 1, 0.1686275, 0, 1,
6.020202, 5.838384, -567.7029, 1, 0.1686275, 0, 1,
6.060606, 5.838384, -567.6976, 1, 0.1686275, 0, 1,
6.10101, 5.838384, -567.7017, 1, 0.1686275, 0, 1,
6.141414, 5.838384, -567.7154, 1, 0.1686275, 0, 1,
6.181818, 5.838384, -567.7387, 1, 0.1686275, 0, 1,
6.222222, 5.838384, -567.7716, 1, 0.1686275, 0, 1,
6.262626, 5.838384, -567.8141, 1, 0.1686275, 0, 1,
6.30303, 5.838384, -567.8661, 1, 0.1686275, 0, 1,
6.343434, 5.838384, -567.9277, 1, 0.1686275, 0, 1,
6.383838, 5.838384, -567.999, 1, 0.1686275, 0, 1,
6.424242, 5.838384, -568.0797, 1, 0.1686275, 0, 1,
6.464646, 5.838384, -568.17, 1, 0.1686275, 0, 1,
6.505051, 5.838384, -568.27, 1, 0.1686275, 0, 1,
6.545455, 5.838384, -568.3795, 1, 0.1686275, 0, 1,
6.585859, 5.838384, -568.4986, 1, 0.1686275, 0, 1,
6.626263, 5.838384, -568.6273, 1, 0.1686275, 0, 1,
6.666667, 5.838384, -568.7655, 1, 0.1686275, 0, 1,
6.707071, 5.838384, -568.9133, 1, 0.1686275, 0, 1,
6.747475, 5.838384, -569.0707, 1, 0.1686275, 0, 1,
6.787879, 5.838384, -569.2377, 1, 0.1686275, 0, 1,
6.828283, 5.838384, -569.4143, 1, 0.1686275, 0, 1,
6.868687, 5.838384, -569.6005, 1, 0.1686275, 0, 1,
6.909091, 5.838384, -569.7961, 1, 0.1686275, 0, 1,
6.949495, 5.838384, -570.0015, 1, 0.1686275, 0, 1,
6.989899, 5.838384, -570.2163, 1, 0.1686275, 0, 1,
7.030303, 5.838384, -570.4408, 1, 0.1686275, 0, 1,
7.070707, 5.838384, -570.6748, 1, 0.1686275, 0, 1,
7.111111, 5.838384, -570.9184, 1, 0.1686275, 0, 1,
7.151515, 5.838384, -571.1716, 1, 0.1686275, 0, 1,
7.191919, 5.838384, -571.4344, 1, 0.1686275, 0, 1,
7.232323, 5.838384, -571.7067, 1, 0.06666667, 0, 1,
7.272727, 5.838384, -571.9886, 1, 0.06666667, 0, 1,
7.313131, 5.838384, -572.2802, 1, 0.06666667, 0, 1,
7.353535, 5.838384, -572.5812, 1, 0.06666667, 0, 1,
7.393939, 5.838384, -572.8919, 1, 0.06666667, 0, 1,
7.434343, 5.838384, -573.2121, 1, 0.06666667, 0, 1,
7.474748, 5.838384, -573.5419, 1, 0.06666667, 0, 1,
7.515152, 5.838384, -573.8813, 1, 0.06666667, 0, 1,
7.555555, 5.838384, -574.2303, 1, 0.06666667, 0, 1,
7.59596, 5.838384, -574.5889, 1, 0.06666667, 0, 1,
7.636364, 5.838384, -574.957, 1, 0.06666667, 0, 1,
7.676768, 5.838384, -575.3347, 1, 0.06666667, 0, 1,
7.717172, 5.838384, -575.722, 1, 0.06666667, 0, 1,
7.757576, 5.838384, -576.1188, 1, 0.06666667, 0, 1,
7.79798, 5.838384, -576.5253, 1, 0.06666667, 0, 1,
7.838384, 5.838384, -576.9413, 1, 0.06666667, 0, 1,
7.878788, 5.838384, -577.3669, 1, 0.06666667, 0, 1,
7.919192, 5.838384, -577.8021, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.838384, -578.2468, 0.9647059, 0, 0.03137255, 1,
8, 5.838384, -578.7012, 0.9647059, 0, 0.03137255, 1,
4, 5.888889, -581.1665, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.888889, -580.6904, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.888889, -580.2238, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.888889, -579.7665, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.888889, -579.3187, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.888889, -578.8802, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.888889, -578.4512, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.888889, -578.0317, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.888889, -577.6215, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.888889, -577.2207, 1, 0.06666667, 0, 1,
4.40404, 5.888889, -576.8294, 1, 0.06666667, 0, 1,
4.444445, 5.888889, -576.4474, 1, 0.06666667, 0, 1,
4.484848, 5.888889, -576.075, 1, 0.06666667, 0, 1,
4.525252, 5.888889, -575.7119, 1, 0.06666667, 0, 1,
4.565657, 5.888889, -575.3582, 1, 0.06666667, 0, 1,
4.606061, 5.888889, -575.0139, 1, 0.06666667, 0, 1,
4.646465, 5.888889, -574.679, 1, 0.06666667, 0, 1,
4.686869, 5.888889, -574.3536, 1, 0.06666667, 0, 1,
4.727273, 5.888889, -574.0375, 1, 0.06666667, 0, 1,
4.767677, 5.888889, -573.731, 1, 0.06666667, 0, 1,
4.808081, 5.888889, -573.4338, 1, 0.06666667, 0, 1,
4.848485, 5.888889, -573.146, 1, 0.06666667, 0, 1,
4.888889, 5.888889, -572.8676, 1, 0.06666667, 0, 1,
4.929293, 5.888889, -572.5986, 1, 0.06666667, 0, 1,
4.969697, 5.888889, -572.3391, 1, 0.06666667, 0, 1,
5.010101, 5.888889, -572.089, 1, 0.06666667, 0, 1,
5.050505, 5.888889, -571.8483, 1, 0.06666667, 0, 1,
5.090909, 5.888889, -571.617, 1, 0.1686275, 0, 1,
5.131313, 5.888889, -571.3951, 1, 0.1686275, 0, 1,
5.171717, 5.888889, -571.1827, 1, 0.1686275, 0, 1,
5.212121, 5.888889, -570.9796, 1, 0.1686275, 0, 1,
5.252525, 5.888889, -570.7859, 1, 0.1686275, 0, 1,
5.292929, 5.888889, -570.6017, 1, 0.1686275, 0, 1,
5.333333, 5.888889, -570.4269, 1, 0.1686275, 0, 1,
5.373737, 5.888889, -570.2615, 1, 0.1686275, 0, 1,
5.414141, 5.888889, -570.1056, 1, 0.1686275, 0, 1,
5.454545, 5.888889, -569.959, 1, 0.1686275, 0, 1,
5.494949, 5.888889, -569.8219, 1, 0.1686275, 0, 1,
5.535354, 5.888889, -569.6942, 1, 0.1686275, 0, 1,
5.575758, 5.888889, -569.5759, 1, 0.1686275, 0, 1,
5.616162, 5.888889, -569.4669, 1, 0.1686275, 0, 1,
5.656566, 5.888889, -569.3674, 1, 0.1686275, 0, 1,
5.69697, 5.888889, -569.2774, 1, 0.1686275, 0, 1,
5.737374, 5.888889, -569.1967, 1, 0.1686275, 0, 1,
5.777778, 5.888889, -569.1255, 1, 0.1686275, 0, 1,
5.818182, 5.888889, -569.0637, 1, 0.1686275, 0, 1,
5.858586, 5.888889, -569.0112, 1, 0.1686275, 0, 1,
5.89899, 5.888889, -568.9683, 1, 0.1686275, 0, 1,
5.939394, 5.888889, -568.9347, 1, 0.1686275, 0, 1,
5.979798, 5.888889, -568.9105, 1, 0.1686275, 0, 1,
6.020202, 5.888889, -568.8958, 1, 0.1686275, 0, 1,
6.060606, 5.888889, -568.8904, 1, 0.1686275, 0, 1,
6.10101, 5.888889, -568.8945, 1, 0.1686275, 0, 1,
6.141414, 5.888889, -568.908, 1, 0.1686275, 0, 1,
6.181818, 5.888889, -568.9309, 1, 0.1686275, 0, 1,
6.222222, 5.888889, -568.9633, 1, 0.1686275, 0, 1,
6.262626, 5.888889, -569.0049, 1, 0.1686275, 0, 1,
6.30303, 5.888889, -569.0562, 1, 0.1686275, 0, 1,
6.343434, 5.888889, -569.1167, 1, 0.1686275, 0, 1,
6.383838, 5.888889, -569.1866, 1, 0.1686275, 0, 1,
6.424242, 5.888889, -569.2661, 1, 0.1686275, 0, 1,
6.464646, 5.888889, -569.3549, 1, 0.1686275, 0, 1,
6.505051, 5.888889, -569.4531, 1, 0.1686275, 0, 1,
6.545455, 5.888889, -569.5607, 1, 0.1686275, 0, 1,
6.585859, 5.888889, -569.6778, 1, 0.1686275, 0, 1,
6.626263, 5.888889, -569.8043, 1, 0.1686275, 0, 1,
6.666667, 5.888889, -569.9402, 1, 0.1686275, 0, 1,
6.707071, 5.888889, -570.0854, 1, 0.1686275, 0, 1,
6.747475, 5.888889, -570.2402, 1, 0.1686275, 0, 1,
6.787879, 5.888889, -570.4043, 1, 0.1686275, 0, 1,
6.828283, 5.888889, -570.5779, 1, 0.1686275, 0, 1,
6.868687, 5.888889, -570.7608, 1, 0.1686275, 0, 1,
6.909091, 5.888889, -570.9532, 1, 0.1686275, 0, 1,
6.949495, 5.888889, -571.155, 1, 0.1686275, 0, 1,
6.989899, 5.888889, -571.3662, 1, 0.1686275, 0, 1,
7.030303, 5.888889, -571.5868, 1, 0.1686275, 0, 1,
7.070707, 5.888889, -571.8168, 1, 0.06666667, 0, 1,
7.111111, 5.888889, -572.0563, 1, 0.06666667, 0, 1,
7.151515, 5.888889, -572.3052, 1, 0.06666667, 0, 1,
7.191919, 5.888889, -572.5634, 1, 0.06666667, 0, 1,
7.232323, 5.888889, -572.8311, 1, 0.06666667, 0, 1,
7.272727, 5.888889, -573.1082, 1, 0.06666667, 0, 1,
7.313131, 5.888889, -573.3948, 1, 0.06666667, 0, 1,
7.353535, 5.888889, -573.6907, 1, 0.06666667, 0, 1,
7.393939, 5.888889, -573.996, 1, 0.06666667, 0, 1,
7.434343, 5.888889, -574.3109, 1, 0.06666667, 0, 1,
7.474748, 5.888889, -574.635, 1, 0.06666667, 0, 1,
7.515152, 5.888889, -574.9686, 1, 0.06666667, 0, 1,
7.555555, 5.888889, -575.3116, 1, 0.06666667, 0, 1,
7.59596, 5.888889, -575.6641, 1, 0.06666667, 0, 1,
7.636364, 5.888889, -576.0259, 1, 0.06666667, 0, 1,
7.676768, 5.888889, -576.3972, 1, 0.06666667, 0, 1,
7.717172, 5.888889, -576.7778, 1, 0.06666667, 0, 1,
7.757576, 5.888889, -577.1679, 1, 0.06666667, 0, 1,
7.79798, 5.888889, -577.5674, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.888889, -577.9763, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.888889, -578.3947, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.888889, -578.8224, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.888889, -579.2596, 0.9647059, 0, 0.03137255, 1,
8, 5.888889, -579.7061, 0.9647059, 0, 0.03137255, 1,
4, 5.939394, -582.1502, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.939394, -581.6822, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.939394, -581.2234, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.939394, -580.7739, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.939394, -580.3337, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.939394, -579.9027, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.939394, -579.481, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.939394, -579.0685, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.939394, -578.6653, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.939394, -578.2713, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.939394, -577.8866, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.939394, -577.5111, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.939394, -577.1449, 1, 0.06666667, 0, 1,
4.525252, 5.939394, -576.788, 1, 0.06666667, 0, 1,
4.565657, 5.939394, -576.4402, 1, 0.06666667, 0, 1,
4.606061, 5.939394, -576.1018, 1, 0.06666667, 0, 1,
4.646465, 5.939394, -575.7726, 1, 0.06666667, 0, 1,
4.686869, 5.939394, -575.4527, 1, 0.06666667, 0, 1,
4.727273, 5.939394, -575.142, 1, 0.06666667, 0, 1,
4.767677, 5.939394, -574.8406, 1, 0.06666667, 0, 1,
4.808081, 5.939394, -574.5485, 1, 0.06666667, 0, 1,
4.848485, 5.939394, -574.2656, 1, 0.06666667, 0, 1,
4.888889, 5.939394, -573.9919, 1, 0.06666667, 0, 1,
4.929293, 5.939394, -573.7275, 1, 0.06666667, 0, 1,
4.969697, 5.939394, -573.4724, 1, 0.06666667, 0, 1,
5.010101, 5.939394, -573.2264, 1, 0.06666667, 0, 1,
5.050505, 5.939394, -572.9898, 1, 0.06666667, 0, 1,
5.090909, 5.939394, -572.7625, 1, 0.06666667, 0, 1,
5.131313, 5.939394, -572.5443, 1, 0.06666667, 0, 1,
5.171717, 5.939394, -572.3354, 1, 0.06666667, 0, 1,
5.212121, 5.939394, -572.1359, 1, 0.06666667, 0, 1,
5.252525, 5.939394, -571.9455, 1, 0.06666667, 0, 1,
5.292929, 5.939394, -571.7644, 1, 0.06666667, 0, 1,
5.333333, 5.939394, -571.5926, 1, 0.1686275, 0, 1,
5.373737, 5.939394, -571.43, 1, 0.1686275, 0, 1,
5.414141, 5.939394, -571.2766, 1, 0.1686275, 0, 1,
5.454545, 5.939394, -571.1326, 1, 0.1686275, 0, 1,
5.494949, 5.939394, -570.9977, 1, 0.1686275, 0, 1,
5.535354, 5.939394, -570.8722, 1, 0.1686275, 0, 1,
5.575758, 5.939394, -570.7559, 1, 0.1686275, 0, 1,
5.616162, 5.939394, -570.6488, 1, 0.1686275, 0, 1,
5.656566, 5.939394, -570.551, 1, 0.1686275, 0, 1,
5.69697, 5.939394, -570.4625, 1, 0.1686275, 0, 1,
5.737374, 5.939394, -570.3832, 1, 0.1686275, 0, 1,
5.777778, 5.939394, -570.3132, 1, 0.1686275, 0, 1,
5.818182, 5.939394, -570.2524, 1, 0.1686275, 0, 1,
5.858586, 5.939394, -570.2009, 1, 0.1686275, 0, 1,
5.89899, 5.939394, -570.1586, 1, 0.1686275, 0, 1,
5.939394, 5.939394, -570.1255, 1, 0.1686275, 0, 1,
5.979798, 5.939394, -570.1018, 1, 0.1686275, 0, 1,
6.020202, 5.939394, -570.0873, 1, 0.1686275, 0, 1,
6.060606, 5.939394, -570.082, 1, 0.1686275, 0, 1,
6.10101, 5.939394, -570.0861, 1, 0.1686275, 0, 1,
6.141414, 5.939394, -570.0993, 1, 0.1686275, 0, 1,
6.181818, 5.939394, -570.1218, 1, 0.1686275, 0, 1,
6.222222, 5.939394, -570.1536, 1, 0.1686275, 0, 1,
6.262626, 5.939394, -570.1946, 1, 0.1686275, 0, 1,
6.30303, 5.939394, -570.2449, 1, 0.1686275, 0, 1,
6.343434, 5.939394, -570.3045, 1, 0.1686275, 0, 1,
6.383838, 5.939394, -570.3733, 1, 0.1686275, 0, 1,
6.424242, 5.939394, -570.4514, 1, 0.1686275, 0, 1,
6.464646, 5.939394, -570.5386, 1, 0.1686275, 0, 1,
6.505051, 5.939394, -570.6352, 1, 0.1686275, 0, 1,
6.545455, 5.939394, -570.741, 1, 0.1686275, 0, 1,
6.585859, 5.939394, -570.8561, 1, 0.1686275, 0, 1,
6.626263, 5.939394, -570.9804, 1, 0.1686275, 0, 1,
6.666667, 5.939394, -571.114, 1, 0.1686275, 0, 1,
6.707071, 5.939394, -571.2568, 1, 0.1686275, 0, 1,
6.747475, 5.939394, -571.4089, 1, 0.1686275, 0, 1,
6.787879, 5.939394, -571.5703, 1, 0.1686275, 0, 1,
6.828283, 5.939394, -571.7409, 1, 0.06666667, 0, 1,
6.868687, 5.939394, -571.9208, 1, 0.06666667, 0, 1,
6.909091, 5.939394, -572.1099, 1, 0.06666667, 0, 1,
6.949495, 5.939394, -572.3083, 1, 0.06666667, 0, 1,
6.989899, 5.939394, -572.5159, 1, 0.06666667, 0, 1,
7.030303, 5.939394, -572.7328, 1, 0.06666667, 0, 1,
7.070707, 5.939394, -572.9589, 1, 0.06666667, 0, 1,
7.111111, 5.939394, -573.1943, 1, 0.06666667, 0, 1,
7.151515, 5.939394, -573.439, 1, 0.06666667, 0, 1,
7.191919, 5.939394, -573.6929, 1, 0.06666667, 0, 1,
7.232323, 5.939394, -573.9561, 1, 0.06666667, 0, 1,
7.272727, 5.939394, -574.2285, 1, 0.06666667, 0, 1,
7.313131, 5.939394, -574.5101, 1, 0.06666667, 0, 1,
7.353535, 5.939394, -574.801, 1, 0.06666667, 0, 1,
7.393939, 5.939394, -575.1013, 1, 0.06666667, 0, 1,
7.434343, 5.939394, -575.4106, 1, 0.06666667, 0, 1,
7.474748, 5.939394, -575.7294, 1, 0.06666667, 0, 1,
7.515152, 5.939394, -576.0573, 1, 0.06666667, 0, 1,
7.555555, 5.939394, -576.3945, 1, 0.06666667, 0, 1,
7.59596, 5.939394, -576.741, 1, 0.06666667, 0, 1,
7.636364, 5.939394, -577.0967, 1, 0.06666667, 0, 1,
7.676768, 5.939394, -577.4617, 1, 0.06666667, 0, 1,
7.717172, 5.939394, -577.8359, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.939394, -578.2194, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.939394, -578.6121, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.939394, -579.0141, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.939394, -579.4254, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.939394, -579.8458, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.939394, -580.2756, 0.9647059, 0, 0.03137255, 1,
8, 5.939394, -580.7146, 0.9647059, 0, 0.03137255, 1,
4, 5.989899, -583.1378, 0.9647059, 0, 0.03137255, 1,
4.040404, 5.989899, -582.6776, 0.9647059, 0, 0.03137255, 1,
4.080808, 5.989899, -582.2266, 0.9647059, 0, 0.03137255, 1,
4.121212, 5.989899, -581.7846, 0.9647059, 0, 0.03137255, 1,
4.161616, 5.989899, -581.3517, 0.9647059, 0, 0.03137255, 1,
4.20202, 5.989899, -580.928, 0.9647059, 0, 0.03137255, 1,
4.242424, 5.989899, -580.5133, 0.9647059, 0, 0.03137255, 1,
4.282828, 5.989899, -580.1078, 0.9647059, 0, 0.03137255, 1,
4.323232, 5.989899, -579.7113, 0.9647059, 0, 0.03137255, 1,
4.363636, 5.989899, -579.324, 0.9647059, 0, 0.03137255, 1,
4.40404, 5.989899, -578.9457, 0.9647059, 0, 0.03137255, 1,
4.444445, 5.989899, -578.5765, 0.9647059, 0, 0.03137255, 1,
4.484848, 5.989899, -578.2165, 0.9647059, 0, 0.03137255, 1,
4.525252, 5.989899, -577.8655, 0.9647059, 0, 0.03137255, 1,
4.565657, 5.989899, -577.5237, 0.9647059, 0, 0.03137255, 1,
4.606061, 5.989899, -577.1909, 1, 0.06666667, 0, 1,
4.646465, 5.989899, -576.8672, 1, 0.06666667, 0, 1,
4.686869, 5.989899, -576.5527, 1, 0.06666667, 0, 1,
4.727273, 5.989899, -576.2473, 1, 0.06666667, 0, 1,
4.767677, 5.989899, -575.9509, 1, 0.06666667, 0, 1,
4.808081, 5.989899, -575.6636, 1, 0.06666667, 0, 1,
4.848485, 5.989899, -575.3854, 1, 0.06666667, 0, 1,
4.888889, 5.989899, -575.1164, 1, 0.06666667, 0, 1,
4.929293, 5.989899, -574.8564, 1, 0.06666667, 0, 1,
4.969697, 5.989899, -574.6056, 1, 0.06666667, 0, 1,
5.010101, 5.989899, -574.3638, 1, 0.06666667, 0, 1,
5.050505, 5.989899, -574.1312, 1, 0.06666667, 0, 1,
5.090909, 5.989899, -573.9076, 1, 0.06666667, 0, 1,
5.131313, 5.989899, -573.6932, 1, 0.06666667, 0, 1,
5.171717, 5.989899, -573.4878, 1, 0.06666667, 0, 1,
5.212121, 5.989899, -573.2916, 1, 0.06666667, 0, 1,
5.252525, 5.989899, -573.1044, 1, 0.06666667, 0, 1,
5.292929, 5.989899, -572.9263, 1, 0.06666667, 0, 1,
5.333333, 5.989899, -572.7574, 1, 0.06666667, 0, 1,
5.373737, 5.989899, -572.5975, 1, 0.06666667, 0, 1,
5.414141, 5.989899, -572.4468, 1, 0.06666667, 0, 1,
5.454545, 5.989899, -572.3051, 1, 0.06666667, 0, 1,
5.494949, 5.989899, -572.1725, 1, 0.06666667, 0, 1,
5.535354, 5.989899, -572.0491, 1, 0.06666667, 0, 1,
5.575758, 5.989899, -571.9348, 1, 0.06666667, 0, 1,
5.616162, 5.989899, -571.8295, 1, 0.06666667, 0, 1,
5.656566, 5.989899, -571.7333, 1, 0.06666667, 0, 1,
5.69697, 5.989899, -571.6462, 1, 0.1686275, 0, 1,
5.737374, 5.989899, -571.5683, 1, 0.1686275, 0, 1,
5.777778, 5.989899, -571.4995, 1, 0.1686275, 0, 1,
5.818182, 5.989899, -571.4397, 1, 0.1686275, 0, 1,
5.858586, 5.989899, -571.389, 1, 0.1686275, 0, 1,
5.89899, 5.989899, -571.3475, 1, 0.1686275, 0, 1,
5.939394, 5.989899, -571.315, 1, 0.1686275, 0, 1,
5.979798, 5.989899, -571.2916, 1, 0.1686275, 0, 1,
6.020202, 5.989899, -571.2774, 1, 0.1686275, 0, 1,
6.060606, 5.989899, -571.2722, 1, 0.1686275, 0, 1,
6.10101, 5.989899, -571.2762, 1, 0.1686275, 0, 1,
6.141414, 5.989899, -571.2892, 1, 0.1686275, 0, 1,
6.181818, 5.989899, -571.3113, 1, 0.1686275, 0, 1,
6.222222, 5.989899, -571.3426, 1, 0.1686275, 0, 1,
6.262626, 5.989899, -571.3829, 1, 0.1686275, 0, 1,
6.30303, 5.989899, -571.4324, 1, 0.1686275, 0, 1,
6.343434, 5.989899, -571.4909, 1, 0.1686275, 0, 1,
6.383838, 5.989899, -571.5586, 1, 0.1686275, 0, 1,
6.424242, 5.989899, -571.6353, 1, 0.1686275, 0, 1,
6.464646, 5.989899, -571.7211, 1, 0.06666667, 0, 1,
6.505051, 5.989899, -571.8161, 1, 0.06666667, 0, 1,
6.545455, 5.989899, -571.9202, 1, 0.06666667, 0, 1,
6.585859, 5.989899, -572.0333, 1, 0.06666667, 0, 1,
6.626263, 5.989899, -572.1555, 1, 0.06666667, 0, 1,
6.666667, 5.989899, -572.2869, 1, 0.06666667, 0, 1,
6.707071, 5.989899, -572.4273, 1, 0.06666667, 0, 1,
6.747475, 5.989899, -572.5768, 1, 0.06666667, 0, 1,
6.787879, 5.989899, -572.7355, 1, 0.06666667, 0, 1,
6.828283, 5.989899, -572.9033, 1, 0.06666667, 0, 1,
6.868687, 5.989899, -573.0801, 1, 0.06666667, 0, 1,
6.909091, 5.989899, -573.266, 1, 0.06666667, 0, 1,
6.949495, 5.989899, -573.4611, 1, 0.06666667, 0, 1,
6.989899, 5.989899, -573.6652, 1, 0.06666667, 0, 1,
7.030303, 5.989899, -573.8784, 1, 0.06666667, 0, 1,
7.070707, 5.989899, -574.1008, 1, 0.06666667, 0, 1,
7.111111, 5.989899, -574.3322, 1, 0.06666667, 0, 1,
7.151515, 5.989899, -574.5728, 1, 0.06666667, 0, 1,
7.191919, 5.989899, -574.8224, 1, 0.06666667, 0, 1,
7.232323, 5.989899, -575.0812, 1, 0.06666667, 0, 1,
7.272727, 5.989899, -575.349, 1, 0.06666667, 0, 1,
7.313131, 5.989899, -575.6259, 1, 0.06666667, 0, 1,
7.353535, 5.989899, -575.912, 1, 0.06666667, 0, 1,
7.393939, 5.989899, -576.2071, 1, 0.06666667, 0, 1,
7.434343, 5.989899, -576.5114, 1, 0.06666667, 0, 1,
7.474748, 5.989899, -576.8247, 1, 0.06666667, 0, 1,
7.515152, 5.989899, -577.1472, 1, 0.06666667, 0, 1,
7.555555, 5.989899, -577.4787, 0.9647059, 0, 0.03137255, 1,
7.59596, 5.989899, -577.8193, 0.9647059, 0, 0.03137255, 1,
7.636364, 5.989899, -578.1691, 0.9647059, 0, 0.03137255, 1,
7.676768, 5.989899, -578.5279, 0.9647059, 0, 0.03137255, 1,
7.717172, 5.989899, -578.8959, 0.9647059, 0, 0.03137255, 1,
7.757576, 5.989899, -579.2729, 0.9647059, 0, 0.03137255, 1,
7.79798, 5.989899, -579.6591, 0.9647059, 0, 0.03137255, 1,
7.838384, 5.989899, -580.0543, 0.9647059, 0, 0.03137255, 1,
7.878788, 5.989899, -580.4586, 0.9647059, 0, 0.03137255, 1,
7.919192, 5.989899, -580.8721, 0.9647059, 0, 0.03137255, 1,
7.959596, 5.989899, -581.2946, 0.9647059, 0, 0.03137255, 1,
8, 5.989899, -581.7263, 0.9647059, 0, 0.03137255, 1,
4, 6.040404, -584.1287, 0.8588235, 0, 0.1372549, 1,
4.040404, 6.040404, -583.6762, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.040404, -583.2327, 0.9647059, 0, 0.03137255, 1,
4.121212, 6.040404, -582.798, 0.9647059, 0, 0.03137255, 1,
4.161616, 6.040404, -582.3724, 0.9647059, 0, 0.03137255, 1,
4.20202, 6.040404, -581.9557, 0.9647059, 0, 0.03137255, 1,
4.242424, 6.040404, -581.548, 0.9647059, 0, 0.03137255, 1,
4.282828, 6.040404, -581.1492, 0.9647059, 0, 0.03137255, 1,
4.323232, 6.040404, -580.7593, 0.9647059, 0, 0.03137255, 1,
4.363636, 6.040404, -580.3784, 0.9647059, 0, 0.03137255, 1,
4.40404, 6.040404, -580.0064, 0.9647059, 0, 0.03137255, 1,
4.444445, 6.040404, -579.6434, 0.9647059, 0, 0.03137255, 1,
4.484848, 6.040404, -579.2894, 0.9647059, 0, 0.03137255, 1,
4.525252, 6.040404, -578.9443, 0.9647059, 0, 0.03137255, 1,
4.565657, 6.040404, -578.6081, 0.9647059, 0, 0.03137255, 1,
4.606061, 6.040404, -578.2809, 0.9647059, 0, 0.03137255, 1,
4.646465, 6.040404, -577.9626, 0.9647059, 0, 0.03137255, 1,
4.686869, 6.040404, -577.6533, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.040404, -577.3529, 1, 0.06666667, 0, 1,
4.767677, 6.040404, -577.0615, 1, 0.06666667, 0, 1,
4.808081, 6.040404, -576.779, 1, 0.06666667, 0, 1,
4.848485, 6.040404, -576.5055, 1, 0.06666667, 0, 1,
4.888889, 6.040404, -576.2409, 1, 0.06666667, 0, 1,
4.929293, 6.040404, -575.9853, 1, 0.06666667, 0, 1,
4.969697, 6.040404, -575.7386, 1, 0.06666667, 0, 1,
5.010101, 6.040404, -575.5009, 1, 0.06666667, 0, 1,
5.050505, 6.040404, -575.2721, 1, 0.06666667, 0, 1,
5.090909, 6.040404, -575.0522, 1, 0.06666667, 0, 1,
5.131313, 6.040404, -574.8414, 1, 0.06666667, 0, 1,
5.171717, 6.040404, -574.6394, 1, 0.06666667, 0, 1,
5.212121, 6.040404, -574.4465, 1, 0.06666667, 0, 1,
5.252525, 6.040404, -574.2624, 1, 0.06666667, 0, 1,
5.292929, 6.040404, -574.0873, 1, 0.06666667, 0, 1,
5.333333, 6.040404, -573.9211, 1, 0.06666667, 0, 1,
5.373737, 6.040404, -573.764, 1, 0.06666667, 0, 1,
5.414141, 6.040404, -573.6157, 1, 0.06666667, 0, 1,
5.454545, 6.040404, -573.4764, 1, 0.06666667, 0, 1,
5.494949, 6.040404, -573.3461, 1, 0.06666667, 0, 1,
5.535354, 6.040404, -573.2247, 1, 0.06666667, 0, 1,
5.575758, 6.040404, -573.1122, 1, 0.06666667, 0, 1,
5.616162, 6.040404, -573.0087, 1, 0.06666667, 0, 1,
5.656566, 6.040404, -572.9142, 1, 0.06666667, 0, 1,
5.69697, 6.040404, -572.8286, 1, 0.06666667, 0, 1,
5.737374, 6.040404, -572.7519, 1, 0.06666667, 0, 1,
5.777778, 6.040404, -572.6842, 1, 0.06666667, 0, 1,
5.818182, 6.040404, -572.6254, 1, 0.06666667, 0, 1,
5.858586, 6.040404, -572.5756, 1, 0.06666667, 0, 1,
5.89899, 6.040404, -572.5347, 1, 0.06666667, 0, 1,
5.939394, 6.040404, -572.5028, 1, 0.06666667, 0, 1,
5.979798, 6.040404, -572.4799, 1, 0.06666667, 0, 1,
6.020202, 6.040404, -572.4658, 1, 0.06666667, 0, 1,
6.060606, 6.040404, -572.4608, 1, 0.06666667, 0, 1,
6.10101, 6.040404, -572.4647, 1, 0.06666667, 0, 1,
6.141414, 6.040404, -572.4775, 1, 0.06666667, 0, 1,
6.181818, 6.040404, -572.4993, 1, 0.06666667, 0, 1,
6.222222, 6.040404, -572.53, 1, 0.06666667, 0, 1,
6.262626, 6.040404, -572.5696, 1, 0.06666667, 0, 1,
6.30303, 6.040404, -572.6182, 1, 0.06666667, 0, 1,
6.343434, 6.040404, -572.6758, 1, 0.06666667, 0, 1,
6.383838, 6.040404, -572.7423, 1, 0.06666667, 0, 1,
6.424242, 6.040404, -572.8178, 1, 0.06666667, 0, 1,
6.464646, 6.040404, -572.9022, 1, 0.06666667, 0, 1,
6.505051, 6.040404, -572.9955, 1, 0.06666667, 0, 1,
6.545455, 6.040404, -573.0979, 1, 0.06666667, 0, 1,
6.585859, 6.040404, -573.2092, 1, 0.06666667, 0, 1,
6.626263, 6.040404, -573.3293, 1, 0.06666667, 0, 1,
6.666667, 6.040404, -573.4585, 1, 0.06666667, 0, 1,
6.707071, 6.040404, -573.5966, 1, 0.06666667, 0, 1,
6.747475, 6.040404, -573.7437, 1, 0.06666667, 0, 1,
6.787879, 6.040404, -573.8997, 1, 0.06666667, 0, 1,
6.828283, 6.040404, -574.0646, 1, 0.06666667, 0, 1,
6.868687, 6.040404, -574.2385, 1, 0.06666667, 0, 1,
6.909091, 6.040404, -574.4213, 1, 0.06666667, 0, 1,
6.949495, 6.040404, -574.6132, 1, 0.06666667, 0, 1,
6.989899, 6.040404, -574.8139, 1, 0.06666667, 0, 1,
7.030303, 6.040404, -575.0236, 1, 0.06666667, 0, 1,
7.070707, 6.040404, -575.2422, 1, 0.06666667, 0, 1,
7.111111, 6.040404, -575.4698, 1, 0.06666667, 0, 1,
7.151515, 6.040404, -575.7064, 1, 0.06666667, 0, 1,
7.191919, 6.040404, -575.9518, 1, 0.06666667, 0, 1,
7.232323, 6.040404, -576.2062, 1, 0.06666667, 0, 1,
7.272727, 6.040404, -576.4697, 1, 0.06666667, 0, 1,
7.313131, 6.040404, -576.7419, 1, 0.06666667, 0, 1,
7.353535, 6.040404, -577.0233, 1, 0.06666667, 0, 1,
7.393939, 6.040404, -577.3135, 1, 0.06666667, 0, 1,
7.434343, 6.040404, -577.6127, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.040404, -577.9208, 0.9647059, 0, 0.03137255, 1,
7.515152, 6.040404, -578.2379, 0.9647059, 0, 0.03137255, 1,
7.555555, 6.040404, -578.5638, 0.9647059, 0, 0.03137255, 1,
7.59596, 6.040404, -578.8988, 0.9647059, 0, 0.03137255, 1,
7.636364, 6.040404, -579.2427, 0.9647059, 0, 0.03137255, 1,
7.676768, 6.040404, -579.5956, 0.9647059, 0, 0.03137255, 1,
7.717172, 6.040404, -579.9575, 0.9647059, 0, 0.03137255, 1,
7.757576, 6.040404, -580.3282, 0.9647059, 0, 0.03137255, 1,
7.79798, 6.040404, -580.7079, 0.9647059, 0, 0.03137255, 1,
7.838384, 6.040404, -581.0966, 0.9647059, 0, 0.03137255, 1,
7.878788, 6.040404, -581.4942, 0.9647059, 0, 0.03137255, 1,
7.919192, 6.040404, -581.9008, 0.9647059, 0, 0.03137255, 1,
7.959596, 6.040404, -582.3162, 0.9647059, 0, 0.03137255, 1,
8, 6.040404, -582.7407, 0.9647059, 0, 0.03137255, 1,
4, 6.090909, -585.1227, 0.8588235, 0, 0.1372549, 1,
4.040404, 6.090909, -584.6777, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.090909, -584.2415, 0.8588235, 0, 0.1372549, 1,
4.121212, 6.090909, -583.814, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.090909, -583.3954, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.090909, -582.9856, 0.9647059, 0, 0.03137255, 1,
4.242424, 6.090909, -582.5846, 0.9647059, 0, 0.03137255, 1,
4.282828, 6.090909, -582.1924, 0.9647059, 0, 0.03137255, 1,
4.323232, 6.090909, -581.809, 0.9647059, 0, 0.03137255, 1,
4.363636, 6.090909, -581.4343, 0.9647059, 0, 0.03137255, 1,
4.40404, 6.090909, -581.0685, 0.9647059, 0, 0.03137255, 1,
4.444445, 6.090909, -580.7115, 0.9647059, 0, 0.03137255, 1,
4.484848, 6.090909, -580.3633, 0.9647059, 0, 0.03137255, 1,
4.525252, 6.090909, -580.0239, 0.9647059, 0, 0.03137255, 1,
4.565657, 6.090909, -579.6932, 0.9647059, 0, 0.03137255, 1,
4.606061, 6.090909, -579.3715, 0.9647059, 0, 0.03137255, 1,
4.646465, 6.090909, -579.0585, 0.9647059, 0, 0.03137255, 1,
4.686869, 6.090909, -578.7542, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.090909, -578.4588, 0.9647059, 0, 0.03137255, 1,
4.767677, 6.090909, -578.1722, 0.9647059, 0, 0.03137255, 1,
4.808081, 6.090909, -577.8944, 0.9647059, 0, 0.03137255, 1,
4.848485, 6.090909, -577.6254, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.090909, -577.3652, 1, 0.06666667, 0, 1,
4.929293, 6.090909, -577.1138, 1, 0.06666667, 0, 1,
4.969697, 6.090909, -576.8712, 1, 0.06666667, 0, 1,
5.010101, 6.090909, -576.6374, 1, 0.06666667, 0, 1,
5.050505, 6.090909, -576.4124, 1, 0.06666667, 0, 1,
5.090909, 6.090909, -576.1962, 1, 0.06666667, 0, 1,
5.131313, 6.090909, -575.9888, 1, 0.06666667, 0, 1,
5.171717, 6.090909, -575.7902, 1, 0.06666667, 0, 1,
5.212121, 6.090909, -575.6003, 1, 0.06666667, 0, 1,
5.252525, 6.090909, -575.4194, 1, 0.06666667, 0, 1,
5.292929, 6.090909, -575.2471, 1, 0.06666667, 0, 1,
5.333333, 6.090909, -575.0837, 1, 0.06666667, 0, 1,
5.373737, 6.090909, -574.9291, 1, 0.06666667, 0, 1,
5.414141, 6.090909, -574.7833, 1, 0.06666667, 0, 1,
5.454545, 6.090909, -574.6464, 1, 0.06666667, 0, 1,
5.494949, 6.090909, -574.5182, 1, 0.06666667, 0, 1,
5.535354, 6.090909, -574.3987, 1, 0.06666667, 0, 1,
5.575758, 6.090909, -574.2881, 1, 0.06666667, 0, 1,
5.616162, 6.090909, -574.1864, 1, 0.06666667, 0, 1,
5.656566, 6.090909, -574.0934, 1, 0.06666667, 0, 1,
5.69697, 6.090909, -574.0092, 1, 0.06666667, 0, 1,
5.737374, 6.090909, -573.9338, 1, 0.06666667, 0, 1,
5.777778, 6.090909, -573.8672, 1, 0.06666667, 0, 1,
5.818182, 6.090909, -573.8094, 1, 0.06666667, 0, 1,
5.858586, 6.090909, -573.7604, 1, 0.06666667, 0, 1,
5.89899, 6.090909, -573.7202, 1, 0.06666667, 0, 1,
5.939394, 6.090909, -573.6888, 1, 0.06666667, 0, 1,
5.979798, 6.090909, -573.6663, 1, 0.06666667, 0, 1,
6.020202, 6.090909, -573.6525, 1, 0.06666667, 0, 1,
6.060606, 6.090909, -573.6475, 1, 0.06666667, 0, 1,
6.10101, 6.090909, -573.6513, 1, 0.06666667, 0, 1,
6.141414, 6.090909, -573.6639, 1, 0.06666667, 0, 1,
6.181818, 6.090909, -573.6853, 1, 0.06666667, 0, 1,
6.222222, 6.090909, -573.7155, 1, 0.06666667, 0, 1,
6.262626, 6.090909, -573.7545, 1, 0.06666667, 0, 1,
6.30303, 6.090909, -573.8024, 1, 0.06666667, 0, 1,
6.343434, 6.090909, -573.8589, 1, 0.06666667, 0, 1,
6.383838, 6.090909, -573.9244, 1, 0.06666667, 0, 1,
6.424242, 6.090909, -573.9986, 1, 0.06666667, 0, 1,
6.464646, 6.090909, -574.0816, 1, 0.06666667, 0, 1,
6.505051, 6.090909, -574.1735, 1, 0.06666667, 0, 1,
6.545455, 6.090909, -574.274, 1, 0.06666667, 0, 1,
6.585859, 6.090909, -574.3835, 1, 0.06666667, 0, 1,
6.626263, 6.090909, -574.5017, 1, 0.06666667, 0, 1,
6.666667, 6.090909, -574.6287, 1, 0.06666667, 0, 1,
6.707071, 6.090909, -574.7645, 1, 0.06666667, 0, 1,
6.747475, 6.090909, -574.9092, 1, 0.06666667, 0, 1,
6.787879, 6.090909, -575.0626, 1, 0.06666667, 0, 1,
6.828283, 6.090909, -575.2248, 1, 0.06666667, 0, 1,
6.868687, 6.090909, -575.3959, 1, 0.06666667, 0, 1,
6.909091, 6.090909, -575.5757, 1, 0.06666667, 0, 1,
6.949495, 6.090909, -575.7643, 1, 0.06666667, 0, 1,
6.989899, 6.090909, -575.9617, 1, 0.06666667, 0, 1,
7.030303, 6.090909, -576.168, 1, 0.06666667, 0, 1,
7.070707, 6.090909, -576.383, 1, 0.06666667, 0, 1,
7.111111, 6.090909, -576.6068, 1, 0.06666667, 0, 1,
7.151515, 6.090909, -576.8394, 1, 0.06666667, 0, 1,
7.191919, 6.090909, -577.0809, 1, 0.06666667, 0, 1,
7.232323, 6.090909, -577.3311, 1, 0.06666667, 0, 1,
7.272727, 6.090909, -577.5901, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.090909, -577.858, 0.9647059, 0, 0.03137255, 1,
7.353535, 6.090909, -578.1346, 0.9647059, 0, 0.03137255, 1,
7.393939, 6.090909, -578.42, 0.9647059, 0, 0.03137255, 1,
7.434343, 6.090909, -578.7143, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.090909, -579.0173, 0.9647059, 0, 0.03137255, 1,
7.515152, 6.090909, -579.3292, 0.9647059, 0, 0.03137255, 1,
7.555555, 6.090909, -579.6498, 0.9647059, 0, 0.03137255, 1,
7.59596, 6.090909, -579.9792, 0.9647059, 0, 0.03137255, 1,
7.636364, 6.090909, -580.3174, 0.9647059, 0, 0.03137255, 1,
7.676768, 6.090909, -580.6645, 0.9647059, 0, 0.03137255, 1,
7.717172, 6.090909, -581.0203, 0.9647059, 0, 0.03137255, 1,
7.757576, 6.090909, -581.3849, 0.9647059, 0, 0.03137255, 1,
7.79798, 6.090909, -581.7584, 0.9647059, 0, 0.03137255, 1,
7.838384, 6.090909, -582.1406, 0.9647059, 0, 0.03137255, 1,
7.878788, 6.090909, -582.5317, 0.9647059, 0, 0.03137255, 1,
7.919192, 6.090909, -582.9315, 0.9647059, 0, 0.03137255, 1,
7.959596, 6.090909, -583.3401, 0.8588235, 0, 0.1372549, 1,
8, 6.090909, -583.7576, 0.8588235, 0, 0.1372549, 1,
4, 6.141414, -586.1194, 0.8588235, 0, 0.1372549, 1,
4.040404, 6.141414, -585.6817, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.141414, -585.2526, 0.8588235, 0, 0.1372549, 1,
4.121212, 6.141414, -584.8322, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.141414, -584.4205, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.141414, -584.0173, 0.8588235, 0, 0.1372549, 1,
4.242424, 6.141414, -583.6229, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.141414, -583.2371, 0.9647059, 0, 0.03137255, 1,
4.323232, 6.141414, -582.86, 0.9647059, 0, 0.03137255, 1,
4.363636, 6.141414, -582.4915, 0.9647059, 0, 0.03137255, 1,
4.40404, 6.141414, -582.1317, 0.9647059, 0, 0.03137255, 1,
4.444445, 6.141414, -581.7805, 0.9647059, 0, 0.03137255, 1,
4.484848, 6.141414, -581.438, 0.9647059, 0, 0.03137255, 1,
4.525252, 6.141414, -581.1041, 0.9647059, 0, 0.03137255, 1,
4.565657, 6.141414, -580.7789, 0.9647059, 0, 0.03137255, 1,
4.606061, 6.141414, -580.4624, 0.9647059, 0, 0.03137255, 1,
4.646465, 6.141414, -580.1545, 0.9647059, 0, 0.03137255, 1,
4.686869, 6.141414, -579.8553, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.141414, -579.5647, 0.9647059, 0, 0.03137255, 1,
4.767677, 6.141414, -579.2828, 0.9647059, 0, 0.03137255, 1,
4.808081, 6.141414, -579.0095, 0.9647059, 0, 0.03137255, 1,
4.848485, 6.141414, -578.7449, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.141414, -578.489, 0.9647059, 0, 0.03137255, 1,
4.929293, 6.141414, -578.2417, 0.9647059, 0, 0.03137255, 1,
4.969697, 6.141414, -578.0031, 0.9647059, 0, 0.03137255, 1,
5.010101, 6.141414, -577.7731, 0.9647059, 0, 0.03137255, 1,
5.050505, 6.141414, -577.5518, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.141414, -577.3391, 1, 0.06666667, 0, 1,
5.131313, 6.141414, -577.1351, 1, 0.06666667, 0, 1,
5.171717, 6.141414, -576.9398, 1, 0.06666667, 0, 1,
5.212121, 6.141414, -576.7531, 1, 0.06666667, 0, 1,
5.252525, 6.141414, -576.575, 1, 0.06666667, 0, 1,
5.292929, 6.141414, -576.4056, 1, 0.06666667, 0, 1,
5.333333, 6.141414, -576.2449, 1, 0.06666667, 0, 1,
5.373737, 6.141414, -576.0929, 1, 0.06666667, 0, 1,
5.414141, 6.141414, -575.9495, 1, 0.06666667, 0, 1,
5.454545, 6.141414, -575.8147, 1, 0.06666667, 0, 1,
5.494949, 6.141414, -575.6886, 1, 0.06666667, 0, 1,
5.535354, 6.141414, -575.5712, 1, 0.06666667, 0, 1,
5.575758, 6.141414, -575.4624, 1, 0.06666667, 0, 1,
5.616162, 6.141414, -575.3622, 1, 0.06666667, 0, 1,
5.656566, 6.141414, -575.2708, 1, 0.06666667, 0, 1,
5.69697, 6.141414, -575.188, 1, 0.06666667, 0, 1,
5.737374, 6.141414, -575.1138, 1, 0.06666667, 0, 1,
5.777778, 6.141414, -575.0483, 1, 0.06666667, 0, 1,
5.818182, 6.141414, -574.9915, 1, 0.06666667, 0, 1,
5.858586, 6.141414, -574.9433, 1, 0.06666667, 0, 1,
5.89899, 6.141414, -574.9037, 1, 0.06666667, 0, 1,
5.939394, 6.141414, -574.8729, 1, 0.06666667, 0, 1,
5.979798, 6.141414, -574.8506, 1, 0.06666667, 0, 1,
6.020202, 6.141414, -574.8371, 1, 0.06666667, 0, 1,
6.060606, 6.141414, -574.8322, 1, 0.06666667, 0, 1,
6.10101, 6.141414, -574.8359, 1, 0.06666667, 0, 1,
6.141414, 6.141414, -574.8483, 1, 0.06666667, 0, 1,
6.181818, 6.141414, -574.8694, 1, 0.06666667, 0, 1,
6.222222, 6.141414, -574.8991, 1, 0.06666667, 0, 1,
6.262626, 6.141414, -574.9375, 1, 0.06666667, 0, 1,
6.30303, 6.141414, -574.9845, 1, 0.06666667, 0, 1,
6.343434, 6.141414, -575.0402, 1, 0.06666667, 0, 1,
6.383838, 6.141414, -575.1046, 1, 0.06666667, 0, 1,
6.424242, 6.141414, -575.1776, 1, 0.06666667, 0, 1,
6.464646, 6.141414, -575.2592, 1, 0.06666667, 0, 1,
6.505051, 6.141414, -575.3495, 1, 0.06666667, 0, 1,
6.545455, 6.141414, -575.4485, 1, 0.06666667, 0, 1,
6.585859, 6.141414, -575.5562, 1, 0.06666667, 0, 1,
6.626263, 6.141414, -575.6724, 1, 0.06666667, 0, 1,
6.666667, 6.141414, -575.7974, 1, 0.06666667, 0, 1,
6.707071, 6.141414, -575.931, 1, 0.06666667, 0, 1,
6.747475, 6.141414, -576.0732, 1, 0.06666667, 0, 1,
6.787879, 6.141414, -576.2241, 1, 0.06666667, 0, 1,
6.828283, 6.141414, -576.3837, 1, 0.06666667, 0, 1,
6.868687, 6.141414, -576.5519, 1, 0.06666667, 0, 1,
6.909091, 6.141414, -576.7288, 1, 0.06666667, 0, 1,
6.949495, 6.141414, -576.9143, 1, 0.06666667, 0, 1,
6.989899, 6.141414, -577.1085, 1, 0.06666667, 0, 1,
7.030303, 6.141414, -577.3114, 1, 0.06666667, 0, 1,
7.070707, 6.141414, -577.5229, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.141414, -577.743, 0.9647059, 0, 0.03137255, 1,
7.151515, 6.141414, -577.9719, 0.9647059, 0, 0.03137255, 1,
7.191919, 6.141414, -578.2094, 0.9647059, 0, 0.03137255, 1,
7.232323, 6.141414, -578.4554, 0.9647059, 0, 0.03137255, 1,
7.272727, 6.141414, -578.7103, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.141414, -578.9737, 0.9647059, 0, 0.03137255, 1,
7.353535, 6.141414, -579.2458, 0.9647059, 0, 0.03137255, 1,
7.393939, 6.141414, -579.5266, 0.9647059, 0, 0.03137255, 1,
7.434343, 6.141414, -579.816, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.141414, -580.1141, 0.9647059, 0, 0.03137255, 1,
7.515152, 6.141414, -580.4208, 0.9647059, 0, 0.03137255, 1,
7.555555, 6.141414, -580.7361, 0.9647059, 0, 0.03137255, 1,
7.59596, 6.141414, -581.0602, 0.9647059, 0, 0.03137255, 1,
7.636364, 6.141414, -581.3929, 0.9647059, 0, 0.03137255, 1,
7.676768, 6.141414, -581.7343, 0.9647059, 0, 0.03137255, 1,
7.717172, 6.141414, -582.0843, 0.9647059, 0, 0.03137255, 1,
7.757576, 6.141414, -582.4429, 0.9647059, 0, 0.03137255, 1,
7.79798, 6.141414, -582.8102, 0.9647059, 0, 0.03137255, 1,
7.838384, 6.141414, -583.1862, 0.9647059, 0, 0.03137255, 1,
7.878788, 6.141414, -583.5709, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.141414, -583.9642, 0.8588235, 0, 0.1372549, 1,
7.959596, 6.141414, -584.3661, 0.8588235, 0, 0.1372549, 1,
8, 6.141414, -584.7767, 0.8588235, 0, 0.1372549, 1,
4, 6.191919, -587.1187, 0.8588235, 0, 0.1372549, 1,
4.040404, 6.191919, -586.688, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.191919, -586.2659, 0.8588235, 0, 0.1372549, 1,
4.121212, 6.191919, -585.8523, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.191919, -585.4473, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.191919, -585.0507, 0.8588235, 0, 0.1372549, 1,
4.242424, 6.191919, -584.6627, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.191919, -584.2831, 0.8588235, 0, 0.1372549, 1,
4.323232, 6.191919, -583.9121, 0.8588235, 0, 0.1372549, 1,
4.363636, 6.191919, -583.5496, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.191919, -583.1956, 0.9647059, 0, 0.03137255, 1,
4.444445, 6.191919, -582.8502, 0.9647059, 0, 0.03137255, 1,
4.484848, 6.191919, -582.5132, 0.9647059, 0, 0.03137255, 1,
4.525252, 6.191919, -582.1848, 0.9647059, 0, 0.03137255, 1,
4.565657, 6.191919, -581.8649, 0.9647059, 0, 0.03137255, 1,
4.606061, 6.191919, -581.5535, 0.9647059, 0, 0.03137255, 1,
4.646465, 6.191919, -581.2506, 0.9647059, 0, 0.03137255, 1,
4.686869, 6.191919, -580.9562, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.191919, -580.6704, 0.9647059, 0, 0.03137255, 1,
4.767677, 6.191919, -580.3931, 0.9647059, 0, 0.03137255, 1,
4.808081, 6.191919, -580.1243, 0.9647059, 0, 0.03137255, 1,
4.848485, 6.191919, -579.864, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.191919, -579.6121, 0.9647059, 0, 0.03137255, 1,
4.929293, 6.191919, -579.3689, 0.9647059, 0, 0.03137255, 1,
4.969697, 6.191919, -579.1342, 0.9647059, 0, 0.03137255, 1,
5.010101, 6.191919, -578.9079, 0.9647059, 0, 0.03137255, 1,
5.050505, 6.191919, -578.6902, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.191919, -578.481, 0.9647059, 0, 0.03137255, 1,
5.131313, 6.191919, -578.2803, 0.9647059, 0, 0.03137255, 1,
5.171717, 6.191919, -578.0881, 0.9647059, 0, 0.03137255, 1,
5.212121, 6.191919, -577.9044, 0.9647059, 0, 0.03137255, 1,
5.252525, 6.191919, -577.7293, 0.9647059, 0, 0.03137255, 1,
5.292929, 6.191919, -577.5627, 0.9647059, 0, 0.03137255, 1,
5.333333, 6.191919, -577.4045, 1, 0.06666667, 0, 1,
5.373737, 6.191919, -577.2549, 1, 0.06666667, 0, 1,
5.414141, 6.191919, -577.1139, 1, 0.06666667, 0, 1,
5.454545, 6.191919, -576.9813, 1, 0.06666667, 0, 1,
5.494949, 6.191919, -576.8572, 1, 0.06666667, 0, 1,
5.535354, 6.191919, -576.7417, 1, 0.06666667, 0, 1,
5.575758, 6.191919, -576.6347, 1, 0.06666667, 0, 1,
5.616162, 6.191919, -576.5362, 1, 0.06666667, 0, 1,
5.656566, 6.191919, -576.4462, 1, 0.06666667, 0, 1,
5.69697, 6.191919, -576.3647, 1, 0.06666667, 0, 1,
5.737374, 6.191919, -576.2918, 1, 0.06666667, 0, 1,
5.777778, 6.191919, -576.2274, 1, 0.06666667, 0, 1,
5.818182, 6.191919, -576.1714, 1, 0.06666667, 0, 1,
5.858586, 6.191919, -576.124, 1, 0.06666667, 0, 1,
5.89899, 6.191919, -576.0851, 1, 0.06666667, 0, 1,
5.939394, 6.191919, -576.0547, 1, 0.06666667, 0, 1,
5.979798, 6.191919, -576.0329, 1, 0.06666667, 0, 1,
6.020202, 6.191919, -576.0196, 1, 0.06666667, 0, 1,
6.060606, 6.191919, -576.0147, 1, 0.06666667, 0, 1,
6.10101, 6.191919, -576.0184, 1, 0.06666667, 0, 1,
6.141414, 6.191919, -576.0306, 1, 0.06666667, 0, 1,
6.181818, 6.191919, -576.0513, 1, 0.06666667, 0, 1,
6.222222, 6.191919, -576.0806, 1, 0.06666667, 0, 1,
6.262626, 6.191919, -576.1183, 1, 0.06666667, 0, 1,
6.30303, 6.191919, -576.1646, 1, 0.06666667, 0, 1,
6.343434, 6.191919, -576.2194, 1, 0.06666667, 0, 1,
6.383838, 6.191919, -576.2827, 1, 0.06666667, 0, 1,
6.424242, 6.191919, -576.3545, 1, 0.06666667, 0, 1,
6.464646, 6.191919, -576.4348, 1, 0.06666667, 0, 1,
6.505051, 6.191919, -576.5237, 1, 0.06666667, 0, 1,
6.545455, 6.191919, -576.621, 1, 0.06666667, 0, 1,
6.585859, 6.191919, -576.7269, 1, 0.06666667, 0, 1,
6.626263, 6.191919, -576.8413, 1, 0.06666667, 0, 1,
6.666667, 6.191919, -576.9642, 1, 0.06666667, 0, 1,
6.707071, 6.191919, -577.0956, 1, 0.06666667, 0, 1,
6.747475, 6.191919, -577.2356, 1, 0.06666667, 0, 1,
6.787879, 6.191919, -577.3841, 1, 0.06666667, 0, 1,
6.828283, 6.191919, -577.541, 0.9647059, 0, 0.03137255, 1,
6.868687, 6.191919, -577.7065, 0.9647059, 0, 0.03137255, 1,
6.909091, 6.191919, -577.8806, 0.9647059, 0, 0.03137255, 1,
6.949495, 6.191919, -578.063, 0.9647059, 0, 0.03137255, 1,
6.989899, 6.191919, -578.2541, 0.9647059, 0, 0.03137255, 1,
7.030303, 6.191919, -578.4537, 0.9647059, 0, 0.03137255, 1,
7.070707, 6.191919, -578.6617, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.191919, -578.8783, 0.9647059, 0, 0.03137255, 1,
7.151515, 6.191919, -579.1034, 0.9647059, 0, 0.03137255, 1,
7.191919, 6.191919, -579.337, 0.9647059, 0, 0.03137255, 1,
7.232323, 6.191919, -579.5792, 0.9647059, 0, 0.03137255, 1,
7.272727, 6.191919, -579.8298, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.191919, -580.089, 0.9647059, 0, 0.03137255, 1,
7.353535, 6.191919, -580.3567, 0.9647059, 0, 0.03137255, 1,
7.393939, 6.191919, -580.6329, 0.9647059, 0, 0.03137255, 1,
7.434343, 6.191919, -580.9176, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.191919, -581.2108, 0.9647059, 0, 0.03137255, 1,
7.515152, 6.191919, -581.5126, 0.9647059, 0, 0.03137255, 1,
7.555555, 6.191919, -581.8228, 0.9647059, 0, 0.03137255, 1,
7.59596, 6.191919, -582.1416, 0.9647059, 0, 0.03137255, 1,
7.636364, 6.191919, -582.4689, 0.9647059, 0, 0.03137255, 1,
7.676768, 6.191919, -582.8047, 0.9647059, 0, 0.03137255, 1,
7.717172, 6.191919, -583.149, 0.9647059, 0, 0.03137255, 1,
7.757576, 6.191919, -583.5018, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.191919, -583.8632, 0.8588235, 0, 0.1372549, 1,
7.838384, 6.191919, -584.2331, 0.8588235, 0, 0.1372549, 1,
7.878788, 6.191919, -584.6115, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.191919, -584.9984, 0.8588235, 0, 0.1372549, 1,
7.959596, 6.191919, -585.3938, 0.8588235, 0, 0.1372549, 1,
8, 6.191919, -585.7977, 0.8588235, 0, 0.1372549, 1,
4, 6.242424, -588.1199, 0.8588235, 0, 0.1372549, 1,
4.040404, 6.242424, -587.6963, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.242424, -587.2809, 0.8588235, 0, 0.1372549, 1,
4.121212, 6.242424, -586.874, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.242424, -586.4755, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.242424, -586.0853, 0.8588235, 0, 0.1372549, 1,
4.242424, 6.242424, -585.7036, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.242424, -585.3301, 0.8588235, 0, 0.1372549, 1,
4.323232, 6.242424, -584.9651, 0.8588235, 0, 0.1372549, 1,
4.363636, 6.242424, -584.6085, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.242424, -584.2602, 0.8588235, 0, 0.1372549, 1,
4.444445, 6.242424, -583.9203, 0.8588235, 0, 0.1372549, 1,
4.484848, 6.242424, -583.5887, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.242424, -583.2656, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.242424, -582.9509, 0.9647059, 0, 0.03137255, 1,
4.606061, 6.242424, -582.6445, 0.9647059, 0, 0.03137255, 1,
4.646465, 6.242424, -582.3465, 0.9647059, 0, 0.03137255, 1,
4.686869, 6.242424, -582.0569, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.242424, -581.7756, 0.9647059, 0, 0.03137255, 1,
4.767677, 6.242424, -581.5027, 0.9647059, 0, 0.03137255, 1,
4.808081, 6.242424, -581.2383, 0.9647059, 0, 0.03137255, 1,
4.848485, 6.242424, -580.9822, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.242424, -580.7344, 0.9647059, 0, 0.03137255, 1,
4.929293, 6.242424, -580.4951, 0.9647059, 0, 0.03137255, 1,
4.969697, 6.242424, -580.2641, 0.9647059, 0, 0.03137255, 1,
5.010101, 6.242424, -580.0415, 0.9647059, 0, 0.03137255, 1,
5.050505, 6.242424, -579.8273, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.242424, -579.6215, 0.9647059, 0, 0.03137255, 1,
5.131313, 6.242424, -579.424, 0.9647059, 0, 0.03137255, 1,
5.171717, 6.242424, -579.2349, 0.9647059, 0, 0.03137255, 1,
5.212121, 6.242424, -579.0543, 0.9647059, 0, 0.03137255, 1,
5.252525, 6.242424, -578.8819, 0.9647059, 0, 0.03137255, 1,
5.292929, 6.242424, -578.718, 0.9647059, 0, 0.03137255, 1,
5.333333, 6.242424, -578.5624, 0.9647059, 0, 0.03137255, 1,
5.373737, 6.242424, -578.4152, 0.9647059, 0, 0.03137255, 1,
5.414141, 6.242424, -578.2764, 0.9647059, 0, 0.03137255, 1,
5.454545, 6.242424, -578.146, 0.9647059, 0, 0.03137255, 1,
5.494949, 6.242424, -578.0239, 0.9647059, 0, 0.03137255, 1,
5.535354, 6.242424, -577.9103, 0.9647059, 0, 0.03137255, 1,
5.575758, 6.242424, -577.805, 0.9647059, 0, 0.03137255, 1,
5.616162, 6.242424, -577.7081, 0.9647059, 0, 0.03137255, 1,
5.656566, 6.242424, -577.6195, 0.9647059, 0, 0.03137255, 1,
5.69697, 6.242424, -577.5394, 0.9647059, 0, 0.03137255, 1,
5.737374, 6.242424, -577.4676, 1, 0.06666667, 0, 1,
5.777778, 6.242424, -577.4042, 1, 0.06666667, 0, 1,
5.818182, 6.242424, -577.3492, 1, 0.06666667, 0, 1,
5.858586, 6.242424, -577.3026, 1, 0.06666667, 0, 1,
5.89899, 6.242424, -577.2643, 1, 0.06666667, 0, 1,
5.939394, 6.242424, -577.2344, 1, 0.06666667, 0, 1,
5.979798, 6.242424, -577.2129, 1, 0.06666667, 0, 1,
6.020202, 6.242424, -577.1998, 1, 0.06666667, 0, 1,
6.060606, 6.242424, -577.195, 1, 0.06666667, 0, 1,
6.10101, 6.242424, -577.1986, 1, 0.06666667, 0, 1,
6.141414, 6.242424, -577.2106, 1, 0.06666667, 0, 1,
6.181818, 6.242424, -577.231, 1, 0.06666667, 0, 1,
6.222222, 6.242424, -577.2598, 1, 0.06666667, 0, 1,
6.262626, 6.242424, -577.2969, 1, 0.06666667, 0, 1,
6.30303, 6.242424, -577.3425, 1, 0.06666667, 0, 1,
6.343434, 6.242424, -577.3964, 1, 0.06666667, 0, 1,
6.383838, 6.242424, -577.4586, 1, 0.06666667, 0, 1,
6.424242, 6.242424, -577.5293, 0.9647059, 0, 0.03137255, 1,
6.464646, 6.242424, -577.6083, 0.9647059, 0, 0.03137255, 1,
6.505051, 6.242424, -577.6957, 0.9647059, 0, 0.03137255, 1,
6.545455, 6.242424, -577.7916, 0.9647059, 0, 0.03137255, 1,
6.585859, 6.242424, -577.8957, 0.9647059, 0, 0.03137255, 1,
6.626263, 6.242424, -578.0083, 0.9647059, 0, 0.03137255, 1,
6.666667, 6.242424, -578.1292, 0.9647059, 0, 0.03137255, 1,
6.707071, 6.242424, -578.2585, 0.9647059, 0, 0.03137255, 1,
6.747475, 6.242424, -578.3962, 0.9647059, 0, 0.03137255, 1,
6.787879, 6.242424, -578.5422, 0.9647059, 0, 0.03137255, 1,
6.828283, 6.242424, -578.6967, 0.9647059, 0, 0.03137255, 1,
6.868687, 6.242424, -578.8596, 0.9647059, 0, 0.03137255, 1,
6.909091, 6.242424, -579.0308, 0.9647059, 0, 0.03137255, 1,
6.949495, 6.242424, -579.2103, 0.9647059, 0, 0.03137255, 1,
6.989899, 6.242424, -579.3983, 0.9647059, 0, 0.03137255, 1,
7.030303, 6.242424, -579.5946, 0.9647059, 0, 0.03137255, 1,
7.070707, 6.242424, -579.7993, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.242424, -580.0125, 0.9647059, 0, 0.03137255, 1,
7.151515, 6.242424, -580.2339, 0.9647059, 0, 0.03137255, 1,
7.191919, 6.242424, -580.4637, 0.9647059, 0, 0.03137255, 1,
7.232323, 6.242424, -580.702, 0.9647059, 0, 0.03137255, 1,
7.272727, 6.242424, -580.9486, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.242424, -581.2036, 0.9647059, 0, 0.03137255, 1,
7.353535, 6.242424, -581.467, 0.9647059, 0, 0.03137255, 1,
7.393939, 6.242424, -581.7387, 0.9647059, 0, 0.03137255, 1,
7.434343, 6.242424, -582.0188, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.242424, -582.3073, 0.9647059, 0, 0.03137255, 1,
7.515152, 6.242424, -582.6042, 0.9647059, 0, 0.03137255, 1,
7.555555, 6.242424, -582.9095, 0.9647059, 0, 0.03137255, 1,
7.59596, 6.242424, -583.2231, 0.9647059, 0, 0.03137255, 1,
7.636364, 6.242424, -583.5451, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.242424, -583.8755, 0.8588235, 0, 0.1372549, 1,
7.717172, 6.242424, -584.2143, 0.8588235, 0, 0.1372549, 1,
7.757576, 6.242424, -584.5615, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.242424, -584.917, 0.8588235, 0, 0.1372549, 1,
7.838384, 6.242424, -585.2809, 0.8588235, 0, 0.1372549, 1,
7.878788, 6.242424, -585.6532, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.242424, -586.0339, 0.8588235, 0, 0.1372549, 1,
7.959596, 6.242424, -586.4229, 0.8588235, 0, 0.1372549, 1,
8, 6.242424, -586.8203, 0.8588235, 0, 0.1372549, 1,
4, 6.292929, -589.1231, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.292929, -588.7062, 0.8588235, 0, 0.1372549, 1,
4.080808, 6.292929, -588.2975, 0.8588235, 0, 0.1372549, 1,
4.121212, 6.292929, -587.8972, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.292929, -587.5049, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.292929, -587.121, 0.8588235, 0, 0.1372549, 1,
4.242424, 6.292929, -586.7454, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.292929, -586.3779, 0.8588235, 0, 0.1372549, 1,
4.323232, 6.292929, -586.0187, 0.8588235, 0, 0.1372549, 1,
4.363636, 6.292929, -585.6678, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.292929, -585.3251, 0.8588235, 0, 0.1372549, 1,
4.444445, 6.292929, -584.9906, 0.8588235, 0, 0.1372549, 1,
4.484848, 6.292929, -584.6644, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.292929, -584.3464, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.292929, -584.0367, 0.8588235, 0, 0.1372549, 1,
4.606061, 6.292929, -583.7352, 0.8588235, 0, 0.1372549, 1,
4.646465, 6.292929, -583.442, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.292929, -583.157, 0.9647059, 0, 0.03137255, 1,
4.727273, 6.292929, -582.8802, 0.9647059, 0, 0.03137255, 1,
4.767677, 6.292929, -582.6117, 0.9647059, 0, 0.03137255, 1,
4.808081, 6.292929, -582.3514, 0.9647059, 0, 0.03137255, 1,
4.848485, 6.292929, -582.0994, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.292929, -581.8557, 0.9647059, 0, 0.03137255, 1,
4.929293, 6.292929, -581.6202, 0.9647059, 0, 0.03137255, 1,
4.969697, 6.292929, -581.3929, 0.9647059, 0, 0.03137255, 1,
5.010101, 6.292929, -581.1738, 0.9647059, 0, 0.03137255, 1,
5.050505, 6.292929, -580.9631, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.292929, -580.7605, 0.9647059, 0, 0.03137255, 1,
5.131313, 6.292929, -580.5662, 0.9647059, 0, 0.03137255, 1,
5.171717, 6.292929, -580.3801, 0.9647059, 0, 0.03137255, 1,
5.212121, 6.292929, -580.2023, 0.9647059, 0, 0.03137255, 1,
5.252525, 6.292929, -580.0328, 0.9647059, 0, 0.03137255, 1,
5.292929, 6.292929, -579.8715, 0.9647059, 0, 0.03137255, 1,
5.333333, 6.292929, -579.7184, 0.9647059, 0, 0.03137255, 1,
5.373737, 6.292929, -579.5735, 0.9647059, 0, 0.03137255, 1,
5.414141, 6.292929, -579.437, 0.9647059, 0, 0.03137255, 1,
5.454545, 6.292929, -579.3086, 0.9647059, 0, 0.03137255, 1,
5.494949, 6.292929, -579.1885, 0.9647059, 0, 0.03137255, 1,
5.535354, 6.292929, -579.0767, 0.9647059, 0, 0.03137255, 1,
5.575758, 6.292929, -578.973, 0.9647059, 0, 0.03137255, 1,
5.616162, 6.292929, -578.8777, 0.9647059, 0, 0.03137255, 1,
5.656566, 6.292929, -578.7906, 0.9647059, 0, 0.03137255, 1,
5.69697, 6.292929, -578.7117, 0.9647059, 0, 0.03137255, 1,
5.737374, 6.292929, -578.6411, 0.9647059, 0, 0.03137255, 1,
5.777778, 6.292929, -578.5787, 0.9647059, 0, 0.03137255, 1,
5.818182, 6.292929, -578.5245, 0.9647059, 0, 0.03137255, 1,
5.858586, 6.292929, -578.4786, 0.9647059, 0, 0.03137255, 1,
5.89899, 6.292929, -578.441, 0.9647059, 0, 0.03137255, 1,
5.939394, 6.292929, -578.4116, 0.9647059, 0, 0.03137255, 1,
5.979798, 6.292929, -578.3904, 0.9647059, 0, 0.03137255, 1,
6.020202, 6.292929, -578.3775, 0.9647059, 0, 0.03137255, 1,
6.060606, 6.292929, -578.3728, 0.9647059, 0, 0.03137255, 1,
6.10101, 6.292929, -578.3764, 0.9647059, 0, 0.03137255, 1,
6.141414, 6.292929, -578.3882, 0.9647059, 0, 0.03137255, 1,
6.181818, 6.292929, -578.4083, 0.9647059, 0, 0.03137255, 1,
6.222222, 6.292929, -578.4366, 0.9647059, 0, 0.03137255, 1,
6.262626, 6.292929, -578.4731, 0.9647059, 0, 0.03137255, 1,
6.30303, 6.292929, -578.5179, 0.9647059, 0, 0.03137255, 1,
6.343434, 6.292929, -578.571, 0.9647059, 0, 0.03137255, 1,
6.383838, 6.292929, -578.6323, 0.9647059, 0, 0.03137255, 1,
6.424242, 6.292929, -578.7018, 0.9647059, 0, 0.03137255, 1,
6.464646, 6.292929, -578.7795, 0.9647059, 0, 0.03137255, 1,
6.505051, 6.292929, -578.8656, 0.9647059, 0, 0.03137255, 1,
6.545455, 6.292929, -578.9598, 0.9647059, 0, 0.03137255, 1,
6.585859, 6.292929, -579.0623, 0.9647059, 0, 0.03137255, 1,
6.626263, 6.292929, -579.1731, 0.9647059, 0, 0.03137255, 1,
6.666667, 6.292929, -579.2921, 0.9647059, 0, 0.03137255, 1,
6.707071, 6.292929, -579.4193, 0.9647059, 0, 0.03137255, 1,
6.747475, 6.292929, -579.5548, 0.9647059, 0, 0.03137255, 1,
6.787879, 6.292929, -579.6985, 0.9647059, 0, 0.03137255, 1,
6.828283, 6.292929, -579.8505, 0.9647059, 0, 0.03137255, 1,
6.868687, 6.292929, -580.0107, 0.9647059, 0, 0.03137255, 1,
6.909091, 6.292929, -580.1792, 0.9647059, 0, 0.03137255, 1,
6.949495, 6.292929, -580.356, 0.9647059, 0, 0.03137255, 1,
6.989899, 6.292929, -580.5409, 0.9647059, 0, 0.03137255, 1,
7.030303, 6.292929, -580.7341, 0.9647059, 0, 0.03137255, 1,
7.070707, 6.292929, -580.9355, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.292929, -581.1452, 0.9647059, 0, 0.03137255, 1,
7.151515, 6.292929, -581.3632, 0.9647059, 0, 0.03137255, 1,
7.191919, 6.292929, -581.5894, 0.9647059, 0, 0.03137255, 1,
7.232323, 6.292929, -581.8237, 0.9647059, 0, 0.03137255, 1,
7.272727, 6.292929, -582.0664, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.292929, -582.3173, 0.9647059, 0, 0.03137255, 1,
7.353535, 6.292929, -582.5765, 0.9647059, 0, 0.03137255, 1,
7.393939, 6.292929, -582.8439, 0.9647059, 0, 0.03137255, 1,
7.434343, 6.292929, -583.1195, 0.9647059, 0, 0.03137255, 1,
7.474748, 6.292929, -583.4034, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.292929, -583.6956, 0.8588235, 0, 0.1372549, 1,
7.555555, 6.292929, -583.9959, 0.8588235, 0, 0.1372549, 1,
7.59596, 6.292929, -584.3046, 0.8588235, 0, 0.1372549, 1,
7.636364, 6.292929, -584.6215, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.292929, -584.9465, 0.8588235, 0, 0.1372549, 1,
7.717172, 6.292929, -585.2799, 0.8588235, 0, 0.1372549, 1,
7.757576, 6.292929, -585.6215, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.292929, -585.9714, 0.8588235, 0, 0.1372549, 1,
7.838384, 6.292929, -586.3295, 0.8588235, 0, 0.1372549, 1,
7.878788, 6.292929, -586.6958, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.292929, -587.0704, 0.8588235, 0, 0.1372549, 1,
7.959596, 6.292929, -587.4532, 0.8588235, 0, 0.1372549, 1,
8, 6.292929, -587.8443, 0.8588235, 0, 0.1372549, 1,
4, 6.343434, -590.1279, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.343434, -589.7176, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.343434, -589.3154, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.343434, -588.9213, 0.8588235, 0, 0.1372549, 1,
4.161616, 6.343434, -588.5354, 0.8588235, 0, 0.1372549, 1,
4.20202, 6.343434, -588.1575, 0.8588235, 0, 0.1372549, 1,
4.242424, 6.343434, -587.7878, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.343434, -587.4262, 0.8588235, 0, 0.1372549, 1,
4.323232, 6.343434, -587.0727, 0.8588235, 0, 0.1372549, 1,
4.363636, 6.343434, -586.7274, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.343434, -586.3901, 0.8588235, 0, 0.1372549, 1,
4.444445, 6.343434, -586.0609, 0.8588235, 0, 0.1372549, 1,
4.484848, 6.343434, -585.7399, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.343434, -585.4269, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.343434, -585.1221, 0.8588235, 0, 0.1372549, 1,
4.606061, 6.343434, -584.8254, 0.8588235, 0, 0.1372549, 1,
4.646465, 6.343434, -584.5369, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.343434, -584.2563, 0.8588235, 0, 0.1372549, 1,
4.727273, 6.343434, -583.984, 0.8588235, 0, 0.1372549, 1,
4.767677, 6.343434, -583.7198, 0.8588235, 0, 0.1372549, 1,
4.808081, 6.343434, -583.4636, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.343434, -583.2156, 0.9647059, 0, 0.03137255, 1,
4.888889, 6.343434, -582.9757, 0.9647059, 0, 0.03137255, 1,
4.929293, 6.343434, -582.7439, 0.9647059, 0, 0.03137255, 1,
4.969697, 6.343434, -582.5203, 0.9647059, 0, 0.03137255, 1,
5.010101, 6.343434, -582.3047, 0.9647059, 0, 0.03137255, 1,
5.050505, 6.343434, -582.0972, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.343434, -581.8979, 0.9647059, 0, 0.03137255, 1,
5.131313, 6.343434, -581.7067, 0.9647059, 0, 0.03137255, 1,
5.171717, 6.343434, -581.5236, 0.9647059, 0, 0.03137255, 1,
5.212121, 6.343434, -581.3486, 0.9647059, 0, 0.03137255, 1,
5.252525, 6.343434, -581.1817, 0.9647059, 0, 0.03137255, 1,
5.292929, 6.343434, -581.0229, 0.9647059, 0, 0.03137255, 1,
5.333333, 6.343434, -580.8723, 0.9647059, 0, 0.03137255, 1,
5.373737, 6.343434, -580.7298, 0.9647059, 0, 0.03137255, 1,
5.414141, 6.343434, -580.5953, 0.9647059, 0, 0.03137255, 1,
5.454545, 6.343434, -580.4691, 0.9647059, 0, 0.03137255, 1,
5.494949, 6.343434, -580.3508, 0.9647059, 0, 0.03137255, 1,
5.535354, 6.343434, -580.2408, 0.9647059, 0, 0.03137255, 1,
5.575758, 6.343434, -580.1388, 0.9647059, 0, 0.03137255, 1,
5.616162, 6.343434, -580.0449, 0.9647059, 0, 0.03137255, 1,
5.656566, 6.343434, -579.9592, 0.9647059, 0, 0.03137255, 1,
5.69697, 6.343434, -579.8816, 0.9647059, 0, 0.03137255, 1,
5.737374, 6.343434, -579.8121, 0.9647059, 0, 0.03137255, 1,
5.777778, 6.343434, -579.7507, 0.9647059, 0, 0.03137255, 1,
5.818182, 6.343434, -579.6974, 0.9647059, 0, 0.03137255, 1,
5.858586, 6.343434, -579.6522, 0.9647059, 0, 0.03137255, 1,
5.89899, 6.343434, -579.6152, 0.9647059, 0, 0.03137255, 1,
5.939394, 6.343434, -579.5862, 0.9647059, 0, 0.03137255, 1,
5.979798, 6.343434, -579.5654, 0.9647059, 0, 0.03137255, 1,
6.020202, 6.343434, -579.5527, 0.9647059, 0, 0.03137255, 1,
6.060606, 6.343434, -579.5481, 0.9647059, 0, 0.03137255, 1,
6.10101, 6.343434, -579.5516, 0.9647059, 0, 0.03137255, 1,
6.141414, 6.343434, -579.5632, 0.9647059, 0, 0.03137255, 1,
6.181818, 6.343434, -579.583, 0.9647059, 0, 0.03137255, 1,
6.222222, 6.343434, -579.6108, 0.9647059, 0, 0.03137255, 1,
6.262626, 6.343434, -579.6468, 0.9647059, 0, 0.03137255, 1,
6.30303, 6.343434, -579.6909, 0.9647059, 0, 0.03137255, 1,
6.343434, 6.343434, -579.7431, 0.9647059, 0, 0.03137255, 1,
6.383838, 6.343434, -579.8034, 0.9647059, 0, 0.03137255, 1,
6.424242, 6.343434, -579.8718, 0.9647059, 0, 0.03137255, 1,
6.464646, 6.343434, -579.9484, 0.9647059, 0, 0.03137255, 1,
6.505051, 6.343434, -580.033, 0.9647059, 0, 0.03137255, 1,
6.545455, 6.343434, -580.1258, 0.9647059, 0, 0.03137255, 1,
6.585859, 6.343434, -580.2267, 0.9647059, 0, 0.03137255, 1,
6.626263, 6.343434, -580.3357, 0.9647059, 0, 0.03137255, 1,
6.666667, 6.343434, -580.4528, 0.9647059, 0, 0.03137255, 1,
6.707071, 6.343434, -580.578, 0.9647059, 0, 0.03137255, 1,
6.747475, 6.343434, -580.7114, 0.9647059, 0, 0.03137255, 1,
6.787879, 6.343434, -580.8528, 0.9647059, 0, 0.03137255, 1,
6.828283, 6.343434, -581.0024, 0.9647059, 0, 0.03137255, 1,
6.868687, 6.343434, -581.16, 0.9647059, 0, 0.03137255, 1,
6.909091, 6.343434, -581.3258, 0.9647059, 0, 0.03137255, 1,
6.949495, 6.343434, -581.4998, 0.9647059, 0, 0.03137255, 1,
6.989899, 6.343434, -581.6818, 0.9647059, 0, 0.03137255, 1,
7.030303, 6.343434, -581.8719, 0.9647059, 0, 0.03137255, 1,
7.070707, 6.343434, -582.0701, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.343434, -582.2765, 0.9647059, 0, 0.03137255, 1,
7.151515, 6.343434, -582.491, 0.9647059, 0, 0.03137255, 1,
7.191919, 6.343434, -582.7136, 0.9647059, 0, 0.03137255, 1,
7.232323, 6.343434, -582.9443, 0.9647059, 0, 0.03137255, 1,
7.272727, 6.343434, -583.1831, 0.9647059, 0, 0.03137255, 1,
7.313131, 6.343434, -583.4301, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.343434, -583.6851, 0.8588235, 0, 0.1372549, 1,
7.393939, 6.343434, -583.9482, 0.8588235, 0, 0.1372549, 1,
7.434343, 6.343434, -584.2195, 0.8588235, 0, 0.1372549, 1,
7.474748, 6.343434, -584.4989, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.343434, -584.7864, 0.8588235, 0, 0.1372549, 1,
7.555555, 6.343434, -585.082, 0.8588235, 0, 0.1372549, 1,
7.59596, 6.343434, -585.3857, 0.8588235, 0, 0.1372549, 1,
7.636364, 6.343434, -585.6976, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.343434, -586.0176, 0.8588235, 0, 0.1372549, 1,
7.717172, 6.343434, -586.3456, 0.8588235, 0, 0.1372549, 1,
7.757576, 6.343434, -586.6818, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.343434, -587.0261, 0.8588235, 0, 0.1372549, 1,
7.838384, 6.343434, -587.3785, 0.8588235, 0, 0.1372549, 1,
7.878788, 6.343434, -587.7391, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.343434, -588.1077, 0.8588235, 0, 0.1372549, 1,
7.959596, 6.343434, -588.4844, 0.8588235, 0, 0.1372549, 1,
8, 6.343434, -588.8693, 0.8588235, 0, 0.1372549, 1,
4, 6.393939, -591.134, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.393939, -590.7302, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.393939, -590.3343, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.393939, -589.9464, 0.7568628, 0, 0.2392157, 1,
4.161616, 6.393939, -589.5665, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.393939, -589.1946, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.393939, -588.8307, 0.8588235, 0, 0.1372549, 1,
4.282828, 6.393939, -588.4749, 0.8588235, 0, 0.1372549, 1,
4.323232, 6.393939, -588.1269, 0.8588235, 0, 0.1372549, 1,
4.363636, 6.393939, -587.7869, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.393939, -587.455, 0.8588235, 0, 0.1372549, 1,
4.444445, 6.393939, -587.131, 0.8588235, 0, 0.1372549, 1,
4.484848, 6.393939, -586.815, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.393939, -586.507, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.393939, -586.207, 0.8588235, 0, 0.1372549, 1,
4.606061, 6.393939, -585.915, 0.8588235, 0, 0.1372549, 1,
4.646465, 6.393939, -585.6309, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.393939, -585.3549, 0.8588235, 0, 0.1372549, 1,
4.727273, 6.393939, -585.0868, 0.8588235, 0, 0.1372549, 1,
4.767677, 6.393939, -584.8267, 0.8588235, 0, 0.1372549, 1,
4.808081, 6.393939, -584.5746, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.393939, -584.3305, 0.8588235, 0, 0.1372549, 1,
4.888889, 6.393939, -584.0944, 0.8588235, 0, 0.1372549, 1,
4.929293, 6.393939, -583.8662, 0.8588235, 0, 0.1372549, 1,
4.969697, 6.393939, -583.6461, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.393939, -583.4339, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.393939, -583.2297, 0.9647059, 0, 0.03137255, 1,
5.090909, 6.393939, -583.0335, 0.9647059, 0, 0.03137255, 1,
5.131313, 6.393939, -582.8453, 0.9647059, 0, 0.03137255, 1,
5.171717, 6.393939, -582.6651, 0.9647059, 0, 0.03137255, 1,
5.212121, 6.393939, -582.4929, 0.9647059, 0, 0.03137255, 1,
5.252525, 6.393939, -582.3286, 0.9647059, 0, 0.03137255, 1,
5.292929, 6.393939, -582.1723, 0.9647059, 0, 0.03137255, 1,
5.333333, 6.393939, -582.024, 0.9647059, 0, 0.03137255, 1,
5.373737, 6.393939, -581.8837, 0.9647059, 0, 0.03137255, 1,
5.414141, 6.393939, -581.7515, 0.9647059, 0, 0.03137255, 1,
5.454545, 6.393939, -581.6271, 0.9647059, 0, 0.03137255, 1,
5.494949, 6.393939, -581.5108, 0.9647059, 0, 0.03137255, 1,
5.535354, 6.393939, -581.4025, 0.9647059, 0, 0.03137255, 1,
5.575758, 6.393939, -581.3021, 0.9647059, 0, 0.03137255, 1,
5.616162, 6.393939, -581.2097, 0.9647059, 0, 0.03137255, 1,
5.656566, 6.393939, -581.1253, 0.9647059, 0, 0.03137255, 1,
5.69697, 6.393939, -581.049, 0.9647059, 0, 0.03137255, 1,
5.737374, 6.393939, -580.9805, 0.9647059, 0, 0.03137255, 1,
5.777778, 6.393939, -580.92, 0.9647059, 0, 0.03137255, 1,
5.818182, 6.393939, -580.8676, 0.9647059, 0, 0.03137255, 1,
5.858586, 6.393939, -580.8232, 0.9647059, 0, 0.03137255, 1,
5.89899, 6.393939, -580.7867, 0.9647059, 0, 0.03137255, 1,
5.939394, 6.393939, -580.7582, 0.9647059, 0, 0.03137255, 1,
5.979798, 6.393939, -580.7377, 0.9647059, 0, 0.03137255, 1,
6.020202, 6.393939, -580.7252, 0.9647059, 0, 0.03137255, 1,
6.060606, 6.393939, -580.7207, 0.9647059, 0, 0.03137255, 1,
6.10101, 6.393939, -580.7241, 0.9647059, 0, 0.03137255, 1,
6.141414, 6.393939, -580.7356, 0.9647059, 0, 0.03137255, 1,
6.181818, 6.393939, -580.755, 0.9647059, 0, 0.03137255, 1,
6.222222, 6.393939, -580.7824, 0.9647059, 0, 0.03137255, 1,
6.262626, 6.393939, -580.8178, 0.9647059, 0, 0.03137255, 1,
6.30303, 6.393939, -580.8612, 0.9647059, 0, 0.03137255, 1,
6.343434, 6.393939, -580.9126, 0.9647059, 0, 0.03137255, 1,
6.383838, 6.393939, -580.972, 0.9647059, 0, 0.03137255, 1,
6.424242, 6.393939, -581.0393, 0.9647059, 0, 0.03137255, 1,
6.464646, 6.393939, -581.1147, 0.9647059, 0, 0.03137255, 1,
6.505051, 6.393939, -581.198, 0.9647059, 0, 0.03137255, 1,
6.545455, 6.393939, -581.2893, 0.9647059, 0, 0.03137255, 1,
6.585859, 6.393939, -581.3886, 0.9647059, 0, 0.03137255, 1,
6.626263, 6.393939, -581.4958, 0.9647059, 0, 0.03137255, 1,
6.666667, 6.393939, -581.6111, 0.9647059, 0, 0.03137255, 1,
6.707071, 6.393939, -581.7344, 0.9647059, 0, 0.03137255, 1,
6.747475, 6.393939, -581.8656, 0.9647059, 0, 0.03137255, 1,
6.787879, 6.393939, -582.0048, 0.9647059, 0, 0.03137255, 1,
6.828283, 6.393939, -582.152, 0.9647059, 0, 0.03137255, 1,
6.868687, 6.393939, -582.3073, 0.9647059, 0, 0.03137255, 1,
6.909091, 6.393939, -582.4705, 0.9647059, 0, 0.03137255, 1,
6.949495, 6.393939, -582.6416, 0.9647059, 0, 0.03137255, 1,
6.989899, 6.393939, -582.8208, 0.9647059, 0, 0.03137255, 1,
7.030303, 6.393939, -583.0079, 0.9647059, 0, 0.03137255, 1,
7.070707, 6.393939, -583.2031, 0.9647059, 0, 0.03137255, 1,
7.111111, 6.393939, -583.4062, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.393939, -583.6172, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.393939, -583.8364, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.393939, -584.0634, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.393939, -584.2985, 0.8588235, 0, 0.1372549, 1,
7.313131, 6.393939, -584.5415, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.393939, -584.7925, 0.8588235, 0, 0.1372549, 1,
7.393939, 6.393939, -585.0516, 0.8588235, 0, 0.1372549, 1,
7.434343, 6.393939, -585.3186, 0.8588235, 0, 0.1372549, 1,
7.474748, 6.393939, -585.5936, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.393939, -585.8766, 0.8588235, 0, 0.1372549, 1,
7.555555, 6.393939, -586.1675, 0.8588235, 0, 0.1372549, 1,
7.59596, 6.393939, -586.4665, 0.8588235, 0, 0.1372549, 1,
7.636364, 6.393939, -586.7734, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.393939, -587.0883, 0.8588235, 0, 0.1372549, 1,
7.717172, 6.393939, -587.4113, 0.8588235, 0, 0.1372549, 1,
7.757576, 6.393939, -587.7421, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.393939, -588.0811, 0.8588235, 0, 0.1372549, 1,
7.838384, 6.393939, -588.4279, 0.8588235, 0, 0.1372549, 1,
7.878788, 6.393939, -588.7828, 0.8588235, 0, 0.1372549, 1,
7.919192, 6.393939, -589.1456, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.393939, -589.5164, 0.7568628, 0, 0.2392157, 1,
8, 6.393939, -589.8952, 0.7568628, 0, 0.2392157, 1,
4, 6.444445, -592.1412, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.444445, -591.7437, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.444445, -591.354, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.444445, -590.9722, 0.7568628, 0, 0.2392157, 1,
4.161616, 6.444445, -590.5982, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.444445, -590.2321, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.444445, -589.8739, 0.7568628, 0, 0.2392157, 1,
4.282828, 6.444445, -589.5236, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.444445, -589.181, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.444445, -588.8464, 0.8588235, 0, 0.1372549, 1,
4.40404, 6.444445, -588.5197, 0.8588235, 0, 0.1372549, 1,
4.444445, 6.444445, -588.2007, 0.8588235, 0, 0.1372549, 1,
4.484848, 6.444445, -587.8896, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.444445, -587.5865, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.444445, -587.2911, 0.8588235, 0, 0.1372549, 1,
4.606061, 6.444445, -587.0037, 0.8588235, 0, 0.1372549, 1,
4.646465, 6.444445, -586.7241, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.444445, -586.4523, 0.8588235, 0, 0.1372549, 1,
4.727273, 6.444445, -586.1884, 0.8588235, 0, 0.1372549, 1,
4.767677, 6.444445, -585.9324, 0.8588235, 0, 0.1372549, 1,
4.808081, 6.444445, -585.6842, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.444445, -585.4439, 0.8588235, 0, 0.1372549, 1,
4.888889, 6.444445, -585.2115, 0.8588235, 0, 0.1372549, 1,
4.929293, 6.444445, -584.9869, 0.8588235, 0, 0.1372549, 1,
4.969697, 6.444445, -584.7702, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.444445, -584.5613, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.444445, -584.3604, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.444445, -584.1672, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.444445, -583.9819, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.444445, -583.8045, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.444445, -583.6349, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.444445, -583.4733, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.444445, -583.3195, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.444445, -583.1735, 0.9647059, 0, 0.03137255, 1,
5.373737, 6.444445, -583.0354, 0.9647059, 0, 0.03137255, 1,
5.414141, 6.444445, -582.9052, 0.9647059, 0, 0.03137255, 1,
5.454545, 6.444445, -582.7828, 0.9647059, 0, 0.03137255, 1,
5.494949, 6.444445, -582.6683, 0.9647059, 0, 0.03137255, 1,
5.535354, 6.444445, -582.5616, 0.9647059, 0, 0.03137255, 1,
5.575758, 6.444445, -582.4628, 0.9647059, 0, 0.03137255, 1,
5.616162, 6.444445, -582.3719, 0.9647059, 0, 0.03137255, 1,
5.656566, 6.444445, -582.2888, 0.9647059, 0, 0.03137255, 1,
5.69697, 6.444445, -582.2136, 0.9647059, 0, 0.03137255, 1,
5.737374, 6.444445, -582.1462, 0.9647059, 0, 0.03137255, 1,
5.777778, 6.444445, -582.0867, 0.9647059, 0, 0.03137255, 1,
5.818182, 6.444445, -582.0351, 0.9647059, 0, 0.03137255, 1,
5.858586, 6.444445, -581.9913, 0.9647059, 0, 0.03137255, 1,
5.89899, 6.444445, -581.9554, 0.9647059, 0, 0.03137255, 1,
5.939394, 6.444445, -581.9274, 0.9647059, 0, 0.03137255, 1,
5.979798, 6.444445, -581.9072, 0.9647059, 0, 0.03137255, 1,
6.020202, 6.444445, -581.8949, 0.9647059, 0, 0.03137255, 1,
6.060606, 6.444445, -581.8904, 0.9647059, 0, 0.03137255, 1,
6.10101, 6.444445, -581.8939, 0.9647059, 0, 0.03137255, 1,
6.141414, 6.444445, -581.9052, 0.9647059, 0, 0.03137255, 1,
6.181818, 6.444445, -581.9243, 0.9647059, 0, 0.03137255, 1,
6.222222, 6.444445, -581.9512, 0.9647059, 0, 0.03137255, 1,
6.262626, 6.444445, -581.9861, 0.9647059, 0, 0.03137255, 1,
6.30303, 6.444445, -582.0288, 0.9647059, 0, 0.03137255, 1,
6.343434, 6.444445, -582.0794, 0.9647059, 0, 0.03137255, 1,
6.383838, 6.444445, -582.1378, 0.9647059, 0, 0.03137255, 1,
6.424242, 6.444445, -582.2041, 0.9647059, 0, 0.03137255, 1,
6.464646, 6.444445, -582.2783, 0.9647059, 0, 0.03137255, 1,
6.505051, 6.444445, -582.3603, 0.9647059, 0, 0.03137255, 1,
6.545455, 6.444445, -582.4502, 0.9647059, 0, 0.03137255, 1,
6.585859, 6.444445, -582.5479, 0.9647059, 0, 0.03137255, 1,
6.626263, 6.444445, -582.6536, 0.9647059, 0, 0.03137255, 1,
6.666667, 6.444445, -582.767, 0.9647059, 0, 0.03137255, 1,
6.707071, 6.444445, -582.8884, 0.9647059, 0, 0.03137255, 1,
6.747475, 6.444445, -583.0175, 0.9647059, 0, 0.03137255, 1,
6.787879, 6.444445, -583.1546, 0.9647059, 0, 0.03137255, 1,
6.828283, 6.444445, -583.2995, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.444445, -583.4523, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.444445, -583.6129, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.444445, -583.7814, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.444445, -583.9578, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.444445, -584.142, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.444445, -584.334, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.444445, -584.534, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.444445, -584.7418, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.444445, -584.9575, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.444445, -585.181, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.444445, -585.4124, 0.8588235, 0, 0.1372549, 1,
7.313131, 6.444445, -585.6517, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.444445, -585.8988, 0.8588235, 0, 0.1372549, 1,
7.393939, 6.444445, -586.1537, 0.8588235, 0, 0.1372549, 1,
7.434343, 6.444445, -586.4166, 0.8588235, 0, 0.1372549, 1,
7.474748, 6.444445, -586.6873, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.444445, -586.9659, 0.8588235, 0, 0.1372549, 1,
7.555555, 6.444445, -587.2523, 0.8588235, 0, 0.1372549, 1,
7.59596, 6.444445, -587.5466, 0.8588235, 0, 0.1372549, 1,
7.636364, 6.444445, -587.8487, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.444445, -588.1587, 0.8588235, 0, 0.1372549, 1,
7.717172, 6.444445, -588.4766, 0.8588235, 0, 0.1372549, 1,
7.757576, 6.444445, -588.8023, 0.8588235, 0, 0.1372549, 1,
7.79798, 6.444445, -589.1359, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.444445, -589.4774, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.444445, -589.8267, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.444445, -590.1838, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.444445, -590.5489, 0.7568628, 0, 0.2392157, 1,
8, 6.444445, -590.9218, 0.7568628, 0, 0.2392157, 1,
4, 6.494949, -593.1493, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.494949, -592.7579, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.494949, -592.3743, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.494949, -591.9984, 0.7568628, 0, 0.2392157, 1,
4.161616, 6.494949, -591.6302, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.494949, -591.2698, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.494949, -590.9171, 0.7568628, 0, 0.2392157, 1,
4.282828, 6.494949, -590.5722, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.494949, -590.235, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.494949, -589.9055, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.494949, -589.5838, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.494949, -589.2698, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.494949, -588.9636, 0.8588235, 0, 0.1372549, 1,
4.525252, 6.494949, -588.6651, 0.8588235, 0, 0.1372549, 1,
4.565657, 6.494949, -588.3743, 0.8588235, 0, 0.1372549, 1,
4.606061, 6.494949, -588.0913, 0.8588235, 0, 0.1372549, 1,
4.646465, 6.494949, -587.816, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.494949, -587.5485, 0.8588235, 0, 0.1372549, 1,
4.727273, 6.494949, -587.2887, 0.8588235, 0, 0.1372549, 1,
4.767677, 6.494949, -587.0367, 0.8588235, 0, 0.1372549, 1,
4.808081, 6.494949, -586.7924, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.494949, -586.5557, 0.8588235, 0, 0.1372549, 1,
4.888889, 6.494949, -586.3269, 0.8588235, 0, 0.1372549, 1,
4.929293, 6.494949, -586.1058, 0.8588235, 0, 0.1372549, 1,
4.969697, 6.494949, -585.8925, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.494949, -585.6868, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.494949, -585.489, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.494949, -585.2988, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.494949, -585.1164, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.494949, -584.9418, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.494949, -584.7748, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.494949, -584.6157, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.494949, -584.4642, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.494949, -584.3205, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.494949, -584.1845, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.494949, -584.0563, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.494949, -583.9359, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.494949, -583.8231, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.494949, -583.7181, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.494949, -583.6208, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.494949, -583.5313, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.494949, -583.4495, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.494949, -583.3755, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.494949, -583.3091, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.494949, -583.2506, 0.9647059, 0, 0.03137255, 1,
5.818182, 6.494949, -583.1998, 0.9647059, 0, 0.03137255, 1,
5.858586, 6.494949, -583.1567, 0.9647059, 0, 0.03137255, 1,
5.89899, 6.494949, -583.1213, 0.9647059, 0, 0.03137255, 1,
5.939394, 6.494949, -583.0938, 0.9647059, 0, 0.03137255, 1,
5.979798, 6.494949, -583.0739, 0.9647059, 0, 0.03137255, 1,
6.020202, 6.494949, -583.0617, 0.9647059, 0, 0.03137255, 1,
6.060606, 6.494949, -583.0574, 0.9647059, 0, 0.03137255, 1,
6.10101, 6.494949, -583.0607, 0.9647059, 0, 0.03137255, 1,
6.141414, 6.494949, -583.0718, 0.9647059, 0, 0.03137255, 1,
6.181818, 6.494949, -583.0906, 0.9647059, 0, 0.03137255, 1,
6.222222, 6.494949, -583.1172, 0.9647059, 0, 0.03137255, 1,
6.262626, 6.494949, -583.1515, 0.9647059, 0, 0.03137255, 1,
6.30303, 6.494949, -583.1935, 0.9647059, 0, 0.03137255, 1,
6.343434, 6.494949, -583.2433, 0.9647059, 0, 0.03137255, 1,
6.383838, 6.494949, -583.3009, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.494949, -583.3661, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.494949, -583.4391, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.494949, -583.5199, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.494949, -583.6084, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.494949, -583.7047, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.494949, -583.8086, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.494949, -583.9203, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.494949, -584.0398, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.494949, -584.1669, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.494949, -584.3019, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.494949, -584.4446, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.494949, -584.595, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.494949, -584.7531, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.494949, -584.919, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.494949, -585.0927, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.494949, -585.274, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.494949, -585.4631, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.494949, -585.66, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.494949, -585.8646, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.494949, -586.0768, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.494949, -586.2969, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.494949, -586.5247, 0.8588235, 0, 0.1372549, 1,
7.313131, 6.494949, -586.7603, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.494949, -587.0036, 0.8588235, 0, 0.1372549, 1,
7.393939, 6.494949, -587.2546, 0.8588235, 0, 0.1372549, 1,
7.434343, 6.494949, -587.5134, 0.8588235, 0, 0.1372549, 1,
7.474748, 6.494949, -587.7798, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.494949, -588.0541, 0.8588235, 0, 0.1372549, 1,
7.555555, 6.494949, -588.3361, 0.8588235, 0, 0.1372549, 1,
7.59596, 6.494949, -588.6258, 0.8588235, 0, 0.1372549, 1,
7.636364, 6.494949, -588.9233, 0.8588235, 0, 0.1372549, 1,
7.676768, 6.494949, -589.2285, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.494949, -589.5414, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.494949, -589.8621, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.494949, -590.1906, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.494949, -590.5267, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.494949, -590.8706, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.494949, -591.2222, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.494949, -591.5816, 0.7568628, 0, 0.2392157, 1,
8, 6.494949, -591.9487, 0.7568628, 0, 0.2392157, 1,
4, 6.545455, -594.158, 0.7568628, 0, 0.2392157, 1,
4.040404, 6.545455, -593.7726, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.545455, -593.395, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.545455, -593.0248, 0.7568628, 0, 0.2392157, 1,
4.161616, 6.545455, -592.6623, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.545455, -592.3074, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.545455, -591.9602, 0.7568628, 0, 0.2392157, 1,
4.282828, 6.545455, -591.6205, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.545455, -591.2885, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.545455, -590.9642, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.545455, -590.6474, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.545455, -590.3382, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.545455, -590.0367, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.545455, -589.7428, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.545455, -589.4565, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.545455, -589.1779, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.545455, -588.9068, 0.8588235, 0, 0.1372549, 1,
4.686869, 6.545455, -588.6434, 0.8588235, 0, 0.1372549, 1,
4.727273, 6.545455, -588.3876, 0.8588235, 0, 0.1372549, 1,
4.767677, 6.545455, -588.1393, 0.8588235, 0, 0.1372549, 1,
4.808081, 6.545455, -587.8988, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.545455, -587.6658, 0.8588235, 0, 0.1372549, 1,
4.888889, 6.545455, -587.4406, 0.8588235, 0, 0.1372549, 1,
4.929293, 6.545455, -587.2228, 0.8588235, 0, 0.1372549, 1,
4.969697, 6.545455, -587.0128, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.545455, -586.8103, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.545455, -586.6155, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.545455, -586.4282, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.545455, -586.2487, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.545455, -586.0767, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.545455, -585.9123, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.545455, -585.7556, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.545455, -585.6064, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.545455, -585.465, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.545455, -585.3311, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.545455, -585.2048, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.545455, -585.0862, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.545455, -584.9752, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.545455, -584.8718, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.545455, -584.7761, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.545455, -584.6879, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.545455, -584.6074, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.545455, -584.5345, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.545455, -584.4692, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.545455, -584.4115, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.545455, -584.3615, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.545455, -584.319, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.545455, -584.2842, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.545455, -584.2571, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.545455, -584.2375, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.545455, -584.2255, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.545455, -584.2213, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.545455, -584.2245, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.545455, -584.2355, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.545455, -584.254, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.545455, -584.2802, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.545455, -584.314, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.545455, -584.3553, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.545455, -584.4044, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.545455, -584.4611, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.545455, -584.5253, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.545455, -584.5972, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.545455, -584.6767, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.545455, -584.7639, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.545455, -584.8586, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.545455, -584.9609, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.545455, -585.0709, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.545455, -585.1885, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.545455, -585.3138, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.545455, -585.4467, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.545455, -585.5871, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.545455, -585.7352, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.545455, -585.8909, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.545455, -586.0543, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.545455, -586.2252, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.545455, -586.4038, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.545455, -586.59, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.545455, -586.7838, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.545455, -586.9853, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.545455, -587.1943, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.545455, -587.411, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.545455, -587.6353, 0.8588235, 0, 0.1372549, 1,
7.313131, 6.545455, -587.8672, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.545455, -588.1068, 0.8588235, 0, 0.1372549, 1,
7.393939, 6.545455, -588.3539, 0.8588235, 0, 0.1372549, 1,
7.434343, 6.545455, -588.6088, 0.8588235, 0, 0.1372549, 1,
7.474748, 6.545455, -588.8712, 0.8588235, 0, 0.1372549, 1,
7.515152, 6.545455, -589.1412, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.545455, -589.4188, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.545455, -589.7041, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.545455, -589.997, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.545455, -590.2975, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.545455, -590.6057, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.545455, -590.9214, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.545455, -591.2448, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.545455, -591.5757, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.545455, -591.9144, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.545455, -592.2606, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.545455, -592.6145, 0.7568628, 0, 0.2392157, 1,
8, 6.545455, -592.976, 0.7568628, 0, 0.2392157, 1,
4, 6.59596, -595.1672, 0.654902, 0, 0.3411765, 1,
4.040404, 6.59596, -594.7878, 0.7568628, 0, 0.2392157, 1,
4.080808, 6.59596, -594.4158, 0.7568628, 0, 0.2392157, 1,
4.121212, 6.59596, -594.0513, 0.7568628, 0, 0.2392157, 1,
4.161616, 6.59596, -593.6943, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.59596, -593.3449, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.59596, -593.0029, 0.7568628, 0, 0.2392157, 1,
4.282828, 6.59596, -592.6685, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.59596, -592.3415, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.59596, -592.0221, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.59596, -591.7101, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.59596, -591.4057, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.59596, -591.1088, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.59596, -590.8193, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.59596, -590.5374, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.59596, -590.263, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.59596, -589.9961, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.59596, -589.7367, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.59596, -589.4848, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.59596, -589.2404, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.59596, -589.0035, 0.8588235, 0, 0.1372549, 1,
4.848485, 6.59596, -588.7741, 0.8588235, 0, 0.1372549, 1,
4.888889, 6.59596, -588.5522, 0.8588235, 0, 0.1372549, 1,
4.929293, 6.59596, -588.3378, 0.8588235, 0, 0.1372549, 1,
4.969697, 6.59596, -588.131, 0.8588235, 0, 0.1372549, 1,
5.010101, 6.59596, -587.9316, 0.8588235, 0, 0.1372549, 1,
5.050505, 6.59596, -587.7397, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.59596, -587.5554, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.59596, -587.3785, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.59596, -587.2092, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.59596, -587.0473, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.59596, -586.8929, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.59596, -586.7461, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.59596, -586.6068, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.59596, -586.475, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.59596, -586.3506, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.59596, -586.2338, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.59596, -586.1245, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.59596, -586.0227, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.59596, -585.9283, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.59596, -585.8416, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.59596, -585.7623, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.59596, -585.6905, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.59596, -585.6262, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.59596, -585.5694, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.59596, -585.5201, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.59596, -585.4783, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.59596, -585.4441, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.59596, -585.4173, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.59596, -585.3981, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.59596, -585.3863, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.59596, -585.382, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.59596, -585.3853, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.59596, -585.3961, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.59596, -585.4143, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.59596, -585.4401, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.59596, -585.4733, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.59596, -585.5141, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.59596, -585.5624, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.59596, -585.6182, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.59596, -585.6815, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.59596, -585.7523, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.59596, -585.8306, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.59596, -585.9163, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.59596, -586.0096, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.59596, -586.1105, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.59596, -586.2188, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.59596, -586.3346, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.59596, -586.4579, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.59596, -586.5887, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.59596, -586.7271, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.59596, -586.8729, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.59596, -587.0262, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.59596, -587.1871, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.59596, -587.3555, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.59596, -587.5313, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.59596, -587.7147, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.59596, -587.9055, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.59596, -588.1039, 0.8588235, 0, 0.1372549, 1,
7.191919, 6.59596, -588.3098, 0.8588235, 0, 0.1372549, 1,
7.232323, 6.59596, -588.5231, 0.8588235, 0, 0.1372549, 1,
7.272727, 6.59596, -588.744, 0.8588235, 0, 0.1372549, 1,
7.313131, 6.59596, -588.9724, 0.8588235, 0, 0.1372549, 1,
7.353535, 6.59596, -589.2083, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.59596, -589.4517, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.59596, -589.7026, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.59596, -589.961, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.59596, -590.2269, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.59596, -590.5004, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.59596, -590.7812, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.59596, -591.0697, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.59596, -591.3656, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.59596, -591.6691, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.59596, -591.98, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.59596, -592.2984, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.59596, -592.6244, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.59596, -592.9578, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.59596, -593.2988, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.59596, -593.6472, 0.7568628, 0, 0.2392157, 1,
8, 6.59596, -594.0032, 0.7568628, 0, 0.2392157, 1,
4, 6.646465, -596.1767, 0.654902, 0, 0.3411765, 1,
4.040404, 6.646465, -595.803, 0.654902, 0, 0.3411765, 1,
4.080808, 6.646465, -595.4366, 0.654902, 0, 0.3411765, 1,
4.121212, 6.646465, -595.0777, 0.654902, 0, 0.3411765, 1,
4.161616, 6.646465, -594.7261, 0.7568628, 0, 0.2392157, 1,
4.20202, 6.646465, -594.382, 0.7568628, 0, 0.2392157, 1,
4.242424, 6.646465, -594.0452, 0.7568628, 0, 0.2392157, 1,
4.282828, 6.646465, -593.7158, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.646465, -593.3938, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.646465, -593.0792, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.646465, -592.772, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.646465, -592.4721, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.646465, -592.1797, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.646465, -591.8947, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.646465, -591.617, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.646465, -591.3467, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.646465, -591.0839, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.646465, -590.8284, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.646465, -590.5803, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.646465, -590.3396, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.646465, -590.1063, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.646465, -589.8804, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.646465, -589.6619, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.646465, -589.4507, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.646465, -589.2469, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.646465, -589.0506, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.646465, -588.8616, 0.8588235, 0, 0.1372549, 1,
5.090909, 6.646465, -588.6801, 0.8588235, 0, 0.1372549, 1,
5.131313, 6.646465, -588.5059, 0.8588235, 0, 0.1372549, 1,
5.171717, 6.646465, -588.3391, 0.8588235, 0, 0.1372549, 1,
5.212121, 6.646465, -588.1797, 0.8588235, 0, 0.1372549, 1,
5.252525, 6.646465, -588.0277, 0.8588235, 0, 0.1372549, 1,
5.292929, 6.646465, -587.8831, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.646465, -587.7458, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.646465, -587.616, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.646465, -587.4936, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.646465, -587.3785, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.646465, -587.2709, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.646465, -587.1706, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.646465, -587.0777, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.646465, -586.9922, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.646465, -586.9141, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.646465, -586.8434, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.646465, -586.7801, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.646465, -586.7242, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.646465, -586.6757, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.646465, -586.6345, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.646465, -586.6008, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.646465, -586.5744, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.646465, -586.5554, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.646465, -586.5438, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.646465, -586.5397, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.646465, -586.5428, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.646465, -586.5535, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.646465, -586.5714, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.646465, -586.5968, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.646465, -586.6296, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.646465, -586.6697, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.646465, -586.7173, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.646465, -586.7722, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.646465, -586.8345, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.646465, -586.9042, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.646465, -586.9814, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.646465, -587.0659, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.646465, -587.1578, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.646465, -587.2571, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.646465, -587.3637, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.646465, -587.4778, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.646465, -587.5992, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.646465, -587.7281, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.646465, -587.8643, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.646465, -588.0079, 0.8588235, 0, 0.1372549, 1,
6.909091, 6.646465, -588.159, 0.8588235, 0, 0.1372549, 1,
6.949495, 6.646465, -588.3174, 0.8588235, 0, 0.1372549, 1,
6.989899, 6.646465, -588.4832, 0.8588235, 0, 0.1372549, 1,
7.030303, 6.646465, -588.6564, 0.8588235, 0, 0.1372549, 1,
7.070707, 6.646465, -588.837, 0.8588235, 0, 0.1372549, 1,
7.111111, 6.646465, -589.025, 0.8588235, 0, 0.1372549, 1,
7.151515, 6.646465, -589.2203, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.646465, -589.4231, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.646465, -589.6332, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.646465, -589.8508, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.646465, -590.0757, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.646465, -590.308, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.646465, -590.5477, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.646465, -590.7948, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.646465, -591.0493, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.646465, -591.3112, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.646465, -591.5804, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.646465, -591.8571, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.646465, -592.1412, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.646465, -592.4326, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.646465, -592.7315, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.646465, -593.0377, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.646465, -593.3513, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.646465, -593.6724, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.646465, -594.0007, 0.7568628, 0, 0.2392157, 1,
7.919192, 6.646465, -594.3365, 0.7568628, 0, 0.2392157, 1,
7.959596, 6.646465, -594.6797, 0.7568628, 0, 0.2392157, 1,
8, 6.646465, -595.0303, 0.654902, 0, 0.3411765, 1,
4, 6.69697, -597.1863, 0.654902, 0, 0.3411765, 1,
4.040404, 6.69697, -596.8181, 0.654902, 0, 0.3411765, 1,
4.080808, 6.69697, -596.4573, 0.654902, 0, 0.3411765, 1,
4.121212, 6.69697, -596.1038, 0.654902, 0, 0.3411765, 1,
4.161616, 6.69697, -595.7574, 0.654902, 0, 0.3411765, 1,
4.20202, 6.69697, -595.4185, 0.654902, 0, 0.3411765, 1,
4.242424, 6.69697, -595.0867, 0.654902, 0, 0.3411765, 1,
4.282828, 6.69697, -594.7623, 0.7568628, 0, 0.2392157, 1,
4.323232, 6.69697, -594.4451, 0.7568628, 0, 0.2392157, 1,
4.363636, 6.69697, -594.1353, 0.7568628, 0, 0.2392157, 1,
4.40404, 6.69697, -593.8326, 0.7568628, 0, 0.2392157, 1,
4.444445, 6.69697, -593.5374, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.69697, -593.2493, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.69697, -592.9686, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.69697, -592.6951, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.69697, -592.4289, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.69697, -592.1699, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.69697, -591.9183, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.69697, -591.674, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.69697, -591.4368, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.69697, -591.207, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.69697, -590.9846, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.69697, -590.7693, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.69697, -590.5613, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.69697, -590.3607, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.69697, -590.1672, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.69697, -589.9811, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.69697, -589.8023, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.69697, -589.6307, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.69697, -589.4664, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.69697, -589.3094, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.69697, -589.1597, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.69697, -589.0173, 0.8588235, 0, 0.1372549, 1,
5.333333, 6.69697, -588.8821, 0.8588235, 0, 0.1372549, 1,
5.373737, 6.69697, -588.7542, 0.8588235, 0, 0.1372549, 1,
5.414141, 6.69697, -588.6336, 0.8588235, 0, 0.1372549, 1,
5.454545, 6.69697, -588.5203, 0.8588235, 0, 0.1372549, 1,
5.494949, 6.69697, -588.4142, 0.8588235, 0, 0.1372549, 1,
5.535354, 6.69697, -588.3155, 0.8588235, 0, 0.1372549, 1,
5.575758, 6.69697, -588.224, 0.8588235, 0, 0.1372549, 1,
5.616162, 6.69697, -588.1398, 0.8588235, 0, 0.1372549, 1,
5.656566, 6.69697, -588.0629, 0.8588235, 0, 0.1372549, 1,
5.69697, 6.69697, -587.9932, 0.8588235, 0, 0.1372549, 1,
5.737374, 6.69697, -587.9308, 0.8588235, 0, 0.1372549, 1,
5.777778, 6.69697, -587.8758, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.69697, -587.8279, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.69697, -587.7874, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.69697, -587.7542, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.69697, -587.7282, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.69697, -587.7095, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.69697, -587.6981, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.69697, -587.694, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.69697, -587.6971, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.69697, -587.7076, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.69697, -587.7253, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.69697, -587.7503, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.69697, -587.7826, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.69697, -587.8221, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.69697, -587.869, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.69697, -587.9231, 0.8588235, 0, 0.1372549, 1,
6.424242, 6.69697, -587.9845, 0.8588235, 0, 0.1372549, 1,
6.464646, 6.69697, -588.0532, 0.8588235, 0, 0.1372549, 1,
6.505051, 6.69697, -588.1291, 0.8588235, 0, 0.1372549, 1,
6.545455, 6.69697, -588.2123, 0.8588235, 0, 0.1372549, 1,
6.585859, 6.69697, -588.3029, 0.8588235, 0, 0.1372549, 1,
6.626263, 6.69697, -588.4006, 0.8588235, 0, 0.1372549, 1,
6.666667, 6.69697, -588.5057, 0.8588235, 0, 0.1372549, 1,
6.707071, 6.69697, -588.618, 0.8588235, 0, 0.1372549, 1,
6.747475, 6.69697, -588.7377, 0.8588235, 0, 0.1372549, 1,
6.787879, 6.69697, -588.8646, 0.8588235, 0, 0.1372549, 1,
6.828283, 6.69697, -588.9988, 0.8588235, 0, 0.1372549, 1,
6.868687, 6.69697, -589.1403, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.69697, -589.289, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.69697, -589.4451, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.69697, -589.6083, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.69697, -589.7789, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.69697, -589.9568, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.69697, -590.142, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.69697, -590.3344, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.69697, -590.5341, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.69697, -590.7411, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.69697, -590.9554, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.69697, -591.1769, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.69697, -591.4058, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.69697, -591.6418, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.69697, -591.8853, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.69697, -592.1359, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.69697, -592.3939, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.69697, -592.6591, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.69697, -592.9316, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.69697, -593.2114, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.69697, -593.4985, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.69697, -593.7928, 0.7568628, 0, 0.2392157, 1,
7.757576, 6.69697, -594.0944, 0.7568628, 0, 0.2392157, 1,
7.79798, 6.69697, -594.4033, 0.7568628, 0, 0.2392157, 1,
7.838384, 6.69697, -594.7195, 0.7568628, 0, 0.2392157, 1,
7.878788, 6.69697, -595.043, 0.654902, 0, 0.3411765, 1,
7.919192, 6.69697, -595.3737, 0.654902, 0, 0.3411765, 1,
7.959596, 6.69697, -595.7118, 0.654902, 0, 0.3411765, 1,
8, 6.69697, -596.0571, 0.654902, 0, 0.3411765, 1,
4, 6.747475, -598.1957, 0.654902, 0, 0.3411765, 1,
4.040404, 6.747475, -597.8331, 0.654902, 0, 0.3411765, 1,
4.080808, 6.747475, -597.4777, 0.654902, 0, 0.3411765, 1,
4.121212, 6.747475, -597.1293, 0.654902, 0, 0.3411765, 1,
4.161616, 6.747475, -596.7883, 0.654902, 0, 0.3411765, 1,
4.20202, 6.747475, -596.4543, 0.654902, 0, 0.3411765, 1,
4.242424, 6.747475, -596.1275, 0.654902, 0, 0.3411765, 1,
4.282828, 6.747475, -595.8079, 0.654902, 0, 0.3411765, 1,
4.323232, 6.747475, -595.4955, 0.654902, 0, 0.3411765, 1,
4.363636, 6.747475, -595.1902, 0.654902, 0, 0.3411765, 1,
4.40404, 6.747475, -594.8922, 0.654902, 0, 0.3411765, 1,
4.444445, 6.747475, -594.6013, 0.7568628, 0, 0.2392157, 1,
4.484848, 6.747475, -594.3175, 0.7568628, 0, 0.2392157, 1,
4.525252, 6.747475, -594.0409, 0.7568628, 0, 0.2392157, 1,
4.565657, 6.747475, -593.7715, 0.7568628, 0, 0.2392157, 1,
4.606061, 6.747475, -593.5093, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.747475, -593.2542, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.747475, -593.0063, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.747475, -592.7656, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.747475, -592.532, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.747475, -592.3057, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.747475, -592.0865, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.747475, -591.8745, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.747475, -591.6696, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.747475, -591.4719, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.747475, -591.2814, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.747475, -591.098, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.747475, -590.9219, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.747475, -590.7529, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.747475, -590.5911, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.747475, -590.4364, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.747475, -590.2889, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.747475, -590.1486, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.747475, -590.0154, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.747475, -589.8895, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.747475, -589.7706, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.747475, -589.659, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.747475, -589.5546, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.747475, -589.4573, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.747475, -589.3671, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.747475, -589.2842, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.747475, -589.2084, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.747475, -589.1398, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.747475, -589.0784, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.747475, -589.0241, 0.8588235, 0, 0.1372549, 1,
5.818182, 6.747475, -588.977, 0.8588235, 0, 0.1372549, 1,
5.858586, 6.747475, -588.9371, 0.8588235, 0, 0.1372549, 1,
5.89899, 6.747475, -588.9044, 0.8588235, 0, 0.1372549, 1,
5.939394, 6.747475, -588.8788, 0.8588235, 0, 0.1372549, 1,
5.979798, 6.747475, -588.8604, 0.8588235, 0, 0.1372549, 1,
6.020202, 6.747475, -588.8491, 0.8588235, 0, 0.1372549, 1,
6.060606, 6.747475, -588.845, 0.8588235, 0, 0.1372549, 1,
6.10101, 6.747475, -588.8481, 0.8588235, 0, 0.1372549, 1,
6.141414, 6.747475, -588.8585, 0.8588235, 0, 0.1372549, 1,
6.181818, 6.747475, -588.8759, 0.8588235, 0, 0.1372549, 1,
6.222222, 6.747475, -588.9005, 0.8588235, 0, 0.1372549, 1,
6.262626, 6.747475, -588.9323, 0.8588235, 0, 0.1372549, 1,
6.30303, 6.747475, -588.9713, 0.8588235, 0, 0.1372549, 1,
6.343434, 6.747475, -589.0174, 0.8588235, 0, 0.1372549, 1,
6.383838, 6.747475, -589.0707, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.747475, -589.1312, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.747475, -589.1989, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.747475, -589.2736, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.747475, -589.3557, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.747475, -589.4448, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.747475, -589.5411, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.747475, -589.6447, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.747475, -589.7553, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.747475, -589.8732, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.747475, -589.9982, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.747475, -590.1304, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.747475, -590.2697, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.747475, -590.4163, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.747475, -590.5699, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.747475, -590.7308, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.747475, -590.8989, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.747475, -591.0741, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.747475, -591.2565, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.747475, -591.446, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.747475, -591.6428, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.747475, -591.8467, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.747475, -592.0577, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.747475, -592.276, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.747475, -592.5014, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.747475, -592.734, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.747475, -592.9738, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.747475, -593.2207, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.747475, -593.4748, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.747475, -593.7361, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.747475, -594.0045, 0.7568628, 0, 0.2392157, 1,
7.636364, 6.747475, -594.2802, 0.7568628, 0, 0.2392157, 1,
7.676768, 6.747475, -594.5629, 0.7568628, 0, 0.2392157, 1,
7.717172, 6.747475, -594.8529, 0.654902, 0, 0.3411765, 1,
7.757576, 6.747475, -595.15, 0.654902, 0, 0.3411765, 1,
7.79798, 6.747475, -595.4543, 0.654902, 0, 0.3411765, 1,
7.838384, 6.747475, -595.7658, 0.654902, 0, 0.3411765, 1,
7.878788, 6.747475, -596.0844, 0.654902, 0, 0.3411765, 1,
7.919192, 6.747475, -596.4102, 0.654902, 0, 0.3411765, 1,
7.959596, 6.747475, -596.7432, 0.654902, 0, 0.3411765, 1,
8, 6.747475, -597.0834, 0.654902, 0, 0.3411765, 1,
4, 6.79798, -599.205, 0.654902, 0, 0.3411765, 1,
4.040404, 6.79798, -598.8477, 0.654902, 0, 0.3411765, 1,
4.080808, 6.79798, -598.4975, 0.654902, 0, 0.3411765, 1,
4.121212, 6.79798, -598.1544, 0.654902, 0, 0.3411765, 1,
4.161616, 6.79798, -597.8183, 0.654902, 0, 0.3411765, 1,
4.20202, 6.79798, -597.4893, 0.654902, 0, 0.3411765, 1,
4.242424, 6.79798, -597.1674, 0.654902, 0, 0.3411765, 1,
4.282828, 6.79798, -596.8525, 0.654902, 0, 0.3411765, 1,
4.323232, 6.79798, -596.5447, 0.654902, 0, 0.3411765, 1,
4.363636, 6.79798, -596.244, 0.654902, 0, 0.3411765, 1,
4.40404, 6.79798, -595.9503, 0.654902, 0, 0.3411765, 1,
4.444445, 6.79798, -595.6636, 0.654902, 0, 0.3411765, 1,
4.484848, 6.79798, -595.3841, 0.654902, 0, 0.3411765, 1,
4.525252, 6.79798, -595.1116, 0.654902, 0, 0.3411765, 1,
4.565657, 6.79798, -594.8462, 0.654902, 0, 0.3411765, 1,
4.606061, 6.79798, -594.5879, 0.7568628, 0, 0.2392157, 1,
4.646465, 6.79798, -594.3366, 0.7568628, 0, 0.2392157, 1,
4.686869, 6.79798, -594.0923, 0.7568628, 0, 0.2392157, 1,
4.727273, 6.79798, -593.8552, 0.7568628, 0, 0.2392157, 1,
4.767677, 6.79798, -593.6251, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.79798, -593.4021, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.79798, -593.1862, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.79798, -592.9772, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.79798, -592.7755, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.79798, -592.5807, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.79798, -592.3929, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.79798, -592.2123, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.79798, -592.0388, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.79798, -591.8723, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.79798, -591.7128, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.79798, -591.5604, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.79798, -591.4152, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.79798, -591.2769, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.79798, -591.1458, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.79798, -591.0216, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.79798, -590.9045, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.79798, -590.7946, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.79798, -590.6917, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.79798, -590.5958, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.79798, -590.507, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.79798, -590.4253, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.79798, -590.3506, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.79798, -590.2831, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.79798, -590.2225, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.79798, -590.1691, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.79798, -590.1227, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.79798, -590.0834, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.79798, -590.0511, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.79798, -590.0259, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.79798, -590.0078, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.79798, -589.9967, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.79798, -589.9927, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.79798, -589.9957, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.79798, -590.0059, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.79798, -590.0231, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.79798, -590.0473, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.79798, -590.0786, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.79798, -590.117, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.79798, -590.1625, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.79798, -590.215, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.79798, -590.2746, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.79798, -590.3412, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.79798, -590.4149, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.79798, -590.4957, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.79798, -590.5836, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.79798, -590.6785, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.79798, -590.7805, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.79798, -590.8895, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.79798, -591.0056, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.79798, -591.1287, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.79798, -591.259, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.79798, -591.3963, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.79798, -591.5406, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.79798, -591.6921, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.79798, -591.8506, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.79798, -592.0161, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.79798, -592.1887, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.79798, -592.3684, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.79798, -592.5552, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.79798, -592.749, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.79798, -592.9499, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.79798, -593.1578, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.79798, -593.3729, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.79798, -593.5949, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.79798, -593.8241, 0.7568628, 0, 0.2392157, 1,
7.434343, 6.79798, -594.0603, 0.7568628, 0, 0.2392157, 1,
7.474748, 6.79798, -594.3036, 0.7568628, 0, 0.2392157, 1,
7.515152, 6.79798, -594.5539, 0.7568628, 0, 0.2392157, 1,
7.555555, 6.79798, -594.8113, 0.7568628, 0, 0.2392157, 1,
7.59596, 6.79798, -595.0757, 0.654902, 0, 0.3411765, 1,
7.636364, 6.79798, -595.3473, 0.654902, 0, 0.3411765, 1,
7.676768, 6.79798, -595.6259, 0.654902, 0, 0.3411765, 1,
7.717172, 6.79798, -595.9116, 0.654902, 0, 0.3411765, 1,
7.757576, 6.79798, -596.2043, 0.654902, 0, 0.3411765, 1,
7.79798, 6.79798, -596.5041, 0.654902, 0, 0.3411765, 1,
7.838384, 6.79798, -596.811, 0.654902, 0, 0.3411765, 1,
7.878788, 6.79798, -597.1249, 0.654902, 0, 0.3411765, 1,
7.919192, 6.79798, -597.4459, 0.654902, 0, 0.3411765, 1,
7.959596, 6.79798, -597.7739, 0.654902, 0, 0.3411765, 1,
8, 6.79798, -598.1091, 0.654902, 0, 0.3411765, 1,
4, 6.848485, -600.2137, 0.654902, 0, 0.3411765, 1,
4.040404, 6.848485, -599.8618, 0.654902, 0, 0.3411765, 1,
4.080808, 6.848485, -599.5167, 0.654902, 0, 0.3411765, 1,
4.121212, 6.848485, -599.1786, 0.654902, 0, 0.3411765, 1,
4.161616, 6.848485, -598.8475, 0.654902, 0, 0.3411765, 1,
4.20202, 6.848485, -598.5233, 0.654902, 0, 0.3411765, 1,
4.242424, 6.848485, -598.2061, 0.654902, 0, 0.3411765, 1,
4.282828, 6.848485, -597.8959, 0.654902, 0, 0.3411765, 1,
4.323232, 6.848485, -597.5926, 0.654902, 0, 0.3411765, 1,
4.363636, 6.848485, -597.2963, 0.654902, 0, 0.3411765, 1,
4.40404, 6.848485, -597.0069, 0.654902, 0, 0.3411765, 1,
4.444445, 6.848485, -596.7245, 0.654902, 0, 0.3411765, 1,
4.484848, 6.848485, -596.449, 0.654902, 0, 0.3411765, 1,
4.525252, 6.848485, -596.1806, 0.654902, 0, 0.3411765, 1,
4.565657, 6.848485, -595.9191, 0.654902, 0, 0.3411765, 1,
4.606061, 6.848485, -595.6645, 0.654902, 0, 0.3411765, 1,
4.646465, 6.848485, -595.4169, 0.654902, 0, 0.3411765, 1,
4.686869, 6.848485, -595.1763, 0.654902, 0, 0.3411765, 1,
4.727273, 6.848485, -594.9426, 0.654902, 0, 0.3411765, 1,
4.767677, 6.848485, -594.7159, 0.7568628, 0, 0.2392157, 1,
4.808081, 6.848485, -594.4962, 0.7568628, 0, 0.2392157, 1,
4.848485, 6.848485, -594.2834, 0.7568628, 0, 0.2392157, 1,
4.888889, 6.848485, -594.0776, 0.7568628, 0, 0.2392157, 1,
4.929293, 6.848485, -593.8787, 0.7568628, 0, 0.2392157, 1,
4.969697, 6.848485, -593.6868, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.848485, -593.5019, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.848485, -593.3239, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.848485, -593.1529, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.848485, -592.9888, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.848485, -592.8317, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.848485, -592.6816, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.848485, -592.5384, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.848485, -592.4022, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.848485, -592.2729, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.848485, -592.1507, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.848485, -592.0353, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.848485, -591.927, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.848485, -591.8256, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.848485, -591.7311, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.848485, -591.6437, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.848485, -591.5631, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.848485, -591.4896, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.848485, -591.423, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.848485, -591.3633, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.848485, -591.3107, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.848485, -591.265, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.848485, -591.2262, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.848485, -591.1944, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.848485, -591.1696, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.848485, -591.1517, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.848485, -591.1408, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.848485, -591.1368, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.848485, -591.1399, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.848485, -591.1498, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.848485, -591.1668, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.848485, -591.1907, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.848485, -591.2216, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.848485, -591.2594, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.848485, -591.3041, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.848485, -591.3559, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.848485, -591.4146, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.848485, -591.4803, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.848485, -591.5529, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.848485, -591.6325, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.848485, -591.7191, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.848485, -591.8126, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.848485, -591.913, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.848485, -592.0204, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.848485, -592.1349, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.848485, -592.2562, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.848485, -592.3845, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.848485, -592.5198, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.848485, -592.662, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.848485, -592.8113, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.848485, -592.9674, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.848485, -593.1306, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.848485, -593.3007, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.848485, -593.4777, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.848485, -593.6617, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.848485, -593.8527, 0.7568628, 0, 0.2392157, 1,
7.232323, 6.848485, -594.0506, 0.7568628, 0, 0.2392157, 1,
7.272727, 6.848485, -594.2555, 0.7568628, 0, 0.2392157, 1,
7.313131, 6.848485, -594.4673, 0.7568628, 0, 0.2392157, 1,
7.353535, 6.848485, -594.6862, 0.7568628, 0, 0.2392157, 1,
7.393939, 6.848485, -594.9119, 0.654902, 0, 0.3411765, 1,
7.434343, 6.848485, -595.1447, 0.654902, 0, 0.3411765, 1,
7.474748, 6.848485, -595.3844, 0.654902, 0, 0.3411765, 1,
7.515152, 6.848485, -595.631, 0.654902, 0, 0.3411765, 1,
7.555555, 6.848485, -595.8846, 0.654902, 0, 0.3411765, 1,
7.59596, 6.848485, -596.1453, 0.654902, 0, 0.3411765, 1,
7.636364, 6.848485, -596.4128, 0.654902, 0, 0.3411765, 1,
7.676768, 6.848485, -596.6873, 0.654902, 0, 0.3411765, 1,
7.717172, 6.848485, -596.9688, 0.654902, 0, 0.3411765, 1,
7.757576, 6.848485, -597.2572, 0.654902, 0, 0.3411765, 1,
7.79798, 6.848485, -597.5526, 0.654902, 0, 0.3411765, 1,
7.838384, 6.848485, -597.8549, 0.654902, 0, 0.3411765, 1,
7.878788, 6.848485, -598.1642, 0.654902, 0, 0.3411765, 1,
7.919192, 6.848485, -598.4805, 0.654902, 0, 0.3411765, 1,
7.959596, 6.848485, -598.8038, 0.654902, 0, 0.3411765, 1,
8, 6.848485, -599.134, 0.654902, 0, 0.3411765, 1,
4, 6.89899, -601.222, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.89899, -600.8751, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.89899, -600.5351, 0.654902, 0, 0.3411765, 1,
4.121212, 6.89899, -600.202, 0.654902, 0, 0.3411765, 1,
4.161616, 6.89899, -599.8757, 0.654902, 0, 0.3411765, 1,
4.20202, 6.89899, -599.5562, 0.654902, 0, 0.3411765, 1,
4.242424, 6.89899, -599.2437, 0.654902, 0, 0.3411765, 1,
4.282828, 6.89899, -598.9379, 0.654902, 0, 0.3411765, 1,
4.323232, 6.89899, -598.639, 0.654902, 0, 0.3411765, 1,
4.363636, 6.89899, -598.347, 0.654902, 0, 0.3411765, 1,
4.40404, 6.89899, -598.0619, 0.654902, 0, 0.3411765, 1,
4.444445, 6.89899, -597.7836, 0.654902, 0, 0.3411765, 1,
4.484848, 6.89899, -597.5122, 0.654902, 0, 0.3411765, 1,
4.525252, 6.89899, -597.2477, 0.654902, 0, 0.3411765, 1,
4.565657, 6.89899, -596.99, 0.654902, 0, 0.3411765, 1,
4.606061, 6.89899, -596.7391, 0.654902, 0, 0.3411765, 1,
4.646465, 6.89899, -596.4951, 0.654902, 0, 0.3411765, 1,
4.686869, 6.89899, -596.2581, 0.654902, 0, 0.3411765, 1,
4.727273, 6.89899, -596.0278, 0.654902, 0, 0.3411765, 1,
4.767677, 6.89899, -595.8044, 0.654902, 0, 0.3411765, 1,
4.808081, 6.89899, -595.5878, 0.654902, 0, 0.3411765, 1,
4.848485, 6.89899, -595.3782, 0.654902, 0, 0.3411765, 1,
4.888889, 6.89899, -595.1754, 0.654902, 0, 0.3411765, 1,
4.929293, 6.89899, -594.9794, 0.654902, 0, 0.3411765, 1,
4.969697, 6.89899, -594.7903, 0.7568628, 0, 0.2392157, 1,
5.010101, 6.89899, -594.608, 0.7568628, 0, 0.2392157, 1,
5.050505, 6.89899, -594.4326, 0.7568628, 0, 0.2392157, 1,
5.090909, 6.89899, -594.2641, 0.7568628, 0, 0.2392157, 1,
5.131313, 6.89899, -594.1025, 0.7568628, 0, 0.2392157, 1,
5.171717, 6.89899, -593.9476, 0.7568628, 0, 0.2392157, 1,
5.212121, 6.89899, -593.7997, 0.7568628, 0, 0.2392157, 1,
5.252525, 6.89899, -593.6586, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.89899, -593.5244, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.89899, -593.397, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.89899, -593.2766, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.89899, -593.1629, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.89899, -593.0561, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.89899, -592.9562, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.89899, -592.8631, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.89899, -592.7769, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.89899, -592.6976, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.89899, -592.6251, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.89899, -592.5594, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.89899, -592.5007, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.89899, -592.4488, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.89899, -592.4037, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.89899, -592.3655, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.89899, -592.3342, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.89899, -592.3098, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.89899, -592.2922, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.89899, -592.2814, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.89899, -592.2775, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.89899, -592.2805, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.89899, -592.2903, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.89899, -592.307, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.89899, -592.3306, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.89899, -592.361, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.89899, -592.3983, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.89899, -592.4424, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.89899, -592.4933, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.89899, -592.5512, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.89899, -592.6159, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.89899, -592.6875, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.89899, -592.7659, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.89899, -592.8512, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.89899, -592.9434, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.89899, -593.0424, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.89899, -593.1483, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.89899, -593.261, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.89899, -593.3806, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.89899, -593.507, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.89899, -593.6403, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.89899, -593.7805, 0.7568628, 0, 0.2392157, 1,
6.949495, 6.89899, -593.9275, 0.7568628, 0, 0.2392157, 1,
6.989899, 6.89899, -594.0814, 0.7568628, 0, 0.2392157, 1,
7.030303, 6.89899, -594.2421, 0.7568628, 0, 0.2392157, 1,
7.070707, 6.89899, -594.4097, 0.7568628, 0, 0.2392157, 1,
7.111111, 6.89899, -594.5842, 0.7568628, 0, 0.2392157, 1,
7.151515, 6.89899, -594.7655, 0.7568628, 0, 0.2392157, 1,
7.191919, 6.89899, -594.9537, 0.654902, 0, 0.3411765, 1,
7.232323, 6.89899, -595.1487, 0.654902, 0, 0.3411765, 1,
7.272727, 6.89899, -595.3506, 0.654902, 0, 0.3411765, 1,
7.313131, 6.89899, -595.5594, 0.654902, 0, 0.3411765, 1,
7.353535, 6.89899, -595.775, 0.654902, 0, 0.3411765, 1,
7.393939, 6.89899, -595.9976, 0.654902, 0, 0.3411765, 1,
7.434343, 6.89899, -596.2269, 0.654902, 0, 0.3411765, 1,
7.474748, 6.89899, -596.4631, 0.654902, 0, 0.3411765, 1,
7.515152, 6.89899, -596.7061, 0.654902, 0, 0.3411765, 1,
7.555555, 6.89899, -596.9561, 0.654902, 0, 0.3411765, 1,
7.59596, 6.89899, -597.2128, 0.654902, 0, 0.3411765, 1,
7.636364, 6.89899, -597.4765, 0.654902, 0, 0.3411765, 1,
7.676768, 6.89899, -597.747, 0.654902, 0, 0.3411765, 1,
7.717172, 6.89899, -598.0244, 0.654902, 0, 0.3411765, 1,
7.757576, 6.89899, -598.3086, 0.654902, 0, 0.3411765, 1,
7.79798, 6.89899, -598.5997, 0.654902, 0, 0.3411765, 1,
7.838384, 6.89899, -598.8976, 0.654902, 0, 0.3411765, 1,
7.878788, 6.89899, -599.2024, 0.654902, 0, 0.3411765, 1,
7.919192, 6.89899, -599.514, 0.654902, 0, 0.3411765, 1,
7.959596, 6.89899, -599.8326, 0.654902, 0, 0.3411765, 1,
8, 6.89899, -600.158, 0.654902, 0, 0.3411765, 1,
4, 6.949495, -602.2296, 0.5490196, 0, 0.4470588, 1,
4.040404, 6.949495, -601.8877, 0.5490196, 0, 0.4470588, 1,
4.080808, 6.949495, -601.5526, 0.5490196, 0, 0.4470588, 1,
4.121212, 6.949495, -601.2242, 0.5490196, 0, 0.4470588, 1,
4.161616, 6.949495, -600.9027, 0.5490196, 0, 0.4470588, 1,
4.20202, 6.949495, -600.5879, 0.654902, 0, 0.3411765, 1,
4.242424, 6.949495, -600.2798, 0.654902, 0, 0.3411765, 1,
4.282828, 6.949495, -599.9785, 0.654902, 0, 0.3411765, 1,
4.323232, 6.949495, -599.684, 0.654902, 0, 0.3411765, 1,
4.363636, 6.949495, -599.3962, 0.654902, 0, 0.3411765, 1,
4.40404, 6.949495, -599.1152, 0.654902, 0, 0.3411765, 1,
4.444445, 6.949495, -598.841, 0.654902, 0, 0.3411765, 1,
4.484848, 6.949495, -598.5735, 0.654902, 0, 0.3411765, 1,
4.525252, 6.949495, -598.3127, 0.654902, 0, 0.3411765, 1,
4.565657, 6.949495, -598.0588, 0.654902, 0, 0.3411765, 1,
4.606061, 6.949495, -597.8116, 0.654902, 0, 0.3411765, 1,
4.646465, 6.949495, -597.5712, 0.654902, 0, 0.3411765, 1,
4.686869, 6.949495, -597.3375, 0.654902, 0, 0.3411765, 1,
4.727273, 6.949495, -597.1105, 0.654902, 0, 0.3411765, 1,
4.767677, 6.949495, -596.8904, 0.654902, 0, 0.3411765, 1,
4.808081, 6.949495, -596.6769, 0.654902, 0, 0.3411765, 1,
4.848485, 6.949495, -596.4703, 0.654902, 0, 0.3411765, 1,
4.888889, 6.949495, -596.2704, 0.654902, 0, 0.3411765, 1,
4.929293, 6.949495, -596.0773, 0.654902, 0, 0.3411765, 1,
4.969697, 6.949495, -595.8909, 0.654902, 0, 0.3411765, 1,
5.010101, 6.949495, -595.7114, 0.654902, 0, 0.3411765, 1,
5.050505, 6.949495, -595.5385, 0.654902, 0, 0.3411765, 1,
5.090909, 6.949495, -595.3724, 0.654902, 0, 0.3411765, 1,
5.131313, 6.949495, -595.2131, 0.654902, 0, 0.3411765, 1,
5.171717, 6.949495, -595.0605, 0.654902, 0, 0.3411765, 1,
5.212121, 6.949495, -594.9147, 0.654902, 0, 0.3411765, 1,
5.252525, 6.949495, -594.7757, 0.7568628, 0, 0.2392157, 1,
5.292929, 6.949495, -594.6434, 0.7568628, 0, 0.2392157, 1,
5.333333, 6.949495, -594.5179, 0.7568628, 0, 0.2392157, 1,
5.373737, 6.949495, -594.3992, 0.7568628, 0, 0.2392157, 1,
5.414141, 6.949495, -594.2872, 0.7568628, 0, 0.2392157, 1,
5.454545, 6.949495, -594.1819, 0.7568628, 0, 0.2392157, 1,
5.494949, 6.949495, -594.0834, 0.7568628, 0, 0.2392157, 1,
5.535354, 6.949495, -593.9917, 0.7568628, 0, 0.2392157, 1,
5.575758, 6.949495, -593.9067, 0.7568628, 0, 0.2392157, 1,
5.616162, 6.949495, -593.8286, 0.7568628, 0, 0.2392157, 1,
5.656566, 6.949495, -593.7571, 0.7568628, 0, 0.2392157, 1,
5.69697, 6.949495, -593.6924, 0.7568628, 0, 0.2392157, 1,
5.737374, 6.949495, -593.6345, 0.7568628, 0, 0.2392157, 1,
5.777778, 6.949495, -593.5834, 0.7568628, 0, 0.2392157, 1,
5.818182, 6.949495, -593.539, 0.7568628, 0, 0.2392157, 1,
5.858586, 6.949495, -593.5013, 0.7568628, 0, 0.2392157, 1,
5.89899, 6.949495, -593.4705, 0.7568628, 0, 0.2392157, 1,
5.939394, 6.949495, -593.4464, 0.7568628, 0, 0.2392157, 1,
5.979798, 6.949495, -593.429, 0.7568628, 0, 0.2392157, 1,
6.020202, 6.949495, -593.4184, 0.7568628, 0, 0.2392157, 1,
6.060606, 6.949495, -593.4146, 0.7568628, 0, 0.2392157, 1,
6.10101, 6.949495, -593.4175, 0.7568628, 0, 0.2392157, 1,
6.141414, 6.949495, -593.4272, 0.7568628, 0, 0.2392157, 1,
6.181818, 6.949495, -593.4437, 0.7568628, 0, 0.2392157, 1,
6.222222, 6.949495, -593.4669, 0.7568628, 0, 0.2392157, 1,
6.262626, 6.949495, -593.4968, 0.7568628, 0, 0.2392157, 1,
6.30303, 6.949495, -593.5336, 0.7568628, 0, 0.2392157, 1,
6.343434, 6.949495, -593.5771, 0.7568628, 0, 0.2392157, 1,
6.383838, 6.949495, -593.6273, 0.7568628, 0, 0.2392157, 1,
6.424242, 6.949495, -593.6843, 0.7568628, 0, 0.2392157, 1,
6.464646, 6.949495, -593.7481, 0.7568628, 0, 0.2392157, 1,
6.505051, 6.949495, -593.8186, 0.7568628, 0, 0.2392157, 1,
6.545455, 6.949495, -593.8959, 0.7568628, 0, 0.2392157, 1,
6.585859, 6.949495, -593.98, 0.7568628, 0, 0.2392157, 1,
6.626263, 6.949495, -594.0708, 0.7568628, 0, 0.2392157, 1,
6.666667, 6.949495, -594.1684, 0.7568628, 0, 0.2392157, 1,
6.707071, 6.949495, -594.2727, 0.7568628, 0, 0.2392157, 1,
6.747475, 6.949495, -594.3838, 0.7568628, 0, 0.2392157, 1,
6.787879, 6.949495, -594.5016, 0.7568628, 0, 0.2392157, 1,
6.828283, 6.949495, -594.6263, 0.7568628, 0, 0.2392157, 1,
6.868687, 6.949495, -594.7576, 0.7568628, 0, 0.2392157, 1,
6.909091, 6.949495, -594.8958, 0.654902, 0, 0.3411765, 1,
6.949495, 6.949495, -595.0406, 0.654902, 0, 0.3411765, 1,
6.989899, 6.949495, -595.1923, 0.654902, 0, 0.3411765, 1,
7.030303, 6.949495, -595.3508, 0.654902, 0, 0.3411765, 1,
7.070707, 6.949495, -595.5159, 0.654902, 0, 0.3411765, 1,
7.111111, 6.949495, -595.6879, 0.654902, 0, 0.3411765, 1,
7.151515, 6.949495, -595.8666, 0.654902, 0, 0.3411765, 1,
7.191919, 6.949495, -596.052, 0.654902, 0, 0.3411765, 1,
7.232323, 6.949495, -596.2443, 0.654902, 0, 0.3411765, 1,
7.272727, 6.949495, -596.4432, 0.654902, 0, 0.3411765, 1,
7.313131, 6.949495, -596.649, 0.654902, 0, 0.3411765, 1,
7.353535, 6.949495, -596.8615, 0.654902, 0, 0.3411765, 1,
7.393939, 6.949495, -597.0807, 0.654902, 0, 0.3411765, 1,
7.434343, 6.949495, -597.3068, 0.654902, 0, 0.3411765, 1,
7.474748, 6.949495, -597.5396, 0.654902, 0, 0.3411765, 1,
7.515152, 6.949495, -597.7791, 0.654902, 0, 0.3411765, 1,
7.555555, 6.949495, -598.0254, 0.654902, 0, 0.3411765, 1,
7.59596, 6.949495, -598.2784, 0.654902, 0, 0.3411765, 1,
7.636364, 6.949495, -598.5383, 0.654902, 0, 0.3411765, 1,
7.676768, 6.949495, -598.8049, 0.654902, 0, 0.3411765, 1,
7.717172, 6.949495, -599.0782, 0.654902, 0, 0.3411765, 1,
7.757576, 6.949495, -599.3583, 0.654902, 0, 0.3411765, 1,
7.79798, 6.949495, -599.6452, 0.654902, 0, 0.3411765, 1,
7.838384, 6.949495, -599.9388, 0.654902, 0, 0.3411765, 1,
7.878788, 6.949495, -600.2392, 0.654902, 0, 0.3411765, 1,
7.919192, 6.949495, -600.5463, 0.654902, 0, 0.3411765, 1,
7.959596, 6.949495, -600.8602, 0.5490196, 0, 0.4470588, 1,
8, 6.949495, -601.1809, 0.5490196, 0, 0.4470588, 1,
4, 7, -603.2363, 0.5490196, 0, 0.4470588, 1,
4.040404, 7, -602.8993, 0.5490196, 0, 0.4470588, 1,
4.080808, 7, -602.569, 0.5490196, 0, 0.4470588, 1,
4.121212, 7, -602.2454, 0.5490196, 0, 0.4470588, 1,
4.161616, 7, -601.9285, 0.5490196, 0, 0.4470588, 1,
4.20202, 7, -601.6182, 0.5490196, 0, 0.4470588, 1,
4.242424, 7, -601.3146, 0.5490196, 0, 0.4470588, 1,
4.282828, 7, -601.0176, 0.5490196, 0, 0.4470588, 1,
4.323232, 7, -600.7273, 0.5490196, 0, 0.4470588, 1,
4.363636, 7, -600.4437, 0.654902, 0, 0.3411765, 1,
4.40404, 7, -600.1667, 0.654902, 0, 0.3411765, 1,
4.444445, 7, -599.8964, 0.654902, 0, 0.3411765, 1,
4.484848, 7, -599.6328, 0.654902, 0, 0.3411765, 1,
4.525252, 7, -599.3758, 0.654902, 0, 0.3411765, 1,
4.565657, 7, -599.1255, 0.654902, 0, 0.3411765, 1,
4.606061, 7, -598.8818, 0.654902, 0, 0.3411765, 1,
4.646465, 7, -598.6448, 0.654902, 0, 0.3411765, 1,
4.686869, 7, -598.4145, 0.654902, 0, 0.3411765, 1,
4.727273, 7, -598.1909, 0.654902, 0, 0.3411765, 1,
4.767677, 7, -597.9739, 0.654902, 0, 0.3411765, 1,
4.808081, 7, -597.7635, 0.654902, 0, 0.3411765, 1,
4.848485, 7, -597.5598, 0.654902, 0, 0.3411765, 1,
4.888889, 7, -597.3629, 0.654902, 0, 0.3411765, 1,
4.929293, 7, -597.1725, 0.654902, 0, 0.3411765, 1,
4.969697, 7, -596.9888, 0.654902, 0, 0.3411765, 1,
5.010101, 7, -596.8118, 0.654902, 0, 0.3411765, 1,
5.050505, 7, -596.6414, 0.654902, 0, 0.3411765, 1,
5.090909, 7, -596.4777, 0.654902, 0, 0.3411765, 1,
5.131313, 7, -596.3207, 0.654902, 0, 0.3411765, 1,
5.171717, 7, -596.1703, 0.654902, 0, 0.3411765, 1,
5.212121, 7, -596.0266, 0.654902, 0, 0.3411765, 1,
5.252525, 7, -595.8896, 0.654902, 0, 0.3411765, 1,
5.292929, 7, -595.7592, 0.654902, 0, 0.3411765, 1,
5.333333, 7, -595.6355, 0.654902, 0, 0.3411765, 1,
5.373737, 7, -595.5184, 0.654902, 0, 0.3411765, 1,
5.414141, 7, -595.4081, 0.654902, 0, 0.3411765, 1,
5.454545, 7, -595.3043, 0.654902, 0, 0.3411765, 1,
5.494949, 7, -595.2073, 0.654902, 0, 0.3411765, 1,
5.535354, 7, -595.1169, 0.654902, 0, 0.3411765, 1,
5.575758, 7, -595.0331, 0.654902, 0, 0.3411765, 1,
5.616162, 7, -594.9561, 0.654902, 0, 0.3411765, 1,
5.656566, 7, -594.8857, 0.654902, 0, 0.3411765, 1,
5.69697, 7, -594.8219, 0.7568628, 0, 0.2392157, 1,
5.737374, 7, -594.7648, 0.7568628, 0, 0.2392157, 1,
5.777778, 7, -594.7144, 0.7568628, 0, 0.2392157, 1,
5.818182, 7, -594.6707, 0.7568628, 0, 0.2392157, 1,
5.858586, 7, -594.6335, 0.7568628, 0, 0.2392157, 1,
5.89899, 7, -594.6031, 0.7568628, 0, 0.2392157, 1,
5.939394, 7, -594.5793, 0.7568628, 0, 0.2392157, 1,
5.979798, 7, -594.5623, 0.7568628, 0, 0.2392157, 1,
6.020202, 7, -594.5518, 0.7568628, 0, 0.2392157, 1,
6.060606, 7, -594.548, 0.7568628, 0, 0.2392157, 1,
6.10101, 7, -594.5509, 0.7568628, 0, 0.2392157, 1,
6.141414, 7, -594.5605, 0.7568628, 0, 0.2392157, 1,
6.181818, 7, -594.5767, 0.7568628, 0, 0.2392157, 1,
6.222222, 7, -594.5995, 0.7568628, 0, 0.2392157, 1,
6.262626, 7, -594.6291, 0.7568628, 0, 0.2392157, 1,
6.30303, 7, -594.6653, 0.7568628, 0, 0.2392157, 1,
6.343434, 7, -594.7082, 0.7568628, 0, 0.2392157, 1,
6.383838, 7, -594.7577, 0.7568628, 0, 0.2392157, 1,
6.424242, 7, -594.8139, 0.7568628, 0, 0.2392157, 1,
6.464646, 7, -594.8768, 0.654902, 0, 0.3411765, 1,
6.505051, 7, -594.9463, 0.654902, 0, 0.3411765, 1,
6.545455, 7, -595.0225, 0.654902, 0, 0.3411765, 1,
6.585859, 7, -595.1053, 0.654902, 0, 0.3411765, 1,
6.626263, 7, -595.1948, 0.654902, 0, 0.3411765, 1,
6.666667, 7, -595.291, 0.654902, 0, 0.3411765, 1,
6.707071, 7, -595.3938, 0.654902, 0, 0.3411765, 1,
6.747475, 7, -595.5033, 0.654902, 0, 0.3411765, 1,
6.787879, 7, -595.6195, 0.654902, 0, 0.3411765, 1,
6.828283, 7, -595.7423, 0.654902, 0, 0.3411765, 1,
6.868687, 7, -595.8718, 0.654902, 0, 0.3411765, 1,
6.909091, 7, -596.0079, 0.654902, 0, 0.3411765, 1,
6.949495, 7, -596.1508, 0.654902, 0, 0.3411765, 1,
6.989899, 7, -596.3002, 0.654902, 0, 0.3411765, 1,
7.030303, 7, -596.4564, 0.654902, 0, 0.3411765, 1,
7.070707, 7, -596.6192, 0.654902, 0, 0.3411765, 1,
7.111111, 7, -596.7886, 0.654902, 0, 0.3411765, 1,
7.151515, 7, -596.9648, 0.654902, 0, 0.3411765, 1,
7.191919, 7, -597.1476, 0.654902, 0, 0.3411765, 1,
7.232323, 7, -597.337, 0.654902, 0, 0.3411765, 1,
7.272727, 7, -597.5331, 0.654902, 0, 0.3411765, 1,
7.313131, 7, -597.7359, 0.654902, 0, 0.3411765, 1,
7.353535, 7, -597.9454, 0.654902, 0, 0.3411765, 1,
7.393939, 7, -598.1615, 0.654902, 0, 0.3411765, 1,
7.434343, 7, -598.3843, 0.654902, 0, 0.3411765, 1,
7.474748, 7, -598.6137, 0.654902, 0, 0.3411765, 1,
7.515152, 7, -598.8498, 0.654902, 0, 0.3411765, 1,
7.555555, 7, -599.0925, 0.654902, 0, 0.3411765, 1,
7.59596, 7, -599.342, 0.654902, 0, 0.3411765, 1,
7.636364, 7, -599.5981, 0.654902, 0, 0.3411765, 1,
7.676768, 7, -599.8608, 0.654902, 0, 0.3411765, 1,
7.717172, 7, -600.1302, 0.654902, 0, 0.3411765, 1,
7.757576, 7, -600.4063, 0.654902, 0, 0.3411765, 1,
7.79798, 7, -600.689, 0.5490196, 0, 0.4470588, 1,
7.838384, 7, -600.9785, 0.5490196, 0, 0.4470588, 1,
7.878788, 7, -601.2745, 0.5490196, 0, 0.4470588, 1,
7.919192, 7, -601.5773, 0.5490196, 0, 0.4470588, 1,
7.959596, 7, -601.8867, 0.5490196, 0, 0.4470588, 1,
8, 7, -602.2027, 0.5490196, 0, 0.4470588, 1
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
6, 1.1525, -722.6, 0, -0.5, 0.5, 0.5,
6, 1.1525, -722.6, 1, -0.5, 0.5, 0.5,
6, 1.1525, -722.6, 1, 1.5, 0.5, 0.5,
6, 1.1525, -722.6, 0, 1.5, 0.5, 0.5
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
3.322, 4.5, -722.6, 0, -0.5, 0.5, 0.5,
3.322, 4.5, -722.6, 1, -0.5, 0.5, 0.5,
3.322, 4.5, -722.6, 1, 1.5, 0.5, 0.5,
3.322, 4.5, -722.6, 0, 1.5, 0.5, 0.5
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
3.322, 1.1525, -606.4028, 0, -0.5, 0.5, 0.5,
3.322, 1.1525, -606.4028, 1, -0.5, 0.5, 0.5,
3.322, 1.1525, -606.4028, 1, 1.5, 0.5, 0.5,
3.322, 1.1525, -606.4028, 0, 1.5, 0.5, 0.5
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
4, 1.925, -695.7853,
8, 1.925, -695.7853,
4, 1.925, -695.7853,
4, 1.79625, -700.2544,
5, 1.925, -695.7853,
5, 1.79625, -700.2544,
6, 1.925, -695.7853,
6, 1.79625, -700.2544,
7, 1.925, -695.7853,
7, 1.79625, -700.2544,
8, 1.925, -695.7853,
8, 1.79625, -700.2544
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
4, 1.53875, -709.1926, 0, -0.5, 0.5, 0.5,
4, 1.53875, -709.1926, 1, -0.5, 0.5, 0.5,
4, 1.53875, -709.1926, 1, 1.5, 0.5, 0.5,
4, 1.53875, -709.1926, 0, 1.5, 0.5, 0.5,
5, 1.53875, -709.1926, 0, -0.5, 0.5, 0.5,
5, 1.53875, -709.1926, 1, -0.5, 0.5, 0.5,
5, 1.53875, -709.1926, 1, 1.5, 0.5, 0.5,
5, 1.53875, -709.1926, 0, 1.5, 0.5, 0.5,
6, 1.53875, -709.1926, 0, -0.5, 0.5, 0.5,
6, 1.53875, -709.1926, 1, -0.5, 0.5, 0.5,
6, 1.53875, -709.1926, 1, 1.5, 0.5, 0.5,
6, 1.53875, -709.1926, 0, 1.5, 0.5, 0.5,
7, 1.53875, -709.1926, 0, -0.5, 0.5, 0.5,
7, 1.53875, -709.1926, 1, -0.5, 0.5, 0.5,
7, 1.53875, -709.1926, 1, 1.5, 0.5, 0.5,
7, 1.53875, -709.1926, 0, 1.5, 0.5, 0.5,
8, 1.53875, -709.1926, 0, -0.5, 0.5, 0.5,
8, 1.53875, -709.1926, 1, -0.5, 0.5, 0.5,
8, 1.53875, -709.1926, 1, 1.5, 0.5, 0.5,
8, 1.53875, -709.1926, 0, 1.5, 0.5, 0.5
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
3.94, 2, -695.7853,
3.94, 7, -695.7853,
3.94, 2, -695.7853,
3.837, 2, -700.2544,
3.94, 3, -695.7853,
3.837, 3, -700.2544,
3.94, 4, -695.7853,
3.837, 4, -700.2544,
3.94, 5, -695.7853,
3.837, 5, -700.2544,
3.94, 6, -695.7853,
3.837, 6, -700.2544,
3.94, 7, -695.7853,
3.837, 7, -700.2544
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
3.631, 2, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 2, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 2, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 2, -709.1926, 0, 1.5, 0.5, 0.5,
3.631, 3, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 3, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 3, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 3, -709.1926, 0, 1.5, 0.5, 0.5,
3.631, 4, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 4, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 4, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 4, -709.1926, 0, 1.5, 0.5, 0.5,
3.631, 5, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 5, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 5, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 5, -709.1926, 0, 1.5, 0.5, 0.5,
3.631, 6, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 6, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 6, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 6, -709.1926, 0, 1.5, 0.5, 0.5,
3.631, 7, -709.1926, 0, -0.5, 0.5, 0.5,
3.631, 7, -709.1926, 1, -0.5, 0.5, 0.5,
3.631, 7, -709.1926, 1, 1.5, 0.5, 0.5,
3.631, 7, -709.1926, 0, 1.5, 0.5, 0.5
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
3.94, 1.925, -695.7853,
3.94, 7.075, -695.7853,
3.94, 1.925, -517.0204,
3.94, 7.075, -517.0204,
3.94, 1.925, -695.7853,
3.94, 1.925, -517.0204,
3.94, 7.075, -695.7853,
3.94, 7.075, -517.0204,
3.94, 1.925, -695.7853,
8.06, 1.925, -695.7853,
3.94, 1.925, -517.0204,
8.06, 1.925, -517.0204,
3.94, 7.075, -695.7853,
8.06, 7.075, -695.7853,
3.94, 7.075, -517.0204,
8.06, 7.075, -517.0204,
8.06, 1.925, -695.7853,
8.06, 7.075, -695.7853,
8.06, 1.925, -517.0204,
8.06, 7.075, -517.0204,
8.06, 1.925, -695.7853,
8.06, 1.925, -517.0204,
8.06, 7.075, -695.7853,
8.06, 7.075, -517.0204
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
var radius = 95.52187,
distance = 424.9876,
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
this.mvMatrix.translate( -6, -4.5, 606.4029 );
this.mvMatrix.scale( 25.06799, 20.05439, 0.5777431 );
this.mvMatrix.multRight( unnamed_chunk_41rgl.userMatrix[1] );
this.mvMatrix.translate(-0, -0, -424.9876);
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
## 2552 6.060606 3.262626 -519.6238 #FFFF00
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
##  6.063679  1.179280 
## 
## $value
## [1] 519.6215
## 
## $counts
## function gradient 
##       55       NA 
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
## [1,] 6.060606 3.262626
## [2,] 6.063679 3.252032
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
##       alpha        beta   log_sigma 
##  3.89843696  9.02887297 -0.06640212 
## 
## $value
## [1] 27.04731
## 
## $counts
## function gradient 
##      158       NA 
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
## #  Elapsed Time: 0.005613 seconds (Warm-up)
## #                0.005264 seconds (Sampling)
## #                0.010877 seconds (Total)
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
## #  Elapsed Time: 0.005805 seconds (Warm-up)
## #                0.005301 seconds (Sampling)
## #                0.011106 seconds (Total)
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
## #  Elapsed Time: 0.005857 seconds (Warm-up)
## #                0.005638 seconds (Sampling)
## #                0.011495 seconds (Total)
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
## #  Elapsed Time: 0.005805 seconds (Warm-up)
## #                0.005447 seconds (Sampling)
## #                0.011252 seconds (Total)
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
## p     0.30    0.00 0.13  0.08  0.20  0.29  0.39  0.59  1299    1
## lp__ -6.61    0.02 0.70 -8.60 -6.76 -6.33 -6.16 -6.11  1259    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:46:30 2015.
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
##           mean se_mean   sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
## beta[1]  -3.12    0.03 0.85  -4.78  -3.66  -3.13  -2.57  -1.39  1083    1
## beta[2]   0.78    0.01 0.38   0.01   0.54   0.79   1.03   1.51  1086    1
## sigma     1.24    0.01 0.22   0.89   1.09   1.21   1.37   1.74   942    1
## lp__    -13.66    0.04 1.28 -16.90 -14.24 -13.33 -12.74 -12.19   875    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:47:02 2015.
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
## [1] 0.978
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
##       2.5%      97.5% 
## 0.01344458 1.50841808
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
## [1] 0.04428111 1.53135419
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
## -2.0971  -0.8279  -0.0589   0.4578   2.0410  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)   0.5182     0.1221   4.244 2.19e-05 ***
## x            -1.0726     0.1975  -5.432 5.57e-08 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 85.632  on 49  degrees of freedom
## Residual deviance: 51.860  on 48  degrees of freedom
## AIC: 165.35
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
## (Intercept)  0.2662044  0.7461540
## x           -1.4714294 -0.6957461
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
##          mean se_mean   sd   2.5%   25%   50%   75% 97.5% n_eff Rhat
## beta[1]  0.51    0.00 0.12   0.26  0.43  0.51  0.59  0.74  1022    1
## beta[2] -1.08    0.01 0.20  -1.48 -1.21 -1.08 -0.94 -0.69  1136    1
## lp__    -9.03    0.03 1.03 -11.86 -9.41 -8.70 -8.30 -8.04  1124    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:47:38 2015.
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
## beta[1]   0.49    0.00 0.11   0.26   0.41   0.49   0.56   0.70  1378    1
## lp__    -41.96    0.02 0.70 -44.05 -42.13 -41.67 -41.50 -41.45  1399    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:47:43 2015.
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
## sigma     1.22    0.01 0.25   0.81   1.05   1.20   1.37   1.78   487 1.01
## beta[1]  -0.17    0.01 0.28  -0.76  -0.34  -0.15   0.03   0.31   552 1.00
## lp__    -29.99    0.36 7.78 -46.22 -35.01 -30.00 -24.55 -15.30   467 1.01
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:48:19 2015.
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
##           mean se_mean   sd   2.5%    25%   50%    75%  97.5% n_eff Rhat
## beta[1]   0.50    0.00 0.20   0.12   0.37   0.5   0.63   0.89  1835    1
## phi       0.86    0.01 0.33   0.41   0.63   0.8   1.01   1.67  1902    1
## lp__    -20.80    0.03 1.00 -23.62 -21.22 -20.5 -20.07 -19.80   941    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:48:54 2015.
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
## -1.4824  -0.8429  -0.6351   0.4857   2.3828  
## 
## Coefficients:
##          Estimate Std. Error z value Pr(>|z|)    
## companyA  -3.5051     0.3836  -9.137  < 2e-16 ***
## companyB  -4.0775     0.5042  -8.087 6.12e-16 ***
## companyC  -2.9444     0.2962  -9.942  < 2e-16 ***
## companyD  -2.3445     0.2284 -10.263  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1075.199  on 80  degrees of freedom
## Residual deviance:   86.034  on 76  degrees of freedom
## AIC: 157.64
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
## 0.02916667 0.01666667 0.05000000 0.08750000
```

```r
plogis(confint(m))
```

```
## Waiting for profiling to be done...
```

```
##                2.5 %     97.5 %
## companyA 0.012637598 0.05563394
## companyB 0.005203111 0.03828885
## companyC 0.027090112 0.08243072
## companyD 0.056130675 0.12761339
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
## beta[1]   -3.45    0.01 0.38   -4.25   -3.69   -3.43   -3.18   -2.75  2438
## beta[2]   -3.94    0.01 0.45   -4.90   -4.23   -3.92   -3.62   -3.12  2808
## beta[3]   -2.93    0.01 0.31   -3.58   -3.12   -2.91   -2.71   -2.37  2718
## beta[4]   -2.33    0.00 0.22   -2.77   -2.47   -2.32   -2.17   -1.90  3203
## lp__    -178.13    0.04 1.48 -181.92 -178.82 -177.79 -177.06 -176.31  1454
##         Rhat
## beta[1]    1
## beta[2]    1
## beta[3]    1
## beta[4]    1
## lp__       1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:49:30 2015.
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
## psi    0.45    0.00 0.07   0.32   0.40   0.45   0.49   0.58  2559    1
## p      0.83    0.00 0.05   0.73   0.80   0.84   0.87   0.92  2619    1
## lp__ -66.72    0.03 1.04 -69.43 -67.11 -66.40 -66.00 -65.73  1033    1
## 
## Samples were drawn using NUTS(diag_e) at Wed Nov 25 19:50:06 2015.
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

$$[y_i \mid p_i] [p_i \mid p_0, \pi_i, \tau_i] [\pi_i \mid \sigma_\pi] [\tau_j \mid \sigma_\tau]$$

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
