# simulating data for bird clutch size poisson-gamma hierarchy example:
n <- 200
Sigma <- matrix(c(.1, 0, 0, .1), nrow=2)
eta <- matrix(rnorm(2*n), ncol=2) %*% chol(Sigma)
eta[, 1] <- eta[, 1]
plot(eta)
mass <- exp(rnorm(n, 1, .3))
p <- plogis(eta[, 1] + 2 * mass - 6)
lambda <- exp(eta[, 2] + .6 * mass - 1)

plot(lambda, p)
y <- rpois(n, lambda)
z <- rbinom(n, size=y, prob=p)

d <- data.frame(eggs = y, 
                fledglings = z, 
                bird_mass = exp(rnorm(n)), 
                population = rep(c('A', 'B', 'C', 'D'), each=25))
write.csv(d, 'eggs.csv', row.names=FALSE)
