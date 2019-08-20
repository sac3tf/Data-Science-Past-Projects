##Unit 9
library(tswge)
##ARMA Estimation function...phis, avar = white noise variance
est.arma.wge(somedata, p =2, q = 1)

##AR Estimation...mle is the default so not necessary
est.ar.wge(somedata, p=4, type = 'mle')

##Example model (1-1.6B+.8B^2)(X_t-50) = (1-.8B)a_t, sigma^2 = 5
x21 = gen.arma.wge(n=100, phi=c(1.6,-.8), theta = .8, vara = 5, sn =55)
x21 = x21 +50
est.arma.wge(x21, p = 2, q = 1)
##Question from 9.3
x = gen.arma.wge(n = 200, phi = c(.3, -.7), theta = -.4, vara = 4, sn = 27)
x = x - 37
est.arma.wge(x, p = 2, q = 1)

##Yule-walker estimate
est.ar.wge(somedata, p =2, type = 'yw')
x.yw = est.ar.wge(somedata, p =2, type = 'yw')
x.yw
##Burg Estimates
x.burg = est.ar.wge(somedata, p =2, type = 'burg')
x.burg
##MLE Estimates
x.mle = est.ar.wge(somedata, p =2, type = 'mle')
x.mle

##Question from 9.5
x = gen.arma.wge(n = 200, phi = c(.3, -.7), vara = 4, sn = 27)
x = x - 37
est.ar.wge(x, p = 2, type = 'burg')

##Estimate White Noise Variance....$res
x = gen.arma.wge(n=100, phi = c(2,195, -1.994, .796), sn = 53)
x.mle=est.ar.wge(x, p=3, type = 'mle')
x.mle
x.mle$avar
##to get the white noise variance
mean(x.mle$res^2)
x.mle$avar

##Question 9.8
data <- read.csv('/Users/stevencocke/Downloads/maybewhitenoise2.csv', sep =",")
plotts.sample.wge(data$x)

##Question 9.9 AIC calculation
x = gen.arma.wge()
##default below
aic.wge(x, p=0:5, q=0:2,type = "aic" )

##9.10 AIC5
data <- read.csv('/Users/stevencocke/Downloads/inflation (1).csv', sep =",")
aic5.wge(data$Inflation, type = 'bic')
pacf(data$Inflation)

##9.11 PACF
pacf(somedata)

data <- read.csv('/Users/stevencocke/Downloads/armawhatpq1 (1).csv', sep =",")
pacf(data$x)
aic5.wge(data$x)

data <- read.csv('/Users/stevencocke/Downloads/texasgasprice (1).csv', sep =",")
aic5.wge(data$Price)
y = est.ar.wge(data$Price, p = 2, type = 'burg')
x = fore.arma.wge(data$Price, y$phi, n.ahead = 24, limits = FALSE, lastn = TRUE)
x$f
mean(x$f[50:56])
data_reduced = data$Price[-(182:205)]
x = fore.arma.wge(data_reduced, phi = c(1.3813631, -.4058498), n.ahead = 24, limits = FALSE)
ASE = mean((x$f-data$Price[182:205])^2)
ASE

##Live Session
data <- read.csv('/Users/stevencocke/Downloads/Unit9_2.csv', sep =",")
data
a_1 = gen.arma.wge(100, phi = -.99999)
a_n1 = gen.arma.wge(100, phi = .99999)
par(mfrow=c(1,1))
plotts.wge(a_1)
plotts.wge(a_n1)

data <- read.csv('/Users/stevencocke/Downloads/Unit9_1.csv', sep =",")
plotts.sample.wge(data$x)
aic5.wge(data$x, p = 0:10, q = 0:4, type = 'bic')
pacf(data$x)
estimate = est.arma.wge(data$x, p = 9, q = 1)
fore.arma.wge(data$x, estimate$phi, limits = FALSE, lastn = FALSE)
