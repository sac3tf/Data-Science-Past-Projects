library(tswge)
library(tseries)
## Unit 10.3
##generate ARIMA(2,1,0) data
xd1 = gen.arima.wge(n=200, phi=c(1.2,-.8), d=1, sn = 56)
##differenced data
xd1.diff = artrans.wge(xd1, phi.tr = 1)
##aic on the differnced (stationary) data
aic5.wge(xd1.diff, p=0:5, q=0:2)
##aic picks AR2
est.ar.wge(xd1.diff, p = 2)
mean(xd1)

##This concept check is to show that the differnced data is white noise and data is actually stationary
data <- read.csv('/Users/stevencocke/Downloads/10_year_bond_rate_2010-2015 (1).csv', sep =",")
plotts.sample.wge(data$Close)
data.dif = artrans.wge(data$Close, phi.tr = 1)
aic5.wge(data.dif)
aic5.wge(data$Close)

##10.5 Forecasting ARIMA review
x1 = gen.arma.wge(n=200, phi=c(.3,-.8))
fore.arma.wge(x1, phi = c(.3,-.8), n.ahead = 50)

x2 = gen.aruma.wge(n=200, d= 2, phi = c(.3, -.8))
fore.aruma.wge(x2, d = 2, phi = c(.3,-.8), n.ahead = 50)

##Unit 10.7
##fit an AR(6) model and confirm there is evidence of a (1-B) factor using burg estimates
est.ar.wge(data$Close, p = 6, type = 'burg')

data = read.csv('/Users/stevencocke/Downloads/zero_one_or_tworootsofone (1).csv', sep =",")
plotts.sample.wge(data$x)
est.ar.wge(data$x, p = 8, type = 'burg')

##Unit 10.8 Dicky Fuller
x = gen.arma.wge(200,phi = c(.9), sn = 5)
adf.test(x)

x = gen.arma.wge(200,phi = c(.9))

adf.test(x)

##10.9 Seasonal Models
x = gen.aruma.wge(n = 48, s = 4, sn = 23)
est.ar.wge(x, p = 8, type = 'burg')
y = artrans.wge(x, phi.tr = c(0,0,0,1))

x = gen.aruma.wge(n = 200, s = 12, phi = c(1.5,-.8), sn = 87)
x=x+50
plotts.sample.wge(x, lag.max = 60)
d15 = est.ar.wge(x, p =15, type = 'burg')
y = artrans.wge(x, phi.tr = c(0,0,0,0,0,0,0,0,0,0,0,1))
aic5.wge(y, p=0:13, q=0:3, type = 'bic')
factor.wge()
?factor.wge

data = read.csv('/Users/stevencocke/Downloads/swadelay (1).csv', sep =",")
est.ar.wge(data$arr_delay, p=15, type = 'burg')
factor.wge(phi = c(0,0,0,0,0,0,0,0,0,0,0,1))

x=gen.aruma.wge(n=200, s=12, phi = c(1.5,-.8), sn = 87)
x=x+50
fore.aruma.wge(x, s=12, phi = c(1.47, -.76), n.ahead = 36, lastn = TRUE)

##10.12 Signal Plus Noise
x = gen.signalplusnoise.wge(100, b0 = 0, b1= 0, phi= .95, sn = 28)
t = seq(1,100,1)
df = data.frame(x = x, t= t)
fit = lm(x~t, data = df)
summary(fit)

x = gen.sigplusnoise.wge(100, b0 = 0, b1= 0, phi= .99)  #note that there is not a seed this time. We will generate a different realization each time.
t = seq(1,100,1)
df = data.frame(x = x, t= t)
fit = lm(x~t, data = df)
summary(fit) # record whether it rejected or failed to reject.

##Cochrane-Orcutt
install.packages('orcutt', dependencies=TRUE, repos='http://cran.rstudio.com/')
library(orcutt)
x = gen.sigplusnoise.wge(100, b0=0, b1=0, phi = .95, sn = 21)
t = seq(1,100,1)
df = data.frame(x = x, t=t)
fit = lm(x~t, data = df)
summary(fit)
cfit = cochrane.orcutt(fit)
summary(cfit)

t = seq(1,69,1)
df = data.frame(x = data$arr_delay, t=t)
fit = lm(x~t, data = df)
summary(fit)
cfit = cochrane.orcutt(fit)
summary(cfit)

##For Live
data = read.csv('/Users/stevencocke/Desktop/Time Series/Data/total-married-families-with-children-under-age-18/total-families-with-children-under-18-years-old-with-married-couple.csv', sep = ",")
data
t = seq(1,69,1)
df = data.frame(x = data$value, t=t)
fit = lm(x~t, data = df)
summary(fit)
cfit = cochrane.orcutt(fit)
summary(cfit)

est.ar.wge(data$value, p = 15, type = 'burg')

plotts.sample.wge(data$value)

aic5.wge(data$value, p = 0:15, q= 0:4, type = 'bic')

est.arma.wge(data$value, p = 2, q = 1)
