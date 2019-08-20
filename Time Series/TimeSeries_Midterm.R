install.packages('PolynomF')

##Unit 1
##View autocorrelation by lags
Y5 = somedata
acf(Y5, plot = TRUE, lag.max = 1 or more)

##find the mean
mean(somedata$number)

##Unit 2
##spectral density
parzen.wge(somedata$number, trunc = somenumber)
##acf and spectral density  and realization
plotts.sample.wge(somedata$number)

##Unit 3
##moving average filter 5 point
ma = filter(somedata, rep(1,5)/5)
plot(ma,type = "l")
##difference filter
dif = dif(somedata, lag=1)
plot(dif, type = "l")

##generate AR model...use sn= for seed
gen.arma.wge(n=100, phi=.95)
gen.arma.wge(n=250, phi=-.7)
##generate white noise
gen.arma.wge(n=100)

##plot true acf and sd
plotts.true.wge(phi=-.95)

##plot true acf for a model
x = gen.arma.wge(n=100,phi = .95)
plotts.sample.wge(x)

##AR(2) with positive and one negative root...switch signs to explore two pos and two neg
x = gen.arma.wge(n=200, phi = c(.2, .-48))

##Unit 4
##Factor Table for AR(3) 1.95x_t-1+1.85x_t-2-.855x_t-3-a_t...remember to add them smallest to largest
factor.wge(phi = c(1.95, -1.85, .855))
##ACF and SD
plotts.true.wge(phi=c(1.95, -1.85, .855))

##Unit5
##generate MA(1) and MA(2) models
gen.arma.wge(n=100, theta = -.99)
gen.arma.wge(n=100, theta = c(.9,-.4))
plotts.true.wge(theta = c(.9,-.6))

##Factor MA(2)
factor.wge(phi = c(.6, -.9, .8, -.2))

##When factoring ARMA models, factor each side separately, both using factor.wge and phi=
##ARMA PLOT
plotts.wge(phi=c(.3,.9,.1,-.8075), theta=c(-.9,-.8,-.72))

##AIC
##Once specific AIC
aic.wge(somedata$number, p = , q = )$value
##Best AICs
aic5.wge(somedata$number)
##Psi Weight
psi.weights.wge(phi = c(1.2, -.6), theta = c(.5), lag.max = 5)

##Unit 6
##signal plus noise
gen.sigplusnoise.wge(100, b0 = 1, b1 = 4, vara = 200)

#periodic signal
gen.sigplusnoise.wge(100, coef = c(5,0), freq = c(.1,0), psi = c(0,.25), phi = .975, vara = 20)
##get coefficients
parms = mult.wge(c(.975), c(.2,-.45), c(-.53))
parms$model.coef
##ARIMA(0,1,0) var = white noise component and d is the season part
x = gen.arima.wge(200, phi = 0, var = 1, d =1)
acf(x)

##Difference the data...we will only have size 1 less than n because of the subtracting
x= gen.arima.wge(200, phi = 0, var = 1, d = 1, sn=31)
firstdif = artrans.wge(x,1)
seconddif = artrans.wge(firstdif, 1)
parzen.wge(seconddif)
aic5.wge(seconddif)

##ARUMA for (1-B^4)
x1 = gen.aruma.wge(n=80, s=4, sn=6)
plotts.sample.wge(x1)

##ARUMA for (1-B^4) with ARMA(2,1)
x2 = gen.aruma.wge(n=80, phi = c(1,-.6), s=4, theta = -.5, sn=6)
factor.wge(phi = 1,-.6)
factor.wge(phi=-.5)
plotts.sample.wge(x2, lag.max = 45)

##Stationarize ARUMA model
x=gen.aruma.wge(n=80, s=4, sn= 81)
dif = artrans.wge(x, c(0,0,0,1)) ##take out the (1-B^4)
aic5.wge(dif) #check the structure of the noise

##Another example
x = gen.aruma.wge(n = 80, phi = c(.4,.6,-.74) , theta = c(-.7), s=12, sn = 31)
dif = artrans.wge(x, c(rep(0,11), 1)) ##take out (1-B^12)
aic5.wge(dif)

##factor the seasonal component of (1-B^12)
##to show if something is seasonal, compare factor of the model to factor of the basic seasonal to see if factors are the same
factor.wge(phi = c(rep(0,11),1))

##Unit 7 
##Forecast AR
fore.arma.wge(somedata, phi = .8, n.ahead = 20, plot = TRUE, limits = FALSE)

##Forecast ARMA
x = gen.arma.wge(n = 75, phi = c(.6), sn=24)
fore.arma.wge(x, phi = c(.6), n.ahead = 20, limits = FALSE)
?fore.arma.wge

#Calculate Psi Weights...these are used to calc prob weights
psi.weights.wge(phi = c(.6, -.4), lag.max = 3)

##Probability limits...f = forecasts, ll = lowerlimits, ul = upper limits
fore.arma.wge(x, phi = c(1.6, -.8), theta = -.9, n.ahead = 20, plot = TRUE, limits = TRUE)

##Forecast ARIMA(0,1,0)
x=gen.aruma.wge(n=50, phi=-.8, d = 1, sn= 15)
fore.aruma.wge(x, d=1, n.ahead = 20, limits = FALSE)
?fore.aruma.wge
##Forecast Seasonal Model (1-B^4)
x = gen.aruma.wge(n=20, s=4, sn = 6)
fore.aruma.wge(x, s = 4, n.ahead = 8, lastn = FALSE, plot = TRUE, limits = FALSE)
fore.aruma.wge(x, s = 4, n.ahead = 8, lastn = TRUE, plot = TRUE, limits = FALSE)

##Forecast Signal Pllus Noise
fore.sigplusnoise.wge(x, linear = TRUE, freq = 0, max.p = 5, n.ahead = 10, lastn = FALSE, plot = TRUE, limits = TRUE)

y = gen.sigplusnoise.wge(n=50,b0=5,b1=.3,coef = c(0,0), freq = c(0,0), psi = c(0,0) , phi = c(-.2,-.6), vara=.5, plot = TRUE, sn = 0)
plotts.sample.wge(y)
fo = fore.sigplusnoise.wge(y, linear = TRUE, freq = 0, max.p = 5, n.ahead = 10, lastn = FALSE, plot = TRUE ,limits = TRUE)

ASE = mean((fo$f-y[41:50])^2)
ASE





