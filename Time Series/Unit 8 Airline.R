Walmart <- read.csv("~/Desktop/Time Series/Walmart.csv")
install.packages('PolynomF')
library(tswge)
library(ggplot2)
factor.wge(phi=c(1.445, -.411, -.038, .170, .362, -.245, -.177, .213))

plotts.true.wge(phi=c(1.445, -.411, -.038, .170, .362, -.245, -.177, .213))

Data_Store9Item50 <- subset(Walmart, store == 9, select = c("sales", "item"))
Data_Item50 <- subset(Data_Store9Item50, item == 50, select = c("sales"))

plotts.wge(Data_Item50$sales)

aic5.wge(Data_Store9Item50$sales)

parzen.wge(Data_Item50$sales)

acf(Data_Item50, lag.max = 25, type = c("correlation", "covariance", "partial"), plot = TRUE, na.action = na.contiguous, demean = TRUE)


plotts.true.wge(theta=c(-.1,.3))

factor.wge(c(1.95,-1.9))
factor.wge(c(1.95,-1.9))

plotts.true.wge(phi=c(.65,-.75), theta=c(-.72,.43))

swa <- read.csv(file="/Users/stevencocke/Desktop/Time Series/Data/swadelay.csv", header = TRUE, sep=",")
aic5.wge(swa$weather_delay)
plotts.sample.wge(swa$arr_delay)
psi.weights.wge(phi=c(1.95,-1.9), lag.max = 5)

aic5.wge(swa$arr_cancelled)


x = plotts.true.wge(theta = c(.8,-.5))

x$aut[2]



x = plotts.true.wge(phi = c(-.3), theta = c(.6, -.8))

x= gen.arima.wge(500, phi = c(-.3), theta = c(.6, -.8), var = 1, d = 2, sn = 35)
firstdf = artrans.wge(x,1)
seconddf = artrans.wge(firstdf,1)
parzen.wge(seconddf)

data()
factor.wge(phi = c(.6, -.8))
aic5.wge(seconddf)

x3 = gen.aruma.wge(n = 200, phi = c(.6, -.94), theta = c(-.3), s = 6, sn = 19)

x4 = gen.aruma.wge(n=500, phi = c(.6,-.8), theta = c(0,1), s = 12, sn =37)
dif = artrans.wge(x4, c(rep(0,11),1))
aic5.wge(dif)

x=gen.aruma.wge(n=500, phi = c(.6,-.8), theta = c(.3, -.7), s=12, sn = 31)
Dif = artrans.wge(x,c(rep(0,11),1)) #Take out the (1-B^12)
aic5.wge(Dif)




factor.wge(c(rep(0,6),1))
factor.wge(c(-.5,.2,0,1,.5,-.2))

stocks <- read.csv("/Users/stevencocke/Downloads/CSV (1).csv", header = TRUE)
plot(x=stocks$Date, y=stocks$Close, type = "l")
ggplot(data=stocks, aes(x=Date, y=Close, group=1)) +
  geom_line()+
  geom_point()
stocks <- stocks["Close"]
firstdf = artrans.wge(stocks$Close, phi.tr = 1, lag.max = 50, plottr = TRUE)
?artrans.wge

x1 = gen.arma.wge(n=75, phi=c(1.6,-.8), sn = 24)
fore.arma.wge(x1, phi = c(1.6,-.8), n.ahead = 20, limits = FALSE)

psi.weights.wge(phi = c(-.72,1.7), lag.max = 5)

swa_an <- swa[(swa$year >= 2004 && swa$month >= 2)]
swa_an <- swa_an[swa_an$year <= 2018 and ]