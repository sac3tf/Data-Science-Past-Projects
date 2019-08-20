data = read.csv('/Applications/Splunk/etc/apps/Splunk_SA_Scientific_Python_darwin_x86_64/bin/darwin_x86_64/lib/python2.7/site-packages/statsmodels/datasets/sunspots/sunspots.csv', sep = ",")
library(tswge)
aic5.wge(data$SUNACTIVITY, p= 0:10, q= 0)

##For live session
data = read.csv('/Users/stevencocke/Desktop/Time Series/Data/SN_y_tot_V2.0.csv', sep = ";")
sunspots$counts = data$X8.3

plotts.sample.wge(sunspots$counts)

aic5.wge(sunspots$counts, p=0:15, q=0:5, type = 'aicc')

x = est.arma.wge(sunspots$counts, p = 6, q=2)
mean(sunspots$counts)

y = fore.arma.wge(sunspots$counts, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = TRUE)

ASE = mean((y$f-sunspots$counts[304:318])^2)
ASE

x = est.arma.wge(sunspots$counts, p = 6, q=2)


y = artrans.wge(sunspots$counts, phi.tr = c(0,0,0,0,1))
aic5.wge(y, p=0:15, q=0:5, type = 'aic')

x = est.arma.wge(sunspots$counts, p = 6, q=4)
y = fore.aruma.wge(sunspots$counts, s = 5, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = TRUE)
ASE = mean((y$f-sunspots$counts[304:318])^2)
ASE

x = est.arma.wge(sunspots$counts, p = 6, q=2)
y = fore.arma.wge(sunspots$counts, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = FALSE)
y$f


data = read.csv('/Users/stevencocke/Desktop/Time Series/Data/accuspike.csv', sep = ",")
plotts.sample.wge(data$Active.Users)
aic5.wge(data$Active.Users, p=0:15, q=0:5)

x=est.arma.wge(data$Active.Users, p=5, q=3)
y = fore.arma.wge(data$Active.Users, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = FALSE)
ASE = mean((y$f-data$Active.Users[178:192])^2)
ASE
