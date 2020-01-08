library(tswge)
BSales = read.csv('/Users/stevencocke/Downloads/businesssales (1).csv', sep = ",")

#without time
ksfit - lm(sales~ad_tv+ad_online+discount, data = BSales)
aic.wge(ksfit$residuals, p = 0:8, q=0)
fit = arima(BSales$sales, order = c(7,0,0), xreg = BSales[,3:5])

#with time
t=1:100
ksfit - lm(sales~t+ad_tv+ad_online+discount, data = BSales)
aic.wge(ksfit$residuals, p = 0:8, q=0)
fit = arima(BSales$sales, order = c(7,0,0), xreg = cbind(t,BSales[,3:5]))

##to lag variables
ad_tv1 = dplyr::lag(BSales$ad_tv,1)

whatislag = read.csv('/Users/stevencocke/Downloads/whatisthelag (1).csv', sep = ",")

ccf(whatislag$X1, whatislag$Y)

##VAR
install.packages('vars', repos='http://cran.us.r-project.org')
library(vars)
?VARselect()
VARselect(X, lag.max = 6, type = "const", season = NULL, exogen = NULL)
lsfit=VAR(X, p=5, type = "const")
preds = predict(lsfit, n.ahead=5)

X=cbind(BSales$ad_tv, BSales$ad_online, BSales$discount)
VARselect(X, lag.max=6, type="const", season = NULL, exogen = NULL)
x = VAR(BSales[,2:5], p = 2, type = "const")
VARselect(BSales[,2:5], lag.max = 6, type = "const")
lsfit=VAR(BSales[,2:5], p=2, type = "const")
summary(lsfit)
lsfit

cmort_data = read.csv('/Users/stevencocke/Downloads/la_cmort_study (1).csv', sep = ",")
plotts.sample.wge(cmort$cmort)

#without time
ksfit = lm(cmort~temp+part, data = cmort_data)
aic.wge(ksfit$residuals, p = 0:8, q=0)
fit = arima(cmort_data$cmort, order = c(4,0,0), xreg = cmort_data[,2:3])
plotts.sample.wge(fit$residuals)

X=cbind(cmort_data$temp, cmort_data$part)
VARselect(X, lag.max=12, type="const", season = NULL, exogen = NULL)
x = VAR(BSales[,2:5], p = 2, type = "const")
VARselect(cmort_data[,2:3], lag.max = 6, type = "const")
lsfit=VAR(cmort_data[,2:3], p=6, type = "const")
summary(lsfit)
lsfit