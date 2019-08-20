
##Load Libraries
library(tswge)
library(datasets)

##Grab dataset
datasets::Seatbelts

#Convert to dataframe
SB <- Seatbelts
class(SB)
df <- data.frame(SB)

#View the structure
str(df)

#View the data
head(df)

#Plot the data
plotts.sample.wge(df$DriversKilled)

##AIC #p=12 and q=9 ##BIC p=12 q=0
aic5.wge(df$DriversKilled, p=0:15, q=0:10, type = "bic")
aic5.wge(df$DriversKilled, p=0:15, q=0:10)

#Estimate phi
x = est.ar.wge(df$DriversKilled, p = 12)
mean(df$DriversKilled)

#Factor tables for seasonality
factor.wge(phi = c(0,0,0,0,0,0,0,0,0,0,0,1))
factor.wge(phi = c(0,0,0,1))

#Transform
y = artrans.wge(df$DriversKilled, phi.tr = c(0,0,0,0,0,0,0,0,0,0,0,1))
y = artrans.wge(df$DriversKilled, phi.tr = c(0,0,0,1))

plotts.sample.wge(y)

#AIC on transformed data s=4...picks AR9...s=12 picks 12 5
aic5.wge(y, p=0:15, q=0:10, type = "bic")

#Try differences...doesn't help so no d term
xd1.diff = artrans.wge(df$DriversKilled, phi.tr = 1)
xd2.diff = artrans.wge(xd1.diff, phi.tr = 1)

#With seasonal component s=12
x = est.arma.wge(df$DriversKilled, p = 1)
y = fore.aruma.wge(df$DriversKilled, s = 12, phi = x$phi, n.ahead = 15, lastn = TRUE, plot = TRUE, limits = TRUE)
ASE = mean((y$f-df$DriversKilled[178:192])^2)
ASE

#Without seasonal component
x = est.arma.wge(df$DriversKilled, p = 12, q=5)
y = fore.arma.wge(df$DriversKilled, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = TRUE, plot = TRUE, limits = TRUE)
ASE = mean((y$f-df$DriversKilled[178:192])^2)
ASE


###VAR
ccf(df$kms, df$DriversKilled, ylim = c(-1,1))

##TAke all except last 15
df_kms = df$kms[1:177]
df_deaths = df$DriversKilled[1:177]
all_data = data.frame(cbind(df$kms, df$DriversKilled))
names(all_data) <- c("kms","DriversKilled")

##chooses 13 1 or 15 0
p_df_kms = aic5.wge(df_kms, p=0:15, q=0:10, type = "bic")
p_df_kms
##chooses 12 0 or 11 2
p_df_deaths = aic5.wge(df_deaths, p=0:15, q=0:10, type = "bic")
p_df_deaths

#estimate and forecast kms Univariate Regression
kms.est = est.ar.wge(df_kms, p = 15)
fore.arma.wge(df_kms, phi = kms.est$phi, n.ahead = 15, lastn = FALSE, limits = TRUE)

#estimate and forecast kms Univariate Regression
deaths.est = est.ar.wge(df_deaths, p = 12)
fore.arma.wge(df_deaths, phi = deaths.est$phi, n.ahead = 15, lastn = FALSE, limits = TRUE)

#VAR SELECT chooses 14
X = cbind(df_kms, df_deaths)
VARselect(X, lag.max = 16, type = "const", season = NULL, exogen = NULL)
lsfit = VAR(X, p=14, type = 'const')
preds = predict(lsfit, n.ahead = 15)
preds

#ASE kms
preds$fcst$df_kms[1:15,1]
ASE = mean((preds$fcst$df_kms[1:15,1]-df$kms[178:192])^2)
ASE
#ASE deaths
preds$fcst$df_deaths[1:15,1]
ASE = mean((preds$fcst$df_deaths[1:15,1]-df$DriversKilled[178:192])^2)
ASE

#plot Deaths
plot(seq(1,192,1), df$DriversKilled, type = "b")
points(seq(178, 192, 1), preds$fcst$df_deaths[1:15,1], type = "b", pch = 15)
#plot kms
plot(seq(1,192,1), df$kms, type = "b")
points(seq(178, 192, 1), preds$fcst$df_kms[1:15,1], type = "b", pch = 15)
#fanchart
fanchart(preds)

##forecast using all the data
X = cbind(df$kms, df$DriversKilled)
VARselect(X, lag.max = 16, type = "const", season = NULL, exogen = NULL)
lsfit = VAR(X, p=14, type = 'const')
preds = predict(lsfit, n.ahead = 15)
preds
#predictions
prediction = data.frame(cbind(preds$fcst$y1[,"fcst"],preds$fcst$y2[,"fcst"]))
names(prediction) <- c("kms","DriversKilled")
forecast = rbind(all_data, prediction)

#forecasts for deaths
plot(seq(1,207,1), forecast$DriversKilled, type = "b")
points(seq(193, 207, 1), forecast$DriversKilled[193:207], type = "b", pch = 15)
#forecasts for kms
plot(seq(1,207,1), forecast$kms, type = "b")
points(seq(193, 207, 1), forecast$kms[193:207], type = "b", pch = 15)

x = est.arma.wge(df$DriversKilled, p = 1)
y = fore.aruma.wge(df$DriversKilled, s = 12, phi = x$phi, n.ahead = 15, lastn = FALSE, plot = TRUE, limits = TRUE)

##Neural Network
library(nnfor)
df_ts = ts(df$DriversKilled[1:177], start = c(1969,1), frequency = 12)
df_ts_test = ts(df$DriversKilled[178:192], start = c(1983,10), frequency = 12)
df_ts_full = ts(df$DriversKilled[1:192], start = c(1969,1), frequency = 12)

#only time as regressor
fit.mlp = mlp(df_ts, reps = 100)
fit.mlp
plot(fit.mlp)
f = forecast(fit.mlp, h=15)
plot(df$DriversKilled[178:192], type = "l")
lines(seq(1,15),f$mean, col = "blue")
ASE = mean((df_ts_test-f$mean)^2)
ASE



#with additional regressors
df_tsx = data.frame(kms = ts(df$kms), petrol = ts(df$PetrolPrice), drivers = ts(df$drivers))
fit2 = mlp(df_ts, xreg = df_tsx, reps = 100)
f2 = forecast(fit2, h = 15, xreg = df_tsx)
plot(df$DriversKilled[178:192], type = "l", ylim = c(60, 160))
lines(seq(1,15), f2$mean, col= "blue")
ASE = mean((df$DriversKilled[178:192]-f2$mean)^2)
ASE




#Without seasonal component ARMA(12,5)
x = est.arma.wge(df$DriversKilled, p = 12, q=5)
y = fore.arma.wge(df$DriversKilled, phi = x$phi, theta = x$theta, n.ahead = 15, lastn = FALSE, plot = TRUE, limits = TRUE)
