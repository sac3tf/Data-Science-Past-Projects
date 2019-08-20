library(tswge)
##create time series object
swatrain = ts(swa$arr_delay[1:141], start = c(2004,1), frequency = 12)
swatest = ts(swa$arr_delay[142:177], start = c(2015, 10), frequency = 12)
set.seed(2)

install.packages('nnfor',repos='http://cran.us.r-project.org')
available.packages()
if (!require("devtools")){install.packages("devtools")}
library(nnfor)
fit.mlp = mlp(swatrain, reps = 50, comb = "mean")
fit.mlp
plot(fit.mlp)
fore.mlp = forecast(fit.mlp, h=36)
ASE = mean((swatest - fore.mlp$mean)^2)
ASE
?mlp

fit.mlp = mlp(swatrain, difforder = c(1,6,12), allow.det.season = FALSE, reps = 100)


install.packages("~/Downloads/r-cran-plotrix_3.7-1.orig.tar", type = "source", repos = NULL)
