data("AirPassengers")
AP <- AirPassengers
AP
class(AP)
start(AP); end(AP); frequency(AP)
summary(AP)
plot(AP,ylab= "Passengers(1000's)",xlab="Time")

layout(1:2)
plot(aggregate(AP))
boxplot(AP ~ cycle(AP))

#if I am accessing github pages, I require the raw data
www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/Maine.dat"
Maine.month <- read.table(www, header=TRUE)
attach(Maine.month)
class(Maine.month)

Maine.month.ts <- ts(unemploy, start = c(1996,1), freq = 12)

Maine.annual.ts <-aggregate(Maine.month.ts)/12

layout(1:2)
plot(Maine.month.ts,ylab="unemployed (%)")
plot(Maine.annual.ts, ylab="unemployed (%)")

Maine.Feb <- window(Maine.month.ts, start = c(1996,2), freq = TRUE)
Maine.Aug <- window(Maine.month.ts, start = c(1996,8), freq = TRUE)

Feb.ratio <- mean(Maine.Feb)/ mean(Maine.month.ts)
Aug.ratio <- mean(Maine.Aug)/ mean(Maine.month.ts)

Feb.ratio
Aug.ratio

www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/USunemp.dat"
US.month <-read.table(www, header = T)
attach(US.month)
US.month.ts <-ts(USun, start= c(1996,1), end = c(2006,10), freq= 12)
plot(US.month.ts, ylab="Unemployed (%)")

www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/cbe.dat"
CBE <- read.table(www, header = T)

CBE[1:4, ]
class(CBE)

Elec.ts <- ts(CBE[,3], start = 1958, freq = 12)
Beer.ts <- ts(CBE[,2], start = 1958, freq = 12)
Choc.ts <-ts(CBE[,1], start = 1958, freq = 12)
plot(cbind(Elec.ts, Beer.ts, Choc.ts))

AP.elec <- ts.intersect(AP, Elec.ts)

start(AP.elec)

end(AP.elec)

AP.elec[1:3,]

AP <-AP.elec [,1]; Elec <- AP.elec[,2]

layout(1:2)
plot(AP, main = "", ylab = "Air passengers/ 100's")
plot(Elec, main="", ylab="Electricity proudction/MkWh")
plot(as.vector(AP), as.vector(Elec),
     xlab = "Air passengers / 1000's",
     ylab = "Electricity production /MWh")
abline(reg = lm(Elec ~ AP))

cor(AP, Elec)

www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/pounds_nz.dat"
Z <-read.table(www, header = T)

Z[1:4, ]

Z.ts <- ts(Z, st= 1991, fr=4)

plot(Z.ts, xlab = "time/years",
     ylab = "Quarterley exchange rate in $NZ / pound")

Z.92.96 <- window(Z.ts, start = c(1992,1), end = c(1996,1))
Z.96.98 <- window(Z.ts, start = c(1996,1), end = c(1998,1))

layout(1:2)
plot(Z.92.96, ylab = "Exchange rate in $NZ/pound",
     xlab = "Time (years)")
plot(Z.96.98, ylab = "Exchange rate in $NZ/pound",
     xlab = "Time (years)")

www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/global.dat"
Global <- scan(www)

Global.ts <- ts(Global, st = c(1856,1), end = c(2005,12),
                fr = 12)
Global.annual <- aggregate(Global.ts, FUN = mean)
plot(Global.ts)
plot(Global.annual)

New.series <- window(Global.ts, start = c(1970,1), end = c(2005,12))
New.time <-time(New.series)
plot(New.series); abline(reg = lm(New.series ~ New.time))

plot(decompose(Elec.ts))
Elec.decom <- decompose(Elec.ts, type = "multi")
plot(Elec.decom)


Trend <- Elec.decom$trend
Seasonal <- Elec.decom$seasonal
ts.plot(cbind(Trend, Trend * Seasonal), lty = 1:2)

www <- "https://raw.githubusercontent.com/dallascard/Introductory_Time_Series_with_R_datasets/refs/heads/master/Herald.dat"
Herald.dat <- read.table(www, header = T)
attach(Herald.dat)

x <- CO; y <- Benzoa; n <- length(x)
sum((x - mean(x))*(y - mean(y))) / (n-1) 

mean((x - mean(x)) * (y - mean(y)))

cov(x, y)
cov(x, y)/ (sd(x)* sd(y))
    
cor(x,y)
