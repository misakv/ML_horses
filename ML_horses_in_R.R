
library(readr)

setwd("C:/Users/misak/Desktop/konici_projekt/")

dataset <- read_delim("sedmileti/dataset_7let.csv", 
                            ";", escape_double = FALSE, col_types = cols(damLevel = col_integer(), 
                                 horseLevel = col_integer(), sireLevel = col_integer(), 
                                 sireOfdamLevel = col_integer()), 
                                 trim_ws = TRUE)


dataset$sqSire <- (dataset$sireLevel)^2
dataset$sqDam<- (dataset$damLevel)^2
dataset$sqSireOfdam <- (dataset$sireOfdamLevel)^2

##################summary statistics###############
length(dataset$horseName)


summary(dataset$horseLevel)
summary(dataset$sireLevel)
summary(dataset$damLevel)
summary(dataset$sireOfdamLevel)

##korelace
cor(dataset$horseLevel, dataset$sireLevel, use = "complete.obs" )
cor(dataset$horseLevel, dataset$damLevel, use = "complete.obs" )
cor(dataset$horseLevel, dataset$sireOfdamLevel, use = "complete.obs" )
##


#NAs
library(ggplot2)
library(naniar)

vis_miss(dataset[,6:8])

#levels big scale
par(mfrow=c(2,2))

hist(dataset$horseLevel, col = "red", breaks=5)
hist(dataset$sireLevel, col = "red", breaks=5)
hist(dataset$damLevel, col = "red", breaks=5)
hist(dataset$sireOfdamLevel, col = "red", breaks=5)

plot(density(dataset$horseLevel) , xlim = c(90, 140) , col = "red" , lwd=2 ,
     main="Kernel Density of Competition levels of young horses")

###RandomForrest imputing#####
##############################


#rfImpute
library(randomForest)
library(caTools)
library(hydroGOF)


imputedData <- tibble::as_tibble(
  randomForest::rfImpute(horseLevel ~ sireLevel + damLevel + sireOfdamLevel, ntree = 200, iter = 5, data = dataset)
) 



set.seed(123)
inximput <- sample.split(seq_len(nrow(imputedData)), 0.8)

fulldataTrain <- imputedData[inximput,]
fulldataTest <- imputedData[!inximput,]


##korelace
cor(imputedData$horseLevel, imputedData$sireLevel, use = "complete.obs" )
cor(imputedData$horseLevel, imputedData$damLevel, use = "complete.obs" )
cor(imputedData$horseLevel, imputedData$sireOfdamLevel, use = "complete.obs" )
##


#truncated
library(truncreg)
library(MLmetrics)

tri <- truncreg( horseLevel ~ sireLevel + damLevel + sireOfdamLevel , data = fulldataTrain, scaled = F, point = 79, direction = "left")
summary(tri)

predictionsTrun1 <- predict(tri, newdata=fulldataTrain) 
rmse(fulldataTrain$horseLevel, predictionsTrun1, na.rm=TRUE)

predictionsTrun2 <- predict(tri, newdata=fulldataTest)
rmse(fulldataTest$horseLevel, predictionsTrun2, na.rm=TRUE)


library(ggplot2)
library(dplyr)
library(mgcv)
library(tidymv)
library(tidyverse)
library(cowplot)
library(modelsummary)


par(mfrow=c(1,2))

plot(predictionsTrun1,fulldataTrain$horseLevel,
     xlab="predicted",ylab="actual", ylim = c(70,150), main = "Train dataset")
abline(a=0,b=1, col='red',  lwd=3)
abline(a=11,b=1, col='blue', lty=2, lwd=3)
abline(a=-11,b=1, col='blue', type='dotdash', lty=2,  lwd=3)


plot(predictionsTrun2,fulldataTest$horseLevel,
     xlab="predicted",ylab="actual", ylim = c(70,150), main = "Test dataset")
abline(a=0,b=1, col='red',  lwd=3)
abline(a=11,b=1, col='blue', lty=2, lwd=3)
abline(a=-11,b=1, col='blue', type='dotdash', lty=2,  lwd=3)




#RF
lapply(fulldataTrain, class)

summary(rf<- randomForest(
  horseLevel ~ sireLevel + damLevel + sireOfdamLevel,
  data=fulldataTrain , ntree=1000 ))


predictionsTree1 <- predict(rf, newdata=fulldataTrain) 

rmse(fulldataTrain$horseLevel, predictionsTree1, na.rm=TRUE)
#rmse 7.29 !


predictionsTree2 <- predict(rf, newdata=fulldataTest) 

rmse(fulldataTest$horseLevel, predictionsTree2, na.rm=TRUE)
#rmse 11.53 !

par(mfrow=c(1,2))

plot(predictionsTree1,fulldataTrain$horseLevel,
     xlab="predicted",ylab="actual", ylim = c(70,150), main = "Train dataset")
abline(a=0,b=1, col='red',  lwd=3)
abline(a=7.29,b=1, col='blue', lty=2, lwd=3)
abline(a=-7.29,b=1, col='blue', type='dotdash', lty=2,  lwd=3)

plot(predictionsTree2,fulldataTest$horseLevel,
     xlab="predicted",ylab="actual", ylim = c(70,150), main = "Train dataset")
abline(a=0,b=1, col='red',  lwd=3)
abline(a=11.53, b=1, col='blue', lty=2, lwd=3)
abline(a=-11.53, b=1, col='blue', type='dotdash', lty=2,  lwd=3)

#######################PREMIUM DETECTION##################
##########################################################
library("kernlab")




fulldataTrain$Premium <- ifelse(fulldataTrain$horseLevel >=125, 1, 0)
fulldataTest$Premium <- ifelse(fulldataTest$horseLevel >=125, 1, 0)

############Logity########

model<-glm(Premium ~ sireLevel + damLevel + sireOfdamLevel , family=binomial(link="logit"),
           data=fulldataTrain)

fulldataTest$pred<-predict(model,newdata=fulldataTest,
                       type="response")

predikceLogit <- print(with(fulldataTest,table(y=Premium,glPred=pred>=0.4)))



# glPred
# y   FALSE TRUE
# 0    22    5
# 1     5    4


##############SVM - Test dataset

#rbsfot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="rbfdot", C=10)

fulldataTest$Kernel<- predict(mSVMV, newdata=fulldataTest,type="response")
TableSVM <- with(fulldataTest,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTest,table(y=Premium,glPred=Kernel>=0.30)))



#vanilladot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="vanilladot", C=10)


fulldataTest$Kernel<- predict(mSVMV, newdata=fulldataTest,type="response")
TableSVM <- with(fulldataTest,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTest,table(y=Premium,glPred=Kernel>=0.0466)))


#polydot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="polydot", C=10)


fulldataTest$Kernel<- predict(mSVMV, newdata=fulldataTest,type="response")
TableSVM <- with(fulldataTest,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTest,table(y=Premium,glPred=Kernel>=0.0466)))




##############SVM - Train dataset

#rbsfot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="rbfdot", C=10)

fulldataTrain$Kernel<- predict(mSVMV, newdata=fulldataTrain,type="response")
TableSVM <- with(fulldataTrain,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTrain,table(y=Premium,glPred=Kernel>=0.30)))



#vanilladot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="vanilladot", C=10)


fulldataTrain$Kernel<- predict(mSVMV, newdata=fulldataTrain,type="response")
TableSVM <- with(fulldataTrain,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTrain,table(y=Premium,glPred=Kernel>=0.0466)))


#polydot
mSVMV<-ksvm(Premium ~ sireLevel + damLevel + sireOfdamLevel , data=fulldataTrain,
            kernel="polydot", C=10)


fulldataTrain$Kernel<- predict(mSVMV, newdata=fulldataTrain,type="response")
TableSVM <- with(fulldataTrain,table(y=Premium,Kernel=Kernel))

predikceSVM <- print(with(fulldataTrain,table(y=Premium,glPred=Kernel>=0.0466)))



