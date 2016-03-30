# data available on github...  https://github.com/CSG-kaggle/santander
# it consists of a train set, test set and sample submission
getwd()
#create a clean workspace
rm(list = ls())
#load data(
d1 <- read.csv("train.csv", stringsAsFactors = F)
dim(d1)
which(colnames(d1)=="TARGET")
#the last column is the outcome variable, #371
table(d1$TARGET)
#no missing class labels
#there are only 3008 "1s" in the dataset, so it is highly unbalanced
3008/76020 #only 4% 

library(caret)
#http://topepo.github.io/caret/index.html
vars <- d1[,1:370]
nzv <- nearZeroVar(vars, saveMetrics= TRUE)
#examine results
nzv #nzv column if "TRUE" equals a near zero variable
table(nzv$nzv)
table(nzv$zeroVar)
# we have only 53 potentially meaningful variables out of the full dataset as one is the ID
x <- vars[nzv$nzv==FALSE]
dim(x)
d2 = cbind(x,d1$TARGET)
dim(d2)
#save this for further analysis
write.csv(d2, "d2.csv", row.names=FALSE)
