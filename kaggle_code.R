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

######### Part 2, univariate feature elimination

#load d2 from github.
library(RCurl) 
library(foreign)
url <- "https://raw.githubusercontent.com/CSG-kaggle/santander/master/d2.csv"
d2 <- getURL(url) #this takes a few seconds
d2 <- read.csv(textConnection(d2))
#With the data unbalanced and also its size, I think we should create a smaller set
#to do the feature elimination
library(dplyr)
colnames(d2)[55] <- "TARGET" #for some reason when I did the cbind it messed up the colname
tgt1 <- filter(d2, TARGET==1) 
dim(tgt1) #3008 by 55
tgt2 <- filter(d2, TARGET==0)
dim(tgt2) #73012 x 55
#sample 10,000 observations from tgt2
set.seed(123)
tgt3 <- sample_n(d2,10000, replace=F)
dim(tgt3) #10,000 by 55
d3 <- rbind(tgt1,tgt3)
dim(d3)
names(d3)
#run the boruta algorithm on a few variables at a time because of the computation time. It also helps to view the plots
library(Boruta)
set.seed(45)
fs1 <- Boruta(TARGET~var15+imp_op_var39_comer_ult3+imp_op_var41_comer_ult3+imp_op_var41_ult1
              +imp_op_var39_ult1+ind_var5+ind_var12_0+ind_var13_0 , data=d3, doTrace=2)
fs1
plot(fs1)
set.seed(45)
fs2 <- Boruta(TARGET~ind_var13+ind_var30+ind_var37_cte+ind_var37_0+ind_var37
              +ind_var39_0+ind_var41_0+num_var4+num_var5, data=d3, doTrace=2)
fs2
plot(fs2)
set.seed(45)
fs3 <- Boruta(TARGET~num_var12_0+num_var30_0+num_var30+num_var35+num_var39_0+num_var41_0+
                num_var42_0+num_var42+saldo_var5+saldo_var30, data=d3, doTrace=2)
fs3
plot(fs3)
set.seed(45)
fs4 <- Boruta(TARGET~saldo_var42+var36+ind_var10_ult1+ind_var10cte_ult1+ind_var9_cte_ult1
              +ind_var9_ult1+ind_var43_emit_ult1+ind_var43_recib_ult1
              +num_var22_hace2+num_var22_hace3+num_var22_ult1, data=d3, doTrace=2)
fs4
plot(fs4)
set.seed(45)
fs5 <- Boruta(TARGET~num_var22_ult3 + num_med_var22_ult3+num_med_var45_ult3
              + num_meses_var5_ult3 + num_meses_var39_vig_ult3 + num_var43_recib_ult1
              + num_var45_ult3 + saldo_medio_var5_hace2 + saldo_medio_var5_hace3
              +saldo_medio_var5_ult1 + saldo_medio_var5_ult3 + var38, data=d3, doTrace=2)
fs5
plot(fs5)

#with no other rejected features, let's examine a heatmap of the full data
#spearman correlation to account for likely non-linearity
dataCor <- as.matrix(cor(d2[,2:54]), method="Spearman")
col <- colorRampPalette(c("red", "white", "green"))(20)
heatmap(x = dataCor, col = col, symm = TRUE)
#a few clusters of correlated variables
