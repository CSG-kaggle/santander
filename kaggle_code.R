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
#also check for missing values
d2[!complete.cases(d2),]
#no missing values!!!
#correlation distribution
summary(dataCor[upper.tri(dataCor)])
#we have some variables with 100% correlation
highCorr <- sum(abs(dataCor[upper.tri(dataCor)]) > .95)
highCorr
#24 have cor above .9 and 4 above .99 and 11 above .95
#doc claims not an issue with boosted trees - 
#http://xgboost.readthedocs.org/en/latest/R-package/discoverYourData.html?highlight=correlation
#issues with indicator variables?
table(d2$ind_var30, d2$ind_var5)
#anyway, try eliminating cor above 0.95...maybe do PCA?
highlyCorData <- findCorrelation(dataCor, cutoff = .99)
highlyCorData #this gives you the column numbers that are high Cor with others for removal
d3 <- d2[,-highlyCorData]
dim(d3) #76020 x 51
names(d3)
#drop ID
d3 <- d3[,-1]

###XGBOOST on d3
#save d3
write.csv(d3, "d3.csv", row.names=F)
library(xgboost)
library(caret)
#OK, now we have to deal with unbalanced response/outcome
#try full data and also SMOTE to see what works best
d3 <- read.csv("d3.csv")
#Full data
#set.seed(123)
#trainIndex <- createDataPartition(d3$TARGET, p = .7,
          #list = FALSE,times = 1)
#Train <- as.matrix(d3[ trainIndex,])
#Validate  <- as.matrix(d3[-trainIndex,])
#xgboost with no hyperparameter tuning
#documentation recommends putting data in a class matrix
d3$var15_30 <- d3$var15*d3$saldo_var30
d3$var15_38 <- d3$var15*d3$var38
d3$var30_38 <- d3$saldo_var30*d3$var38
hist(d3$var15)
full <- as.matrix(d3)
full <- full[,c(1:49,51:53,50)]
dtrain <- xgb.DMatrix(full[,c(1:52)], label=full[,53])
#dval <- xgb.DMatrix(Validate[,1:49], label=Validate[,50])
weight<- 1/(3008/73012)
weight
model1 <- xgboost(data=dtrain,
                scale_pos_weight=weight,
                nround=300,
                max.depth = 6, 
                eta = 0.1, #0.01,.0001,.001,.005
                subsample = 0.5,
                objective="binary:logistic",
                eval.metric="auc")
pred <- predict(model1, full)
summary(pred)
prediction <- as.numeric(pred > 0.5)
table(full[,53],prediction)
impMatrix <- xgb.importance(feature_names=colnames(full),model=model1)
impMatrix #var15, saldo_var30, var38
#library(Ckmeans.1d.dp)
xgb.plot.importance(impMatrix)
#ROC and other model performance measures
library(ROCR)
roc_pred <- prediction(pred, full[,53])
performance(roc_pred, "auc")
roc_perf <- performance(roc_pred,"tpr","fpr")
plot(roc_perf, col=rainbow(10))
fScore <- performance(roc_pred, "f")
plot(fScore)



#hybrid sampling for trainControl
library(DMwR)
t <- d3
#SMOTE requires response as factor
t$TARGET <- as.factor(t$TARGET)
set.seed(456)
smote_train <- SMOTE(TARGET ~ ., data=t,perc.over=250,perc.under=175)
table(smote_train$TARGET) #10528 x 9024
str(smote_train)
x <- as.matrix(smote_train[,c(1:49,51,52,53)])
y <- ifelse(smote_train[,50]=="1",1,0)
sm_train <- cbind(x,y)
#tune parameters
# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = c(300,500),
  eta = 0.1,#.3 is default,
  max_depth = 6
)
xgb_grid_1
# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        
  classProbs=TRUE,
  summaryFunction = twoClassSummary
)
#df3 into a matrix
x <- as.matrix(smote_train[c(1:49,51:53)])
y <- as.factor(ifelse(smote_train$TARGET==1,"lost","maintained"))
table(y)
# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train1 = train(
  x = x,
  y = y,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)
xgb_train1
c# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")


#load test set
test <- read.csv("test.csv", stringsAsFactors = F)
dim(test)
#create interaction terms
test$var15_30 <- test$var15*test$saldo_var30
test$var15_38 <- test$var15*test$var38
test$var30_38 <- test$saldo_var30*test$var38
#create reduced test set and drop TARGET
myvars <- names(d3[,-50]) #%in% c("v1", "v2", "v3") 
myvars
#vars <- paste(myvars, collapse=",")
testdata <- test[,colnames(test)%in%myvars] 
dim(testdata)
names(testdata)
#turn to matrix
testdata <- as.matrix(testdata)
#test set predictions
predTest <- predict(model1, testdata)
summary(predTest)
predictionTest <- as.numeric(predTest > 0.5)
table(predictionTest)
#create .csv for submission
sub1 <- data.frame(cbind(test$ID,predictionTest))
dim(sub1)
names(sub1) <- c("ID","TARGET")
head(sub1)
str(sub1)
sub1$ID <- as.character(sub1$ID)
write.csv(sub1, "sub1.csv", row.names=F)
