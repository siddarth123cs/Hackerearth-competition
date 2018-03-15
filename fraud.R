rm(list=ls(all=T))
setwd("C:\\Users\\SIDDARTH\\Desktop\\hacker earth\\fraudulent trans")
library(glmnet)
library(caret)
library(MASS)
library(vegan)
library(data.table)
library(doParallel)
library(DMwR)
library(dummies)
library(e1071)
registerDoParallel(10)
train=read.csv("train.csv",na.strings=c("","NA"),stringsAsFactors = F)
str(train)
colSums(is.na(train))
table(train[,51])
table(train$cat_var_3)
test=read.csv("test.csv",na.strings=c("","NA"),stringsAsFactors = F)
colSums(is.na(test))
data=rbind(train[,-51],test)

data$num_var_1=ifelse(data$num_var_1==0,0,log(data$num_var_1))
skewness((data$num_var_1))

data$num_var_2=ifelse(data$num_var_2==0,0,log(data$num_var_2))
skewness(data$num_var_2)


data$num_var_4=ifelse(data$num_var_4==0,0,log(data$num_var_4))
skewness(data$num_var_4)

data$num_var_5=ifelse(data$num_var_5==0,0,log(data$num_var_5))
skewness(data$num_var_5)

data$num_var_6=ifelse(data$num_var_6==0,0,log(data$num_var_6))
skewness(data$num_var_6)

data$num_var_7=ifelse(data$num_var_7==0,0,log(data$num_var_7))
skewness(data$num_var_7)
data_imp=centralImputation((data))
colSums(is.na(data_imp))
str(data_imp)
num_attr<-c("num_var_1","num_var_2","num_var_3","num_var_4","num_var_5","num_var_6","num_var_7")
cat_attr<-setdiff(x=names(data_imp),y=num_attr)
cat_data<-data.frame(sapply(data_imp[,cat_attr],as.factor))
num_data<-data.frame(sapply(data_imp[,num_attr],as.numeric))
data2 = cbind(num_data, cat_data)
str(data2)
finaldata=data2[1:348978,setdiff(names(data2),c("transaction_id","cat_var_42","cat_var_38","cat_var_37","cat_var_40", "cat_var_36", "cat_var_31", "cat_var_35"))]
finaldata$target=train$target
finaltest=data2[348979:872444,setdiff(names(data2),c("transaction_id","cat_var_42","cat_var_38","cat_var_37","cat_var_40", "cat_var_36", "cat_var_31", "cat_var_35"))]

write.csv(finaldata,file="finaldata.csv")
write.csv(finaltest,file="finaltest.csv")


str(data2)
a=createDataPartition(y=finaldata$target,p=1,list = F)
finaltrain=finaldata[a,]
str(finaltrain)
finaltest2=finaldata[-a,]
#svm
modelsvm = svm(target~.,finaltrain,gamma=1/5,cost=72,epsilon=0.1)
predsvm=predict(modelsvm,finaltest2)
rm(x)
#h2o
library(h2o)
h2o.init()
finaldata[,43]=as.factor(finaldata[,43])
finaldata.h2o=as.h2o(finaldata)
finaltest.h2o=as.h2o(finaltest)
y.dep<-43
x.indep<-1:42

#random forest
rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = finaldata.h2o, ntrees = 151, mtries = 6, max_depth = 25, seed = 1122,col_sample_rate_per_tree = 0.9,binomial_double_trees = T,balance_classes = T,nfolds=5)
save(rforest.model,file="bestrfmodel")
load("bestrfmodel")
perf=h2o.performance(rforest.model)
h2o.auc(perf)
predict.dl2_random <- as.data.frame(h2o.predict(rforest.model, finaltest.h2o))
pred_random=as.data.frame(h2o.predict(rforest.model,finaldata.h2o))
pred_random[,1]=ifelse(pred_random[,1]==1,1,0)

write.csv(x=predict.dl2_random,file="try.csv")

#gbm
gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = finaldata.h2o, ntrees = 151, max_depth = 3, learn_rate = 0.151, seed = 1122,sample_rate = 0.8,col_sample_rate=1,min_split_improvement=1e-5,nfolds=5)
h2o.auc(h2o.performance(gbm.model))
predict.dl2 <- as.data.frame(h2o.predict(gbm.model, finaltest.h2o))
pred_gbm=as.data.frame(h2o.predict(gbm.model,finaldata.h2o))
pred_gbm[,1]=ifelse(pred_gbm[,1]==1,1,0)
write.csv(x=predict.dl2,file="try.csv")

#deep learning
dlearning.model <- h2o.deeplearning(y = y.dep,
                                    x = x.indep,
                                    training_frame = finaldata.h2o,
                                    epoch = 75,
                                    distribution = "bernoulli",
                                    nfolds = 5,
                                    hidden = c(150,150),
                                    
                                    
                                    activation = "Rectifier",
                                    seed = 1122)


h2o.auc(h2o.performance(dlearning.model))
predict.dl2_dp <- as.data.frame(h2o.predict(dlearning.model, finaltest.h2o))
write.csv(x=predict.dl2_dp,file="try.csv")
pred_dl=as.data.frame(h2o.predict(dlearning.model,finaldata.h2o))
pred_dl[,1]=ifelse(pred_dl[,1]==1,1,0)



#lasso
x=data.matrix(finaldata[,setdiff(names(finaldata),"target")])
colSums(is.na(balancedData))
str(balancedData)
Model = cv.glmnet(x,finaldata$target, type.measure="auc", 
                  family="binomial",nfolds=10,parallel=TRUE)
pred_test=predict(Model,data.matrix(finaltest),type="response")
pred_test_org=ifelse(pred_test>0.5,1,0)
pred_lass=predict(Model,x,type="response")
pred_lass=ifelse(pred_lass>0.5,1,0)
typeof(pred_test)
write.csv(x=pred_test,file = "try.csv")

#xgboost
library(xgboost, quietly=TRUE)
a=createDataPartition(y=finaldata$target,p=1,list = F)
finaldata$target=ifelse(as.numeric(finaldata$target)==1,1,0)
finaltrain=finaldata[a,]
str(finaltrain)
finaltest2=finaldata[-a,]
xgb.data.train <- xgb.DMatrix(data.matrix(finaldata[, colnames(finaldata) != "target"]), label = finaldata$target)
xgb.data.test <- xgb.DMatrix(data.matrix(finaltest2[, colnames(finaltest2) != "target"]), label = finaltest2$target)

xgb.model.speed <- xgb.train(data = xgb.data.train
                               , params = list( booster="gbtree",
                                                objective = "binary:logistic"
                                               , eta = 0.09
                                               , max.depth = 4
                                               , min_child_weight = 100
                                               , subsample = 1
                                               , colsample_bytree = 0.7
                                               , nthread = 2
                                               ,nfolds=5
                                               , eval_metric = "auc"
                                               
                               )
                             #  , watchlist = list(test = xgb.data.test)
                               , nrounds = 500
                              # , early_stopping_rounds = 400
                               #, print_every_n = 20
  )
  

print(xgb.model.speed)
xgb.feature.imp = xgb.importance(model = xgb.model.speed)
pred=predict(xgb.model.speed,data.matrix(finaltest), ntreelimit = xgb.model.speed$bestInd)
pred_xgb=predict(xgb.model.speed,xgb.data.train)
plot(roc(finaldata$target,pred_xgb))
typeof(pred)
write(x=(1-pred),file="try.csv",sep="\n")
a=data.frame(pred)
xgb.feature.imp = xgb.importance(model = xgb.model.hist)



#stacking
xgb = 1-pred
random = predict.dl2_random[,3]
deeplearnin=predict.dl2_dp[,3]
gbm=predict.dl2[,3]
finalpred=(xgb+gbm+deeplearnin+random)/4
save(finalpred,file="bestROC")
load("bestROC")
write.csv(x=finalpred,file ="try.csv")








