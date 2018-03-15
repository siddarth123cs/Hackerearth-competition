rm(list=ls(all=T))
setwd("C:\\Users\\SIDDARTH\\Desktop\\hacker earth\\Annual Returns")
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


#finding skewness and density
train$return=log(train$return+0.0099)
test=read.csv("test.csv",na.strings=c("","NA"),stringsAsFactors = F)
data=rbind(train[,-18],test)

s=(data[,2])
s1 = unlist(strsplit(s, split='DSK0000', fixed=T))[]
a=as.numeric(s1)
data[is.na(data$desk_id)==F,2]=a[is.na(a)==F]
data[,2]=as.numeric(data[,2])
s2=(data[,1])
s3 = unlist(strsplit(s2, split='PF000', fixed=T))[]
a=as.numeric(s3)
data[,1]=a[is.na(a)==F]

impute=data[,c(1,2,5)]

new=centralImputation(impute)
data[,c(1,2,5)]=new
data$desk_id=as.integer(data$desk_id)
#converting start_data
data$start_year=(as.integer(data$start_date/10000))
data$start_month=(as.integer((data$start_date%%10000)/100))
data$start_day=(as.integer(data$start_date%%100))
data$start_date_new=as.Date(paste(data$start_month,data$start_day,data$start_year,sep = "/"),format = "%m/%d/%Y")


#converting creation_date
data$creation_year=(as.integer(data$creation_date/10000))
data$creation_month=(as.integer((data$creation_date%%10000)/100))
data$creation_day=(as.integer(data$creation_date%%100))
data$creation_date_new=as.Date(paste(data$creation_month,data$creation_day,data$creation_year,sep = "/"),format = "%m/%d/%Y")

#converting sell_date
data$sell_year=(as.integer(data$sell_date/10000))
data$sell_month=(as.integer((data$sell_date%%10000)/100))
data$sell_day=(as.integer(data$sell_date%%100))
data$sell_date_new=as.Date(paste(data$sell_month,data$sell_day,data$sell_year,sep = "/"),format = "%m/%d/%Y")

#subtracting all the dates
data$sell_start=data$sell_date-data$start_date
data$sell_creation=data$creation_date-data$sell_date
data$creation_start=data$creation_date-data$start_date
data$sell_start2=data$sell_date+data$start_date
data$sell_creation2=data$creation_date+data$sell_date
data$creation_start2=data$creation_date+data$start_date


#removing unwanted columns
data=data[,setdiff(names(data),c("portfolio_id","start_date","creation_star","creation_date","sell_date","start_day","start_date_new","creation_day","creation_date_new","sell_day","sell_date_new","sell_month","creation_month","start_month","sell_creation2","creation_start2"))]

#removing Na in libor rate by converting them to '0' as they arent thr in places other than LONDON
data$libor_rate[is.na(data$libor_rate)]=0

#filling NAs in indicator and status with false
data$indicator_code[is.na(data$indicator_code)]=FALSE
data$indicator_code=as.factor(ifelse(data$indicator_code==FALSE,0,1))
data$status[is.na(data$status)]=FALSE
data$status=as.factor(ifelse(data$status==FALSE,0,1))



data=data[is.na(data$sold)==F,]

data$sold=log(data$sold)^4

data$bought=log(data$bought)^4
data$profit=data$bought-data$sold

data$profit=log((data$profit- min(data$profit)+1))^2

str(data)
num_attr<-c("desk_id","sold","euribor_rate","bought","libor_rate","profit","sell_start2","sell_start","sell_creation")
cat_attr<-setdiff(x=names(data),y=num_attr)
cat_data<-data.frame(sapply(data[,cat_attr],as.factor))
num_data<-data.frame(sapply(data[,num_attr],as.numeric))
data2 = cbind(num_data, cat_data)
data2=data2[,setdiff(names(data2),c("sold","bought","sell_year","start_year","creation_year"))]



data2$hedge_value[is.na(data2$hedge_value)==T]=TRUE

finaldata=data2[1:9364,]
finaldata$return=train[is.na(train$sold)==F,18]
cor(finaldata$return,finaldata$creation_start2)
finaltest=data2[9365:14165,]


library(xgboost)
a=createDataPartition(y=finaldata$return,p=1,list = F)
finaltrain=finaldata[a,]
str(finaltrain)
finaltest2=finaldata[-a,]

xgb.data.train <- xgb.DMatrix(data.matrix(finaltrain[, colnames(finaltrain) != "return"]), label = finaltrain$return)
xgb.data.test <- xgb.DMatrix(data.matrix(finaltest2[, colnames(finaltest2) != "return"]), label = finaltest2$return)



xgb.model.speed <- xgb.train(data = xgb.data.train
                             , params = list(booster="gbtree",
                               objective = "reg:linear"
                                             , eta = 0.075
                                             
                                             , max.depth =7
                                             , min_child_weight = 95
                                             , subsample = 0.65
                                             , colsample_bytree = 0.55
                                             , nthread = 3
                                             , eval_metric = "rmse"
                               
                                             ,colsample_bylevel=1
                              ,predictor='cpu_predictor'
                             )
                            # , watchlist = list(test = xgb.data.test)
                             , nrounds = 20000
                            # , early_stopping_rounds = 10000
                            # , print_every_n = 100
)
xgb.feature.imp = xgb.importance(model = xgb.model.speed)
save(xgb.model.speed,file="bestmodel")
load(file = "bestmodel")
print(xgb.feature.imp)
predict=predict(xgb.model.speed,xgb.data.test)
regr.eval(finaltest2$return,predict)
pred2=predict(xgb.model.speed,data.matrix(finaltest))
pred2_org=exp(pred2)-(0.0099)
pred_xgb=predict(xgb.model.speed,data.matrix(finaldata))

write.csv(x=pred2_org,file = "try.csv")


#SVM
a=createDataPartition(y=finaldata$return,p=1,list = F)
finaltrain=finaldata[a,]
str(finaltrain)
finaltest2=finaldata[-a,]
modelsvm = svm(return~.,finaltrain,gamma=1/5,cost=72,epsilon=0.05)
save(modelsvm,file="bestsvmmodel")
load("bestsvmmodel")
summary(modelsvm)
#OptModelsvm=tune(svm,return~.,data=finaltrain,ranges=list(elsilon=seq(0,1,0.1), cost=1:100))
#BstModel=OptModelsvm$best.model
predsvm=predict(modelsvm,finaltest2)
regr.eval(finaltest2$return,predsvm)
pred2svm=predict(modelsvm,(finaltest))
pred2svm_org=exp(pred2svm)-(0.0099)
pred_svm=predict(modelsvm,finaldata)
write.csv(x=pred2svm_org,file = "try.csv")

library(h2o)
h2o.init()
finaldata.h2o=as.h2o(finaldata)
finaltest.h2o=as.h2o(finaltest)
finaltrain=finaldata.h2o[a,]
y.dep<-16
x.indep<-1:15
rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = finaldata.h2o, ntrees = 151, mtries = 6, max_depth = 4, seed = 1122,nfolds=5,binomial_double_trees = T,balance_classes = T)
h2o.performance(rforest.model)
predict.dl2_random <- as.data.frame(h2o.predict(rforest.model, finaltest.h2o))
predict.dl2_org_random=exp(predict.dl2_random)-(0.0099)
pred_random=as.data.frame(h2o.predict(rforest.model,finaldata.h2o))
write.csv(x=predict.dl2_org_random,file="try.csv")
save(rforest.model,file="rfbestmodel")
load("rfbestmodel")

gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = finaldata.h2o, ntrees = 551, max_depth = 7, learn_rate = 0.151, seed = 1122,sample_rate = 0.85,col_sample_rate=0.8,min_split_improvement=1e-5,nfolds = 5)
h2o.performance(gbm.model)
predict.dl2 <- as.data.frame(h2o.predict(gbm.model, finaltest.h2o))
predict.dl2_org=exp(predict.dl2)-(0.0099)
pred_gbm=as.data.frame(h2o.predict(gbm.model,finaldata.h2o))
save(gbm.model,file = "gbmbest")
load("gbmbest")

write.csv(x=predict.dl2,file="try.csv")

dlearning.model <- h2o.deeplearning(y = y.dep,
                                    x = x.indep,
                                    training_frame = finaldata.h2o,
                                    epoch = 160,
                                    distribution = "huber",
                                    nfolds =0,
                                    hidden = c(250,250),
                                    activation = "Rectifier",
                                    seed = 1122)
h2o.performance(dlearning.model)
predict.dl2_dp <- as.data.frame(h2o.predict(dlearning.model, finaltest.h2o))
predict.dl2_dp_org=exp(predict.dl2_dp)-(0.0099)
pred_dp=as.data.frame(h2o.predict(dlearning.model,finaldata.h2o))
write.csv(x=predict.dl2,file="try.csv")
save(dlearning.model,file = "bestdlmodel")


#stacking
train_Pred_All_Models = data.frame(
                                   xgb = (pred_xgb),
                                   random = (pred_random),
                                   deeplearnin=(pred_dp),
                                   svm=(pred_svm),
                                   gbm=(pred_gbm))
train_Pred_All_Models = data.frame(sapply(train_Pred_All_Models, 
                                          as.numeric))
train_Pred_All_Models = cbind(train_Pred_All_Models, return = (finaldata$return))
train_Pred_All_Models.h2o=as.h2o(train_Pred_All_Models)

#xgboost with stacking
a=createDataPartition(y=train_Pred_All_Models$return,p=1,list = F)
finaltrain=train_Pred_All_Models[a,]
str(finaltrain)
finaltest2=train_Pred_All_Models[-a,]

xgb.data.train <- xgb.DMatrix(data.matrix(finaltrain[, colnames(finaltrain) != "return"]), label = finaltrain$return)
xgb.data.test <- xgb.DMatrix(data.matrix(finaltest2[, colnames(finaltest2) != "return"]), label = finaltest2$return)



xgb.model.speed.ensemble <- xgb.train(data = xgb.data.train
                             , params = list(booster="gbtree",
                                             objective = "reg:linear"
                                             , eta = 0.15
                                             ,nfolds=5
                                             , max.depth =4
                                             , min_child_weight = 2.9
                                             , subsample = 0.9
                                             , colsample_bytree = 1
                                             , nthread = 3
                                             , eval_metric = "rmse"
                                            # ,distribution="huber"
                                             ,colsample_bylevel=1
                                             #  ,predictor='cpu_predictor'
                             )
                             # , watchlist = list(test = xgb.data.test)
                             , nrounds = 450
                             # , early_stopping_rounds = 3000
                             #, print_every_n = 100
)



test_Pred_All_Models = data.frame(
                                  xgb = pred2,
                                  random = predict.dl2_random,
                                  deeplearning=predict.dl2_dp,
                                  svm=pred2svm,
                                  gbm=predict.dl2) 
test_Pred_All_Models = data.frame(sapply(test_Pred_All_Models, as.numeric))
test_Pred_All_Models=data.matrix(test_Pred_All_Models)
ensemble_pred = predict(xgb.model.speed.ensemble,test_Pred_All_Models)
pred_Test4=exp(ensemble_pred)-(0.0099)
write.csv(x=pred_Test4,file = "try.csv")

