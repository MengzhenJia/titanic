#------setup-----
#20210608titanic
setwd("d:/research/projects/titanic/")
install.packages("MLmetrics",destdir = "d://Application//RStudio//packages" )
library(dplyr)
library(tidyverse)
library(kableExtra)
library(corrplot)
library(xgboost)
library(MLmetrics)
library('randomForest')
library('rpart')
library('rpart.plot')
library('car')
library('e1071')
library(vcd)
library(ROCR)
library(pROC)
library(glmnet)
library(caret) 



df_train<- read.csv("./train.csv")
df_test<-read.csv("./test.csv")
df_train$set<-"train"
df_test$set<-"test"
df_test$Survived<- NA
df_full<-rbind(df_test,df_train)
colnames(df_train)
colnames(df_test)
head(df_train)
rm(df_t)
##处理缺失值
sapply(df_full,function(x){sum(is.na(x))})
#填补缺失值
#麻烦的方法：
df_full$Age<- replace(df_full$Age,is.na(df_full$Age),mean(df_full$Age,na.rm=T))
df_full$Fare<-replace(df_full$Fare,is.na(df_full$Fare),mean(df_full$Fare,na.rm = TRUE))
#家庭随行人数
df_full<- df_full %>%
  mutate(familysize=SibSp+Parch+1) 

df_full$familysized[df_full$familysize==1]<- "small"
df_full$familysized[df_full$familysize>=2]<- "middle"
df_full$familysized[df_full$familysize>=5]<- "big"  

df_full$Title<-gsub('(.*, )|(\\..*)','',df_full$Name)
table(df_full$Sex,df_full$Title)
rare_title<- c("Capt","Col","Don","Dr","Jonkheer","Major","Rev","Sir","the Countess")
df_full$Title[df_full$Title %in% c("Mlle", "Ms","Lady")] <- "Miss"
df_full$Title[df_full$Title== "Mme"] <- "Mrs"
df_full$Title[df_full$Title %in% rare_title] <- "Officer"
df_full$Title <- as.factor(df_full$Title)
kable(table(df_full$Sex, df_full$Title))
res<- table(df_full$Sex, df_full$Title)
knitr::kable(res,format = "latex")

#---数据探索---
train<-df_full()
ggplot(df_full %>% filter(set=="train"),aes(Title,fill=Survived))+
  geom_bar(aes(fill=factor(Survived)),position = "stack")+
  scale_fill_brewer(palette = "Set1")+
  #scale_y_continuous(labels = comma)+
  ylab("Survival")+
  ggtitle("survival by Title")

ggplot(df_full%>% filter(set=="train"),aes(Sex,fill=Survived))+
  geom_bar(aes(fill=factor(Survived)),position="stack")+
scale_fill_brewer(palette="Set1")+
  ylab("survival")+
  ggtitle("survival by sex")

ggplot(df_full%>% filter(set=="train"),aes(familysize,fill=Survived))+
  geom_bar(aes(fill=factor(Survived)),position="stack")+
  scale_fill_brewer(palette="Set1")+
  ylab("survival")+
  ggtitle("survival by familysize")

ggplot(df_full %>% filter(set=="train"),aes(x="Age",y="Survived"))+
  geom_jitter(aes(color=Sex))+
  facet_wrap(~Pclass)


tbl_age<-df_full %>% filter(set=="train") %>%
  select(Age,Survived) %>%
  group_by(Survived) %>%
  summarise(mean.age=mean(Age,na.rm=TRUE))
tbl_age
ggplot(df_full %>% filter(set=="train"),aes(Age,fill=factor(Survived)))+
  geom_histogram(aes(y=..density..),alpha=0.5)+
  geom_density(alpha=0.2,aes(colour=Survived))+
  geom_vline(data=tbl_age,aes(xintercept=mean.age,colour=Survived),lty=2,size=1)+
  scale_fill_brewer(palette = "Set1")+
  #scale_y_continuous(labels = percent)+
  ylab("density")+
  ggtitle("survival rate by age")+
  theme_minimal()
#这幅图的竖直线分别为幸存和未幸存人的平均年龄，以及不同年龄段的幸存比例。18-40岁幸存比例最大。

#舱位等级
ggplot(df_full%>% filter(set=="train",Pclass!=3),aes(Age))+
  geom_density(alpha=0.5,aes(fill=factor(Survived)))+
  labs(title = "头等舱和二等舱不同年龄段的生存密度图")

tbl_corr<-df_full %>% filter(set=="train") %>%
  select(-PassengerId,-SibSp,-Parch) %>% #加-表示不需要的变量因子
  select_if(is.numeric) %>%
  cor(use="complete.obs") %>%
  corrplot.mixed(tl.cex=0.85)



#ML
feather<-df_full[1:891,c("Pclass","Title","Sex","Pclass","Parch","familysize")]
response<-as.factor(df_train$Survived)
feather$Survived=as.factor(df_train$Survived)
ind=createDataPartition(feather$Survived,time=1,p=0.8,list = FALSE)
train_val=feather[ind,]
test_val=feather[-ind,]


Model_DT=rpart(Survived~.,data=train_val,method="class")
rpart.plot(Model_DT,extra=3,fallen.leaves=T)
pre_DT=predict(Model_DT,data=train_val,type="class")
confusionMatrix(pre_DT,train_val$Survived)

##logistic
log.mod<-glm(Survived~.,family=binomial(link=logit),
             data=train_val)
summary(log.mod)
confint(log.mod)
train.probs<-predict(log.mod,data=train_val,type="response")
table(train_val$Survived,train.probs>0.5)

test.probs<-predict(log.mod,data=test_val,type="response")
table(test_val$Survived,test.probs>0.5)

#svm
liner.tune=svm(Survived~.,data=train_val,kernel="linear",                  cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))
liner.tune
#best.linear=liner.tune$best.model
#best.linear招不到bestone用第一次生成的代替
pre_svm=predict(liner.tune,newdata=test_val,type="class")
confusionMatrix(pre_svm,test_val$Survived)
svm_roc <- roc(test_val$Survived,as.numeric(pre_svm))
plot(svm_roc, 
     print.auc=TRUE, 
     auc.polygon=TRUE, 
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", 
     print.thres=TRUE,
     main='SVM模型ROC曲线',
     kernel = "radial")

#randomforest
set.seed(754)
test<-df_full[1:891,]
train<-df_full[892:1309,]
rf_model<-randomForest(factor(Survived)~Pclass+Sex+Age+
                         SibSp+Parch+Fare+Title+familysize,
                       data = train)

rf_model$confusion
varImpPlot(rf_model)
plot(rf_model,ylim=c(0,0.36))
legend("topright",colnames(rf_model$err.rate),col=1:3,fill=1:3)
importance<-importance(rf_model)
varImportance<-data.frame(Variables=row.names(importance),
                          Importance=round(importance[,'MeanDecreaseGini'],2))

RankImportance<-varImportance %>%
  mutate(rank(paste0('#',dense_rank(desc(importance)))))

ggplot(RankImportance,aes(x=reorder(Variables,Importance),
                          y=Importance,fill=Importance))+
  geom_bar(stat = "identity")+
  labs(x="Variables")+
  ggtitle("variable importance random forest")+
  theme(plot.title = element_text(hjust = 0.5))+
  coord_flip()

prediction<-predict(rf_model,test)
solution<-data.frame(PassengerID=test$PassengerId,Survived=prediction)


prediction <- predict(rf_model, test)
submission <- data.frame(PassengerId=names(prediction),Survived=prediction)
if(!file.exists("./predictions.csv")) {
  write.csv(submission, file = "./predictions.csv",row.names = F)}



