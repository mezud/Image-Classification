#DS8004 Data Mining
#Amir Ghaderi and Mezbah Uddin 

#install packages
install.packages("dplyr")
install.packages("stats")
install.packages("e1071")
install.packages("plyr")
install.packages("class")
install.packages("randomForest")
install.packages("ipred")
install.packages('neuralnet')
install.packages("RSNNS")
install.packages("Hmisc")
#load libraries 
library(dplyr)
library(stats)
library(e1071)
library(plyr)
library(class)
library(randomForest)
library(ipred)
library(neuralnet)
library(RSNNS)
library(Hmisc)
#Import data
nam <- read.table("data.csv", nrow = 1, stringsAsFactors = FALSE, sep = ",")
df <- read.table("data.csv", skip = 1, stringsAsFactors = FALSE, sep = ",")
df <- df[, 1:32]
nam$V33<-NULL
names(df) <- nam 

# class info
class_info <-df %>%count(diagnosis, sort = TRUE) 
class_info$n<-as.numeric(class_info$n)
class_info$pct<-(class_info$n/(357+212))
print(' 67% of class B and 33% of class M')


# normalize dataset
f <- df[,3:32]

norm <- function(x){
  z<-(x-min(x))/(max(x)-min(x))
  return(z)
}

df_n<- sapply(f,norm)


#PCA
pca <- prcomp(df_n,center = TRUE,scale. = TRUE) 
summary(pca) 
print('pick 10 Principal components')
df_pca<-pca$x
df_pca<- df_pca[,1:10]
df_pca<- as.data.frame(df_pca)

#add class
df_pca$class <- df[,2]

#convert class to factor
df_pca$class <- as.factor(df_pca$class)

#Data Splitting
xc<- function(x){
  if (x=='M') {
    return("0")
  } else {
    return("1")
  }
}

df_pca$cn<-lapply(df_pca$class,xc)
df_pca$cn<-as.numeric(df_pca$cn)
class(df_pca$cn)
df_pca$cn<-as.factor(df_pca$cn)

df_pca1<-df_pca[,-c(11)]
data_train <-df_pca1[1:455,]
data_test <- df_pca1[456:569,]


#knn k = 1,3,5,11,19,21
model_knn <- knn(train = data_train[,1:10], test = data_test[,1:10], cl = data_train[,11], k = 21)
table(data_test[,11], model_knn)
mean(model_knn == data_test[,11])


#Cross Validation 5 Fold SVM

#K fold validation iris
k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
df_pca$id <- sample(1:k, nrow(df_pca), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()


for (i in 1:k){
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(df_pca, id %in% list[-i])
  testset <- subset(df_pca, id %in% list[i])
  
  #SVM 
  #mymodel <- svm(trainingset$class ~ ., data = trainingset,kernel="linear",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$class ~ ., data = trainingset,kernel="polynomial",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$class ~ ., data = trainingset,kernel="sigmoid",cost=1,scale=FALSE)
  #mymodel <- svm(trainingset$class ~ ., data = trainingset,kernel="radial",cost=1,scale=FALSE)
  
  #Random Forest
  #mymodel <- randomForest(trainingset$class ~ ., data = trainingset, ntree = 1000)
  
  #bagging
  #mymodel <- bagging(class~., data=trainingset,type="class")
  
  
  # remove response column 1, 
  temp <- as.data.frame(predict(mymodel, testset[,-11]))
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, temp)
  
  # append this iteration's test set to the test set copy data frame
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,11]))
  
  result <- cbind(prediction,testsetCopy)
  names(result) <- c("Predicted", "Actual")
  cat(paste0("iteration: ", i))
  print(head(result))
  count<-0
  for (i in 1:nrow(result)){
    if(result$Predicted[i]!=result$Actual[i]){count<-count+1}
    
  }
  print(count/nrow(result))
  
}

#Neural Networks
data_train<-as.data.frame(data_train)
data
str(data_train)
data_train$cn<-as.numeric(data_train$cn)
nnp <- neuralnet(cn~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10, data = data_train, hidden=1,lifesign = "minimal",linear.output = FALSE,threshold = 0.1)

temp_test <- subset(data_test, select = c("SEX","AGE","LIMIT_BAL"))
creditnet.results <- compute(nnp,temp_test)
table(data_test[,12],round(creditnet.results$net.result))
mean(round(creditnet.results$net.result) == data_test[,12])


####
df1_pca<-df_pca[,-11]
df1_pca <- df1_pca[sample(1:nrow(df1_pca),length(1:nrow(df1_pca))),1:ncol(df1_pca)]

df1_pcaValues <- df1_pca[,1:10]
df1_pcaTargets <- decodeClassLabels(df1_pca[,11])


df1_pca <- splitForTrainingAndTest(df1_pcaValues, df1_pcaTargets, ratio=0.15)

model <- mlp(df1_pca$inputsTrain, df1_pca$targetsTrain, size=5, learnFuncParams=c(0.1), 
             maxit=50, inputsTest=df1_pca$inputsTest, targetsTest=df1_pca$targetsTest)

summary(model)
model
weightMatrix(model)
extractNetInfo(model)

par(mfrow=c(2,2))
plotIterativeError(model)

predictions <- predict(model,df1_pca$inputsTest)

plotRegressionError(predictions[,2], df1_pca$targetsTest[,2])

confusionMatrix(df1_pca$targetsTrain,fitted.values(model))
confusionMatrix(df1_pca$targetsTest,predictions)

plotROC(fitted.values(model)[,2], df1_pca$targetsTrain[,2])
plotROC(predictions[,2], df1_pca$targetsTest[,2])

ggplot(df, aes(x=df_pca$PC6, y=df_pca$PC7, fill=df_pca$cn))+geom_point(aes(color=df_pca$cn))

rcorr(df_pca1)

df_pca2<-df[,3:32]

df_pca3<-df_pca2[,1:10]
df_pca4<-df_pca2[,16:30]
rcorr(as.matrix(df_pca))

library(corrplot)
M <- cor(df_pca3)
corrplot(M, method="circle")
head(M)
