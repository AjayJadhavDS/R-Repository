# Loading the File
bk <-read.csv(file.choose(), sep = ";")

str(bk)

# Using lapply converting all columns to numeric
bk1 <- lapply(bk,as.numeric)
bk1 <- data.frame(bk1)
str(bk1)

##########################
# Class Attribute - > y
#############################


bk_preprocess <- bk1[,c(1,3,4,14,15,21)]
str(bk_preprocess)

## min,max

summary(bk1$age)
summary(bk1$emp.var.rate)

## mode
names(table(bk1$age))[table(bk1$age)==max(table(bk1$age))]

names(table(bk1$emp.var.rate))[table(bk1$emp.var.rate)==max(table(bk1$emp.var.rate))]

## sd

sd(bk1$age)

sd(bk1$emp.var.rate)

## Plot ##
boxplot(bk1$y ~ bk1$previous)
plot(bk1$y ~ bk1$poutcome)
plot(bk1$y ~ bk1$education)
plot(bk1$y ~ bk1$marital)
plot(bk1$y ~ bk1$age)

## Splitting Dataset ##

##Splitting dataset ##

bk_preprocess = bk_preprocess[1:20000,]

ind <- sample(2, nrow(bk_preprocess), replace=TRUE, prob=c(0.75, 0.25))
trainDataset <- bk_preprocess[ind==1,]
testDataset <- bk_preprocess[ind==2,]
trainDataset$y = as.factor(trainDataset$y)


str(trainDataset)

library(mlbench)

library(e1071)


## Naive Bayes Method ##

nb_model_bk <- naiveBayes(y ~.,data = trainDataset)
nb_model_bk
summary(nb_model_bk)
str(nb_model_bk)
nbm_test_predict_bk <- predict(nb_model_bk,testDataset[,-6])
mean(nbm_test_predict_bk==testDataset$y)

## Output ##
# 82.64585


## Random Forest ##
library(randomForest)
rf_bk <- randomForest(y ~ ., data=trainDataset, ntree=15, proximity=TRUE)
rf_pred = predict(rf_bk,newdata = testDataset)
mean(rf_pred == testDataset[,6])

#output
#0.9550031
importance(rf_bk)
varImpPlot(rf_bk)

Rand_Pred_bk <- predict(rf_bk, newdata=testDataset)
mean(Rand_Pred_bk==testDataset$y)
100*sum(Rand_Pred_bk==testDataset$y)/length(testDataset$y)


## clustering ##
## k means

normalize = function(x){
  return( (x - min(x)) / (max(x) - min(x)) )
}

testDataset_new = normalize(testDataset)


testDataset_new$y <- NULL

(kmeans.result <- kmeans(testDataset_new,2))

table(testDataset$y,kmeans.result$cluster)

plot(testDataset_new[c("previous","poutcome")], col=kmeans.result$cluster)

points(kmeans.result$centers[,c("poutcome","education")], col = 1:3, pch=8, cex=2)


## DBSCAN

library(fpc)

dbscan_data <- dbscan(testDataset[,-6], eps=1.5, MinPts=5)

## Let's compare the clusters with the original class labels
table(dbscan_data$cluster, testDataset$y)
plot(dbscan_data, testDataset[,-6])
plotcluster(testDataset[,-6], dbscan_data$cluster)
