install.packages("tree")
install.packages("party")
install.packages("rpart")
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
install.packages("randomForest")
install.packages("h2o")

library(car)

### Q1 ###

# In my dataset i had chosen Block has Class Attribute becasuse after seeing my dataset i came to conclusion that this is all about virus in the chicago 
# SO if we had known at what place the virus is more we can make some preventive measures in that areas to save peple from diseases. 
# So if there are some mosquitoes killers sprays like hit which are not available in O hare blocks they can use this data class attributes and implement 
# their products in that areas and gets profits and also save health of people.

### Mean, Median, Mode & sd ###

getwd()
virus <- read.csv("./West_Nile_Virus__WNV__Mosquito_Test_Results.csv")[1:100,]

str(virus)
virus <- na.omit(virus)

summary(virus)

summary(virus$TEST.ID)
mode(virus$TEST.ID)
sd(virus$TEST.ID)

summary(virus$NUMBER.OF.MOSQUITOES)
mode(virus$NUMBER.OF.MOSQUITOES)
sd(virus$NUMBER.OF.MOSQUITOES)

virus$BLOCK<-as.numeric(virus$BLOCK)

### Scatter plot ###


plot (virus$BLOCK,virus$NUMBER.OF.MOSQUITOES, main = " Scatterplot" , xlab = "No.of Mosquitoes", ylab = "Block", pch = 20, col=" red")

### KNN ###

install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
##################################################
#       k Nearest Neighbor       #
##################################################


library(textir) ## needed to standardize the data
library(MASS)   ## a library of example datasets

getwd()
virusq4 <- read.csv("./West_Nile_Virus__WNV__Mosquito_Test_Results.csv")
virusq4 <- na.omit(virusq4)
str(virusq4)

virusq4$RESULT<-as.numeric(virusq4$RESULT)
virusq4$BLOCK<-as.numeric(virusq4$BLOCK)


par(mfrow=c(1,3), mai=c(.3,.6,.1,.1))
plot( RESULT ~ BLOCK , data=virusq4, col=c(grey(.2),2:6))
plot(WEEK ~ BLOCK, data=virusq4, col=c(grey(.2),2:6))
plot(SEASON.YEAR ~ BLOCK, data=virusq4, col=c(grey(.2),2:6))

n=length(virusq4$BLOCK)
nt=560
set.seed(1) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)


## x <- normalize(virusq4[,c(4,3)])

x=virusq4[,c(4,2)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

### KNN Algorithm ###

library(class)  
nearest3 <- knn(train=x[train,],test=x[-train,],cl=virusq4$BLOCK[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=virusq4$BLOCK[train],k=5)
data.frame(virusq4$BLOCK[-train],nearest3,nearest5)

## ploting them to see how these works

par(mfrow=c(1,2))

## plot for k=3 (single) nearest neighbor

plot(x[train,],col=virusq4$BLOCK[train],cex=.8,main="3-nearest neighbor")
points(x[-train,],bg=nearest3,pch=21,col=grey(.9),cex=1.25)

## plot for k=5 nearest neighbors

plot(x[train,],col=virusq4$BLOCK[train],cex=.8,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)

## calculating the proportion of correct classifications on this one 
## training set

pcorrn1=100*sum(virusq4$BLOCK[-train]==nearest3)/(n-nt)
pcorrn5=100*sum(virusq4$BLOCK[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5

## cross-validation (leave one out)
pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,virusq4$BLOCK,k)
  pcorr[k]=100*sum(virusq4$BLOCK==pred)/n
}

pcorr


######################################
########## NaviBayes ##############
######################################

library(mlbench)

virusq4[sapply(virusq4, is.numeric)] <- lapply(virusq4[sapply(virusq4, is.numeric)], as.factor)

plot(as.factor(virusq4[,4]))

title(main=" Nile Virus ", xlab="Block", ylab="#reps")## looking around that dataset 

virusq4[,"train"] <- ifelse(runif(nrow(virusq4))<0.25,1,0)

virusq4$train = as.factor(virusq4$train)

str(virusq4)

## Getting col number of train / test indicator column (needed later)

trainColNum <- grep('train', names(virusq4))

## separating training and test sets and removing training column before we model the data

trainvirusq4 <- virusq4[virusq4$train==1,-trainColNum]
testvirusq4 <- virusq4[virusq4$train==0,-trainColNum]
testvirusq4

str(virusq4)

## building the Naive Bayes model

## Loading e1071 library and invoking naiveBayes method

library(e1071)
pb6 <- naiveBayes(BLOCK ~ .,data = trainvirusq4)
pb6
summary(pb6)
str(pb6)

pb6_test_predict <-predict(pb6, testvirusq4[,-14])

## Building confusion matrix

table(pred=pb6_test_predict,true=testvirusq4$BLOCK)


### Neural Networks ###


install.packages("neuralnet")
library(neuralnet)
library(ggplot2)
library(nnet)
library(dplyr)
library(reshape2)


set.seed(123)


testvirusq4[sapply(testvirusq4, is.factor)] <- lapply(testvirusq4[sapply(testvirusq4, is.factor)], as.numeric)

testvirusq4[,c(5:7)] <- NULL

testvirusq4[,c(2:3)]<-NULL

testvirusq4[,c(6:8)]<-NULL


# Converting observation class and BLock into one vector.

labels <- class.ind(as.factor(testvirusq4$BLOCK))

# Generic function to standardize a column of data.

standardizer <- function(x){(x-min(x))/(max(x)-min(x))}

# Performing Normalization predictors. We need lapply to do this.

testvirusq4[, 1:5] <- lapply(testvirusq4[, 1:5], standardizer)
testvirusq4        

# Reviewing the data for Normalization

# Combining labels and standardized predictors.

pre_process_mb9n <- cbind(testvirusq4[,1:5], labels)

View(pre_process_mb9n)

# Formula for the neuralnet using the as.formula function

nw <- as.formula(" BLOCK ~ RESULT+SPECIES")

# Creating a neural network object using the tanh function and two hidden layers of size 10 and 8. 

mb9nw_net <- neuralnet(nw, data = pre_process_mb9n, hidden = c(10,8), act.fct = "tanh", linear.output = FALSE)

# Ploting the neural network.

plot(mb9nw_net)

# I am using the compute function and the neural network object's net.result attribute for 
# Calculating the overall accuracy of the  neural network.

mb9nw_preds <-  neuralnet::compute(mb9nw_net, pre_process_mb9n[, 1:5])
origi_values <- max.col(pre_process_mb9n[, 6:9])
pr.nn_limit <- max.col(mb9nw_preds$net.result)
print(paste("Model Accuracy: ", round(mean(pr.nn_limit==origi_values)*100, 2), "%.", sep = ""))

### Accuracy 16.67% ###

### KMEANS ###

install.packages("cluster")  
install.packages("fpc") 


testvirusq4[sapply(testvirusq4, is.factor)] <- lapply(testvirusq4[sapply(testvirusq4, is.factor)], as.numeric)
testvirusq4

virus5<-testvirusq4

str(virus5)

virus5$BLOCK <- NULL


match.kmeans.result <- kmeans(virus5, 3)


table(testvirusq4$BLOCK, match.kmeans.result$cluster)

par(mfrow=c(1,2), mai=c(.3,.6,.1,.1))

plot(virus5[c("SEASON.YEAR", "RESULT")], col = match.kmeans.result$cluster)

points(match.kmeans.result$centers[,c("SEASON.YEAR", "RESULT")], col = 1:3, pch = 8, cex=2)


###  Density-based Clustering ###

library(fpc)

ds.virus <- dbscan(virus5, eps=0.42, MinPts=3)

### we are choosing Win_Type has the Class variable.
### Comparing the clusters with the original class labels


table(ds.virus$cluster, testvirusq4$BLOCK)

plot(ds.virus, virus5)

### Displaying the clusters in a scatter plot using the first and 4 th column of the data.


plot(ds.virus, virus5[c(1,4)])

### plotcluster plotting.


plotcluster(virus5, ds.virus$cluster)


### Create a new dataset for labeling

set.seed(435)
idx.win <- sample(1:nrow(testvirusq4), 10)
newData <- testvirusq4[idx.win,-1]
newData <- newData + matrix(runif(10*4, min=0, max=0.2), nrow=10, ncol=4)

### Labeling new data

myPred.virus <- predict(ds.virus, virus5, newData)

### Plot the result with new data as asterisks

plot(virus5[c(1,4)], col=1+ds.virus$cluster)
points(newData[c(1,4)], pch="*", col=1+myPred.virus, cex=3)

### Check cluster labels

table(myPred.virus, testvirusq4$BLOCK[idx.win])

### C TREE ###

getwd()
virus4 <- read.csv("./West_Nile_Virus__WNV__Mosquito_Test_Results.csv")
virus4 <- na.omit(virus4)
# virus4$RESULT <- as.factor(virus4$RESULT)
virus4$BLOCK <- as.numeric(virus4$BLOCK)

str(virus4)
virus4

set.seed(1234)
v4 <- sample(2, nrow(virus4), replace=TRUE, prob=c(0.25, 0.75))
trainvirus <- virus4[v4==1,]
testvirus <- virus4[v4==2,]

## I am building a decision tree, and checking the prediction results. 
## Here v4a specifies that RESULT is the target variable and all other variables are independent variables.


library(party)
v4a <-  BLOCK~ WEEK + SEASON.YEAR
v4ctree <- ctree(v4a, data=trainvirus)

## Now I am making some predictions. 
## I am doing this by comparing the train and test data outputs. 
## Iam also performing the confusion matrix.
## which is a table where we can get the true and predicted values in matrix form.

table(predict(v4ctree), trainvirus$RESULT)

## Printing the Ctree 

print(v4ctree)

## Plotting the tree.

plot(v4ctree)

## The barplot for each Block node shows the probabilities falling into the either 0 or 1.
## Now iam testing the test data (75%) with the bulit tree data.

testPred8a <- predict(v4ctree, newdata = testvirus)
table(testPred8a, testvirus$BLOCK)


