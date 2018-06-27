#~~~~~~~~Set the working directory ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
setwd("C:/Working Directory/Office/Data Science/Coursera/Practical Machine Learning")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~Download the files from the url and store them in training and testing data frame~~~#

url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

destfile_train <- "pml-training.csv"
destfile_test <- "pml-testing.csv"

training <- download.file(url = url_train,destfile = destfile_train)
testing <- download.file(url = url_test, destfile = destfile_test)

library(caret)
library(dplyr)
library(randomForest)
library(rpart)

training_data <- read.csv(destfile_train, na.strings = c("NA","","#DIV/0!"))
validation_data <- read.csv(destfile_test, na.strings = c("NA","","#DIV/0!"))

head(training_data)
names(training_data)
head(validation_data)
names(validation_data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~ Find out and remove the variables out of 160 which have lot of NA values ~~~#

training_clean <- training_data 
rmcol <- NULL

for(i in 1:160){ 
  
  nas <-  sum(is.na(training_clean[,i]))
  if (nas > 10000) { 
    rmcol <- c(rmcol,i)   # Creare a vector of all columns which needs to be removed #
        }
    }  
training_clean <- training_clean[-rmcol]
train_data <- training_clean
validation_data <-  validation_data[-rmcol]
dim(train_data)
dim(validation_data)
validation_data <- select(validation_data,-problem_id)
dim(validation_data)
names(validation_data)

#levels(validation_data$cvtd_timestamp) <- levels(train_data$cvtd_timestamp)
#levels(validation_data$new_window) <- levels(train_data$new_window)
#levels(validation_data$user_name) <- levels(train_data$user_name)

train_data <- select(train_data,-c(1:5))
validation_data <- select(validation_data,-c(1:5))


#~~~~~~Splitting the train data into training and testing data p = 0.6 ~~~~~~#

set.seed(23456)
inTrain <- createDataPartition(y=train_data$classe, p = 0.6,list = FALSE)
mytraining <- train_data[inTrain,]
mytesting <- train_data[-inTrain,]

#~~~~~~ Check and remove variables which have near zero values ~~~~~#
nzvalue_train <- nearZeroVar(mytraining,saveMetrics = TRUE)
nzvalue_train
mytraining <- mytraining[,nzvalue_train$nzv==FALSE]
dim(mytraining)

nzvalue_test <- nearZeroVar(mytesting,saveMetrics = TRUE)
mytesting <- mytesting[,nzvalue_test$nzv==FALSE]
dim(mytesting)

#~~~~~~~~~~~~Model Based on Random Forest~~~~~~~~~~~~#
model_rf <- randomForest(classe ~ ., data = mytraining)
print(model_rf)
pred_rf <- predict(model_rf, newdata = mytesting)
confusionMatrix(mytesting$classe,pred_rf)
confusionMatrix(mytesting$classe,pred_rf)$overall[1]
outsamplerror_rf <- 1 - confusionMatrix(mytesting$classe,pred_rf)$overall[1]

#~~~~~~~~Model based on lda~~~~~~~~~~~#
model_lda <- train(classe ~ ., data = mytraining,method = "lda")
print(model_lda)
pred_lda <- predict(model_lda, newdata = mytesting)
confusionMatrix(mytesting$classe,pred_lda)
confusionMatrix(mytesting$classe,pred_lda)$overall[1]
outsamplerror_lda <- 1 - confusionMatrix(mytesting$classe,pred_lda)$overall[1]

#~~~~~~~~~~~Model based on rpart~~~~~~#
model_rpart <- train(classe ~ ., data = mytraining,method = "rpart")
model_rpart$finalModel
plot(model_rpart$finalModel, col ="red")
text(model_rpart$finalModel, col = "blue")
pred_rpart <- predict(model_rpart, newdata = mytesting)
confusionMatrix(mytesting$classe,pred_rpart)$overall[1]
outsamplerror_rpart <- 1 - confusionMatrix(mytesting$classe,pred_rpart)$overall[1]

# Get the model accuracy of all the models used for predictions # 
confusionMatrix(mytesting$classe,pred_rf)$overall[1]
confusionMatrix(mytesting$classe,pred_lda)$overall[1]
confusionMatrix(mytesting$classe,pred_rpart)$overall[1]

# Out of Sample Error of the models #
outsamplerror_rf
outsamplerror_lda
outsamplerror_rpart

#~~~~~~ Predict on the validation test data based on the model with the highest accuracy ~~~~~#
pred <-  predict(model_rf, newdata = validation_data,type = "class")
pred

