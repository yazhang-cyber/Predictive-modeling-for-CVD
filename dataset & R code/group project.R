# Library
library(rpart.plot)
library(caret)
library(dplyr)
library(randomForest)
library(neuralnet)
library(pROC)
library(ggplot2)
library(lattice)
library(smallstuff)
library(class)      
library(e1071)         
library(ROCR)
library(skimr)

# Naive Bayes as benchmark
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# transform some variables into categorical
hd.df$Sex <- as.factor(hd.df$Sex)
hd.df$cp <- as.factor(hd.df$cp)
hd.df$FBS.over.120 <- as.factor(hd.df$FBS.over.120)
hd.df$EKG.results <- as.factor(hd.df$EKG.results)
hd.df$Exercise.angina <- as.factor(hd.df$Exercise.angina)
hd.df$slope <- as.factor(hd.df$slope)
hd.df$v.f <- as.factor(hd.df$v.f)
hd.df$Thallium <- as.factor(hd.df$Thallium)
hd.df$Heart.Disease <- as.factor(hd.df$Heart.Disease)

#load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")

# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")

# transform the same variables in the validation dataset into categorical
validation.df$Sex <- as.factor(validation.df$Sex)
validation.df$cp <- as.factor(validation.df$cp)
validation.df$FBS.over.120 <- as.factor(validation.df$FBS.over.120)
validation.df$EKG.results <- as.factor(validation.df$EKG.results)
validation.df$Exercise.angina <- as.factor(validation.df$Exercise.angina)
validation.df$slope <- as.factor(validation.df$slope)
validation.df$v.f <- as.factor(validation.df$v.f)
validation.df$Thallium <- as.factor(validation.df$Thallium)

# Naive Bayes model
nb <- naiveBayes(Heart.Disease ~ ., data = hd.df)

# predict on the validation data
nb.prob <- predict(nb, validation.df, type = "raw")[, 2]

# predict on the test data
predictions_nb <- predict(nb, validation.df,type="class")

# ensure both predictions and actual values are factors with matching levels
nb.pred <- ifelse(nb.prob > 0.5, 1, 0)
nb.pred <- factor(nb.pred, levels = c(0, 1))
validation.df$Heart.Disease <- factor(validation.df$Heart.Disease, levels = c(0, 1))
predictions_nb <- factor(predictions_nb, levels = c(0, 1))

# confusion matrix
confusionMatrix(predictions_nb, validation.df$Heart.Disease, positive = "1")



# Decision Tree
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# transform some variables into categorical
hd.df$Sex <- as.factor(hd.df$Sex)
hd.df$cp <- as.factor(hd.df$cp)
hd.df$FBS.over.120 <- as.factor(hd.df$FBS.over.120)
hd.df$EKG.results <- as.factor(hd.df$EKG.results)
hd.df$Exercise.angina <- as.factor(hd.df$Exercise.angina)
hd.df$slope <- as.factor(hd.df$slope)
hd.df$v.f <- as.factor(hd.df$v.f)
hd.df$Thallium <- as.factor(hd.df$Thallium)

set.seed(123)
# fit the decision tree model
train.ct <- rpart(Heart.Disease ~ .,
                  data = hd.df,
                  method = "class")

# prune the decision tree model
prune.train.ct <- prune(train.ct, cp = train.ct$cptable[which.min(train.ct$cptable[,"xerror"]),"CP"])

# plot the pruned tree
prp(prune.train.ct,
    type = 1,
    extra = 1,
    under = TRUE,
    split.font = 1,
    cex = 1.1,
    box.palette = c("green", "red"))

# load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")

# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")

# transform the same variables in the validation dataset into categorical
validation.df$Sex <- as.factor(validation.df$Sex)
validation.df$cp <- as.factor(validation.df$cp)
validation.df$FBS.over.120 <- as.factor(validation.df$FBS.over.120)
validation.df$EKG.results <- as.factor(validation.df$EKG.results)
validation.df$Exercise.angina <- as.factor(validation.df$Exercise.angina)
validation.df$slope <- as.factor(validation.df$slope)
validation.df$v.f <- as.factor(validation.df$v.f)
validation.df$Thallium <- as.factor(validation.df$Thallium)

# use the pruned model to make predictions on the validation dataset
predictions <- predict(prune.train.ct, newdata = validation.df, type = "class")

# ensure both predictions and actual values are factors with the same levels
predictions <- as.factor(predictions)
validation.df$Heart.Disease <- as.factor(validation.df$Heart.Disease)

# align levels between predictions and actual values
levels(predictions) <- levels(validation.df$Heart.Disease)

# compute the confusion matrix 
confusionMatrix(predictions, validation.df$Heart.Disease)

# get the predicted probabilities for the positive class
dt.predictions <- predict(prune.train.ct, newdata = validation.df, type = "prob")[, 2]

# create the ROC curve
roc_curve <- roc(validation.df$Heart.Disease, dt.prdictions)

# plot the ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue")



# Random Forest
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# transform some variables into categorical
hd.df$Sex <- as.factor(hd.df$Sex)
hd.df$cp <- as.factor(hd.df$cp)
hd.df$FBS.over.120 <- as.factor(hd.df$FBS.over.120)
hd.df$EKG.results <- as.factor(hd.df$EKG.results)
hd.df$Exercise.angina <- as.factor(hd.df$Exercise.angina)
hd.df$slope <- as.factor(hd.df$slope)
hd.df$v.f <- as.factor(hd.df$v.f)
hd.df$Thallium <- as.factor(hd.df$Thallium)
hd.df$Heart.Disease <- as.factor(hd.df$Heart.Disease)

set.seed(123)
# random forest
rf <- randomForest(Heart.Disease ~ ., data = hd.df, ntree = 500, nodesize = 5, importance = TRUE)

# variable importance plot
varImpPlot(rf, type = 1)

# load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")

# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")
# align the levels of factors in the validation set to match the training set
validation.df$cp <- factor(validation.df$cp, levels = levels(hd.df$cp))
validation.df$slope <- factor(validation.df$slope, levels = levels(hd.df$slope))
validation.df$v.f <- factor(validation.df$v.f, levels = levels(hd.df$v.f))
validation.df$Thallium <- factor(validation.df$Thallium, levels = levels(hd.df$Thallium))
validation.df$Sex <- factor(validation.df$Sex, levels = levels(hd.df$Sex))
validation.df$FBS.over.120 <- factor(validation.df$FBS.over.120, levels = levels(hd.df$FBS.over.120))
validation.df$EKG.results <- factor(validation.df$EKG.results, levels = levels(hd.df$EKG.results))
validation.df$Exercise.angina <- factor(validation.df$Exercise.angina, levels = levels(hd.df$Exercise.angina))
validation.df$Heart.Disease <- factor(validation.df$Heart.Disease, levels = levels(hd.df$Heart.Disease))

# make predictions on the validation data
rf.pred <- predict(rf, validation.df)

# compute the confusion matrix
confusionMatrix(rf.pred, validation.df$Heart.Disease, positive = "1")

# predict probabilities on the validation data
rf.predictions <- predict(rf, validation.df, type = "prob")[, 2]



# Neural Network
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# create dummy variables for categorical variables
dummies <- dummyVars(" ~ .", data = hd.df)
hd.df <- predict(dummies, newdata = hd.df)

# convert the data frame to a data frame
hd.df <- as.data.frame(hd.df)

# normalize the numerical variables
norm_df <- preProcess(hd.df, method = 'range')
norm.df <- predict(norm_df, hd.df)

# load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")
# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")
# create dummy variables for categorical variables
dummies <- dummyVars(" ~ .", data = validation.df)
validation.df <- predict(dummies, newdata = validation.df)

# convert the data frame to a data frame
validation.df <- as.data.frame(validation.df)

# normalize the numerical variables
norm_df <- preProcess(validation.df, method = 'range')
vnorm.df <- predict(norm_df, validation.df)

set.seed(123)
# define ranges for hyperparameters
hidden_layers <- list(c(2), c(5), c(5, 5))  # Different hidden layer structures to try
stepmax_values <- c(1e5, 1e6, 1e7)          # Different stepmax values to try

# initialize a data frame to store results
results <- data.frame(hidden = character(), stepmax = numeric(), accuracy = numeric(), auc = numeric(), stringsAsFactors = FALSE)

# loop through all combinations of hidden layers and stepmax
for (hidden in hidden_layers) {
  for (stepmax in stepmax_values) {
    # train the neural network model with the current combination
    set.seed(123)
    nn <- neuralnet(
      Heart.Disease ~ ., 
      data = norm.df, 
      hidden = hidden, 
      stepmax = stepmax, 
      linear.output = FALSE
    )
    
    # predict on validation data
    predictions <- as.vector(compute(nn, vnorm.df[, -ncol(vnorm.df)])$net.result)
    predicted_class <- ifelse(predictions > 0.5, 1, 0)  # Convert probabilities to binary
    
    # calculate accuracy and AUC for this combination
    accuracy <- mean(predicted_class == vnorm.df$Heart.Disease)
    roc_curve <- roc(vnorm.df$Heart.Disease, predictions)
    auc_value <- auc(roc_curve)
    
    # store the results
    results <- rbind(results, data.frame(
      hidden = paste(hidden, collapse = "-"),  # Store hidden layer structure as a string
      stepmax = stepmax,
      accuracy = accuracy,
      auc = auc_value
    ))
  }
}

# print all results
print(results)

# select the best combination based on AUC or accuracy
best_model <- results %>% 
  arrange(desc(auc)) %>% 
  slice(1)

print(best_model)

# fit the neural network model
nn <- neuralnet(Heart.Disease ~ .,
                data = norm.df,
                hidden = c(5,5),
                linear.output = FALSE,
                stepmax = 1e5)


# make predictions on the validation data
predictions <- compute(nn, vnorm.df[, -ncol(vnorm.df)])$net.result

# convert neural network output probabilities to binary classes
nn.pred <- ifelse(predictions > 0.5, 1, 0)

# convert to factor with levels matching the original target variable
nn.pred <- factor(nn.pred, levels = c(0, 1))
validation_target <- factor(validation.df$Heart.Disease, levels = c(0, 1))

# compute the confusion matrix
conf_matrix <- confusionMatrix(nn.pred, validation_target, positive = "1")
print(conf_matrix)

# convert the predictions to a vector of probabilities
nn.predictions <- as.vector(predictions)

# the ROC curve
roc_curve <- roc(validation_target, nn.predicitons)

# plot ROC curve
plot(roc_curve, main = "ROC Curve for Neural Network Model", col = "blue")



# Logistic Regression
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# transform some variables into categorical
hd.df$Sex <- as.factor(hd.df$Sex)
hd.df$cp <- as.factor(hd.df$cp)
hd.df$FBS.over.120 <- as.factor(hd.df$FBS.over.120)
hd.df$EKG.results <- as.factor(hd.df$EKG.results)
hd.df$Exercise.angina <- as.factor(hd.df$Exercise.angina)
hd.df$slope <- as.factor(hd.df$slope)
hd.df$v.f <- as.factor(hd.df$v.f)
hd.df$Thallium <- as.factor(hd.df$Thallium)
hd.df$Heart.Disease <- as.factor(hd.df$Heart.Disease)

set.seed(123)
# build the logistic regression model
lr <- glm(Heart.Disease ~ Sex + cp + BP + ST.depression + slope + v.f + Thallium, 
          data = hd.df, family = "binomial")
summary(lr)

# load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")

# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")

# transform the same variables in the validation dataset into categorical
validation.df$Sex <- as.factor(validation.df$Sex)
validation.df$cp <- as.factor(validation.df$cp)
validation.df$FBS.over.120 <- as.factor(validation.df$FBS.over.120)
validation.df$EKG.results <- as.factor(validation.df$EKG.results)
validation.df$Exercise.angina <- as.factor(validation.df$Exercise.angina)
validation.df$slope <- as.factor(validation.df$slope)
validation.df$v.f <- as.factor(validation.df$v.f)
validation.df$Thallium <- as.factor(validation.df$Thallium)

# make predictions on the validation data
lr.predictions <- predict(lr, newdata = validation.df, type = "response")
lr.class.predictions <- ifelse(lr.predictions > 0.4, 1, 0)

# ensure both predictions and actual values are factors with matching levels
lr.class.predictions <- factor(lr.class.predictions, levels = c(0, 1))
validation_target <- factor(validation.df$Heart.Disease, levels = c(0, 1))

# compute the confusion matrix
confusionMatrix(lr.class.predictions, validation_target)



# KNN
# loading the data
train_data <- read.csv("Heart_Disease_Training data.csv")
valid_data <- read.csv("Heart_Disease_Validation data.csv")

# convert the target variable "Heart Disease" to factor (for classification)
train_data$Heart.Disease <- as.factor(train_data$Heart.Disease)
valid_data$Heart.Disease <- as.factor(valid_data$Heart.Disease)
# feature scaling: Standardize features using preProcess (excluding the target column)
preProcessRange <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
train_data_scaled <- predict(preProcessRange, train_data[, -ncol(train_data)])
valid_data_scaled <- predict(preProcessRange, valid_data[, -ncol(valid_data)])
# range of K values to test
k_values <- seq(1, 20, 2)
# cross-validation using 5-folds
train_control <- trainControl(method = "cv", number = 5)
knn_cv <- train(x = train_data_scaled,
                y = train_data$Heart.Disease,
                method = "knn",
                tuneGrid = data.frame(k = k_values),
                trControl = train_control)
# print the best K found from cross-validation
best_k <- knn_cv$bestTune$k
print(paste("Best K: ", best_k)) # Best K: 5

# use the best K to train the final model and predict on validation set
knn_pred <- knn(train = train_data_scaled,
                test = valid_data_scaled,
                cl = train_data$Heart.Disease,
                k = 5,prob=TRUE)
# evaluate the model: Calculate accuracy and sensitivity
conf_matrix <- confusionMatrix(knn_pred, valid_data$Heart.Disease, positive = "1")

# extract accuracy and sensitivity (recall)
accuracy <- conf_matrix$overall['Accuracy']
sensitivity <- conf_matrix$byClass['Sensitivity']

# print the results
print(paste("Accuracy: ", accuracy))
print(paste("Sensitivity: ", sensitivity))
print(conf_matrix)

# obtain probabilities for ROC curve calculation
knn_prob <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == levels(train_data$Heart.Disease)[1], 1 - knn_prob, knn_prob) 
knn.predictions <- knn(train = train_data_scaled, test = valid_data_scaled, cl =train_data$Heart.Disease, k = 5, prob=TRUE)



# SVM
# load the data
hd.df <- read.csv("Heart_Disease_Training data.csv")

# rename the columns
hd.df <- hd.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")


# transform some variables into categorical
hd.df$Sex <- as.factor(hd.df$Sex)
hd.df$cp <- as.factor(hd.df$cp)
hd.df$FBS.over.120 <- as.factor(hd.df$FBS.over.120)
hd.df$EKG.results <- as.factor(hd.df$EKG.results)
hd.df$Exercise.angina <- as.factor(hd.df$Exercise.angina)
hd.df$slope <- as.factor(hd.df$slope)
hd.df$v.f <- as.factor(hd.df$v.f)
hd.df$Thallium <- as.factor(hd.df$Thallium)
hd.df$Heart.Disease <- as.factor(hd.df$Heart.Disease)

# load the validation data
validation.df <- read.csv("Heart_Disease_Validation data.csv")

# rename the columns
validation.df <- validation.df %>%
  rename("cp" = "Chest.pain.type",
         "chol" = "Cholesterol",
         "v.f" = "Number.of.vessels.fluro",
         "slope" = "Slope.of.ST")

# transform the same variables in the validation dataset into categorical
validation.df$Sex <- as.factor(validation.df$Sex)
validation.df$cp <- as.factor(validation.df$cp)
validation.df$FBS.over.120 <- as.factor(validation.df$FBS.over.120)
validation.df$EKG.results <- as.factor(validation.df$EKG.results)
validation.df$Exercise.angina <- as.factor(validation.df$Exercise.angina)
validation.df$slope <- as.factor(validation.df$slope)
validation.df$v.f <- as.factor(validation.df$v.f)
validation.df$Thallium <- as.factor(validation.df$Thallium)

# SVM model
svm <- svm(Heart.Disease ~ ., data = hd.df, kernel = "linear", cost = 10, scale = TRUE, probability = TRUE)

# make predictions on the validation data with SVM
svm.prob <- predict(svm, validation.df, probability = TRUE)
svm.prob <- attr(svm.prob, "probabilities")[, 2]

# predict on the test data
predictions <- predict(svm, validation.df, type = "response")

# ensure both predictions and actual values are factors with matching levels
predictions <- as.factor(predictions)
validation.df$Heart.Disease <- as.factor(validation.df$Heart.Disease)

# confusion matrix
confusionMatrix(predictions, validation.df$Heart.Disease, positive = "1")



# Performance Evaluation
# compare the models based on ROC curve
validation_target <- validation.df$Heart.Disease

# calculate ROC curves for all models
roc_nn <- roc(validation_target, nn.predictions)
roc_dt <- roc(validation_target, dt.predictions)
roc_rf <- roc(validation_target, rf.predictions)
roc_lr <- roc(validation_target, lr.predictions)
roc_knn <- roc(validation_target, knn_prob) 
roc_svm <- roc(validation_target, svm.prob)
roc_nb <- roc(validation_target, nb.prob)

# plot ROC curves
plot(roc_nn, col = "green", main = "ROC Curve Comparison", lwd = 2)
lines(roc_dt, col = "yellow", lwd = 2)
lines(roc_rf, col = "purple", lwd = 2)
lines(roc_lr, col = "blue", lwd = 6)
lines(roc_knn, col = "orange", lwd = 2)
lines(roc_svm, col = "black", lwd = 2)
lines(roc_nb, col = "red", lwd = 3)

# add a legend to the plot
legend("bottomright", 
       legend = c("Neural Network", "Decision Tree", "Random Forest", "Logistic Regression", "KNN", "SVM", "Naive Bayes"), 
       col = c("green", "yellow", "purple", "blue", "orange", "black", "red"), 
       lwd = 2,
       cex = 0.7)
