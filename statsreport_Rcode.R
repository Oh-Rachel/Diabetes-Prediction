set.seed(1101)
         
setwd("C:/Users/ohkel/Documents/DSA1101/data files")
data <- read.csv("diabetes_5050.csv")

head(data)
dim(data)
#  70692    22


attach(data)

########### checking input variables #############
# using odds ratio


#### check HighBP ###
table(HighBP)
table_bp <- table(HighBP, Diabetes_binary);table_bp
prop_bp <- prop.table(table_bp, "HighBP"); prop_bp
# Among respondents with HighBP No, 71.7% are having no diabetes while 28.3% are having diabetes
# we consider having diabetes as success

# for those with HighBP No, odds of success:
odds_Highbp_no <- prop_bp[3]/(1-prop_bp[3]) ; odds_Highbp_no # = 0.3952437

# for those with HighBP Yes, odds of success:
odds_Highbp_yes <- prop_bp[4]/(1-prop_bp[4]) ; odds_Highbp_yes # = 2.011188

#odds ratio:
odds_Highbp_no/odds_Highbp_yes # = 0.1965225

# since odds ratio is far from 1, there is strong (negative) association
# hence we will use HighBP to build our model



# we can create a loop to do this for the selected categorical variables

binary_vars <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
                 "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex")



odds_ratio <- numeric(length(binary_vars)) # store odds ratio

for (i in 1:length(binary_vars)) {
  var_name <- binary_vars[i]
  var_data <- data[[var_name]]
  
  table_var <- table(var_data, Diabetes_binary)
  prop_var <- prop.table(table_var, 1) 
  
  odds_no <- prop_var[3]/(1 - prop_var[3])
  odds_yes <- prop_var[4]/(1 - prop_var[4])
  
  odds_ratio[i] <- odds_no/odds_yes
}

index <- c(which(odds_ratio <= 0.5), which(odds_ratio >= 1.5)) # get index of those far from 1
include <- binary_vars[index] ;include

# features that have strong assocation with the response and should be included in the model:
# "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack", "DiffWalk", "PhysActivity", 
# "HvyAlcoholConsump"

new_data <- data[, c("Diabetes_binary", include)] 




boxplot(BMI ~  Diabetes_binary)
boxplot(GenHlth ~  Diabetes_binary)
boxplot(PhysHlth ~  Diabetes_binary)
boxplot(Age ~  Diabetes_binary)
boxplot(Education ~  Diabetes_binary)
boxplot(Income ~  Diabetes_binary)

# include features with observable association

new_data <- data[, c("Diabetes_binary", include, c("BMI", "GenHlth", "PhysHlth", "Age", "Income"))]
# work with this data set from now on
dim(new_data)




#########################   MODEL 1 (M1): Naive Bayes   ##############################

library(e1071)
library(ROCR)


# initial run
test.index <- sample(1:dim(new_data)[1], 0.2*dim(new_data)[1]) # get random index (20%)


train.data <- new_data[-test.index, ]
test.data <- new_data[test.index, ]


M1 <- naiveBayes(Diabetes_binary ~., data = train.data) # form initial model



pred_M1 <- predict(M1, test.data[, -1], type = 'class') # using test data, predict response

acc <- sum(pred_M1 == test.data$Diabetes_binary) / length(test.data$Diabetes_binary) ; acc # compute accuracy

# find auc
pred_M1 <- predict(M1, test.data[, -1], type = 'raw')
score1 <- pred_M1[, "1"]
prediction_M1 <- prediction(score1, test.data$Diabetes_binary)
roc_M1 <- performance(prediction_M1, "tpr", "fpr")
auc_M1 <- performance(prediction_M1 , measure ="auc")
auc <- auc_M1@y.values[[1]] ;auc

# roc curve example
plot(roc_M1 , col = "red", main = paste(" Area under the curve :", round(auc ,4)))




# n-fold cross validation

n_folds <- 5
folds_j <- sample(rep(1:n_folds, length.out = dim(new_data)[1]))

acc_M1 <- numeric(n_folds) # store accuracy for each fold
auc_M1 <- numeric(n_folds) # store auc for each fold

# pick out 20% for testing
for (j in 1:n_folds) {
  
  # test 20%
  test.index <- which(folds_j == j)
  train.data <- new_data[-test.index, ]
  test.data <- new_data[test.index, ]
  
  # build model
  M1 <- naiveBayes(Diabetes_binary ~., data = train.data)
  
  # pred based on class to calculate accuracy
  pred_M1 <- predict(M1, test.data[, -1], type = 'class')
  acc_M1[j] <- sum(pred_M1 == test.data$Diabetes_binary) / length(test.data$Diabetes_binary)
  
  # calculate AUC
  pred_M1 <- predict(M1, test.data[, -1], type = 'raw')
  score1 <- pred_M1[, "1"]
  
  prediction_M1 <- prediction(score1, test.data$Diabetes_binary)
  # roc_M1 <- performance(prediction_M1, measure = "tpr", x.measure = "fpr")
  auc_M1[j] <- performance(prediction_M1, "auc")@y.values[[1]]
}

# find mean accuracy and mean AUC
mean_acc_M1 <- mean(acc_M1) ;mean_acc_M1 # 0.7278758
mean_auc_M1 <- mean(auc_M1) ;mean_auc_M1 # 0.7898671








#########################   MODEL 2 (M2): Decision Tree   ##############################
library("rpart")
library("rpart.plot")

# corss validation to find best cp

n_folds = 5
n = dim(new_data)[1]
folds_j <- sample(rep(1:n_folds, length.out = n))

cp <- 10^(-10:0)
length(cp)
acc <- numeric(length(cp)) 

for (i in 1:length(cp)) {
  correct <- 0
  
  for (j in 1:n_folds) {
    test.index <- which(folds_j == j)
    train.data <- new_data[-test.index,]
    test.data <- new_data[test.index,]
    
    M2 <- rpart(Diabetes_binary ~ .,
                method = "class", 
                data = train.data,
                control = rpart.control(cp = cp[i]),
                parms = list(split = 'information'))
    
    ## predict label for test data based on fitted tree
    pred_M2 <- predict(M2, test.data[,-1], type = 'class')
    
    correct <- correct + sum(pred_M2 == test.data[,1])
  }
  
  acc[i] <- correct / n
}


plot(-log(cp,base=10),acc,type='b')

## determine the best cp based on highest accuracy
max(acc)
best.cp =cp[which(acc == max(acc))] ; best.cp # 1e-04




# fit decision tree with chosen cp
M2 <- rpart(Diabetes_binary ~ .,
            method = "class", 
            data = train.data,
            control = rpart.control(cp = best.cp),
            parms = list(split = 'information'))


# to get the tree plotted:
rpart.plot(M2, type=4, extra=2, clip.right.labs=FALSE, varlen=0)


# finding mean auc

auc_M2 <- numeric(length(n_folds))

for (j in 1:n_folds) {
  
  # test 20%
  test.index <- which(folds_j == j)
  train.data <- new_data[-test.index, ]
  test.data <- new_data[test.index, ]
  
  # build model
  M2 <- rpart(Diabetes_binary ~ .,
              method = "class", 
              data = train.data,
              control = rpart.control(cp = best.cp),
              parms = list(split = 'information'))
  
  
  # calculate AUC
  pred_M2 <- predict(M2, test.data[, -1], type = 'prob')
  score2 <- pred_M2[, "1"]
  
  prediction_M2 <- prediction(score2, test.data$Diabetes_binary)
  # roc_M2 <- performance(prediction_M2, measure = "tpr", x.measure = "fpr")
  auc_M2[j] <- performance(prediction_M2, "auc")@y.values[[1]]
}


mean_auc_M2 <- mean(auc_M2) ;mean_auc_M2





#########################   MODEL 3 (M3): Logistic Regression   ##############################

# declaring factor variabes

for (vars in names(new_data)[1:9]) {
  new_data[[vars]] <- as.factor(new_data[[vars]])
}


# build initial model and check significance of regressors
M3<- glm( Diabetes_binary ~., data = new_data,family = binomial)
summary(M3)

# p-value of PhysActivity = 0.05127 is large hence PhysActivity is not very significant,
# we can remove the variable PhysActivity from our model

#rebuild model without PhysActivity

M3 <- glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + Stroke + HeartDiseaseorAttack + DiffWalk +
            HvyAlcoholConsump + BMI + GenHlth + PhysHlth + Age + Income, 
          data = new_data, family = binomial(link = "logit"))
summary(M3)



n_folds <- 5
folds_j <- sample(rep(1:n_folds, length.out = dim(new_data)[1]))

acc_M3 <- numeric(n_folds)
auc_M3 <- numeric(n_folds)

for (j in 1:n_folds) {
  
  # test 20%
  test.index <- which(folds_j == j)
  train.data <- new_data[-test.index, ]
  test.data <- new_data[test.index, ]
  
  # build model, removed PhysActivity
  M3 <- glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + Stroke + HeartDiseaseorAttack + DiffWalk +
              HvyAlcoholConsump + BMI + GenHlth + PhysHlth + Age + Income, 
            data = train.data, family = binomial(link = "logit"))
  
  # predict test set
  pred_M3 <- predict(M3, test.data[, -1], type = 'response')
  
  # convert probabilities to binary
  pred_binary <- ifelse(pred_M3 > 0.5, 1, 0)
  
  # calculate accuracy
  acc_M3[j] <- sum(pred_binary == test.data$Diabetes_binary) / length(test.data$Diabetes_binary)
  
  # calculate AUC
  prediction_M3 <- prediction(pred_M3, test.data$Diabetes_binary)
  auc_M3[j] <- performance(prediction_M3, "auc")@y.values[[1]]
}

mean_acc_M3 <- mean(acc_M3) ;mean_acc_M3 # 0.7472273
mean_auc_M3 <- mean(auc_M3) ;mean_auc_M3 # 0.8235211







# analysis on tpr, fpr and exploring threshold values


prob = predict(M3, type ="response")
pred = prediction(prob , Diabetes_binary )
roc = performance(pred , "tpr", "fpr")


alpha <- round (as.numeric(unlist(roc@alpha.values)) ,4)
length(alpha) 
fpr <- round(as.numeric(unlist(roc@x.values)) ,4)
tpr <- round(as.numeric(unlist(roc@y.values)) ,4)


par(mar = c(5 ,5 ,2 ,5))

# plot FPR, TPR and threshold
plot(alpha ,tpr , xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "blue")
par( new ="True")
plot(alpha ,fpr , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "red" )
axis( side =4) # to create an axis at the 4th side
mtext(side =4, line =3, "False positive rate")
text(0.18 ,0.18 , "FPR")
text(0.58 ,0.58 , "TPR")

# to compare tpr, fpr and threshold
x = cbind(alpha, tpr, fpr)

# to find whichh index gives biggest difference in tpr and fpr
best.threshold <- which((tpr - fpr) == max((tpr - fpr)))
x[best.threshold,1]





