library(readr)
library(lubridate)
library(tidyverse)
library(glmnet)
library(randomForest)
library(tree)

## Prediction Model

### lasso for variable selection
library(glmnet)
kickstarter <- na.omit(kickstarter_model)
Mx<- model.matrix(state~.,data=kickstarter_model)[,-1]
My<- kickstarter_model$state == 1

##
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.n
num.success <- sum(My)
w <- (num.success/num.n)*(1-(num.success/num.n))

## theoretical lambda:
lasso <- glmnet(Mx,My, family="binomial")

lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)

## running lasso using theoretical lambda
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory) 
summary(lassoTheory)
source("DataAnalyticsFunctions.R")

### these are the indices
support(lassoTheory$beta)
colnames(Mx)[support(lassoTheory$beta)]
length(support(lassoTheory$beta))


#lasso path
# changing the value of lambda computations to 20 instead of 100 to reduce computation time
lassocv1 <- cv.glmnet(Mx,My,nlambda = 20,family = "binomial")
lassocv1
plot(lassocv1)
support(lassoTheory$beta)
summary(lassocv1)
support(lassocv1$beta)
lassof <- glmnet(Mx,My,family = "binomial", lambda = lassocv1$lambda.1se)
lassomin <- glmnet(Mx,My,family = "binomial", lambda = lassocv1$lambda.min)

# checking indices for both
support(lassof$beta)
colnames(Mx)[support(lassof$beta)]
length(support(lassof$beta))

support(lassomin$beta)
colnames(Mx)[support(lassomin$beta)]
length(support(lassomin$beta))

#checking all lambda values
lambda.theory
lassocv1$lambda.1se
lassocv1$lambda.min

#storing features for all lasso calculations
features.min <- support(lasso$beta[,which.min(lassocv1$cvm)])
length(features.min)
features.1se <- support(lasso$beta[,which.min( (lassocv1$lambda-lassocv1$lambda.1se)^2)])
length(features.1se) 
features.theory <- support(lassoTheory$beta)
length(features.theory)

#datasets of given features
data.min <- data.frame(Mx[,features.min],My)
data.1se <- data.frame(Mx[,features.1se],My)
data.theory <- data.frame(Mx[,features.theory],My)

#K fold cv to evaluate difference between Lasso and Post Lasso
n <- nrow(data.1se)
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

PL.OOS <- data.frame(PL.1se=rep(NA,nfold)) 
L.OOS <- data.frame(L.1se=rep(NA,nfold)) 

for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  # 
  # ### This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  if ( length(features.1se) == 0){  r1se <- glm(state~1, data=kickstarter_model, subset=train, family="binomial")
  } else {r1se <- glm(My~., data=data.1se, subset=train, family="binomial")
  }
  
  if ( length(features.theory) == 0){ 
    rtheory <- glm(state~1, data=kickstarter_model, subset=train, family="binomial") 
  } else {rtheory <- glm(My~., data=data.1se, subset=train, family="binomial") }
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.1se[-train,], type="response")
  PL.OOS$PL.min[k] <- R2(y=My[-train], pred=predmin, family="binomial")
  PL.OOS$PL.1se[k] <- R2(y=My[-train], pred=pred1se, family="binomial")
  PL.OOS$PL.theory[k] <- R2(y=My[-train], pred=predtheory, family="binomial")
  
  ### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassocv1$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassocv1$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  L.OOS$L.min[k] <- R2(y=My[-train], pred=predlassomin, family="binomial")
  L.OOS$L.1se[k] <- R2(y=My[-train], pred=predlasso1se, family="binomial")
  L.OOS$L.theory[k] <- R2(y=My[-train], pred=predlassotheory, family="binomial")
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

#Checking performance of different models using OOS

## Out of sample prediction
### create an empty dataframe of results
OOS <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold),forest=rep(NA,nfold), null=rep(NA,nfold)) 

### Set the second part for testing (first for training)
nfold <- 5
n <- nrow(data.min) # the number of observations
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
k <- 2
set.seed(64)
### Set the other part for training (if not k)
train <- which(foldid!=k) # train on all but fold `k'
test  <- which(foldid==k) # test on fold k

data.theory$My[data.theory$My == 1] <- TRUE
data.theory$My[data.theory$My == 0] <- FALSE

model.logistic.interaction <-glm(My~.^2, data=data.theory, subset=train, family="binomial")
model.logistic <-glm(My~., data=data.theory, subset=train,family="binomial")
model.tree <- tree(My~ ., data=data.theory, subset=train)
model.forest <- randomForest(My~.,data=data.theory, subset = train, ntree = 30)
model.nulll <-glm(My~1, data=data.theory, subset=train,family="binomial")

## get predictions: type=response so we have probabilities
pred.logistic.interaction <- predict(model.logistic.interaction, newdata=data.min[-train,], type="response")
pred.logistic <- predict(model.logistic, newdata=data.min[-train,], type="response")
pred.tree <- predict(model.tree, newdata=data.min[-train,],type = "vector")
pred.forest <- predict(model.forest, newdata=data.min[-train,])
pred.null <- predict(model.nulll, newdata=data.min[-train,], type="response")

# Logistic Interaction
values <- FPR_TPR( (pred.logistic.interaction >= val) , My[-train] )

OOS$logistic.interaction[k] <- values$ACC
OOS.TP$logistic.interaction[k] <- values$TP
OOS.FP$logistic.interaction[k] <- values$FP
OOS.TN$logistic.interaction[k] <- values$TN
OOS.FN$logistic.interaction[k] <- values$FN
# Logistic
values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
OOS$logistic[k] <- values$ACC
OOS.TP$logistic[k] <- values$TP
OOS.TN$logistic[k] <- values$TN
OOS.FP$logistic[k] <- values$FP
OOS.FN$logistic[k] <- values$FN
# Tree
values <- FPR_TPR( (pred.tree >= val) , My[-train] )
OOS$tr[k] <- values$ACC
OOS.TP$tr[k] <- values$TP
OOS.TN$tr[k] <- values$TN
OOS.FP$tr[k] <- values$FP
OOS.FN$tr[k] <- values$FN
# Forest
values <- FPR_TPR( (pred.forest >= val) , My[-train] )
OOS$forest[k] <- values$ACC
OOS.TP$forest[k] <- values$TP
OOS.TN$forest[k] <- values$TN
OOS.FP$forest[k] <- values$FP
OOS.FN$forest[k] <- values$FN

#Null
values <- FPR_TPR( (pred.null >= val) , My[-train] )
OOS$null[k] <- values$ACC
OOS.TP$null[k] <- values$TP
OOS.TN$null[k] <- values$TN
OOS.FP$null[k] <- values$FP
OOS.FN$null[k] <- values$FN

View(OOS)

#accuracy on original models
model.logistic_o <-glm(state~., data=kickstarter_model, subset=train,family="binomial")
model.tree_o <- tree(state~ ., data=kickstarter_model, subset=train) 
model.forest_o <- randomForest(state~.,data=kickstarter_model, subset = train, ntree = 10)
model.nulll_o <-glm(state~1, data=kickstarter_model, subset=train,family="binomial")

# ## get predictions: type=response so we have probabilities
pred.logistic_o             <- predict(model.logistic_o, newdata=kickstarter_model[-train,], type="response")
pred.tree_o                 <- predict(model.tree_o, newdata=kickstarter_model[-train,], type="vector")
pred.null_o <- predict(model.nulll_o, newdata=kickstarter_model[-train,], type="response")
pred.forest_o                 <- predict(model.forest_o, newdata=kickstarter_model[-train,], type="response")

values <- FPR_TPR( (pred.logistic_o  >= val) , My[-train] )
Acc_Lo <- values$ACC
values <- FPR_TPR( (pred.tree_o  >= val) , My[-train] )
Acc_Tree <- values$ACC
values <- FPR_TPR( (pred.forest_o  >= val) , My[-train] )
Acc_Forest <- values$ACC
values <- FPR_TPR( (pred.null_o  >= val) , My[-train] )
Acc_Null <- values$ACC

# there is not much difference between Lasso datasets and original datasets and so we use 
# k fold cv for models on Lasso datasets

### create an empty dataframe of results
OOS <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), forest=rep(NA,nfold), null=rep(NA,nfold)) 
PL.OOS.TP <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.TP <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
PL.OOS.TN <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.TN <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
PL.OOS.FP <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.FP <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
PL.OOS.FN <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.FN <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 

OOS.TP <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), forest=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.TN <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), forest=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.FP <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), forest=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.FN <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), forest=rep(NA,nfold), null=rep(NA,nfold))

library(glmnet)
val <- .5
nfold <- 5
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  if ( length(features.1se) == 0){  r1se <- glm(My~1, data=kickstarter_model, subset=train, family="binomial") 
  } else {r1se <- glm(My~., data=data.1se, subset=train, family="binomial")
  }
  
  if ( length(features.theory) == 0){ 
    rtheory <- glm(My~1, data=kickstarter_model, subset=train, family="binomial") 
  } else {rtheory <- glm(My~., data=data.theory, subset=train, family="binomial") }
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  
  values <- FPR_TPR( (predmin >= val) , My[-train] )
  PL.OOS$PL.min[k] <- values$ACC
  PL.OOS.TP$PL.min[k] <- values$TP
  PL.OOS.TN$PL.min[k] <- values$TN
  PL.OOS.FP$PL.min[k] <- values$FP
  PL.OOS.FN$PL.min[k] <- values$FN
  
  values <- FPR_TPR( (pred1se >= val) , My[-train] )
  PL.OOS$PL.1se[k] <- values$ACC
  PL.OOS.TP$PL.1se[k] <- values$TP
  PL.OOS.FP$PL.1se[k] <- values$FP
  PL.OOS.TN$PL.1se[k] <- values$TN
  PL.OOS.FN$PL.1se[k] <- values$FN
  
  values <- FPR_TPR( (predtheory >= val) , My[-train] )
  PL.OOS$PL.theory[k] <- values$ACC
  PL.OOS.TP$PL.theory[k] <- values$TP
  PL.OOS.TN$PL.theory[k] <- values$TN
  PL.OOS.FP$PL.theory[k] <- values$FP
  PL.OOS.FN$PL.theory[k] <- values$FN
  
  ### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassocv1$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassocv1$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  values <- FPR_TPR( (predlassomin >= val) , My[-train] )
  L.OOS$L.min[k] <- values$ACC
  L.OOS.TP$L.min[k] <- values$TP
  L.OOS.TN$L.min[k] <- values$TN
  L.OOS.FP$L.min[k] <- values$FP
  L.OOS.FN$L.min[k] <- values$FN
  values <- FPR_TPR( (predlasso1se >= val) , My[-train] )
  L.OOS$L.1se[k] <- values$ACC
  L.OOS.TP$L.1se[k] <- values$TP
  L.OOS.TN$L.1se[k] <- values$TN
  L.OOS.FP$L.1se[k] <- values$FP
  L.OOS.FN$L.1se[k] <- values$FN
  values <- FPR_TPR( (predlassotheory >= val) , My[-train] )
  L.OOS$L.theory[k] <- values$ACC
  L.OOS.TP$L.theory[k] <- values$TP
  L.OOS.TN$L.theory[k] <- values$TN
  L.OOS.FP$L.theory[k] <- values$FP
  L.OOS.FN$L.theory[k] <- values$FN
  
  # ## fit the two regressions and null model
  # ##### full model uses all 200 signals
  model.logistic <-glm(My~., data=data.theory, subset=train,family="binomial")
  model.tree <- tree(My~ ., data=data.theory, subset=train) 
  model.forest <- randomForest(My~.,data=data.theory, subset = train, ntree = 10)
  model.nulll <-glm(My~1, data=data.theory, subset=train,family="binomial")
  # ## get predictions: type=response so we have probabilities
  pred.logistic <- predict(model.logistic, newdata=data.theory[-train,], type="response")
  pred.tree <- predict(model.tree, newdata=data.theory[-train,], type="vector")
  pred.forest <- predict(model.forest, newdata=data.theory[-train,], type="response")
  pred.null <- predict(model.nulll, newdata=data.theory[-train,], type="response")
  
  # ## calculate accuracy
  # Logistic
  values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
  OOS$logistic[k] <- values$ACC
  OOS.TP$logistic[k] <- values$TP
  OOS.TN$logistic[k] <- values$TN
  OOS.FP$logistic[k] <- values$FP
  OOS.FN$logistic[k] <- values$FN
  # # Tree
  values <- FPR_TPR( (pred.tree >= val) , My[-train] )
  OOS$tree[k] <- values$ACC
  OOS.TP$tree[k] <- values$TP
  OOS.TN$tree[k] <- values$TN
  OOS.FP$tree[k] <- values$FP
  OOS.FN$tree[k] <- values$FN
  # # Forest
  values <- FPR_TPR( (pred.forest >= val) , My[-train] )
  OOS$forest[k] <- values$ACC
  OOS.TP$forest[k] <- values$TP
  OOS.TN$forest[k] <- values$TN
  OOS.FP$forest[k] <- values$FP
  OOS.FN$forest[k] <- values$FN  # #Null
  #Null
  values <- FPR_TPR( (pred.null >= val) , My[-train] )
  OOS$null[k] <- values$ACC
  OOS.TP$null[k] <- values$TP
  OOS.TN$null[k] <- values$TN
  OOS.FP$null[k] <- values$FP
  OOS.FN$null[k] <- values$FN
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

write.csv(ACCperformance,"Accuracy.csv")

summary(model.forest)
importance(model.forest, type = 1)

#plotting accuracy
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))
names(OOS)[1] <-"logistic"
ACCperformance <- cbind(PL.OOS,L.OOS,OOS)
#names(OOS)[1] <-"logistic.interaction"
barplot(colMeans(ACCperformance), xpd=FALSE, ylim=c(0,1), xlab="Method", ylab = "Accuracy")
m.OOS <- as.matrix(ACCperformance)
rownames(m.OOS) <- c(1:nfold)
par(mar=c(1.5,1.5,1.5,1))
par(mai=c(1.5,1.5,1.5,1))
barplot(t(as.matrix(m.OOS)), beside=TRUE, legend=TRUE, args.legend=list(x= "topright", bty = "n", inset = c(-0.28,-0.50)),
        ylab= bquote( "Out of Sample Accuracy"), xlab="Fold", names.arg = c(1:5))

#Lasso path
plot(lassocv1)
library(ggplot2)
colnames(m.OOS)
summary(m.OOS)

### plot FPR and TPR
plot( c( 0, 1 ), c(0, 1), type="n", xlim=c(0,0.5), ylim=c(0,1), bty="n", xlab = "False positive rate", ylab="True positive rate")
lines(c(0,1),c(0,1), lty=2)
#
TPR = sum(OOS.TP$tree)/(sum(OOS.TP$tree)+sum(OOS.FN$tree))  
FPR = sum(OOS.FP$tree)/(sum(OOS.FP$tree)+sum(OOS.TN$tree))  
text( FPR, TPR, labels=c("Tr"))
points( FPR , TPR )
#
TPR = sum(OOS.TP$logistic)/(sum(OOS.TP$logistic)+sum(OOS.FN$logistic))  
FPR = sum(OOS.FP$logistic)/(sum(OOS.FP$logistic)+sum(OOS.TN$logistic))  
text( FPR, TPR, labels=c("LR"))
points( FPR , TPR )
#
TPR = sum(OOS.TP$forest)/(sum(OOS.TP$forest)+sum(OOS.FN$forest))  
FPR = sum(OOS.FP$forest)/(sum(OOS.FP$forest)+sum(OOS.TN$forest))  
text( FPR, TPR, labels=c("RF"))
points( FPR , TPR )
