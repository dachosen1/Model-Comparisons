# load libraries
library(data.table)
library(DT)
library(caret)
library(randomForest)
library(nnet)
library(class)
library(e1071)
library(glmnet)
library(xgboost)
library(lubridate)
library(xgboost)
library(archdata)
library(Ckmeans.1d.dp)
library(plyr)
library(dplyr)
library(gbm)
library(class)
library(nnet)

# functions
create.formula <-
  function(outcome.name,
           input.names,
           input.patterns = NA,
           all.data.names = NA,
           return.as = "character") {
    variable.names.from.patterns <- c()
    if (!is.na(input.patterns[1]) & !is.na(all.data.names[1])) {
      pattern <- paste(input.patterns, collapse = "|")
      variable.names.from.patterns <-
        all.data.names[grep(pattern = pattern, x = all.data.names)]
    }
    all.input.names <-
      unique(c(input.names, variable.names.from.patterns))
    all.input.names <- all.input.names[all.input.names !=
                                         outcome.name]
    if (!is.na(all.data.names[1])) {
      all.input.names <-
        all.input.names[all.input.names %in% all.data.names]
    }
    input.names.delineated <- sprintf("`%s`", all.input.names)
    the.formula <-
      sprintf("`%s` ~ %s",
              outcome.name,
              paste(input.names.delineated, collapse = " + "))
    if (return.as == "formula") {
      return(as.formula(the.formula))
    }
    if (return.as != "formula") {
      return(the.formula)
    }
  }

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

standardize <- function(x) {
  return((x - mean(x)) / sd(x))
}

round.numerics <- function(x, digits) {
  if (is.numeric(x)) {
    x <- round(x = x, digits = digits)
  }
  return(x)
}

# load data
test.file.path <- '../Data/MNIST-fashion testing set-49.csv'
train.file.path <- '../Data/MNIST-fashion training set-49.csv'

train <- fread('../Data/MNIST-fashion training set-49.csv')
test <- fread(test.file.path)

# constants
n.values <- c(500, 1000, 2000)
iterations <- 3
WeightRow <- 0.25
WeightTime <- 0.25
WeightError <- 0.5

label.name = 'label'
input.names = names(train)[-1]


# sample data names
sampledata.names <- c()
for (n in n.values) {
  for (k in (1:iterations)) {
    index <- sample(1:nrow(train), n)
    dat_name <- paste('dat', n, k, sep = '_')
    assign(dat_name, train[index,])
    sampledata.names <- c(sampledata.names, dat_name)
  }
}

# multinomial model
formula.ml <-
  create.formula(outcome.name = label.name, input.names = input.names)

mlr.ml <- function(dat.name) {
  dat <- get(dat.name)
  Model = 'Multinomial Logistic regression'
  `Sample Size` = nrow(dat)
  Data = dat.name
  A = round(nrow(dat) / nrow(train), 4)
  start = Sys.time()
  ml.model = multinom(formula.ml,
                      data = dat,
                      maxit = 300,
                      trace = FALSE)
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  result = predict(ml.model, test[, -1])
  C = round(1 - mean(result == test[, get(label.name)]), 4)
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

# load results
ml.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- MLR(sampledata.names[i])
  ml.result <- rbind(ml.result, result)
}

# KNN model
knn.ml <- function(dat.name, K = 10) {
  dat <- get(dat.name)
  Model = 'KNN'
  `Sample Size` = nrow(dat)
  Data = dat.name
  A = round(nrow(dat) / nrow(train), 4)
  start = Sys.time()
  result = knn(dat[, -1], test[, -1], dat[, get(label.name)], k = K)
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  C = round(1 - mean(result == test[, get(label.name)]), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}
# load results
knn.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- KNN(sampledata.names[i], K = 5)
  knn.result <- rbind(knn.result, result)
}

# Descsion trees
formula.ct = create.formula(outcome.name = label.name, input.names = input.names)

CT <- function(dat.name, Test = test) {
  dat <- get(dat.name)
  
  Model = 'Classification Tree'
  `Sample Size` = nrow(dat)
  Data = dat.name
  A = round(nrow(dat) / nrow(train), 4)
  start = Sys.time()
  model <- rpart(formula = formula.ct,
                 data = dat,
                 method = 'class')
  pred <-
    predict(object = model,
            newdata = Test[, -1],
            type = 'class')
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  C = round(1 - mean(pred == Test[, get(label.name)]), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

# load results
ct.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- CT(sampledata.names[i])
  ct.result <- rbind(ct.result, result)
}


# random forest
rf.ml <- function (dat.name,
                   Test = test,
                   ntree = 100) {
  dat <- get(dat.name)
  
  Model = 'Random Forest'
  `Sample Size` = nrow(dat)
  Data = dat.name
  
  A = round(nrow(dat) / nrow(train), 4)
  dat$label = factor(dat$label)
  start = Sys.time()
  forest = randomForest(label ~ ., data = dat, n.tree = ntree)
  pred = predict(forest, newdata = test)
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  C = round(1 - mean(unlist(pred) == unlist(test[, 1])), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

# load results
rf_result <- data.frame()

for (i in 1:length(sampledata.names)) {
  rf_model <- rf.ml(dat.name = sampledata.names[i])
  rf_result <- rbind(rf_model, rf_result)
}

# Support Vector machine
svm.ml <- function (dat, test.name = test) {
  Data = dat
  dat.name <- as.data.frame(get(dat))
  test.set <- test.name[, 1]
  Model = 'Support Vector Machine'
  `Sample Size` = nrow(dat.name)
  A = round(nrow(dat.name) / nrow(train), 4)
  tic = Sys.time()
  model.svm <-
    train(
      label ~ .,
      data = dat.name,
      method = "svmLinear",
      preProcess = c("center", "scale"),
      tuneLength = 10
    )
  toc = Sys.time()
  results <- predict(model.svm, test.name[, -1])
  B = round(min(1, as.numeric(toc - tic) / 60), 4)
  C = round(1 - mean(results == test.name[, get(label.name)]), 4)
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, Data, `Sample Size`, A, B, C, Points))
}

# load results
svm_result <- data.frame()
for (i in 1:length(sampledata.names)) {
  model <- svm.ml(dat = sampledata.names[i], test.name = test)
  svm_result <- rbind(model, svm_result)
}

# lasso regression
lr.lasso.ml <- function(dat.name) {
  dat <- get(dat.name)
  Model = 'Lasso Regression'
  `Sample Size` = nrow(dat)
  Data = dat.name
  A = round(nrow(dat) / nrow(train), 4)
  start = Sys.time()
  y <- as.factor(unclass(unlist(dat[, get(label.name)])))
  alpha1.fit <-
    glmnet(as.matrix(dat[, -1]), y, alpha = 1, family = 'multinomial')
  alpha1.pred <-
    predict(alpha1.fit, newx = as.matrix(test[, -1]), type = 'class')
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  test.pred = as.factor(unclass(unlist(test[, get(label.name)])))
  C = round(1 - mean(alpha1.pred == test.pred), 4)
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}
# load results
lr.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- lr.lasso.ml(sampledata.names[i])
  lr.result <- rbind(lr.result, result)
}


# Lasso regression model
lr.ridge.ml <- function(dat.name) {
  dat <- get(dat.name)
  Model = 'Lasso Regression'
  `Sample Size` = nrow(dat)
  Data = dat.name
  A = round(nrow(dat) / nrow(train), 4)
  start = Sys.time()
  y <- as.factor(unclass(unlist(dat[, get(label.name)])))
  alpha1.fit <-
    glmnet(as.matrix(dat[, -1]), y, alpha = 0, family = 'multinomial')
  alpha1.pred <-
    predict(alpha1.fit, newx = as.matrix(test[, -1]), type = 'class')
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  
  test.pred = as.factor(unclass(unlist(test[, get(label.name)])))
  C = round(1 - mean(alpha1.pred == test.pred), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

# load results
rr.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- lr.ridge.ml(sampledata.names[i])
  rr.result <- rbind(rr.result, result)
}

#--- XG Boost
# Parameter selection
hyper_grid <- expand.grid(
  eta = c(.01, .05, .07, .1),
  max_depth = c(7, 10, 15),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1),
  colsample_bytree = c(.8, .9, 1)
)

for (i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  set.seed(123)
  
  xgb.tune <- xgb.cv(
    params = params,
    data = data.matrix(x_train),
    label = train$price,
    eval_metric = "merror",
    nrounds = 150,
    num_class = 10,
    objective = "multi:softprob",
    # for regression models
    verbose = 0,
    # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <-
    which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] <-
    min(xgb.tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)


# model development
train$label <- as.numeric(train$label) - 1
test$label <- as.numeric(test$label) - 1

sampledata.names <- c()
for (n in n.values) {
  for (k in (1:iterations)) {
    index <- sample(1:nrow(train), n)
    dat_name <- paste('dat', n, k, sep = '_')
    assign(dat_name, train[index, ])
    sampledata.names <- c(sampledata.names, dat_name)
  }
}
train <- data.matrix(train)
train_x <- t(train[, -1])
train_y <- train[, 1]
train_array <- train_x
dim(train_array) <- c(7, 7, 1, ncol(train_x))

test <- data.matrix(test)
test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x

xgb_params <- list(
  "objective" = "multi:softprob",
  "eval_metric" = "merror",
  "num_class" = 10
)
nround    <- 120 # number of XGBoost rounds
train_data   <- train[, -1]
train_label  <- train[, 1]
train_matrix <-
  xgb.DMatrix(data = data.matrix(train_data), label = train_label)
# split test data and make xgb.DMatrix
test_data  <- test[, -1]
test_label <- test[, 1]
test_matrix <-
  xgb.DMatrix(data = data.matrix(test_data), label = test_label)

xgb.ml <- function(dat.name) {
  dat <- get(dat.name)
  label <- dat[, 1]
  dat_matrix <-
    xgb.DMatrix(data = data.matrix(dat[, -1]), label = label)
  
  Model = 'XGBoosting'
  `Sample Size` = nrow(dat)
  Data = dat.name
  
  A = round(nrow(dat) / nrow(train), 4)
  
  start = Sys.time()
  xgb.model =  xgb.train(params = xgb_params,
                         data = dat_matrix,
                         nrounds = 200)
  
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  
  xgb_val_preds = predict(xgb.model, newdata = test_matrix)
  
  xgb_val_out = matrix(xgb_val_preds,
                       nrow = 10,
                       ncol = length(xgb_val_preds) / 10) %>%
    t() %>%
    data.frame() %>%
    mutate(max = max.col(., ties.method = 'last'), label = test_label + 1)
  result = xgb_val_out$max
  
  C = round(1 - mean(result == test_label + 1), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

# load results
xgb.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- xgb.ml(sampledata.names[i])
  xgb.result  <- rbind(xgb.result , result)
}

# gradient boosting model
gbm.ml <- function(dat.name) {
  dat <- get(dat.name)
  label <- dat[, 1]
  
  
  Model = 'GBM'
  `Sample Size` = nrow(dat)
  Data = dat.name
  
  A = round(nrow(dat) / nrow(train), 4)
  
  start = Sys.time()
  gbm.model =  gbm(
    label ~ .,
    data = dat[-1],
    n.trees = 150,
    distribution = 'multinomial'
  )
  
  end = Sys.time()
  B = round(min(1, as.numeric(end - start, units = 'secs') / 60), 4)
  
  predictionMatrix = predict(gbm.model,
                             newdata = test_data,
                             n.trees = 150,
                             type = 'response')
  p.predictionMatrixT = apply(predictionMatrix, 1, which.max)

  result = p.predictionMatrixT
  
  C = round(1 - mean(result == test_label + 1), 4)
  
  Points = round(WeightRow * A + WeightTime * B + WeightError * C, 4)
  
  return(data.frame(Model, `Sample Size`, Data, A, B, C, Points))
}

gbm.result <- data.frame()
for (i in 1:length(sampledata.names)) {
  result <- gbm.ml(sampledata.names[i])
  gbm.result  <- rbind(gbm.result , result)
}

# comparison
score <-
  data.table(
    rbind(
      ml.result,
      knn.result,
      ct.result,
      e.result,
      rf_result,
      gbm.result,
      svm_result,
      lr.result,
      rr.result,
      xgb.result
    )
  )
model = 'Model'
size = "Sample.Size"
meanscore <- score[, .(
  'A' = mean(A),
  'B' = mean(B),
  'C' = mean(C),
  'Points' = mean(Points)
), by = c(model, size)]
score <- score[, lapply(X = .SD, FUN = 'round.numerics', digits = 4)]
meanscore <-
  meanscore[, lapply(X = .SD, FUN = 'round.numerics', digits = 4)]
datatable(score[order(Points), ])
datatable(meanscore[order(Points)])
