library(caret)
library(h2o)
h2o.init()

#training models

#creating variable names as vector for models: (Title)
xt <- setdiff(names(train_title), "SentimentTitle")
yt <- "SentimentTitle"

#creating variable names as vector for models: (Headline)
xh <- setdiff(names(train_Headline),"SentimentHeadline")
yh <- "SentimentHeadline"

#####################################################################################################################
################################### model:1 random forest with basic parameters #####################################
#Training Title: 
random.h2o_title<-h2o.randomForest(x=xt,y=yt,training_frame = train.h2o_title,ntrees=50,seed = 10)
h2o.mae(random.h2o_title, train = T)
h2o.performance(model = random.h2o_title, newdata = test.h2o_title)

#Training Headline:
random.h2o_hl<-h2o.randomForest(x=xh,y=yh,training_frame = train.h2o_hl,ntrees=50,seed = 10)
h2o.mae(random.h2o_hl, train = T)
h2o.performance(model = random.h2o_hl, newdata = test.h2o_hl)

#predicting sentimentTitle in final test dataset
p_test_title <- as.data.frame(h2o.predict(random.h2o_title,test_final_ti_h2o))
colnames(p_test_title) <- "SentimentTitle"

#predicting sentimentHeadline in final test dataset:
p_test_Headline <- as.data.frame(h2o.predict(random.h2o_hl,test_final_hl_h2o))
colnames(p_test_Headline) <- "SentimentHeadline"

#creating submission file
submissionfile <- cbind(testfinal[,"IDLink"],p_test_title,p_test_Headline)
write_csv(submissionfile,path = "results/submission_01.csv")

########################################################################################################################
####################################### model:2 random forest with tuned parameter #####################################
#Training Title: 
random.h2o_title<-h2o.randomForest(x=xt,y=yt,training_frame = train.h2o_title,
                                   ntrees=100,seed = 10,nfolds = 10,
                                   stopping_rounds = 10)

h2o.mae(random.h2o_title, train = T)
h2o.performance(model = random.h2o_title, newdata = test.h2o_title)

#Training Headline:
random.h2o_hl<-h2o.randomForest(x=xh,y=yh,training_frame = train.h2o_hl,
                                ntrees=100,seed = 10,
                                nfolds = 10,stopping_rounds = 10)
h2o.mae(random.h2o_hl, train = T)
h2o.performance(model = random.h2o_hl, newdata = test.h2o_hl)

#predicting sentimentTitle in final test dataset
p_test_title <- as.data.frame(h2o.predict(random.h2o_title,test_final_ti_h2o))
colnames(p_test_title) <- "SentimentTitle"

#predicting sentimentHeadline in final test dataset:
p_test_Headline <- as.data.frame(h2o.predict(random.h2o_hl,test_final_hl_h2o))
colnames(p_test_Headline) <- "SentimentHeadline"

#creating submission file
submissionfile <- cbind(testfinal[,"IDLink"],p_test_title,p_test_Headline)
write_csv(submissionfile,path = "results/submission_02.csv")

##########################################################################################################################
################################### model3: gradient boosting with tuned parameters: #####################################
#training title:
gbm.h2o_title <- h2o.gbm(x=xt, y=yt, training_frame = train.h2o_title,
                         nfolds = 5,seed=10,
                         ntrees = 500,stopping_rounds = 10)

h2o.mae(gbm.h2o_title, train = T)
h2o.performance(model = gbm.h2o_title, newdata = test.h2o_title)

#training headline:
gbm.h2o_hl<-h2o.gbm(x=xh,y = yh,training_frame = train.h2o_hl,
                             ntrees=500,seed = 10,
                             nfolds = 5,stopping_rounds = 10)

h2o.mae(gbm.h2o_hl, train = T)
h2o.performance(model = gbm.h2o_hl, newdata = test.h2o_hl)

#predicting sentimentTitle in final test dataset
p_test_title <- as.data.frame(h2o.predict(gbm.h2o_title,test_final_ti_h2o))
colnames(p_test_title) <- "SentimentTitle"

#predicting sentimentHeadline in final test dataset:
p_test_Headline <- as.data.frame(h2o.predict(gbm.h2o_hl,test_final_hl_h2o))
colnames(p_test_Headline) <- "SentimentHeadline"

#creating submission file
submissionfile <- cbind(testfinal[,"IDLink"],p_test_title,p_test_Headline)
write_csv(submissionfile,path = "results/submission_03.csv")

###########################################################################################################################
########################################model 4: using gbm grid search: ###################################################

#for grid search using training and validation dataframe
split_ti <- h2o.splitFrame(train.h2o_title, ratios = 0.75)
train_ti <- split_ti[[1]]
valid_ti <- split_ti[[2]]

split_hl <- h2o.splitFrame(train.h2o_hl, ratios = 0.75)
train_hl <- split_hl[[1]]
valid_hl <- split_hl[[2]]

#hyper parameter grid:
hyper_grid <- list(max_depth = c(1,3,5),
                   min_rows = c(1,5,10),
                   learn_rate = c(0.01,0.05,0.1),
                   learn_rate_annealing=c(.99,1),
                   sample_rate=c(.5,.75,1),
                   col_sample_rate=c(.8,.9,1)
)

#perform grid search on title:
grid_ti <- h2o.grid(algorithm = "gbm",
                    grid_id = "gbm_ti_grid1",
                    x=xt,
                    y=yt,
                    training_frame = train_ti,
                    validation_frame = valid_ti,
                    hyper_params = hyper_grid,
                    ntrees=5000,
                    stopping_rounds=10,
                    stopping_tolerance=0,
                    seed=12)

#due to time constraint, didn't the full grid search on headline, used the parameters from full grid search on title
#perform grid search on headline:
# grid_hl <- h2o.grid(algorithm = "gbm",
#                     grid_id = "gbm_hl_grid1",
#                     x=xh,
#                     y=yh,
#                     training_frame = train_hl,
#                     validation_frame = valid_hl,
#                     hyper_params = hyper_grid,
#                     ntrees=5000,
#                     stopping_rounds=10,
#                     stopping_tolerance=0,
#                     seed=12)

grid_ti@model_ids[1]
best_model_id_ti <- grid_ti@model_ids[[1]]
best_model <- h2o.getModel(best_model_id_ti)
#using best model from grid search and running with cross validation
gbm.h2o_gridti <- h2o.gbm(x=xt,y=yt,training_frame = train.h2o_title,nfolds = 5,
                          ntrees = 5000,learn_rate = 0.1,max_depth = 5,
                          min_rows = 5,sample_rate = 0.75,
                          col_sample_rate = 0.8,stopping_rounds = 10, stopping_tolerance = 0,
                          seed = 12)
h2o.mae(gbm.h2o_gridti, train = T)
h2o.performance(model = gbm.h2o_gridti, newdata = test.h2o_title)

#using the parameters for grid search on title and applying the same parameters for headline:
gbm.h2o_gridhl <- h2o.gbm(x=xh,y=yh,training_frame = train.h2o_hl,nfolds = 5,
                          ntrees = 5000,learn_rate = 0.1,max_depth = 5,
                          min_rows = 5,sample_rate = 0.75,
                          col_sample_rate = 0.8,stopping_rounds = 10, stopping_tolerance = 0,
                          seed = 12)

h2o.mae(gbm.h2o_gridhl, train = T)
h2o.performance(model = gbm.h2o_gridhl, newdata = test.h2o_hl)

#predicting title:
p_test_title <- as.data.frame(h2o.predict(gbm.h2o_gridti,test_final_ti_h2o))
colnames(p_test_title) <- "SentimentTitle"

#predicting headline:
p_test_Headline <- as.data.frame(h2o.predict(gbm.h2o_gridhl,test_final_hl_h2o))
colnames(p_test_Headline) <- "SentimentHeadline"

#creating submission file:
submissionfile <- cbind(testfinal[,"IDLink"],p_test_title,p_test_Headline)
write_csv(submissionfile,path = "results/submission_05.csv") ### score of 90.30

###########################################################################################################################
############################################## model 5: gbm with random search ####################################

search_criteria <- list(strategy = "RandomDiscrete",
                        stopping_metric="mae",
                        stopping_tolerance=0.005,
                        stopping_rounds=10,
                        max_runtime_secs=60*60)

#radnom search on title:
grid_gbm_ti_sc <- h2o.grid(algorithm = "gbm",
                           grid_id = "gbm_grid2_ti",
                           x=xt,
                           y=yt,
                           training_frame = train_ti,
                           validation_frame = valid_ti,
                           hyper_params = hyper_grid,
                           search_criteria = search_criteria,
                           ntrees=5000,
                           stopping_rounds=10,
                           stopping_tolerance=0,
                           seed=12)

#random search on headline:
grid_gbm_hl_sc <- h2o.grid(algorithm = "gbm",
                           grid_id = "gbm_grid2_hl",
                           x=xh,
                           y=yh,
                           training_frame = train_hl,
                           validation_frame = valid_hl,
                           hyper_params = hyper_grid,
                           search_criteria = search_criteria,
                           ntrees=5000,
                           stopping_rounds=10,
                           stopping_tolerance=0,
                           seed=12)

#applying best model for title from random search criteria on full training dataset with cross validation
grid_gbm_ti_sc@grid_id
grid_perf_ti <- h2o.getGrid(grid_id = "gbm_grid2_ti",sort_by = "mae",decreasing = F)
best_model_ti <- grid_perf_ti@model_ids[[1]]
best_model_ti <- h2o.getModel(best_model_ti)
h2o.performance(best_model_ti,valid = T)
best_model_ti@parameters
gbm.h2o_gridtisc <- h2o.gbm(x=xt,y=yt,training_frame = train.h2o_title,nfolds = 5,
                            ntrees = 5000,learn_rate = 0.01,max_depth = 3,
                            min_rows = 1,sample_rate = 0.75,
                            col_sample_rate = 0.9,stopping_rounds = 10, stopping_tolerance = 0,
                            seed = 12, distribution = "gaussian")
h2o.mae(gbm.h2o_gridtisc, train = T)
h2o.performance(model = gbm.h2o_gridtisc, newdata = test.h2o_title)

#best model for headline from random search criteria on full training dataset with cross validation
grid_gbm_hl_sc@grid_id
grid_perf_hl <- h2o.getGrid(grid_id = "gbm_grid2_hl",sort_by = "mae",decreasing = F)
best_model_hl <- grid_perf_hl@model_ids[[1]]
best_model_hl <- h2o.getModel(best_model_hl)
h2o.performance(best_model_hl,valid = T)
best_model_hl@parameters
gbm.h2o_gridhlsc <- h2o.gbm(x=xh,y=yh,training_frame = train.h2o_hl,nfolds = 5,
                            ntrees = 5000,learn_rate = 0.01,max_depth = 5,
                            min_rows = 1,sample_rate = 0.5,
                            col_sample_rate = 1,stopping_rounds = 10, stopping_tolerance = 0,
                            seed = 12, distribution = "gaussian")
h2o.mae(gbm.h2o_gridhlsc, train = T)
h2o.performance(model = gbm.h2o_gridhlsc, newdata = test.h2o_hl)

#predicting title:
p_test_title <- as.data.frame(h2o.predict(gbm.h2o_gridtisc,test_final_ti_h2o))
colnames(p_test_title) <- "SentimentTitle"

#predicting headline
p_test_Headline <- as.data.frame(h2o.predict(gbm.h2o_gridhlsc,test_final_hl_h2o))
colnames(p_test_Headline) <- "SentimentHeadline"

#submission for gbm using random search criteria:
submissionfile <- cbind(testfinal[,"IDLink"],p_test_title,p_test_Headline)
write_csv(submissionfile,path = "results/submission_08.csv")