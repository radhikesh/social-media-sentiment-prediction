#loading libraries:
library(tm)
source("utils.R")
library(h2o)
h2o.init()

#############################################################################################################################
#####################################processing title from training dataset##################################################
titleSparse <- word_corpus_df(data = dataset, colname = "Title")
#adding other columns from dataset
titleSparse <- cbind(titleSparse,dataset[,c("Source","Topic","Facebook","GooglePlus","LinkedIn","SentimentTitle")])

#splitting training into train and test(validation):
trainIndex <- createDataPartition(titleSparse$SentimentTitle,p=0.75, list = F)
train_title <- titleSparse[trainIndex,] 
test_title <- titleSparse[-trainIndex,]

#creating h2o dataframes:
train.h2o_title<-as.h2o(train_title)
test.h2o_title<-as.h2o(test_title)

###############################################################################################################################
######################################processing headline from training dataset################################################
HeadlineSparse <- word_corpus_df(data = dataset, colname = "Headline")
#adding other columns from dataset
HeadlineSparse <- cbind(HeadlineSparse,dataset[,c("Source","Topic","Facebook","GooglePlus","LinkedIn","SentimentHeadline")])

#splitting training into train and test(validation):
trainIndex <- createDataPartition(HeadlineSparse$SentimentHeadline,p=0.75, list = F)
train_Headline <- HeadlineSparse[trainIndex,] 
test_Headline <- HeadlineSparse[-trainIndex,]

#creating h2o dataframes:
train.h2o_hl<-as.h2o(train_Headline)
test.h2o_hl<-as.h2o(test_Headline)

################################################################################################################################
######################################processing title from test dataset########################################################
testSparse_title <- word_corpus_df(data = testfinal, colname = "Title")
#adding other columns from test dataset
testSparse_title <- cbind(testSparse_title,testfinal[,c("Source","Topic","Facebook","GooglePlus","LinkedIn")])
test_final_ti_h2o <- as.h2o(testSparse_title)

############################################################################################################################
#######################################processing Headline from test dataset################################################
testSparse_hl <- word_corpus_df(data = testfinal, colname = "Headline")
#adding other columns from dataset
testSparse_hl <- cbind(testSparse_hl,testfinal[,c("Source","Topic","Facebook","GooglePlus","LinkedIn")])
test_final_hl_h2o <- as.h2o(testSparse_hl)
