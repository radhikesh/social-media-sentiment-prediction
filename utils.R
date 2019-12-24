#functions to create word corpus and dataframe out of it.

word_corpus_df <- function(data,colname){
#creating word corpus:
corpus <- VCorpus(VectorSource(data[[colname]]))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stemDocument)

#creating document matrix:
frequencies <- DocumentTermMatrix(corpus)
sparse <- removeSparseTerms(frequencies, 0.99)
dfsparse <- as.data.frame(as.matrix(sparse))
colnames(dfsparse) <- make.names(colnames(dfsparse))
return(dfsparse)
}


