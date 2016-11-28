basedir <- "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/"
setwd(basedir)

rawdf <- read.csv("/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
testdf <- read.csv("/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_testing.csv")


# boolean 1 or 0
# multiplication

bfeatures <- tail(head(colnames(rawdf),ncol(rawdf)-1),ncol(rawdf)-2)

addFeatures <- function(df,flist,type="multiplication"){
  for(i in 1:(length(bfeatures)-1)){
    flist <- bfeatures[i]
    for(j in (i+1):(length(bfeatures))){
      nfeature <- bfeatures[j]
      if (type=="boolean"){
        df[paste(flist,nfeature,"b",sep="_")] <- ifelse(df[flist] > 0 & df[nfeature] > 0,1,0)  
      }
      if(type=="max"){
        df[paste(flist,nfeature,"max",sep="_")] <- pmax(df[flist],df[nfeature])
      }
      if(type=="min"){
        df[paste(flist,nfeature,"min",sep="_")] <- pmin(df[flist],df[nfeature])
      }
      if(type=="multiply"){
        df[paste(flist,nfeature,"m",sep="_")] <- df[flist] *df[nfeature]
      }
    }
  }
  return(df)  
}


rawdf <- addFeatures(rawdf,bfeatures,"boolean")
testdf <- addFeatures(testdf,bfeatures,"boolean")

# rawdf <- addFeatures(rawdf,bfeatures,"max")
# testdf <- addFeatures(testdf,bfeatures,"max")
# 
# rawdf <- addFeatures(rawdf,bfeatures,"min")
# testdf <- addFeatures(testdf,bfeatures,"min")
# 
# rawdf <- addFeatures(rawdf,bfeatures,"multiply")
# testdf <- addFeatures(testdf,bfeatures,"multiply")

write.csv(rawdf,"data/train.csv",row.names = F)
write.csv(testdf,"data/test.csv",row.names = F)

