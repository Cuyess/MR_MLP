library(readxl)
library(multcomp)

setwd("/Users/zhaociyun/PycharmProjects/DNN MR")

del <- read.csv("/Users/zhaociyun/PycharmProjects/DNN MR include outer test111/external test.csv",header=F)[,1]

alff <- read_excel("ALFF.xlsx")
names(alff)[-1] <- paste("alff", names(alff)[-1],sep = "_")
falff <- read_excel("fALFF.xlsx")
names(falff)[-1] <- paste("falff", names(falff)[-1],sep = "_")
gmv <- read_excel("GMV.xlsx")
names(gmv)[-1] <- paste("gmv", names(gmv)[-1],sep = "_")
ReHo <- read_excel("ReHo.xlsx")
names(ReHo)[-1] <- paste("ReHo", names(ReHo)[-1],sep = "_")
scfc <- read_excel("ALFF_sc_fc_coordination.xlsx")
names(scfc)[-1] <- paste("scfc", names(scfc)[-1],sep = "_")
cov <- read_excel("cov.xlsx")
cov$Group <- as.factor(cov$Group)
cov$Gender <- as.factor(cov$Gender)

calc <- function(df){
  df <- merge(df, cov, by = "编号")
  df <- df[!(df$`编号` %in% del),]
  p <- c()
  p_12 <- c()
  for(i in 2:247){
    ancova_model <- aov(as.formula(paste0("`", colnames(df)[i],
                "` ~ Gender + Age + education + headmotion + eTIV + mFD + 全脑耦合系数 + Group")),
                data = df)
    p <- c(p, summary(ancova_model)[[1]][8,5])
    postHocs <- glht(ancova_model, linfct = mcp(Group = "Tukey"))
    p_12 <- c(p_12, summary(postHocs)$test$pvalues[1])
  }
  adjusted_p <- p.adjust(p, method = "fdr")
  adjusted_p_12 <- p.adjust(p_12, method = "fdr")
  print(adjusted_p)
  return(colnames(df)[which(adjusted_p < 0.05 & adjusted_p_12 < 0.05) + 1])
}

Data <- cov
Data <- cbind(Data, alff[,calc(alff)])
Data <- cbind(Data, falff[,calc(falff)])
Data <- cbind(Data, gmv[,calc(gmv)])
Data <- cbind(Data, ReHo[,calc(ReHo)])
Data <- cbind(Data, scfc[,calc(scfc)])
write.csv(Data,"fdr.csv")