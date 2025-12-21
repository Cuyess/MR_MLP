library(readxl)
library(multcomp)
library(writexl)

setwd("/Users/zhaociyun/PycharmProjects/DNN MR")
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
  p <- c()
  p_12 <- c()
  var_names <- c()
  
  for(i in 2:247){
    var_name <- colnames(df)[i]
    tryCatch({
      ancova_model <- aov(as.formula(paste0("`", var_name, "` ~ Gender + Age + education + headmotion + eTIV + mFD + 全脑耦合系数 + Group")), data = df)
      model_summary <- summary(ancova_model)
      
      # Check if Group effect exists
      if("Group" %in% trimws(rownames(model_summary[[1]]))) {
        group_p <- model_summary[[1]][trimws(rownames(model_summary[[1]])) == "Group", "Pr(>F)"]
        postHocs <- glht(ancova_model, linfct = mcp(Group = "Tukey"))
        pairwise_p <- summary(postHocs)$test$pvalues[1]  # nTRD vs TRD (2 vs 1)
        
        p <- c(p, group_p)
        p_12 <- c(p_12, pairwise_p)
        var_names <- c(var_names, var_name)
      }
    }, error = function(e) {
      # Skip variables that cause errors
    })
  }
  
  if(length(p) == 0) {
    return(list(selected_vars = character(0), results_table = data.frame()))
  }
  
  adjusted_p <- p.adjust(p, method = "bonferroni")
  adjusted_p_12 <- p.adjust(p_12, method = "bonferroni")
  names(adjusted_p) <- var_names
  names(adjusted_p_12) <- var_names
  
  # Find significant variables
  sig_mask <- adjusted_p < 0.05 & adjusted_p_12 < 0.05
  sig_var_names <- var_names[sig_mask]
  
  if(length(sig_var_names) > 0) {
    results_table <- data.frame(
      Variable = sig_var_names,
      Adjusted_Overall_P_Bonferroni = adjusted_p[sig_var_names],
      Adjusted_P_nTRD_vs_TRD_Bonferroni = adjusted_p_12[sig_var_names],
      check.names = FALSE
    )
    return(list(selected_vars = sig_var_names, results_table = results_table))
  } else {
    return(list(selected_vars = character(0), results_table = data.frame()))
  }
}

Data <- cov
results_list <- list()

alff_res <- calc(alff)
Data <- cbind(Data, alff[, alff_res$selected_vars, drop = FALSE])
if (nrow(alff_res$results_table) > 0) {
  results_list$alff <- alff_res$results_table
}

falff_res <- calc(falff)
Data <- cbind(Data, falff[, falff_res$selected_vars, drop = FALSE])
if (nrow(falff_res$results_table) > 0) {
  results_list$falff <- falff_res$results_table
}

gmv_res <- calc(gmv)
Data <- cbind(Data, gmv[, gmv_res$selected_vars, drop = FALSE])
if (nrow(gmv_res$results_table) > 0) {
  results_list$gmv <- gmv_res$results_table
}

ReHo_res <- calc(ReHo)
Data <- cbind(Data, ReHo[, ReHo_res$selected_vars, drop = FALSE])
if (nrow(ReHo_res$results_table) > 0) {
  results_list$ReHo <- ReHo_res$results_table
}

scfc_res <- calc(scfc)
Data <- cbind(Data, scfc[, scfc_res$selected_vars, drop = FALSE])
if (nrow(scfc_res$results_table) > 0) {
  results_list$scfc <- scfc_res$results_table
}

if (length(results_list) > 0) {
  write_xlsx(results_list, "significant_p_values_Bonferroni.xlsx")
  cat("Results saved to significant_p_values_Bonferroni.xlsx\n")
  
  # Print summary
  cat("\nSummary of significant variables:\n")
  for(name in names(results_list)) {
    cat(name, ":", nrow(results_list[[name]]), "variables\n")
  }
} else {
  cat("No significant variables found.\n")
}