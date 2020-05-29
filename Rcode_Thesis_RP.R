library(MASS)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(extraDistr)
library(ggplot2)
library(brms)
library(rstan)
rstan_options(auto_write = TRUE)

library(bayesplot)
library(tictoc)
library(gridExtra)
library(parallel)
library(DataCombine)
library(MCMCvis)

set.seed(333)
ftrs <- read_csv("CSAI/Thesis/ftrs (1).csv")

## Distribution of Mistakes 
boxplot_mis = boxplot(ftrs$Nr_Mistakes, main='Boxplot: Nr of Mistakes')
distr_mis = hist(ftrs$Nr_Mistakes) 

## Group PoS Tags 
# count occurences of every tag, 
levels(ftrs$PoS)
fill <- NULL
for(i in 1:15){
  data = (filter(ftrs, PoS==levels(ftrs$PoS)[i]))
  fill[i] <- nrow(data)
}

# Group PoS tags with less than 100.000, under others. Final categories: Adj, Det, Noun, Pron, Propn, Verb, and Others.
to_replace <- c("ADP", "ADV", "AUX", "CONJ", "INTJ", "NUM", "PART", "SCONJ", "X")
for (tag in to_replace){
  ftrs[ftrs==tag]<-"OTHER"
}
ftrs <- filter(ftrs, PoS!='OTHER')

## Log transform freq and exp and scale ftrs
# Numerical: 
ftrs$Frequency <- log(ftrs$Frequency)
ftrs$Word_exp <- log(ftrs$Word_exp)
ftrs[c(6, 8, 9, 10, 11)] <- scale(ftrs[c(6, 8, 9, 10, 11)])

##  Contrast Coding Categorical ftrs 
ftrs$Task_Format <- as.factor(ftrs$Task_Format)
ftrs$PoS <- as.factor(ftrs$PoS)
ftrs$Track <- as.factor(ftrs$Track)
ftrs$Unfamiliar_Sound <- as.factor(ftrs$Unfamiliar_Sound)

# Sum contrast coding, and treatment for unfamiliar sound
contrasts(ftrs$Task_Format) <- contr.sum(3)
contrasts(ftrs$PoS) <- contr.sum(6)
contrasts(ftrs$Track) <- contr.sum(3)
contrasts(ftrs$Unfamiliar_Sound) <- contr.treatment(2)  


## Summary Statistics 
es_data <- filter(ftrs, Track=='es_en')
en_data <- filter(ftrs, Track=='en_es')
fr_data <- filter(ftrs, Track=='fr_en')

# number of zero mistakes
zero_mis <- filter(ftrs, Nr_Mistakes == 0)
# 1.004.753

#Five or less mistakes:
five_or_less <- filter(ftrs, Nr_Mistakes < 5)
# 1.290.460

en_users = unique(en_data$User)
es_users = unique(es_data$User)
fr_users = unique(fr_data$User)

en_words = unique(en_data$Word)
es_words = unique(es_data$Word)
fr_words = unique(fr_data$Word)

length(en_users)
length(en_words)
length(es_users)
length(es_words)
length(fr_users)
length(fr_words)

listen <- filter(ftrs, Task_Format == "listen")    ## A tibble: 421,999 x 13
tap <- filter(ftrs, Task_Format == "reverse_tap")  ## A tibble: 396,796 x 13
translate <- filter(ftrs, Task_Format =="reverse_translate") # A tibble: 476,098 x 13

## Collinearity Check
freq_len <- cbind(ftrs$Frequency, ftrs$Word_length)
freq_len_plot <- pairs(freq_len)

## Analysis
# Set Priors
get_prior(formula = Nr_Mistakes ~ Task_Format + PoS + Word_length + Track + Word_exp + Concreteness + Frequency + Distance + 
            Unfamiliar_Sound + (1 + Task_Format | Word) + (Task_Format + Concreteness + Frequency + Distance + Unfamiliar_Sound + Word_length | User), 
          data = ftrs, family = zero_inflated_poisson())


priors <- c(set_prior("normal(0, 5)", class = "Intercept"), 
            set_prior("normal(0, 1)", class = "b"),
            set_prior("normal(0, 1)", class = "sd"))


## Train the model on 2 different stratified samples (20 users per track), total of approximately 14000 observations per sample
## to check if coefficient values are relatively consistent across samples, indicating that samples are statistically representative
Summs_1 <- NULL
for(i in 1:2){
  sample_en <- sample(en_users, 20)
  sample_es <- sample(es_users, 20)
  sample_fr <- sample(fr_users, 20)
  sample_users <- c(sample_en, sample_es, sample_fr)
  strat_sample <- filter(ftrs, User %in% sample_users)
  SS <- brm(formula = Nr_Mistakes ~  Task_Format + PoS + Word_length + Track + Word_exp + Concreteness + Frequency + Distance + Unfamiliar_Sound +
              + (1 | Word) + (1| User),
            data = strat_sample, family = zero_inflated_poisson(), prior = priors,
            iter = 3000, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))
  Summs_1[[i]] <- summary(SS)
}

Summs_2 <- NULL
for(i in 1:2){
  sample_en <- sample(en_users, 25)
  sample_es <- sample(es_users, 25)
  sample_fr <- sample(fr_users, 25)
  sample_users <- c(sample_en, sample_es, sample_fr)
  strat_sample <- filter(ftrs, User %in% sample_users)
  SS <- brm(formula = Nr_Mistakes ~  Task_Format + PoS + Word_length + Track + Word_exp + Concreteness + Frequency + Distance + Unfamiliar_Sound +
              + (1 | Word) + (1| User),
            data = strat_sample, family = zero_inflated_poisson(), prior = priors,
            iter = 3000, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))
  Summs_2[[i]] <- summary(SS)
}

# Sample for training and comparing models 
set.seed(248)
sample_en <- sample(en_users, 20)
sample_es <- sample(es_users, 20)
sample_fr <- sample(fr_users, 20)
sample_users <- c(sample_en, sample_es, sample_fr)
strat_sample <- filter(ftrs, User %in% sample_users)

# Baseline model
MC1 <- brm(formula = Nr_Mistakes ~ Task_Format + PoS + Track + Word_exp + Concreteness + Frequency + Distance + Word_length +  Unfamiliar_Sound +
             (1 | Word) + (1 | User),
           data = strat_sample, family = zero_inflated_poisson(), prior = priors,
           iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))

# check model/likelihood fit
pp_check(MC1, nsamples = 200)

# Feature Selection
MC2 <- brm(formula = Nr_Mistakes ~  Task_Format + PoS + Word_exp + Concreteness  + Distance  + Word_length +
             (1 | Word) + (1 | User),
           data = strat_sample, family = zero_inflated_poisson(), prior = priors,
           iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))

# Model comparison 
MC1 <- add_criterion(MC1,'waic')
MC2 <- add_criterion(MC2, 'waic')
comp <- loo_compare(MC1, MC2, criterion="waic")
comp


# Add Interaction Terms
IM1 <- brm(formula = Nr_Mistakes ~  Task_Format  + PoS  + Word_exp + Concreteness + Distance + Word_length +
               PoS: Task_Format + Concreteness:Task_Format+ Distance:Task_Format + Word_length:Task_Format + 
               (1 | Word) + (1 | User),
               data = strat_sample, family = zero_inflated_poisson(), prior = priors,
               iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))

IM1 <- add_criterion(IM1, "waic")
comp_2 <- loo_compare(MC2, IM1, criterion="waic")



# Add Random Slopes
priors <- c(set_prior("normal(0, 5)", class = "Intercept"), 
            set_prior("normal(0, 1)", class = "b"),
            set_prior("normal(0, 1)", class = "sd"),
            set_prior("lkj(2)", class = "cor")) 

RS2 <- brm(formula = Nr_Mistakes ~  Task_Format  + PoS  + Word_exp + Concreteness + Distance + Word_length 
           + Task_Format:PoS + Task_Format:Concreteness + Task_Format:Distance + Task_Format:Word_length + 
            (1 | Word) + (Task_Format + PoS + Concreteness + Distance + Word_length + Task_Format:PoS +
            Task_Format:Concreteness + Task_Format:Distance + Task_Format:Word_length | User),
           data = strat_sample, family = zero_inflated_poisson(), prior = priors, inits = 0,
           iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 12))

RS2 <- add_criterion(RS2, "waic")
comp_3 <- loo_compare(MC2, IM1, RS1, criterion="waic")


## New stratified sample, to ensure best model is still best model on new sample and not 'overfitted'
sample_en_2 <- sample(en_users, 20)
sample_es_2 <- sample(es_users, 20)
sample_fr_2 <- sample(fr_users, 20)
sample_users_2 <- c(sample_en_2, sample_es_2, sample_fr_2)
strat_sample_2 <- filter(ftrs, User %in% sample_users_2)

FMC1 <- brm(formula = Nr_Mistakes ~  Task_Format  + PoS  + Word_exp + Concreteness + Frequency + Distance + Word_length +
              (1 | Word) + (1 | User),
            data = strat_sample_2, family = zero_inflated_poisson(), prior = priors,
            iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))

FMC2 <- brm(formula = Nr_Mistakes ~  Task_Format  + PoS  + Word_exp + Concreteness + Distance + Word_length +
             PoS:Task_Format: Concreteness:Task_Format + Distance:Task_Format + Word_length:Task_Format + 
              (1 | Word) + (1 | User),
            data = strat_sample_2, family = zero_inflated_poisson(), prior = priors,
            iter = 2500, chains = 4, cores = 4, control = list(adapt_delta = 0.99, max_treedepth = 10))

FMC3 <- brm(formula = Nr_Mistakes ~  Task_Format  + PoS  + Word_exp + Concreteness + Distance + Word_length +
              + Task_Format:PoS + Task_Format:Concreteness + Task_Format:Distance + Task_Format:Word_length + 
              (1 | Word) + (Task_Format + PoS + Frequency + Concreteness + Distance +
              Task_Format:PoS + Task_Format:Concreteness + Task_Format:Distance + Task_Format:Word_length | User),
            data = strat_sample_2, family = zero_inflated_poisson(), prior = priors,
            iter = 2500, chains = 4, cores = 4, inits = 0, control = list(adapt_delta = 0.99, max_treedepth = 12))

FMC1 <- add_criterion(FMC1, "waic")
FMC2 <- add_criterion(FMC2, "waic")
FMC3 <- add_criterion(FMC3, "waic")

final_comp <- loo_compare(FMC1, FMC2, FMC3, criterion="waic")
final_comp

## Plots for Visualizing Results

# Coefficents and CI for Baseline model
BL_coef <- MCMCplot(MC1, params = c('Intercept', 'Task_Format1', 'Task_Format2', 'PoS1', 'PoS2', 'PoS3', 'PoS4', 'PoS5',  
                                    'Track1', 'Track2', 'Word_exp', 'Concreteness', 'Frequency', 'Distance', 'Word_length', 'Unfamiliar_Sound2'),
                    labels = c('Intercept', 'Task:Listen', 'Task:Tap', 'PoS:Adj', 'PoS:Det', 'PoS:Noun', 'PoS:Pron', 'PoS:PropN', 
                               'Track:En_Es', 'Track:Es_En', 'Word_exp', 'Concreteness', 'Frequency', 'Distance', 'Word_length', 'Unfamiliar_Sound') )

# Plots conditional effects of the interaction terms 
int_effects <- plot(conditional_effects(IM3))


## Posterior Distributions of maximal model

# Population-level effects
mcmc_plot(RS1, pars = c("^b_"), type="hist")

# Group-level effects
mcmc_plot(RS1, pars = c("sd_"), type="hist")

# Seperate Random intercept User and Word
sd_random_int <- mcmc_plot(RS1, pars = c("sd_User__I", "sd_Word__I"), type="hist")









