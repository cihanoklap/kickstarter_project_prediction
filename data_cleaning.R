library(readr)
library(lubridate)
library(tidyverse)
library(glmnet)
library(randomForest)
library(tree)

rm(list=ls())


### The dataset we've been used for this analysis is coming from Kaggle Public Datasets.
### You can check it from: https://www.kaggle.com/kemical/kickstarter-projects#ks-projects-201801.csv

kickstarter <- read.csv("final_kickstarter.csv")


# Data cleaning and transformations

# removing irrelevant columns like id, name etc.
kickstarter <- kickstarter[,-1]
kickstarter <- kickstarter[,-9]
kickstarter <- kickstarter[,-7]
kickstarter <- kickstarter[,-5]

# converting category columns to factor
kickstarter$state <- as.factor(kickstarter$state)

# retaining year and months but converting to factor in order to avoid time series effect
kickstarter$year <- as.factor(kickstarter$year)
kickstarter$month <- as.factor(kickstarter$month)

kickstarter$main_category <- as.factor(kickstarter$main_category)
kickstarter$category <- as.factor(kickstarter$category)

#converting outcome to 1 and 0
kickstarter$state <- as.character(kickstarter$state)
kickstarter$state[kickstarter$state == "failed"] <- 0
kickstarter$state[kickstarter$state == "successful"] <- 1
kickstarter$state <- as.numeric(kickstarter$state)

# checking distribution
hist(kickstarter$usd_goal_real,bin = 100)
ggplot(kickstarter, aes(y = usd_goal_real)) + 
  geom_boxplot(outlier.colour="red",
               outlier.size=2)

# exploratory checks for country
summary(kickstarter$usd_goal_real)
check <- kickstarter %>% filter(usd_goal_real < 5) %>%
  group_by(country) %>% summarise(counts = n())

## Using Log of Pledge and Goal because the ranges are widely distributed ($ amounts)
kickstarter$log_usd_pledged_real <- log(kickstarter$usd_pledged_real)
kickstarter$log_usd_pledged_real[!is.finite(kickstarter$log_usd_pledged_real)] <- 0
kickstarter$log_usd_goal_real <- log(kickstarter$usd_goal_real)
kickstarter$log_usd_goal_real[!is.finite(kickstarter$log_usd_goal_real)] <- 0

#treating 0 values for log
kickstarter$log_usd_goal_real <- log(kickstarter$usd_goal_real)
kickstarter$log_usd_goal_real[!is.finite(kickstarter$log_usd_goal_real)] <- 0

#transformation of launch date and deadline date to difference between them
kickstarter_model$days <- as.Date(kickstarter_model$deadline) - as.Date(kickstarter_model$launched)

# removing unneccesary variables
kickstarter_model <- kickstarter_model[,-c(1,3,4,10)]
