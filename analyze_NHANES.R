library(survey)
library(stringr)

df <- read.csv("~/Desktop/transformed_NHANES.csv")

age_df <- subset(df, AGE >= 18 & AGE <= 24)
age_df <- subset(df, AGE >= 25 & AGE <= 34)
age_df <- subset(df, AGE >= 35 & AGE <= 44)
age_df <- subset(df, AGE >= 45 & AGE <= 54)
age_df <- subset(df, AGE >= 55 & AGE <= 64)
age_df <- subset(df, AGE >= 65 & AGE <= 74)
age_df <- subset(df, AGE >= 75)



model <- glm(
  formula = OBESE ~ MALE + HISPANIC + BLACK + ASIAN + LS_HS + HS + LOW_INCOME
  + MALE:HISPANIC + MALE:BLACK + MALE:ASIAN + 
    HISPANIC:LOW_INCOME + BLACK:LOW_INCOME + ASIAN:LOW_INCOME, 
  family = "binomial", 
  data = age_df
)



sum_model <- summary(model)
cl = sum_model$coefficients[0:14]

p1 = cl[1]
p2 = cl[2]
p3 = cl[3]
p4 = cl[4]
p5 = cl[5]
p6 = cl[6]
p7 = cl[7]
p8 = cl[8]
p9 = cl[9]
p10 = cl[10]
p11 = cl[11]
p12 = cl[12]
p13 = cl[13]
p14 = cl[14]

str_glue('(1 / (1 + math.exp(-( {p1} + {p2}*MALE +  {p3}*HISPANIC + {p4}*BLACK +  {p5}*ASIAN + {p6}*LS_HS + {p7}*HS + {p8}*LOW_INCOME +  {p9}*MALE*HISPANIC + {p10}*MALE*BLACK +  {p11}*MALE*ASIAN + {p12}*HISPANIC*LOW_INCOME +  {p13}*BLACK*LOW_INCOME + {p14}*ASIAN*LOW_INCOME ))))')



MALE = 1
HISPANIC = 1
BLACK = 0
ASIAN = 0
LS_HS = 0
HS = 1
LOW_INCOME = 1

p = 1 / (1 + exp(-(-0.59278 + 0.15686*MALE + 0.18525*HISPANIC + 0.63385*BLACK + -1.17589*ASIAN + 0.17082*LS_HS + 0.46123*HS + -0.26761*LOW_INCOME + -0.05495*MALE*HISPANIC + -0.81182*MALE*HISPANIC + -0.02498*MALE*ASIAN + 0.41395*HISPANIC*LOW_INCOME + -0.02153*BLACK*LOW_INCOME + 0.16342*ASIAN*LOW_INCOME) ))












# Here we use "subset" to tell "nhanesDesign" that we want to only look at a
# specific subpopulation (i.e., those age between 18-79 years). This is
# important to do. If you don't do this and just restrict it in a different way
# your estimates won't have correct SEs.

df <- read.csv("~/Desktop/transformed_NHANES.csv")

nhanesDesign <- svydesign(
  id = ~SDMVPSU,
  strata  = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest = TRUE,
  data = df
)


ageDesign <- subset(nhanesDesign, AGE >= 18 & AGE <= 24)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 25 & AGE <= 34)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 35 & AGE <= 44)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 45 & AGE <= 54)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 55 & AGE <= 64)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 65 & AGE <= 74)
svymean(~OBESE, ageDesign)

ageDesign <- subset(nhanesDesign, AGE >= 75)
svymean(~OBESE, ageDesign)






