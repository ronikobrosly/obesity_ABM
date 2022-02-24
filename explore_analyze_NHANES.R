library(survey)
library(stringr)

df <- read.csv("~/Desktop/transformed_NHANES.csv")


# Here we use "subset" to tell "nhanesDesign" that we want to only look at a
# specific subpopulation (i.e., those age between 18-79 years). This is
# important to do. If you don't do this and just restrict it in a different way
# your estimates won't have correct SEs.

df <- read.csv("~/Desktop/transformed_NHANES.csv")
df2 <- read.csv("~/Desktop/transformed_NHANES_2.csv")
new_df <- rbind(df, df2)
new_df$WTMEC2YR <- new_df$WTMEC2YR / 2

nhanesDesign <- svydesign(
  id = ~SDMVPSU,
  strata  = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest = TRUE,
  data = new_df
)


for (val in ((3:15)*5))
{
  ageDesign <- subset(nhanesDesign, AGE >= val & AGE <= (val + 5))
  print(val + 2.5)
  print(svymean(~OBESE, ageDesign))
  print("")
}

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






