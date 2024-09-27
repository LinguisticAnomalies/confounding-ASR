library(ggplot2)
library(tidyverse)
library(lme4)
library(broom)
library(texreg)
val_df <- read.csv("total_val.csv")
val_df$loc <- factor(
    val_df$loc,
    levels = c("ATL", "DCB", "DCA", "LES", "ROC", "PRV", "VLD"))
cor(val_df$MOS, val_df$ppl)
cor(val_df$MOS, val_df$pt_wer)
cor(val_df$MOS, val_df$ft_wer)

ggplot(val_df, aes(x = MOS, y = ppl)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", color = "red") +
    facet_wrap(~ loc) +
    theme_minimal() +
    labs(
         x = "Mean Opinion Score (MOS)",
         y = "Perpelxity")

# mixed effect linear model
pt_model <- lmer(pt_wer ~ MOS + ppl + (1 | loc), data = val_df)
summary(pt_model)
pt_model_interaction <- lmer(pt_wer ~ MOS * ppl + (1 | loc), data = val_df)
anova(pt_model, pt_model_interaction)
summary(pt_model_interaction)


ft_model <- lmer(ft_wer ~ MOS + ppl + (1 | loc), data = val_df)
summary(ft_model)
ft_model_interaction <- lmer(ft_wer ~ MOS * ppl + (1 | loc), data = val_df)
anova(ft_model, ft_model_interaction)
summary(ft_model_interaction)
