library(ggplot2)
library(tidyverse)
library(broom)
library(oaxaca)
library(lme4)
library(lmerTest)
mos_df <- read.csv("mos_total.csv")

median_data <- mos_df %>%
    group_by(loc) %>%
    summarise(median_MOS = median(MOS))


violin <- ggplot(mos_df, aes(x = loc, y = MOS, fill=loc)) +
    geom_violin(trim = FALSE) +
    geom_boxplot(width = 0.1, fill = "white", color = "black") +
    labs(x = "Location",
         y = "Mean Opinion Score (MOS)") +
    theme_minimal() +
    theme(legend.position = "none") +  # Remove legend as it's redundant
    coord_cartesian(clip = "off")

violin

ggsave("mos.png", violin, dpi = 300)

# ---------- confounding effect -------
large_wer <- c(25.22, 35.39, 32.56, 41.29, 55.53, 
               31.73, 26.53,17.30, 29.43, 25.34, 
               35.12, 48.11, 24.21, 20.02)
locs <- c("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
whisper_large_wer <- data.frame(
    location=rep(locs, 2),
    wer=large_wer,
    condition = factor(
        rep(c("pre-trained", "fine-tuned"),each = 7), 
        levels = c("pre-trained", "fine-tuned")))

sub_mos_df <- mos_df %>% filter(split %in% c("val", "river"))
wilcox.test(
    sub_mos_df[sub_mos_df$loc == 'river', "MOS"],
    sub_mos_df[sub_mos_df$loc == 'PRV', "MOS"],
    alternative = "greater")
sub_mos_df <- sub_mos_df %>%
    group_by(loc) %>%
    summarise(mos = mean(MOS))

provenance_map <- tibble(
    location = c("ATL", "DCA", "DCB", "LES", "PRV", "VLD", "ROC"),
    provenance = c("mono", "reel", "mono", "mono", "reel", "mono", "mono")
)

urban_map <- tibble(
    location = c("ATL", "DCA", "DCB", "LES", "PRV", "VLD", "ROC"),
    urban_indicator = c("urban", "urban", "urban", "urban", "rural", "rural", "urban")
)

merged_data <- whisper_large_wer %>%
    left_join(sub_mos_df, by = c("location" = "loc")) %>%
    left_join(provenance_map, by = "location") %>%
    left_join(urban_map, by = "location")

merged_data <- merged_data %>%
    mutate(across(c(condition, provenance, urban_indicator), as.factor))

# ---------- regression analysis -----
model <- lm(
    wer ~ mos + provenance + urban_indicator,
    data = merged_data)
summary(model)
tidy_model <- tidy(model)
print(tidy_model)

conf_int <- confint(model)
print(conf_int)

model1 <- lm(
    wer~mos*urban_indicator, data = merged_data)
summary(model1)

model2 <- lmerTest::lmer(wer ~ mos + (1 + mos | location), data = merged_data)
summary(model2)

# ------- plot on MOS vs. WER by provenance
ggplot(merged_data, aes(x = mos, y = wer, color = provenance)) +
    geom_point() +
    geom_smooth(method = "lm", se = TRUE) +
    labs(x = "Median MOS",
         y = "Word Error Rate (WER)",
         color = "Device") +
    theme_minimal()
merged_data <- merged_data %>% filter(! location %in% c("river"))

# --------- correlation --------
cor(merged_data$wer, merged_data$mos, method = "pearson")
cor(merged_data$wer, merged_data$mos, method = "spearman")


