library(ggplot2)
library(viridis)
library(tidyverse)
locs <- c("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
# pre-trained, fine-tuned
large_wer <- c(25.22, 35.39, 32.56, 41.29, 55.53, 31.73, 26.53,
               17.30, 29.43, 25.34, 35.12, 48.11, 24.21, 20.02)
large_cer <- c(13.69, 23.76, 23.22, 29.75, 43.54, 23.07, 17.26,
               11.16, 20.98, 18.93, 26.58, 37.36, 18.87, 12.98)
wilcox.test(large_wer[1:7], large_wer[8:14], paired = TRUE, alternative = "greater")
whisper_large_wer <- data.frame(
    location=rep(locs, 2),
    wer=large_wer,
    condition = factor(rep(c("pre-trained", "fine-tuned"), each = 7), levels = c("pre-trained", "fine-tuned")))
whisper_large_cer <- data.frame(
    location=rep(locs, 2),
    cer=large_cer,
    condition = factor(rep(c("pre-trained", "fine-tuned"), each = 7), levels = c("pre-trained", "fine-tuned")))

whisper_large_wer_plot <- ggplot(
    whisper_large_wer, aes(fill=condition, y=wer, x=location)) +
    geom_bar(position="dodge", stat="identity") + 
    geom_text(
        aes(label = sprintf("%.2f", wer)),
        position = position_dodge(width = 0.9),
        vjust = -0.5,
        size = 3) +
    labs(fill = "Condition") +
    # ggtitle("WER on CORAAL Validation Set") +
    theme_minimal() +
    ylim(0, 60) +
    theme(legend.position = c(0.90, 0.90))
whisper_large_cer_plot <- ggplot(
    whisper_large_cer, aes(fill=condition, y=cer, x=location)) +
    geom_bar(position="dodge", stat="identity") + 
    geom_text(
        aes(label = sprintf("%.2f", cer)),
        position = position_dodge(width = 0.9),
        vjust = -0.5,
        size = 3) +
    labs(fill = "Condition") +
    ggtitle("CER on CORAAL Validation Set") +
    ylim(0, 60) +
    theme_minimal() +
    theme(legend.position = c(0.90, 0.90))
whisper_large_wer_plot
whisper_large_cer_plot
ggsave("whisper_large_wer_plot.png", whisper_large_wer_plot, dpi = 300)
ggsave("whisper_large_cer_plot.png", whisper_large_cer_plot, dpi = 300)
# ----------------------
pt_ag1 <- c(27.92, 34.54, 34.34, 41.30, 47.09, 19.29, 24.05)
ft_ag1 <- c(19.28, 28.39, 28.11, 34.73, 39.11, 11.84, 16.41)
pt_ag2 <- c(21.78, 34.30, 31.04, 45.46, 58.23, 27.89, 26.47)
ft_ag2 <- c(14.78, 28.76, 23.76, 36.77, 49.36, 19.57, 20.40)
pt_ag3 <- c(NA, 45.28, 31.92, 38.92, 58.67, 60.36, 28.53)
ft_ag3 <- c(NA, 39.65, 23.54, 34.30, 52.62, 53.95, 22.32)
df_pt <- data.frame(locs, pt_ag1, pt_ag2, pt_ag3)
df_ft <- data.frame(locs, ft_ag1, ft_ag2, ft_ag3)
wilcox.test(ft_ag2, ft_ag3, paired = TRUE)

data <- data.frame(
    age_group = rep(c("< 30", "31 - 50", "< 50"), each = 7),
    pt_ag1 = pt_ag1,
    ft_ag1 = ft_ag1,
    pt_ag2 = pt_ag2,
    ft_ag2 = ft_ag2,
    pt_ag3 = pt_ag3,
    ft_ag3 = ft_ag3
)
data_long <- pivot_longer(
    data, cols = starts_with("pt_ag") | starts_with("ft_ag"), 
    names_to = "condition", values_to = "WER")

# Create ggplot2 plot
ggplot(data_long, aes(x = age_group, y = WER, fill = condition)) +
    geom_bar(position = "dodge", stat = "identity", width = 0.7, color = "black") +
    scale_fill_manual(
        values = c("pt_ag1" = "blue", "ft_ag1" = "red", "pt_ag2" = "lightblue", 
                   "ft_ag2" = "pink", "pt_ag3" = "darkblue", "ft_ag3" = "darkred")) +
    labs(title = "Pre-trained vs. Fine-tuned Comparison",
         x = "Age Group",
         y = "WER",
         fill = "Condition") +
    theme_minimal() +
    # Add a meaningful legend
    theme(
        legend.title = element_blank(), 
        legend.position = "top", legend.text = element_text(size = 10))

