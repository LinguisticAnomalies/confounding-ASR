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
