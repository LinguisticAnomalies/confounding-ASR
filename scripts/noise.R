library(ggplot2)
library(tidyverse)
locs <- c("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
wer_values_nr <- c(24.74, 36.05, 31.32, 37.52, 52.47, 27.43, 27.05,
                10.02, 18.34, 17.37, 24.20, 38.28, 13.65, 13.58)
conditions <- rep(c("pre-trained", "fine-tuned"), each = 7)

wer_values <- c(25.22, 35.39, 32.56, 41.29, 55.53, 31.73, 26.53,
                17.30, 29.43, 25.34, 35.12, 48.11, 24.21, 20.02)


whisper_large_nr <- data.frame(
    location = factor(rep(locs, 2), levels = locs),
    wer = as.numeric(wer_values_nr),
    condition = factor(conditions, levels = c("pre-trained", "fine-tuned"))
)

whisper_large_wer <- data.frame(
    location = factor(rep(locs, 2), levels = locs),
    wer = as.numeric(wer_values),
    condition = factor(conditions, levels = c("pre-trained", "fine-tuned"))
)


whisper_combined <- rbind(
    transform(whisper_large_wer, noise_cancellation = "Original"),
    transform(whisper_large_nr, noise_cancellation = "Noise Cancelled")
)

whisper_combined_plot <- ggplot(
    whisper_combined, 
    aes(fill = condition, y = wer, x = location, group = interaction(condition, noise_cancellation))
) +
    geom_bar(position = position_dodge2(width = 0.9, preserve = "single"), stat = "identity", alpha = 0.7) + 
    geom_text(
        aes(label = sprintf("%.2f", wer)),
        position = position_dodge2(width = 0.9, preserve = "single"),
        vjust = -0.5,
        size = 2.5
    ) +
    labs(
        x = "Geographical location",
        y = "Word Error Rate (WER) (%)",
        fill = "Whisper model type"
    ) + 
    scale_fill_manual(values = c("pre-trained" = "#1f77b4", "fine-tuned" = "#ff7f0e")) +
    facet_wrap(~ noise_cancellation, nrow = 2) +
    theme_minimal() +
    ylim(0, 90) +
    theme(
        legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

whisper_combined_plot
ggsave("whisper_large_wer_nr.png", whisper_combined_plot, dpi = 300)

whisper_large_wer_plot <- ggplot(
    whisper_large_nr, aes(fill=condition, y=wer, x=location)) +
    geom_bar(position="dodge", stat="identity") + 
    geom_text(
        aes(label = sprintf("%.2f", wer)),
        position = position_dodge(width = 0.9),
        vjust = -0.5,
        size = 3) +
    labs(fill = "Condition") +
    # ggtitle("WER on CORAAL Validation Set") +
    labs(
        x = "Geographical location",           # Change x-axis label
        y = "Word Error Rate (WER) (%)", # Change y-axis label
        fill = "Whisper model type",       # Change legend label
    ) + 
    theme_minimal() +
    ylim(0, 90) +
    theme(legend.position = c(0.15, 0.90))
whisper_large_wer_plot
