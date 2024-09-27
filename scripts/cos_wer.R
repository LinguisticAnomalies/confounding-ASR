library(ggplot2)

wer_data <- read.csv("cos_res_wer.csv")
cer_data <- read.csv("cos_res_cer.csv")
wer_data <- subset(wer_data, (score <1) & (score > 0) & (cos > 0) & (cos < 1))

cer_data <- subset(cer_data, score <1)

wer_wrong <- subset(wer_data, cos >= 0.5)
wer_30 <- subset(wer_data, score > 0.3)


plot_scatter <- function(df, cutoff=cutoff) {
    df <- subset(df, score > cutoff)
    # Plot the scatter plot with jittering and adjusted point size
    ggplot(df, aes(x = score, y = cos, color = model)) +
        geom_point(
            position = position_jitter(width = 0.1, height = 0),
            size = 2) +
        labs(
            x = "Word Error Rate",
            y = "Cosine Similarity") +
        scale_x_continuous(breaks = seq(cutoff, 1, by = 0.05)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

plot_scatter_cer <- function(df, cutoff=cutoff) {
    df <- subset(df, score > cutoff)
    # Plot the scatter plot with jittering and adjusted point size
    ggplot(df, aes(x = score, y = cos, color = model)) +
        geom_point(
            position = position_jitter(width = 0.1, height = 0),
            size = 2) +
        labs(
            x = "Character Error Rate",
            y = "Cosine Similarity") +
        scale_x_continuous(breaks = seq(cutoff, 1, by = 0.05)) +
        theme(axis.text.x=element_text(angle=90, hjust=1))
        theme_minimal()
}

p1 <- plot_scatter(wer_data, 0)
p1
p2 <- plot_scatter_cer(cer_data, 0.3)
p2
ggsave("cos-wer-coraal.png", p1, dpi=300)
ggsave("cos-cer-coraal.png", p2, dpi=300)
