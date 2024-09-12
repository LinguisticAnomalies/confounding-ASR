library(ggplot2)
library(dplyr)
ppls <- read.csv("../data/char_gpt_ppl.csv")
ppls <- ppls %>%
    mutate(
        region = ifelse(train_corpus %in% c("PRV", "VLD"), "Rural", "Urban")
    )

ppls$train_corpus <- factor(
    ppls$train_corpus,
    levels = c("ATL", "DCB", "DCA", "LES", "ROC", "PRV", "VLD"))

p <- ggplot(
    ppls, aes(x = train_corpus, 
              y = ppl, 
              color = eval_corpus,
              shape = region)) +
    geom_point(position = position_dodge(width = 0.3),
               size = 3) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), 
                  position = position_dodge(width = 0.3),
                  width = 0.5) +
    labs(x = "Training Component",
         y = "Perplexity",
         color = "Validating Component",
         shape = "Region") +
    theme_minimal() +
    theme(legend.position = c(0.90, 0.80),
          legend.text = element_text(size = 10)) +
    guides(color = guide_legend(ncol = 2))
p
ggsave("char_gpt.png", p, dpi = 300)
