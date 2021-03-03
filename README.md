# ubiquitous-guide
#RStudio
#RAINFALL-INDUCED LANDSLIDES PREDICTION USING ARTIFICIAL INTELLIGENCE (AI) FOR RISK REDUCTION IN BANJARNEGARA INDONESIA
#Anistia Malinda Hidayat, Adi Mulsandi, Hastuardi Harsa, Bambang Suprihadi
#this script used for plotting the data and predicting the probability of rainfall-induced landslide based on artificial neural network. The type of ANN to be used in this research is ANN-based on probabilistic for gaining classification, so-called Probabilistic Neural Network (PNN).

# clear all
rm(list = ls())
while(!is.null(dev.list())) dev.off()
setwd("D:/script")

# apply function
library(dplyr)
library(moments)

# open/read the data
contoh.data <- read.csv('D:/script/data.csv', header=TRUE, sep = ',')

# understand the distribution and the summary of the data
contoh.data %>% select(-landslide) %>% sapply(function(input) {
  c(summary(input), Skewness = skewness(input), Kurtosis = kurtosis(input))
})

# identify the number of landslide cases
contoh.data %>% mutate(landslide = as.factor(landslide)) %>% select(landslide) %>% summary()

# show the graphic
library(reshape2)
library(ggplot2)
contoh.data.semua <- contoh.data %>% mutate(indeks = 1:nrow(.)) %>% select(-landslide) %>% 
  melt(id = "indeks")
contoh.data.longsor <- contoh.data %>% mutate(indeks = 1:nrow(.)) %>% filter(landslide == 
                                                                               1) %>% select(-landslide) %>% melt(id = "indeks")
# plot
ggplot(mapping = aes(indeks,value)) + geom_line(data = contoh.data.semua) + geom_point(data = contoh.data.longsor,
                                                                                       color = "red", size = 1) + facet_wrap(~variable, scales = "free_y", labeller = label_parsed) + 
  xlab("Index") + ylab("Value") + theme(text = element_text(size=20))

# show the probability value of each variable by separating landslide cases 'Yes' and 'No'.
contoh.data %>% mutate(landslide = ifelse(landslide == 0, "No", "Yes")) %>% melt(id = c("landslide")) %>% 
  ggplot(aes(value)) + geom_density(aes(fill = landslide), alpha = 0.5) + facet_wrap(~variable, 
                                                                                     scale = "free") + xlab("Value") + ylab("Probability") + theme(text = element_text(size=20))

# plot value of variables
library(GGally)
contoh.data %>% select(-landslide) %>% ggpairs(mapping = aes(color = contoh.data$landslide %>% 
                                                               as.factor, alpha = 0.5))

# to decrease its range between variables by utilising log function
contoh.data.log <- contoh.data %>% select(-landslide) %>% +1 %>% log %>% mutate(landslide = contoh.data$landslide %>% 
                                                                                  factor)

contoh.data.log %>% melt(id = "landslide") %>% mutate(landslide = factor(ifelse(landslide == 
                                                                                  0, "No", "Yes"))) %>% ggplot(aes(value)) + geom_density(aes(fill = landslide), 
                                                                                                                                          alpha = 0.5) + facet_wrap(~variable, scale = "free") + xlab("Value") + ylab("Probability") + theme(text = element_text(size=20))
# applying PCA
contoh.data.pca <- prcomp(contoh.data.log %>% select(-landslide), center = TRUE, scale. = TRUE)
contoh.data.pca$variasi <- contoh.data.pca$sdev^2
contoh.data.pca$variasi.persen.akumulasi <- (contoh.data.pca$variasi %>% cumsum) / sum(contoh.data.pca$variasi) * 100

contoh.data.pca$n <- which(contoh.data.pca$variasi.persen.akumulasi >= 90)[1]
contoh.data.pca$n
contoh.data.pca$x[, 1:contoh.data.pca$n] %>% 
  data.frame %>% 
  mutate(landslide = factor(ifelse(contoh.data.log$landslide == 0, 'No', 'Yes'))) %>% 
  ggplot(aes(PC1, PC2, color = landslide)) +
  geom_point()

# PNN function
p.x.new.ci <- function(x.new, data.input, sigma.input, gamma.input = NULL) {
  distance <- dist(rbind(x.new, data.input))[1:nrow(data.input)]
  distance <- distance^2
  distance <- -distance
  argument <- exp(distance / (2 * (sigma.input^2)))
  omega <- argument / (
    ((2 * pi)^(length(x.new) / 2)) * (sigma.input^length(x.new))
  )
  if(!is.null(gamma.input) & is.numeric(gamma.input)) {
    omega.sum <- data.input %>% nrow %>% '*'(gamma.input) %>% round
    return(rev(sort(omega))[1:omega.sum] %>% mean(na.rm = TRUE))
  } else {
    omega %>% mean(na.rm = TRUE)
  }
}

c.pnn <- function(x.new, data.input, class.input, sigma.input, gamma.input = NULL) {
  class.index <- tapply(1:length(class.input), class.input, function(input1) return(input1))
  p.x.new.result <- sapply(
    class.index, function(input1) {
      p.x.new.ci(x.new, data.input[input1, ], sigma.input, gamma.input)
    }
  )
  p.x.new.result <- (p.x.new.result / sum(p.x.new.result)) %>% round(3)
  output <- data.frame(
    class.output = (class.input %>% as.factor %>% levels)[p.x.new.result %>% which.max],
    probability = p.x.new.result[p.x.new.result %>% which.max]
  )
  rownames(output) <- NULL
  return(output)
}

# defining training and testing data
training <- contoh.data.pca$x[1:1461, 1:contoh.data.pca$n] %>% data.frame
testing <- contoh.data.pca$x[1462:1826, 1:contoh.data.pca$n] %>% data.frame

# defining response variable of the data
respon.training <- contoh.data.log$landslide[1:1461]
respon.testing <- contoh.data.log$landslide[1462:1826]


# run pnn program using sigma value = 1.0
hasil.pnn <- apply(
  testing, 1, function(input) {
    c.pnn(
      x.new = input,
      data.input = training,
      class.input = respon.training,
      sigma.input = 1.0
    )
  }
)

# PNN result:
tabel.kontingensi <- table(
  Observasi = contoh.data.log$landslide[1462:1826] %>% factor,
  Model = sapply(hasil.pnn, function(input) input$class.output)
)

# showing the result in table
tabel.kontingensi
