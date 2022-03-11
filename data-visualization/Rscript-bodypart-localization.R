# https://www.youtube.com/watch?v=93_JSqQ3aG4&ab_channel=MakingSenseRemotely
# body map plot with IDW

library(ggplot2)
library(png)
library(gstat)
library(raster)
library(rgdal)
library(dplyr)

# import data
# file format: "","Var1","Var2","Freq","bpSum","PercRelative"
diagBp <- read.csv("2021.11.07-diagBp.csv")
background<- png::readPNG("../manuscript/figures/bp-distribution/body-shape.png")

# plot relative percentages of bp/diagnosis
ggplot(diagBp, aes(x=Var1, y=PercRelative, fill=Var2)) + geom_bar(stat = "identity")

# filter diagnoses (only >150 images, no other, no ND)
diagBpFiltered <- diagBp[diagBp$bpSum>150,]
diagBpFiltered <- diagBpFiltered[diagBpFiltered$Var1 != "",]
diagBpFiltered <- diagBpFiltered[diagBpFiltered$Var1 != "ND",]

unique(diagBpFiltered$Var1)

# for which diagnosis should we generate + cleanup
diagBpForX <- diagBp[diagBp$Var1 == "Psoriasis",]
diagBpForX$X <- NULL
diagBpForX$Var1 <- as.character.factor(diagBpForX$Var1)
diagBpForX$Var2 <- as.character.factor(diagBpForX$Var2)
#diagBpForX$Var1 <- NULL

# import bodypart to coordinate mapping
#bpToCoord <- data.frame(bps, bpsX, bpsY)
bpToCoord <- read.csv("bpToCoord.csv")
bpToCoord = bpToCoord %>% arrange(bps)

# prepare data (anal + genitals* -> genitalsAnal)
data = diagBpForX
data = data %>%
  mutate(Var2 = replace(Var2, Var2=="anal", "genitalsAnal")) %>%
  mutate(Var2 = replace(Var2, Var2=="genitalsFemale", "genitalsAnal")) %>%
  mutate(Var2 = replace(Var2, Var2=="genitalsMale", "genitalsAnal")) %>%
  #filter(Var2 == "genitalsAnal") %>%
  group_by(Var1, Var2) %>%
  summarise(Freq=sum(Freq)) %>%
  as.data.frame()
data$z = data$Freq / sum(data$Freq) * 100
data$bps = data$Var2
data$Var2 = NULL
data = data %>% arrange(bps)

# merge data
data = full_join(data, bpToCoord, on="bps")

# mirror of arms, armsAndHands, hands, legs, legsAndFeet, feet
data = data %>%
  filter(bps == "arms") %>%
  mutate(bps="armsX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
data = data %>%  
  filter(bps == "armsAndHands") %>%
  mutate(bps="armsAndHandsX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
data = data %>% 
  filter(bps == "hands") %>%
  mutate(bps="handsX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
data = data %>% 
  filter(bps == "legs") %>%
  mutate(bps="legsX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
data = data %>% 
  filter(bps == "legsAndFeet") %>%
  mutate(bps="legsAndFeetX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
data = data %>% 
  filter(bps == "feet") %>%
  mutate(bps="feetX", bpsX=100-bpsX) %>%
  bind_rows(data, .)
  

# generate grid
grid <-expand.grid(x=seq(0, 100), y=seq(0, 100))
coordinates(grid) = ~x+y

pts = data
coordinates(pts) = ~ bpsX + bpsY
proj4string(pts) = proj4string(grid)
idw = idw(formula = z~1, locations = pts, newdata = grid)
idwdf = data.frame(idw)

ggplot() + 
  #geom_point(data= grid, aes(x, y), size=0.1)+
  #geom_point(data=bpToCoord, aes(bpsX, bpsY), color="red")+
  geom_tile(data=idwdf, aes(x, y, fill=var1.pred))+
  scale_fill_gradientn(colors = terrain.colors(10))+
  xlim(0, 100)+
  ylim(0, 100)+
  coord_fixed(ratio=2.5)+
  annotation_raster(background, xmin=0, xmax = 100, ymin=0, ymax=100)+
  labs(fill = "% of images")
  #geom_text(data=bpToCoord, aes(bpsX, bpsY, label=bps), hjust=0, vjust=0)
  

  
