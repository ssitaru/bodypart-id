library(mltest)

# file format: photoid,path,sessionid,diagnosis,date,dataset,bodypart
data <- read.csv("raw/20210628-stats.csv")
# file format: filename,predictedBp,realBp,isCorrect
mlData <- read.csv("../nets/xception_datav3/predict_top1.csv")

data$date <- as.Date(data$date)

# no others
data <- data[data$bodypart!= "other",]

# fix names
data$bodypart[data$bodypart == "torsoback"] <- "torso"
data$bodypart[data$bodypart == "torsofront"] <- "torso"
data$bodypart[data$bodypart == "torsoBack"] <- "torso"
data$bodypart[data$bodypart == "torsoFront"] <- "torso"
data$bodypart[data$bodypart == "legsAndfeet"] <- "legsAndFeet"

hist(data$date[data$date>=as.Date("2005-01-10")], breaks="weeks", freq=TRUE, xlab="Picture date by week")

countsByBp <- as.data.frame(table(data$bodypart))

mlData <- mlData[mlData$predictedBp != "other",]
mlData$predictedBp[mlData$predictedBp == "legsAndfeet"] <- "legsAndFeet"
mlData$realBp[mlData$realBp == "torsoback"] <- "torso"
mlData$realBp[mlData$realBp == "torsofront"] <- "torso"
mlData$realBp[mlData$realBp == "torsoBack"] <- "torso"
mlData$realBp[mlData$realBp == "torsoFront"] <- "torso"
mlData$realBp[mlData$realBp == "legsAndfeet"] <- "legsAndFeet"

mlDataStat <- ml_test(as.factor(mlData$predictedBp), as.factor(mlData$realBp))

# diagnosis groups
data$diaggroup <- ""
data$diaggroup[grep("BCC|SCC|Basalzell|Spino(c|z)ell|PEK|Plattenepi", data$diagnosis, ignore.case=TRUE)] <- "NMSC"
data$diaggroup[grep("Melanom|(Lentigo maligna)", data$diagnosis, ignore.case=TRUE)] <- "Melanoma"
data$diaggroup[grep("Pso", data$diagnosis, ignore.case=TRUE)] <- "Psoriasis"
data$diaggroup[grep("Exanthem", data$diagnosis, ignore.case=TRUE)] <- "Exanthema"
data$diaggroup[grep("pemphig", data$diagnosis, ignore.case=FALSE)] <- "AIBD"
data$diaggroup[grep("N(e|ae)v(u|i)", data$diagnosis, ignore.case=FALSE)] <- "Nevus"
data$diaggroup[grep("Ul(c|z)era", data$diagnosis, ignore.case=TRUE)] <- "Ulcus"
data$diaggroup[grep("Lichen ruber", data$diagnosis, ignore.case=TRUE)] <- "Lichen ruber"
data$diaggroup[grep("Lupus|erythemato", data$diagnosis, ignore.case=TRUE)] <- "Lupus"
data$diaggroup[grep("CTCL|T-Zelll|lymphom|MF|mycosis", data$diagnosis, ignore.case=TRUE)] <- "Lymphoma"
data$diaggroup[grep("scler|morph", data$diagnosis, ignore.case=TRUE)] <- "Sclerodermy"
data$diaggroup[grep("LSA|Lichen scler", data$diagnosis, ignore.case=TRUE)] <- "LSA"
data$diaggroup[grep("dar(r|)ier", data$diagnosis, ignore.case=TRUE)] <- "M. Darier"
data$diaggroup[grep("urticaria pigmen|masto(c|z)ytose", data$diagnosis, ignore.case=TRUE)] <- "Mastocytosis"
data$diaggroup[grep("BCC|SCC|Basalzell|Spino(c|z)ell|PEK|Plattenepi|Spinaliom", data$diagnosis, ignore.case=TRUE)] <- "NMSC"
data$diaggroup[grep("Alope(c|z)", data$diagnosis, ignore.case=TRUE)] <- "Alopecia"
data$diaggroup[grep("A(c|k)ne inversa", data$diagnosis, ignore.case=TRUE)] <- "HS"
data$diaggroup[grep("Zoster", data$diagnosis, ignore.case=TRUE)] <- "Zoster"
data$diaggroup[grep("Dupuytren", data$diagnosis, ignore.case=TRUE)] <- "Dupuytren"
data$diaggroup[grep("Herpes", data$diagnosis, ignore.case=TRUE)] <- "Herpes simplex"
data$diaggroup[grep("eosinophil", data$diagnosis, ignore.case=TRUE)] <- "Hypereosinophilia"
data$diaggroup[grep("e(c|k)zem|atop", data$diagnosis, ignore.case=TRUE)] <- "Eczema"
data$diaggroup[grep("AE", data$diagnosis, ignore.case=FALSE)] <- "Eczema"

# figure confusion matrix
# https://stackoverflow.com/questions/7421503/how-to-plot-a-confusion-matrix-using-heatmaps-in-r
mConfusion <- data.frame(table(mlData$realBp, mlData$predictedBp))
#heatmap(mConfusion, Colv = NA, Rowv = NA, xlab="Labelled body part", ylab="Predicted body part", margins = c(9,9), scale="row")
#legend(x="bottomright", legend=c(0, "ave", max(table(mlData$realBp, mlData$predictedBp))), fill=brewer.pal(3, "OrRd"))
ggplot(mConfusion, aes(x=Var1, y=Var2, fill=Freq)) + 
  geom_tile() + theme_bw()+  coord_equal() +scale_x_discrete(guide = guide_axis(n.dodge=2))+
  labs(x="Labelled", y="Predicted")+guides(fill="none")+
  scale_fill_distiller(palette="YlOrRd", direction=1)+
  geom_text(aes(label=Freq), color="black")

# figure diagnosis % pie
diags <- data.frame(table(data$diaggroup))
diags$Perc <- diags$Freq / sum(diags$Freq) * 100
# filter <2%
diags <- diags[diags$Perc > 2, ]
# sort
diags <- diags[order(diags$Freq, decreasing = TRUE),]
# pie
pie(diags$Freq, labels = diags$Var1)

# figure diagnosis group/body part (run diags first)
diagBp <- data.frame(table(data$diaggroup, data$bodypart))
diagBp <- diagBp[diagBp$Var1 %in% diags$Var1,]
diagBp$bpSum <- ave(diagBp$Freq, diagBp$Var1, FUN=sum)
diagBp$PercRelative <- diagBp$Freq / diagBp$bpSum * 100
diagBpFiltered <- diagBp[diagBp$PercRelative>5, ]
ggplot(diagBpFiltered, aes(x=Var1, y=PercRelative, fill=Var2)) + geom_bar(stat = "identity")

# figure accuracy by bodypart
mlDataFiltered <- mlData
mlDataFiltered$predictedBp[mlDataFiltered$predictedBp == "legsAndfeet"] <- "legsAndFeet" 
mlTest <- ml_test(mlDataFiltered$predictedBp, mlDataFiltered$realBp, output.as.table = TRUE)
mlTest <- data.frame(mlTest)
mlTest$bp <- row.names(mlTest)
ggplot(mlTest, aes(x=bp, y=balanced.accuracy, fill=balanced.accuracy)) + geom_bar(stat="identity", color="black") +
  scale_x_discrete(guide = guide_axis(n.dodge=2))+
  labs(x="Predicted body part", y = "Balanced accuracy")+
  scale_fill_distiller(palette="Greens", direction=1)+
  guides(fill="none")

