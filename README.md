[Data Mining & Machine Learning final project.pptx](https://github.com/agnesstuprojects/Data-Mining-Machine-Learning-final-project-Spring-1-2022/files/8158039/Data.Mining.Machine.Learning.final.project.pptx)
[Blog_Data Mining Project ANN and SVM.docx](https://github.com/agnesstuprojects/Data-Mining-Machine-Learning-final-project-Spring-1-2022/files/8155132/Blog_Data.Mining.Project.ANN.and.SVM.docx)
[Data Mining & Machine Learning final project.pptx](https://github.com/agnesstuprojects/Data-Mining-Machine-Learning-final-project-Spring-1-2022/files/8155133/Data.Mining.Machine.Learning.final.project.pptx)
[Readme_Data Mining Project.docx](https://github.com/agnesstuprojects/Data-Mining-Machine-Learning-final-project-Spring-1-2022/files/8155134/Readme_Data.Mining.Project.docx)
[Telco_Customer_Churn.csv](https://github.com/agnesstuprojects/Data-Mining-Machine-Learning-final-project-Spring-1-2022/files/8155135/Telco_Customer_Churn.csv)
# Data-Mining-Machine-Learning-final-project-Spring-1-2022
Churn Prediction in Telecoms Industry Using R 

Telecommunication market is expanding day by day. Companies are facing a severe loss of revenue due to increasing competition hence the loss of customers. They are trying to find the reasons of losing customers by measuring customer loyalty to regain the lost customers. The customers leaving the current company and moving to another telecom company are called churn. This Analysis will use ANN and SVM models to find the best model for the study.
This Analysis will use ANN and SVM models to find the best model for the study.

Code in R
# Read data
```{r}
Customer_Churn <- read.csv("Telco_Customer_Churn.csv", na.strings = c("null", "NA"), stringsAsFactors = TRUE)
head(Customer_Churn)
```

#Names of Columns
```{r}
names(Customer_Churn)
```

#Table Organization
```{r}
glimpse(Customer_Churn)
```

# Clean the Data
```{r}
##Find NA observations
Customer_Churn_NA <- subset(Customer_Churn, is.na(Customer_Churn$TotalCharges))

##Remove them 
Customer_Churn <- na.omit(Customer_Churn)

##Change SeniorCitizen (0,1) to factors
Customer_Churn$SeniorCitizen <- as.factor(
  mapvalues(Customer_Churn$SeniorCitizen,
            from=c("0","1"),
            to=c("No", "Yes"))
)
##Change tenure to numerical
Customer_Churn[5] <- lapply(Customer_Churn[5], as.numeric)
```
#Analyzing the three continuous variables CHURN:
#Tenure: The median tenure for customers who have left is around 10 months.
```{r}
options(repr.plot.width =6, repr.plot.height = 2)
ggplot(Customer_Churn, aes(y= tenure, x = "", fill = Churn)) + 
geom_boxplot()+ 
theme_bw()+
xlab(" ")
```
#MonthlyCharges: Customers who have churned, have high monthly charges. The median is above 75.
```{r}
ggplot(Customer_Churn, aes(y= MonthlyCharges, x = "", fill = Churn)) + 
geom_boxplot()+ 
theme_bw()+
xlab(" ")
```
# TotalCharges:* The median Total charges of customers who have churned is low.
```{r}
ggplot(Customer_Churn, aes(y= TotalCharges, x = "", fill = Churn)) + 
geom_boxplot()+ 
theme_bw()+
xlab(" ")
```
#Checking the correlation between continuous variables
#Total Charges has positive correlation with MonthlyCharges and tenure.
```{r}
options(repr.plot.width =6, repr.plot.height = 4)
churn_cor <- round(cor(Customer_Churn[,c("tenure", "MonthlyCharges", "TotalCharges")]), 1)

ggcorrplot(churn_cor,  title = "Correlation")+theme(plot.title = element_text(hjust = 0.5))
```

#Subset numeric values
```{r}
Customer_Churn1 <- Customer_Churn[c(21,6,19,20)]
head(Customer_Churn1)
```

### Data Analysis with ANN
# Splitting the dataset into training and test set
```{r}
#Create training and testing set
set.seed(12345) #to guarantee repeatable results
Customer_Churn1_train <- Customer_Churn1[2:801, ]
Customer_Churn1_test <- Customer_Churn1[802:1601, ]
```

#Simple ANN with only a single hidden neuron
```{r}
Customer_Churn_model <- neuralnet(formula = Churn ~ tenure + MonthlyCharges + TotalCharges, data = Customer_Churn1_train)

#Visualize the network topology
plot(Customer_Churn_model)
```
#Compute values
```{r}
#Predict Churn 
model_results <- neuralnet::compute(Customer_Churn_model, Customer_Churn1_test[1:4])

#Obtain predicted Churn values
predicted_churn <- model_results$net.result

#Examine the correlation between predicted and actual values

cor(predicted_churn, Customer_Churn1_test$tenure)
```

#Neurons for the hidden layer
```{r}
set.seed(12345) #to guarantee repeatable results

Customer_Churn_model2 <- neuralnet(formula = Churn ~ tenure + MonthlyCharges + TotalCharges, data = Customer_Churn1_train, hidden = 3)

#Visualize the network topology
plot(Customer_Churn_model2)
```

#More Hidden Layers
```{r}
set.seed(12345) #to guarantee repeatable results

Customer_Churn_model3 <- neuralnet(formula = Churn ~ tenure + MonthlyCharges + TotalCharges, data = Customer_Churn1_train,hidden = c(4,2))  

#Visualize the network topology
plot(Customer_Churn_model3)
```
##Examine normalized data to unnormalized values(actual)
```{r}
#note that the predicted and actual values are on different scales
predictions <- data.frame ( 
  actual = Customer_Churn1_train$Churn,
  pred = Customer_Churn1_test$Churn)

head(predictions, n = 3)
```
#Unnormalize prediction
```{r}
#Unnormalize function to reverse the normalization

unnormalize <- function(x) {
  return ((x * (max(predictions$Churn))-
            min(predictions$Churn)) + min(predictions$Churn))
}
predictions$pred_new <- unnormalize(predictions$pred)

#Now we can calculate the error (it's fair)
predictions$error <- predictions$pred_new - predictions$actual

head(predictions, n=3)
##KVM create model by using kernel to transform the data
```

#SVM
```{r}
#Split samples
Cust_Churn1_train <- Customer_Churn1[2:501, ]
Cust_Churn1_validation <- Customer_Churn1[1001:1005, ]
Cust_Churn1_test <- Customer_Churn1[1501:2000, ]
```

#SVM Model
#Train a simple linear SVM
```{r}
set.seed(12345)
modelSVM <- ksvm(Churn~., data = Cust_Churn1_train, kernel = "vanilladot")

# look at basic information about the model
modelSVM
```

#Predict SVM
```{r}
pred <- predict(modelSVM, Cust_Churn1_test)

# Confusion Matrix
# Generate confusion Matrix
ConfusionMatrix1 <- data.frame (
  actual = Cust_Churn1_train$Churn,
  pred = Cust_Churn1_test$Churn)
ConfusionMatrix1
```

#Accuracy
```{r}
#Calculate Accuracy
vanilla_accuracy <- sum(diag(ConfusionMatrix1)) / sum(ConfusionMatrix1)
cat("Vanilla Kernel Accuracy:", vanilla_accuracy)
```

#RBFDOT
```{r}
set.seed(12345)

modelSVM <- ksvm(Churn ~.,
                 data = Cust_Churn1_train,
                 kernel = "rbfdot")
modelSVM
#Predict
pred <- predict(modelSVM, Cust_Churn1_test)

# Generate confusion Matrix
ConfusionMatrix2 <- data.frame ( 
  actual = Cust_Churn1_train$Churn,
  pred = Cust_Churn1_test$Churn)
ConfusionMatrix2
```

#Accuracy
```{r}
#Calculate Accuracy
accuracy1 <- sum(diag(ConfusionMatrix)) / sum(ConfusionMatrix)
cat("Gaussian RBF Kernel Accuracy:", accuracy)
```
