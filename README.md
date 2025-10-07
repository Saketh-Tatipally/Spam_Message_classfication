# Spam_Message_classfication
Spam Text Message Classification using Machine Learning and Deep Learning models  a Kaggle-based NLP project analyzing and predicting ham vs spam messages.

### 📄 Overview
This project builds a **text-classification model** to distinguish between *ham* and *spam* SMS messages using classical ML and a neural network.



### 📊 Dataset
- **Source:** [Spam Text Message Classification – Kaggle](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)  
- **Shape:** 5572 rows × 2 columns (`Category`, `Message`)  
- **Classes:** 4825 ham (86.6 %) | 747 spam (13.4 %)


### 🧹 Data Preparation
1. Downloaded and loaded data from Kaggle.  
2. Cleaned text – lowercased, removed punctuation/digits/stop words.  
3. Encoded labels (`ham → 0`, `spam → 1`).  
4. Split train/test using `train_test_split`.  
5. Vectorized text with `TfidfVectorizer`.


### 🤖 Models Evaluated
| Algorithm | Notes |
|------------|--------|
| Multinomial Naive Bayes | Baseline TF-IDF |
| Linear SVM | High-margin linear classifier |
| Logistic Regression | Simple baseline |
| Decision Tree | Interpretable tree |
| Random Forest | Bagged ensemble |
| Gradient Boosting | Sequential boosting |
| XGBoost | Optimized boosted trees |
| K-Nearest Neighbors | Distance-based |
| Multilayer Perceptron | Deep feed-forward NN |


### 📈 Evaluation Metrics
- **Accuracy**
- **Classification Report (Precision/Recall/F1)**
- **Confusion Matrix**



### 🏆 Results
| Model | Accuracy |
|--------|-----------|
| Multilayer Perceptron | **≈ 98 % (best)** |
| Linear SVM | 97 % |
| Random Forest | 96 % |

The **Multilayer Perceptron** achieved the highest accuracy and balanced precision-recall scores.


### 🔧 Hyperparameter Tuning Suggestions
Grid/Random search for:
- `alpha` (Naive Bayes)
- `C` (SVM, LogReg)
- Tree depth, min samples split/leaf (Tree, RF, GB)
- Learning rate (XGBoost)
- Hidden layers & activation (MLP)



### 💡 Insights & Challenges
- Spam texts tend to be longer and contain commercial keywords.  
- Balancing classes and tuning text cleaning significantly improved performance.  
- Deep models need careful regularization to avoid overfitting.


### 🚀 Future Work
- Try word embeddings (GloVe, Word2Vec, BERT).  
- Explore ensemble and deep architectures.  
- Deploy as Flask/FastAPI microservice for real-time prediction.


