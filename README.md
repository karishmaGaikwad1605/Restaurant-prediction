# Restaurant Rating Prediction Project 

 **Introduction**
With the growing popularity of online food delivery services and restaurant review platforms such as Zomato, Yelp, and Google Reviews, customer opinions play a vital role in influencing dining choices. Given the vast amount of user-generated content available, manually analyzing reviews is impractical. Thus, this project aims to automate the process by predicting restaurant ratings based on customer sentiments expressed in textual reviews.
The core idea behind this project is to use NLP techniques to process and analyze review texts, extracting patterns that correlate with ratings. Machine learning models are trained on historical review data to learn these relationships, allowing the system to predict ratings for new reviews with high accuracy. Furthermore, an interactive web application enables users to input reviews and receive predicted ratings instantly. To enhance decision-making, visualizations are created using Tableau to provide a clearer understanding of trends and customer sentiment patterns.
 Methodology
**a) Data Collection**
The dataset comprises customer reviews and corresponding restaurant ratings collected from various online platforms. Each review includes textual feedback from users and a numerical rating (e.g., on a scale of 1 to 5). This structured and unstructured data combination enables the system to learn patterns between textual sentiment and numerical ratings.
**b) Data Preprocessing**
Before training the predictive model, the collected review data undergoes preprocessing to enhance the quality of input features. The preprocessing steps include:
•	Text Cleaning: Removal of unnecessary characters, special symbols, stopwords, and punctuation.
•	Tokenization: Splitting textual data into individual words or phrases.
•	Lemmatization & Stemming: Reducing words to their base forms to ensure consistency.
•	Feature Extraction: Transforming textual reviews into numerical representations using the Term Frequency-Inverse Document Frequency (TF-IDF) method.
**c) Model Development**
To predict restaurant ratings accurately, multiple machine learning models were implemented and evaluated:
•	Logistic Regression: A simple and interpretable model for binary classification.
•	Random Forest: A robust ensemble learning method that improves prediction stability.
Each model was trained and evaluated using performance metrics such as:
•	Accuracy: Measures the percentage of correct predictions.
•	Precision & Recall: Evaluate the model’s ability to correctly classify positive and negative sentiments.
•	F1-Score: A balance between precision and recall for better overall performance assessment.
**d) Web Application**
A web-based interface was developed to enable users to interact with the prediction system. The application consists of:
•	Backend: Implemented using Flask, allowing seamless integration with the machine learning model.
•	Frontend: Developed with HTML, CSS, and JavaScript for an interactive user experience.
•	Deployment: Hosted on AWS to ensure accessibility and scalability.
**e) Data Visualization**
To facilitate better interpretation of restaurant review trends, Tableau dashboards were created. These visualizations include:
•	Sentiment Analysis Trends: Displaying the distribution of positive and negative sentiments.
•	Rating Distribution: Understanding how ratings are spread across different restaurants.
•	Keyword Analysis: Identifying frequently mentioned words and their impact on ratings.
**Results & Discussion**
The results indicate that the XGBoost model outperformed other models, achieving the highest accuracy in rating prediction. The web application allows users to input textual reviews and receive real-time predictions, making it a useful tool for restaurant owners and customers alike. The Tableau dashboard further enhances decision-making by providing insights into customer preferences and sentiment trends.

