# Spam-Email-Detection-Using-Machine-Learning
A supervised machine learning project that classifies emails as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

📌 Project Overview

Spam emails are a persistent problem in digital communication. This project demonstrates how machine learning and NLP can be used to automatically classify email messages as spam or legitimate.

The system converts raw text into numerical features using Bag-of-Words (CountVectorizer) and applies a Naive Bayes model for classification.

🚀 Features

Text preprocessing and vectorization

Spam vs Ham classification

Model training and evaluation

Confusion matrix and classification report

High accuracy with simple and interpretable model

🧠 Technologies Used

Python 3

Pandas

Scikit-learn

Natural Language Processing (NLP)

📂 Project Structure
spam-email-classifier/
│
├── spam_ham_dataset.csv     # Dataset file
├── spam_classifier.py       # Main Python script
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── LICENSE                  # Open-source license
└── .gitignore               # Ignored files


📊 Dataset Description

The dataset contains labeled email messages with the following fields:
| Column Name | Description           |
| ----------- | --------------------- |
| `text`      | Email message content |
| `label`     | spam or ham           |
| `label_num` | 1 = spam, 0 = ham     |

