# GA Project 3: Classifying Reddit Posts by NLP
## Goh Wen Xuan

## Problem Statement

We are developing a new wellness app that prompts the User to write a short journal entry, which will be analysed to determined the User's philosophical inclination and return a relevant message or thought of the day. Our app focuses on two philosophical beliefs - Stoicism and Buddhism.

To understand the topics from each philosophical groups, data is extracted from two subreddits - r/Stoicism and r/Buddhism. We will use the data to attempt to train a classifier model to predict the User's philosophical inclination based on the journal entry.

We believe we can adapt the model in the future into other subreddits.

## Executive Summary

Reddit is a social news platform where Users are able to post contents, images and links which other Users can participate in the discussion. Posts are organized into User-created boards called 'subreddits' where each subreddit caters to a specific topics or subject. We are focusing on two subreddits - Stoicism and Buddhism which have 461k and 621k members respectively.

Firstly, data extraction is performed using [pushshift.io](https://github.com/pushshift/api) API. Approximately 10,000 posts were extracted each subreddit.

After narrowing down the required columns, initial data cleaning is performed and EDA revealed several occuring keywords each subreddit has. This includes the discovery of a popular book on stocisim by Ryan Holiday which was frequently brought up in the stoicism subreddit. The text is processed using various NLP methods before feeding into Pycaret to determine the best model. LightGBM is determined to be the best model achieving the highest accuracy compared to other classifier models. 

## Data Dictionary
| Features              | Type            | Description                                           |
|-----------------------|-----------------|-------------------------------------------------------|
| subreddit             | int64           | The forum within Reddit dedicated to a specific topic |
| created_date          | datetime object | Date and time of the post created                     |
| content               | object          | The title and selftext column merged                  |
| tokenized_content     | object          | Content after tokenization                            |
| lem_tokenized_content | object          | Content after tokenization and lemmatized             |

## Methodology

**Data Extraction using pushshift.io API**  
It allows reddit comments and submissions to be extracted with various optional parameters. However, it is limited to 100 posts per request and for-loops is written to run the request multiple times. A total of 19,979 posts were extracted from both subreddits.

**Initial Data Cleaning and EDA**  
The raw data collected includes several columns such as domain of the post, awards received by the post (if any), number of comments and many more. Only several columns were selected as described in the data dictionary. Duplicated posts with the same author and title are dropped. Though many missing values are found in the body of the post, looking at the rows revealed there are blank posts and the row should not be dropped as some posts contain the title only. Several undesired values such as [removed] or [deleted] are discovered in the selftext column which indicates the text was either removed or deleted and the values were replaced with a blank and not removed because the title could still be useful. 

EDA was performed to gain insights into popular keywords. It is no surprise that obvious keywords such as 'stoicism', 'buddhism', 'buddhist' appears as the top few occuring words. In fact, by looking at other keywords, we discovered that Ryan Holiday is an author of a popular book 'The Daily Stoic' and we may be able to tap into the book for additional insights or even return quotes from his book (with permission) in our app. The author and dates were examined for any patterns but none were found. 

**NLP Preprocessing**  
Since we are trying to train a model to identify between two possible responses - r/Stocisim and r/Buddhism, it is a binary classification and we will map the responses into 1 (r/Stoicism) and 0 (r/Buddhism). The feature matrix will be the subreddit text. To train a model, the text needs to be stripped from any special characters or strings (such as 'www.', 'https://', '.com'), symbols, emojis, numbers and non-english text. We attempt to remove them using regex functions. 

Next, the text has to be tokenized to help to model understand the context of the word. Word tokenization is the process of splitting large text into smaller chunks called 'tokens' and this is achieved by nltk library. Next, we attempt to reduce the number of words by lemmatizing the tokenized text. Lemmatizing refers to the process of reducing the word to its root form so that they can be analysed as a single word. 

At this stage, the text has undergone tokenizating and lemmatizing. However the text remains as a string and for it to be processed by models, we will use vectorization. Vectorization converts each unique word into a column and count the number of times the word appears. This is achieved by nltk library CountVectorizer. Tfidfvectorizer is an alternative algorithm to transform text. Instead of counting the number of times each word appears like CountVectorizer, Tfidfvectorizer considers the overall texts and gives a weight to each words depending on the frequency it appears. CountVectorizer gave a decent score at the end though Tfidfvectorizer could be explored coupled with additional feature engineering.

After tokenizing, lemmatizing and transforming into a matrix form using CountVectorizer, the final shape of the dataset has 1961 rows and 2 columns - text and subreddit. It is now ready for modelling.

**Modelling**
To recap, this is a binary classification problem as we are predicting the posts which it belongs to - r/Stoicism or r/Buddhism. Pycaret is used in this process. It is a library that automates machine learning workflow and runs various models under the hood and summaries the result. The Pycaret's classification module is instantiate in this process. 

We chose Accuracy as the model evaluation metric we we want to ensure the model is able to classify between the two subreddit. Pycaret reports Light Gradient Boosting Machine (lightgbm) as the model with the best accuracy (0.9246), and it also performs the best in terms of F1, Kappa and MCC score. Logistic regression is the next best model with an accuracy of 0.9211, a 0.38% only decreased compared to lightgbm. 

Pycaret also allows further hyperparameter tuning by Random Grid Search though it does not provide any sigificant improvement to the accuracy. The accuracy on the test data achieved 0.9229 compared to 0.9214 on training test. It is not a sigificiant difference which means there is no overfitting. 

## Conclusions and Recommendations
We have extracted approximately 10,000 posts from r/Stoicism and /rBuddhism to analyse and determine the keywords from each subreddit. The text were cleaned and preprocessed using NLP methods before selecting the modelling using Pycaret. The preprocessed data provided some insights in certain keywords in each subreddit - such as Ryan and Abdomen in r/Stoicism and r/Buddhism respectively. In fact, we discover Ryan Holiday is an author of a popular book on Stoicism, and we may consider using his book as additional information sources or even quoting his book as the message of the day.

Next, the classification model, lightgbm was selected as the final model, having the best accuracy score compared to other classification models. The model selected achieved an accuracy of 92% in identifying between stocisim and buddhism. We are able to use the model to predict the User's philosophical incline based on the journal entry written in the wellness app we are developing and a relevant thought of the day or message will be returned.

We believe that the model can be adjusted and updated in our wellness app to expand into other philosophical believes.
