import sqlite3
import pandas as pd
import numpy as np
import random as rd

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import wordcloud

import collections
import copy
import re
import string

import nltk
from nltk import stem
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.metrics import confusion_matrix
import seaborn as sns

import time
start_time = time.time()

class AmazonBookReviewProject:
    def __init__(self, use_stemming=True):  # Change the default value to False for non-stemmed word cloud.
        self.use_stemming = use_stemming
        self.books_df = pd.DataFrame()
        self.books_data = pd.DataFrame()
        self.books_rating = pd.DataFrame()
        self.stopWords = []
        # Please enter your file path in this location below once and need not enter anywhere else.
        self.filepath =  r'C:\Users\saran\Desktop\Sara\Data'

    def run_project_work(self):
        self.loadData()
        self.books_df = self.mergeData(self.books_data, self.books_rating)
        self.books_df = self.cleanData(self.books_df)
        self.analyseRating()    # Generate histogram
        self.stopWords = self.getStopWords()
        self.analyzeSentiments(self.use_stemming)  # To generate both options: Stemmed and non-stemmed.
        self.runTextPredictions()

    def loadData(self):
        print("\n*******   Beginning to load Data   *******\n")
        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)

        """ Load data into books_data. """
        self.books_data = pd.read_csv(self.filepath + r"\books_data.csv")

        # """ Load data into books_rating. """
        # Establishing connection to SQLite database
        conn = sqlite3.connect('books_rating.db')

        query = "SELECT * FROM books_rating_tbl"
        # Using pandas to read SQL query results to data frame
        self.books_rating = pd.read_sql_query(query, conn)

        conn.close()

        """ To understand the structure print the head of the DataFrame. """
        # print(self.books_data.head(5))
        # print(self.books_rating.head(5))

        print("\n*******   Load Data Completed   *******\n")


    def mergeData(self, books_data, books_rating):
        print("\n*******   Beginning to Merge Data   *******\n")
        # Merging both dataframes into one dataframe
        self.books_df = self.books_rating.merge(self.books_data, how='outer', on='Title')
        self.books_df = self.books_df[
            ['Id', 'Title', 'profileName', 'review/score', 'review/summary',
             'review/text', 'description', 'authors', 'publisher', 'publishedDate', 'categories', 'ratingsCount']]
        print("\n*******   Merge Data Completed  *******\n")
        return self.books_df


    def cleanData(self, books_df):
        print("\n*******   Beginning to Clean and Sanitize Data   *******\n")
        # print(books_df.columns.tolist())

        # Rename columns as per naming conventions
        books_df.columns = ['id', 'title', 'profileName', 'review_score',  'review_summary',
                            'review_text', 'description', 'authors', 'publisher', 'publishedDate', 'categories', 'ratingsCount']

        """ TREAT MISSING VALUES"""
        print("\n*******   Treating Missing Values   *******\n")
        print("\nNUMBER OF ROWS BEFORE HANDLING MISSING VALUES:\n", len(books_df))

        # Print missing values summary for column in 'books_df'
        print("\nBOOKS_DF MISSING VALUES SUMMARY : BEFORE HANDLING MISSING VALUES\n", books_df.isnull().sum())

        # Treat missing values in the observations. In this case, dropna has been chosen as the strategy.
        books_df.dropna(subset=['profileName', 'review_summary', 'review_text', 'description', 'authors', 'publisher', 'publishedDate',
                                'categories', 'ratingsCount'], inplace=True)

        print("\nNUMBER OF ROWS AFTER HANDLING MISSING VALUES\n", len(books_df))

        # print(books_df.head(5))
        print("\nBOOKS_DF MISSING VALUES SUMMARY AFTER HANDLING MISSING VALUES \n", books_df.isnull().sum())

        print("\nRemoving Special Characters in 'authors' and 'categories' columns\n")
        # Remove all occurrences of [ " ' enclosing ' " ] in authors and categories
        books_df.loc[:, 'authors'] = books_df['authors'].astype(str).str.replace(r"[\[\]\'\"]+", "", regex=True)
        books_df.loc[:, 'categories'] = books_df['categories'].astype(str).str.replace(r"[\[\]\'\"]+", "", regex=True)

        print("\nAltering 'published_date' to datetime and cleaning incomplete observations\n")
        def standardize_date(date_str):
            # Check if the date string is NaN
            if pd.isna(date_str):
                return np.nan
            # Check if the date string is just a year
            elif len(date_str) == 4:
                return f"01/01/{date_str}"
            # Check if the date string is in the format "YYYY-MM"
            elif len(date_str) == 7 and date_str.count('-') == 1:
                return f"{date_str.replace('-', '/')}/01"
            # If the date string is already in the format "MM/DD/YYYY", return it as is
            elif len(date_str) == 10 and date_str.count('/') == 2:
                return date_str
            # If none of the above conditions are met, return the original string
            else:
                return date_str

        # Apply the standardize_date function to the 'publishedDate' column
        books_df.loc[:, 'publishedDate'] = books_df['publishedDate'].apply(standardize_date)

        # Now convert the standardized date strings into datetime objects
        books_df['publishedDate'] = pd.to_datetime(books_df['publishedDate'], format='%m/%d/%Y', errors='coerce')

        # Format the datetime objects to a string in the format MM/DD/YYYY
        books_df.loc[:, 'publishedDate'] = books_df['publishedDate'].dt.strftime('%m/%d/%Y')
        books_df.loc[:, 'publishedDate'] = books_df[books_df['publishedDate'].dt.year >= 1900]
        books_df.dropna(subset=['publishedDate'], inplace=True)

        books_df['year'] = books_df['publishedDate'].dt.year.astype(int)

        """ HANDLE DUPLICATE ROWS"""
        print("\n*******   Handling Duplicate Observations   *******\n")
        # Check for duplicates in entire dataframe
        print("\nBOOKS_DF DUPLICATES PRESENT OR NOT\n", books_df.duplicated().any())
        print("\nLength of books_df before removing duplicates: \n", len(books_df))

        # To remove duplicates considering all columns
        books_df.drop_duplicates(keep='first', inplace=True)
        print("\nLength of books_df after removing duplicates: \n", len(books_df))

        # books_df.to_csv('Clean.csv', index=False)
        print("\n*******   End of Cleaning and Sanitizing Data   *******\n")
        return books_df


    def analyseRating(self):
        print("\n*******   Generating Histogram for Periodic Evolution of Review Ratings: A Comparative Analysis from  1900 to  2023  *******\n")
        # Define the year ranges
        year_ranges = [(1900, 1940), (1941, 1980), (1981, 2000), (2000, 2023)]

        # Create a subplot for each year range
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Years {start}-{end}" for start, end in year_ranges])

        # Define a list of colors for the histograms
        colors = ['#515c72', '#6e8666', '#574a42', 'saddlebrown']

        # Creating a copy for filtering data as per years.
        df_filtered = copy.deepcopy(self.books_df)
        # df_filtered['review_text'] = df_filtered['review_text'].apply(self.remove_punc_stopwords) # removing stop words and puncutations

        # Plot a histogram for each year range
        for i, (start, end) in enumerate(year_ranges):
            # Filter the DataFrame for the current year range
            df_filtered_year = df_filtered[(df_filtered['year'] >= start) & (df_filtered['year'] <= end)]
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # WordClouds for different ranges of years
            corpus_txt = " ".join(x for x in df_filtered_year['review_text'])
            # print(corpus_txt)
            frequencyDict_wc = self.calculate_word_frequencies(corpus_txt, False, True)
            fileName_wc = self.filepath + r"\Histogram_WordCloud" + str(i) + ".png"

            # print(fileName)
            self.generateWordCloud(frequencyDict_wc, fileName_wc)

            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

            # Add the histogram to the subplot
            hist = px.histogram(df_filtered, x="review_score", nbins=10, color_discrete_sequence=[colors[i]])
            hist.update_traces(marker_line_color='rgba(0,  0,  0,  0.2)', marker_line_width=1)
            fig.add_trace(hist.data[0], row=(i // 2) + 1, col=(i % 2) + 1)

        shapes = []
        # Define the coordinates of the corners of each subplot These coordinates are in figure fraction (from 0 to 1) and include a small margin
        subplot_coords = [(-0.05, 0.48, 1.05, 0.52),  # top left
                          (-0.05, 0.48, 0.42, -0.08),  # bottom left
                          (0.51, 1.03, 1.05, 0.52),   # top-right
                          (0.51, 1.03, 0.42, -0.08)]    # bottom-right

        # Add a shape for each subplot using the defined coordinates
        for coords in subplot_coords:
            shapes.append(go.layout.Shape(type="rect", xref="paper", yref="paper",
                                          x0=coords[0], y0=coords[2], x1=coords[1], y1=coords[3],
                                          line=dict(color="black", width=2)))

        # Update layout with all shapes for subplot borders
        fig.update_layout( plot_bgcolor="#fffff3", paper_bgcolor="#fffff3",
            title={'text': "<b>Periodic Evolution of Review Ratings: A Comparative Analysis from  1900 to  2023</b>",
                'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 18, 'color': 'black', 'family': 'Courier New, monospace'}},
            width=1000, height=800, margin=dict(l=50, r=50, t=130, b=50), shapes=shapes)

        # Set the x-axis label for all subplots
        for i in range(1, 5):
            fig.update_xaxes(title_text="Review Score", row=i, col=1)
            fig.update_xaxes(title_text="Review Score", row=i, col=2)

        # Set the y-axis label for all subplots
        for i in range(1, 5):
            fig.update_yaxes(title_text="Count", row=i, col=1)
            fig.update_yaxes(title_text="Count", row=i, col=2)

        # Show the plot
        fig.show()


    def getStopWords(self):
        nltk.download('stopwords')
        sWords = set(stopwords.words('english'))
        sWords.update({'the', 'would', 'could', 'should', 'i', 'we', 'she', 'he', 'it'})
        # print("###### STOP WORDS ######")
        # print(sWords, "\n###############\n")
        return sWords

    def analyzeSentiments(self, use_stemming=True):
        # Use the next line to get the non-stemmed word cloud:
        # def analyzeSentiments(self, use_stemming=False):
        self.books_df['sentiment'] = ['positive' if rating > 3 else
                                      'negative' if rating < 3
                                      else 'neutral' for rating in self.books_df['review_score']]
        self.generate_special_word_clouds(sentimentType='all', use_stemming=use_stemming)
        self.generate_special_word_clouds(sentimentType='positive', use_stemming=use_stemming)
        self.generate_special_word_clouds(sentimentType='negative', use_stemming=use_stemming)

    # Use this line to get the Stemmed word cloud:
    def generate_special_word_clouds(self, sentimentType='all', use_stemming=True):

        # Use this line to get the non-stemmed word cloud:
        # def generate_special_word_clouds(self, sentimentType='all', use_stemming=False):
        corpus_txt = ""
        if (sentimentType.lower() == 'all'):
            corpus_txt = " ".join(x for x in self.books_df['review_text'])
        else:
            reviews_all_sub = self.books_df[self.books_df['sentiment'] == sentimentType]
            corpus_txt = " ".join(x for x in reviews_all_sub['review_text'])
        frequencyDict = self.calculate_word_frequencies(corpus_txt, False, True)
        fileName = ""
        match sentimentType.lower():
            case 'all':
                fileName = self.filepath + r"\AllReviews.png"
                print(fileName)
            case 'positive':
                fileName = self.filepath + r"\PosReviews.png"
                print(fileName)
            case 'negative':
                fileName = self.filepath + r"\NegReviews.png"
                print(fileName)
            case _:
                fileName = self.filepath + r"\OtherReviews.png"
                print(fileName)
        self.generateWordCloud(frequencyDict, fileName)


    # def generateWordCloud(self, frequencies, path):
    #     # use this code to generate the word cloud
    #     cloud = wordcloud.WordCloud(width=800, height=400)
    #     cloud.generate_from_frequencies(frequencies)
    #     cloud.to_file(path)
    #     print("FILE GENERATED:", path)
    #
    #     # plt.interactive(True)
    #     # plt.imshow(cloud, interpolation='bilinear')
    #     # plt.axis('off')
    #     # plt.title('Word Cloud: ' + path, fontweight="bold")
    #     # plt.show()

    def generateWordCloud(self, frequencies, path):
        # Create a WordCloud object with a larger canvas size and a smaller maximum font size
        cloud = wordcloud.WordCloud(width=800, height=400, max_font_size=110)

        # Generate the word cloud from the frequencies dictionary
        cloud.generate_from_frequencies(frequencies)

        # Save the word cloud to a file
        cloud.to_file(path)
        print("FILE GENERATED:", path)

        # Display the word cloud
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: ' + path, fontweight="bold")
        plt.show()

    def calculate_word_frequencies(self, text, stemmed=False, withoutStopWords=True):
        tokens = []
        if stemmed:
            tokens = self.tokenize_withstemming(text)
        else:
            tokens = self.tokenize(text)
        final_tokens = []
        if withoutStopWords:
            final_tokens = self.remove_stopwords(tokens)
        else:
            final_tokens = tokens
        tkn_count_dict = self.get_token_counts(final_tokens)
        return tkn_count_dict


    def tokenize_withstemming(self, doc):
        tokens = self.tokenize(doc)
        # Stemming
        porterstem = stem.PorterStemmer()
        stemTokens = [porterstem.stem(x) for x in tokens]
        return stemTokens


    def tokenize(self, doc):
        text = doc.lower().strip()
        text = re.sub(f'[{string.punctuation}]', " ", text)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


    def remove_stopwords(self, token_list):
        tokensFiltered = [token for token in token_list if token not in self.stopWords]
        return tokensFiltered


    def get_token_counts(self, token_list):
        token_counter = collections.Counter([txt.lower() for txt in token_list])
        return dict(sorted(token_counter.items(), key=lambda item: item[1], reverse=True))


    def runTextPredictions(self):
        print("\n**********  PERFORMING BINOMIAL REGRESSION  *************\n")

        """ Remove punctuations and stopwords from the text data in ReviewText and Title"""
        self.books_df['title'] = [str(x) for x in self.books_df['title']]
        self.books_df['review_text'] = [str(x) for x in self.books_df['review_text']]

        books_reg = copy.deepcopy(self.books_df)

        # The following applies remove_punc_stopwords function to each value in the given column. The result is a column with lower case values
        # that have no punctuations, no stop words,
        books_reg['title'] = books_reg['title'].apply(self.remove_punc_stopwords)
        books_reg['review_text'] = books_reg['review_text'].apply(self.remove_punc_stopwords)

        """ Add a new variable called sentiment; if Rating is greater than 3, then sentiment = 1, else sentiment = -1 """
        books_reg['sentiment_value'] = [1 if x >= 3 else -1 for x in books_reg.review_score]

        """ split the dataset into two: train (85% of the obs.) and test (15% of the obs.)"""

        books_reg['random_index'] = [rd.uniform(0, 1) for x in range(len(books_reg))]
        # print(books_all.head(5))

        reviews_sub_train = books_reg[books_reg.random_index < 0.85][
            ['review_text', 'review_score', 'title', 'sentiment_value']]
        reviews_sub_test = books_reg[books_reg.random_index >= 0.85][
            ['review_text', 'review_score', 'title', 'sentiment_value']]

        # vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        train_matrix = vectorizer.fit_transform(reviews_sub_train['title'])
        test_matrix = vectorizer.transform(reviews_sub_test['title'])

        # print("\n****************************************\n")
        # print("\ntrain_matrix\n", train_matrix)
        # #
        # print("\n****************************************\n")
        # print("\ntest_matrix\n", test_matrix)

        # Perform Logistic Regression"""
        lr = LogisticRegression(max_iter=1000, solver='sag')

        X_train = train_matrix
        # print(X_train)
        X_test = test_matrix
        y_train = reviews_sub_train['sentiment_value']
        y_test = reviews_sub_test['sentiment_value']

        lr.fit(X_train, y_train)
        print("\nCoefficients1: ")
        print(lr.coef_)
        print("\nIntercept1: ")
        print(lr.intercept_)

        # Generate the predictions for the test dataset"""
        predictions = lr.predict(X_test)
        reviews_sub_test['predictions'] = predictions
        # print(reviews_sub_test.head(30))

        # Calculate the prediction accuracy"""
        reviews_sub_test['match'] = reviews_sub_test['sentiment_value'] == reviews_sub_test['predictions']

        print("\nPrediction Accuracy1: ")
        print(sum(reviews_sub_test['match']) / len(reviews_sub_test))

        # Generate confusion matrix and display"""
        actual = y_test
        predicted = predictions
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        # plt.show()

        # Generate classification report"""
        report = classification_report(y_test, predictions)

        print("\nClassification Report:\n")
        print(report)

        # Modifying the model accuracy considering the sample of 1 million from each instances negative and positive.
        # Remove punctuations and stopwords from the text data in ReviewText and Title

        """ 
        self.books_df['title'] = [str(x) for x in self.books_df['title']]
        self.books_df['review_text'] = [str(x) for x in self.books_df['review_text']]

        books_reg = copy.deepcopy(self.books_df)

        # The following applies remove_punc_stopwords function to each value in the given column. The result is a column with lower case values
        # that have no punctuations, no stop words,
        books_reg['title'] = books_reg['title'].apply(self.remove_punc_stopwords)
        books_reg['review_text'] = books_reg['review_text'].apply(self.remove_punc_stopwords)

        # Add a new variable called sentiment; if Rating is greater than 3, then sentiment = 1, else sentiment = -1
        books_reg['sentiment_value'] = [1 if x >= 3 else -1 for x in books_reg.review_score]"""

        # Count the number of positive and negative instances
        num_positive_instances = sum(books_reg['sentiment_value'] == 1)
        num_negative_instances = sum(books_reg['sentiment_value'] == -1)

        # Specify the sample size for positive and negative instances
        sample_size = min(1000000,
                          num_positive_instances)  # Limiting to 1 million or the number of positive instances, whichever is smaller
        positive_data = books_reg[books_reg['sentiment_value'] == 1].sample(n=sample_size, random_state=42)

        sample_size = min(1000000,
                          num_negative_instances)  # Limiting to 1 million or the number of negative instances, whichever is smaller
        negative_data = books_reg[books_reg['sentiment_value'] == -1].sample(n=sample_size, random_state=42)

        reviews_sub_train = pd.concat([positive_data, negative_data])

        # Resetting index
        reviews_sub_train.reset_index(drop=True, inplace=True)

        # Vectorization
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        train_matrix = vectorizer.fit_transform(reviews_sub_train['title'])

        # Splitting the data into training and testing sets (85% train, 15% test)
        X_train, X_test, y_train, y_test = train_test_split(train_matrix, reviews_sub_train['sentiment_value'],
                                                            test_size=0.15, random_state=42)

        # Performing Logistic Regression on the training data
        lr = LogisticRegression(max_iter=1000, solver='sag')
        lr.fit(X_train, y_train)

        # Predicting on the testing set
        predictions = lr.predict(X_test)

        # Calculate prediction accuracy
        accuracy = accuracy_score(y_test, predictions)
        print("MODEL ACCURACY: [BINOMIAL REGRESSION]", accuracy)

        print("\n**********  PERFORMING MULTINOMIAL REGRESSION  *************\n")

        # Preprocessing and feature extraction for the final books dataframe
        X = self.books_df['review_text']  # Text-based features
        y = self.books_df['review_score']  # Target variable

        # Split the dataset into training and testing sets with 15% as test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

        # TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Train multinomial logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)

        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print("MODEL ACCURACY: [MULTINOMIAL REGRESSION]", accuracy)

        # Assuming you have already split your data into X_test and y_test and you have predictions in y_pred
        print('\nEvaluating Model\n')
        accuracy = accuracy_score(y_test, y_pred)

        print("\nAccuracy:", accuracy)
        target_names = ['1', '2', '3', '4', '5']

        print('Classification report:')
        print(classification_report(y_test, y_pred, target_names=target_names))
        # Confusion Matrix
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:\n")
        print(conf_matrix)

        # Assuming you have already split your data into X_test and y_test and you have predictions in y_pred
        print('\nConfusion Matrix Plot\n')
        cm = metrics.confusion_matrix(y_test, y_pred)

        # Create a DataFrame from the confusion matrix
        classes = range(1, 6)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        print('\nHeat map generated\n')
        sns.heatmap(cm_df, annot=True, cmap='crest', fmt='g')
        plt.title('Confusion Matrix - Multinomial Regression', y=1.1)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        end_time = time.time()
        execution_time = end_time - start_time
        minutes, seconds = divmod(execution_time, 60)
        print(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
        plt.show()
        print("End of Code")

    def remove_punc_stopwords(self, text):
        text_tokens = self.tokenize(text)
        text_tokens  = self.remove_stopwords((text_tokens))
        return "".join(text_tokens)

# arp = AmazonBookReviewProject()

# Use this line for stemming word clouds:
arp = AmazonBookReviewProject(use_stemming=True)

#Use this line for non-stemming word clouds:
# arp = AmazonBookReviewProject(use_stemming=False)

arp.run_project_work()
