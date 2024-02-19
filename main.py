import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline, AutoModelForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import requests
import time
from plotly.subplots import make_subplots
import streamlit as st
import warnings

# Filter out Future Warning
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Load sentiment analysis model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Read the CSV file into a DataFrame
companies_df = pd.read_csv('companies.csv')

# Display a dropdown menu for selecting companies
selected_companies = st.multiselect("Select one or more companies:", companies_df['Company Name'].tolist(), placeholder="Select here")
enter_button = st.button("Enter")

if selected_companies and enter_button:
    # Define a dictionary to store historical stock data
    hists = {}

    # Loop through selected companies
    for user_input_company in selected_companies:
        # Find the corresponding ticker for the entered company name
        selected_tickers = companies_df.loc[
            companies_df['Company Name'].str.contains(user_input_company, case=False), 'Ticker'].values

        if not any(selected_tickers):
            st.write(f"No matching company found for '{user_input_company}'. Please check the company name.")
        else:
            # Convert selected_tickers to strings (it may be an array if there are multiple matches)
            selected_tickers = [str(ticker) for ticker in selected_tickers]

            # Fetch latest stock prices for selected companies for 1 month
            for ticker in selected_tickers:
                tkr = yf.Ticker(ticker)
                latest_data = tkr.history(period="1mo")  # Fetch data for the last month
                hists[ticker] = latest_data

            sentiment_scores_dict = {}  # Dictionary to store sentiment scores for each stock
            newsapi_key=""    # Register for News API key on https://newsapi.org/

            # Fetch latest news articles about each stock and perform sentiment analysis
            for ticker in selected_tickers:
                # Retrieve news articles related to the selected company
                url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey=newsapi_key'
                response = requests.get(url)

                if response.status_code == 200:
                    articles = response.json()['articles']
                    sentiment_scores = []

                    with tqdm(total=len(articles), desc=f"Analyzing Sentiment for {user_input_company}") as pbar:
                        for article in articles:
                            # Perform sentiment analysis using the Transformers pipeline
                            sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                            sentiment_score = sentiment(article['content'])[0]['score']
                            sentiment_scores.append(sentiment_score)
                            time.sleep(0.1)  # Simulating some processing time
                            pbar.update(1)

                    # Store sentiment scores for the current stock
                    sentiment_scores_dict[ticker] = sentiment_scores

                else:
                    st.write(f"Error retrieving news articles for {user_input_company}. Status code: {response.status_code}")

            # Combine sentiment scores with historical stock price data
            for ticker, hist in hists.items():
                if ticker in sentiment_scores_dict:
                    sentiment_scores = sentiment_scores_dict[ticker]
                    if len(sentiment_scores) > len(hist):
                        # st.write(f"Truncating sentiment scores for {user_input_company} to match stock data length")
                        sentiment_scores = sentiment_scores[:len(hist)]  # Truncate sentiment scores
                    elif len(sentiment_scores) < len(hist):
                        st.write(
                            f"Skipping {user_input_company}: Mismatch between sentiment scores ({len(sentiment_scores)}) and stock data ({len(hist)})")
                        continue
                    # Create DataFrame for sentiment scores
                    avg_sentiment_df = pd.DataFrame(sentiment_scores, columns=["Sentiment Score"])
                    avg_sentiment_df.index = hist.index  # Match index with stock price data
                    # Merge sentiment scores with stock price data
                    combined_df = pd.concat([hist, avg_sentiment_df], axis=1)
                    # Create subplot grid with secondary y-axis
                    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
                    # Add traces for stock price and sentiment score
                    fig_combined.add_trace(go.Scatter(x=combined_df.index, y=combined_df["Close"], name="Stock Price"),
                                           secondary_y=False)
                    fig_combined.add_trace(
                        go.Scatter(x=combined_df.index, y=combined_df["Sentiment Score"], name="Sentiment Score"),
                        secondary_y=True)
                    # Update layout
                    fig_combined.update_yaxes(title_text="<b>Stock Price</b>", secondary_y=False)
                    fig_combined.update_yaxes(title_text="<b>Sentiment Score</b>", secondary_y=True)
                    fig_combined.update_layout(title=f"Stock Price and Sentiment Score for {user_input_company}",
                                               xaxis_title="Date",
                                               yaxis_title="Stock Price",
                                               legend_title="Legend")
                    # Show figure
                    st.plotly_chart(fig_combined)

            # Calculate average sentiment score
            avg_sentiment_scores = {}
            for ticker, sentiment_scores in sentiment_scores_dict.items():
                avg_sentiment_scores[ticker] = sum(sentiment_scores) / len(
                    sentiment_scores) if sentiment_scores else 0

            # Display average sentiment score for each company
            for ticker, avg_sentiment_score in avg_sentiment_scores.items():
                st.write(f"**Average sentiment score for {user_input_company}** : {avg_sentiment_score}")
elif enter_button:
    st.write("**Please select at least one company.**")
