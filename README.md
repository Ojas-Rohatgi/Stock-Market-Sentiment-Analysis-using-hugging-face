# Stock-Market-Sentiment-Analysis-using-hugging-face
This project helps to visualize relation between News articles and stock market.

# Stock Market Sentiment Analysis

This repository contains code for analyzing market sentiment using stock prices and news articles. It includes three main Python files:

# Ticker Collection Script (ticker_collection.py)

This script is designed to collect ticker symbols and corresponding company names for companies listed in the S&P 500 index.

## Usage

### Dependencies

- `yfinance`: Used to fetch financial data, including stock prices, from Yahoo Finance.
- `pandas`: Utilized for data manipulation and analysis.

### Functionality

1. **`get_company_name` Function**:
   - Fetches the company name for a given ticker symbol using the Yahoo Finance API.
   - Returns the full name of the company if successful; otherwise, prints an error message and returns `None`.

2. **Fetching S&P 500 Tickers**:
   - Fetches the list of S&P 500 tickers from Wikipedia.
   - Extracts ticker symbols from the HTML table and converts them to a list, limiting it to the first 500 tickers.

3. **Creating an Empty DataFrame**:
   - Initializes an empty DataFrame to store the results, with columns for ticker symbols and company names.

4. **Collecting Tickers and Company Names**:
   - Iterates through each ticker symbol in the S&P 500 tickers list.
   - Calls the `get_company_name` function to fetch the corresponding company name.
   - Appends the ticker and company name to the DataFrame if the company name is retrieved successfully.
   - Prints an error message if the company name cannot be retrieved for a ticker.

5. **Saving Data to CSV**:
   - Saves the DataFrame containing ticker symbols and company names to a CSV file named `companies.csv`, excluding the index.

## Files

- `ticker_collection.py`: Python script containing the ticker collection functionality.
- `companies.csv`: CSV file containing the collected ticker symbols and company names.

# Companies CSV File

The `companies.csv` file contains the collected ticker symbols and corresponding company names retrieved using the ticker collection script.

## Description

This CSV file serves as a dataset that stores information about companies listed in the S&P 500 index. It includes two columns:

1. **Ticker**: Represents the unique ticker symbol associated with each company.
2. **Company Name**: Represents the full name of the company corresponding to the ticker symbol.

## Usage

1. **Data Retrieval**: The `main.py` script accesses the `companies.csv` file to retrieve the ticker symbols and company names of companies listed in the S&P 500 index.

2. **User Selection**: In the Streamlit web application generated by `main.py`, users can select one or more companies from a dropdown menu populated with the company names listed in the `companies.csv` file.

3. **Data Processing**: After the user selects companies, the `main.py` script retrieves historical stock data and sentiment analysis scores for the selected companies.

4. **Integration**: The ticker symbols obtained from the `companies.csv` file are used to fetch historical stock data from the Yahoo Finance API. Additionally, the company names are used for display purposes in the Streamlit application.

5. **Error Handling**: In case a company's ticker symbol or name is not found, appropriate error messages are displayed to the user within the Streamlit application, ensuring a smooth user experience.

## Example

Suppose a user selects the company "Apple Inc." from the dropdown menu in the Streamlit application. The `main.py` script retrieves the corresponding ticker symbol "AAPL" from the `companies.csv` file. Subsequently, it uses this ticker symbol to fetch historical stock data and sentiment analysis scores for Apple Inc.

## File Format

- **File Name**: companies.csv
- **Format**: Comma-separated values (CSV)
- **Encoding**: UTF-8

# Main Script: Market Sentiment Analysis

The `main.py` script is the primary component of the Market Sentiment Analysis project. It is responsible for generating a Streamlit web application that allows users to select companies from the S&P 500 index, retrieve their historical stock data, and analyze their sentiment scores based on news articles.

## File Structure

- **File Name**: main.py
- **Dependencies**: 
  - yfinance: For retrieving historical stock data.
  - pandas: For data manipulation and processing.
  - plotly.graph_objects: For generating interactive charts.
  - transformers: For sentiment analysis using pre-trained models.
  - tqdm: For displaying progress bars during sentiment analysis.
  - requests: For fetching news articles from the News API.
  - time: For adding delays during sentiment analysis.
  - plotly.subplots: For creating subplot grids in the charts.
  - streamlit: For building the web application user interface.


## Usage

1. **Initialization**: Upon execution, the `main.py` script initializes the sentiment analysis model and tokenizer using the Twitter-RoBERTa model.

2. **Company Selection**: Users are presented with a dropdown menu populated with company names fetched from the `companies.csv` file. They can select one or more companies they are interested in.

3. **Data Retrieval**: After selecting companies, the script retrieves historical stock data for the last month and news articles related to the selected companies.

4. **Sentiment Analysis**: The script performs sentiment analysis on the retrieved news articles using the initialized sentiment analysis model. It calculates sentiment scores for each article and aggregates them to obtain an average sentiment score for each company.

5. **Data Visualization**: Using Plotly, the script generates interactive charts that display the historical stock prices and average sentiment scores for the selected companies over the past month.

6. **User Interaction**: Users can interact with the generated charts to analyze the relationship between stock prices and sentiment scores. Additionally, error messages are displayed for any issues encountered during data retrieval or analysis.

7. **Integration**: The `main.py` script integrates various libraries and APIs, including yfinance for stock data retrieval, Transformers for sentiment analysis, Plotly for data visualization, and Streamlit for web application development.

## Example

Suppose a user selects the companies "Apple Inc." and "Microsoft Corporation" from the dropdown menu in the Streamlit application. The `main.py` script retrieves their historical stock data and sentiment analysis scores, generates interactive charts displaying the data, and presents them to the user for analysis.

