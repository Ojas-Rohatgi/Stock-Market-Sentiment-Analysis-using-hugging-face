import yfinance as yf
import pandas as pd


def get_company_name(ticker):
    """Fetches the company name for a given ticker symbol."""
    try:
        company = yf.Ticker(ticker)
        company_name = company.info['longName']  # Use 'longName' for full name
        return company_name
    except Exception as e:
        print(f"Error fetching company name for {ticker}: {e}")
        return None


# Fetch S&P 500 tickers
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_data = pd.read_html(url)
sp500_tickers = sp500_data[0]['Symbol'][:500].tolist()

# Create an empty DataFrame to store results
data = {'Ticker': [], 'Company Name': []}
df = pd.DataFrame(data)

# Collect tickers and company names
for ticker in sp500_tickers:
    company_name = get_company_name(ticker)

    if company_name:
        # Append data to DataFrame
        df = pd.concat([df, pd.DataFrame({'Ticker': [ticker], 'Company Name': [company_name]})], ignore_index=True)
    else:
        print(f"Could not find company name for {ticker}")

# Save the DataFrame to a CSV file
df.to_csv('companies.csv', index=False)
