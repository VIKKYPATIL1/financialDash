import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pyrebase
count=0



firebase_config = {
    "apiKey": "AIzaSyBl8E4cnWrLupsQrGEP_PGJfyKLCwkcqd8",
    "authDomain": "finance-f4e49.firebaseapp.com",
    "databaseURL": "https://finance-f4e49.firebaseio.com",
    "projectId": "finance-f4e49",
    "storageBucket": "finance-f4e49.appspot.com",
    "messagingSenderId": "960348831110",
    "appId": "1:960348831110:web:fde77542b44c7cd2b44007",
    "measurementId": "G-WRQPZGEE2G",
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def sign_up(name, email, password):
    if len(password) < 6:
        st.error("Password must be at least 6 characters long.")
        return None
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.success("User account created successfully!")
        return user
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return None

def login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.page = 'dashboard'  # Navigate to the dashboard after login
        st.success("Logged in successfully!")
        return user
    except Exception as e:
        st.error("Invalid email or password.")
        return None
def logout():
    st.session_state.page = "home"  # Redirect to home page
    st.session_state.userauth = False
    st.session_state.user = None
    st.success("You have been logged out!")
def dashboard():
    # Set up the Streamlit app
    st.title('Stock Market Analysis Dashboard')

    # Load the dataset (change the path as needed)
    file_path = 'prices-split-adjusted.csv'
    prices_data = pd.read_csv(file_path)
    prices_data['date'] = pd.to_datetime(prices_data['date'])

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    options = ['Heatmap of Stock Price Performance', 'P/E and P/B Ratio Analysis', 'Top 10 Profitable Companies', 'Return and Volatility', 'Gross Profit Prediction', 'Stock Price Comparison']
    selection = st.sidebar.radio("Choose a visualization:", options)

    # Visualization 1: Heatmap of Stock Price Performance
    if selection == 'Heatmap of Stock Price Performance':
        selected_year = st.slider("Select Year", 2010, 2016, 2014)
        # Filter the data for the selected year
        filtered_data = prices_data[prices_data['date'].dt.year == selected_year]
        top_50_stocks = filtered_data.groupby('symbol')['volume'].sum().nlargest(50).index
        top_50_data = filtered_data[filtered_data['symbol'].isin(top_50_stocks)]

        top_50_data['month'] = top_50_data['date'].dt.to_period('M')
        monthly_avg_close = top_50_data.groupby(['symbol', 'month'])['close'].mean().unstack(level='month')
        if not monthly_avg_close.empty and monthly_avg_close.shape[1] > 0:
            monthly_price_change = (monthly_avg_close.divide(monthly_avg_close.iloc[:, 0], axis=0) - 1) * 100
        else:
            st.error("The monthly_avg_close DataFrame is empty or does not have the expected columns.")

        st.write(f"Heatmap of Top 50 Traded Stocks' Monthly Price Performance for {selected_year}")

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(monthly_price_change, cmap='RdYlGn', annot=False, fmt=".1f", linewidths=0.5, cbar_kws={'label': '% Change from Start of Year'}, ax=ax)
        ax.set_title(f"Heatmap of Top 50 Traded Stocks' Monthly Price Performance ({selected_year})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Stock Symbol")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Visualization 2: P/E and P/B Ratio Analysis
    elif selection == 'P/E and P/B Ratio Analysis':
        # Create mock data for P/E and P/B ratios
        unique_symbols = prices_data['symbol'].unique()
        dates = pd.to_datetime(prices_data['date'].unique())
        mock_data = []
        for symbol in unique_symbols:
            for date in dates:
                mock_data.append({
                    'date': date,
                    'symbol': symbol,
                    'P/E': np.random.uniform(10, 30),
                    'P/B': np.random.uniform(1, 10)
                })
        pe_pb_df = pd.DataFrame(mock_data)
        merged_df = pd.merge(prices_data, pe_pb_df, on=['symbol', 'date'])

        stock_dropdown = st.selectbox('Select Stock Symbol', unique_symbols)
        year_dropdown = st.selectbox('Select Year', sorted(merged_df['date'].dt.year.unique()))

        stock_data = merged_df[(merged_df['symbol'] == stock_dropdown) & (merged_df['date'].dt.year == year_dropdown)]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(stock_data['date'], stock_data['P/E'], label='P/E Ratio', marker='o')
        ax.plot(stock_data['date'], stock_data['P/B'], label='P/B Ratio', marker='x')
        ax.set_title(f"P/E and P/B Ratios for {stock_dropdown} in {year_dropdown}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Visualization 3: Top 10 Profitable Companies by Gross Profit
    elif selection == 'Top 10 Profitable Companies':
        df = pd.read_csv('fundamentals.csv')
        df['For Year'] = pd.to_numeric(df['For Year'], errors='coerce')
        df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')
        df = df.dropna(subset=['For Year', 'Gross Profit'])
        df['For Year'] = df['For Year'].astype(int)

        year = st.number_input("Enter Year to Filter (e.g., 2023)", 1900, 2100, 2023)
        df_filtered = df[df['For Year'] == year]
        
        df_grouped = df_filtered.groupby('Ticker Symbol').agg({
            'Gross Profit': 'sum',
            'Period Ending': 'first'
        }).reset_index()
        top_10_stocks = df_grouped.sort_values('Gross Profit', ascending=False).head(10)

        st.write(f"Top 10 Profitable Companies in {year}")
        st.table(top_10_stocks[['Ticker Symbol', 'Gross Profit', 'Period Ending']])

        companies = top_10_stocks['Ticker Symbol']
        gross_profit = top_10_stocks['Gross Profit']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(companies, gross_profit, color='skyblue')
        ax.set_title(f'Top 10 Companies by Gross Profit for {year}', fontsize=14)
        ax.set_xlabel('Companies (Ticker Symbol)', fontsize=12)
        ax.set_ylabel('Gross Profit', fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Visualization 4: Return and Volatility of Top Companies
    elif selection == 'Return and Volatility':
        df = pd.read_csv('fundamentals.csv')
        df = df[['Ticker Symbol', 'For Year', 'Net Income']].dropna()
        df['For Year'] = df['For Year'].astype(int)
        df['Net Income'] = df['Net Income'].astype(float)

        data = df.sort_values(by=['Ticker Symbol', 'For Year'])
        data['Return'] = data.groupby('Ticker Symbol')['Net Income'].pct_change()

        top_companies = data.groupby('Ticker Symbol')['Net Income'].mean().nlargest(10).index
        top_data = data[data['Ticker Symbol'].isin(top_companies)]
        
        top_volatility = top_data.groupby('Ticker Symbol')['Return'].std().sort_values(ascending=False)
        
        year = st.selectbox('Select Year for Returns', sorted(top_data['For Year'].unique()))
        year_data = top_data[top_data['For Year'] == year]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Ticker Symbol', y='Return', data=year_data, palette="viridis", hue='Ticker Symbol', dodge=False, legend=False, ax=ax)
        ax.set_title(f'Returns of Top Companies in {year}')
        ax.set_ylabel('Returns')
        ax.set_xlabel('Ticker Symbol')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_volatility.index, y=top_volatility.values, palette="magma", hue=top_volatility.index, dodge=False, legend=False, ax=ax)
        ax.set_title('Volatility of Returns (Top Companies)')
        ax.set_ylabel('Volatility')
        ax.set_xlabel('Ticker Symbol')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Visualization 5: Gross Profit Prediction using Linear Regression
    elif selection == 'Gross Profit Prediction':
        df = pd.read_csv('fundamentals.csv')
        df['For Year'] = pd.to_numeric(df['For Year'], errors='coerce')
        df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')
        df = df.dropna(subset=['For Year', 'Gross Profit'])

        year = st.number_input("Enter Year for Data Filtering", 1900, 2100, 2023)
        future_year = st.number_input("Enter Year for Prediction", 1900, 2100, 2025)

        df_filtered = df[df['For Year'] == year]
        df_grouped = df_filtered.groupby('Ticker Symbol').agg({'Gross Profit': 'sum'}).reset_index()
        top_10_stocks = df_grouped.sort_values('Gross Profit', ascending=False).head(10)

        projections = []
        model = LinearRegression()

        for _, row in top_10_stocks.iterrows():
            ticker = row['Ticker Symbol']
            company_data = df[df['Ticker Symbol'] == ticker]
            
            if len(company_data) > 1:
                X = company_data['For Year'].values.reshape(-1, 1)
                y = company_data['Gross Profit'].values
                model.fit(X, y)
                predicted_profit = model.predict(np.array([[future_year]]))
                projections.append({
                    'Ticker Symbol': ticker,
                    'Predicted Year': future_year,
                    'Predicted Gross Profit': predicted_profit[0]
                })

        projections_df = pd.DataFrame(projections)

        if not projections_df.empty:
            st.write(f"Predictions for Gross Profit in {future_year}")
            st.table(projections_df)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(projections_df['Ticker Symbol'], projections_df['Predicted Gross Profit'], color='lightgreen')
            ax.set_title(f'Gross Profit Prediction for {future_year}')
            ax.set_xlabel('Companies')
            ax.set_ylabel('Predicted Gross Profit')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Visualization 6: Stock Price Comparison
    elif selection == 'Stock Price Comparison':
        
       
        

        # Load the stock prices dataset (assuming it is already present)
        file_path = 'prices-split-adjusted.csv'
        prices_df = pd.read_csv(file_path)

        # Convert the 'date' column to datetime
        prices_df['date'] = pd.to_datetime(prices_df['date'])

        # Define the interactive function for stock comparison
        def plot_stock_comparison(stock1, stock2, start_date, end_date):
            # Convert start_date and end_date to datetime64[ns] type
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Filter data for the given date range
            filtered_data = prices_df[(prices_df['date'] >= start_date) & (prices_df['date'] <= end_date)]
            
            # Filter data for the selected stocks
            stock1_data = filtered_data[filtered_data['symbol'] == stock1]
            stock2_data = filtered_data[filtered_data['symbol'] == stock2]
            
            # Plot the stock prices
            plt.figure(figsize=(14, 7))
            plt.plot(stock1_data['date'], stock1_data['close'], label=f"{stock1} Closing Price", marker='o')
            plt.plot(stock2_data['date'], stock2_data['close'], label=f"{stock2} Closing Price", marker='x')
            plt.title(f"Stock Price Movement Comparison: {stock1} vs {stock2}")
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        # Streamlit widgets for stock selection and date range
        st.title('Stock Price Comparison')

        # Get unique stock symbols
        unique_symbols = prices_df['symbol'].unique()

        # Get the min and max dates
        min_date = prices_df['date'].min()
        max_date = prices_df['date'].max()

        # Dropdown widgets for stock selection
        stock1 = st.selectbox('Select Stock 1', unique_symbols, index=0)
        stock2 = st.selectbox('Select Stock 2', unique_symbols, index=1)

        # Date range slider
        start_date, end_date = st.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)

        # Plot the comparison when the user selects stocks and date range
        if stock1 and stock2:
            plot_stock_comparison(stock1, stock2, start_date, end_date)

# Main login and authentication interface
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'login'  # Default page

    # Login page
    if st.session_state.page == 'login':
        st.title("Login Page")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(email, password)
        if st.button("Sign Up"):
            name = st.text_input("Full Name")
            sign_up(name, email, password)

    # Dashboard page
    elif st.session_state.page == 'dashboard':
        st.title("Dashboard")
        dashboard()
        
        if st.button("Logout"):
            logout()
        # Logout Button
if __name__ == "__main__":
    main()
