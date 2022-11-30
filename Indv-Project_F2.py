# -*- coding: utf-8 -*-
"""
Sources: 
    
[1] https://github.com/didizhx/Streamlit-Financial-Dashboard/blob/main/Streamlit%20Application%20Financial%20Programming%20(Yahoo).py

[2] https://github.com/luigibr1/Streamlit-StockSearchWebApp

[3] https://github.com/dataman-git/codes_for_articles/blob/master/stock.py
    
[4] https://github.com/ranaroussi/yfinance
    
[5] https://towardsdatascience.com/creating-a-financial-dashboard-using-python-and-streamlit-cccf6c026676
    
[6] https://python.plainenglish.io/creating-an-awesome-web-app-with-python-and-streamlit-728fe100cf7
    
[7] https://python.plainenglish.io/building-a-stock-market-app-with-python-streamlit-in-20-minutes-2765467870ee
    
[8] https://github.com/hackingthemarkets/streamlit-dashboards
    
[9] https://www.youtube.com/watch?v=0ESc1bh3eIg
    
[10] https://medium.datadriveninvestor.com/created-a-dashboard-on-streamlit-using-python-for-stock-comparison-and-fundamental-analysis-286219f6e4da
    

"""

#-----------------------------------------------------------------------------#
#                               Initiating                                    #
#-----------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas_datareader.data as web
import bs4 as bs
import requests


#-----------------------------------------------------------------------------#
#                             Tab 1 - Summary                                 #
#-----------------------------------------------------------------------------#

# periods to select for the summary chart
period = {'DurationText':['1M','3M','6M','YTD','1Y','2Y','5Y','MAX'], 'DurationN':[30,90,120,335,365,730,1825,18250]}
periods = pd.DataFrame(period)


# start and end date of the plot data 
today = datetime.today().date()


def PageStart():
    
    #display chosen ticker name
    st.title(ticker)

    return

def tab1():

    PageStart()
    
    #buttons to select desired period of data as defined in the periods dataframe
    buttons_tab1 = st.selectbox("select period",periods['DurationText'])
  
    

    #plots for each selected duration: 
    def PlotSummary(buttons_tab1): 
        
        #start date that varies with selected period
        x = today - timedelta(periods.loc[periods['DurationText'] == buttons_tab1,'DurationN'].iloc[0].item()) 
        
        #data for the plots
        closing_price = yf.Ticker(ticker).history(period = '1d', start = x, end = today) 
               
        fig,ax = plt.subplots(figsize = (12,7))
             
        ax.plot(closing_price['Close'], label='Closing Price',color='blue')
        
        #fill the color under the line
        plt.fill_between(closing_price.index, closing_price['Close'], color='lightgreen') 
        
        #twinning bar chart plot
        ax2 = ax.twinx() 
        
        #plotting bar chart
        ax2.bar(closing_price.index,closing_price['Volume'], label='Volume (in millions)',color='red') 
        
        # diminishing the scale of the bar char 
        ax2.set_ylim(0,((closing_price['Volume'].max())*7)) 
        
        #hiding y ticks of bar chart from the plot
        ax2.set_yticks([])
        
        #moving ticks to the right side
        ax.yaxis.tick_right() 
        
        #store ticks in np array
        my_xticks = ax.get_xticks()
        
        # only show 1st and median ticks in the plot
        ax.set_xticks([my_xticks[0], np.median(my_xticks),my_xticks[-1]])
        
        #Legend labels for both axes shown together in one legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax.set_frame_on(False) 
        ax2.set_frame_on(False)

        st.pyplot(fig)
    
    #call the function
    PlotSummary(buttons_tab1 = buttons_tab1) 

    #defining columns and tables to show summary data in two columns
    col1, col2 = st.columns(2)
    QuoteTable = yf.Ticker(ticker).info
    
    
    QuoteTable = pd.DataFrame(QuoteTable.items(), columns=['Attribute', 'Value'])
    
    st.dataframe(QuoteTable)
    

  

#-----------------------------------------------------------------------------#
#                             Tab 2 - Chart                                   #
#-----------------------------------------------------------------------------#


def tab2():
    
    PageStart()

    # Add select begin-end date options
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    # added selectbox for data intervals
    interval = {'IntervalButton':['Daily','Weekly','Monthly'],'IntervalCode':['1d','1wk','1mo']}
    intervals= pd.DataFrame(interval)
    time_interval_button = st.selectbox('Interval',intervals['IntervalButton'])
    
    #added radio boxes to choose a graph type
    plot_type = st.radio('Plot type',['Line','Candle'])
   

    if plot_type =='Line':
    #plotting 
        def PlotTab2(start_date, end_date, time_interval_button):
            
            # data used in tab2 for closing price
            Chart_ClosingPrice = yf.Ticker(ticker).history(
                                                    start_date = start_date, 
                                                    end_date = end_date, 
                                                    interval=(intervals.loc[intervals['IntervalButton'] == time_interval_button,
                                                    'IntervalCode'].iloc[0]))
            
            
            fig,ax = plt.subplots(figsize=(12,7))
            
            ax.plot(Chart_ClosingPrice['Close'], label='Closing Price', color='blue')
            
            #twinning the first ax
            ax2=ax.twinx()
            
            #plotting bar chart
            ax2.bar(Chart_ClosingPrice.index,Chart_ClosingPrice['Volume'], label='Volume (in Millions)',color='green')
            
            # diminishing the scale of the bar char
            ax2.set_ylim(0,((Chart_ClosingPrice['Volume'].max())*7))
            
            #hiding y ticks of bar chart from the plot
            ax2.set_yticks([])
            
            #moving ticks to the right side
            ax.yaxis.tick_right()
            
            #store ticks in np array
            my_xticks = ax.get_xticks()
            
            # only show 1st and median ticks in the plot
            ax.set_xticks([my_xticks[0], 
                           np.median(my_xticks), 
                           my_xticks[-1]])
            
            #Legend labels for both axes shown together in one legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            
            ax2.legend(lines + lines2, 
                       labels + labels2, loc=0)
            
            ax.set_frame_on(False)
            ax2.set_frame_on(False)

            return st.pyplot(fig) 
        
        PlotTab2(start_date = start_date, 
                 end_date = end_date, 
                 time_interval_button = time_interval_button)
        
    else: 
        #plotting candle chart using pyplot
        Chart_Candle = yf.Ticker(ticker).history(start_date = start_date, 
                                                  end_date = end_date, 
                                                  interval = (intervals.loc[intervals['IntervalButton'] == time_interval_button, 
                                                  'IntervalCode'].iloc[0]))
        
        fig = go.Figure(data=[go.Candlestick(x=Chart_Candle.index,
                open = Chart_Candle['Open'],
                high = Chart_Candle['High'],
                low = Chart_Candle['Low'],
                close = Chart_Candle['Close'])])
        

        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)


#-----------------------------------------------------------------------------#
#                           Tab 3 - Financials                                #
#-----------------------------------------------------------------------------# 


# NOT WORKING || UNABLE TO UNDERSTAND THE ERROR #

def tab3():
    
    #selectbox to select Financial Statement type
    FinancialReportType = st.selectbox('Show:',['Income Statement','Balance Sheet','Cash Flow']) 
    PeriodDict = {'Report':['Annual','Quarterly'],'ReportCode':[True,False]}
    PeriodDF=pd.DataFrame(PeriodDict)
    
    
    print(PeriodDF)
    
    #radio button to select report period
    PeriodType = st.radio('Report:',PeriodDF['Report'])


    #defining a function to display the data according to selected parameters
    def ShowReport(FinancialReportType,PeriodType): 
        if FinancialReportType == 'Income Statement': 
            st.subheader(PeriodType +' Income Statement for '+ ticker)
            x = yf.Ticker(ticker).financials(yearly=(PeriodDF[PeriodDF['Report']==PeriodType,'ReportCode'].iloc[0]))
            
        elif FinancialReportType == 'Balance Sheet':
            st.subheader(PeriodType + ' Balance Sheet Statement for '+ ticker)
            x = yf.Ticker(ticker).balance_sheet(yearly=(PeriodDF[PeriodDF['Report']==PeriodType,'ReportCode'].iloc[0]))
            
        elif FinancialReportType == 'Cash Flow':
            st.subheader(PeriodType +' Cash Flow Statement for '+ ticker)
            x= yf.Ticker(ticker).cashflow(yearly=(PeriodDF[PeriodDF['Report']==PeriodType,'ReportCode'].iloc[0]))
            
        return st.dataframe(x)
    
    ShowReport(FinancialReportType=FinancialReportType, PeriodType=PeriodType)
    



    
#-----------------------------------------------------------------------------#
#                       Tab 4 - Monte Carlo simulation                        #
#-----------------------------------------------------------------------------#

# Source: [1] https://github.com/didizhx/Streamlit-Financial-Dashboard/blob/main/Streamlit%20Application%20Financial%20Programming%20(Yahoo).py

def tab4():
    N_Simualtions = st.selectbox('Number of simulations',[200,500,1000])
    T_Horizon = st.selectbox('Time horizon',[30,60,90])

    st.subheader('MonteCarlo simulation of a stock price for '+ ticker )
    class MonteCarlo(object):
        
        def __init__(self, ticker, data_source, start_date, end_date, time_horizon, n_simulation, seed):
            
            # Initiate class variables
            self.ticker = ticker  # Stock ticker
            self.data_source = data_source  # Source of data, e.g. 'yahoo'
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')  # Text, YYYY-MM-DD
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')  # Text, YYYY-MM-DD
            self.time_horizon = time_horizon  # Days
            self.n_simulation = n_simulation  # Number of simulations
            self.seed = seed  # Random seed
            self.simulation_df = pd.DataFrame()  # Table of results
            
            # Extract stock data
            self.stock_price = web.DataReader(ticker, data_source, self.start_date, self.end_date)
            
            # Calculate financial metrics
            # Daily return (of close price)
            self.daily_return = self.stock_price['Close'].pct_change()
            # Volatility (of close price)
            self.daily_volatility = np.std(self.daily_return)
            
        def run_simulation(self):
            
            # Run the simulation
            np.random.seed(self.seed)
            self.simulation_df = pd.DataFrame()  # Reset
            
            for i in range(self.n_simulation):

                # The list to store the next stock price
                next_price = []

                # Create the next stock price
                last_price = self.stock_price['Close'][-1]

                for j in range(self.time_horizon):
                    
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, self.daily_volatility)

                    # Generate the random future price
                    future_price = last_price * (1 + future_return)

                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price

                # Store the result of the simulation
                self.simulation_df[i] = next_price

        def plot_simulation_price(self):
            
            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10, forward=True)

            plt.plot(self.simulation_df)
            plt.title('Monte Carlo simulation for ' + self.ticker + \
                    ' stock price in next ' + str(self.time_horizon) + ' days')
            plt.xlabel('Day')
            plt.ylabel('Price')

            plt.axhline(y=self.stock_price['Close'][-1], color='red')
            plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
            ax.get_legend().legendHandles[0].set_color('red')

            st.pyplot(fig)
    # Initiate
    today = datetime.today().date().strftime('%Y-%m-%d')
    mc_sim = MonteCarlo(ticker=ticker, data_source='yahoo',
                    start_date='2021-01-01', end_date=today,
                    time_horizon=T_Horizon, n_simulation=N_Simualtions, seed=123)
    # Run simulation
    mc_sim.run_simulation()
    # Plot the results
    mc_sim.plot_simulation_price()
    

    
#-----------------------------------------------------------------------------#
#                         Tab 5 - Analysts Forecasting & News                 #
#-----------------------------------------------------------------------------#

# NOT WORKING || UNABLE TO UNDERSTAND THE ERROR #
 
def tab5():
     st.subheader('Analysts info for '+ ticker)
     st.dataframe(yf.Ticker(ticker).recommendations['Recommendations'].assign(hack='').set_index('hack'))
     st.dataframe(yf.Ticker(ticker).recommendations_summary['Recommendations Summary'].assign(hack='').set_index('hack'))
     st.dataframe(yf.Ticker(ticker).analyst_price_target['Price Target'].assign(hack='').set_index('hack')) #again, assign hides index values from printing
     st.dataframe(yf.Ticker(ticker).revenue_forecasts['Revenue Forecast'].assign(hack='').set_index('hack'))
     st.dataframe(yf.Ticker(ticker).earnings_forecasts['Earnings Forecasts'].assign(hack='').set_index('hack'))
     st.dataframe(yf.Ticker(ticker).earnings_trend['Earnings Trend'].assign(hack='').set_index('hack'))
     
     st.dataframe(yf.Ticker(ticker).news)

    
#-----------------------------------------------------------------------------#
#                                     Main                                    #
#-----------------------------------------------------------------------------#

#Creating a sidebar menu
def run(): #function to run the entire dashboard at once
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker1 = row.findAll('td')[0].text
        tickers.append(ticker1)
    
    tickers = [s.replace('\n', '') for s in tickers]
    
    # Add selection box
    global ticker #move the ticker variable to global var names.
    ticker = st.sidebar.selectbox("Select a ticker", tickers)
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart','Financials','Monte Carlo Simulation', 'Analysts Forecasts & News'])
     
    # defining an update button:
    run_button = st.sidebar.button('Update Data')
    if run_button:
        st.experimental_rerun()

     # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1
        tab1()
    elif select_tab == 'Chart':
        # Run tab 2
        tab2()
    elif select_tab == 'Financials':
        tab3()
        # Run tab 3
    elif select_tab == 'Monte Carlo Simulation':
        tab4()
        # Run tab 4
    elif select_tab == 'Analysts Forecasts & News':
        tab5()
        # Run tab 5

    
if __name__ == "__main__":
    run()

