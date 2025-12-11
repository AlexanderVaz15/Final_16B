# Final_16B
Investors often struggle to develop an asset portfolio that balances risk and returns across a diverse range of security holdings. The Bullfolio aims to elucidate this process by analyzing individual stock volatility and inter-stock relationships to assess overall risk and determine optimal portfolio allocations. 

API's Needed:
StockNews API – Retrieves stock-related news articles and sentiment scores.

FRED API – Provides macroeconomic indicators (CPI, GDP, interest rates, employment data, etc.)

YFinance API – Delivers historical stock price

Necessary Libraries:

import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

import os

import yfinance as yf

from fredapi import Fred

fred = Fred(api_key='Your_own_key')

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, Ridge

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from reportlab.lib import colors

from reportlab.lib.pagesizes import letter

To run the project, ensure that all of the libraries are installed on your device and import them into your notebook. Then, input your CSV file with all of your stocks and the total amount invested. Then run the functions and add the list of your stocks, as well as your invested amount, to get a reallocation of your investments. Lastly, get the PDF version of the dataframe and print it out to align your investments.

By: Alexander Vazquez, Joseph Kigler, Wildan Levitt
