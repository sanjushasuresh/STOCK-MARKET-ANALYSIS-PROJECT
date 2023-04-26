import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

import warnings
warnings.filterwarnings('ignore')

## Import Data

Data=pd.read_csv(r"C:\Users\Shine Caleb\Reliance_Stock_Forecast_Vir\Reliance\Reliance Daily Dataset.csv")
Data.dropna(inplace=True)

Data.drop(columns=["Unnamed: 0"],inplace=True)
Data

Data["Date"]=pd.to_datetime(Data["Date"])

Data["month"] = Data.Date.dt.strftime("%b") # month extraction
Data["year"] = Data.Date.dt.strftime("%Y") # year extraction

## January Month Prices

Jan_Data=Data.tail(31)

plt.figure(facecolor='black',figsize=[10,7])
 
ax = plt.axes()
ax.set_facecolor("black")

plt.plot(Jan_Data["High"])
plt.plot(Jan_Data["Low"])

plt.legend(["High", "Low"], loc ="upper right")

ax.set_xlabel('Days')
ax.set_ylabel('Price')

ax.set_xticklabels([4,8,12,16,20,24,28])

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

ax.tick_params(axis='x', colors='White')
ax.tick_params(axis='y', colors='White')

ax.spines['left'].set_color('White')  
ax.spines['bottom'].set_color('White')

plt.savefig(r'C:\Users\Shine Caleb\Reliance_Stock_Forecast_Vir\Reliance\static\HL.jpeg')

plt.figure(facecolor='black',figsize=[10,7])
 
ax = plt.axes()
ax.set_facecolor("black")

plt.plot(Jan_Data["Open"])

ax.set_xlabel('Days')
ax.set_ylabel('Opening Price')

ax.set_xticklabels([4,8,12,16,20,24,28])

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

ax.tick_params(axis='x', colors='White')
ax.tick_params(axis='y', colors='White')

ax.spines['left'].set_color('White')  
ax.spines['bottom'].set_color('White')

plt.savefig(r'C:\Users\Shine Caleb\Reliance_Stock_Forecast_Vir\Reliance\static\Open.jpeg')

## Model Building

Train=Data.head(1964)
Test=Data.tail(30)

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

### Holts winter exponential smoothing with Multiplicative seasonality and additive trend


hwe_model_Open = ExponentialSmoothing(Train["Open"],seasonal="mul",trend="add",seasonal_periods=365).fit()
pred_hwe_Open = hwe_model_Open.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_Open,Test["Open"])

hwe_model_Close = ExponentialSmoothing(Train["Close"],seasonal="mul",trend="add",seasonal_periods=365).fit()
pred_hwe_Close = hwe_model_Close.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_Close,Test["Close"])

hwe_model_High = ExponentialSmoothing(Train["High"],seasonal="mul",trend="add",seasonal_periods=365).fit()
pred_hwe_High = hwe_model_High.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_High,Test["High"])

hwe_model_Low = ExponentialSmoothing(Train["Low"],seasonal="mul",trend="add",seasonal_periods=365).fit()
pred_hwe_Low = hwe_model_Low.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_Low,Test["Low"])

def Forecast(Days):
    import datetime
    df=pd.DataFrame()
    df["Open"]= hwe_model_Open.forecast(Days).round(1)
    df["Close"]= hwe_model_Close.forecast(Days).round(1)
    df["High"]=hwe_model_High.forecast(Days).round(1)
    df["Low"]=hwe_model_Low.forecast(Days).round(1)
    test_date = datetime.datetime.strptime("01-02-2023", "%d-%m-%Y")
    date_generated = pd.date_range(test_date, periods=Days)
    df.index=date_generated.strftime("%d-%m-%Y")
    return df

import plotly.express as px
figplotly = px.line(Data,x="Date", y="Open", title='Reliance Stock Price',template = "plotly_dark")

from flask import *
import numpy as np
import pandas as pd

app = Flask(__name__)  
  
@app.route('/')  
def customer():  
   return render_template('Home.html')  

@app.route("/About")
def About():
    return render_template('About.html',figure=figplotly.to_html())

@app.route("/Analysis")
def Analysis():
    return render_template('Analysis.html')

@app.route('/Rec',methods = ["POST", "GET"])  
def html_table():
    if request.method == 'POST':
        result = request.form
        # return result["Days"]
        Day=(int(result["Days"]))
        df=pd.DataFrame(Forecast(int(result["Days"])))

        import datetime
        test_date = datetime.datetime.strptime("01-02-2023", "%d-%m-%Y")
        date_generated = pd.date_range(test_date, periods=int(result["Days"]))

        plt.figure(facecolor='black',figsize=[10,7])

        ax = plt.axes()
        ax.set_facecolor("black")

        plt.plot(df["High"])
        plt.plot(df["Low"])

        plt.legend(["High", "Low"], loc ="upper right")

        ax.set_xticklabels(date_generated.strftime("%d"))

        ax.set_xlabel('Day')
        ax.set_ylabel('Price')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        ax.tick_params(axis='x', colors='White')
        ax.tick_params(axis='y', colors='White')

        ax.spines['left'].set_color('White')  
        ax.spines['bottom'].set_color('White')

        plt.savefig(r'C:\Users\Shine Caleb\Reliance_Stock_Forecast_Vir\Reliance\static\HL_Forecast.jpeg')

        plt.figure(facecolor='black',figsize=[10,7])
        
        ax1 = plt.axes()
        ax1.set_facecolor("black")

        plt.plot(df["Open"])

        ax1.set_xticklabels(date_generated.strftime("%d"))

        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price')

        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')

        ax1.tick_params(axis='x', colors='White')
        ax1.tick_params(axis='y', colors='White')

        ax1.spines['left'].set_color('White')  
        ax1.spines['bottom'].set_color('White')

        plt.savefig(r'C:\Users\Shine Caleb\Reliance_Stock_Forecast_Vir\Reliance\static\Open_Forecast.jpeg')
        
        return render_template('Rec.html',tables=[df.to_html(classes='data')], titles=df.columns.values)
        # df.index=range(1,11)
        # return df.to_html(header="true", table_id="table")
        # return result
        # return render_template("Rec.html",)
        # Song1=df["Songs"][1],


if __name__ == '__main__':  
   app.run(debug=True)
   