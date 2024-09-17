import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\Indicators\Support_Resistance\EURUSD_Candlestick_1_D_ASK_05.05.2003-19.10.2019.csv")
df = df[df['volume']!=0]
df.reset_index(drop=True, inplace=True)
df.isna().sum()

def support(df1,l,n1,n2):
    for i in range(l-n1+1, l+1):
        if(df1.low[i] > df1.low[i-1]):
            return 0
        
    for i in range(l+1, l+n2+1):
        if(df1.low[i]<df1.low[i-1]):
            return 0
        
    return 1

def resistance(df1, l ,n1, n2):
    for i in range(l-n1+1,l+1):
        if(df1.high[i] < df1.high[i-1]):
            return 0
    for i in range(l+1, l+n2+1):
        if(df1.high[i]>df1.high[i-1]):
            return 0
        
    return 1

    
ss=[]
rr=[]
n1=3
n2=2
for row in range(3,205):
    if support(df, row, n1,n2):
        ss.append((row,df.low[row]))
    if resistance(df, row, n1, n2):
        rr.append((row,df.high[row]))    

s = 0
e=200
dfpl = df[s:e]

fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                                     open = dfpl['open'],
                                     high=dfpl['high'],
                                     low=dfpl['low'],
                                     close=dfpl['close'])])

c = 0
while(1):
    if(c>len(ss)-1): #or sr[c][0] > e
        break
    fig.add_shape(type='line', x0=ss[c][0]-3, y0=ss[c][1],
                  x1=e,
                  y1=ss[c][1],
                  line=dict(color="MediumPurple", width=3))
    c+=1
c=0
while(1):
    if(c>len(rr)-1):
        break
    fig.add_shape(type='line', x0=rr[c][0]-3, y0=rr[c][1],
                  x1=e,
                  y1=rr[c][1],
                  line=dict(color="RoyalBlue", width=1))
    c+=1

fig.show()