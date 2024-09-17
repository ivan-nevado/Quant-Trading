import pandas as pd

# ####How to turn a csv into a dataframe
# df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Pandas Tutorial1\BTC-USD.csv')
# print (df)
# print('')
# print('----')
# ####How to drop a column in pandas
# #df = df.drop('Adj Close' , axis =1) #1 is for column and 0 for rows
# #print(df)

# ####How to rename a column in pandas
# #df.rename(columns = {'Volume':'Vol','Date':'Time'},inplace = True)
# #print(df)

# ####How do you change the index?

# # df = df.set_index('Date')
# # print(df)

# # ####How do you reset the index in pandas?
# # df = df.reset_index()
# # print(df) 

# ####How to make a dataframe from scratch
# # df2 = pd.DataFrame({
# #     'name':['jon','jon','jon','mindy'],
# #     'thing':['food','food','chair','chair'],
# #     'rating':[5,5,6,22]
# # })
# # print(df2)
# # print('')
# # print('----')

# # ####How to drop duplicates
# # df2 = df2.drop_duplicates()
# # print(df2)
# # print('')
# # print('----')

# # ####How to drop dupes based on columns
# # df2 = df2.drop_duplicates(subset =['name'])
# # print(df2)
# # print('')
# # print('----')

# ####How to group by in pandas
# # df = pd.DataFrame({'food':['chicken','chicken','bacon','ranch'],
# #                     'cals':[500.,300.,600.,52.]})
# # print (df)
# # print('')
# # print('----')

# # df = df.groupby(['food']).mean()#mean is average
# # print(df)

# #How to filter in pandas

# #use operators for numbers

# # df = df[df.High > 4500]
# # print(df)
# # print('')
# # print('----')

# # #use multiple operators
# # df = df[(df.High > 700000) | (df.Volume == 2.486080e+09)]
# # print(df)
# # print('')
# # print('----')

# # #filter with list
# # highs = [468.174011,67779.015625, 69371.476563]
# # df = df[df.High.isin(highs)]
# # print(df)

# # df = pd.DataFrame({'food':['chicken','chicken','bacon','ranch'],
# #                    'cals':[500.,390.,600.,52.,]})
# # df = df[df.food.str.startswith('b')]
# # print(df)

# ####Sometimes we want the largest or smallest numbers
# # df = df.nsmallest(2,'Volume')
# # print(df)

# # df = df.iloc[7:10,:]
# # print (df)

# #### how to sort in pandas

# # df = df.sort_values(by = ['Close'])
# # print(df)

# #### How to sort by multiple columns

# df = df.sort_values(by = ['Close','Volume'])
# print(df)

####How to read in an XLSX File

df = pd.read_excel(r'C:\Users\Iván\Desktop\Ivan\Pandas Tutorial1\file_example_XLS_10.xls',index_col= 0)
print (df)