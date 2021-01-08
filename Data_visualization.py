# This script is used to do the Data visualiation and Data Preprocessing

# Data visualiztion and manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 获取文件路径和表格名
S = 'C://Users/26529/Desktop/Data/2018_MCMProblemC_DATA/MCM_NFLIS_Data.xlsx'

s = 'Data'

def Get_Data(file, sheetname):
    
    # 获取数据
    
    Data_itself = pd.read_excel(file, sheet_name=sheetname)
    return Data_itself

Data = Get_Data(S,s)

Data['Total_name'] = Data['State']+' '+Data['COUNTY'] #通过这个标签来唯一标识





# Draw the frequency accumulation curve for the states

# 画出频率累积曲线对States来划分而言

X = Data[['YYYY','TotalDrugReportsState','State']]

X=X.drop_duplicates()

S = X.groupby('YYYY')['TotalDrugReportsState'].cumsum()

X=X.pivot(index='YYYY',columns='State',values='TotalDrugReportsState')

# 计算总的各州的数据用于计算后面的比例

X.plot(title='numbers',grid=True)

X['T']=X['KY']+X['OH']+X['PA']+X['VA']+X['WV']

for i in X.columns:
    if i!='T':
        X[i] = np.array(X[i])/np.array(X['T'])
    

# Make the plot
plt.figure()
plt.stackplot(X.index,X['KY'],X['OH'],X['PA'],X['VA'],X['WV'],labels=['KY','OH','PA','VA','WV'])
plt.legend(loc='upper left')
plt.margins(0,0)
plt.title('stacked area chart for States')
plt.show()


# 在这里我们计算一个总的趋势

S = S[4::5]  #这里我们取出来的是五个州的案例数的加和

S.index= list(range(2010,2018))

S.plot(title='total number',grid=True)

# Draw the histogram for the annual increase rate for each counties

#  计算county增长频率分布 
X = Data[['YYYY','Total_name','TotalDrugReportsCounty']]

X = X.drop_duplicates()


n_day=7

def calcu_rate(A,n):
        A['return'] =A['TotalDrugReportsCounty']/A['TotalDrugReportsCounty'].shift(n) -1
        return A
    # Using the apply function to calculate the historical return for all the indexes respectly
X=X.groupby('Total_name').apply(lambda x:calcu_rate(x,n_day))

X = X.dropna()

A = X[X['return']<1]

B = X[X['return']>1]

B = B[B['return']<60]

plt.figure()
plt.hist(A['return'].values,bins=50)
plt.show()

plt.figure()
plt.hist(B['return'].values,bins=50)
plt.show()
