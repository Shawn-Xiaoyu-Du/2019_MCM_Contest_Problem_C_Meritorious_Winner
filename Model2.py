# Codes for Part2

# 包的加载
# Package loaded
# 这里有很多包其实在这个方法里是用不着的，比如我们仅仅用了GBDT
import pandas as pd
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline   #做模型之间的管子链接


# Getdata_ML
# Data Preprocessing




S = 'C://Users/26529/Desktop/Data/2018_MCMProblemC_DATA/ML_Dataset'

s=['ACS_10_5YR_DP02_with_ann.csv','ACS_11_5YR_DP02_with_ann.csv','ACS_12_5YR_DP02_with_ann.csv','ACS_13_5YR_DP02_with_ann.csv','ACS_14_5YR_DP02_with_ann.csv','ACS_15_5YR_DP02_with_ann.csv','ACS_16_5YR_DP02_with_ann.csv']

def Get_Data1(filename,path):
    
    
    Whole_Data_set=dict()
    
    c = list(range(2010,2017))
    
    j=0
    #  Get Data
    for i in filename:
        
        Data_itself = pd.read_csv(path+'/'+i)
        
        Whole_Data_set[c[j]] = Data_itself
        
        j = j+1
        
    
    return Whole_Data_set


Whole_Data_set = Get_Data1(s,S)

# 下面是数据预处理的过程

# 删除缺失数据和不需要的数据，正则表达式 0.5h
# 将剩下的数据进行标准化处理   0.2h
# 将所在county的该年毒品数做连接 0.3h

def Data_drop(Whole_Data_set):
    
    # 在这里我们丢弃所有的含有(X)的变量
    
    for i in list(Whole_Data_set.keys()):
        
        
        #print(Whole_Data_set[i].columns)
        cols=[x for j,x in enumerate(Whole_Data_set[i].columns) if Whole_Data_set[i].iat[3,j]=='(X)']
        Whole_Data_set[i]=Whole_Data_set[i].drop(cols,axis=1)
        Whole_Data_set[i]=Whole_Data_set[i].drop(['GEO.id'],axis=1)
        Whole_Data_set[i].drop(axis=0, index=0, inplace=True)   #删除第一行重复数据
        
        
    # 接下来我们将通过交叉处理取得最小的数据共同交集；
    # 利用这个数据交叉集合
    
    A0=set(Whole_Data_set[2010].columns)
    A1=set(Whole_Data_set[2011].columns)
    A2=set(Whole_Data_set[2012].columns)
    A3=set(Whole_Data_set[2013].columns)
    A4=set(Whole_Data_set[2014].columns)
    A5=set(Whole_Data_set[2015].columns)
    A6=set(Whole_Data_set[2016].columns)
    
    A = list(A0.intersection(A1).intersection(A3).intersection(A4).intersection(A5).intersection(A6))
    
    for i in list(Whole_Data_set.keys()):
        
        
        Whole_Data_set[i]=Whole_Data_set[i][A]
    
    return Whole_Data_set
        
Whole_Data_set = Data_drop(Whole_Data_set)

# 将得到的DataFrame打上标签，我们先获取一下被解释变量数据

S1 = 'C://Users/26529/Desktop/Data/2018_MCMProblemC_DATA/MCM_NFLIS_Data.xlsx'

s1 = 'Data'

def Get_Data(file, sheetname):
    
    # 获取数据
    
    Data_itself = pd.read_excel(file, sheet_name=sheetname)
    return Data_itself

Data1 = Get_Data(S1,s1)

def Join_Data(Whole_Data_set,Data1):
    
    Data = Data1[['YYYY','FIPS_Combined','TotalDrugReportsCounty']]
    
    Data = Data.drop_duplicates()
    
    for i in list(Whole_Data_set.keys()):
        
        data = Data[Data['YYYY']==i]
        #print(data.shape)
        
        Whole_Data_set[i]['GEO.id2']=Whole_Data_set[i]['GEO.id2'].values.astype(int)
        Whole_Data_set[i] = Whole_Data_set[i].merge(data,left_on='GEO.id2',right_on='FIPS_Combined')
        print(i)
    
    return Whole_Data_set
    
    


Whole_Data_set = Join_Data(Whole_Data_set,Data1)

L = list(Whole_Data_set[2010].columns)
L.remove('YYYY')
L.remove('FIPS_Combined')

# 丢弃所有缺失值，正态化

from sklearn import preprocessing

def Dropspecialstr_columns(L,Whole_Data_set):
    
    c = 0    
    
    for i in list(Whole_Data_set.keys()):
        
        if c==0:
            
            New_Df = Whole_Data_set[i][L]
            c = c+1
            
        else:
            AAA = Whole_Data_set[i][L]
            
            New_Df = pd.concat([New_Df,AAA])
    
    New_Df = New_Df.apply(pd.to_numeric, errors='coerce')
    
    New_Df = New_Df.dropna(axis=1)
    
    # 正态化
    
    New_Df=pd.DataFrame(data=preprocessing.scale(New_Df),index=New_Df.index,columns=New_Df.columns)

    return New_Df


New_Df = Dropspecialstr_columns(L,Whole_Data_set)




# 用GBDT模型做特征处理和特征筛选并输出权重 0.5h
# 用该权重加权数据获得一个指标 0.5h
# 用该指标作为原模型的一个变量加入回归之中去，但是这个时候无法预测，因为没有未来的因子数据 1h
# 通过进行参数估计，对这个变量进行分解后得到每一部分的灵敏度，据此进行分析，并给出一些假设数据进行未来模拟1h
# 写后面的所有回答和两页信纸  你们的工作
# 进行分析  你们的工作
# 修改 剩余时间





    ## 设置机器学习的参数，区分预测集和训练集
    ## Set parameters and load data
    
X =  New_Df.drop('TotalDrugReportsCounty',axis=1)
y = pd.DataFrame(New_Df.TotalDrugReportsCounty)
    
    #print(y.columns)
    #print(X.columns)
n_estimator = 5

    # 在这里我们实际上不需要做测试集和训练集的区分，因为本部分本来就是训练的部分
X_train = X
y_train = y

    # 需要将对 LR和GBDT的训练集给区分开来
    # It is important to train the ensemble of trees on a different subset of the training data than the linear regression model to avoid overfitting, in particular if the total number of leaves is similar to the number of training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Supervised transformation based on gradient boosted trees
    # 这里是训练好的模型，GBDT模型，编码模型和逻辑回归模型
grd = GradientBoostingRegressor(n_estimators=10)   

grd.fit(X_train, y_train)

Impact = list(list(grd.feature_importances_))
    
plt.figure()
plt.plot(Impact)
plt.title('Feature_importance')
plt.show()


y_pred_grd = list(grd.predict(X_test))

y_test.reset_index(inplace=True)

y_test=list(y_test.TotalDrugReportsCounty.values)
plt.figure()
plt.scatter(range(len(y_test)),y_test)

plt.scatter(range(len(y_pred_grd)),y_pred_grd)
plt.title('error analysis')
plt.grid()
plt.show()

RMSD = np.sqrt(np.square(np.array(y_pred_grd)-np.array(y_test)).sum()/len(y_test))

Iter =10
s_lo = list()
s_value = list()

a=Impact.copy()

for i in range(Iter):
    
    s_lo.append(a.index(max(a)))
    s_value.append(a[s_lo[i]])
    a[s_lo[i]] = 0
    
plt.figure()
plt.plot(range(10),s_value)
plt.grid()
plt.title('Feature selection')
plt.show()

# 开始着手处理part2参数

Feature = ['HC01_VC66', 'HC01_VC70', 'HC01_VC88', 'HC01_VC71']
def Extract_data(Whole_Data_set,Feature):
    
    a = dict()
    
    
    for i in list(Whole_Data_set.keys()):
        
        a[i] = Whole_Data_set[i][Feature]
        a[i] = a[i].apply(pd.to_numeric, errors='coerce')
        a[i] = a[i].dropna(axis=1)
        a[i] = pd.DataFrame(data=preprocessing.scale(a[i]),index=a[i].index,columns=a[i].columns)
        a[i]['GEO.id2']=Whole_Data_set[i]['GEO.id2']
    return a


a = Extract_data(Whole_Data_set,Feature)

Weight = s_value[0:4]

for i in list(a.keys()):
    a[i]['Index'] =Weight[0]*a[i].iloc[:,0]+Weight[1]*a[i].iloc[:,1]+Weight[2]*a[i].iloc[:,2]+Weight[3]*a[i].iloc[:,3]
    a[i] = a[i][['Index','GEO.id2']]