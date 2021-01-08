# Codes for Part1

# 1) package imported

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import statsmodels.regression as streg




# 获取文件路径和表格名
# 2) Get data source path and sheet name

S = 'C://Users/26529/Desktop/Data/2018_MCMProblemC_DATA/MCM_NFLIS_Data.xlsx'
s = 'Data'

def Get_Data(file, sheetname):
    
    #  Get Data
    
    Data_itself = pd.read_excel(file, sheet_name=sheetname)
    
    return Data_itself

Data = Get_Data(S,s)

Data['Total_name'] = Data['State']+' '+Data['COUNTY'] #通过这个标签Total_name来唯一标识


# 获取地区数据并计算他们之间的相关关系
# 3) Divide and aggregate data by county then calculate the relationship

X = Data[['YYYY','Total_name','TotalDrugReportsCounty']]

X = X.drop_duplicates()

X = X.pivot(index='YYYY',values='TotalDrugReportsCounty',columns='Total_name')

X = X.fillna(value=0)

T = X.corr()

# 计算每个药品的时间序列数据

D = Data[['YYYY','SubstanceName','Total_name','DrugReports']]

D=D.sort_index(by=['SubstanceName','YYYY'])

Substance_name = list(set(D.SubstanceName))  # 这里存放所有药物名

Total__name = list(set(D.Total_name)) # 这里存放所有county名

Raw_Data = dict()  # 这里存放所有药品数据

for i in Substance_name:
        
    d = D[D['SubstanceName']==i]
    
    #name_temp = list(set(d.Total_name.values))
    
    d=d.pivot(columns='Total_name',values='DrugReports',index='YYYY')
    
    dd = pd.DataFrame(columns=Total__name,index=d.index)
    dd[d.columns] = d[d.columns]
    dd = dd.sort_index(level=T.columns,axis=1)
    dd = dd.fillna(0)
    Raw_Data[i] = dd
    
# 这是我们第一个核心的线性模型；
'''
我们在这里对一阶滞后项
'''    

def Model1(Corr, r):
    
    Future_data = dict() #为预测准备好起始的解释变量数据
    
    # 因子的构造
    
    n = r.shape[0]-1
    print(n)
    
    c = np.array(Corr)
    
    r_dff = np.diff(r,axis=0)   #成功构造被解释变量
    
    c=c-np.diag(np.diag(c)) 
    
    Impact = np.dot(r,c)[0:n]   #解释变量1：外部影响因子 n*461
    Future_data['1'] = np.dot(r,c)[n].reshape([461,1]) #未来因子1
    
    Impact_plus = Impact*r[0:n,:] #解释变量2： 外部感染因子
    Future_data['2'] = Future_data['1']*(r[n,:].reshape([461,1])) #未来因子2
    
    r1 = r[0:n,]
    Future_data['3'] = r[n,].reshape([461,1])
    
    r1_square = np.power(r[0:n,],2)
    Future_data['4'] = np.power(r[n,].reshape([461,1]),2)
    
    Beta0 = dict()    
    Beta1 = dict()
    Beta2 = dict()
    Beta3 = dict()
    
    Result = dict()
    
    for i in range(461):
        
        Explain = np.array([Impact[:,i],Impact_plus[:,i],r1[:,i],r1_square[:,i]]).T
        
        Md2 = streg.linear_model.OLS(r_dff[:,i].reshape([n,1]),Explain)
        result = Md2.fit()
        Beta0[i] = float(result.params[0])
        Beta1[i] = float(result.params[1])
        Beta2[i] = float(result.params[2])
        Beta3[i] = float(result.params[3])
        Result[i] = result
    return Beta0, Beta1, Beta2, Beta3, Result, Future_data

#[a1,a2,a3,a4,R,F] = Model1(T,Raw_Data['Heroin'])


def Predict(Future_data, Result,r,a1,a2,a3,a4):
    
    Ran = 3
    
    Mean1 = np.mean(list(a1.values()))
    Mean2 = np.mean(list(a2.values()))
    Mean3 = np.mean(list(a3.values()))
    Mean4 = np.mean(list(a4.values()))
    Std1 = np.std(list(a1.values()))
    Std2 = np.std(list(a2.values()))
    Std3 = np.std(list(a3.values()))
    Std4 = np.std(list(a4.values()))
    Max1 = Mean1+Ran*Std1
    Max2 = Mean2+Ran*Std2
    #Max3 = Mean3+Ran*Std3
    #Max4 = Mean4+Ran*Std4
    Min1 = Mean1-Ran*Std1
    Min2 = Mean2-Ran*Std2
    #Min3 = Mean3-Ran*Std3
    Min4 = Mean4-Ran*Std4
    Max_3 = np.max([abs(np.min(list(a3.values()))),abs(np.max(list(a3.values())))])
    explain = np.array([Future_data['1'],Future_data['2'],Future_data['3'],Future_data['4']])[:,:,0]
    
    Temp =np.zeros(461)
    
    
    '''
    进行严格的系数预处理和限制，
    a1和a2必须符号相反
    且a2必须很小
    a3在-1-1之间
    a4必须也很小并且是负数
    '''
    
    
    for i in range(461):
        
        if(Result[i].params[3]>0):
            
            Result[i].params[3]=0
            
        if(Result[i].params[3]<Min4):
            
            Result[i].params[3]=Min4
         
        #Result[i].params[2]= Result[i].params[2]/Max_3
        
        if(Result[i].params[2]>0):
            Result[i].params[2]=0
            
        
        if(Result[i].params[0]<Min1):
            
            Result[i].params[0]=Min1
            
        if(Result[i].params[0]>Max1):
            
            Result[i].params[0]=Max1
            
        if(Result[i].params[1]<Min2):
            
            Result[i].params[1]=Min2
            
        if(Result[i].params[1]>Max2):
            
            Result[i].params[1]=Max2
            
        #Result[i].params[2]=1
        Result[i].params[1]=0.001
        Temp[i] = Result[i].predict(explain[:,i])
        
        
        
    Temp[np.where(Temp==Temp.max())]=np.log(abs(Temp.max()))
    
    Temp[np.where(Temp==Temp.min())]=-np.log(abs(Temp.min()))
    
    k=1
    
    Temp = Temp+k*np.random.standard_normal(Temp.shape)
    
    r = np.insert(r,r.shape[0],values=Temp+r[r.shape[0]-1,:],axis=0)
    
    r[r<0]=0

    return r

def Find_the_first_State(Data):
    
    # Try to find where the drug derived from, the state
    # 找毒品的起源地
    
    # N 将是我们算法操作的唯一数据集；
    N = Data[['YYYY','SubstanceName','State','DrugReports']]
    
    # 我们通过一系列操作将N取出每一个substance最新的一年,并合理排序
    
    def Find_first(A):
        A['first_year'] =np.min(list(set(A.YYYY.values)))
        return A
    
    N=N.groupby('SubstanceName').apply(lambda x:Find_first(x))
    N=N.sort_index(by=['SubstanceName','YYYY'])

    # 药品的总名字和地区的总名字
    S_name = list(set(N.SubstanceName))
    ST_name = list(set(N.State))


    # 这里提前我们的模型的可能会在的州
    df_result = dict()


    for i in range(len(set(N.SubstanceName))):
    
        # 将每一个数据的药品都读取过来
        D = N[N['SubstanceName']==S_name[i]]
    
        # 如果一直在一个洲里面活动，就是那个洲
        if len(set(D.State.values))==1:
            df_result[S_name[i]]=list(set(D.State.values))[0]
        else:
            D = D[D['YYYY']==D['first_year']]
            D = D.groupby(by=['State'])['DrugReports'].sum()
            
            # 我们其实已经很轻松的将最先的一年数据找出来了
            # 接下来将最大的作为起始点就可以了
            df_result[S_name[i]]=D[D.values==D.max()].index[0]
            
    
    DF_RESULT = dict()
    # 将键值对调转
    
    for j in ST_name:
        
        DF_RESULT[j]=list()
        
    
    for k in list(df_result.keys()):
        
        DF_RESULT[df_result[k]].append(k)
        
    return DF_RESULT

# s=Find_the_first_State(Data)
   
def Get_New_Explain(Corr,r):
    
    
    Future_data = dict() #为预测准备好起始的解释变量数据
    
    
    
    n = r.shape[0]-1
    
    c = np.array(Corr)
        
    c=c-np.diag(np.diag(c)) 
    
    Future_data['1'] = np.dot(r,c)[n].reshape([461,1]) #未来因子1
    
    Future_data['2'] = Future_data['1']*(r[n,:].reshape([461,1])) #未来因子2
    
    Future_data['3'] = r[n,].reshape([461,1])
    
    Future_data['4'] = np.power(r[n,].reshape([461,1]),2)
    
    return Future_data

def Run_predict(Corr, rawdata,k_steps):
    
    r = np.array(rawdata)
    
    [a1,a2,a3,a4,R,F] = Model1(Corr, r)
    r=Predict(F,R,r,a1,a2,a3,a4)
    S = dict()
    for i in range(k_steps-1):
        
        [a1,a2,a3,a4,R,F] = Model1(Corr, r)
        
        S[i]=R
        
        r=Predict(F,R,r,a1,a2,a3,a4)


    return r,S


def Combine_Data(Raw_Data, Name):
    
    S=pd.DataFrame(index=Raw_Data['Heroin'].index,columns=Raw_Data['Heroin'].columns)
    
    S = S.fillna(0)
    
    n = len(Name)
    
    tem1 = pd.DataFrame(data=Raw_Data[Name[0]],index=Raw_Data['Heroin'].index,columns=Raw_Data['Heroin'].columns)
    
    tem1 = tem1.fillna(0)
    
    # 这里我们输入列名就将想要的毒品数据进行加总计算
    S = S + tem1

    if n>1:
        for i in range(n-1):
            tem1 = pd.DataFrame(data=Raw_Data[Name[i]],index=Raw_Data['Heroin'].index,columns=Raw_Data['Heroin'].columns)
            tem1 = tem1.fillna(0)
            S = S + tem1
    return S
    

def Get_Balance_point(x,k):
    
    n1 = x.shape[0]
    r = (x[1:n1]-x[0:n1-1])/x[0:n1-1]
    r[np.isnan(r)]=0
    r=list(r)
    r.reverse()
    for i in range(len(r)):
        if abs(r[i])>k:
            break
    C = len(x)-i
    return C
    
    
    
def Generate_chart(MMM,Raw_Data):
    
    
    # 我们先将原始数据转换为DataFrame格式
    Source_chart = pd.DataFrame(data=MMM,index=list(range(2010,MMM.shape[0]+2010)),columns=Raw_Data['Heroin'].columns)
    
    # 然后再将他进行查分之后取出差分部分
    S_diff = Source_chart.diff().iloc[1:,:]
    
    
    Information = pd.DataFrame(index=Source_chart.columns,columns=['B_time','B_value','G_time','G_value'])
    for i in S_diff.columns.values:
        
        Information.loc[i,'G_value']=S_diff[i].max()
        Information.loc[i,'G_time']=S_diff.index[np.where(S_diff[i]==S_diff[i].max())].values[0]
        
        a = Get_Balance_point(Source_chart[i].values,0.05)
        
        Information.loc[i,'B_time']=a+2010-1
        Information.loc[i,'B_value']=Source_chart.loc[a+2010-1,i]
        
    return Information     


#D1 = pd.DataFrame(data=D_Heroin,columns=Raw_Data['Heroin'].columns,index=range(2010,2018))

Substance_name.remove('Oxycodone')
Substance_name.remove('Heroin')

Data_1 = Combine_Data(Raw_Data,['Heroin'])
D_Heroin,S=Run_predict(T,Data_1,5)
D1 = pd.DataFrame(data=D_Heroin,columns=Raw_Data['Heroin'].columns,index=range(2010,2023))

Data_2 = Combine_Data(Raw_Data,['Oxycodone'])
D_Oxy,S=Run_predict(T,Data_2,5)
D2 = pd.DataFrame(data=D_Oxy,columns=Raw_Data['Heroin'].columns,index=range(2010,2023))

Data_3 = Combine_Data(Raw_Data,Substance_name)
D_allother,S=Run_predict(T,Data_3,5)
D3 = pd.DataFrame(data=D_allother,columns=Raw_Data['Heroin'].columns,index=range(2010,2023))

#D1.sum(axis=1).plot(title='Aggregation',grid=True,figsize=[15,10])
'''
plt.scatter(range(len(I1['G_value'].values)),I1['G_value'].values)
'''
'''
Real_data = Data_3


Predicted_data = pd.DataFrame(columns=Real_data.columns,index=Real_data.index[1:])

for i in range(461):
    r_p = list(R[i].predict())
    for j in range(7):
        Predicted_data.iloc[j,i]=r_p[j]
        

Real_data1=Real_data1.loc[2011:]

PPP=Predicted_data.sum(axis=1)
RRR=Real_data1.sum(axis=1)

plt.figure()
plt.scatter(np.array(range(2011,2018)).T,np.array(PPP.values).T)
plt.scatter(np.array(range(2011,2018)).T,np.array(RRR.values).T)
plt.title('Model erro analysis')
plt.grid()
plt.show()

np.sqrt(np.square(PPP.values-RRR.values).sum()/7)
'''

plt.figure(figsize=[15,10])
plt.grid()
plt.title('Compare')
plt.plot(d1.sum(axis=1),label='Beta<0')

plt.legend('1')
plt.plot(d2.sum(axis=1),label='Beta=0')


#plt.plot(d3.sum(axis=1),label='Beta>0')


#plt.plot(d4.sum(axis=1),label='Index*2.5')


#plt.plot(d5.sum(axis=1),label='Index*3')

plt.legend()
plt.show()
