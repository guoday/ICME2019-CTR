import os
import pandas as pd
import numpy as np
import random
import json
import gc
from gensim.models import Word2Vec
import multiprocessing
from collections import Counter
from sklearn import preprocessing
import scipy.special as special
from pandas import DataFrame, Series
from collections import Counter
np.random.seed(2019)
random.seed(2019)


def all_cont(train_df,test_df,f):    
    print("all cont:",f)
    dic={}
    for item in train_df[f].values:
        try:
            dic[item]+=1
        except:
            dic[item]=1    
    for item in test_df[f].values:
        try:
            dic[item]+=1
        except:
            dic[item]=1     
    cont=[]
    for item in train_df[f].values:
        cont.append(dic[item])        
    train_df[f+'_all_cont']=cont
    print('train','done')
    
    cont=[]
    for item in test_df[f].values:
        cont.append(dic[item])        
    test_df[f+'_all_cont']=cont
    print('test','done')            
    print(f+'_all_cont')
    print('avg of cont',np.mean(train_df[f+'_all_cont']),np.mean(test_df[f+'_all_cont']))   
    
def day_cont(train_df,test_df,f):
    print("day cont:",f)
    dic={}
    cont=[]
    dics=[]
    day=0
    for item in train_df[['day',f]].values:
        item[0]=int(item[0])
        if day!=item[0]:
            dics.append(dic)
            dic={}
            day+=1
            print(day)
        try:
            dic[item[1]]+=1
        except:
            dic[item[1]]=1
    dics.append(dic)        
    day=0
    dic=dics[day]
    for item in train_df[['day',f]].values:
        item[0]=int(item[0])
        if day!=item[0]:
            day+=1 
            dic=dics[day]
            print(day) 
        cont.append(dic[item[1]])        
    train_df[f+'_day_cont']=cont
    print('train','done')
    
    dic={}
    for item in test_df[f].values:
        try:
            dic[item]+=1
        except:
            dic[item]=1            
    cont=[]
    for item in test_df[f].values:
        cont.append(dic[item])
    test_df[f+'_day_cont']=cont    
    print('test','done')
    print(f+'_day_cont')
    print('avg of cont',np.mean(train_df[f+'_day_cont']),np.mean(test_df[f+'_day_cont']))   
    
def combine(train_df,test_df,f1,f2):
    train_df[f1+'_'+f2]=train_df[f1]*1e7+train_df[f2]
    train_df[f1+'_'+f2]=train_df[f1+'_'+f2].astype(int)
    test_df[f1+'_'+f2]=test_df[f1]*1e7 +test_df[f2]
    test_df[f1+'_'+f2]=test_df[f1+'_'+f2].astype(int)
    
def kfold_static(train_df,test_df,f,label):
    print("K-fold static:",f+'_'+label)
    #K-fold positive and negative num
    avg_rate=train_df[label].mean()
    num=len(train_df)//5
    index=[0 for i in range(num)]+[1 for i in range(num)]+[2 for i in range(num)]+[3 for i in range(num)]+[4 for i in range(len(train_df)-4*num)]
    random.shuffle(index)
    train_df['index']=index

    dic=[{} for i in range(5)]
    dic_all={}
    for item in train_df[['index',f,label]].values:
        try:
            dic[item[0]][item[1]][item[2]]+=1
        except:
            dic[item[0]][item[1]]=[0,0]
            dic[item[0]][item[1]][item[2]]+=1
        try:
            dic_all[item[1]][item[2]]+=1
        except:
            dic_all[item[1]]=[0,0]
            dic_all[item[1]][item[2]]+=1
    print("static done!")
                
    positive=[]
    negative=[]
    rate=[]
    for item in train_df[['index',f]].values:
        n,p=dic_all[item[1]]
        try:
            p-=dic[item[0]][item[1]][1]
            n-=dic[item[0]][item[1]][0] 
        except:
            pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(avg_rate)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))  
            
    train_df[f+'_'+label+'_positive_num']=positive
    train_df[f+'_'+label+'_negative_num']=negative
    train_df[f+'_'+label+'_rate']=rate
    print("train done!")
    #for test
    positive=[]
    negative=[]
    rate=[]
    for uid in test_df[f].values:
        p=0
        n=0
        try:
            p=dic_all[uid][1]
            n=dic_all[uid][0]
        except:
            pass
        if p==0 and n==0:
            positive.append(-1)
            negative.append(-1)
            rate.append(avg_rate)
        else:
            positive.append(p)
            negative.append(n)
            rate.append(p/(p+n))            
        
    test_df[f+'_'+label+'_positive_num']=positive
    test_df[f+'_'+label+'_negative_num']=negative  
    test_df[f+'_'+label+'_rate']=rate
    print("test done!")
    del train_df['index']
    print(f+'_'+label+'_positive_num')
    print(f+'_'+label+'_negative_num')
    print(f+'_'+label+'_rate')
    print('avg of positive num',np.mean(train_df[f+'_'+label+'_positive_num']),np.mean(test_df[f+'_'+label+'_positive_num']))
    print('avg of negative num',np.mean(train_df[f+'_'+label+'_negative_num']),np.mean(test_df[f+'_'+label+'_negative_num']))
    print('avg of rate',np.mean(train_df[f+'_'+label+'_rate']),np.mean(test_df[f+'_'+label+'_rate']))

    
    
def w2v(train_df,test_df,f,flag,L):
    print("w2v:",f)
    sentence=[]
    dic={}
    day=0
    for item in train_df[['day','uid',f]].values:
        if day!=item[0]:
            for key in dic:
                sentence.append(dic[key])
            dic={}
            day=item[0]
            print(day)
        try:
            dic[item[1]].append(str(item[2]))
        except:
            dic[item[1]]=[str(item[2])]
    for key in dic:
        sentence.append(dic[key])
    dic={}       
    for item in test_df[['uid',f]].values:
        try:
            dic[item[0]].append(str(item[1]))
        except:
            dic[item[0]]=[str(item[1])]
    for key in dic:
        sentence.append(dic[key])
    print(len(sentence))
    print('training...')
    random.shuffle(sentence)
    if f=='item_id':
        model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=10,iter=50)
    else:
        model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=10,iter=10)
    print('outputing...')
    values=set(train_df[f].values)|set(test_df[f].values)
    w2v=[]
    for v in values:
        a=[v]
        a.extend(model[str(v)])
        w2v.append(a)
    out_df=pd.DataFrame(w2v)
    names=[f]
    for i in range(L):
        names.append(names[0]+'_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' + f +'_'+flag +'_w2v_'+str(L)+'.pkl') 
    
def w2v_1(train_df,test_df,f,flag,L):
    print("w2v:",f)
    sentence=[]
    dic={}
    day=0
    for item in train_df[['day','author_id',f]].values:
        if day!=item[0]:
            for key in dic:
                sentence.append(dic[key])
            dic={}
            day=item[0]
            print(day)
        try:
            dic[item[1]].append(str(item[2]))
        except:
            dic[item[1]]=[str(item[2])]
    for key in dic:
        sentence.append(dic[key])
    dic={}       
    for item in test_df[['author_id',f]].values:
        try:
            dic[item[0]].append(str(item[1]))
        except:
            dic[item[0]]=[str(item[1])]
    for key in dic:
        sentence.append(dic[key])
    print(len(sentence))
    print('training...')
    random.shuffle(sentence)
    model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=10,iter=10)
    print('outputing...')
    values=set(train_df[f].values)|set(test_df[f].values)
    w2v=[]
    for v in values:
        a=[v]
        a.extend(model[str(v)])
        w2v.append(a)
    out_df=pd.DataFrame(w2v)
    names=[f]
    for i in range(L):
        names.append(names[0]+'_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' + f +'_'+flag +'_w2v_'+str(L)+'.pkl') 
    
            
    
def var_mean(train_df,test_df):
    data=train_df[['uid','author_id']].append(test_df[['uid','author_id']])
    group=data[['uid','author_id']].groupby('uid')
    group=group.apply(lambda x:np.var(np.fft.fft(list(Counter(list(x['author_id'])).values()))))
    train_df['uid_num_of_author_fft_var']=train_df['uid'].apply(lambda x:group[x])
    test_df['uid_num_of_author_fft_var']=test_df['uid'].apply(lambda x:group[x])
    print(train_df[train_df['finish']==1]['uid_num_of_author_fft_var'].mean())
    print(train_df[train_df['finish']==0]['uid_num_of_author_fft_var'].mean())

    group=data[['uid','author_id']].groupby('uid')
    group=group.apply(lambda x:np.var(list(Counter(list(x['author_id'])).values())))
    train_df['uid_num_of_author_var']=train_df['uid'].apply(lambda x:group[x])
    test_df['uid_num_of_author_var']=test_df['uid'].apply(lambda x:group[x])
    print(train_df[train_df['finish']==1]['uid_num_of_author_var'].mean())
    print(train_df[train_df['finish']==0]['uid_num_of_author_var'].mean())

    group=data[['uid','author_id']].groupby('uid')
    group=group.apply(lambda x:np.mean(list(Counter(list(x['author_id'])).values())))
    train_df['uid_num_of_author_mean']=train_df['uid'].apply(lambda x:group[x])
    test_df['uid_num_of_author_mean']=test_df['uid'].apply(lambda x:group[x])
    print(train_df[train_df['finish']==1]['uid_num_of_author_mean'].mean())
    print(train_df[train_df['finish']==0]['uid_num_of_author_mean'].mean())
    
    
def did_features(train_df,test_df):
    data=train_df[['uid','did','author_id']].append(test_df[['uid','did','author_id']])

    group=data[['uid','did']].groupby('uid')
    group=group.apply(lambda x: len(set(x['did'])))
    train_df['uid_has_num_of_did']=train_df['uid'].apply(lambda x: group[x])
    test_df['uid_has_num_of_did']=test_df['uid'].apply(lambda x: group[x])

    group=data[['did','author_id']].groupby('did')
    group=group.apply(lambda x: len(set(x['author_id'])))
    train_df['did_has_num_of_author']=train_df['did'].apply(lambda x: group[x])
    test_df['did_has_num_of_author']=test_df['did'].apply(lambda x: group[x])

def author_features(train_df,test_df):
    data=train_df.append(test_df)
    groupby=data[['author_id','item_id']].drop_duplicates().groupby('author_id')
    groupby=groupby.apply(lambda x: len(set(x['item_id'])))
    print('author_include_num_of_item')
    train_df['author_include_num_of_item']=train_df['author_id'].apply(lambda x:groupby[x])
    test_df['author_include_num_of_item']=test_df['author_id'].apply(lambda x:groupby[x])
    print('avg of author_include_num_of_item',np.mean(train_df['author_include_num_of_item']),\
          np.mean(test_df['author_include_num_of_item']))  
    
    groupby=data[['author_id','uid']].drop_duplicates().groupby('author_id')
    groupby=groupby.apply(lambda x: len(set(x['uid'])))
    print('author_include_num_of_uid')
    train_df['author_include_num_of_uid']=train_df['author_id'].apply(lambda x:groupby[x])
    test_df['author_include_num_of_uid']=test_df['author_id'].apply(lambda x:groupby[x])
    print('avg of author_include_num_of_uid',np.mean(train_df['author_include_num_of_uid']),\
          np.mean(test_df['author_include_num_of_uid'])) 
     
def deepwalk(train_df,test_df,f1,f2,flag,L):
    print("deepwalk:",f1,f2)
    dic={}
    for item in train_df[[f1,f2]].values:
        try:
            dic['item_'+str(item[1])].add('user_'+str(item[0]))
        except:
            dic['item_'+str(item[1])]=set(['user_'+str(item[0])])
        try:
            dic['user_'+str(item[0])].add('item_'+str(item[1]))
        except:
            dic['user_'+str(item[0])]=set(['item_'+str(item[1])])

    for item in test_df[[f1,f2]].values:
        try:
            dic['item_'+str(item[1])].add('user_'+str(item[0]))
        except:
            dic['item_'+str(item[1])]=set(['user_'+str(item[0])])
        try:
            dic['user_'+str(item[0])].add('item_'+str(item[1]))
        except:
            dic['user_'+str(item[0])]=set(['item_'+str(item[1])])
    print("creating")        
    path_length=10        
    sentences=[]
    length=[]
    for key in dic:
        sentence=[key]
        while len(sentence)!=path_length:
            key=random.sample(dic[sentence[-1]],1)[0]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)

        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%100000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4,min_count=1,sg=1, workers=10,iter=20)
    print('outputing...')
    
    values=set(train_df[f1].values)|set(test_df[f1].values)
    w2v=[]
    for v in values:
        a=[v]
        a.extend(model['user_'+str(v)])
        w2v.append(a)
    out_df=pd.DataFrame(w2v)
    names=[f1]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    
    values=set(train_df[f2].values)|set(test_df[f2].values)
    w2v=[]
    for v in values:
        a=[v]
        a.extend(model['item_'+str(v)])
        w2v.append(a)
    out_df=pd.DataFrame(w2v)
    names=[f2]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/' +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl') 
    
def get_agg_features(train_df,test_df,f1,f2,agg):
    if type(f1)==str:
        f1=[f1]
    data=train_df[f1+[f2]].append(test_df[f1+[f2]])
    if agg == "count":
        tmp = pd.DataFrame(data.groupby(f1)[f2].count()).reset_index()
    elif agg=="mean":
        tmp = pd.DataFrame(data.groupby(f1)[f2].mean()).reset_index()
    elif agg=="nunique":
        tmp = pd.DataFrame(data.groupby(f1)[f2].nunique()).reset_index()
    elif agg=="max":
        tmp = pd.DataFrame(data.groupby(f1)[f2].max()).reset_index()
    elif agg=="min":
        tmp = pd.DataFrame(data.groupby(f1)[f2].min()).reset_index()
    elif agg=="sum":
        tmp = pd.DataFrame(data.groupby(f1)[f2].sum()).reset_index()
    elif agg=="std":
        tmp = pd.DataFrame(data.groupby(f1)[f2].std()).reset_index()
    elif agg=="median":
        tmp = pd.DataFrame(data.groupby(f1)[f2].median()).reset_index()
    elif agg=="skew":
        tmp = pd.DataFrame(data.groupby(f1)[f2].skew()).reset_index()
    elif agg=="unique_mean":
        group=data.groupby(f1)
        group=group.apply(lambda x:np.mean(list(Counter(list(x[f2])).values())))
        tmp = pd.DataFrame(group.reset_index())
    elif agg=="unique_var":
        group=data.groupby(f1)
        group=group.apply(lambda x:np.var(list(Counter(list(x[f2])).values())))
        tmp = pd.DataFrame(group.reset_index())
    else:
        raise "agg error"
    tmp.columns = f1+['_'.join(f1)+"_"+f2+"_"+agg]
    print('_'.join(f1)+"_"+f2+"_"+agg)
    train_df=train_df.merge(tmp, on=f1, how='left')
    test_df=test_df.merge(tmp, on=f1, how='left')
    del tmp
    del data
    gc.collect()
    print(train_df.shape,test_df.shape)
    return train_df,test_df  
    
    
def title_mean(train_df,test_df):
    temp=[]
    for item in train_df[['title_keys','title_values']].values:
        mean = 0
        if item[0]=="empty":
            temp.append(-1)
        else:
            for t1,t2 in zip(item[0].split(),item[1]):
                mean+=int(t1)*t2
            mean/=sum(item[1])
            temp.append(mean)
    train_df['title_mean']=temp
    temp=[]
    for item in test_df[['title_keys','title_values']].values:
        mean = 0
        if item[0]=="empty":
            temp.append(-1)
        else:
            for t1,t2 in zip(item[0].split(),item[1]):
                mean+=int(t1)*t2
            mean/=sum(item[1])
            temp.append(mean)
    test_df['title_mean']=temp    
    print('title_mean')
    print(train_df['title_mean'].mean(),test_df['title_mean'].mean()) 
    
def titile_kfold_static(train_df,test_df,label):
    print("K-fold static:",'title_'+label)
    #K-fold positive and negative num
    avg_rate=train_df[label].mean()
    num=len(train_df)//5
    index=[0 for i in range(num)]+[1 for i in range(num)]+[2 for i in range(num)]+[3 for i in range(num)]+[4 for i in range(len(train_df)-4*num)]
    random.shuffle(index)
    train_df['index']=index
    dic=[{} for i in range(5)]
    dic_all={}
    for item in train_df[['index','title_keys',label]].values:
        for t in item[1].split():
            try:
                dic[item[0]][t][item[2]]+=1
            except:
                dic[item[0]][t]=[0,0]
                dic[item[0]][t][item[2]]+=1
            try:
                dic_all[t][item[2]]+=1
            except:
                dic_all[t]=[0,0]
                dic_all[t][item[2]]+=1
    print("static done!")
    min_rate=[]
    max_rate=[]
    mean_rate=[]
    for item in train_df[['index','title_keys']].values:
        temp=[]
        for t in item[1].split():
            n=0
            p=0
            for j in range(5):
                if j!=item[0]:
                    try:
                        p+=dic[j][t][1]
                        n+=dic[j][t][0] 
                    except:
                        pass
            if p==0 and n==0:
                temp.append(avg_rate)
            else:
                temp.append(p/(p+n))
        min_rate.append(min(temp))
        max_rate.append(max(temp))
        mean_rate.append(np.mean(temp))

    train_df['title_max_rate_'+label]=max_rate
    train_df['title_min_rate_'+label]=min_rate
    train_df['title_mean_rate_'+label]=mean_rate
    
    print("train done!")
    #for test
    min_rate=[]
    max_rate=[]
    mean_rate=[]
    for item in test_df['title_keys'].values:
        temp=[]
        for t in item.split():
            n=0
            p=0
            for j in range(5):
                try:
                    p+=dic[j][t][1]
                    n+=dic[j][t][0] 
                except:
                    pass
            if p==0 and n==0:
                temp.append(avg_rate)
            else:
                temp.append(p/(p+n))
        min_rate.append(min(temp))
        max_rate.append(max(temp))
        mean_rate.append(np.mean(temp)) 

    test_df['title_max_rate_'+label]=max_rate
    test_df['title_min_rate_'+label]=min_rate
    test_df['title_mean_rate_'+label]=mean_rate
    print("test done!")
    
if __name__ == "__main__":
    for path1,path2,flag in [('data/train_dev.pkl','data/dev.pkl','dev')]:
            print(path1,path2)
            train_df=pd.read_pickle(path1)
            test_df=pd.read_pickle(path2) 
            train_df['ids']=list(range(len(train_df)))
            test_df['ids']=list(range(len(test_df)))
            #word2vec
            w2v(train_df,test_df,'author_id',flag,128)
            w2v(train_df,test_df,'item_id',flag,128)
            w2v_1(train_df,test_df,'uid',flag,128)
            w2v_1(train_df,test_df,'did',flag,128)
            #kfold
            for f in ['item_id','author_id','did']:
                kfold_static(train_df,test_df,f,'finish') 
            #deepwalk
            deepwalk(train_df,test_df,'uid','item_id',flag,64)
            deepwalk(train_df,test_df,'uid','author_id',flag,64)
            deepwalk(train_df,test_df,'did','item_id',flag,64)
            deepwalk(train_df,test_df,'did','author_id',flag,64)
            #aggregate features
            #================================================================================
            combine(train_df,test_df,'author_id','did')
            for f in ['did','author_id','item_id','author_id_did']:
                all_cont(train_df,test_df,f)
                day_cont(train_df,test_df,f)
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"uid","item_id","unique_mean")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","author_id","unique_mean")  
            train_df,test_df=get_agg_features(train_df,test_df,"uid","channel","unique_mean") 
            train_df,test_df=get_agg_features(train_df,test_df,"did","item_id","unique_mean")
            train_df,test_df=get_agg_features(train_df,test_df,"did","author_id","unique_mean")  
            train_df,test_df=get_agg_features(train_df,test_df,"did","channel","unique_mean")
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"uid","item_id","unique_var")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","author_id","unique_var")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","channel","unique_var")
            train_df,test_df=get_agg_features(train_df,test_df,"did","item_id","unique_var")
            train_df,test_df=get_agg_features(train_df,test_df,"did","author_id","unique_var")
            train_df,test_df=get_agg_features(train_df,test_df,"did","channel","unique_var")
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_cont","skew")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_cont","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_cont","std")
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_cont","skew")
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_cont","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_cont","std")
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_cont","skew")
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_cont","mean")
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_cont","std")
            #================================================================================ 
            train_df,test_df=get_agg_features(train_df,test_df,"uid","did","nunique")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","did","count")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","channel","nunique")
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"did","video_duration","min")
            train_df,test_df=get_agg_features(train_df,test_df,"did","video_duration","max") 
            train_df,test_df=get_agg_features(train_df,test_df,"did","video_duration","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"did","video_duration","std")
            train_df,test_df=get_agg_features(train_df,test_df,"channel","video_duration","min")
            train_df,test_df=get_agg_features(train_df,test_df,"channel","video_duration","max") 
            train_df,test_df=get_agg_features(train_df,test_df,"channel","video_duration","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"channel","video_duration","std")
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"item_id","uid","nunique")
            train_df,test_df=get_agg_features(train_df,test_df,"item_id","uid","count")        
            #================================================================================
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","item_id","nunique")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","item_id","count")        
            #================================================================================  
            train_df,test_df=get_agg_features(train_df,test_df,"uid","user_city","nunique")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","author_id","nunique")        
            #================================================================================          
            train_df,test_df=get_agg_features(train_df,test_df,"channel","user_city","nunique")          
            #================================================================================         
            train_df,test_df=get_agg_features(train_df,test_df,"did","video_duration","skew")
            train_df,test_df=get_agg_features(train_df,test_df,"channel","video_duration","skew")        
            #================================================================================
            title_mean(train_df,test_df)
            train_df,test_df=get_agg_features(train_df,test_df,"uid","title_mean","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","title_mean","std")
            train_df,test_df=get_agg_features(train_df,test_df,"uid","title_mean","skew")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_mean","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_mean","std")
            train_df,test_df=get_agg_features(train_df,test_df,"author_id","title_mean","skew")  
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_mean","mean")
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_mean","std")
            train_df,test_df=get_agg_features(train_df,test_df,"did","title_mean","skew") 
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_mean","mean")
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_mean","std")
            train_df,test_df=get_agg_features(train_df,test_df,["uid","channel"],"title_mean","skew") 
            #================================================================================
            titile_kfold_static(train_df,test_df,'finish')
            titile_kfold_static(train_df,test_df,'like')
            var_mean(train_df,test_df)
            author_features(train_df,test_df)
            did_features(train_df,test_df) 
            #================================================================================
            
            print(train_df.shape,test_df.shape)
            print(list(train_df))
            train_df.to_pickle(path1) 
            test_df.to_pickle(path2)  
            print("*"*80)
            print("done!")

    
    