import sklearn.decomposition as sk_decomposition
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

## SVD to 
for file,dim in [('data/video_w2v.pkl',32),('data/audio_w2v.pkl',64)]:
    if file[-3:]=="pkl":
        df=pd.read_pickle(file)
    else:
        df=pd.read_csv(file)
    pca = sk_decomposition.PCA(n_components=dim,whiten=False,svd_solver='auto')
    pca.fit(df[df.columns[1:]])
    df1=pd.DataFrame(pca.transform(df[df.columns[1:]]))
    df1.columns=df.columns[1:dim+1]
    df1[df.columns[0]]=df[df.columns[0]].values
    df1.to_pickle(file[:-4]+'_svd_'+str(dim)+'.pkl')



    


def norm(train_df,test_df,features):   
    df=pd.concat([train_df,test_df])[features]
    scaler = preprocessing.QuantileTransformer(random_state=0)
    scaler.fit(df[features]) 
    train_df[features]=scaler.transform(train_df[features])
    test_df[features]=scaler.transform(test_df[features])


    
for path1,path2,flag in [('data/train_dev.pkl','data/dev.pkl','dev')]:
        print(path1,path2)
        train_df=pd.read_pickle(path1)
        test_df=pd.read_pickle(path2)
        print(train_df.shape,test_df.shape)
        float_features=['uid_did_nunique', 'uid_did_count', 'uid_channel_nunique', 'did_video_duration_min', 
                        'did_video_duration_max', 'did_video_duration_mean', 'did_video_duration_std', 
                        'channel_video_duration_min', 'channel_video_duration_max', 'channel_video_duration_mean', 
                        'channel_video_duration_std', 'uid_item_id_unique_mean', 'uid_author_id_unique_mean', 
                        'uid_channel_unique_mean', 'did_item_id_unique_mean', 'did_author_id_unique_mean', 
                        'did_channel_unique_mean', 'uid_item_id_unique_var', 'uid_author_id_unique_var', 
                        'uid_channel_unique_var', 'did_item_id_unique_var', 'did_author_id_unique_var', 
                        'did_channel_unique_var', 'author_id_title_cont_skew', 'author_id_title_cont_mean', 
                        'author_id_title_cont_std', 'did_title_cont_skew', 'did_title_cont_mean', 
                        'did_title_cont_std', 'uid_channel_title_cont_skew', 'uid_channel_title_cont_mean', 
                        'uid_channel_title_cont_std', 'item_id_uid_nunique', 'item_id_uid_count', 
                        'author_id_item_id_nunique', 'author_id_item_id_count', 'uid_user_city_nunique', 
                        'uid_author_id_nunique', 'channel_user_city_nunique', 'did_video_duration_skew', 
                        'channel_video_duration_skew', 'title_mean', 'uid_title_mean_mean', 
                        'uid_title_mean_std', 'uid_title_mean_skew', 'author_id_title_mean_mean', 
                        'author_id_title_mean_std', 'author_id_title_mean_skew', 'did_title_mean_mean', 
                        'did_title_mean_std', 'did_title_mean_skew', 'uid_channel_title_mean_mean', 
                        'uid_channel_title_mean_std', 'uid_channel_title_mean_skew',
                        'uid_num_of_author_mean','uid_num_of_author_var','uid_num_of_author_fft_var',]
        train_df=train_df.fillna(-1)
        test_df=test_df.fillna(-1)
        norm(train_df,test_df,float_features)
        print(train_df[float_features])
        
        k=10
        train_df=train_df.sample(frac=1)
        test_df=test_df.sample(frac=0.1)
        train=[(path2[:-4]+'_NN.pkl',test_df)]
        for i in range(k):
            train.append((path1[:-4]+'_NN_'+str(i)+'.pkl',train_df.iloc[int(i/k*len(train_df)):int((i+1)/k*len(train_df))]))
        del train_df
        gc.collect()
        for file,temp in train:
            print(file,temp.shape)
            for f1,f2 in [('uid','item_id'),('uid','author_id'),('did','item_id'),('did','author_id')]:
                col=f1
                df = pd.read_pickle( 'data/' +f1+'_'+ f2+'_'+col +'_'+flag +'_deepwalk_64.pkl')
                df = df.drop_duplicates([col])
                fs = list(df)
                fs.remove(col)
                temp = pd.merge(temp, df, on=col, how='left')
                print(temp.shape) 
                col=f2
                df = pd.read_pickle( 'data/' +f1+'_'+ f2+'_'+col +'_'+flag +'_deepwalk_64.pkl')
                df = df.drop_duplicates([col])
                fs = list(df)
                fs.remove(col)
                temp = pd.merge(temp, df, on=col, how='left')
                print(temp.shape)                 
            print("done 1!")
            
            for col in ['video','audio','author_id','uid','did','item_id']:
                if col in ['audio']:
                    df = pd.read_pickle( 'data/' + col + '_w2v_svd_64.pkl')
                    col='item_id'
                elif col in ['video']:
                    df = pd.read_pickle( 'data/' + col + '_w2v_svd_32.pkl')
                    col='item_id'    
                else:
                    df = pd.read_pickle( 'data/' + col + '_'+flag+'_w2v_128.pkl')
                df = df.drop_duplicates([col])
                fs = list(df)
                fs.remove(col)    
                print(temp.shape)
                temp = pd.merge(temp, df, on=col, how='left')
                print(temp.shape) 
            print("done 2!")
            temp=temp.fillna(0)           
            temp.to_pickle(file)
            del temp
            gc.collect()
        


            
    