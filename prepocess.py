import tqdm 
import pandas as pd
import numpy as np
import random 
import json
import os
import pickle
import gc
from collections import Counter
import math
from sklearn.model_selection import train_test_split

def split_data(reprepocess=False):
    #split data, (train,test) for inference, (train_dev,dev) for dev
    if reprepocess:
        train_df =pd.read_csv('data/final_track2_train.txt', sep='\t', names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'create_time', 'video_duration'])       
        day=[i*7//len(train_df) for i in range(len(train_df))]
        train_df['day']=day
        train_dev_df=train_df[train_df['day']<=4]   
        dev_df=train_df[train_df['day']==6]
        return (train_dev_df,dev_df)
    else:
        train_dev_df=pd.read_pickle('data/train_dev.pkl')
        dev_df=pd.read_pickle('data/dev.pkl')
        return (train_dev_df,dev_df)


def parsing_item_title_features():
    if os.path.exists("data/item_title_features.pkl"):
        print("Reload title features")
        return pickle.load(open('data/item_title_features.pkl','rb'))
    print("process title features")
    item_dict={}
    cont=0
    with open("data/track2_title.txt",'r') as f:
        for line in f:
            line=json.loads(line.strip())
            item_id=int(line['item_id'])
            if item_id not in item_dict:
                item_dict[item_id]={}
            keys=list(line['title_features'].keys())
            values=[line['title_features'][x] for x in keys]
            if len(keys)==0:
                keys=['empty']
                values=[1]
            item_dict[item_id]['title_keys']=' '.join(keys)           
            item_dict[item_id]['title_values']=values
    pickle.dump(item_dict,open('data/item_title_features.pkl','wb'))
    
    return item_dict

def parsing_item_face_features():
    if os.path.exists("data/item_face_features.pkl"):
        print("Reload face features")
        return pickle.load(open('data/item_face_features.pkl','rb'))
    print("process face features")
    item_dict={}
    with open("data/track2_face_attrs.txt",'r') as f:
        for line in f:
            line=json.loads(line.strip())
            item_id=int(line['item_id'])
            if item_id not in item_dict:
                item_dict[item_id]={}
            people=len(line["face_attrs"])
            if people==0:
                continue
            gender=[]
            beauty=[]
            for item in line["face_attrs"]:
                gender.append(item['gender'])
                beauty.append(item["beauty"])
            item_dict[item_id]['gender']=gender
            item_dict[item_id]['beauty']=beauty
    pickle.dump(item_dict,open('data/item_face_features.pkl','wb'))
    
    return item_dict                    


def create_titile_features(df,Title_dict):
    print("Title features")
    if 'title_cont' not in list(df):
        item_ids=df['item_id'].values
        keys=[]
        values=[]
        hit=[]
        cont=[]
        for idx in item_ids:
            try:
                keys.append(Title_dict[idx]['title_keys'])
                values.append(Title_dict[idx]['title_values'])
                hit.append(True)
                cont.append(sum(Title_dict[idx]['title_values']))
            except:
                keys.append('empty')
                values.append([1])
                cont.append(0)
                hit.append(False)   
        print("Hit rate",round(np.mean(hit)*100,3))
        df['title_keys']=keys
        df['title_values']=values
        df['title_cont']=cont
    return df
    
def create_face_features(df,face_dict):
    print("Face features")
    if 'gender_cont' not in list(df):
        item_ids=df['item_id'].values
        gender_cont=[]
        gender_0=[]
        gender_1=[]
        beauty_max=[]
        beauty_min=[]
        beauty_mean=[]
        beauty_var=[]
        beauty_fft_var=[]
        hit=[]
        for idx in item_ids:
            try:
                beauty_min.append(min(face_dict[idx]['beauty']))
                hit.append(True)  
            except:
                beauty_min.append(-1)
                hit.append(False)  
        print("Hit rate",round(np.mean(hit)*100,3))
        df['beauty_min']=beauty_min
    return df    



def audio_w2v():
    print("process audio features")
    w2v=[]

    with open("data/track2_audio_features.txt",'r') as f:
        for line in f:
            line=json.loads(line.strip())
            item_id=int(line['item_id'])
            if len(line['audio_feature_128_dim'])==128:
                a=[item_id]
                a.extend(line['audio_feature_128_dim'])
                w2v.append(a)

    out_df=pd.DataFrame(w2v)
    names=['item_id']
    for i in range(128):
        names.append('audio_embedding_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/audio_w2v.pkl')

    
def video_w2v():
    print("process video features")
    w2v=[]

    with open("data/track2_video_features.txt",'r') as f:
        for line in f:
            line=json.loads(line.strip())
            item_id=int(line['item_id'])
            if len(line['video_feature_dim_128'])==128:
                a=[item_id]
                a.extend(line['video_feature_dim_128'])
                w2v.append(a)

    out_df=pd.DataFrame(w2v)
    names=['item_id']
    for i in range(128):
        names.append('video_embedding_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('data/video_w2v.pkl')   
    
        
if __name__ == "__main__":
    dev_group=split_data(True)
    Title_dict=parsing_item_title_features() 
    Face_dict=parsing_item_face_features() 
    video_w2v()
    audio_w2v()
    
    for train_df,test_df,path1,path2 in [dev_group+('data/train_dev.pkl','data/dev.pkl')]:
        print(train_df.shape,test_df.shape)
        for df in [train_df,test_df]:
            create_titile_features(df,Title_dict)
            create_face_features(df,Face_dict)
        print(train_df.shape,test_df.shape)
        train_df.to_pickle(path1) 
        test_df.to_pickle(path2) 
        del train_df
        del test_df
        gc.collect()

    
 