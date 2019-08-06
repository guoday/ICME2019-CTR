import numpy as np
import pandas as pd
import ctrNet
import tensorflow as tf
from src import misc_utils as utils
import os
import gc
from sklearn import metrics
import random
np.random.seed(2019)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

#################################################################################################
label='finish'
single_features=['did_finish_positive_num', 'did_finish_negative_num','did_all_cont','did_day_cont','author_id_finish_positive_num', 'author_id_finish_negative_num',
                'author_id_all_cont','author_id_day_cont','item_id_finish_positive_num', 'item_id_finish_negative_num',
                'item_id_all_cont','item_id_day_cont','author_id_did_all_cont','author_include_num_of_item','author_include_num_of_uid',
                'title_cont','uid_has_num_of_did','uid_channel_nunique','did_video_duration_max','channel_video_duration_min',
                'item_id_uid_nunique']
cross_features=['uid','user_city','item_id','author_id','item_city','channel','music_id','did','video_duration','did_day_cont','author_id_did_all_cont']
multi_features=None
multi_weights=None
dense_features=['audio_embedding_'+str(i) for i in range(64)]+['video_embedding_'+str(i) for i in range(32)]+['author_id_embedding_128_'+str(i) for i in range(128)]+['uid_embedding_128_'+str(i) for i in range(128)]+['did_embedding_128_'+str(i) for i in range(128)]
dense_features+=['did_finish_rate','author_id_finish_rate','item_id_finish_rate', 'title_mean_rate_finish','title_max_rate_finish','title_min_rate_finish','beauty_min','title_mean',  'uid_num_of_author_mean','uid_num_of_author_var','uid_num_of_author_fft_var','uid_channel_title_mean_mean', 'uid_channel_title_mean_std', 'uid_channel_title_mean_skew']
for f1,f2 in [('uid','item_id'),('uid','author_id'),('did','item_id'),('did','author_id')]:
    for i in range(64):
        dense_features.append(f1+'_'+ f2+'_'+f1+'_deepwalk_embedding_64_'+str(i))
        dense_features.append(f1+'_'+ f2+'_'+f2+'_deepwalk_embedding_'+str(i))  
kv_features=['uid_did_nunique', 'uid_did_count', 'uid_channel_nunique', 'did_video_duration_min', 
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
                        'channel_video_duration_skew',  'title_mean', 'uid_title_mean_mean', 
                        'uid_title_mean_std', 'uid_title_mean_skew', 'author_id_title_mean_mean', 
                        'author_id_title_mean_std', 'author_id_title_mean_skew', 'did_title_mean_mean', 
                        'did_title_mean_std', 'did_title_mean_skew']
#################################################################################################
hparam=tf.contrib.training.HParams(
            model='CIN',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[1024,512],
            dense_hidden_size=[300],
            cross_layer_sizes=[128,128],
            k=16,
            single_k=16,
            cross_hash_num=int(5e6),
            single_hash_num=int(5e6),
            multi_hash_num=int(1e6),
            batch_size=4096,
            infer_batch_size=2**14,
            optimizer="adam",
            dropout=0,
            kv_batch_num=10,
            learning_rate=0.0002,
            num_display_steps=100,
            epoch=1, #don't modify
            metric='auc',
            activation=['relu','relu','relu'],
            init_method='tnormal',
            cross_activation='relu',
            init_value=0.001,
            single_features=single_features,
            cross_features=cross_features,
            multi_features=multi_features,
            multi_weights=multi_weights,
            dense_features=dense_features,
            kv_features=kv_features,
            model_name="xdeepfm",
            test=True,
            label=label)
utils.print_hparams(hparam)

#################################################################################################
model=ctrNet.build_model(hparam)
test=pd.read_pickle('data/dev_NN.pkl')
k=10
for i in range(k):
    print(i)
    train=pd.read_pickle('data/train_dev_NN_'+str(i)+'.pkl')
    model.train(train,test)    
#################################################################################################    
print("Dev Inference:")
preds=model.infer(test)
test[hparam.label+'_probability']=preds.round(6) 
fpr, tpr, thresholds = metrics.roc_curve(test[hparam.label]+1, test[hparam.label+'_probability'], pos_label=2)
auc=metrics.auc(fpr, tpr)
print(hparam.label,round(auc,6))
score=str(round(auc,6))
del train
del test
gc.collect()
