from collections import defaultdict
import os
from tqdm import tqdm

from sklearn.metrics import roc_auc_score,auc,average_precision_score,roc_curve

# 全部写入all.txt
# with open('all_compare_result/all.txt','w') as out:
#     for txt in os.listdir('compare_result_adjust_train'):
#         file_path=os.path.join('compare_result_adjust_train',txt)
#         with open(file_path,'r') as rd:
#             for pairs_and_labels in rd.readlines():
#                 img0,img1,label=pairs_and_labels.split(',')
#                 label=label.strip()
#                 out.write(f'{img0},{img1},{label},{txt}\n')





# 将修改过label的数据统一写入到upload_deal.txt,upload-->upload_deal
# with open('upload_deal.txt','w') as out:
#     with open('upload.txt','r') as f:
#         for line in f.readlines():
#             img0,img1,label,pred,remark=line.strip().split(',')
#             if remark!='-2':
#                 out.write(f'{img0},{img1},{label},{pred},{remark}\n')




# 将upload_deal.txt中的数据写入到compare_result中
# upload=[]
# with open('upload_deal.txt','r') as f:
#     upload=f.readlines()

# change_count=0
# for txt in os.listdir('compare_result'):
#     file_path=os.path.join('compare_result',txt)
#     new_content=[]
#     with open(file_path,'r') as rd:
#         for pairs_and_labels in rd.readlines():
#             img0,img1,label=pairs_and_labels.strip().split(',')
#             for up in upload:
#                 img0_up,img1_up,label_up,pred_up,remark_up=up.strip().split(',')
#                 if img0==img0_up and img1==img1_up:
#                     change_count+=1
#                     label=remark_up
#             new_content.append(f'{img0},{img1},{label}\n')
#     with open(os.path.join('new_compare',txt),'w') as out:
#         out.writelines(new_content)
# print(change_count)





# 将online_data_test_result.txt中每一行逗号前面的数字加到online_test_img中对应在逗号后面的图片名字的前面
# save_file_dir='url_100000_img'
# txt_name='url_100000_result.txt'
# with open(txt_name,'r') as f:
#     for line in tqdm(f.readlines()):
#         num=line.strip().split(',')[0]
#         img=line.strip().split(',')[1]
#         os.rename(os.path.join(save_file_dir,img),os.path.join(save_file_dir,num+'_'+img))





# 将online_test_img中的图片名字前面的数字去掉
# for img in tqdm(os.listdir('online_test_img')):
#     os.rename(os.path.join('online_test_img',img),os.path.join('online_test_img',img.split('_')[1]))


# 从csv文件中读取分数和图片名字，将分数最小的前4000的图片的分数加到url_100000_img下对应图片的前面，然后将这4000张图片移动到url_nums4000_img_score中




# 找出compare_result_adjust_train中的最长序列
# max_len=6
# global_longest_road=[]
# dict_road=defaultdict(list)
# for txt in os.listdir('compare_result_adjust_train'):
#     file_path=os.path.join('compare_result_adjust_train',txt)
#     with open(file_path,'r') as rd:
#         for pairs_and_labels in rd.readlines():
#             img0,img1,label=pairs_and_labels.strip().split(',')
#             if label=='-1':
#                 dict_road[img0].append(img1)
#             elif label=='1':
#                 dict_road[img1].append(img0)
# print(len(dict_road))
# road=[]
# def dfs(dict_road,road:list,start):
#         road.append(start)
#         next_node_list=dict_road[start]
#         if len(next_node_list)==0:
#             global max_len
#             global global_longest_road
#             if len(road)==max_len:
#                 max_len=len(road)
#                 global_longest_road.append(road.copy())
#             road.pop()
#             return
#         for node in next_node_list:
#             if node not in road:
#                 dfs(dict_road,road,node)
#         road.pop()
# count=0
# for start in list(dict_road.keys()):
#     count+=1
#     dfs(dict_road,road,start)
# print('count:',count)   
# print('max_len:',max_len)
# print(global_longest_road)


# ------------------------------从这里开始是将url_100000_img中的图片进行处理去交给gpt标记--------------------------------

# 将新随机sample的imgurl的txt文件中的每一行url利用request获取到图片并保存到online_test_img2中
# import requests as req
# from PIL import Image
# from io import BytesIO
# import os
# from tqdm import tqdm
# start_line=0
# end_line=2
# print(f'start_line:{start_line},end_line:{end_line}')
# with open('url_270000_url_ongly.txt','r') as f:
#     # with open('url_100000.txt','w') as out:
#         for line in tqdm(f.readlines()[15000:20000]):
#             url=line.strip()
#             # url=url.split('\t')[0]
#             img_name=url.split('/')[-1].split('.')[0]+'.jpg'
#             img_path=os.path.join('url_270000_img_1w-2w',img_name)
#             if not os.path.exists(img_path):
#                 try:
#                     response = req.get(url)
#                     image=Image.open(BytesIO(response.content)).convert('RGB')
#                     image.save(img_path)
#                     # out.write(f'{img_name}\n')
#                 except Exception as e:
#                     print('error :',e)

# 将某个img文件夹中的图片名字写入到txt文件中
# img_dir='url_270000_img_1w-2w'
# txt='url_270000/url_270000_img_1w-2w.txt'
# print(len(os.listdir(img_dir)))
# with open(txt,'w') as out:
#     for img in tqdm(os.listdir(img_dir)):
#         out.write(f'{img}\n')

# 预测完之后，将txt直接读取为csv文件，并进行处理
# import pandas as pd
# url_prefix='https://img-s-msn-com.akamaized.net/tenant/amp/entityid/'
# file='url_270000_img_1w-2w.txt'
# df=pd.read_csv(file,header=None,names=['img_name','pred'])
# df['img_name']=df['img_name'].apply(lambda x:url_prefix+x.split('.')[0]+'.img')
# df=df.sort_values(by='pred',ascending=True)
# print(df.head())
# df=df['img_name']
# df.to_csv('url_270000_img_1w-2w_after_sort.txt',index=False,header=False)



# -------------------------------------在这里结束-------------------------------------------------


# 将两个txt文件的信息进行融合
info_dict={}
with open('model_scope_iter2_single_img.txt','r') as one:
    for content in one.readlines():
        content=content.strip()
        img_path,pred=content.split(',')
        img_name=img_path.split('.')[0]
        # img_name=img_path.split('/')[-1].split('.')[0]
        info_dict[img_name]=1-float(pred)
with open('test_blur_iter2_single_gptvscore.txt','r') as two:
    with open('model_scope_merge.txt','w') as out:
        for content in two.readlines():
            content=content.strip()
            img_url,gtpv_score=content.split('\t')
            img_name=img_url.split('/')[-1].split('.')[0]
            if img_name in info_dict.keys():
                pred=info_dict[img_name]
            else:
                print(img_url)
                pred,label='',''
            out.write(f'{img_url}\t{gtpv_score}\t{str(pred)}\n')

# 分析txt文件中的auc值
# from  sklearn.metrics import precision_recall_curve,precision_score
# from matplotlib import pyplot as plt

# file='test_blur_iter2_merge.txt'
# labels=[]
# logits=[]
# with open(file,'r') as f:
#     for line in f.readlines():
#         img_url,gtpv_score,label,pred=line.strip().split('\t')
#         if float(gtpv_score)>20.0:
#             fake_label=1
#         else:
#             fake_label=0
#         labels.append(fake_label)
#         logits.append(round(float(pred),5))
# # # 计算Precision-Recall Curve中的precision、recall和阈值
# precision, recall, thresholds = precision_recall_curve(labels, logits)

# # 计算平均精度（Average Precision）
# # avg_precision = average_precision_score(labels, logits)
# roc_auc=auc(recall,precision)

# # # 绘制PR Curve
# # plt.figure(figsize=(8, 6))
# # plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP={avg_precision:.2f})')
# # plt.xlabel('Recall')
# # plt.ylabel('Precision')
# # plt.title('Precision-Recall Curve')
# # plt.legend(loc='best')
# # plt.savefig('test.jpg')

# roc_auc=roc_auc_score(labels,logits)
# avg_precision=average_precision_score(labels,logits)
# print("ROC AUC Score:", roc_auc)
# print("Average Precision Score:", avg_precision)

