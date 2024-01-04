from collections import defaultdict
import os
from tqdm import tqdm



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
save_file_dir='online_test_img2'
txt_name='online_data_test2_result_iteration13.txt'
with open(txt_name,'r') as f:
    for line in tqdm(f.readlines()):
        num=line.strip().split(',')[0]
        img=line.strip().split(',')[1]
        os.rename(os.path.join(save_file_dir,img),os.path.join(save_file_dir,num+'_'+img))





# 将online_test_img中的图片名字前面的数字去掉
# for img in tqdm(os.listdir('online_test_img')):
#     os.rename(os.path.join('online_test_img',img),os.path.join('online_test_img',img.split('_')[1]))




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




# 将新随机sample的imgurl的txt文件中的每一行url利用request获取到图片并保存到online_test_img2中
# import requests as req
# from PIL import Image
# from io import BytesIO
# import os
# from tqdm import tqdm
# with open('sample500.txt','r') as f:
#     with open('test2.txt','w') as out:
#         for line in tqdm(f.readlines()):
#             url=line.strip()
#             img_name=url.split('/')[-1].split('.')[0]+'.jpg'
#             img_path=os.path.join('online_test_img2',img_name)
#             if not os.path.exists(img_path):
#                 try:
#                     response = req.get(url)
#                     image=Image.open(BytesIO(response.content)).convert('RGB')
#                     image.save(img_path)
#                     out.write(f'{img_name}\n')
#                 except Exception as e:
#                     print('error :',e)