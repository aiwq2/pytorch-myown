from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import time
import cv2
from extract_features.blurry import *
from extract_features.NRSS import *
import traceback
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, f1_score,roc_auc_score,average_precision_score,precision_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb
import itertools
from multiprocessing import Pool
from functools import partial
from joblib import Parallel, delayed
from itertools import combinations
from sklearn import svm
from sklearn.decomposition import PCA


def get_fetures(iamge_path):
    img=cv2.imread(iamge_path,0).astype(np.int64)
    height,width=img.shape
    features=[]
    canny_feature=canny(img)
    laplacian_feature=variance_of_laplacian(img)
    fft_feature=fourier_transform(img)
    entropy_feature=image_entropy(img)
    saturation_feature=color_saturation(iamge_path)
    brightness_feature=image_brightness(iamge_path)
    # SMD_feature=SMD(img)
    SMD2_feature=SMD2(img)
    variance_feature=variance(img)
    energy_feature=energy(img)
    nrss_feature=NRSS(iamge_path)
    brenner_feature=brenner(img)
    Vollath_feature=Vollath(img)
    svd_blur_degree=get_svd_blur_degree(img)

    # features.append([float(canny_feature),float(laplacian_feature),float(fft_feature),
    #                 float(entropy_feature),float(saturation_feature),float(brightness_feature),
    #                 float(SMD_feature),float(SMD2_feature),float(brenner_feature),float(variance_feature),
    #                 float(energy_feature),float(Vollath_feature)
    #                 ])

    features.append([float(canny_feature),float(laplacian_feature),float(fft_feature),float(entropy_feature),
                    float(saturation_feature),float(brightness_feature),float(SMD2_feature),
                    float(variance_feature),float(energy_feature),float(nrss_feature),float(height),float(width),
                    float(brenner_feature),float(Vollath_feature),float(svd_blur_degree)
                    ])

    return features

def write_features_to_txt(f,features,image_path,label=None):
    if label is not None:
        if label.startswith(normal_img_label):
            label_write_in=0
        elif label.startswith(blurry_img_label):
            label_write_in=1
        elif label.startswith(mid_img_label):
            label_write_in=2
        f.write(f'{image_path},{label_write_in}\t')
    else:
        f.write(f'{image_path}\t')
    for index,feature in enumerate(features):
        if index!=len(features)-1:
            f.write(f'{feature:.2f},')
        else:
            f.write(f'{feature:.2f}\n')

def extract_feature_to_txt(train_dir,test_dir,increment_train_dir,train_txt_save_path,text_txt_save_path,increment_train_txt_save_path,is_add):

    if is_add:
        count=0
        total_images = sum(len(files) for _, _, files in os.walk(increment_train_dir))
        def process_image(image_path):
            try:
                features = get_fetures(image_path)
                return features[0], image_path
            except Exception as e:
                print(e)
                return None
        def write_to_txt(f,label, result):
            try:
                nonlocal count
                features, image_path = result
                write_features_to_txt(f, features, image_path, label)
                count += 1
                pbar.update(1)
            except Exception as e:
                print(e)
                return None
        # with open(increment_train_txt_save_path,'w') as f:
        #     for label in os.listdir(increment_train_dir):
        #         label_dir=os.path.join(increment_train_dir,label)
        #         for img in tqdm(os.listdir(label_dir),desc='increment_train_dir'):
        #             try:
        #                 image_path=os.path.join(label_dir,img)
        #                 features=get_fetures(image_path)
        #                 write_features_to_txt(f,features[0],image_path,label)
        #             except Exception as e:
        #                 print(e)
        with open(increment_train_txt_save_path, 'w') as f:
            with ThreadPoolExecutor(max_workers=4) as executor:  # 设置最大线程数
                pbar = tqdm(total=total_images, desc='Progress')  # 创建进度条
                for label in os.listdir(increment_train_dir):
                    label_dir = os.path.join(increment_train_dir, label)
                    for img in tqdm(os.listdir(label_dir), desc='increment_train_dir'):
                        image_path = os.path.join(label_dir, img)
                        future = executor.submit(process_image, image_path)
                        future.add_done_callback(lambda x: write_to_txt(f,label, x.result()))
    else:
        # with open(train_txt_save_path,'w') as f:
        #     for label in os.listdir(train_dir)[:1000]:
        #         label_dir=os.path.join(train_dir,label)
        #         for img in tqdm(os.listdir(label_dir),desc='train_dir'):
        #             try:
        #                 image_path=os.path.join(label_dir,img)
        #                 features=get_fetures(image_path)
        #                 write_features_to_txt(f,features[0],image_path,label)
        #             except Exception as e:
        #                 print(e)


        with open(text_txt_save_path,'w') as f:
            for label in os.listdir(test_dir):
                label_dir=os.path.join(test_dir,label)
                for img in tqdm(os.listdir(label_dir),desc='test_dir'):
                    try:
                        image_path=os.path.join(label_dir,img)
                        features=get_fetures(image_path)
                        write_features_to_txt(f,features[0],image_path,label)
                    except Exception as e:
                        print(e)
         
       
def extract_feature_to_txt_inference(inference_dir,inference_txt_save_path):
    with open(inference_txt_save_path,'w') as f:
        for img in tqdm(os.listdir(inference_dir),desc='inference_dir'):
            try:
                image_path=os.path.join(inference_dir,img)
                features=get_fetures(image_path)
                write_features_to_txt(f,features[0],image_path)
            except Exception as e:
                print(e)

def use_GBDT(is_adjustParam,train_txt_save_path,test_txt_save_path,inference_txt_save_path,model_path,predict_result_path,is_train,normal_img_label,analyze,threshold):
    X_train=[]
    y_train=[]
    img_train=[]
    X_test=[]
    y_test=[]
    img_test=[]
    if is_train:
        with open(train_txt_save_path,'r') as f:
            for content in f.readlines():
                content=content.strip()
                img_path,label=content.split('\t')[0].split(',')
                features=content.split('\t')[1].split(',')
                # features[-1]=0
                X_train.append([*features])
                if label=='0':
                    y_train.append(0)
                elif label=='1':
                    y_train.append(1)
                elif label=='2':
                    y_train.append(2)
                img_train.append(img_path)
        

        with open(test_txt_save_path,'r') as f:
            for content in f.readlines():
                content=content.strip()
                img_path,label=content.split('\t')[0].split(',')
                features=content.split('\t')[1].split(',')
                # features[-1]=0
                X_test.append([*features])
                if label=='0':
                    y_test.append(0)
                elif label=='1':
                    y_test.append(1)
                elif label=='2':
                    y_test.append(2)
                img_test.append(img_path)

    else:
        with open(inference_txt_save_path,'r') as f:
            for content in f.readlines():
                content=content.strip()
                img_path=content.split('\t')[0]
                features=content.split('\t')[1].split(',')
                X_test.append([*features])
                img_test.append(img_path)

    # 定义参数网格
    param_grid = {
        'n_estimators': range(50,200,10),
        'learning_rate':[0.001,0.01,0.1,0.5,0.05],
        'max_depth':[3,4,5,6],
        'subsample':[0.6,0.8,1],
    }

    best_n=-1
    best_l=-1
    best_d=-1
    best_sub=-1
    best_split=-1
    best_leaf=-1
    best_accu=-1
    best_p_score=-1
    best_auc=-1

    # 初始化梯度提升分类器模型
    if not is_train:
        with open(model_path,'rb') as f:
            gbdt=pickle.load(f)
        print(f'model loaded from {model_path}')
    else:
    # 调参
    # 交叉验证去除不好的数据
    # --------------------------------------------------------------------------------
        def model_cv_train(model, model_name,X_train,label, kfold=5):
            oof_preds = np.zeros((X_train.shape[0]))
            test_preds = np.zeros(X_train.shape[0])
            skf = StratifiedKFold(n_splits=kfold)
            print(f"Model = {model_name}")
            for k, (train_index, test_index) in tqdm(enumerate(skf.split(X_train, label)),total=kfold):
                x_train, x_test = X_train[train_index], X_train[test_index]
                y_train, y_test = label[train_index], label[test_index]

                model.fit(x_train,y_train)

                y_pred = model.predict_proba(x_test)[:,1]
                oof_preds[test_index] = y_pred.ravel()
                auc = roc_auc_score(y_test,y_pred)
                print(f'test_index:{test_index}')
                print("- KFold = %d, val_auc = %.4f" % (k, auc))
                # test_fold_preds = model.predict_proba(y_train)[:, 1]
                # test_preds += test_fold_preds.ravel()
            print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
            return test_preds / kfold
    # 多进程调参，速度更快
        if is_adjustParam:
            def evaluate_params(params):
                n_est, lear, max_dep, sub= params['n_estimators'], params['learning_rate'], params['max_depth'], params['subsample']
                gbdt = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lear, max_depth=max_dep, subsample=sub, random_state=42)
                gbdt.fit(X_train, y_train)
                y_pred=gbdt.predict(X_test)
                y_prob = gbdt.predict_proba(X_test)[:,1]
                y_pred=np.where(y_prob>=threshold,1,0)
                # roc_auc = roc_auc_score(y_test, y_prob)
                roc_auc=f1_score(y_test,y_pred)
                return roc_auc, n_est, lear, max_dep, sub
            grid = ParameterGrid(param_grid)
            start_time=time.time()
            print(f'start time:{start_time}')
            results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in tqdm(grid))
            for roc_auc, n_est, lear, max_dep, sub in results:
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_n = n_est
                    best_l = lear
                    best_d = max_dep
                    best_sub = sub
            end_time=time.time()
            print('time used: ',end_time-start_time)
            print(f'best f1 score:{best_auc},best n:{best_n},best l:{best_l},best d:{best_d},best sub:{best_sub}')
            return

        # --------------------------------------------------------------------------------
        # 单进程跑调参，速度稍微有点慢
        # for n_est in tqdm(param_grid['n_estimators']):
        #     for lear in tqdm(param_grid['learning_rate']):
        #         for max_dep in param_grid['max_depth']:
        #             for sub in param_grid['subsample']:
        #                 gbdt = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lear, max_depth=max_dep, subsample=sub,random_state=42)
        #                 # X_train=scaler.fit_transform(X_train)
        #                 gbdt.fit(X_train, y_train)

        #                 # 使用训练好的模型进行预测
        #                 # X_test=scaler.transform(X_test)
        #                 y_pred = gbdt.predict(X_test)
        #                 y_prob=gbdt.predict_proba(X_test)[:,1]

        #                 roc_auc=roc_auc_score(y_test,y_prob)
        #                 # p_score=precision_score(y_test,y_pred,zero_division=np.nan)
        #                 # print(f'y_test:{y_test}')
        #                 # print(f'y_pred:{y_pred}')
        #                 if roc_auc>best_auc:
        #                     best_auc=roc_auc
        #                     best_n=n_est
        #                     best_l=lear
        #                     best_d=max_dep
        #                     best_sub=sub
        # print(f'best auc score:{best_auc},best n:{best_n},best l:{best_l},best d:{best_d},best sub:{best_sub}')
        # return
        # 多种模型选择
        gbdt = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample,random_state=42)
        # gbdt = GradientBoostingClassifier(n_estimators=140, learning_rate=0.05, max_depth=4, subsample=0.8,random_state=42)
        # binary_gbdt_models = {}
        # y_train=np.array(y_train)
        # for label in range(3):
        #     binary_y_train = (y_train == label).astype(int)  # 将标签转换为二分类问题
        #     gbdt = GradientBoostingClassifier(n_estimators=40, learning_rate=0.5, max_depth=3, subsample=0.6,random_state=42)
        #     gbdt.fit(X_train, binary_y_train)
        #     binary_gbdt_models[label] = gbdt
        # # 使用三个模型进行预测
        # y_probs = []
        # for label, gbdt_model in binary_gbdt_models.items():
        #     y_prob = gbdt_model.predict_proba(X_test)[:, 1]  # 获取预测为正类别的概率
        #     y_probs.append(y_prob)
        # y_pred = []
        # for probs in zip(*y_probs):
        #     predicted_label = max(range(3), key=lambda i: probs[i])
        #     y_pred.append(predicted_label)
        # accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)
        # return
        # X_train=np.array(X_train)
        # y_train=np.array(y_train)
        # model_cv_train(gbdt, 'gbdt',X_train,y_train,kfold=60)
        # print('cv finished')
        # return
        # gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.6,random_state=42)
        # grd_enc = OneHotEncoder()
        # gbdt = LogisticRegression()
        # gbdt_2.fit(X_train, y_train)
        # grd_enc.fit(gbdt_2.apply(X_train)[:, :, 0])
        # grd_lr.fit(grd_enc.transform(gbdt_2.apply(X_train)[:, :, 0]), y_train)
        # svc = svm.SVC()
        # LR=LogisticRegression()
        # RF=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
        # gbdt=StackingClassifier(estimators=[('gbdt',gbdt),('svm',svc),('lr',LR),('rf',RF)],final_estimator=LogisticRegression())
        # gbdt=lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,max_depth=7,subsample=1.0,num_leaves=31,random_state=42)
        # gbdt=XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=7,subsample=1.0,random_state=42)

    # 初始化 GridSearchCV
    # gbdt= GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=5, scoring='accuracy')
    # 在训练集上训练模型
    standard_scaler=StandardScaler()
    minmax_scaler=MinMaxScaler()
    pca=PCA(n_components=15)
    if is_train:
        
        # X_train=standard_scaler.fit_transform(X_train)
        # X_train=minmax_scaler.fit_transform(X_train)
        # X_train=pca.fit_transform(X_train)
        # 进行特征交叉
        # X_train=np.array(X_train,dtype=np.float32)
        # X_cross1=X_train[:,0]+X_train[:,1]+X_train[:,2]
        # X_cross2=X_train[:,6]+X_train[:,7]+X_train[:,8]
        # X_cross3=X_train[:,10]+X_train[:,11]
        # X_train=np.concatenate((X_train,X_cross1.reshape(-1,1),X_cross2.reshape(-1,1),X_cross3.reshape(-1,1)),axis=1)        
        # X_train=pca.fit_transform(X_train)

        gbdt.fit(X_train, y_train)
        print('feature importance:',gbdt.feature_importances_)     # [0.29872611 0.17596116 0.06042178 0.46489096]
        # print(gbdt.n_estimators_)
        # print(gbdt.train_score_)
        # print(len(gbdt.estimators_))

        # # 输出最佳参数组合和对应的准确率
        # print("Best parameters found: ", gbdt.best_params_)
        # print("Best accuracy found: ", gbdt.best_score_)
        # # 使用最佳参数的模型进行预测
        # gbdt = gbdt.best_estimator_

        y_pred_train = gbdt.predict(X_train)
        y_prob_train=gbdt.predict_proba(X_train)[:,1]
        y_pred_train=np.where(y_prob_train>=threshold,1,0)

        # 计算预测准确率
        accuracy = accuracy_score(y_train, y_pred_train)
        ap=average_precision_score(y_train,y_prob_train)
        roc_auc=roc_auc_score(y_train,y_prob_train)
        print('training ap:',ap)
        print('training roc_auc:',roc_auc)
        print('training accuracy:',accuracy) 
        report=classification_report(y_train,y_pred_train)
        print(report)
        matrix=confusion_matrix(y_train,y_pred_train)
        print(matrix.T)
        # with open('train.txt','w') as out:
        #     for img,y_t,y_p,y_p_p in zip(img_train,y_train,y_pred_train,y_prob_train):
        #         out.write(f'{img},{y_t},{y_p},{y_p_p}\n')
        # print(f'result saved in train.txt')
        


        # 使用训练好的模型进行预测
        # X_test=standard_scaler.transform(X_test)
        # X_test=minmax_scaler.transform(X_test)
        # X_test=pca.transform(X_test)
        # X_test=np.array(X_test,dtype=np.float32)
        # X_cross1=X_test[:,0]+X_test[:,1]+X_test[:,2]
        # X_cross2=X_test[:,6]+X_test[:,7]+X_test[:,8]
        # X_cross3=X_test[:,10]+X_test[:,11]
        # X_test=np.concatenate((X_test,X_cross1.reshape(-1,1),X_cross2.reshape(-1,1),X_cross3.reshape(-1,1)),axis=1)
        # X_test=pca.transform(X_test)

        y_pred_evl = gbdt.predict(X_test)
        y_prob_evl=gbdt.predict_proba(X_test)[:,1]
        y_pred_evl=np.where(y_prob_evl>=threshold,1,0)

        # 计算预测准确率
        accuracy = accuracy_score(y_test, y_pred_evl)
        ap=average_precision_score(y_test,y_prob_evl)
        roc_auc=roc_auc_score(y_test,y_prob_evl)
        print('test ap:',ap)
        print('test roc_auc:',roc_auc)
        print('test accuracy:',accuracy) # 0.9012345679012346
        report=classification_report(y_test,y_pred_evl)
        print(report)
        matrix=confusion_matrix(y_test,y_pred_evl)
        print(matrix.T)
        # with open('evl.txt','w') as out:
        #     for img,y_t,y_p,y_p_p in zip(img_test,y_test,y_pred_evl,y_prob_evl):
        #         out.write(f'{img},{y_t},{y_p},{y_p_p}\n')
        # print(f'result saved in evl.txt')
        with open(model_path,'wb') as f:
            pickle.dump(gbdt,f)
        print(f'model saved in {model_path}')
    else:
        # X_test=scaler.fit_transform(X_test)
        y_pred_test = gbdt.predict(X_test)
        y_prob_test=gbdt.predict_proba(X_test)[:,1]
        # X_test=np.array(X_test,dtype=np.float32)
        # y_prob_test[np.where(X_test[:,-1]<threshold)]=1
        with open(predict_result_path,'w') as f:
            for img,y_p,y_p_p in zip(img_test,y_pred_test,y_prob_test):
                f.write(f'{img},{y_p},{round(y_p_p,2)}\n')
        print(f'predict result saved in {predict_result_path}')
    if analyze:
        if os.path.exists(os.path.join(ana_dir,'clear_wrong_R')):
            shutil.rmtree(os.path.join(ana_dir,'clear_wrong_R'))
            os.makedirs(os.path.join(ana_dir,'clear_wrong_R'),exist_ok=True)
        else:
            os.makedirs(os.path.join(ana_dir,'clear_wrong_R'),exist_ok=True)
        if os.path.exists(os.path.join(ana_dir,'blurry_wrong_R')):
            shutil.rmtree(os.path.join(ana_dir,'blurry_wrong_R'))
            os.makedirs(os.path.join(ana_dir,'blurry_wrong_R'),exist_ok=True)
        else:
            os.makedirs(os.path.join(ana_dir,'blurry_wrong_R'),exist_ok=True)
        # if os.path.exists(os.path.join(ana_dir,'clear_wrong_P')):
        #     shutil.rmtree(os.path.join(ana_dir,'clear_wrong_P'))
        #     os.makedirs(os.path.join(ana_dir,'clear_wrong_P'),exist_ok=True)
        # if os.path.exists(os.path.join(ana_dir,'blurry_wrong_P')):
        #     shutil.rmtree(os.path.join(ana_dir,'blurry_wrong_P'))
        #     os.makedirs(os.path.join(ana_dir,'blurry_wrong_P'),exist_ok=True)
        # 使用训练集还是验证集自己改变变量
        print('start move wrong image to analyze dir...')
        # for y_t,y_p,y_prob,img_t in tqdm(zip(y_train,y_pred_train,y_prob_train,img_train)):
        for y_t,y_p,y_prob,img_t in tqdm(zip(y_test,y_pred_evl,y_prob_evl,img_test)):
            y_prob=round(y_prob,2)
            if y_t==0 and y_p==1:
                if not os.path.exists(os.path.join(ana_dir,'clear_wrong_R',img_t.split('/')[-1])):                    
                    shutil.copy(img_t,os.path.join(ana_dir,'clear_wrong_R',f'{y_prob}_'+img_t.split('/')[-1]))
            if y_t==1 and y_p==0:
                if not os.path.exists(os.path.join(ana_dir,'blurry_wrong_R',img_t.split('/')[-1])):       
                    shutil.copy(img_t,os.path.join(ana_dir,'blurry_wrong_R',f'{y_prob}_'+img_t.split('/')[-1]))
            # shutil.copy(img_t,os.path.join(ana_dir,'blurry_wrong_R',f'{y_prob}_'+img_t.split('/')[-1]))
        print(f'clear_wrong_R length:{len(os.listdir(os.path.join(ana_dir,"clear_wrong_R")))}')
        print(f'blurry_wrong_R length:{len(os.listdir(os.path.join(ana_dir,"blurry_wrong_R")))}')
        print('analyze finish')
                    # if accuracy>best_accu:
                    #     best_accu=accuracy
                    #     best_n=n_est
                    #     best_l=lear
                    #     best_d=max_dep
                    #     best_sub=sub


        # print(best_accu)
        # print(best_n)
        # print(best_l)
        # print(best_d)
        # print(best_sub)
def evl(test_txt_save_path,model_path):
    print('start evaluate...')
    X_test=[]
    y_test=[]
    img_test=[]
    with open(test_txt_save_path,'r') as f:
        for content in f.readlines():
            content=content.strip()
            img_path,label=content.split('\t')[0].split(',')
            features=content.split('\t')[1].split(',')
            # features[-1]=0
            X_test.append([*features])
            if label=='0':
                y_test.append(0)
            else:
                y_test.append(1)
            img_test.append(img_path)
    with open(model_path,'rb') as f:
        gbdt=pickle.load(f)
        print(f'model loaded from {model_path}')
    y_pred_test = gbdt.predict(X_test)
    y_prob_test=gbdt.predict_proba(X_test)[:,1]
    print(f'type of y_prob_test:{type(y_prob_test)}')
    # thresholds=np.arange(0.3,0.96,0.01).tolist()
    thresholds=[0.6]
    max_acc=0
    for threshold in thresholds:
        print(f'threshold:{threshold}')
        y_pred_test[y_prob_test<threshold]=0
        y_pred_test[y_prob_test>=threshold]=1
        acc_score=accuracy_score(y_test,y_pred_test)
        if acc_score>max_acc:
            max_acc=acc_score
            best_threshold=threshold
            print(f'accuracy:{acc_score},best_threshold:{best_threshold}')
        
    report=classification_report(y_test,y_pred_test)
    conf_matrix=confusion_matrix(y_test,y_pred_test)
    ap=average_precision_score(y_test,y_prob_test)
    roc_auc=roc_auc_score(y_test,y_prob_test)
    print('test ap:',ap)
    print('test roc_auc:',roc_auc)
    print(report)
    print(conf_matrix.T)




if __name__ == "__main__":
    ana_dir='wrong_classify_analyze'



    # 一些超参数
    # 为True则训练，为False则进行inference
    is_train=True
    normal_img_label='clear'
    mid_img_label='mid'
    blurry_img_label='blurry'
    # 是否抽增量数据的特征
    is_add=False
    

    # data location
    train_dir='../dataset/BlUR/iter9_image/train_3879_iter4'
    test_dir='../dataset/BlUR/iter9_image/val modified by PMs'
    inference_dir='../dataset/BlUR/improve_test'

    train_txt_save_path='data_features/blurry/balance_1.txt'
    test_txt_save_path='data_features/blurry/test_blurry_val modified by PMs.txt'
    
    inference_txt_save_path='data_features/blurry/inference.txt'
    predict_result_path='predict_result.txt'
    model_path='model/gbdt_model.pkl'

    increment_train_dir='../dataset/BlUR/iter9_image/val modeified  by PM add wh suffix'
    increment_train_txt_save_path='data_features/blurry/val_iter6.txt'
    
    # 首先提取train和val特征放入txt文件中，之后要手动把increment_train_txt_save_path中的内容复制粘贴添加到train_txt_save_path的后面，后续训练的时候就注释掉这行代码
    # extract_feature_to_txt(train_dir,test_dir,increment_train_dir,train_txt_save_path,test_txt_save_path,increment_train_txt_save_path,is_add)
    # 这是提取inference的特征并放入txt文件
    # extract_feature_to_txt_inference(inference_dir,inference_txt_save_path)

    # 通过前面提取到特征的txt文件训练GBDT模型，或利用GBDT模型进行推理
    # 是否分析错误分类的样本
    analyze=True
    threshold=0.7
    is_adjustParam=False
    n_estimators=100
    learning_rate=0.1
    max_depth=4
    subsample=0.6
    use_GBDT(is_adjustParam,train_txt_save_path,test_txt_save_path,inference_txt_save_path,model_path,predict_result_path,is_train,normal_img_label,analyze,threshold)
    # evl(test_txt_save_path,model_path)