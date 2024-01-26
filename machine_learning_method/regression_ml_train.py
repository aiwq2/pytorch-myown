import os
import cv2
from matplotlib import pyplot as plt
from extract_features.blurry import *
from extract_features.NRSS import *
import traceback
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,average_precision_score,mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier,GradientBoostingRegressor)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb

def get_fetures(iamge_path):
    img=cv2.imread(iamge_path,0)
    height,width=img.shape
    features=[]
    canny_feature=canny(img)
    laplacian_feature=variance_of_laplacian(img)
    fft_feature=fourier_transform(img)
    # contrast_feature=image_contrast(img)
    entropy_feature=image_entropy(img)
    saturation_feature=color_saturation(iamge_path)
    brightness_feature=image_brightness(iamge_path)
    # SMD_feature=SMD(img)
    SMD2_feature=SMD2(img)
    # brenner_feature=brenner(img)
    variance_feature=variance(img)
    energy_feature=energy(img)
    # Vollath_feature=Vollath(img)
    nrss_feature=NRSS(iamge_path)

    # features.append([float(canny_feature),float(laplacian_feature),float(fft_feature),
    #                 float(entropy_feature),float(saturation_feature),float(brightness_feature),
    #                 float(SMD_feature),float(SMD2_feature),float(brenner_feature),float(variance_feature),
    #                 float(energy_feature),float(Vollath_feature)
    #                 ])

    features.append([float(canny_feature),float(laplacian_feature),float(fft_feature),float(entropy_feature),
                    float(saturation_feature),float(brightness_feature),float(SMD2_feature),
                    float(variance_feature),float(energy_feature),float(nrss_feature)
                    ])

    return features

def write_features_to_txt(f,features,image_path,gptv_score=None):
    if gptv_score is not None:
        f.write(f'{image_path},{gptv_score}\t')
    else:
        f.write(f'{image_path}\t')
    for index,feature in enumerate(features):
        if index!=len(features)-1:
            f.write(f'{feature},')
        else:
            f.write(f'{feature}\n')

def extract_feature_to_txt(train_dir,test_dir,train_txt_save_path,text_txt_save_path):
    with open(train_txt_save_path,'w') as f:
        for img in tqdm(os.listdir(train_dir),desc='train_dir'):
            try:
                image_path=os.path.join(train_dir,img)
                features=get_fetures(image_path)
                gptv_score=img.split('_')[0]
                write_features_to_txt(f,features[0],image_path,gptv_score)
            except Exception as e:
                print(f'error:{e},img:{img}')


    with open(text_txt_save_path,'w') as f:
        for img in tqdm(os.listdir(test_dir),desc='test_dir'):
            try:
                image_path=os.path.join(test_dir,img)
                features=get_fetures(image_path)
                gptv_score=img.split('_')[0]
                write_features_to_txt(f,features[0],image_path,gptv_score)
            except Exception as e:
                print(f'error:{e},img:{img}')
       
def extract_feature_to_txt_inference(inference_dir,inference_txt_save_path):
    with open(inference_txt_save_path,'w') as f:
        for img in tqdm(os.listdir(inference_dir),desc='inference_dir'):
            try:
                image_path=os.path.join(inference_dir,img)
                features=get_fetures(image_path)
                gptv_score=img.split('_')[0]
                write_features_to_txt(f,features[0],image_path,gptv_score)
            except Exception as e:
                print(e)

def draw(X,y_ture,y_pred,location):
    X=np.linspace(0,1,len(y_ture))
    y_ture=np.array(y_ture)
    plt.scatter(X, y_ture, s=1,color='black', label='Actual')
    plt.scatter(X, y_pred, s=1,color='red', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Actual vs Predicted Values '+location)
    plt.savefig(location+'.jpg')
    print(f'save {location}.jpg')
    plt.clf()

def use_GBDT(train_txt_save_path,test_txt_save_path,inference_txt_save_path,model_path,predict_result_path,is_train,normal_img_label,analyze):
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
                img_path,gpt_score=content.split('\t')[0].split(',')
                features=content.split('\t')[1].split(',')
                X_train.append([*features])
                y_train.append(float(gpt_score))
                img_train.append(img_path)
        with open(test_txt_save_path,'r') as f:
            for content in f.readlines():
                content=content.strip()
                img_path,gpt_score=content.split('\t')[0].split(',')
                features=content.split('\t')[1].split(',')
                X_test.append([*features])
                y_test.append(float(gpt_score))
                img_test.append(img_path)
    else:
        with open(inference_txt_save_path,'r') as f:
            for content in f.readlines():
                content=content.strip()
                img_path,gpt_score=content.split('\t')[0].split(',')
                features=content.split('\t')[1].split(',')
                X_test.append([*features])
                y_test.append(float(gpt_score))
                img_test.append(img_path)

    # 定义参数网格
    param_grid = {
        'n_estimators': range(20,200,20),
        'learning_rate':[0.001,0.01,0.1,0.5,0.05],
        'max_depth':[3,4,5,6,7],
        'subsample':[0.6,0.8,1],
        'min_samples_split':[2]
    }

    best_n=-1
    best_l=-1
    best_d=-1
    best_sub=-1
    best_accu=-1
    best_min_samples_split=-1
    best_mse=1000.0

    # 初始化梯度提升分类器模型
    if not is_train:
        with open(model_path,'rb') as f:
            gbdt=pickle.load(f)
        print(f'regression model loaded from {model_path}')
    else:
    # 调参
        # for n_est in tqdm(param_grid['n_estimators']):
        #     for lear in tqdm(param_grid['learning_rate']):
        #         for max_dep in param_grid['max_depth']:
        #             for sub in param_grid['subsample']:
        #                 for min_samples_split in param_grid['min_samples_split']:
        #                     gbdt = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lear, max_depth=max_dep, subsample=sub,min_samples_split=min_samples_split,random_state=42)
        #                     # X_train=scaler.fit_transform(X_train)
        #                     gbdt.fit(X_train, y_train)

        #                     # 使用训练好的模型进行预测
        #                     # X_test=scaler.transform(X_test)
        #                     y_pred = gbdt.predict(X_test)

        #                     mse=mean_squared_error(y_test,y_pred)
        #                     if mse<best_mse:
        #                         best_mse=mse
        #                         best_n=n_est
        #                         best_l=lear
        #                         best_d=max_dep
        #                         best_sub=sub
        # print(f'best mse:{best_mse},best n:{best_n},best l:{best_l},best d:{best_d},best sub:{best_sub}')
        # return
        # 多种模型选择
        gbdt = GradientBoostingRegressor(n_estimators=180,learning_rate=0.05,max_depth=7,subsample=0.8,min_samples_split=2,random_state=42)
        # gbdt = KNeighborsRegressor(n_neighbors=20)
        # svc = svm.SVC()
        # LR=LogisticRegression()
        # RF=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
        # gbdt=StackingClassifier(estimators=[('gbdt',gbdt),('svm',svc),('lr',LR),('rf',RF)],final_estimator=LogisticRegression())
        # gbdt=lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,max_depth=7,subsample=1.0,num_leaves=31,random_state=42)
        # gbdt=XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=7,subsample=1.0,random_state=42)

    # 初始化 GridSearchCV
    # gbdt= GridSearchCV(estimator=gbdt, scoring='neg_mean_squared_error',param_grid=param_grid, cv=5,verbose=1)
    # 在训练集上训练模型
    standard_scaler=StandardScaler()
    minmax_scaler=MinMaxScaler()
    if is_train:
        
        # X_train=standard_scaler.fit_transform(X_train)
        # X_train=minmax_scaler.fit_transform(X_train)
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
        mse_train = mean_squared_error(y_train, y_pred_train)
        print(f'training mse:{mse_train}')
        draw(X_train,y_train,y_pred_train,'train')
        

        # 使用训练好的模型进行预测
        # X_test=standard_scaler.fit_transform(X_test)
        # X_test=minmax_scaler.fit_transform(X_test)
        y_pred_evl = gbdt.predict(X_test)
        mse_test=mean_squared_error(y_test,y_pred_evl)
        draw(X_test,y_test,y_pred_evl,'evl')
        print(f'evl mse:{mse_test}')
        with open('train.txt','w') as out:
            for img,y_t,y_p in zip(img_train,y_train,y_pred_train):
                out.write(f'{img},{y_t},{y_p}\n')
        print(f'train result save in train.txt')

        with open('evl.txt','w') as out:
            for img,y_t,y_p in zip(img_test,y_test,y_pred_evl):
                out.write(f'{img},{y_t},{y_p}\n')
        print(f'train result save in evl.txt')
        with open(model_path,'wb') as f:
            pickle.dump(gbdt,f)
        print(f'model saved in {model_path}')
    else:
        # X_test=scaler.fit_transform(X_test)
        y_pred_regression = gbdt.predict(X_test)
        # y_pred_classfication=gbdt_classfication.predict_proba(X_test)[:,1]*100.0
        with open(predict_result_path,'w') as f:
            for img,y_p,y_p_p in zip(img_test,y_test,y_pred_regression):
                f.write(f'{img},{y_p},{y_p_p}\n')
        mse_test=mean_squared_error(y_test,y_pred_regression)
        print(f'test mse:{mse_test}')
        print(f'predict result saved in {predict_result_path}')
    if analyze:
        if os.path.exists(os.path.join(ana_dir,'normal_wrong_R')):
            shutil.rmtree(os.path.join(ana_dir,'normal_wrong_R'))
        if os.path.exists(os.path.join(ana_dir,'much_text_wrong_R')):
            shutil.rmtree(os.path.join(ana_dir,'much_text_wrong_R'))
        for y_t,y_p,img_t in zip(y_test,y_pred,img_test):
            if y_t==0 and y_p==1:
                os.makedirs(os.path.join(ana_dir,'normal_wrong_R'),exist_ok=True)
                shutil.copy(os.path.join(test_dir,'normal_text',img_t),os.path.join(ana_dir,'normal_wrong_R',img_t))
            if y_t==1 and y_p==0:
                os.makedirs(os.path.join(ana_dir,'much_text_wrong_R'),exist_ok=True)
                shutil.copy(os.path.join(test_dir,'too_much_text',img_t),os.path.join(ana_dir,'much_text_wrong_R',img_t))
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

if __name__ == "__main__":
    ana_dir='cnstd_analyze_img/Percentage of text/ana'




    # 一些超参数
    # 为True则训练，为False则进行inference
    is_train=True
    normal_img_label='clear'
    # 是否分析错误分类的样本
    analyze=False

    # data location
    train_dir='../dataset/BlUR/gptv_regression_dataset/train'
    test_dir='../dataset/BlUR/gptv_regression_dataset/val'
    inference_dir='../dataset/BlUR/gptv_2cls/test_blurry_100'
    train_txt_save_path='data_features/blurry/train_blurry_feature.txt'
    test_txt_save_path='data_features/blurry/test_blurry_feature.txt'
    inference_txt_save_path='data_features/blurry/inference.txt'
    predict_result_path='predict_result.txt'
    model_path='model/gbdt_model.pkl'
    
    # 首先提取train和val特征放入txt文件中，后续训练的时候就注释掉这行代码
    # extract_feature_to_txt(train_dir,test_dir,train_txt_save_path,test_txt_save_path)
    # 这是提取inference的特征并放入txt文件
    # extract_feature_to_txt_inference(inference_dir,inference_txt_save_path)

    # 通过前面提取到特征的txt文件训练GBDT模型，或利用GBDT模型进行推理
    use_GBDT(train_txt_save_path,test_txt_save_path,inference_txt_save_path,model_path,predict_result_path,is_train,normal_img_label,analyze)