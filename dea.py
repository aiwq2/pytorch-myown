# 针对list中的txt文件，对于每个txt文件，比较每一行中第一列元素的值的第三列元素形成的方差，并按方差从大到小排列放入另一个txt为你教案
import numpy as np
file1='predict_result_0.2.txt'
file2='predict_result_0.4.txt'
file3='predict_result_0.6.txt'
img_var_dict={}
with open(file1,'r') as f:
    lines1=f.readlines()
    with open(file2,'r') as f2:
        lines2=f2.readlines()
        with open(file3,'r') as f3:
            lines3=f3.readlines()
            for l1,l2,l3 in zip(lines1,lines2,lines3):
                img1,label1,score1=l1.strip().split(',')
                img2,label2,score2=l2.strip().split(',')
                img3,label3,score3=l3.strip().split(',')
                scores=[float(score1),float(score2),float(score3)]
                var=np.var(scores)
                img_var_dict[img1]=var
            var_tupel=list(img_var_dict.items())
            var_tupel.sort(key=lambda x:x[1],reverse=True)
            with open('var_sort.txt','w') as out:
                for line in var_tupel:
                    out.write(f'{line[0]},{line[1]}\n')