import numpy as np
'''
split=200000
a=[i for i in range(1999800)]
seed = 547
np.random.seed(seed)
np.random.shuffle(a)

ids=a[:split]
print(len(ids))
with open('./train_20w_ids.txt','w') as f_train:
    for j in range(len(ids)):
        f_train.write(str(ids[j])+'\n')
'''
#max number of patches per image 
#orig:10w*20=200w
#posneg30:33333*20=666666
#posneg60:2w*20=40w
#posneg90:15000*20=30w
#posneg120:12000*20=24w
#posneg150:1w*20=20w
#=====select different rotation angles,so every line is different
rotate_angle=['orig'] #1
#rotate_angle=['orig','pos30','neg30'] #3
#rotate_angle=['orig','pos30','neg30','pos60','neg60'] #5
#rotate_angle=['orig','pos30','neg30','pos60','neg60','pos90','neg90'] #7
#rotate_angle=['orig','pos30','neg30','pos60','neg60','pos90','neg90','pos120','neg120'] #9
#rotate_angle=['pos30','neg30','pos60','neg60','pos90','neg90','pos120','neg120','pos150','neg150'] #11 'orig'
#rotate_angle=['pos30_first5','neg30_first5','pos60_first5','neg60_first5','pos90_first5','neg90_first5',
#             'pos120_first5','neg120_first5','pos150_first5','neg150_first5']
#rotate_angle=['pos30_first1','neg30_first1','pos60_first1','neg60_first1','pos90_first1','neg90_first1',
#              'pos120_first1','neg120_first1','pos150_first1','neg150_first1']
#=====the numbers from per full image,so every box is different
split=90910
orig_img_numbers=1
#datarootdir/masks/neg90/6_4.npy
#datarootdir/imgs/neg90/6_4.npy
print(len(rotate_angle)*orig_img_numbers*split)


with open('All_angle_data/train_1_number5_total100w_ids.txt','w') as f_train:
    for angle in rotate_angle:
        for i in range(orig_img_numbers):
            for j in range(split):
                f_train.write(angle+'/'+str(i)+'_'+str(j)+'\n')
              
'''
with open('All_angle_data/train_11_number1_total100w_ids.txt',mode='a') as f_train:
    for angle in rotate_angle:
        for i in range(orig_img_numbers):
            for j in range(split):
                f_train.write(angle+'/'+str(i)+'_'+str(j)+'\n')
                   

'''