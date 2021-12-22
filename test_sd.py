import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.stats
import math

'''======================== parameters ================================'''
# test directory path
path = "./test/"
# txt save path
save_txt_path="./test_sd.txt"

'''======================== Main Body ==========================='''

def point_distance_line(point,line_point1,line_point2):
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

def judgeSD_new(path):
    #使用累计能量谱评价
    # img = cv2.imread(path)
    img = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    dct_img = cv2.dct(img)
    m,n= dct_img.shape
    power_x=np.zeros((n,1))
    power_y =np.zeros((m,1))
    power_dct_square = np.square(np.abs(dct_img))
    sum_rows = np.sum(power_dct_square, axis=0).reshape((n, 1))
    sum_cols = np.sum(power_dct_square, axis=1).reshape((m, 1))
    power_x = power_x + sum_rows / m  # Average power across rows
    power_y = power_y + sum_cols / n  # Average power across cols

    power_x=np.log10(power_x+1)
    power_y=np.log10(power_y+1)
    max_x = np.max(power_x)
    max_y = np.max(power_y)
    power_x = power_x / max_x
    power_y = power_y / max_y
    Sx = np.sum(power_x)
    Sy = np.sum(power_y)
    S_x = np.zeros((n, 1))
    S_y = np.zeros((m, 1))
    power_x_accumulate_sum = 0
    power_y_accumulate_sum = 0
    for i in range(0,n):
        power_x_accumulate_sum += power_x[i]
        S_x[i] = power_x_accumulate_sum / Sx
    for i in range(0,m):
        power_y_accumulate_sum += power_y[i]
        S_y[i] = power_y_accumulate_sum / Sy
    gradient_y = S_y[1:] - S_y[:-1]
    gradient_x = S_x[1:] - S_x[:-1]

    point_x = np.argmin(abs(gradient_x - 1 / n))  # 找到转折点
    point_y=  np.argmin(abs(gradient_y - 1 / m))
    # 计算转折点到直线的距离
    P_x = np.array([point_x,float(S_x[point_x])])  # 转折点坐标
    P_y =  np.array([point_y,float(S_y[point_y])])
    Q1 =  np.array([0,0])
    Q2 =  np.array([n,1])
    Q3=   np.array([m,1])
    distance_x=point_distance_line(P_x,Q1,Q2)
    distance_y = point_distance_line(P_y, Q1, Q3)
    distance_final =np.sqrt(pow(distance_x,2)+ pow(distance_y , 2))
    result = distance_final
    #print(result)
    return result



def get_frame_from_video2(video_name, interval):
    final_co=[]#前后帧对比
    # 保存图片的路径
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0
    while True:
        success, frame = video_capture.read()
        i += 1
        if(i==2):
            reso = int(min(frame.shape[0], frame.shape[1]))
        if not success:
            print('video is all read')
            break
        if i % interval == 0:
            j += 1
            if j>1:
               try:
                   co =judgeSD_new(frame)
                   print("Sd on frame " + str(j) + ": " + str(co))
                   final_co.append(co)
               except:
                   print("Frame error!")
                   final_co.append(final_co[-1])
    return final_co,reso

def final2(video_name,interval):
    final_co,reso = get_frame_from_video2(video_name, interval)
    sd=np.mean(np.array(final_co))
    if(reso==720):
        if (float(sd) > 0.4):
            score = 1
        elif (0.3 <= float(sd) <= 0.4):
            score = 2
        else:
            score = 3
    elif ( 1080==reso):
        if (float(sd) > 0.58):
            score = 1
        elif (0.39 <= float(sd) <= 0.58):
            score = 2
        else:
            score = 3
    elif ( reso==2160):
        if (float(sd) > 0.63):
            score = 1
        elif (0.49 <= float(sd) <= 0.63):
            score = 2
        else:
            score = 3
    else:
        print("This resolution has not been test!")
        score=3

    return  [sd, score]


dirs = os.listdir(path)
fo = open(save_txt_path, "w")
# fo.write( "video_id || sd || score "+ "\n")
for item in dirs:
    print(str(item))
    sd=final2(path+item,1)
    res = ""
    for i in range(len(sd)):
        res = res + " " + str(sd[i])
    fo.write(item.split(".mp4")[0] + res + "\n")
fo.close()


