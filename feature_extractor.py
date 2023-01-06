import os
import cv2
import numpy as np
import math
import pandas as pd

PATH = os.getcwd()
DATA_PATH = PATH + "/Data"
classes = os.listdir(DATA_PATH)

cols = ["class"]
cols = ["f"+str(i) for i in range(1,66)]
cols = ["class"] + cols

df = pd.DataFrame(columns=cols)

def get_points_straight(img,line_coord, limit,dir):
    first, last=None,None
    for i in range(limit):
        if dir == 1:
            if img[line_coord][i] == 0:
                if first==None:first = [line_coord, i]
                else: last = [line_coord, i]
        else:
            if img[i][line_coord] == 0:
                if first==None:first= [i, line_coord]
                else: last = [i, line_coord]
    return [first,last]

def diag_traverser(img, x, y, get_points, dir):
    first, last=None,None
    count = 0

    i,j=x,y
    while True:
        try:
            if i<0 or j<0:break
            if img[i,j] == 0:
                if first==None:first= [j,i]
                else: last = [j,i]
                count+=1

            i+=1
            if dir==1: j+=1
            else: j-=1
        except:
            break
    
    if get_points:return [first,last]
    return count

def get_distance(point1, point2):
    if point1==None or point2==None:return -1
    return math.dist(point1,point2)

def get_distance_between_centers(line1_points, line2_points):
    point11, point12 = line1_points[0], line1_points[1]
    point21, point22 = line2_points[0], line2_points[1]

    if None in [point11, point12, point21, point22]: return -1

    line1_center = [(point11[0] + point12[0])/2, (point11[1] + point12[1])/2]
    line2_center = [(point21[0] + point22[0])/2, (point21[1] + point22[1])/2]
    
    return get_distance(line1_center, line2_center)

for c in classes:
    print(c)
    class_path = DATA_PATH+"/"+c
    imgs = os.listdir(class_path)

    if c == "c": continue
    for imgp in imgs:
        img_path = class_path + "/" +imgp
        img = cv2.imread(img_path, 0)
        copy = cv2.imread(img_path)

        (thresh, bw) = cv2.threshold(img, 100,255, cv2.THRESH_BINARY)

        h,w = bw.shape
        cropped = bw[:,20:w-20]
        # copy = copy[:,20:w-20]
        h,w = cropped.shape
        
        partition_offset_x = w//4
        partition_offset_y = h//4
        line1_x = partition_offset_x
        line2_x = partition_offset_x*2
        line3_x = partition_offset_x*3
        line1_y = partition_offset_y
        line2_y = partition_offset_y*2
        line3_y = partition_offset_y*3

        points_hor1 = get_points_straight(cropped, line1_y, w, 1)
        points_hor2 = get_points_straight(cropped, line2_y, w, 1)
        points_hor3 = get_points_straight(cropped, line3_y, w, 1)
        points_ver1 = get_points_straight(cropped, line1_x, h, 2)
        points_ver2 = get_points_straight(cropped, line2_x, h, 2)
        points_ver3 = get_points_straight(cropped, line3_x, h, 2)
        points_diag11 = diag_traverser(cropped, 0, 0, True, 1)
        points_diag12 = diag_traverser(cropped, 0, w//2, True, 1)
        points_diag13 = diag_traverser(cropped, h//2, 0, True, 1)
        points_diag21 = diag_traverser(cropped, 0,w-1, True, 2)
        points_diag22 = diag_traverser(cropped, 0,w//2, True, 2)
        points_diag23 = diag_traverser(cropped, h//2,w-1, True, 2)

        disth1 = get_distance(points_hor1[0],points_hor1[1])
        disth2 = get_distance(points_hor2[0],points_hor2[1])
        disth3 = get_distance(points_hor3[0],points_hor3[1])
        distv1 = get_distance(points_ver1[0],points_ver1[1])
        distv2 = get_distance(points_ver2[0],points_ver2[1])
        distv3 = get_distance(points_ver3[0],points_ver3[1])
        distd11 = get_distance(points_diag11[0],points_diag11[1])
        distd12 = get_distance(points_diag12[0],points_diag12[1])
        distd13 = get_distance(points_diag13[0],points_diag13[1])
        distd21 = get_distance(points_diag21[0],points_diag21[1])
        distd22 = get_distance(points_diag22[0],points_diag22[1])
        distd23 = get_distance(points_diag23[0],points_diag23[1])

        cbp_11 = diag_traverser(cropped, 0, 0, False, 1)
        cbp_12 = diag_traverser(cropped, 0, w//2, False, 1)
        cbp_13 = diag_traverser(cropped, h//2, 0, False, 1)
        cbp_21 = diag_traverser(cropped, 0, w-1, False, 2)
        cbp_22 = diag_traverser(cropped, 0, w//2, False, 2)
        cbp_23 = diag_traverser(cropped, h//2, w-1, False, 2)
        
        block_counts = []
        for i in range(4):
            for j in range(4):
                block_cropped = cropped[partition_offset_x*i: partition_offset_x*(i+1), partition_offset_y*j: partition_offset_y*(j+1)]
                block_counts.append(np.count_nonzero(block_cropped==0))

        distc1 = get_distance_between_centers(points_hor1, points_ver3)
        distc2 = get_distance_between_centers(points_hor2, points_ver2)
        distc3 = get_distance_between_centers(points_hor3, points_ver1)
        distc4 = get_distance_between_centers(points_diag11, points_diag21)
        distc5 = get_distance_between_centers(points_diag12, points_diag23)
        distc6 = get_distance_between_centers(points_diag13, points_diag22)

        count = np.count_nonzero(img==0)

        # if c == "c":
        #     cv2.circle(copy, (points_hor1[0][0], points_hor1[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_hor1[1][0], points_hor1[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_hor2[0][0], points_hor2[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_hor2[1][0], points_hor2[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_hor3[0][0], points_hor3[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_hor3[1][0], points_hor3[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_ver1[0][0], points_ver1[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_ver1[1][0], points_ver1[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_ver2[0][0], points_ver2[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_ver2[1][0], points_ver2[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_ver3[0][0], points_ver3[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_ver3[1][0], points_ver3[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag11[0][0], points_diag11[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag11[1][0], points_diag11[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag12[0][0], points_diag12[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag12[1][0], points_diag12[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag13[0][0], points_diag13[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag13[1][0], points_diag13[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag21[0][0], points_diag21[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag21[1][0], points_diag21[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag22[0][0], points_diag22[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag22[1][0], points_diag22[1][1]), 2, (0, 0, 255), 2)

        #     cv2.circle(copy, (points_diag23[0][0], points_diag23[0][1]), 2, (0, 0, 255), 2)
        #     cv2.circle(copy, (points_diag23[1][0], points_diag23[1][1]), 2, (0, 0, 255), 2)
           
        #     cv2.imshow("img", copy)
        #     if cv2.waitKey(0) == ord("q"):exit()
        #     cv2.destroyAllWindows()
        
        data = [c]
        data.append(points_hor1[0][1] if (points_hor1[0] is not None) else -1)
        data.append(points_hor2[0][1] if (points_hor2[0] is not None) else -1)
        data.append(points_hor3[0][1] if (points_hor3[0] is not None) else -1)
        data.append(points_ver1[0][0] if (points_ver1[0] is not None) else -1)
        data.append(points_ver2[0][0] if (points_ver2[0] is not None) else -1)
        data.append(points_ver3[0][0] if (points_ver3[0] is not None) else -1)
        data.append(points_hor1[1][1] if (points_hor1[1] is not None) else -1)
        data.append(points_hor2[1][1] if (points_hor2[1] is not None) else -1)
        data.append(points_hor3[1][1] if (points_hor3[1] is not None) else -1)
        data.append(points_ver1[1][0] if (points_ver1[1] is not None) else -1)
        data.append(points_ver2[1][0] if (points_ver2[1] is not None) else -1)
        data.append(points_ver3[1][0] if (points_ver3[1] is not None) else -1) #12

        data.append(points_diag11[0][0] if (points_diag11[0] is not None) else -1)
        data.append(points_diag12[0][0] if (points_diag12[0] is not None) else -1)
        data.append(points_diag13[0][0] if (points_diag13[0] is not None) else -1)
        data.append(points_diag21[0][1] if (points_diag21[0] is not None) else -1)
        data.append(points_diag22[0][1] if (points_diag22[0] is not None) else -1)
        data.append(points_diag23[0][1] if (points_diag23[0] is not None) else -1)
        data.append(points_diag11[1][0] if (points_diag11[1] is not None) else -1)
        data.append(points_diag12[1][0] if (points_diag12[1] is not None) else -1)
        data.append(points_diag13[1][0] if (points_diag13[1] is not None) else -1)
        data.append(points_diag21[1][1] if (points_diag21[1] is not None) else -1)
        data.append(points_diag22[1][1] if (points_diag22[1] is not None) else -1)
        data.append(points_diag23[1][1] if (points_diag23[1] is not None) else -1) #24

        data.append(disth1)
        data.append(disth2)
        data.append(disth3)
        data.append(distv1)
        data.append(distv2)
        data.append(distv3) #30

        data.append(distd11)
        data.append(distd12)
        data.append(distd13)
        data.append(distd21)
        data.append(distd22)
        data.append(distd23) #36

        data.append(cbp_11)
        data.append(cbp_12)
        data.append(cbp_13)
        data.append(cbp_21)
        data.append(cbp_22)
        data.append(cbp_23) #42

        data = data + block_counts #58

        data.append(distc1)
        data.append(distc2)
        data.append(distc3)
        data.append(distc4)
        data.append(distc5)
        data.append(distc6) #64
    
        data.append(count) #65

        row = pd.DataFrame([data], columns=cols)
        df = df.append(row)

df.to_csv(PATH+"/data.csv", index=False) 