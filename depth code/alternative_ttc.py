import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
def FOE_calculate(img1,img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    params = dict(pyr_scale=0.5,
             levels=3,
             winsize=15,
             iterations=10,
             poly_n=5,
             poly_sigma=1.2,
             flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **params)

       # Compute the FOE
    h, w = flow.shape[:2]
    min_diff = float('inf')
    for x in range(w):
           h_left = (flow[:, :x, 0] < 0).sum()
           h_right = (flow[:, x:, 0] > 0).sum()
           diff = abs(h_left - h_right)
           if diff < min_diff:
               min_diff = diff
               foe_x = x

    min_diff = float('inf')
    for y in range(h):
           v_top = (flow[:y, :, 1] < 0).sum()
           v_bottom = (flow[y:, :, 1] > 0).sum()
           diff = abs(v_top - v_bottom)
           if diff < min_diff:
               min_diff = diff
               foe_y = y

    foe = (foe_x, foe_y)
    return foe
def matching(img1,img2):
    sift = cv2.SIFT_create(contrastThreshold=0.01,edgeThreshold=15)

# Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create a BFMatcher object
    bf = cv2.BFMatcher()

# Perform feature matching
    matches = bf.knnMatch(descriptors1, descriptors2, k=3)

# Apply ratio test to filter good matches
    good_matches = []
    for m, n,_ in matches:
     if m.distance <  0.75 * n.distance:
        good_matches.append(m)
    print(len(good_matches))

# Draw matches on a new image
# output_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the output image
# cv2.imshow('Matches', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    img1_m = []
    img2_m = []
# print(keypoints1)
    for i in good_matches:
     ind1 = i.queryIdx
     ind2 = i.trainIdx

     img1_m.append(keypoints1[ind1].pt)
     img2_m.append(keypoints2[ind2].pt)
    return img1_m,img2_m


def E_P_tran_rota(pts1,pts2,intrin):

     pts1_n = cv2.undistortPoints(np.expand_dims(pts1,axis=1), cameraMatrix=intrin,distCoeffs=None)
     pts2_n = cv2.undistortPoints(np.expand_dims(pts2,axis=1), cameraMatrix=intrin,distCoeffs=None)

     E, mask = cv2.findEssentialMat(pts1_n,pts2_n, focal=667,pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
     _, Ro, Tr, mask = cv2.recoverPose(E, pts1_n, pts2_n)
     return E,Ro, Tr
     
     
     

cap = cv2.VideoCapture('1.avi')
count = 0
frame_list = []
fps = cap.get(cv2.CAP_PROP_FPS)
while (cap.isOpened()):
    
    ret, frame = cap.read()
    if ret:
        
        frame_list.append(frame)

        count += 1

    else:
        break
cap.release()
prev_dis_of_foe = None
prev_foe = None
prev_frame = frame_list[0]
gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
# prev_cor = cv2.goodFeaturesToTrack(gray,200,0.01,30,3)

# new_frame_list = []
fource = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('method2_1.avi', fource, 10.0,(820,616), isColor=False)

for i in range(len(frame_list) - 2):
    
    frame1 = frame_list[i+1]
    frame2 = frame_list[i+2]
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    print(gray1.shape)
    # cv2.imshow('',frame1)
    # cv2.waitKey(10)
    


    foe = FOE_calculate(frame1, frame2)
    intrinsic_m = np.array([[664.68,0,413.48],
                           [0,664.87,322.61],
                           [0,0,1]])
    pts1, pts2 = matching(frame1,frame2)

    E_m, R_m, T_m = E_P_tran_rota(pts1,pts2,intrinsic_m)
    # print(foe)
    p_m1 = np.hstack([np.eye(3), np.zeros((3,1))])
    p_m2 = np.hstack([R_m,T_m.reshape(-1,1)])
    
    # nextpts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, prev_cor, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT, 10 ,0.03))
    # cor = []
    # new_next = []
    
    # for a in range(len(status)):
    #     if status[a] == 1:
    #         cor.append(prev_cor[a])
    #         new_next.append(nextpts[a])
    prev_cor_tri=np.array(pts1).reshape((2,len(pts2)))
    new_next_tri = np.array(pts2).reshape((2,len(pts2)))
    # if len(new_next) < len(prev_cor):
    #     num = len(new_next)
    #     prev_dis_of_foe = prev_dis_of_foe[:num]
    
        
    

    points3D = cv2.triangulatePoints(p_m1,p_m2,prev_cor_tri,new_next_tri)
    pts3D = points3D / points3D[3]
    pts3D = np.array(pts3D[:3,:].reshape(pts3D.shape[1],3)).astype(np.uint8)
    new_frame = np.zeros(gray1.shape)
    depth_inf = []
    for i in pts3D:
        # print(i[:2])
        h = np.linalg.norm(i)
        depth_inf.append(h)
    dmax = max(depth_inf)
    dmin = min(depth_inf)
    
    for i in range(len(pts1)):
        # print([int(pts1[i][0])])
        
        new_frame[int(pts1[i][1])][int(pts1[i][0])] = (dmax - depth_inf[i]) / (dmax -dmin) *255
    new_frame = new_frame.astype(np.uint8)
    out.write(new_frame)



    
    




       
    
    # prev_cor = copy.deepcopy(new_next)
    # prev_cor = np.array(prev_cor)
    # # prev_cor = np.array(prev_cor)
    # prev_foe = copy.deepcopy(foe)
    
    # new_frame_list.append(frame1)
    
out.release()
    
# for i in new_frame_list:
#     cv2.imshow('',i)
#     cv2.waitKey(25)

        



    


    


# if prev_foe is not None:
#        foe_3D = cv2.triangulatePoints(p_m1,p_m2,prev_foe,foe)
#        foe_3D = foe_3D / foe_3D[3]
     
#        distance_of_foe = pts3D.T - foe_3D.T
#     #    print(len(distance_of_foe))

#        if prev_dis_of_foe is not None:
#            velocity = np.linalg.norm((distance_of_foe - prev_dis_of_foe), axis=1) * fps
#            time = np.linalg.norm(distance_of_foe) / velocity
#            time_min = time[0]
#            order = 0

#            for j in range(len(time)):
#               if time[j] < time_min:
#                 time_min = time[j]
#                 order = 0
#             #   print(int(nextpts[order][0]))

#            cv2.circle(frame1,nextpts[order][0].astype(int),3,(255,0,0), -1)
#            cv2.putText(frame1,f'minimum time is{time_min}s',nextpts[order][0].astype(int),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    







    # prev_dis_of_foe = copy.deepcopy(distance_of_foe)