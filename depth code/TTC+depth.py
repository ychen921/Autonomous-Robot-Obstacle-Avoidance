import cv2
import numpy as np
import matplotlib as plt
import statistics
# Change video path
path = "1.avi"
def matching(img1,img2):
    orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

# Initialize Brute-Force matcher and perform matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

# Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)


    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return pts1,pts2


def E_P_tran_rota(pts1,pts2,intrin):

     pts1_n = cv2.undistortPoints(np.expand_dims(pts1,axis=1), cameraMatrix=intrin,distCoeffs=None)
     pts2_n = cv2.undistortPoints(np.expand_dims(pts2,axis=1), cameraMatrix=intrin,distCoeffs=None)

     E, mask = cv2.findEssentialMat(pts1_n,pts2_n, focal=667,pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
     _, Ro, Tr, mask = cv2.recoverPose(E, pts1_n, pts2_n)
     return E,Ro, Tr


# Lucas-Kanade Optical Flow
def LK_Optical_Flow(prev_frame, new_frame, window_size, feature_list):
    
    tau = 0.04
    
    w = int(window_size/2)
    
    # Normalize the pixel
    prev_frame = prev_frame/255
    new_frame = new_frame/255
    
    # Gradient Operator
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])
    
    # Optical Flow matrix
    u = np.zeros(prev_frame.shape)
    v = np.zeros(prev_frame.shape)
    
    # Gradient over x, y, t
    fx = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_x) #+ cv2.filter2D(src=new_frame, ddepth=-1, kernel=kernel_x)
    fy = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_y) #+ cv2.filter2D(src=new_frame, ddepth=-1, kernel=kernel_y)
    ft = cv2.filter2D(src=new_frame, ddepth=-1, kernel=kernel_t) - cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_t)
    
    
    # Compute every deteted corner
    for feature in feature_list:
            
        j, i = feature.ravel()		
        i, j = int(i), int(j) 
            
        Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            
        b = np.reshape(It, (It.shape[0],1))
        
        A = np.vstack((Ix, Iy)).T
        
        # compute velocity
        if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
            U = np.matmul(np.linalg.pinv(A), b)
                    
            u[i,j] = U[0]
            v[i,j] = U[1]
    # print(type(u), type(v))
    # print(u.shape, v.shape)
    # print('u: ', u)
    # print('v: ', v)
            
    return (u, v)

def draw_optical_flow(frame, new_frame, u, v):
    
    # arrow color
    line_color = (0, 0, 255)
    
    arrow_img = frame.copy()

    row, col = new_frame.shape

    # Plot the optical flow
    for i in range(row):
        for j in range(col):
            
            if u[i,j] and v[i,j]:
                arrow_img = cv2.arrowedLine(arrow_img, (j+int(round(u[i,j])), int(i+round(v[i,j]))), (j, i), line_color, thickness=1)
    
    return arrow_img
            
cap = cv2.VideoCapture(path)

# Define the Farneback optical flow parameters
params = dict(pyr_scale=0.5,
              levels=3,
              winsize=15,
              iterations=10,
              poly_n=5,
              poly_sigma=1.2,
              flags=0)

# Initialize the previous frame and the FOE
prev_frame = None
foe = None
BLOCK_NUM_X = 8
BLOCK_NUM_Y = 6
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))
four = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('outp.avi', four,30.0,(820,616), isColor=False)

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    frequency = cap.get(cv2.CAP_PROP_FPS)
    if  ret:
       print('s')
    #    print(frame.shape)
       
        

    # Convert the frame to grayscale
       gray_d = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur the frame
       gray = cv2.GaussianBlur(gray_d, (5, 5), 0)

       if prev_frame is not None:
        
        # Find the corners in previous frame
        feature_list,_ = matching(prev_frame_d,gray_d)
        
        # Implementing Lucas-Kanade Optical Flow algorithm
        u, v = LK_Optical_Flow(prev_frame, gray, 15, feature_list)
        
        # Compute the optical flow between the current and previous frames
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, **params)
        # print(type(flow), flow.shape)

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
        # print('foe_x: ', foe_x)
        # print('foe_y: ', foe_y)
        
        # Calculate every blokc of the time to collision (TTC)
        kpCnt = np.zeros((BLOCK_NUM_Y, BLOCK_NUM_X))
        ttcSum = np.zeros((BLOCK_NUM_Y, BLOCK_NUM_X))
        ttcMean = np.zeros((BLOCK_NUM_Y, BLOCK_NUM_X))
        ttc = np.zeros(flow.shape[:2])
        
        # print('TTC shape: ', ttc.shape)
        for feature in feature_list:
            j, i = feature.ravel()		
            i, j = int(i), int(j)
            if u[i,j] and v[i,j]:
                ttc[i,j] = np.sqrt((np.square(j - foe_x) + np.square(i - foe_y)) / (np.square(u[i,j]) + np.square(v[i,j])))
                
            else:
                ttc[i,j] = 0
            # print('(i, j)', i, j, '(u, v): ', u[i,j], v[i,j])
            idxX = int(j * BLOCK_NUM_X / w)
            idxY = int(i * BLOCK_NUM_Y / h)
            kpCnt[idxY, idxX] += 1
            ttcSum[idxY, idxX] += ttc[i,j]
        
        # cv2.imshow('depth', depth)
        # if cv2.waitKey(25) == ord('q'):
        #  break
            
        # print(kpCnt, ttcSum)
        # for i in range(BLOCK_NUM_X):
        #     for j in range(BLOCK_NUM_Y):
        #         ttcMean[j,i] = ttcSum[j,i] / kpCnt[j,i]
        
        # Draw the TTC on the frame
        for x in range(BLOCK_NUM_X):
            for y in range(BLOCK_NUM_Y):
                cv2.putText(frame, str(round(ttcMean[y, x], 2)), (int(w/BLOCK_NUM_X*x), int(h/BLOCK_NUM_Y*y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.2, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw the FOE on the frame
        if foe is not None and isinstance(foe, tuple) and len(foe) == 2:
            cv2.circle(frame, foe, 10, (0, 0, 255), -1)
        else:
            print(f'Invalid FOE: {foe}')

        # Draw the flow vectors on the frame
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                fx, fy = flow[y, x]
                cv2.line(frame,
                         (x, y),
                         (int(x + fx), int(y + fy)),
                         (0, 255, 0),
                         1)
        op_flow = (np.square(u) + np.square(v))
        
        intrinsic_m = np.array([[664.68,0,413.48],
                           [0,664.87,322.61],
                           [0,0,1]])
         
        pts1, pts2 = matching(prev_frame_d,gray_d)

        #  print(pts1)

        E_m, R_m, T_m = E_P_tran_rota(pts1,pts2,intrinsic_m)
        velocity = np.linalg.norm(T_m) 
        depth = ttc.astype(np.uint8)
        max = np.max(depth)
        min = np.min(depth)
        # m = statistics.median(depth)
        
        for i in range(len(depth)):
            for j in range(len(depth[i])):
                if depth[i][j] !=0:
                    depth[i][j] = (max - depth[i][j]) / (max - min) * 255
        depth = depth.astype(np.uint8)
        
        # img = ((max - i) / (max - min) * 255 for i in depth)
        
        out.write(depth)
    # Show the frame
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(25) == ord('q'):
    #     break

    # Save the current frame for the next iteration
       prev_frame = gray.copy()
       prev_frame_d = gray_d.copy()
    else:
        
        break

        
out.release()

cap.release()