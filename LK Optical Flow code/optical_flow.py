'''
    File name: optical_flow.py
    Author: Yi-Chung Chen
    Date created: 5/6/2023
    Date last modified: 5/6/2023
'''

import cv2
import numpy as np

# Change video path
path = "G:/我的雲端硬碟/Maryland/UMD/Robotics/Courses/2023 Spring/ENPM673 - Perception for Autonomous Robots/Project 5/video/video1.avi"

# Lucas-Kanade Optical Flow
def LK_Optical_Flow(prev_frame, new_frame, window_size, feature_list):
    
    tau = 0.04
    
    w = int(window_size/2)
    
    # Normalize the pixels
    prev_frame = prev_frame/255
    new_frame = new_frame/255
    
    # Gradient Operators
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])
    
    # Optical Flow matrix
    u = np.zeros(prev_frame.shape)
    v = np.zeros(prev_frame.shape)
    
    # Compute Spatial and Temporal derivative
    fx = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_x) 
    fy = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_y)
    ft = cv2.filter2D(src=new_frame, ddepth=-1, kernel=kernel_t) - cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_t)
    
    
    # Compute every detected corner
    for feature in feature_list:
            
        j, i = feature.ravel()		
        i, j = int(i), int(j) 
            
        Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            
        b = np.reshape(It, (It.shape[0],1))
        
        A = np.vstack((Ix, Iy)).T
        
        # Find two eigenvalues are large (find the corners) tracking point
        if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
            
            # compute velocity by least square method
            U = np.matmul(np.linalg.pinv(A), b)
                    
            u[i,j] = U[0]
            v[i,j] = U[1]
    
            
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

ret, first_frame = cap.read()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

window = 15

# Blur the previous frame
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

while(cap.isOpened()):
    
    # Find the corners in previous frame
    feature_list = cv2.goodFeaturesToTrack(prev_gray, 10000, 0.05, 0.1)
    
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur the frame
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Implementing Lucas-Kanade Optical Flow algorithm
    u, v = LK_Optical_Flow(prev_gray, gray, window, feature_list)
    
    # Draw Optical flow
    displacement = draw_optical_flow(frame, gray, u, v)
    
    cv2.imshow("optical flow", displacement)
    cv2.waitKey(5)
    
    prev_gray = gray
    
cap.release()
cv2.destroyAllWindows()