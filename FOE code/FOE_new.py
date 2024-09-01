import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('fp1.avi')

# Define the Farneback optical flow parameters
params = dict(pyr_scale=0.5,
              levels=3,
              winsize=15,
              iterations=3,
              poly_n=5,
              poly_sigma=1.2,
              flags=0)

# Initialize the previous frame and the FOE
prev_frame = None
foe = None

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Compute the optical flow between the current and previous frames
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, **params)

        # Compute the FOE
        h, w = flow.shape[:2]
        h_mid, w_mid = h // 2, w // 2
        h_left = flow[:, :w_mid, 0].sum()
        h_right = flow[:, w_mid:, 0].sum()
        v_top = flow[:h_mid, :, 1].sum()
        v_bottom = flow[h_mid:, :, 1].sum()
        if abs(h_left - h_right) > abs(v_top - v_bottom):
            foe = (w_mid, h_mid)
        else:
            foe = (int(w // 2 - v_top // h_left), int(h // 2 - h_left // v_top))

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

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Save the current frame for the next iteration
    prev_frame = gray.copy()

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
