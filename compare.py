import cv2
import numpy as np
from pathlib import Path
from time import time
import datetime

# Open the 23.976 fps video
cap1 = cv2.VideoCapture(str(Path('~').expanduser())+'/Documents/Evangelion - 3.33 You Can (Not) Redo (2012) CUSTOM MULTi 1080p 10bits BluRay x265 EAC3 AC3 5.1 v3 -Punisher694.mkv')

# Open the 24 fps video
cap2 = cv2.VideoCapture(str(Path('~').expanduser())+'/Documents/Evangelion - 3.333 You Can (Not) Redo (2012) CUSTOM MULTi 1080p 10bits BluRay x265 DDP 5.1 v2 -Punisher694.mkv')

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec
target_fps = 23.976023976023978
frameSize = (4040,1832)
out = cv2.VideoWriter('output.mp4', fourcc, target_fps, frameSize)
# Get the original framerates
orig_fps1 = cap1.get(cv2.CAP_PROP_FPS)
orig_fps2 = cap2.get(cv2.CAP_PROP_FPS)
# Define the target framerate

print(orig_fps1)
print(orig_fps2)

# Calculate the frame duration for the target framerate
target_frame_duration = 1 / target_fps

x_seconds = 0
cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
x_seconds = 0
cap2.set(cv2.CAP_PROP_POS_FRAMES, 24)

# Initialize a counter and timer for each video
frame_counter1 = 1
prev_frame_time1 = 0

frame_counter2 = cap2.get(cv2.CAP_PROP_POS_FRAMES)
prev_frame_time2 = 0

cv2.namedWindow('video', cv2.WINDOW_NORMAL)

f = open('values.txt','w')
f.write(f'time,mseg,msec,msegb\n')

# Loop through each frame of the videos
while True:
    loop_start = time()
    # Read the next frames from each video
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    # Break out of the loop if either video has reached its end
    if not ret1 or not ret2:
        break
    
    # If the first video has a different framerate, resample it
    if orig_fps1 != target_fps:
        curr_frame_time1 = frame_counter1 * orig_fps1
        if curr_frame_time1 - prev_frame_time1 >= target_frame_duration:
            # Write the resampled frame to the output
            # Note that we need to use interpolation when resizing to prevent artifacts
            prev_frame_time1 = curr_frame_time1
        else:
            continue

    # If the second video has a different framerate, resample it
    if orig_fps2 != target_fps:
        curr_frame_time2 = frame_counter2 * orig_fps2
        if curr_frame_time2 - prev_frame_time2 >= target_frame_duration:
            # Write the resampled frame to the output
            # Note that we need to use interpolation when resizing to prevent artifacts
            prev_frame_time2 = curr_frame_time2
        else:
            continue
    
   
    # Increment the frame counters for each video
    frame_counter1 += 1
    frame_counter2 += 1

    frame1g = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2g = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    avg_luma = frame1.mean()
    avg_luma2 = frame2.mean()
    luma_diff =  (avg_luma - avg_luma2)
    #print(f'lumadiff:{luma_diff}')
    frame2g = cv2.add(frame2g,luma_diff)
    diff_img = cv2.GaussianBlur(frame1g, (15, 15), 0)
    diff_img2 = cv2.GaussianBlur(frame2g, (15, 15), 0)

    diff = cv2.absdiff( frame1g, frame2g)
    diff2 = cv2.absdiff( frame1, frame2)
    diff3 = cv2.absdiff(diff_img,diff_img2)
    mse = np.mean(diff ** 2)
    mse2 = np.mean(diff2 ** 2)
    mse3 = np.mean(diff3 ** 2)
    f.write(f'{cap1.get(cv2.CAP_PROP_POS_MSEC)/1000},{mse},{mse2},{mse3}\n')

    layout = cv2.hconcat([frame1,frame2])

    diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
    
    layout2 = cv2.hconcat([diff,diff2])
    layout3 = cv2.vconcat([layout,layout2])
    layout3 = cv2.copyMakeBorder(layout3, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.putText(layout3, f'3.33 BD: {str(datetime.timedelta(seconds=cap1.get(cv2.CAP_PROP_POS_MSEC)//1000))}',(110,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(layout3, f'3.333 BD: {str(datetime.timedelta(seconds=cap2.get(cv2.CAP_PROP_POS_MSEC)//1000))}',(110+1920,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(layout3, f'mse: {mse}, msec: {mse2}, msegb: {mse3}',(110,160+816+816),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('video', layout3)
    out.write(layout3)
    print(layout3.shape)

    key = cv2.waitKey(1)

    if key == ord('a'):
        cap1.set(cv2.CAP_PROP_POS_MSEC, cap1.get(cv2.CAP_PROP_POS_MSEC)+20000)
        cap2.set(cv2.CAP_PROP_POS_MSEC, cap2.get(cv2.CAP_PROP_POS_MSEC)+20000)
    if key == ord('d'):
        cap1.set(cv2.CAP_PROP_POS_MSEC, cap1.get(cv2.CAP_PROP_POS_MSEC)-20000)
        cap2.set(cv2.CAP_PROP_POS_MSEC, cap2.get(cv2.CAP_PROP_POS_MSEC)-20000)
    if key == ord(' '):
        cv2.waitKey(0)
    if key == ord('s'):
        cap1.release()
        cap2.release()
        f.close()
        out.release()

# Release the video capture objects
cap1.release()
cap2.release()
f.close()
out.release() 