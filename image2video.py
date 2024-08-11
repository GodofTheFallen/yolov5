# -*- coding: utf-8 -*-
import cv2
import os





if __name__ == '__main__':
     # parse xml file to get label
     video_dir = "./runs/detect/6.avi"
     fps = 2
     img_size = (1920, 1080)
     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
     videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
     path = "./runs/detect/exp2"
     for i in range(990,1010):
         img_filename = path + "/" + "%06d.jpg" % i
         img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
         videoWriter.write(img)
videoWriter.release()


