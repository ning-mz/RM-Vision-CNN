import sys
sys.path.append("../")
from buffers import DataFrameBuffer
from executionBase import ExecutionBase

import cv2 
import argparse
import platform
import subprocess
import time
import math
import numpy as np
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

def ReadLabelFile(file_path):
      with open(file_path, 'r') as f:
          lines = f.readlines()
      ret = {}
      for line in lines:
          pair = line.strip().split(maxsplit=1)
          ret[int(pair[0])] = pair[1].strip()
      return ret

class execeutionObject(ExecutionBase):
    """
    """
    def __init__(self, dataBuffer):
        self.dataBuffer = dataBuffer

        super().__init__()

    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='Path of the detection model.', required=True)
        parser.add_argument('--label', help='Path of the labels file.')
        parser.add_argument('--input', help='File path of the input image.')
        args = parser.parse_args()
    
    
        # Initialize engine.
        engine = DetectionEngine(args.model)
        labels = ReadLabelFile(args.label) if args.label else None
    
        # Open image.
        #img = Image.open(args.input)
        #draw = ImageDraw.Draw(img)
    
        #Open video
        cv2.namedWindow('detect_demo')
        capture = cv2.VideoCapture('/home/mike/infantry-blue-light.avi')# This is pointing at a file as the source video
        capture.isOpened() 
    
        count = 0 
        init_time = time.time()
        self.__live = True
        while self.__live:
            while self.running:
                ret, frame = capture.read()
                if frame is None:
                    break
                count = count+1
                local_time = time.time()
                time_space = local_time-init_time
                if time_space > 2:
                    init_time = time.time()
                    fps = count/time_space
                    count = 0
                    print("FPS: ", fps)
                    print(local_time)
                cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2img = cv2.resize(cv2img, (300, 300))
                pilimg = Image.fromarray(cv2img)
                #pilimg = pilimg.resize((416, 416)) #,Image.ANTIALIAS

                #result = cv2.imread(pilimg) that's wrong
                #frame.astype(np.uint8)
                #img = np.array(frame, dtype=np.uint8)
                ans = engine.DetectWithImage(pilimg, threshold=0.05, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=10)
                #draw = ImageDraw.Draw(pilimg)

                shoot_flag = 0
                shoot_middle = [0, 0]
                armor_buffer = [0,0,0,0]
                if ans:
                    newDataFrame = []
                    for obj in ans:
                        if obj.score >0.5:
                            if labels:
                                box = obj.bounding_box.flatten().tolist()
                                if obj.label_id == 2:             
                                    shoot_middle[0] = int((box[0] + box[2]) / 2)
                                    shoot_middle[1] = int((box[1] + box[3]) / 2)
                                    shoot_flag = 1
       
                                if obj.label_id == 0:
                                    result_x = abs(shoot_middle[0] - ((box[0] + box[2]) / 2))
                                    result_y = abs(shoot_middle[1] - ((box[1] + box[3]) / 2))              
                                    result = math.sqrt(result_x*result_x + result_y*result_y)
                                    if result < 22:
                                        armor_buffer = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

                                newDataFrame.append(armor_buffer)#TODO: Alter this!

                    self.dataBuffer.append(newDataFrame)

                key = cv2.waitKey(1)
    
        capture.release()
        cv2.destroyAllWindows()