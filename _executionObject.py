import sys
sys.path.append("../")
from buffers import DataFrameBuffer
from executionBase import ExecutionBase

import cv2 
import numpy as np

class executionObject(ExecutionBase):
    """
    Encapsulates the running of the detection network and tracking
    """

    def __init__(self, dataBuffer: DataFrameBuffer):
        self.dataBuffer = dataBuffer

        super().__init__()

        #TODO: add initialisation of variables here

    def run(self):
        """
        WARNING: This is a blocking method. Run in a seperate thread.

        Runns the object detection and tracking in an externaly controllable perpetual loop.

        Requires the "resume" method to be called to start actual running.
        """

        while self.isLive():
            while self.running:
                pass#TODO: add code here

        #TODO: release assets here if needed