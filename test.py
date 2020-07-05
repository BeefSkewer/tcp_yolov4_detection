"""
有趣的事情
没有结束
2020/4/2 16:28
"""
from pypylon import pylon
import numpy as np
import time

import cv2 as cv
 
# 连接Basler相机列表的第一个相机
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
 
# 开始读取图像
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
 
# 转换为OpenCV的BGR彩色格式
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
 
while camera.IsGrabbing():
    prev_time = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
 
    if grabResult.GrabSucceeded():
        # 转换为OpenCV图像格式
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv.namedWindow('title', cv.WINDOW_NORMAL)
        cv.imshow('title', img)
        print(1 / (time.time() - prev_time))
        k = cv.waitKey(1)
        if k == 27:
            break
    grabResult.Release()
 
# 关闭相机
camera.StopGrabbing()
# 关闭窗口
cv.destroyAllWindows()


