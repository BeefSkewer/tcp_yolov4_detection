from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import datetime
import darknet
from pypylon import pylon
import _thread
from socket import *
import json
import struct
global person_num,preTime,detect_num,flag
#检测到的目标数目，当变动的时候，保存图片
person_num=0
car_num=0
bicycle_num=0
detect_num=0
preTime=time.time()
flag=0
frame=0
#

def save_detections(box_num,image):
    global preTime,detect_num,flag,frame
    # 保存检测到目标的图片
    # detections:目标信息列表
    frame=frame+1
    #print(frame)
    if(box_num!=detect_num)and (box_num>0):
        detect_num=box_num
        preTime=time.time()
        flag=1
        frame=0
    #若发现检测的目标个数发生变化时，若0.5秒个数不变则传输图片。若检测目标数目不变则1.5秒传输一次图片
    if(frame>3)and flag==1and (box_num>0)or((frame>10)and flag==0and (box_num>0)):
      flag=0
      img_name=time.strftime('%Y-%m-%d_%H_%M_%S')
      cv2.imwrite('/home/nvidia/Object_Detect/image/%s.jpg'%(img_name),
                                                       image)

      #darknet.send_mjpeg('/home/nvidia/Object_Detect/image/2020-07-08_11_05_46.jpg',8090,400000,40)
      frame=0
      print("图片存储成功")
      f=open("/home/nvidia/Object_Detect/txt/img_msg.txt","a+",encoding="utf-8")
      f.write("\n time:%s   person_num:%s   car_num:%s   bicycle_num:%s"%(img_name,person_num,car_num,bicycle_num))

      #tcp 发送图片部分
      # with open ('/home/nvidia/Object_Detect/%s.jpg'%(img_name),'rb') as f:
      #         img_bytes=f.read()
      #         msg={
      #
      #             "filename": str(img_name),
      #             "total_size":len(img_bytes),
      #             "person_num":person_num,
      #             "car_num":car_num,
      #             "bicycle_num":bicycle_num
      #         }
      #         headers_json=json.dumps(msg)
      #         headers_json_bytes=bytes(headers_json,encoding="utf-8")
      #         skt.send(struct.pack('i',len(headers_json_bytes)))
      #         skt.send(headers_json_bytes)
      #         skt.sendall(img_bytes)
      #
      #         # for data in f:
      #         #   skt.send(data)
      #         f.close()
      #
      #         print("发送完成")
    detect_num =box_num
    preTime = time.time()





def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
#存储检测目标的信息
#detection(
#   [0]=类别（bytes）
#   [1]=置信度(flout)
#   [2]=x,y,w,h 中心坐标和长宽(tuple:4)
#   例： [(b'person', 0.32106196880340576, (273.49481201171875, 172.2395782470703, 291.85736083984375, 364.177001953125))]
# )

def cvDrawBoxes(detections, img):
    #从识别到的目标列表中，获取每个目标的信息并根据坐标画框
    global person_num,car_num,bicycle_num
    box_num=0
    person_num=0
    bicycle_num=0
    car_num=0
    for detection in detections:
        if(detection[0].decode() not in ['person','car'
                                                  '','bicycle']): break
        if(detection[0].decode()=='person'): person_num=person_num+1
        if(detection[0].decode() == 'car'): car_num = car_num + 1
        if(detection[0].decode() == 'bicycle_num'): bicycle_num = bicycle_num + 1
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        box_num=box_num+1
    return img,box_num


netMain = None
metaMain = None
altNames = None
def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4-tiny.cfg"
    weightPath = "./yolov4-tiny.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain),3)
    # 连接Basler相机列表的第一个相机
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # 开始读取图像
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    #输出源视频
    out = cv2.VideoWriter(
             "/home/nvidia/Object_Detect/original_video/%s_origital.avi"%(time.strftime('%Y-%m-%d_%H_%M_%S')), cv2.VideoWriter_fourcc(*"MP42"), 20.0,
             (darknet.network_width(netMain), darknet.network_height(netMain)))
    out_detect = cv2.VideoWriter(
        "/home/nvidia/Object_Detect/detect_video/%s_detect.avi"%(time.strftime('%Y-%m-%d_%H_%M_%S')), cv2.VideoWriter_fourcc(*"MP42"), 20.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
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
            # darknet 处理部分

            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            frame_resized_color = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            out.write(frame_resized_color)
            image,box_num = cvDrawBoxes(detections, frame_resized)
            cv2.imwrite('/home/nvidia/Object_Detect/temp.jpg', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 缓存

            #img=darknet.change_cvmat("/home/nvidia/Object_Detect/image/2020-07-06_17_29_46.jpg",1)
            darknet.send_mjpeg('/home/nvidia/Object_Detect/temp.jpg'.encode("utf8"), 8090, 400000, 40)
            print("转换成功")
            im, arr = darknet.array_to_image(image)
            #print(box_num)



            save_detections(box_num,image)
            out_detect.write(image)
            #输出fps
            #print(1 / (time.time() - prev_time))

            cv2.imshow('Demo', image)
            k = cv2.waitKey(1)
            if k == 27:
                break
        grabResult.Release()


    # 关闭相机
    camera.StopGrabbing()
    # 关闭窗口
    cv2.destroyAllWindows()
    out.release()
    out_detect.release()
    # cap = cv2.VideoCapture("video.mp4")
    # cap.set(3, 1280)
    # cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    # print("Starting the YOLO loop...")
    #
    # # Create an image we reuse for each detect
    # darknet_image = darknet.make_image(darknet.network_width(netMain),
    #                                 darknet.network_height(netMain),3)
    # while True:
    #     prev_time = time.time()
    #     ret, frame_read = cap.read()
    #     frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    #     frame_resized = cv2.resize(frame_rgb,
    #                                (darknet.network_width(netMain),
    #                                 darknet.network_height(netMain)),
    #                                interpolation=cv2.INTER_LINEAR)
    #
    #     darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    #
    #     detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    #     image = cvDrawBoxes(detections, frame_resized)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     print(1/(time.time()-prev_time))
    #     cv2.imshow('Demo', image)
    #     cv2.waitKey(3)
    # cap.release()
    # out.release()

if __name__ == "__main__":
    #tcp与局域网主机通信
    # Host = '192.168.1.80'
    # Port = 5005
    # ADDR = (Host, Port)
    #
    # server = socket(AF_INET, SOCK_STREAM)
    # server.bind(ADDR)
    # server.listen(5)
    #
    #
    # print("等待客户端连接")
    # skt, addr = server.accept()
    # print(skt)
    YOLO()

   # server.close()
