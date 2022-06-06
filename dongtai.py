import cv2
import numpy as np
import os


def take_photo(path):
    cap=cv2.VideoCapture(0)
    face_detector=cv2.CascadeClassifier('E:\\pythonProject6\\vc\\haarcascade_frontalface_alt.xml')
    filname=1
    flag_writ=False

    while True:
       flag,frame= cap.read()
       """key=cv2.waitKey(0)
       if key==ord('q'):
           break"""
       """elif flag==False:
           break
       elif not flag:
           break"""
       gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
       faces=face_detector.detectMultiScale(gray,minNeighbors=10)
       for x,y,w,h in faces:
        if flag_writ:
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,dsize=(64,64))
            cv2.imwrite('./face_damik/%s/%d.jpg'%(path,filname),face)
            filname+=1
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
       if filname>10:
           break
       cv2.imshow('face',frame)
       key1=cv2.waitKey(1000//24)
       if key1==ord('q'):
           break
       if key1==ord('w'):
           flag_writ=True

    cv2.destroyAllWindows()
    cap.release()


def take_facEs():
    while True:
        path = input('请输入文件夹的名字，拼音缩写，如按q退出！')
        if path == 'q':
            break
        os.makedirs('./face_damik/%s' % (path), exist_ok=True)
        take_photo(path)


if __name__=='__main__':
    take_facEs()