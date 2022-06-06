"""import cv2
import numpy as np
v=cv2.VideoCapture(0)
h=v.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
w=v.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
face_dater=cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
while True:
    head=cv2.imread('旦.png',-1)
    flag,frame=v.read()
    if flag==False:
        break
    frame=cv2.resize(frame,dsize=(int(w // 2),int(h // 2)))
    gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
    faces=face_dater.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3)
    flag=0
    for x,y,w,h in faces:
        head=cv2.resize(head,dsize=(w,h))
        head_channels=cv2.split(head)
        frame_channels=cv2.split(frame)
        b,g,r,a=cv2.split(head)
        for c in range(0,3):
            frame_channels[c]=np.array(frame_channels[c],dtype=np.uint8)
            k=np.uint8((255.0-a)/255.0)
            frame_channels[c][y:y + h,x:x +w]+=frame_channels[c][y:y + h,x:x + w]*k
            head_channels[c] *=np.array(a /255,dtype=np.uint8)
            frame_channels[c][y:y + h, x:x +w] +=np.array(head_channels[c],dtype=np.unit8)
        ans=cv2.merge(frame_channels)
        flag=1
    if flag:
       cv2.imshow(ans)
    else:
        pass
    key=cv2.waitKey(10)
    if key ==ord('q'):
        break
cv2.destroyAllWindows()
v.release()"""
import cv2
scap=cv2.VideoCapture(0)
filename=1
flage_write=False
face_dectteror=cv2.CascadeClassifier('E:\\pythonProject5\\vc\\haarcascade_frontalface_alt.xml')
while True:
    flag,frame=scap.read()
    if flag==False:
        break
    gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
    facse=face_dectteror.detectMultiScale(gray,minNeighbors=10)
    for x,y,w,h in facse:
      if flage_write:
         face= gray[y:y+h,x:x+w]
         face=cv2.resize(face,dsize=(64,64))
         cv2.imwrite('./face/lfk/%d.jpg'%(filename),face)
         filename +=1
         cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),
                      color=[0,0,255],thickness=2)
    if filename >30:
        break
    cv2.imshow('face',frame)
    key=cv2.waitKey(1000//24)
    if key==ord('q'):
        break
    if key==ord('w'):
        flage_write=True
cv2.destroyAllWindows()
scap.release()