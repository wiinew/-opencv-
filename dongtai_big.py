import cv2
import numpy as np
import os

def load():
    lisdir=os.listdir('./face_damik')
    dirs=[i for i in  lisdir if not i.startswith('.')]
    print(dirs)
    face=[]
    target=[]
    for index,dir in enumerate(dirs):
        for i in range(1,10):
            print('------%s-------%d'%(dir,i))
            gray = cv2.imread('./face_damik/%s/%d.jpg' % (dir, i))
            gray_ = gray[:, :, 0]
            gray_ = cv2.resize(gray_, dsize=(64, 64))
            gray_=cv2.equalizeHist(gray_)
            face.append(gray_)
            target.append(index)
        #print(gray.shape)
    face=np.asarray(face)
    target=np.asarray(target)
    print(face.shape)
    return face,target,dirs
"""    faces=[]
    target=[i for i in range(len(dirs))]*1
    for dir in enumerate(dirs):
        for  i in range(1,10):
            gray=cv2.imread('./face/%s/%d.jpg' % (dir,13))
           #print(gray)
            print(dir)
            gray_=gray[:,:, 0]
            gray_=cv2.resize(gray_,dsize=(64,64))
            faces.append(gray_)
           # target.append(dir)
    faces=np.asarray(faces)
    target=np.asarray(target)
    target.sort()
    #print('---------', faces.shape)
    return faces,target"""
def nmlio():
    global faace, target ,dirs
    # print(target)
    # print(faace)
    index = np.arange(27)
    np.random.shuffle(index)
    # print(index)
    faces = faace[index]
    #print(faces)
    target = target[index]
    x_tran, x_text = faces[:22], faces[22:]
    y_tran, y_text = target[:22], target[22:]
    return x_tran,x_text,y_text,y_tran


def method_name(face_reconginzer,names):
  cap= cv2.VideoCapture(0)
  face_deter=cv2.CascadeClassifier('E:\\pythonProject6\\vc\\haarcascade_frontalface_alt.xml')
  while True:
      flag,frame=cap.read()
      if not flag:
          break
      gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
      faces=face_deter.detectMultiScale(gray,minNeighbors=10)
      for x,y,w,h in faces:
          face=gray[x:x+w,y:y+h]
          face=cv2.resize(face,dsize=(64,64))
          face=cv2.equalizeHist(face)
          y_,confidence=face_reconginzer.predict(face)
          lable=names[y_]
          print('这个人是: %s ,置信度是%d'%(lable,confidence))
          cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
          cv2.putText(frame,text=lable,org=(x,y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.5,color=[0,0,255],thickness=2)
      cv2.imshow('face',frame)
      key=cv2.waitKey(1000//24)
      if key==ord('q'):
        break
  cap.release()
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
    faces, target, names = load()
    # face_reconginzer = cv2.face.EigenFaceRecognizer_create()
    # face_reconginzer=cv2.face.FisherFaceRecognizer_create()
    face_reconginzer = cv2.face.LBPHFaceRecognizer_create()
    face_reconginzer.train(faces, target)
    method_name(face_reconginzer, names)