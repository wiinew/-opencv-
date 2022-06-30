import matplotlib
from cvzone.HandTrackingModule import HandDetector
import cv2
startDist=None
scale=0
cx,cy=500,500
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
detetcor = HandDetector(detectionCon=0.8)
while True:
    flag, img = cap.read()
    hands, img = detetcor.findHands(img)
    img1 = cv2.imread("11.jpg")
    # print(img1)
    if len(hands) == 2:
        # print("Rwo")
        # print(detetcor.fingersUp(hand[0]),detetcor.fingersUp(hand[1]))
        if detetcor.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detetcor.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            # print("flag")

             hand1 = hands[0]
             lmList1 = hand1["lmList"]  # List of 21 Landmark points
             bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
             centerPoint1 = hand1['center']  # center of the hand cx,cy
             handType1 = hand1["type"]  # Handtype Left or Right
             fingers1 = detetcor.fingersUp(hand1)
             hand2 = hands[1]
             lmList2 = hand2["lmList"]  # List of 21 Landmark points
             bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
             centerPoint2 = hand2['center']  # center of the hand cx,cy
             handType2 = hand2["type"]  # Hand Type "Left" or "Right"

             fingers2 = detetcor.fingersUp(hand2)

        # Find Distance between two Landmarks. Could be same hand or different hands
             if startDist is None:
              length, info, img = detetcor.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
             # print(length)
              startDist=length
             length, info, img = detetcor.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
             scale=int((startDist-length)//2)
             cx, cy = info[4:]
             print(scale)
    else:
        startDist=None
            # print(detetcor.findDistance(None,lmList1[8],lmList2[8],img))
    try:
     h1,w1,_=img1.shape
     newH,newW=((h1+scale)//2)*2,((w1+scale)//2)*2
     img1=cv2.resize(img1,(newW,newH))

     img[cy-newH//2:cy+newH//2, cx-newW//2:cx+newW//2] = img1
    except:
        pass
    cv2.imshow("image", img)
    cv2.waitKey(1)
