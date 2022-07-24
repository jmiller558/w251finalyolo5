import torch
import cv2

model = torch.hub.load('/usr/src/app/yolo5', 'custom', path='runs/train/yolov5s_results/weights/last.pt', source='local')

model.conf = .7

model.classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29]

cap = cv2.VideoCapture(0)
captured = False
while(True):
    # Capture frame-by-frame
    if captured == False: 
        ret, frame = cap.read()

    # Display the resulting frame
    #cv2.imshow('frame',frame)
        image = frame[..., ::-1]
        results = model(image, size=640)
        to_show = results.render()[0][..., ::-1]
    
    cv2.imshow('results',to_show)
    
    if results.pandas().xyxy[0].empty == False:
        to_show = results.render()[0][..., ::-1]
        captured = True
    
    if cv2.waitKey(1) & 0xFF == ord('r'):
        captured = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
