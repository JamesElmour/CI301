import cv2

def DecodeVideo(video, save, frames):
    print("Saving: " + video)
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    success = True
    while count <= frames:
        success,image = vidcap.read()
        
        if success is False:
            return
            
        cv2.imwrite(save + "/frame_%d.jpg" % count, image)     # save frame as JPEG file
        count += 1

        if count % 1000 == 0:
            print("Frame: ", count)
            print(save + "/frame_%d.jpg" % count)