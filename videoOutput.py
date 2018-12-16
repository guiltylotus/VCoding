import cv2
import os

image_folder = './images'
video_name = 'video.avi'
width = 352
height = 288

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

for i in range(100):
  image = "BGRimage" + str(i) + ".png"
  video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()