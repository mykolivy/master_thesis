from PIL import Image
import numpy as np
#from scipy.misc import imsave

resolution = 64
fps = 30
duration = 5

data = np.zeros((resolution, resolution))
row = np.zeros(resolution)

image_num = fps*duration
move_ratio = int(image_num / resolution)
print(move_ratio)

j = 0
for t in range(0,image_num):
    if t % move_ratio == 0:
        for i in range(0,resolution):
            data[i][j] = 255
        j=(j+1)%resolution
    image = Image.fromarray(np.uint8(data))
    #image.show(command='fim -a')
    image.save(f'img_{t}.pgm')
