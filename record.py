import numpy as np
import cv2
import os

# Takes in a camera object and any overlay information.
# Returns the raw RGB image from the camera.
def record(camera, overlay_text):
    # Continually display raw output of webcam
    ret, raw_frame = camera.read() # We could convert color spaces to HSV here
    frame = raw_frame 
    #cv2.putText(frame, overlay_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', raw_frame)
    return raw_frame

'''
# Takes in a label for a file.
# Returns a new file with format <label><index>.
def name_this_file(label, dir):
    max_index = 0
    for file in os.listdir(dir):
        # Get next lowest index for this label
        if label in file:
            index = int(file[len(label):file.find('.')])
            if index > max_index:
                max_index = index
    return label + str(max_index + 1)
'''

def name_this_dir(cwd):
    max_index = 0
    for dir in os.listdir(cwd):
        index = int(dir)
        if index > max_index:
            max_index = index
    return str(max_index)

if __name__ == '__main__':
    label = ''
    while label not in ('good', 'bad'):
        label = raw_input('LABEL FOR THIS DATASET (good / bad): ')

    temp_dir = '%s/data/%s' % (os.getcwd(), label)
    dir = '%s/%s/' % (temp_dir, name_this_dir(temp_dir))
    if not os.path.exists(dir):
        os.makedirs(dir)
    print dir
    index = 0 

    camera = cv2.VideoCapture(0)

    while True:
        frame = record(camera, label)
        filename = dir + str(index) + '.jpg'
        height, width, channels = frame.shape # Ensure we actually took an image.
        if width and height:
            cv2.imwrite(filename, frame) # Save the image.
            index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): # Quit on 'q' and clean up.
            break

    # Destroy camera object and GUI
    camera.release()
    cv2.destroyAllWindows()
