import numpy as np
import cv2
import os

# Takes in a camera object and any overlay information.
# Returns the raw RGB image from the camera.
def record(camera, overlay_text):
    # Continually display raw output of webcam
    ret, raw_frame = camera.read() # We could convert color spaces to HSV here
    frame = raw_frame 
    cv2.putText(frame, overlay_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('frame', frame)

# Takes in a label for a file.
# Returns a new file with format <label><index>.
def name_this_file(label):
    max_index = 0
    for file in os.listdir(os.getcwd() + '/data'):
    
        # Get next lowest index for this label
        if label in file:
            index = int(file[len('good'):file.find('.')])
            if index > max_index:
                max_index = index

    return label + str(max_index + 1)

# Saves a video with the given label.
def save_video(video, label):
    out = cv2.VideoWriter(label, video, 20.0, (640, 480))

if __name__ == '__main__':
    label = ''
    while label not in ('good', 'bad'):
        label = raw_input('LABEL FOR THIS DATASET (good / bad): ')
    label = name_this_file(label) + '.avi'

    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.getcwd() + '/data/' + label, fourcc, 20.0, (1280, 720))

    while True:
        frame = record(camera, label)
        out.write(frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy camera object and GUI
    camera.release()
    out.release()
    cv2.destroyAllWindows()
