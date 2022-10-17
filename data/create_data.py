"""
creating dataset
"""
from curses import has_colors
import os
import sys
import json
import time
import shutil
from multiprocessing import Process
import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
sys.path.append("../")


DIR = 'data'
NAME = str(input("Enter Gesture name: "))
IMAGE_NUM = int(input("Enter number of images: "))
NUM = open(f'{DIR}/class_num', encoding="utf-8").read()
ENCODINGS = f'{DIR}/encodings.json'

with open(DIR+'/'+"EVENT.json", encoding="utf-8") as event_file:
    EVENT = json.load(event_file)
    EVENT["EVENT"] = False
    json.dump(EVENT, open(DIR+'/'+"EVENT.json", 'w', encoding="utf-8"))


# To Generate data from live video feed
def run_camera():
    """
    This function runs the camera pipeline.
    """

    with open(ENCODINGS, encoding="utf-8") as encodings:
        encoded = json.load(encodings)
    encoded[int(NUM)] = NAME

    try:
        os.mkdir(DIR + '/dataset/' + NUM)
        camera = cv2.VideoCapture(0)

        num_frames = 1

        print("[INFO] warming up...")
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
            ) as face_detection:
            while camera.isOpened():

                global EVENT

                if not EVENT["EVENT"]:
                    with open(DIR+'/'+"EVENT.json", encoding="utf-8") as new_event:
                        EVENT = json.load(new_event)

                success, image = camera.read()
                h, w, _ = image.shape
                x_max = 0
                y_max = 0
                x_min = h
                y_min = w

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Detect Face
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_h, img_w, _ = image.shape
                cv2.imshow("video`  feed", cv2.flip(image,1))
                _ = cv2.waitKey(1)
                if results.detections:
                    for face_keypoints in results.detections:
                        x_min = min(x_min, face_keypoints.location_data.relative_bounding_box.xmin)
                        y_min = min(y_min, face_keypoints.location_data.relative_bounding_box.ymin)
                        h_ = max(x_max, face_keypoints.location_data.relative_bounding_box.height)
                        w_ = max(y_max, face_keypoints.location_data.relative_bounding_box.width)
                        x_1 = int((x_min)*img_w)
                        x_2 = int((x_min + w_)*img_w)
                        y_1 = int((y_min)*img_h)
                        y_2 = int((y_max + h_)*img_h)

                    cv2.rectangle(image,(x_1,y_1),(x_2,y_2),(0, 255, 0), 2)

                    # Crop the image
                    roi = image[y_1:y_2, x_1:x_2]
                    try:
                        roi = cv2.resize(roi, (224, 224))
                        if EVENT["EVENT"]:

                            # Saving the image
                            if num_frames%5 == 0:
                                cv2.imwrite(filename=f"{DIR}/dataset/{NUM}/image"+
                                str(int(num_frames/5))+".jpg",img = roi )
                                print(f"image_{int(num_frames/5)}.jpg saved")

                            # Termination Condition
                            if num_frames == 5*IMAGE_NUM+1:
                                camera.release()
                                cv2.destroyAllWindows()
                                break

                            num_frames += 1

                        cv2.imshow('Hand', cv2.flip(roi, 1))
                        _ = cv2.waitKey(1)
                    except SystemExit:
                        continue

    except KeyboardInterrupt:
        print("\n\n[INFO] exiting...")
        shutil.rmtree(DIR + '/dataset/' + NUM)
        open(f"{DIR}/class_num", 'w', encoding="utf-8").write(str(int(NUM)))
        sys.exit()
    except:
        print("Error")
        camera.release()
        cv2.destroyAllWindows()
        shutil.rmtree(DIR + '/dataset/' + NUM)
        raise

    json.dump(encoded, open(ENCODINGS, 'w', encoding="utf-8"))
    open(f"{DIR}/class_num", 'w',encoding="utf-8").write(str(int(NUM)+1))

def wait_response():
    """
    This function waits for the response from the user.
    """
    global EVENT
    global NUM
    while not EVENT["EVENT"]:
        try:
            # Waiting for the user to press a key.
            time.sleep(2)
            print("[INFO] starting in 5 seconds... Press 'ctrl+c' to quit")
            time.sleep(5)
            EVENT["EVENT"] = True
            print("[INFO] starting...")
            json.dump(EVENT, open(DIR+'/'+"EVENT.json", 'w', encoding="utf-8"))
            break
        except KeyboardInterrupt:
            print("\n\n[INFO] exiting...")
            shutil.rmtree(DIR + '/dataset/' + NUM)
            open(f"{DIR}/class_num", 'w', encoding='utf-8').write(str(int(NUM)))
            sys.exit()

if __name__ == "__main__":

    # Thread for the camera
    p1 = Process(target=run_camera)
    p1.start()
    # Thread for the wait_response
    p2 = Process(target=wait_response)
    p2.start()
    try:
        p1.join()
    except SystemExit:
        pass
    try:
        p2.join()
    except SystemExit:
        pass

# EOL
