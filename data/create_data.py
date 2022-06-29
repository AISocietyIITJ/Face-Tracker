from concurrent.futures import process
import os, sys
sys.path.append(os.getcwd())
from assets.utils import shutil, time, imutils, cv2, json, Process


# config_file = 'configurations.json'

# with open(config_file) as load_value:
#     configurations = json.load(load_value)

DIR = 'data'
IMAGE_DIR = 'data_dir_1'
NAME = str(input("Enter Person's name: "))
IMAGE_NUM = int(input("Enter number of images: "))
NUM = open(f'{DIR}/class_num', encoding="utf-8").read()
ENCODINGS = f'{DIR}/encodings.json'

with open(DIR+'/'+"EVENT.json", encoding="utf-8") as event_begin:
    EVENT = json.load(event_begin)
    EVENT["EVENT"] = False
    json.dump(EVENT, open(DIR+'/'+"EVENT.json", 'w',encoding="utf-8"))

def run_camera():
    """
    Run the camera
    """
    print("[INFO] starting video stream...")
    # global DIR
    # global IMAGE_DIR
    # global NAME
    # global IMAGE_NUM
    # global NUM
    # global ENCODINGS

    with open(ENCODINGS, encoding="utf-8") as encodings:
        encoded = json.load(encodings)
    encoded[int(NUM)] = NAME

    try:
        os.mkdir(DIR + '/' + IMAGE_DIR + '/' + NUM)
        camera = cv2.VideoCapture(0)

        num_frames = 1
        left = 300
        right = 700
        top = 64
        bottom = 464

        print("[INFO] warming up...")
        while (True):

            global EVENT

            if not EVENT["EVENT"]:
                with open(DIR+'/'+"EVENT.json", encoding="utf-8") as event_check:
                    EVENT = json.load(event_check)

            (_, frame) = camera.read()

            frame = imutils.resize(frame, width = 640, height = 800)
            frame = cv2.flip(frame,1)

            clone = frame.copy()
            roi = frame[top:bottom, left:right]
            blur = cv2.GaussianBlur(roi, (7, 7), 0)

            # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            cv2.imshow("video feed", clone)
            cv2.imshow("ROI", blur)

            _ = cv2.waitKey(1)

            if EVENT["EVENT"]:

                if num_frames%50 == 0:
                    cv2.imwrite(
                        filename=f"{DIR}/{IMAGE_DIR}/{NUM}/image"+
                        str(int(num_frames/50))+".jpg",img = blur )
                    print(f"image_{int(num_frames/50)}.jpg saved")

                if num_frames == 50*IMAGE_NUM+1:
                    camera.release()
                    cv2.destroyAllWindows()
                    break

                num_frames += 1
    except KeyboardInterrupt:
        print("\n\n[INFO] exiting...")
        shutil.rmtree(DIR + '/' + IMAGE_DIR + '/' + NUM)
        open(f"{DIR}/class_num", 'w', encoding="utf-8").write(str(int(NUM)))
        sys.exit()
    except:
        print("Error")
        camera.release()
        cv2.destroyAllWindows()
        shutil.rmtree(DIR + '/' + IMAGE_DIR + '/' + NUM)
        raise

    json.dump(encoded, open(ENCODINGS, 'w', encoding="utf-8"))
    open(f"{DIR}/class_num", 'w', encoding="utf-8").write(str(int(NUM)+1))

def wait_response():
    """
    Wait for the user to press a key
    """
    global EVENT
    global NUM
    while not EVENT["EVENT"]:
        try:
            time.sleep(2)
            print("[INFO] starting in 5 seconds... Press 'ctrl+c' to quit")
            time.sleep(5)
            EVENT["EVENT"] = True
            print("[INFO] starting...")
            json.dump(EVENT, open(DIR+'/'+"EVENT.json", 'w', encoding="utf-8"))
            break
        except KeyboardInterrupt:
            print("\n\n[INFO] exiting...")
            shutil.rmtree(DIR + '/' + IMAGE_DIR + '/' + NUM)
            open(f"{DIR}/class_num", 'w', encoding="utf-8").write(str(int(NUM)))
            sys.exit()

if __name__ == "__main__":
    process_1 = Process(target=run_camera)
    process_1.start()
    process_2 = Process(target=wait_response)
    process_2.start()


# EOL
