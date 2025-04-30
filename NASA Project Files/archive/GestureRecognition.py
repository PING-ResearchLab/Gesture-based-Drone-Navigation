import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
 
 
# Initialize the low-level drivers
mouse = Controller()
 
 
 
screen_width, screen_height = pyautogui.size()
 
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)
 

 
def rock(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[0], landmark_list[5], landmark_list[6]) > 120 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        #util.get_distance([landmark_list[8], landmark_list[5]]) > 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 130 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 130 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[17], landmark_list[18]) > 120
        #thumb_index_dist > 50
    )
 
def hand_open(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[0], landmark_list[5], landmark_list[6]) > 120 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        #util.get_distance([landmark_list[8], landmark_list[5]]) > 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 120 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) > 120 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 120 and
        thumb_index_dist > 50
    )
 
def three_fing(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[0], landmark_list[5], landmark_list[6]) > 120 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        #util.get_distance([landmark_list[8], landmark_list[5]]) > 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 120 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[17], landmark_list[20]) < 120
    )
 
def hand_closed(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[7]) < 130 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 130 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 130 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 130 and
        util.get_distance([landmark_list[4], landmark_list[17]]) < 225
    )
 
def thumb_up(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[7]) < 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 100 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 100 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 100 and
        util.get_distance([landmark_list[4], landmark_list[17]]) > 225
    )
 
def point(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[0], landmark_list[5], landmark_list[6]) > 120 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 100 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 100 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 100 #and
        #util.get_angle(landmark_list[1], landmark_list[2], landmark_list[3]) < 100
        #thumb_index_dist > 50
    )
 
def peace(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[0], landmark_list[5], landmark_list[6]) > 120 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[9], landmark_list[10]) > 120 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 120 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 100 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 100 #and
        #util.get_angle(landmark_list[1], landmark_list[2], landmark_list[3]) < 100
        #thumb_index_dist > 50
    )
 
def fudge(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 100 and
        util.get_angle(landmark_list[0], landmark_list[9], landmark_list[10]) > 120 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 120 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 100 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 100
        #thumb_index_dist > 50
    )
 
def nice(landmark_list, thumb_index_dist):
    return (
         util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 100 and
        util.get_angle(landmark_list[0], landmark_list[9], landmark_list[10]) > 120 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[13], landmark_list[14]) > 120 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[17], landmark_list[18]) > 120 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 120 and
        util.get_distance([landmark_list[4], landmark_list[8]]) < 70
        #util.get_angle(landmark_list[1], landmark_list[2], landmark_list[3]) < 100
        #thumb_index_dist > 50
    )
 
 
 
#def is_right_click(landmark_list, thumb_index_dist):
#    return (
#            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
#            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
#            thumb_index_dist > 50
#    )
 
 
#def is_double_click(landmark_list, thumb_index_dist):
#    return (
#            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
#            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
#            thumb_index_dist > 50
#    )
 
 
#def is_screenshot(landmark_list, thumb_index_dist):
#    return (
#            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
#            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
#            thumb_index_dist < 50
#    )
 
 
def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
 
        if hand_open(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Hand Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif hand_closed(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Hand Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif point(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Point", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif peace(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif fudge(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Fudge", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif nice(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Nice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif thumb_up(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif three_fing(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Three", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif rock(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Land", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
 
#        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50  and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
#            move_mouse(index_finger_tip)
#        elif is_left_click(landmark_list,  thumb_index_dist):
#            mouse.press(Button.left)
#            mouse.release(Button.left)
#            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#        elif is_right_click(landmark_list, thumb_index_dist):
#            mouse.press(Button.right)
#            mouse.release(Button.right)
#            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#        elif is_double_click(landmark_list, thumb_index_dist):
#            pyautogui.doubleClick()
#            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#        elif is_screenshot(landmark_list,thumb_index_dist ):
#            im1 = pyautogui.screenshot()
#            label = random.randint(1, 1000)
#            im1.save(f'my_screenshot_{label}.png')
#            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
 
def main():
    import argparse
    import time
    import socket,os,struct, time
    import numpy as np

    # Args for setting IP/port of AI-deck. Default settings are for when
    # AI-deck is in AP mode.
    parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
    parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
    parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
    parser.add_argument('--save', action='store_true', help="Save streamed images")
    args = parser.parse_args()

    deck_port = args.p
    deck_ip = args.n

    print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((deck_ip, deck_port))
    print("Socket connected")

    imgdata = None
    data_buffer = bytearray()

    def rx_bytes(size):
        data = bytearray()
        while len(data) < size:
            data.extend(client_socket.recv(size-len(data)))
        return data

    import cv2

    start = time.time()
    count = 0

    # while(1):
    #     # First get the info
    #     packetInfoRaw = rx_bytes(4)
    #     #print(packetInfoRaw)
    #     [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
    #     #print("Length is {}".format(length))
    #     #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
    #     #print("Function is 0x{:02X}".format(function))

    #     imgHeader = rx_bytes(length - 2)
    #     #print(imgHeader)
    #     #print("Length of data is {}".format(len(imgHeader)))
    #     [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)
    #     if magic == 0xBC:
    #         #print("Magic is good")
    #         #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
    #         #print("Image format is {}".format(format))
    #         #print("Image size is {} bytes".format(size))

    #         # Now we start rx the image, this will be split up in packages of some size
    #         imgStream = bytearray()

    #         while len(imgStream) < size:
    #             packetInfoRaw = rx_bytes(4)
    #             [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
    #             #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
    #             chunk = rx_bytes(length - 2)
    #             imgStream.extend(chunk)

    #         count = count + 1
    #         meanTimePerImage = (time.time()-start) / count
    #         print("{}".format(meanTimePerImage))
    #         print("{}".format(1/meanTimePerImage))

    #         if format == 0:
    #             bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
    #             bayer_img.shape = (244, 324)
    #             color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
    #             cv2.imshow('Raw', bayer_img)
    #             cv2.imshow('Color', color_img)
    #             if args.save:
    #                 cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
    #                 cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
    #             cv2.waitKey(1)
    #         else:
    #             with open("img.jpeg", "wb") as f:
    #                 f.write(imgStream)
    #             nparr = np.frombuffer(imgStream, np.uint8)
    #             decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
    #             cv2.imshow('JPEG', decoded)
    #             cv2.waitKey(1)


    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():

            # First get the info
            packetInfoRaw = rx_bytes(4)
            #print(packetInfoRaw)
            [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
            #print("Length is {}".format(length))
            #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
            #print("Function is 0x{:02X}".format(function))

            imgHeader = rx_bytes(length - 2)
            #print(imgHeader)
            #print("Length of data is {}".format(len(imgHeader)))
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)
            if magic == 0xBC:
                #print("Magic is good")
                #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
                #print("Image format is {}".format(format))
                #print("Image size is {} bytes".format(size))

                # Now we start rx the image, this will be split up in packages of some size
                imgStream = bytearray()

                while len(imgStream) < size:
                    packetInfoRaw = rx_bytes(4)
                    [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                    #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
                    chunk = rx_bytes(length - 2)
                    imgStream.extend(chunk)

                count = count + 1
                meanTimePerImage = (time.time()-start) / count
                print("{}".format(meanTimePerImage))
                print("{}".format(1/meanTimePerImage))

                if format == 0:
                    bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
                    bayer_img.shape = (244, 324)
                    color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                    color_img[:, :, 0] = cv2.multiply(color_img[:, :, 0], 1.0)  # Boost blue
                    color_img[:, :, 1] = cv2.multiply(color_img[:, :, 1], 0.7)  # Keep green neutral
                    color_img[:, :, 2] = cv2.multiply(color_img[:, :, 2], 0.8)  # Reduce red
                    color_img = cv2.convertScaleAbs(color_img, alpha=1.5, beta=0)  # Increase brightness
                    cv2.imshow('Raw', bayer_img)
                    cv2.imshow('Color', color_img)
                    if args.save:
                        cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                        cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
                    cv2.waitKey(1)
                else:
                    with open("img.jpeg", "wb") as f:
                        f.write(imgStream)
                    nparr = np.frombuffer(imgStream, np.uint8)
                    decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
                    cv2.imshow('JPEG', decoded)
                    cv2.waitKey(1)

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
 
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))
 
            # Detect gestures
            detect_gesture(frame, landmark_list, processed)
 
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()