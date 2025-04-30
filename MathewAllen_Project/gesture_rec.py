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
 
 
#def find_finger_tip(processed):
#   if processed.multi_hand_landmarks:
#        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
#        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
#        middle_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
#        ring_finger_tip=hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP]
#        pinky_tip=hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP]
#        return index_finger_tip
#    return None, None
 
 
#def move_mouse(index_finger_tip):
#    if index_finger_tip is not None:
#        x = int(index_finger_tip.x * screen_width)
#        y = int(index_finger_tip.y / 2 * screen_height)
#        pyautogui.moveTo(x, y)
 
 
#def is_left_click(landmark_list, thumb_index_dist):
#    return (
#            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
#            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
#            thumb_index_dist > 50
#    )
 
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
 
def pinky(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 130 and
        #util.get_distance([landmark_list[8], landmark_list[5]]) > 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 130 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 130 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) > 120 and
        util.get_angle(landmark_list[0], landmark_list[17], landmark_list[18]) > 120
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
            cv2.putText(frame, "Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif hand_closed(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif point(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Foward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif peace(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Back", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif fudge(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Fudge", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif nice(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Land", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif thumb_up(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif pinky(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Right", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif rock(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Spin", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
 
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
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
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
