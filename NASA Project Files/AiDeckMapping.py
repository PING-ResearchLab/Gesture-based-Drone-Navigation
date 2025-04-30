import cv2
import mediapipe as mp
import logging
import sys
import pyautogui
import random
import util
import cv2
import time
import cflib
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger
 
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd  # New import for creating the table
import plotly.graph_objs as go
from cflib.crazyflie.log import LogConfig

uri = 'radio://0/80/2M'

x = 0.0
y = 0.0
z = 0.0

# Initialize the low-level drivers
cflib.crtp.init_drivers()
logging.basicConfig(level=logging.ERROR)
 
 
screen_width, screen_height = pyautogui.size()
 
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
 
def is_close(range):
    MIN_DISTANCE = 0.35  # m
 
    if range is None:
        return False
    else:
        return range < MIN_DISTANCE
    
# Variables to store position data for plotting and table
x_positions = [0]
y_positions = [0]
z_positions = [0]
coordinates = []  # List to store coordinate data for the table
 
# Variables to store detection points
detection_x_positions = []
detection_y_positions = []
detection_z_positions = []
 
# Create a 3D plot for tracking the drone's path
trace = go.Scatter3d(x=x_positions, y=y_positions, z=z_positions, mode='lines+markers', name='Path', line=dict(width=2), marker=dict(size=2))
 
# Create an empty obstacle trace to store columns dynamically
obstacle_traces = []

# Function to add detection points to the plot
def add_detection_point(x, y, z):
    detection_x_positions.append(x)
    detection_y_positions.append(y)
    detection_z_positions.append(z)

# Function to add a column representing an obstacle
def add_obstacle_column(x, y, z_start=0, height=0.8):
    """
    Adds a column at position (x, y) with a starting z value of z_start and height of `height`.
    """
    # Define the base and top coordinates of the column
    # x_coords = [x, x, x + 0.1, x + 0.1, x]
    # y_coords = [y, y + 0.1, y + 0.1, y, y]
    # z_coords = [z_start, z_start, z_start, z_start, z_start + height]  # From z=0 to z=height
 
# Define the four corner points of the base
    x_corners = [x, x + 0.3, x + 0.3, x]
    y_corners = [y, y, y + 0.3, y + 0.3]
    z_base = [z_start] * 4  # Bottom face
    z_top = [z_start + height] * 4  # Top face
 
    # Combine into a single list for `Mesh3d`
    x_all = x_corners + x_corners  # Base + Top
    y_all = y_corners + y_corners
    z_all = z_base + z_top
 
    # Define the triangles (faces of the column)
    column = go.Mesh3d(
        x=x_all,
        y=y_all,
        z=z_all,
        i=[0, 1, 2, 0, 2, 3,  # Bottom face
        4, 5, 6, 4, 6, 7,  # Top face
        0, 1, 5, 0, 5, 4,  # Side face 1
        1, 2, 6, 1, 6, 5,  # Side face 2
        2, 3, 7, 2, 7, 6,  # Side face 3
        3, 0, 4, 3, 4, 7],  # Side face 4
        j=[1, 2, 3, 0, 3, 2,
        5, 6, 7, 4, 7, 6,
        1, 5, 4, 0, 1, 4,
        2, 6, 5, 1, 2, 5,
        3, 7, 6, 2, 3, 6,
        0, 4, 7, 3, 0, 7],
        k=[2, 3, 0, 3, 0, 1,
        6, 7, 4, 7, 4, 5,
        5, 4, 0, 1, 5, 0,
        6, 5, 1, 2, 6, 1,
        7, 6, 2, 3, 7, 2,
        4, 7, 3, 0, 4, 3],
        color='red',
        opacity=0.6,
        name=f'Obstacle at ({x:.2f}, {y:.2f})'
    )
 
    # Append the column to the obstacle traces
    obstacle_traces.append(column)
 
    # Add the column to the figure
    fig.add_trace(column)
 
layout = go.Layout(title='Crazyflie 3D Path', scene=dict(
    xaxis=dict(range=[-1, 4], title='X Position (m)'),
    yaxis=dict(range=[-1, 1], title='Y Position (m)'),
    zaxis=dict(range=[0, 1], title='Z Position (m)')
))
fig = go.Figure(data=[trace], layout=layout)


# Function to update the 3D plot with new positions and dynamically change axis limits
def update_plot():
    global fig
    fig.data[0].x = x_positions
    fig.data[0].y = y_positions
    fig.data[0].z = z_positions
    # fig.data[1].x = detection_x_positions
    # fig.data[1].y = detection_y_positions
    # fig.data[1].z = detection_z_positions
    
    # Get current axis limits
    x_min, x_max = fig.layout.scene.xaxis.range
    y_min, y_max = fig.layout.scene.yaxis.range
    z_min, z_max = fig.layout.scene.zaxis.range
 
    # Check if the drone's position is outside the current limits and update limits dynamically
    if x_positions[-1] > x_max or x_positions[-1] < x_min:
        fig.layout.scene.xaxis.range = [min(x_positions[-1], x_min) - 1, max(x_positions[-1], x_max) + 1]
 
    if y_positions[-1] > y_max or y_positions[-1] < y_min:
        fig.layout.scene.yaxis.range = [min(y_positions[-1], y_min) - 1, max(y_positions[-1], y_max) + 1]
 
    if z_positions[-1] > z_max or z_positions[-1] < z_min:
        fig.layout.scene.zaxis.range = [min(z_positions[-1], z_min) - 0.5, max(z_positions[-1], z_max) + 0.5]
 
# Log configuration for the flow deck
log_conf = LogConfig(name='Position', period_in_ms=100)
log_conf.add_variable('kalman.stateX', 'float')
log_conf.add_variable('kalman.stateY', 'float')
log_conf.add_variable('kalman.stateZ', 'float')
 
 
# Variable to store the initial timestamp
initial_timestamp = None
 
# Log callback function to capture and plot position data
def log_pos_callback(timestamp, data, logconf):
    global initial_timestamp
    if initial_timestamp is None:
        initial_timestamp = timestamp  # Store the first timestamp as the initial one
 
    # Calculate the offset timestamp starting from zero
    adjusted_timestamp = timestamp - initial_timestamp
 
    # Get X, Y, and Z positions from the Kalman filter (flow deck data)
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    x_positions.append(x)
    y_positions.append(y)
    z_positions.append(z)
    
    # Append the adjusted timestamp and coordinates to the list for the table
    coordinates.append({'Timestamp': adjusted_timestamp, 'X': x, 'Y': y, 'Z': z})
    
    # Update the plot each time new data comes in
    update_plot()
 


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
        util.get_angle(landmark_list[0], landmark_list[17], landmark_list[19]) > 120
    )
 
def hand_closed(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[7]) < 100 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[11]) < 100 and
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[15]) < 100 and
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[19]) < 100 and
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
 
#turning is mc.turn_left(180)
 
def detect_gesture(frame, landmark_list, processed, mc):
    if len(landmark_list) >= 21:
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
 
        if hand_open(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.up(0.05)
        elif hand_closed(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.down(0.05)
        elif point(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.forward(0.05)  
        elif peace(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Back", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.back(0.05)
        elif fudge(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Turn Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.turn_left(45)
        elif nice(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Land", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.land()
            log_conf.stop()
            # Save the interactive 3D plot as an HTML file
            pio.write_html(fig, file='crazyflie_3d_path.html', auto_open=True)

        elif thumb_up(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.right(0.05)  
        elif rock(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Turn Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.turn_right(45)
        elif pinky(landmark_list, thumb_index_dist):
            cv2.putText(frame, "Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mc.left(0.05)
 
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

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
 

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(uri, cf=cf) as scf:
        with MotionCommander(scf) as mc:
            with Multiranger(scf) as multi_ranger:
                scf.cf.log.add_config(log_conf)
                log_conf.data_received_cb.add_callback(log_pos_callback)
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
                            #print("{}".format(meanTimePerImage))
                            #print("{}".format(1/meanTimePerImage))

                            if format == 0:
                                bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
                                bayer_img.shape = (244, 324)
                                color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
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




                        log_conf.start()


                        if is_close(multi_ranger.front):
                            mc.back(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            add_obstacle_column(x_positions[-1]+.35, y_positions[-1]-.15, z_start=0, height=0.8)  # Add a column on the map
                        elif is_close(multi_ranger.back):
                            mc.forward(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            add_obstacle_column(x_positions[-1]-.35, y_positions[-1]-.15, z_start=0, height=0.8)  # Add a column on the map
                        elif is_close(multi_ranger.left):
                            mc.right(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            add_obstacle_column(x_positions[-1], y_positions[-1]+.5, z_start=0, height=0.8)  # Add a column on the map
                        elif is_close(multi_ranger.right):
                            mc.left(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            add_obstacle_column(x_positions[-1], y_positions[-1]-.5, z_start=0, height=0.8)  # Add a column on the map
                        elif is_close(multi_ranger.up):
                            mc.down(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            add_obstacle_column(x_positions[-1], y_positions[-1], z_start=0, height=0.8)  # Add a column on the map
                        elif is_close(multi_ranger.down):
                            mc.up(0.2)
                            # Add detection point to the plot
                            add_detection_point(x, y, z)
                            #add_obstacle_column(x+multi_ranger.down, y, z_start=0, height=0.8)  # Add a column on the map
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
 
                        # Detect gestures and control Crazyflie
                        detect_gesture(frame, landmark_list, processed, mc)
                        
                        cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()
 