import cv2
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Dash
from dash.dependencies import Output, Input
import threading
import time
from collections import deque

# Global variables
positions = deque(maxlen=100)  # Store up to 100 LED positions (x, y, z)
keep_running = True  # Control flag for threads

# Calibration factor for pixel-to-meter conversion
CALIBRATION_FACTOR = 0.01  # Example: 1 pixel = 0.01 meters (adjust as needed)

# Fixed bounding box size
BOX_SIZE = 50  # Fixed size for the bounding box around the detected fire
prev_center = None  # Store the previous center position for smooth transitions

# Dash app for Plotly visualization
app = Dash(__name__)

# Initial layout for the Dash app
app.layout = html.Div([
    html.H1("Brightest RGB LED Trajectory (in Meters)", style={"text-align": "center"}),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=500, n_intervals=0)  # Update every 500ms
])

# Dash callback to update the graph dynamically
@app.callback(
    Output("live-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_graph(n_intervals):
    global positions

    # Extract x, y, z coordinates from positions
    x_vals = [pos[0] * CALIBRATION_FACTOR for pos in positions]
    y_vals = [pos[1] * CALIBRATION_FACTOR for pos in positions]
    z_vals = [pos[2] for pos in positions]  # Assuming z is already in meters

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=np.linspace(0, 1, len(z_vals)),  # Gradient coloring
            colorscale='Viridis',
            opacity=0.8
        ),
        line=dict(
            color='blue',
            width=2
        )
    )])

    # Set plot layout
    fig.update_layout(
        title="Brightest RGB LED Trajectory (in Meters)",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        )
    )

    return fig


def detect_brightest_led(frame, threshold=200):
    """
    Detects the brightest RGB LED while excluding white light.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 150, threshold])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 150, threshold])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 150, threshold])
    upper_blue = np.array([140, 255, 255])

    # Create masks for red, green, and blue
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brightest = None
    brightest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > brightest_area:  # Keep the largest area (brightest object)
            brightest_area = area
            brightest = contour

    if brightest is not None:
        x, y, w, h = cv2.boundingRect(brightest)
        x_center = x + w // 2
        y_center = y + h // 2
        return (x_center, y_center)

    return None


def log_led_positions():
    """
    Logs the positions of the brightest RGB LED for 3D plotting.
    """
    global positions, keep_running, prev_center
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    try:
        while keep_running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            center = detect_brightest_led(frame)
            
            if center is not None:
                prev_center = center if prev_center is None else (
                    int(0.8 * prev_center[0] + 0.2 * center[0]),
                    int(0.8 * prev_center[1] + 0.2 * center[1])
                )
                x_center, y_center = prev_center
                z = 1.0  # Simulated depth in meters
                positions.append((x_center, y_center, z))
                
                # Define fixed bounding box size
                half_size = BOX_SIZE // 2
                x1, y1 = int(x_center - half_size), int(y_center - half_size)
                x2, y2 = int(x_center + half_size), int(y_center + half_size)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Brightest RGB LED Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_running = False
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Start LED detection in a separate thread
        logging_thread = threading.Thread(target=log_led_positions)
        logging_thread.start()

        # Start the Dash app
        app.run_server(debug=False, use_reloader=False)

        # Wait for the logging thread to complete
        logging_thread.join()

    except KeyboardInterrupt:
        print("Exiting...")
        keep_running = False
        logging_thread.join()
