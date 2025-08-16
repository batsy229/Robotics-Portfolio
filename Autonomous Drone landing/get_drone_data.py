from pymavlink import mavutil
import time
import math

### ✅ 1. Connect to Pixhawk 6C
pixhawk = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)  # Adjust port if needed
pixhawk.wait_heartbeat()  # Confirm connection

print("✅ Pixhawk 6C is communicating successfully!")

### ✅ 2. Read Live Drone Data
def get_drone_data():
    """Reads key telemetry data from Pixhawk 6C."""
    while True:
        msg = pixhawk.recv_match(blocking=True)

        if msg.get_type() == "GLOBAL_POSITION_INT":
            lat = msg.lat * 1e-7  # Convert to degrees
            lon = msg.lon * 1e-7
            alt = msg.alt * 1e-3  # Convert to meters
            print(f"📡 GPS: Lat={lat}, Lon={lon}, Alt={alt:.1f}m")

        elif msg.get_type() == "VFR_HUD":
            airspeed = msg.airspeed
            groundspeed = msg.groundspeed
            print(f"🚀 Speed: Air={airspeed:.1f}m/s, Ground={groundspeed:.1f}m/s")

        elif msg.get_type() == "ATTITUDE":
            roll = math.degrees(msg.roll)
            pitch = math.degrees(msg.pitch)
            yaw = math.degrees(msg.yaw)
            print(f"🎯 Attitude: Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}")

        elif msg.get_type() == "HEARTBEAT":
            mode = msg.custom_mode
            print(f"💡 Flight Mode: {mode}")

        time.sleep(1)  # Adjust frequency to limit excessive processing

### ✅ 3. Run the Telemetry Monitor
get_drone_data()