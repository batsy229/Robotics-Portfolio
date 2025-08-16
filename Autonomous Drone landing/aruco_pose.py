import numpy as np
import cv2
import cv2.aruco as arucoSqS
import time
import math
from picamera2 import Picamera2
import pickle


class ArucoSingleTracker():
    def __init__(self,
                 id_to_find,
                 marker_size,
                 camera_matrix,
                 camera_distortion,
                 camera_size=[2028, 1520],
                 show_video=False):

        self.id_to_find = id_to_find
        self.marker_size = marker_size
        self._show_video = show_video
        self._camera_matrix = camera_matrix
        self._camera_distortion = camera_distortion
        self.is_detected = False
        self._kill = False

        # Initialize Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(main={"size": (2028, 1520), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()

        # 180 degree rotation matrix around the x-axis (adjusts orientation alignment)
        self._R_flip = np.eye(3, dtype=np.float32)
        self._R_flip[1, 1] = -1
        self._R_flip[2, 2] = -1

        # Define the ArUco dictionary
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._parameters = aruco.DetectorParameters_create()

        # Font for text display
        self.font = cv2.FONT_HERSHEY_PLAIN

    def _rotationMatrixToEulerAngles(self, R):
        """Calculates rotation matrix to Euler angles."""
        assert (np.allclose(np.dot(R.T, R), np.identity(3))), "R is not a valid rotation matrix"

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def track(self, loop=True, verbose=False, show_video=None):

        self._kill = False
        if show_video is None:
            show_video = self._show_video

        marker_found = False
        x = y = z = 0

        while not self._kill:

            # Capture a frame from Picamera2
            frame = self.picam2.capture_array()

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find all the ArUco markers in the image
            corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=self._aruco_dict,
                                                         parameters=self._parameters)

            if ids is not None and self.id_to_find in ids:
                marker_found = True

                # Estimate pose of the detected marker
                ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self._camera_matrix,
                                                      self._camera_distortion)

                # Unpack the output, get only the first detected marker's rotation and translation vectors
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

                x, y, z = tvec[0], tvec[1], tvec[2]

                # Draw the detected marker and put a reference frame over it
                aruco.drawDetectedMarkers(frame, corners)

                # Use cv2.drawFrameAxes with a slightly larger axis length
                cv2.drawFrameAxes(frame, self._camera_matrix, self._camera_distortion, rvec, tvec, self.marker_size)

                # Obtain the rotation matrix from the rotation vector
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc = R_ct.T

                # Apply the R_flip to adjust orientation alignment
                R_aligned = self._R_flip * R_tc

                # Get the attitude in terms of Euler angles
                roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(R_aligned)

                # Print the position and attitude of the marker relative to the camera
                if verbose:
                    print("Marker Position: x={:.1f} cm  y={:.1f} cm  z={:.1f} cm".format(x, y, z))
                    print("Marker Attitude: roll={:.1f}°  pitch={:.1f}°  yaw={:.1f}°".format(
                        math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)))

                if show_video:
                    # Display position and attitude on the frame
                    cv2.putText(frame, f"Position x={x:.1f} y={y:.1f} z={z:.1f}", (10, 30), self.font, 1, (0, 255, 0),
                                2, cv2.LINE_AA)
                    cv2.putText(frame,
                                f"Attitude roll={math.degrees(roll_marker):.1f} pitch={math.degrees(pitch_marker):.1f} yaw={math.degrees(yaw_marker):.1f}",
                                (10, 60), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                if verbose:
                    print("No marker detected")

            if show_video:
                # Display the frame
                cv2.imshow("frame", frame)

                # Use 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            if not loop:
                return marker_found, x, y, z

        # Stop the Picamera2
        self.picam2.stop()


if __name__ == "__main__":
    # Define the marker ID and size (150mm)
    id_to_find = 69
    marker_size = 15  # in centimeters

    # Load camera calibration parameters
    with open('/home/goku/python/cameraMatrix.pkl', 'rb') as f:
        camera_matrix = pickle.load(f)
    with open('/home/goku/python/dist.pkl', 'rb') as f:
        camera_distortion = pickle.load(f)

    # Initialize the tracker with ArUco marker ID and size
    aruco_tracker = ArucoSingleTracker(id_to_find=id_to_find,
                                       marker_size=marker_size,
                                       camera_matrix=camera_matrix,
                                       camera_distortion=camera_distortion,
                                       show_video=True)

    # Run the tracking
    aruco_tracker.track(verbose=True)
