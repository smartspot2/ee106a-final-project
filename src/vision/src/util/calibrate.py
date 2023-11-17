"""
Calibration of the camera using an AR tag.
"""
import tf2_ros
import rospy

def calibrate(marker_name):
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer, queue_size=10)

    try:
        transform = buffer.lookup_transform(marker_name, "usb_cam", rospy.Time())

        # TODO
    except tf2_ros.LookupException as e:
        print("Exception", e)