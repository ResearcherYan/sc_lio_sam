import rospy
from sensor_msgs.msg import Imu
import numpy as np

# Global variables
w_x = []
w_y = []
w_z = []
a_x = []
a_y = []
a_z = []
duration = 10


def imu_cb(imu_data, begin_time):
  # # Read the quaternion of the robot IMU
  # x = imu_data.orientation.x
  # y = imu_data.orientation.y
  # z = imu_data.orientation.z
  # w = imu_data.orientation.w

  # # Read the angular velocity of the robot IMU
  # w_x = imu_data.angular_velocity.x
  # w_y = imu_data.angular_velocity.y
  # w_z = imu_data.angular_velocity.z

  # # Read the linear acceleration of the robot IMU
  # a_x = imu_data.linear_acceleration.x
  # a_y = imu_data.linear_acceleration.y
  # a_z = imu_data.linear_acceleration.z

  w_x.append(imu_data.angular_velocity.x)
  w_y.append(imu_data.angular_velocity.y)
  w_z.append(imu_data.angular_velocity.z)

  a_x.append(imu_data.linear_acceleration.x)
  a_y.append(imu_data.linear_acceleration.y)
  a_z.append(imu_data.linear_acceleration.z)

  if rospy.get_time() - begin_time > duration:
    rospy.signal_shutdown("Data collection over!")

  return


if __name__ == '__main__':
  rospy.init_node('imu_node', anonymous=True)
  begin_time = rospy.get_time()
  print("Begin time: ", begin_time)
  rospy.Subscriber("/handsfree/imu", Imu, imu_cb, begin_time)
  rospy.spin()

  print("End time: ", rospy.get_time())
  print("Data collection over!")
  np_w_x = np.array(w_x)
  np_w_y = np.array(w_y)
  np_w_z = np.array(w_z)
  mean_w_x = np_w_x.mean()
  mean_w_y = np_w_y.mean()
  mean_w_z = np_w_z.mean()

  np_a_x = np.array(a_x)
  np_a_y = np.array(a_y)
  np_a_z = np.array(a_z)
  mean_a_x = np_a_x.mean()
  mean_a_y = np_a_y.mean()
  mean_a_z = np_a_z.mean()

  print("mean_w_x: ", mean_w_x)
  print("mean_w_y: ", mean_w_y)
  print("mean_w_z: ", mean_w_z)
  print("mean_a_x: ", mean_a_x)
  print("mean_a_y: ", mean_a_y)
  print("mean_a_z: ", mean_a_z)
