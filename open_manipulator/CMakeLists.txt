cmake_minimum_required(VERSION 3.5)
project(open_manipulator)
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  follow_camera.py
  camera_calibration.py
  follow_with_camera.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
