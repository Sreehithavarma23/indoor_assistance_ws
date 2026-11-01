cmake_minimum_required(VERSION 3.8)
project(indoor_assistance_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  urdf
  worlds
  config
  maps
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/go_to_location.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
