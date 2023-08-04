cmake_minimum_required(VERSION 3.7)
project(fable LANGUAGES C)
find_package(OpenCL REQUIRED)
add_executable(${PROJECT_NAME} src/vec_add.c)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL m)

