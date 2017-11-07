cmake_minimum_required(VERSION 3.1)
project(face_landmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC -DCPU_ONLY -fopenmp")
 
add_definitions(-Wall)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(PROJECT_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    )

set(SOURCE_FILES
    main.cpp
    fp_detector.cc
    )
set(PROJECT_LIBS
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui
    opencv_dnn
    boost_system
    glog
    cblas
    lapack
    dlib
    ${PROJECT_SOURCE_DIR}/lib/libcaffe.so
    )

find_package(OpenCV)
if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Test will not be built.")
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND facedet_required_libs ${OpenCV_LIBS})
endif()

include_directories(${PROJECT_INCLUDE_DIRS}) 
add_executable(face_landmark ${SOURCE_FILES})
target_link_libraries(face_landmark ${PROJECT_LIBS})