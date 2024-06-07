# Минимальная версия CMake
cmake_minimum_required(VERSION 3.15)

# Имя проекта
project(CV-SAS VERSION 0.1 LANGUAGES CXX)

# Включаем генерацию файла компиляции
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Находим OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dlib
find_package(dlib REQUIRED)
include_directories(${DLIB_INCLUDE_DIRS})

add_subdirectory(sources/FaceRecognition)

# Исполняемый файл
add_executable(${PROJECT_NAME} sources/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} dlib::dlib face_recognition)