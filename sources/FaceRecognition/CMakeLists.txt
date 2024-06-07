# Минимальная версия CMake
cmake_minimum_required(VERSION 3.15)

# Имя библиотеки
project(face_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Источники
set(SOURCES
    FaceRecognition.cpp
)

set(HEADERS
    FaceRecognition.hpp
)

# Создание статической библиотеки
add_library(face_recognition STATIC ${SOURCES} ${HEADERS})

# Включение директорий для поиска заголовочных файлов
target_include_directories(face_recognition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Линковка с внешними библиотеками
target_link_libraries(face_recognition ${OpenCV_LIBS} dlib::dlib)