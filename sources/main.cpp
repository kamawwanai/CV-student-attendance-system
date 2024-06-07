/*
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

using namespace dlib; //NOLINT
using namespace std; //NOLINT

// Определение модели нейронной сети для извлечения лицевых признаков
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


class Models {
public:
    Models() {
        try {
            dlib::deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;
        } catch (const std::exception& e) {
            std::cout << "Error loading shape_predictor_68_face_landmarks.dat" << std::endl;
            std::cerr << "Error loading shape_predictor_68_face_landmarks.dat: " << e.what() << std::endl;
            throw;
        }

        try {
            dlib::deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;
        } catch (const std::exception& e) {
            std::cout << "Error loading dlib_face_recognition_resnet_model_v1.dat" << std::endl;
            std::cerr << "Error loading dlib_face_recognition_resnet_model_v1.dat: " << e.what() << std::endl;
            throw;
        }
    }

    dlib::shape_predictor sp;
    anet_type net;
};


dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

// Функция для извлечения дескрипторов лиц
auto getFaceDescriptors(const cv::Mat &frame, Models& models,
                        std::vector<cv::Point>& left_corners)
                        -> std::vector<dlib::matrix<float, 0, 1>> {
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    
    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    std::vector<dlib::rectangle> faces = detector(cimg);

    for (auto face : faces) {
        dlib::full_object_detection shape = models.sp(cimg, face);
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
        face_descriptors.push_back(models.net(face_chip));
        cv::rectangle(frame, 
                      cv::Point(face.left(), face.top()), 
                      cv::Point(face.right(), face.bottom()), 
                      cv::Scalar(0, 255, 0), 2);
        left_corners.emplace_back(face.left(), face.top());
    }
    return face_descriptors;
}


void getKnownFacesDescriptors(std::map<std::string, dlib::matrix<float, 0, 1>>& known_faces_descriptors,
                            const std::vector<std::string>& names, Models& models) {
        std::vector<cv::Point> p;
        for (const auto& name: names) {
            std::string path = "../data/" + name + ".jpg";
            cv::Mat person_img = cv::imread(path);
            known_faces_descriptors[name] = getFaceDescriptors(person_img, models, p)[0];
        }
    }


auto identifyFace(const dlib::matrix<float, 0, 1> &descriptor, const std::map<std::string, dlib::matrix<float, 0, 1>>& known_faces) -> std::string {
    std::string identified_name = "Unknown";
    double min_distance = 0.6; // Пороговое значение расстояния для идентификации

    for (const auto &known_face : known_faces) {
        double distance = dlib::length(known_face.second - descriptor);
        if (distance < min_distance) {
            min_distance = distance;
            identified_name = known_face.first;
        }
    }

    return identified_name;
}


auto main() -> int {
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        return -1;
    }

    Models models;

    std::vector<std::string> names {"harry_potter", "ksenia_karimova", "lady_gaga",
                                    "sergey_razumovskiy", "thimotee_chamolete", "dima_kutuzov"};

    

    std::map<std::string, dlib::matrix<float, 0, 1>> known_faces_descriptors;
    getKnownFacesDescriptors(known_faces_descriptors, names, models);


    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    
    while (true) {
        cv::Mat frame;
        cap >> frame; // Чтение кадра из камеры

        if (frame.empty()) { break; }
        
        std::vector<cv::Point> left_corners;
        std::vector<dlib::matrix<float, 0, 1>> face_descriptors = getFaceDescriptors(frame, models, left_corners);

        int temp = 0;
        for (const auto &descriptor : face_descriptors) {
            std::string name = identifyFace(descriptor, known_faces_descriptors);
            // имя на кадре
            cv::putText(frame, name, left_corners[temp], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            temp++; 
        }

        cv::imshow("Camera", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 27) {
            break; // Выход при нажатии ESC
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
} */





#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv; // NOLINT
using namespace cv::face; // NOLINT
using namespace std; // NOLINT

void createFaceRecognizerModel(const std::vector<std::string>& names) {
    std::vector<Mat> images;
    std::vector<int> labels;

    int temp = 1;
    for (const auto& name: names) {
        std::string path = "../data/" + name + ".jpg";
        images.push_back(imread(path, IMREAD_GRAYSCALE));
        labels.push_back(temp);
        ++temp;
    }

    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();

    model->train(images, labels);

    model->save("../models/face_recognizer.yml");
}

auto main() -> int {
    std::vector<std::string> names {"harry_potter", "ksenia_karimova", "lady_gaga",
                                    "sergey_razumovskiy", "thimotee_chamolete", "dima_kutuzov"};
    std::vector<int> labels {1,2,3,4,5, 6};
    
    // createFaceRecognizerModel(names);
    CascadeClassifier face_cascade;
    if (!face_cascade.load("../models/haarcascade_frontalface_default.xml")) {
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->read("../models/face_recognizer.yml");

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) { break; }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto face : faces) {
            Mat faceROI = gray(face);
            int label;
            double confidence;
            recognizer->predict(faceROI, label, confidence);

            rectangle(frame, face, Scalar(255, 0, 0), 2);
            string text = "Unknown";
            if (confidence < 100) {
                text = names[label];
            }
            putText(frame, text, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 0, 0), 2);
        }

        imshow("Camera", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 27) {
            break; // Выход при нажатии ESC
        }
    }

    return 0;
}

