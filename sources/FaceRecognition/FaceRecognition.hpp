#pragma once

// #include <opencv2/opencv.hpp>

// #include <dlib/opencv.h>

// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

// #include <dlib/image_processing.h>
// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing/render_face_detections.h>
// #include <dlib/image_processing.h>
// #include <dlib/image_io.h>
// #include <dlib/opencv/cv_image.h>
// #include <dlib/opencv/to_open_cv.h>
// #include <dlib/dnn.h>
// #include <dlib/clustering.h>
// #include <dlib/string.h>


// // Определение архитектуры сети
// template <template <int, template<typename> class, int, typename> class block, int N, template<typename> class BN, typename SUBNET>
// using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

// template <template <int, template<typename> class, int, typename> class block, int N, template<typename> class BN, typename SUBNET>
// using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

// template <int N, template <typename> class BN, int stride, typename SUBNET>
// using block  = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

// template <int N, typename SUBNET> using ares      = residual<block, N, dlib::affine, SUBNET>;
// template <int N, typename SUBNET> using ares_down = residual_down<block, N, dlib::affine, SUBNET>;

// template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
// template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
// template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
// template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
// template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, dlib::max_pool<3, 3, 2, 2, SUBNET>>>>;

// using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
//                             alevel0<
//                             alevel1<
//                             alevel2<
//                             alevel3<
//                             alevel4<
//                             dlib::input_rgb_image_sized<150>
//                             >>>>>>>>;


// class Models {
// public:
//     Models() {
//         dlib::deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;
//         dlib::deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;
//     }

//     dlib::shape_predictor sp;
//     anet_type net;
// };


// // Функция для извлечения дескрипторов лиц
// auto getFaceDescriptors(const cv::Mat &frame, Models& models) -> std::vector<dlib::matrix<float, 0, 1>> {
//     std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
    
//     dlib::cv_image<dlib::bgr_pixel> cimg(frame);
//     dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
//     std::vector<dlib::rectangle> faces = detector(cimg);

//     for (auto face : faces) {
//         dlib::full_object_detection shape = models.sp(cimg, face);
//         dlib::matrix<dlib::rgb_pixel> face_chip;
//         dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
//         face_descriptors.push_back(models.net(face_chip));
//     }
//     return face_descriptors;
// }


// class KnownFaces {
// public:
//     KnownFaces(const std::vector<std::string>& names, Models& models) {
//         for (const auto& name: names) {
//             std::string path = "../data/" + name + ".jpg";
//             cv::Mat person_img = cv::imread(path);
//             known_faces[name] = getFaceDescriptors(person_img, models)[0];
//         }
//     }

//     std::map<std::string, dlib::matrix<float, 0, 1>> known_faces;
// };


// auto identifyFace(const dlib::matrix<float, 0, 1> &descriptor, const KnownFaces& db) -> std::string {
//     std::string identified_name = "Unknown";
//     double min_distance = 0.6; // Пороговое значение расстояния для идентификации

//     for (const auto &known_face : db.known_faces) {
//         double distance = dlib::length(known_face.second - descriptor);
//         if (distance < min_distance) {
//             min_distance = distance;
//             identified_name = known_face.first;
//         }
//     }

//     return identified_name;
// }


// class FaceRecognition {
// public:
//     explicit FaceRecognition(cv::Mat  image) : m_image(std::move(image)) { }
// private:
//     cv::Mat m_image;
// };