//
// Created by Hrach Ayunts on 07.06.22.
//

#include "image_components.h"
#include "utils.h"

const float eps = 0.000001f;

cv::Mat getImageComponent(const cv::Mat& source, ImageComponents ic, float coef) {
    cv::Mat gray(source.rows, source.cols, CV_8UC1);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            auto r = static_cast<float>(source.at<cv::Vec3b>(i, j)[2]) / 255.f;
            auto g = static_cast<float>(source.at<cv::Vec3b>(i, j)[1]) / 255.f;
            auto b = static_cast<float>(source.at<cv::Vec3b>(i, j)[0]) / 255.f;
            switch (ic) {
                case ImageComponents::R:
                    gray.at<uchar>(i, j) = source.at<cv::Vec3b>(i, j)[2];
                    break;
                case ImageComponents::G:
                    gray.at<uchar>(i, j) = source.at<cv::Vec3b>(i, j)[1];
                    break;
                case ImageComponents::B:
                    gray.at<uchar>(i, j) = source.at<cv::Vec3b>(i, j)[0];
                    break;
                case ImageComponents::R_mul_G:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(r * g * 255.f));
                    break;
                case ImageComponents::R_mul_B:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(r * b * 255.f));
                    break;
                case ImageComponents::G_mul_B:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(g * b * 255.f));
                    break;
                case ImageComponents::R_pow_2:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(pow(r, coef) * 255.f));
                    break;
                case ImageComponents::G_pow_2:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(pow(g, coef) * 255.f));
                    break;
                case ImageComponents::B_pow_2:
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(pow(b, coef) * 255.f));
                    break;
                case ImageComponents::R_dif_G: {
//                    float rg = abs(log(eps + r / (g + 1.f)) * r / (g + 1.f));
//                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(rg * 255.f));

                    float rg = abs(pow((r) / (g + 1.f), 0.9f));
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(rg * 255.f));
                    break;
                }
                case ImageComponents::G_dif_B: {
//                    float gb = abs(log(eps + g / (b + 1.f)) * g / (b + 1.f));

                    float gb = abs(pow((g) / (b + 1.f), 0.9f));
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(gb * 255.f));
                    break;
                }
                case ImageComponents::B_dif_R: {
                    float br = abs(pow((b) / (r + 1.f), 0.9f));
//                    float br = abs(log(eps + b / (r + 1.f)) * b / (r + 1.f));
                    gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(br * 255.f));
                    break;
                }
            }
        }
    }
    return gray;
}