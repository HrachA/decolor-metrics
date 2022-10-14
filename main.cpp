#include <iostream>
//#include <pair>
#include "opencv2/opencv.hpp"
#include "DecolorMeasurer.h"

float clampFloat(float val) {
    float res = val;
    if (val < 0.f) {
        res = 0.f;
    } else if (val > 255.f) {
        res = 255.f;
    }
    return res;
}

cv::Vec3f clampVec(cv::Vec3f v) {
    return {clampFloat(v[0]),
            clampFloat(v[1]),
            clampFloat(v[2])};
}

cv::Mat ColorLog(const cv::Mat& src) {
    cv::Mat color(src.rows, src.cols, src.type());
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            auto p = static_cast<cv::Vec3f>(src.at<cv::Vec3b>(i, j));
            color.at<cv::Vec3b>(i, j)[0] = (uchar) clampFloat(
                    abs(256.f * log((p[0] + 1.f) / (p[1] + 1.f)) / log(256.f)));
            color.at<cv::Vec3b>(i, j)[1] = (uchar) clampFloat(
                    abs(256.f * log((p[1] + 1.f) / (p[2] + 1.f)) / log(256.f)));
            color.at<cv::Vec3b>(i, j)[2] = (uchar) clampFloat(
                    abs(256.f * log((p[2] + 1.f) / (p[0] + 1.f)) / log(256.f)));
        }
    }
    return color;
}

cv::Mat ColorSpace(const cv::Mat& src) {
    cv::Mat color(src.rows, src.cols, src.type());
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            auto p = static_cast<cv::Vec3f>(src.at<cv::Vec3b>(i, j));
            color.at<cv::Vec3b>(i, j)[2] = (uchar) (clampFloat(abs(
                    (p[2] - p[1]) / sqrt(2.f))));
            color.at<cv::Vec3b>(i, j)[1] = (uchar) (clampFloat(abs(
                    (p[2] + p[1] - 2.f * p[0]) / sqrt(6.f))));
            color.at<cv::Vec3b>(i, j)[0] = (uchar) (clampFloat(abs(
                    (p[2] + p[1] + p[0]) / sqrt(3.f))));
        }
    }
    return color;
}

// GRAY = a * R + b * G + c * B
cv::Mat grayFromCoefs(const cv::Mat& src, cv::Vec3f coefs) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            gray.at<uchar>(i, j) = static_cast<uchar>(
                    clampFloat(coefs.dot(static_cast<cv::Vec3f>(src.at<cv::Vec3b>(i, j)))
                    ));
        }
    }
    return gray;
}

cv::Mat grayLightness(const cv::Mat& src) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            auto vec = static_cast<cv::Vec3f>(src.at<cv::Vec3b>(i, j));
            auto list = {src.at<cv::Vec3b>(i, j)[0],
                         src.at<cv::Vec3b>(i, j)[1],
                         src.at<cv::Vec3b>(i, j)[2]};
            float mx = cv::max(list);
            float mn = cv::min(list);
            gray.at<uchar>(i, j) = static_cast<uchar>((mn + mx) / 2.f);
        }
    }
    return gray;
}

cv::Mat grayAverage(const cv::Mat& src) {
    return grayFromCoefs(src, {1.f / 3, 1.f / 3, 1.f / 3});
}

cv::Mat grayLuminosity(const cv::Mat& src) {
    return grayFromCoefs(src, {0.07, 0.72, 0.21});
}

cv::Mat grayScotopic(const cv::Mat& src) {
    return grayFromCoefs(src, {4.33, 1.039, -0.702});
}


std::pair<cv::Vec3f, float> grayBest(const cv::Mat& src) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);

    float maxTIS = 0.f;
    cv::Vec3f vec = {0, 0, 0};

    const int step = 50;
    const int begin = 0;
    const int end = 100;

//    const int step = 50;
//    const int begin = -100;
//    const int end = 200;

    for (int a = begin; a <= end; a += step) {
        for (int b = begin; b <= end; b += step) {
//            if (a + b <= 100) {
            int c = 100 - a - b;
            float aF = a / 100.f;
            float bF = b / 100.f;
            float cF = c / 100.f;
//                printf("a %f, b %f, c %f\n", aF, bF, cF);
            gray = grayFromCoefs(src, {aF, bF, cF});
            DecolorMeasurer measurer(gray, src, 1, 15, 10, 5);
            const float tis = abs(measurer.TIS());

            const float ame = abs(AME(gray, 10));
            const float comb = tis * ame;

            if (comb > maxTIS) {
                maxTIS = comb;
                vec = cv::Vec3f(aF, bF, cF);
            }

            printf("%.5f %.5f %.5f %.2f %.2f -> comb ame tis B G\n", comb, ame, tis, aF, bF);
            std::string path = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/";
            imwrite(path + "all3/" + std::to_string(comb).substr(0, 7)
                    + "_" + std::to_string(aF).substr(0, 4)
                    + "_" + std::to_string(bF).substr(0, 4)
                    + "_" + std::to_string(cF).substr(0, 4) +
                    ".png", gray);
//            }
        }
    }

    return {vec, maxTIS};
}

int main() {

    std::string path = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/";

    cv::Mat src = imread(path + "8" + ".png", cv::IMREAD_COLOR);

    cv::Mat srcLog = ColorLog(src);
    imwrite(path + "all8/log.png", srcLog);

    cv::Mat srcCol = ColorSpace(src);
    imwrite(path + "all8/col.png", srcCol);
//    src_global = src.clone();

//    std::string imagePath = "11";
//    cv::Mat src1 = imread(path + "1.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat src2 = imread(path + "22.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat src3 = imread(path + "3.png", cv::IMREAD_GRAYSCALE);
//
//    const float a = 0.25;
//    const float b = 0.5;
//    cv::Mat final = a * src1 + b * src2 + (1 - a - b) * (255 - src3);

//    cv::imshow("res", final);
//    cv::waitKey(0);

///*
//    auto p = grayBest(src);
//    cv::Vec3f coefs = p.first;
//    printf("Best coeffs: B, G, R\n %f, %f, %f", coefs[0], coefs[1], coefs[2]);
//    cv::Mat bestGray = grayFromCoefs(src, coefs);
//    imwrite(path + "Best1_" + std::to_string(p.second).substr(0, 7)
//            + "_" + std::to_string(coefs[0]).substr(0, 4)
//            + "_" + std::to_string(coefs[1]).substr(0, 4)
//            + "_" + std::to_string(coefs[2]).substr(0, 4) +
//            ".png", bestGray);


    auto p = grayBest(srcLog);
    auto coefs = p.first;
    printf("Best coeffs: B, G, R\n %f, %f, %f", coefs[0], coefs[1], coefs[2]);
    cv::Mat bestGray = grayFromCoefs(src, coefs);
    imwrite(path + "Best2_" + std::to_string(p.second).substr(0, 7)
            + "_" + std::to_string(coefs[0]).substr(0, 4)
            + "_" + std::to_string(coefs[1]).substr(0, 4)
            + "_" + std::to_string(coefs[2]).substr(0, 4) +
            ".png", bestGray);

//    std::pair<cv::Vec3f, float> p({{0.1, 0.3, 0.6}, 0});
    p = grayBest(srcCol);
    coefs = p.first;
    printf("Best coeffs: B, G, R\n %f, %f, %f", coefs[0], coefs[1], coefs[2]);
    bestGray = grayFromCoefs(src, coefs);
    imwrite(path + "Best3_" + std::to_string(p.second).substr(0, 7)
            + "_" + std::to_string(coefs[0]).substr(0, 4)
            + "_" + std::to_string(coefs[1]).substr(0, 4)
            + "_" + std::to_string(coefs[2]).substr(0, 4) +
            ".png", bestGray);

// */
    return 0;
}
