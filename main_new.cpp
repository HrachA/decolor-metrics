//
// Created by Hrach Ayunts on 27.05.22.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "DecolorMeasurer.h"
#include "utils/image_components.h"
#include "utils/utils.h"

void printTIAInfo(const cv::Mat& src, const cv::Mat& gray, const cv::Mat& mask) {
    DecolorMeasurer measurer1(gray, src, 1, 10, 10, 5);
    const float tia1 = abs(measurer1.TIA());

    DecolorMeasurer measurer2(gray, src, mask, 1, 10, 10, 5);
    const float tia2 = abs(measurer2.TIA());

    printf("%.5f %.5f -> old, weighted\n", tia1, tia2);
}

int main() {
    const std::string path = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/";

    const int imageIndex = 12;
    const std::string ext = ".png";
    const std::string filename = st(imageIndex) + ext;
//    std::vector<std::string> names = {
//            "12.png",
//            "11.png",
//            "8.png",
//            "23.png",
//    };


    cv::Mat src = cv::imread(path + filename);

//    cv::Mat mask = cv::imread(path + "masks/12_2.png");
    cv::Mat mask = cv::imread(path + "temp/attention.png", cv::IMREAD_GRAYSCALE);

    mask = negative(mask);
    std::vector<int> indices = {1, 3, 6};
//    for (int i = 1; i <= 7; ++i) {
    for (int i: indices) {
        cv::Mat gray1 =  cv::imread(path + "temp/" + st(i) + ".png", cv::IMREAD_GRAYSCALE);
        cv::Mat gray2 = gray1(cv::Rect2i(3, 2, gray1.cols - 6, gray1.rows - 4));
//        print(gray2.cols);
//        print(gray2.rows);
        print(i);
        printTIAInfo(src, gray2, mask);
    }



//    cv::Mat r = getImageComponent(src, ImageComponents::R);
//    cv::Mat g = getImageComponent(src, ImageComponents::G);
//    cv::Mat b = getImageComponent(src, ImageComponents::B);
//
//    cv::Mat avg = grayFromCoefs(src, {1.f/3, 1.f/3, 1.f/3});
//    cv::Mat lum1 = grayFromCoefs(src, {0.114f, 0.587f, 0.299f});
//    cv::Mat lum2 = grayFromCoefs(src, {0.0722f, 0.7152f, 0.2126f});
//
//    cv::Mat lum3 = grayFromCoefsPow(src, {0.0722f, 0.7152f, 0.2126f}, 2.2);

//    cv::Mat rg = getImageComponent(src, ImageComponents::R_mul_G);
//    cv::Mat rb = getImageComponent(src, ImageComponents::R_mul_B);
//    cv::Mat gb = getImageComponent(src, ImageComponents::G_mul_B);
//
//    cv::Mat r2 = getImageComponent(src, ImageComponents::R_pow_2);
//    cv::Mat g2 = getImageComponent(src, ImageComponents::G_pow_2);
//    cv::Mat b2 = getImageComponent(src, ImageComponents::B_pow_2);
//
//    cv::Mat r_g = getImageComponent(src, ImageComponents::R_dif_G);
//    cv::Mat g_b = getImageComponent(src, ImageComponents::G_dif_B);
//    cv::Mat b_r = getImageComponent(src, ImageComponents::B_dif_R);

//    std::string folder = "components/" + st(imageIndex) + "/";
//    cv::imwrite(path + folder + "r.png", r);
//    cv::imwrite(path + folder + "g.png", g);
//    cv::imwrite(path + folder + "b.png", b);

//    cv::imwrite(path + folder + "avg.png", avg);
//    cv::imwrite(path + folder + "lum1.png", lum1);
//    cv::imwrite(path + folder + "lum2.png", lum2);
//    cv::imwrite(path + folder + "lum3.png", lum3);


//    printTIAInfo(src, r, mask);
//    printTIAInfo(src, g, mask);
//    printTIAInfo(src, b, mask);
//
//    printTIAInfo(src, avg, mask);
//    printTIAInfo(src, lum1, mask);
//    printTIAInfo(src, lum2, mask);
//    printTIAInfo(src, lum3, mask);

//    auto res = grayBestNew(src, mask, 0, 100, path + "masks/best/" + st(imageIndex) + "/");
//
//    cv::imwrite(path + folder  + "rg.png", rg);
//    cv::imwrite(path + folder  + "rb.png", rb);
//    cv::imwrite(path + folder  + "gb.png", gb);
//
//    cv::imwrite(path + folder + "r2_.png", r2);
//    cv::imwrite(path + folder + "g2_.png", g2);
//    cv::imwrite(path + folder + "b2_.png", b2);

//    cv::imwrite(path + folder + "r_g_.png", r_g);
//    cv::imwrite(path + folder + "g_b_.png", g_b);
//    cv::imwrite(path + folder + "b_r_.png", b_r);

    return 0;
}