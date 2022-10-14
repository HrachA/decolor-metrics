//
// Created by Hrach Ayunts on 05.05.22.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "DecolorMeasurer.h"
#include "decolor.h"
#include "utils/utils.h"
#include "AME.h"

void ameInfo(const cv::Mat& src) {
    const float ame3 = AME(src, 3);
    const float ame5 = AME(src, 5);
    const float ame7 = AME(src, 7);
    const float ame9 = AME(src, 10);

    printf("AME 3: %f, 5: %f, 7: %f, 9: %f\n", ame3, ame5, ame7, ame9);
}

std::vector<std::string> tisInfo(const cv::Mat& src, const cv::Mat& gray) {
    DecolorMeasurer measurer(gray, src, 1, 15, 10, 5);
    const float tis = abs(measurer.TIS());
    const float ame = abs(MeanAME(gray));
    const float mul = tis * ame;
    const float mean = (tis + ame) / 2;
    const float harm = mul / mean;

    printf("%.5f %.5f %.5f %.5f %.5f -> tis, ame, mul, mean, harm\n", tis, ame, mul, mean, harm);

    return {std::to_string(tis), std::to_string(ame),
            std::to_string(mul), std::to_string(mean), std::to_string(harm)};
}

int main() {

    const std::string path =
            "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/new/ame_test/";
//    std::vector<std::string> names = {
//            "3.png",
//            "11.png",
//            "8.png",
//            "12.png",
//            "2.png",
//            "6.png"
//    };
    cv::Mat their = cv::imread(path + "8_cv.png", cv::IMREAD_GRAYSCALE);
    cv::Mat our = cv::imread(path + "8.png", cv::IMREAD_GRAYSCALE);

    const float ame1 = abs(MeanAME(their));
    const float ame2 = abs(MeanAME(our));

    printf("AME: Their: %f, Our: %f\n", ame1, ame2);
    cv::Mat best = cv::imread(path + "11.png", cv::IMREAD_GRAYSCALE);
    cv::Mat source = cv::imread("/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/11.png");


    for (int i = 1; i <= 8; i ++) {
        cv::Mat level = cv::imread( path + "levels/" + st(i) + ".png", cv::IMREAD_GRAYSCALE);
        const float ame = abs(MeanAME(level));

//        source.at<cv::Vec3b>(0, 0) = {0, 0, 0};
//        source.at<cv::Vec3b>(0, 1) = {255, 255, 255};
//        level.at<uchar>(0, 0) = 0;
//        level.at<uchar>(0, 1) = 255;
        const float newM = newAME(source, level);
        const float newMAvg = newMeanAME(source, level);
//        level = addText(level, st(ame));
        level = addText(level, "k = 5: " + st(newM));
        level = addText(level, "avg (3, 5, 9): " + st(newMAvg));
        imwrite(path + "levels/_" + st(i) + ".png", level);

//        cv::Mat dark = source * (float)i / 8;
//        const float ame = abs(MeanAME(dark));
//        dark = addText(dark, st(ame));
//        imwrite(path + "dark/" + st(i) + ".png", dark);

        printf("AME: %f, new: %f\n", ame, newM);
    }
//    printf("AME: %f, source\n", ame, i);
//    for (int i = 1; i < 25; i += 4) {
//        cv::Mat blurred;
//        cv::blur(source, blurred, Size(i, i));
//        const float ame = abs(MeanAME(blurred));
//        printf("AME: %f, ksize: %i\n", ame, i);

//        cv::imwrite(path + "11_" + std::to_string(i) + ".png", blurred);
//    }


//    for (int i = 0; i < names.size(); ++i) {
//        names[i] = "8.png";
//        print(names[i]);
//        cv::Mat src = cv::imread(path + names[i]);
//        cv::Mat cvDecolor, ourDecolor, cvBoosted, ourBoosted;
//
//        auto res = grayBest(src);
//
//        ourDecolor = grayFromCoefs(src, res.first);
//
//        auto infoVec = tisInfo(src, ourDecolor);
//        infoVec.push_back({std::to_string(res.first[0])});
//        infoVec.push_back({std::to_string(res.first[1])});
//        infoVec.push_back({std::to_string(res.first[2])});
//        std::vector<cv::Scalar> colors;
//        cv::Mat text = addText(ourDecolor, infoVec, colors);
//
//        cv::imwrite(path + "new/opt5/" + names[i], ourDecolor);
//        cv::imwrite(path + "new/opt5/" + names[i] + names[i], text);


//        cv::Mat their = cv::imread(path + "new/cv/" + names[i], cv::IMREAD_GRAYSCALE);
//        cv::Mat our = cv::imread(path + "new/dc/" + names[i], cv::IMREAD_GRAYSCALE);
//        cv::Mat opt = cv::imread(path + "new/opt2/" + names[i], cv::IMREAD_GRAYSCALE);

//        auto infoVec = tisInfo(src, their);
//        std::vector<cv::Scalar> colors;
//        cv::Mat text = addText(their, infoVec, colors);
//        cv::imwrite(path + "new/cv/" + names[i] + names[i], text);
//
//        infoVec = tisInfo(src, our);
//        text = addText(our, infoVec, colors);
//        cv::imwrite(path + "new/dc/" + names[i] + names[i], text);
//
//        infoVec = tisInfo(src, opt);
//        text = addText(opt, infoVec, colors);
//        cv::imwrite(path + "new/opt2/" + names[i] + names[i], text);

//        cv::decolor(src, cvDecolor, cvBoosted);
//        cv::imwrite(path + "new/cv/_" + names[i], cvDecolor);
//        auto infoVec = tisInfo(src, cvDecolor);

//        decolorization2(src, ourDecolor, ourBoosted);
//        cv::imwrite(path + "new/dc/" + names[i] + names[i], ourDecolor);

//        ourDecolor = grayFromCoefs(src, {0.2, 0, 0.8});
//        cv::imwrite(path + "new/opt/" + names[i], ourDecolor);
//


//        const std::string writeName = "their_" + names[i];
//        const std::string writeName = "our_opt_" + names[i];
//        cv::imwrite(path + "new/" + writeName, cvText);
//        cv::imwrite(path + "new/_" + names[i], ourDecolor);

//        cv::Mat sourceText = addText(src, {"TIS", "AME", "MUL", "MEAN", "HARM"}, colors);
//        const std::string writeNameS = "source_" + names[i];
//        cv::imwrite(path + "new/" + writeNameS, sourceText);
//    }


    return 0;
}