//
// Created by Hrach Ayunts on 05.05.22.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "DecolorMeasurer.h"
#include "decolor.h"
#include "utils/utils.h"
#include "AME.h"

std::vector<std::string> tisInfo(const cv::Mat& src, const cv::Mat& gray) {
    DecolorMeasurer measurer(gray, src, 1, 10, 10, 5);

    const float tis = abs(measurer.TIS());
    const float tia = abs(measurer.TIA());
    auto scores = measurer.getEScore();
//    const float ame = abs(MeanAME(gray));
//    const float newAME = newMeanAME(src, gray, true);
//    const float mul = tis * newAME;
//    const float mean = (tis + ame) / 2;
//    const float harm = mul / mean;

    printf("%.5f %.5f -> tia, tis\n", tia, tis);

    for (int i = 3; i <= 5; i++) {
        printf("t: %d, E-score: %.5f\n", i, scores[i]);
    }
    print("");

    return {"TIA: " + std::to_string(tia),
            "TIS: " + std::to_string(tis),
//            "NEW: " + std::to_string(newAME),
//            "MUL: " + std::to_string(mul),
    };
}

#define GAMMA 1.5
#define decolorization(source, gray, boosted) \
dc::decolor(source, gray, boosted, {GAMMA, GAMMA, GAMMA}, true, 10, 0.2)

#define GAMMA 1.5
#define decolorization2(source, gray, boosted) \
dc::decolor(source, gray, boosted, {GAMMA, GAMMA, GAMMA}, false)

int main() {
    const std::string path = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/new/ame_test/levels/";


//    const std::string path =
//            "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/";
    std::vector<std::string> names = {
//            "3.png",
            "11.png",
//            "8.png",
//            "12.png",
//            "2.png",
//            "23.png",
//            "6.png",
//            "5.png",
//            "20.png",
//            "1.png"
    };

    std::string pathCadik = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/";

    std::vector<std::string> folders = {
//            "cp/",
//            "cieY/",
//            "spdecolor/",
//            "svd/",
            "our_final/"
    };

    cv::Mat src = cv::imread(pathCadik + "11.png");

    for (int i = 1; i <= 8; ++i) {
        print(i);
        cv::Mat gray = cv::imread(path + st(i) + ".png", cv::IMREAD_GRAYSCALE);
        auto infoVec = tisInfo(src, gray);
    }

//    for (int i = 1; i <= 24; ++i) {
//        std::string name = st(i) + ".png";
//        print(name);
//        cv::Mat src = cv::imread(pathCadik + name);
//        std::string writePath = pathCadik + "new/" + st(i) + "/";
//        auto res = grayBest9(src, -100, 200, writePath);
//    }

    float meanTIS[5] = {0.f};
    float meanTIA[5] = {0.f};

    for (const auto& name: names) {

        print(name);
        cv::Mat src = cv::imread(pathCadik + name);
//        cv::Mat cvDecolor, ourDecolor, cvBoosted, ourBoosted;

        for (const auto& folder: folders) {
            print(folder);
            cv::Mat gray = cv::imread(pathCadik + folder + name, cv::IMREAD_GRAYSCALE);
            auto infoVec = tisInfo(src, gray);
        }

//        splitAndShowLab(src);

//        cv::decolor(src, ourDecolor, cvBoosted);

//        cv::imwrite(path + "boosted_" + name, cvBoosted);


//
//        {
//            auto res = grayBest9(src);
//        }

//        {
//            cv::Mat gray2 = cv::imread(path + "/new/cv/8.png", cv::IMREAD_GRAYSCALE);
//            auto infoVec = tisInfo(src, gray2);
//        }
//        {
//            cv::Mat gray1 = cv::imread(path + "/new/opt_ame-/8.png", cv::IMREAD_GRAYSCALE);
//            auto infoVec = tisInfo(src, gray1);
//        }
//        {
//            cv::Mat gray2 = cv::imread(path + "/new/opt_tia-/8.png", cv::IMREAD_GRAYSCALE);
//            auto infoVec = tisInfo(src, gray2);
//        }

//        {
//            ourDecolor = cv::imread(path + "new/cv/" + name, cv::IMREAD_GRAYSCALE);
//            auto infoVec = tisInfo(src, ourDecolor);
//            std::vector<cv::Scalar> colors;
//            cv::Mat text = addText(ourDecolor, infoVec, colors);
//            cv::imwrite(path + "new/cv/" + name + name, text);
//        }
//
//        {
//            ourDecolor = cv::imread(path + "new/dc/" + name, cv::IMREAD_GRAYSCALE);
//            auto infoVec = tisInfo(src, ourDecolor);
//            std::vector<cv::Scalar> colors;
//            cv::Mat text = addText(ourDecolor, infoVec, colors);
//            cv::imwrite(path + "new/dc/" + name + name, text);
//        }

//        Metrics testM = Metrics::AMEG;
//        {
//            print("---------- Test 1 ----------");
//            auto res = grayBest(src, testM, 0, 100, 5);
//            ourDecolor = grayFromCoefs(src, res.first);
//            auto infoVec = tisInfo(src, ourDecolor);
//            infoVec.push_back({"B: " + std::to_string(res.first[0])});
//            infoVec.push_back({"G: " + std::to_string(res.first[1])});
//            infoVec.push_back({"R: " + std::to_string(res.first[2])});
//            std::vector<cv::Scalar> colors;
//            cv::Mat text = addText(ourDecolor, infoVec, colors);
//            cv::imwrite(path + "new/opt_ame/" + name, ourDecolor);
//            cv::imwrite(path + "new/opt_ame/" + name + name, text);
//        }
//
//        {
//            print("---------- Test 2 ----------");
//            auto res = grayBest(src, testM, -100, 200, 20);
//            ourDecolor = grayFromCoefs(src, res.first);
//            auto infoVec = tisInfo(src, ourDecolor);
//            infoVec.push_back({"B: " + std::to_string(res.first[0])});
//            infoVec.push_back({"G: " + std::to_string(res.first[1])});
//            infoVec.push_back({"R: " + std::to_string(res.first[2])});
//            std::vector<cv::Scalar> colors;
//            cv::Mat text = addText(ourDecolor, infoVec, colors);
//            cv::imwrite(path + "new/opt_ame-/" + name, ourDecolor);
//            cv::imwrite(path + "new/opt_ame-/" + name + name, text);
//        }
    }
    return 0;
}