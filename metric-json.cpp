//
// Created by Hrach Ayunts on 05.09.22.
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "DecolorMeasurer.h"
#include "utils/image_components.h"
#include "utils/utils.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

//void printTIAInfo(const cv::Mat& src, const cv::Mat& gray, const cv::Mat& mask) {
//    DecolorMeasurer measurer1(gray, src, 1, 10, 10, 5);
//    const float tia1 = abs(measurer1.TIA());
//
//    DecolorMeasurer measurer2(gray, src, mask, 1, 10, 10, 5);
//    const float tia2 = abs(measurer2.TIA());
//
//    printf("%.5f %.5f -> old, weighted\n", tia1, tia2);
//}

std::string findPathWithKey(const std::vector<std::string>& paths, const std::string& key) {
    for (auto path: paths) {
        if (path.find(key) != std::string::npos) {
            return path;
        }
    }
    printf("No path with key %ss\n", key.c_str());
    return "";
}

int main() {
    const std::string prefix = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization"
                               "/images/Cadik/evaluation/";
    const std::string pathIm = prefix + "cadik08perceptualEvaluation-images/";
    const std::string pathSal = prefix + "saliency/";
    const std::string pathAt = prefix + "attention/";
    const std::string ext = ".png";

    std::vector<int> attentionLayer = {0, 76, 81, 164, 166, 168};

    std::vector<std::string> methods = {
            "source",
            "CIE_Y",
            "Color2Gray",
            "Grundland", // decolorize?
            "Smith",
            "rasche",
            "Bala",
            "theta" // Neumann
    };

    std::vector<MetricType> mt = {
            MetricType::local,
//            MetricType::global,
//            MetricType::mixed
    };
    std::vector<std::string> mts = {
            "local",
//            "global",
//            "mixed"
    };

    const int imageCount = 24;

    std::vector<std::string> paths;
    json js, jsSal, jsAt;

//    MetricType type = MetricType::local;
//    std::string typeStr = "local_escore";

    const int testSamples = 2;
    const int pairNumber = 10;

    const int minThresh = 1;
    const int maxThresh = 10;

//    for (int lay: attentionLayer) {
    int lay = 166;
        for (int mm = 0; mm < mt.size(); ++mm) {

            for (int i = 1; i <= imageCount; ++i) {
                const std::string dir = pathIm + st(i);

                cv::glob(dir, paths, false);

                const std::string filename = st(i) + ext;
                cv::Mat src = cv::imread(findPathWithKey(paths, methods[0]));

                splitAndShowYCC(src);
//        cv::imwrite("/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/src/"
//        + st(i) + ".png", src);

//                std::vector<float> tiaVec(7);
//                std::vector<float> tiaSalVec(7);
                std::vector<float> tiaAtVec(7);

                // sal
//            cv::Mat maskSal = cv::imread(pathSal + filename, cv::IMREAD_GRAYSCALE);
                // attention -> negate!!!!!!!
                cv::Mat maskAt;
                if (lay != 0) {
                    maskAt = cv::imread(
                            "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/masks/attention"
                            + st(lay) + "/" + filename, cv::IMREAD_GRAYSCALE);
                }
//        maskAt = 255 - maskAt;
//                 TODO: save negated version

                for (int m = 1; m <= 7; ++m) {
                    cv::Mat gray = cv::imread(findPathWithKey(paths, methods[m]), cv::IMREAD_GRAYSCALE);
                    cv::Mat tempMask;
//                DecolorMeasurer measurer(gray, src, tempMask, minThresh, maxThresh, pairNumber, testSamples, type);
//                DecolorMeasurer measurerSal(gray, src, maskSal, minThresh, maxThresh, pairNumber, testSamples, type);
                    DecolorMeasurer measurerAt(gray, src, maskAt, minThresh, maxThresh, pairNumber, testSamples, mt[mm]);

//            const float tia = abs(measurer.TIA());
//            const float tia1 = abs(measurerSal.TIA());
            const float tia2 = abs(measurerAt.TIA());

//                const float tia = abs(measurer.getEScore()[4]);
//                const float tia1 = abs(measurerSal.getEScore()[4]);
//                    const float tia2 = abs(measurerAt.getEScore()[4]);
                    printf("No mask: , Saliency: , Attention: %f\n", tia2);

//                    tiaVec[m - 1] = tia;
//                    tiaSalVec[m - 1] = tia1;
                    tiaAtVec[m - 1] = tia2;
                }
//                js[st(i)] = tiaVec;
//                jsSal[st(i)] = tiaSalVec;
                jsAt[st(i)] = tiaAtVec;
            }

//            std::string jsonPath = prefix + "res-jsons/" + typeStr + ".json";
//            std::string jsonPathSal = prefix + "res-jsons/" + typeStr + "_sal.json";
//            std::string jsonPathAt = prefix + "res-jsons/new" + mts[mm] + "_at_" + st(lay) + ".json";

//            std::ofstream o(jsonPath);
//            std::ofstream o1(jsonPathSal);
//            std::ofstream o2(jsonPathAt);

//            o << std::setw(4) << std::fixed << std::setprecision(8) << js << std::endl;
//            o1 << std::setw(4) << std::fixed << std::setprecision(8) << jsSal << std::endl;
//            o2 << std::setw(4) << std::fixed << std::setprecision(8) << jsAt << std::endl;
        }
//    }

    return 0;
}