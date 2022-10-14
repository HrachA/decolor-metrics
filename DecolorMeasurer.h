//
// Created by Hrach Ayunts on 03.03.22.
//

#ifndef DECOLORIZATION_DECOLORMEASURER_H
#define DECOLORIZATION_DECOLORMEASURER_H

#include "opencv2/opencv.hpp"

enum class Metrics {
    TIS,
    TIA,
    AMEG,
    MUL
};

enum class MetricType {
    local,
    global,
    mixed
};

std::vector<float> grayMeasures(const cv::Mat& gray, const cv::Mat& source, int threshold = 6);

std::pair<cv::Vec3f, float> grayBest(const cv::Mat& src, Metrics opt, int start = 0, int end = 100, int step = 5);

std::pair<std::vector<float>, float> grayBest9(const cv::Mat& src, int start = -100, int end = 200, std::string path = "");

std::pair<std::vector<float>, float> grayBestNew(const cv::Mat& src, const cv::Mat& mask, int start, int end, std::string path);


cv::Mat grayFromCoefs(const cv::Mat& src, const cv::Vec3f& coefs);

cv::Mat grayFromCoefsPow(const cv::Mat& src, const cv::Vec3f& coefs, float gamma = 1.f);

class DecolorMeasurer {
public:
    DecolorMeasurer(const cv::Mat& gray, const cv::Mat& source, int minThresh = 1, int maxThresh = 16,
                    int pairNumber = 10, int testSamples = 15);

    DecolorMeasurer(const cv::Mat& gray, const cv::Mat& source, cv::Mat& mask, int minThresh = 1, int maxThresh = 16,
                    int pairNumber = 10, int testSamples = 15, MetricType type = MetricType::mixed);

    float slope();

    float TIS();

    float TIA();

    std::unordered_map<int, float> getCCPR() {
        return ccpr;
    }

    std::unordered_map<int, float> getCCFR() {
        return ccfr;
    }

    std::unordered_map<int, float> getEScore() {
        return eScore;
    }

    float getMeanCCPR() {
        float sum = 0;
        for (auto& it: ccpr) {
            sum += it.second;
        }
        return sum / static_cast<float>(ccpr.size());
    }

    float getMeanCCFR() {
        float sum = 0;
        for (auto& it: ccfr) {
            sum += it.second;
        }
        return sum / static_cast<float>(ccfr.size());
    }

    float getMeanEScore() {
        float sum = 0;
        for (auto& it: eScore) {
            sum += it.second;
        }
        return sum / static_cast<float>(eScore.size());
    }

private:
    inline bool isIndexValid(int index, int size) {
        return index >= 0 && index < size;
    }

private:
    std::unordered_map<int, float> ccpr;
    std::unordered_map<int, float> ccfr;
    std::unordered_map<int, float> eScore;
    std::vector<int> thresholds;
};

class DecolorMeasurerOnData {
public:
    explicit DecolorMeasurerOnData(int minThresh = 1, int maxThresh = 16);

    void addDecolorMeasurer(DecolorMeasurer& measurer);

    float getMeanCCPR() {
        return sumCCPR / samples;
    }

    float getMeanCCFR() {
        return sumCCFR / samples;
    }

    float getMeanEScore() {
        return sumEScore / samples;
    }

    float getMeanSlope() {
        return sumSlope / samples;
    }

    std::unordered_map<int, float> getEScore() {
        for (int t = minThreshold; t <= maxThreshold; ++t) {
            eScore[t] /= samples;
        }
        return eScore;
    }

private:
    int minThreshold;
    int maxThreshold;
    float samples = 0;
    float sumCCPR = 0;
    float sumCCFR = 0;
    float sumEScore = 0;
    float sumSlope = 0;

    // TODO: add other 2 metrics
//    std::unordered_map<int, float> ccpr;
//    std::unordered_map<int, float> ccfr;
    std::unordered_map<int, float> eScore;
    std::vector<int> thresholds;
};


#endif //DECOLORIZATION_DECOLORMEASURER_H
