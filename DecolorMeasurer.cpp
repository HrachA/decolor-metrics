//
// Created by Hrach Ayunts on 03.03.22.
//

#include "DecolorMeasurer.h"
#include "utils/utils.h"
#include <random>
#include "AME.h"


// GRAY = a * B + b * G + c * R
cv::Mat grayFromCoefs(const cv::Mat& src, const cv::Vec3f& coefs) {
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

cv::Mat grayFromCoefsPow(const cv::Mat& src, const cv::Vec3f& coefs, float gamma) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            float b = static_cast<float>(src.at<cv::Vec3b>(i, j)[0]) / 255.f;
            float g = static_cast<float>(src.at<cv::Vec3b>(i, j)[1]) / 255.f;
            float r = static_cast<float>(src.at<cv::Vec3b>(i, j)[2]) / 255.f;

            gray.at<uchar>(i, j) = static_cast<uchar>(pow(
                    coefs[0] * pow(b, gamma) +
                    coefs[1] * pow(g, gamma) +
                    coefs[2] * pow(r, gamma), 1.f / gamma) * 255.f);
        }
    }
    return gray;
}

cv::Mat grayFromCoefs9(const cv::Mat& src, std::vector<float> coefs) {

    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            auto r = static_cast<float>(src.at<cv::Vec3b>(i, j)[2]) / 255.f;
            auto g = static_cast<float>(src.at<cv::Vec3b>(i, j)[1]) / 255.f;
            auto b = static_cast<float>(src.at<cv::Vec3b>(i, j)[0]) / 255.f;

            const float eps = 0.000001f;
            float rg = abs(log(eps + r / (g + 1.f)) * r / (g + 1.f));
            float gb = abs(log(eps + g / (b + 1.f)) * g / (b + 1.f));
            float br = abs(log(eps + b / (r + 1.f)) * b / (r + 1.f));

//            printf("Test log: %f, %f, %f\n", rg, gb, br);

            const float res = b * coefs[0] + g * coefs[1] + r * coefs[2] +
                              b * g * coefs[3] + g * r * coefs[4] + b * r * coefs[5] +
                              rg * coefs[6] + gb * coefs[7] + br * coefs[8];

            gray.at<uchar>(i, j) = static_cast<uchar>(clampFloat(res * 255.f));
        }
    }
    return gray;
}

std::pair<cv::Vec3f, float> grayBest(const cv::Mat& src, Metrics opt, int start, int end, int step) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);

    float maxMeasure = 0.f;
    cv::Vec3f vec = {0, 0, 0};

//    std::vector<int> values{-100, -50, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300};
//    std::vector<int> values{0, 10, 20, 30, 50, 60, 80, 90, 100};

//    for (int a: values) {
//        for (int b: values) {
    for (int a = start; a <= end; a += step) {
        for (int b = start; b <= end; b += step) {
            if (start < 0 || a + b <= 100) {
                int c = 100 - a - b;
                float aF = a / 100.f;
                float bF = b / 100.f;
                float cF = c / 100.f;

                gray = grayFromCoefs(src, {aF, bF, cF});

                float curMeasure;
                if (opt == Metrics::AMEG) {
                    curMeasure = newMeanAME(src, gray, true);
                    printf("B: %f, G: %f, R: %f, newAME: %f\n",
                           aF, bF, cF, curMeasure);
//                    cv::imwrite("/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/new/8/"
//                                + std::to_string(curMeasure) + ".png", gray);
                } else {
                    DecolorMeasurer measurer(gray, src, 1, 10, 10, 5);
//                    const float tis = abs(measurer.TIS());
                    const float tia = abs(measurer.TIA());

//                    B: -0.500000, G: 0.500000, R: 1.000000, 8.png
//                    B: 0.500000, G: -0.500000, R: 1.000000,

//                    cv::imwrite("/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/new/8/"
//                                + std::to_string(tia) + ".png", gray);

                    if (opt == Metrics::MUL) {
                        const float ame = newMeanAME(src, gray, true);
                        curMeasure = tia * ame;
//                        printf("B: %f, G: %f, R: %f, TIS: %f, newAME: %f, mul: %f\n",
//                               aF, bF, cF, tis, ame, curMeasure);
                    } else {
                        curMeasure = tia;
                        printf("B: %f, G: %f, R: %f, TIA: %f\n",
                               aF, bF, cF, tia);
                    }
//                    const float mul = tis * ame;
//                    const float mean = (tis + ame) / 2;
//                    const float harm = mul / mean;
                }
                if (maxMeasure < curMeasure) {
                    maxMeasure = curMeasure;
                    vec = cv::Vec3f(aF, bF, cF);
                }
            }
        }
    }
    return {vec, maxMeasure};
}

std::pair<std::vector<float>, float> grayBest9(const cv::Mat& src, int start, int end, std::string path) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);

    float maxMeasure = 0.f;
    std::vector<float> vec = {0, 0, 0};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(start, end);

    const int PARAMS = 9;

    for (int i = 0; i <= 100; i++) {
        std::vector<float> coefs(PARAMS);
        for (int k = 0; k < PARAMS; k++) {
            coefs[k] = dist(mt) / 100.f;
        }

        gray = grayFromCoefs9(src, coefs);

        DecolorMeasurer measurer(gray, src, 1, 10, 10, 5);
        const float tia = abs(measurer.TIA());

        if (tia > 0.9) {
            printf("TIA: %f, ", tia);
            for (int k = 0; k < PARAMS; k++) {
                printf("%.5f ", coefs[k]);
            }
            print("");
            cv::imwrite(path + std::to_string(tia) + ".png", gray);
        }


//                    cv::imwrite("/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/new/8/"
//                                + std::to_string(tia) + ".png", gray);

//            if (opt == Metrics::MUL) {
//                const float ame = newMeanAME(src, gray, true);
//                curMeasure = tia * ame;
//                        printf("B: %f, G: %f, R: %f, TIS: %f, newAME: %f, mul: %f\n",
//                               aF, bF, cF, tis, ame, curMeasure);
//            } else {
//                curMeasure = tia;
//                printf("B: %f, G: %f, R: %f, TIA: %f\n",
//                       aF, bF, cF, tia);
//            }
        if (maxMeasure < tia) {
            maxMeasure = tia;
            vec = coefs;
        }
    }

    return {vec, maxMeasure};
}

std::pair<std::vector<float>, float>
grayBestNew(const cv::Mat& src, const cv::Mat& mask, int start, int end, std::string path) {
    cv::Mat gray(src.rows, src.cols, CV_8UC1);

    float maxMeasure = 0.f;
    std::vector<float> vec = {0, 0, 0};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(start, end);

    for (int i = 0; i <= 1000; i++) {
        cv::Vec3f coefs; // b g r
        coefs[0] = static_cast<float>(dist(mt)) / 100.f;
        coefs[1] = static_cast<float>(dist(mt)) / 100.f;
        coefs[2] = 1.f - coefs[0] - coefs[1];

        gray = grayFromCoefs(src, coefs);

        DecolorMeasurer measurer(gray, src, 1, 10, 10, 5);
        const float tia = abs(measurer.TIA());

        if (tia > 0.92) {
            printf("TIA: %f, ", tia);
            for (int k = 0; k < 3; k++) {
                printf("%.5f ", coefs[k]);
            }
            print("");
            cv::imwrite(path + std::to_string(tia) + ".png", gray);
        }

        if (maxMeasure < tia) {
            maxMeasure = tia;
//            vec = coefs;
        }
    }

    return {vec, maxMeasure};
}

std::pair<float, float> linearRegression(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        std::cout << "Invalid input for Linear Regression" << std::endl;
        return {0, 0};
    }

    float sumX = 0, sumX2 = 0;
    float sumY = 0, sumXY = 0;
    for (int i = 0; i < x.size(); i++) {
        sumX += x[i];
        sumX2 += x[i] * x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
    }

    auto size = static_cast<float>(x.size());
    float k = (size * sumXY - sumX * sumY) / (size * sumX2 - sumX * sumX);
    float b = (sumY - k * sumX) / size;

    return {k, b};
}

DecolorMeasurer::DecolorMeasurer(const cv::Mat& gray, const cv::Mat& source, int minThresh, int maxThresh,
                                 int pairNumber, int testSamples) {

//    print("Without mask");
    for (int t = minThresh; t <= maxThresh; ++t) {
        thresholds.push_back(t);
        ccpr.insert({t, 0.f});
        ccfr.insert({t, 0.f});
        eScore.insert({t, 0.f});
    }

    cv::Mat grayFloat, sourceNorm, lab;
    gray.convertTo(grayFloat, CV_32F);

    std::vector<cv::Vec3f> labVec;

    source.convertTo(sourceNorm, CV_32F, 1.0 / 255.0, 0);
    cvtColor(sourceNorm, lab, cv::COLOR_BGR2Lab);

    labVec = std::vector<cv::Vec3f>(gray.rows * gray.cols);
    int ind = 0;
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            labVec[ind] = lab.at<cv::Vec3f>(i, j);
            ind++;
        }
    }

    std::vector<float> grayVec((float*) grayFloat.data, (float*) grayFloat.data + grayFloat.rows * grayFloat.cols);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(0, static_cast<int>(grayVec.size() - 1));

    const int size = grayVec.size();

    for (int test = 0; test < testSamples; ++test) {
        std::unordered_map<int, double> ccprTemp;
        std::unordered_map<int, double> ccfrTemp;
        std::unordered_map<int, double> omegaTemp;
        std::unordered_map<int, double> thetaTemp;

        for (int threshold: thresholds) {
            ccprTemp.insert({threshold, 0.f});
            ccfrTemp.insert({threshold, 0.f});
            omegaTemp.insert({threshold, 0.f});
            thetaTemp.insert({threshold, 0.f});
        }

        for (int i = 1; i < size - 1; ++i) {

//            std::vector<int> compInd(pairNumber - 2);
//            for (int k = 0; k < pairNumber - 2; ++k) {
//                compInd[k] = dist(mt);
//            }
            std::vector<int> compInd;
            compInd.push_back(i + 1);
            compInd.push_back(i - 1);


            for (int j: compInd) {
                float delta = distanceLAB(labVec[i], labVec[j]);
                float grayDist = abs(grayVec[i] - grayVec[j]) / 2.55f;

                for (int t: thresholds) {
                    auto threshFloat = static_cast<float>(t);
                    // ccpr
                    if (delta >= threshFloat) {
                        omegaTemp[t]++;
                        if (grayDist >= threshFloat) {
                            ccprTemp[t]++;
                        }
                    }

                    // ccfr
                    if (grayDist > threshFloat) {
                        thetaTemp[t]++;
                        if (delta <= threshFloat) {
                            ccfrTemp[t]++;
                        }
                    }
                }
            }
        }

        for (int t: thresholds) {
            ccprTemp[t] = omegaTemp[t] < 0.01 ? 1.0 : ccprTemp[t] / omegaTemp[t];
            ccfrTemp[t] = thetaTemp[t] < 0.01 ? 0.0 : 1.0 - ccfrTemp[t] / thetaTemp[t];
            ccpr[t] += static_cast<float>(ccprTemp[t]);
            ccfr[t] += static_cast<float>(ccfrTemp[t]);
        }
    }

    for (int t: thresholds) {
        ccpr[t] /= static_cast<float>(testSamples);
        ccfr[t] /= static_cast<float>(testSamples);
        if (ccpr[t] + ccfr[t] < 0.01) {
            eScore[t] = 0;
        } else {
            eScore[t] = 2.f * ccpr[t] * ccfr[t] / (ccpr[t] + ccfr[t]);
        }
    }
}

// weighted
DecolorMeasurer::DecolorMeasurer(const cv::Mat& gray, const cv::Mat& source, cv::Mat& mask, int minThresh,
                                 int maxThresh, int pairNumber, int testSamples, MetricType type) {

//    print("With mask");
    if (mask.cols == 0 || mask.rows == 0) {
        mask = cv::Mat(gray.rows, gray.cols, CV_8U, 255);
    }

    for (int t = minThresh; t <= maxThresh; ++t) {
        thresholds.push_back(t);
        ccpr.insert({t, 0.f});
        ccfr.insert({t, 0.f});
        eScore.insert({t, 0.f});
    }

    cv::Mat grayFloat, sourceNorm, lab, maskFloat;
    gray.convertTo(grayFloat, CV_32F);
    mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

    std::vector<cv::Vec3f> labVec;

    source.convertTo(sourceNorm, CV_32F, 1.0 / 255.0, 0);
    cvtColor(sourceNorm, lab, cv::COLOR_BGR2Lab);

    labVec = std::vector<cv::Vec3f>(gray.rows * gray.cols);
    int ind = 0;
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            labVec[ind] = lab.at<cv::Vec3f>(i, j);
            ind++;
        }
    }

    std::vector<float> grayVec((float*) grayFloat.data, (float*) grayFloat.data + grayFloat.rows * grayFloat.cols);
    std::vector<float> maskVec((float*) maskFloat.data, (float*) maskFloat.data + maskFloat.rows * maskFloat.cols);

    const int size = grayVec.size();
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(0, static_cast<int>(size - 1));

    const int width = grayFloat.cols;
    const int height = grayFloat.rows;

    for (int test = 0; test < testSamples; ++test) {
        std::unordered_map<int, double> ccprTemp;
        std::unordered_map<int, double> ccfrTemp;
        std::unordered_map<int, double> omegaTemp;
        std::unordered_map<int, double> thetaTemp;

        for (int threshold: thresholds) {
            ccprTemp.insert({threshold, 0.f});
            ccfrTemp.insert({threshold, 0.f});
            omegaTemp.insert({threshold, 0.f});
            thetaTemp.insert({threshold, 0.f});
        }

        // range in [1, size - 2] for 2 neighbors be always in range
        for (int i = 0; i < size; ++i) {
            if (maskVec[i] < 0.1) {
                continue;
            }

            std::vector<int> compInd;
            if (type == MetricType::global || type == MetricType::mixed) {
                compInd = std::vector<int>(pairNumber);
                for (int k = 0; k < pairNumber; ++k) {
                    compInd[k] = dist(mt);
                }
            }

            std::array<int, 4> indices = {i + 1, i - 1, i - width, i + width};
            if (type == MetricType::local || type == MetricType::mixed) {
                for (int index: indices) {
                    if (isIndexValid(index, size)) {
                        compInd.push_back(index);
                    }
                }
            }

            for (int j: compInd) {
                float delta = distanceLAB(labVec[i], labVec[j]);
                float grayDist = abs(grayVec[i] - grayVec[j]) / 2.55f;

                for (int t: thresholds) {
                    auto threshFloat = static_cast<float>(t);
                    // ccpr
                    if (delta >= threshFloat) {
                        omegaTemp[t] += maskVec[i] * maskVec[j];
                        if (grayDist >= threshFloat) {
                            ccprTemp[t] += maskVec[i] * maskVec[j];
                        }
                    }

                    // ccfr
                    if (grayDist > threshFloat) {
                        thetaTemp[t] += maskVec[i] * maskVec[j];
                        if (delta <= threshFloat) {
                            ccfrTemp[t] += maskVec[i] * maskVec[j];
                        }
                    }
                }
            }
        }

        for (int t: thresholds) {
            ccprTemp[t] = omegaTemp[t] < 0.01 ? 1.0 : ccprTemp[t] / omegaTemp[t];
            ccfrTemp[t] = thetaTemp[t] < 0.01 ? 0.0 : 1.0 - ccfrTemp[t] / thetaTemp[t];
            ccpr[t] += static_cast<float>(ccprTemp[t]);
            ccfr[t] += static_cast<float>(ccfrTemp[t]);
        }
    }

    for (int t: thresholds) {
        ccpr[t] /= static_cast<float>(testSamples);
        ccfr[t] /= static_cast<float>(testSamples);
        if (ccpr[t] + ccfr[t] < 0.01) {
            eScore[t] = 0;
        } else {
            eScore[t] = 2.f * ccpr[t] * ccfr[t] / (ccpr[t] + ccfr[t]);
        }
    }
}

float DecolorMeasurer::TIS() {
    std::vector<float> X;
    std::vector<float> Y;

    const auto it = std::minmax_element(thresholds.begin(), thresholds.end());
    const auto min = static_cast<float>(*it.first);
    const auto max = static_cast<float>(*it.second);

    for (int t: thresholds) {
        Y.push_back(eScore[t]);
        X.push_back((static_cast<float>(t) - min) / (max - min));
    }

    auto p = linearRegression(X, Y);
    const float tis = std::max(1.f - abs(p.first * p.second), 0.f);
    return tis;
}

float DecolorMeasurer::TIA() {
    std::vector<float> X;
    std::vector<float> Y;

    const auto it = std::minmax_element(thresholds.begin(), thresholds.end());
    const auto min = static_cast<float>(*it.first);
    const auto max = static_cast<float>(*it.second);

    for (int t: thresholds) {
        Y.push_back(eScore[t]);
        X.push_back((static_cast<float>(t) - min) / (max - min));
    }

    auto p = linearRegression(X, Y);
    const float tia = std::max(abs((p.second + (p.first + p.second)) / 2.f), 0.f);
    return tia;
}

float DecolorMeasurer::slope() {
    std::vector<float> X;
    std::vector<float> Y;

    const auto it = std::minmax_element(thresholds.begin(), thresholds.end());
    const auto min = static_cast<float>(*it.first);
    const auto max = static_cast<float>(*it.second);

    for (int t: thresholds) {
        Y.push_back(eScore[t]);
        X.push_back((static_cast<float>(t) - min) / (max - min));
    }

    auto p = linearRegression(X, Y);
    return p.first;
}

DecolorMeasurerOnData::DecolorMeasurerOnData(int minThresh, int maxThresh)
        : minThreshold(minThresh), maxThreshold(maxThresh) {
    for (int t = minThreshold; t <= maxThreshold; ++t) {
        thresholds.push_back(t);
        eScore.insert({t, 0.f});
    }
}

void DecolorMeasurerOnData::addDecolorMeasurer(DecolorMeasurer& measurer) {
    samples++;
    sumCCPR += measurer.getMeanCCPR();
    sumCCFR += measurer.getMeanCCFR();
    sumEScore += measurer.getMeanEScore();
    sumSlope += measurer.slope();

    auto score = measurer.getEScore();
    for (int t = minThreshold; t <= maxThreshold; ++t) {
        eScore[t] += score[t];
    }
}


/*

#include <chrono>

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    using namespace std::chrono_literals;


auto t1 = high_resolution_clock::now();

auto t2 = high_resolution_clock::now();

auto ms_int = duration_cast<milliseconds>(t2 - t1);

 std::cout << ms_int.count() << "ms\n";
*/


//        printf("T: %d, ccpr: %f, ccfr: %f, escore: %f\n", t, ccpr[t], ccfr[t], eScore[t]);

//        printf("T: %d, ccpr: %f, ccfr: %f, escore: %f\n", t, ccpr[t], ccfr[t], eScore[t]);

//    printf("TIA: %f, slope: %f, intercept: %f\n", tia, p.first, p.second);

//    printf("TIS: %f, slope: %f, intercept: %f\n", tis, p.first, p.second);