//
// Created by Hrach Ayunts on 18.05.22.
//

#include "AME.h"
#include "utils/utils.h"

const float eps = 0.000001;

float modulation(float mx, float mn) {
    return (mx - mn) / (mx + mn + eps);
}

float AME(const cv::Mat& source, int kernelSize) {
    float avg = 0;
    float count = 0;

    for (int i = 0; i < source.rows - kernelSize; i++) {
        for (int j = 0; j < source.cols - kernelSize; j++) {
            cv::Rect roi(j, i, kernelSize - 1, kernelSize - 1);

            double minVal;
            double maxVal;
            minMaxLoc(source(roi), &minVal, &maxVal, nullptr, nullptr);
            if (maxVal - minVal < 5) { // || maxVal - minVal > 250) {
                continue;
            }
            float mod = modulation(maxVal, minVal);

//            printf("Modulation: %f, max: %f, min %f\n", mod, maxVal, minVal);

            const float alpha = eps;
            const float exp = 0.95f;
            const float ent = fabs(exp * powf((mod + alpha), exp) * logf(mod + alpha));

//            printf("Min: %f, Max: %f, Mod: %f, Entropy: %f\n", minVal, maxVal, mod, ent);
            avg += ent;
            count++;
        }
    }

    return avg / count;
}

float MeanAME(const cv::Mat& source) {
    const float ame3 = AME(source, 3);
    const float ame5 = AME(source, 5);
    const float ame9 = AME(source, 9);

    return pow(ame3 * ame5 * ame9, 1.f / 3);
}

//
float newAME(const cv::Mat& source, const cv::Mat& gray, int kernelSize) {
    cv::Mat grayFloat, sourceNorm, lab;
    gray.convertTo(grayFloat, CV_32F);
    source.convertTo(sourceNorm, CV_32F, 1.0 / 255.0, 0);
    cvtColor(sourceNorm, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    split(source, channels);

    float avg = 0;
    float count = 0;
    for (int i = 0; i < gray.rows - kernelSize; i++) {
        for (int j = 0; j < gray.cols - kernelSize; j++) {
            cv::Rect roi(j, i, kernelSize, kernelSize);

            cv::Mat grayROI = grayFloat(roi);
            cv::Mat labROI = lab(roi);

            double minVal, maxVal;
            double diff = 0;
            cv::Point maxP, minP;
            std::vector<cv::Point> minPV(3), maxPV(3);
//            minMaxLoc(grayROI, &minVal, &maxVal, &minP, &maxP);
            for (int k = 0; k < 3; k++) {
                minMaxLoc(channels[k](roi), &minVal, &maxVal, &minPV[k], &maxPV[k]);
//                printf("K = %d, MaxVal: %f, minVal: %f\n", k, maxVal, minVal);
                if (diff < maxVal - minVal) {
                    diff = maxVal - minVal;
                    maxP = maxPV[k];
                    minP = minPV[k];
                }
            }

            const float deltaColor = fabs(distanceLAB(
                    labROI.at<cv::Vec3f>(maxP), labROI.at<cv::Vec3f>(minP)));
            const float grayDist = fabs(grayROI.at<float>(maxP) - grayROI.at<float>(minP)) / 2.55f;
//            if (i == 35 && j == 35 && deltaColor > 2) {
//                printf("Diff: %f , deltaColor: %f, grayDist: %f\n\n", diff, deltaColor, grayDist);
//            }
//            const float grayDist2 = maxVal - minVal;
//
//            if (fabs(grayDist - grayDist2) > 0.00001) {
//                printf("%f == %f \n", grayDist, grayDist2);
//            }

//            if (grayDist < 3) {
//                continue;
//            }

//            float delta = fabs( grayDist - deltaColor) / 100.f;
            if (grayDist > 5) {
                avg++;
            }
//            if (deltaColor <= 5) {
//                delta = 0.f;
//            }
//            printf("Modulation: %f, max: %f, min %f\n", mod, maxVal, minVal);

//            avg += delta;
            count++;
        }
    }
    return avg / count;
//    return 1 - avg / count;
}

float newMeanAME(const cv::Mat& source, const cv::Mat& gray, bool calc, const std::vector<int>& sizes) {
    if (calc) {
        int k = 2 * (gray.cols + gray.rows) / 100 + 1;
//        printf("Calculated kernel size for this image is %d\n", k);
        return newAME(source, gray, k);
    }

    const float ame1 = newAME(source, gray, sizes[0]);
    const float ame2 = newAME(source, gray, sizes[1]);
    const float ame3 = newAME(source, gray, sizes[2]);

    return pow(ame1 * ame2 * ame3, 1.f / 3);
}

