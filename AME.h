//
// Created by Hrach Ayunts on 18.05.22.
//

#ifndef DECOLOR_METRICS_AME_H
#define DECOLOR_METRICS_AME_H

#include "opencv2/opencv.hpp"

float AME(const cv::Mat& source, int kernelSize = 5);

float MeanAME(const cv::Mat& source);

float newAME(const cv::Mat& source, const cv::Mat& gray, int kernelSize = 5);

float
newMeanAME(const cv::Mat& source, const cv::Mat& gray, bool calc = false, const std::vector<int>& sizes = {3, 9, 15});

#endif //DECOLOR_METRICS_AME_H
