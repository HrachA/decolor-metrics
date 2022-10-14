//
// Created by Hrach Ayunts on 07.06.22.
//

#ifndef DECOLOR_METRICS_IMAGE_COMPONENTS_H
#define DECOLOR_METRICS_IMAGE_COMPONENTS_H

#include <opencv2/core.hpp>

enum class ImageComponents {
    R,
    G,
    B,
    R_mul_G,
    R_mul_B,
    G_mul_B,
    R_pow_2,
    G_pow_2,
    B_pow_2,
    R_dif_G,
    G_dif_B,
    B_dif_R
};

cv::Mat getImageComponent(const cv::Mat& source, ImageComponents ic, float coef = 1.f);

#endif //DECOLOR_METRICS_IMAGE_COMPONENTS_H
