#ifndef INTERACTIVE_SEGMENTATION_UTILS_H
#define INTERACTIVE_SEGMENTATION_UTILS_H

#include <string>
#include <opencv2/core.hpp>
#include <iostream>

#define print(x) std::cout << x << std::endl

#define st(x) std::to_string(x)

cv::Mat concatHorizontal(std::vector<cv::Mat> images);

cv::Mat concatHorizontal(std::vector<std::pair<cv::Mat, std::string>> images);

cv::Mat addText(const cv::Mat& image, const std::string& text);

cv::Mat addText(const cv::Mat& image, const std::vector<std::string>& strings, std::vector<cv::Scalar>& colors);

std::string getImageType(int number);

void printImageInfo(const cv::Mat& image, const std::string& name = "");

void showHistogram(cv::Mat& image);

void showAndWait(const std::string& window, const cv::Mat& img, int waitKey = 0);

void splitAndShowYCC(const cv::Mat& img);

void splitAndShowLab(const cv::Mat& img);

void splitAndShowHSV(const cv::Mat& img);

cv::Mat negative(const cv::Mat& src);

bool imageReadAndResize(const std::string& path, cv::Mat& dst, int maxSide = 1000);

void gammaCorrection(const cv::Mat& src, cv::Mat& dest, float gamma = 0.8);

void getCIEY(cv::Mat& source, cv::Mat& dest);

float distanceLAB(const cv::Vec3f& v1, const cv::Vec3f& v2);

float distanceLAB(const cv::Vec3b& v1, const cv::Vec3b& v2);

float clampFloat(float val, float low = 0.f, float high = 255.f);

#endif //INTERACTIVE_SEGMENTATION_UTILS_H
