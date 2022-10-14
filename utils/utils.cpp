#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "gui/window_QT.h"

cv::Mat concatHorizontal(std::vector<cv::Mat> images) {
    cv::Mat result;

    if (images.size() > 1) {
        for (int i = 0; i < images.size(); ++i) {
            if (images[i].channels() == 1) {
                cv::cvtColor(images[i], images[i], cv::COLOR_GRAY2BGR);
            }
            if (i == 0) {
                result = images[0];
            } else {
                cv::hconcat(result, images[i], result);
            }
        }
    }

    return result;
}

cv::Mat addText(const cv::Mat& image, const std::string& text) {
    cv::Mat result = image.clone();

    cv::Scalar forColor({20, 0, 20});
    cv::Scalar backColor({255, 255, 255});

    int delta = image.rows / 10;
    cv::copyMakeBorder(result, result, 0, delta, 0, 0, cv::BORDER_CONSTANT, backColor);

    double fontScale = image.rows / 800.0;
    cv::Point point(2 * image.cols / 5, 21 * image.rows / 20);
    cv::putText(result, text, point, cv::FONT_HERSHEY_COMPLEX, fontScale, forColor, (int) (3.0 * fontScale));

    cv::copyMakeBorder(result, result, 1, 0, 1, 1, cv::BORDER_CONSTANT, backColor);
    return result;
}

cv::Mat addText(const cv::Mat& image, const std::vector<std::string>& strings, std::vector<cv::Scalar>& colors) {
    cv::Mat result = image.clone();

    if (colors.empty()) {
        for (int i = 0; i < strings.size(); ++i) {
            colors.push_back(cv::Scalar(0,0, 0));
        }
    }

//    cv::Scalar forColor({20, 0, 20});
    cv::Scalar backColor({255, 255, 255});

    int delta = image.rows / 10;
    double fontScale = image.rows / 500.0;
    for (int i = 0; i < strings.size(); ++i) {
        cv::copyMakeBorder(result, result, 0, delta, 0, 0, cv::BORDER_CONSTANT, backColor);

        cv::Point point(2 * image.cols / 5, i * delta + 21 * image.rows / 20);
        cv::putText(result, strings[i], point, cv::FONT_HERSHEY_COMPLEX, fontScale, colors[i], (int) (3.0 * fontScale));
    }

    cv::copyMakeBorder(result, result, 1, 0, 1, 1, cv::BORDER_CONSTANT, backColor);
    return result;
}

cv::Mat concatHorizontal(std::vector<std::pair<cv::Mat, std::string>> images) {
    cv::Mat result;
    std::vector<cv::Mat> newImages(images.size());

    for (int i = 0; i < images.size(); ++i) {
        newImages[i] = addText(images[i].first, images[i].second);
    }

    return concatHorizontal(newImages);

//    if (images.size() > 1) {
//        for (int i = 0; i < images.size(); ++i) {
//            if (images[i].channels() == 1) {
//                cv::cvtColor(images[i], images[i], cv::COLOR_GRAY2BGR);
//            }
//            if (i == 0) {
//                result = images[0];
//            } else {
//                cv::hconcat(result, images[i], result);
//            }
//        }
//    }
//
//    return result;
}

std::string getImageType(int number) {
    int imgTypeInt = number % 8;
    std::string imgTypeString;

    switch (imgTypeInt) {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    int channel = (number / 8) + 1;
    std::stringstream type;
    type << "CV_" << imgTypeString << "C" << channel;

    return type.str();
}

void printImageInfo(const cv::Mat& image, const std::string& name) {

    if (!name.empty()) {
        std::cout << name << " image" << std::endl;
    }

    double min, max;
    minMaxLoc(image, &min, &max);

    std::cout << "Image width is " << image.size().width << " and height is " << image.size().height << std::endl;
    std::cout << "Image is in range (" << min << ", " << max << ")" << std::endl;
    std::cout << "Image type is " << getImageType(image.type()) << std::endl;
}

void showHistogram(cv::Mat& image) {
    using namespace cv;
    int histSize = 256;

    float range[] = {0, 256}; //the upper boundary is exclusive
    const float* histRange = {range};

    bool uniform = true, accumulate = false;

    Mat hist;
    calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8U, Scalar(0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             Scalar(255), 2, 8, 0);
    }

    imshow("Source image", image);
    imshow("Histogram", histImage);
}

void showAndWait(const std::string& window, const cv::Mat& img, int waitKey) {
    CvMat cImage = cvMat(img);
    cvShowImage(window.c_str(), &cImage);
    cvWaitKey(waitKey);
}

// TODO: make one function
void splitAndShowYCC(const cv::Mat& img) {
    cv::Mat imageYCC;
    cv::cvtColor(img, imageYCC, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels(3);
    split(imageYCC, channels);

    printImageInfo(channels[0], "Y");
    printImageInfo(channels[1], "Cr");
    printImageInfo(channels[2], "Cb");
//    showAndWait("YCC", concatHorizontal(channels));
}

void splitAndShowLab(const cv::Mat& img) {
    cv::Mat imageLab;
    cv::cvtColor(img, imageLab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels(3);
    split(imageLab, channels);

    printImageInfo(channels[0], "L");
    printImageInfo(channels[1], "a");
    printImageInfo(channels[2], "b");

    std::string path = "/Users/hrachayunts/Documents/Thesis/Colorization/Decolorization/images/Cadik/paper_images/3/boost/";
    cv::imwrite(path + "L.png", channels[0]);
    cv::imwrite(path + "a.png", channels[1]);
    cv::imwrite(path + "b.png", channels[2]);

//    showAndWait("Lab", concatHorizontal(channels));
}

void splitAndShowHSV(const cv::Mat& img) {
    cv::Mat imageHSV;
    cv::cvtColor(img, imageHSV, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels(3);
    split(imageHSV, channels);

    printImageInfo(channels[0], "H");
    printImageInfo(channels[1], "S");
    printImageInfo(channels[2], "V");
//    showAndWait("HSV", concatHorizontal(channels));
}

cv::Mat negative(const cv::Mat& src) {
    cv::Mat dst = src.clone();
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src.channels() == 3) {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255) - src.at<cv::Vec3b>(i, j);
            } else {
                dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
            }
        }
    }
    return dst;
}

bool imageReadAndResize(const std::string& path, cv::Mat& dst, int maxSide) {
    dst = cv::imread(path);
    if (dst.empty()) {
        print("Image is empty!");
        return false;
    }

    cv::Size size = dst.size();
    if (std::max(size.width, size.height) > maxSide) {
        if (size.height > size.width) {
            size = {maxSide * size.width / size.height, maxSide};
        } else {
            size = {maxSide, maxSide * size.height / size.width};
        }
    }

    cv::resize(dst, dst, size, 0, 0);
    return true;
}

void gammaCorrection(const cv::Mat& src, cv::Mat& dest, float gamma) {
    cv::Mat1d dsrc;
    src.convertTo(dsrc, CV_64F);

    double min, max;
    minMaxLoc(dsrc, &min, &max);

    cv::Mat ddst;
    cv::pow(dsrc / max, gamma, ddst);

    ddst = ddst * 255.;
    ddst.convertTo(dest, CV_8U);
}

void getCIEY(cv::Mat& source, cv::Mat& dest) {
    cv::Mat imgXYZ;
    cv::cvtColor(source, imgXYZ, cv::COLOR_BGR2XYZ);

    std::vector<cv::Mat> channels(3);
    split(imgXYZ, channels);
    channels[1].copyTo(dest);
}

float distanceLAB(const cv::Vec3f& v1, const cv::Vec3f& v2) {
    return sqrtf((v1[0] - v2[0]) * (v1[0] - v2[0]) +
                 (v1[1] - v2[1]) * (v1[1] - v2[1]) +
                 (v1[2] - v2[2]) * (v1[2] - v2[2]));
}

float distanceLAB(const cv::Vec3b& v1, const cv::Vec3b& v2) {
    float l1 = static_cast<float>(v1[0]) / 2.55;
    float l2 = static_cast<float>(v2[0]) / 2.55;
    float a1 = static_cast<float>(v1[1]) - 128.f;
    float a2 = static_cast<float>(v2[1]) - 128.f;
    float b1 = static_cast<float>(v1[2]) - 128.f;
    float b2 = static_cast<float>(v2[2]) - 128.f;
    return sqrtf((l1 - l2) * (l1 - l2) +
                 (a1 - a2) * (a1 - a2) +
                 (b1 - b2) * (b1 - b2));
}

float clampFloat(float val, float low, float high) {
    if (val < low) return low;
    if (val > high) return high;
    return val;
}