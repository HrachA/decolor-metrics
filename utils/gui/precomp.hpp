/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __HIGHGUI_H_
#define __HIGHGUI_H_

#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgcodecs.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>


inline void convertToShow(const cv::Mat &src, cv::Mat &dst, bool toRGB = true) {
    const int src_depth = src.depth();
    CV_Assert(src_depth != CV_16F && src_depth != CV_32S);
    cv::Mat tmp;
    switch (src_depth) {
        case CV_8U:
            tmp = src;
            break;
        case CV_8S:
            cv::convertScaleAbs(src, tmp, 1, 127);
            break;
        case CV_16S:
            cv::convertScaleAbs(src, tmp, 1 / 255., 127);
            break;
        case CV_16U:
            cv::convertScaleAbs(src, tmp, 1 / 255.);
            break;
        case CV_32F:
        case CV_64F: // assuming image has values in range [0, 1)
            src.convertTo(tmp, CV_8U, 255., 0.);
            break;
    }
    cv::cvtColor(tmp, dst, toRGB ? cv::COLOR_BGR2RGB : cv::COLOR_BGRA2BGR, dst.channels());
}

inline void convertToShow(const cv::Mat &src, const CvMat *arr, bool toRGB = true) {
    cv::Mat dst = cv::cvarrToMat(arr);
    convertToShow(src, dst, toRGB);
    CV_Assert(dst.data == arr->data.ptr);
}


namespace cv {

    CV_EXPORTS Mutex &getWindowMutex();

    static inline Mutex &getInitializationMutex() { return getWindowMutex(); }

}  // namespace

#endif /* __HIGHGUI_H_ */
