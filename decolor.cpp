//
// Created by Hrach Ayunts on 28.08.21.
//

#include "decolor.h"
#include <cmath>
#include <vector>
#include <limits>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include "utils/utils.h"

using namespace std;
using namespace cv;

// updated eps and coef to normalize r/b like fractions
const float eps = 1.0f;
const float coef = 1.0f;

namespace dc {

    double Decolor::energyCalcu(const vector<double>& Cg, const vector<vector<double> >& polyGrad,
                                const vector<double>& wei) const {
        const size_t size = polyGrad[0].size();
        vector<double> energy(size);
        vector<double> temp(size);
        vector<double> temp1(size);

        for (size_t i = 0; i < polyGrad[0].size(); i++) {
            double val = 0.0;
            for (size_t j = 0; j < polyGrad.size(); j++)
                val = val + (polyGrad[j][i] * wei[j]);
            temp[i] = val - Cg[i];
            temp1[i] = val + Cg[i];
        }

        for (size_t i = 0; i < polyGrad[0].size(); i++)
            energy[i] = -1.0 * log(exp(-1.0 * pow(temp[i], 2) / sigma) + exp(-1.0 * pow(temp1[i], 2) / sigma));

        double sum = 0.0;
        for (size_t i = 0; i < polyGrad[0].size(); i++)
            sum += energy[i];

        return (sum / polyGrad[0].size());

    }

    // constructor
    Decolor::Decolor() {
        kernelx = Mat(1, 2, CV_32FC1);
        kernely = Mat(2, 1, CV_32FC1);
        kernelx.at<float>(0, 0) = 1.0;
        kernelx.at<float>(0, 1) = -1.0;
        kernely.at<float>(0, 0) = 1.0;
        kernely.at<float>(1, 0) = -1.0;
        order = 2;
        sigma = 0.02f;

        nonLocal = false;
    }

    vector<double> Decolor::product(const vector<Vec3i>& comb, const double initRGB[3]) {
        vector<double> res(comb.size());
        for (size_t i = 0; i < comb.size(); i++) {
            double dp = 0.0;
            for (int j = 0; j < 3; j++)
                dp = dp + (comb[i][j] * initRGB[j]);
            res[i] = dp;
        }
        return res;
    }

    // [1, -1] kernel convolution, last column -> 0
    void Decolor::singleChannelGradX(const Mat& img, Mat& dest) const {
        const int w = img.size().width;
        const Point anchor(kernelx.cols - kernelx.cols / 2 - 1, kernelx.rows - kernelx.rows / 2 - 1);
        filter2D(img, dest, -1, kernelx, anchor, 0.0, BORDER_CONSTANT);
        dest.col(w - 1) = 0.0;
    }

    // [1, -1]^T kernel convolution, last row -> 0
    void Decolor::singleChannelGradY(const Mat& img, Mat& dest) const {
        const int h = img.size().height;
        const Point anchor(kernely.cols - kernely.cols / 2 - 1, kernely.rows - kernely.rows / 2 - 1);
        filter2D(img, dest, -1, kernely, anchor, 0.0, BORDER_CONSTANT);
        dest.row(h - 1) = 0.0;
    }

    // vector of partial gradients ([gx..., gy...]), size: width * height * 2
    void Decolor::gradvector(const Mat& img, vector<double>& grad) const {
        Mat dest;
        Mat dest1;
        singleChannelGradX(img, dest);
        singleChannelGradY(img, dest1);

        Mat d_trans = dest.t();
        Mat d1_trans = dest1.t();

        const int height = d_trans.size().height;
        const int width = d_trans.size().width;

        grad.resize(width * height * 2);

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                grad[i * width + j] = d_trans.at<float>(i, j);

        const int offset = width * height;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                grad[offset + i * width + j] = d1_trans.at<float>(i, j);
    }

    // vector of LAB distances of Von Neumann neighborhood (local pixel pairs)
    void Decolor::colorGrad(const Mat& img, vector<double>& Cg) const {
        Mat lab;

        cvtColor(img, lab, COLOR_BGR2Lab);

        vector<Mat> lab_channel;
        split(lab, lab_channel);

        vector<double> ImL;
        vector<double> Ima;
        vector<double> Imb;

        gradvector(lab_channel[0], ImL);
        gradvector(lab_channel[1], Ima);
        gradvector(lab_channel[2], Imb);

        Cg.resize(ImL.size());
        for (size_t i = 0; i < ImL.size(); i++) {
            const double res = sqrt(pow(ImL[i], 2) + pow(Ima[i], 2) + pow(Imb[i], 2)) / 100;
            Cg[i] = res;
        }
    }

    // add r, g, b to comb vector & increase idx
    void Decolor::add_vector(vector<Vec3i>& comb, int& idx, int r, int g, int b) {
        comb.emplace_back(r, g, b);
        idx++;
    }

    // add curGrad vector to polyGrad & increase idx1
    void Decolor::add_to_vector_poly(vector<vector<double> >& polyGrad, const vector<double>& curGrad, int& idx1) {
        polyGrad.push_back(curGrad);
        idx1++;
    }

    // alf - vector of weak orders (1, 0, -1), same indexing system (alf[0] -> [0][0] <? [0][1])
    void Decolor::weak_order(const Mat& im, vector<double>& alf) const {
        Mat img;
        const int h = im.size().height;
        const int w = im.size().width;
        if ((h + w) > 800) {
            const double sizefactor = double(800) / (h + w);
            resize(im, img, Size(cvRound(w * sizefactor), cvRound(h * sizefactor)));
        } else {
            img = im;
        }

        Mat curIm = Mat(img.size(), CV_32FC1);
        vector<Mat> rgb_channel;
        split(img, rgb_channel);

        vector<double> Rg, Gg, Bg;
        gradvector(rgb_channel[2], Rg);
        gradvector(rgb_channel[1], Gg);
        gradvector(rgb_channel[0], Bg);

        // thresholding the values
        vector<double> t1(Rg.size()), t2(Rg.size()), t3(Rg.size());
        vector<double> tmp1(Rg.size()), tmp2(Rg.size()), tmp3(Rg.size());

        const double level = .05;

        for (size_t i = 0; i < Rg.size(); i++) {
            t1[i] = (Rg[i] > level) ? 1.0 : 0.0;
            t2[i] = (Gg[i] > level) ? 1.0 : 0.0;
            t3[i] = (Bg[i] > level) ? 1.0 : 0.0;
            tmp1[i] = (Rg[i] < -1.0 * level) ? 1.0 : 0.0;
            tmp2[i] = (Gg[i] < -1.0 * level) ? 1.0 : 0.0;
            tmp3[i] = (Bg[i] < -1.0 * level) ? 1.0 : 0.0;
        }

        alf.resize(Rg.size());
        for (size_t i = 0; i < Rg.size(); i++)
            alf[i] = (t1[i] * t2[i] * t3[i]);

        for (size_t i = 0; i < Rg.size(); i++)
            alf[i] -= tmp1[i] * tmp2[i] * tmp3[i];
    }

    // returns neighbor diffs for 9 weights, LAB diff & combinations vector
    void Decolor::grad_system(const Mat& im, vector<vector<double> >& polyGrad,
                              vector<double>& Cg, vector<Vec3i>& comb, std::vector<float> gamma) const {
        Mat img;
        int h = im.size().height;
        int w = im.size().width;
        if ((h + w) > 800) {
            const double sizefactor = double(800) / (h + w);
            resize(im, img, Size(cvRound(w * sizefactor), cvRound(h * sizefactor)));
        } else {
            img = im;
        }

        h = img.size().height;
        w = img.size().width;
        colorGrad(img, Cg);

        Mat curIm = Mat(img.size(), CV_32FC1);
        vector<Mat> rgb_channel;
        split(img, rgb_channel);

        int idx = 0, idx1 = 0;
        for (int r = 0; r <= order; r++) {
            for (int g = 0; g <= order; g++) {
                for (int b = 0; b <= order; b++) {
                    if ((r + g + b) <= order && (r + g + b) > 0) {
                        add_vector(comb, idx, r, g, b);
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
//                                if (r == 1 && g == 1) {
                                if (b == 2) {
//                                    curIm.at<float>(i, j) = pow(coef * rgb_channel[2].at<float>(i, j) /
//                                                                (rgb_channel[1].at<float>(i, j) + eps), gamma[0]);
                                    curIm.at<float>(i, j) = fabs(rgb_channel[2].at<float>(i, j) -
                                                                rgb_channel[1].at<float>(i, j));
//                                } else if (g == 1 && b == 1) {
                                } else if (r == 2) {
//                                    curIm.at<float>(i, j) = pow(coef * rgb_channel[1].at<float>(i, j) /
//                                                                (rgb_channel[0].at<float>(i, j) + eps), gamma[1]);
                                    curIm.at<float>(i, j) = fabs(rgb_channel[1].at<float>(i, j) -
                                                                 rgb_channel[0].at<float>(i, j));
//                                } else if (b == 1 && r == 1) {
                                } else if (g == 2) {
//                                    curIm.at<float>(i, j) = pow(coef * rgb_channel[0].at<float>(i, j) /
//                                                                (rgb_channel[2].at<float>(i, j) + eps), gamma[2]);
                                    curIm.at<float>(i, j) = fabs(rgb_channel[0].at<float>(i, j) -
                                                                 rgb_channel[2].at<float>(i, j));
//                                } else if (r == 2) {
//                                    curIm.at<float>(i, j) = (rgb_channel[2].at<float>(i, j) *
//                                                             log2(rgb_channel[2].at<float>(i, j) + 1.f));
//                                } else if (g == 2) {
//                                    curIm.at<float>(i, j) = (rgb_channel[1].at<float>(i, j) *
//                                                             log2(rgb_channel[1].at<float>(i, j) + 1.f));
//                                } else if (b == 2) {
//                                    curIm.at<float>(i, j) = (rgb_channel[0].at<float>(i, j) *
//                                                             log2(rgb_channel[0].at<float>(i, j) + 1.f));
                                } else {
                                    curIm.at<float>(i, j) = static_cast<float>(
                                            pow(rgb_channel[2].at<float>(i, j), r) *
                                            pow(rgb_channel[1].at<float>(i, j), g) *
                                            pow(rgb_channel[0].at<float>(i, j), b));
                                }
                            }
                        }
                        vector<double> curGrad;
                        gradvector(curIm, curGrad);
                        add_to_vector_poly(polyGrad, curGrad, idx1);
                    }
                }
            }
        }

//        print("I am here!");
//        print(comb.size());
//        print(polyGrad.size());
    }

    // solve ???
    void Decolor::wei_update_matrix(const vector<vector<double> >& poly, const vector<double>& Cg, Mat& X) {
        const int size = static_cast<int>(poly.size());
        const int size0 = static_cast<int>(poly[0].size());
        Mat P = Mat(size, size0, CV_32FC1);

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size0; j++)
                P.at<float>(i, j) = static_cast<float>(poly[i][j]);

        const Mat P_trans = P.t();
        Mat B = Mat(size, size0, CV_32FC1);
        for (int i = 0; i < size; i++) {
            for (int j = 0, end = int(Cg.size()); j < end; j++)
                B.at<float>(i, j) = static_cast<float>(poly[i][j] * Cg[j]);
        }

        Mat A = P * P_trans;
        solve(A, B, X, DECOMP_NORMAL);
    }

    // initializes wights for corresponding combination
    void Decolor::wei_inti(const vector<Vec3i>& comb, vector<double>& wei) {
        double initRGB[3] = {.33, .33, .33};
//        double initRGB[3] = {0.15, 0.05, 0.8};
//        double initRGB[3] = {.299, .587, .114};

        wei = product(comb, initRGB);

        vector<int> sum(comb.size());

        for (size_t i = 0; i < comb.size(); i++)
            sum[i] = (comb[i][0] + comb[i][1] + comb[i][2]);

        for (size_t i = 0; i < sum.size(); i++) {
            if (sum[i] == 1)
                wei[i] = wei[i] * double(1);
            else
                wei[i] = wei[i] * double(0);
        }

//        printf("Initial weights ");
//        for (double w : wei) {
//            printf("%f, ", w);
//        }
//        printf("\n");

        sum.clear();
    }

    void Decolor::grayImContruct(vector<double>& wei, const Mat& img, Mat& Gray, std::vector<float> gamma) const {
        const int h = img.size().height;
        const int w = img.size().width;

        vector<Mat> rgb_channel;
        split(img, rgb_channel);

        int kk = 0;

        for (int r = 0; r <= order; r++)
            for (int g = 0; g <= order; g++)
                for (int b = 0; b <= order; b++)
                    if ((r + g + b) <= order && (r + g + b) > 0) {
                        for (int i = 0; i < h; i++)
                            for (int j = 0; j < w; j++) {

                                float color;
//                                if (r == 1 && g == 1) {
                                if (b == 2) {
//                                    color = pow(coef * rgb_channel[2].at<float>(i, j) /
//                                                (rgb_channel[1].at<float>(i, j) + eps), gamma[0]);
                                    color = fabs(rgb_channel[2].at<float>(i, j) -
                                            rgb_channel[1].at<float>(i, j));
//                                } else if (g == 1 && b == 1) {
                                } else if (r == 2) {
//                                    color = pow(coef * rgb_channel[1].at<float>(i, j) /
//                                                (rgb_channel[0].at<float>(i, j) + eps), gamma[1]);
                                    color = fabs(rgb_channel[1].at<float>(i, j) -
                                                 rgb_channel[0].at<float>(i, j));
//                                } else if (b == 1 && r == 1) {
                                } else if (g == 2) {
//                                    color = pow(coef * rgb_channel[0].at<float>(i, j) /
//                                                (rgb_channel[2].at<float>(i, j) + eps), gamma[2]);
                                    color = fabs(rgb_channel[0].at<float>(i, j) -
                                                 rgb_channel[2].at<float>(i, j));
//                                } else if (r == 2) {
//                                    color = (rgb_channel[2].at<float>(i, j) *
//                                             log2(rgb_channel[2].at<float>(i, j) + 1.f));
//                                } else if (g == 2) {
//                                    color = (rgb_channel[1].at<float>(i, j) *
//                                             log2(rgb_channel[1].at<float>(i, j) + 1.f));
//                                } else if (b == 2) {
//                                    color = (rgb_channel[0].at<float>(i, j) *
//                                             log2(rgb_channel[0].at<float>(i, j) + 1.f));
                                } else {
                                    color = pow(rgb_channel[2].at<float>(i, j), r) *
                                            pow(rgb_channel[1].at<float>(i, j), g) *
                                            pow(rgb_channel[0].at<float>(i, j), b);
                                }

                                Gray.at<float>(i, j) = Gray.at<float>(i, j) +
                                                       static_cast<float>(wei[kk]) * color;

                            }
                        kk = kk + 1;
                    }

        double minval, maxval;
        minMaxLoc(Gray, &minval, &maxval);

        Gray -= minval;
        Gray /= maxval - minval;
    }

    Mat color_clustering(const Mat& img, int clusterCount = 50) {
        Mat labels, centers;
        Mat points = img.reshape(1, img.rows*img.cols);
        kmeans(points, clusterCount, labels,
               TermCriteria(TermCriteria::EPS, 10, 0.0001),
               20, KMEANS_PP_CENTERS, centers);

        Mat clustered = img.clone();
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                int ind = labels.at<int>(i * img.cols + j);
                clustered.at<Vec3f>(i, j) = centers.at<Vec3f>(ind);
            }
        }

        return clustered;
    }

    void decolor(InputArray _src, OutputArray _dst, OutputArray _color_boost, const std::vector<float>& gamma,
                 bool nonLocal, int clusterCount, float resizeScale) {
        Mat I = _src.getMat();
        _dst.create(I.size(), CV_8UC1);
        Mat dst = _dst.getMat();

        _color_boost.create(I.size(), CV_8UC3);
        Mat color_boost = _color_boost.getMat();

        CV_Assert(!I.empty() && (I.channels() == 3));

        // Parameter Setting
        const int maxIter = 15;
        const double tol = .0001;
        int iterCount = 0;
        double E = 0;
        double pre_E = std::numeric_limits<double>::infinity();

        Mat img;
        I.convertTo(img, CV_32FC3, 1.0 / 255.0);

        // Initialization
        Decolor obj;
        obj.nonLocal = nonLocal;

        vector<double> Cg;
        vector<vector<double> > polyGrad;
        vector<Vec3i> comb;
        vector<double> alf;

        cv::Mat imgCopy = img.clone();

        cv::Mat clustered;
        if (nonLocal) {
            static int k = 1;
            clustered = color_clustering(imgCopy, clusterCount);
            if (resizeScale > 0) {
                cv::resize(clustered, clustered, cv::Size(), resizeScale, 1, cv::INTER_LINEAR);
            }
            cv::hconcat(imgCopy, clustered, imgCopy);
//            cv::imwrite("../images/test_results/cons_concat_" + std::to_string(k++) + "_.png", imgCopy * 255.);
//            showAndWait("Wow", img);
        }

        obj.grad_system(imgCopy, polyGrad, Cg, comb, gamma);
        obj.weak_order(imgCopy, alf);

        // Solver
        Mat Mt = Mat(int(polyGrad.size()), int(polyGrad[0].size()), CV_32FC1);
        obj.wei_update_matrix(polyGrad, Cg, Mt);

        vector<double> wei;
        obj.wei_inti(comb, wei);

        //////////////////////////////// main loop starting ////////////////////////////////////////

        vector<double> G_pos(alf.size());
        vector<double> G_neg(alf.size());
        vector<double> EXPsum(G_pos.size());
        vector<double> EXPterm(G_pos.size());
        vector<double> temp(polyGrad[0].size());
        vector<double> temp1(polyGrad[0].size());
        vector<double> temp2(EXPsum.size());
        vector<double> wei1(polyGrad.size());

        while (sqrt(pow(E - pre_E, 2)) > tol) {
            iterCount += 1;
            pre_E = E;

            for (size_t i = 0; i < polyGrad[0].size(); i++) {
                double val = 0.0;
                for (size_t j = 0; j < polyGrad.size(); j++)
                    val = val + (polyGrad[j][i] * wei[j]);
                temp[i] = val - Cg[i];
                temp1[i] = val + Cg[i];
            }

            for (size_t i = 0; i < alf.size(); i++) {
                const double sqSigma = obj.sigma * obj.sigma;
                const double pos = ((1 + alf[i]) / 2) * exp(-1.0 * 0.5 * (temp[i] * temp[i]) / sqSigma);
                const double neg = ((1 - alf[i]) / 2) * exp(-1.0 * 0.5 * (temp1[i] * temp1[i]) / sqSigma);
                G_pos[i] = pos;
                G_neg[i] = neg;
            }

            for (size_t i = 0; i < G_pos.size(); i++)
                EXPsum[i] = G_pos[i] + G_neg[i];

            for (size_t i = 0; i < EXPsum.size(); i++)
                temp2[i] = (EXPsum[i] == 0) ? 1.0 : 0.0;

            for (size_t i = 0; i < G_pos.size(); i++)
                EXPterm[i] = (G_pos[i] - G_neg[i]) / (EXPsum[i] + temp2[i]);

            for (int i = 0; i < int(polyGrad.size()); i++) {
                double val1 = 0.0;
                for (int j = 0; j < int(polyGrad[0].size()); j++) {
                    val1 = val1 + (Mt.at<float>(i, j) * EXPterm[j]);
                }
                wei1[i] = val1;
            }

//            printf("Weights: ");
            for (size_t i = 0; i < wei.size(); i++) {
                wei[i] = wei1[i];
//                printf("%f, ", wei[i]);
            }
//            printf("\n");

            E = obj.energyCalcu(Cg, polyGrad, wei);

            if (iterCount > maxIter)
                break;
        }

//        printf("Iteration count: %d\n", iterCount);
//        printf("Final Weights: ");
        double sum = 0;
        for (double w: wei) {
//            printf("%f, ", w);
            sum += abs(w);
        }

        Mat Gray = Mat::zeros(img.size(), CV_32FC1);
        obj.grayImContruct(wei, img, Gray, gamma);

        Gray.convertTo(dst, CV_8UC1, 255);
//        printf("Weights sum: %f\n", sum);
        if (sum < 0.001) {
            print("Weights sum is 0");
            cvtColor(I, dst, COLOR_BGR2GRAY);
        }

        ///////////////////////////////////       Contrast Boosting   /////////////////////////////////

//        cv::hconcat(I, clustered, I);
        Mat lab;
        cvtColor(I, lab, COLOR_BGR2Lab);

        vector<Mat> lab_channel;
        split(lab, lab_channel);

        dst.copyTo(lab_channel[0]);

        merge(lab_channel, lab);

        cvtColor(lab, color_boost, COLOR_Lab2BGR);
    }
}