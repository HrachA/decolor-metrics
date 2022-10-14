//
// Created by Hrach Ayunts on 28.08.21.
//

#ifndef DECOLORIZATION_DECOLOR_H
#define DECOLORIZATION_DECOLOR_H

#include "opencv2/photo.hpp"
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

namespace dc {

    class Decolor {
    private:
        Mat kernelx;
        Mat kernely;
        int order;

    public:
        float sigma;
        bool nonLocal;

        Decolor();

        static vector<double> product(const vector<Vec3i> &comb, const double initRGB[3]);

        double
        energyCalcu(const vector<double> &Cg, const vector<vector<double> > &polyGrad, const vector<double> &wei) const;

        void singleChannelGradX(const Mat &img, Mat &dest) const;

        void singleChannelGradY(const Mat &img, Mat &dest) const;

        void gradvector(const Mat &img, vector<double> &grad) const;

        void colorGrad(const Mat &img, vector<double> &Cg) const;

        static void add_vector(vector<Vec3i> &comb, int &idx, int r, int g, int b);

        static void add_to_vector_poly(vector<vector<double> > &polyGrad, const vector<double> &curGrad, int &idx1);

        void weak_order(const Mat &img, vector<double> &alf) const;

        void grad_system(const Mat &img, vector<vector<double> > &polyGrad,
                         vector<double> &Cg, vector<Vec3i> &comb, std::vector<float> gamma) const;

        static void wei_update_matrix(const vector<vector<double> > &poly, const vector<double> &Cg, Mat &X);

        static void wei_inti(const vector<Vec3i> &comb, vector<double> &wei);

        void grayImContruct(vector<double> &wei, const Mat &img, Mat &Gray, std::vector<float> gamma) const;
    };

    void decolor(InputArray _src, OutputArray _dst, OutputArray _color_boost, const std::vector<float>& gamma,
                 bool nonLocal = false, int clusterCount = 50, float resizeScale = 0.3);
}

#endif //DECOLORIZATION_DECOLOR_H
