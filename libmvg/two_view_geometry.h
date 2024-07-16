//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


#include <vector>
#include <Eigen/Eigen>
#include "ransac.h"

namespace YH3DR
{
    namespace MultiviewGeometry
    {
        class UFMatrix
        {
            private:
                std::vector<std::pair<int, int>> matchingFeatures;
                Eigen::Matrix3d F;
                int sampleCount;
                
                int GetSampleCount();
                
                
                
                
            public:
                UFMatrix();
                UFMatrix(std::vector<std::pair<int, int>>& inMatchingFeatures);
                ~UFMatrix();
                
                Eigen::Matrix3d GetFMatrix();
                
                void FilterCorrespondences();
                void SetMatchingFeatures(std::vector<std::pair<int, int>>& inMatchingFeatures);
                void CalSampleCount();
                void CalFMatrix();
                void CalFundamentalMatrixNEP(); //normalized eight-point algorithm
                void CalFundamentalMatrixAM(); //algebraic minimization algorithm
                void CalFundamentalMatrixMSE(); //reprojection error with the Levenberg-Marquardt algorithm
                
                
            
        };
    }
}
