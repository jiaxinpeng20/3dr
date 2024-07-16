//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


#include <vector>
#include <Eigen/Eigen>
#include "polynomial.h"

namespace YH3DR
{
    namespace MultiviewGeometry
    {
        class UFundamentalMatrix
        {
            private:
            enum class EstimationMethod{
                M7Point,
                M8Point
            };

			void NormalizeImagePoints(Eigen::Matrix2d& P, Eigen::Matrix2d& Q, Eigen::MatrixXd& NM);
    
            std::vector<Eigen::Matrix3d> SolveBy7PointMethod(Eigen::Matrix2d& p1, Eigen::Matrix2d& p2);
            std::vector<Eigen::Matrix3d> SolveBy8PointMethod(Eigen::Matrix2d& p1, Eigen::Matrix2d& p2);
    
            public:
				UFundamentalMatrix();
				~UFundamentalMatrix();
				static std::vector<Eigen::Matrix3d> Solve(EstimationMethod method, Eigen::Matrix2d& p1, Eigen::Matrix2d& p2);
        };



    }
}




