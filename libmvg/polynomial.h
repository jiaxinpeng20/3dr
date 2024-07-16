//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)

#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/Polynomials>


namespace YH3DR
{
    namespace MultiviewGeometry
    {
        class UPolynomial
        {
            private:
            
            public:
                UPolynomial();
                ~UPolynomial();
                static std::vector<double> GetRealRoots(Eigen::VectorXd& inCoefficients);
        };
    }
}

