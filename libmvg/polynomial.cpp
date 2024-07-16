//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/Polynomials>
#include "polynomial.h"


namespace YH3DR
{
    namespace MultiviewGeometry
    {
        UPolynomial::UPolynomial()
        {
            
        }
        
        UPolynomial::~UPolynomial()
        {
            
        }
        
        std::vector<double> UPolynomial::GetRealRoots(Eigen::VectorXd& inCoefficients)
        {
            Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
            solver.compute(inCoefficients);
            std::vector<double> _real_roots;
            solver.realRoots(_real_roots);
            
            return _real_roots;
        }
    }
}
