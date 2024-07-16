//Copyright (c) 2024, National University of Defense Technology (NUDT)
// All rights reserved.
//
//
// Written by Dr. Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)

#include "two_view_geometry.h"

namespace YH3DR
{
    namespace MultiviewGeometry
    {
        UFMatrix::UFMatrix()
        {
            
        }
        
        UFMatrix::UFMatrix(std::vector<std::pair<int, int>>& inMatchingFeatures)
        {
            this->matchingFeatures = inMatchingFeatures;
        }
        
        UFMatrix::~UFMatrix()
        {
            
        }
        
        void UFMatrix::SetMatchingFeatures(std::vector<std::pair<int, int>>& inMatchingFeatures)
        {
            this->matchingFeatures = inMatchingFeatures;
        }
        
        
        void UFMatrix::CalSampleCount()
        {
            //Calculate the number of RANSAC samples iteratively
            int sample_count = 0;

            
        }
        
        void UFMatrix::CalFMatrix()
        {
            
        }
        
        Eigen::Matrix3d UFMatrix::GetFMatrix()
        {
            return this->F;
        }
    }
}
