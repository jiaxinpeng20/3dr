// Copyright (c) 2024, National University of Defense Technology.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of National University of Defense Technology nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Jiaxin Peng (jiaxinpeng.nudt-at-gmail-dot-com)

#include "UImage.h"
#include <omp.h>

namespace XSFM
{
    namespace Algorithm
    {
        UImage::UImage(std::vector<std::string>& inImageFileList, int inNumOfThreads)
        {
            imageFileList = inImageFileList;
            numOfImages = inImageFileList.size();

            int maxThreads = std::thread::hardware_concurrency() + 1;
            if(inNumOfThreads <= 0 || inNumOfThreads > maxThreads)
            {
                inNumOfThreads = maxThreads;
            }

            numOfThreads = inNumOfThreads;
        }

        UImage::UImage(std::vector<std::string>& inImageFileList)
        {
            imageFileList = inImageFileList;
            numOfImages = inImageFileList.size();

            int maxThreads = std::thread::hardware_concurrency();
            numOfThreads = maxThreads;
        }
        
        UImage::~UImage()
        {
            
        }
        
        void UImage::ReadImageFromPath()
        {
            imageDataList.resize(numOfImages);//allocate memory spacess

            #pragma omp parallel for num_threads(numOfThreads)
            for(int i = 0; i < numOfImages; i++)
            {
                cv::Mat _imageData = cv::imread(imageFileList.at(i), cv::IMREAD_COLOR);
                imageDataList.at(i) = _imageData;
            }  

            //Remove invalid image data
            std::vector<std::string>::iterator iter1 = imageFileList.begin();
            for(std::vector<cv::Mat>::iterator iter0 = imageDataList.begin() ;iter0 != imageDataList.end(); )
            {
                if(iter0->empty())
                {
                    iter0 = imageDataList.erase(iter0);
                    iter1 = imageFileList.erase(iter1);
                    numOfImages--;
                }
                else
                {
                    iter0++;
                    iter1++;
                }
            }
            
        }
        

        void UImage::ComputeImageFeatures()
        {
            //#pragma omp parallel for num_threads(numOfThreads)
            for(int i = 0; i < numOfImages; i++)//Make sure that openCV is already multi-threaded.
            {
                cv::Mat imageGrayData;
                cv::cvtColor(imageDataList.at(i), imageGrayData, cv::COLOR_BGR2GRAY);

                cv::Ptr<cv::SIFT> sift = cv::SIFT::create(numOfKeyPoints);
                std::vector<cv::KeyPoint> keyPoints;
                sift->detect(imageGrayData, keyPoints);
                
                cv::Mat logoImage;
                cv::drawKeypoints(imageGrayData, keyPoints, logoImage);

                //Compute descriptors for all features 
                cv::Mat _cv_image_descriptors;
                sift->compute(imageGrayData, keyPoints, _cv_image_descriptors);

                Eigen::MatrixXd _eigen_image_descriptors;
                cv::cv2eigen(_cv_image_descriptors, _eigen_image_descriptors);
                descriptors.push_back(_eigen_image_descriptors);

            }
            
            
            


            //cv::imwrite("TEST", logoImage);
            //cv::waitKey(0);

        }

        std::vector<Eigen::MatrixXd> UImage::GetImageDescriptors()
        {
            return descriptors;
        }

        void UImage::GetImageExif()
        {
            
        }
        
    }
}
