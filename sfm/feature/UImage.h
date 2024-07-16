//Copyright (c) 2024, National University of Defense Technology (NUDT)
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
//     * Neither the name of the copyright holder nor the names of its 
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
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
// Author: Jiaxin Peng (jiaxinpeng-dot-nudt-at-gmail-dot-com)


//Image class for data reading and fearure extraction, one image is for one computing node

#ifndef YH3DR_LIB3DR_UIMAGE_H
#define YH3DR_LIB3DR_UIMAGE_H

#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

#include "thirdParty/TinyEXIF-master/TinyEXIF.h"
#include "feature/UCamera.h"
#include "feature/serialize.h"


namespace YH3DR
{
    namespace Algorithm
    {
    
        class UImage
        {
            
            friend class boost::serialization::access;
            
            template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar& imageID;
                ar& imagePath;
                ar& imageData;
                
                ar& exifAvailable;
                ar& imageWidth;
                ar& imageHeight;
                ar& cameraMaker;
                ar& cameraModel;
                ar& serialNumber;
                
                ar& orientation;
                ar& xResolution;
                ar& yResolution;
                ar& resolutionUnit;
                
                ar& focalLength;
                ar& principalPointX;
                ar& principalPointY;
                
                ar& latitude;
                ar& longitude;
                ar& altitude;
                ar& altitudeRef;
                ar& relativeAltitude;
                ar& rollDegree;
                ar& pitchDegree;
                ar& yawDegree;
                ar& speedX;
                ar& speedY;
                ar& speedZ;
                ar& accuracyXY;
                ar& accuracyZ;
                ar& GPSDOP;
                
                
                ar& featureDescriptor;
                ar& featureDim;
                ar& featureCount;
            }
        
            private:

                int imageID;
                std::string imagePath;           //image full path
                cv::Mat imageData;
                
                
                bool exifAvailable = false;
                int imageWidth;
                int imageHeight;
                std::string cameraMaker;
                std::string cameraModel;
                std::string serialNumber;    
                
                uint16_t orientation; 
                double xResolution;
                double yResolution;
                uint16_t resolutionUnit;
                
                double focalLength;
                double principalPointX;            //pixels
                double principalPointY;            //pixels
                
                //GPS information
                double latitude;
                double longitude;               
		        double altitude;               
		        int8_t altitudeRef;         
		        double relativeAltitude;       
		        double rollDegree;            
		        double pitchDegree;         
		        double yawDegree;       
		        double speedX;  
		        double speedY;    
		        double speedZ;   
		        double accuracyXY;       
		        double accuracyZ;       
		        double GPSDOP;                     //degree of precision
                
                
                
                Eigen::MatrixXi featureDescriptor; //integer type
                int featureDim = 128;              //default feature vector dimension
                int featureCount = 8192;           //default feature number
               
                /*YH3DR::Algorithm::UCamera camera;*/
                
            public:
                UImage();
                UImage(int inImageID);
                UImage(std::string inImagePath);
                ~UImage();
                
                void SetImageID(int inImageID);
                int GetImageID();
                void SetImagePath(std::string inImagePath);
                std::string GetImagePath();
                
                Eigen::MatrixXi GetFeatureDescriptor();
                
                void SetFeatureDim(int inFeatureDim);
                int GetFeatureDim();
                void SetFeatureCount(int inFeatureCount);
                int GetFeatureCount();
                
                bool isExifAvailable();
                int GetExifImageWidth();
                int GetExifImageHeight();
                std::string GetExifCameraMaker();
                std::string GetExifCameraModel();
                std::string GetExifSerialNumber();
                uint16_t GetExifOrientation();
                double GetExifXResolution();
                double GetExifYResolution();
                uint16_t GetExifResolutionUnit();
                double GetExifFocalLength();
                double GetExifPrincipalPointX();
                double GetExifPrincipalPointY();
                double GetExifLatitude();
                double GetExifLongitude();               
		        double GetExifAltitude();               
		        int8_t GetExifAltitudeRef();         
		        double GetExifRelativeAltitude();       
		        double GetExifRollDegree();            
		        double GetExifPitchDegree();         
		        double GetExifYawDegree();       
		        double GetExifSpeedX();  
		        double GetExifSpeedY();    
		        double GetExifSpeedZ();   
		        double GetExifAccuracyXY();       
		        double GetExifAccuracyZ();       
		        double GetExifGPSDOP(); 
                
                
                //void SetCameraParameter(YH3DR::Algorithm::UCamera inCamera);
                //YH3DR::Algorithm::UCamera GetCameraParameter();
                
                void ReadImageFromPath();
                void ComputeImageFeatures();
                void ExportMatchingFeatures(cv::Mat secondImage);
                bool CheckValidEXIF();
                bool CheckValidImage();
                void GetImageExif();
                //void GetCameraIntrinsic(YH3DR::Algorithm::UCamera& camera);
        };
        
        class UPackedImage
        {
            friend class boost::serialization::access;
            
            template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar& packedImageData;
            }
            
            private:
            
                std::vector<YH3DR::Algorithm::UImage> packedImageData;
                
            public:
                //Ensure that there must be a contructor function with empty parameters.
                //If not gcc reports error: no matching function for call to ‘YH3DR::Algorithm::UPackedImage::UPackedImage()
                //for boost::mpi::gather()
                UPackedImage()
                {
                    
                }
            
                UPackedImage(std::vector<YH3DR::Algorithm::UImage>& inImages)
                {
                    for(auto iter = inImages.begin(); iter != inImages.end(); iter++)
                    {
                        packedImageData.push_back(*iter);
                    }
                }
                
                ~UPackedImage()
                {
                    
                }
                
                void SetData(std::vector<YH3DR::Algorithm::UImage>& inImages)
                {
                    for(auto iter = inImages.begin(); iter != inImages.end(); iter++)
                    {
                        packedImageData.push_back(*iter);
                    }  
                }
                
                
                
                std::vector<YH3DR::Algorithm::UImage> Unpack()
                {
                    return packedImageData;
                }
                
        };
        
    }
}
#endif
