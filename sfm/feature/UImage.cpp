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

#include "UImage.h"

namespace YH3DR
{
    namespace Algorithm
    {
        UImage::UImage()
        {
            
        }
        
        UImage::UImage(int inImageID)
        {
            this->imageID = inImageID;
        }
        
        UImage::UImage(std::string inImagePath)
        {
            this->imagePath= inImagePath;
        }
        
        UImage::~UImage()
        {
            
        }
        
        
        
        void UImage::SetImageID(int inImageID)
        {
            this->imageID = inImageID;
        }
        
        int UImage::GetImageID()
        {
            return this->imageID;
        }

        void UImage::SetImagePath(std::string inImagePath)
        {
            this->imagePath = inImagePath;
        }
        
        std::string UImage::GetImagePath()
        {
            return this->imagePath;
        }
        
        
        
        
        Eigen::MatrixXi UImage::GetFeatureDescriptor()
        {
            return this->featureDescriptor;
        }
        
        
        void UImage::SetFeatureDim(int inFeatureDim)
        {
            this->featureDim = inFeatureDim;
        }
                
        int UImage::GetFeatureDim()
        {
            return this->featureDim;
        }
                
        void UImage::SetFeatureCount(int inFeatureCount)
        {
            this->featureCount = inFeatureCount;
        }
                
        int UImage::GetFeatureCount()
        {
            return this->featureCount;
        }
         
        bool  UImage::isExifAvailable()
        {
            return this->exifAvailable;
        }
        
        int UImage::GetExifImageWidth()
        {
            return this->imageWidth;
        }
        
        int UImage::GetExifImageHeight()
        {
            return this->imageHeight;
        }
        
        std::string UImage::GetExifCameraMaker()
        {
            return this->cameraMaker;
        }
                
        std::string UImage::GetExifCameraModel()
        {
            return this->cameraModel;
        }
                
        std::string UImage::GetExifSerialNumber()
        {
            return this->serialNumber;
        }
                
        uint16_t UImage::GetExifOrientation()
        {
            return this->orientation;
        }
                
        double UImage::GetExifXResolution()
        {
            return this->xResolution;
        }
                
        double UImage::GetExifYResolution()
        {
            return this->yResolution;
        }
                
        uint16_t UImage::GetExifResolutionUnit()
        {
            return this->resolutionUnit;
        }
                
        double UImage::GetExifFocalLength()
        {
            return this->focalLength;
        }
                
        double UImage::GetExifPrincipalPointX()
        {
            return this->principalPointX;
        }
                
        double UImage::GetExifPrincipalPointY()
        {
            return this->principalPointY;
        }
                
        double UImage::GetExifLatitude()
        {
            return this->latitude;
        }
                
        double UImage::GetExifLongitude()
        {
            return this->longitude;
        }           
		        
		double UImage::GetExifAltitude()
		{
		    return this->altitude;
		}              
		        
		int8_t UImage::GetExifAltitudeRef()
		{
		    return this->altitudeRef;
		}       
		        
		double UImage::GetExifRelativeAltitude()
		{
		    return this->relativeAltitude;
		}  
		        
		double UImage::GetExifRollDegree()
		{
		    return this->rollDegree;
		}            
		        
		double UImage::GetExifPitchDegree()
		{
		    return this->pitchDegree;
		}     
		       
		double UImage::GetExifYawDegree()
		{
		    return this->yawDegree;
		}    
		        
		double UImage::GetExifSpeedX()
		{
		    return this->speedX;
		} 
		        
		double UImage::GetExifSpeedY()
		{
		    return this->speedY;
		}  
		
		double UImage::GetExifSpeedZ()
		{
		    return this->speedZ;
		} 
		        
		double UImage::GetExifAccuracyXY()
		{
		    return this->accuracyXY;
		}    
		        
		double UImage::GetExifAccuracyZ()
		{
		    return this->accuracyZ;
		}     
		        
		double UImage::GetExifGPSDOP()
		{
		    return this->GPSDOP;
		}
        
        
                
        /*void UImage::SetCameraParameter(YH3DR::Algorithm::UCamera inCamera)
        {
            this->camera = inCamera;
        }
                
        YH3DR::Algorithm::UCamera UImage::GetCameraParameter()
        {
            return this->camera;
        }*/
        
         
        void UImage::ReadImageFromPath()
        {
            if(imagePath.empty())
            {
                std::cout<<"Invalid image path..."<<std::endl;
                return;
            }

            imageData = cv::imread(imagePath, cv::IMREAD_COLOR);
            
            if(imageData.empty())
            {
                std::cout<<"Failed to read image data..."<<std::endl;
            }
            
        }
        
       

        void UImage::ComputeImageFeatures()
        {
                
            cv::Mat imageGrayData;
            cv::cvtColor(imageData, imageGrayData, cv::COLOR_BGR2GRAY);

            cv::Ptr<cv::SIFT> sift = cv::SIFT::create(featureCount);
            std::vector<cv::KeyPoint> keyPoints;
            sift->detect(imageGrayData, keyPoints);
                
            cv::Mat logoImage;
            cv::drawKeypoints(imageGrayData, keyPoints, logoImage);

            //Compute descriptors for all features 
            cv::Mat _cv_image_descriptors;
            sift->compute(imageGrayData, keyPoints, _cv_image_descriptors);

            cv::cv2eigen(_cv_image_descriptors, this->featureDescriptor);
            this->featureDescriptor = this->featureDescriptor.transpose();

             std::cout<<this->featureDescriptor.rows()<<" x "<<featureDescriptor.cols()<<std::endl; 

        }
        
        void UImage::ExportMatchingFeatures(cv::Mat secondImage)
        {
            cv::Mat imageGrayData1, imageGrayData2;
            cv::cvtColor(imageData, imageGrayData1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(secondImage, imageGrayData2, cv::COLOR_BGR2GRAY);

            cv::Ptr<cv::SIFT> sift = cv::SIFT::create(featureCount);
            std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
            sift->detect(imageGrayData1, keyPoints1);
            sift->detect(imageGrayData2, keyPoints2);

            cv::Mat logoImage1, logoImage2;
            cv::drawKeypoints(imageGrayData1, keyPoints1, logoImage1);
            cv::drawKeypoints(imageGrayData2, keyPoints2, logoImage2);

            cv::Mat _cv_image_descriptors1, _cv_image_descriptors2;
            sift->compute(imageGrayData1, keyPoints1, _cv_image_descriptors1);
            sift->compute(imageGrayData2, keyPoints2, _cv_image_descriptors2);

            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
            std::vector<cv::DMatch> matches1, matches;
            matcher->match(_cv_image_descriptors1, _cv_image_descriptors2, matches);

            cv::Mat ImageMatches;
            cv::drawMatches(imageGrayData1, keyPoints1, imageGrayData2, keyPoints2, matches, ImageMatches);
            
            //Generate export file for matching features
            std::filesystem::path _path(this->imagePath);
            std::string exportFolder = _path.parent_path().string();
            std::string exportFile = _path.stem().string() + "_matching" + _path.extension().string();
            cv::imwrite(exportFolder + exportFile, ImageMatches);
        }


        void UImage::GetImageExif()
        {
            std::ifstream ifs(this->imagePath, std::ifstream::binary);
            if(ifs.is_open())
            {
                TinyEXIF::EXIFInfo exif(ifs);
                if(exif.Fields)
                {
                    this->imageWidth = exif.ImageWidth;
                    this->imageHeight = exif.ImageHeight;
                    
                    int cvImageWidth = imageData.rows;
                    int cvImageHeight = imageData.cols;
                    if(cvImageWidth > 0 && cvImageHeight > 0)
                    {
                        if(cvImageWidth != exif.ImageWidth || cvImageHeight != exif.ImageHeight)
                        {
                            this->imageWidth = cvImageWidth;
                            this->imageHeight = cvImageHeight;
                        }
                    }
                    
                    this->cameraMaker = exif.Make;
                    this->cameraModel = exif.Model;
                    this->serialNumber = exif.SerialNumber;
                    
                    this->orientation = exif.Orientation;
                    this->xResolution = exif.XResolution;
                    this->yResolution = exif.YResolution;
                    this->resolutionUnit = exif.ResolutionUnit;
                    
                    this->focalLength = exif.Calibration.FocalLength;
                    this->principalPointX = exif.Calibration.OpticalCenterX;
                    this->principalPointY = exif.Calibration.OpticalCenterY;
                    
                    this->latitude = exif.GeoLocation.Latitude;
                    this->longitude = exif.GeoLocation.Longitude;               
		            this->altitude = exif.GeoLocation.Altitude;               
		            this->altitudeRef = exif.GeoLocation.AltitudeRef;         
		            this->relativeAltitude = exif.GeoLocation.RelativeAltitude;       
		            this->rollDegree = exif.GeoLocation.RollDegree;            
		            this->pitchDegree = exif.GeoLocation.PitchDegree;         
		            this->yawDegree = exif.GeoLocation.YawDegree;       
		            this->speedX = exif.GeoLocation.SpeedX;  
		            this->speedY = exif.GeoLocation.SpeedY;    
		            this->speedZ = exif.GeoLocation.SpeedZ;   
		            this->accuracyXY = exif.GeoLocation.AccuracyXY;       
		            this->accuracyZ = exif.GeoLocation.AccuracyZ;       
		            this->GPSDOP = exif.GeoLocation.GPSDOP; 
		            
		            this->exifAvailable = true;
                }
                else
                {
                    std::cout<<"Invalid image exif information for image "<<imagePath<<std::endl;
                }
                
                ifs.close();
            }
            
        }
        
    }
}
