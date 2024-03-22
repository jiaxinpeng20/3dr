//Image class for data reading and fearure extraction


#include<iostream>
#include<thread>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<Eigen/Eigen>
#include <opencv2/core/eigen.hpp>



namespace XSFM
{
    namespace Algorithm
    {
        class UImage
        {
            private:
                std::vector<std::string> imageFileList;//Image files for feature extraction
                int numOfImages;
                int numOfThreads;
                

                cv::Mat imageData;

                std::vector<cv::Mat> imageDataList;//B, G, R color space whereas R, G, B color space for Matlab
                
                int numOfKeyPoints = 8192;//default number of key points
                std::vector<Eigen::MatrixXd> descriptors;//openCV descriptors for SIFT
                




                std::string exif;//camera information

                double CameraPoseX;
                double CameraPoseY;
                double CameraPoseZ;

                double CameraPoseAlpha;
                double CameraPoseBeta;
                double CameraPoseGamma;

                double k1;
                double k2;
                double f;
                double k0;


                
                
            public:
                void ReadImageFromPath();
                void ComputeImageFeatures();

                std::vector<Eigen::MatrixXd> GetImageDescriptors();
                void GetImageExif();
                UImage(std::vector<std::string>& inImageFileList, int inNumOfThreads);
                UImage(std::vector<std::string>& inImageFileList);
                ~UImage();
        };
    }
}
