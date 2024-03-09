#include<iostream>
#include<opencv2/opencv.hpp>

#include<vector>

class CFeatureExtractor
{
    private:
    std::vector<cv::KeyPoints> keyPoints;
    int numFeatures = 500;
     
    
    public:
    CFeatureExtractor()
    {
        
    }
    
    ~CFeatureExtractor()
    {
        
    }
    
    void FeatureImport(onst ImageReaderOptions& reader_options, const std::string& import_path);
    bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options, const Bitmap& bitmap, FeatureKeypoints* keypoints, FeatureDescriptors* descriptors);
    bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options, const Bitmap& bitmap, FeatureKeypoints* keypoints, FeatureDescriptors* descriptors);
    
}

class FeatureExtractor
{
    private:
    std::vector<cv::KeyPoints> keyPoints;
    int numFeatures = 1000;
    
    public:
    FeatureExtractor()
    {
    
    }
    
    ~FeatureExtractor()
    {
    
    }
}
