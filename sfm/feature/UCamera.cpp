#include <iostream>
#include "UCamera.h"
namespace YH3DR{

namespace Algorithm{

    UCamera::UCamera()
    {
        
    }
    
    UCamera::~UCamera()
    {
        
    }
    
    bool UCamera::CheckCameraParameters()
    {
        
    }
    
    void UCamera::SetFocalLength(double inFocalLength)
    {
        this->focalLength = inFocalLength;
    }
    
    void UCamera::SetCalibratedParameters(double inK1)
    {
        this->k1 = inK1;
    }
    
    void UCamera::SetCalibratedParameters(double inK1, double inK2)
    {
        this->k1 = inK1;
        this->k2 = inK2;
    }
    
    void UCamera::SetPrincipalPointX(double inPrincipalPointX)
    {
        this->principalPointX = inPrincipalPointX;
    }
    
    void UCamera::SetPrincipalPointY(double inPrincipalPointY)
    {
        this->principalPointY = inPrincipalPointY;
    }
    
    void UCamera::SetSensorWidth(double inSensorWidth)
    {
        this->sensorWidth = inSensorWidth;
    }
    
    void UCamera::SetSensorHeight(double inSensorHeight)
    {
        this->sensorHeight = inSensorHeight;
    }
    
    void UCamera::SetSensorSize(double inSensorWidth, double inSensorHeight)
    {
        this->sensorWidth = inSensorWidth;
        this->sensorHeight = inSensorHeight;
    }
    
    void UCamera::SetImageWidth(int inImageWidth)
    {
        this->imageWidth = inImageWidth;
    }
    
    void UCamera::SetImageHeight(int inImageHeight)
    {
        this->imageHeight = inImageHeight;
    }
    
    void UCamera::SetImageSize(int inImageWidth, int inImageHeight)
    {
        this->imageWidth = inImageWidth;
        this->imageHeight = inImageHeight;
    }
    
    void UCamera::SetCameraPoseX(double inCameraPoseX)
    {
        this->cameraPoseX = inCameraPoseX;
    }
    
    void UCamera::SetCameraPoseY(double inCameraPoseY)
    {
        this->cameraPoseY = inCameraPoseY;
    }
    
    void UCamera::SetCameraPoseZ(double inCameraPoseZ)
    {
        this->cameraPoseZ = inCameraPoseZ;
    }
    
    void UCamera::SetCameraPosition(double inCameraPoseX, double inCameraPoseY, double inCameraPoseZ)
    {
        this->cameraPoseX = inCameraPoseX;
        this->cameraPoseY = inCameraPoseY;
        this->cameraPoseZ = inCameraPoseZ;
    }
    
    void UCamera::SetCameraPoseThera(double inCameraPoseTheta)
    {
        this->cameraPoseTheta = inCameraPoseTheta;
    }
    
    void UCamera::SetCameraPosePhi(double inCameraPosePhi)
    {
        this->cameraPosePhi = inCameraPosePhi;
    }
    
    void UCamera::SetCameraAngle(double inCameraPoseTheta, double inCameraPosePhi)
    {
        this->cameraPoseTheta = inCameraPoseTheta;
        this->cameraPosePhi = inCameraPosePhi;
    }

}
}
