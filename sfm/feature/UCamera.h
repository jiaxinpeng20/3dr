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


//Camera class for camera intrinsic and extrinsic parameters

#ifndef YH3DR_LIB3DR_UCAMERA_H
#define YH3DR_LIB3DR_UCAMERA_H
#include<iostream>

namespace YH3DR{

namespace Algorithm{
class UCamera
{
    private:
        //intrinsic parameters
        double focalLength;
        double k1;
        double k2;
        
        double principalPointX;
        double principalPointY;
        
        double sensorWidth;
        double sensorHeight;
        
        int imageWidth;
        int imageHeight;
        
        
        //camera poses
        double cameraPoseX;
        double cameraPoseY;
        double cameraPoseZ;
        double cameraPoseTheta;
        double cameraPosePhi;
        
        //int 
        
    public:
        UCamera();
        ~UCamera();
        
        bool CheckCameraParameters();
        
        void SetFocalLength(double inFocalLength);
        void SetCalibratedParameters(double inK1);
        void SetCalibratedParameters(double inK1, double inK2);
        void SetPrincipalPointX(double inPrincipalPointX);
        void SetPrincipalPointY(double inPrincipalPointY);
        void SetSensorWidth(double inSensorWidth);
        void SetSensorHeight(double inSensorHeight);
        void SetSensorSize(double inSensorWitdh, double inSensorHeight);
        void SetImageWidth(int inImageWidth);
        void SetImageHeight(int inImageHeight);
        void SetImageSize(int inImageWidth, int inImageHeight);
        
        void SetCameraPoseX(double inCameraPoseX);
        void SetCameraPoseY(double inCameraPoseY);
        void SetCameraPoseZ(double inCameraPoseZ);
        void SetCameraPosition(double inCameraPoseX, double inCameraPoseY, double inCameraPoseZ);
        void SetCameraPoseThera(double inCameraPoseTheta);
        void SetCameraPosePhi(double inCameraPosePhi);
        void SetCameraAngle(double inCameraPoseTheta, double inCameraPosePhi);
        
        
        double GetFocalLength();
        double GetCalibratedParameterK1();
        double GetCalibratedParameterK2();
        double GetPrincipalPointX();
        double GetPrincipalPointY();
        double GetSensorWidth();
        double GetSensorHeight();
        int GetImageWidth();
        int GetImageHeight();
        
        double GetCameraPoseX();
        double GetCameraPoseY();
        double GetCameraPoseZ();
        double GetCameraPoseThera();
        double GetCameraPosePhi();
};
}
}
#endif
