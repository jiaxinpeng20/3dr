#include<iostream>
#include<fstream>
#include <>




struct vertexData
{
    double x;
    double y;
    double z;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
}

struct triangleData
{
    int p0;
    int p1;
    int p2;
}

struct cameraData
{
    Eigen::MatrixXd R;
    Eigen::VectorXd t;
}


void compressRodriguesRotation(double r[3], const double R[9])
{
        double a = (R[0]+R[4]+R[8]-1.0)/2.0;
        const double epsilon = 0.01;
        if( fabs(R[1] - R[3]) < epsilon &&
            fabs(R[5] - R[7]) < epsilon && 
            fabs(R[2] - R[6]) < epsilon )
        {
            if( fabs(R[1] + R[3]) < 0.1 &&
                fabs(R[5] + R[7]) < 0.1 && 
                fabs(R[2] + R[6]) < 0.1 && a > 0.9)
            {
                r[0]    =    0;
                r[1]    =    0;
                r[2]    =    0;
            }
            else
            {
                const double ha = sqrt(0.5) * 3.14159265358979323846; 
                double xx = (R[0]+1.0)/2.0;
                double yy = (R[4]+1.0)/2.0;
                double zz = (R[8]+1.0)/2.0;
                double xy = (R[1]+R[3])/4.0;
                double xz = (R[2]+R[6])/4.0;
                double yz = (R[5]+R[7])/4.0;

                if ((xx > yy) && (xx > zz)) 
                { 
                    if (xx< epsilon) 
                    {
                        r[0] = 0;    r[1] = r[2] = ha; 
                    } else 
                    {
                        double t = sqrt(xx) ;
                        r[0] = double(t * 3.14159265358979323846);
                        r[1] = double(xy/t * 3.14159265358979323846);
                        r[2] = double(xz/t * 3.14159265358979323846);
                    }
                } else if (yy > zz) 
                { 
                    if (yy< epsilon)
                    {
                        r[0] = r[2]  = ha; r[1] = 0;
                    } else
                    {
                        double t = sqrt(yy);
                        r[0] = double(xy/t* 3.14159265358979323846);
                        r[1] = double( t * 3.14159265358979323846);
                        r[2] = double(yz/t* 3.14159265358979323846);
                    }    
                } else 
                {
                    if (zz< epsilon) 
                    {
                        r[0] = r[1] = ha; r[2] = 0;
                    } else
                    {
                        double t  = sqrt(zz);
                        r[0]  = double(xz/ t* 3.14159265358979323846);
                        r[1]  = double(yz/ t* 3.14159265358979323846);
                        r[2]  = double( t * 3.14159265358979323846);
                    }
                }
            }
        }
        else
        {
            a = acos(a);
            double b = 0.5*a/sin(a);
            r[0]    =    double(b*(R[7]-R[5]));
            r[1]    =    double(b*(R[2]-R[6]));
            r[2]    =    double(b*(R[3]-R[1]));
        }
}

void UncompressRodriguesRotation(const double d[3], double DR[9])
{
        double a = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        double ct = a==0.0?0.5f:(1.0f-cos(a))/a/a;
        double st = a==0.0?1:sin(a)/a;
        DR[0]=double(1.0 - (d[1]*d[1] + d[2]*d[2])*ct);
        DR[1]=double(d[0]*d[1]*ct - d[2]*st);
        DR[2]=double(d[2]*d[0]*ct + d[1]*st);
        DR[3]=double(d[0]*d[1]*ct + d[2]*st);
        DR[4]=double(1.0f - (d[2]*d[2] + d[0]*d[0])*ct);
        DR[5]=double(d[1]*d[2]*ct - d[0]*st);
        DR[6]=double(d[2]*d[0]*ct - d[1]*st);
        DR[7]=double(d[1]*d[2]*ct + d[0]*st);
        DR[8]=double(1.0 - (d[0]*d[0] + d[1]*d[1])*ct );
}


int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout<<"Please enter bundler out file for translation."<<std::endl;
    }
    std::ifstream in(argv[1]);

    int nCamera, nStructure, nProjection;
    //Read bundler out file
    if(!(in>>nCamera>>nStructure>>nProjection))
    {
	    return 0; 
    }


    //Camera Model
    std::vector<std::tuple<double, double, double>> cameraModel;
    cameraModel.push_back(std::make_tuple(0.0, 0.0, 0.0));
    cameraModel.push_back(std::make_tuple(2.0, 1.0, 1.0));
    cameraModel.push_back(std::make_tuple(2.0, -1.0, 1.0));
    cameraModel.push_back(std::make_tuple(2.0, 1.0, -1.0));
    cameraModel.push_back(std::make_tuple(2.0, -1.0, -1.0));



    //Define vertex array and face array
    std::vector<vertexData> vertexArray;
    std::vector<triangleData> triangleArray;
    
    //Skip observation data
    for(unsigned int i = 0; i < nProjection; i++)
	{
		int camidx, optidx;
		double x, y;

		if(!(in>>camidx>>optidx>>x>>y))
			return 0;
    }
    
    
    //Read camera data
    for(unsigned int i = 0; i < nCamera; i++)
    {
        double p[9];
		for(int j = 0; j < 9; j++) in>>p[j];
        //of<<p[3]<<"  "<<p[4]<<"  "<<p[5]<<std::endl;
        
        
        
        
    }
    
    
    
    //Draw cameras for bundle adjustment dataset
    

    std::ofstream of("trajectory.ply");
	of<<"ply"<<std::endl;
	of<<"format ascii 1.0"<<std::endl;
	of<<"comment author: Dr. Jiaxin Peng"<<std::endl;
	of<<"element vertex "<<nStructure<<std::endl;
	of<<"property double x"<<std::endl;
	of<<"property double y"<<std::endl;
	of<<"property double z"<<std::endl;
	of<<"property uchar red"<<std::endl;
	of<<"property uchar green"<<std::endl;
	of<<"property uchar blue"<<std::endl;
	of<<"element face "<<nStructure<<std::endl;
	of<<"end_header"<<std::endl;


    

    for(unsigned int i = 0; i < nStructure; i++)
    {
        double p0, p1, p2;
		in>>p0>>p1>>p2;
        of<<p0<<"  "<<p1<<"  "<<p2<<std::endl;
    }

    in.close();
    of.close();
}
