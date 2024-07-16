#include<iostream>
#include<vector>
#include<fstream>
#include<GL/glut.h>

std::vector<GLfloat> vStructureList;
std::vector<GLfloat> vCameraList;

static GLfloat x_max, x_min, y_max, y_min, z_max, z_min;
static GLfloat lookAtx = 0.0f, lookAty = 0.0f, lookAtz = 10.0f;

void init()
{

/*GLfloat mat_specular [ ] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat mat_shininess [ ] = { 50.0 };
GLfloat light_position [ ] = { 1.0, 1.0, 1.0, 0.0 }; 

glClearColor( 0.0, 0.0, 0.0, 0.0 );
glShadeModel( GL_SMOOTH );

glMaterialfv( GL_FRONT, GL_SPECULAR, mat_specular);
glMaterialfv( GL_FRONT, GL_SHININESS, mat_shininess);
glLightfv( GL_LIGHT0, GL_POSITION, light_position);

glEnable(GL_LIGHTING);
glEnable(GL_LIGHT0);
glEnable(GL_DEPTH_TEST);*/
}

bool readBundlerFile(std::string filePath)
{

    x_min = y_min = z_min = +1.0e+10;
    x_max = y_max = z_max = -1.0e+10;

    unsigned int nProjection, nStructure, nCamera;
    std::ifstream ifs(filePath);
    if(!(ifs>>nCamera>>nStructure>>nProjection))
    {
        return false;
    }
    vStructureList.reserve(nStructure*3);
    vCameraList.reserve(nCamera*6);

    double x, y, z;int k;
    //scanning projections
    for(int i = 0; i < nProjection; i++)
    {
        ifs>>k>>k>>x>>x;
    }
    //scanning structure parameters  
    for(int i = 0; i < nStructure; i++)
    {
        ifs>>x>>y>>z;
        vStructureList.push_back((GLfloat)x);
        vStructureList.push_back((GLfloat)y);
        vStructureList.push_back((GLfloat)z);
        if(z < z_min)
        {
            z_min = z;
        }
        if(z > z_max)
        {
            z_max = z;
        }
    }
    std::cout<<z_min<<"  "<<z_max<<std::endl;
    //scanning camera parameters
    for(int i = 0; i < nCamera; i++)
    {
        ifs>>x>>y>>z;
        vCameraList.push_back((GLfloat)x);
        vCameraList.push_back((GLfloat)y);
        vCameraList.push_back((GLfloat)z);
        ifs>>x>>y>>z;
        vCameraList.push_back((GLfloat)x);
        vCameraList.push_back((GLfloat)y);
        vCameraList.push_back((GLfloat)z);
        ifs>>x>>y>>z;
    }
    
    ifs.close();
}


void display()
{

    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

    //Draw object points
    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(2.0f);
    glBegin(GL_POINTS);
        for(int i = 0; i < vStructureList.size(); i+=3)
        {
            glVertex3f(vStructureList.at(i), vStructureList.at(i+1), vStructureList.at(i+2));
        }
    glEnd();
    //Draw camera
    glColor3f(0.0f, 0.0f, 1.0f);
    glPointSize(2.0f);
    glBegin(GL_POINTS);
        for(int i = 0; i < vCameraList.size(); i+=6)
        {
            glVertex3f(vCameraList.at(i+3), vCameraList.at(i+4), vCameraList.at(i+5));
        }
    glEnd();
    std::cout<<lookAtx<<"  "<<lookAty<<"  "<<lookAtz<<std::endl;
    glLoadIdentity();
    gluLookAt (lookAtx, lookAty, lookAtz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glFlush();
}


void reshape(int w, int h)
{

   glViewport(0, 0, (GLsizei) w, (GLsizei) h); 
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(170.0, (GLfloat)w/(GLfloat)h, 1.0, 20.0);
}


void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 'w':
        case 'W':
            lookAtz = lookAtz - 1.0f;
            glutPostRedisplay();
            break;
        case 's':
        case 'S':
            lookAtz = lookAtz + 1.0f;
            glutPostRedisplay();
            break;
        default:
            break;
        
    }

}

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH); 
    glutInitWindowSize(1280, 800); 
    glutInitWindowPosition(100, 100); 
    glutCreateWindow("Point Cloud Viewer"); 
    init(); 

    if(argc >= 2)
    {
        readBundlerFile(argv[1]);
    }
    else
    {
        std::cout<<"ERROR: please enter a valid bundler file."<<std::endl;
    }
    glutDisplayFunc(display); 
    glutReshapeFunc(reshape); 
    glutKeyboardFunc(keyboard); 
    glutMainLoop(); 

	return 0;
}
