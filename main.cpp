#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_io.h"
#include "fp_detector.h"

using namespace std;
using namespace cv;
using namespace dlib;
using namespace cv::dnn;



// initialize 
string config_path = "./model/face_landmark.prototxt";
string model_path = "./model/face_landmark.caffemodel";

FPDetector fp_detector;

cv::TickMeter timer;


int main()
{
    string img = "./images/face1.jpg";
    Mat image = imread(img);
    frontal_face_detector detector = get_frontal_face_detector();
    array2d< bgr_pixel> arrimg(image.rows, image.cols);
    for(int i=0; i < image.rows; i++) {
      for(int j=0; j < image.cols; j++) {
        arrimg[i][j].blue = image.at< cv::Vec3b>(i,j)[0];
        arrimg[i][j].green=image.at< cv::Vec3b>(i,j)[1];
        arrimg[i][j].red = image.at< cv::Vec3b>(i,j)[2];
      }
    }
    
    std::vector<dlib::rectangle> dets = detector(arrimg);

    fp_detector.LoadModel(config_path, model_path);
    



    for(size_t i = 0;i < dets.size();i++) {
        dlib::rectangle tmp = dets[i];

        Rect face_rect = Rect(Point(tmp.left(), tmp.top()), Point(tmp.right(), tmp.bottom()));
        cv::rectangle(image, face_rect, Scalar(0,0,255), 1,4,0);
        Mat srcROI(image, face_rect);
        
        Mat img2;
        cvtColor(srcROI,img2,CV_RGB2GRAY);
        
        timer.reset();
        timer.start();

        fp_detector.FindFaceLandMark(img2);
        timer.stop();
        std::cout << " total time = " << timer.getTimeMilli() << std::endl;

        for(size_t i = 0;i < fp_detector.facial_points.size() ;i++) {
          Point2f x = Point2f(fp_detector.facial_points.at(i).x*face_rect.width,
                              fp_detector.facial_points.at(i).y*face_rect.height);
            cv::circle(image(face_rect), x, 0.1, Scalar(0, 0, 255), 4, 8, 0);
        }   


    }

    imshow("result", image);
    waitKey(0);
   //cout << feat_dim << endl;
    imwrite("./results/face1.jpg", image);
    //free(net);
    
    image.release();
    return 0;
    
}
