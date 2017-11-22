#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"

class Utils{
  private:
    cv::Mat imageRGBA;
    cv::Mat imageGray;
    cv::Mat imageBlurred;
    cv::Mat imageGrad_x;
    cv::Mat imageGrad_y;
    cv::Mat imageGrad_val;
    cv::Mat imageGrad_thin;
    cv::Mat imageGrad_thresh;
    cv::Mat imageGrad_hyster;
    cv::Mat imageGrad_dir;
    cv::Mat image;

  public: 
    Utils(){};
    ~Utils(){};

    void loadImage(const std::string &filename);
    void memAlloc(uchar4 ** h_rgbaImage, unsigned char ** h_grayImage,
		  uchar4 ** d_rgbaImage, unsigned char ** d_grayImage,
           unsigned char ** h_blurredImage, unsigned char ** d_blurredImage,
                    char ** h_gradImage_x,           char ** d_gradImage_x,
                    char ** h_gradImage_y,           char ** d_gradImage_y,
           unsigned char ** h_gradImage_val, unsigned char ** d_gradImage_val,
           unsigned char ** h_gradImage_thin, unsigned char ** d_gradImage_thin,
           unsigned char ** h_gradImage_thresh, unsigned char ** d_gradImage_thresh,
           unsigned char ** h_gradImage_hyster, unsigned char ** d_gradImage_hyster,
                   float ** h_gradImage_dir, float ** d_gradImage_dir);

    void displayImage(cv::Mat imageMat);
    size_t getNumPixels(cv::Mat imageMat);

    cv::Mat getImage(){return image;};
    cv::Mat getImageRGBA(){return imageRGBA;};
    cv::Mat getImageGray(){return imageGray;};
    cv::Mat getImageBlurred(){return imageBlurred;};
    cv::Mat getImageGrad_x(){return imageGrad_x;};
    cv::Mat getImageGrad_y(){return imageGrad_y;};
    cv::Mat getImageGrad_val(){return imageGrad_val;};
    cv::Mat getImageGrad_thin(){return imageGrad_thin;};
    cv::Mat getImageGrad_thresh(){return imageGrad_thresh;};
    cv::Mat getImageGrad_hyster(){return imageGrad_hyster;};
    cv::Mat getImageGrad_dir(){return imageGrad_dir;};

    void createGaussianFilter(float **h_filter, float **d_filter, int *filterWidth);
    void createGradientFilter_x(float **h_filter, float **d_filter, int *filterWidth);
    void createGradientFilter_y(float **h_filter, float **d_filter, int *filterWidth);
};

