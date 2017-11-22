#include <iostream>
#include "Utils.h"

// load image given filename
void Utils::loadImage(const std::string &filename){

  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't load file: " << filename << std::endl;
    exit(1);
  }
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
  imageGray.create(image.rows, image.cols, CV_8UC1); 
  imageBlurred.create(image.rows, image.cols, CV_8UC1); 
  imageGrad_x.create(image.rows, image.cols, CV_8SC1); 
  imageGrad_y.create(image.rows, image.cols, CV_8SC1); 
  imageGrad_val.create(image.rows, image.cols, CV_8UC1); 
  imageGrad_thin.create(image.rows, image.cols, CV_8UC1); 
  imageGrad_thresh.create(image.rows, image.cols, CV_8UC1); 
  imageGrad_hyster.create(image.rows, image.cols, CV_8UC1); 
  imageGrad_dir.create(image.rows, image.cols, CV_8UC1); 
}

void Utils::memAlloc(uchar4 ** h_rgbaImage, unsigned char ** h_grayImage,
		     uchar4 ** d_rgbaImage, unsigned char ** d_grayImage,
           unsigned char ** h_blurredImage, unsigned char ** d_blurredImage,
                    char ** h_gradImage_x,           char ** d_gradImage_x,
                    char ** h_gradImage_y,           char ** d_gradImage_y,
           unsigned char ** h_gradImage_val, unsigned char ** d_gradImage_val,
           unsigned char ** h_gradImage_thin, unsigned char ** d_gradImage_thin,
           unsigned char ** h_gradImage_thresh, unsigned char ** d_gradImage_thresh,
           unsigned char ** h_gradImage_hyster, unsigned char ** d_gradImage_hyster,
                   float ** h_gradImage_dir, float ** d_gradImage_dir){

  *h_rgbaImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0); 
  *h_grayImage = imageGray.ptr<unsigned char>(0);
  *h_blurredImage = imageBlurred.ptr<unsigned char>(0);
  *h_gradImage_x = imageGrad_x.ptr<char>(0);
  *h_gradImage_y = imageGrad_y.ptr<char>(0);
  *h_gradImage_val = imageGrad_val.ptr<unsigned char>(0);
  *h_gradImage_thin = imageGrad_thin.ptr<unsigned char>(0);
  *h_gradImage_thresh = imageGrad_thresh.ptr<unsigned char>(0);
  *h_gradImage_hyster = imageGrad_hyster.ptr<unsigned char>(0);
  *h_gradImage_dir = imageGrad_dir.ptr<float>(0);

  // allocate memory on the device
  cudaMalloc(d_rgbaImage, sizeof(uchar4) * getNumPixels(image));
  cudaMalloc(d_grayImage, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_blurredImage, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_gradImage_x, sizeof(char) * getNumPixels(image));
  cudaMalloc(d_gradImage_y, sizeof(char) * getNumPixels(image));
  cudaMalloc(d_gradImage_val, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_gradImage_thin, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_gradImage_thresh, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_gradImage_hyster, sizeof(unsigned char) * getNumPixels(image));
  cudaMalloc(d_gradImage_dir, sizeof(float) * getNumPixels(image));

  cudaMemset(*d_gradImage_thin, 0, sizeof(unsigned char) * getNumPixels(image));
  cudaMemset(*d_gradImage_thresh, 0, sizeof(unsigned char) * getNumPixels(image));
  cudaMemset(*d_gradImage_hyster, 0, sizeof(unsigned char) * getNumPixels(image));

  // copy data from host to device
  GpuTimer timer;
  timer.Start();
  cudaMemcpy(*d_rgbaImage, *h_rgbaImage, sizeof(uchar4) * getNumPixels(image),cudaMemcpyHostToDevice);
  timer.Stop();
  printf("Copy Image from host to device elpased timer is: %f msecs. \n",timer.Elapsed());
}

size_t Utils::getNumPixels(cv::Mat imageMat){
  return imageMat.rows * imageMat.cols; 
}

void Utils::displayImage(cv::Mat imageMat){
  
  cv::namedWindow("DisplayImage", CV_WINDOW_AUTOSIZE );
  cv::imshow("DisplayImage", imageMat);

}


void Utils::createGaussianFilter(float **h_filter, float **d_filter, int * filterWidth) {
  
  // Create the filter
  const int blurKernelWidth = 3;
  const float blurKernelSigma = 0.2;

  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];

  float filterSum = 0.f; 

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      int temp = (r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2;
      (*h_filter)[temp] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      int temp = (r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2;
      (*h_filter)[temp] *= normalizationFactor;
    }
  }

  cudaMalloc((void **) d_filter, sizeof(float) * blurKernelWidth * blurKernelWidth);

  // copy data from host to device
  cudaMemcpy(*d_filter, *h_filter, sizeof(float) * blurKernelWidth * blurKernelWidth  ,cudaMemcpyHostToDevice);

}



void Utils::createGradientFilter_x(float **h_filter, float **d_filter, int * filterWidth) {
  
  // Create the filter
  const int gradKernelWidth = 3;

  *filterWidth = gradKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[gradKernelWidth * gradKernelWidth];

  // define x direction gradient filter
  (*h_filter)[0] = -1;
  (*h_filter)[1] =  0;
  (*h_filter)[2] =  1;
  (*h_filter)[3] = -2;
  (*h_filter)[4] =  0;
  (*h_filter)[5] =  2;
  (*h_filter)[6] = -1;
  (*h_filter)[7] =  0;
  (*h_filter)[8] =  1;

  cudaMalloc((void **) d_filter, sizeof(float) * gradKernelWidth * gradKernelWidth);

  // copy data from host to device
  cudaMemcpy(*d_filter, *h_filter, sizeof(float) * gradKernelWidth * gradKernelWidth  ,cudaMemcpyHostToDevice);
}



void Utils::createGradientFilter_y(float **h_filter, float **d_filter, int * filterWidth) {
  
  // Create the filter
  const int gradKernelWidth = 3;

  *filterWidth = gradKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[gradKernelWidth * gradKernelWidth];

  // define y direction gradient filter
  (*h_filter)[0] = -1;
  (*h_filter)[1] = -2;
  (*h_filter)[2] = -1;
  (*h_filter)[3] =  0;
  (*h_filter)[4] =  0;
  (*h_filter)[5] =  0;
  (*h_filter)[6] =  1;
  (*h_filter)[7] =  2;
  (*h_filter)[8] =  1;

  cudaMalloc((void **) d_filter, sizeof(float) * gradKernelWidth * gradKernelWidth);

  // copy data from host to device
  cudaMemcpy(*d_filter, *h_filter, sizeof(float) * gradKernelWidth * gradKernelWidth  ,cudaMemcpyHostToDevice);
}
