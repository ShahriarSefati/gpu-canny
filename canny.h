#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class Canny{
  private:

  public: 
    Canny(){};
    ~Canny(){};

    void rgbaToGrayscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols);
    
    void gaussianBlur(unsigned char *d_grayImage, unsigned char *d_blurredImage,  size_t numRows, size_t numCols, const int filterWidth, const float *d_filter );
    
    void gaussianBlurShared(unsigned char *d_grayImage, unsigned char *d_blurredImage,  size_t numRows, size_t numCols, const int filterWidth, const float *d_filter );
    
    void grad_x(unsigned char *d_grayImage, char *d_gradImage_x,  size_t numRows, size_t numCols, const int filterWidth, const float *d_filter );
    
    void grad_y(unsigned char *d_grayImage, char *d_gradImage_y,  size_t numRows, size_t numCols, const int filterWidth, const float *d_filter );
    
    void grad_calc(char *d_gradImage_x, char *d_gradImage_y, unsigned char *d_gradImage_val, float *d_gradImage_dir, size_t numRows,  size_t numCols);

    void nonMax_suppress(unsigned char *d_gradImage_val, unsigned char *d_gradImage_thin, float *d_gradImage_dir, size_t numRows,  size_t numCols);
    
    void threshold(unsigned char *d_gradImage_thin, unsigned char *d_gradImage_thresh, size_t numRows,  size_t numCols);
    
    void hysteresis( unsigned char *d_gradImage_thin, unsigned char *d_gradImage_thresh, unsigned char *d_gradImage_hyster, size_t numRows,  size_t numCols);
};

