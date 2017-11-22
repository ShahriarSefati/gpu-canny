#include <iostream>
#include "canny.h"

__global__ void toGrayscaleKer(const uchar4* const rgbaImage,
                       unsigned char* const grayImage,
                       int numRows, int numCols)
{
  int y = threadIdx.y+ blockIdx.y* blockDim.y;
  int x = threadIdx.x+ blockIdx.x* blockDim.x;
  if (y < numCols && x < numRows) {
  	int index = numRows*y +x;
  uchar4 color = rgbaImage[index];
  unsigned char gray = (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
  grayImage[index] = gray;
  }
}

// rgba to gray scale image
////////////////////////////////////////////////////////////////////////

void Canny::rgbaToGrayscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_grayImage, size_t numRows, size_t numCols)
{
  
  int   blockWidth = 32;
  
  const dim3 blockSize(blockWidth, blockWidth, 1);
  int   blocksX = numRows/blockWidth+1;
  int   blocksY = numCols/blockWidth+1;
  const dim3 gridSize( blocksX, blocksY, 1);
  toGrayscaleKer<<<gridSize, blockSize>>>(d_rgbaImage, d_grayImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
}

////////////////////////////////////////////////////////////////////////

__global__
void gaussianBlurKer(unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    
    float color = 0.0f;
    
    for(int f_y = 0; f_y < filterWidth; f_y++) {
        for(int f_x = 0; f_x < filterWidth; f_x++) {
   
            int c_x = p.x + f_x - filterWidth/2;
            int c_y = p.y + f_y - filterWidth/2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y*filterWidth + f_x];
            color += filter_value*static_cast<float>(inputChannel[c_y*numCols + c_x]);
            
        }
    }
    
    outputChannel[m] = color;
}

// gaussian blur kernel using shared memory for optimized performance
///////////////////////////////////////////////////////////////////////

__global__
void gaussianBlurKerShared(unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    // radius of filter
    int R = 1;

    // shared memory
    __shared__ unsigned char subMatrix[1156];

    int x = blockIdx.x * blockDim.x + threadIdx.x - R;
    int y = blockIdx.y * blockDim.y + threadIdx.y - R;
    
    x = max(x, 0);
    x = min(x, numCols -1);
    y = max(y, 0);
    y = min(y, numRows -1);
    
    unsigned int index = y * numCols + x;
    unsigned int blockIndex = (R+threadIdx.y)*(blockDim.y + 2*R) + R + threadIdx.x;
    subMatrix[blockIndex] = inputChannel[index];
    
    // depending on the thread location, set outside boundry values to zero   
    if (threadIdx.x == 0 ){
        for (int i = 1 ; i <= R ; i++){
          subMatrix[blockIndex - i] = (index % numCols == 0 ? 0 : inputChannel[index -i]);
        }
    }
      
    if (threadIdx.x == blockDim.y -1 ){
      for (int i = 1 ; i <= R ; i++){
        subMatrix[blockIndex + i] = (index % numCols == numCols - 1 ? 0 : inputChannel[index + i]);
      }
    }
    
    if (threadIdx.y == 0 ){
      for (int i = 1 ; i <= R ; i++){
        subMatrix[(R+threadIdx.y - i)*(blockDim.y + 2*R) + R + threadIdx.x] = (index < numCols ? 0 : inputChannel[ (y - i) * numCols + x]) ;
      }
    }

    if (threadIdx.y == blockDim.x - 1 ){
      for (int i = 1 ; i <= R ; i++){
        subMatrix[(R+threadIdx.y + i)*(blockDim.y + 2*R) + R + threadIdx.x] = (index >= (numRows - 1 ) * numCols ? 0 : inputChannel[ (y + i) * numCols + x]);
      }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 ){
      for (int i = 1 ; i <= R ; i++){
        for (int j = 1 ; j <= R ; j ++){
          subMatrix[(R+threadIdx.y - j)*(blockDim.y + 2*R) + R + threadIdx.x - i] = (index == 0 ? 0 :  inputChannel[ (y - j) * numCols + x - i]);
        }
      }
    }
    
    if (threadIdx.x == 0 && threadIdx.y == blockDim.x - 1 ){
      for (int i = 1 ; i <= R ; i++){
        for (int j = 1 ; j <= R ; j ++){
          subMatrix[(R+threadIdx.y + j)*(blockDim.y + 2*R) + R + threadIdx.x - i] = (index == (numRows - 1 )* numCols ? 0 :  inputChannel[ (y + j) * numCols + x - i]);
        }
      }
    }

    if (threadIdx.x == blockDim.y - 1 && threadIdx.y == 0 ){
      for (int i = 1 ; i <= R ; i++){
        for (int j = 1 ; j <= R ; j ++){
          subMatrix[(R+threadIdx.y - j)*(blockDim.y + 2*R) + R + threadIdx.x + i] = (index == numCols - 1 ? 0 :inputChannel[ (y - j) * numCols + x + i]);
        }
      }
    }
    
    if (threadIdx.x == blockDim.y - 1 && threadIdx.y == blockDim.x - 1 ){
      for (int i = 1 ; i <= R ; i++){
        for (int j = 1 ; j <= R ; j ++){
          subMatrix[(R+threadIdx.y + j)*(blockDim.y + 2*R) + R + threadIdx.x + i] = (index == numRows * numCols - 1 ? 0 :inputChannel[ (y + j) * numCols + x + i]);
        }
      }
    }

    __syncthreads();

      // running the average blur filter
      float sum = 0;
      for (int dy = -R; dy <= R ; dy++){
        for (int dx = -R; dx <= R; dx++){
          float i = static_cast<float>(subMatrix[blockIndex + (dy * blockDim.x) + dx]);
          sum += i;
        }
      } 
    outputChannel[index] = sum/ ( (2*R +1) * (2*R+1));
}


// gaussian blur function 
///////////////////////////////////////////////////////////////////////
void Canny::gaussianBlur(unsigned char *d_grayImage, unsigned char *d_blurredImage,
                  size_t numRows,  size_t numCols, const int filterWidth, const float *d_filter )
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  gaussianBlurKer<<<gridSize, blockSize>>>(
      d_grayImage,
      d_blurredImage,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize();
    
}

///////////////////////////////////////////////////////////////////////
void Canny::gaussianBlurShared(unsigned char *d_grayImage, unsigned char *d_blurredImage,
                  size_t numRows,  size_t numCols, const int filterWidth, const float *d_filter )
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  gaussianBlurKerShared<<<gridSize, blockSize>>>(
      d_grayImage,
      d_blurredImage,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize();
    
}
///////////////////////////////////////////////////////////////////////


__global__
void grad_x_ker(unsigned char* const inputChannel,
                   char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    
    float color = 0.0f;
    
    for(int f_y = 0; f_y < filterWidth; f_y++) {
        for(int f_x = 0; f_x < filterWidth; f_x++) {
   
            int c_x = p.x + f_x - filterWidth/2;
            int c_y = p.y + f_y - filterWidth/2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y*filterWidth + f_x];
            color += filter_value*static_cast<float>(inputChannel[c_y*numCols + c_x]);
            
        }
    }
    
    outputChannel[m] = color;
}
//////////////////////////////////////////////////////////////////////

void Canny::grad_x(unsigned char *d_grayImage, char *d_gradImage_x,
                  size_t numRows,  size_t numCols, const int filterWidth, const float *d_filter )
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  grad_x_ker<<<gridSize, blockSize>>>(
      d_grayImage,
      d_gradImage_x,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); 
    
}


///////////////////////////////////////////////////////////////////////


__global__
void grad_y_ker(unsigned char* const inputChannel,
                         char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    
    float color = 0.0f;
    
    for(int f_y = 0; f_y < filterWidth; f_y++) {
        for(int f_x = 0; f_x < filterWidth; f_x++) {
   
            int c_x = p.x + f_x - filterWidth/2;
            int c_y = p.y + f_y - filterWidth/2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y*filterWidth + f_x];
            color += filter_value*static_cast<float>(inputChannel[c_y*numCols + c_x]);

            
        }
    }
    
    outputChannel[m] = color;
}
//////////////////////////////////////////////////////////////////////

void Canny::grad_y(unsigned char *d_grayImage, char *d_gradImage_y,
                  size_t numRows,  size_t numCols, const int filterWidth, const float *d_filter )
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  grad_y_ker<<<gridSize, blockSize>>>(
      d_grayImage,
      d_gradImage_y,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); 
    
}

///////////////////////////////////////////////////////////////////

__global__
void grad_calc_ker(char* const inputGrad_x,
                char* const inputGrad_y,
                unsigned char* const outputGrad_val,
                float* const outputGrad_dir,
                int numRows, int numCols)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    // calculate gradient values at each pixel
    outputGrad_val[m] = sqrt( static_cast<float>(inputGrad_x[m]) *  static_cast<float>(inputGrad_x[m]) + static_cast<float>(inputGrad_y[m]) *  static_cast<float>(inputGrad_y[m]));
    
    // caculate gradient directions at each pixel
    outputGrad_dir[m] = atan((static_cast<float>(inputGrad_y[m])) / (static_cast<float>(inputGrad_x[m])));
   

      

}

//////////////////////////////////////////////////////////////////////

void Canny::grad_calc(char *d_gradImage_x, char *d_gradImage_y, unsigned char *d_gradImage_val, float *d_gradImage_dir, size_t numRows,  size_t numCols)
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  grad_calc_ker<<<gridSize, blockSize>>>(
      d_gradImage_x,
      d_gradImage_y,
      d_gradImage_val,
      d_gradImage_dir,
      numRows,
      numCols);
  cudaDeviceSynchronize();   
}


///////////////////////////////////////////////////////////////////

__global__
void nonMax_ker(unsigned char* const outputGrad_val,
                unsigned char* const outputGrad_thin,
                float* const outputGrad_dir,
                int numRows, int numCols)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    

    // non-max suppression
    int east = m + 1;
    int west = m - 1;
    int north = (p.y - 1) * numCols + p.x;
    int south = (p.y + 1) * numCols + p.x;
    int ne = (p.y - 1) * numCols + p.x + 1;
    int nw = (p.y - 1) * numCols + p.x - 1;
    int se = (p.y + 1) * numCols + p.x + 1;
    int sw = (p.y + 1) * numCols + p.x - 1;
    
    // rounding direction values to 0, 45, 90 ,135
    if (p.x > 0 && p.x < numCols - 1 && p.y > 0 && p.y < numRows - 1){
      if (-0.4 < outputGrad_dir[m] < 0.4){
        if (outputGrad_val[m] > outputGrad_val[east] || outputGrad_val[m] > outputGrad_val[west]){
          outputGrad_thin[m] = outputGrad_val[m];
        }
      } else if ( 0.4 < outputGrad_dir[m] < 1.35) {
        if (outputGrad_val[m] > outputGrad_val[ne] || outputGrad_val[m] > outputGrad_val[sw]){
          outputGrad_thin[m] = outputGrad_val[m];
        }
      } else if ( 1.35 < outputGrad_dir[m] || outputGrad_dir[m] < -1.35) {
        if (outputGrad_val[m] > outputGrad_val[north] || outputGrad_val[m] > outputGrad_val[south]){
          outputGrad_thin[m] = outputGrad_val[m];
        } 
      } else if ( -1.35 < outputGrad_dir[m] < -0.4) {
        if (outputGrad_val[m] > outputGrad_val[nw] || outputGrad_val[m] > outputGrad_val[se]){
          outputGrad_thin[m] = outputGrad_val[m];
        }
        }
    }   
}

//////////////////////////////////////////////////////////////////////

void Canny::nonMax_suppress(unsigned char *d_gradImage_val, unsigned char *d_gradImage_thin, float *d_gradImage_dir, size_t numRows,  size_t numCols)
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  nonMax_ker<<<gridSize, blockSize>>>(
      d_gradImage_val,
      d_gradImage_thin,
      d_gradImage_dir,
      numRows,
      numCols);
  cudaDeviceSynchronize();
    
}

////////////////////////////////////////////////////////////////////////
__global__
void threshold_ker(unsigned char* const outputGrad_thin,
                   unsigned char* const outputGrad_thresh,
                   int numRows, int numCols)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;
    
      if( outputGrad_thin[m] >= 50 ){
        outputGrad_thresh[m] = outputGrad_thin[m];
      }
      
}

//////////////////////////////////////////////////////////////////////

void Canny::threshold(unsigned char *d_gradImage_thin, unsigned char *d_gradImage_thresh, size_t numRows,  size_t numCols)
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  threshold_ker<<<gridSize, blockSize>>>(
      d_gradImage_thin,
      d_gradImage_thresh,
      numRows,
      numCols);
  cudaDeviceSynchronize();
    
}

////////////////////////////////////////////////////////////////////////
__global__
void hysteresis_ker(unsigned char* const outputGrad_thin,
                    unsigned char* const outputGrad_thresh,
                    unsigned char* const outputGrad_hyster,
                    int numRows, int numCols)
{

    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if(p.x >= numCols || p.y >= numRows)
         return;

    // define neighbor locations
    int east = m + 1;
    int west = m - 1;
    int north = (p.y - 1) * numCols + p.x;
    int south = (p.y + 1) * numCols + p.x;
    int ne = (p.y - 1) * numCols + p.x + 1;
    int nw = (p.y - 1) * numCols + p.x - 1;
    int se = (p.y + 1) * numCols + p.x + 1;
    int sw = (p.y + 1) * numCols + p.x - 1;
    
    // keep the high thresholds    
    outputGrad_hyster[m] = outputGrad_thresh[m];

    // keep the pixels with gradient between thresholds that are neighbor to high thresholds
    if (p.x > 0 && p.x < numCols - 1 && p.y > 0 && p.y < numRows - 1){
      if( 10 < outputGrad_thin[m] < 50 ){
        if(outputGrad_thresh[east] != 0 || outputGrad_thresh[west] != 0 ||
           outputGrad_thresh[north] != 0 || outputGrad_thresh[south] != 0 ||
           outputGrad_thresh[ne] != 0 ||outputGrad_thresh[nw] != 0 ||
           outputGrad_thresh[se] != 0 ||outputGrad_thresh[sw] != 0 ){
          outputGrad_hyster[m] = outputGrad_thin[m];
        }
      }
    }
}

//////////////////////////////////////////////////////////////////////

void Canny::hysteresis(unsigned char *d_gradImage_thin, unsigned char *d_gradImage_thresh, unsigned char *d_gradImage_hyster, size_t numRows,  size_t numCols)
{
  const dim3 blockSize(32, 32);

  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  hysteresis_ker<<<gridSize, blockSize>>>(
      d_gradImage_thin,
      d_gradImage_thresh,
      d_gradImage_hyster,
      numRows,
      numCols);
  cudaDeviceSynchronize(); 
    
}
