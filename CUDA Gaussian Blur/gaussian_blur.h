void gaussian_blur(const uchar4 *const d_inputImageRGBA,
                         uchar4 *const d_outputImageRGBA, 
                   const int           imageHeight,
                   const int           imageWidth,
                   const float  *const d_filter,
                   const int           filterWidth);
