NVCC=nvcc
NVCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -m64
CC=g++
GCC_OPTS=-O3 -Wall -Wextra -m64

ifeq ($(shell sw_vers 2>/dev/null | grep Mac | awk '{ print $$2}'),Mac)
### default locations on mac os x
	OPENCV_LIBPATH=/usr/local/lib
	OPENCV_INCLUDEPATH=/usr/local/include
	CUDA_INCLUDEPATH=/usr/local/cuda/include
	CUDA_LIBPATH=/usr/local/cuda/lib

else
### default locations on most linux distros
	OPENCV_LIBPATH=/usr/lib
	OPENCV_INCLUDEPATH=/usr/include
	CUDA_INCLUDEPATH=/usr/cuda/include
	CUDA_LIBPATH=/usr/cuda/lib
endif

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

all: blur Makefile

blur: main.o gaussian_blur.o
	$(NVCC) -o blur main.o gaussian_blur.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp utils.h
	$(CC) -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

gaussian_blur.o: gaussian_blur.cu utils.h
	$(NVCC) -c gaussian_blur.cu $(NVCC_OPTS)

clean:
	rm -f *.o *.png blur
