comp=clang++
cflags= -std=c++11 -stdlib=libc++
lflags= -stdlib=libc++

all: examples
	./examples

examples: examples.o
	$(comp) $(lflags) examples.o -o examples

examples.o: examples.cpp
	$(comp) $(cflags) -c examples.cpp
