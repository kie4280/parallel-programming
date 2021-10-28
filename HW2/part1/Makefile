all: pi.out


pi.out: pi.cpp
	g++ -o pi.out -lpthread -g -mavx2 -O3 -std=c++11 -I . -ftree-vectorize \
	-ftree-vectorizer-verbose=7 -ffast-math -fopt-info-vec-missed pi.cpp simdxorshift.cpp

clean:
	rm *.out