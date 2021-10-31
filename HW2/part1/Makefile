all: pi.out


pi.out: pi.cpp
	g++ -o pi.out -mavx2 -O3 -ftree-vectorize \
	-ftree-vectorizer-verbose=7 -ffast-math -fopt-info-vec-missed pi.cpp simdxorshift.cpp simdxorshift128plus.h -lpthread 

clean:
	rm *.out