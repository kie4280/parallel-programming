NVCC = nvcc
CXX=g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm

APP_NAME=mandelbrot
OBJDIR=objs
COMMONDIR=./common

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))


default: $(APP_NAME)

.PHONY: dirs clean sync zip

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/kernel.o $(OBJDIR)/mandelbrotSerial.o $(OBJDIR)/mandelbrotThread.o $(PPM_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) mandelbrotThreadRef.a

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/CycleTimer.h kernel.h

$(OBJDIR)/kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@


sync:
	rsync -aP ~/MyProjects/parallel/HW5 pp2:/home/0710018/

zip:
	zip HW5_0710018.zip kernel1.cu kernel2.cu kernel3.cu kernel4.cu url.txt
