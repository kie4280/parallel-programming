
matmul: main.cc matmul.cc
	mpicxx -c main.cc matmul.cc
	mpicxx main.o matmul.o -o matmul
	rm main.o matmul.o

hello:
	mpicxx ./hello.cc -o mpi_hello

sync:
	rsync -aP ~/MyProjects/parallel/HW4 pp5:/home/0710018/

zip:
	cp Makefile part2/Makefile;
	zip HW4_0710018.zip -r part1 part2 url.txt

.PHONY:
	zip