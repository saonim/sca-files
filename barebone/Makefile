SCRIPT_VERSION=v1.0
SCRIPT_AUTHOR=Saoni Mukherjee

all:
	nvcc -arch=compute_35 -code=sm_35 -m 64 template.cu -cubin -o template.cubin
	KeplerAs.pl -i barebone.sass template.cubin barebone.cubin
	rm -rf a_dlink.*
	rm -rf barebone.cpp*
	rm -rf a.out
	rm barebone.cu.cpp.ii
	rm barebone.fatbin*
	rm barebone.o
	fatbinary --create="barebone.fatbin" -64 "--image=profile=sm_35,file=barebone.cubin" "--image=profile=compute_35,file=barebone.ptx" --embedded-fatbin="barebone.fatbin.c" --cuda
	gcc -D__CUDA_ARCH__=350 -E -x c++           -DCUDA_DOUBLE_MATH_FUNCTIONS   -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "barebone.cudafe1.cpp" > "barebone.cu.cpp.ii" 
	gcc -c -x c++ "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "barebone.o" "barebone.cu.cpp.ii" 
	nvlink --arch=sm_35 --register-link-binaries="a_dlink.reg.c" -m64   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "barebone.o"  -o "a_dlink.sm_35.cubin"
	fatbinary --create="a_dlink.fatbin" -64 -link "--image=profile=sm_35,file=a_dlink.sm_35.cubin" --embedded-fatbin="a_dlink.fatbin.c" 
	gcc -c -x c++ -DFATBINFILE="\"a_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"a_dlink.reg.c\"" -I. "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -D"__CUDACC_VER__=80026" -D"__CUDACC_VER_BUILD__=26" -D"__CUDACC_VER_MINOR__=0" -D"__CUDACC_VER_MAJOR__=8" -m64 -o "a_dlink.o" "/usr/local/cuda/bin/crt/link.stub" 
	g++ -m64 -o "a.out" -Wl,--start-group "a_dlink.o" "barebone.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

gensass:
	nvcc -keep -arch=compute_35 -code=sm_35 barebone.cu
	nvcc -arch=compute_35 -code=sm_35 barebone.cu -cubin -o barebone.cubin
	KeplerAs.pl -e barebone.cubin > barebone.sass

clean:
	rm -rf a_dlink.*
	rm -rf a.out
	rm -rf barebone.cpp*
	rm -rf barebone.cu.*
	rm -rf barebone.cuda*
	rm -rf barebone.fatbin*
	rm -rf barebone.module_id
	rm -rf barebone.ptx
	rm -rf barebone.o
	rm -rf *~
	rm -rf barebone.cubin
	rm -rf template.cubin
	rm -rf barebone.sass


help:
	@echo "Usage: make [target] ...\n"
	@echo "\t gensass \t Generate the sass file from .cu file"
	@echo "\t all     \t Generate the executable from .cubin"
	@echo "\t clean   \t Remove all intermediate objects"
	@echo "\nWritten by $(SCRIPT_AUTHOR), version $(SCRIPT_VERSION)"
	@echo "Please report any bug or error to the author." 	
