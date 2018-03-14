default: no-cuda-debug

debug: main.cu
	@echo "target: debug"
	@if [ ! -d bin ]; then mkdir bin; fi
	nvcc -xc -ggdb -o bin/main main.cu

no-cuda-debug: main.cu
	@echo "target: no-cuda-debug"
	@if [ ! -d bin ]; then mkdir bin; fi
	gcc -xc -ggdb -o bin/main main.cu

clean:
	@echo "target: clean"
	@rm -rf bin
