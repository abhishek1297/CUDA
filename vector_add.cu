
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#define N 65535

__global__ void vecadd(int *a, int *b, int *c) {

	int id = blockIdx.x;
	if (id < N)
		c[id] = a[id] * b[id];
}

void add(int *a, int *b, int *c) {
	for (int id=0; id < N; id++)
		c[id] = a[id] * b[id];
}

/*
int main() {

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc(&dev_a, N *sizeof(int));
	cudaMalloc(&dev_b, N *sizeof(int));
	cudaMalloc(&dev_c, N *sizeof(int));

	for (int i=0; i<N; ++i) {

		a[i] = i + 1;
		b[i] = i + 2;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);


	clock_t gpu_t = std::clock();
	vecadd<<<N, 256>>>(dev_a, dev_b, dev_c);
	gpu_t  = std::clock()  - gpu_t;

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	clock_t cpu_t = std::clock();
	add(a, b, c);
	cpu_t = std::clock() - cpu_t;

	std::cout << std::setprecision(10) << "GPU: " << double(gpu_t) / double(CLOCKS_PER_SEC) << " sec" << std::endl <<
	"CPU: " << double(cpu_t) / double(CLOCKS_PER_SEC) << " sec" << std::endl;
	std::cout << std::setprecision(5) << double(gpu_t) / double(cpu_t) << std::endl;
	return 0;
}

*/
