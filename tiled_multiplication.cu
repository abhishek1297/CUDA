#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

#define TILE_WIDTH 20
#define WIDTH 10000
#define MATSIZE 256
#define SIZE (WIDTH * WIDTH * sizeof(float))

float A[WIDTH][WIDTH], B[WIDTH][WIDTH], C[WIDTH][WIDTH];
float *dev_A, *dev_B, *dev_C;

__global__ void matmul(float *A, float *B, float *C) {


	__shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	if (row > WIDTH || col > WIDTH)
		return;
	int partial_val = 0;

	for (int tile_id=0; tile_id<WIDTH/TILE_WIDTH; ++tile_id) {

		sh_A[threadIdx.y][threadIdx.x] = A[row * WIDTH + (tile_id * TILE_WIDTH + threadIdx.x)];
		sh_B[threadIdx.y][threadIdx.x] = B[col + (tile_id * TILE_WIDTH + threadIdx.y) * WIDTH];

		__syncthreads();

		for (int i=0; i<TILE_WIDTH; ++i) {
			partial_val += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];
			__syncthreads();
		}
	}


	C[row * WIDTH + col] = partial_val;
}

std::string getDimString(dim3 blocks, dim3 threads) {

	std::stringstream ss;
	ss << "BLOCKS: (" <<  blocks.x << ", " << blocks.y << ")";
	ss << " THREADS: (" << threads.x << ", " << threads.y << ")";

	return ss.str();
}

int main () {

	std::cout << "Not Done" << std::endl;



	for (int i=0; i<WIDTH; ++i) {
		for (int j=0; j<WIDTH; ++j) {
			A[i][j] = B[i][j] = 2.5;
		}
	}

	cudaMalloc(&dev_A, SIZE);
	cudaMalloc(&dev_B, SIZE);
	cudaMalloc(&dev_C, SIZE);

	cudaMemcpy(dev_A, A, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_C, C, SIZE, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 blocksPerGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);

	std::cout << "Calculating GPU" << std::endl;
	std::clock_t t = std::clock();
	matmul<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C);
	cudaDeviceSynchronize();
	t = std::clock() - t;
	std::ofstream out{"time_logs.log", std::ios_base::app};

	std::cout << std::setprecision(5) << getDimString(blocksPerGrid, threadsPerBlock) <<
	", TIME: "<< double(t) / double(CLOCKS_PER_SEC) << std::endl;

	cudaMemcpy(C, dev_C, SIZE, cudaMemcpyDeviceToHost);
//	std::cout << "Calculating CPU" << std::endl;
//	t = std::clock();
//	for (int i=0; i<WIDTH; ++i) {
//		for (int j=0; j<WIDTH; ++j) {
//			int partial_val{0};
//			for (int k=0; k<WIDTH; ++k) {
//				partial_val += A[i][k] * B[k][j];
//			}
//		}
//	}
//	t = std::clock() - t;
//	std::cout << std::setprecision(5) << double(t) / double(CLOCKS_PER_SEC) << std::endl;

	out.close();
	cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
	std::cout << "Done!" << std::endl;

	return 0;

}

