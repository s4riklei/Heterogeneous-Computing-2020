
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>
#include <utility>
#include <cstddef>
#include <cfloat>
#include <chrono>

namespace {

	struct Metadata {
		int desiredBlockCount;		// minimum number of blocks to create
		int threadCountPerBlock;	// threads per block
		int pointCount;				// number of points for the traveling salesman to visit
		int randomSeed;				// random seed used to generate the distance matrix
		int threadCount;			// actual number of generated threads
		int prefixSize;				// amount of static points per thread, that don't get permutated
	};

	struct DevicePointers {
		float* distanceMatrix = 0;
		// memory block containing the best result permutations found by each thread (device)
		int* allThreadsBestResultPermutations = 0;
		// array containing the best distance found by each thread (device)
		float* allThreadsBestResults = 0;
		// memory block containing the current permutation state threads are operating on
		int* allThreadsPermutations = 0;
		// array containing the temp state while executing Heap's algorithm
		int* allThreadsC = 0;
	};

	Metadata parseArgs(int argc, char** argv) {
		int desiredBlockCount;
		int threadCountPerBlock;
		int pointCount;
		int randomSeed;
		int threadCount;
		int prefixSize;

		// validate user input
		if (argc != 5) {
			throw std::invalid_argument("Error: Incorrect number of arguments");
		}

		try {
			desiredBlockCount = std::stoi(argv[1]);
			if (desiredBlockCount < 1) {
				throw std::invalid_argument("Error: Please set block count to a value greater than 0");
			}

			threadCountPerBlock = std::stoi(argv[2]);
			if (threadCountPerBlock < 1 || threadCountPerBlock > 1024) {
				throw std::invalid_argument("Error: Please set thread count per block to a value between 1 and 1024");
			}

			pointCount = std::stoi(argv[3]);
			if (pointCount < 2) {
				throw std::invalid_argument("Error: Please set point count to a value greater than 1");
			}

			randomSeed = std::stoi(argv[4]);
		}
		catch (const std::out_of_range& e) {
			std::cout << e.what() << std::endl;
			throw std::invalid_argument("Error: A parameter value was out of range");
		}

		// calculate thread count and prefix size
		([&] {
			int desiredThreadCount = desiredBlockCount * threadCountPerBlock;
			threadCount = 1;
			prefixSize = 0;
			for (int i = pointCount; (i > 0 && threadCount < desiredThreadCount); i--) {
				threadCount *= i;
				prefixSize++;
			}
		})();

		Metadata m{
			desiredBlockCount,
			threadCountPerBlock,
			pointCount,
			randomSeed,
			threadCount,
			prefixSize
		};
		return m;
	}

	std::vector<std::vector<float>> createDistanceMatrix(const Metadata& metadata) {
		std::default_random_engine generator(metadata.randomSeed);
		std::uniform_real_distribution<float> distribution(0.0f, 1000.0f);

		std::vector<std::vector<float>> matrix;
		matrix.reserve(metadata.pointCount);
		for (int i = 0; i < metadata.pointCount; i++) {
			std::vector<float> row;
			row.reserve(metadata.pointCount);
			for (int i = 0; i < metadata.pointCount; i++) {
				row.push_back(distribution(generator));
			}
			matrix.push_back(std::move(row));
		}

		return matrix;
	}

	void printDistanceMatrix(const std::vector<std::vector<float>>& matrix) {
		std::cout << "Distance Matrix:" << std::endl;
		std::cout << "------" << "\t";
		for (int i = 0; i < matrix.size(); i++) {
			std::cout << i << ":\t";
		}
		std::cout << std::endl;

		int counter = 0;
		for (auto& row : matrix) {
			std::cout << counter++ << ":\t";
			for (auto& val : row) {
				std::cout << val << "\t";
			}
			std::cout << std::endl;
		}
	}

	DevicePointers allocateDeviceMemory(const Metadata& metadata) {
		cudaError_t cudaStatus;

		DevicePointers d;

		// since metadata.threadCount is probably not always divisible by metadata.threadCountPerBlock,
		// some "overhead" memory needs to be allocated to not cause segfaults when coalescing memory
		int actualThreadCount = ([&] {
			int result = metadata.threadCount / metadata.threadCountPerBlock;
			result += metadata.threadCount % metadata.threadCountPerBlock == 0 ? 0 : 1;
			result *= metadata.threadCountPerBlock;
			return result;
		})();

		std::size_t size = metadata.pointCount * metadata.pointCount * sizeof(float);
		cudaStatus = cudaMalloc(&d.distanceMatrix, size);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("Error allocating memory!");
		}

		size = actualThreadCount * metadata.pointCount * sizeof(int);
		cudaStatus = cudaMalloc(&d.allThreadsBestResultPermutations, size);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("Error allocating memory!");
		}

		size = actualThreadCount * sizeof(float);
		cudaStatus = cudaMalloc(&d.allThreadsBestResults, size);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("Error allocating memory!");
		}

		size = actualThreadCount * metadata.pointCount * sizeof(int);
		cudaStatus = cudaMalloc(&d.allThreadsPermutations, size);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("Error allocating memory!");
		}

		size = actualThreadCount * (metadata.pointCount - metadata.prefixSize) * sizeof(int);
		cudaStatus = cudaMalloc(&d.allThreadsC, size);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("Error allocating memory!");
		}

		return d;
	}

	void freeDeviceMemory(DevicePointers& d) {
		cudaFree(d.distanceMatrix);
		cudaFree(d.allThreadsBestResultPermutations);
		cudaFree(d.allThreadsBestResults);
		cudaFree(d.allThreadsPermutations);
		cudaFree(d.allThreadsC);

		d.distanceMatrix = 0;
		d.allThreadsBestResultPermutations = 0;
		d.allThreadsBestResults = 0;
		d.allThreadsPermutations = 0;
		d.allThreadsC = 0;
	}

	void copyDataHostToDevice(const DevicePointers& d, const std::vector<std::vector<float>>& distanceMatrix) {
		float* currentPointer = d.distanceMatrix;
		cudaError_t cudaStatus;
		for (auto& row : distanceMatrix) {
			std::size_t count = row.size() * sizeof(float);
			cudaStatus = cudaMemcpy(currentPointer, row.data(), count, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				throw std::runtime_error("Error copying distance matrix onto device!");
			}
			currentPointer += row.size();
		}
	}

	__device__ float calculatePermutationTotalDistance(float* distanceMatrix, int* permutation, int pointCount) {
		float distance = distanceMatrix[permutation[blockDim.x * (pointCount - 1)] * pointCount + permutation[0]];
		for (int i = 0; i < pointCount - 1; i++) {
			distance += distanceMatrix[permutation[blockDim.x * i] * pointCount + permutation[blockDim.x * (i + 1)]];
		}
		return distance;
	}

	__device__ void swap(int& a, int& b) {
		int tmp = a;
		a = b;
		b = tmp;
	}

	__device__ void copyPermutationToBestArray(int* dest, int* src, int size) {
		for (int i = 0; i < size; i++) {
			dest[i] = src[blockDim.x * i];
		}
	}

	__device__ void prepareSharedDistanceMatrix(const int id, const int pointCount, const float* global, float* shared) {
		int size = pointCount * pointCount;
		for (int i = threadIdx.x; i < size; i += blockDim.x) {
			shared[i] = global[i];
		}
	}

	__global__ void calculateBestRoute(const int threadCount, const int prefixSize, const int pointCount, const DevicePointers d) {
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadCount <= id) {
			return;
		}
		
		// initialize permutation
		int* permutation = d.allThreadsPermutations + blockIdx.x * blockDim.x * pointCount + threadIdx.x;

		for (int i = 0; i < pointCount; i++) {
			permutation[blockDim.x * i] = i;
		}

		int quotient = id;
		for (int i = prefixSize - 1; i >= 0; i--) {
			int factoradic = quotient % (pointCount - i);
			quotient /= pointCount - i;
			swap(permutation[blockDim.x * i], permutation[blockDim.x * (factoradic + i)]);
		}

		// prepare pointers
		int* permutable = permutation +  blockDim.x * prefixSize;
		int permutableSize = pointCount - prefixSize;
		int* bestResultPermutations = d.allThreadsBestResultPermutations + id * pointCount;
		extern __shared__ float sharedDistanceMatrix[];

		prepareSharedDistanceMatrix(id, pointCount, d.distanceMatrix, sharedDistanceMatrix);
		__syncthreads();

		// initial solution is initial best
		copyPermutationToBestArray(bestResultPermutations, permutation, pointCount);
		float bestResult = calculatePermutationTotalDistance(sharedDistanceMatrix, permutation, pointCount);

		if (permutableSize != 0) {
			int* c = d.allThreadsC + blockIdx.x * blockDim.x * permutableSize + threadIdx.x;
			for (int i = 0; i < permutableSize; i++) {
				c[blockDim.x * i] = 0;
			}
			int i = 0;
			while (i < permutableSize) {
				if (c[blockDim.x * i] < i) {
					if (i % 2 == 0) {
						swap(permutable[0], permutable[blockDim.x * i]);
					}
					else {
						swap(permutable[blockDim.x * c[blockDim.x * i]], permutable[blockDim.x * i]);
					}

					float tempResult = calculatePermutationTotalDistance(sharedDistanceMatrix, permutation, pointCount);
					if (tempResult < bestResult) {
						copyPermutationToBestArray(bestResultPermutations, permutation, pointCount);
						bestResult = tempResult;
					}


					c[blockDim.x * i]++;
					i = 0;
				}
				else {
					c[blockDim.x * i] = 0;
					i++;
				}
			}
		}

		d.allThreadsBestResults[id] = bestResult;
	}

	void executeThreads(const Metadata& metadata, const DevicePointers& d) {
		int blockCount = ([&] {
			int result = metadata.threadCount / metadata.threadCountPerBlock;
			result += metadata.threadCount % metadata.threadCountPerBlock == 0 ? 0 : 1;
			return result;
		})();
		std::size_t sharedMemorySize = metadata.pointCount * metadata.pointCount * sizeof(float);
		calculateBestRoute<<<blockCount, metadata.threadCountPerBlock, sharedMemorySize>>>(metadata.threadCount, metadata.prefixSize, metadata.pointCount, d);
	}

	void copyDataDeviceToHostAndEvaluate(const Metadata& metadata, const DevicePointers& d) {
		// copy data
		cudaError_t cudaStatus;
		
		std::vector<std::vector<int>> allThreadsBestResultPermutations = ([&] {
			std::vector<std::vector<int>> result;
			result.reserve(metadata.threadCount);
			int* tmp = new int[metadata.threadCount * metadata.pointCount];
			std::size_t size = metadata.threadCount * metadata.pointCount * sizeof(int);
			cudaStatus = cudaMemcpy(tmp, d.allThreadsBestResultPermutations, size, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				throw std::runtime_error("Error copying data from device to host");
			}

			int* intervalStart = tmp;
			int* intervalEnd = tmp + metadata.pointCount;
			for (int i = 0; i < metadata.threadCount; i++) {
				result.push_back(std::vector<int>(intervalStart, intervalEnd));
				intervalStart += metadata.pointCount;
				intervalEnd += metadata.pointCount;
			}

			delete[] tmp;
			return result;
		})();

		std::vector<float> allThreadsBestResults = ([&] {
			float* tmp = new float[metadata.threadCount];
			std::size_t size = metadata.threadCount * sizeof(float);
			cudaStatus = cudaMemcpy(tmp, d.allThreadsBestResults, size, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				throw std::runtime_error("Error copying data from device to host");
			}

			std::vector<float> result(tmp, tmp + metadata.threadCount);
			delete[] tmp;
			return result;
		})();

		// evaluate
		float bestResult = FLT_MAX;
		std::vector<int>* bestResultPermutation = 0;
		for (int i = 0; i < allThreadsBestResults.size(); i++) {
			if (allThreadsBestResults[i] < bestResult) {
				bestResult = allThreadsBestResults[i];
				bestResultPermutation = &allThreadsBestResultPermutations[i];
			}
		}

		std::cout << "Best Route:" << std::endl;
		for (int i = 0; i < bestResultPermutation->size(); i++) {
			std::cout << bestResultPermutation->at(i) << " -> ";
		}
		std::cout << bestResultPermutation->at(0) << std::endl;
		std::cout << "Route Length: " << bestResult << std::endl;
	}
}

int main(int argc, char** argv) {
	DevicePointers devicePointers;

	try {
		std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

		const Metadata METADATA = parseArgs(argc, argv);
		
		std::vector<std::vector<float>> distanceMatrix = createDistanceMatrix(METADATA);
		printDistanceMatrix(distanceMatrix);

		devicePointers = allocateDeviceMemory(METADATA);

		copyDataHostToDevice(devicePointers, distanceMatrix);

		std::chrono::steady_clock::time_point t_threadStart = std::chrono::steady_clock::now();
		executeThreads(METADATA, devicePointers);
		cudaDeviceSynchronize();
		std::chrono::steady_clock::time_point t_threadEnd = std::chrono::steady_clock::now();

		copyDataDeviceToHostAndEvaluate(METADATA, devicePointers);

		freeDeviceMemory(devicePointers);

		cudaDeviceReset();

		std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
		std::chrono::duration<double> totalExecutionTime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
		std::chrono::duration<double> threadExecutionTime = std::chrono::duration_cast<std::chrono::duration<double>>(t_threadEnd - t_threadStart);
		std::cout << "Total execution time (s): " << totalExecutionTime.count() << std::endl;
		std::cout << "Thread execution time (s): " << threadExecutionTime.count() << std::endl;
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Usage: " << argv[0] << "<desired block count (int)> <thread count per block (int)>" <<
			" <point count (int)> <random seed (int)>" << std::endl;
		std::cerr << "desired block count = Minimum number of blocks to create" << std::endl;
		std::cerr << "thread count per block = Number of threads for each block" << std::endl;
		std::cerr << "point count = Number of points to be generated for the traveling salesman problem" << std::endl;
		std::cerr << "random seed = random seed used to generate the distances between points" << std::endl;
		return 1;
	}
	catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		freeDeviceMemory(devicePointers);
		return 1;
	}
}