#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <random>
#include <algorithm>
#include <thread>
#include <chrono>



float** createDistanceMatrix(int, int);
void calculateThreadCountAndPrefixSize(int&, int&, int, int);
int* calculateInitialPermutationFromThreadID(int, int, int);
float calculatePermutationTotalDistance(float**, int*, int);
void executeThread(int, float**, int, int, int*, float*);
void printDistanceMatrix(float**, int);



int main(int argc, char** argv) {
	std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

	int pointCount{};
	int coreCount{};
	int randomSeed{};

	if (argc == 4) {
		pointCount = std::atoi(argv[1]);
		coreCount = std::atoi(argv[2]);
		randomSeed = std::atoi(argv[3]);
		if (pointCount <= 2) {
			std::cerr << "Please set Point Count greater than 2" << std::endl;
			std::exit(1);
		}
		if (coreCount < 1) {
			std::cerr << "Please set core count greater than 1" << std::endl;
			std::exit(1);
		}
	}
	else {
		std::cerr << "Usage: " << argv[0] << " <point count (int)> <core count (int)> <random seed (int)>" << std::endl
			<< "point count = Number of points to be generated for the traveling salesman problem" << std::endl
			<< "core count = Number of available CPU cores" << std::endl
			<< "random seed = random seed used to generate the distances between points" << std::endl;
		std::exit(1);
	}


	float** distanceMatrix = createDistanceMatrix(pointCount, randomSeed);
	printDistanceMatrix(distanceMatrix, pointCount);

	int threadCount, prefixSize;
	calculateThreadCountAndPrefixSize(threadCount, prefixSize, pointCount, coreCount);

	int** allThreadsBestResultPermutations = new int* [threadCount];
	for (int i = 0; i < threadCount; i++) {
		allThreadsBestResultPermutations[i] = new int[pointCount];
	}
	float* allThreadsBestResults = new float[threadCount];

	std::chrono::steady_clock::time_point t_threadStart = std::chrono::steady_clock::now();
	std::thread* threads = new std::thread[threadCount];
	for (int i = 0; i < threadCount; i++) {
		threads[i] = std::thread(executeThread, i, distanceMatrix, pointCount, prefixSize, allThreadsBestResultPermutations[i], allThreadsBestResults + i);
	}
	for (int i = 0; i < threadCount; i++) {
		threads[i].join();
	}
	std::chrono::steady_clock::time_point t_threadEnd = std::chrono::steady_clock::now();
	


	float bestResult = FLT_MAX;
	int* bestResultPermutation = allThreadsBestResultPermutations[0];
	for (int i = 0; i < threadCount; i++) {
		if (allThreadsBestResults[i] < bestResult) {
			bestResult = allThreadsBestResults[i];
			bestResultPermutation = allThreadsBestResultPermutations[i];
		}
	}
	std::cout << "Best Route:" << std::endl;
	for (int i = 0; i < pointCount; i++) {
		std::cout << bestResultPermutation[i] << " -> ";
	}
	std::cout << bestResultPermutation[0] << std::endl;
	std::cout << "Route Length: " << bestResult << std::endl;


	//delete stuff
	for (int i = 0; i < pointCount; i++) {
		delete[] distanceMatrix[i];
	}
	delete[] distanceMatrix;
	for (int i = 0; i < threadCount; i++) {
		delete[]  allThreadsBestResultPermutations[i];
	}
	delete[] allThreadsBestResultPermutations;
	delete[] threads;
	delete[] allThreadsBestResults;

	// print execution times
	std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> totalExecutionTime = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
	std::chrono::duration<double> threadExecutionTime = std::chrono::duration_cast<std::chrono::duration<double>>(t_threadEnd - t_threadStart);
	std::cout << "Total execution time (s): " << totalExecutionTime.count() << std::endl;
	std::cout << "Thread execution time (s): " << threadExecutionTime.count() << std::endl;

	return 0;
}

// delete this, when done
float** createDistanceMatrix(int matrixSize, int randomSeed) {
	float** matrix = new float* [matrixSize];
	std::default_random_engine generator(randomSeed);
	std::uniform_real_distribution<float> distribution(0.0f, 1000.0f);
	for (int i = 0; i < matrixSize; i++) {
		matrix[i] = new float[matrixSize];
		for (int j = 0; j < matrixSize; j++) {
			matrix[i][j] = distribution(generator);
		}
	}
	return matrix;
}

void calculateThreadCountAndPrefixSize(int& threadCount, int& prefixSize, int pointCount, int coreCount) {
	threadCount = 1;
	prefixSize = 0;
	for (int i = pointCount; i > 0; i--) {
		if (threadCount < coreCount) {
			threadCount *= i;
			prefixSize++;
		}
		else {
			break;
		}
	}
}

// delete this, when done
int* calculateInitialPermutationFromThreadID(int id, int pointCount, int prefixSize) {
	int* factoradic = new int[prefixSize];

	int* out = new int[pointCount];
	for (int i = 0; i < pointCount; i++) {
		out[i] = i;
	}

	int quotient = id;
	for (int i = prefixSize - 1; i >= 0; i--) {
		factoradic[i] = quotient % (pointCount - i);
		quotient /= pointCount - i;
	}

	for (int i = 0; i < prefixSize; i++) {
		std::swap(out[i], out[factoradic[i] + i]);
	}

	delete[] factoradic;
	return out;
}

float calculatePermutationTotalDistance(float** distanceMatrix, int* permutation, int pointCount) {
	float distance = distanceMatrix[permutation[pointCount - 1]][permutation[0]];
	for (int i = 0; i < pointCount - 1; i++) {
		distance += distanceMatrix[permutation[i]][permutation[i + 1]];
	}
	return distance;
}

void executeThread(int id, float** distanceMatrix, int pointCount, int prefixSize, int* bestResultPermutation, float* bestResult) {
	int* permutation = calculateInitialPermutationFromThreadID(id, pointCount, prefixSize);
	int* prefix = permutation;
	int* permutable = permutation + prefixSize;
	int permutableSize = pointCount - prefixSize;

	std::memcpy(bestResultPermutation, permutation, sizeof(int) * pointCount);
	*bestResult = calculatePermutationTotalDistance(distanceMatrix, permutation, pointCount);

	//Heap's Algorithm
	int* c = new int[permutableSize];
	for (int i = 0; i < permutableSize; i++) {
		c[i] = 0;
	}
	int i = 0;
	while (i < permutableSize) {
		if (c[i] < i) {
			if (i % 2 == 0) {
				std::swap(permutation[0], permutation[i]);
			}
			else {
				std::swap(permutation[c[i]], permutation[i]);
			}

			float tempResult = calculatePermutationTotalDistance(distanceMatrix, permutation, pointCount);
			if (tempResult < *bestResult) {
				*bestResult = tempResult;
				std::memcpy(bestResultPermutation, permutation, sizeof(int) * pointCount);
			}


			c[i]++;
			i = 0;
		}
		else {
			c[i] = 0;
			i++;
		}
	}

	delete[] permutation;
	delete[] c;
}

void printDistanceMatrix(float** distanceMatrix, int pointCount) {
	std::cout << "Distance Matrix:" << std::endl;
	std::cout << "------" << "\t";
	for (int i = 0; i < pointCount; i++) {
		std::cout << i << ":\t";
	}
	std::cout << std::endl;
	for (int i = 0; i < pointCount; i++) {
		std::cout << i << ":\t";
		for (int j = 0; j < pointCount; j++) {
			std::cout << distanceMatrix[i][j] << "\t";
		}
		std::cout << std::endl;
	}
}