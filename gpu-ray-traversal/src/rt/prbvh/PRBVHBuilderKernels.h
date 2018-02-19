/**
 * \file	PRBVHBuilderKernels.h
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	PRBVHBuilder kernels header file.
 */

#ifndef _PRBVH_BUILDER_KERNELS_H_
#define PRBVH_BUILDER_KERNELS_H_

#include "CudaBVHNode.h"

#define BLOCK_THREADS 256
#define FAST 1

namespace FW {

struct Path {

	const static int SIZE = 2;
	const static int BITS = sizeof(unsigned long long) * SIZE * 8;
	unsigned long long bitset[SIZE];

	FW_CUDA_FUNC Path(void) {
		for (int i = 0; i < SIZE; ++i)
			bitset[i] = 0;
	}

	FW_CUDA_FUNC Path(const Path & other) {
		for (int i = 0; i < SIZE; ++i)
			bitset[i] = other.bitset[i];
	}

	FW_CUDA_FUNC Path(unsigned long long bitset0) {
		bitset[0] = bitset0;
		for (int i = 1; i < SIZE; ++i)
			bitset[i] = 0;
	}

	FW_CUDA_FUNC void pushZero(void) {
		for (int i = SIZE - 1; i >= 1; --i) {
			bitset[i] <<= 1ull;
			if (bitset[i - 1] & (1ull << unsigned long long(63)))
				bitset[i] |= 1ull;
		}
		bitset[0] <<= 1ull;
	}

	FW_CUDA_FUNC void pushOne(void) {
		pushZero();
		bitset[0] |= 1ull;
	}

	FW_CUDA_FUNC int popBack(void) {
		int bit = bitset[0] & 1ull;
		for (int i = 0; i < SIZE - 1; ++i) {
			bitset[i] >>= 1;
			if (bitset[i + 1] & 1ull)
				bitset[i] |= (1ull << unsigned long long(63));
		}
		bitset[SIZE - 1] >>= 1ull;
		return bit;
	}

	FW_CUDA_FUNC int popFront(void) {
		int bit = bitset[SIZE - 1] & (1ull << unsigned long long(63)) ? 1 : 0;
		pushZero();
		return bit;
	}

	FW_CUDA_FUNC int skipFrontZeros(void) {
		int skip = 0;
		while (popFront() == 0 && skip < BITS)
			++skip;
		return skip;
	}

	FW_CUDA_FUNC int skipFrontOnes(void) {
		int skip = 0;
		while (popFront() == 1 && skip < BITS)
			++skip;
		return skip;
	}

	FW_CUDA_FUNC int highestBitIndex(void) {
		Path temp = *this;
		temp.skipFrontZeros();
		return temp.skipFrontOnes() - 1;
	}

};

#ifdef __CUDACC__
extern "C" {

__device__ int foundNodes;
__device__ int insertedNodes;
__device__ float cost;

__global__ void findBestNode(
	const int numberOfNodes,
	const int numberOfTriangles,
	const int mod,
	const int remainder,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * outNodeIndices,
	float * areaReductions,
	Path * paths,
	unsigned short * lockBounds,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
);

__global__ void lockNodes(
	const int numberOfNodes,
	const int mod,
	const int remainder,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * outNodeIndices,
	float * areaReductions,
	Path * paths,
	unsigned long long * locks,
	unsigned short * zeroBounds
);

__global__ void checkLocks(
	const int numberOfNodes,
	const int mod,
	const int remainder,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * outNodeIndices,
	float * areaReductions,
	Path * paths,
	unsigned long long * locks,
	unsigned short * zeroBounds
);

__global__ void reinsert(
	const int numberOfNodes,
	const int mod,
	const int remainder,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * outNodeIndices,
	float * areaReductions
);

__global__ void computeSizes(
	const int threads,
	const int numberOfNodes,
	int * termCounters,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes
);

__global__ void computeCost(
	const int numberOfNodes,
	const int numberOfTriangles,
	const float sceneBoxArea,
	const float ct, 
	const float ci, 
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
);

}
#endif

};

#endif /* _PRBVH_BUILDER_KERNELS_H_ */
