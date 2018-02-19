/**
 * \file	LBVHBuilderKernels.h
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	LBVHBuilder kernels header file.
 */

#ifndef _LBVH_BUILDER_KERNELS_H_
#define _LBVH_BUILDER_KERNELS_H_

#include "CudaBVHNode.h"

#define BLOCK_THREADS 256

namespace FW {

#ifdef __CUDACC__
extern "C" {

	__constant__ float sceneBoxConst[6];
	__device__ float sceneBox[6];
	__device__ float cost;
	__device__ int prefixScanOffset;

	__global__ void computeSceneBox(
		const int threads,
		const int numberOfVertices
	);

	__global__ void computeMortonCodes30(
		const int threads,
		const int numberOfTriangles, 
		unsigned int * mortonCodes, 
		int * triangleIndices
	);

	__global__ void computeMortonCodes60(
		const int threads,
		const int numberOfTriangles, 
		unsigned long long * mortonCodes, 
		int * triangleIndices
	);

	__global__  void setupLeaves(
		const int threads,
		const int numberOfTriangles,
		int * triangleIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax
	);

	__global__ void construct30(
		const int n,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		unsigned int * mortonCodes
		);

	__global__ void construct60(
		const int n,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		unsigned long long * mortonCodes
	);

	__global__ void refit(
		const int threads,
		const int numberOfNodes,
		int * termCounters,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax
	);

	__global__ void woopifyTriangles(
		const int threads,
		const int numberOfTriangles,
		int * triangleIndices,
		Vec4f * triWoopsA,
		Vec4f * triWoopsB,
		Vec4f * triWoopsC
	);

	__global__ void computeCost(
		const int threads,
		const int numberOfNodes,
		const float sceneBoxArea,
		const float ct,
		const float ci,
		CudaBVHNode * nodes
	);

}
#endif

};

#endif /* _LBVH_BUILDER_KERNELS_H_ */
