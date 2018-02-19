/**
 * \file	LBVHBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	LBVHBuilder kernels soruce file.
 */

#include "LBVHBuilderKernels.h"
#include "CudaBVHUtil.cuh"

using namespace FW;

template <typename T>
__device__ void construct(
	const int n,
	const int i,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	T * mortonCodes
) {

	// Determine direction of the range (+1 or -1).
	const int d = sgn(delta(i, i + 1, n, mortonCodes) - delta(i, i - 1, n, mortonCodes));

	// Compute upper bound for the length of the range.
	const int deltaMin = delta(i, i - d, n, mortonCodes);
	int lmax = 2;
	while (delta(i, i + lmax * d, n, mortonCodes) > deltaMin) lmax <<= 1;

	// Find the other end using binary search.
	int l = 0;
	for (int t = lmax >> 1; t >= 1; t >>= 1)
	if (delta(i, i + (l + t) * d, n, mortonCodes) > deltaMin)
		l += t;
	const int j = i + l * d;

	// Find the split position using binary search.
	const int deltaNode = delta(i, j, n, mortonCodes);
	int s = 0;
	int k = 2;
	int t;
	do {
		t = divCeil(l, k);
		k <<= 1;
		if (delta(i, i + (s + t) * d, n, mortonCodes) > deltaNode)
			s += t;
	} while (t > 1);
	const int gamma = i + s * d + min<int>(d, 0);

	// Output child pointers.
	int left = gamma;
	int right = gamma + 1;
	if (min<int>(i, j) == gamma) left += n - 1;
	if (max<int>(i, j) == gamma + 1) right += n - 1;

	// Write node etc.
	nodeLeftIndices[i] = left;
	nodeRightIndices[i] = right;
	nodeSizes[i] = l + 1;

	// Parent indices.
	nodeParentIndices[left] = i;
	nodeParentIndices[right] = i;

}

extern "C" __global__ void computeSceneBox(
	const int threads,
	const int numberOfVertices
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Bounding box within the thread.
	AABB box; Vec3f vertex;
	for (int vertexIndex = threadIndex; vertexIndex < numberOfVertices; vertexIndex += threads) {
		vertexFromTexture(vertexIndex, vertex);
		box.grow(vertex);
	}

	// Cache.
	__shared__ float cache[3 * BLOCK_THREADS];
	Vec3f * bound = (Vec3f*)cache;

	// Min.
	bound[threadIdx.x] = box.min();
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

	__syncthreads();
	if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

	__syncthreads();
	if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

	__syncthreads();
	if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 128]);

	// Update global bounding box.
	if (threadIdx.x == 0) {
		atomicMin(&sceneBox[0], bound[threadIdx.x].x);
		atomicMin(&sceneBox[1], bound[threadIdx.x].y);
		atomicMin(&sceneBox[2], bound[threadIdx.x].z);
	}

	// Max.
	bound[threadIdx.x] = box.max();
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

	__syncthreads();
	if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

	__syncthreads();
	if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

	__syncthreads();
	if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 128]);

	// Update global bounding box.
	if (threadIdx.x == 0) {
		atomicMax(&sceneBox[3], bound[threadIdx.x].x);
		atomicMax(&sceneBox[4], bound[threadIdx.x].y);
		atomicMax(&sceneBox[5], bound[threadIdx.x].z);
	}

}

extern "C" __global__ void computeMortonCodes30(
	const int threads,
	const int numberOfTriangles, 
	unsigned int * mortonCodes, 
	int * triangleIndices
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Scene box.
	AABB _sceneBox = *(AABB*)sceneBoxConst;
	Vec3f scale = 1.0f / (_sceneBox.max() - _sceneBox.min());

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.		
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndex, v0, v1, v2);

		// Box.
		AABB box;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);

		// Triangle index, node index and Morton code.
		triangleIndices[triangleIndex] = triangleIndex;
		mortonCodes[triangleIndex] = mortonCode((box.midPoint() - _sceneBox.min()) * scale);

	}

}

extern "C" __global__ void computeMortonCodes60(
	const int threads,
	const int numberOfTriangles, 
	unsigned long long * mortonCodes,
	int * triangleIndices
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Scene box.
	AABB _sceneBox = *(AABB*)sceneBoxConst;
	Vec3f scale = 1.0f / (_sceneBox.max() - _sceneBox.min());

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.		
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndex, v0, v1, v2);

		// Box.
		AABB box;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);

		// Triangle index, node index and Morton code.
		triangleIndices[triangleIndex] = triangleIndex;
		mortonCodes[triangleIndex] = mortonCode64((box.midPoint() - _sceneBox.min()) * scale);

	}

}

extern "C" __global__  void setupLeaves(
	const int threads,
	const int numberOfTriangles,
	int * triangleIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	for (int boxIndex = threadIndex; boxIndex < numberOfTriangles; boxIndex += threads) {

		// Triangle index.
		const int triangleIndex = triangleIndices[boxIndex];
		
		// Triangle.		
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndex, v0, v1, v2);
		
		// Box.
		AABB box;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);

		// Leaf node.
		const int nodeIndex = boxIndex + numberOfTriangles - 1;
		nodeLeftIndices[nodeIndex] = boxIndex;
		nodeRightIndices[nodeIndex] = boxIndex + 1;
		nodeSizes[nodeIndex] = 1;
		nodeBoxesMin[nodeIndex] = Vec4f(box.min(), 0.0f);
		nodeBoxesMax[nodeIndex] = Vec4f(box.max(), 0.0f);

	}

}

extern "C" __global__ void construct30(
	const int n,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	unsigned int * mortonCodes
	) {

	// Thread index.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n - 1) {
		construct(
			n,
			i,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			nodeSizes,
			mortonCodes
			);
	}

	// Root parent index.
	if (i == 0)
		nodeParentIndices[0] = -1;


}

extern "C" __global__ void construct60(
	const int n,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	unsigned long long * mortonCodes
) {

	// Thread index.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n - 1) {
		construct(
			n,
			i,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			nodeSizes,
			mortonCodes
		);
	}

	// Root parent index.
	if (i == 0)
		nodeParentIndices[0] = -1;

	
}

extern "C" __global__ void refit(
	const int threads,
	const int numberOfNodes,
	int * termCounters,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Number of interior nodes.
	const int numberOfInteriorNodes = numberOfNodes >> 1;

	for (int leafIndex = threadIndex + numberOfInteriorNodes; leafIndex < numberOfNodes; leafIndex += threads) {

		// Node index.
		int nodeIndex = nodeParentIndices[leafIndex];

		// Go up to the root.
		while (nodeIndex >= 0 && atomicAdd(&termCounters[nodeIndex], 1) > 0) {
			
			// Node.
			int nodeLeftIndex = nodeLeftIndices[nodeIndex];
			int nodeRightIndex = nodeRightIndices[nodeIndex];

			// Box.
			AABB box;

			// Min.
			const Vec4f nodeLeftBoxMin = nodeBoxesMin[nodeLeftIndex];
			const Vec4f nodeRightBoxMin = nodeBoxesMin[nodeRightIndex];
			box.grow(Vec3f(nodeLeftBoxMin.x, nodeLeftBoxMin.y, nodeLeftBoxMin.z));
			box.grow(Vec3f(nodeRightBoxMin.x, nodeRightBoxMin.y, nodeRightBoxMin.z));
			nodeBoxesMin[nodeIndex] = Vec4f(box.min(), 0.0f);

			// Max.
			const Vec4f nodeLeftBoxMax = nodeBoxesMax[nodeLeftIndex];
			const Vec4f nodeRightBoxMax = nodeBoxesMax[nodeRightIndex];
			box.grow(Vec3f(nodeLeftBoxMax.x, nodeLeftBoxMax.y, nodeLeftBoxMax.z));
			box.grow(Vec3f(nodeRightBoxMax.x, nodeRightBoxMax.y, nodeRightBoxMax.z));
			nodeBoxesMax[nodeIndex] = Vec4f(box.max(), 0.0f);

			// Go to the parent.
			nodeIndex = nodeParentIndices[nodeIndex];

		}

	}

}

extern "C" __global__ void woopifyTriangles(
	const int threads,
	const int numberOfTriangles,
	int * triangleIndices,
	Vec4f * triWoopsA,
	Vec4f * triWoopsB,
	Vec4f * triWoopsC
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Woop's matrix.
	Mat4f im;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndices[triangleIndex], v0, v1, v2);

		// Woopify triangle.
		im.setCol(0, Vec4f(v0 - v2, 0.0f));
		im.setCol(1, Vec4f(v1 - v2, 0.0f));
		im.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
		im.setCol(3, Vec4f(v2, 1.0f));
		im = invert(im);

		triWoopsA[triangleIndex] = Vec4f(im(2, 0), im(2, 1), im(2, 2), -im(2, 3));
		triWoopsB[triangleIndex] = im.getRow(0);
		triWoopsC[triangleIndex] = im.getRow(1);

	}

}

extern "C" __global__ void computeCost(
	const int threads,
	const int numberOfNodes,
	const float sceneBoxArea,
	const float ct,
	const float ci,
	CudaBVHNode * nodes
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Cost.
	float _cost = 0.0f;

	for (int nodeIndex = threadIndex; nodeIndex < numberOfNodes; nodeIndex += threads) {

		CudaBVHNode node = nodes[nodeIndex];
		float P = node.getSurfaceArea() / sceneBoxArea;

		// Leaf.
		if (node.isLeaf()) {
			_cost += ci * P * node.getSize();
		}

		// Interior node.
		else {
			_cost += ct * P;
		}
	}

	// Cache.
	__shared__ volatile float cache[BLOCK_THREADS];

	// Cost reduction.
	cache[threadIdx.x] = _cost;
	cache[threadIdx.x] += cache[threadIdx.x ^ 1];
	cache[threadIdx.x] += cache[threadIdx.x ^ 2];
	cache[threadIdx.x] += cache[threadIdx.x ^ 4];
	cache[threadIdx.x] += cache[threadIdx.x ^ 8];
	cache[threadIdx.x] += cache[threadIdx.x ^ 16];

	__syncthreads();
	if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];

	__syncthreads();
	if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];

	__syncthreads();
	if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];

	// Update total cost.
	if (threadIdx.x == 0) {
		atomicAdd(&cost, cache[threadIdx.x]);
	}

}

