/**
 * \file	PRBVHBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	PRBVHBuilder kernels soruce file.
 */

#include "PRBVHBuilderKernels.h"
#include "CudaBVHUtil.cuh"

extern "C" __global__ void findBestNode(
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
	unsigned short * zeroBounds,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
) {

	// Thread index.
	const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

	// Best node found so far.
	float bestAreaReduction, areaReduction;
	int bestOutNodeIndex;
#if !FAST
	Path bestPath, path;
	unsigned short zeroBound;
	unsigned short bestZeroBound;
#endif

	// Node state.
	float inNodeParentArea, directAreaReductionBound;
	int inNodeParentIndex, inNodeSiblingIndex, outNodeIndex, highestIndex;
	AABB inNodeBox, outNodeBox, mergedBox, highestBox;

	// Down flag.
	bool down;

	if (inNodeIndex > 0 && inNodeIndex < numberOfNodes) {
		inNodeParentIndex = nodeParentIndices[inNodeIndex];
		
		// Node boxes.
		inNodeBox = AABB(nodeBoxesMin[inNodeIndex], nodeBoxesMax[inNodeIndex]);
		inNodeParentArea = AABB(nodeBoxesMin[inNodeParentIndex], nodeBoxesMax[inNodeParentIndex]).area();
		directAreaReductionBound = inNodeParentArea - inNodeBox.area();
		highestBox.reset();

		// Sibling index.
		outNodeIndex = inNodeIndex;
		if (nodeLeftIndices[inNodeParentIndex] == outNodeIndex)
			inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
		else
			inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];

		// Switch to sibling and go down.
		outNodeIndex = inNodeSiblingIndex;
		highestIndex = inNodeParentIndex;
		down = true;

		// Best node found so far.
		bestAreaReduction = 0.0f;
		areaReduction = 0.0f;
#if !FAST
		bestPath = 0;
		path = 2;
		zeroBound = 0;
#endif

		// Main loop.
		while (true) {

			// Bounding boxes.
			outNodeBox = AABB(nodeBoxesMin[outNodeIndex], nodeBoxesMax[outNodeIndex]);
			mergedBox = AABB(inNodeBox, outNodeBox);

			// Down.
			if (down) {

				// Check area reduction.
				float directAreaReduction = inNodeParentArea - mergedBox.area();
				if (bestAreaReduction < directAreaReduction + areaReduction) {
					bestAreaReduction = directAreaReduction + areaReduction;
					bestOutNodeIndex = outNodeIndex;
#if !FAST
					bestPath = path;
					bestZeroBound = zeroBound;
#endif
				}

				// Add area reduction.
				float areaReductionCur = outNodeBox.area() - mergedBox.area();
				areaReduction += areaReductionCur;

#if !FAST
				// Area doesn't change => Inc. end of zero zone.
				if (areaReductionCur == 0.0f) {
					++zeroBound;
				}
#endif

				// Leaf or pruning => Go up.
				if (outNodeIndex >= numberOfTriangles - 1 || areaReduction + directAreaReductionBound <= bestAreaReduction) {
					down = false;
				}

				// Interior => Go to the left child.
				else {
					outNodeIndex = nodeLeftIndices[outNodeIndex];
#if !FAST
					path.pushZero();
#endif
				}

			}

			// Up.
			else {

				// Parent index.
				int outNodeParentIndex = nodeParentIndices[outNodeIndex];

				// Subtract node's area.
				float areaReductionCur = outNodeBox.area() - mergedBox.area();
				areaReduction -= areaReductionCur;

#if !FAST
				// Area doesn't change => Dec. end of zero zone.
				if (areaReductionCur == 0.0f) {
					--zeroBound;
				}
#endif

				// Back to the highest node.
				if (outNodeParentIndex == highestIndex) {

#if !FAST
					// Back to highest.
					path.popBack();
#endif

					// Update cumulative box.
					highestBox.grow(outNodeBox);

					// Go back to the highest node.
					outNodeIndex = outNodeParentIndex;

					// Check area reduction, skip the node's parent.
					if (outNodeIndex != inNodeParentIndex) {

						mergedBox = AABB(inNodeBox, highestBox);
						float directAreaReduction = inNodeParentArea - mergedBox.area();
						if (bestAreaReduction < directAreaReduction + areaReduction) {
							bestAreaReduction = directAreaReduction + areaReduction;
							bestOutNodeIndex = outNodeIndex;
#if !FAST
							bestPath = path;
							bestZeroBound = zeroBound;
#endif
						}

						// Add area reduction.
						outNodeBox = AABB(nodeBoxesMin[outNodeIndex], nodeBoxesMax[outNodeIndex]);
						float areaReductionCur = outNodeBox.area() - highestBox.area();
						areaReduction += areaReductionCur;

#if !FAST
						// Area changed => Update bounds of zero zone.
						if (areaReductionCur > 0.0f) {
							zeroBound = ((zeroBound >> 8) + 1);
							zeroBound |= zeroBound << 8;
						}
#endif

					}

					// The highest node is root => Done.
					outNodeParentIndex = nodeParentIndices[outNodeIndex];
					if (outNodeParentIndex < 0) {
						break;
					}

#if !FAST
					// Inc. end of zero zone.
					++zeroBound;
#endif

					// Update the highest node.
					highestIndex = outNodeParentIndex;

					// Go down.
					down = true;

					// Switch to sibling.
#if !FAST
					path.pushOne();
					path.pushZero();
#endif
					if (nodeLeftIndices[highestIndex] == outNodeIndex) outNodeIndex = nodeRightIndices[highestIndex];
					else outNodeIndex = nodeLeftIndices[highestIndex];

				}

				// Still bellow the highest node.
				else {

					// Switch to right sibling.
					if (nodeLeftIndices[outNodeParentIndex] == outNodeIndex) {
						down = true;
						outNodeIndex = nodeRightIndices[outNodeParentIndex];
#if !FAST
						path.popBack();
						path.pushOne();
#endif
					}

					// Go up.
					else {
						outNodeIndex = outNodeParentIndex;
#if !FAST
						path.popBack();
#endif
					}

				}

			}

		}

		// Save the best out node and area reduction.
		areaReductions[inNodeIndex] = bestAreaReduction;
		outNodeIndices[inNodeIndex] = bestOutNodeIndex;
#if !FAST
		paths[inNodeIndex] = bestPath;
		zeroBounds[inNodeIndex] = bestZeroBound;
#endif

	}

}

extern "C" __global__ void lockNodes(
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
) {

	// Thread index.
	const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

	if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

		// Area reduction.
		float areaReduction = areaReductions[inNodeIndex];

		// Area reduction and out node index.
		unsigned long long lock = (unsigned long long(__float_as_int(areaReduction)) << 32ull) | unsigned long long(inNodeIndex << 3);

		// Only nodes with positive area reduction.
		if (areaReduction > 0.0f) {

			// In node parent index.
			int inNodeParentIndex = nodeParentIndices[inNodeIndex];

			// In node sibling index.
			int inNodeSiblingIndex;

			// State.
			int state;

			// Came from left.
			if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
				inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
				state = 1;
			}
			// Came from right.
			else {
				inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
				state = 2;
			}

			// Lock in node, its parent, and sibling.
			atomicMax(&locks[inNodeIndex], lock | 4);
			atomicMax(&locks[inNodeSiblingIndex], lock | 4);
			atomicMax(&locks[inNodeParentIndex], lock | 4);

			// Parent is not the root => lock parent's parent.
			if (inNodeParentIndex > 0) {
				int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
				atomicMax(&locks[inNodeParentParentIndex], lock | 4);
			}
			// Parent is root => lock sibling's children.
			else if (inNodeSiblingIndex < numberOfNodes - 1) {
				int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
				int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
				atomicMax(&locks[inNodeSiblingLeftIndex], lock | 4);
				atomicMax(&locks[inNodeSiblingRightIndex], lock | 4);
			}

			// Out node index.
			int outNodeIndex = outNodeIndices[inNodeIndex];

#if !FAST
			// Out node index.
			outNodeIndex = inNodeParentIndex;

			// Path.
			Path path = paths[inNodeIndex];

			// Zero bounds.
			unsigned short zeroBound = zeroBounds[inNodeIndex];
			unsigned char zeroBegin = zeroBound >> 8;
			unsigned char zeroEnd = zeroBound & 255;

			// Reverse and skip leading zeros.
			int highestBitIndex = path.highestBitIndex();
			int skip = path.skipFrontZeros();

			for (int bitIndex = 0; bitIndex < Path::BITS - 1 - skip; ++bitIndex) {

				// Extract bit.
				int bit = path.popFront();

				// From parent.
				if (state == 0) {
					// To right.
					if (bit) {
						outNodeIndex = nodeRightIndices[outNodeIndex];
					}
					// To left.
					else {
						outNodeIndex = nodeLeftIndices[outNodeIndex];
					}
				}

				// From left.
				else if (state == 1) {
					// To parent.
					if (bit) {
						int outNodeParentIndex = nodeParentIndices[outNodeIndex];
						if (outNodeIndex == nodeLeftIndices[outNodeParentIndex])
							state = 1;
						else
							state = 2;
						outNodeIndex = outNodeParentIndex;
					}
					// To right.
					else {
						outNodeIndex = nodeRightIndices[outNodeIndex];
						state = 0;
					}
				}

				// From right.
				else {
					// To parent.
					if (bit) {
						int outNodeParentIndex = nodeParentIndices[outNodeIndex];
						if (outNodeIndex == nodeLeftIndices[outNodeParentIndex])
							state = 1;
						else
							state = 2;
						outNodeIndex = outNodeParentIndex;
					}
					// To left.
					else {
						outNodeIndex = nodeLeftIndices[outNodeIndex];
						state = 0;
					}
				}

				// Lock the node.
				if (bitIndex < zeroBegin || bitIndex >= zeroEnd) {
					if (bitIndex < highestBitIndex)
						atomicMax(&locks[outNodeIndex], lock | 1 | 2);
					else
						atomicMax(&locks[outNodeIndex], lock | 1);
				}
				else {
					if (bitIndex < highestBitIndex)
						atomicMax(&locks[outNodeIndex], lock | 2);
					else
						atomicMax(&locks[outNodeIndex], lock);
				}

			}
#endif

			// Lock out node.
			atomicMax(&locks[outNodeIndex], lock | 4);

			// Parent is not the root => lock parent's parent.
			if (outNodeIndex > 0) {
				int outNodeParentIndex = nodeParentIndices[outNodeIndex];
				atomicMax(&locks[outNodeParentIndex], lock | 4);
			}
			// Parent is root => lock root's children.
			else {
				int outNodeLeftIndex = nodeLeftIndices[outNodeIndex];
				int outNodeRightIndex = nodeRightIndices[outNodeIndex];
				atomicMax(&locks[outNodeLeftIndex], lock | 4);
				atomicMax(&locks[outNodeRightIndex], lock | 4);
			}

		}

	}

}

extern "C" __global__ void checkLocks(
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
	) {

	// Thread index.
	const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

	if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

		// Best out node index and area reduction.
		float areaReduction = areaReductions[inNodeIndex];

		// Area reduction and out node index.
		unsigned long long lock0 = (unsigned long long(__float_as_int(areaReduction)) << 32) | unsigned long long(inNodeIndex << 3);

		// Only nodes with positive area reductions.
		if (areaReduction > 0.0f) {

			// Increment inserted nodes.
			atomicAdd(&foundNodes, 1);

			// In node parent index.
			int inNodeParentIndex = nodeParentIndices[inNodeIndex];

			// In node sibling index and state.
			int inNodeSiblingIndex;

			// State.
			int state;

			// Came from left.
			if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
				inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
				state = 1;
			}
			// Came from right.
			else {
				inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
				state = 2;
			}

			// Check in node, its parent, and sibling.
			if (locks[inNodeIndex] != (lock0 | 4)) areaReduction = 0.0f;
			if (locks[inNodeSiblingIndex] != (lock0 | 4)) areaReduction = 0.0f;
			if (locks[inNodeParentIndex] != (lock0 | 4)) areaReduction = 0.0f;

			// Parent is not the root => check parent's parent.
			if (inNodeParentIndex > 0) {
				int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
				if (locks[inNodeParentParentIndex] != (lock0 | 4)) areaReduction = 0.0f;
			}
			// Parent is root => check sibling's children.
			else if (inNodeSiblingIndex < numberOfNodes - 1) {
				int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
				int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
				if (locks[inNodeSiblingLeftIndex] != (lock0 | 4)) areaReduction = 0.0f;
				if (locks[inNodeSiblingRightIndex] != (lock0 | 4)) areaReduction = 0.0f;
			}

			// Out node index.
			int outNodeIndex = outNodeIndices[inNodeIndex];

#if !FAST
			// Out node index.
			outNodeIndex = inNodeParentIndex;

			// Path.
			Path path = paths[inNodeIndex];

			// Zero bounds.
			unsigned short zeroBound = zeroBounds[inNodeIndex];
			unsigned char zeroBegin = zeroBound >> 8;
			unsigned char zeroEnd = zeroBound & 255;

			// Reverse and skip leading zeros.
			int highestBitIndex = path.highestBitIndex();
			int skip = path.skipFrontZeros();

			for (int bitIndex = 0; bitIndex < Path::BITS - 1 - skip; ++bitIndex) {

				// Extract bit.
				int bit = path.popFront();

				// From parent.
				if (state == 0) {
					// To right.
					if (bit) {
						outNodeIndex = nodeRightIndices[outNodeIndex];
					}
					// To left.
					else {
						outNodeIndex = nodeLeftIndices[outNodeIndex];
					}
				}

				// From left.
				else if (state == 1) {
					// To parent.
					if (bit) {
						int outNodeParentIndex = nodeParentIndices[outNodeIndex];
						if (outNodeIndex == nodeLeftIndices[outNodeParentIndex])
							state = 1;
						else
							state = 2;
						outNodeIndex = outNodeParentIndex;
					}
					// To right.
					else {
						outNodeIndex = nodeRightIndices[outNodeIndex];
						state = 0;
					}
				}

				// From right.
				else {
					// To parent.
					if (bit) {
						int outNodeParentIndex = nodeParentIndices[outNodeIndex];
						if (outNodeIndex == nodeLeftIndices[outNodeParentIndex])
							state = 1;
						else
							state = 2;
						outNodeIndex = outNodeParentIndex;
					}
					// To left.
					else {
						outNodeIndex = nodeLeftIndices[outNodeIndex];
						state = 0;
					}
				}

				// Check node lock.
				unsigned long long lock1 = locks[outNodeIndex];
				int topologyBit = lock1 & 4;
				int changeBit0 = bitIndex < zeroBegin || bitIndex >= zeroEnd;
				int changeBit1 = lock1 & 1;
				int upBit0 = bitIndex < highestBitIndex;
				int upBit1 = lock1 & 2;
				if (lock1 > (lock0 | 7)) {
					if (topologyBit && bitIndex != highestBitIndex) areaReduction = 0.0f;
					if (changeBit0 && changeBit1) areaReduction = 0.0f;
					if (changeBit0 && upBit0 && !upBit1) areaReduction = 0.0f;
					if (!changeBit0 && !upBit0 && changeBit1 &&  upBit1) areaReduction = 0.0f;
				}

			}
#endif

			// Lock out node.
			if (locks[outNodeIndex] != (lock0 | 4)) areaReduction = 0.0f;

			// Parent is not the root => check parent's parent.
			if (outNodeIndex > 0) {
				int outNodeParentIndex = nodeParentIndices[outNodeIndex];
				if (locks[outNodeParentIndex] != (lock0 | 4)) areaReduction = 0.0f;
			}

			// Area rediction.
			areaReductions[inNodeIndex] = areaReduction;

		}

	}

}

extern "C" __global__ void reinsert(
	const int numberOfNodes,
	const int mod,
	const int remainder,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * outNodeIndices,
	float * areaReductions
	) {

	// Thread index.
	const int inNodeIndex = mod * (blockDim.x * blockIdx.x + threadIdx.x) + remainder;

	if (inNodeIndex < numberOfNodes && inNodeIndex > 0) {

		// Best out node index and area reduction.
		float areaReduction = areaReductions[inNodeIndex];

		// Only nodes with positive area reductions.
		if (areaReduction > 0.0f) {

			// In node parent index.
			int inNodeParentIndex = nodeParentIndices[inNodeIndex];

			// In node sibling index.
			int inNodeSiblingIndex;

			// Came from left.
			if (inNodeIndex == nodeLeftIndices[inNodeParentIndex]) {
				inNodeSiblingIndex = nodeRightIndices[inNodeParentIndex];
			}
			// Came from right.
			else {
				inNodeSiblingIndex = nodeLeftIndices[inNodeParentIndex];
			}

			// Remove.
			if (inNodeParentIndex != 0) {
				int inNodeParentParentIndex = nodeParentIndices[inNodeParentIndex];
				if (nodeLeftIndices[inNodeParentParentIndex] == inNodeParentIndex)
					nodeLeftIndices[inNodeParentParentIndex] = inNodeSiblingIndex;
				else
					nodeRightIndices[inNodeParentParentIndex] = inNodeSiblingIndex;
				nodeParentIndices[inNodeSiblingIndex] = inNodeParentParentIndex;
			}
			else {
				int inNodeSiblingLeftIndex = nodeLeftIndices[inNodeSiblingIndex];
				int inNodeSiblingRightIndex = nodeRightIndices[inNodeSiblingIndex];
				nodeLeftIndices[0] = inNodeSiblingLeftIndex;
				nodeRightIndices[0] = inNodeSiblingRightIndex;
				nodeParentIndices[inNodeSiblingLeftIndex] = 0;
				nodeParentIndices[inNodeSiblingRightIndex] = 0;
				inNodeParentIndex = inNodeSiblingIndex;
			}

			// Insert.
			int outNodeIndex = outNodeIndices[inNodeIndex];
			if (outNodeIndex != 0) {
				int outNodeParentIndex = nodeParentIndices[outNodeIndex];
				if (nodeLeftIndices[outNodeParentIndex] == outNodeIndex)
					nodeLeftIndices[outNodeParentIndex] = inNodeParentIndex;
				else
					nodeRightIndices[outNodeParentIndex] = inNodeParentIndex;
				nodeParentIndices[inNodeParentIndex] = outNodeParentIndex;
				nodeLeftIndices[inNodeParentIndex] = outNodeIndex;
				nodeRightIndices[inNodeParentIndex] = inNodeIndex;
				nodeParentIndices[outNodeIndex] = inNodeParentIndex;
				nodeParentIndices[inNodeIndex] = inNodeParentIndex;
			}
			else {
				int outNodeLeftIndex = nodeLeftIndices[0];
				int outNodeRightIndex = nodeRightIndices[0];
				nodeLeftIndices[inNodeParentIndex] = outNodeLeftIndex;
				nodeRightIndices[inNodeParentIndex] = outNodeRightIndex;
				nodeParentIndices[outNodeLeftIndex] = inNodeParentIndex;
				nodeParentIndices[outNodeRightIndex] = inNodeParentIndex;
				nodeParentIndices[inNodeParentIndex] = 0;
				nodeParentIndices[inNodeIndex] = 0;
				nodeLeftIndices[0] = inNodeIndex;
				nodeRightIndices[0] = inNodeParentIndex;
			}

			// Increment inserted nodes.
			atomicAdd(&insertedNodes, 1);


		}

	}

}

extern "C" __global__ void computeSizes(
	const int threads,
	const int numberOfNodes,
	int * termCounters,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Number of interior nodes.
	const int numberOfInteriorNodes = numberOfNodes >> 1;

	for (int leafIndex = threadIndex + numberOfInteriorNodes; leafIndex < numberOfNodes; leafIndex += threads) {

		// Node index.
		int nodeIndex = nodeParentIndices[leafIndex];

		// Go up to the root.
		while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

			// Node.
			int nodeLeftIndex = nodeLeftIndices[nodeIndex];
			int nodeRightIndex = nodeRightIndices[nodeIndex];

			// Size.
			int nodeLeftSize = nodeSizes[nodeLeftIndex];
			int nodeRightSize = nodeSizes[nodeRightIndex];
			nodeSizes[nodeIndex] = nodeLeftSize + nodeRightSize;

			// Root.
			if (nodeIndex == 0) break;

			// Go to the parent.
			nodeIndex = nodeParentIndices[nodeIndex];

		}

	}

}

extern "C" __global__ void computeCost(
	const int numberOfNodes,
	const int numberOfTriangles,
	const float sceneBoxArea,
	const float ct,
	const float ci,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax
	) {

	// Thread index.
	const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Cost.
	float _cost = 0.0f;

	if (nodeIndex < numberOfNodes) {

		// Node box and area.
		AABB nodeBox = AABB(nodeBoxesMin[nodeIndex], nodeBoxesMax[nodeIndex]);
		float P = nodeBox.area() / sceneBoxArea;

		// Leaf.
		if (nodeIndex >= numberOfTriangles - 1) {
			_cost += ci * P;
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
