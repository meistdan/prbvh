/**
* \file		BVHCollapser.cpp
* \author	Daniel Meister
* \date		2017/01/23
* \brief	BVHCollapser class source file.
*/

#include "BVHCollapser.h"
#include "BVHCollapserKernels.h"
#include "BVHUtil.h"
#include "CudaBVHNode.h"

using namespace FW;

float BVHCollapser::computeNodeStatesAdaptive(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices,
	Buffer & nodeBoxesMin,
	Buffer & nodeBoxesMax,
	Buffer & termCounters
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeNodeStatesAdaptive");

	// Block.
	int blockThreads = BLOCK_THREADS;

	// Resize buffers.
	nodeStates.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeCosts.resizeDiscard(sizeof(float)* (2 * numberOfTriangles - 1));

	// Clear termination counters.
	termCounters.clear();

	// Set params.
	kernel.setParams(
		numberOfTriangles,
		ci,
		ct,
		termCounters,
		nodeCosts,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		nodeStates,
		nodeBoxesMin,
		nodeBoxesMax
	);

	// Launch.
	float time = kernel.launchTimed(numberOfTriangles, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::computeNodeStates(
	int numberOfTriangles,
	int maxLeafSize,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & termCounters
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeNodeStates");

	// Block.
	int blockThreads = BLOCK_THREADS;

	// Resize buffers.
	nodeStates.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));

	// Clear termination counters.
	termCounters.clear();

	// Set params.
	kernel.setParams(
		numberOfTriangles,
		maxLeafSize,
		termCounters,
		nodeParentIndices,
		nodeSizes,
		nodeStates
	);

	// Launch.
	float time = kernel.launchTimed(numberOfTriangles, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}


float BVHCollapser::computeLeafIndices(
	int numberOfTriangles,
	Buffer & nodeParentIndices
	) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeLeafIndices");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = numberOfTriangles;

	// Resize buffer.
	leafIndices.resizeDiscard(sizeof(int)* numberOfTriangles);

	// Set params.
	kernel.setParams(
		threads,
		numberOfTriangles,
		leafIndices,
		nodeParentIndices,
		nodeStates
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::invalidateCollapsedNodes(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & termCounters
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("invalidateCollapsedNodes");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = numberOfTriangles;

	// Clear termination counters.
	termCounters.clear();

	// Set params.
	kernel.setParams(
		threads,
		numberOfTriangles,
		termCounters,
		leafIndices,
		nodeParentIndices,
		nodeSizes,
		nodeStates
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::computeNodeOffsetsBFS(
	int numberOfTriangles,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeNodeOffsetsBFS");

	// Clear prefix scan offsets.
	module->getGlobal("leafPrefixScanOffset").clear();
	*(int*)module->getGlobal("interiorPrefixScanOffset").getMutablePtr() = 1;

	// Resize buffer.
	nodeOffsets.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeIndices.resizeDiscard(sizeof(int)* (numberOfTriangles - 1));
	nodeIndices.clearRange(0, 0, sizeof(int));

	int taskOffset = 0;
	int numberOfTasks = 1;
	float time = 0.0f;

	while (numberOfTasks > 0) {

		// Threads and blocks.
		int blockThreads = BLOCK_THREADS;
		int threads = divCeilLog(numberOfTasks, LOG_WARP_THREADS) << LOG_WARP_THREADS;

		// Set params.
		kernel.setParams(
			threads,
			taskOffset,
			numberOfTasks,
			nodeLeftIndices,
			nodeRightIndices,
			nodeIndices,
			nodeOffsets,
			nodeStates
			);

		// Launch.
		time += kernel.launchTimed(threads, Vec2i(blockThreads, 1));

		// Update task offset and number of tasks.
		taskOffset += numberOfTasks;
		numberOfTasks = (*(int*)module->getGlobal("interiorPrefixScanOffset").getPtr()) - taskOffset;

	}

	// Kernel time.
	return time;

}

float BVHCollapser::computeNodeOffsets(
	int numberOfTriangles
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeNodeOffsets");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = 2 * numberOfTriangles - 1;

	// Clear prefix scan offsets.
	module->getGlobal("leafPrefixScanOffset").clear();
	*(int*)module->getGlobal("interiorPrefixScanOffset").getMutablePtr() = 1;

	// Resize buffer.
	nodeOffsets.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));

	// Set params.
	kernel.setParams(
		threads,
		2 * numberOfTriangles - 1,
		nodeOffsets,
		nodeStates
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::computeTriangleOffsets(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & triangleOffsets
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeTriangleOffsets");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = 2 * numberOfTriangles - 1;

	// Resize buffer.
	triangleOffsets.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));

	// Clear prefix scan offset.
	module->getGlobal("prefixScanOffset").clear();

	// Set params.
	kernel.setParams(
		threads,
		2 * numberOfTriangles - 1,
		nodeStates,
		nodeSizes,
		triangleOffsets
		);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::compact(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices,
	Buffer & nodeBoxesMin,
	Buffer & nodeBoxesMax,
	CudaBVH & bvh
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("compact");

	// Block.
	int blockThreads = BLOCK_THREADS;

	// Leaf offset.
	int newNumberOfInteriorNodes = *(int*)module->getGlobal("interiorPrefixScanOffset").getPtr();
	int newNumberOfLeafNodes = *(int*)module->getGlobal("leafPrefixScanOffset").getPtr();

	// Resize buffer.
	int numberOfNodes = newNumberOfInteriorNodes + newNumberOfLeafNodes;
	bvh.getNodeBuffer().resizeDiscard(numberOfNodes * sizeof(CudaBVHNode));

	// Set params.
	kernel.setParams(
		2 * numberOfTriangles - 1,
		newNumberOfInteriorNodes,
		nodeStates,
		nodeOffsets,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		triangleOffsets,
		nodeBoxesMin,
		nodeBoxesMax,
		bvh.getNodeBuffer()
	);

	// Launch.
	float time = kernel.launchTimed(2 * numberOfTriangles - 1, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::reorderTriangles(
	int numberOfTriangles,
	Buffer & trinagleIndices,
	CudaBVH & bvh
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("reorderTriangles");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = numberOfTriangles;

	// Set params.
	kernel.setParams(
		threads,
		numberOfTriangles,
		triangleOffsets,
		trinagleIndices,
		bvh.getTriIndexBuffer(),
		leafIndices
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float BVHCollapser::convert(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices,
	Buffer & nodeBoxesMin,
	Buffer & nodeBoxesMax,
	Buffer & triangleIndices,
	CudaBVH & bvh
) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("convert");

	// Copy triangle indices.
	bvh.getTriIndexBuffer() = triangleIndices;

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = 2 * numberOfTriangles - 1;

	// Set params.
	kernel.setParams(
		threads,
		2 * numberOfTriangles - 1,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		nodeBoxesMin,
		nodeBoxesMax,
		bvh.getNodeBuffer()
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

BVHCollapser::BVHCollapser() : ct(3.0f), ci(2.0f) {
	compiler.setSourceFile("src/rt/prbvh/BVHCollapserKernels.cu");
	compiler.addOptions("-use_fast_math");
	compiler.include("src/rt");
	compiler.include("src/framework");
}

BVHCollapser::~BVHCollapser() {
}

float BVHCollapser::collapseAdaptive(
	int numberOfTriangles,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices,
	Buffer & nodeBoxesMin,
	Buffer & nodeBoxesMax,
	Buffer & triangleIndices,
	Buffer & termCounters,
	CudaBVH & bvh
) {

	// Node states.
	float nodeStatesTime = computeNodeStatesAdaptive(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, termCounters);

	// Leaf indices.
	float leafIndicesTime = computeLeafIndices(numberOfTriangles, nodeParentIndices);

	// Invalidate collapsed leaves.
	float invalidateTime = invalidateCollapsedNodes(numberOfTriangles, nodeSizes, nodeParentIndices, termCounters);

	// Node offsets.
#if BFS_LAYOUT
	float nodeOffsetsTime = computeNodeOffsetsBFS(numberOfTriangles, nodeLeftIndices, nodeRightIndices);
#else
	float nodeOffsetsTime = computeNodeOffsets(numberOfTriangles);
#endif

	// Triangle offsets.
	float triangleOffsetsTime = computeTriangleOffsets(numberOfTriangles, nodeSizes, triangleOffsets);

	// Compaction.
	float compactTime = compact(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, bvh);

	// Reorder triangles.
	float reorderTrianglesTime = reorderTriangles(numberOfTriangles, triangleIndices, bvh);

	return nodeStatesTime + leafIndicesTime + invalidateTime + nodeOffsetsTime + triangleOffsetsTime + compactTime + reorderTrianglesTime;

}

float BVHCollapser::collapse(
	int numberOfTriangles,
	int maxLeafSize,
	Buffer & nodeSizes,
	Buffer & nodeParentIndices,
	Buffer & nodeLeftIndices,
	Buffer & nodeRightIndices,
	Buffer & nodeBoxesMin,
	Buffer & nodeBoxesMax,
	Buffer & triangleIndices,
	Buffer & termCounters,
	CudaBVH & bvh
) {

	// Max. leaf size 1 => Just convert.
	if (maxLeafSize <= 1)
		return convert(numberOfTriangles, nodeSizes, nodeParentIndices, nodeLeftIndices,
		nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);

	// Node states.
	float nodeStatesTime = computeNodeStates(numberOfTriangles, maxLeafSize, nodeSizes, nodeParentIndices, termCounters);

	// Leaf indices.
	float leafIndicesTime = computeLeafIndices(numberOfTriangles, nodeParentIndices);

	// Invalidate collapsed leaves.
	float invalidateTime = invalidateCollapsedNodes(numberOfTriangles, nodeSizes, nodeParentIndices, termCounters);

	// Node offsets.
#if BFS_LAYOUT
	float nodeOffsetsTime = computeNodeOffsetsBFS(numberOfTriangles, nodeLeftIndices, nodeRightIndices);
#else
	float nodeOffsetsTime = computeNodeOffsets(numberOfTriangles);
#endif

	// Triangle offsets.
	float triangleOffsetsTime = computeTriangleOffsets(numberOfTriangles, nodeSizes, triangleOffsets);

	// Compaction.
	float compactTime = compact(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, bvh);

	// Reorder triangles.
	float reorderTrianglesTime = reorderTriangles(numberOfTriangles, triangleIndices, bvh);

	return nodeStatesTime + leafIndicesTime + invalidateTime + nodeOffsetsTime + triangleOffsetsTime + compactTime + reorderTrianglesTime;

}

float BVHCollapser::getCi() {
	return ci;
}

void BVHCollapser::setCi(float ci) {
	this->ci = ci;
}

float BVHCollapser::getCt() {
	return ct;
}

void BVHCollapser::setCt(float ct) {
	this->ct = ct;
}
