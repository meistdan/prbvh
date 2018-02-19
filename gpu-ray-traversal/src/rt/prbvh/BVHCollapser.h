/**
* \file		BVHCollapser.h
* \author	Daniel Meister
* \date		2017/01/22
* \brief	BVHCollapser class header file.
*/

#ifndef _BVH_COLLAPSER_H_
#define _BVH_COLLAPSER_H_

#include "cuda/CudaBVH.hpp"
#include "gpu/CudaCompiler.hpp"

#define BFS_LAYOUT 1

namespace FW {

class BVHCollapser {

private:

	CudaCompiler compiler;

	Buffer nodeCosts;
	Buffer nodeStates;
	Buffer nodeOffsets;

	Buffer leafIndices;
	Buffer triangleOffsets;

	Buffer nodeIndices;

	float ci;
	float ct;

	float computeNodeStatesAdaptive(
		int numberOfTriangles,
		Buffer & nodeSizes,
		Buffer & nodeParentIndices,
		Buffer & nodeLeftIndices,
		Buffer & nodeRightIndices,
		Buffer & nodeBoxesMin,
		Buffer & nodeBoxesMax,
		Buffer & termCounters
	);

	float computeNodeStates(
		int numberOfTriangles,
		int maxLeafSize,
		Buffer & nodeSizes,
		Buffer & nodeParentIndices,
		Buffer & termCounters
	);

	float computeLeafIndices(
		int numberOfTriangles,
		Buffer & nodeParents
	);

	float invalidateCollapsedNodes(
		int numberOfTriangles,
		Buffer & nodeSizes,
		Buffer & nodeParentIndices,
		Buffer & termCounters
	);

	float computeNodeOffsetsBFS(
		int numberOfTriangles,
		Buffer & nodeLeftIndices,
		Buffer & nodeRightIndices
	);

	float computeNodeOffsets(
		int numberOfTriangles
	);

	float computeTriangleOffsets(
		int numberOfTriangles,
		Buffer & nodeSizes,
		Buffer & triangleOffsets
	);

	float compact(
		int numberOfTriangles,
		Buffer & nodeSizes,
		Buffer & nodeParentIndices,
		Buffer & nodeLeftIndices,
		Buffer & nodeRightIndices,
		Buffer & nodeBoxesMin,
		Buffer & nodeBoxesMax,
		CudaBVH & bvh
	);

	float reorderTriangles(
		int numberOfTriangles,
		Buffer & trinagleIndices,
		CudaBVH & bvh
	);

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
	);

public:

	BVHCollapser(void);
	~BVHCollapser(void);

	float collapseAdaptive(
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
	);

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
	);

	float getCi(void);
	void setCi(float ci);
	float getCt(void);
	void setCt(float ct);

};

};

#endif /* _BVH_COLLAPSER_H_ */
