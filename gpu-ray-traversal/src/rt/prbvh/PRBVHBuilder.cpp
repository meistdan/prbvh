/**
 * \file	PRBVHBuilder.cpp
 * \author	Daniel Meister
 * \date	2018/01/22
 * \brief	PRBVHBuilder class source file.
 */

#include "PRBVHBuilder.h"
#include "BVHUtil.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace FW;

void PRBVHBuilder::configure() {
	
	const std::string configFile = "prbvh.cfg";

	std::ifstream file(configFile);
	std::string line, key, value;

	int modTmp;

	if (file.is_open()) {
		while (getline(file, line)) {
			std::istringstream ss(line);
			getline(ss, key, '=');
			getline(ss, value, '=');

			if (key == "mod") {
				std::istringstream(value) >> modTmp;
				setMod(modTmp);
			}

		}
		file.close();
	}

}

void PRBVHBuilder::allocate(int numberOfTriangles) {
	LBVHBuilder::allocate(numberOfTriangles);
	locks.resizeDiscard(sizeof(unsigned long long) * (2 * numberOfTriangles - 1));
	areaReductions.resizeDiscard(sizeof(float) * (2 * numberOfTriangles - 1));
	outNodeIndices.resizeDiscard(sizeof(int) * (2 * numberOfTriangles - 1));
#if !FAST
	paths.resizeDiscard(sizeof(Path) * (2 * numberOfTriangles - 1));
	zeroBounds.resizeDiscard(sizeof(unsigned short)* (2 * numberOfTriangles - 1));
#endif
}

float PRBVHBuilder::optimize(int numberOfTriangles) {

	// Kernel times.
	float findBestNodeTime = 0.0f;
	float findBestNodeTimeT = 0.0f;
	float lockNodesTime = 0.0f;
	float checkLocksTime = 0.0f;
	float reinsertTime = 0.0f;
	float refitTime = 0.0f;
	float computeCostTime = 0.0f;

	// Kernels.
	CudaModule * prbvhModule = prbvhCompiler.compile();
	CudaKernel findBestNodeKernel = prbvhModule->getKernel("findBestNode");
	CudaKernel lockNodesKernel = prbvhModule->getKernel("lockNodes");
	CudaKernel checkLocksKernel = prbvhModule->getKernel("checkLocks");
	CudaKernel reinsertKernel = prbvhModule->getKernel("reinsert");
	CudaKernel computeCostKernel = prbvhModule->getKernel("computeCost");
	CudaModule * module = compiler.compile();
	CudaKernel refitKernel = module->getKernel("refit");

	// Threads and blocks.
	int findBestNodeBlockThreads = BLOCK_THREADS;
	int lockNodesBlockThreads = BLOCK_THREADS;
	int checkLocksBlockThreads = BLOCK_THREADS;
	int reinsertBlockThreads = BLOCK_THREADS;
	int refitBlockThreads = BLOCK_THREADS;
	int computeCostBlockThreads = BLOCK_THREADS;

	// Scene box.
	AABB sceneBox = *(AABB*)module->getGlobal("sceneBox").getPtr();
	float sceneBoxArea = sceneBox.area();

	// Step counter.
	int steps = 0;

	// Mod.
	int modCur = mod;

	// Number of nodes.
	int numberOfNodes = 2 * numberOfTriangles - 1;

	// Number of inserted nodes.
	int insertedNodesTotal = 0;
	int insertedNodes;
	int foundNodes;

	// Costs.
	float cost = FW_F32_MAX;
	float prevCost;

	while (true) {

		// Previous cost.
		prevCost = cost;

		// Clear termination counters.
		termCounters.clear();

		// Find the best node.
		findBestNodeKernel.setParams(
			numberOfNodes,
			numberOfTriangles,
			modCur,
			steps % modCur,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			outNodeIndices,
			areaReductions,
			paths,
			zeroBounds,
			nodeBoxesMin,
			nodeBoxesMax
		);
		findBestNodeTimeT = findBestNodeKernel.launchTimed(divCeil(numberOfNodes, modCur), Vec2i(findBestNodeBlockThreads, 1));
		findBestNodeTime += findBestNodeTimeT;

		// Clear locks.
		locks.clear();

		// Lock nodes on paths.
		lockNodesKernel.setParams(
			numberOfNodes,
			modCur,
			steps % modCur,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			outNodeIndices,
			areaReductions,
			paths,
			locks,
			zeroBounds
		);
		lockNodesTime += lockNodesKernel.launchTimed(divCeil(numberOfNodes, modCur), Vec2i(lockNodesBlockThreads, 1));

		// Clear number of inserted nodes.
		prbvhModule->getGlobal("insertedNodes").clear();
		prbvhModule->getGlobal("foundNodes").clear();

		// Check locks.
		checkLocksKernel.setParams(
			numberOfNodes,
			modCur,
			steps % modCur,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			outNodeIndices,
			areaReductions,
			paths,
			locks,
			zeroBounds
			);
		checkLocksTime += checkLocksKernel.launchTimed(divCeil(numberOfNodes, modCur), Vec2i(checkLocksBlockThreads, 1));

		// Remove and insert nodes.
		reinsertKernel.setParams(
			numberOfNodes,
			modCur,
			steps % modCur,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			outNodeIndices,
			areaReductions
		);
		reinsertTime += reinsertKernel.launchTimed(divCeil(numberOfNodes, modCur), Vec2i(reinsertBlockThreads, 1));

		// Check number of inserted nodes.
		foundNodes = *(int*)prbvhModule->getGlobal("foundNodes").getPtr();
		insertedNodes = *(int*)prbvhModule->getGlobal("insertedNodes").getPtr();
		insertedNodesTotal += insertedNodes;

		// Clear termination counters.
		termCounters.clear();

		// Refit.
		refitKernel.setParams(
			numberOfTriangles,
			numberOfNodes,
			termCounters,
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			nodeBoxesMin,
			nodeBoxesMax
		);
		refitTime += refitKernel.launchTimed(numberOfTriangles, Vec2i(refitBlockThreads, 1));

		// Clear cost.
		prbvhModule->getGlobal("cost").clear();

		// Compute cost.
		computeCostKernel.setParams(
			numberOfNodes,
			numberOfTriangles,
			sceneBoxArea,
			collapser.getCt(),
			collapser.getCi(),
			nodeBoxesMin,
			nodeBoxesMax
		);
		computeCostTime += computeCostKernel.launchTimed(numberOfNodes, Vec2i(computeCostBlockThreads, 1));

		// Cost.
		cost = *(float*)prbvhModule->getGlobal("cost").getPtr();

		// Increment step counter.
		++steps;

		// Log.
		printf("<InsertionBuilder> inserted nodes %d / %d (%f), time %f, cost %f.\n", insertedNodes, foundNodes, float(insertedNodes) / float(foundNodes), findBestNodeTimeT, cost);

		// Break conditions.
		const float BETA = 0.1f;
		if ((fabs(prevCost - cost) <= BETA || insertedNodes == 0) && modCur == 1) {
			break;
		}

		// Decrease mod. parameter.
		if (fabs(prevCost - cost) <= BETA) {
			modCur = max(1, modCur - 1);
		}

	}

	// Log.
	printf("<PRBVHBuilder> BVH optimized in %d steps and %d nodes were reinserted.\n", steps, insertedNodesTotal);
	printf("<PRBVHBuilder> Nodes found in %fs.\n", findBestNodeTime);
	printf("<PRBVHBuilder> Nodes locked in %fs.\n", lockNodesTime);
	printf("<PRBVHBuilder> Locks checked in %fs.\n", checkLocksTime);
	printf("<PRBVHBuilder> Nodes removed and inserted in %fs.\n", reinsertTime);
	printf("<PRBVHBuilder> Bounding boxes refitted in %fs.\n", refitTime);
	printf("<PRBVHBuilder> Cost computed in %fs.\n", computeCostTime);

	return findBestNodeTime + lockNodesTime + checkLocksTime + reinsertTime + refitTime + computeCostTime;

}

float PRBVHBuilder::computeSizes(int numberOfTriangles) {

	// Kernel.
	CudaModule * module = prbvhCompiler.compile();
	CudaKernel kernel = module->getKernel("computeSizes");

	// Threads and blocks.
	int blockThreads = BLOCK_THREADS;
	int threads = 2 * numberOfTriangles - 1;

	// Clear termination counters.
	termCounters.clear();

	// Set params.
	kernel.setParams(
		threads,
		2 * numberOfTriangles - 1,
		termCounters,
		nodeParentIndices,
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes
	);
	
	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float PRBVHBuilder::build(CudaBVH & bvh, Scene * scene) {

	// Number of triangles.
	int numberOfTriangles = scene->getNumTriangles();

	// Allocate buffers.
	allocate(numberOfTriangles);

	// Compute scene box.
	float sceneBoxTime = computeSceneBox(scene);
	printf("<PRBVHBuilder> Scene box computed in %fs.\n", sceneBoxTime);

	// Morton codes.
	float mortonCodesTime = computeMortonCodes(scene);
	printf("<PRBVHBuilder> Morton codes (%d bits) computed in %fs.\n", mortonCodes60Bits ? 60 : 30, mortonCodesTime);

	// Sort.
	float sortTime = sortTriangles(numberOfTriangles, bvh);
	printf("<PRBVHBuilder> Triangles sorted in %fs.\n", sortTime);

	// Setup leaves.
	float setupLeavesTime = setupLeaves(numberOfTriangles);
	printf("<PRBVHBuilder> Leaves setup in %fs.\n", setupLeavesTime);

	// Construction.
	float constructTime = construct(numberOfTriangles);
	printf("<PRBVHBuilder> Topology constructed in %fs.\n", constructTime);

	// Refit.
	float refitTime = refit(numberOfTriangles);
	printf("<PRBVHBuilder> Bounding boxes refitted in %fs.\n", refitTime);

	// Optimize by insertion.
	float optimizeTime = optimize(numberOfTriangles);
	printf("<PRBVHBuilder> BVH optimized by insertion in %fs.\n", optimizeTime);

	// Compute sizes.
	float sizesTime = computeSizes(numberOfTriangles);
	printf("<PRBVHBuilder> Node sizes computed in %fs.\n", sizesTime);
	
	// Collapse.
	float collapseTime;
	if (adaptiveLeafSize)
		collapseTime = collapser.collapseAdaptive(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, termCounters, bvh);
	else
		collapseTime = collapser.collapse(numberOfTriangles, maxLeafSize, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, termCounters, bvh);
	printf("<PRBVHBuilder> BVH collapsed and converted in %fs.\n", collapseTime);

	// Woopify triangles.
	float woopTime = woopifyTriangles(bvh, scene);
	printf("<PRBVHBuilder> Triangles woopified in %fs.\n", woopTime);

	return sceneBoxTime + mortonCodesTime + sortTime + setupLeavesTime + constructTime + 
		refitTime + optimizeTime + sizesTime + collapseTime + woopTime;

}

PRBVHBuilder::PRBVHBuilder() : mod(8) {
	prbvhCompiler.setSourceFile("src/rt/prbvh/PRBVHBuilderKernels.cu");
	prbvhCompiler.addOptions("-use_fast_math");
	prbvhCompiler.include("src/rt");
	prbvhCompiler.include("src/framework");
	configure();
}

PRBVHBuilder::~PRBVHBuilder() {
}

CudaBVH * PRBVHBuilder::build(Scene * scene) {
	
	// Create BVH.
	CudaBVH * bvh = new CudaBVH(BVHLayout::BVHLayout_AOS_SOA);

	// Resize buffers.
	const int TRIANGLE_ALIGN = 4096;
	bvh->getNodeBuffer().resizeDiscard(sizeof(CudaBVHNode)* (2 * scene->getNumTriangles() - 1));
	bvh->getTriIndexBuffer().resizeDiscard(sizeof(int)* scene->getNumTriangles());
	bvh->getTriWoopBuffer().resizeDiscard((4 * sizeof(Vec4f)* scene->getNumTriangles() + TRIANGLE_ALIGN - 1) & -TRIANGLE_ALIGN);

	// Settings.
	printf("<PRBVHBuilder> Morton codes %d bits, mod %d.\n", mortonCodes60Bits ? 60 : 30, mod);
	printf("<PRBVHBuilder> Adaptive collapse %d, Max. leaf size %d.\n", int(adaptiveLeafSize), maxLeafSize);

	// Build.
	float time = build(*bvh, scene);

	// Cost.
	float cost = computeCost(*bvh);

#if VALIDATE_BVH
	// Validate.
	validate(*bvh, scene);
#endif

	// Stats.
	printf("<PRBVHBuilder> BVH built in %fs from %d triangle.\n", time, scene->getNumTriangles());
	printf("<PRBVHBuilder> %f MTriangles/s.\n", (scene->getNumTriangles() * 1.0e-3f / time), scene->getNumTriangles());
	printf("<PRBVHBuilder> BVH cost is %f.\n", cost);

	return bvh;

}

int PRBVHBuilder::getMod() {
	return mod;
}

void PRBVHBuilder::setMod(int mod) {
	if (mod >= 1 || mod <= 64) this->mod = mod;
}
