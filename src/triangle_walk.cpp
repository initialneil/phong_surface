/* Walk on triangle mesh.
*  https://arxiv.org/abs/2007.04940
*  https://kdocs.cn/l/ciGn4GHqHj8f
*  All rights reserved. Prometheus 2023.
*  Contributor(s): Neil Z. Shao.
*/
#include "triangle_walk.h"

using namespace std;

namespace prometheus
{
	template <typename T1, typename T2>
	static bool isKeyInMap(const std::map<T1, T2>& map, T1 key)
	{
		return (map.find(key) != map.end());
	}

	static bool isBaryInside(Eigen::Vector3f bary, float tol = 1e-5)
	{
		if (bary[0] >= -tol && bary[0] <= 1 + tol &&
			bary[1] >= -tol && bary[1] <= 1 + tol &&
			bary[2] >= -tol && bary[2] <= 1 + tol)
			return true;
		return false;
	}

	// https://math.stackexchange.com/a/3996095
	// line p1-p2 intersect with line p3-p4
	static void calcLineIntersectBarycentric(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3, Eigen::Vector3f p4,
		Eigen::Vector2f& t12, Eigen::Vector3f& intersect)
	{
		float u1 = p1[0], v1 = p1[1], w1 = p1[2];
		float u2 = p2[0], v2 = p2[1], w2 = p2[2];
		float u3 = p3[0], v3 = p3[1], w3 = p3[2];
		float u4 = p4[0], v4 = p4[1], w4 = p4[2];

		t12[0] = (u1 * (v4 - v3) + u3 * (v1 - v4) + u4 * (v3 - v1)) / 
			((u1 - u2) * (v4 - v3) - (u4 - u3) * (v1 - v2));

		t12[1] = (u1 * (v2 - v3) + u2 * (v3 - v1) + u3 * (v1 - v2)) /
			((u1 - u2) * (v4 - v3) - (u4 - u3) * (v1 - v2));

		intersect[0] = u1 + t12[0] * (u2 - u1);
		intersect[1] = v1 + t12[0] * (v2 - v1);
		intersect[2] = w1 + t12[0] * (w2 - w1);
	}

	static Eigen::Vector2f readAB(Eigen::Vector3f abc)
	{
		return Eigen::Vector2f(abc[0], abc[1]);
	}

	static int findCrossingEdge(Eigen::Vector3f p_bary, Eigen::Vector3f q_bary)
	{
		// find intersect edge
		for (int j = 0; j < 3; ++j) {
			Eigen::Vector3f edge_v0(0, 0, 0), edge_v1(0, 0, 0);
			edge_v0[j] = 1;
			edge_v1[(j + 1) % 3] = 1;

			Eigen::Vector2f t12;
			Eigen::Vector3f intersect;
			calcLineIntersectBarycentric(edge_v0, edge_v1, p_bary, q_bary,
				t12, intersect);

			// crossing and not on the start
			if (t12[0] >= 0.0 && t12[0] <= 1.0 && t12[1] > 1e-5 && t12[1] <= 1.0) {
				return j;
			}
		}

		return -1;
	}

	static int findOnEdgeIndex(Eigen::Vector3f p_bary)
	{
		for (int j = 0; j < 3; ++j) {
			if (fabsf(p_bary[j]) < 1e-5)
				return j;
		}

		return -1;
	}

	static TriangleWalk::ReorderTriangle reorderTriangle(TriangleWalk::SurfacePoint spt, int edge_i)
	{
		TriangleWalk::ReorderTriangle tri;
		tri.f_idx = spt.f_idx;
		tri.order_idxs << edge_i, (edge_i + 1) % 3, (edge_i + 2) % 3;
		return tri;
	}

	static Eigen::Vector3f reorderBarycentric(Eigen::Vector3f bary, int edge_i)
	{
		Eigen::Vector3f reorder_bary(
			bary[edge_i], bary[(edge_i + 1) % 3], bary[(edge_i + 2) % 3]
		);
		return reorder_bary;
	}

	// resolve numerical issue of edge point
	static void resetBaryOnEdge(Eigen::Vector3f& bary)
	{
		int idx = 0;
		float min_val = bary[0];
		for (int i = 1; i < 3; ++i) {
			if (fabsf(bary[i]) < fabsf(min_val)) {
				min_val = bary[i];
				idx = i;
			}
		}

		for (int i = 0; i < 3; ++i) {
			if (i == idx) {
				bary[i] = 0;
			}
			else {
				bary[i] += min_val / 2;
			}
		}
	}

	///////////////////////////////////// reorder /////////////////////////////////////

	void TriangleWalk::WalkingPoint::finalize(SurfacePoint& spt, Eigen::Vector3f& shift)
	{
		Eigen::Vector3f p;
		p[0] = this->intersect_ab[0];
		p[1] = this->intersect_ab[1];
		p[2] = 1.0f - p[0] - p[1];

		Eigen::Vector3f q;
		q[0] = p[0] + this->shift_ab[0];
		q[1] = p[1] + this->shift_ab[1];
		q[2] = 1.0f - q[0] - q[1];

		Eigen::Vector3f _shift = q - p;

		spt.f_idx = this->triangle.f_idx;
		for (int j = 0; j < 3; ++j) {
			spt.bary[this->triangle.order_idxs[j]] = p[j];
			shift[this->triangle.order_idxs[j]] = _shift[j];
		}
	}

	///////////////////////////////////// walk /////////////////////////////////////
	TriangleWalk::TriangleWalk()
	{}

	TriangleWalk::~TriangleWalk()
	{}

	// init mesh
	void TriangleWalk::initTriangleMesh(const Eigen::MatrixXi& F)
	{
		m_buffer.F = F;
		initTriangleNeighborV2(F);
	}

	void TriangleWalk::initTriangleNeighborV1(const Eigen::MatrixXi& F)
	{
		m_buffer.nbr_table.resize(F.rows());
		if (F.rows() == 0)
			return;

		printf("[TriangleWalk] init triangle mesh ...");
#pragma omp parallel for num_threads(8)
		for (int i0 = 0; i0 < F.rows() - 1; ++i0) {
			for (int j0 = 0; j0 < 3; ++j0) {
				// edge vert
				int cur_idx0 = F(i0, j0);
				int cur_idx1 = F(i0, (j0 + 1) % 3);

				// triangle shifted order for edge
				TriangleWalk::ReorderTriangle cur_tri;
				cur_tri.f_idx = i0;
				cur_tri.order_idxs << j0, (j0 + 1) % 3, (j0 + 2) % 3;

				// search for edge idx1-idx0 on other triangles
				for (int i1 = i0 + 1; i1 < F.rows(); ++i1) {
					for (int j1 = 0; j1 < 3; ++j1) {
						// edge vert
						int other_idx0 = F(i1, j1);
						int other_idx1 = F(i1, (j1 + 1) % 3);

						if (cur_idx1 == other_idx0 && cur_idx0 == other_idx1) {
							// triangle shifted order for edge
							TriangleWalk::ReorderTriangle other_tri;
							other_tri.f_idx = i1;
							other_tri.order_idxs << j1, (j1 + 1) % 3, (j1 + 2) % 3;

							// set mutual neighbor
							m_buffer.nbr_table[i0][j0] = other_tri;
							m_buffer.nbr_table[i1][j1] = cur_tri;
						}
					}
				}
			}
		}
		printf("[done]\n");
	}

	void TriangleWalk::initTriangleNeighborV2(const Eigen::MatrixXi& F)
	{
		m_buffer.nbr_table.resize(F.rows());
		if (F.rows() == 0)
			return;

		// build edge table
		printf("[TriangleWalk] init edge table ...");
		std::map<string, Eigen::Vector2i> edge_table;
		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				// edge vert
				int cur_idx0 = F(i, j);
				int cur_idx1 = F(i, (j + 1) % 3);

				// current edge key
				string key = to_string(cur_idx0) + "-" + to_string(cur_idx1);
				edge_table[key] = Eigen::Vector2i(i, j);
			}
		}
		printf("[done]\n");

		// build neighbor table
		printf("[TriangleWalk] init triangle neighbor ...");
		for (int i0 = 0; i0 < F.rows(); ++i0) {
			for (int j0 = 0; j0 < 3; ++j0) {
				// edge vert
				int cur_idx0 = F(i0, j0);
				int cur_idx1 = F(i0, (j0 + 1) % 3);

				// triangle shifted order for edge
				TriangleWalk::ReorderTriangle cur_tri;
				cur_tri.f_idx = i0;
				cur_tri.order_idxs << j0, (j0 + 1) % 3, (j0 + 2) % 3;

				// neighbor edge key
				string nbr_key = to_string(cur_idx1) + "-" + to_string(cur_idx0);
				if (isKeyInMap(edge_table, nbr_key)) {
					Eigen::Vector2i nbr_edge = edge_table[nbr_key];
					int i1 = nbr_edge[0];
					int j1 = nbr_edge[1];

					// triangle shifted order for edge
					TriangleWalk::ReorderTriangle other_tri;
					other_tri.f_idx = i1;
					other_tri.order_idxs << j1, (j1 + 1) % 3, (j1 + 2) % 3;

					// set mutual neighbor
					m_buffer.nbr_table[i0][j0] = other_tri;
					m_buffer.nbr_table[i1][j1] = cur_tri;
				}
			}
		}
		printf("[done]\n");
	}

	// walk on triangle mesh
	TriangleWalk::SurfacePoint TriangleWalk::walkSurfacePoint(SurfacePoint spt, Eigen::Vector3f shift)
	{
		// finish if ending point is inside triangle
		Eigen::Vector3f q_bary = spt.bary + shift;
		if (isBaryInside(q_bary)) {
			spt.bary = q_bary;
			signalWalkingPoint(spt, shift);
			return  spt;
		}

		// check start point inside triangle
		if (!isBaryInside(spt.bary)) {
			// check if starting point is on edge
			int edge_idx = findOnEdgeIndex(spt.bary);
			if (edge_idx == -1) {
				// restart from a triangle point
				spt.bary << 0.5, 0.5, 0;
				shift = q_bary - spt.bary;
				signalWalkingPoint(spt, shift);
				return walkSurfacePoint(spt, shift);
			}
		}

		// find crossing edge
		int cross_edge_idx = findCrossingEdge(spt.bary, q_bary);
		if (cross_edge_idx != -1) {
			signalWalkingPoint(spt, shift);
			return walkCrossEdge(spt, shift, cross_edge_idx);
		}

		// check if starting point is on edge
		int on_edge_idx = findOnEdgeIndex(spt.bary);
		if (on_edge_idx != -1) {
			signalWalkingPoint(spt, shift);
			return walkCrossEdge(spt, shift, on_edge_idx);
		}

		return spt;
	}

	TriangleWalk::SurfacePoint TriangleWalk::walkCrossEdge(SurfacePoint spt, Eigen::Vector3f shift, int edge_idx)
	{
		Eigen::Vector3f q_bary = spt.bary + shift;

		Eigen::Vector3f edge_v0(0, 0, 0), edge_v1(0, 0, 0);
		edge_v0[edge_idx] = 1;
		edge_v1[(edge_idx + 1) % 3] = 1;

		Eigen::Vector2f t12;
		Eigen::Vector3f intersect;
		calcLineIntersectBarycentric(edge_v0, edge_v1, spt.bary, q_bary,
			t12, intersect);

		// check neighbor exist
		ReorderTriangle nbr_tri = m_buffer.nbr_table[spt.f_idx][edge_idx];

		// no neighbor, stop on edge intersection
		if (nbr_tri.f_idx == -1) {
			spt.bary = intersect;
			return spt;
		}

		// neighbor exist, continue from edge intersection
		SurfacePoint spt_inter = spt;
		spt_inter.bary = intersect;
		Eigen::Vector3f remain_shift = q_bary - intersect;
		signalWalkingPoint(spt_inter, remain_shift);

		// the reordered cur_tri is aligned with reordered nbr_tri
		ReorderTriangle cur_tri = reorderTriangle(spt_inter, edge_idx);
		WalkingPoint cur_point;
		cur_point.triangle = cur_tri;
		cur_point.intersect_ab = readAB(reorderBarycentric(intersect, edge_idx));
		cur_point.shift_ab = readAB(reorderBarycentric(remain_shift, edge_idx));

		WalkingPoint nbr_point = walkToNeighbor(cur_point, nbr_tri);

		SurfacePoint nbr_spt;
		Eigen::Vector3f nbr_shift;
		nbr_point.finalize(nbr_spt, nbr_shift);

		// resolve numerical issue of edge point
		resetBaryOnEdge(nbr_spt.bary);

		return walkSurfacePoint(nbr_spt, nbr_shift);
	}

	TriangleWalk::WalkingPoint TriangleWalk::walkToNeighbor(WalkingPoint wpt, ReorderTriangle nbr_tri)
	{
		// current walking point transfer to neighbor triangle
		WalkingPoint nbr_point;
		nbr_point.triangle = nbr_tri;

		/* current bary of AB, transfer to BA of neighbor
		*  B A' - C'
		*  | \    |
		*  |  \   |
		*  |   \  |
		*  C -- A B'
		*/
		nbr_point.intersect_ab << wpt.intersect_ab[1], wpt.intersect_ab[0];
		nbr_point.shift_ab << -wpt.shift_ab[0], -wpt.shift_ab[1];

		return nbr_point;
	}

	// verbose: signal intermediate walking point
	void TriangleWalk::signalWalkingPoint(SurfacePoint spt, Eigen::Vector3f shift)
	{
		if (m_callback_walking_spt) {
			m_callback_walking_spt(spt, shift);
		}
	}

}
