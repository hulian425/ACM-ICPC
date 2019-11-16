#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;

#define MAXN 100
struct ArcNode
{
	int adjvex;	// 有向边的另一个邻接点序号
	ArcNode* nextarc;	// 指向下一个边节点的指针
};

struct VNode
{
	int data;	// 顶点信息
	ArcNode* head1;	// 出边表的表头指针
	ArcNode* head2;		// 入边表的表头指针
};

struct LGraph
{
	VNode vertexs[MAXN];	// 顶点数组
	int vexnum, arcnum;		// 顶点数和边(弧)数
};

LGraph lg;	// 图(邻接表存储)

void CreateLG()	// 采用邻接表存储表示构造有向图G
{
	int i = 0;
	ArcNode* pi;
	int v1, v2;
	lg.vexnum = lg.arcnum = 0;

	scanf("%d %d", &lg.vexnum, &lg.arcnum);
	for (i = 0; i < lg.arcnum; i++)
	{
		scanf("%d%d", &v1, &v2);
		v1--; v2--;
		pi = new ArcNode;
		pi->adjvex = v2;
		pi->nextarc = lg.vertexs[v1].head1;
		lg.vertexs[v1].head1 = pi;
		pi = new ArcNode;
		pi->adjvex = v1;
		pi->nextarc = lg.vertexs[v2].head2;
		lg.vertexs[v2].head2 = pi;
	}
}
