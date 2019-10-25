// convex hull EE O(n^4)

#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;

struct Point
{
	int x;
	int y;
	bool extreme;
};

int area2(Point p, Point q, Point s)
{
	return p.x * (q.y - s.y) + q.x * (s.y - p.y) + s.x * (p.y - q.y);
}
int ToLeft(Point p, Point q, Point s)
{
	int ans = area2(p, q, s);
	if (ans > 0)
		return 1;
	else if (ans == 0)
		return 0;
	else return -1;
}
void checkEdge(Point S[], int n, int p, int q) {
	bool LEmpty = true, REmpty = true;
	for (int k = 0; k < n && (LEmpty || REmpty); k++)
	{
		if (k != p && k != q)
		{
			int flag = ToLeft(S[p], S[q], S[k]);
			if (flag > 0) LEmpty = false;
			else if (flag < 0) REmpty = false;
		}
	}

	if (LEmpty || REmpty) {
		S[p].extreme = S[q].extreme = true;
	}
}
void markEE(Point S[], int n)
{
	for (int k = 0; k < n; k++)
		S[k].extreme = false;
	for (int p = 0; p < n; p++) // test
	{
		for (int q = p + 1; q < n; q++) // each
			checkEdge(S, n, p, q); // directed edge pq;
	}
}
int main()
{
	Point points[] = { {0, 3}, {2, 2}, {1, 1}, {2, 1},
					  {3, 0}, {0, 0}, {3, 3} };
	int n = sizeof(points) / sizeof(points[0]);
	markEE(points, n);

	for (int i = 0; i < n; i++)
	{
		if (points[i].extreme)
			printf("x = %d, y = %d\n", points[i].x, points[i].y);
	}
	return 0;
}
