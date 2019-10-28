#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

struct point
{
	int x;
	int y;
};
point p0;
void LTL(point points[], int n)
{
	int ltl = 0;
	p0.x = points[0].x;
	p0.y = points[0].y;
	for (int i = 0; i < n; i++)
	{
		if (points[i].x < p0.x)
		{
			p0.x = points[i].x;
			p0.y = points[i].y;
			ltl = i;
		}
		else if (points[i].x == p0.x)
		{
			if (points[i].y < p0.y)
			{
				p0.x = points[i].x;
				p0.y = points[i].y;
				ltl = i;
			}
		}
	}
	swap(points[0], points[ltl]);
}

int area2(point p, point q, point s)
{
	return p.x * (q.y - s.y) + q.x * (s.y - p.y) + s.x * (p.y - q.y);
}

int ToLeft(point p, point q, point s)
{
	int ans = area2(p, q, s);
	if (ans == 0) return 0;
	else if (ans > 0) return 1;
	else return -1;
}
// sort cmp
// 极角排序
bool cmp(point a, point b)
{
	if (ToLeft(p0, a, b) > 0) return true;
	else if (ToLeft(p0, a, b) == 0)
	{
		return a.x > b.x;
	}
	return false;
}
void GrahamScan(point points[], int n)
{
	// ltl
	LTL(points, n);
	sort(points + 1, points + n, cmp);

}

int main()
{
	point points[] = { {0, 3}, {1, 1}, {2, 2}, {4, 4},
					  {0, 0}, {1, 2}, {3, 1}, {3, 3} };
	int n = sizeof(points) / sizeof(points[0]);
	GrahamScan(points, n);
	for (int i = 0; i < n; i++)
	{
		printf("x = %d, y = %d\n", points[i].x, points[i].y);
	}
	return 0;
}
