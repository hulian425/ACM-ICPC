#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
using namespace std;

struct point
{
	int x;
	int y;
};
point p0;
stack<point> S, T;
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
	for (int i = n - 1; i >= 2; i--)
	{
		T.push(points[i]);
	}
	S.push(points[0]);
	S.push(points[1]);

	while (!T.empty())
	{
		point tmp = S.top();
		S.pop();
		int flag = ToLeft(S.top(), tmp, T.top());
		if ( flag > 0)
		{
			S.push(tmp);
			S.push(T.top());
			T.pop();
		}
		else
		{
			T.pop();
		}
	}
}

int main()
{
	point points[] = { {0, 3}, {1, 1}, {2, 2}, {4, 4},
					  {0, 0}, {1, 2}, {3, 1}, {3, 3} };
	int n = sizeof(points) / sizeof(points[0]);
	GrahamScan(points, n);
	while (!S.empty())
	{
		printf("x = %d, y = %d\n", S.top().x, S.top().y);
		S.pop();
	}
	
	return 0;
}
