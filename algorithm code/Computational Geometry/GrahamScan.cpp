// GrahamScan algorithm O(nlogn)
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
using namespace std;

const int N = 5e5 + 5;
int n;
struct point
{
	int x;
	int y;
};
point points[N];
point p0;
stack<point> S, T;
vector<point> ans;
void LTL(point points[], int n)
{
	int ltl = 0;
	p0.x = points[0].x;
	p0.y = points[0].y;
	for (int i = 0; i < n; i++)
	{
		if (points[i].y < p0.y)
		{
			p0.x = points[i].x;
			p0.y = points[i].y;
			ltl = i;
		}
		else if (points[i].y == p0.y)
		{
			if (points[i].x < p0.x)
			{
				p0.x = points[i].x;
				p0.y = points[i].y;
				ltl = i;
			}
		}
	}
	//printf("ltl : x = %d, y = %d\n", points[ltl].x, points[ltl].y);
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
	/*for (int i = 0; i < n; i++)
	{
		printf("x = %d, y = %d\n", points[i].x, points[i].y);

	}
	cout << "paixu" << endl;*/
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
		if ( flag >= 0)
		{
			S.push(tmp);
			S.push(T.top());
			T.pop();
		}
		else if ( flag < 0)
		{
			continue;
		}


	}
	while (!S.empty())
	{
		//printf("x = %d, y = %d\n", S.top().x, S.top().y);
		ans.push_back(S.top()); S.pop();
	}
	return;
}

