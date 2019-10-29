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

long long  area2(point p, point q, point s)
{
	return p.x * (q.y - s.y) + q.x * (s.y - p.y) + s.x * (p.y - q.y);
}

int ToLeft(point p, point q, point s)
{
	long long ans = area2(p, q, s);
	//printf("area2 = %d\n", ans);
	if (ans == 0) return 0;
	else if (ans > 0) return 1;
	else return -1;
}
// sort cmp
bool cmp(point a, point b)
{
	return a.x < b.x;
}
void GrahamScan(point points[], int n)
{
	sort(points, points + n, cmp);
	/*for (int i = 0; i < n; i++)
	{
		printf("x = %d, y = %d\n", points[i].x, points[i].y);

	}
	cout << "paixu" << endl;*/
	
	// 凸包的上部分
	for (int i = 0; i < n - 1; i++)
	{
		T.push(points[i]);
	}
	point t = { points[n - 1].x, -1000000000 };
	S.push(t);
	S.push(points[n - 1]);
	// printf("x = %d, y = %d\n", S.top().x, S.top().y);
	while (!T.empty())
	{
		point tmp = S.top();
		S.pop();
		int flag = ToLeft(S.top(), tmp, T.top());
		if (flag >= 0)
		{
			S.push(tmp);
			S.push(T.top());
			T.pop();
		}
		else if (flag < 0)
		{
			continue;
		}


	}
	while (!S.empty())
	{
		//printf("x = %d, y = %d\n", S.top().x, S.top().y);
		ans.push_back(S.top()); S.pop();
	}
	ans.pop_back();

	// 凸包的下部分
	for (int i = n - 1; i > 0; i--) T.push(points[i]);
	t.x = points[0].x; t.y = 1000000;
	S.push(t); S.push(points[0]);
	while (!T.empty())
	{
		point tmp = S.top();
		S.pop();
		
		/*printf("x = %d, y = %d\n", S.top().x, S.top().y);
		printf("x = %d, y = %d\n", tmp.x, tmp.y);
		printf("x = %d, y = %d\n", T.top().x, T.top().y);*/
		int flag = ToLeft(S.top(), tmp, T.top());
		//printf("flag = %d\n", flag);
		if (flag >= 0)
		{
			S.push(tmp);
			S.push(T.top());
			T.pop();
		}
		else if (flag < 0)
		{
			continue;
		}


	}
	S.pop();
	while (!S.empty())
	{
		T.push(S.top()); S.pop();
	}
	T.pop(); T.pop();
	while (!T.empty())
	{
		ans.push_back(T.top()); T.pop();
	}

	/*for (int i = 0; i < ans.size(); i++)
	{
		printf("x = %d, y = %d\n", ans[i].x, ans[i].y);
	}*/
	return;
}

int dist(point a, point b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
void solve()
{
	GrahamScan(points, n);
	int res = 0;
	for (int i = 0; i < ans.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			res = max(res, dist(ans[i], ans[j]));
		}
	}
	printf("%d\n", res);
}
int main()
{
	scanf("%d", &n);
	for (int i = 0; i < n; i++)
	{
		scanf("%d %d", &points[i].x, &points[i].y);
	}
	if (n > 2)
		solve();
	else printf("%d\n", dist(points[0], points[1]));

	//int n = sizeof(points) / sizeof(points[0]);

	return 0;
}
