#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <queue>
using namespace std;

const int N = 15;
int n;
struct point
{
	int x, y;
};

point ps[2 * N];

int par[N];
int rank1[N];

void init(int n)
{
	for (int i = 1; i <= n; i++)
	{
		par[i] = i;
		rank1[i] = 0;
	}
}

int find(int x)
{
	if (par[x] == x)
	{
		return x;
	}
	else return par[x] = find(par[x]);
}

void unite(int x, int y)
{
	x = find(x);
	y = find(y);
	if (x == y) return;
	if (rank1[x] < rank1[y])
	{
		par[x] = y;
	}
	else
	{
		par[y] = x;
		if (rank1[x] == rank1[y]) rank1[x]++;
	}
}

int area2(point a, point b, point c)
{
	return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
}

int ccw(point a, point b, point c)
{
	int flag = area2(a, b, c);
	if (flag > 0) return 1;
	else if (flag < 0) return -1;
	else return 0;
}

bool intersect(int a, int b)
{
	
	int j1 = ccw(ps[a], ps[a + n], ps[b]);
	int j2 = ccw(ps[a], ps[a + n], ps[b + n]);
	int g1 = ccw(ps[b], ps[b + n], ps[a]);
	int g2 = ccw(ps[b], ps[b + n], ps[a + n]);
	if ((j1 * j2 < 0 && g1 * g2 < 0))
		return 1;
	if (j1 == 0 && ps[b].x <= max(ps[a].x, ps[a + n].x) && ps[b].x >= min(ps[a].x, ps[a + n].x) && ps[b].y <= max(ps[a].y, ps[a + n].y) && ps[b].y >= min(ps[a].y, ps[a + n].y))
	{
		return 1;
	}
	if (j2 == 0 && ps[b + n].x <= max(ps[a].x, ps[a + n].x) && ps[b + n].x >= min(ps[a].x, ps[a + n].x) && ps[b +n].y <= max(ps[a].y, ps[a + n].y) && ps[b+n].y >= min(ps[a].y, ps[a + n].y))
	{
		return 1;
	}
	if (g1 == 0 && ps[b].x <= max(ps[b].x, ps[b + n].x) && ps[a].x >= min(ps[b].x, ps[b + n].x) && ps[a].y <= max(ps[b].y, ps[b + n].y) && ps[a + n].y >= min(ps[b].y, ps[b + n].y))
	{
		return 1;
	}
	if (g2 == 0 && ps[a + n].x <= max(ps[b].x, ps[b + n].x) && ps[a + n].x >= min(ps[b].x, ps[b + n].x) && ps[a + n].y <= max(ps[b].y, ps[b + n].y) && ps[a + n].y >= min(ps[b].y, ps[b + n].y))
	{
		return 1;
	}
	else return 0;
}

int main()
{
	while (~scanf("%d", &n) && n)
	{
		init(n);
		
		for (int i = 1; i <= n; i++)
		{
			scanf("%d %d %d %d", &ps[i].x, &ps[i].y, &ps[i + n].x, &ps[i + n].y);
		}

		for (int i = 1; i < n; i++)
		{
			for (int j = i+1; j <= n; j++)
			{
				if (intersect(i, j) == 1)
				{
					//printf("x = %d y = %d, x = %d y = %d\n", ps[i].x, ps[i].y, ps[i + n].x, ps[i + n].y);
					//printf("x = %d y = %d, x = %d y = %d\n\n", ps[j].x, ps[j].y, ps[j + n].x, ps[j + n].y);

					unite(i, j);
				}
			}
		}
		int a, b;
		while (scanf("%d %d", &a, &b) && a && b)
		{
			if (find(a) == find(b))
			{
				
				printf("CONNECTED\n");
			}
			else puts("NOT CONNECTED");
		}
	}
	return 0;

}
