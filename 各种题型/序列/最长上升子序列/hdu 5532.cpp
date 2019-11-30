#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <list>
#include<numeric>
#include <deque>
#include <vector>
using namespace std;

typedef long long ll;
const int MAX_N = 1e5 + 5;
inline int read() {
	int s = 0, w = 1;
	char ch = getchar();
	while (ch<'0' || ch>'9') { if (ch == '-')w = -1; ch = getchar(); }
	while (ch >= '0' && ch <= '9') s = s * 10 + ch - '0', ch = getchar();
	return s * w;
}
const int INF = 1e9;
int n;
int a[MAX_N];
int dp[MAX_N];


int LIS()
{
	for (int i = 0; i < n; i++)
	{
		dp[i] = INF;
	}
	for (int i = 0; i < n; i++)
	{
		*upper_bound(dp, dp + n, a[i]) = a[i];
	}
	for (int i = 0; i < n; i++)
	{
		printf("%d%c", dp[i], i == n - 1 ? '\n' : ' ');
	}
	return (lower_bound(dp, dp + n, INF) - dp);
}

int LDS()
{
	for (int i = 0; i < n; i++)
	{
		dp[i] = INF;
		
	}for (int i = n - 1; i >= 0; i--)
	{
		*upper_bound(dp, dp + n, a[i]) = a[i];
	}
	
	return (lower_bound(dp, dp + n, INF) - dp);
}



int main()
{
	freopen("in.txt", "r", stdin);
	int t = read();
	while (t--)
	{
		n = read();
		for (int i = 0; i < n; i++)
		{
			a[i] = read();
		}
		int lis = LIS();
		int lds = LDS();

		if (lis == n || lis == n - 1 || lds == n || lds == n - 1)
		{
			puts("YES");
		}
		else puts("NO");
	}

}

