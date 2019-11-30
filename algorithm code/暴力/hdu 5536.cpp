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
const int MAX_N = 1003;
inline int read() {
	int s = 0, w = 1;
	char ch = getchar();
	while (ch<'0' || ch>'9') { if (ch == '-')w = -1; ch = getchar(); }
	while (ch >= '0' && ch <= '9') s = s * 10 + ch - '0', ch = getchar();
	return s * w;
}
const int INF = 1e9 + 5;

int a[MAX_N];

int solve(int a, int b, int c)
{
	return (a + b) ^ c;
}
int main()
{
    // freopen("in.txt", "r", stdin);
	int t = read();
	while (t--)
	{
		int n;
		n = read();
		for (int i = 0; i < n; i++)
		{
			a[i] = read();
		}
		int ans = -INF;
		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				for (int k = j + 1; k < n; k++)
				{
					int t = max(max(solve(a[i], a[j], a[k]), solve(a[j], a[k], a[i])),solve( a[k], a[i], a[j]));
					ans = max(t, ans);
				}
			}
		}
		cout << ans << endl;
	}

}

