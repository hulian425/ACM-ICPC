#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;



int m, n;

int xx[4] = { 0, 1, -1, 0 };
int yy[4] = { 1, 0, 0, -1 };
struct node
{
	bool water;
	bool up;
	bool down;
	bool right;
	bool left;
};

node grid[55][55];
node Creatnode(char temp)
{
	node t{ 0,0,0,0,0 };
	if (temp == 'A')
	{
		t.up = t.left = 1;
	}
	else if (temp == 'B')
	{
		t.up = t.right = 1;
	}

	else if (temp == 'C')
	{
		t.left = t.down = 1;
	}

	else if (temp == 'D')
	{
		t.right = t.down = 1;
	}
	else if (temp == 'E')
	{
		t.up = t.down = 1;
	}

	else if (temp == 'F')
	{
		t.left = t.right = 1;
	}

	else if (temp == 'G')
	{
		t.up = t.left = t.right = 1;
	}
	else if (temp == 'H')
	{
		t.left = t.up = t.down = 1;
	}
	else if (temp == 'I')
	{
		t.left = t.right = t.down = 1;
	}
	else if (temp == 'J')
	{
		t.up = t.right = t.down = 1;
	}
	else if (temp == 'K')
	{
		t.up = t.left = t.right = t.down = 1;
	}
	return t;
}

void dfs(int x, int y)
{
	grid[x][y].water = 1;
	for (int i = 0; i < 4; i++)
	{
		int dx = x + xx[i];
		int dy = y + yy[i];
		if (i == 0 && grid[x][y].right == 1 && dx < m && dx >= 0 && dy < n && dy >= 0 && grid[dx][dy].water == 0)
		{
			dfs(dx, dy);
		}

		else if (i == 1 && grid[x][y].down == 1 && dx < m && dx >= 0 && dy < n && dy >= 0 && grid[dx][dy].water == 0)
		{
			dfs(dx, dy);
		}
		else if (i == 2 && grid[x][y].up == 1 && dx < m && dx >= 0 && dy < n && dy >= 0 && grid[dx][dy].water == 0)
		{
			dfs(dx, dy);
		}

		else if (i == 3 && grid[x][y].left == 1 && dx < m && dx >= 0 && dy < n && dy >= 0 && grid[dx][dy].water == 0)
		{
			dfs(dx, dy);
		}
	}
	return;

}
int main()
{
	while (1)
	{
		
		scanf("%d%d", &m, &n);
		if (m < 0 || n < 0) break;
		int ans = 0;
		for (int i = 0; i < m; i++)
		{
			getchar();
			for (int j = 0; j < n; j++)
			{
				
				char temp = getchar();
				grid[i][j] = Creatnode(temp);
				
			}
		}

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (grid[i][j].water == 0)
				{
					ans++;
					dfs(i, j);
				}
			}
		}
		printf("%d\n", ans);
	}
}
