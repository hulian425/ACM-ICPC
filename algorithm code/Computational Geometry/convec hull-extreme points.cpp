#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

// O(n^4) extreme point
struct Point
{
	int x;
	int y;
	bool extreme;
};
int area2(Point p, Point q, Point s)
{
	return p.x * q.y - p.y * q.x + q.x * s.y - q.y * s.x + s.x * p.y - s.y * p.x;
}
int ToLeft(Point p, Point q, Point s)
{
	int dir =  area2(p, q, s);
	if (dir == 0)
		return 0;
	else if (dir > 0)
		return 1;
	else 
		return -1;
}  

bool InTriangle(Point p, Point q, Point r, Point s)
{
	int pqLeft = ToLeft(p, q, s);
	int qrLeft = ToLeft(q, r, s);
	int rpLeft = ToLeft(r, p, s);

	return (pqLeft == qrLeft && qrLeft == rpLeft && pqLeft != 0);
}
void extremePoint(Point S[], int n)
{
	for (int s = 0; s < n; s++)
		S[s].extreme = true;
	for (int p = 0; p < n; p++)
		for (int q = p + 1; q < n; q++)
			for (int r = q + 1; r < n; r++)
				for (int s = 0; s < n; s++)
				{
					if (s == p || s == q || s == r || S[s].extreme == false)
					{
						continue;
					}
					else if (InTriangle(S[p], S[q], S[r], S[s]))
					{
						S[s].extreme = false;
					}
				}
}

int main()
{
	Point S[] = { {0, 3,0}, {2,2,0}, {1, 1,0}, {2, 1,0},
					  {3, 0,0}, {0, 0,0}, {3, 3,0} };
	int n = sizeof(S) / sizeof(S[0]);
	extremePoint(S, n);
	for (int i = 0; i < n; i++)
	{
		if (S[i].extreme == true)
		{
			printf("x = %d, y = %d\n", S[i].x, S[i].y);
		}
	}
	return 0;
}
