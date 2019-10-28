#include <algorithm>
#include <iostream>
using namespace std;
// Finding LTL
// 三点一线未解决
struct Point
{
	int x;
	int y;
	bool extreme;
	int succ;
};
int LTL(Point S[], int n) // n > 2
{
	int ltl = 0; // the lowest-thrn-leftmost point
	for (int k = 1; k < n; k++) // test all points
	{
		if (S[k].y < S[ltl].y || (S[k].y == S[ltl].y && S[k].x < S[ltl].x))
			ltl = k;
	}
	return ltl;
}
int area2(Point A, Point B, Point C)
{
	return (A.x * (B.y - C.y)) + B.x * (C.y - A.y) + C.x * (A.y - B.y);
}

int ToLeft(Point p, Point q, Point s)
{
	int ans = area2(p, q, s);
	if (ans > 0) return 1;
	else if (ans == 0) return 0;
	else return -1;
}
void Jarvis(Point S[], int n)
{
	for (int k = 0; k < n; k++)
	{
		S[k].extreme = false;
		S[k].succ = -1;
	}

	int ltl = LTL(S, n); int k = ltl;

	do { // start with LTL
		S[k].extreme = true; int s = -1;
		for (int t = 0; t < n; t++) // check
			if (t != k && t != s && (s == -1) || ToLeft(S[k], S[s], S[t]) == -1) // candidate t
			{
				s = t; // update s if t lies right to  ks		
			}
		S[k].succ = s; k = s; // new EE(p, q)identified
	} while (ltl != k); // quit when LTL reached

	printf("x = %d, y = %d\n", S[ltl].x, S[ltl].y);
	for (int i = S[ltl].succ; i != ltl; i = S[i].succ)
		printf("x = %d, y = %d\n", S[i].x, S[i].y);
}
int main()
{
	Point points[] = { {0, 3}, {2, 2}, {1, 1}, {2, 1},
					  {3, 0}, {0, 0}, {3, 3} };
	int n = sizeof(points) / sizeof(points[0]);
	Jarvis(points, n);
	

	return 0;
}
