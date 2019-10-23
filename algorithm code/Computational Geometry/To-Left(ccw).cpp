#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

struct Point
{
	double x;
	double y;
};
/*three points A (start point), O (center point) and B (end point) then you can compute
c=(Ax−Ox)(By−Oy)−(Ay−Oy)(Bx−Ox).

If c>0
the arc is counterclockwise, if c<0 it is clockwise, if c=0 then A, O and B are aligned.*/
double area(Point A, Point B, Point C)
{
	return (A.x * (B.y - C.y)) + B.x * (C.y - A.y) + C.x * (A.y - B.y);
}

bool ToLeft(Point p, Point q, Point s)
{
	return area(p, q, s) > 0;
}

int main()
{
	// test
	Point p = { 0, 0 };
	Point q = { 3, 3 };
	// Point s = { -1, -5.3 };
	Point s = { 0, 0 };
	if (ToLeft(p, q, s)) // p to q;
		printf("To left\n");
	else
		printf("To right\n"); // if point is lies in the edge also return false;
	return 0;
}
