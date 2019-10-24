#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

/*
A utility function to calculate area of triangle formed by (x1, y1),
(x2, y2) and (x3,y3)
*/


struct Point
{
	double x;
	double y;
};

/*Calculate area of the given triangle, i.e., area of the triangle ABC in the above diagram. 
Area A = [ x1(y2 – y3) + x2(y3 – y1) + x3(y1-y2)]/2*/

double area(Point A, Point B, Point C)
{
	return abs((A.x * (B.y - C.y)) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
}

bool isInside(Point A, Point B, Point C, Point p)
{
	double a = area(A, B, C);
	double a1 = area(A, B, p);
	double a2 = area(B, C, p);
	double a3 = area(A, C, p);
	return (a == a1 + a2 + a3);
}

int main()
{
	/*Let us check whether the point P(10, 15) lies inside the triangle fprmed by
	A(0,0),B(20,0)and C(10,30)*/
	Point A = { 0, 0 };
	Point B = { 20, 0 };
	Point C = { 10, 30 };
	Point p = { 10,  15};
	if (isInside(A, B, C, p)) // if point is on the edge of triangle return true;
	{
		printf("Inside\n");
	}
	else printf("Not Inside\n");
	return 0;
}

/* 
// another method
struct Point
{
	double x;
	double y;
};
double area(Point A, Point B, Point C)
{
	return (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
}
bool ToLeft(Point p, Point q, Point s)
{
	return area(p, q, s) > 0;
}  
int main()
{
	Point A = { 0, 0 };
	Point B = { 20, 0 };
	Point C = { 10, 30 };
	Point p = { -10, 15 };

	if (ToLeft(A, B, p) && ToLeft(B, C, p) && ToLeft(C, A, p))
	{
		cout << "p is in traingle" << endl;
	}
	else
		cout << "p is not in  vctraingle" << endl;
	return 0;
} */
