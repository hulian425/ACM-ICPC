
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

                            
