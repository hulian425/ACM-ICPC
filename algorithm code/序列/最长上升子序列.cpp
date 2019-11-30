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
	
	return (lower_bound(dp, dp + n, INF) - dp);
}
