[C. George and Job k个同长子段最大和](#George_and_Job)

## George_and_Job

[C. George and Job](https://codeforces.com/problemset/problem/467/C)

**题意**
给你一个序列，将序列分成k个长度为m的连续子段，使各个子段的区间和的和最大

**题解**

令dp[i][j] 代表在以第i个数结尾，j个子段的区间和

那么 dp[i][j] = max(dp[i-1][j], dp[i][j-m] + sum[i] - sum[i-m])

```
  rep(i, 1, n)
  {
    a[i] = re;
    a[i] += a[i - 1];
  }
  memset(dp, 0, sizeof dp);
  dp[m][1] = a[m];  
  for (int j = 1; j <= k; j++)
  {
    for (int i = m + 1; i <= n; i++)
    {
      dp[i][j] = max(dp[i-1][j], dp[i-m][j-1] + a[i] - a[i - m]);
    }
  }
```
