## Brackets
[poj 2955](http://poj.org/problem?id=2955)
求序列的子序列的最大括号匹配数量
初始化长度为2的
```c++
if (check(s[l-1], s[r-1]))
  dp[l][r] = max(dp[l][r], dp[l+1][r-1] + 2);
for (int k = l; k <= r - 1; k++)
  {
    dp[l][r] = max(dp[l][r], dp[l][k] + dp[k+1][r]);
  }
```

