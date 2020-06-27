[toc]

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
## Halloween Costumes
[Halloween Costumes](https://vjudge.net/problem/LightOJ-1422)

**题意**

穿衣服，脱衣服，脱过的衣服不能再穿,问最少要准备几件衣服？

转移 + 区间dp的固定写法

a[i]和a[j]相等的时候  

dp[i][j]=dp[i][j-1]
```c++
if (a[l] == a[r])
  dp[l][r] = min(dp[l][r], dp[l][r-1]);
  rep(k, l, r-1)
   {                 
      dp[l][r] = min(dp[l][r], dp[l][k] + dp[k+1][r]);
   }
```

