[一、 Brackets](#Brackets)

[二、Halloween Costumes](#Costumes)

[三、Multiplication  Puzzle](#Multiplication)

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
## Costumes

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

## Multiplication
[Multiplication Puzzle](https://vjudge.net/problem/POJ-1651)

**题意**

每次删掉一个数`a[i]`, `sum += a[i] * a[i-1] * a[i+1]`, 求最小sum。

**题解**

枚举k， k作为区间的中点 `dp[l][r] = min(dp[l][k]+dp[k][r]+a[k]*a[l]*a[r], dp[l][r])`;


## 加分二叉树

**题意**

给你一个二叉树的中序遍历，subtree的左子树的加分 × subtree的右子树的加分 ＋ subtree的根的分数 

若某个子树为空，规定其加分为1。叶子的加分就是叶节点本身的分数，不考虑它的空子树。

试求一棵符合中序遍历为（1,2,3,…,n）且加分最高的二叉树tree。


**题解**

令f[l][r]表示中序遍历为w[l~r]的所有二叉树的分值最大值

状态计算：f[i][j] = max(f[i][k - 1] * f[k + 1][j] + w[k])，
即将f[i][j]表示的二叉树集合按根节点分类，则根节点在 k 时的最大得分即为 f[i][k - 1] * f[k + 1][j] + w[k]，
则f[i][j]即为遍历 k 所取到的最大值

```c++
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 50;

int n;
int w[N];
unsigned f[N][N];
int root[N][N];

void dfs(int l, int r)
{
    if (l > r) return;

    int k = root[l][r];
    printf("%d ", k);
    dfs(l, k - 1);
    dfs(k + 1, r);
}

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    for (int len = 1; len <= n; len ++ )
        for (int l = 1; l + len - 1 <= n; l ++ )
        {
            int r = l + len - 1;

            for (int k = l; k <= r; k ++ )
            {
                int left = k == l ? 1 : f[l][k - 1];
                int right = k == r ? 1 : f[k + 1][r];
                int score = left * right + w[k];
                if (l == r) score = w[k];
                if (f[l][r] < score)
                {
                    f[l][r] = score;
                    root[l][r] = k;
                }
            }
        }

    printf("%d\n", f[1][n]);
    dfs(1, n);
    puts("");

    return 0;
}

```
