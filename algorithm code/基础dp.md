[C. George and Job **k个同长子段最大和**](#George_and_Job)

[多个状态线性dp  Make_The_Fence_Great_Again](#Make_The_Fence_Great_Again)

[求个数大于总数一般的数](#Ignatius_and_the_Princess_IV)

[Doing_Homework状态压缩dp](#Doing_Homework)

## George_and_Job

[C. George and Job](https://codeforces.com/problemset/problem/467/C)

**题意**
给你一个序列，将序列分成k个长度为m的连续子段，使各个子段的区间和的和最大

**题解**

令dp[i][j] 代表在以第i个数结尾，j个子段的区间和

那么 dp[i][j] = max(dp[i-1][j], dp[i][j-m] + sum[i] - sum[i-m])

```c++
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

## Make_The_Fence_Great_Again

[D. Make The Fence Great Again](https://codeforces.com/problemset/problem/1221/D)

**题意**

一个序列，每个位置的数`a[i]`对应一个操作消耗`b[i]`，每操作一次，这个数+1，消耗增加`b[i]`,经过一系列操作，使序列的每一个数与他相邻的数大小不一样，问最小消耗多大？

**题解**

每个数最多+2
用dp[N][3] 线性递推

```c++
case{
        int n; cin >> n;
        rep(i,1, n)
        {
            cin >> a[i];
            cin >> b[i];
            dp[i][0] = dp[i][1] = dp[i][2] = 0x3f3f3f3f3f3f3f3f;
        }
        dp[1][0] = 0, dp[1][1] = b[1], dp[1][2] = b[1]*2;
        rep(i, 2, n)                        
        {
            if (a[i] != a[i-1])
            dp[i][0] = min(dp[i-1][0], dp[i][0]);
            if (a[i] != a[i-1] + 1)
            dp[i][0] = min(dp[i-1][1], dp[i][0]);
            if (a[i]!= a[i-1] +2) dp[i][0] = min(dp[i][0], dp[i-1][2]);
            if (a[i] +1 != a[i-1]) dp[i][1] = min(dp[i-1][0]+b[i] , dp[i][1]);
            if (a[i] +1 != a[i-1] + 1) dp[i][1] = min(dp[i-1][1]+b[i] , dp[i][1]);
            if (a[i] +1 != a[i-1] + 2) dp[i][1] = min(dp[i-1][2]+b[i] , dp[i][1]);
            if (a[i] +1  +1!= a[i-1] + 2) dp[i][2] = min(dp[i-1][2]+2*b[i] , dp[i][2]);
            if (a[i] +1  +1!= a[i-1] + 1) dp[i][2] = min(dp[i-1][1]+2*b[i] , dp[i][2]);
            if (a[i] +1  +1!= a[i-1] ) dp[i][2] = min(dp[i-1][0]+2*b[i] , dp[i][2]);
        }
        cout << min(min(dp[n][1], dp[n][0]),dp[n][2]) << endl;
```

## Ignatius_and_the_Princess_IV

[https://vjudge.net/problem/HDU-1029](https://vjudge.net/problem/HDU-1029)

**题意**

求序列种个数大于(n+1)/2的数

**题解**

因为有个数大于一半，考虑每出现一个数记录值x，cnt = 1，再出现一个不同的数cnt--,再出现一个相同的数cnt++,cnt = 0时，将x变为最新出现的数，最后x的值就是个数大于个数一半的数

```c++
    int n;
    while (~scanf("%lld", &n)){
        int ans = -1;
        int cnt = 0;
        rep(i, 1, n)
        {
            int t;
            t = re;
            if (cnt == 0)
            {
                ans = t;
                cnt++;
            }
            else {
                if (ans!=t)cnt--;
                else cnt++;
            }
        }
        cout << ans << endl;
    }
```

## Doing_Homework

[Doing Homework](https://vjudge.net/problem/HDU-1074#author=634579757)

**题意**

有n个任务，每个任务有一个截止时间，超过截止时间一天，要扣一个分。
求如何安排任务，使得扣的分数最少。

**题解**

因为数据的关系，用状压dp，已完成的任务个数为阶段递推，因为输入已经给的是字典序，所以输出倒着输出就可以了

```c++
const int N = 16;
pair<string, PII> a[N];
int dp[1 << N];
int pre[1 << N];
int n;
void output(int status)
{
    if (status == 0)
        return;
    int t = 0;
    for (int i = 0; i < n; i++)
    {
        if ((status & (1 << i)) != 0 && (pre[status] & (1 << i)) == 0)
        {
            t = i;
            break;
        }
    }
    output(pre[status]);
    cout << a[t].first << endl;
}
signed main()
{
    STDIN
    case{
        cin >> n;
        rep(i ,0, n-1)  
        {
            cin >> a[i].first >> a[i].second.first>>a[i].second.second;
        }
        for (int i = 0; i < 1<< n; i++)
            dp[i]=INF1;
        dp[0] = 0;
        for (int i = 0; i < (1 << n); i++ )
        {
            for (int j = 0; j < n; j++)
            {
                if (i & (1<<j))continue;
                int s = 0;
                for (int k = 0; k< n; k++)
                {
                    if (i&(1<<k))
                    {
                        s += a[k].second.second;
                    }
                }
                s += a[j].second.second;
                if (s > a[j].second.first) s-=a[j].second.first;
                else s = 0;
                if (dp[i|(1<<j)] > s + dp[i])
                {
                    dp[i|(1<<j)]  = s + dp[i];
                    pre[i|(1<<j)] = i;
                }
            }
        }
        cout << dp[(1<<n)-1] << endl;
        output((1<<n)-1);
    }
}
```
