[C. George and Job **k个同长子段最大和**](#George_and_Job)

[多个状态线性dp  Make_The_Fence_Great_Again](#Make_The_Fence_Great_Again)

[求个数大于总数一般的数](#Ignatius_and_the_Princess_IV)

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
