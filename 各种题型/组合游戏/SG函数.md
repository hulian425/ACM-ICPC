```c++
#include<iostream>
#include<algorithm>
#include<cstdio>
#include<cstring>
using namespace std;

int index[11] = {1,2,4,8,16,32,128,256,512,1024};

int n;
int SG[1002] = {0};
int hash[1000000] = {0};
void getSG(int n)
{
    int i,j;
    memset(SG,0,sizeof(SG));
    for(i=1;i<=n;i++)
    {
        memset(hash,0,sizeof(hash));
        for(j=0;index[j]<=i;j++)
            hash[SG[i-index[j]]]=1;
        for(j=0;j<=n;j++)    //求mes{}中未出现的最小的非负整数
        {
            if(hash[j]==0)
            {
                SG[i]=j;
                break;
            }
        }
    }
}

int main()
{
    /*int i;
    for (i = 1; i < 1002; i++)
    {
        int flag = 0;
        for (int j = 0; j < 11; j++)
        {

            if (index[j] > i)
                break;
            if (!SG[i - index[j]])
            {
                flag = 1;
                break;
            }
        }
         if (flag) SG[i] = 1;
        else SG[i] = 0;
    }*/

    getSG(1000);
    for (int i = 0; i < 1002; i++)
    {
        printf("i = %d %d\n", i, SG[i]);
    }

    /* while (scanf("%d", &n) == 1)
    {
        if (vic[n]) printf("KiKi\n");
        else printf("Cici\n");
    }
    return 0; */

    //int f[N],sg[N],hash[N];


}

```
