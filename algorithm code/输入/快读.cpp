// getchar()比scanf要快,故为了加快读入,可以用getchar()代替scanf

inline int read(){
   int s=0,w=1;
   char ch=getchar();
   while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
   while(ch>='0'&&ch<='9') s=s*10+ch-'0',ch=getchar();
   return s*w;
}

// 如何在c++中提升cin和cout的速度
int main(){
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	
}
