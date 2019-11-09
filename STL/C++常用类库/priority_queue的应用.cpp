typedef struct HMT
{
	char a;
	int num;
	HMT* l;
	HMT* r;
	vector<char> code;
}*hmt;
struct cmp
{
	bool operator ()(hmt &a, hmt &b)const
	{
		return a->num > b->num;
	}
};

priority_queue<hmt,vector<hmt>, cmp> huf;
