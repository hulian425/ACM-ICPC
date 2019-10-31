#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <string>
using namespace std;

/*Definition. A binary search tree is a binary tree that is either empty or satisfies the following
conditions:
• All values occurring in the left subtree are smaller than that of the root.
• All values occurring in the right subtree are larger than that of the root.
• The left and right subtrees are themselves binary search trees.*/

struct node {
	int val;
	node* lch, * rch;
};
node* insert(int v, node* p)
{
	if (p == NULL)
	{
		node* q = new node;
		q->val = v;
		q->lch = NULL;
		q->rch = NULL;
		return q;
	}
	else
	{
		if (v < p->val)
			p->lch = insert(v, p->lch);
		else p->rch = insert(v, p->rch);
		return p;
	}
}

// 查找数值x
bool find(node* p, int v)
{
	if (p == NULL) return false;
	else if (v == p->val) return false;
	else if (v < p->val) return find(p->lch, v);
	else return find(p->rch, v);
}

// 删除数值
node* remove(node* p, int v)
{
	if (p == NULL) return NULL;
	else if (v < p->val) p->lch = remove(p->lch, v);
	else if (v > p->val) p->rch = remove(p->rch, v);
	else if (p->lch == NULL)
	{
		node* q = p->rch;
		delete p;
		return q;
	}
	else if (p->lch->rch == NULL)
	{
		node* q = p->lch;
		q->rch = p->rch;
		delete p;
		return q;
	}
	else
	{
		node* q;
		for (q = p->lch; q->rch->rch != NULL; q = q->rch);
		node* r = q->rch; // 左子树最大的那个节点
		q->rch = r->lch;
		r->lch = p->lch;
		r->rch = p->rch;
		delete p;
		return r;
	}
}
