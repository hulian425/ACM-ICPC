// c++ program to insert a node in AVL tree
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <string>
using namespace std;

// An AVL tree node
class Node
{
public:
	int key;
	Node* left;
	Node* right;
	int height;
};

// A utility function to get maximum
// of two intergers
int max(int a, int b)
{
	return a > b ? a : b;
}

int height(Node* N)
{
	if (N == NULL)
		return 0;
	return N->height;
}

Node* newNode(int key)
{
	Node* node = new Node();
	node->key = key;
	node->left = NULL;
	node->right = NULL;
	node->height = 1; 
	return node;
}

Node* rightRotation(Node* y)
{
	Node* x = y->left;
	Node* T2 = x->right;
	x->right = y;
	y->left = T2;
	y->height = 1 + max(height(y->left), height(y->right));
	x->height = 1 + max(height(x->left), height(x->right));
	return  x;
}
Node* leftRotation(Node* x)
{
	Node* y = x->right;
	Node* T2 = y->left;
	y->left = x;
	x->right = T2;
	x->height = 1 + max(height(x->left), height(x->right));
	y->height = 1 + max(height(y->left), height(y->right));
	return y;
}

int getBalance(Node* N)
{
	if (N == NULL)
		return 0;
	return height(N->left) - height(N->right);

}
int gerBalance(Node* N)
{
	if (N == NULL)
		return 0;
	return height(N->left) - height(N->right);
}
Node* insert(Node* node, int key)
{
	if (node == NULL)
		return newNode(key);
	if (key < node->key)
		node->left = insert(node->left, key);
	else if (key > node->key)
		node->right = insert(node->right, key);
	else return node;

	node->height = 1 + max(height(node->left) , height(node->right));
	int balance = (getBalance(node));
	if (balance > 1 && key < node->left->key)
		return rightRotation(node);
	if (balance > 1 && key > node->left->key)
	{
		node->left = leftRotation(node->left);
		return rightRotation(node);
	}
	if (balance < -1 && key > node->right->key)
		return leftRotation(node);
	if (balance < -1 && key < node->right->key)
	{
		node->right = rightRotation(node->right);
		return leftRotation(node);
	}

	return node;
}
/*Given a non-empty binary search tree,
return the node with minimum key value
found in that tree. Note that the entire tree does not need to be searched*/
Node* minValueNode(Node* node)
{
	Node* current = node;
	/* loop dowen to find the leftmost leaf */
	while (current->left != NULL)
		current = current->left;
	return current;
}

// Recursive function to delete a node
// with given key from subtree with
// given root. It returns root of the
// modified subtree

Node* deleteNode(Node* root, int key)
{
	// STEP 1: perform standard BST delete
	if (root == NULL)
		return root;

	// If the key to be delete is smaller 

	// than the root's key, then it lies
	// in left subtree
	if (key < root->key)
		root->left = deleteNode(root->left, key);
	else if (key > root->key)
		root->right = deleteNode(root->right, key);

	// if key is same as root's key, then
	// This is the node to be deleted
	else
	{
		// node with only one child or no child
		if ((root->left == NULL) || (root->right == NULL))
		{
			Node* temp = root->left ? root->left : root->right;
			if (temp == NULL)
			{
				temp = root;
				root = NULL;
			}
			else
				*root = *temp;
			free(temp);
		}
		else
			{
			// node with two children: Get the inorder
			// successor (smallest in the right subtree)
			Node* temp = minValueNode(root->right);

			// Copy the inorder successor's
			// data to this node
			root->key = temp->key;
			root->right = deleteNode(root->right, temp->key);

			}
	}

	// If the tree had only one node
	// then return
	if (root == NULL)
		return root;
	root->height = 1 + max(height(root->left), height(root->right));
	int balance = getBalance(root);
	if (balance > 1 && getBalance(root->left) >= 0)
		return rightRotation(root);
	if (balance > 1 && getBalance(root->left) < 0)
	{
		root->left = leftRotation(root->left);
		return rightRotation(root);
	}

	if (balance < -1 && getBalance(root->right) <= 0)
		return leftRotation(root);
	if (balance < -1 && getBalance(root->right) > 0)
	{
		root->right = rightRotation(root->right);
		return leftRotation(root);
	}
	return root;
}

// A utility function to print preorder
// traversal of the tree
// The function also prints height
// of every node;

void preOrder(Node* root)
{
	if (root != NULL)
	{
		cout << root->key << " ";
		preOrder(root->left);
		preOrder(root->right);
	}
}

// Driver Code 
int main()
{
	Node* root = NULL;

	/* Constructing tree given in
	the above figure */
	root = insert(root, 9);
	root = insert(root, 5);
	root = insert(root, 10);
	root = insert(root, 0);
	root = insert(root, 6);
	root = insert(root, 11);
	root = insert(root, -1);
	root = insert(root, 1);
	root = insert(root, 2);

	/* The constructed AVL Tree would be
			9
		/ \
		1 10
		/ \ \
	0 5 11
	/ / \
	-1 2 6
	*/

	cout << "Preorder traversal of the "
		"constructed AVL tree is \n";
	preOrder(root);

	root = deleteNode(root, 10);

	/* The AVL Tree after deletion of 10
			1
		/ \
		0 9
		/ / \
	-1 5     11
		/ \
		2 6
	*/

	cout << "\nPreorder traversal after"
		<< " deletion of 10 \n";
	preOrder(root);

	return 0;
}


