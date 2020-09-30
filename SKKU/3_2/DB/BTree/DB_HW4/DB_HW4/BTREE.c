#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "BTREE.h"
void BTreeInit(int _t)
{
	root = NULL;  t = _t - 1;
	max_key = t;
	max_child = t + 1;
	min_child = (max_child / 2);
	if (max_child % 2 != 0)
		min_child++;
	min_key = min_child - 1;
}


void traverse()
{
	if (root != NULL) _traverse(root);
}


BTreeNode* search(int k)
{
	return (root == NULL) ? NULL : _search(root, k);
}


BTreeNode* _createNode(bool _leaf)
{
	BTreeNode* newNode = (BTreeNode*)malloc(sizeof(BTreeNode));
	int i;

	// Copy the given minimum degree and leaf property
	newNode->leaf = _leaf;

	// Allocate memory for maximum number of possible keys
	// and child pointers
	newNode->keys = (int*)malloc(sizeof(int) * (t + 1));
	newNode->C = (BTreeNode**)malloc(sizeof(BTreeNode*) * (t + 2));

	// Initialize child
	for (i = 0; i < t + 2; i++)
		newNode->C[i] = NULL;

	// Initialize the number of keys as 0
	newNode->n = 0;

	// Initialize parent
	newNode->P = NULL;

	return newNode;
}




void _traverse(BTreeNode* present)
{
	// There are n keys and n+1 children, travers through n keys and first n children
	int i;
	for (i = 0; i < present->n; i++)
	{
		// If this is not leaf, then before printing key[i],
		// traverse the subtree rooted with child C[i].
		if (present->leaf == false)
			_traverse(present->C[i]);

		printf(" ");
		printf("%d", present->keys[i]);
	}

	// Print the subtree rooted with last child
	if (present->leaf == false)
		_traverse(present->C[i]);
}


BTreeNode* _search(BTreeNode* present, int k)
{
	// Find the first key greater than or equal to k
	int i = 0;
	while (i < present->n && k > present->keys[i])
		i++;

	// If the found key is equal to k, return this node
	if (present->keys[i] == k)
		return present;

	// If key is not found here and this is a leaf node
	if (present->leaf == true)
		return NULL;

	// Go to the appropriate child
	return _search(present->C[i], k);
}

BTreeNode* _search_for_insertion(BTreeNode* present, int k)
{
	// If  this is a leaf node
	if (present->leaf == true)
		return present;
	// Find the first key greater than or equal to k
	int i = 0;
	while (i < present->n && k > present->keys[i])
		i++;
	// Go to the appropriate child
	return _search_for_insertion(present->C[i], k);
}


void insertElement(int k)
{
	// Find key in this tree, and If there is a key, it prints error message.
	if (search(k) != NULL)
	{
		printf("The tree already has %d \n", k);
		return;
	}

	// If tree is empty
	if (root == NULL)
	{
		// Allocate memory for root
		root = _createNode(true);
		root->P = NULL; // Init parent
		root->keys[0] = k;  // Insert key
		root->n = 1;  // Update number of keys in root
	}
	else // If tree is not empty
		_insert(root, k);
}


void _insert(BTreeNode* present, int k)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	//find the leaf node to be added 
	//and rebalance from the leaf
	BTreeNode* leafNode = _search_for_insertion(present, k);
	_add_key_to_leaf(leafNode, k);
	_balancing(leafNode);
	//-------------------------------------------------------------------------------------------------------
}
void _add_key_to_leaf(BTreeNode* leaf, int k)
{
	// Find the first key greater than or equal to k
	int i = 0;
	while (i < leaf->n && k > leaf->keys[i])
		i++;
	for (int j = (leaf->n); j >= (i + 1); j --)
		leaf->keys[j] = leaf->keys[j - 1];
	leaf->keys[i] = k;
	leaf->n++;
	return;
}
//suppose the src node is the new parent node from splitNode
void _add_node_to_node(BTreeNode* dst, BTreeNode* src)
{
	int i = 0;
	while (i < dst->n && src->keys[0] > dst->keys[i])
		i++;
	for (int j = (dst->n); j >= (i + 1); j -- )
		dst->keys[j] = dst->keys[j - 1];
	dst->keys[i] = src->keys[0];
	for (int j = (dst->n); j >= (i + 1); j -- )
		dst->C[j + 1] = dst->C[j];
	dst->C[i] = src->C[0];
	dst->C[i + 1] = src->C[1];
	src->C[0]->P = dst;
	src->C[1]->P = dst;
	dst->n++;
	return;
}
void _balancing(BTreeNode* present)
{
	BTreeNode* parent;

	if (present->n <= t)
	{
		return;
	}
	else if (present->P == NULL)
	{
		root = _splitChild(present);
		return;
	}
	else
	{
		parent = _splitChild(present);
		_balancing(parent);
	}
}
SplitResult* splitNode(BTreeNode* node)
{
	SplitResult* ret = (SplitResult*)malloc(sizeof(SplitResult));
	//if this node is full, split this node into three nodes, including one parent and two child nodes.
	int parent_idx = node->n / 2;
	if (node->n % 2 == 0)
		parent_idx--;
	int parent_key = node->keys[parent_idx];
	ret->newP = _createNode(false);
	ret->newP->keys[ret->newP->n++] = node->keys[parent_idx];
	ret->newP->P = node->P;
	ret->newChildL = _createNode(node->leaf);
	ret->newChildR = _createNode(node->leaf);
	copyNode(ret->newChildL, node, 0, parent_idx - 1);
	copyNode(ret->newChildR, node, parent_idx + 1, max_key);
	ret->newChildL->P = ret->newP;
	ret->newChildR->P = ret->newP;
	ret->newP->C[0] = ret->newChildL;
	ret->newP->C[1] = ret->newChildR;
	freeNode(node);
	return ret;
}
void freeNode(BTreeNode* node)
{
	free(node->keys);
	free(node->C);
}

void copyNode(BTreeNode* dst, BTreeNode* src, int startIdx, int endIdx)
{
	dst->n = (endIdx - startIdx + 1);
	for (int i = startIdx; i <= endIdx; i++)
	{
		dst->keys[i - startIdx] = src->keys[i];
		dst->C[i - startIdx] = src->C[i];
		if (dst->C[i - startIdx] != NULL)
			dst->C[i - startIdx]->P = dst;
	}
	dst->C[dst->n] = src->C[endIdx + 1];
	if(dst->C[dst->n]!=NULL)
		dst->C[dst->n]->P = dst;
	return;
}

BTreeNode* _splitChild(BTreeNode* present)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	BTreeNode* org_P = present->P;
	SplitResult* ret = splitNode(present);
	if (org_P == NULL)
		return ret->newP;
	else
	{
		_add_node_to_node(org_P, ret->newP);
		return org_P;
	}
	//-------------------------------------------------------------------------------------------------------
}


void removeElement(int k)
{
	if (!root)
	{
		printf("The tree is empty\n");
		return;
	}

	// Call the remove function for root
	_remove(root, k);

	// If the root node has 0 keys, make its first child as the new root
	//  if it has a child, otherwise set root as NULL
	if (root->n == 0)
	{
		BTreeNode* tmp = root;
		if (root->leaf)
		{
			root = NULL;
		}
		else
		{
			root = root->C[0];
			root->P = NULL;
		}

		// Free the old root
		free(tmp);
	}
	return;
}

void _remove(BTreeNode* present, int k)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	BTreeNode * containNode=_search(present, k);
	if (containNode->leaf)
	{
		removeFromLeaf(containNode, k);
		return _balancingAfterDel(containNode);
	}
	else
	{
		// if the containNode is not a leaf node.
		int i = 0;
		int containNode_org_n = containNode->n;
		while (i < containNode->n && k > containNode->keys[i])
			i++;
		BTreeNode* leaf = findLeftMost(containNode, i);
		containNode->keys[i] = leaf->keys[leaf->n - 1];
		leaf->keys[leaf->n - 1] = k;
		removeFromLeaf(leaf, k);
		return _balancingAfterDel(leaf);
	}
	//-------------------------------------------------------------------------------------------------------
}

void _balancingAfterDel(BTreeNode* present) // repair After Delete
{
	int minKeys = (t + 2) / 2 - 1;
	BTreeNode* parent;
	BTreeNode* next;
	int parentIndex = 0;

	if (present->n < minKeys)
	{
		if (present->P == NULL)
		{
			if (present->n == 0)
			{
				root = present->C[0];
				if (root != NULL)
					root->P = NULL;
			}
		}
		else
		{
			parent = present->P;
			for (parentIndex = 0; parent->C[parentIndex] != present; parentIndex++);
			if (parentIndex > 0 && parent->C[parentIndex - 1]->n > minKeys)
			{
				_borrowFromLeft(present, parentIndex);
			}
			else if (parentIndex < parent->n && parent->C[parentIndex + 1]->n >minKeys)
			{
				_borrowFromRight(present, parentIndex);
			}
			else if (parentIndex == 0)
			{
				// Merge with right sibling
				next = _merge(present,parentIndex);
				_balancingAfterDel(next->P);
			}
			else
			{
				// Merge with left sibling
				next = _merge(present,parentIndex);
				_balancingAfterDel(next->P);

			}

		}
	}
}


void _borrowFromRight(BTreeNode* present, int parentIdx)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	BTreeNode* parentNode;
	BTreeNode* rightSibling;
	parentNode = present->P;
	rightSibling = parentNode->C[parentIdx + 1];

	present->keys[present->n] = parentNode->keys[parentIdx];
	present->C[present->n + 1] = rightSibling->C[0];
	if(present->C[present->n+1]!=NULL)
		present->C[present->n + 1]->P = present;
	present->n++;
	parentNode->keys[parentIdx] = rightSibling->keys[0];

	for (int i = 1; i < rightSibling->n; i++)
	{
		rightSibling->keys[i - 1] = rightSibling->keys[i];
		rightSibling->C[i - 1] = rightSibling->C[i];
	}
	rightSibling->keys[rightSibling->n - 1] = 0;
	rightSibling->C[rightSibling->n-1] = rightSibling->C[rightSibling->n];
	rightSibling->C[rightSibling->n] = NULL;
	rightSibling->n--;
	return;
	//-------------------------------------------------------------------------------------------------------
}


void _borrowFromLeft(BTreeNode* present, int parentIdx)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	BTreeNode* parentNode;
	BTreeNode* leftSibling;
	parentNode = present->P;
	leftSibling = parentNode->C[parentIdx - 1];
	for (int i = present->n; i > 0; i--)
	{
		present->keys[i] = present->keys[i - 1];
		present->C[i+1] = present->C[i];
	}
	present->C[1] = present->C[0];
	present->keys[0] = parentNode->keys[parentIdx - 1];
	present->C[0] = leftSibling->C[leftSibling->n];
	//leaf node였을때 present의 child없는거 예외처리 하기 
	if(present->C[0]!=NULL)
		present->C[0]->P = present;
	parentNode->keys[parentIdx - 1] = leftSibling->keys[leftSibling->n - 1];
	present->n++;

	leftSibling->C[leftSibling->n] = NULL;
	leftSibling->keys[leftSibling->n-1] = 0;
	leftSibling->n--;
	return;
	//-------------------------------------------------------------------------------------------------------
}


BTreeNode* _merge(BTreeNode* present,int parentIdx)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	if (parentIdx == 0)
	{
		//merge with right sibling
		BTreeNode* parentNode = present->P, * rightSibling=parentNode->C[parentIdx+1];
		int present_org_n = present->n;
		int rightSibling_org_n = rightSibling->n;
		int parent_org_n = parentNode->n;
		present->keys[present_org_n] = parentNode->keys[parentIdx];
		int i;
		for (i = 0; i < rightSibling_org_n; i++)
		{
			present->keys[i + present_org_n + 1] = rightSibling->keys[i];
			present->C[i + present_org_n + 1] = rightSibling->C[i];
			if (present->C[i + present_org_n + 1] != NULL)
				present->C[i + present_org_n + 1]->P = present;
		}
		present->C[i + present_org_n + 1] = rightSibling->C[i];
		if (present->C[i + present_org_n + 1] != NULL)
			present->C[i + present_org_n + 1]->P = present;
		present->n += 1 + rightSibling_org_n;

		int parent_right_num = parent_org_n - (parentIdx + 1);
		for (i = 0; i < parent_right_num; i++)
		{
			parentNode->keys[i + parentIdx] = parentNode->keys[i + parentIdx + 1];
			parentNode->C[i + parentIdx+1] = parentNode->C[i + parentIdx + 2];
		}
		parentNode->keys[i + parentIdx] = 0;
		parentNode->C[i + parentIdx + 1] = NULL;

		parentNode->n -= 1;
		freeNode(rightSibling);
	}
	else
	{
		//merge with left sibling.
		BTreeNode* parentNode = present->P, * leftSibling = parentNode->C[parentIdx - 1];
		int present_org_n = present->n;
		int leftSibling_org_n = leftSibling->n;
		int parent_org_n = parentNode->n;
		int present_result_n = present_org_n + leftSibling_org_n + 1;

		int i;
		for (i = 0; i < present_org_n; i++)
		{
			present->keys[leftSibling_org_n + 1 + i] = present->keys[i];
			present->C[leftSibling_org_n + 2 + i] = present->C[i+1];
		}
		present->C[leftSibling_org_n + 1] = present->C[0];

		for (i = 0; i < leftSibling_org_n; i++)
		{
			present->keys[i] = leftSibling->keys[i];
			present->C[i] = leftSibling->C[i];
			if (present->C[i] != NULL)
				present->C[i]->P = present;
		}
		present->C[i] = leftSibling->C[i];
		if (present->C[i] != NULL)
			present->C[i]->P = present;
		present->keys[leftSibling_org_n] = parentNode->keys[parentIdx - 1];
		present->n = present_result_n;
		parentNode->C[parentIdx - 1] = present;
		for (i = parentIdx; i < parent_org_n; i++)
		{
			parentNode->keys[i - 1] = parentNode->keys[i];
			parentNode->C[i] = parentNode->C[i+1];
		}
		parentNode->keys[i-1] = 0;
		parentNode->C[i] = NULL;
		parentNode->n-=1;
		freeNode(leftSibling);
	}

	return present;
	//-------------------------------------------------------------------------------------------------------
}


int _getLevel(BTreeNode* present)
{
	int i;
	int maxLevel = 0;
	int temp;
	if (present == NULL) return maxLevel;
	if (present->leaf == true)
		return maxLevel + 1;

	for (i = 0; i < present->n + 1; i++)
	{
		temp = _getLevel(present->C[i]);

		if (temp > maxLevel)
			maxLevel = temp;
	}

	return maxLevel + 1;
}

void _getNumberOfNodes(BTreeNode* present, int* numNodes, int level)
{
	int i;
	if (present == NULL) return;

	if (present->leaf == false)
	{
		for (i = 0; i < present->n + 1; i++)
			_getNumberOfNodes(present->C[i], numNodes, level + 1);
	}
	numNodes[level] += 1;
}

void _mappingNodes(BTreeNode* present, BTreeNode*** nodePtr, int* numNodes, int level)
{
	int i;
	if (present == NULL) return;

	if (present->leaf == false)
	{
		for (i = 0; i < present->n + 1; i++)
			_mappingNodes(present->C[i], nodePtr, numNodes, level + 1);
	}

	nodePtr[level][numNodes[level]] = present;
	numNodes[level] += 1;
}
void print_TH2(BTreeNode* node)
{
	if (node != NULL)
	{
		printf("node key :");
		for (int i = 0; i < node->n; i++)
		{
			printf("%d ", node->keys[i]);
		}
	}
	printf("\n");
}
void print_TH(BTreeNode* node,int level)
{
	if (node != NULL)
	{
		printf("level : %d node key num : %d ", level, node->n);
		printf("keys : ");
		for (int i = 0; i < node->n; i++)
		{
			printf("%d ", node->keys[i]);
		}
		printf("\n");
		printf("parent : ");
		print_TH2(node->P);
		for (int i = 0; i <= node->n; i++)
		{
			print_TH(node->C[i], level + 1);
		}
	}
	else
		return;
}

void printTree()
{
	int level;
	int* numNodes;
	int i, j, k;
	level = _getLevel(root);
	numNodes = (int*)malloc(sizeof(int) * (level));
	memset(numNodes, 0, level * sizeof(int));

	_getNumberOfNodes(root, numNodes, 0);

	BTreeNode*** nodePtr;
	nodePtr = (BTreeNode***)malloc(sizeof(BTreeNode**) * level);
	for (i = 0; i < level; i++) {
		nodePtr[i] = (BTreeNode**)malloc(sizeof(BTreeNode*) * numNodes[i]);
	}

	memset(numNodes, 0, level * sizeof(int));
	_mappingNodes(root, nodePtr, numNodes, 0);

	for (i = 0; i < level; i++) {
		for (j = 0; j < numNodes[i]; j++) {
			printf("[");

			for (k = 0; k < nodePtr[i][j]->n; k++)
				printf("%d ", nodePtr[i][j]->keys[k]);

			printf("] ");
		}
		printf("\n");
	}

	for (i = 0; i < level; i++) {
		free(nodePtr[i]);
	}
	free(nodePtr);
}
BTreeNode* findLeftMost(BTreeNode* present, int idx)
{
	BTreeNode* ret = present->C[idx];
	while (!ret->leaf)
	{
		ret = ret->C[ret->n];
	}
	return ret;
}
void removeFromLeaf(BTreeNode* leaf, int k)
{
	BTreeNode* containNode = leaf;
	//if the containNode is a leaf node.
	int i = 0;
	int containNode_org_n = containNode->n;
	while (i < containNode->n && k > containNode->keys[i])
		i++;
	int j;
	for (j = i + 1; j < containNode_org_n; j++)
	{
		containNode->keys[j - 1] = containNode->keys[j];
	}
	containNode->keys[j] = 0;
	containNode->n--;
	return;
}
