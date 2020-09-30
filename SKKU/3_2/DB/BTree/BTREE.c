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
	newNode->C = (BTreeNode**)malloc(sizeof(BTreeNode*)*(t + 2));

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
	// Find the first key greater than or equal to k
	int i = 0;
	while (i < present->n && k > present->keys[i])
		i++;

	// If key is not found here and this is a leaf node
	if (present->leaf == true)
		return present;

	// Go to the appropriate child
	return _search(present->C[i], k);
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
	for (int j = (leaf->n); j >= (i + 1); j - )
		leaf->keys[j] = leaf->keys[j - 1];
	leaf->keys[i] = k;
	return;
}
void _add_node_to_node(BTreeNode* dst, BTreeNode* src)
{
	int i = 0;
	while (i < dst->n && src->keys[0] > dst->keys[i])
		i++;
	for (int j = (leaf->n); j >= (i + 1); j - )
		dst->keys[j] = dst->keys[j - 1];
	dst->keys[i] = k;
	for (int j = (leaf->n); j >= (i + 1); j - )
		dst->C[j+1] = dst->C[j];
	dst->C[i] = src->C[0];
	dst->C[i + 1] = src->C[1];
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
	//if this node is not full, return NULL
	if (node->n <= max_key)
		return NULL;
	SplitResult* ret = (SplitResult*)malloc(sizeof(SplitResult));
	//if this node is full, split this node into three nodes, including one parent and two child nodes.
	int parent_idx = node->n / 2;
	if (node->n % 2 == 0)
		parent_idx--;
	int parent_key = node->keys[parent_idx];
	ret->newP=_createNode(false);
	ret->newP->keys[ret->newP->n++] = node->keys[parent_idx];
	ret->newP->P = node->P;
	ret->newChildL = _createNode(node->leaf);
	ret->newChildR = _createNode(node->leaf);
	copyNode(ret->newChildL, node, 0, parent_idx - 1);
	copyNode(ret->newChildR, node, parent_idx +1, max_key);
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
	for (int i = startIdx; i < endIdx; i++)
	{
		dst->keys[i-startIdx] = src->keys[i];
		dst->C[i - startIdx] = src->C[i];
	}
	dst->C[dst->n] = src[endIdx + 1];
}

BTreeNode * _splitChild(BTreeNode* present)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	SplitResult* ret = splitNode(present);
	//if the present node is root.
	if(ret->newP->P==NULL)
		return ret->newP;
	else
	{
		_add_node_to_node(present->P, ret->newP);
		return present->P;
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
		BTreeNode *tmp = root;
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
				next = _merge(present);
				_balancingAfterDel(next->P);
			}
			else
			{
				// Merge with left sibling
				next = _merge(parent->C[parentIndex - 1]);
				_balancingAfterDel(next->P);

			}

		}
	}
}


void _borrowFromRight(BTreeNode* present, int parentIdx)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	
	
	//-------------------------------------------------------------------------------------------------------
}


void _borrowFromLeft(BTreeNode* present, int parentIdx)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	
	
	//-------------------------------------------------------------------------------------------------------
}


BTreeNode* _merge(BTreeNode* present)
{
	//-------------------------------------------------------------------------------------------------------
	//Write your code.
	
	
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

void _mappingNodes(BTreeNode* present, BTreeNode ***nodePtr, int* numNodes, int level)
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


void printTree()
{
	int level;
	int *numNodes;
	int i, j, k;

	level = _getLevel(root);
	numNodes = (int *)malloc(sizeof(int) * (level));
	memset(numNodes, 0, level * sizeof(int));

	_getNumberOfNodes(root, numNodes, 0);

	BTreeNode ***nodePtr;
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