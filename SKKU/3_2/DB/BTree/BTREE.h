#ifndef BTREE_H_
#define BTREE_H_
#include <stdbool.h>

typedef struct BTreeNode {
	int *keys;  // An array of keys
	struct BTreeNode *P; // An array of parent pointer
	struct BTreeNode **C; // An array of child pointers
	int n;     // Current number of keys
	bool leaf; // Is true when node is leaf. Otherwise false
} BTreeNode;
typedef struct
{
	struct BTreeNode * newP;
	struct BTreeNode* newChildL;
	struct BTreeNode* newChildR;
}SplitResult;

BTreeNode *root; // Pointer to root node	
int t;  // Minimum degree
int max_child, min_child, max_key, min_key;


void traverse(); // A function to traverse all nodes in a subtree rooted with this node
BTreeNode* search(int k); // function to search a key in this tree
void insertElement(int k); // The main function that inserts a new key in this B-Tree
void removeElement(int k); // The main function that removes a new key in thie B-Tree
void BTreeInit(int _t); // Initializes tree as empty
SplitResult* splitNode(BTreeNode* node);
void copyNode(BTreeNode* dst, BTreeNode* src, int startIdx, int endIdx);
void freeNode(BTreeNode* node);
BTreeNode* _createNode(bool _leaf);
void _insert(BTreeNode* present, int k);
void _add_key_to_leaf(BTreeNode* leaf, int k);
void _add_node_to_node(BTreeNode* dst, BTreeNode* src);
void _balancing(BTreeNode* present);
void _balancingAfterDel(BTreeNode* present); 


BTreeNode* _splitChild(BTreeNode* present);
void _traverse(BTreeNode* present);
BTreeNode* _search(BTreeNode* present, int k);
BTreeNode* _search_for_insertion(BTreeNode* present, int k);
void _remove(BTreeNode* present, int k);


void _borrowFromRight(BTreeNode* present, int idx);
void _borrowFromLeft(BTreeNode* present, int idx);
BTreeNode* _merge(BTreeNode* present);

int _getLevel(BTreeNode* present);
void _getNumberOfNodes(BTreeNode* present, int* numNodes, int level);
void _mappingNodes(BTreeNode* present, BTreeNode ***nodePtr, int* numNodes, int level);
void printTree();

#endif /* BTREE_H_ */