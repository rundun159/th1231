#include <stdio.h>
#include <string.h>
union Data {   
    int boolean; 	//If there is no data, boolean = 0/ there is data, boolean=1
    int intData;
    float floatData;     
    char * stringData;    
};
struct Node
{
	//Informations about node
	char  * nodeType;	//which part of syntax is this node
	char  * detType;	//detail type of this node
	char  * dataType;	//int, float, 
	union Data data;	
	struct Node ** childArr;
	int childNum;
	int nodeNum;
};
struct Node * stack[10000];
struct Node * queue[10000];
int head;
int tail;
int sp;
int count;
struct Node * makeNewNode(char * nodeType,char *detType, char * dataType,union Data data);
struct Node * rootNode;
void printNode(struct Node * node);
void makechildArr(struct Node * node, int num);
void travelTree(struct Node *node,int depth);
void queue_init();
void enqueue(struct Node * node);
struct Node * dequeue();

