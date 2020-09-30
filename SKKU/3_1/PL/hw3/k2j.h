#ifndef __K2J_H__
#define __K2J_H__
#include <stdio.h>
#include <string.h>
#include "Kotlin.h"
/*
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
*/
int travelListOfList(char * compare);
int travelIfList(char * compare);
struct Node * k2jTree(struct Node * rootNode);
struct Node * k2jFunction(struct Node * funDefNode);
void travelForJava(struct Node * nowNode);
struct Node * k2jBody(struct Node ** funArr,int num); 	//list of function is passed
struct Node * changedefPar(struct Node * kdefPar);
void printToken(struct Node * jrootNode);
struct Node * change_ret_statement(struct Node * knode);
struct Node * change_expr_statement(struct Node * knode);
struct Node * change_branch_statement(struct Node * knode);
struct Node * change_for_statement(struct Node *knode);
struct Node * change_when_statement(struct Node * knode);
struct Node * change_when_sentence(struct Node *knode);
struct Node * change_whenBodySet(struct Node * knode);
struct Node * change_if_statement(struct Node * knode);
struct Node * change_fun_statement(struct Node * knode);
struct Node * change_val_statement(struct Node * knode);
struct Node * change_var_statement(struct Node * knode);
struct Node * change_def_statement(struct Node * knode);
struct Node * change_statement(struct Node * knode);
struct Node * change_body(struct Node * knode);
int funNum=0;
int isinFun=0;
struct Node ** kotlinFunArr;
void printOutAsJava(struct Node * jnode);
#endif 
