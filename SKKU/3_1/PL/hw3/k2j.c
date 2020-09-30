#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Kotlin.h"
#include "k2j.h"
struct Node* jrootNode;
char** listof_types;
char** listof_names;
int listof_num;
char** if_types;
char** if_names;
int if_num;
struct Node* k2jTree(struct Node* rootNode)
{
	union Data defData;
	defData.boolean = 0;
	jrootNode = makeNewNode("MainClass", "class Main { Body }", "NULL", defData);
	makechildArr(jrootNode, 5);
	char* class_char;
	char* Main_char;
	char* OPENA_char;
	char* CLOSEA_char;
	class_char = strndup("class", 5);
	Main_char = strndup("Main", 4);
	OPENA_char = strndup("{", 1);
	CLOSEA_char = strndup("}", 1);


	defData.stringData = class_char;
	jrootNode->childArr[0] = makeNewNode("TOKEN", "class", "STRING", defData);
	defData.stringData = Main_char;
	jrootNode->childArr[1] = makeNewNode("TOKEN", "Main", "STRING", defData);
	defData.stringData = OPENA_char;
	jrootNode->childArr[2] = makeNewNode("TOKEN", "{", "STRING", defData);
	jrootNode->childArr[3] = k2jBody(kotlinFunArr, funNum);
	defData.stringData = CLOSEA_char;
	jrootNode->childArr[4] = makeNewNode("TOKEN", "}", "STRING", defData);


	return jrootNode;

}
struct Node* k2jBody(struct Node** funArr, int num) 	//list of function is passed
{
	union Data defData;
	defData.boolean = 0;
	struct Node* classBody = makeNewNode("Functions in Main Class", "Functions", "NULL", defData);
	makechildArr(classBody, num);
	for (int i = 0; i < num; i++)
	{
		classBody->childArr[i] = k2jFunction(funArr[i]);
	}
	return classBody;
}
struct Node* k2jFunction(struct Node* funDefNode) //in case This is not a nested function . convert kotlin fun_def_statement into java fun_def_statement 
{

	//modify Kotlin functions into java functions. 
	union Data defData;
	defData.boolean = 0;
	struct Node* javafunNode = makeNewNode("FUN_DEF_STATEMENT", "public static ret_type ID (defPar) assnStatement", "NULL", defData);
	makechildArr(javafunNode, 8);
	defData.stringData = strndup("public", strlen("public"));
	javafunNode->childArr[0] = makeNewNode("TOKEN", "public", "STRING", defData);
	defData.stringData = strndup("static", strlen("static"));
	javafunNode->childArr[1] = makeNewNode("TOKEN", "static", "STRING", defData);
	// if there is no return type specified in Kotlin function, 
	//java function should return void
	listof_types = (char**)malloc(sizeof(char*) * 100);
	listof_names = (char**)malloc(sizeof(char*) * 100);
	listof_num = 0;

	if (funDefNode->childArr[5]->childNum == 0)
	{
		defData.stringData = strndup("void", strlen("void"));
		javafunNode->childArr[2] = makeNewNode("TOKEN", "void", "STRING", defData);
	}
	else
	{
		javafunNode->childArr[2] = funDefNode->childArr[5]->childArr[1];
		if (funDefNode->childArr[5]->childArr[2]->childNum != 0)//If there is ? in return type of Kotlin code
		{
			//change primitive type to refernce type
			if (strcmp(javafunNode->childArr[2]->detType, "Int") == 0||strcmp(javafunNode->childArr[2]->detType, "INT") == 0||strcmp(javafunNode->childArr[2]->detType, "int") == 0)
			{
				javafunNode->childArr[2]->data.stringData = strndup("Integer", strlen("Integer"));
				javafunNode->childArr[2]->childArr[0]->data.stringData = strndup("Integer", strlen("Integer"));
			}
		}
		else
		{
			//change cpital character to small character 
			if (strcmp(javafunNode->childArr[2]->detType, "Int") == 0)
			{
				javafunNode->childArr[2]->childArr[0]->detType[0] -= ('I' - 'i');
				javafunNode->childArr[2]->detType[0] -= ('I' - 'i');
			}
			else if (strcmp(javafunNode->childArr[2]->detType, "Double") == 0)
			{
				javafunNode->childArr[2]->childArr[0]->detType[0] -= ('I' - 'i');
				javafunNode->childArr[2]->detType[0] -= ('I' - 'i');
			}
			else if (strcmp(javafunNode->childArr[2]->detType, "Float") == 0)
			{
				javafunNode->childArr[2]->childArr[0]->detType[0] -= ('I' - 'i');
				javafunNode->childArr[2]->detType[0] -= ('I' - 'i');
			}
			else if (strcmp(javafunNode->childArr[2]->detType, "Long") == 0)
			{
				javafunNode->childArr[2]->childArr[0]->detType[0] -= ('I' - 'i');
				javafunNode->childArr[2]->detType[0] -= ('I' - 'i');
			}
			else if (strcmp(javafunNode->childArr[2]->detType, "Boolean") == 0)
			{
				javafunNode->childArr[2]->childArr[0]->detType[0] -= ('I' - 'i');
				javafunNode->childArr[2]->detType[0] -= ('I' - 'i');
			}
			else if (strcmp(javafunNode->childArr[2]->detType, "Any") == 0)
			{
				javafunNode->childArr[2]->detType = strndup("Object", strlen("Object"));
				javafunNode->childArr[2]->childArr[0]->detType = strndup("Object", strlen("Object"));
			}
		}
	}
	javafunNode->childArr[3] = funDefNode->childArr[1];
	javafunNode->childArr[4] = funDefNode->childArr[2];
	//should convert Kotlin's function parameter to Java's funciton parameter
	javafunNode->childArr[5] = funDefNode->childArr[3];
	if (javafunNode->childArr[5]->childNum != 0) //not empty defPar
	{
		javafunNode->childArr[5] = changedefPar(funDefNode->childArr[3]);
	}
	if (strcmp(javafunNode->childArr[3]->data.stringData, "main") == 0)//corresponding function is main function of kotlin
	{
		defData.stringData = strndup("String args[]", strlen("String args[]"));
		javafunNode->childArr[5] = makeNewNode("TOKEN", "String args[]", "STRING", defData);
	}
	javafunNode->childArr[6] = funDefNode->childArr[4];
	javafunNode->childArr[7] = funDefNode->childArr[6];
	if (funDefNode->childArr[6]->childArr[1]->childNum != 0) //BodySet is not empty
		javafunNode->childArr[7]->childArr[1] = change_body(funDefNode->childArr[6]->childArr[1]->childArr[0]);
	listof_types = (char**)malloc(sizeof(char*) * 100);
	listof_names = (char**)malloc(sizeof(char*) * 100);
	listof_num = 0;

	return javafunNode;
}
//in java's function definition node 
//it has childeren
//public static types ID ( par ) assnStatement
void travelForJava(struct Node* nowNode)
{
	if (strcmp(nowNode->nodeType, "FUN_DEF_STATEMENT") == 0 && isinFun == 0)
	{
		kotlinFunArr[funNum] = nowNode;
		funNum++;
		isinFun = 1;
	}
	for (int i = 0; i < nowNode->childNum; i++)
		travelForJava(nowNode->childArr[i]);
	if (strcmp(nowNode->nodeType, "FUN_DEF_STATEMENT") == 0 && isinFun != 0)
		isinFun = 0;
}
struct Node* changedefPar(struct Node* kdefPar)
{
	struct Node* jdefPar;
	union Data defData;
	if (kdefPar->childNum == 3)
	{
		defData.stringData = strndup("jdefPar", strlen("jdefPar"));
		jdefPar = makeNewNode("jdefPar", "Types ID", "STRING", defData);
		makechildArr(jdefPar, 2);
		jdefPar->childArr[0] = kdefPar->childArr[2];
		jdefPar->childArr[1] = kdefPar->childArr[0];
	}
	else
	{
		defData.stringData = strndup("jdefPar", strlen("jdefPar"));
		jdefPar = makeNewNode("jdefPar", "Types ID COMMA jdefPar", "STRING", defData);
		makechildArr(jdefPar, 4);
		jdefPar->childArr[0] = kdefPar->childArr[2];
		jdefPar->childArr[1] = kdefPar->childArr[0];
		jdefPar->childArr[2] = kdefPar->childArr[3];
		jdefPar->childArr[3] = changedefPar(kdefPar->childArr[4]);
	}
	//modify jdefPar types.
	if (strcmp(jdefPar->childArr[0]->detType, "Int") == 0)
	{
		jdefPar->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->data.stringData[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->data.stringData[0] -= ('I' - 'i');

	}
	else if (strcmp(jdefPar->childArr[0]->detType, "Float") == 0)
	{
		jdefPar->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->data.stringData[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->data.stringData[0] -= ('I' - 'i');
	}
	else if (strcmp(jdefPar->childArr[0]->detType, "Double") == 0)
	{
		jdefPar->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->data.stringData[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->data.stringData[0] -= ('I' - 'i');
	}
	else if (strcmp(jdefPar->childArr[0]->detType, "Long") == 0)
	{
		jdefPar->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->data.stringData[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->data.stringData[0] -= ('I' - 'i');
	}
	else if (strcmp(jdefPar->childArr[0]->detType, "Boolean") == 0)
	{
		jdefPar->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->detType[0] -= ('I' - 'i');
		jdefPar->childArr[0]->data.stringData[0] -= ('I' - 'i');
		jdefPar->childArr[0]->childArr[0]->data.stringData[0] -= ('I' - 'i');
	}
	else if (strcmp(jdefPar->childArr[0]->detType, "Any") == 0)
	{
		jdefPar->childArr[0]->detType = strndup("Object", strlen("Object"));
		jdefPar->childArr[0]->childArr[0]->detType = strndup("Object", strlen("Object"));
		printf("Let's see\n");
		travelTree(jdefPar->childArr[0],0);
	}
	return jdefPar;
}
void printTOKEN(struct Node* jrootNode)
{
	if (jrootNode->childNum == 0)
	{
		if (strcmp(jrootNode->nodeType, "TOCKEN") == 0 || strcmp(jrootNode->nodeType, "TOKEN") == 0)
		{
			printNode(jrootNode);
			return;
		}
	}
	for (int i = 0; i < jrootNode->childNum; i++)
		printTOKEN(jrootNode->childArr[i]);
}
struct Node* change_expr_statement(struct Node* knode)
{
	union Data defData;
	struct Node* jnode;
	if (strcmp(knode->detType, "FUN_STATE") == 0)
	{
		defData.stringData = strndup("EXPR_STATEMENT", strlen("EXPR_STATEMENT"));
		jnode = makeNewNode("EXPR_STATEMENT", "FUN_STATE", "STRING", defData);
		makechildArr(jnode, 1);
		struct Node* funnode;
		defData.stringData = strndup("FUN_STATE", strlen("FUN_STATE"));
		funnode = makeNewNode("FUN_STATE", "ID OPENC PAR CLOSEC ;", "STRING", defData);
		makechildArr(funnode, 5);
		funnode->childArr[0] = knode->childArr[0]->childArr[0];
		funnode->childArr[1] = knode->childArr[0]->childArr[1];
		funnode->childArr[2] = knode->childArr[0]->childArr[2];
		funnode->childArr[3] = knode->childArr[0]->childArr[3];
		defData.stringData = strndup(";", strlen(";"));
		funnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);
		if (strcmp(funnode->childArr[0]->data.stringData, "println") == 0)
		{
			funnode->childArr[0]->data.stringData = strndup("System.out.println", strlen("System.out.println"));
			funnode->childArr[0]->childArr[0]->data.stringData = strndup("System.out.println", strlen("System.out.println"));
		}
		else if (strcmp(funnode->childArr[0]->data.stringData, "print") == 0)
		{
			funnode->childArr[0]->data.stringData = strndup("System.out.print", strlen("System.out.print"));
			funnode->childArr[0]->childArr[0]->data.stringData = strndup("System.out.print", strlen("System.out.print"));
		}
		jnode->childArr[0] = funnode;
		return jnode;
	}
	else
	{
		defData.stringData = strndup("EXPR_STATEMENT", strlen("EXPR_STATEMENT"));
		jnode = makeNewNode("EXPR_STATEMENT", "EXPR_STATEMENT ;", "STRING", defData);
		makechildArr(jnode, 2);
		jnode->childArr[0] = knode;
		defData.stringData = strndup(";", strlen(";"));
		jnode->childArr[1] = makeNewNode("TOKEN", ";", "STRING", defData);
		return jnode;
	}
}
struct Node* change_branch_statement(struct Node* knode)
{
	union Data defData;
	struct Node* jnode;
	if (strcmp(knode->childArr[0]->detType, "IF") == 0)
	{
		defData.stringData = strndup("BRANCH_STATEMENT", strlen("BRANCH_STATEMENT"));
		jnode = makeNewNode("BRANCH_STATEMENT", "IF_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_if_statement(knode);
		return jnode;
	}
	else if (strcmp(knode->childArr[0]->detType, "WHEN") == 0)
	{
		defData.stringData = strndup("BRANCH_STATEMENT", strlen("BRANCH_STATEMENT"));
		jnode = makeNewNode("BRANCH_STATEMENT", "WHEN_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_when_statement(knode);
		return jnode;
	}
	else if (strcmp(knode->childArr[0]->detType, "FOR") == 0)
	{
		defData.stringData = strndup("BRANCH_STATEMENT", strlen("BRANCH_STATEMENT"));
		jnode = makeNewNode("BRANCH_STATEMENT", "FOR_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_for_statement(knode);
		return jnode;
	}
	else
	{
		return knode;
	}
}
struct Node* change_for_statement(struct Node* knode)
{
	union Data defData;
	struct Node* jnode;
	if (strcmp(knode->childArr[2]->childArr[0]->childArr[3]->childArr[0]->nodeType, "LIST_STATEMENT") == 0) //conver to int x=4; x<=.... format
	{
		struct Node* rangeNode;
		struct Node* is_in_node = knode->childArr[2]->childArr[0];
		rangeNode = knode->childArr[2]->childArr[0]->childArr[3]->childArr[0]->childArr[0];

		if (rangeNode->childNum == 3) //there is no step part
		{	//if it is range
			defData.stringData = strndup("RANGE_STATEMENT", strlen("RANGE_STATEMENT"));
			jnode = makeNewNode("RANGE_STATEMENT", "int EXPR_STATEMENT = EXPR_STATEMENT ; EXPR_STATEMENT <= EXPR_STATEMENT ; EXPR_STATEMENT ++", "STRING", defData);
			makechildArr(jnode, 11);
			defData.stringData = strndup("int", strlen("int"));
			jnode->childArr[0] = makeNewNode("TOKEN", "int", "STRING", defData);
			jnode->childArr[1] = is_in_node->childArr[0];
			defData.stringData = strndup("=", strlen("="));
			jnode->childArr[2] = makeNewNode("TOKEN", "=", "STRING", defData);
			jnode->childArr[3] = rangeNode->childArr[0];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);
			jnode->childArr[5] = is_in_node->childArr[0];
			jnode->childArr[7] = rangeNode->childArr[2];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[8] = makeNewNode("TOKEN", ";", "STRING", defData);
			jnode->childArr[9] = is_in_node->childArr[0];

			if (strcmp(rangeNode->childArr[1]->nodeType, "RANGE") == 0)
			{
				defData.stringData = strndup("<=", strlen("<="));
				jnode->childArr[6] = makeNewNode("TOKEN", "<=", "STRING", defData);
				defData.stringData = strndup("++", strlen("++"));
				jnode->childArr[10] = makeNewNode("TOKEN", "++", "STRING", defData);
			}
			else
			{
				defData.stringData = strndup(">=", strlen(">="));
				jnode->childArr[6] = makeNewNode("TOKEN", ">=", "STRING", defData);
				defData.stringData = strndup("--", strlen("--"));
				jnode->childArr[10] = makeNewNode("TOKEN", "--", "STRING", defData);
			}

		}
		else //there is a step part
		{
			defData.stringData = strndup("RANGE_STATEMENT", strlen("RANGE_STATEMENT"));
			jnode = makeNewNode("RANGE_STATEMENT", "int EXPR_STATEMENT = EXPR_STATEMENT ; EXPR_STATEMENT <= EXPR_STATEMENT ; EXPR_STATEMENT += EXPR_STATEMENT", "STRING", defData);
			makechildArr(jnode, 12);
			defData.stringData = strndup("int", strlen("int"));
			jnode->childArr[0] = makeNewNode("TOKEN", "int", "STRING", defData);
			jnode->childArr[1] = is_in_node->childArr[0];
			defData.stringData = strndup("=", strlen("="));
			jnode->childArr[2] = makeNewNode("TOKEN", "=", "STRING", defData);
			jnode->childArr[3] = rangeNode->childArr[0];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);
			jnode->childArr[5] = is_in_node->childArr[0];
			jnode->childArr[7] = rangeNode->childArr[2];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[8] = makeNewNode("TOKEN", ";", "STRING", defData);
			jnode->childArr[9] = is_in_node->childArr[0];

			if (strcmp(rangeNode->childArr[1]->nodeType, "RANGE") == 0)
			{
				defData.stringData = strndup("<=", strlen("<="));
				jnode->childArr[6] = makeNewNode("TOKEN", "<=", "STRING", defData);
				defData.stringData = strndup("+=", strlen("+="));
				jnode->childArr[10] = makeNewNode("TOKEN", "+=", "STRING", defData);
				jnode->childArr[11] = rangeNode->childArr[4];
			}
			else
			{
				defData.stringData = strndup(">=", strlen(">="));
				jnode->childArr[6] = makeNewNode("TOKEN", ">=", "STRING", defData);
				defData.stringData = strndup("-=", strlen("-="));
				jnode->childArr[10] = makeNewNode("TOKEN", "-=", "STRING", defData);
				jnode->childArr[11] = rangeNode->childArr[4];
			}

		}
		knode->childArr[2] = jnode;
		//		knode->childArr[2]->childArr[0]->childArr[3]->childArr[0]->childArr[0]=jnode;
	}
	else //convert to types A : B format
	{
		struct Node* exprNode = knode->childArr[2];
		struct Node* listNode = exprNode->childArr[0];
		defData.stringData = strndup("LIST_STATEMENT", strlen("LIST_STATEMENT"));
		jnode = makeNewNode("LIST_STATEMENT", "types EXPR_STATEMENT : EXPR_STATEMENT", "STRING", defData);
		makechildArr(jnode, 4);

		jnode->childArr[1] = listNode->childArr[0];
		defData.stringData = strndup(":", strlen(":"));
		jnode->childArr[2] = makeNewNode("TOKEN", ":", "STRING", defData);
		jnode->childArr[3] = listNode->childArr[3];
		char* varname = strndup(listNode->childArr[3]->data.stringData, strlen(listNode->childArr[3]->data.stringData));
		int idx = travelListOfList(listNode->childArr[3]->data.stringData);
		defData.stringData = strndup(listof_types[idx], strlen(listof_types[idx]));
		if (strcmp(defData.stringData, "INT") == 0)
			defData.stringData = strndup("int", strlen("int"));
		else if (strcmp(defData.stringData, "FLOAT") == 0)
			defData.stringData = strndup("float", strlen("float"));
		else if (strcmp(defData.stringData, "DOUBLE") == 0)
			defData.stringData = strndup("double", strlen("double"));
		else if (strcmp(defData.stringData, "STRING") == 0)
			defData.stringData = strndup("String", strlen("String"));
		jnode->childArr[0] = makeNewNode("TOKEN", defData.stringData, "STRING", defData);
		exprNode->childArr[0] = jnode;
	}

	struct Node* bodySetNode = knode->childArr[5];
	bodySetNode->childArr[0]=change_body(bodySetNode->childArr[0]);

	return knode;
}
struct Node* change_when_sentence(struct Node* knode)
{
	if (strcmp(knode->nodeType, "WHEN_STATEMENT") == 0) //.if knode is when_statement
	{
		union Data defData;
		struct Node* jnode;
		if (strcmp(knode->childArr[0]->detType, "ELSE") == 0)	//if the statement starts with ELSE
		{		//structure is | ELSE  Arrow  retStatement | |
				//have to change to
				//default : retStatement ;
			defData.stringData = strndup("WHEN_SENTENCE", strlen("WHEN_SENTENCE"));
			jnode = makeNewNode("WHEN_SENTENCE", "default : retStatement ;", "STRING", defData);
			makechildArr(jnode, 4);
			defData.stringData = strndup("default", strlen("default"));
			jnode->childArr[0] = makeNewNode("TOKEN", "default", "STRING", defData);
			defData.stringData = strndup(":", strlen(":"));
			jnode->childArr[1] = makeNewNode("TOKEN", ":", "STRING", defData);
			jnode->childArr[2] = knode->childArr[2];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[3] = makeNewNode("TOKEN", ";", "STRING", defData);
			knode = jnode;
		}
		else //it doesn't start with ELSE  | exprStatement  Arrow  retStatement
		{
			defData.stringData = strndup("WHEN_SENTENCE", strlen("WHEN_SENTENCE"));
			jnode = makeNewNode("WHEN_SENTENCE", "case exprStatement : retStatement ;", "STRING", defData);
			makechildArr(jnode, 5);
			defData.stringData = strndup("case", strlen("case"));
			jnode->childArr[0] = makeNewNode("TOKEN", "case", "STRING", defData);
			jnode->childArr[1] = knode->childArr[0];
			defData.stringData = strndup(":", strlen(":"));
			jnode->childArr[2] = makeNewNode("TOKEN", ":", "STRING", defData);
			jnode->childArr[3] = knode->childArr[2];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);
			knode = jnode;
		}
	}
	else
	{
		for (int i = 0; i < knode->childNum; i++)
			knode->childArr[i] = change_when_sentence(knode->childArr[i]);
	}
	return knode;
}
struct Node* change_whenBodySet(struct Node* knode) //change kotlin whenBodyset to java style traveling the whenBodysetTree
{
	union Data defData;
	struct Node* jnode;
	knode = change_when_sentence(knode);
	return knode;
}
struct Node* change_when_statement(struct Node* knode)
{
	union Data defData;
	struct Node* jnode;
	defData.stringData = strndup("WHEN_Statement", strlen("WHEN_Statement"));
	jnode = makeNewNode("WHEN_Statement", "switch PENC  exprStatement  CLOSEC  OPENA  whenBodySet  CLOSEA ;", "STRING", defData);
	makechildArr(jnode, 7);
	defData.stringData = strndup("switch", strlen("switch"));
	jnode->childArr[0] = makeNewNode("TOKEN", "switch", "STRING", defData);
	jnode->childArr[1] = knode->childArr[1];
	jnode->childArr[2] = knode->childArr[2];
	jnode->childArr[3] = knode->childArr[3];
	jnode->childArr[4] = knode->childArr[4];
	jnode->childArr[5] = change_whenBodySet(knode->childArr[5]);
	jnode->childArr[6] = knode->childArr[6];
	defData.stringData = strndup(";", strlen(";"));
	return jnode;
}
struct Node* change_if_statement(struct Node* knode)
{
	if_types = (char**)malloc(sizeof(char*) * 100);;
	if_names = (char**)malloc(sizeof(char*) * 100);;
	if_num = 0;

	union Data defData;
	struct Node* jnode;
	defData.stringData = strndup("IF_Statement", strlen("IF_Statement"));
	jnode = makeNewNode("IF_STATEMENT", "IF OPENC exprStatement CLOSEC then ;", "STRING", defData);
	makechildArr(jnode, 6);
	jnode->childArr[0] = knode->childArr[0];
	jnode->childArr[1] = knode->childArr[1];
	jnode->childArr[3] = knode->childArr[3];
	defData.stringData = strndup(";", strlen(";"));
	jnode->childArr[5] = makeNewNode("TOKEN", ";", "STRING", defData);

	//fill in expr part
	//In the case of it is IS_IN_STATE
	if (strcmp(knode->childArr[2]->detType, "IS_IN_STATEMENT") == 0)
	{
		//in the case it is IS_Statement
		struct Node* is_in_node = knode->childArr[2]->childArr[0];

		if_names[if_num] = strndup(is_in_node->childArr[0]->data.stringData, strlen(is_in_node->childArr[0]->data.stringData));

		if (strcmp(is_in_node->childArr[3]->childArr[0]->detType, "Int") == 0)
			is_in_node->childArr[3]->childArr[0]->detType = strndup("int", strlen("int"));
		else if (strcmp(is_in_node->childArr[3]->childArr[0]->data.stringData, "Float") == 0)
			is_in_node->childArr[3]->childArr[0]->detType = strndup("float", strlen("float"));
		else if (strcmp(is_in_node->childArr[3]->childArr[0]->data.stringData, "Double") == 0)
			is_in_node->childArr[3]->childArr[0]->detType = strndup("double", strlen("double"));
		else if (strcmp(is_in_node->childArr[3]->childArr[0]->data.stringData, "Boolean") == 0)
			is_in_node->childArr[3]->childArr[0]->detType = strndup("bool", strlen("bool"));
		if_types[if_num] = strndup(is_in_node->childArr[3]->childArr[0]->detType, strlen(is_in_node->childArr[3]->detType));
		if_num++;
		if (strcmp(knode->childArr[2]->childArr[0]->childArr[2]->detType, "IS") == 0)
		{
			jnode->childArr[2] = knode->childArr[2]->childArr[0];
			defData.stringData = strndup("instanceof", strlen("instanceof"));
			jnode->childArr[2]->childArr[2] = makeNewNode("TOKEN", "instanceof", "STRING", defData);
			//types conversion
			if (strcmp(jnode->childArr[2]->childArr[3]->detType, "Int") == 0 || strcmp(jnode->childArr[2]->childArr[3]->detType, "Float") == 0 || strcmp(jnode->childArr[2]->childArr[3]->detType, "Double") == 0 || strcmp(jnode->childArr[2]->childArr[3]->detType, "Boolean") == 0)
				jnode->childArr[2]->childArr[3]->detType[0] -= 'I' - 'i';
		}
		else if (strcmp(knode->childArr[2]->childArr[2]->detType, "IN") == 0)
		{
			jnode->childArr[2] = knode->childArr[2];
		}
		else
		{
			jnode->childArr[2] = knode->childArr[2];
		}
	}
	else
	{
		jnode->childArr[2] = knode->childArr[2];
	}
	//then part
	struct Node* thenNode = knode->childArr[4];
	if (strcmp(knode->childArr[4]->childArr[0]->detType, "OPENA") == 0)
	{
		change_body(thenNode->childArr[1]->childArr[0]);
		jnode->childArr[4] = knode->childArr[4];
	}
	else
	{	//Then | RET_STATEMENT
		struct Node* retNode = knode->childArr[4]->childArr[0]->childArr[1];
		if (strcmp(retNode->detType, "EXPR_STATEMENT") == 0)
		{
			struct Node* exprNode = retNode->childArr[0];
			if (strcmp(exprNode->detType, "BINARY_OP_STATE") == 0)
			{
				struct Node* biNode = exprNode->childArr[0];
				int idx = travelIfList(biNode->childArr[0]->data.stringData);
				if (idx != -1)
				{
					defData.boolean = 0;
					struct Node* newExprNode = makeNewNode("BINARY_OP_STATE", "((types)ID).ID()", "NULL", defData);
					makechildArr(newExprNode, 7);
					defData.stringData = strndup("((", strlen("(("));
					newExprNode->childArr[0] = makeNewNode("TOKEN", "((", "STRING", defData);
					defData.stringData = strndup(if_types[idx], strlen(if_types[idx]));
					newExprNode->childArr[1] = makeNewNode("TOKEN", if_types[idx], "STRING", defData);
					defData.stringData = strndup(")", strlen(")"));
					newExprNode->childArr[2] = makeNewNode("TOKEN", ")", "STRING", defData);
					newExprNode->childArr[3] = biNode->childArr[0];
					defData.stringData = strndup(").", strlen(")."));
					newExprNode->childArr[4] = makeNewNode("TOKEN", ").", "STRING", defData);
					newExprNode->childArr[5] = biNode->childArr[2];
					defData.stringData = strndup("()", strlen("()"));
					newExprNode->childArr[6] = makeNewNode("TOKEN", "()", "STRING", defData);
					exprNode->childArr[0] = newExprNode;
				}
			}
		}
		jnode->childArr[4] = knode->childArr[4];
	}
	free(if_types);
	free(if_names);
	if_num = 0;
	return jnode;
}
struct Node* change_fun_statement(struct Node* knode)
{
	return knode;
}
struct Node* change_var_statement(struct Node* knode)
{
	union Data defData;
	defData.boolean = 0;
	struct Node* jnode;
	if (knode->childArr[4]->childNum != 0) //with assnBlock
	{
		jnode = makeNewNode("VAR_DEFSTATEMENT", "type ID ASSN exprStatement ; ", "NULL", defData);
		makechildArr(jnode, 5);
		jnode->childArr[1] = knode->childArr[2];
		defData.stringData = strndup("=", strlen("="));
		jnode->childArr[2] = makeNewNode("TOKEN", "=", "STRING", defData);
		jnode->childArr[3] = knode->childArr[4]->childArr[0]->childArr[1]->childArr[0];
		defData.stringData = strndup(";", strlen(";"));
		jnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);
	}
	else //without assBlock
	{
		jnode = makeNewNode("VAR_DEFSTATEMENT", "type ID  ; ", "NULL", defData);
		makechildArr(jnode, 3);
		jnode->childArr[1] = knode->childArr[2];
		defData.stringData = strndup(";", strlen(";"));
		jnode->childArr[2] = makeNewNode("TOKEN", ";", "STRING", defData);
	}

	//insert Type part
	if (knode->childArr[3]->childNum != 0) //no need to type inference
	{
		jnode->childArr[0] = knode->childArr[3]->childArr[1];

		if (strcmp(jnode->childArr[0]->detType, "Int") == 0 || strcmp(jnode->childArr[0]->detType, "Float") == 0 || strcmp(jnode->childArr[0]->detType, "Double") == 0)
		{
			jnode->childArr[0]->detType[0] -= 'I' - 'i';
			jnode->childArr[0]->childArr[0]->detType[0] -= 'I' - 'i';
			jnode->childArr[0]->data.stringData[0] -= 'I' - 'i';
			jnode->childArr[0]->childArr[0]->data.stringData[0] -= 'I' - 'i';
		}
	}
	else //shoud do type inference
	{
		//assnBlock
		//ASSN assnBlock2
		//exprStatement
		//factor
		//number
		// number->detType  | INT | FLOAT | MINUS INT | MINUS FLOAT
		char* typeInfo = knode->childArr[4]->childArr[0]->childArr[1]->childArr[0]->childArr[0]->childArr[0]->detType;
		if (strcmp(typeInfo, "INT") == 0 || strcmp(typeInfo, "MINUS INT") == 0)
		{
			defData.stringData = strndup("int", strlen("int"));
			jnode->childArr[0] = makeNewNode("TOKEN", "int", "STRING", defData);
		}
		else
		{
			defData.stringData = strndup("float", strlen("float"));
			jnode->childArr[0] = makeNewNode("TOKEN", "float", "STRING", defData);
		}
	}
	return jnode;
}
int travelIfList(char* compare)
{
	for (int i = 0; i < if_num; i++)
		if (strcmp(if_names[i], compare) == 0)
			return i;
	return -1;
}
int travelListOfList(char* compare)
{
	for (int i = 0; i < listof_num; i++)
		if (strcmp(listof_names[i], compare) == 0)
			return i;
	return -1;
}
struct Node* change_val_statement(struct Node* knode)
{
	union Data defData;
	defData.boolean = 0;
	struct Node* jnode;
	if (knode->childNum == 4 || knode->childNum == 6) //with assnBlock
	{

		if ((knode->childNum == 4) && (strcmp(knode->childArr[3]->childArr[1]->childArr[0]->detType, "LIST_STATEMENT") == 0))			//if The variable is an instance of a list 
		{
			struct Node* exprNode = knode->childArr[3]->childArr[1]->childArr[0];
			listof_types[listof_num] = strndup(exprNode->dataType, strlen(exprNode->dataType));
			listof_names[listof_num] = strndup(knode->childArr[2]->childArr[0]->data.stringData, strlen(knode->childArr[2]->childArr[0]->data.stringData));
			listof_num++;
			int idx = travelListOfList("items2222");
			defData.boolean = 0;
			jnode = makeNewNode("VAL_DEF_STATEMENT", "List< types > ID = Listof ( par ) ;", "NULL", defData);
			makechildArr(jnode, 10);
			defData.stringData = strndup("List<", strlen("List<"));
			jnode->childArr[0] = makeNewNode("TOKEN", "List<", "STRING", defData);
			if (strcmp(exprNode->childArr[0]->dataType, "INT") == 0)
				defData.stringData = strndup("int", strlen("int"));
			else if (strcmp(exprNode->childArr[0]->dataType, "FLOAT") == 0)
				defData.stringData = strndup("float", strlen("float"));
			else if (strcmp(exprNode->childArr[0]->dataType, "DOUBLE") == 0)
				defData.stringData = strndup("double", strlen("double"));
			else if (strcmp(exprNode->childArr[0]->dataType, "BOOLEAN") == 0)
				defData.stringData = strndup("bool", strlen("bool"));
			else if (strcmp(exprNode->childArr[0]->dataType, "STRING") == 0)
				defData.stringData = strndup("String", strlen("String"));
			jnode->childArr[1] = makeNewNode("TOKEN", defData.stringData, "STRING", defData);
			defData.stringData = strndup(">", strlen(">"));
			jnode->childArr[2] = makeNewNode("TOKEN", ">", "STRING", defData);
			jnode->childArr[3] = knode->childArr[2];
			jnode->childArr[4] = knode->childArr[3]->childArr[0];
			defData.stringData = strndup("List.of", strlen("List.of"));
			jnode->childArr[5] = makeNewNode("TOKEN", "List.of", "STRING", defData);
			jnode->childArr[6] = exprNode->childArr[0]->childArr[1];
			jnode->childArr[7] = exprNode->childArr[0]->childArr[2];
			jnode->childArr[8] = exprNode->childArr[0]->childArr[3];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[9] = makeNewNode("TOKEN", ";", "STRING", defData);
		}
		else if ((knode->childNum == 4) && (strcmp(knode->childArr[3]->childArr[1]->childArr[0]->detType, "SET_STATEMENT") == 0))			//if The variable is an instance of a list 
		{
			struct Node* exprNode = knode->childArr[3]->childArr[1]->childArr[0];
			defData.boolean = 0;
			listof_types[listof_num] = strndup(exprNode->dataType, strlen(exprNode->dataType));
			listof_names[listof_num] = strndup(knode->childArr[2]->childArr[0]->data.stringData, strlen(knode->childArr[2]->childArr[0]->data.stringData));
			listof_num++;
			jnode = makeNewNode("VAL_DEF_STATEMENT", "Set< types > ID = Listof ( par ) ;", "NULL", defData);
			makechildArr(jnode, 10);
			defData.stringData = strndup("Set<", strlen("Set<"));
			jnode->childArr[0] = makeNewNode("TOKEN", "List<", "STRING", defData);
			if (strcmp(exprNode->childArr[0]->dataType, "INT") == 0)
				defData.stringData = strndup("int", strlen("int"));
			else if (strcmp(exprNode->childArr[0]->dataType, "FLOAT") == 0)
				defData.stringData = strndup("float", strlen("float"));
			else if (strcmp(exprNode->childArr[0]->dataType, "DOUBLE") == 0)
				defData.stringData = strndup("double", strlen("double"));
			else if (strcmp(exprNode->childArr[0]->dataType, "BOOLEAN") == 0)
				defData.stringData = strndup("bool", strlen("bool"));
			else if (strcmp(exprNode->childArr[0]->dataType, "STRING") == 0)
				defData.stringData = strndup("String", strlen("String"));
			jnode->childArr[1] = makeNewNode("TOKEN", defData.stringData, "STRING", defData);
			defData.stringData = strndup(">", strlen(">"));
			jnode->childArr[2] = makeNewNode("TOKEN", ">", "STRING", defData);
			jnode->childArr[3] = knode->childArr[2];
			jnode->childArr[4] = knode->childArr[3]->childArr[0];
			defData.stringData = strndup("Set.of", strlen("Set.of"));
			jnode->childArr[5] = makeNewNode("TOKEN", "List.of", "STRING", defData);
			jnode->childArr[6] = exprNode->childArr[0]->childArr[1];
			jnode->childArr[7] = exprNode->childArr[0]->childArr[2];
			jnode->childArr[8] = exprNode->childArr[0]->childArr[3];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[9] = makeNewNode("TOKEN", ";", "STRING", defData);
		}
		else
		{
			jnode = makeNewNode("VAL_DEF_STATEMENT", "final type ID assnBlock ; ", "NULL", defData);
			makechildArr(jnode, 5);
			defData.stringData = strndup("final", strlen("final"));
			jnode->childArr[0] = makeNewNode("TOKEN", "final", "STRING", defData);
			jnode->childArr[2] = knode->childArr[2];
			if (knode->childNum == 4)
				jnode->childArr[3] = knode->childArr[3];
			else if (knode->childNum == 6)
				jnode->childArr[3] = knode->childArr[5];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[4] = makeNewNode("TOKEN", ";", "STRING", defData);

/*			jnode = makeNewNode("VAL_DEF_STATEMENT", "final type ID ASSN exprStatement ; ", "NULL", defData);
			makechildArr(jnode, 6);
			defData.stringData = strndup("final", strlen("final"));
			jnode->childArr[0] = makeNewNode("TOKEN", "final", "STRING", defData);
			jnode->childArr[2] = knode->childArr[2];
			defData.stringData = strndup("=", strlen("="));
			jnode->childArr[3] = makeNewNode("TOKEN", "=", "STRING", defData);
			if (knode->childNum == 4)
				jnode->childArr[4] = knode->childArr[3];
			else if (knode->childNum == 6)
				jnode->childArr[4] = knode->childArr[5];
			defData.stringData = strndup(";", strlen(";"));
			jnode->childArr[5] = makeNewNode("TOKEN", ";", "STRING", defData);
*/		}
	}
	else
	{
		jnode = makeNewNode("VAL_DEF_STATEMENT", "final type ID  ; ", "NULL", defData);
		makechildArr(jnode, 4);
		defData.stringData = strndup("final", strlen("final"));
		jnode->childArr[0] = makeNewNode("TOKEN", "final", "STRING", defData);
		jnode->childArr[2] = knode->childArr[2];
		defData.stringData = strndup(";", strlen(";"));
		jnode->childArr[3] = makeNewNode("TOKEN", ";", "STRING", defData);
	}
	if (strcmp(knode->childArr[3]->detType, "TYPE") == 0) //no need to type inference
	{
		jnode->childArr[1] = knode->childArr[4];
		if (strcmp(jnode->childArr[1]->detType, "Int") == 0 || strcmp(jnode->childArr[1]->detType, "Float") == 0 || strcmp(jnode->childArr[1]->detType, "Double") == 0)
		{
			jnode->childArr[1]->detType[0] -= 'I' - 'i';
			jnode->childArr[1]->childArr[0]->detType[0] -= 'I' - 'i';
			jnode->childArr[1]->data.stringData[0] -= 'I' - 'i';
			jnode->childArr[1]->childArr[0]->data.stringData[0] -= 'I' - 'i';
		}
	}
	else //shoud do type inference
	{
		//assnBlock
		//ASSN assnBlock2
		//exprStatement
		//factor
		//number
		// number->detType  | INT | FLOAT | MINUS INT | MINUS FLOAT
		char* typeInfo = knode->childArr[3]->childArr[1]->childArr[0]->childArr[0]->childArr[0]->detType;
		if (strcmp(typeInfo, "INT") == 0 || strcmp(typeInfo, "MINUS INT") == 0)
		{
			defData.stringData = strndup("int", strlen("int"));
			jnode->childArr[1] = makeNewNode("TOKEN", "int", "STRING", defData);
		}
		else
		{
			if (strcmp(knode->childArr[3]->dataType, "NULL") == 0)
			{
				defData.stringData = strndup("float", strlen("float"));
				jnode->childArr[1] = makeNewNode("TOKEN", "float", "STRING", defData);
			}
			else
			{
				defData.stringData = knode->childArr[3]->dataType;
				if (strcmp(defData.stringData, "INT") == 0)
					defData.stringData = strndup("int", strlen("int"));
				else if (strcmp(defData.stringData, "FLOAT") == 0)
					defData.stringData = strndup("float", strlen("float"));
				else if (strcmp(defData.stringData, "STRING") == 0)
					defData.stringData = strndup("String", strlen("String"));
				else if (strcmp(defData.stringData, "BOOLEAN") == 0)
					defData.stringData = strndup("bool", strlen("bool"));

				jnode->childArr[1] = makeNewNode("TOKEN", knode->childArr[3]->dataType, "STRING", defData);
			}

		}
	}
	return jnode;
}
struct Node* change_def_statement(struct Node* knode)
{
	struct Node* jnode;
	union Data defData;
	if (strcmp(knode->detType, "VAL_DEF_STATEMENT") == 0)
	{
		defData.boolean = 0;
		jnode = makeNewNode("DEF_STATEMENT", "VAL_DEF_STATE", "NULL", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_val_statement(knode->childArr[0]);
	}
	else if (strcmp(knode->detType, "VAR_DEF_STATEMENT") == 0)
	{
		defData.boolean = 0;
		jnode = makeNewNode("DEF_STATEMENT", "VAR_DEF_STATE", "NULL", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_var_statement(knode->childArr[0]);
	}
	else if (strcmp(knode->detType, "FUN_DEF_STATEMENT") == 0)
	{
		defData.boolean = 0;
		jnode = makeNewNode("DEF_STATEMENT", "FUN_DEF_STATE", "NULL", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_fun_statement(knode->childArr[0]);
	}
	else if (strcmp(knode->detType, "FUN_DEF_STATEMENT") == 0)
	{

	}
	return jnode;
}
struct Node* change_ret_statement(struct Node* knode)
{
	union Data defData;
	defData.stringData = strndup("RET_STATEMENT", strlen("RET_STATEMENT"));
	struct Node* jnode = makeNewNode("RET_STATEMENT", "RETURN RET_STATEMENT2 ;", "STRING", defData);
	makechildArr(jnode, 2);
	struct Node* ret2Node = knode->childArr[1];
	if (strcmp(ret2Node->detType, "EXPR_STATEMENT") == 0)
	{
		if (strcmp(ret2Node->childArr[0]->detType, "BINARY_OP_STATE") == 0)
		{	//will substitute expr_statement. put in ret2Node->childArr[0]
			union Data defData;

			struct Node* exprNode = ret2Node->childArr[0];
			struct Node* biNode = exprNode->childArr[0];

			int idx = travelIfList(biNode->childArr[0]->data.stringData);
			if (idx != -1)
			{
				defData.boolean = 0;
				struct Node* newExprNode = makeNewNode("BINARY_OP_STATE", "((types)ID).ID()", "NULL", defData);
				makechildArr(newExprNode, 7);
				defData.stringData = strndup("((", strlen("(("));
				newExprNode->childArr[0] = makeNewNode("TOKEN", "((", "STRING", defData);
				defData.stringData = strndup(if_types[idx], strlen(if_types[idx]));
				newExprNode->childArr[1] = makeNewNode("TOKEN", if_types[idx], "STRING", defData);
				defData.stringData = strndup(")", strlen(")"));
				newExprNode->childArr[2] = makeNewNode("TOKEN", ")", "STRING", defData);
				newExprNode->childArr[3] = biNode->childArr[0];
				defData.stringData = strndup(").", strlen(")."));
				newExprNode->childArr[4] = makeNewNode("TOKEN", ").", "STRING", defData);
				newExprNode->childArr[5] = biNode->childArr[2];
				defData.stringData = strndup("()", strlen("()"));
				newExprNode->childArr[6] = makeNewNode("TOKEN", "()", "STRING", defData);
				exprNode->childArr[0] = newExprNode;
			}


		}
	}
	jnode->childArr[0] = knode;
	defData.stringData = strndup(";", strlen(";"));
	jnode->childArr[1] = makeNewNode("TOKEN", ";", "STRING", defData);
	return jnode;
}
struct Node* change_statement(struct Node* knode)
{
	struct Node* jnode;
	union Data defData;
	if (strcmp(knode->detType, "DEF_STATEMENT") == 0)
	{
		defData.stringData = strndup("DEF_STATEMENT", strlen("DEF_STATEMENT"));
		jnode = makeNewNode("STATEMENT", "DEF_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_def_statement(knode->childArr[0]);
	}
	/*	else if(strcmp(knode->detType,"ASSN_STATEMENT")==0){
			defData.stringData=strndup("ASSN_STATEMENT",strlen("ASSN_STATEMENT"));
			jnode=makeNewNode("STATEMENT","ASSN_STATEMENT","STRING",defData);
			makechildArr(jnode,1);
			jnode->childArr[0]=knode->childArr[0];
		}*/
	else if (strcmp(knode->detType, "BRANCH_STATEMENT") == 0) {
		defData.stringData = strndup("BRANCH_STATEMENT", strlen("BRANCH_STATEMENT"));
		jnode = makeNewNode("STATEMENT", "BRANCH_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_branch_statement(knode->childArr[0]);
	}
	else if (strcmp(knode->detType, "EXPR_STATEMENT") == 0) {
		defData.stringData = strndup("EXPR_STATEMENT", strlen("EXPR_STATEMENT"));
		jnode = makeNewNode("STATEMENT", "EXPR_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_expr_statement(knode->childArr[0]);
	}
	else if (strcmp(knode->detType, "RET_STATEMENT") == 0) {
		defData.stringData = strndup("RET_STATEMENT", strlen("RET_STATEMENT"));
		jnode = makeNewNode("STATEMENT", "RET_STATEMENT", "STRING", defData);
		makechildArr(jnode, 1);
		jnode->childArr[0] = change_ret_statement(knode->childArr[0]);
	}
	else {
		defData.stringData = strndup("ELSE_OTHER_STATEMENT", strlen("ELSE_OTHER_STATEMENT"));
		jnode = makeNewNode("STATEMENT", "ELSE_OTHER_STATEMENT ;", "STRING", defData);
		makechildArr(jnode, 2);
		jnode->childArr[0] = knode->childArr[0];
		defData.stringData = strndup(";", strlen(";"));
		jnode->childArr[1] = makeNewNode("TOKEN", ";", "STRING", defData);
	}
	return jnode;
}
struct Node* change_body(struct Node* knode)
{
	struct Node* jnewnode;
	union Data defData2;
	// There are two cases of its children
	// 1. statment Body
	// 2. satement 
	if (knode->childNum == 2) //Body | statement Body
	{
		defData2.boolean = 0;
		jnewnode = makeNewNode("BODY", "statement_Body", "NULL", defData2);
		makechildArr(jnewnode, 2);
		jnewnode->childArr[0] = change_statement(knode->childArr[0]);
		jnewnode->childArr[1] = change_body(knode->childArr[1]);
	}
	else //Body | statement 
	{
		defData2.boolean = 0;
		jnewnode = makeNewNode("BODY", "statement", "NULL", defData2);
		makechildArr(jnewnode, 1);
		jnewnode->childArr[0] = change_statement(knode->childArr[0]);
	}
	return jnewnode;
}
void printOutAsJava(struct Node* jnode)
{
	if (strcmp(jnode->nodeType, "TOKEN") == 0)
	{
		if (strcmp(jnode->dataType, "NULL") != 0)
		{

			if (strcmp(jnode->dataType, "STRING") == 0)
			{
				if (strcmp(jnode->data.stringData, ";") == 0 || strcmp(jnode->data.stringData, "}") == 0 || strcmp(jnode->data.stringData, "{") == 0)
					printf("%s\n", jnode->data.stringData);
				else
					printf("%s ", jnode->data.stringData);
			}
			else if (strcmp(jnode->dataType, "INT") == 0)
				printf("%d ", jnode->data.boolean);
			else if (strcmp(jnode->dataType, "FLOAT") == 0)
				printf("%lf ", jnode->data.floatData);
			else if (strcmp(jnode->dataType, "DOUBLE") == 0)
				printf("%lf ", jnode->data.floatData);
			else if (strcmp(jnode->dataType, "BOOLEAN") == 0)
				printf("%d ", jnode->data.boolean);
			else
				printf("%d ", jnode->data.intData);
		}
		else
		{
			printf("%s ", jnode->detType);			
		}
	}
	for (int i = 0; i < jnode->childNum; i++)
		printOutAsJava(jnode->childArr[i]);
}