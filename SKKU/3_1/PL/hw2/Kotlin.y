%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "expr.tab.h"
#include "calc.h"
extern int yylex(void);
extern void yyterminate();
extern int yyerror(const char *s);
union Data parData2;
%}
%error-verbose
%union {
	double dval;
	char * str;	
	int ival;
	struct Node * ptr;
}
%token <ptr> null DOT
%token <ptr> Any
%token <ptr> Int Long String List Double interface abstract override get class
%token <ptr> Float Unit Boolean
%token <ptr> NOT
%token <ptr> downTo
%token <ptr> Q
%token <ptr> Arrow
%token <ptr> DOWN
%token <ptr> step RETURN
%token <ptr> fun
%token <ptr> PRINTLN
%token <ptr> PRINT
%token <ptr> SAME DIFF
%token <ptr> TYPE COMMA import package
%token <ptr> IF WHILE FOR WHEN ELSE BR 
%token <ptr> VAR VAL
%token <ptr> STRING IDENTIFIER
%token <ptr> INT
%token <ptr> FLOAT
%token <ptr> EOL OPENA OPENB OPENC CLOSEA CLOSEB CLOSEC OPEND CLOSED IS IN listOf
%token <ptr> AND OR
%token <ptr> LESS LESS_EQ LARGER LARGER_EQ PLUS MINUS MULT DIV INCRE DECRE ASSN PLUS_ASSN MINUS_ASSN MULT_ASSN DIV_ASSN
//%left  PLUS MINUS
//%left  MULT DIV
//%right ASSN
%precedence RETURN
%precedence ASSN PLUS_ASSN MINUS_ASSN MULT_ASSN DIV_ASSN
%precedence INCRE DECRE
%precedence AND OR
%precedence SAME DIFF
%precedence PLUS MINUS
%precedence MULT DIV
%precedence NOT
%nonassoc UMINUS
%nonassoc UPLUS
%type <ptr> NUMBER body  exprStatement ID defStatement BodySet goal statement RANGE branchStatement whenBodySet whenStatement whenBody NotBlock retStatement retStatement2 assnStatement assnOP is_in_Statement funDefStatement2 defStatement2 lambdaStatement types varDefBlock1 varDefBlock2 lambda2 lambda1 packStatement2 packStatement importStatement importStatement2 
%type <ptr> binaryOpStatement unaryOpStatement factor funStatement  listStatement then elseBlock ElseBlock interfaceDefStatement classDefStatement parBlock parentBlock parentBlock2 parStatement par defPar rangeStatement funDefStatement valDefStatement varDefStatement funAssnBlock overrideBlock funTypeBlock QBlock assnBlock2 assnBlock
//%locations %define api.pure full
//%define api.prefix {c}
%%
/* Rules */
goal :
	statement
{
	struct Node * newNode = makeNewNode("GOAL","STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
		rootNode=newNode;
}
	|	statement goal	
{
	struct Node * newNode = makeNewNode("GOAL","STATEMENT GOASL",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
		rootNode=newNode;
}
;
packStatement:
	package ID packStatement2
{
	struct Node * newNode = makeNewNode("PACK_STATE","package ID packStatement2",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
packStatement2:
	DOT ID
{
	struct Node * newNode = makeNewNode("PACK_STATE2","DOT ID",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	| DOT ID packStatement2
{
	struct Node * newNode = makeNewNode("PACK_STATE2","DOT ID PACK_STATE2",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[1]=$3;
		$$=newNode;
		printNode(newNode);
}
;
importStatement:
	import ID importStatement2
{
	struct Node * newNode = makeNewNode("IMPORT_STATE","IMPORT ID IMPORT_Statement2",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
importStatement2:
	DOT ID
{
	struct Node * newNode = makeNewNode("IMPORT_STATE2","DOT ID",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	| DOT MULT
{
	struct Node * newNode = makeNewNode("IMPORT_STATE2","DOT ALL",$2->dataType,$2->data);
		$2->detType=strndup("ALL",strlen("ALL"));
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	| DOT ID importStatement2
{
	struct Node * newNode = makeNewNode("IMPORT_STATE2","DOT ID IMPORT_Statement2",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
statement:
	packStatement
{
	struct Node * newNode = makeNewNode("STATEMENT","PACK_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|importStatement
{
	struct Node * newNode = makeNewNode("STATEMENT","IMPORT_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|retStatement 
{
	struct Node * newNode = makeNewNode("STATEMENT","RET_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|defStatement 
{
	struct Node * newNode = makeNewNode("STATEMENT","DEF_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|assnStatement
{
	struct Node * newNode = makeNewNode("STATEMENT","ASSN_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|branchStatement	
{
	struct Node * newNode = makeNewNode("STATEMENT","BRANCH_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|exprStatement
{
	struct Node * newNode = makeNewNode("STATEMENT","EXPR_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|lambdaStatement
{
	struct Node * newNode = makeNewNode("STATEMENT","LAMBDA_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
RANGE:
	DOT DOT
{
	struct Node * newNode = makeNewNode("RANGE","DOT DOT",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
BodySet:
	%empty	
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("BODY_SET","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	body 
{
	struct Node * newNode = makeNewNode("BODY_SET","BODY",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
body: 
	statement
{
	struct Node * newNode = makeNewNode("BODY","STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	statement body
{
	struct Node * newNode = makeNewNode("BODY","STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
    ; 
branchStatement: //if, while, for, when....
		IF OPENC exprStatement CLOSEC then	
		{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","IF OPENC exprStatement CLOSEC then",$1->dataType,$1->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	|	WHILE OPENC  exprStatement  CLOSEC  OPENA  BodySet  CLOSEA 
		{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","WHILE OPENC exprStatement CLOSEC  OPENA  BodySet  CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,7);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		$$=newNode;
		printNode(newNode);
}	|	FOR  OPENC  exprStatement  CLOSEC  OPENA  BodySet  CLOSEA 
		{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","FOR OPENC exprStatement CLOSEC  OPENA  BodySet  CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,7);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		$$=newNode;
		printNode(newNode);
}	|	WHEN  OPENC  exprStatement  CLOSEC  OPENA  whenBodySet  CLOSEA 
		{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","WHEN OPENC expr_Statement CLOSEC  OPENA  WHEN_BodySet  CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,7);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		$$=newNode;
		printNode(newNode);
}	|	WHEN  OPENA  whenBodySet  CLOSEA
{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","WHEN  OPENA  whenBodySet  CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
;
then:
	OPENA  BodySet  CLOSEA ElseBlock 
{
	struct Node * newNode = makeNewNode("THEN","OPENA  BodySet  CLOSEA ElseBlock ",$2->dataType,$2->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement  ELSE  exprStatement 
{
	struct Node * newNode = makeNewNode("THEN","exprStatement  ELSE  exprStatement ",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	retStatement
{
	struct Node * newNode = makeNewNode("THEN","RET_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
};
ElseBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("ELSE_BLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	| elseBlock
{
	struct Node * newNode = makeNewNode("ELSE_BLOCK","ELSE_BLOCK",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
elseBlock:
	ELSE  OPENA  BodySet  CLOSEA
{
	struct Node * newNode = makeNewNode("else_BLOCK","ELSE  OPENA  BodySet  CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
};
whenBodySet:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("WHEN_BODY_SET","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}	|	whenBody 
{
	struct Node * newNode = makeNewNode("WHEN_BODY_SET","when_BODY",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
whenStatement:
		exprStatement  Arrow  STRING
{
	struct Node * newNode = makeNewNode("WHEN_STATEMENT","EXPR_STATEMENT ARROW STRING",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	NotBlock  IS  types  Arrow  STRING
		{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NOT_BLOCK IS TYPES ARROW STRING ",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
	|	NotBlock  IS  exprStatement  Arrow  STRING
		{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NotBlock  IS  exprStatement  Arrow  STRING",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
	|	NotBlock  IN  listStatement  Arrow  STRING
		{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NotBlock  IN  LIST_Statement  Arrow  STRING",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}

	|	ELSE  Arrow  STRING
{
	struct Node * newNode = makeNewNode("NUMBER","ELSE  Arrow  STRING",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}

	|	exprStatement  Arrow funStatement
{
	struct Node * newNode = makeNewNode("NUMBER","ELSE  Arrow  funStatement",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	NotBlock  IS  types  Arrow  funStatement
{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NotBlock  IS  types  Arrow  funStatement",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	|	NotBlock  IS  exprStatement  Arrow  funStatement
{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NotBlock  IS  EXPR  Arrow  funStatement",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	|	NotBlock  IN  listStatement  Arrow  funStatement
{
		struct Node * newNode = makeNewNode("WHEN_STATEMENT","NotBlock  IS  LIST_STATE  Arrow  funStatement",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	|	ELSE  Arrow  funStatement
{
	struct Node * newNode = makeNewNode("NUMBER","ELSE  Arrow  funStatement",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
whenBody:
	whenStatement
{
	struct Node * newNode = makeNewNode("WHEN_BODY","WHEN_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	whenStatement whenBody
{
	struct Node * newNode = makeNewNode("WHEN_BODY","WHEN_STATE WHEN_BODY",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
NotBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("NOT_BLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}

	| NOT
{
	struct Node * newNode = makeNewNode("NOT_BLOCK","NOT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
retStatement: //return statement
    //return exprStatement
		RETURN retStatement2
{
	struct Node * newNode = makeNewNode("RET_STATEMENT","RETURN RET_STATEMENT2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
retStatement2:
	exprStatement
{
	struct Node * newNode = makeNewNode("RET_STATEMENT2","EXPR_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| null
{
	struct Node * newNode = makeNewNode("RET_STATEMENT2","NULL",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| %empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("RET_STATEMENT2","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
;
assnStatement:
   		ID assnOP exprStatement
{
	struct Node * newNode = makeNewNode("ASSN_STATEMENT","ID ASSN_OP EXPR_STATEMENT",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	ID assnOP branchStatement	
{
	struct Node * newNode = makeNewNode("ASSN_STATEMENT","ID ASSN_OP BRANCH_STATE",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	ID INCRE
{
	struct Node * newNode = makeNewNode("ASSN_STATEMENT","ID INCRE",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	|	ID DECRE
{
	struct Node * newNode = makeNewNode("ASSN_STATEMENT","ID DECRE",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
   ;
assnOP:
		ASSN
{
	struct Node * newNode = makeNewNode("ASSN_OP","ASSN",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	PLUS_ASSN
{
	struct Node * newNode = makeNewNode("ASSN_OP","PLUS_ASSN",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	MINUS_ASSN
{
	struct Node * newNode = makeNewNode("ASSN_OP","MINUS_ASSN",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	MULT_ASSN
{
	struct Node * newNode = makeNewNode("ASSN_OP","MULT_ASSN",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	DIV_ASSN
{
	struct Node * newNode = makeNewNode("ASSN_OP","DIV_ASSN",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
	
listStatement:
   // listOf("a", "b")
	listOf OPENC  par  CLOSEC
{
	struct Node * newNode = makeNewNode("LIST_STATEMENT","LISTOF (PAR)",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
	|	rangeStatement
{
	struct Node * newNode = makeNewNode("LIST_STATEMENT","RANGE_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Any OPENC CLOSEC
{
	struct Node * newNode = makeNewNode("LIST_STATEMENT","Any OPENC CLOSEC",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
   ;
is_in_Statement:
		exprStatement NotBlock IS exprStatement
{
	struct Node * newNode = makeNewNode("IS_INT_STATE","exprStatement NotBlock IS exprStatement",$3->dataType,$3->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement NotBlock IN exprStatement
{
	struct Node * newNode = makeNewNode("IS_INT_STATE","exprStatement NotBlock IN exprStatement",$3->dataType,$3->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement NotBlock IS types
{
	struct Node * newNode = makeNewNode("IS_INT_STATE","exprStatement NotBlock IS TYPES",$3->dataType,$3->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
;
defStatement:	  
	funDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","FUN_DEF_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| varDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","VAR_DEF_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| valDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","VAL_DEF_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| classDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","CLASS_DEF_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| interfaceDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","INTERFACEW_DEF_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| override funDefStatement2
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","override funDefStatement2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	| abstract defStatement2
{
	struct Node * newNode = makeNewNode("DEF_STATEMENT","abstract defStatement2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
    ;
funDefStatement2:
	abstract funDefStatement
{
	struct Node * newNode = makeNewNode("FUN_DEF_STATE2","abstract FUN_def_State",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	|	funDefStatement
{
	struct Node * newNode = makeNewNode("FUN_DEF_STATE2","FUN_def_State",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
defStatement2:
	funDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATE2","FUN_def_State",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| classDefStatement
{
	struct Node * newNode = makeNewNode("DEF_STATE2","CLASS_def_State",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
interfaceDefStatement :
	interface ID OPENA BodySet CLOSEA
		{
	struct Node * newNode = makeNewNode("INTERFACE_DEF_STATE","interface ID OPENA BodySet CLOSEA",$1->dataType,$1->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
;
classDefStatement:
	class ID parBlock parentBlock OPENA BodySet CLOSEA 
		{
	struct Node * newNode = makeNewNode("class_def_STATEMENT","class ID parBlock parentBlock OPENA BodySet CLOSEA ",$1->dataType,$1->data);
		makechildArr(newNode,7);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		$$=newNode;
		printNode(newNode);
}

;
parBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("PAR_BLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);}	
| OPENC parStatement CLOSEC
{
	struct Node * newNode = makeNewNode("PAR_BLOCK","EMPTY",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
parentBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("PARENT_BLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);}	
	|	TYPE parentBlock2
{
	struct Node * newNode = makeNewNode("PARENT_BLOCK","TYPE parentBlock2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
parentBlock2:
		ID OPENC exprStatement CLOSEC
{
	struct Node * newNode = makeNewNode("PARENT_BLOCK2","ID (EXPR_STATEMENT)",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
	|	ID OPENC exprStatement CLOSEC COMMA parentBlock2
		{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","ID OPENC exprStatement CLOSEC COMMA parentBlock2",$1->dataType,$1->data);
		makechildArr(newNode,6);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		$$=newNode;
		printNode(newNode);
}

	|	ID
{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","ID",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|valDefStatement	
{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","val_def_state",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|ID COMMA parentBlock2
{
	struct Node * newNode = makeNewNode("BRANCH_STATEMENT","ID COMMA PARENT_BLOCK2",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
parStatement:
	varDefStatement 
{
	struct Node * newNode = makeNewNode("PAR_STATEMENT","VAR_DEF_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|valDefStatement
{
	struct Node * newNode = makeNewNode("PAR_STATEMENT","VAL_DEF_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|varDefStatement COMMA parStatement
{
	struct Node * newNode = makeNewNode("PAR_STATEMENT","VAR_DEF_STATE COMMA PAR_STATE",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|valDefStatement COMMA parStatement
{
	struct Node * newNode = makeNewNode("PAR_STATEMENT","VAL_DEF_STATE COMMA PAR_STATE",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
par : 
		%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("PAR","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	exprStatement
{
	struct Node * newNode = makeNewNode("PAR","EXPR_STATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement  COMMA  par 
{
	struct Node * newNode = makeNewNode("PAR","EXPR_STATEMENT COMMA PAR",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
defPar: 
		%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("DEF_PAR","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	ID  TYPE  types 
{
	struct Node * newNode = makeNewNode("DEF_PAR","ID TYPE TYPES",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);}
	|	ID  TYPE  types  COMMA  defPar 
		{
	struct Node * newNode = makeNewNode("DEF_PAR","ID  TYPE  types  COMMA  defPar",$1->dataType,$1->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}

;
funDefStatement:
	fun  ID  OPENC  defPar  CLOSEC  funTypeBlock  funAssnBlock
{
	struct Node * newNode = makeNewNode("FUN_DEF_STATEMENT","FUN ID DEFPAR FUN_TYPE_BLOCK FUN_ASSN_BLOCK",$1->dataType,$1->data);
		makechildArr(newNode,7);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		$$=newNode;
		printNode(newNode);
}
;
overrideBlock :
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("OVERRIDE_BLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	override
{
	struct Node * newNode = makeNewNode("OVERRIDE_BLOCK","OVERRIDE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
funTypeBlock:
		%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("FUNTYPEBLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	 TYPE  types  QBlock 
{
	struct Node * newNode = makeNewNode("FUNTYPEBLOCK","TYPE TYPES QBLOCK",$1->dataType,$1->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
QBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("QBLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|	Q
{	
	struct Node * newNode = makeNewNode("QBLOCK","Q",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
};
assnBlock:
	ASSN assnBlock2
{
	struct Node * newNode = makeNewNode("ASSNBLOCK","ASSN ASSNBLOCK2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
assnBlock2:
	exprStatement
{
	struct Node * newNode = makeNewNode("ASSNBLOCK2","EXPRSTATEMENT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| branchStatement
{
	struct Node * newNode = makeNewNode("ASSNBLOCK2","BRANCH_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
funAssnBlock:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("FUNASSNBLOCK","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	| OPENA  BodySet  CLOSEA
{
	struct Node * newNode = makeNewNode("FUNASSNBLOCK","{ BODY_SET }",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|  assnBlock
{
	struct Node * newNode = makeNewNode("NUMBER","ASSN_BLOCK",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
varDefStatement:		//variance defintion
 	overrideBlock VAR  ID varDefBlock1 varDefBlock2
{
	struct Node * newNode = makeNewNode("VARDEFSTATEMENT","OVERRIDEBLOCK VAR ID VAR DEFBLOCK1 VARDEFBLCOK2",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
; 
varDefBlock1:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("VARDEFBLOCK1","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|TYPE  types
{
	struct Node * newNode = makeNewNode("VARDEFBLOCK2","TYPE TYPES",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
varDefBlock2:
	%empty
{
	parData2.boolean=0;
	struct Node * newNode = makeNewNode("VARDEFBLOCK2","EMPTY","",parData2);
	$$=newNode;
	printNode(newNode);
}
	|assnBlock
{
	struct Node * newNode = makeNewNode("VARDEFBLOCK2","ASSNBLOCK",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| get OPENC CLOSEC assnBlock
	{
	struct Node * newNode = makeNewNode("VARDEFBLOCK2","GET () ASSNBLOCK",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}
;
valDefStatement:		//value definition
   // val a: Int
		overrideBlock VAL  ID  assnBlock
{
	struct Node * newNode = makeNewNode("VALDEFSTATEMENT","OVERRIDEBLOCK VAL ID",$2->dataType,$2->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
}	
	|	overrideBlock VAL  ID  TYPE  types
{
	struct Node * newNode = makeNewNode("VALDEFSTATEMENT","OVERRIDEBLOCK VAL ID TYPE TYPES",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	
	|	overrideBlock VAL  ID  TYPE  types  assnBlock
{
	struct Node * newNode = makeNewNode("VALDEFSTATEMENT","OVERRIDEBLOCK VAL ID TYPE TYPE TYPES ASSNBLOCK",$2->dataType,$2->data);
		makechildArr(newNode,6);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		$$=newNode;
		printNode(newNode);
}	
	|	overrideBlock VAL  ID  TYPE  types get OPENC CLOSEC assnBlock
{
	struct Node * newNode = makeNewNode("VALDEFSTATEMENT","OVERRIDEBLOCK VAL ID TYPE TYPE TYPES GET ASSNBLOCK",$2->dataType,$2->data);
		makechildArr(newNode,9);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		newNode->childArr[5]=$6;
		newNode->childArr[6]=$7;
		newNode->childArr[7]=$8;
		newNode->childArr[8]=$9;
		$$=newNode;
		printNode(newNode);
}	
    ;
rangeStatement:
		exprStatement RANGE exprStatement 
{
	struct Node * newNode = makeNewNode("RANGESTATEMENT","EXPR RANGE EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}	
	|	exprStatement RANGE exprStatement step exprStatement
{	
	struct Node * newNode = makeNewNode("RANGESTATEMENT","EXPR RANGE EXPR STEP EXPR",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}	|	exprStatement downTo exprStatement
{
	struct Node * newNode = makeNewNode("RANGESTATEMENT","EXPR DOWNTO EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}	|	exprStatement downTo exprStatement step exprStatement
{
	struct Node * newNode = makeNewNode("RANGESTATEMENT","EXPR downTo exprStatement step exprStatement",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
;

binaryOpStatement:
		exprStatement PLUS exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR PLUS EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement MINUS exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR MINUS EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement MULT exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR MULT EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement DIV exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR DIV EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement LESS exprStatement 
{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR LESS EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement LESS_EQ exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR LESS_EQ EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement LARGER exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR LARGER EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement LARGER_EQ exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR LARGER_EQ EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement SAME exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR SAME EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement SAME types
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR SAME TYPES",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	| 	exprStatement DIFF exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR DIFF EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	| 	exprStatement DIFF types
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR DIFF TYPES",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	exprStatement AND exprStatement
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","EXPR AND EXPR",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	ID DOT ID
	{
	struct Node * newNode = makeNewNode("BINARY_OPSTATEMENT","ID DOT ID",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
	|	ID DOT funStatement
	{
	struct Node * newNode = makeNewNode("NUMBER","ID DOT FUNSTATEMENT",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
;
lambdaStatement:
		ID lambda2
		{
	struct Node * newNode = makeNewNode("LAMBDA_STATE","ID LAMBDA2",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
lambda2:
	lambda1 
{
	struct Node * newNode = makeNewNode("LAMBDA2","LAMBDA1",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	| lambda1 lambda2
{
	struct Node * newNode = makeNewNode("LAMBDA2","LAMBDA1 LAMBDA2",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
lambda1:
	DOT ID OPENA exprStatement CLOSEA 
		{
	struct Node * newNode = makeNewNode("LAMBDA1","	DOT ID OPENA exprStatement CLOSEA ",$2->dataType,$2->data);
		makechildArr(newNode,5);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		newNode->childArr[4]=$5;
		$$=newNode;
		printNode(newNode);
}
;
unaryOpStatement:
		NOT exprStatement 
		{
	struct Node * newNode = makeNewNode("UNARY_OP_STATE","NOT exprStatement",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;

exprStatement:	
        STRING {
	struct Node * newNode = makeNewNode("EXPR","STRING",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |   ID {
	struct Node * newNode = makeNewNode("EXPR","ID",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |	binaryOpStatement {
	struct Node * newNode = makeNewNode("EXPR","BINARY_OP_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |   unaryOpStatement
{
	struct Node * newNode = makeNewNode("EXPR","UNARY_OP_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |  	OPENC  exprStatement  CLOSEC {
	struct Node * newNode = makeNewNode("EXPR","( EXPR )",$2->dataType,$2->data);
		makechildArr(newNode,3);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		$$=newNode;
		printNode(newNode);
}
    |	factor {
	struct Node * newNode = makeNewNode("EXPR","FACTOR",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |	is_in_Statement {
	struct Node * newNode = makeNewNode("EXPR","IS_IN_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}

    |	listStatement 
{
	struct Node * newNode = makeNewNode("EXPR","LIST_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
    |	funStatement 
{
	struct Node * newNode = makeNewNode("EXPR","FUN_STATE",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
    ;
funStatement:
	ID OPENC  par  CLOSEC
{
	struct Node * newNode = makeNewNode("EXPR","ID OPENC  par  CLOSEC",$1->dataType,$1->data);
		makechildArr(newNode,4);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		newNode->childArr[2]=$3;
		newNode->childArr[3]=$4;
		$$=newNode;
		printNode(newNode);
};
factor: 
    	NUMBER		
{
	struct Node * newNode = makeNewNode("FACTOR","NUMBER",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	
;
ID: 
	IDENTIFIER 
{
	struct Node * newNode = makeNewNode("ID","IDENTIFIER",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
types :
		Int	
{
	struct Node * newNode = makeNewNode("TYPES","Int",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Float
{
	struct Node * newNode = makeNewNode("TYPES","Float",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Double	
{
	struct Node * newNode = makeNewNode("TYPES","Double",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	String	
{
	struct Node * newNode = makeNewNode("TYPES","String",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Unit	
{
	struct Node * newNode = makeNewNode("TYPES","Unit",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	null
{
	struct Node * newNode = makeNewNode("TYPES","null",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Any	
{
	struct Node * newNode = makeNewNode("TYPES","Any",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	Boolean	
{
	struct Node * newNode = makeNewNode("TYPES","Boolean",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
	|	List types 
{
	struct Node * newNode = makeNewNode("TYPES","LIST TYPES",$1->dataType,$1->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}	|	Long		{ 	
	struct Node * newNode = makeNewNode("TYPES","LONG",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}
;
NUMBER : 
		INT 
	{	struct Node * newNode = makeNewNode("NUMBER","INT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}	|	FLOAT 
	{ 	
	struct Node * newNode = makeNewNode("NUMBER","FLOAT",$1->dataType,$1->data);
		makechildArr(newNode,1);
		newNode->childArr[0]=$1;
		$$=newNode;
		printNode(newNode);
}	|	MINUS INT 
{
	struct Node * newNode = makeNewNode("NUMBER","MINUS INT",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
	|	MINUS FLOAT 
{
	struct Node * newNode = makeNewNode("NUMBER","MINUS FLOAT",$2->dataType,$2->data);
		makechildArr(newNode,2);
		newNode->childArr[0]=$1;
		newNode->childArr[1]=$2;
		$$=newNode;
		printNode(newNode);
}
;
%%
/* User code */
/* User code */
/*
typedef struct NODE
{
	//Informations about node
	char  nodeType[20];	//which part of syntax is this node
	char  detType[20];	//detail type of this node
//	char  dataType[20];	//int, float, 
	union Data data;	
//	int parent;		//If there exists parent, parent v =1, no parent v=0
	struct NODE ** children;	//list of children of this node.

}Node;

union Data {   
    int boolean; 	//If there is no data, boolean = 0/ there is data, boolean=1
    int intData;
    float floatData;     
    char * stringData;    
};
*/
int yyerror(const char *s)
{
	return printf("%s\n", s);
}

