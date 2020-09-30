%{
#include <stdlib.h>
#include <string.h>
#include "Kotlin.tab.h"
#include "Kotlin.h"
extern int yyerror(const char*);
union Data parData;
%}
INT [0-9]+
INTL [0-9]+L
FLOAT [0-9]+"."[0-9]+
ID [a-zA-Z][a-zA-Z0-9]*
SPACE [\t\r]+
COMMENT \/\/[^\t\r\n]*
STRING \"[^\”\"]*[\”\"]
%%
"\n"	{	
}
{SPACE} { }
{COMMENT} {
}
{INT}	{ 
	parData.intData=atoi(yytext);
	struct Node * newNode = makeNewNode("TOKEN","INT","INT",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return INT;
}
{INTL} {
	//SHOULD FILL IN
}
{FLOAT} { 
	parData.floatData=atof(yytext);
	struct Node * newNode = makeNewNode("TOKEN","FLOAT","FLOAT",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return FLOAT;
}
"package" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","PACKAGE","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return package;
}
"import" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","IMPORT","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return import;
}
"get" { 
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","GET","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return get;
}
"Long" {
	parData.stringData=strndup("Long",strlen("Long"));
	struct Node * newNode = makeNewNode("TOKEN","Long","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Long;
}
"Int" {
	parData.stringData=strndup("Int",strlen("Int"));
	struct Node * newNode = makeNewNode("TOKEN","Int","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Int;
}
"List" {
	parData.stringData=strndup("List",strlen("List"));
	struct Node * newNode = makeNewNode("TOKEN","LIST","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return List;
}
"Double" {
	parData.stringData=strndup("Double",strlen("Double"));
	struct Node * newNode = makeNewNode("TOKEN","Double","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Double;
}
"interface" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","INTERFACE","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return interface;
}
"abstract" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","ABSTRACT","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return abstract;
}
"override" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","OVERRIDE","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return override;
}
"null"	{
	parData.stringData=strndup("null",strlen("null"));
	struct Node * newNode = makeNewNode("TOKEN","NULL","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return null;
}
"Boolean"	{
	parData.stringData=strndup("Booelan",strlen("Booelan"));
	struct Node * newNode = makeNewNode("TOKEN","Boolean","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Boolean;
}
"Float"	{
	parData.stringData=strndup("Float",strlen("Float"));
	struct Node * newNode = makeNewNode("TOKEN","Float","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Float;
}
"if"	{
	parData.stringData=strndup("if",strlen("if"));
	struct Node * newNode = makeNewNode("TOKEN","IF","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return	IF;
}
"String" {
	parData.stringData=strndup("String",strlen("String"));
	struct Node * newNode = makeNewNode("TOKEN","String","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return String;
}
"Any"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","ANY","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Any;
}
"downTo" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","DOWNTO","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return downTo;
}
"step"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","STEP","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return step;
}
"fun"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","FUN","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return fun;
}
"Unit"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","UNIT","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Unit;
}
"return" {
	parData.stringData=strndup("return",strlen("return"));
	struct Node * newNode = makeNewNode("TOKEN","RETURN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return RETURN;
}
"else" {
	parData.stringData=strndup("else",strlen("else"));
	struct Node * newNode = makeNewNode("TOKEN","ELSE","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return ELSE;
}
"while" {	
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","WHILE","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return WHILE;
}
"for" {
	parData.stringData=strndup("for",strlen("for"));
	struct Node * newNode = makeNewNode("TOKEN","FOR","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return FOR;
}
"when" {
	parData.stringData=strndup("when",strlen("when"));
	struct Node * newNode = makeNewNode("TOKEN","WHEN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return WHEN;
}
"is" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","IS","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IS;
}
"in" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","IN","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IN;
}
"setOf" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","SETOf","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return setOf;

}
"listOf" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","LISTOF","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return listOf;
}
"var" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","VAR","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return VAR;
}
"val" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","VAL","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return VAL;
}
"class" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","CLASS","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return class;
}
{ID}	{
	char * str = strndup(yytext,yyleng);
	parData.stringData=str;
	struct Node * newNode = makeNewNode("TOKEN","ID","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IDENTIFIER;	
}
"="	{
	parData.stringData=strndup("=",strlen("="));
	struct Node * newNode = makeNewNode("TOKEN","ASSN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return ASSN;
}
"+="	{
	parData.stringData=strndup("+=",strlen("+="));
	struct Node * newNode = makeNewNode("TOKEN","PLUS_ASSN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return PLUS_ASSN;
}
"-="	{
	parData.stringData=strndup("-=",strlen("-="));
	struct Node * newNode = makeNewNode("TOKEN","MINUS_ASSN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MINUS_ASSN;
}
"*="	{
	parData.stringData=strndup("*=",strlen("*="));
	struct Node * newNode = makeNewNode("TOKEN","MULT_ASSN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MULT_ASSN;
}
"/="	{
	parData.stringData=strndup("/=",strlen("/="));
	struct Node * newNode = makeNewNode("TOKEN","DIV_ASSN","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIV_ASSN;
}
"+"	{ 	
	parData.stringData=strndup("+",strlen("+"));
	struct Node * newNode = makeNewNode("TOKEN","PLUS","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return PLUS; 
}
"-"	{ 
	parData.stringData=strndup("-",strlen("-"));
	struct Node * newNode = makeNewNode("TOKEN","MINUS","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MINUS;
}

"*"	{
	parData.stringData=strndup("*",strlen("*"));
	struct Node * newNode = makeNewNode("TOKEN","MULT","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MULT; 
}

"/"	{ 	
	parData.stringData=strndup("/",strlen("/"));
	struct Node * newNode = makeNewNode("TOKEN","DIV","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIV;
}
"{" 	{
	parData.stringData=strndup("{",strlen("{"));
	struct Node * newNode = makeNewNode("TOKEN","OPENA","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENA;
}
"}"	{ 	
	parData.stringData=strndup("}",strlen("}"));
	struct Node * newNode = makeNewNode("TOKEN","CLOSEA","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEA;
}
"["	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","OPENB","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENB;
}
"]"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","CLOSEB","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEB;
}
"("	{ 
	parData.stringData=strndup("(",strlen("("));
	struct Node * newNode = makeNewNode("TOKEN","OPENC","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENC;
}
")"	{ 
	parData.stringData=strndup(")",strlen(")"));
	struct Node * newNode = makeNewNode("TOKEN","CLOSEC","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEC;
}
"<Int>" { 	
	parData.stringData=strndup("Int",strlen("Int"));
	struct Node * newNode = makeNewNode("TOKEN","Int","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Int;
}
"<Float>" {
	parData.stringData=strndup("Float",strlen("Float"));
	struct Node * newNode = makeNewNode("TOKEN","Float","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Float;
}
"<Double>" {
	parData.stringData=strndup("Double",strlen("Double"));
	struct Node * newNode = makeNewNode("TOKEN","Double","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Double;
}
"<String>" {
	parData.stringData=strndup("String",strlen("String"));
	struct Node * newNode = makeNewNode("TOKEN","String","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return String;
}
"!"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","NOT","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return NOT;
}
"?"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","Q","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Q;
}
"->" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOKEN","ARROW","NULL",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Arrow;
}
"."	{
	parData.stringData=strndup(".",strlen("."));
	struct Node * newNode = makeNewNode("TOKEN","DOT","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DOT;
}
"=="    {
	parData.stringData=strndup("==",strlen("=="));
	struct Node * newNode = makeNewNode("TOKEN","SAME","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return SAME;
}
"!="	{
	parData.stringData=strndup("!=",strlen("!="));
	struct Node * newNode = makeNewNode("TOKEN","DIFF","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIFF;
}
"&&"	{
	parData.stringData=strndup("&&",strlen("&&"));
	struct Node * newNode = makeNewNode("TOKEN","AND","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return AND;
}
"||"	{
	parData.stringData=strndup("||",strlen("||"));
	struct Node * newNode = makeNewNode("TOKEN","OR","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OR;
}
":"	{ 	
	parData.stringData=strndup(":",strlen(":"));
	struct Node * newNode = makeNewNode("TOKEN","TYPE","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return TYPE;
}
","	{
	parData.stringData=strndup(",",strlen(","));
	struct Node * newNode = makeNewNode("TOKEN","COMMA","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return COMMA;
}
"\n"	{ 
	//return NL;
}
">"	{
	parData.stringData=strndup(">",strlen(">"));
	struct Node * newNode = makeNewNode("TOKEN",">","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LESS;
}
">="	{
	parData.stringData=strndup(">=",strlen(">="));
	struct Node * newNode = makeNewNode("TOKEN",">=","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LESS_EQ;
}
"<"	{
	parData.stringData=strndup("<",strlen("<"));
	struct Node * newNode = makeNewNode("TOKEN","<","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LARGER;
}
"<="	{ 	
	parData.stringData=strndup("<=",strlen("<="));
	struct Node * newNode = makeNewNode("TOKEN","<=","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LARGER_EQ;
}
"++"	{
	parData.stringData=strndup("++",strlen("++"));
	struct Node * newNode = makeNewNode("TOKEN","++","STRING",parData);
	enqueue(newNode);
	printNode(newNode);
	return INCRE;
}
"--"	{ 
	parData.stringData=strndup("--",strlen("--"));
	struct Node * newNode = makeNewNode("TOKEN","--","STRING",parData);
	enqueue(newNode);
	printNode(newNode);
	return DECRE;
}
{STRING}	{
	char * str = strndup(yytext,yyleng);
	yylval.str= str;
	parData.stringData=yylval.str;
	struct Node * newNode = makeNewNode("TOKEN","STRING","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return STRING;
}
%%
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


struct Node * makeNewNode(char * nodeType,char *detType, char * dataType,union Data data)
{
	struct Node * node = (struct Node*)malloc(sizeof(struct Node));
	node->nodeType = strndup(nodeType,strlen(nodeType));
	node->detType = strndup(detType,strlen(detType));
	node->dataType = strndup(dataType,strlen(dataType));
	if(strcmp(dataType,"INT")==0)
		node->data.intData=data.intData;
	else if(strcmp(dataType,"FLOAT")==0)
		node->data.floatData=data.floatData;
	else if(strcmp(dataType,"STRING")==0)
		node->data.stringData = strndup(data.stringData,strlen(data.stringData));
	else if(strcmp(dataType,"NULL")==0)
		node->data.boolean=0;		
	else
		node->data.boolean=0;
	node->childArr=NULL;
	node->childNum=0;
	node->nodeNum=count++;
	return node;
}
void printNode(struct Node * node)
{
	printf("%s |",node->nodeType);
	printf("%s |",node->detType);
	printf("%s |",node->dataType);
	if(strcmp(node->dataType,"INT")==0)
		printf("%d", node->data.intData);
	else if(strcmp(node->dataType,"FLOAT")==0)
		printf("%lf",node->data.floatData);
	else if(strcmp(node->dataType,"STRING")==0)
		printf("%s  ",node->data.stringData);
/*	printf("Node Type: %s  |      ",node->nodeType);
	printf("detail Type: %s |     ",node->detType);
	printf("data Type: %s |       ",node->dataType);
	if(strcmp(node->dataType,"INT")==0)
		printf("data: %d", node->data.intData);
	else if(strcmp(node->dataType,"FLOAT")==0)
		printf("data: %lf",node->data.floatData);
	else if(strcmp(node->dataType,"STRING")==0)
		printf("data: %s | ",node->data.stringData);
*/
//	printf("     child: %d | nodeNum: %d ",node->childNum,node->nodeNum);
	printf("\n");
}
void makechildArr(struct Node * node, int num)
{	
	node->childArr=(struct Node **)malloc(sizeof(struct Node *)*num);
	node->childNum=num;
}
void travelTree(struct Node *node,int depth)
{
	for(int j=0;j<depth;j++)
		printf(" ");
	printNode(node);
	int i=0;

	if(strcmp(node->nodeType,"GOAL")==0)
		if(node->childNum==2)
		{
			travelTree(node->childArr[0],depth+1);
			travelTree(node->childArr[1],depth);
			return;
		}		
	if(strcmp(node->nodeType,"BODY")==0 || strcmp(node->nodeType,"WHEN_BODY")==0 || strcmp(node->nodeType,"LAMBDA2")==0)
		if(node->childNum==2)
		{
			travelTree(node->childArr[0],depth+1);
			travelTree(node->childArr[1],depth);
			return;
		}		
	if(strcmp(node->nodeType,"PAR")==0)
		if(node->childNum==3)
		{
			travelTree(node->childArr[0],depth+1);
			travelTree(node->childArr[1],depth+1);
			travelTree(node->childArr[2],depth);
			return;
		}		
	if(strcmp(node->nodeType,"DEF_PAR")==0)
		if(node->childNum==5)
		{
			travelTree(node->childArr[0],depth+1);
			travelTree(node->childArr[1],depth+1);
			travelTree(node->childArr[2],depth+1);
			travelTree(node->childArr[3],depth+1);
			travelTree(node->childArr[4],depth);
			return;
		}		
	while(i<node->childNum)
	{	
		travelTree(node->childArr[i],depth+1);
		i++;
	}		
}
void queue_init()
{
	head=0;
	tail=0;
	//head==tail->empty
}
void enqueue(struct Node * node)
{
	queue[tail]=node;
	tail=(tail+1)%10000;
}
struct Node * dequeue()
{
	head=(head+1)%10000;
	return queue[head-1];	
}

