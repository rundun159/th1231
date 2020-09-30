%{
#include <stdlib.h>
#include <string.h>
#include "expr.tab.h"
#include "calc.h"
extern int yyerror(const char*);
union Data parData;
%}
INT [0-9]+
INTL [0-9]+L
FLOAT [0-9]+"."[0-9]+
ID [a-zA-Z][a-zA-Z0-9]*
SPACE [\t\r]+
COMMENT \/\/[^\t\r\n]*
STRING \"[^\t\r\"]*\"
%%
"\n"	{	
}
{SPACE} { }
{COMMENT} {
}
{INT}	{ 
	parData.intData=atoi(yytext);
	struct Node * newNode = makeNewNode("TOCKEN","INT","INT",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return INT;
}
{INTL} {
	//SHOULD FILL IN
}
{FLOAT} { 
	parData.floatData=atof(yytext);
	struct Node * newNode = makeNewNode("TOCKEN","FLOAT","FLOAT",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return FLOAT;
}
"package" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","PACKAGE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return package;
}
"import" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","IMPORT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return import;
}
"get" { 
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","GET","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return get;
}
"Long" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Long","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Long;
}
"Int" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Int","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Int;
}
"List" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LIST","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return List;
}
"Double" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Double","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Double;
}
"interface" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","INTERFACE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return interface;
}
"abstract" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","ABSTRACT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return abstract;
}
"override" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","OVERRIDE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return override;
}
"null"	{
	parData.intData=0;
	struct Node * newNode = makeNewNode("TOCKEN","NULL","INT",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return null;
}
"Boolean"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Boolean","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Boolean;
}
"Float"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Float","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Float;
}
"if"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","IF","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return	IF;
}
"String" {
		parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","String","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return String;
}
"Any"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","ANY","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Any;
}
"downTo" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DOWNTO","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return downTo;
}
"step"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","STEP","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return step;
}
"fun"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","FUN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return fun;
}
"Unit"	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","UNIT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Unit;
}
"return" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","RETURN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return RETURN;
}
"else" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","ELSE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return ELSE;
}
"while" {	
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","WHILE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return WHILE;
}
"for" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","FOR","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return FOR;
}
"when" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","WHEN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return WHEN;
}
"is" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","IS","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IS;
}
"in" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","IN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IN;
}
"listOf" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LISTOF","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return listOf;
}
"var" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","VAR","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return VAR;
}
"val" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","VAL","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return VAL;
}
"class" {
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","CLASS","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return class;
}
{ID}	{
	char * str = strndup(yytext,yyleng);
	parData.stringData=str;
	struct Node * newNode = makeNewNode("TOCKEN","ID","STRING",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return IDENTIFIER;	
}
"="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","ASSN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return ASSN;
}
"+="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","PLUS_ASSN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return PLUS_ASSN;
}
"-="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","MINUS_ASSN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MINUS_ASSN;
}
"*="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","MULT_ASSN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MULT_ASSN;
}
"/="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DIV_ASSN","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIV_ASSN;
}
"+"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","PLUS","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return PLUS; 
}
"-"	{ 
		parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","MINUS","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MINUS;
}

"*"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","MULT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return MULT; 
}

"/"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DIV","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIV;
}
"{" 	{
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","OPENA","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENA;
}
"}"	{ 	
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","CLOSEA","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEA;
}
"["	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","OPENB","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENB;
}
"]"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","CLOSEB","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEB;
}
"("	{ 
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","OPENC","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OPENC;
}
")"	{ 
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","CLOSEC","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return CLOSEC;
}
"<Int>" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Int","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Int;
}
"<Float>" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Float","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Float;
}
"<Double>" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Double","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Double;
}
"<String>" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","String","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return String;
}
"!"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","NOT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return NOT;
}
"?"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","Q","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Q;
}
"->" { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","ARROW","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return Arrow;
}
"."	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DOT","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DOT;
}
"=="    { 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","SAME","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return SAME;
}
"!="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DIFF","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return DIFF;
}
"&&"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","AND","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return AND;
}
"||"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","OR","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return OR;
}
":"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","TYPE","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return TYPE;
}
","	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","COMMA","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return COMMA;
}
"\n"	{ 
	//return NL;
}
">"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LESS","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LESS;
}
">="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LESS_EQ","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LESS_EQ;
}
"<"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LARGER","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LARGER;
}
"<="	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","LARGER_EQ","",parData);
	printNode(newNode);
	yylval.ptr=newNode;
	return LARGER_EQ;
}
"++"	{ 	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","INCRE","",parData);
	enqueue(newNode);
	printNode(newNode);
	return INCRE;
}
"--"	{ 
	parData.boolean=0;
	struct Node * newNode = makeNewNode("TOCKEN","DECRE","",parData);
	enqueue(newNode);
	printNode(newNode);
	return DECRE;
}
{STRING}	{
	char * str = strndup(yytext,yyleng);
	yylval.str= str;
	parData.stringData=yylval.str;
	struct Node * newNode = makeNewNode("TOCKEN","STRING","STRING",parData);
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
	printf("%s ",node->detType);
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

