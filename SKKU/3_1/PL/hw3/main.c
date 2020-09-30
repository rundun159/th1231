#include <stdio.h>
#include <stdlib.h>
#include "Kotlin.h"
#include "k2j.c"
#include <unistd.h>
#include <fcntl.h>
extern FILE *yyin;
extern int yy_scan_string(const char *);
extern int yyparse(void);

int main(int argc, char** argv) 
{
/*
 	FILE* fp = fopen(argv[1], "a");
 	yyin=fp;
	string str = argv[1];
	str += '\n';
	yy_scan_string(str.c_str());
*/
/* 	FILE* fp = fopen(argv[1], "a");
	yyin=fp;
*/	
	int fd=open(argv[1],O_RDONLY);
	printf("%d\n",fd);
	dup2(fd,0);
	int fd2;
	if( argc== 3)
	{
		fd2=open(argv[2],O_WRONLY|O_CREAT);
	}
	else
	{
		fd2=open("output.java",O_WRONLY|O_CREAT|O_TRUNC,0644);
	}
	queue_init();
	yyparse();
/*	printf("=========================================Travel Started");
	printf("=================================================\n");
	travelTree(rootNode,0);
*/	
	printf("=========================================Java");
	printf("=================================================\n");
	kotlinFunArr=(struct Node**)malloc(sizeof(struct Node *)*1000);
	printf("kotlinFunArr\n");
	travelForJava(rootNode);
	printf("kotlinFunArr2\n");
	k2jTree(rootNode);
	printf("kotlinFunArr3\n");
	travelTree(jrootNode,0);
	printf("kotlinFunArr4\n");
	printf("%d \n",funNum);
	dup2(fd2,1);
	printf("import java.util.*;\n");
	printOutAsJava(jrootNode);

//	printToken(jrootNode);
	return 0;
}
