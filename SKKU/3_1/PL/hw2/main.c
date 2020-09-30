#include <stdio.h>
#include <stdlib.h>
#include "calc.h"
extern int yyparse(void);

void main()
{
	queue_init();
	yyparse();
	printf("=========================================Travel Started");
	printf("=================================================\n");
	travelTree(rootNode,0);
}
