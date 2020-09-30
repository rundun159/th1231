#include <stdio.h>
#include <stdlib.h>

extern int yyparse(void);

void main()
{
	yyparse();
}
