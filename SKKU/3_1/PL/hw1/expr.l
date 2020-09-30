%{
#include <stdlib.h>
#include <string.h>

#include "expr.tab.h"

extern int yyerror(const char*);
double valuesV[1000];
char* valuesID[1000];
int top=0; 
%}

NUMBER ([0-9]+|[0-9]+"."[0-9]+)
ID [a-zA-Z]+[a-zA-Z0-9]*
SPACE [ \t\r\n]+

%%


{NUMBER} { yylval.dval = atof(yytext);
	return NUMBER;
}
{ID}	{ 
		yylval.str= strndup(yytext,yyleng);
		return ID;	
	}
"="	{ return ASSN;
}
"+"	{ return PLUS; 
}

"-"	{ return MINUS;
}

"*"	{ return MULT; 
}

"/"	{ return DIV;
}

";"	{ return EOL;
}
"("	{ return OPEN;
}
")"	{ return CLOSE;
}

{SPACE} { /* ignore */ 
}

.	{  
		printf("%d\n", yytext[0]);
           yyerror("unexpected input"); 
}


%%
/* User code */

