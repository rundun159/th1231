%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <iostream>
extern int yylex(void);
extern void yyterminate();
extern int yyerror(const char *s);
void assign(char *s, double eval);
double findvalue(char *s);
extern double valuesV[200];
extern char* valuesID[200];
extern int top; 
%}
%union {
	double dval;
	char * str;	
	int ival;
}
%token <dval> NUMBER 	
%token <str> ID
%token EOL OPEN CLOSE ASSN
%left  PLUS MINUS
%left  MULT DIV
%nonassoc UMINUS
%nonassoc UPLUS
%type <dval> eval 
%type <dval> expr
%type <dval> factor
%%
/* Rules */
goal: 	goal eval 	 { printf("%lf\n", $2);}
    |   eval		{ printf("%lf\n", $1);}
    |   ID   ASSN eval  { assign($1,$3); free $1;
}
    |   goal ID ASSN eval { assign($2,$4); free $2;
}
 	;
eval:	expr EOL	{ $$=$1;
	}
	;
expr:	
	ID		{ $$ = findvalue($1); free $1;}
    |   PLUS ID 	{ $$ = findvalue($2); free $2;}
    |   MINUS ID 	{ $$ = (-1)*findvalue($2); free $2;}
    |	expr PLUS expr	{ $$ = $1 + $3;
    	}
    |	expr MINUS expr	{ $$ = $1 - $3;
	} 
    |	expr MULT expr	{ $$ = $1 * $3;
	} 
    |	expr DIV expr	{
//			if($3==0) 
//			{ yyerror("Divided by zero");}
			$$ = $1 / $3;
	} 
    |  	OPEN expr CLOSE  { $$ = $2;
	}
    |	factor		{ $$ = $1;
	} 
    ;
factor: MINUS factor {  $$ = -$2;
	}
    |   PLUS factor  {  $$ = $2;
	}
    |	NUMBER		{ $$ = $1;
        }
    ;

%%
/* User code */
int yyerror(const char *s)
{
	return printf("%s\n", s);
}
double findvalue(char *s)
{
	for(int i=0;i<top;i++)
		if(strcmp(s,valuesID[i])==0)
			return valuesV[i];
}
void assign(char *s, double eval)
{
	for(int i=0;i<top;i++)
		if(strcmp(s,valuesID[i])==0)
		{
			valuesV[i]=eval;
			return;
		}
	valuesID[top]=strndup(s,strlen(s));
	valuesV[top]=eval;
	top=top+1;
	return;
}



