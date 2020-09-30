#Makefiel for expression calculator

TARGET=Kotlin2Java
CFG=Kotlin
LEX_SRC=$(CFG).l
YACC_SRC=$(CFG).y
SRC=main.c

all:
		flex $(LEX_SRC)
		bison -d -v -t $(YACC_SRC) 
		gcc -o $(TARGET) $(CFG).tab.c lex.yy.c $(SRC) -lfl -DYYDEBUG -g  
clean:
		rm -f lex.yy.c $(CFG).tab.c $(CFG).tab.h $(TARGET)
