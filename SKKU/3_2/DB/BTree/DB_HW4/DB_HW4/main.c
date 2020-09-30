#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BTREE.h"

int main(void) {
	printf(" Max degree = 3 \n");
	printf("Insert\(1, 3, 7, 10, 11, 13, 14, 15, 18, 16, 19, 24, 25, 26\)\n");
	BTreeInit(3); // Max degree 3
	insertElement(1); 
	insertElement(3);
	insertElement(7);
	insertElement(10);
	insertElement(11);
	insertElement(13);
	insertElement(14); //where is 3???
	insertElement(15);
	insertElement(18);
	insertElement(16);
	insertElement(19);
	insertElement(24);
	insertElement(25);
	insertElement(26);
	printTree();
	printf("\n\nMax degree = 3 \n");
	printf("Insert\(1, 3, 7, 10, 11, 13, 14, 15, 18, 16, 19, 24, 25, 26\)Remove\(13\)\n");
	removeElement(13);
	printTree();



	printf("It's insertion\n");
 	BTreeInit(4); // Max degree 4
	insertElement(1); insertElement(3); insertElement(7); insertElement(10); insertElement(11); insertElement(13);	insertElement(14);
	insertElement(15); insertElement(18); insertElement(16); insertElement(19); insertElement(24); insertElement(25); insertElement(26);
	printTree();

	removeElement(13);
	printTree();


	BTreeInit(5); // Max degree 5
	insertElement(1); insertElement(2); insertElement(3); insertElement(4);
	printTree();
	printf("\n");

	printf("====== split ======\n");
	insertElement(5); // split
	printTree();
	printf("\n");

	printf("====== balanced tree ======\n");
	insertElement(6); insertElement(7); insertElement(8); insertElement(9);
	insertElement(10); insertElement(11); insertElement(12); insertElement(13);
	insertElement(14); insertElement(15); insertElement(16); insertElement(17);
	printTree();
	printf("\n");

	printf("====== merge ======\n");
	removeElement(12); // merge
	printTree();
	printf("\n");

	insertElement(12);
	printTree();
	printf("\n");

	printf("====== remove root ======\n");
	removeElement(9); // remove root
	printTree();
	printf("\n");


	printf("====== remove leaf node ======\n");
	removeElement(11); // remove leaf node
	printTree();
	printf("\n");




	return 0;
}