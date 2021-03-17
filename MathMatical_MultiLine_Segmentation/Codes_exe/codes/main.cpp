#include "LineSegmentation.hpp"
#pragma warning(disable:4996)
#define _CRT_SECURE_NO_WARNINGS



//./Image2Lines (get_lines) {input} {output}
int main(int argc,char *argv[])
{

	LineSegmentation line_segmentation(
		argv[1],
		argv[2]);
	line_segmentation.rotate_img();
	line_segmentation.segment();

	//scanf("%*d");

	return 0;
}


