//#include <iostream>
//#include <vector>
//#include <numeric>
//using namespace std;
//typedef vector<vector<int>> Matrix;
//void fill_input_V(vector<int>& input_V, int n);
//void fill_index_Mat(Matrix& index_Mat, vector<bool> & isBackWard,int n);
//int retBackWards(const Matrix& index_Mat, int n);
//void printMat(const Matrix& mat);
//int main()
//{
//	freopen("input.txt", "r", stdin);
////	freopen("output.txt", "w", stdout);
//	int TC; cin >> TC;
//	for (int cases = 0; cases < TC; cases++)
//	{
//		int n; cin >> n;
//		vector<int> input_V(n);
//		Matrix index_Mat(n + 1, vector<int>(2,0));
//		vector<bool> isLarge(n+1,false);
//	//	fill_input_V(input_V, n);
//		fill_index_Mat(index_Mat, isLarge, n);
////		printMat(index_Mat);
//		cout << "#" << cases + 1 << " " << accumulate(isLarge.begin(),isLarge.end(),0) << endl;
//	}
//}
//void fill_input_V(vector<int>& input_V, int n)
//{
//	for (int i = 0; i < n; i++)
//		cin >> input_V[i];
//}
//void fill_index_Mat(Matrix& index_Mat, vector<bool>& isBackWard, int n)
//{
//	int book;
//	for (int i = 0; i < n; i++)
//	{
//		cin >> book;
//		index_Mat[book][0] = i;
//		index_Mat[book-1][1] = i;
//		if (index_Mat[book][0] >= index_Mat[book][1])
//		{
//			isBackWard[book] = true;
//		}
//		else
//		{
//			isBackWard[book] = false;
//		}
//		if (index_Mat[book-1][0] >= index_Mat[book-1][1])
//			isBackWard[book-1] = true;
//		else
//			isBackWard[book-1] = false;
//	}
//	
//}
//int retBackWards(const Matrix& index_Mat, int n)
//{
//	int ret = 1;
//	for (int i = 1; i < n; i++)
//	{
//		if (index_Mat[i][1] < index_Mat[i][0])
//			ret++;
//	}
//	return ret;
//}
//void printMat(const Matrix& mat)
//{
//	for (int i = 0; i < mat.size(); i++)
//	{
//		cout <<mat[i][0]<<" "<<mat[i][1]<< endl;
//	}
//}
