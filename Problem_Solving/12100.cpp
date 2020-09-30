////두번째 삼성 기출문제.
//// 3시간 안에 풀긴 함.
//// 알고리즘을 깊게 생각하지 못한게 패인
//// queue 에 넣을때 Node *를 넣자 
//// heap에 만들도록 struct 객체를 heap에 쌓자
//// 방향이 4방향으로 같을때 
//// 하나를 진짜 잘 짜서 차이만 잘 조정하거나 
//// 여기에서는 prev, now가 되는것만 잘 비교하면 됐다.
//
//#include<iostream>
//#include<queue>
//#include<vector>
//#include<algorithm>
//using namespace std;
//int N;
//typedef vector<vector<int>> Mat;
//Mat input_mat;
//int search_way_arr[4][2] = { {0,-1},{0,1},{1,0},{-1,0} };
//typedef struct NODE
//{
//	Mat mat;
//	int tryNum;
//	NODE() {
//
//	}
//}Node ;
//Node* init_Node(Mat mat, int tryNum)
//{
//	Node *newNode =new Node();
//	newNode->mat = mat;
//	newNode->tryNum = tryNum;
//	return newNode;
//}
//queue<Node> q;
//int retMax(const Mat& mat)
//{
//	int ret = -1;
//	for (int i = 0; i < N; i++)
//		for (int j = 0; j < N; j++)
//			if (mat[i][j] > ret)
//				ret = mat[i][j];
//	return ret;
//}
//Mat shift_Mat(int way,const Mat & mat)
//{
//	//way 0 : right , 1: left 
//	Mat newMat = Mat(N, vector<int>(N, 0));
//	if (way == 0)
//	{
////		cout << "Way : " << way << endl;
//		for (int i = 0; i < N; i++)
//		{
//			vector<int> result = vector<int>(N,0);
//			int idx = 0;
//			int prev_num = 0;
//			bool changed = false;
//			for (int j = N - 1; j >= 0; j--)
//			{
//				int now_num = mat[i][j];
//				if (now_num == 0)
//				{
//					continue;
//				}
//				if ((prev_num != 0) && (prev_num == now_num))
//				{
//					changed = true;
//					result[idx++] = prev_num * 2;
//					prev_num = 0;
//				}
//				else if ((prev_num != now_num) && (prev_num != 0))
//				{
//					changed = true;
//					result[idx++] = prev_num;
//					prev_num = now_num;
//				}
//				else
//				{
//					prev_num = now_num;
//				}
//			}
//			if ((prev_num != 0))
//				result[idx++] = prev_num;
//			for (int j = 0; j < N; j++)
//			{
//				newMat[i][N - j - 1] = result[j];
//			}
//		}
//	}
//	else if (way == 1)
//	{
//		//		cout << "Way : " << way << endl;
//		for (int i = 0; i < N; i++)
//		{
//			vector<int> result = vector<int>(N, 0);
//			int idx = 0;
//			int prev_num = 0;
//			bool changed = false;
//			for (int j = 0; j < N; j++)
//			{
//				int now_num = mat[i][j];
//				if (now_num == 0)
//				{
//					continue;
//				}
//				if ((prev_num != 0) && (prev_num == now_num))
//				{
//					changed = true;
//					result[idx++] = prev_num * 2;
//					prev_num = 0;
//				}
//				else if ((prev_num != now_num) && (prev_num != 0))
//				{
//					changed = true;
//					result[idx++] = prev_num;
//					prev_num = now_num;
//				}
//				else
//				{
//					prev_num = now_num;
//				}
//			}
//			if ((prev_num != 0))
//				result[idx++] = prev_num;
//			for (int j = 0; j < N; j++)
//			{
//				newMat[i][j] = result[j];
//			}
//		}
//	}
//	else if (way == 2)
//	{
//		//		cout << "Way : " << way << endl;
//		for (int i = 0; i < N; i++)
//		{
//			vector<int> result = vector<int>(N, 0);
//			int idx = 0;
//			int prev_num = 0;
//			bool changed = false;
//			for (int j = 0; j <N  ; j++)
//			{	
//				changed = false;
//				int now_num = mat[j][i];
//				if (now_num == 0)
//				{
//					continue;
//				}
//				if ((prev_num != 0) && (prev_num == now_num))
//				{
//					changed = true;
//					result[idx++] = prev_num * 2;
//					prev_num = 0;
//				}
//				else if ((prev_num != now_num)&&(prev_num!=0))
//				{
//					changed = true;
//					result[idx++] = prev_num;
//					prev_num = now_num;
//				}
//				else
//				{
//					prev_num = now_num;
//				}
//			}
//			if ((prev_num != 0))
//				result[idx++] = prev_num;
//			for (int j = 0; j < N; j++)
//			{
//				newMat[j][i] = result[j];
//			}
//		}
//	}
//	else
//	{
//	//		cout << "Way : " << way << endl;
//		for (int i = 0; i < N; i++)
//		{
//			vector<int> result = vector<int>(N, 0);
//			int idx = 0;
//			int prev_num = 0;
//			bool changed = false;
//			for (int j = N - 1; j >= 0; j--)
//			{
//				changed = false;
//				int now_num = mat[j][i];
//				if (now_num == 0)
//				{
//					continue;
//				}
//				if ((prev_num != 0) && (prev_num == now_num))
//				{
//					changed = true;
//					result[idx++] = prev_num * 2;
//					prev_num = 0;
//				}
//				else if ((prev_num != now_num) && (prev_num != 0))
//				{
//					changed = true;
//					result[idx++] = prev_num;
//					prev_num = now_num;
//				}
//				else
//				{
//					prev_num = now_num;
//				}
//			}
//			if ((prev_num != 0))
//				result[idx++] = prev_num;
//			for (int j = 0; j < N; j++)
//			{
//				newMat[N - j - 1][i] = result[j];
//			}
//		}
//	}
//	return newMat;
//}
//void print_Mat(const Mat& mat);
//int pickNode_shift()
//{
//	Node first_node= q.front();
//	q.pop();
////	cout << "tryNum: " << first_node.tryNum << endl;
////	print_Mat(first_node.mat);
//	if (first_node.tryNum == 5)
//	{
//		return retMax(first_node.mat);
//	}
//	else
//	{
//		Mat nowMat = first_node.mat;
//		for (int i = 0; i < 4; i++)
//		{
//			Mat newMat = shift_Mat(i, nowMat);
////			print_Mat(newMat);
//			Node newNode = *init_Node(newMat, first_node.tryNum + 1);
//			q.push(newNode);
//		}
//		return -1;
//	}
//}
//void print_Mat(const Mat& mat)
//{
//	cout << "Print" << endl;
//	for (int i = 0; i < N; i++)
//	{
//		for (int j = 0; j < N; j++)
//			cout << mat[i][j] << " ";
//		cout << endl;
//	}
//}
//int main()
//{
//	freopen("input.txt", "r", stdin);
////	freopen("output.txt", "w", stdout);
//	cin >> N;
//	//cout << "N : " << N << endl;
//	Mat input_mat = Mat(N, vector<int>(N,0));
//	for (int i = 0; i < N; i++)
//		for (int j = 0; j < N; j++)
//			cin >> input_mat[i][j];
//	Node firstNode = *init_Node(input_mat, 0);
//	int maxNum = -1;
//	q.push(firstNode);
//	int index = 0;
//	while (!q.empty())
//	{
//		//cout << "index: " << index << endl;
//		maxNum = max(maxNum, pickNode_shift());
//		index++;
//	}
//	cout << maxNum << endl;
//}
