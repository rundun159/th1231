//#include<iostream>
//#include<vector>
//#include<string>
//using namespace std;
//#define A_NUM 4
//#define A_SIZE 8
//typedef vector<int> A; //톱니바퀴
//typedef vector<A> V_A; //톱니바퀴 arr
//int k;
//typedef pair<int, int> Rotate;
//vector<Rotate> r;
//void rotate(A& a, int r);
//int ret_Rotate_from_Right(const A& a1, const A& a2, int r);
//int ret_Rotate_from_Left(const A& a1, const A& a2, int r);
//void print_V_A(const V_A& v_a);
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	V_A v_a= V_A(A_NUM,A(A_SIZE,0));
//	for (int i = 0; i < A_NUM; i++)
//	{
//		string str;
//		cin >> str;
//		for (int j = 0; j < A_SIZE; j++)
//		{
//			v_a[i][j] = str[j] - '0';
//		}
//	}
//	cin >> k; 
//	r = vector<Rotate>(k);
//	for (int i = 0; i < k; i++)
//		cin >> r[i].first >> r[i].second;
//	for (int i = 0; i < k; i++)
//	{
//		vector<int> R(4, 0);
//		int rotate_idx = r[i].first - 1;
//		int rotate_way = r[i].second;
//		R[rotate_idx] = rotate_way;
//		if (rotate_idx != 0)
//		{
//			for (int left = rotate_idx - 1; left >= 0; left--)
//			{
//				R[left] = ret_Rotate_from_Right(v_a[left], v_a[left + 1], R[left + 1]);
//			}
//		}
//		if (rotate_idx != (A_NUM - 1))
//		{
//			for (int right = rotate_idx + 1; right < A_NUM; right++)
//			{
//				R[right] = ret_Rotate_from_Right(v_a[right-1], v_a[right], R[right-1]);
//			}
//		}
//		for (int i = 0; i < A_NUM; i++)
//		{
//			rotate(v_a[i], R[i]);
//		}
//		//cout << "rotate " << i + 1 << endl;
//		//print_V_A(v_a);
//	}
//	cout << v_a[0][0] * 1 + v_a[1][0] * 2 + v_a[2][0] * 4 + v_a[3][0] * 8 << endl;
//	return 0;
//} 
//void rotate(A& a, int r)	//r : rotate 방향. 1 : 시계, -1 : 반시계, 0 안움직임
//{
//	A temp= A(8);
//	if (r == 0)
//	{
//		return;
//	}
//	else if (r == 1)
//	{
//		for (int i = 0; i < A_SIZE; i++)
//		{
//			temp[(i + 1) % A_SIZE] = a[i];
//		}
//	}
//	else if (r == -1)
//	{
//		for (int i = 0; i < A_SIZE; i++)
//		{
//			temp[(i +7 ) % A_SIZE] = a[i];
//		}
//	}
//	for (int i = 0; i < A_SIZE; i++)
//	{
//		a[i] = temp[i];
//	}
//	return;
//}
//int ret_Rotate_from_Right(const A& a1, const A& a2, int r) //r : 회전 방향 of a2
//{
//	int RIGHT_IDX = 6, LEFT_IDX = 2;
//	if (r == 0)
//		return 0;
//	else
//	{
//		if (a1[LEFT_IDX] == a2[RIGHT_IDX])
//			return 0;
//		else
//		{
//			return -1 * r;
//		}
//	}		
//}
//int ret_Rotate_from_Left(const A& a1, const A& a2, int r) //r : 회전 방향 of a1
//{
//	int RIGHT_IDX = 6, LEFT_IDX = 2;
//	if (r == 0)
//		return 0;
//	else
//	{
//		if (a1[LEFT_IDX] == a2[RIGHT_IDX])
//			return 0;
//		else
//		{
//			return -1 * r;
//		}
//	}
//}
//void print_V_A(const V_A& v_a)
//{
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 8; j++)
//			cout << v_a[i][j];
//		cout << endl;
//	}
//}
