////25분 걸림.
////
//
//#include<iostream>
//#include<algorithm>
//#include<vector>
//using namespace std;
//const int MAX_LONG =2000001;
//int n;
//vector<int> num_v;
//int cache[MAX_LONG];
//void check_Sum(int start_idx, int num_left, int sum);
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	cin >> n;
//	num_v = vector<int>(n);
//	for (int i = 0; i < n; i++)
//		cin >> num_v[i];
//	for (int num_left = 1; num_left <= n; num_left++)
//	{
//		for (int start_idx = -1; start_idx <= n - num_left; start_idx++)
//			check_Sum(start_idx, num_left, 0);
//	}
//	for(int i=1;i<MAX_LONG;i++)
//		if (!cache[i])
//		{
//			cout << i << endl;
//			return 0;
//		}
//}
//void check_Sum(int start_idx, int num_left, int sum) //이 함수 이후로 num_left개수 만큼 더해주면 됨. start_idx 를 제외한 이후부터 더해줌.
//{
//	if (num_left == 0)
//	{
//		cache[sum] = 1;
//		return;
//	}
//	int last_idx = n - num_left;
//	for (int next_idx = start_idx + 1; next_idx <= last_idx; next_idx++)
//	{
//		check_Sum(next_idx, num_left - 1, sum + num_v[next_idx]);
//	}
//	return;
//}
