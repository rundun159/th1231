////public:::꼭써야함
////40분 걸림. 60% 정답률이니까 꽤 쉬운 문제임. 30분 정도 안에 했어야 함.
////dfs가 좀 오랜만이라 당황한 느낌은 있었다
////꼭 해놓는 루틴이 없어서 그런건데,
////들어가기 전에 조작하고, 나와서 조작하는 툴로 해야한다.
//
//#include<iostream>
//#include<vector>
//#include<limits.h>
//#include<algorithm>
//using namespace std;
//class node
//{
//public:
//	int min_score;
//	int max_score;
//	long int sum;
//	node(int _min, int _max, int _sum):min_score(_min),max_score(_max),sum(_sum) 
//	{
//	}
//};
//int n, l, r, x;
//int cntcase(int min_idx, int num_left, node* node);
//vector<int> scores;
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	cin >> n >> l >> r >> x;
//	scores = vector<int>(n, 0);	
//	for (int i = 0; i < n; i++)
//		cin >> scores[i];
//	sort(scores.begin(), scores.end());
//	int sum = 0;
//	for (int num_left = 2; num_left <= n; num_left++)
//	{
//		f	or (int start_idx = 0; start_idx <= n - num_left; start_idx++)
//		{
//			node* node = new node(scores[start_idx], 0, 0);
//			node->sum += scores[start_idx];
//			node->max_score = scores[start_idx];
//			sum += cntcase(start_idx, num_left-1, node);
//			delete(node);
//		}
//	}
//	cout << sum << endl;		
//}
//int cntcase(int min_idx, int num_left, node* node)
//{
//	if (num_left == 0)
//	{
////		cout << node->max_score << " " << node->min_score << " " << node->sum << endl;
//		if (node->sum >= l && node->sum <= r)
//			if (node->max_score - node->min_score >= x)
//				return 1;
//		return 0;
//	}
//	int ret = 0;
//	int last_idx = n - num_left;
//	for (int idx = min_idx + 1; idx <= last_idx; idx++)
//	{
//		node->sum += scores[idx];
//		node->max_score = scores[idx];
//		ret += cntcase(idx, num_left - 1, node);
//		node->sum -= scores[idx];
//		node->max_score = scores[min_idx];
//	}
//	return ret;
//}
