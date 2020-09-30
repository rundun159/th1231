//#include <string>
//#include <vector>
//#include <algorithm>
//using namespace std;
//typedef struct Node
//{
//	int n;
//	vector<int> weak;
//	vector<int> dist;
//};
//typedef struct Temp
//{
//	int m;
//	int now_m;
//	int d_idx;
//	int start_w_idx;
//	vector<int> check_weak;
//	vector<int> d_idx_list;
//	vector<int> start_w_idx_list;
//	int way;
//};
//int solution(int n, vector<int> weak, vector<int> dist) {
//	Node node;
//	node.weak = weak;
//	node.dist = dist;
//	node.n = n;
//	int answer = 0;
//	return answer;
//}
//bool check_condition(Temp const& temp);
//bool use_m(int m, vector<int>const& weak, vector<int>const& dist);
//void check_weak(Node const& node, Temp& temp);
//void uncheck_weak(Node const& node, Temp& temp);
//int solution(int n, vector<int> weak, vector<int> dist) {
//	sort(dist.begin(), dist.end(), greater<>());
//	int answer = -1;
//	for (int i = 1; i < dist.size(); i++)
//	{
//		if (use_m(i, weak, dist))
//			return i;
//	}
//	return -1;
//}
//bool use_m(int m, vector<int>const & weak, vector<int>const & dist)
//{
//
//}
//void check_weak(Node const & node, Temp & temp) 
//{
//	int start_pos = node.weak[temp.start_w_idx];
//	int dist = node.dist[temp.d_idx];
//	int end_pos;
//	if (temp.way == 1)
//	{
//		end_pos = (start_pos + dist)%node.n;
//	}
//	else if (temp.way==0)
//	{
//		end_pos = start_pos;
//		if (start_pos < dist)
//			end_pos += node.n;
//		end_pos -= dist;
//	}
//	for (int i = 0; i < node.weak.size(); i++)
//	{
//		if (temp.way == 1)
//		{
//			if (node.weak[i] <= end_pos)
//				temp.check_weak[i] +=1;
//			else if (node.weak[i] >= start_pos)
//				temp.check_weak[i] += 1;
//		}
//		else if (temp.way == 0)
//		{
//			if (node.weak[i] <= start_pos)
//				temp.check_weak[i] += 1;
//			else if (node.weak[i] >= end_pos)
//				temp.check_weak[i] += 1;
//		}
//	}
//}
//void uncheck_weak(Node const& node, Temp& temp)
//{
//	int start_pos = node.weak[temp.start_w_idx];
//	int dist = node.dist[temp.d_idx];
//	int end_pos;
//	if (temp.way == 1)
//	{
//		end_pos = (start_pos + dist) % node.n;
//	}
//	else if (temp.way == 0)
//	{
//		end_pos = start_pos;
//		if (start_pos < dist)
//			end_pos += node.n;
//		end_pos -= dist;
//	}
//	for (int i = 0; i < node.weak.size(); i++)
//	{
//		if (temp.way == 1)
//		{
//			if (node.weak[i] <= end_pos)
//				temp.check_weak[i] -= 1;
//			else if (node.weak[i] >= start_pos)
//				temp.check_weak[i] -= 1;
//		}
//		else if (temp.way == 0)
//		{
//			if (node.weak[i] <= start_pos)
//				temp.check_weak[i] -= 1;
//			else if (node.weak[i] >= end_pos)
//				temp.check_weak[i] -= 1;
//		}
//	}
//}
//bool check_condition(Temp const& temp)
//{
//	int n = temp.check_weak.size();
//	int sum = 0;
//	for (int i = 0; i < temp.check_weak.size(); i++)
//		if(temp.check_weak[i]!=0)
//			sum += 1;
//	if (n == sum)
//		return true;
//	else
//		return false;
//}
//void fill(Temp & temp)
//{
//	if (temp.m == temp.now_m)
//		return;
//	
//}
//void pop(Node const & node, Temp& temp)
//{
//	temp.d_idx_list.pop_back();
//	temp.start_w_idx_list.pop_back();
//	uncheck_weak(node,temp);
//}
//typedef struct Temp
//{
//	int m;
//	int now_m;
//	int d_idx;
//	int start_w_idx;
//	vector<int> check_weak;
//	vector<int> d_idx_list;
//	vector<int> start_w_idx_list;
//	int way;
//};
