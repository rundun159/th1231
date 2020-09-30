//#include<iostream>
//#include<vector>
//#include<math.h>
//#include<algorithm>
//using namespace std;
//int n, m, l;
//typedef struct M
//{
//	int x;
//	int y;
//};
//class Node
//{
//public:
//	int x;
//	int y;
//	Node(int x_, int y_) :x(x_), y(y_) {}
//};
//bool get_caught(int x, M const & m)
//{
//	int dist = m.y;
//	dist += abs(x - m.x);
//	return dist <= l;
//}
//bool compare_M(M& m1, M& m2)
//{
//	return m1.x < m2.x;
//}
//int ret_cnt(vector<int> const& n_list, vector<M> const & m_list)
//{
//	int ret = 0;
//	int last_idx = 0;
//	for (int i = 0; i < m_list.size(); i++)
//	{
//		bool got_c = false;
//		while ((last_idx < n_list.size()) && m_list[i].x >= n_list[last_idx] )
//			last_idx++;
//		if (last_idx == n_list.size())
//			last_idx--;
//		if (last_idx != 0)
//			if (get_caught(n_list[last_idx-1], m_list[i]))
//				got_c = true;
//		if(!got_c)
//			if (get_caught(n_list[last_idx], m_list[i]))
//				got_c = true;
//		if (got_c)
//			ret++;
//	}
//	return ret;
//}
//int main()
//{
//	//freopen("input.txt", "r", stdin);
//	cin >> n >> m >> l;
//	vector<int> n_list(n);
//	vector<M> m_list(m);
//	for (int i = 0; i < n; i++)
//		cin >> n_list[i];
//	for (int i = 0; i < m; i++)
//		cin >> m_list[i].x >> m_list[i].y;
//	sort(n_list.begin(), n_list.end());
//	sort(m_list.begin(), m_list.end(), compare_M);
//	cout << ret_cnt(n_list, m_list)<<endl;
//}