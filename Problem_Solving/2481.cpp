//#include<iostream>
//#include<string>
//#include<vector>
//#include<algorithm>
//#include<queue>
//using namespace std;
//int n, k, m;
//bool ret_close(string const& s1, string const& s2);
//vector<int> make_ret(vector<vector<int>>& maps, vector<int>& hist);
//vector<int> make_hist(vector<int> const& hist, int goal);
//vector<int> find_path(vector<vector<int>>& maps, int goal);
//class Node
//{
//public:
//	int idx;
//	int time;
//	Node(int i, int t) :idx(i), time(t) {}
//};
//int main()
//{
//	//freopen("input.txt", "r", stdin);
//	cin >> n >> k;
//	vector<string> input_codes(n);
//	for (int i = 0; i < n; i++)
//		cin >> input_codes[i];
//	cin >> m;
//	vector<int> ends(m);
//	for (int i = 0; i < m; i++)
//		cin >> ends[i];
//	vector<vector<int>> maps(n+1, vector<int>(0));
//	for (int i = 0; i < n; i++)
//		for (int j = i+1; j < n; j++)
//			if (ret_close(input_codes[i], input_codes[j]))
//			{
//				maps[i + 1].push_back(j + 1);
//				maps[j + 1].push_back(i + 1);
//			}
//	for (int i = 0; i < m; i++)
//	{
//		vector<int> hist = find_path(maps, ends[i]);
//		if (hist.size() == 1)
//			cout << -1 << endl;
//		else
//		{
//			vector<int> hist2 = make_hist(hist, ends[i]);
//			for (int i = hist2.size() - 2; i >= 0; i--)
//				cout << hist2[i] << " ";
//			cout << endl;
//		}
//	}
//	//vector<int> hist;
//	//vector<int> ret = make_ret(maps,hist);
//	//for (int i = 0; i < m; i++)
//	//{
//	//	if (ret[ends[i]] == -1)
//	//		cout << -1 << endl;
//	//	else
//	//	{
//	//		vector<int> hist2 = make_hist(hist, ends[i]);
//	//		for (int i = hist2.size() - 2; i >= 0; i--)
//	//			cout << hist2[i] << " ";
//	//		cout << endl;
//	//	}
//	//}
//}
//bool ret_close(string const& s1, string const& s2)
//{
//	int diff = 0;
//	for (int i = 0; i < k; i++)
//	{
//		if (s1[i] != s2[i])
//			diff++;
//		if (diff > 1)
//			return false;
//	}
//	if (diff == 1)
//		return true;
//	else
//		return false;
//}
//vector<int> make_ret(vector<vector<int>>& maps,vector<int> & hist)
//{
//	queue<Node*> q;
//	vector<int> ret(n + 1, -1);
//	hist = vector<int>(n + 1, -1);
//	ret[1] = 0;
//	hist[1] = 0;
//	q.push(new Node(1, 0));
//	while (!q.empty())
//	{
//		Node* front = q.front();
//		q.pop();
//		int idx = front->idx;
//		int time = front->time;
//		for (int i = 0; i < maps[idx].size(); i++)
//		{
//			int close_node = maps[idx][i];
//			if (ret[close_node] == -1)
//			{
//				ret[close_node] = time + 1;
//				hist[close_node] = idx;
//				q.push(new Node(close_node, time + 1));
//			}
//		}
//		delete(front);
//	}
//	return ret;
//}
//vector<int> make_hist(vector<int> const& hist, int goal)
//{
// // there should be a path to 1
//	vector<int> ret;
//	ret.push_back(goal);
//	int last = goal;
//	while (last != 0)
//	{
//		last = hist[last];
//		ret.push_back(last);
//	}
//	return ret;
//}
//vector<int> find_path(vector<vector<int>>& maps, int goal)
//{
//	int n = maps.size() - 1;
//	queue<Node*> q;
//	vector<int> ret;
//	vector<int> visited(n + 1, false);
//	vector<int> hist(n + 1,-1);
//	visited[1] = true;
//	ret.push_back(1);
//	hist[1] = 0;
//	q.push(new Node(1, 0));
//	while (!q.empty())
//	{
//		Node* front = q.front();
//		q.pop();
//		int idx = front->idx;
//		int time = front->time;
//		for (int i = 0; i < maps[idx].size(); i++)
//		{
//			int close_node = maps[idx][i];
//			if (visited[close_node] == false)
//			{
//				visited[close_node] = true;
//				ret.push_back(close_node);
//				hist[close_node] = idx;
//				if (close_node == goal)
//					return hist;
//				q.push(new Node(close_node, time + 1));
//			}
//		}
//		delete(front);
//	}
//	return vector<int>(1,-1);
//}
