//#include<iostream>
//#include<vector>
//#include<math.h>
//#include<algorithm>
//using namespace std;
//class Node
//{
//public:
//	int sum, num1, num2,num3;
//	Node(int sum_,int num1_, int num2_):sum(sum_),num1(num1_),num2(num2_)
//	{	}
//	Node(int sum_, int num1_, int num2_,int num3_) :sum(sum_), num1(num1_), num2(num2_), num3(num3_)
//	{	}
//};
//bool check_prime_number(int num);
//vector<int> array1(vector<bool> &v);
//vector<Node> array2(vector<bool>& v, vector<int> const& v2);
//vector<Node> array3(vector<int> const& v1, vector<Node> const& v2, vector<int> & v3);
//
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	int T, K;
//	cin >> T;
//	vector<bool> v1(1001, false);
//	vector<bool> v2(1001, false);
//	vector<int> v3(1001, -1);
//	vector<int> ret1 =	array1(v1);
//	vector<Node> ret2 = array2(v2, ret1);
//	vector<Node> ret3= array3(ret1, ret2,v3 );
//	
//	for (int t = 0; t < T; t++)
//	{
//		cin >> K;
//		if (v3[K] != -1)
//		{
//			Node node = ret3[v3[K]];
//			vector<int> ans;
//			ans.push_back(node.num1);
//			ans.push_back(node.num2);
//			ans.push_back(node.num3);
//			sort(ans.begin(), ans.end());
//			cout << ans[0] << " " << ans[1] << " " << ans[2] << endl;
//		}
//		else
//			cout << 0 << endl;
//	}
//}
//
//bool check_prime_number(int num)
//{
//	if (num < 2)
//		return false;
//	else if (num >= 1000)
//		return false;
//	int num_sqrt = sqrt(num);
//	for (int i = 2; i <= num_sqrt; i++)
//	{
//		if ((num % i) == 0)
//			return false;
//	}
//	return true;
//}
//vector<int> array1(vector<bool>& v)
//{
//	vector<int> ret;
//	for (int i = 2; i < 1000; i++)
//		if (check_prime_number(i))
//		{
//			v[i] = true;
//			ret.push_back(i);
//		}
//	return ret;
//}
//vector<Node> array2(vector<bool> & v,vector<int> const &v2)
//{
//	vector<Node> ret;
//	int sum;
//	for (int i = 0; i < v2.size(); i++)
//		for (int j = 0; j < v2.size(); j++)
//		{
//			sum = v2[i] + v2[j];
//			if (sum < 1000)
//				ret.push_back(Node(sum, v2[i], v2[j]));
//		}
//	return ret;
//}
//vector<Node> array3(vector<int> const & v1, vector<Node> const & v2,vector<int> &v3)
//{
//	vector<Node> ret;
//	vector<bool> record(1001, false);
//	int cnt = 0;
//	for(int i=0;i<v1.size();i++)
//		for (int j = 0; j < v2.size(); j++)
//		{
//			int sum = v1[i] + v2[j].sum;
//			if ((sum < 1000) && (sum % 2 != 0) && (sum>0))
//			{
//				if (!record[sum])
//				{
//					record[sum] = true;
//					ret.push_back(Node(sum, v2[j].num1, v2[j].num2, v1[i]));
//					v3[sum] = cnt;
//					cnt++;
//				}
//			}
//		}
//	return ret;
//}
