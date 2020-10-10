#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef struct Node
{
	int time;
	int money;
}Node;
int n;
vector< Node> works;
vector<int> max_money;
int ret_max_money(int idx);
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n;
	works = vector<Node>(n);
	for (int i = 0; i < n; i++)
		cin >> works[i].time >> works[i].money;
	max_money = vector<int>(n, -1);
	cout << ret_max_money(0) << endl;
}
int ret_max_money(int idx)
{
	if (idx >= n)
		return 0;
	int& ret = max_money[idx];
	int sum;
	if (ret != -1)
		return ret;
	ret = 0;
	if (idx + works[idx].time <= n)
	{
		sum = works[idx].money;
		if (idx + works[idx].time < n)
			sum += ret_max_money(idx + works[idx].time);
		ret = max(ret, sum);
	}
	if (idx < n - 1)
		ret = max(ret, ret_max_money(idx + 1));
	return ret;
}
