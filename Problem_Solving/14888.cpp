//50�� ��.
//index�� ��������.
//index �Ҽ��� �� ����س��� �򰥸��� ����.
//dfs ������ ��츦 �� �����س���

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<int> op_given;
vector<int> num_given;
int n;
int cal_opt(const vector<int> &op_v);
void doDFS(int opIdx, int startIdx, int cnt, vector<int>& op_v, vector<int>& op_cnt);
long long int max_ret, min_ret;
void main()
{
	freopen("input.txt", "r", stdin);
	cin >> n;
	num_given = vector<int>(n);
	op_given = vector<int>(4);
	for (int i = 0; i < n; i++)
		cin >> num_given[i];
	for (int i = 0; i < 4; i++)
		cin >> op_given[i];
	max_ret = -9999999999;
	min_ret = 9999999999;
	vector<int> op_v = vector<int>(n - 1, -1);
	vector<int> op_cnt = vector<int>(4, 0);	
	doDFS(0, 0, 0, op_v, op_cnt);
	cout << max_ret << endl;
	cout << min_ret << endl;
}
int cal_opt(const vector<int>& op_v)
{
	long long int ret = 0;
	if (op_v[0] == 0)
		ret = num_given[0] + num_given[1];
	else if (op_v[0] == 1)
		ret = num_given[0] - num_given[1];
	else if (op_v[0] == 2)
		ret = num_given[0] * num_given[1];
	else if (op_v[0] == 3)
		ret = num_given[0] / num_given[1];
	if (n == 2)
		return ret;
	for (int i = 1; i < n - 1; i++)
	{
		if (op_v[i] == 0)
			ret = ret + num_given[i + 1];
		else if (op_v[i] == 1)
			ret = ret - num_given[i + 1];
		else if (op_v[i] == 2)
			ret = ret * num_given[i + 1];
		else if (op_v[i] == 3)
			ret = ret / num_given[i + 1];
	}
	return ret;
}
void doDFS(int opIdx, int startIdx, int cnt, vector<int>& op_v, vector<int>& op_cnt)
{
	if (opIdx == 4)
	{
		int sum = 0;
		for (int i = 0; i < 4; i++)
			sum += op_cnt[i];
		if (sum == (n - 1))
		{
			long long int ret = cal_opt(op_v);
			if (ret > max_ret)
				max_ret = ret;
			if (ret < min_ret)
				min_ret = ret;
			return;
		}
		else
		{
			return;
		}
	}
	if (cnt < op_given[opIdx])
	{
		if (startIdx >= n-1)
			return;
		for (int i = startIdx; i < n-1; i++)
		{
			if (op_v[i] == -1)
			{
				op_v[i] = opIdx;
				op_cnt[opIdx]++;
				doDFS(opIdx, i + 1, cnt + 1, op_v, op_cnt);
				op_v[i] = -1;
				op_cnt[opIdx]--;
			}
		}
	}
	else
	{
		doDFS(opIdx + 1, 0, 0, op_v, op_cnt);
	}
}
