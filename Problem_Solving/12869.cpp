//24�� �ɸ�
//3���� ���� ������ϱ� �������
//sort �����ϱ� �򰥷���
//customizing ����� ��������
//�����ؼ� �����ϴ� ����� �� ���̴� ����. ��������

#include<vector>
#include<iostream>
#include<algorithm>
#include<limits.h>
using namespace std;
int n;
int cache[61][61][61];
int retMin(int p0, int p1, int p2);
const int ways[6][3] = {
	{-1,-3,-9},{-1,-9,-3},{-3,-1,-9},{-3,-9,-1},{-9,-1,-3},{-9,-3,-1}
};
bool int_swap(int& n1, int& n2)
{
	return n1 > n2;
}
int main()
{
	freopen("input.txt", "r", stdin);
	vector<vector<vector<int>>> v(61,vector<vector<int>>(61, vector<int>(61, 0)));
	cin >> n;
	vector<int> init_p(3, 0);
	for (int i = 0; i < 3; i++)
		cin >> init_p[i];
	sort(init_p.begin(), init_p.end(),int_swap);
	cout << init_p[0] << " " << init_p[1] << " " << init_p[2] << endl;
	cout << retMin(init_p[0], init_p[1], init_p[2]) << endl;
}
int retMin(int p0,int p1, int p2)
{
	if (p0 == p1 && p1 == p2 && p0 == 0)
		return 0;
	int& ret = cache[p0][p1][p2];
	if (ret != 0)
		return ret;
	ret = INT_MAX;
	for (int i = 0; i < 6; i++)
	{
		int _p0 = p0 + ways[i][0];
		int _p1 = p1 + ways[i][1];
		int _p2 = p2 + ways[i][2];
		if (_p0 < 0)
			_p0 = 0;
		if (_p1 < 0)
			_p1 = 0;
		if (_p2 < 0)
			_p2 = 0;
		ret = min(ret, retMin(_p0, _p1, _p2));
	}
	return ++ret;
}
