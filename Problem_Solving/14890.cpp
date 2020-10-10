//1�ð� ���� �ɷȳ�
//�������̴� ������ ö���ϰ� �ľ��ϰ� ���� ��������
//����ϴ��� ���̷� ���̽� �и� �� �ϰ� 
//���� ���� Ȯ���� �ϰ� ��������
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef vector<vector<int>> Map;
typedef vector<int> Arr;
int n, l;
vector<int> *ret_v_row(const Map & map,int rowIdx);
vector<int> * ret_v_col(const Map& map,int colIdx);
bool possible(vector<int>  * arr);
void print_v(const vector<int>& v);
bool debug = true;
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n >> l;
	Map map(n, vector<int>(n, 0));
	int ret;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			cin >> map[i][j];
	int sum = 0;
	if(!debug)
		cout << "row" << endl;
	for (int i = 0; i < n; i++)
	{
		ret= possible(ret_v_row(map, i));
		if (!debug)
		{
			if (ret == true)
			{
				cout << "success row " << i << "th: ";
				print_v(*ret_v_row(map, i));
			}
			if (ret == false)
			{
				cout << "fail row " << i << "th: ";
				print_v(*ret_v_row(map, i));
			}
		}
		sum += ret;
	}
	if (!debug)
		cout << "col" << endl;
	for (int i = 0; i < n; i++)
	{
		ret = possible(ret_v_col(map, i));
		if (!debug)
		{
			if (ret == true)
			{
				cout << "success col " << i << "th: ";
				print_v(*ret_v_col(map, i));
			}
			if (ret == false)
			{
				cout << "fail col " << i << "th: ";
				print_v(*ret_v_col(map, i));
			}
		}
		sum += ret;
	}
	cout << sum << endl;
	return 0;
}
vector<int> *ret_v_row(const Map& map, int rowIdx)
{
	Arr* new_arr = new Arr(n);
	for (int i = 0; i < n; i++)
		(*new_arr)[i] = map[rowIdx][i];
	return new_arr;
}
vector<int>* ret_v_col(const Map& map, int colIdx)
{
	Arr* new_arr = new Arr(n);
	for (int i = 0; i < n; i++)
		(*new_arr)[i] = map[i][colIdx];
	return new_arr;
}
bool possible(vector<int> * arr)
{
	int lastH, nowH, cnt = 1;
	bool isUse=false;
	lastH = (*arr)[0];
	for (int i = 1; i < n; i++)
	{
		nowH = (*arr)[i];
		if (nowH == lastH)
		{
			cnt++;
			isUse = false;
		}
		else if (nowH - lastH == 1)
		{
			if (cnt < l || isUse==true)
				return false;
			cnt = 1;
			if(l!=1)
				isUse = true;
			lastH = nowH;
		}
		else if (nowH - lastH == -1)
		{
			if ((n - i) >= l)
			{
				int j;
				for (j = i + 1; j < (i + l); j++)
				{
					if ((*arr)[j] != nowH)
						return false;
				}
				i = (i + l - 1);
				cnt = 0;
				isUse = true;
				lastH = nowH;
			}
			else
				return false;
		}
		else
		{
			return false;
		}
	}
	return true;
}
void print_v(const vector<int>& v)
{
	for (int i = 0; i < v.size(); i++)
		cout << v[i] << " ";
	cout << endl;
}
