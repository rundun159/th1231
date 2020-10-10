//43��
//�׷��� ����� ������ �ƴϾ��µ�
//���ϴ� �˰����򿡼�, �߰��ϴ� ���͸� �߸� ������.
//print�����鼭 ��� Ʋ������ ����ã�� ����� ���� ����


#include<iostream>
#include<vector>
using namespace std;
typedef vector<vector<int>> Map;
const int ways[4][2] =
{
	{0,1},{-1,0},{0,-1},{1,0}
};
int n;
void make_dir_v(vector<int>& dir, const vector<int>& v, int start, int gen);
void make_len_v(vector<int> &v);
int countRect(const Map& map);
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n;
	Map map = Map(101, vector<int>(101,0));
	vector<int> v;
	make_len_v(v);
	for (int t = 0; t < n; t++)
	{
		int x, y, d, g;
		cin >> y >> x >> d >> g; //r=y, c=x
		vector<int> dir;
		make_dir_v(dir, v, d, g);
		map[x][y] = 1;
		for (int i = 0; i < dir.size(); i++)
		{
			x += ways[dir[i]][0];
			y += ways[dir[i]][1];
			//cout << x << " , " << y << endl;
			map[x][y] = 1;
		}
	}
	cout << countRect(map) << endl;
	return 0;
}

void make_len_v(vector<int> &v)
{
	v = vector < int>(11, 0);
	v[0] = 1;
	for (int i = 1; i < 11; i++)
	{
		v[i] = v[i - 1] * 2;
	}
	return;
}
void make_dir_v(vector<int>& dir, const vector<int>& v, int start, int gen)
{	
	dir = vector<int>(v[gen]);
	dir[0] = start;
	for (int i = 1; i <= gen; i++)
	{
		int start_idx = v[i] / 2;
		for (int j = start_idx - 1; j >= 0; j--)
			dir[v[i] - 1 - j] = (dir[j] + 1) % 4;
	}
	//cout << "gen : "<<gen<<"start : " << start << endl;
	//for (int i = 0; i < dir.size(); i++)
	//	cout << dir[i] << " ";
	//cout << endl;
	return;
}
int countRect(const Map& map)
{
	int ret = 0;
	for (int i = 0; i < 100; i++)
		for (int j = 0; j < 100; j++)
			if (map[i][j] * map[i + 1][j] * map[i][j + 1] * map[i + 1][j + 1])
				ret++;
	return ret;
}
