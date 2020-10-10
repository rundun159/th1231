//������ ���� ���� �ڵ�� ¥�� ����
//���� �밡�ٸ� �ؼ� ��Ȯ�� �ڵ带 ¥��

#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;
typedef vector<vector<int>> Map;
typedef vector<int> Row;
const int blocks[5][4][2] = {
	{{0,0},{0,1},{0,2},{0,3}},
	{{0,0},{0,1},{1,0},{1,1}},
	{{0,0},{0,1},{1,1},{0,2}},
	{{0,0},{1,0},{2,0},{2,1}},
	{{0,0},{1,0},{1,1},{2,1}}
};
const int blocks2[5][2] =
{
	{0,3},{1,1},{1,2},{3,1},{2,1}
};
void turn_map(vector<Map*> &map_list);
void mirror_map(vector<Map*>& map_list);
void print_Map(const Map& map);
int ret_max(int block_num, const Map& map);
int n, m;
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n >> m;
	Map map(n, Row(m,0));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			cin >> map[i][j];
	vector<Map*> map_list(8);
	map_list[0] = &map;
	mirror_map(map_list);
	turn_map(map_list);
	int ret = -1;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 8; j++)
			ret = max(ret, ret_max(i, *map_list[j]));
	cout << ret << endl;
}
void turn_map(vector<Map*>& map_list)
{
	for (int i = 0; i < 3; i++)
	{
		int org_row = (*map_list[i]).size(), org_col = (*map_list[i])[0].size();
		int new_row = org_col, new_col = org_row;
		map_list[i + 1] = new Map(new_row, Row(new_col, 0));
		for(int row=0;row<org_row;row++)
			for (int col = 0; col < org_col; col++)
				(*map_list[i + 1])[col][new_col - (row + 1)] = (*map_list[i])[row][col];
	}
	for (int i = 4; i < 7; i++)
	{
		int org_row = (*map_list[i]).size(), org_col = (*map_list[i])[0].size();
		int new_row = org_col, new_col = org_row;
		map_list[i + 1] = new Map(new_row, Row(new_col, 0));
		for (int row = 0; row < org_row; row++)
			for (int col = 0; col < org_col; col++)
				(*map_list[i + 1])[col][new_col - (row + 1)] = (*map_list[i])[row][col];
	}
}
void mirror_map(vector<Map*>& map_list)
{
	int org_row = (*map_list[0]).size(), org_col = (*map_list[0])[0].size();
	map_list[4] = new Map(org_row, Row(org_col, 0));
	for (int row = 0; row < org_row; row++)
		for (int col = 0; col < org_col; col++)
			(*map_list[4])[row][org_col - (col + 1)] = (*map_list[0])[row][col];
}
void print_Map(const Map& map)
{
	for (int i = 0; i < map.size(); i++)
	{
		for (int j = 0; j < map[0].size(); j++)
			cout << map[i][j] << " ";
		cout << endl;
	}
	cout << endl << endl;
}
int ret_max(int block_num, const Map& map)
{
	int ret = -1;
	int row = map.size(), col = map[0].size();
	for(int i=0;i<row-blocks2[block_num][0];i++)
		for (int j = 0; j < col- blocks2[block_num][1]; j++)
		{
			int sum = map[i][j] + map[i + blocks[block_num][1][0]][j + blocks[block_num][1][1]] + map[i + blocks[block_num][2][0]][j + blocks[block_num][2][1]] + map[i + blocks[block_num][3][0]][j + blocks[block_num][3][1]];
			ret = max(ret, sum);
		}
	return ret;
}
