//50�� �ɸ�
//���������� main�Լ����� �ѹ� �� �����ؼ� ���� ������
//�˰����� ������ �־���. �ѹ� �� Ȯ���ϰ� �߾�� �ߴ�.
//�и��� ���������� ���� ���� �ִ°� ������ Ȯ���غ����Ѵ�.
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int way_dir[4][2]
{
	{0,1},{0,-1},{-1,0},{1,0}
};
typedef pair<int, int> Pos;
typedef vector<vector<int>> Map;
typedef struct NODE
{
	vector<int> ver;
	vector<int> hor;
	Pos pos;
}Node;
int N, M, x, y, K;
Map map;
vector<int> k_list;
int move(int way, Node & node);
void copy_value(const vector<int>& from, vector<int>& to);
void print_dice(const Node& node);
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> N >> M >> x >> y >> K;
	map = Map(N, vector<int>(M, 0));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			cin >> map[i][j];
	k_list = vector<int>(K);
	for (int i = 0; i < K; i++)
		cin >> k_list[i];
	Node dice;
	dice.ver = vector<int>(4,0);
	dice.hor = vector<int>(4, 0);
	dice.pos = Pos(x, y);

	for (int i = 0; i < K; i++)
	{
//		print_dice(dice);
		int head=move(k_list[i], dice);
		if (head == -1)
			continue;
		else
			cout << head << endl;
	}
}
int move(int way, Node& node)
{
	Pos next_pos = node.pos;
	next_pos.first += way_dir[way-1][0];
	next_pos.second += way_dir[way-1][1];
	if (next_pos.first < 0 || next_pos.first >= N)
		return -1;
	if (next_pos.second < 0 || next_pos.second >= M)
		return -1;
	node.pos = next_pos;
	if (way == 1)
	{
		vector<int> temp(4);
		for (int i = 0; i < 4; i++)
		{
			temp[(i + 1) % 4] = node.hor[i];
		}
		node.ver[0] = node.hor[3];
		node.ver[2] = node.hor[1];
		copy_value(temp, node.hor);
	}
	else if (way == 2)
	{
		vector<int> temp(4);
		for (int i = 0; i < 4; i++)
		{
			temp[(i + 3) % 4] = node.hor[i];
		}
		node.ver[0] = node.hor[1];
		node.ver[2] = node.hor[3];
		copy_value(temp, node.hor);
	}
	else if (way == 3)
	{
		vector<int> temp(4);
		for (int i = 0; i < 4; i++)
		{
			temp[(i + 1) % 4] = node.ver[i];
		}
		node.hor[0] = node.ver[3];
		node.hor[2] = node.ver[1];
		copy_value(temp, node.ver);
	}
	else if (way == 4)
	{	
		vector<int> temp(4);
		for (int i = 0; i < 4; i++)
		{
			temp[(i + 3) % 4] = node.ver[i];
		}
		node.hor[0] = node.ver[1];
		node.hor[2] = node.ver[3];
		copy_value(temp, node.ver);
	}
	int & now_num = map[node.pos.first][node.pos.second];
	if ( now_num== 0)
	{
		now_num = node.ver[2];
	}
	else
	{
		node.ver[2] = now_num;
		node.hor[2] = now_num;
		now_num = 0;
	}
	return node.ver[0];
}
void copy_value(const vector<int>& from, vector<int>& to)
{
	for (int i = 0; i < from.size(); i++)
	{
		to[i] = from[i];
	}
}
void print_dice(const Node& node)
{
	cout << "ver :";
	for (int i = 0; i < 4; i++)
		cout << " " << node.ver[i];
	cout << endl;
	cout << "hor :";
	for (int i = 0; i < 4; i++)
		cout << " " << node.hor[i];
	cout << endl;
}
