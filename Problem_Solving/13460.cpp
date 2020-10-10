#include <iostream>
#include <vector>
#include <queue>
using namespace std;
typedef struct NODE
{
	pair<int, int> Red, Blue;
	int tryNum;
}Node;
vector<vector<int>> mat;
int n, m;
queue<Node> q;
pair<int, int> goal;
int retNum()
{
	
}
Node newNode_init(pair<int, int>red, pair<int, int>blue, int tryNum)
{
	Node newnode;
	newnode.Red = red;
	newnode.Blue = blue;
	newnode.tryNum = tryNum;
	return newnode;
}
int popFunc()
{
	Node front = q.front();
	Node newNode = front;
	newNode.tryNum++;
	pair<int, int> red, blue;
	red = front.Red, blue = front.Blue;
	vector<pair<int, int>> way(4);
	//up
	way[0] = pair<int, int>(-1, 0);
	//down
	way[1] = pair<int, int>(1, 0);
	//left
	way[2] = pair<int, int>(0, -1);
	//right
	way[3] = pair<int, int>(0, 1);
	//up
	bool up = true;
	if (red.second == blue.second)
		if ((red.first + way[0].first) == (blue.first))
			if (mat[blue.first+way[0].first][blue.second] == -1)
				up = false;
	if (red.second == blue.second)
		if ((red.first ) == (blue.first + way[0].first))
			if (mat[red.first + way[0].first][red.second] == -1)
				up = false;
	if (mat[red.first + way[0].first][red.second] == -1)
		if (mat[blue.first + way[0].first][blue.second] == -1)
			up = false;

	if (blue.second == goal.second)
	{
		bool warning = true;
		for (int i = blue.first + 1; i < goal.first; i++)
			if (mat[i][blue.second] == -1)
				warning = false;
		if (warning)
			up = false;
	}
	if (up)
	{
		pair<int, int> red_pnt = red;
		red_pnt.first--;
		while (mat[red_pnt.first][red_pnt.second] != -1)
			red_pnt.first--;
		pair<int, int> blue_pnt = blue;
		blue_pnt.first--;
		while (mat[blue_pnt.first][blue_pnt.second] != -1)
			blue_pnt.first--;
		Node up_node = newNode_init(red_pnt, blue_pnt, newNode.tryNum);
	}

	//down
	bool down = true;
	if (red.second == blue.second)
		if ((red.first + way[1].first) == (blue.first))
			if (mat[blue.first + way[1].first][blue.second] == -1)
				down = false;
	if (red.second == blue.second)
		if ((red.first) == (blue.first + way[1].first))
			if (mat[red.first + way[1].first][red.second] == -1)
				down = false;
	if (mat[red.first + way[1].first][red.second] == -1)
		if (mat[blue.first + way[1].first][blue.second] == -1)
			down = false;

	if (blue.second == goal.second)
	{
		bool warning = true;
		for (int i = goal.first + 1; i < blue.first; i++)
			if (mat[i][blue.second] == -1)
				warning = false;
		if (warning)
			down = false;
	}

	//left
	bool left = true;
	if (red.first == blue.first)
		if ((red.second + way[2].second) == (blue.second))
			if (mat[blue.first][blue.second + way[2].second] == -1)
				left = false;
	if (red.first == blue.first)
		if ((red.second) == (blue.second + way[2].second))
			if (mat[red.first][red.second + way[2].second] == -1)
				left = false;
	if (mat[red.first][red.second + way[2].second] == -1)
		if (mat[blue.first][blue.second + way[2].second] == -1)
			left = false;

	if (blue.first == goal.first)
	{
		bool warning = true;
		for (int i = goal.second + 1; i < blue.second; i++)
			if (mat[blue.first][i] == -1)
				warning = false;
		if (warning)
			left = false;
	}

	//right
	bool right = true;
	if (red.first == blue.first)
		if ((red.second +way[3].second) == (blue.second))
			if (mat[blue.first][blue.second +way[3].second] == -1)
				right = false;
	if (red.first == blue.first)
		if ((red.second ) == (blue.second+way[3].second))
			if (mat[red.first][red.second +way[3].second] == -1)
				right = false;
	if (mat[red.first][red.second + way[3].second] == -1)
		if (mat[blue.first][blue.first + way[3].second] == -1)
			right = false;
	if (blue.first == goal.first)
	{
		bool warning = true;
		for (int i = blue.second + 1; i < goal.second; i++)
			if (mat[blue.first][i] == -1)
				warning = false;
		if (warning)
			right = false;
	}
	//up

	//down
}
int popAndDo()
{
	Node front = q.front();
	Node newNode=front;
	newNode.tryNum++;
	pair<int, int> red, blue;
	red = front.Red, blue = front.Blue;
	//down
	bool down = true;
	if (mat[red.first + 1][red.second] == 1)
		return front.tryNum+1;

	if (mat[blue.first + 1][blue.second] == -1)
		if (red.first + 1 == blue.first)
			if (red.second == blue.second)
				down = false;

	if (mat[red.first + 1][red.second] == -1)
		if (blue.first + 1 == red.first)
			if (red.second == blue.second)
				down = false;

	if (mat[red.first + 1][red.second] == -1)
		if (mat[blue.first + 1][blue.second] == -1)
			down = false;
	if (down)
	{
		if (mat[red.first + 1][red.second] == 0)
			newNode.Red.first += 1;
		if (mat[blue.first + 1][blue.second] == 0)
			newNode.Blue.first += 1;
		q.push(newNode);
		if (mat[red.first + 1][red.second] == 0)
			newNode.Red.first -= 1;
		if (mat[blue.first + 1][blue.second] == 0)
			newNode.Blue.first -= 1;
	}
	//up
	if (mat[red.first -1 ][red.second] == 1)
		return front.tryNum+1;
	bool up = true;
	
	if (mat[blue.first - 1][blue.second] == -1)
		if (red.first - 1 == blue.first)
			if (red.second == blue.second)
				down = false;

	if (mat[red.first - 1][red.second] == -1)
		if (blue.first - 1 == red.first)
			if (red.second == blue.second)
				down = false;

	if (mat[red.first -1][red.second] == -1)
		if (mat[blue.first - 1][blue.second] == -1)
			up = false;
	if (up)
	{
		if (mat[red.first-1][red.second] == 0)
			newNode.Red.first -= 1;
		if (mat[blue.first-1][blue.second] == 0)
			newNode.Blue.first -= 1;
		q.push(newNode);
		if (mat[red.first-1][red.second ] == 0)
			newNode.Red.first += 1;
		if (mat[blue.first-1][blue.second] == 0)
			newNode.Blue.first += 1;
	}

	//left
	if (mat[red.first][red.second -1] == 1)
		return front.tryNum;
	bool left = true;
	if (red.second - 1 == blue.second)
		if (red.first == blue.first)
			left = false;
	if (mat[red.first ][red.second-1] == -1)
		if (mat[blue.first][blue.second - 1] == -1)
			left = false;
	if (left)
	{
		if (mat[red.first][red.second - 1] == 0)
			newNode.Red.second -= 1;
		if (mat[blue.first][blue.second - 1] == 0)
			newNode.Blue.second -= 1;
		q.push(newNode);
		if (mat[red.first][red.second - 1] == 0)
			newNode.Red.second += 1;
		if (mat[blue.first][blue.second - 1] == 0)
			newNode.Blue.second += 1;
	}

	//right
	if (mat[red.first ][red.second +1] == 1)
		return front.tryNum;
	bool right = true;
	if (red.second + 1 == blue.second)
		if (red.first == blue.first)
			right = false;
	if (mat[red.first ][red.second +1] == -1)
		if (mat[blue.first][blue.second + 1] == -1)
			right = false;
	if (right)
	{
		if (mat[red.first][red.second + 1] == 0)
			newNode.Red.second += 1;
		if (mat[blue.first][blue.second + 1] == 0)
			newNode.Blue.second += 1;
		q.push(newNode);
		if (mat[red.first][red.second + 1] == 0)
			newNode.Red.second -= 1;
		if (mat[blue.first][blue.second + 1] == 0)
			newNode.Blue.second -= 1;
	}
}
int main()
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	cin >> n >> m;
	vector<string> mirror = vector<string>(n);
	mat = vector<vector<int>>(n, vector<int>(m, 0));
	pair<int, int> red, blue;
	for (int i = 0; i < n; i++)
		cin >> mirror[i];
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (mirror[i][j] == '#')
				mat[i][j] = -1;
			else if (mirror[i][j] == '.')
				mat[i][j] = 0;
			else if (mirror[i][j] == 'O')
			{
				mat[i][j] = 1;
				goal = pair<int, int>(i, j);
			}
			else if (mirror[i][j] == 'R')
				red = pair<int, int>(i, j);
			else
				blue = pair<int, int>(i, j);
	Node first;
	first.Red = red;
	first.Blue = blue;
	first.tryNum = 0;
}