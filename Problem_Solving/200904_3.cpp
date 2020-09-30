//#include <string>
//#include <vector>
//#include <iostream>
//#include <queue>
//#include <algorithm>
//using namespace std;
//typedef struct Pos
//{
//	int x, y;
//};
//class Node
//{
//public:
//	Pos pos;
//	int dir;
//	int time;
//	bool possible;
//	Node(int x, int y, int dir_, int time_, bool possible_)
//	{
//		pos.x = x;
//		pos.y = y;
//		dir = dir_;
//		time = time_;
//		possible = possible_;
//	}
//};
//typedef struct Inf
//{
//	int n;
//	vector<vector<int>> MOV_DIR;
//	vector<vector<int>> reocord_0;
//	vector<vector<int>> reocord_1;
//	vector<vector<int>> board;
//	queue<Node *> q;
//};
//Node* move(int way, Node const& node, Inf& inf);
//Node *rotate_ccw(int way, Node const& node, Inf& inf);
//Node *rotate_cw(int way, Node const& node, Inf& inf);
//int ret_time(Inf &inf); 
//void check_record(Inf& inf, Node const& node);
//bool check_poss(Inf const& inf, Pos const& pos, int dir);
//int check_rotate_safe(Inf const& inf, int x, int y);
//bool check_cond(Inf const& inf, Node const& node);
//int solution(vector<vector<int>> board);
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	int n;
//	cin >> n;
//	vector<vector<int>> board(vector<vector<int>>(n, vector<int>(n, 0)));
//	for (int i = 0; i < n; i++)
//		for (int j = 0; j < n; j++)
//			cin >> board[i][j];
//	solution(board);
//}
//int solution(vector<vector<int>> board) {
//	Inf inf;
//	inf.n = board.size();
//	inf.board = board;
//	inf.MOV_DIR = {
//		{0,1},{1,0},{0,-1},{-1,0}
//	};
//	inf.reocord_0 = vector<vector<int>>(inf.n, vector<int>(inf.n, 0));
//	inf.reocord_1 = vector<vector<int>>(inf.n, vector<int>(inf.n, 0));
//	inf.reocord_0[0][0] = 1;
//	Node * first_node= new Node(1,1,0,0,true);
//	inf.q.push(first_node);
//	check_record(inf, *first_node);
//	int answer = 0;
//	answer = ret_time(inf);
//	return answer;
//}
//Node * move(int way, Node const& node, Inf & inf)
//{
//	Pos next_pos;
//	next_pos.x = node.pos.x + inf.MOV_DIR[way][0];
//	next_pos.y = node.pos.y + inf.MOV_DIR[way][1];
//	bool poss = check_poss(inf, next_pos, node.dir);
//	Node* ret;
//	ret = new Node(next_pos.x, next_pos.y, node.dir, node.time + 1, poss);
//	return ret;
//}
//Node * rotate_ccw(int way, Node const& node, Inf & inf)
//{
//	Pos next_pos;
//	bool poss;
//	Node* ret;
//	if (way == 0)
//	{
//		if (node.dir == 0)
//		{
//			next_pos.x = node.pos.x - 1;
//			next_pos.y = node.pos.y;
//			poss = check_poss(inf, next_pos, 1);
//			int safe = check_rotate_safe(inf, next_pos.x, next_pos.y + 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 1, node.time + 1, poss);
//		}
//		else if (node.dir ==1)
//		{
//			next_pos.x = node.pos.x;
//			next_pos.y = node.pos.y;
//			poss = check_poss(inf, next_pos, 0);
//			int safe = check_rotate_safe(inf, next_pos.x +1, next_pos.y + 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 0, node.time + 1, poss);
//		}
//	}
//	else
//	{
//		if (node.dir == 0)
//		{
//			next_pos.x = node.pos.x;
//			next_pos.y = node.pos.y+1;
//			poss = check_poss(inf, next_pos, 1);
//			int safe = check_rotate_safe(inf, next_pos.x + 1, next_pos.y - 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 1, node.time + 1, poss);
//		}
//		else if (node.dir == 1)
//		{
//			next_pos.x = node.pos.x +1;
//			next_pos.y = node.pos.y - 1;
//			poss = check_poss(inf, next_pos, 0);
//			int safe = check_rotate_safe(inf, next_pos.x - 1, next_pos.y + 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 0, node.time + 1, poss);
//		}
//	}
//	return ret;
//}
//Node * rotate_cw(int way, Node const& node, Inf & inf)
//{
//	Pos next_pos;
//	bool poss;
//	Node* ret;
//	if (way == 0)
//	{
//		if (node.dir == 0)
//		{
//			next_pos.x = node.pos.x;
//			next_pos.y = node.pos.y;
//			poss = check_poss(inf, next_pos, 1);
//			int safe = check_rotate_safe(inf, next_pos.x+1, next_pos.y + 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 1, node.time + 1, poss);
//		}
//		else if (node.dir == 1)
//		{
//			next_pos.x = node.pos.x;
//			next_pos.y = node.pos.y-1;
//			poss = check_poss(inf, next_pos, 0);
//			int safe = check_rotate_safe(inf, next_pos.x + 1, next_pos.y);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 0, node.time + 1, poss);
//		}
//	}
//	else
//	{
//		if (node.dir == 0)
//		{
//			next_pos.x = node.pos.x -1;
//			next_pos.y = node.pos.y + 1;
//			poss = check_poss(inf, next_pos, 1);
//			int safe = check_rotate_safe(inf, next_pos.x , next_pos.y - 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 1, node.time + 1, poss);
//		}
//		else if (node.dir == 1)
//		{
//			next_pos.x = node.pos.x + 1;
//			next_pos.y = node.pos.y ;
//			poss = check_poss(inf, next_pos, 0);
//			int safe = check_rotate_safe(inf, next_pos.x - 1, next_pos.y + 1);
//			if (safe != 0)
//				poss = false;
//			ret = new Node(next_pos.x, next_pos.y, 0, node.time + 1, poss);
//		}
//	}
//	return ret;
//}
//int ret_time(Inf &inf)
//{
//	int ret;
//	while (!inf.q.empty())
//	{
//		Node * node = inf.q.front();
//		//cout << "x : " << node->pos.x << " y : " << node->pos.y << " dir " << node->dir<<endl;
//		inf.q.pop();
//		for (int i = 0; i < 4; i++)
//		{
//			Node* temp = move(i, *node,inf);
//			if (!temp->possible)
//				delete(temp);
//			else
//			{
//				check_record(inf, *temp);
//				inf.q.push(temp);
//				if (check_cond(inf, *temp))
//					return temp->time;
//			}
//		}
//		for (int i = 0; i < 2; i++)
//		{
//			Node* temp = rotate_ccw(i, *node,inf);
//			if (!temp->possible)
//				delete(temp);
//			else
//			{
//				check_record(inf, *temp);
//				inf.q.push(temp);
//				if (check_cond(inf, *temp))
//					return temp->time;
//			}
//		}
//		for (int i = 0; i < 2; i++)
//		{
//			Node* temp = rotate_cw(i, *node,inf);
//			if (!temp->possible)
//				delete(temp);
//			else
//			{
//				check_record(inf, *temp);
//				inf.q.push(temp);
//				if (check_cond(inf, *temp))
//					return temp->time;
//			}
//		}
//		delete(node);
//	}
//	return ret;
//}
//void check_record(Inf& inf, Node const& node)
//{
//	if (node.dir == 0)
//	{
//		inf.reocord_0[node.pos.x-1][node.pos.y-1] = 1;
//	}
//	else
//	{
//		inf.reocord_1[node.pos.x-1][node.pos.y-1] = 1;
//	}
//}
//bool check_poss(Inf const& inf, Pos const& pos, int dir)
//{
//	if (pos.x < 1 || pos.y < 1)
//		return false;
//	if (pos.x > inf.n || pos.y > inf.n)
//		return false;
//	if (dir == 0)
//	{
//		if (pos.y == inf.n)
//			return false;
//		if (inf.reocord_0[pos.x-1][pos.y-1] != 0)
//			return false;
//		if (inf.board[pos.x - 1][pos.y - 1] == 1 || inf.board[pos.x - 1][pos.y ] == 1)
//			return false;
//	}
//	else
//	{
//		if (pos.x == inf.n)
//			return false;
//		if (inf.reocord_1[pos.x - 1][pos.y - 1] != 0)
//			return false;
//		if (inf.board[pos.x - 1][pos.y - 1] == 1 || inf.board[pos.x][pos.y - 1] == 1)
//			return false;
//	}
//	return true;
//}
//int check_rotate_safe(Inf const& inf, int x, int y)
//{
//	if (x < 1 || y < 1)
//		return -1;
//	if (x > inf.n || y > inf.n)
//		return -1;
//	return inf.board[x-1][y-1];
//}
//bool check_cond(Inf const& inf, Node const& node)
//{
//	if (!node.possible)
//		return false;
//	if (node.dir == 0)
//	{
//		if ((node.pos.x == inf.n) && (node.pos.y == (inf.n - 1)))
//			return true;
//	}
//	else if (node.dir==1)
//	{
//		if (((node.pos.x-1) == inf.n) && (node.pos.y == (inf.n )))
//			return true;
//	}
//	return false;
//}
