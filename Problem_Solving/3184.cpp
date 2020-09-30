////자꾸 전역변수랑 지역변수랑 이름이 겹쳐서 고치느라 시간 씀
////class로 짜서 생성자 만드는 시간 줄이자
////빈 공간 detection이 완벽하지 못했음.
////완벽하지 못한 알고리즘으로 구현해서 실패함.
//
//
//#include <iostream>
//#include <vector>
//#include <queue>
//using namespace std;
//typedef vector<vector<int>> Map;
//typedef struct 
//{
//	int r;
//	int c;
//	int field_num;
//}Node;
//typedef struct
//{
//	int o;
//	int v;
//}Return;
//class Pos
//{
//public:
//	int r;
//	int c;
//	Pos(int r, int c) :r(r), c(c) {}
//};
//int ways[4][2] = {
//	{-1,0}, {0,1} ,{1,0},{0,-1}
//};
//Map map;
//int init_o_num, init_v_num;
//int r_glo, c_glo;
//int field_num_glo;
//Return* cal_field(int field_num, int r, int c);
//void node_init(Node* node, int r, int c, int field_num);
//bool check_map(int r, int c);
//vector< Pos* > o_pos;
//vector<Pos*> v_pos;
//int main()
//{
//	init_v_num = 0, init_o_num = 0;
//	freopen("input.txt", "r", stdin);
//	field_num_glo = -2;
//	cin >> r_glo >> c_glo;
//	map = Map(r_glo, vector<int>(c_glo, 0));
//	Return* retNode;
//	int ret_o=0, ret_v=0;
//	for (int i = 0; i < r_glo; i++)
//	{
//		string input_str;
//		cin >> input_str;
//		for (int j = 0; j < c_glo; j++)
//		{
//			if (input_str[j] == '.')
//				map[i][j] = 0;
//			else if (input_str[j] == '#')
//				map[i][j] = -1;
//			else if (input_str[j] == 'v')
//			{
//				init_v_num++;
//				map[i][j] = 2;
//				v_pos.push_back(new Pos(i, j));
//			}
//			else
//			{
//				init_o_num++;
//				map[i][j] = 1;
//				o_pos.push_back(new Pos(i, j));
//			}
//		}
//	}
//	for (int i = 0; i < o_pos.size(); i++)
//	{
//		Pos* pos = o_pos[i];
//		if (!check_map(pos->r, pos->c))
//			continue;
//		Return * ret =cal_field(field_num_glo--, pos->r, pos->c);
//		if (ret->o > ret->v)
//		{
//			ret_o += ret->o;
//		}
//		else
//		{
//			ret_v += ret->v;
//		}
//	}
//	for (int i = 0; i < v_pos.size(); i++)
//	{
//		Pos* pos = v_pos[i];
//		if (!check_map(pos->r, pos->c))
//			continue;
//		Return* ret = cal_field(field_num_glo--, pos->r, pos->c);
//		if (ret->o > ret->v)
//		{
//			ret_o += ret->o;
//		}
//		else
//		{
//			ret_v += ret->v;
//		}
//	}
//	cout << ret_o << " " << ret_v << endl;
//	return 0;
//}
//Return* cal_field(int field_num, int r, int c)
//{
//	int num_o = 0, num_v = 0;
//	queue<Node* > q;
//	Node * node = new Node();
//	node_init(node, r, c, field_num);
//	int& num2 = map[r][c];
//	if (num2 == 1)
//		num_o++;
//	else if (num2 == 2)
//		num_v++;
//	num2 = field_num;
//	q.push(node);
//	while(!q.empty())
//	{
//		Node *front = q.front();
//		q.pop();
//		int node_r = front->r, node_c = front->c;
//		for (int i = 0; i < 4; i++)
//		{
//			int next_r = node_r + ways[i][0], next_c = node_c + ways[i][1];
//			if (!check_map(next_r, next_c))
//				continue;
//			Node* newNode = new Node();
//			node_init(newNode, next_r, next_c, field_num);
//			int& num = map[next_r][next_c];
//			if (num == 1)
//				num_o++;
//			else if (num == 2)
//				num_v++;			
//			num = field_num;
//			q.push(newNode);
//		}
//	}
//	Return* ret = new Return();
//	if (num_o > num_v)
//	{
//		ret->o = num_o;
//		ret->v = 0;
//	}
//	else
//	{
//		ret->o = 0;
//		ret->v = num_v;
//	}
//	return ret;
//}
//void node_init(Node* node, int r, int c, int field_num)
//{
//	node->r = r;
//	node->c = c;
//	node->field_num = field_num;
//
//}
//bool check_map(int r, int c)
//{
//	if (r < 0 || r >= r_glo)
//		return false;
//	if (c < 0 || c >= c_glo)
//		return false;
//	if (map[r][c] < 0)
//		return false;
//	else
//		return true;
//}
