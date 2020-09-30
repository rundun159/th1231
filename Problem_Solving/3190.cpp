////3번째 푼 삼성 기출문제.
////문제 이해를 정확히 못해서 시간이 좀 걸렸다.
////F9가 디버깅 중단점
////ctrl + alt+ V > A
////방향 설정하도록 move_dir한건 잘한것 같다
////turn_head 설정해놓는것에서 오류가 있었다.
////구현은 금방했던 것 같다.
////틀린게 없는지, 빠진게 없는지 끝까지 잘 보고 구현 시작하는게 좋은것 같다.
////out_of_idex에러가 발생하는지 잘 생각해보자.
////L_list에서 발생했었다.
////iterator 사용하는 방법 잘 익혀놓자
////
//
//#include<iostream>
//#include<vector>
//#include<deque>
//
//using namespace std;
//int N, K, L;
//int move_dir[4][2] = {
//	{-1,0},{0,1},{1,0},{0,-1}
//};
//typedef pair<int, int> Pair_int;
//vector<Pair_int> K_list;
//typedef struct NODE
//{
//	Pair_int pos;
//	int way;
//	NODE()
//	{
//
//	}
//}Node;
//
//void turn_head(Node & node, char way)
//{
//	if (way == 'D')
//	{
//		node.way = (node.way + 1) % 4;
//	}
//	else
//		node.way = (node.way + 3) % 4;
//}
//deque<Node*> q;
//bool is_finished();
//bool move_head();
//typedef struct L_NODE
//{
//	int time;
//	char way;
//}L_node;
//vector<L_node> L_list;
//void print_body();
//int main()
//{
//	bool is_apple;
//	freopen("input.txt", "r", stdin);
//	cin >> N;
//	cin >> K;
//	K_list = vector<Pair_int>(K);
//	for (int i = 0; i < K; i++)
//	{
//		Pair_int input;
//		cin >> input.first >> input.second;
//		K_list[i] = input;
//	}
//	cin >> L;
//	L_list = vector<L_node>(L);
//	int L_list_idx = 0;
//	for (int i = 0; i < L; i++)
//	{
//		cin >> L_list[i].time >> L_list[i].way;
//	}
//	int time = 1;
//	Node* head = new Node();
//	head->pos.first = 1;
//	head->pos.second = 1;
//	head->way = 1;
//	q.push_back(head);
//	while (1)
//	{
////		print_body();
//		is_apple=move_head();
//		if (is_finished())
//			break;
//		if (!is_apple)
//		{
//			q.pop_back();
//		}
//		if (L_list_idx == L_list.size())
//		{
//			
//		}
//		else if (time == L_list[L_list_idx].time)
//		{
//			turn_head(*q.front(), L_list[L_list_idx].way);
//			L_list_idx++;
//		}
//		time++;
//	}
//	cout << time << endl;
//}
//bool is_finished()
//{
//	Node * head = q.front();
//	if (head->pos.first<1 || head->pos.first>N)
//		return true;
//	if (head->pos.second<1 || head->pos.second>N)
//		return true;
//	deque<Node*>::iterator iter = q.begin();
//	iter++;
//	for (; iter != q.end(); iter++)
//	{
//		if (head->pos == (*iter)->pos)
//			return true;
//	}
//	return false;
//}
//bool move_head()
//{
//	Node* newNode = new Node();
//	newNode->way = q.front()->way;
//	newNode->pos.first = q.front()->pos.first + move_dir[newNode->way][0];
//	newNode->pos.second = q.front()->pos.second + move_dir[newNode->way][1];
//	q.push_front(newNode);
//	for(vector<Pair_int>::iterator iter=K_list.begin();iter!=K_list.end();iter++)
//	{
//		if (iter->first == newNode->pos.first)
//			if (iter->second == newNode->pos.second)
//			{
//				K_list.erase(iter);
//				return true;
//			}
//	}
//	return false;
//}
//void print_body()
//{
//	for (deque<Node*>::iterator iter = q.begin(); iter != q.end(); iter++)
//		cout << "( " << (*iter)->pos.first << ", " << (*iter)->pos.second << ") ";
//	cout << endl;
//
//}
