////1시간 10분
////중간에 답도 봤다.
////계산해서 괜찮은거 같으면, 시간 내에 들어오는 것 같고
////알고리즘 틀린거 없으면 그냥 당당하게 하자. 그게 합격의 방법이다.
////first, second. 조심하자. 빠르게 코딩하지 말고 천천히 코딩하자. 하나하나씩 코딩하자.
////index, 인자로 넘겨줄때 어떻게 하는지 조심하자.
////당황하지 말자
//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <queue>
//using namespace std;
//typedef pair<int, int> Pos;
//typedef vector<vector<int>> Map;
//vector<Pos> virus;
//vector<Pos> block;
//vector<Pos> blank;
//vector<bool> choosed;
//const int way[4][2] = {
//	{-1,0}, {1,0},{0,1},{0,-1}
//};
//int n, m;
//int getVirus(const Map& map);
//int doDfs(int cnt, int lastIdx, Map& map);
//void print_Map(const Map& map);
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	cin >> n >> m;
//	Map map = Map(n, vector<int>(m, 0));
//	int block_num, virus_num;
//	for (int i = 0; i < n; i++)
//		for (int j = 0; j < m; j++)
//		{
//			cin >> map[i][j];
//			if (map[i][j] == 1)
//				block.push_back(Pos(i, j));
//			else if (map[i][j] == 0)
//			{
//				blank.push_back(Pos(i, j));
//				choosed.push_back(false);
//			}			
//			else if (map[i][j] == 2)
//			{
//				virus.push_back(Pos(i, j));
//			}
//		}
////	print_Map(map);
//	int maxPlace = doDfs(0, -1, map);
//	cout << maxPlace << endl;
//}
//int getVirus(const Map& map)
//{	
////	cout << "in get_virus print" << endl;
//	Map temp = map;
//	int ret = virus.size();
////	cout << "print virus" << endl;
//	queue<Pos*> q;
//	for (int i = 0; i < virus.size(); i++)
//	{
////		cout << virus[i].first << ", " << virus[i].second << endl;
//		q.push(&virus[i]);
//	}
//	while (!q.empty())
//	{
//		Pos front_pos = *q.front();
//		q.pop();
////		cout << "front_pos : " << front_pos.first << " , " << front_pos.second << endl;
//		for (int i = 0; i < 4; i++)
//		{
//			Pos * newPos= new Pos(front_pos.first + way[i][0], front_pos.second + way[i][1]);
////			cout << "newPos : " << newPos->first << " , " << newPos->second << endl;
//			bool found = false;
//			if((newPos->first>=0)&&(newPos->first<n))
//				if ((newPos->second >= 0) &&( newPos->second < m))
//					if (temp[newPos->first][newPos->second] == 0)
//					{
//						temp[newPos->first][newPos->second] = 2;
//						ret++;
//						found = true;
//						q.push(newPos);
//					}
//			if (!found)
//				delete(newPos);
//		}
//	}
//	//cout << "in get_virus" << endl;
//	//print_Map(temp);
//	return ret;
//}
//int doDfs(int cnt, int lastIdx, Map& map)
//{
//	int ret = 0;
//	if (cnt == 3)
//	{
//		//cout << "doDfs" << " lastIdx : " << lastIdx << " getvirus: " << getVirus(map) << endl;
//		//print_Map(map);
//		return n * m - (block.size() + 3) - getVirus(map);
//	}
//	int k = blank.size();
//	if (lastIdx == blank.size() - 1)
//		return ret;
//	int nextIdx = lastIdx + 1;
//	for (int i = nextIdx; i < k; i++)
//	{
//		if (!choosed[i])
//		{
//			choosed[i] = true;
//			map[blank[i].first][blank[i].second] = 1;
//			ret = max(ret, doDfs(cnt + 1, i, map));
//			map[blank[i].first][blank[i].second] = 0;
//			choosed[i] = false;
//		}
//	}
//	return ret;
//}
//void print_Map(const Map& map)
//{
//	cout << "print_Map" << endl;
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < m; j++)
//			cout << map[i][j] << " ";
//		cout << endl;
//	}
//}
