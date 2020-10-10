//�޸� �ʰ��� ������.
//�� ������ ����ð��� �����ϰ� ������ © �� ������ ���� ��ư� ¥�� ����
//�޸� �ʰ� �ߴ��� Ȯ���ϰ�,
//�ʹ� ���ϰ� ¥�ٰ� R,C�� �ٲ㼭 �־�����.
//�̷� �Ǽ� ���� ����
//50�ۼ�Ʈ ������� �����ε� 1�ð� 10�� �ɷȴ�
//����� ��û�ؼ� ã�Ҵ� ������ �������� ���� ����� ������ ����

#include<iostream>
#include<queue>
#include<deque>
#include<vector>
using namespace std;
typedef vector<vector<int>> Map;
class Node
{
public:
	int r;
	int c;
	Node(int _r, int _c) :r(_r), c(_c) {}
};
int n;
int regions2(const Map& map);
const int ways[4][2] =
{
	{-1,0},{0,1},{1,0},{0,-1}
};
// R: 1, G : 2, B: 3
pair<int, int> findnext(const Map& visited);
void printMap(const Map& map);
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n;
	Map org = Map(n, vector<int>(n, 0));
	Map map2 = Map(n, vector<int>(n, 0));
	for (int i = 0; i < n; i++)
	{
		string str;
		cin >> str;
		for (int j = 0; j < n; j++)
		{
			int num;
			if (str[j] == 'R')
				num = 1;
			else if (str[j] == 'G')
				num = 2;
			else if (str[j] == 'B')
				num = 3;
			if (num == 2)
			{
				org[i][j] = 2;
				map2[i][j] = 1;
			}
			else
			{
				org[i][j] = num;
				map2[i][j] = num;
			}
		}
	}
	cout << regions2(org)<<" ";
	cout<< regions2(map2) << endl;
	return 0;
}
int regions2(const Map& map)
{
	//printMap(map);
	int ret = 1;
	Map visited = Map(n, vector<int>(n, 0));
	queue<Node*> q;
	q.push(new Node(0, 0));
	visited[0][0] = ret;
	while (1)
	{
		while (!q.empty())
		{
			Node* thisNode = q.front();
			q.pop();
			int this_r = thisNode->r;
			int this_c = thisNode->c;
			for (int i = 0; i < 4; i++)
			{
				int next_r = this_r + ways[i][0];
				int next_c = this_c + ways[i][1];
				bool isOkay = true;
				if (next_r < 0 || next_r >= n)
					isOkay = false;
				if (next_c < 0 || next_c >= n)
					isOkay = false;
				if (isOkay)
				{
					if ((visited[next_r][next_c] == 0) && (map[this_r][this_c] == map[next_r][next_c]))
						isOkay = true;
					else
						isOkay = false;
				}
				if (isOkay)
				{
					q.push(new Node(next_r, next_c));
					visited[next_r][next_c] = visited[this_r][this_c];
				}
			}
			delete(thisNode);
		}

		int next_r = -1, next_c = -1;
		bool found_next = false;
		for (int i = 0; i < n; i++)
		{
			if (found_next)
				break;
			for (int j = 0; j < n; j++)
			{
				if (visited[i][j] == 0)
				{
					next_r = i, next_c = j;
					//cout << next_r << ", " << next_c << endl;
					q.push(new Node(next_r, next_c));
					ret++;
					visited[next_r][next_c] = ret;
					found_next = true;
					break;
				}
			}
			if (found_next)
				break;
		}
		if (next_r == -1)
		{
			//cout << "function enede" << endl;
			return ret;
		}
	}
}
void printMap(const Map& map)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			cout << map[i][j] << " ";
		cout << endl;
	}
}