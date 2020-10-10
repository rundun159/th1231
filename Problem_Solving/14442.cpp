#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include <limits>
using namespace std;
typedef vector<vector<int>> Map;
const int MAX = INT_MAX;
class Node
{
public:
	int r;
	int c;
	int l;	 //�� �Ѽ� Ƚ��.
	Node(int _r, int _c, int _l) :r(_r), c(_c), l(_l) {}
};
vector<Map> visited;
int n, m, k;
const int ways[4][2] =
{
	{-1,0},{0,1},{1,0},{0,-1}
};
int doBFS();
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n >> m >> k;
	visited = vector<Map>(k + 1, Map(n, vector<int>(m, 0)));
	for (int i = 0; i < n; i++)
	{
		string str_input;
		cin >> str_input;
		for (int j = 0; j < m; j++)
		{
			for (int l = 0; l <= k; l++)
			{
				if (str_input[j] == '1')
					visited[l][i][j] = -1;
				else
					visited[l][i][j] = 0;
			}
		}
	}		
	int ret = doBFS();
	if (ret == MAX)
		cout << -1 << endl;
	else
		cout << ret+1 << endl;
	return 0;
}
int doBFS()
{
	if (n == m == 1)
		return 0;
	queue<Node* >q;
	q.push(new Node(0, 0, 0));
	while (!q.empty())
	{
		Node* nodeFront = q.front();
		int this_r = nodeFront->r;
		int this_c = nodeFront->c;
		int this_l = nodeFront->l;
		q.pop();
		if (nodeFront->r == n - 1 && nodeFront->c == m - 1)
			continue;
		int next_layer[2] = { nodeFront->l,nodeFront->l + 1 };
		for (int i = 0; i < 2; i++)
		{
			if (next_layer[i] > k)
				continue;
			for (int wayIdx = 0; wayIdx < 4; wayIdx++)
			{
				int next_r = nodeFront->r + ways[wayIdx][0];
				int next_c = nodeFront->c + ways[wayIdx][1];
				int next_l = next_layer[i];
				bool isOkay = true;
				if (next_r < 0 || next_r >= n)
					isOkay = false;
				if (next_c < 0 || next_c >= m)
					isOkay = false;
				if (isOkay)
				{
					if (this_l == next_l)
					{
						if (visited[next_l][next_r][next_c] == 0)
							isOkay = true;
						else
							isOkay = false;
					}
					else
					{
						if (visited[next_l][next_r][next_c] == -1)
							isOkay = true;
						else
							isOkay = false;
					}
				}
				if (isOkay)
				{
					q.push(new Node(next_r, next_c, next_l));
					visited[next_l][next_r][next_c] = visited[this_l][this_r][this_c] + 1;
				}
			}
		}
		delete(nodeFront);
	}
	int ret = MAX;
	for (int l = 0; l <= k; l++)
		if(visited[l][n - 1][m - 1]!=0)
			ret = min(ret, visited[l][n - 1][m - 1]);
	return ret;
}
