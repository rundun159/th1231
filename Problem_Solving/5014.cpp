#include<iostream>
#include<vector>
#include<queue>
#include<math.h>
using namespace std;
class Node
{
public:
	int floor_idx;
	int steps;
	Node(int f, int s) :floor_idx(f), steps(s) {}
};
int Floor, Start, Goal, Up, Down;
int retSteps();
bool isOkay(int idx, int steps);
const int MAX = 1000000 + 1;
int visited[MAX];
int main()
{
	//freopen("input.txt", "r", stdin);
	cin >> Floor >> Start >> Goal >> Up >> Down;
	int ret = retSteps();
	if (ret == -1)
		cout << "use the stairs" << endl;
	else
		cout << ret << endl;
	return 0;
}
int retSteps()
{
	int way[2] = { Up,-1 * Down };
	queue<Node*>q;
	q.push(new Node(Start, 1));
	visited[Start] = 1;
	while (!q.empty())
	{
		Node* nowNode = q.front();
		q.pop();
		for (int i = 0; i < 2; i++)
		{
			int this_step = way[i];
			int next_f = nowNode->floor_idx + this_step;
			if ((next_f >= 1) && (next_f <= Floor))
			{
				if (next_f == Goal)
					return nowNode->steps;
				if (visited[next_f] == 0)
				{
					q.push(new Node(next_f, nowNode->steps + 1));
					visited[next_f] = nowNode->steps + 1;
				}
			}
		}
	}
	return -1;
}
bool isOkay(int idx, int steps)
{
	if ((idx + steps < 1) || (idx + steps > Floor))
		return false;
	else
		return true;
}
