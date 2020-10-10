#include<iostream>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
class Node
{
public:
	int now_a;
	int now_b;
	int now_c;
	Node(int _a, int _b, int _c) :now_a(_a), now_b(_b), now_c(_c) {}
};
const int MAX = 500 + 1;
int cache[MAX][MAX][MAX];
int a, b, c;
int ret();
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> a >> b >> c;
	cout << ret() << endl;
	return 0;
}
int ret()
{
	if (a == b && b == c && c == a)
		return 1;
	queue<Node*>q;
	q.push(new Node(a, b, c));
	cache[a][b][c] = 1;
	int cases_arr[3][3] = {
		{0,1,2},{1,2,0},{0,2,1}
	};
	while(!q.empty())
	{ 
		Node* nowNode = q.front();
		q.pop();
		int nums[3] = { nowNode->now_a,nowNode->now_b,nowNode->now_c };
		for (int k = 0; k < 3; k++)
		{
			int left = cases_arr[k][0], right = cases_arr[k][1],other = cases_arr[k][2];
			if (nums[left] != nums[right])
			{
				int sum = nums[left] + nums[right];
				int max_num = max(nums[left], nums[right]);
				for (int i = 0; i <= max_num; i++)
				{
					if (sum - i == i && i == nums[other] && sum - i == nums[other])
						return 1;
					if (nums[left] < nums[right])
					{
						if (cache[sum - i][i][nums[other]] == 0)
						{
							q.push(new Node(sum - i, i, nums[other]));
							cache[sum - i][i][nums[other]] = 1;
						}
					}
					else
					{
						if (cache[i][sum - i][nums[other]] == 0)
						{
							q.push(new Node(i, sum - i, nums[other]));
							cache[i][sum - i][nums[other]] = 1;
						}
					}
				}
			}
		}
	}
	return 0;
}
