#include<iostream>
#include<vector>
using namespace std;
void make_matrix1(vector<vector<int> >& matrix1, const int & n);
int make_matrix2(vector<vector<int> >& matrix2, const vector<vector<int> > & matrix1,const int &n);
class Node
{
	int idx;
	vector<Node> child;
	int salary;
public:
	Node(int idx_, vector<bool>& selected)
	{
		idx = idx_;
		selected[idx] = true;
	}
	void makeChild(const vector<vector<int> >& matrix2, vector<bool>& selected, const int& n)
	{
		for (int j = 0; j < n; j++)
			if (matrix2[idx][j] && !selected[j])
				child.push_back(Node(j, selected));
		for (int i = 0; i < child.size(); i++)
			child[i].makeChild(matrix2, selected, n);
	}
	int retChildNum() const 
	{
		return child.size();
	}
	int retIdx()
	{
		return idx;
	}
	Node& retChild(int idx)
	{
		return child[idx];
	}
	const Node& retChild(int idx) const
	{
		return child[idx];
	}
	void setSalary(int sal)
	{
		salary = sal;
	}
	int retSalary() const
	{
		return salary;
	}
};
int countSalary(Node& node);
void travelTree(Node node)
{
	cout << "Idx: "<< node.retIdx()<<" Salary: "<<node.retSalary()<<" child :";
	for (int i = 0; i < node.retChildNum(); i++)
		cout << node.retChild(i).retIdx() << " ";
	cout << endl;
	for (int i = 0; i < node.retChildNum(); i++)
		travelTree(node.retChild(i));
}
int sumSalary(const Node & node);
int main()
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	int testCase;
	cin >> testCase;
	for (int cases = 0; cases < testCase; cases++)
	{
		int n;
		cin >> n;
		vector<vector<int> > matrix1(n, vector<int>(n, 0));
		vector<vector<int> > matrix2(n, vector<int>(n, 0));
		make_matrix1(matrix1, n);
		int rootNodeIdx=make_matrix2(matrix2, matrix1, n);
		int minSum = 987654321;
		for (int i = 0; i < n; i++)
		{
			vector<bool> selected(n, false);
			Node rootNode(i, selected);
			rootNode.makeChild(matrix2, selected, n);
			int sum = countSalary(rootNode);
			sum = sumSalary(rootNode);
			int selectedNum = 0;
			for (int i = 0; i < n; i++)
				selectedNum += selected[i];
			if (selectedNum==n&&sum < minSum)
				minSum=sum;
		}
		cout << "#" << cases + 1 << " " << minSum<< endl;
	}
}
void make_matrix1(vector<vector<int> >& matrix1,const int & n)
{
	for (int i = 0; i < n; i++)
	{
		int k;
		int m;
		cin >> k;
		for (int j = 0; j < k; j++)
		{
			cin >> m;
			matrix1[i][m-1] = 1;
		}
	}
}
int make_matrix2(vector<vector<int> >& matrix2, const vector<vector<int> >& matrix1, const int& n)
{
	int maxChild = -1;
	int rootNode = -1;
	for (int i = 0; i < n; i++)
	{
		int sum = 0;
		for (int j = 0; j < n; j++)
		{
			sum += matrix1[j][i];
			matrix2[i][j] = matrix1[j][i];
		}
		if (sum > maxChild)
		{
			maxChild = sum;
			rootNode = i;
		}
	}
	return maxChild;
}
int countSalary(Node & node)
{
	int sum = 0;
	int childNum = node.retChildNum();
	if (childNum == 0)
	{
		node.setSalary(1);
		return 1;
	}
	for (int i = 0; i < childNum; i++)
		sum += countSalary(node.retChild(i));
	node.setSalary(sum + 1);
	return sum + 1;
}
int sumSalary(const Node& node)
{
	int sum = node.retSalary();
	for (int i = 0; i < node.retChildNum(); i++)
		sum += sumSalary(node.retChild(i));
	return sum;
}
