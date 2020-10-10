#include<iostream>
#include<vector>
using namespace std;
typedef pair<int, int> Point;
typedef vector<vector<int> > Matrix;
void initPoints(vector<Point>& points,const int & n);
void makeGraph(const vector<Point>& points, Matrix& Graph,const int &n, const int &k);
int retL1(const Point& p1, const Point& p2);
void showMatrix(const Matrix& matrix);
class State {
	public:
		int n;
		int k;
		int p;
		int q;
		State(int n_, int k_)
		{
			n = n_; k = k_;
	}
};
int findShortest(Matrix& cache, const Matrix& graph, State& state);
int main()
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	int TC; cin >> TC;
	for (int cases = 0; cases < TC; cases++)
	{
		int n, k;
		cin >> n >> k;
		vector<Point>points(n);
		Matrix graph(n, vector<int>(n, 0));
		Matrix cache(n, vector<int>(k + 1, -1));
		initPoints(points, n);
		makeGraph(points, graph, n, k);
		State state(n, k);
		state.p = 0;
		state.q = k;
		int ret = findShortest(cache, graph, state);
		cout << "#" << cases + 1 << " " << ret<< endl;
		showMatrix(graph);
		showMatrix(cache);
	}
}
void initPoints(vector<Point>& points, const int& n)
{
	for (int i = 0; i < n; i++)
		cin >> points[i].first >> points[i].second;
}
void makeGraph(const vector<Point>& points, Matrix& graph, const int& n, const int& k)
{
	for(int i=0;i<n-1;i++)
		for (int j = i + 1; j < i + k + 2; j++)
		{
			if (j >= n)
				break;
			graph[i][j] = retL1(points[i], points[j]);
		}
}
int retL1(const Point& p1, const Point& p2)
{
	int x = p1.first - p2.first;
	int y = p1.second - p2.second;
	if (x < 0)
		x *= -1;
	if (y < 0)
		y *= -1;
	return x + y;
}
int findShortest(Matrix& cache, const Matrix& graph, State &state)
{
	State orgState = state;
	if (state.p == (state.n - 1))
		return 0;
	int& ret = cache[state.p][state.q];
	if (ret != -1)
		return ret;
	int minLength = 987654321;
	for (int i = state.p+1; i <= state.p+state.q+1; i++)
	{
		if (i >= state.n)
			break;
		int length = graph[state.p][i];
		state.q = state.q-(i - state.p - 1);
		state.p = i;
		length += findShortest(cache, graph, state);
		state = orgState;		
		if (length < minLength)
			minLength = length;
	}
	return ret = minLength;	
}
void showMatrix(const Matrix& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
			cout << matrix[i][j] << " ";
		cout << endl;
	}
}

