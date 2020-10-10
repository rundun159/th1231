#include<iostream>
#include<vector>
#include<list>
#include<algorithm>
#define INT_MAX 987654321
using namespace std;
typedef pair<int, int> TF; //stands for Two and Five
typedef vector<vector<TF> > Matrix;
typedef list<TF> Cand;
typedef vector<vector<Cand> > CandMatrix;
TF retTF(int num);
void mapInit(Matrix& map, const int n);
int retState(const TF& mapTF, const TF& scoreTF1, const TF& scoreTF2);
int retState(const TF& prev, const TF& scoreTF2);
void fillScore(const Matrix& map, CandMatrix& score, const int n);
const TF INITIAL(-2, -2);
const TF ZERO(-1, -1);
const TF MAX_TF(987654321, 987654321);
void printCandMatrix(CandMatrix& candmatrix, int n);
int retMin(CandMatrix& candmatrix, int n);
int main()
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	int TC; cin >> TC;
	for (int cases = 0; cases < TC; cases++)
	{
		int n; cin >> n;
		Matrix map(n, vector<TF>(n, TF(0,0)));
		CandMatrix score(n, vector<Cand>(n));
		mapInit(map, n);
		fillScore(map, score, n);
		printCandMatrix(score,n);
		cout << "#"<<cases+1<<" "<<retMin(score, n) << endl;
	}
}
TF retTF(int num)
{
	TF ret(0, 0);
	if (num == 0)
		return TF(-1,-1);
	else
	{
		while ((num % 2) == 0)
		{
			ret.first++;
			num /= 2;
		}
		while ((num % 5) == 0)
		{
			ret.second++;
			num /= 5;
		}
	}
	return ret;
}
void mapInit(Matrix& map, const int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			int a; cin >> a;
			map[i][j] = retTF(a);
		}
}
void fillScore(const Matrix& map, CandMatrix & score, const int n)
{
	score[0][0].push_back(map[0][0]);
	for (int i = 1; i < n; i++)
	{
		for (int x = 0; x < i; x++)
		{
			int y = i - 1 - x;
			//x, y+1
			if (map[x][y+1]!=ZERO)
			{	
				for (Cand::iterator iter = score[x][y].begin(); iter != score[x][y].end(); iter++)
					score[x][y + 1].push_back(TF(iter->first+map[x][y+1].first,iter->second+map[x][y+1].second));
			}
		}
		for (int x = 0; x < i; x++)
		{
			int y = i  -1 - x;
			//x+1, y
			if (score[x+1][y].empty()&&(map[x+1][y] != ZERO))
			{
				for (Cand::iterator iter = score[x][y].begin(); iter != score[x][y].end(); iter++)
					score[x+1][y].push_back(TF(iter->first + map[x + 1][y].first, iter->second + map[x + 1][y].second));
			}
			else if(map[x + 1][y] != ZERO)
			{
				for (Cand::iterator iter = score[x][y].begin(); iter != score[x][y].end(); iter++)
				{
					int state;
					TF prev(map[x+1][y].first + iter->first, map[x + 1][y].second + iter->second);
					Cand::iterator iter2;
					for (iter2 = score[x+1][y].begin(); iter2 != score[x+1][y].end(); )
					{
						state = retState(prev, *iter2);
						if (state == -1)
							break;
						if (state == 1)
							iter2 = score[x + 1][y].erase(iter2);
						else
							iter2++;
					}
					if (state == -1)
						continue;
					score[x + 1][y].insert(iter2,prev);
				}
			}
		}
	}
	for (int i = 1; i <= (n - 1); i++)
	{
		int y = n - 1;
		for (int x = i; x <= (n - 1); x++)
		{
//			cout << x << ", " << y << "(first loop)" <<endl;
			if (map[x][y] == ZERO)
				continue;
			for (Cand::iterator iter = score[x - 1][y].begin(); iter != score[x - 1][y].end(); iter++)
				score[x][y].push_back(TF(iter->first + map[x][y].first, iter->second + map[x][y].second));
			y--;
		}
		y = n - 1;
		for (int x = i; x <= (n - 1); x++)
		{
//			cout << x << ", " << y << "(second loop)" << endl;
			if (map[x][y] == ZERO)
				continue;
			if (score[x][y].empty())
			{
				for (Cand::iterator iter = score[x][y - 1].begin(); iter != score[x][y - 1].end(); iter++)
					score[x][y].push_back(TF(iter->first + map[x][y].first, iter->second + map[x][y].second));
			}
			else
			{
				for (Cand::iterator iter = score[x][y-1].begin(); iter != score[x][y-1].end(); iter++)
				{
					int state;
					TF prev(map[x][y].first + iter->first, map[x][y].second + iter->second);
					Cand::iterator iter2;
					for (iter2 = score[x][y].begin(); iter2 != score[x][y].end(); )
					{
						state = retState(prev, *iter2);
						if (state == -1)
							break;
						if (state == 1)
							iter2 = score[x][y].erase(iter2);
						else
							iter2++;
					}
					if (state == -1)
						continue;
					score[x][y].insert(iter2, prev);
				}
			}
			y--;
		}
	}
}
int retState(const TF& mapTF, const TF& scoreTF1, const TF& scoreTF2)
{
	TF prev(mapTF.first + scoreTF1.first, mapTF.second + scoreTF1.second);
	if (prev.first >= scoreTF2.first)
	{
		if (prev.second >= scoreTF2.second)
			return -1;
		else 
			return 0;
	}
	else
	{
		if (prev.second >= scoreTF2.second)
			return 0;
		else
			return 1;
	}
}
int retState(const TF& prev, const TF& scoreTF2)
{
	if (prev.first > scoreTF2.first)
	{
		if (prev.second > scoreTF2.second)
			return -1;
		else
			return 0;
	}
	else
	{
		if (prev.second > scoreTF2.second)
			return 0;
		else
			return 1;
	}
}
void printCandMatrix( CandMatrix& candmatrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << i << ", " << j << " : ";
			for (Cand::iterator iter = candmatrix[i][j].begin(); iter != candmatrix[i][j].end(); iter++)
				cout << iter->first << ", " << iter->second << " ||";
			cout << endl;
		}
	}
}
int retMin(CandMatrix& candmatrix, int n)
{
	int ret = INT_MAX;
	int minNow;
	for(Cand::iterator iter = candmatrix[n-1][n-1].begin();iter!=candmatrix[n-1][n-1].end();iter++)
		ret = min(ret, min(iter->first, iter->second));
	return ret;
}
