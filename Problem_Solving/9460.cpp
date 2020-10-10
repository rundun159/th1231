#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int T, n, k;
pair<double, double> pos[10001];
bool get_pos(double max_dist);
int main()
{
//	freopen("input.txt", "r", stdin);
	cin >> T;
	for (int tCase = 0; tCase < T; tCase++)
	{
		cin >> n >> k;
		for (int i = 0; i < n; i++)
			cin >> pos[i].first >> pos[i].second;
		double lo = 0.0;
		double hi = 100000000000.0;
		sort(pos, pos + n);
		while ((hi - lo) > 0.001)
		{
			double mid = (lo + hi) / 2;
			if (get_pos(mid))
				hi = mid;
			else
				lo = mid;
		}
		printf("%.1lf\n", lo);
	}
	return 0;
}
bool get_pos(double max_dist)
{
	int cnt = 1;
	double maxx = pos[0].second;
	double minn = pos[0].second;
	for (int i = 1; i < n; i++)
	{
		if (max(abs(maxx - pos[i].second) / 2, abs(minn - pos[i].second) / 2) <= max_dist)
		{
			maxx = max(maxx,pos[i].second);
			minn = min(minn, pos[i].second);
		}
		else
		{
			maxx = pos[i].second;
			minn = pos[i].second;
			cnt++;
		}
		if (cnt > k)
			return false;
	}
	if (cnt > k)
		return false;
	else
		return true;
}
