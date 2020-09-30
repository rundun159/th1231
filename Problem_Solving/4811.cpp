//#include <iostream>
//#include <vector>
//using namespace std;
//typedef vector<vector<long long int>> CACHE;
//typedef pair<int, int> HW;
//int n;
//long long int  retCases(CACHE& cache,HW hw);
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	cin >> n;
//	while (n != 0)
//	{
//		CACHE cache = CACHE(n+1, vector<long long int>(n+1, -1));
//		cout << retCases(cache, HW(0, n))<<endl;
//		cin >> n;
//	}
//	return 0;
//}
//long long int retCases(CACHE& cache, HW hw)
//{
//	//cout << hw.first << ", " << hw.second << endl;
//	long long int& ret = cache[hw.first][hw.second];
//	if (ret != -1)
//		return ret;
//	else if (hw == HW(0, 0))
//		return ret = 1;
//	ret = 0;
//	if (hw.first != 0)
//		ret += retCases(cache, HW(hw.first - 1, hw.second));
//	if (hw.second != 0)
//		ret += retCases(cache, HW(hw.first+1, hw.second-1));
//	return ret;
//}
