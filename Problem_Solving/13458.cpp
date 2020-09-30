////문제이해를 잘 못함. 문제 이해에 많은 투자가 필요함.
////20분 걸림
//
//#include<iostream>
//#include<vector>
//#include<algorithm>
//
//using namespace std;
//
//int N, B, C;
//vector<int> A;
//int main()
//{
//	freopen("input.txt", "r", stdin);
//	cin >> N;
//	A = vector<int>(N);
//	for (int i = 0; i < N; i++)
//		cin >> A[i];
//	cin >> B >> C;
//	int maxBC = max(B, C);
//	long long int sum = 0;
//	for (int i = 0; i < N; i++)
//	{
//		if (A[i]<=B)
//			sum++;
//		else
//		{
//			int x = A[i] - B;
//			sum++;
//			sum += (x / C);
//			if ((x % C) != 0)
//				sum++;
//		}
//	}
//	cout << sum << endl;
//}