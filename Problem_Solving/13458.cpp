//�������ظ� �� ����. ���� ���ؿ� ���� ���ڰ� �ʿ���.
//20�� �ɸ�

#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

int N, B, C;
vector<int> A;
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> N;
	A = vector<int>(N);
	for (int i = 0; i < N; i++)
		cin >> A[i];
	cin >> B >> C;
	int maxBC = max(B, C);
	long long int sum = 0;
	for (int i = 0; i < N; i++)
	{
		if (A[i]<=B)
			sum++;
		else
		{
			int x = A[i] - B;
			sum++;
			sum += (x / C);
			if ((x % C) != 0)
				sum++;
		}
	}
	cout << sum << endl;
}