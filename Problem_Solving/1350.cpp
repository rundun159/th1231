#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;
int n;
string str_input;
bool ret_poss(int n);
int main()
{
	//freopen("input.txt", "r", stdin);
	cin >> n;
	cin >> str_input;
	bool found = false;
	int i;
	for (i = 1; i < n; i++)
	{
		found = ret_poss(i);
		if (found)
			break;
	}
	if (found)
		cout << i << endl;
	else
		cout << n << endl;
}

bool ret_poss(int len)
{
	int blockLen = n / len;
	int b;
	int l;
	for (int l = 0; l < len; l++)
	{
		int first = str_input[l];
		for (b = 1; b < blockLen; b++)
		{
			int now = str_input[b * len + l];
			if (first != now)
				return false;
		}
	}
	int left = n % len;
	for (int i = 0; i < left; i++)
	{
		int first = str_input[i];
		int now = str_input[b * len + i];
		if (first != now)
			return false;
	}
	return true;
}