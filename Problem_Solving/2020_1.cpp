#include<iostream>
#include<vector>
#include<string>
#include<stack>
using namespace std;
bool check_right(string s); //check the string is correct or not
string get_right(string s); //return corrected string 
int split_s(string s); //return the index of start of v
bool DEBUG = false;
string solution(string p)
{
	string answer = "";
	return answer;
}
bool check_right(string s)
{
	stack<char> stack_s;
	for (int i = 0; i < s.size(); i++)
	{
		if (s[i] == '(')
			stack_s.push(s[i]);
		else
		{
			if (stack_s.empty())
				return false;
			else
				stack_s.pop();
		}
	}
	if (stack_s.empty())
		return true;
	else
		return false;
}
string get_right(string org_s)
{
	if (org_s.size() == 0)
	{
		return "";
	}
	int v_idx = split_s(org_s);
	string v(org_s.size() - v_idx, 'x');
	string u (v_idx,'x');
	for (int i = 0; i < u.size(); i++)
		u[i] = org_s[i];
	for (int i = 0; i < v.size(); i++)
		v[i] = org_s[v_idx + i];
	string ret;
	string new_v = get_right(v);
	if (check_right(u))
	{
		ret = u + new_v;
	}
	else
	{
		ret.push_back('(');
		ret += new_v;
		ret.push_back(')');
		if(u.size()!=2)
			for (int i = 1; i < u.size() - 1; i++)
			{
				if (u[i] == '(')
					ret += ')';
				else
					ret += '(';
			}
	}
	return ret;
}
int split_s(string s)
{
	int num_l = 0;
	int num_r = 0;
	int i;
	for (i = 0; i < s.size(); i++)
	{
		if (s[i] == '(')
		{
			num_l += 1;
		}
		else
		{
			num_r += 1;
		}
		if (num_l == num_r)
			return i+1;
	}
	return i+1;
}
int main()
{
	freopen("input.txt", "r",stdin);
	string org_input_s;
	cin >> org_input_s;
	if (org_input_s.size() == 2)
	{
		return 1;
	}
	string input_s (org_input_s.size() - 2, 'x');
	for (int i = 1; i < org_input_s.size() - 1; i++)
		input_s[i - 1] = org_input_s[i];
	cout <<"\""<< get_right(input_s) << "\"" <<endl;
}