#include<iostream>
#include<vector>
using namespace std;
void vectorsInit(vector<int>& layers, vector<int>& orders);
int retCounts(const vector<int>& layers, const vector<int>& orders, const int& higher, const int& lower);
int main()
{	
	vector<int> LayerNum(10012, 0);
	vector<int> OrderNum(10012, 0);
	vectorsInit(LayerNum,OrderNum);
	freopen("input.txt", "r", stdin);
	int TC; cin >> TC;
	//freopen("output.txt", "w", stdout); 
	for (int cases = 0; cases < TC; cases++)
	{
		int a, b;
		int higher, lower;
		cin >> a >> b;
		if (a < b)
		{
			higher = a;
			lower = b;
		}
		else
		{
			higher = b;
			lower = a;
		}
		cout << "#" << cases + 1 << " " << retCounts(LayerNum,OrderNum,higher,lower) << endl;
	}
}
void vectorsInit(vector<int>& layers, vector<int>& orders)
{
	int location = 1;
	for (int i = 1; i <= 141; i++)
		for (int j = 0; j < i; j++)
		{
			//cout << location << ", " << i<<" ||" ;
			//cout << location << ", " << j << endl;
			orders[location] = j;
			layers[location++] = i;
		}
}
int retCounts(const vector<int>& layers, const vector<int>& orders, const int& higher, const int& lower)
{
	int layerDiff = layers[lower] - layers[higher];
	int orderDiff = orders[lower] - orders[higher];
	if (orderDiff < 0)
		orderDiff *= -1;
	if (layerDiff==0)
		return orderDiff;
	//If lower point is in the boundary of the triangle
	if (orders[lower] >= orders[higher] && orders[lower] <= (orders[higher] + layerDiff))
		return layerDiff;
	else if (orders[lower] < orders[higher])
		return orderDiff + layerDiff;
	else
		return orderDiff;
}
