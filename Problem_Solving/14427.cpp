#include<iostream>
#include<vector>
using namespace std;
typedef struct HEAPNODE
{
	int num;
	int ArrIdx;
}HeapNode;
typedef struct ARRNODE
{
	int num;
	int HeapIdx;
}ArrNode;
typedef struct
{
	int targetIdx;
	int newNum;
	int endIdx;
}NewInfo;
void init(vector<HeapNode>& Heap, vector<ArrNode>& NumIdx, int n);
void switchNode (vector<HeapNode>& Heap, vector<ArrNode>& Arr, int childIdx);
void sortHeap(vector<HeapNode> & Heap, vector<ArrNode> & Arr, const NewInfo & q);
int compNode(const vector<HeapNode>& Heap, int nodeIdx1, int nodeIdx2);
void showHeap(const vector<HeapNode>& Heap,int n);
void showArr(const vector<ArrNode>& Arr, int n);
int pro14427()//
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	NewInfo info;
	int N; cin >> N;
	info.endIdx = N;
	vector<HeapNode> Heap(N + 1);
	vector<ArrNode> Arr(N + 1);
	init(Heap, Arr, N);
	//cout << "Show Heap" << endl;
	//showHeap(Heap,N);
	//cout << "Show Arr" << endl;
	//showArr(Arr,N);
	int M; cin >> M;
	for (int i = 0; i < M; i++)
	{
		int types; cin >> types;
		//cout << "Types: " << types << endl;
		if (types == 1)
		{
			cin >> info.targetIdx >> info.newNum;
			sortHeap(Heap, Arr, info);
			//cout << "Show Heap" << endl;
			//showHeap(Heap, N);
			//cout << "Show Arr" << endl;
			//showArr(Arr, N);
		}
		else
			cout << Heap[1].ArrIdx << endl;
	}
	cout << "Show Heap" << endl;
	showHeap(Heap,N);
	cout << "Show Arr" << endl;
	showArr(Arr,N);
}
void init(vector<HeapNode>& Heap, vector<ArrNode>& Arr, int n)
{
	for (int i = 1; i <= n; i++)
	{
		cin >> Arr[i].num;
		Arr[i].HeapIdx = i;
		int nowHeapIdx = i;
		Heap[i].num = Arr[i].num;
		Heap[i].ArrIdx = i;
		while (nowHeapIdx != 1)
		{
			if (Heap[nowHeapIdx / 2].num < Heap[nowHeapIdx].num)
				break;
			else if (Heap[nowHeapIdx / 2].num == Heap[nowHeapIdx].num)
			{
				if (Heap[nowHeapIdx / 2].ArrIdx < Heap[nowHeapIdx].ArrIdx)
					break;
				else
					switchNode(Heap, Arr, nowHeapIdx);
			}
			else
			{
				switchNode(Heap, Arr, nowHeapIdx);
			}
			nowHeapIdx /= 2;
		}
	}
}
void switchNode(vector<HeapNode>& Heap, vector<ArrNode>& Arr, int childIdx)
{
	Arr[Heap[childIdx / 2].ArrIdx].HeapIdx = childIdx;
	Arr[Heap[childIdx].ArrIdx].HeapIdx = childIdx / 2;
	HeapNode temp = Heap[childIdx / 2];
	Heap[childIdx / 2] = Heap[childIdx];
	Heap[childIdx] = temp;
	return;
}
void sortHeap(vector<HeapNode>& Heap, vector<ArrNode>& Arr, const NewInfo& q)
{
	if (q.newNum < Arr[q.targetIdx].num)
	{
		Arr[q.targetIdx].num = q.newNum;
		Heap[Arr[q.targetIdx].HeapIdx].num = q.newNum;
		int nowHeapIdx = Arr[q.targetIdx].HeapIdx;
		while (nowHeapIdx != 1)
			if (compNode(Heap, nowHeapIdx, nowHeapIdx / 2))
			{
				switchNode(Heap, Arr, nowHeapIdx);
				nowHeapIdx = nowHeapIdx / 2;
			}
			else
				return;
	}
	else
	{
		Arr[q.targetIdx].num = q.newNum;
		Heap[Arr[q.targetIdx].HeapIdx].num = q.newNum;
		int nowHeapIdx = Arr[q.targetIdx].HeapIdx;
		while (q.endIdx>=(nowHeapIdx*2))
		{
			int smallerIdx;
			if ((nowHeapIdx * 2 + 1) > q.endIdx)
			{
				smallerIdx = nowHeapIdx * 2;
			}
			else
			{
				if (Heap[nowHeapIdx * 2].num < Heap[nowHeapIdx * 2 + 1].num)
				{
					smallerIdx = nowHeapIdx * 2;
				}
				else if(Heap[nowHeapIdx * 2].num == Heap[nowHeapIdx * 2 + 1].num)
				{
					if (Heap[nowHeapIdx * 2].ArrIdx < Heap[nowHeapIdx * 2 + 1].ArrIdx)
						smallerIdx = nowHeapIdx * 2;
					else
						smallerIdx = nowHeapIdx * 2+1;
				}
				else
				{
					smallerIdx = nowHeapIdx * 2+1;
				}
			}
			if (compNode(Heap, nowHeapIdx, smallerIdx))
				return;
			else
				switchNode(Heap, Arr, smallerIdx);
			nowHeapIdx = smallerIdx;
		}
	}
}
int compNode(const vector<HeapNode>& Heap, int nodeIdx1, int nodeIdx2)	//node1 < node2 => return 1 , node1 > node2 return 0
{
	if (Heap[nodeIdx1].num < Heap[nodeIdx2].num)
		return 1;
	else if (Heap[nodeIdx1].num == Heap[nodeIdx2].num)
	{
		if (Heap[nodeIdx1].ArrIdx < Heap[nodeIdx2].ArrIdx)
			return 1;
		else
			return 0;
	}
	else
	{
		return 0;
	}
}
void showHeap(const vector<HeapNode>& Heap, int n)
{
	for (int i = 1; i <= n; i++)
		cout << "HeapIdx : " << i << " num: " << Heap[i].num << " ArrIdx : " << Heap[i].ArrIdx << endl;
}
void showArr(const vector<ArrNode>& Arr, int n)
{
	for (int i = 1; i <= n; i++)
		cout << "ArrIdx : " << i << " num: " << Arr[i].num << " HeapIdx : " << Arr[i].HeapIdx << endl;

}
