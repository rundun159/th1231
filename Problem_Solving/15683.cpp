//1�ð� 16��. �������� �ʰ� ������߰�
//���ܰ� � ��Ȳ�� ������
//���� ���α׷��� �� �۵��ϴ��� ���ø� �����ذ��� Ȯ���ߴ�.

#include<iostream>
#include<vector>
#include<algorithm>
#include<limits.h>
using namespace std;
typedef vector<vector<int>> Map;
class Cam
{
public:
	int r;
	int c;
	int k;
	Cam(int _r, int _c, int _k) :r(_r), c(_c), k(_k) {}
};
int n, m;
int cam_num;
const int MAX_WAY[6] = { 0,4,2,4,4,1 };
const int WAY_DIR[4][2] = {
	{0,1}, {1,0}, {0,-1}, {-1,0}
};
vector<Cam*> cam_v;
vector<int> max_way;
vector<int> CAM_K;
int mainDFS(const Map& map);
void painMap(Cam * cam, Map& map,int way);
void eraseMap(Cam* cam, Map& map, int way);
void drawWayMap(Map& map, int way_idx, Cam* cam);
void eraseWayMap(Map& map, int way_idx, Cam* cam);
int normalize(vector<int>& cam_way);
int main()
{
	freopen("input.txt", "r", stdin);
	cin >> n >> m;
	Map org_map = Map(n, vector<int>(m, 0));
	cam_num = 0;
	for(int i=0;i<n;i++)
		for (int j = 0; j < m; j++)
		{
			cin >> org_map[i][j];
			if (org_map[i][j] != 0 && org_map[i][j] != 6)
			{
				cam_num++;
				cam_v.push_back(new Cam(i, j, org_map[i][j]));
				max_way.push_back(MAX_WAY[cam_v[cam_num - 1]->k]);
				CAM_K.push_back(cam_v[cam_num - 1]->k);
			}
		}
	cout << mainDFS(org_map) << endl;
	return 0;
}
int mainDFS(const Map & map)
{
	Map map_dfs = map;
	vector<int> before_ways(cam_num, 0);
	vector<int> ways(cam_num, 0);
	bool finished = false;
	int ret = INT_MAX;
	if (cam_num == 0)
	{
		int blank = 0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				if (map_dfs[i][j] == 0)
					blank++;
		return blank;
	}
	while (!finished)
	{
		//for (int i = 0; i < cam_num; i++)
		//	cout << ways[i] << " ";
		//cout << endl;
		if (!normalize(ways))
			finished = true;
		if (finished)
			return ret;
		for (int i = 0; i < cam_num; i++)
			painMap(cam_v[i], map_dfs, ways[i]);
		int blank = 0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				if (map_dfs[i][j] == 0)
					blank++;
		ret = min(ret, blank);
		for (int i = 0; i < cam_num; i++)
			eraseMap(cam_v[i], map_dfs, ways[i]);
		ways[0]++;
	}
}
void painMap(Cam* cam, Map& map, int way)
{
	if (cam->k == 1)
	{
		drawWayMap(map, way, cam);
	}
	else if (cam->k == 2)
	{	
		if (way == 0)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 2, cam);
		}
		else if (way == 1)
		{
			drawWayMap(map, 1, cam);
			drawWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 3)
	{
		if (way == 0)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 3, cam);
		}
		else if (way == 1)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 1, cam);
		}
		else if (way == 2)
		{
			drawWayMap(map, 1, cam);
			drawWayMap(map, 2, cam);
		}
		else if (way == 3)
		{
			drawWayMap(map, 2, cam);
			drawWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 4)
	{
		if (way == 0)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 2, cam);
			drawWayMap(map, 3, cam);
		}
		else if (way == 1)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 1, cam);
			drawWayMap(map, 3, cam);
		}
		else if (way == 2)
		{
			drawWayMap(map, 0, cam);
			drawWayMap(map, 1, cam);
			drawWayMap(map, 2, cam);
		}
		else if (way == 3)
		{
			drawWayMap(map, 1, cam);
			drawWayMap(map, 2, cam);
			drawWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 5)
	{
		drawWayMap(map, 0, cam);
		drawWayMap(map, 1, cam);
		drawWayMap(map, 2, cam);
		drawWayMap(map, 3, cam);
	}
}
void eraseMap(Cam* cam, Map& map, int way)
{
	if (cam->k == 1)
	{
		eraseWayMap(map, way, cam);
	}
	else if (cam->k == 2)
	{
		if (way == 0)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 2, cam);
		}
		else if (way == 1)
		{
			eraseWayMap(map, 1, cam);
			eraseWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 3)
	{
		if (way == 0)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 3, cam);
		}
		else if (way == 1)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 1, cam);
		}
		else if (way == 2)
		{
			eraseWayMap(map, 1, cam);
			eraseWayMap(map, 2, cam);
		}
		else if (way == 3)
		{
			eraseWayMap(map, 2, cam);
			eraseWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 4)
	{
		if (way == 0)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 2, cam);
			eraseWayMap(map, 3, cam);
		}
		else if (way == 1)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 1, cam);
			eraseWayMap(map, 3, cam);
		}
		else if (way == 2)
		{
			eraseWayMap(map, 0, cam);
			eraseWayMap(map, 1, cam);
			eraseWayMap(map, 2, cam);
		}
		else if (way == 3)
		{
			eraseWayMap(map, 1, cam);
			eraseWayMap(map, 2, cam);
			eraseWayMap(map, 3, cam);
		}
	}
	else if (cam->k == 5)
	{
		eraseWayMap(map, 0, cam);
		eraseWayMap(map, 1, cam);
		eraseWayMap(map, 2, cam);
		eraseWayMap(map, 3, cam);
	}

}
void drawWayMap(Map& map, int way_idx, Cam * cam)
{
	int t_r = cam->r;
	int t_c = cam->c;
	t_r += WAY_DIR[way_idx][0];
	t_c += WAY_DIR[way_idx][1];
	while (t_r >= 0 && t_r < n && t_c >= 0 && t_c < m)
	{
		//cout << " r : " << t_r << " c: " << t_c << endl;
		if (map[t_r][t_c] == 6)
			return;
		else if (map[t_r][t_c] == 0)
			map[t_r][t_c] = -1;
		t_r += WAY_DIR[way_idx][0];
		t_c += WAY_DIR[way_idx][1];
	}
	return;
}
void eraseWayMap(Map& map, int way_idx, Cam* cam)
{
	int t_r = cam->r;
	int t_c = cam->c;
	t_r += WAY_DIR[way_idx][0];
	t_c += WAY_DIR[way_idx][1];
	while (t_r >= 0 && t_r < n && t_c >= 0 && t_c < m)
	{
		if (map[t_r][t_c] == 6)
			return;
		else if (map[t_r][t_c] == -1)
			map[t_r][t_c] = 0;
		t_r += WAY_DIR[way_idx][0];
		t_c += WAY_DIR[way_idx][1];
	}
	return;
}
int normalize(vector<int>& cam_way) //return 0 if it is finished
{
	for (int i = 0; i < cam_num; i++)
	{
		if (cam_way[i] == MAX_WAY[CAM_K[i]])
		{
			if (i == cam_num - 1)
				return 0;
			cam_way[i] = 0;
			if(i!=cam_num-1)
				cam_way[i + 1] += 1;
		}
		else
			return 1;
	}
	return 1;
}
