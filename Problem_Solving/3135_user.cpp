#define ALPHABET_NUM 26
int char2int(char chr);
struct Trie
{
	bool is_Terminal;
	int childNum;
	Trie* child[ALPHABET_NUM];
	Trie() :is_Terminal(false), childNum(0)
	{
		for (int i = 0; i < ALPHABET_NUM; i++)
			child[i] = 0;
	}
	void insert_Trie(int buff_size, char* buf)
	{
		childNum++;
		if (buff_size == 0)
		{
			is_Terminal = true;
			return;
		}
		int chrInt = char2int(*buf);
		if (child[chrInt] == 0)
			child[chrInt] = new Trie();
		child[chrInt]->insert_Trie(buff_size - 1, buf + 1);
	}
	Trie* find(int buff_size, char* buf)
	{
		if (buff_size == 0)
			return this;
		else
		{
			int chrInt = char2int(*buf);
			if (child[chrInt] == 0)
				return nullptr;
			return child[chrInt]->find(buff_size - 1, buf + 1);
		}
	}
	int TreeNum()
	{
		return childNum;
	}
};
Trie* rootNode;
void init(void) {
	rootNode = new Trie();
}

void insert(int buffer_size, char* buf) {
	rootNode->insert_Trie(buffer_size, buf);
}

int query(int buffer_size, char* buf) {
	Trie* foundOne = rootNode->find(buffer_size, buf);
	if (foundOne == nullptr)
		return 0;
	return foundOne->childNum;
}
int char2int(char chr)
{
	return chr - 'a';
}
