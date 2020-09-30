//#include "Trie.h"
//#include <cstring>
//using namespace std;
//Trie::Trie() :is_Terminal(false) 
//{
//	memset(child, 0, sizeof(child));
//}
//Trie::~Trie()
//{
//	for (int i = 0; i < ALPHA_NUM; i++)
//		if (child[i] != 0)
//			delete child[i];
//}
//void Trie::insert(const char* key)
//{
//	if (*key == '\0')
//	{
//		is_Terminal = true;
//	}
//	else
//	{
//		int chrIdx = char2int(*key);
//		if (child[chrIdx] == 0)
//			child[chrIdx] = new Trie();
//		child[chrIdx]->insert(key + 1);
//	}
//}
//Trie* Trie::find(const char* key)
//{
//	if (*key == '\0')
//	{
//		return this;
//	}
//	else
//	{
//		int chrIdx = char2int(*key);
//		if (child[chrIdx] == 0)
//			return NULL;
//		return child[chrIdx]->find(key + 1);
//	}
//}
//bool Trie::string_exist(const char* key)
//{
//	if (*key == '\0')
//	{
//		if (is_Terminal == true)
//			return true;
//		else
//			return false;
//	}
//	int chrIdx = char2int(*key);
//	if (child[chrIdx] == 0)
//		return false;
//	return child[chrIdx]->string_exist(key + 1);
//}
//int char2int(char chr)
//{
//	return chr - 'a';
//}
