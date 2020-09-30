//int t;
//int* array;
//int length;
//int count;
//
//int count3()
//{
//	int i;
//	count = 0;
//	for (i = 0; i < t; i++)
//		count += thread_create(count3s_thread, i);
//	
//	return count;
//}
//
//int count3s_thread(int id)
//{
//	int length_per_thread = length / t;
//	int start = id * length_per_thread;
//	int count_t = 0;
//	for (int i = start; i < start + length_per_thread; i++)
//	{
//		if (array[i] == 3)
//			count_t++;
//	}
//	return count_t;
//}