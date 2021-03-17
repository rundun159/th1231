#include"2015313754_pa3_code.h"
#include<stdio.h>
#include<stdlib.h>
int main()
{
    FILE * fd_input;
    FILE * fd_output1;
    FILE * fd_output2;
    open_input_file(&fd_input);
    open_output_file1_W(&fd_output1);
    Node * rootNode;
    read_input_file(fd_input);
    close_file(&fd_input);

    print_alpha_cnt();
    for(int i=0;i<ALPHA_NUM;i++)
        if(alpha_cnt[i]!=0)
            heap_push(make_new_Node(1,'a'+i,NULL,NULL,alpha_cnt[i]));
    printHeap();
    while(heap_size!=1)
    {
        Node * node1 = heap_pop();
        printHeap();
        Node * node2 = heap_pop();
        printHeap();
        if(BOOL_DEBUG)
        {
            if(node1->is_leaf==1)
                printf("node 1 : (%c %d) ",node1->alpha,node1->cnt);
            else
                printf("node 1 : (%d) ",node1->cnt);
            if(node2->is_leaf==1)
                printf("node 2 : (%c %d) ",node2->alpha,node2->cnt);
            else
                printf("node 2 : (%d) ",node2->cnt);        
            printf("\n");
        }
        Node * new_internal_node = make_new_Node(0,0,node1,node2,node1->cnt+node2->cnt);
        heap_push(new_internal_node);
        printHeap();
    }
    rootNode=node_heap[1];
    printf("PreTravel\n");
    preTravel(rootNode);
    printf("\n End of PreTravel\n");

    encode_tree_2_code(rootNode,fd_output1);
    fputs(HEADERED,fd_output1);
    record_masking(rootNode,0,0);
    check_record();

    open_input_file(&fd_input);
    encode_input_string(fd_input,fd_output1);

    close_file(&fd_input);
    close_file(&fd_output1);

    open_output_file1_R(&fd_output1);
    open_output_file2_W(&fd_output2);
    
    construct_huffman_tree(fd_output1,fd_output2);

}
void read_input_file(FILE * fp)
{
    int input_c;
    while((input_c=fgetc(fp))!=EOF)
    {
        if(input_c>AL_z)
            continue;
        if(input_c<AL_A)
            continue;
        if(input_c>AL_Z && input_c<AL_a)
            continue;
        if(input_c<AL_a)
            input_c += AL_a-AL_A;
        alpha_cnt[input_c-AL_a]+=1;
    }
}
void print_alpha_cnt()
{
    if(BOOL_DEBUG)
    {
        for(int i=0;i<ALPHA_NUM;i++)
        {
            printf("%c : %d\n",i+AL_a,alpha_cnt[i]);
        }
    }
}
void heap_init()
{
    heap_size=0;
    for(int i=0;i<MAX_HEAP_SIZE;i++)
    {
        cnt_heap[i]=0;
        node_heap[i]=NULL;
    }
}
void heap_push(Node * node)
{
    int parent_idx;
    int now_idx;

    heap_size++;
    now_idx=heap_size;
    parent_idx = now_idx/2;

    cnt_heap[now_idx]=node->cnt;
    node_heap[now_idx]=node;
    while(parent_idx!=0)
    {
        int parent_cnt = cnt_heap[parent_idx];
        int now_cnt = cnt_heap[now_idx];
        if(parent_cnt<=now_cnt)
            return;
        else
        {
            swap_heap_node(parent_idx,now_idx);
            parent_idx/=2;
            now_idx/=2;
        }
    }
}
Node * heap_pop(void)
{
    if(heap_size==1)
    {
        heap_size--;
        return node_heap[1];
    }
    Node * return_node = node_heap[1];
    node_heap[1] = node_heap[heap_size];
    cnt_heap[1] = cnt_heap[heap_size];
    heap_size--;
    int now_idx = 1;
    int child_idx;
    while(now_idx<=heap_size)
    {
        int left_child_idx = now_idx*2;
        int right_child_idx = now_idx*2+1;
        int small_child_idx;
        if(left_child_idx>heap_size)
            break;
        if(right_child_idx>heap_size)
            small_child_idx=left_child_idx;
        else if(cnt_heap[left_child_idx]<=cnt_heap[right_child_idx])
            small_child_idx=left_child_idx;
        else
            small_child_idx=right_child_idx;
        if(cnt_heap[now_idx]>cnt_heap[small_child_idx])
            swap_heap_node(now_idx,small_child_idx);
        else
            break;
        now_idx=small_child_idx;
    }
    return return_node;
}
void swap_heap_node(int a, int b)
{
    int temp_cnt = cnt_heap[a];
    Node * temp_node = node_heap[a];  
    cnt_heap[a] = cnt_heap[b];
    node_heap[a] = node_heap[b];
    cnt_heap[b] = temp_cnt;
    node_heap[b] = temp_node;
    return;
}
Node * make_new_Node(short is_leaf, short alpha, Node * child_0, Node * child_1, int cnt)
{
    Node * newNode = (Node*)malloc(sizeof(Node));
    newNode->is_leaf=is_leaf;
    newNode->alpha=alpha;
    newNode->child_0=child_0;
    newNode->child_1=child_1;
    newNode->cnt=cnt;
    return newNode;
}
void preTravel(Node * node)
{
    // if(BOOL_DEBUG)
    // {
    if(node->is_leaf==1)
        printf("(%c %d) ",node->alpha,node->cnt);
    else
        printf("(%d) ",node->cnt);
    if(node->child_0!=NULL)
        preTravel(node->child_0);
    if(node->child_1!=NULL)
        preTravel(node->child_1);
    // return;
    // }
}
void printHeap()
{
    if(BOOL_DEBUG)
    {
        printf("printHeap : (heap size :%d) ",heap_size);
        for(int i=1;i<=heap_size;i++)
            printf("(%d) ",cnt_heap[i]);
        printf("\n");
    }
}

void encode_tree_2_code(Node * node, FILE * fd)
{
    if(node==NULL)
        return;
    if(node->is_leaf)
        fputc(node->alpha,fd);
    else
    {
        fputc('(',fd);
        encode_tree_2_code(node->child_0,fd);
        fputc(',',fd);
        encode_tree_2_code(node->child_1,fd);
        fputc(')',fd);
    }   
}
void record_masking(Node * node, int len, long long int num)
{
    if(node->is_leaf==1)
    {
        alpha_code_len[node->alpha-'a']=len;
        alpha_code_num[node->alpha-'a']=num;
    }
    else
    {
        if(node->child_0!=NULL)
            record_masking(node->child_0,len+1,(num<<1));
        if(node->child_1!=NULL)
            record_masking(node->child_1,len+1,(num<<1)+1);
    }    
}
void check_record()
{
    // if(BOOL_DEBUG)
        for(int i=0;i<ALPHA_NUM;i++)
        {
            printf("%c : ",'a'+i);
            printf("%d %lld \n",alpha_code_len[i],alpha_code_num[i]);
            decode_mask(alpha_code_len[i],alpha_code_num[i]);
            printf("\n");
        }
}
void decode_mask(int len, long long int num)
{
    long long int now;
    // if(BOOL_DEBUG)
    // {
        for(int i=len-1;i>=0;i--)
        {
            now = 1<<i;
            // printf("num: %lld, now : %lld\n",num,now);
            if(num>=now)
            {
                printf("1");
                num-=now;
            }
            else
                printf("0");
        }
    // }
}
void decode_mask_out(int len, long long int num,FILE * fd)
{
    long long int now;
    for(int i=len-1;i>=0;i--)
    {
        now = 1<<i;
        // printf("num: %lld, now : %lld\n",num,now);
        if(num>=now)
        {
            fputc('1',fd);
            num-=now;
        }
        else
            fputc('0',fd);
    }
}
void encode_input_string(FILE * fd_input, FILE * fd_output)
{
    int input_c;
    char alpha_idx;
    while((input_c=fgetc(fd_input))!=EOF)
    {
        if(input_c>AL_z)
            continue;
        if(input_c<AL_A)
            continue;
        if(input_c>AL_Z && input_c<AL_a)
            continue;
        if(input_c<AL_a)
            input_c += AL_a-AL_A;
        alpha_idx = input_c-AL_a;
        decode_mask_out(alpha_code_len[alpha_idx],alpha_code_num[alpha_idx],fd_output);
        if(BOOL_DEBUG)
            decode_mask(alpha_code_len[alpha_idx],alpha_code_num[alpha_idx]);
    }    
}

Node * decode_code_2_tree(char * code,int idx)
{
    Node * node;
    if(code[idx]=='(')
        node = make_new_Node(0,0,NULL,NULL,0);
    else if(code[idx]==')')
        return NULL;
    else
        return make_new_Node(1,code[idx],NULL,NULL,0);
    node->child_0 = decode_code_2_tree(code,idx+1);
}
void decode_Heap_init()
{
    decode_heap_size=0;
}
void decode_Heap_push(Node * heap_node)
{
    decode_heap[++decode_heap_size]=heap_node;
}
void decode_one_char(char * encoded_tree,int idx)
{
    Node * new_heap_node;
    if(encoded_tree[idx]=='(')
        new_heap_node = make_new_Node(0,'(',NULL,NULL,0);
    else if(encoded_tree[idx]==',')
        new_heap_node = make_new_Node(0,',',NULL,NULL,0);
    else if(encoded_tree[idx]==')')
    {
        new_heap_node = make_new_Node(0,' ',decode_heap[decode_heap_size-2],decode_heap[decode_heap_size],0);
        decode_heap_size-=4;
    }
    else
        new_heap_node = make_new_Node(1,encoded_tree[idx],NULL,NULL,0);
    decode_Heap_push(new_heap_node);
    return;    
}
void open_input_file(FILE ** fd)
{
    *fd = fopen(input_file,"r");
}
void open_output_file1_R(FILE ** fd)
{
    *fd = fopen(output_file1,"r");
}
void open_output_file2_R(FILE ** fd)
{
    *fd = fopen(output_file2,"r");
}
void open_output_file1_W(FILE ** fd)
{
    *fd = fopen(output_file1,"w");
}
void open_output_file2_W(FILE ** fd)
{
    *fd = fopen(output_file2,"w");
}
void close_file(FILE ** fd)
{
    fclose(*fd);
}
Node * construct_huffman_tree(FILE * fp_input, FILE * fp_output)
{
    char c;
    int idx =0;
    int tree_str_len =0;
    char * tree_str;
    Node * rootNode;
    Node * node;
    int start = 1;
    decode_Heap_init();
    while((c=fgetc(fp_input))!='H')
        tree_str_len++;
    fseek(fp_input,0,SEEK_SET);
    tree_str = (char *)malloc(sizeof(tree_str_len));
    for(int i=0;i<tree_str_len;i++)
        tree_str[i]=fgetc(fp_input);
    for(int i=0;i<tree_str_len;i++)
        decode_one_char(tree_str,i);

    rootNode = decode_heap[1];

    printf("Start of encode_tree_2_code\n");
    preTravel(rootNode);
    encode_tree_2_code(rootNode,stdout);

    printf("\nWorks\n");                                    

    for(int i=0;i<HEADERED_LEN;i++)
        fgetc(fp_input);
    rootNode = decode_heap[1];

    for(int i=0;i<ALPHA_NUM;i++)
        if(alpha_code_len[i]!=0)
        {
            fputc(AL_a+i,fp_output);
            fputs(" :",fp_output);
            decode_mask_out(alpha_code_len[i],alpha_code_num[i],fp_output);
            fputs("\n",fp_output);            
        }

    while((c=fgetc(fp_input))!=EOF)
    {
        if(start==1)
            node = rootNode;
        
        if(c=='0')
        {
            // printf("0");
            node = node->child_0;
        }
        else
        {
            // printf("1");
            node = node->child_1;
        }

        if(node->is_leaf==1)
        {
            fputc(node->alpha,fp_output);
            start=1;
        }
        else
        {
            start=0;
        }
    }
}
