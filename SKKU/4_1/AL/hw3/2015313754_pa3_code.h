#ifndef __2015313754_PA3_CODE_H__
#define __2015313754_PA3_CODE_H__
#include<stdio.h>
#include<stdlib.h>

#define ALPHA_NUM 26
#define MAX_HEAP_SIZE 10000000
#define MAX_STR_LEN 300
typedef struct NODE
{
    short is_leaf;        //if this node is a leaf, this value is 1, or this one is internal
    short alpha;
    struct NODE * child_0;
    struct NODE * child_1;
    int cnt;
}Node;
typedef struct HEAP_NODE
{
    short is_leaf;
    short alpha;
    struct HEAP_NODE * child_0;
    struct NODE * child_1;

}Heap_Node;
const char * input_file = "hw3_input.txt";
// const char * input_file = "./inputs/input_0.txt";
const char * output_file1 = "hw3_output1.txt";
const char * output_file2 = "hw3_output2.txt";
const char * HEADERED = "HEADEREND";
const int HEADERED_LEN = 9;
const char AL_a = 'a';
const char AL_z = 'z';
const char AL_A = 'A';
const char AL_Z = 'Z';
const int BOOL_DEBUG=0;

static int alpha_cnt[ALPHA_NUM];
static int cnt_heap[MAX_HEAP_SIZE];
static Node * node_heap[MAX_HEAP_SIZE];
static int heap_size;
static Node * decode_heap[MAX_HEAP_SIZE];
static int decode_heap_size;

static int alpha_code_len[ALPHA_NUM];
static long long int alpha_code_num[ALPHA_NUM];



void read_input_file(FILE *);
void print_alpha_cnt();
void open_input_file(FILE ** fd);
void open_output_file1_R(FILE ** fd);
void open_output_file2_R(FILE ** fd);
void open_output_file1_W(FILE ** fd);
void open_output_file2_W(FILE ** fd);
void close_file(FILE ** fd);

void heap_init();
void heap_push(Node *);
void swap_heap_node(int a, int b);
Node * heap_pop(void);
Node * make_new_Node(short is_leaf, short alpha, Node * child_0, Node * child_1, int cnt);

void preTravel(Node * node);
void printHeap();

void encode_tree_2_code(Node * node, FILE * fd);
void record_masking(Node * node, int len, long long int num);
void check_record();
void decode_mask(int len, long long int num);
void decode_mask_out(int len, long long int num,FILE * fd);
void encode_input_string(FILE * fd_input, FILE * fd_output);


Node * decode_code_2_tree(char * code,int idx);
void decode_Heap_init();
void decode_Heap_push(Node * heap_node);
void decode_one_char(char * encoded_tree,int idx);

Node * construct_huffman_tree(FILE * fp_input, FILE * fp_output);

#endif