#include <stdio.h>
#define BUFFER_SIZE 1<<8
#define MAX_BLOCK_N 1<<20
#define MAX_M 1<<4
#define MAX_R 1<<6
#define MAX_T 10
const char * input_file = "hw1_input.txt";
const char * output_file = "hw1_output.txt";
const char * KEY = "(Key)";
const char OPEN_BR = '(';
const char DELIMITER = ':';
const char NEW_LINE = '\n';
const char SPACE = ' ';
const char NUM_0 = '0';
const char NUM_9 = '9';
const char AL_a = 'a';
const char AL_z = 'z';
const char AL_A = 'A';
const char AL_Z = 'Z';
static char org_data[MAX_BLOCK_N][(MAX_M)*(MAX_T)];
static int org_data_len[MAX_BLOCK_N];
static unsigned char data_block[MAX_BLOCK_N][MAX_M];
static unsigned char temp_block[MAX_BLOCK_N][MAX_M];
static int now_block_num=0; // 0:data_block 1:temp_block
static int cnt_list[MAX_R];
static int data_map_list[MAX_BLOCK_N]; // records where the original index maps to where in block
static int data_org_map_list[MAX_BLOCK_N]; //records the block contains which original index information
static int temp_map_list[MAX_BLOCK_N]; // records where the original index maps to where in block
static int temp_org_map_list[MAX_BLOCK_N]; //records the block contains which original index information
int SET_ZERO=1;

void cnt_sort(int n, int idx);

int main()
{
    int input;
    int data_size=0;
    int attribute_num=1;
    int key_attribute_idx = 0;
    int key_same_cnt=0;
    const int key_same_cond = 5;
    static char input_buffer[BUFFER_SIZE];
    char * input_state;
    int idx=0;
    int org_idx;
    FILE * fd_input = fopen(input_file,"r");
    FILE * fd_output = fopen(output_file,"w");
    if(fd_input==NULL)
    {
        printf("Not open\n");
        return 0;
    }
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
    //n is stored in input buffer
    while(input_buffer[idx]!='\n')
    {
        data_size*=10;
        data_size += input_buffer[idx]-'0';
        idx++;
    }
    // gets $
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
    // gets list of attributes
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
    idx=0;
    while(input_buffer[idx]!='\n')
    {
        if(input_buffer[idx]==DELIMITER)
            attribute_num++;
        else if(input_buffer[idx]==OPEN_BR)
            key_attribute_idx=attribute_num-1;
        idx++;
    }
    // gets $
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
    // save starting pointer of data
    // get data
    if(SET_ZERO)
    {
        for(int i=0;i<MAX_BLOCK_N;i++)
            for(int j=0;j<MAX_M;j++)
                data_block[i][j]=0;
        for(int i=0;i<MAX_BLOCK_N;i++)
            for(int j=0;j<MAX_M;j++)
                temp_block[i][j]=0;
        for(int i=0;i<MAX_R;i++)
            cnt_list[i]=0;
    }
    for(int data_idx=0;data_idx<data_size;data_idx++)
    {
        input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
        int deli_cnt=0;
        int buff_idx=0;
        int data_m_idx=0;
        while(deli_cnt!=key_attribute_idx)
        {
            if(input_buffer[buff_idx]==DELIMITER)
            {
                deli_cnt++;
                buff_idx++;
            }
            else
                buff_idx++;
        }
        // printf("deli_cnt : %d key_attribute_idx : %d \n",deli_cnt,key_attribute_idx);
        while(input_buffer[buff_idx]!=DELIMITER && input_buffer[buff_idx]!=NEW_LINE && input_buffer[buff_idx]!=EOF && input_buffer[buff_idx]!=0 )
        {
            unsigned char data=0;
            char b_data = input_buffer[buff_idx];
            if(b_data<NUM_0) //space
                data_block[data_idx][data_m_idx]=1;
            else if(b_data<AL_A) //numerical
                data_block[data_idx][data_m_idx]=b_data-NUM_0+2;
            else if(b_data<AL_a) //Upper letters
                data_block[data_idx][data_m_idx]=2*(b_data-AL_A)+13;
            else                //Lower letters
                data_block[data_idx][data_m_idx]=2*(b_data-AL_a)+12;
            // printf("%c -> %d \n",b_data,data_block[data_idx][data_m_idx]);
            buff_idx++;
            data_m_idx++;
        }
        buff_idx=-1;
        do
        {
            buff_idx++;
            // printf("buff_idx : %d\n",buff_idx);
            org_data[data_idx][buff_idx]=input_buffer[buff_idx];
        } while(input_buffer[buff_idx]!=NEW_LINE&&input_buffer[buff_idx]!=0);        
        org_data_len[data_idx]=buff_idx+1;
    }


    for(int data_idx=0;data_idx<data_size;data_idx++)
    {
        data_map_list[data_idx]=data_idx;
        data_org_map_list[data_idx]=data_idx;
        temp_map_list[data_idx]=data_idx;
        temp_org_map_list[data_idx]=data_idx;
    }
    for(int col_idx=14;col_idx>=0;col_idx--)
        cnt_sort(data_size,col_idx);

    for(int i=0;i<data_size-1;i++)
    {
        org_idx = temp_org_map_list[i];
        fwrite(org_data[org_idx],1,org_data_len[org_idx]-1,fd_output);
        fputc(NEW_LINE,fd_output);
    }
    org_idx = temp_org_map_list[data_size-1];
    fwrite(org_data[org_idx],1,org_data_len[org_idx]-1,fd_output);

    return 0;    
}
void cnt_sort(int data_size, int col_idx) //count sort the directed index
{
    for(int i=0;i<MAX_R;i++)
        cnt_list[i]=0;
    if(now_block_num==0)
    {
        for(int data_idx=0;data_idx<data_size;data_idx++)
        {
            unsigned char d_data = data_block[data_idx][col_idx];
            cnt_list[d_data]++;
        }
        for(int i=1;i<MAX_R;i++)        //getting prefix sum
            cnt_list[i]+=cnt_list[i-1];
        for(int data_idx=data_size-1;data_idx>=0;data_idx--)
        {
            unsigned char d_data = data_block[data_idx][col_idx];
            int next_idx = cnt_list[d_data]-1;
            for(int m_idx=0;m_idx<MAX_M;m_idx++)  
            {
                temp_block[next_idx][m_idx]=data_block[data_idx][m_idx];
            }
            int org_idx = data_org_map_list[data_idx];
            temp_map_list[org_idx]=next_idx;
            temp_org_map_list[next_idx]=org_idx;
            cnt_list[d_data]-=1;
        }
        now_block_num=1;
    }
    else
    {
        for(int data_idx=0;data_idx<data_size;data_idx++)
        {
            unsigned char d_data = temp_block[data_idx][col_idx];
            cnt_list[d_data]++;
        }
        for(int i=1;i<MAX_R;i++)        //getting prefix sum
            cnt_list[i]+=cnt_list[i-1];

        for(int data_idx=data_size-1;data_idx>=0;data_idx--)
        {
            unsigned char d_data = temp_block[data_idx][col_idx];
            int next_idx = cnt_list[d_data]-1;
            for(int m_idx=0;m_idx<MAX_M;m_idx++)  
            {
                data_block[next_idx][m_idx]=temp_block[data_idx][m_idx];
            }
            int org_idx = temp_org_map_list[data_idx];
            data_map_list[org_idx]=next_idx;
            data_org_map_list[next_idx]=org_idx;
            cnt_list[d_data]-=1;
        }
        now_block_num=0;
    }

    return;
}
