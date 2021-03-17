#include<stdio.h>
#include<stdlib.h>
#define MAX_K 5
#define MAX_N 200
#define BUFFER_SIZE 1<<8
#define INT_MIN -987654321
#define INT_MAX 987654321
#define MULTIPLER 100
typedef struct Pos2
{
    int x,y;
}Pos2;
typedef struct Node2
{
    Pos2 pos;
    struct Node2 * prev;
    /* data */
}Node2;
typedef struct Pos3
{
    int x,y,z;
}Pos3;
typedef struct Node3
{
    Pos3 pos;
    struct Node3 * prev;
    /* data */
}Node3;
typedef struct Pos4
{
    int x,y,z,w;
}Pos4;
typedef struct Node4
{
    Pos4 pos;
    struct Node4 * prev;
    /* data */
}Node4;
typedef struct Pos5
{
    int x,y,z,w,p;
}Pos5;
typedef struct Node5
{
    Pos5 pos;
    struct Node5 * prev;
    /* data */
}Node5;

static int k;
static char input_sequence[MAX_K][MAX_N+1];
static char output_sequence[MAX_K][MAX_N*MULTIPLER];
static char star_sequence[MAX_N*MULTIPLER];
static short **array2;
static short ***array3;
static short ****array4;
static short *****array5;
static int input_sequence_len[MAX_K];
static int output_sequence_now_idx[MAX_K];
static char * input_file;
static char * output_file;
static int start_num=0;
int SET_ZERO=1;
void dp2_new();
void dp3_new();
void dp4_new();
void dp5_new();
void print_result2_new(FILE * fd_output);
void print_result3_new(FILE * fd_output);
void print_result4_new(FILE * fd_output);
void print_result5_new(FILE * fd_output);

const int way2[3][2]={
    {-1,-1},{-1,0},{0,-1}
}; 
const int way3[4][3]={
    {-1,-1,-1},{-1,0,0},{0,-1,0},{0,0,-1}
};
const int way4[5][4]={
    {-1,-1,-1,-1},{-1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,-1}
};
const int way5[6][5]={
    {-1,-1,-1,-1,-1},{-1,0,0,0,0},{0,-1,0,0,0},{0,0,-1,0,0},{0,0,0,-1,0},{0,0,0,0,-1}
};

const int bway2[3][2]={
    {-1,-1},{-1,0},{0,-1}
};
const int bway3[4][3]={
    {-1,-1,-1},{-1,0,0},{0,-1,0},{0,0,-1}
};
const int bway4[5][4]={
    {-1,-1,-1,-1},{-1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,-1}
};
const int bway5[6][5]={
    {-1,-1,-1,-1,-1},{-1,0,0,0,0},{0,-1,0,0,0},{0,0,-1,0,0},{0,0,0,-1,0},{0,0,0,0,-1}
};
void alloc2(int x, int y);
void alloc3(int x, int y, int z);
void alloc4(int x, int y, int z, int w);
void alloc5(int x, int y, int z, int w, int p);
void push2(Node2 **top,int x, int y);
Pos2* pop2(Node2 **top);
void push3(Node3 **top,int x, int y, int z);
Pos3* pop3(Node3 **top);
void push4(Node4 **top,int x, int y, int z, int w);
Pos4* pop4(Node4 **top);
void push5(Node5 **top,int x, int y, int z, int w,int p);
Pos5* pop5(Node5 **top);

int main(int argc, char * argv[])
{
    if(argc!=1)
    {
        input_file=argv[1];
        output_file=argv[2];
    }
    else
    {
        input_file="hw2_input.txt";
        output_file="hw2_output.txt";
    }
    static char input_buffer[BUFFER_SIZE];
    char * input_state;
    FILE * fd_input = fopen(input_file,"r");
    FILE * fd_output = fopen(output_file,"w");
    int idx=0;
    k=0;
    if(fd_input==NULL)
    {
        printf("Not open\n");
        return 0;
    }
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);

    while(input_buffer[idx]!='\n')
    {
        k*=10;
        k += input_buffer[idx]-'0';
        idx++;
    }
    input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
    for(int i=0;i<k;i++)
    {
        int buff_idx=0;
        input_sequence_len[i]=0;
        output_sequence_now_idx[i]=0;
        star_sequence[i]=0;
        input_state = fgets(input_buffer,BUFFER_SIZE,fd_input);
        while(input_buffer[buff_idx]!='\n'&&input_buffer[buff_idx]!=0)
        {
            input_sequence[i][buff_idx]=input_buffer[buff_idx];
            buff_idx++;
            input_sequence_len[i]++;
        }    
    }
    for(int i=k;i<=MAX_K;i++)
    {
        input_sequence_len[i]=0;
        output_sequence_now_idx[i]=0;
        star_sequence[i]=0;
    }

    switch(k)
    {
        case 2:
            alloc2(input_sequence_len[0],input_sequence_len[1]);
            dp2_new();
            print_result2_new(fd_output);
        break;
        case 3:
            alloc3(input_sequence_len[0],input_sequence_len[1],input_sequence_len[2]);
            dp3_new(); 
            print_result3_new(fd_output);
        break;
        case 4:
            alloc4(input_sequence_len[0],input_sequence_len[1],input_sequence_len[2],input_sequence_len[3]);
            dp4_new();
            print_result4_new(fd_output);
        break;
        case 5:
            alloc5(input_sequence_len[0],input_sequence_len[1],input_sequence_len[2],input_sequence_len[3],input_sequence_len[4]);
            dp5_new();
            print_result5_new(fd_output);
        break;
    }
}
void push2(Node2 **top,int x, int y)
{
    Node2 * new = (Node2*)malloc(sizeof(Node2));
    new->pos.x=x;
    new->pos.y=y;
    new->prev=*top;
    *top=new;
}
Pos2* pop2(Node2 **top)
{
    Pos2 * ret= (Pos2*)(malloc(sizeof(Pos2)));
    Node2 * prev_top= *top;
    ret->x = (*top)->pos.x;
    ret->y = (*top)->pos.y;
    *top=(*top)->prev;
    free(prev_top);
    return ret;
}

void push3(Node3 **top,int x, int y, int z)
{
    Node3 * new = (Node3*)malloc(sizeof(Node3));
    new->pos.x=x;
    new->pos.y=y;
    new->pos.z=z;
    new->prev=*top;
    *top=new;
}

Pos3* pop3(Node3 **top)
{
    Pos3 * ret= (Pos3*)(malloc(sizeof(Pos3)));
    Node3 * prev_top= *top;
    ret->x = (*top)->pos.x;
    ret->y = (*top)->pos.y;
    ret->z = (*top)->pos.z;
    *top=(*top)->prev;
    free(prev_top);
    return ret;
}
void push4(Node4 **top,int x, int y, int z, int w)
{
    Node4 * new = (Node4*)malloc(sizeof(Node4));
    new->pos.x=x;
    new->pos.y=y;
    new->pos.z=z;
    new->pos.w=w;
    new->prev=*top;
    *top=new;
}
Pos4* pop4(Node4 **top)
{
    Pos4 * ret= (Pos4*)(malloc(sizeof(Pos4)));
    Node4 * prev_top= *top;
    ret->x = (*top)->pos.x;
    ret->y = (*top)->pos.y;
    ret->z = (*top)->pos.z;
    ret->w = (*top)->pos.w;
    *top=(*top)->prev;
    free(prev_top);
    return ret;
}


void push5(Node5 **top,int x, int y, int z, int w, int p)
{
    Node5 * new = (Node5*)malloc(sizeof(Node5));
    new->pos.x=x;
    new->pos.y=y;
    new->pos.z=z;
    new->pos.w=w;
    new->pos.p=p;
    new->prev=*top;
    *top=new;
}
Pos5* pop5(Node5 **top)
{
    Pos5 * ret= (Pos5*)(malloc(sizeof(Pos5)));
    Node5 * prev_top= *top;
    ret->x = (*top)->pos.x;
    ret->y = (*top)->pos.y;
    ret->z = (*top)->pos.z;
    ret->w = (*top)->pos.w;
    ret->p = (*top)->pos.p;
    *top=(*top)->prev;
    free(prev_top);
    return ret;
}

void alloc2(int x, int y)
{
    array2 = (short**)malloc(sizeof(short*)*(x+1));
    for(int i=0;i<=x;i++)
        array2[i]=(short*)malloc(sizeof(short)*(y+1));
    for(int i=0;i<=x;i++)
        for(int j=0;j<=y;j++)
            array2[i][j]=0;
}
void alloc3(int x, int y, int z)
{
    array3 = (short***)malloc(sizeof(short**)*(x+1));
    for(int i=0;i<=x;i++)
    {
        array3[i]=(short**)malloc(sizeof(short*)*(y+1));
        for(int j=0;j<=y;j++)
            array3[i][j] = (short*)malloc(sizeof(short)*(z+1));
    }
    for(int i=0;i<=x;i++)
        for(int j=0;j<=y;j++)
            for(int k=0;k<=z;k++)
                array3[i][j][k]=0;
}
void alloc4(int x, int y, int z, int w)
{
    array4 = (short****)malloc(sizeof(short***)*(x+1));
    for(int i=0;i<=x;i++)
    {
        array4[i]=(short***)malloc(sizeof(short**)*(y+1));
        for(int j=0;j<=y;j++)
        {
            array4[i][j] = (short**)malloc(sizeof(short*)*(z+1));
            for(int k=0;k<=z;k++)
                array4[i][j][k]=(short*)malloc(sizeof(short)*(w+1));
        }
    }
    for(int i=0;i<=x;i++)
        for(int j=0;j<=y;j++)
            for(int k=0;k<=z;k++)
                for(int l=0;l<=w;l++)
                    array4[i][j][k][l]=0;

}

void alloc5(int x, int y, int z, int w, int p)
{
    array5 = (short*****)malloc(sizeof(short****)*(x+1));
    for(int i=0;i<=x;i++)
    {
        array5[i]=(short****)malloc(sizeof(short***)*(y+1));
        for(int j=0;j<=y;j++)
        {
            array5[i][j] = (short***)malloc(sizeof(short**)*(z+1));
            for(int k=0;k<=z;k++)
            {
                array5[i][j][k]=(short**)malloc(sizeof(short*)*(w+1));
                for(int m=0;m<=w;m++)
                   array5[i][j][k][m]=(short*)malloc(sizeof(short)*(p+1));
            }
        }
    }
    for(int i=0;i<=x;i++)
        for(int j=0;j<=y;j++)
            for(int k=0;k<=z;k++)
                for(int l=0;l<=w;l++)
                    for(int n=0;n<=p;n++)
                        array5[i][j][k][l][n]=0;

}


void dp2_new()
{
    for(int x=0;x<=input_sequence_len[0];x++)
        for(int y=0;y<=input_sequence_len[1];y++)
        {
            // printf("%d %d \n",i,j);
            if(x==0 && x==y)
                continue;
            int max_score = INT_MIN;
            for(int i=0;i<3;i++)
            {
                int x_=x+way2[i][0];
                int y_=y+way2[i][1];
                if(x_<0||y_<0)
                    continue;
                short score = array2[x_][y_];
                if(i==0)
                {
                    if(input_sequence[0][x-1]==input_sequence[1][y-1])
                        score+=1;
                }
                if(score>max_score)
                    max_score=score;
            }
            array2[x][y]=max_score;
        }      
}
void dp3_new()
{
    for(int x=0;x<=input_sequence_len[0];x++)
        for(int y=0;y<=input_sequence_len[1];y++)
            for(int z=0;z<=input_sequence_len[2];z++)
        {
            // printf("%d %d \n",i,j);
            if(x==0 && x==y && y==z)
                continue;
            int max_score = INT_MIN;
            for(int i=0;i<4;i++)
            {
                int x_=x+way3[i][0];
                int y_=y+way3[i][1];
                int z_=z+way3[i][2];
                if(x_<0||y_<0||z_<0)
                    continue;
                short score = array3[x_][y_][z_];
                if(i==0)
                {
                    if(input_sequence[0][x-1]==input_sequence[1][y-1]&&input_sequence[1][y-1]==input_sequence[2][z-1])
                        score+=1;
                }
                if(score>max_score)
                    max_score=score;
            }
            array3[x][y][z]=max_score;
        }      
}
void dp4_new()
{
    for(int x=0;x<=input_sequence_len[0];x++)
        for(int y=0;y<=input_sequence_len[1];y++)
            for(int z=0;z<=input_sequence_len[2];z++)
                for(int w=0;w<=input_sequence_len[3];w++)
        {
            // printf("%d %d \n",i,j);
            if(x==0 && x==y && y==z && z==w)
                continue;
            int max_score = INT_MIN;
            for(int i=0;i<5;i++)
            {
                int x_=x+way4[i][0];
                int y_=y+way4[i][1];
                int z_=z+way4[i][2];
                int w_=w+way4[i][3];
                if(x_<0||y_<0||z_<0||w_<0)
                    continue;
                short score = array4[x_][y_][z_][w_];
                if(i==0)
                {
                    if(input_sequence[0][x-1]==input_sequence[1][y-1]&&input_sequence[1][y-1]==input_sequence[2][z-1]&&input_sequence[2][z-1]==input_sequence[3][w-1])
                        score+=1;
                }
                if(score>max_score)
                    max_score=score;
            }
            array4[x][y][z][w]=max_score;
        }      
}


void dp5_new()
{
    for(int x=0;x<=input_sequence_len[0];x++)
        for(int y=0;y<=input_sequence_len[1];y++)
            for(int z=0;z<=input_sequence_len[2];z++)
                for(int w=0;w<=input_sequence_len[3];w++)
                    for(int p=0;p<=input_sequence_len[4];p++)
        {
            // printf("%d %d \n",i,j);
            if(x==0 && x==y && y==z && z==w && w==p)
                continue;
            int max_score = INT_MIN;
            for(int i=0;i<6;i++)
            {
                int x_=x+way5[i][0];
                int y_=y+way5[i][1];
                int z_=z+way5[i][2];
                int w_=w+way5[i][3];
                int p_=p+way5[i][4];
                if(x_<0||y_<0||z_<0||w_<0||p_<0)
                    continue;
                short score = array5[x_][y_][z_][w_][p_];
                if(i==0)
                {
                    if(input_sequence[0][x-1]==input_sequence[1][y-1]&&input_sequence[1][y-1]==input_sequence[2][z-1]&&input_sequence[2][z-1]==input_sequence[3][w-1]&&input_sequence[3][w-1]==input_sequence[4][p-1])
                        score+=1;
                }
                if(score>max_score)
                    max_score=score;
            }
            array5[x][y][z][w][p]=max_score;
        }      
}

void print_result2_new(FILE * fd_output)
{
    Node2 * top =NULL;
    int now_x=input_sequence_len[0];
    int now_y=input_sequence_len[1];
    int ret_x=0;
    int ret_y=0;
    int ret_x_len=0;
    int ret_y_len=0;
    int output_len=0;
    push2(&top,now_x,now_y);
    while(now_x!=0||now_y!=0)
    {
        if(input_sequence[0][now_x-1]==input_sequence[1][now_y-1])
            if(array2[now_x][now_y]==(array2[now_x-1][now_y-1]+1))
            {
                now_x--;
                now_y--;
                push2(&top,now_x,now_y);
                continue;
            }
        int max_idx=-1;
        int max_score = INT_MIN;
        int x_,y_;
        for(int i=0;i<3;i++)
        {
            x_ = now_x + bway2[i][0];
            y_ = now_y + bway2[i][1];
            if(x_<0||y_<0)
                continue;
            int score = array2[x_][y_];
            if(score>max_score)
            {
                max_idx=i;
                max_score=score;
            }
        }
        now_x = now_x + bway2[max_idx][0];
        now_y = now_y + bway2[max_idx][1];
        push2(&top,now_x,now_y);
    }
    // top = (Node*)malloc(sizeof(Node));
    output_sequence_now_idx[0]=0;
    output_sequence_now_idx[1]=0;
    while(top!=NULL)
    {
        Pos2* top_pos = pop2(&top);

        if(ret_x==top_pos->x&&ret_y==top_pos->y)
        {
            output_len--;
        }
        else if(ret_x==top_pos->x&&ret_y!=top_pos->y) //move down
        {
            output_sequence[0][output_len]='-';
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];            
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y==top_pos->y) //move right
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]='-';            
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y!=top_pos->y) //Move cross
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];
            if(output_sequence[0][output_len]==output_sequence[1][output_len])
            {
                star_sequence[output_len]='*';
                start_num++;
            }
            else
                star_sequence[output_len]=' ';
        }
        ret_x=top_pos->x;
        ret_y=top_pos->y;
        output_len++;
        free(top_pos);
    }
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<output_len;j++)
            fputc(output_sequence[i][j],fd_output);
        fputc('\n',fd_output);
    }
    for(int j=0;j<output_len;j++)
        fputc(star_sequence[j],fd_output);    
}
void print_result3_new(FILE * fd_output)
{
    Node3 * top =NULL;
    int now_x=input_sequence_len[0];
    int now_y=input_sequence_len[1];
    int now_z=input_sequence_len[2];
    int ret_x=0;
    int ret_y=0;
    int ret_z=0;
    int ret_x_len=0;
    int ret_y_len=0;
    int ret_z_len=0;
    int output_len=0;
    push3(&top,now_x,now_y,now_z);
    while(now_x!=0||now_y!=0||now_z!=0)
    {
        if(input_sequence[0][now_x-1]==input_sequence[1][now_y-1]
            &&input_sequence[1][now_y-1]==input_sequence[2][now_z-1])
            if(array3[now_x][now_y][now_z]==(array3[now_x-1][now_y-1][now_z-1]+1))
            {
                now_x--;
                now_y--;
                now_z--;
                push3(&top,now_x,now_y,now_z);
                continue;
            }
        int max_idx=-1;
        int max_score = INT_MIN;
        int x_,y_,z_;
        for(int i=0;i<4;i++)
        {
            x_ = now_x + bway3[i][0];
            y_ = now_y + bway3[i][1];
            z_ = now_z + bway3[i][2];
            if(x_<0||y_<0||z_<0)
                continue;
            int score = array3[x_][y_][z_];
            if(score>max_score)
            {
                max_idx=i;
                max_score=score;
            }
        }
        now_x = now_x + bway3[max_idx][0];
        now_y = now_y + bway3[max_idx][1];
        now_z = now_z + bway3[max_idx][2];
        
        push3(&top,now_x,now_y,now_z);
    }
    // top = (Node*)malloc(sizeof(Node));
    output_sequence_now_idx[0]=0;
    output_sequence_now_idx[1]=0;
    output_sequence_now_idx[2]=0;
    while(top!=NULL)
    {
        Pos3* top_pos = pop3(&top);

        if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z)
        {
            output_len--;
        }
        else if(ret_x==top_pos->x&&ret_y!=top_pos->y&&ret_z==top_pos->z) //move down
        {
            output_sequence[0][output_len]='-';
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];            
            output_sequence[2][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z) //move right
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z!=top_pos->z) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y!=top_pos->y&&ret_z!=top_pos->z) //Move cross
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            if(output_sequence[0][output_len]==output_sequence[1][output_len]
                &&output_sequence[1][output_len]==output_sequence[2][output_len])
                {
                star_sequence[output_len]='*';
                start_num++;
                }
            else
                star_sequence[output_len]=' ';
        }
        ret_x=top_pos->x;
        ret_y=top_pos->y;
        ret_z=top_pos->z;
        output_len++;
        free(top_pos);
    }
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<output_len;j++)
            fputc(output_sequence[i][j],fd_output);
        fputc('\n',fd_output);
    }
    for(int j=0;j<output_len;j++)
        fputc(star_sequence[j],fd_output);
}
void print_result4_new(FILE * fd_output)
{
    Node4 * top =NULL;
    int now_x=input_sequence_len[0];
    int now_y=input_sequence_len[1];
    int now_z=input_sequence_len[2];
    int now_w=input_sequence_len[3];
    int ret_x=0;
    int ret_y=0;
    int ret_z=0;
    int ret_w=0;
    int ret_x_len=0;
    int ret_y_len=0;
    int ret_z_len=0;
    int ret_w_len=0;
    int output_len=0;
    push4(&top,now_x,now_y,now_z,now_w);
    while(now_x!=0||now_y!=0||now_z!=0||now_w!=0)
    {
        if(input_sequence[0][now_x-1]==input_sequence[1][now_y-1]
            &&input_sequence[1][now_y-1]==input_sequence[2][now_z-1]
            &&input_sequence[2][now_z-1]==input_sequence[3][now_w-1])
            if(array4[now_x][now_y][now_z][now_w]==(array4[now_x-1][now_y-1][now_z-1][now_w-1]+1))
            {
                now_x--;
                now_y--;
                now_z--;
                now_w--;
                push4(&top,now_x,now_y,now_z,now_w);
                continue;
            }
        int max_idx=-1;
        int max_score = INT_MIN;
        int x_,y_,z_,w_;
        for(int i=0;i<5;i++)
        {
            x_ = now_x + bway4[i][0];
            y_ = now_y + bway4[i][1];
            z_ = now_z + bway4[i][2];
            w_ = now_w + bway4[i][3];
            if(x_<0||y_<0||z_<0||w_<0)
                continue;
            short score = array4[x_][y_][z_][w_];
            if(score>max_score)
            {
                max_idx=i;
                max_score=score;
            }
        }
        now_x = now_x + bway4[max_idx][0];
        now_y = now_y + bway4[max_idx][1];
        now_z = now_z + bway4[max_idx][2];
        now_w = now_w + bway4[max_idx][3];
        
        push4(&top,now_x,now_y,now_z,now_w);
    }
    // top = (Node*)malloc(sizeof(Node));
    output_sequence_now_idx[0]=0;
    output_sequence_now_idx[1]=0;
    output_sequence_now_idx[2]=0;
    output_sequence_now_idx[3]=0;
    while(top!=NULL)
    {
        Pos4* top_pos = pop4(&top);

        if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w)
        {
            output_len--;
        }
        else if(ret_x==top_pos->x&&ret_y!=top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w) //move down
        {
            output_sequence[0][output_len]='-';
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];            
            output_sequence[2][output_len]='-';
            output_sequence[3][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w) //move right
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            output_sequence[3][output_len]='-';            
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z!=top_pos->z&&ret_w==top_pos->w) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            output_sequence[3][output_len]='-';            
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w!=top_pos->w) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            output_sequence[3][output_len]=input_sequence[3][top_pos->w-1];
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y!=top_pos->y&&ret_z!=top_pos->z&&ret_w!=top_pos->w) //Move cross
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            output_sequence[3][output_len]=input_sequence[3][top_pos->w-1];
            if(output_sequence[0][output_len]==output_sequence[1][output_len]&&output_sequence[1][output_len]==output_sequence[2][output_len]&&output_sequence[2][output_len]==output_sequence[3][output_len])                
            {
                star_sequence[output_len]='*';
                start_num++;
                }
            else
                star_sequence[output_len]=' ';
        }
        ret_x=top_pos->x;
        ret_y=top_pos->y;
        ret_z=top_pos->z;
        ret_w=top_pos->w;
        output_len++;
        free(top_pos);
    }
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<output_len;j++)
            fputc(output_sequence[i][j],fd_output);
        fputc('\n',fd_output);
    }
    for(int j=0;j<output_len;j++)
        fputc(star_sequence[j],fd_output);
}
void print_result5_new(FILE * fd_output)
{
    Node5 * top =NULL;
    int now_x=input_sequence_len[0];
    int now_y=input_sequence_len[1];
    int now_z=input_sequence_len[2];
    int now_w=input_sequence_len[3];
    int now_p=input_sequence_len[4];
    int ret_x=0;
    int ret_y=0;
    int ret_z=0;
    int ret_w=0;
    int ret_p=0;
    int ret_x_len=0;
    int ret_y_len=0;
    int ret_z_len=0;
    int ret_w_len=0;
    int ret_p_len=0;
    int output_len=0;
    push5(&top,now_x,now_y,now_z,now_w,now_p);
    while(now_x!=0||now_y!=0||now_z!=0||now_w!=0||now_p!=0)
    {
        if(input_sequence[0][now_x-1]==input_sequence[1][now_y-1]
            &&input_sequence[1][now_y-1]==input_sequence[2][now_z-1]
            &&input_sequence[2][now_z-1]==input_sequence[3][now_w-1]
            &&input_sequence[3][now_w-1]==input_sequence[4][now_p-1])
            if(array5[now_x][now_y][now_z][now_w][now_p]==(array5[now_x-1][now_y-1][now_z-1][now_w-1][now_p-1]+1))
            {
                now_x--;
                now_y--;
                now_z--;
                now_w--;
                now_p--;
                push5(&top,now_x,now_y,now_z,now_w,now_p);
                continue;
            }
        int max_idx=-1;
        int max_score = INT_MIN;
        int x_,y_,z_,w_,p_;
        for(int i=0;i<6;i++)
        {
            x_ = now_x + bway5[i][0];
            y_ = now_y + bway5[i][1];
            z_ = now_z + bway5[i][2];
            w_ = now_w + bway5[i][3];
            p_ = now_p + bway5[i][4];
            if(x_<0||y_<0||z_<0||w_<0||p_<0)
                continue;
            short score = array5[x_][y_][z_][w_][p_];
            if(score>max_score)
            {
                max_idx=i;
                max_score=score;
            }
        }
        now_x = now_x + bway5[max_idx][0];
        now_y = now_y + bway5[max_idx][1];
        now_z = now_z + bway5[max_idx][2];
        now_w = now_w + bway5[max_idx][3];
        now_p = now_p + bway5[max_idx][4];
        
        push5(&top,now_x,now_y,now_z,now_w,now_p);
    }
    // top = (Node*)malloc(sizeof(Node));
    output_sequence_now_idx[0]=0;
    output_sequence_now_idx[1]=0;
    output_sequence_now_idx[2]=0;
    output_sequence_now_idx[3]=0;
    output_sequence_now_idx[4]=0;
    while(top!=NULL)
    {
        Pos5* top_pos = pop5(&top);

        if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w&&ret_p==top_pos->p)
        {
            output_len--;
        }
        else if(ret_x==top_pos->x&&ret_y!=top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w&&ret_p==top_pos->p) //move down
        {
            output_sequence[0][output_len]='-';
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];            
            output_sequence[2][output_len]='-';
            output_sequence[3][output_len]='-';
            output_sequence[4][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w&&ret_p==top_pos->p) //move right
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            output_sequence[3][output_len]='-';            
            output_sequence[4][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z!=top_pos->z&&ret_w==top_pos->w&&ret_p==top_pos->p) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            output_sequence[3][output_len]='-';            
            output_sequence[4][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w!=top_pos->w&&ret_p==top_pos->p) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            output_sequence[3][output_len]=input_sequence[3][top_pos->w-1];
            output_sequence[4][output_len]='-';
            star_sequence[output_len]=' ';
        }
        else if(ret_x==top_pos->x&&ret_y==top_pos->y&&ret_z==top_pos->z&&ret_w==top_pos->w&&ret_p!=top_pos->p) //move right
        {
            output_sequence[0][output_len]='-';            
            output_sequence[1][output_len]='-';            
            output_sequence[2][output_len]='-';            
            output_sequence[3][output_len]='-';
            output_sequence[4][output_len]=input_sequence[4][top_pos->p-1];
            star_sequence[output_len]=' ';
        }
        else if(ret_x!=top_pos->x&&ret_y!=top_pos->y&&ret_z!=top_pos->z&&ret_w!=top_pos->w&&ret_p!=top_pos->p) //Move cross
        {
            output_sequence[0][output_len]=input_sequence[0][top_pos->x-1];
            output_sequence[1][output_len]=input_sequence[1][top_pos->y-1];
            output_sequence[2][output_len]=input_sequence[2][top_pos->z-1];
            output_sequence[3][output_len]=input_sequence[3][top_pos->w-1];
            output_sequence[4][output_len]=input_sequence[4][top_pos->p-1];
            if(output_sequence[0][output_len]==output_sequence[1][output_len]&&output_sequence[1][output_len]==output_sequence[2][output_len]&&output_sequence[2][output_len]==output_sequence[3][output_len]&&output_sequence[3][output_len]==output_sequence[4][output_len])                
            {
                star_sequence[output_len]='*';
                start_num++;
                }
            else
                star_sequence[output_len]=' ';
        }
        ret_x=top_pos->x;
        ret_y=top_pos->y;
        ret_z=top_pos->z;
        ret_w=top_pos->w;
        ret_p=top_pos->p;
        output_len++;
        free(top_pos);
    }
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<output_len;j++)
            fputc(output_sequence[i][j],fd_output);
        fputc('\n',fd_output);
    }
    for(int j=0;j<output_len;j++)
        fputc(star_sequence[j],fd_output);
}
