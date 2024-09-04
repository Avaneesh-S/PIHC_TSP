#include"stdio.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include"math.h"
#include <ctype.h>
#include <assert.h>

/*this code shows that starting from the least cost initial solution through NN and performing 2 opt only on the best initial solution 
may or may not reach a better solution than IHC.c (varies for instances - gives improvement for 300 and 1000 instances, but not for
instance size 100)*/

/* Euclidean distance calculation */
__device__ long GPU_distD(int i,int j,float *x,float*y)
{
	float dx=x[i]-x[j];
	float dy=y[i]-y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

long distD(int i,int j,float *x,float*y)
{
	float dx=x[i]-x[j];
	float dy=y[i]-y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

// __device__ void routeChecker(long N,int *r)
// {
// 	int *v,i,flag=0;
// 	v=(int*)calloc(N,sizeof(int));	

// 	for(i=0;i<N;i++)
// 		v[r[i]]++;
// 	for(i=0;i<N;i++)
// 	{
// 		if(v[i] != 1 )
// 		{
// 			flag=1;
// 			printf("breaking at %d",i);
// 			break;
// 		}
// 	}
// 	if(flag==1)
// 		printf("\nroute is not valid");
// 	// else
// 	// 	printf("\nroute is valid");
// }

/* Initial solution construction using NN */
__global__ void nn_init(int *route,long cities,float *posx,float*posy,int *visited,long *dst)
{
	int id = threadIdx.x+blockIdx.x*blockDim.x;
	if(id<cities)
	{
		dst[id]=0;
		int start_index=id;
		route[start_index*cities+0]=start_index;
		int k=1,i=start_index,j;
		float min;
		int minj,mini,count=1,flag=0;
		// long dst=0;
		// int *visited=(int*)calloc(cities,sizeof(int));
		visited[start_index*cities+start_index]=1;
		while(count!=cities)
		{
			flag=0;
			for(j=0;j<cities;j++)
			{
				if(i!=j && !visited[start_index*cities+j])
				{
					min=GPU_distD(i,j,posx,posy);
					minj=j;
					break;	
				}
			}

			for(j=minj+1;j<cities;j++)
			{
				
				if( !visited[start_index*cities+j])
				{
					if(min>GPU_distD(i,j,posx,posy))
					{
						min=GPU_distD(i,j,posx,posy);
						mini=j;
						flag=1;				
					}
				}
			}
			if(flag==0)
				i=minj;
			else
				i=mini;
			dst[id]+=min;
			route[start_index*cities+k++]=i;
			visited[start_index*cities+i]=1;
			count++;
		}
		// free(visited);
		dst[id]+=GPU_distD(route[start_index*cities+0],route[start_index*cities+cities-1],posx,posy);
		// routeChecker(cities, route);
	}

	
}


/* Arrange coordinate in initial solution's order*/
void setCoord(int *r,float *posx,float *posy,float *px,float *py,long cities)
{
	int i;
	for(i=0;i<cities;i++)
	{
		px[i]=posx[r[i]];
		py[i]=posy[r[i]];
	}
}

long distH(float *px,float *py,long cit)
{
	float dx,dy;
	long cost=0;
	int i;
	for(i=0;i<(cit-1);i++)
	{
		dx=px[i]-px[i+1];
		dy=py[i]-py[i+1]; 
		cost+=sqrtf( (dx*dx) + (dy*dy) );
	}
	dx=px[i]-px[0];
	dy=py[i]-py[0]; 
	cost+=sqrtf( (dx*dx) + (dy*dy) );
	return cost;

}

int minn(int a,int b)
{
	if(a<b)
	{
		return a;
	}
	return b;
}

int main(int argc, char *argv[])
{
	int ch, cnt, in1;
	float in2, in3;
	FILE *f;
	float *posx, *posy;
	float *px, *py,tm;
	char str[256];  
	int *r;
	long sol,d,cities,no_pairs,tid=0;
	int i,j,intl,count;
	
	clock_t start,end,start1,end1;

	f = fopen(argv[1], "r");
	if (f == NULL) {fprintf(stderr, "could not open file \n");  exit(-1);}

	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

	ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
	fscanf(f, "%s\n", str);
	cities = atoi(str);
	if (cities <= 2) {fprintf(stderr, "only %ld cities\n", cities);  exit(-1);}

	sol=cities*(cities-1)/2;
	posx = (float *)malloc(sizeof(float) * cities);  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
	posy = (float *)malloc(sizeof(float) * cities);  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
	px = (float *)malloc(sizeof(float) * cities);  if (px == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
	py = (float *)malloc(sizeof(float) * cities);  if (py == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
	
	r = (int *)malloc(sizeof(int) * cities);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	fscanf(f, "%s\n", str);
	if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

	cnt = 0;

	while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) 
	{
		posx[cnt] = in2;
		posy[cnt] = in3;
		cnt++;
		if (cnt > cities) {fprintf(stderr, "input too long\n");  exit(-1);}
		if (cnt != in1) {fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);  exit(-1);}
	}

	if (cnt != cities) {fprintf(stderr, "read %d instead of %ld cities\n", cnt, cities);  exit(-1);}
	fscanf(f, "%s", str);
	if (strcmp(str, "EOF") != 0) {fprintf(stderr, "didn't see 'EOF' at end of file\n");  exit(-1);}

	// int dst_final=INT_MAX;
	// int count_final;
	// int best_initial_dst=INT_MAX;
	// double best_initial_time;
	// int best_start_city;
    // int*best_initial_route = (int *)malloc(sizeof(int) * cities);

	// if(cities>1000)
	// {
	// 	printf("too many cities, code does not support yet");
	// 	return 0;
	// }

	if(cities<1)
	{
		printf("too less cities");
		return 0;
	}

	long *dst;
	int *visited;
	long *dst_host;
	int *r_device;

	if(cudaSuccess!=cudaMalloc((void**)&dst,sizeof(long)*cities))
	printf("\nCan't allocate memory for dst in device");

	dst_host=(long*)malloc(sizeof(long)*(cities));	

	if(cudaSuccess!=cudaMalloc((void**)&visited,sizeof(int)*(cities*cities)))
	printf("\nCan't allocate memory for visited in device");

	if(cudaSuccess!=cudaMalloc((void**)&r_device,sizeof(int)*(cities*cities)))
	printf("\nCan't allocate memory for r i.e route in device");

	float *d_posx, *d_posy;

	if(cudaSuccess!=cudaMalloc((void**)&d_posx,sizeof(float)*cities))
	printf("\nCan't allocate memory for coordinate x on GPU");
	if(cudaSuccess!=cudaMalloc((void**)&d_posy,sizeof(float)*cities))
	printf("\nCan't allocate memory for coordinate y on GPU");

	if(cudaSuccess!=cudaMemcpy(d_posx,posx,sizeof(float)*cities,cudaMemcpyHostToDevice))
	printf("\nCan't transfer px on GPU");
	if(cudaSuccess!=cudaMemcpy(d_posy,posy,sizeof(float)*cities,cudaMemcpyHostToDevice))
	printf("\nCan't transfer py on GPU");

	start = clock();
	
	/*Calling NN algo for initial solution creation*/
	nn_init<<<(cities-1/1024)+1,minn(cities,1024)>>>(r_device,cities,d_posx,d_posy,visited,dst);

	end = clock();

	if(cudaSuccess!=cudaMemcpy(dst_host,dst,sizeof(long)*cities,cudaMemcpyDeviceToHost))
	printf("\nCan't transfer dst values back to CPU");

	tm = ((double) (end - start)) / CLOCKS_PER_SEC;

	long best_initial_dst=INT_MAX;
	int best_start_city;

	for(int i=0;i<cities;i++)
	{
		// printf("\nindex : %d , value at index : %ld",i,dst_host[i]);
		if(dst_host[i]<best_initial_dst)
		{
			best_initial_dst=dst_host[i];
			best_start_city=i;
		}
	}

	int *req_r=r_device+best_start_city*cities; //move only the route which corresponds to minimum initial dst

	if(cudaSuccess!=cudaMemcpy(r,req_r,sizeof(int)*cities,cudaMemcpyDeviceToHost))
	printf("\nCan't transfer best route values back to CPU");

	// for(int i=0;i<cities;i++)
	// {
	// 	printf("\n%d",r[i]);
	// }

    setCoord(r,posx,posy,px,py,cities);
    
    /*Iterative hill approch */
    start1 = clock();
	long dist=best_initial_dst;
	long dst2=best_initial_dst;
    float cost=0;
    float x=0,y=0;
    register int change=0;
    count=0;

    do{
        cost=0;
        dist=dst2;
        for(i=0;i<(cities-1);i++)
        {	
    
            for(j = i+1; j < cities; j++)
            {
                cost = dist;			
                change = distD(i,j,px,py) 
                + distD(i+1,(j+1)%cities,px,py) 
                - distD(i,(i+1)%cities,px,py)
                - distD(j,(j+1)%cities,px,py);
                cost += change;	
                if(cost < dst2)
                {
                    x = i;
                    y = j;
                    dst2 = cost;
                }
            }

        }
        if(dst2<dist)
        {
            float *tmp_x,*tmp_y;
            tmp_x=(float*)malloc(sizeof(float)*(y-x));	
            tmp_y=(float*)malloc(sizeof(float)*(y-x));	
            for(j=0,i=y;i>x;i--,j++)
            {
                tmp_x[j]=px[i];
                tmp_y[j]=py[i];
            }
            for(j=0,i=x+1;i<=y;i++,j++)
            {
                px[i]=tmp_x[j];
                py[i]=tmp_y[j];
            }
            free(tmp_x);
            free(tmp_y);
        }
        count++;
    }while(dst2<dist);


	printf("\n-------------------------------------------------------------------");
	printf("\nleast initial cost is %d",best_initial_dst);
	printf("\ntime taken is %f",tm);
	printf("\ninitial start city is %d",best_start_city);
	printf("\nMinimal distance found %ld\n",dst2);
	printf("\nnumber of times hill climbed in minimal distance solution %d\n",count);
	end1 = clock();
	printf("\ntime : %f\n",((double) (end1 - start1)) / CLOCKS_PER_SEC);

	free(posx);
	free(posy);

	free(dst_host);

	cudaFree(d_posx);
	cudaFree(d_posy);
	cudaFree(dst);
	cudaFree(visited);
	cudaFree(r_device);
	return 0;
}

