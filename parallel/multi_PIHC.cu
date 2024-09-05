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
__host__ __device__ long distD(int i,int j,float *x,float*y)
{
	float dx=x[i]-x[j];
	float dy=y[i]-y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

// long distD(int i,int j,float *x,float*y)
// {
// 	float dx=x[i]-x[j];
// 	float dy=y[i]-y[j]; 
// 	return(sqrtf( (dx*dx) + (dy*dy) ));
// }

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
					min=distD(i,j,posx,posy);
					minj=j;
					break;	
				}
			}

			for(j=minj+1;j<cities;j++)
			{
				
				if( !visited[start_index*cities+j])
				{
					if(min>distD(i,j,posx,posy))
					{
						min=distD(i,j,posx,posy);
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
		dst[id]+=distD(route[start_index*cities+0],route[start_index*cities+cities-1],posx,posy);
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

/*A kenel function that finds a minimal weighted neighbor using TPR mapping strategy*/
__global__ void tsp_tpr(float *pox,float *poy,long initcost,unsigned long long *dst_tid,long cit)
{

	long id,j;
	register long change,mincost=initcost,cost;
	long i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i < cit)
	{
		
		for(j=i+1;j<cit;j++)
		{
			change = 0; cost=initcost;
			change=distD(i,j,pox,poy)+distD((i+1)%cit,(j+1)%cit,pox,poy)-distD(i,(i+1)%cit,pox,poy)-distD(j,(j+1)%cit,pox,poy);
			cost+=change;	
			if(cost < mincost)
			{
				mincost = cost;
				id = i * (cit-1)+(j-1)-i*(i+1)/2;	
			}	 

		}
		if(mincost < initcost)
			 atomicMin(dst_tid, ((unsigned long long)mincost << 32) | id);

	}
	
}

/*A kenel function that finds a minimal weighted neighbor using TPRED mapping strategy*/
__global__ void tsp_tpred(float *pox,float *poy,long initcost,unsigned long long *dst_tid,long cit,long itr)
{
	long id,j,k;
	register long change,mincost=initcost,cost;
	long i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i < cit)
	{
		
		for(k=0;k<itr;k++)
		{
			change = 0; cost=initcost;
			j=(i+1+k)%cit;
			change=distD(i,j,pox,poy)+distD((i+1)%cit,(j+1)%cit,pox,poy)-distD(i,(i+1)%cit,pox,poy)-distD(j,(j+1)%cit,pox,poy);
			cost+=change;	
			if(cost < mincost)
			{
				mincost = cost;
				if(i < j)
					id = i * (cit-1)+(j-1)-i*(i+1)/2;	
				else
					id = j * (cit-1)+(i-1)-j*(j+1)/2;	

			}	 

		}
		if(mincost < initcost)
			 atomicMin(dst_tid, ((unsigned long long)mincost << 32) | id);
	}
}

/*A kenel function that finds a minimal weighted neighbor using TPRC mapping strategy*/
__global__ void tsp_tprc(float *pox,float *poy,long initcost,unsigned long long *dst_tid,long cit)
{

	long id;
	long change,cost;
	long i=threadIdx.x+blockIdx.x*blockDim.x;
	long j=threadIdx.y+blockIdx.y*blockDim.y;
	if(i < cit && j < cit && i < j)
	{
		
			change = 0; cost = initcost;
			change=distD(i,j,pox,poy)+distD((i+1)%cit,(j+1)%cit,pox,poy)-distD(i,(i+1)%cit,pox,poy)-distD(j,(j+1)%cit,pox,poy);
			cost+=change;	
			if(change < 0)
			{
				id = i * (cit - 1) + (j - 1) - i * (i + 1) / 2;	
				atomicMin(dst_tid, ((unsigned long long)cost << 32) | id);
			}	 

	}
	
}

/*A kenel function that finds a minimal weighted neighbor using TPN mapping strategy*/
__global__ void tsp_tpn(float *pox,float *poy,long cost,unsigned long long *dst_tid,long cit,long sol)
{

	long i,j;
	register long change=0;
	int id=threadIdx.x+blockIdx.x*blockDim.x;
	if(id<sol)
	{
		
		i=cit-2-floorf(((int)__dsqrt_rn(8*(sol-id-1)+1)-1)/2);
		j=id-i*(cit-1)+(i*(i+1)/2)+1;
		change=distD(i,j,pox,poy)+distD((i+1)%cit,(j+1)%cit,pox,poy)-distD(i,(i+1)%cit,pox,poy)-distD(j,(j+1)%cit,pox,poy);
		cost+=change;	
		if(change < 0)
			 atomicMin(dst_tid, ((unsigned long long)cost << 32) | id);
		
	}
	
}

/* At each IHC steps, XY coordinates are arranged using next initial solution's order*/
void twoOpt(long x,long y,float *pox,float *poy)
{
	float *tmp_x,*tmp_y;
	int i,j;
	tmp_x=(float*)malloc(sizeof(float)*(y-x));	
	tmp_y=(float*)malloc(sizeof(float)*(y-x));
	for(j=0,i=y;i>x;i--,j++)
	{
		tmp_x[j]=pox[i];
		tmp_y[j]=poy[i];
	}
	for(j=0,i=x+1;i<=y;i++,j++)
	{
		pox[i]=tmp_x[j];
		poy[i]=tmp_y[j];
	}
	free(tmp_x);
	free(tmp_y);

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

	if(cudaSuccess!=cudaMemcpy(dst_host,dst,sizeof(long)*cities,cudaMemcpyDeviceToHost))
	printf("\nCan't transfer dst values back to CPU");

	end = clock();

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

	int blk,thrd;
	unsigned long long *d_dst_tid;
	long dst2=best_initial_dst;
	long x,y;

	start1 = clock();
	count = 1;
	unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
        unsigned long long dtid;
	long itr=floor(cities/2);
	int nx, ny;
	if(cities <= 32)
	{
		blk = 1 ;
		nx = cities;
		ny = cities;
	}
	else
	{
		blk = (cities - 1) / 32 + 1;
		nx = 32;
		ny = 32;
	}
	dim3 thrds (nx,ny);
	dim3 blks (blk,blk);
	if(cudaSuccess!=cudaMalloc((void**)&d_posx,sizeof(float)*cities))
	printf("\nCan't allocate memory for coordinate x on GPU");
	if(cudaSuccess!=cudaMalloc((void**)&d_posy,sizeof(float)*cities))
	printf("\nCan't allocate memory for coordinate y on GPU");
	if(cudaSuccess!=cudaMalloc((void**)&d_dst_tid,sizeof(unsigned long long)))
	printf("\nCan't allocate memory for dst_tid on GPU");
    	if(cudaSuccess!=cudaMemcpy(d_dst_tid,&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
	printf("\nCan't transfer dst_tid on GPU");
	if(cudaSuccess!=cudaMemcpy(d_posx,px,sizeof(float)*cities,cudaMemcpyHostToDevice))
	printf("\nCan't transfer px on GPU");
	if(cudaSuccess!=cudaMemcpy(d_posy,py,sizeof(float)*cities,cudaMemcpyHostToDevice))
	printf("\nCan't transfer py on GPU");

	int strat;	
	printf("\n Choose a CUDA thread mapping strategy\n1.TPR\n2.TPRED\n3.TPRC\n4.TPN\n");
	scanf("%d",&strat);
	switch(strat)
	{
		case 1:

			if(cities<=1024)
			{
				blk=1;
				thrd=cities;
			}
			else
			{
				blk=(cities-1)/1024+1;
				thrd=1024;
			}
			
			tsp_tpr<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities);
			
			if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
			printf("\nCan't transfer minimal cost back to CPU");

			d = dtid >> 32;
			
			while( d < dst2 )
			{
				dst2=d;
				tid = dtid & ((1ull<<32)-1); 
				x=cities-2-floor((sqrt(8*(sol-tid-1)+1)-1)/2);
				y=tid-x*(cities-1)+(x*(x+1)/2)+1;
				twoOpt(x,y,px,py);
				if(cudaSuccess!=cudaMemcpy(d_posx,px,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer px on GPU");
				if(cudaSuccess!=cudaMemcpy(d_posy,py,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer py on GPU");
				unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
				if(cudaSuccess!=cudaMemcpy(d_dst_tid,&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
				printf("\nCan't transfer dst_tid on GPU");

				tsp_tpr<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities);
				if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
				printf("\nCan't transfer minimal cost back to CPU");
			  	d = dtid >> 32;
				count++;
			}
		break;
		case 2:
			
			if(cities<1024)
			{
				blk=1;
				thrd=cities;
			}
			else
			{
				blk=(cities-1)/1024+1;
				thrd=1024;
			}	

			tsp_tpred<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities,itr);
			
			if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
			printf("\nCan't transfer minimal cost back to CPU");

			d = dtid >> 32;
			
			while( d < dst2 )
			{

				dst2=d;
				tid = dtid & ((1ull<<32)-1); 
				x=cities-2-floor((sqrt(8*(sol-tid-1)+1)-1)/2);
				y=tid-x*(cities-1)+(x*(x+1)/2)+1;
				twoOpt(x,y,px,py);
				if(cudaSuccess!=cudaMemcpy(d_posx,px,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer px on GPU");
				if(cudaSuccess!=cudaMemcpy(d_posy,py,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer py on GPU");
				unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
				if(cudaSuccess!=cudaMemcpy(d_dst_tid,&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
				printf("\nCan't transfer dst_tid on GPU");

				tsp_tpred<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities,itr);
				
				if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
				printf("\nCan't transfer minimal cost back to CPU");
			  	d = dtid >> 32;
				count++;
			}
		break;
		case 3:
			
			tsp_tprc<<<blks,thrds>>>(d_posx,d_posy,dst2,d_dst_tid,cities);
	
			if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
			printf("\nCan't transfer minimal cost back to CPU");
		  	d = dtid >> 32;
			
			while( d < dst2 )
			{
				dst2=d;
				tid = dtid & ((1ull<<32)-1); 
				x=cities-2-floor((sqrt(8*(sol-tid-1)+1)-1)/2);
				y=tid-x*(cities-1)+(x*(x+1)/2)+1;
				twoOpt(x,y,px,py);
				if(cudaSuccess!=cudaMemcpy(d_posx,px,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer px on GPU");
				if(cudaSuccess!=cudaMemcpy(d_posy,py,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer py on GPU");
				unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
				if(cudaSuccess!=cudaMemcpy(d_dst_tid,&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
				printf("\nCan't transfer dst_tid on GPU");

				tsp_tprc<<<blks,thrds>>>(d_posx,d_posy,dst2,d_dst_tid,cities);
				if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
				printf("\nCan't transfer minimal cost back to CPU");
			  	d = dtid >> 32;
				count++;
			}
		break;
		case 4:
			if(sol < 1024)
			{
				blk=1;
				thrd=sol;
			}
			else
			{
				blk=(sol-1)/1024+1;
				thrd=1024;
			}

			tsp_tpn<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities,sol);

			if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
			printf("\nCan't transfer minimal cost back to CPU");
			d = dtid >> 32;
			
			while( d < dst2 )
			{
				dst2=d;
				tid = dtid & ((1ull<<32)-1); 
				x=cities-2-floor((sqrt(8*(sol-tid-1)+1)-1)/2);
				y=tid-x*(cities-1)+(x*(x+1)/2)+1;
				twoOpt(x,y,px,py);
				if(cudaSuccess!=cudaMemcpy(d_posx,px,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer px on GPU");
				if(cudaSuccess!=cudaMemcpy(d_posy,py,sizeof(float)*cities,cudaMemcpyHostToDevice))
				printf("\nCan't transfer py on GPU");
				unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
				if(cudaSuccess!=cudaMemcpy(d_dst_tid,&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
				printf("\nCan't transfer dst_tid on GPU");

				tsp_tpn<<<blk,thrd>>>(d_posx,d_posy,dst2,d_dst_tid,cities,sol);

				if(cudaSuccess!=cudaMemcpy(&dtid,d_dst_tid,sizeof(unsigned long long),cudaMemcpyDeviceToHost))
				printf("\nCan't transfer minimal cost back to CPU");
			  	d = dtid >> 32;
				count++;
			}
		break;
	}
    
    /*Iterative hill approch */
    // start1 = clock();
	// long dist=best_initial_dst;
	// long dst2=best_initial_dst;
    // float cost=0;
    // float x=0,y=0;
    // register int change=0;
    // count=0;

    // do{
    //     cost=0;
    //     dist=dst2;
    //     for(i=0;i<(cities-1);i++)
    //     {	
    
    //         for(j = i+1; j < cities; j++)
    //         {
    //             cost = dist;			
    //             change = distD(i,j,px,py) 
    //             + distD(i+1,(j+1)%cities,px,py) 
    //             - distD(i,(i+1)%cities,px,py)
    //             - distD(j,(j+1)%cities,px,py);
    //             cost += change;	
    //             if(cost < dst2)
    //             {
    //                 x = i;
    //                 y = j;
    //                 dst2 = cost;
    //             }
    //         }

    //     }
    //     if(dst2<dist)
    //     {
    //         float *tmp_x,*tmp_y;
    //         tmp_x=(float*)malloc(sizeof(float)*(y-x));	
    //         tmp_y=(float*)malloc(sizeof(float)*(y-x));	
    //         for(j=0,i=y;i>x;i--,j++)
    //         {
    //             tmp_x[j]=px[i];
    //             tmp_y[j]=py[i];
    //         }
    //         for(j=0,i=x+1;i<=y;i++,j++)
    //         {
    //             px[i]=tmp_x[j];
    //             py[i]=tmp_y[j];
    //         }
    //         free(tmp_x);
    //         free(tmp_y);
    //     }
    //     count++;
    // }while(dst2<dist);


	printf("\n-------------------------------------------------------------------");
	printf("\nleast initial cost is %d",best_initial_dst);
	printf("\nInitial solution time taken is %f",tm);
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

