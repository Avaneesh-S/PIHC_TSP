#include"stdio.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include"math.h"
#include <ctype.h>
#include <assert.h>

/*code to perform 2opt on every initial solution, that is construct initial solution with every city as start city and run 2 opt on 
all in parallel*/

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
__global__ void nn_init(int *route,long cities,float *posx,float*posy,int *visited,long *dst,unsigned long long *dst_tid)
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
		dst_tid[id]=(((long)dst[id]+1) << 32) -1;
		// routeChecker(cities, route);
	}

	
}


/* Arrange coordinate in initial solution's order*/
__global__ void setCoord(int *r,float *posx,float *posy,float *px,float *py,long cities)
{
	int id= threadIdx.x+blockIdx.x*blockDim.x;
	if(id<cities)
	{
		int i;
		for(i=id*cities;i<id*cities+cities;i++)
		{
			px[i]=posx[r[i]];
			py[i]=posy[r[i]];
		}
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
__global__ void tsp_tpr(float *pox,float *poy,long *initcost,unsigned long long *dst_tid,long cit)
{

	long id,j;
	long i=threadIdx.x+blockIdx.x*blockDim.x;
	register long change,mincost=initcost[i%cit],cost;
	if(i < cit*(cit-1))
	{
		long limit = ((long)(i/cit)*cit)+cit;
		for(j=i+1;j<limit;j++)
		{
			change = 0; cost=initcost[i%cit];
			change=distD(i,j,pox,poy)+distD((i+1)%(cit*cit),(j+1)%(cit*cit),pox,poy)-distD(i,(i+1)%(cit*cit),pox,poy)-distD(j,(j+1)%(cit*cit),pox,poy);
			cost+=change;	
			if(cost < mincost)
			{
				mincost = cost;
				id = i%cit * (cit-1)+(j%cit-1)-i%cit*(i%cit+1)/2;	
			}	 

		}
		if(mincost < initcost[i%cit])
			 atomicMin(dst_tid+(i%cit), ((unsigned long long)mincost << 32) | id);

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
	long sol,cities,no_pairs;
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
	px = (float *)malloc(sizeof(float) * (cities*cities));  if (px == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
	py = (float *)malloc(sizeof(float) * (cities*cities));  if (py == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
	
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


	if(cities<1)
	{
		printf("too less cities");
		return 0;
	}

	long *dst;
	unsigned long long *d_dst_tid;
	int *visited;
	long *dst_host;
	int *r_device;

	if(cudaSuccess!=cudaMalloc((void**)&dst,sizeof(long)*cities))
	printf("\nCan't allocate memory for dst in device");

	if(cudaSuccess!=cudaMalloc((void**)&d_dst_tid,sizeof(unsigned long long)*cities))
	printf("\nCan't allocate memory for dst_id in device");

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

	float *d_px, *d_py;
	if(cudaSuccess!=cudaMalloc((void**)&d_px,sizeof(float)*(cities*cities)))
	printf("\nCan't allocate memory for coordinate x on GPU");
	if(cudaSuccess!=cudaMalloc((void**)&d_py,sizeof(float)*(cities*cities)))
	printf("\nCan't allocate memory for coordinate y on GPU");


	start = clock();
	
	/*Calling NN algo for initial solution creation*/
	nn_init<<<(cities-1/1024)+1,minn(cities,1024)>>>(r_device,cities,d_posx,d_posy,visited,dst,d_dst_tid);
	// cudaDeviceSynchronize();

	if(cudaSuccess!=cudaMemcpy(dst_host,dst,sizeof(long)*cities,cudaMemcpyDeviceToHost))
	printf("\nCan't transfer dst values back to CPU");

	end = clock();

	tm = ((double) (end - start)) / CLOCKS_PER_SEC;

	long least_dst=LONG_MAX;
	// int best_start_city;

	for(int itr=0;itr<cities;itr++)
	{
		// printf("\nindex : %d , value at index : %ld",i,dst_host[i]);
		if(dst_host[itr]<least_dst)
		{
			least_dst=dst_host[itr];
			// best_start_city=i;
		}
	}

	printf("\nNN running complete");
	free(posx);
	free(posy);

	free(dst_host);

	// int *req_r=r_device+best_start_city*cities; //move only the route which corresponds to minimum initial dst

	// if(cudaSuccess!=cudaMemcpy(r,req_r,sizeof(int)*cities,cudaMemcpyDeviceToHost))
	// printf("\nCan't transfer best route values back to CPU");

    setCoord<<<(cities-1/1024)+1,minn(cities,1024)>>>(r_device,d_posx,d_posy,d_px,d_py,cities);

	if(cudaSuccess!=cudaMemcpy(px,d_px,sizeof(float)*(cities*cities),cudaMemcpyDeviceToHost))
	printf("\nCan't transfer px values back to CPU");

	if(cudaSuccess!=cudaMemcpy(py,d_py,sizeof(float)*(cities*cities),cudaMemcpyDeviceToHost))
	printf("\nCan't transfer py values back to CPU");

	printf("\ninitial solution part done");

	int blk,thrd;
	// unsigned long long *d_dst_tid;
	// long dst2=best_initial_dst;

	start1 = clock();
	count = 1;
	// unsigned long long dst_tid = (((long)dst2+1) << 32) -1;
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

	unsigned long long *dtid=(unsigned long long*)malloc(sizeof(unsigned long long)*(cities));
	long *tid=(long*)malloc(sizeof(long)*(cities));	
	long *d=(long*)malloc(sizeof(long)*(cities));
	long min_d=LONG_MAX;


	blk=((cities*(cities-1)-1)/1024+1);
	thrd=1024;
	
	
	
	
	tsp_tpr<<<blk,thrd>>>(d_px,d_py,dst,d_dst_tid,cities);
	
	if(cudaSuccess!=cudaMemcpy(dtid,d_dst_tid,sizeof(unsigned long long)*cities,cudaMemcpyDeviceToHost))
	printf("\nCan't transfer minimal dtid to CPU");

	printf("\ntpr finished running");

	for(int itr=0;itr<cities;itr++)
	{
		d[itr] = dtid[itr] >> 32;
		if(d[itr]<min_d)
		{
			min_d=d[itr];
		}
	}

	printf("\n first tpr call complete moved min d");
	long *x=(long*)malloc(sizeof(long)*(cities));
	long *y=(long*)malloc(sizeof(long)*(cities));
	
	
	while( min_d < least_dst )
	{
		least_dst=min_d;
		for(int itr=0;itr<cities;itr++)
		{
			tid[itr] = dtid[itr] & ((1ull<<32)-1);
			x[itr]=cities-2-floor((sqrt(8*(sol-tid[itr]-1)+1)-1)/2);
			y[itr]=tid[itr]-x[itr]*(cities-1)+(x[itr]*(x[itr]+1)/2)+1;
			twoOpt(x[itr],y[itr],px+(cities*itr),py+(cities*itr));
			if(cudaSuccess!=cudaMemcpy(d_px+(itr*cities),px+(cities*itr),sizeof(float)*cities,cudaMemcpyHostToDevice))
			printf("\nCan't transfer px on GPU");
			if(cudaSuccess!=cudaMemcpy(d_py+(itr*cities),py+(cities*itr),sizeof(float)*cities,cudaMemcpyHostToDevice))
			printf("\nCan't transfer py on GPU");
			// unsigned long long dst_tid = (((long)least_dst+1) << 32) -1;
			// if(cudaSuccess!=cudaMemcpy(d_dst_tid[itr],&dst_tid,sizeof(unsigned long long),cudaMemcpyHostToDevice))
			// printf("\nCan't transfer dst_tid on GPU");
		} 

		tsp_tpr<<<blk,thrd>>>(d_px,d_py,dst,d_dst_tid,cities);

		if(cudaSuccess!=cudaMemcpy(dtid,d_dst_tid,sizeof(unsigned long long)*cities,cudaMemcpyDeviceToHost))
		printf("\nCan't transfer minimal dtid to CPU inside loop");


		for(int itr=0;itr<cities;itr++)
		{
			d[itr] = dtid[itr] >> 32;
			if(d[itr]<min_d)
			{
				min_d=d[itr];
			}
		}
		count++;
	}


	printf("\n-------------------------------------------------------------------");
	// printf("\nleast initial cost is %d",best_initial_dst);
	printf("\nInitial solution time taken is %f",tm);
	// printf("\ninitial start city is %d",best_start_city);
	printf("\nMinimal distance found %ld\n",min_d);
	printf("\nnumber of times hill climbed in minimal distance solution %d\n",count);
	end1 = clock();
	printf("\ntime : %f\n",((double) (end1 - start1)) / CLOCKS_PER_SEC);


	free(x);
	free(y);
	free(tid);
	free(d);

	cudaFree(d_posx);
	cudaFree(d_posy);
	cudaFree(dst);
	cudaFree(visited);
	cudaFree(r_device);
	cudaFree(d_dst_tid);
	
	return 0;
}

