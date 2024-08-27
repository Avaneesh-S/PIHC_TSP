#include"stdio.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include"math.h"
#include <ctype.h>
#include <assert.h>

/* Euclidean distance calculation */
long distD(int i,int j,float *x,float*y)
{
	float dx=x[i]-x[j];
	float dy=y[i]-y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

/* Initial solution construction using NN */
long nn_init(int *route,long cities,float *posx,float*posy)
{
	route[0]=0;
	int k=1,i=0,j;
	float min;
	int minj,mini,count=1,flag=0;
	long dst=0;
	int *visited=(int*)calloc(cities,sizeof(int));
	visited[0]=1;
	while(count!=cities)
	{
		flag=0;
		for(j=1;j<cities;j++)
		{
			if(i!=j && !visited[j])
			{
				min=distD(i,j,posx,posy);
				minj=j;
				break;	
			}
		}

		for(j=minj+1;j<cities;j++)
		{
			
			 if( !visited[j])
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
		dst+=min;
		route[k++]=i;
		visited[i]=1;
		count++;
	}
	free(visited);
	dst+=distD(route[0],route[cities-1],posx,posy);
	return dst;
}

void routeChecker(long N,int *r)
{
	int *v,i,flag=0;
	v=(int*)calloc(N,sizeof(int));	

	for(i=0;i<N;i++)
		v[r[i]]++;
	for(i=0;i<N;i++)
	{
		if(v[i] != 1 )
		{
			flag=1;
			printf("breaking at %d",i);
			break;
		}
	}
	if(flag==1)
		printf("\nroute is not valid");
	else
		printf("\nroute is valid");
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

int main(int argc, char *argv[])
{
	int ch, cnt, in1;
	float in2, in3;
	FILE *f;
	float *posx, *posy;
	float *px, *py,tm;
	char str[256];  
	int *r;
	long dst,sol,d,cities,no_pairs,tid=0;
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

	start = clock();
    dst = nn_init(r,cities,posx,posy);
    routeChecker(cities, r);
    setCoord(r,posx,posy,px,py,cities);

	end = clock();
	tm = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\ninitial cost : %ld time : %f\n",dst,tm);

	start1 = clock();
	float cost=0,dist=dst;
	float x=0,y=0;
	register int change=0;
	count=0;	
	/*Iterative hill approch */

	do{
		cost=0;
		dist=dst;
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
				if(cost < dst)
				{
					x = i;
					y = j;
					dst = cost;
				}
			}

		}
		if(dst<dist)
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
	}while(dst<dist);

printf("\nMinimal distance found %ld\n",dst);
printf("\nnumber of time hill climbed %d\n",count);
end1 = clock();
printf("\ntime : %f\n",((double) (end1 - start1)) / CLOCKS_PER_SEC);

free(posx);
free(posy);
return 0;
}

