#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"
#include <cuda.h>
#include <cuda_profiler_api.h>
using namespace std;

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

#define MASK_SIZE 3
#define TILE_SIZE_O 4
const int BLOCK_WIDTH = TILE_SIZE_O+MASK_SIZE-1; //TILE_SIZE

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rs, columns, n, i, j, value) ( (M)[( ((n)*(rs)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rs, columns, n, i, j) ( M[( ((n)*(rs)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrs, int &ncs, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncs = atoi(str);      //ncs
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrs = atoi(str);      //nrs
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rs, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rs; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rs, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rs; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInit_Kernel(int r, int c, double* Sz, double* Sh, int i_s, int i_e, int j_s, int j_e)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; //righe
  int j = blockIdx.y * blockDim.y + threadIdx.y; //colonne

  if(i < i_s || i >= i_e)
    return;
  if(j < j_s || j >= j_e)
    return;
 
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0)
  {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }
  

}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

__global__ void sciddicaTResetFlows_Kernel(int r, int c, double nodata, double* Sf, int i_s, int i_e, int j_s, int j_e)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i < i_s || i >= i_e)
    return;
  if(j < j_s || j >= j_e)
    return;

    

  BUF_SET(Sf, r, c, 0, i, j, 0.0);
  BUF_SET(Sf, r, c, 1, i, j, 0.0);
  BUF_SET(Sf, r, c, 2, i, j, 0.0);
  BUF_SET(Sf, r, c, 3, i, j, 0.0);
}

__global__ void sciddicaTFlowsComputation_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon, int i_s, int i_e, int j_s, int j_e)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.x*TILE_SIZE_O + tx; //+ i_s;
  int row_i = row_o - i_s;
  int col_o = blockIdx.y*TILE_SIZE_O + ty;// + j_s;
  int col_i = col_o - j_s;

  __shared__ double Sz_shared[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ double Sh_shared[BLOCK_WIDTH][BLOCK_WIDTH];

  // Tutti i thread partecipano al caricamento delle tessere di input nella  memoria condivisa. Ogni thread controlla se gli indici y e 
//x dei suoi  elementi della piastrella di input rientrano nell'intervallo valido dell'input. 
// In caso contrario, l'elemento di input che sta tentando di caricare è effettivamente
// una ghost cells e un valore 0.0 viene  inserito  nella memoria condivisa.

  if((row_i >= 0) && (row_i < r)  && (col_i >= 0) && (col_i < c))
  {
    Sz_shared[tx][ty] = GET(Sz, c, row_i, col_i);
    Sh_shared[tx][ty] = GET(Sh, c, row_i, col_i);
  }
  else
  {
    Sz_shared[tx][ty] = 0.0f;
    Sh_shared[tx][ty] = 0.0f;
  }

  __syncthreads();

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  double cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;
  
  if(tx == BLOCK_WIDTH-1 || ty == BLOCK_WIDTH-1)
    return;

  if((blockIdx.x == 0) || (blockIdx.x == blockDim.x-1))
  {
    if(tx < 2 || tx >= i_e-1)
      return;
  }
  else
  {
    if(tx < 1 || tx >= i_e)
      return;
  }

  if((blockIdx.y == 0) || (blockIdx.y == blockDim.y-1))
  {
    if(ty < 2 || ty >= j_e-1)
      return;
  }
  else
  {
    if(ty < 1 || ty >= j_e)
      return;
  }


  m = Sh_shared[tx][ty] - p_epsilon;
  u[0] = Sz_shared[tx][ty] + p_epsilon;
  z = Sz_shared[tx - 1][ty];
  h = Sh_shared[tx - 1][ty];
  u[1] = z + h;                                         
  z = Sz_shared[tx + Xi[2]][ty + Xj[2]];
  h = Sh_shared[tx + Xi[2]][ty + Xj[2]];
  u[2] = z + h;                                         
  z = Sz_shared[tx][ty + 1];
  h = Sh_shared[tx][ty + 1];
  u[3] = z + h;                                         
  z = Sz_shared[tx + 1][ty];
  h = Sh_shared[tx + 1][ty];
  u[4] = z + h;

  do
  {
    again = false;
    average = m;
    cells_count = 0;

    for (n = 0; n < 5; n++)
      if (!eliminated_cells[n])
      {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (n = 0; n < 5; n++)
    {
      if ((average <= u[n]) && (!eliminated_cells[n]))
      {
        eliminated_cells[n] = true;
        again = true;
      }
    }
  } while (again);

  if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, row_i, col_i, (average - u[1]) * p_r);
  if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, row_i, col_i, (average - u[2]) * p_r);
  if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, row_i, col_i, (average - u[3]) * p_r);
  if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, row_i, col_i, (average - u[4]) * p_r);
}

__global__ void sciddicaTWidthUpdate_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, int i_s, int i_e, int j_s, int j_e)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.x*TILE_SIZE_O + tx;
  int row_i = row_o - i_s;
  int col_o = blockIdx.y*TILE_SIZE_O + ty;
  int col_i = col_o - j_s;

  __shared__ double shared[BLOCK_WIDTH][BLOCK_WIDTH][4];
 

  if((row_i >= 0) && (col_i >= 0) && (row_i < r) && (col_i < c))
  {
    shared[tx][ty][0] = BUF_GET(Sf, r, c, 0, row_i, col_i);
    shared[tx][ty][1] = BUF_GET(Sf, r, c, 1, row_i, col_i);
    shared[tx][ty][2] = BUF_GET(Sf, r, c, 2, row_i, col_i);
    shared[tx][ty][3] = BUF_GET(Sf, r, c, 3, row_i, col_i);
  }
  else
  {
    shared[tx][ty][0] = 0.0f;
    shared[tx][ty][1] = 0.0f;
    shared[tx][ty][2] = 0.0f;
    shared[tx][ty][3] = 0.0f;
  }

  __syncthreads();

  if(tx == BLOCK_WIDTH-1 || ty == BLOCK_WIDTH-1)
    return;

  if((blockIdx.x == 0) || (blockIdx.x == blockDim.x-1))
  {
    if(tx < 2 || tx >= i_e-1)
      return;
  }
  else
  {
    if(tx < 1 || tx >= i_e)
      return;
  }

  if((blockIdx.y == 0) || (blockIdx.y == blockDim.y-1))
  {
    if(ty < 2 || ty >= j_e-1)
      return;
  }
  else
  {
    if(ty < 1 || ty >= j_e-1)
      return;
  }

  double h_next;

  if(row_i == r-1 || col_i == c-1)
    return;

  if((row_i >= 0) && (row_i < r)  && (col_i >= 0) && (col_i < c))
  {
    h_next = GET(Sh, c, row_i, col_i);
  }
  else
    return;

  h_next += shared[tx + Xi[1]][ty + Xj[1]][3] - shared[tx][ty][0];
  h_next += shared[tx + Xi[2]][ty + Xj[2]][2] - shared[tx][ty][1];
  h_next += shared[tx + Xi[3]][ty + Xj[3]][1] - shared[tx][ty][2];
  h_next += shared[tx + Xi[4]][ty + Xj[4]][0] - shared[tx][ty][3];
  
  SET(Sh, c, row_i, col_i, h_next );
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rs, cs;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rs, cs, nodata);

  int r = rs;                  // r: grid rs
  int c = cs;                  // c: grid columns
  int i_s = 1, i_e = r-1;  // [i_s,i_e[: kernels application range along the rs
  int j_s = 1, j_e = c-1;  // [i_s,i_e[: kernels application range along the rs
  double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
  int *Xi;// Xj: von Neuman neighborhood row coordinates (see below)
  int *Xj;// Xj: von Neuman neighborhood col coordinates (see below)
  double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
  //   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //


  cudaMallocManaged(&Xi, sizeof(int) * 5);
  cudaMallocManaged(&Xj, sizeof(int) * 5);

  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;

  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;
  

  cudaMallocManaged(&Sz, sizeof(double) * r * c);
  cudaMallocManaged(&Sh, sizeof(double) * r * c);
  cudaMallocManaged(&Sf, sizeof(double) * ADJACENT_CELLS * r * c);
  

 

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file


  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  


  dim3 dimGrid((r-1)/TILE_SIZE_O  , (c-1)/TILE_SIZE_O , 1);
  dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);


  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
  
  sciddicaTSimulationInit_Kernel<<<dimGrid, dimBlock>>>(r, c, Sz, Sh, i_s, i_e, j_s, j_e);
  cudaDeviceSynchronize();
  
  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
    // Apply the resetFlow kernel to the whole domain
    sciddicaTResetFlows_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Sf, i_s, i_e, j_s, j_e);
    cudaDeviceSynchronize();

    // Apply the FlowComputation kernel to the whole domain
    
    sciddicaTFlowsComputation_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon, i_s, i_e, j_s, j_e);
    cudaDeviceSynchronize();

    // Apply the WidthUpdate mass balance kernel to the whole domain
    sciddicaTWidthUpdate_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, i_s, i_e, j_s, j_e);
    cudaDeviceSynchronize();

  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);


  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

  printf("Releasing memory...\n");
  
  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);
  
  cudaFree(Xi);
  cudaFree(Xj);

  return 0;
}
