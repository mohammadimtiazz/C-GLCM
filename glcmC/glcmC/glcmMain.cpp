/*Mohammad Shamim Imtiaz*/

#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h> 
#include <iomanip>
#include <conio.h>


#define PGM_MAXMAXVAL 255
#define EPSILON 0.000000001
#define RADIX 2.0
#define SIGN(x,y) ((y)<0 ? -fabs(x) : fabs(x))
#define SWAP(a,b) {y=(a);(a)=(b);(b)=y;}


using namespace std;
using namespace cv;
typedef unsigned char u_int8_t;


//Initialize functions that have been used for measuring co-occurence matrixes for 0,45,90,135 degree angle
double** CoOcMat_Angle_0   (int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
double** CoOcMat_Angle_45  (int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
double** CoOcMat_Angle_90  (int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);
double** CoOcMat_Angle_135 (int distance, u_int8_t **grays, int rows, int cols, int* tone_LUT, int tone_count);


//Initializing functions used for quantifying co-occurance matrixes
double f1_asm (double **P, int Ng);
double f2_contrast (double **P, int Ng);
double f3_corr (double **P, int Ng);
double f4_var (double **P, int Ng);
double f5_idm (double **P, int Ng);
double f6_savg (double **P, int Ng);
double f7_svar (double **P, int Ng, double S);
double f8_sentropy (double **P, int Ng);
double f9_entropy (double **P, int Ng);
double f10_dvar (double **P, int Ng);
double f11_dentropy (double **P, int Ng);
double f12_icorr (double **P, int Ng);
double f13_icorr (double **P, int Ng);
double f14_maxcorr (double **P, int Ng);

//Supporting matrix for above calculation
double *allocate_vector (int nl, int nh);
double **allocate_matrix (int nrl, int nrh, int ncl, int nch);
void free_matrix(double **matrix,int nrh);




//Defining Supporting matrix
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Allocates a double matrix with range [nrl..nrh][ncl..nch] */
double **allocate_matrix (int nrl, int nrh, int ncl, int nch)
{
	int i;
	double **m;

	/* allocate pointers to rows */
	m = (double **) malloc ((unsigned) (nrh - nrl + 1) * sizeof (double *));
	if (!m) fprintf (stderr, "memory allocation failure (allocate_matrix 1) "), exit (1);
	m -= ncl;

	/* allocate rows and set pointers to them */
	for (i = nrl; i <= nrh; i++) {
		m[i] = (double *) malloc ((unsigned) (nch - ncl + 1) * sizeof (double));
		if (!m[i]) fprintf (stderr, "memory allocation failure (allocate_matrix 2) "), exit (2);
		m[i] -= ncl;
	}

	/* return pointer to array of pointers to rows */
	return m;
}


double *allocate_vector (int nl, int nh) {
	double *v;
	
	v = (double *) calloc (1, (unsigned) (nh - nl + 1) * sizeof (double));
	if (!v) fprintf (stderr, "memory allocation failure (allocate_vector) "), exit (1);
	
	return v - nl;
}


/* support functions to compute f14_maxcorr */
void mkbalanced (double **a, int n)
{
  int last, j, i;
  double s, r, g, f, c, sqrdx;

  sqrdx = RADIX * RADIX;
  last = 0;
  while (last == 0)
  {
    last = 1;
    for (i = 1; i <= n; i++)
    {
      r = c = 0.0;
      for (j = 1; j <= n; j++)
	if (j != i)
	{
	  c += fabs (a[j][i]);
	  r += fabs (a[i][j]);
	}
      if (c && r)
      {
	g = r / RADIX;
	f = 1.0;
	s = c + r;
	while (c < g)
	{
	  f *= RADIX;
	  c *= sqrdx;
	}
	g = r * RADIX;
	while (c > g)
	{
	  f /= RADIX;
	  c /= sqrdx;
	}
	if ((c + r) / f < 0.95 * s)
	{
	  last = 0;
	  g = 1.0 / f;
	  for (j = 1; j <= n; j++)
	    a[i][j] *= g;
	  for (j = 1; j <= n; j++)
	    a[j][i] *= f;
	}
      }
    }
  }
}


void reduction (double **a, int n)
{
  int m, j, i;
  double y, x;

  for (m = 2; m < n; m++)
  {
    x = 0.0;
    i = m;
    for (j = m; j <= n; j++)
    {
      if (fabs (a[j][m - 1]) > fabs (x))
      {
	x = a[j][m - 1];
	i = j;
      }
    }
    if (i != m)
    {
      for (j = m - 1; j <= n; j++)
	SWAP (a[i][j], a[m][j])  
	for (j = 1; j <= n; j++)
	  SWAP (a[j][i], a[j][m])
	  a[j][i] = a[j][i];
    }
    if (x)
    {
      for (i = m + 1; i <= n; i++)
      {
	if ( (y = a[i][m - 1]) )
	{
	  y /= x;
	  a[i][m - 1] = y;
	  for (j = m; j <= n; j++)
	    a[i][j] -= y * a[m][j];
	  for (j = 1; j <= n; j++)
	    a[j][m] += y * a[j][i];
	}
      }
    }
  }
}


int hessenberg (double **a, int n, double wr[], double wi[])
{
  int nn, m, l, k, j, its, i, mmin;
  double z, y, x, w, v, u, t, s, r=0.0, q=0.0, p=0.0, anorm;

  anorm = fabs (a[1][1]);
  for (i = 2; i <= n; i++)
    for (j = (i - 1); j <= n; j++)
      anorm += fabs (a[i][j]);
  nn = n;
  t = 0.0;
  while (nn >= 1)
  {
    its = 0;
    do
    {
      for (l = nn; l >= 2; l--)
      {
	s = fabs (a[l - 1][l - 1]) + fabs (a[l][l]);
	if (s == 0.0)
	  s = anorm;
	if ((double) (fabs (a[l][l - 1]) + s) == s)
	  break;
      }
      x = a[nn][nn];
      if (l == nn)
      {
	wr[nn] = x + t;
	wi[nn--] = 0.0;
      }
      else
      {
	y = a[nn - 1][nn - 1];
	w = a[nn][nn - 1] * a[nn - 1][nn];
	if (l == (nn - 1))
	{
	  p = 0.5 * (y - x);
	  q = p * p + w;
	  z = sqrt (fabs (q));
	  x += t;
	  if (q >= 0.0)
	  {
	    z = p + SIGN (z, p); 
	    wr[nn - 1] = wr[nn] = x + z;
	    if (z)
	      wr[nn] = x - w / z;
	    wi[nn - 1] = wi[nn] = 0.0;
	  }
	  else
	  {
	    wr[nn - 1] = wr[nn] = x + p;
	    wi[nn - 1] = -(wi[nn] = z);
	  }
	  nn -= 2;
	}
	else
	{
	  if (its == 30)
	    {
	     return 0;
	     }			
	  if (its == 10 || its == 20)
	  {
	    t += x;
	    for (i = 1; i <= nn; i++)
	      a[i][i] -= x;
	    s = fabs (a[nn][nn - 1]) + fabs (a[nn - 1][nn - 2]);
	    y = x = 0.75 * s;
	    w = -0.4375 * s * s;
	  }
	  ++its;
	  for (m = (nn - 2); m >= l; m--)
	  {
	    z = a[m][m];
	    r = x - z;
	    s = y - z;
	    p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
	    q = a[m + 1][m + 1] - z - r - s;
	    r = a[m + 2][m + 1];
	    s = fabs (p) + fabs (q) + fabs (r);
	    p /= s;
	    q /= s;
	    r /= s;
	    if (m == l)
	      break;
	    u = fabs (a[m][m - 1]) * (fabs (q) + fabs (r));
	    v = fabs (p) * (fabs (a[m - 1][m - 1]) + 
			    fabs (z) + fabs (a[m + 1][m + 1]));
	    if ((double) (u + v) == v)
	      break;
	  }
	  for (i = m + 2; i <= nn; i++)
	  {
	    a[i][i - 2] = 0.0;
	    if (i != (m + 2))
	      a[i][i - 3] = 0.0;
	  }
	  for (k = m; k <= nn - 1; k++)
	  {
	    if (k != m)
	    {
	      p = a[k][k - 1];
	      q = a[k + 1][k - 1];
	      r = 0.0;
	      if (k != (nn - 1))
		r = a[k + 2][k - 1];
	      if ( (x = fabs (p) + fabs (q) + fabs (r)) )
	      {
		p /= x;
		q /= x;
		r /= x;
	      }
	    }
	    if ( (s = SIGN (sqrt (p * p + q * q + r * r), p)) ) 
	    {
	      if (k == m)
	      {
		if (l != m)
		  a[k][k - 1] = -a[k][k - 1];
	      }
	      else
		a[k][k - 1] = -s * x;
	      p += s;
	      x = p / s;
	      y = q / s;
	      z = r / s;
	      q /= p;
	      r /= p;
	      for (j = k; j <= nn; j++)
	      {
		p = a[k][j] + q * a[k + 1][j];
		if (k != (nn - 1))
		{
		  p += r * a[k + 2][j];
		  a[k + 2][j] -= p * z;
		}
		a[k + 1][j] -= p * y;
		a[k][j] -= p * x;
	      }
	      mmin = nn < k + 3 ? nn : k + 3;
	      for (i = l; i <= mmin; i++)
	      {
		p = x * a[i][k] + y * a[i][k + 1];
		if (k != (nn - 1))
		{
		  p += z * a[i][k + 2];
		  a[i][k + 2] -= p * r;
		}
		a[i][k + 1] -= p * q;
		a[i][k] -= p;
	      }
	    }
	  }
	}
      }
    } while (l < nn - 1);
  }
return 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Defining co-occurance matrix
/* Compute gray-tone spatial dependence matrix at 0 degree angle*/
double** CoOcMat_Angle_0 (int distance, u_int8_t **grays,
						 int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	double count=0.0; /* normalizing factor */
	

	double** matrix = allocate_matrix (0, tone_count, 0, tone_count);
	
	/* zero out matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix[itone][jtone] = 0.0;

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			//if (grays[row][col] == 0)
			//	continue;

			/* find x tone */
			if (col + d < cols) {		// previously stated condition(col + d < cols && grays[row][col + d])
				x = tone_LUT[grays[row][col]];
				y = tone_LUT[grays[row][col + d]];
				matrix[x][y]++;
				matrix[y][x]++;
				count += 2.0 ;
			}
		}

	// resultant matrix is the symmetrical co-occurence matrix before normalization
	// the results are exact match as python sci-kit lib
	cout << "the symmetrical co-occurence matrix before normalization at 0 degree: " << endl << endl;
	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			cout << setprecision(10) << matrix[row][col] << ' ';
		}
		cout << endl;
	}


	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone){
          for (jtone = 0; jtone < tone_count; ++jtone){
            if (count==0.0)   /* protect from error */
               matrix[itone][jtone]=0.0;
               else matrix[itone][jtone] /= count;
		  }
	}


	// resultant matrix is the NORMALIZE symmetrical co-occurence matrix
	// the results are exact match as python sci-kit lib
	cout << "the NORMALIZE symmetrical co-occurence matrix before normalization at 0 degree: " << endl << endl;
	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			//printf("%.5f", matrix[row][col]);
			cout << setprecision(10) << matrix[row][col] << ' ';

		}
		cout << endl;
	}

	return matrix;
}


/* Compute gray-tone spatial dependence matrix at 90 degree angle*/
double** CoOcMat_Angle_90 (int distance, u_int8_t **grays,
						   int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count=0; /* normalizing factor */
	
	double** matrix = allocate_matrix (0, tone_count, 0, tone_count);

	/* zero out matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix[itone][jtone] = 0;
			
	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			//if (grays[row][col] == 0)
			//	continue;
				
			/* find x tone */
			if (row + d < rows) {			// previously stated condition(row + d < rows && grays[row + d][col])
				x = tone_LUT [grays[row][col]];
				y = tone_LUT [grays[row + d][col]];
				matrix[x][y]++;
				matrix[y][x]++;
				count += 2 ;
			}
		}
	

	// resultant matrix is the symmetrical co-occurence matrix before normalization
	// the results are exact match as python sci-kit lib
	cout << "the symmetrical co-occurence matrix before normalization at 90 Degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			cout << setprecision(10) << matrix[row][col] << ' ';
		}
		cout << endl;
	}


	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
          for (jtone = 0; jtone < tone_count; ++jtone)
            if (count==0) matrix[itone][jtone]=0;
            else matrix[itone][jtone] /= count;


	// resultant matrix is the NORMALIZE symmetrical co-occurence matrix
	// the results are exact match as python sci-kit lib
	cout << "the NORMALIZE symmetrical co-occurence matrix before normalization at 90 degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			//printf("%.5f", matrix[row][col]);
			cout << setprecision(10) << matrix[row][col] << ' ';

		}
		cout << endl;
	}


	return matrix;
}


/* Compute gray-tone spatial dependence matrix at 45 degree angle*/
double** CoOcMat_Angle_45 (int distance, u_int8_t **grays,
						    int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count=0; /* normalizing factor */

	double** matrix = allocate_matrix (0, tone_count, 0, tone_count);
	
	/* zero out matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix[itone][jtone] = 0;
			
	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			//if (grays[row][col] == 0)
			//	continue;
				
			/* find x tone */
			if (row + d < rows && col + d < cols) {		//previously stated condition (row + d < rows && col + d < cols && grays[row + d][col + d])
				x = tone_LUT [grays[row][col]];
				y = tone_LUT [grays[row + d][col + d]];
				matrix[x][y]++;
				matrix[y][x]++;
				count += 2 ;
			}
		}
	
	// resultant matrix is the symmetrical co-occurence matrix before normalization
	// the results are exact match as python sci-kit lib
	cout << "the symmetrical co-occurence matrix before normalization at 45 Degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			cout << setprecision(10) << matrix[row][col] << ' ';
		}
		cout << endl;
	}	

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
                if (count==0) matrix[itone][jtone]=0;   /* protect from error */
                else matrix[itone][jtone] /= count;


	// resultant matrix is the NORMALIZE symmetrical co-occurence matrix
	// the results are exact match as python sci-kit lib
	cout << "the NORMALIZE symmetrical co-occurence matrix before normalization at 45 degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			//printf("%.5f", matrix[row][col]);
			cout << setprecision(10) << matrix[row][col] << ' ';

		}
		cout << endl;
	}

	return matrix;
}


/* Compute gray-tone spatial dependence matrix at 135 degree angle*/
double** CoOcMat_Angle_135 (int distance, u_int8_t **grays,
						   int rows, int cols, int* tone_LUT, int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count=0; /* normalizing factor */
	
	double** matrix = allocate_matrix (0, tone_count, 0, tone_count);
	
	/* zero out matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			matrix[itone][jtone] = 0;
			
	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			/* only non-zero values count*/
			//if (grays[row][col] == 0)
			//	continue;
				
			/* find x tone */
			if (row + d < rows && col - d >= 0) {		//previously stated condition (row + d < rows && col - d >= 0 && grays[row + d][col - d])
				x = tone_LUT [grays[row][col]];
				y = tone_LUT [grays[row + d][col - d]];
				matrix[x][y]++;
				matrix[y][x]++;
				count += 2 ;
			}
		}
	

	// resultant matrix is the symmetrical co-occurence matrix before normalization
	// the results are exact match as python sci-kit lib
	cout << "the symmetrical co-occurence matrix before normalization at 135 Degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			cout << setprecision(10) << matrix[row][col] << ' ';
		}
		cout << endl;
	}


	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
                   if (count==0) matrix[itone][jtone]=0;       /* protect from error */
                   else matrix[itone][jtone] /= count;

	// resultant matrix is the NORMALIZE symmetrical co-occurence matrix
	// the results are exact match as python sci-kit lib
	cout << "the NORMALIZE symmetrical co-occurence matrix before normalization at 135 degree: " << endl << endl;

	for (row = 0; row < rows; ++row){
		for (col = 0; col < cols; ++col) {
			//printf("%.5f", matrix[row][col]);
			cout << setprecision(10) << matrix[row][col] << ' ';

		}
		cout << endl;
	}

	return matrix;
}






// Defining all quantifying functions derived from co-occurance matrix

/* Angular Second Moment
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/

double f1_asm (double **P, int Ng) {
	int i, j;
	double sum = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			sum += P[i][j] * P[i][j];
	
	return sum;
}



/* Contrast
* The contrast feature is a difference moment of the P matrix and is a
* measure of the contrast or the amount of local variations present in an
* image.
*/
double f2_contrast (double **P, int Ng) {
	int i, j, n;
	double sum = 0, bigsum = 0;
	
	for (n = 0; n < Ng; ++n) {
		for (i = 0; i < Ng; ++i)
			for (j = 0; j < Ng; ++j) {
				if ((i - j) == n || (j - i) == n)
					sum += P[i][j];
				}
		bigsum += n * n * sum;
		sum = 0;
	}
	
	return bigsum;
}


/* Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*/
double f3_corr (double **P, int Ng) {
	int i, j;
	double sum_sqrx = 0, sum_sqry = 0, tmp, *px;
	double meanx =0 , meany = 0 , stddevx, stddevy;

	px = allocate_vector (0, Ng);
	for (i = 0; i < Ng; ++i)
		px[i] = 0;
	
	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			px[i] += P[i][j];
	
	
	/* Now calculate the means and standard deviations of px and py */
	/*- fix supplied by J. Michael Christensen, 21 Jun 1991 */
	/*- further modified by James Darrell McCauley, 16 Aug 1991
	*     after realizing that meanx=meany and stddevx=stddevy
	*/
	for (i = 0; i < Ng; ++i) {
		meanx += px[i]*i;
		sum_sqrx += px[i]*i*i;
	}
	
	/* M. Boland meanx = meanx/(sqrt(Ng)); */
	meany = meanx;
	sum_sqry = sum_sqrx;
	stddevx = sqrt (sum_sqrx - (meanx * meanx));
	stddevy = stddevx;
	
	/* Finally, the correlation ... */
	for (tmp = 0, i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			  tmp += i*j*P[i][j];
	
	free(px);
        if (stddevx * stddevy==0) return(1);  /* protect from error */
        else return (tmp - meanx * meany) / (stddevx * stddevy);
}


/* Sum of Squares: Variance */
double f4_var (double **P, int Ng) {
	int i, j;
	double mean = 0, var = 0;
	
	/*- Corrected by James Darrell McCauley, 16 Aug 1991
	*  calculates the mean intensity level instead of the mean of
	*  cooccurrence matrix elements
	*/
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			mean += i * P[i][j];
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
		  /*  M. Boland - var += (i + 1 - mean) * (i + 1 - mean) * P[i][j]; */
		  var += (i - mean) * (i - mean) * P[i][j];
	
	return var;
}


/* Inverse Difference Moment */
double f5_idm (double **P, int Ng) {
	int i, j;
	double idm = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			idm += P[i][j] / (1 + (i - j) * (i - j));

	return idm;
}


/* Sum Average */
double f6_savg (double **P, int Ng) {
	int i, j;
	double savg = 0;
	double *Pxpy = allocate_vector (0, 2*Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
		  /* M. Boland Pxpy[i + j + 2] += P[i][j]; */
		  /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
		  Pxpy[i + j] += P[i][j];
		  
	/* M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		savg += i * Pxpy[i];
	
	free (Pxpy);
	return savg;
}


/* Sum Variance */
double f7_svar (double **P, int Ng, double S) {
	int i, j;
	double var = 0;
	double *Pxpy = allocate_vector (0, 2*Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
		  /* M. Boland Pxpy[i + j + 2] += P[i][j]; */
		  /* Indexing from 2 instead of 0 is inconsistent with rest of code*/
		  Pxpy[i + j] += P[i][j];

	/*  M. Boland for (i = 2; i <= 2 * Ng; ++i) */
	/* Indexing from 2 instead of 0 is inconsistent with rest of code*/
	for (i = 0; i <= (2 * Ng - 2); ++i)
		var += (i - S) * (i - S) * Pxpy[i];
	
	free (Pxpy);
	return var;
}


/* Sum Entropy */
double f8_sentropy (double **P, int Ng) {
	int i, j;
	double sentropy = 0;
	double *Pxpy = allocate_vector (0, 2*Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
		  Pxpy[i + j + 2] += P[i][j];
	
	for (i = 2; i <= 2 * Ng; ++i)
		/*  M. Boland  sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sentropy -= Pxpy[i] * log10 (Pxpy[i] + EPSILON)/log10(2.0) ;

	free (Pxpy);
	return sentropy;
}


/* Entropy */
double f9_entropy (double **P, int Ng) {
	int i, j;
	double entropy = 0;
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			/*      entropy += P[i][j] * log10 (P[i][j] + EPSILON); */
			entropy += P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0) ;
	
	return -entropy; 
}


/* Difference Variance */
double f10_dvar (double **P, int Ng) {
	int i, j;
	double sum = 0, sum_sqr = 0, var = 0;
	double *Pxpy = allocate_vector (0, 2*Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs (i - j)] += P[i][j];
	
	/* Now calculate the variance of Pxpy (Px-y) */
	for (i = 0; i < Ng; ++i) {
		sum += i * Pxpy[i] ;
		sum_sqr += i * i * Pxpy[i] ;
		/* M. Boland sum += Pxpy[i];
		sum_sqr += Pxpy[i] * Pxpy[i];*/
	}

	/*tmp = Ng * Ng ;  M. Boland - wrong anyway, should be Ng */
	/*var = ((tmp * sum_sqr) - (sum * sum)) / (tmp * tmp); */
	
	var = sum_sqr - sum*sum ;
	
	free (Pxpy);
	return var;
}


/* Difference Entropy */
double f11_dentropy (double **P, int Ng) {
	int i, j;
	double sum = 0;
	double *Pxpy = allocate_vector (0, 2*Ng);

	for (i = 0; i <= 2 * Ng; ++i)
		Pxpy[i] = 0;

	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j)
			Pxpy[abs (i - j)] += P[i][j];
	
	for (i = 0; i < Ng; ++i)
		/*    sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON); */
		sum += Pxpy[i] * log10 (Pxpy[i] + EPSILON)/log10(2.0) ;
		
	free (Pxpy);
	return -sum;
}


/* Information Measures of Correlation */
double f12_icorr (double **P, int Ng) {
	int i, j;
	double *px, *py;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;
	
	px = allocate_vector (0, Ng);
	py = allocate_vector (0, Ng);
	/* All /log10(2.0) added by M. Boland */
	
	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P[i][j];
			py[j] += P[i][j];
		}
	}
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j) {
			hxy1 -= P[i][j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
			hxy2 -= px[i] * py[j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
			hxy -= P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0);
		}
	
	/* Calculate entropies of px and py - is this right? */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10 (px[i] + EPSILON)/log10(2.0);
		hy -= py[i] * log10 (py[i] + EPSILON)/log10(2.0);
	}

	free(px);
	free(py);
        if ((hx > hy ? hx : hy)==0) return(1);
        else
	return ((hxy - hxy1) / (hx > hy ? hx : hy));
}


/* Information Measures of Correlation */
double f13_icorr (double **P, int Ng) {
	int i, j;
	double *px, *py;
	double hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;
	
	px = allocate_vector (0, Ng);
	py = allocate_vector (0, Ng);
	
	/* All /log10(2.0) added by M. Boland */

	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
		  px[i] += P[i][j];
		  py[j] += P[i][j];
		}
	}
	
	for (i = 0; i < Ng; ++i)
		for (j = 0; j < Ng; ++j) {
			hxy1 -= P[i][j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
			hxy2 -= px[i] * py[j] * log10 (px[i] * py[j] + EPSILON)/log10(2.0);
			hxy -= P[i][j] * log10 (P[i][j] + EPSILON)/log10(2.0);
		}

	/* Calculate entropies of px and py */
	for (i = 0; i < Ng; ++i) {
		hx -= px[i] * log10 (px[i] + EPSILON)/log10(2.0);
		hy -= py[i] * log10 (py[i] + EPSILON)/log10(2.0);
	}

	free(px);
	free(py);
	return (sqrt (fabs (1 - exp (-2.0 * (hxy2 - hxy)))));
}


/* Returns the Maximal Correlation Coefficient */
double f14_maxcorr (double **P, int Ng) {
	int i, j, k;
	double *px, *py, **Q;
	double *x, *iy, tmp;
	double f=0.0;
	
	px = allocate_vector (0, Ng);
	py = allocate_vector (0, Ng);
	Q = allocate_matrix (1, Ng + 1, 1, Ng + 1);
	x = allocate_vector (1, Ng);
	iy = allocate_vector (1, Ng);
	
	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P[i][j];
			py[j] += P[i][j];
		}
	}
	
	/* Find the Q matrix */
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			Q[i + 1][j + 1] = 0;
			for (k = 0; k < Ng; ++k)
                          if (px[i] && py[k])  /* make sure to protect division by zero */
  			    Q[i + 1][j + 1] += P[i][k] * P[j][k] / px[i] / py[k];
		}
	}

	/* Balance the matrix */
	mkbalanced (Q, Ng);
	/* Reduction to Hessenberg Form */
	reduction (Q, Ng);
	/* Finding eigenvalue for nonsymetric matrix using QR algorithm */
	if (!hessenberg (Q, Ng, x, iy)) {
		/* Memmory cleanup */
		for (i=1; i<=Ng+1; i++) free(Q[i]+1);
		free(Q+1);
		free((char *)px);
		free((char *)py);
		free((x+1));
		free((iy+1));

		/* computation failed ! */
		return 0.0;
	}

	/* simplesrt(Ng,x); */
	/* Returns the sqrt of the second largest eigenvalue of Q */
	for (i = 2, tmp = x[1]; i <= Ng; ++i)
		tmp = (tmp > x[i]) ? tmp : x[i];

	if (x[Ng - 1]>=0)
	  f = sqrt(x[Ng - 1]);

	for (i=1; i<=Ng+1; i++) free(Q[i]+1);
	free(Q+1);
	free((char *)px);
	free((char *)py);
	free((x+1));
	free((iy+1));

	return f;
}


int main(){
	IplImage *img;
	uchar intensity;
	
	img = cvLoadImage("z.bmp", 0);
	// image z has following matrix:
	// 0 0 1 1
	// 0 0 1 1
	// 0 2 2 2
	// 2 2 3 3

	// prinitng the image values
	cout << endl << "Printing the image vaules: " << endl << endl;
	for(int y = 0; y < img->height; y++){
		for(int x = 0; x < img->width; x++){
			intensity = img->imageData[img->widthStep * y + x * 1];
			cout << (unsigned int)intensity << "   ";
		}
		cout << endl;
	}



	int row, col, rows, cols;
	cols = img->width;
	rows = img->height;


	// create a 2-dim array **pGray and copy all the image values in **pGray
	unsigned char **pGray;
	pGray = new unsigned char *[rows];

	for(int i = 0; i < rows; i++){
		pGray[i] = new unsigned char[cols];
	}

	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			pGray[y][x] = img->imageData[img->widthStep * y + x * 1];
		}
	}

	//just for varification the pGray has the same value as the image
	cout << endl << "Printing the image vaules from **pGray: " << endl << endl;
	for(int y = 0; y < img->height; y++){
		for(int x = 0; x < img->width; x++){
			intensity = pGray[y][x];
			cout << (unsigned int)intensity << "   ";
		}
		cout << endl;
	}


	int toneLUT[PGM_MAXMAXVAL + 1];		// toneLUT is an array that can hold 256 values
	int toneCount = 0;
	int iTone;
	

	//FIll toneLUT with -1
	for(row = PGM_MAXMAXVAL; row >= 0; --row)
		toneLUT[row] = -1;

	//fill toneLUT with those 8 bit values which contains in the image
	//example: z image has value 0,1,2,3,4
	// so toneLUT will have values only from 0 - 4. 
	for(row = rows - 1; row >= 0; --row){
		for(col = 0; col < cols; ++col){
			//cout << (unsigned int)img->imageData[img->widthStep * row + col * 1] << endl;
			toneLUT[(u_int8_t)img->imageData[img->widthStep * row + col * 1]] = (u_int8_t)img->imageData[img->widthStep * row + col * 1];
		}
	}

	//toneCount contains the number of 8-bit value variation in the image 
	for (row = PGM_MAXMAXVAL, toneCount = 0; row >= 0; --row){
		if (toneLUT[row] != -1)
			toneCount++;
		else
			;
	}
		

	/* Use the number of different tones to build LUT */
	for (row = 0, iTone = 0; row <= PGM_MAXMAXVAL; row++){
		if (toneLUT[row] != -1)
		  toneLUT[row] = iTone++;
	}

	double **pMatrix;
	int distance = 1;	// here the default distance is 1
	//pMatrix will contain the normalize symmetrical co-occurance matrix calculated from the input image for distance 1 and angle 0 
	pMatrix = CoOcMat_Angle_0(distance, pGray, rows, cols, toneLUT, toneCount);

	//cout << endl << endl;

	//for(int y = 0; y < img->height; y++){
	//	for(int x = 0; x < img->width; x++){
	//		cout << pMatrix[y][x] << "   ";
	//	}
	//	cout << endl;
	//}

	double m_asm, m_contrast, m_corr, m_var, m_idm, m_savg, m_svar, m_sentropy, m_entropy, m_dvar, m_dentropy, m_icorr1, m_icorr2, m_maxcorr;

	//measuring asm at 0 degree
	m_asm = f1_asm (pMatrix , toneCount);

	//measuring contrast at 0 degree
	m_contrast = f2_contrast(pMatrix , toneCount);

	//measuring correlation at 0 degree
	m_corr = f3_corr(pMatrix, toneCount);

	//measuring Sum of Squares: Variance at 0 degree
	m_var = f4_var(pMatrix, toneCount);

	//measuring Inverse Difference Moment at 0 degree
	m_idm = f5_idm(pMatrix, toneCount);

	//measuring Sum Average at 0 degree
	m_savg = f6_savg(pMatrix, toneCount);

	//measuring Sum Entropy at 0 degree
	m_sentropy = f8_sentropy(pMatrix, toneCount);

	//measuring Sum Variance at 0 degree
	m_svar = f7_svar(pMatrix, toneCount, m_sentropy);

	//measuring Entropy at 0 degree
	m_entropy = f9_entropy(pMatrix, toneCount);

	//measuring Difference Variance at 0 degree
	m_dvar = f10_dvar(pMatrix, toneCount);

	//measuring Difference Entropy at 0 degree
	m_dentropy = f11_dentropy(pMatrix, toneCount);

	//measuring Information Measures of Correlation1 at 0 degree
	m_icorr1 = f12_icorr(pMatrix, toneCount);

	//measuring Information Measures of Correlation2 at 0 degree
	m_icorr2 = f13_icorr(pMatrix, toneCount);

	//measuring Returns the Maximal Correlation Coefficient at 0 degree
	m_maxcorr = f14_maxcorr(pMatrix, toneCount);


	cout << endl << "ASM:: " << m_asm << endl;
	cout <<  "Contrast:: " << m_contrast << endl;
	cout << "correlation:: " << m_corr << endl;
	cout << "Sum of Squares: Variance:: " << m_var << endl;
	cout <<  "Inverse Difference Moment:: " << m_idm << endl;
	cout << "Sum Average:: " << m_savg << endl;
	cout << "Sum Variance:: " << m_svar << endl;
	cout <<  "Sum Entropy:: " << m_sentropy << endl;
	cout << "Entropy:: " << m_entropy << endl;
	cout << "Difference Variance:: " << m_dvar << endl;
	cout << "Difference Entropy:: " << m_dentropy << endl;
	cout <<  "Information Measures of Correlation1:: " << m_icorr1 << endl;
	cout << "Information Measures of Correlation2:: " << m_icorr2 << endl;
	cout << "Returns the Maximal Correlation Coefficient:: " << m_maxcorr << endl;
	cout << endl << "Press any key to exit the program" << endl;

	getch();
	return 0;



}

