## Homework 3

### Part 1

The file `oil_price.dat` contains the monthly oil price since 1985. The file contains three columns: year, month, price in Euros (from the european bank website). Make a plot of the oil price (put numbers along the horizontal axis; we will learn how to do dates in another notebook) and determine the month and year the oil price first rose above 40 euros, above 60 euros, and above 80 euros. You need to write to the screen something like The oil price exceeds 40 euros for the first time in month xx of year yyyy where xx and yyyy are the correct month and year. Use a double loop. Can you modify the code such that it doesn't print the number of the month but the name of the month?

### Part 2

Find the value of x for which F(x) = p, where p is a probablity of interest (so it is between 0 and 1). Check you answer for &mu; = 3, σ = 2, and find x for p = 0.1 and p = 0.9. Substitute the roots you determine with fsolve back into F(x) to make sure your code works properly.

### Part 3 

With the file, `douglas_data.csv`, the moisture content is defined as the mass of moisture in a beam divided by the total mass of the beam (including the moisture) and is recorded as a percentage. Compute and report the mean and standard deviation of the moisture content, and make a box plot.

When you look at the data, it is obvious that there is one outlier. Create a new boxplot for all the data except for the one outlier, for example by making a boxplot for all moisture data below a certain value. Make sure you choose correct limits for the vertical axis, so that the whiskers are visible.

### Part 4

In this homework you will write four basic matrix operations. Recall, that a matrix is a two dimensional list containing only integers, used in linear algebra to represent the coefficients of multivariable expressions. Submit the following in `matrix.py`. 

### Part 5

1. Write a function `matrix_transpose(matrix_A)` returns the transpose of matrix A as a two dimensional list. The definition of transpose can be found here: https://en.wikipedia.org/wiki/Transpose

2. Write a function `matrix_det(matrix_A)` that calculates the determinant (an integer) of matrix A if it is a 1x1 or 2x2 matrix. If the input matrix is not square, print this error message { \Cannot calculate determinant of a non-square matrix"; If the input matrix is square but has a higher dimension, print this error message { \Cannot calculate determinant of a n-by-n matrix", where you substitute n with the dimension of the input matrix. (from wikipedia https://en.wikipedia.org/wiki/Determinant)

3. Recall matrix addition here https://en.wikipedia.org/wiki/Matrix_addition. Write a function `matrix_add(matrix_A, matrix_B)` that performs matrix addition if the dimensionality is valid. Note that the dimensionality is only valid if input matrix A and input matrix B are of the same dimension in both their row and column lengths.  
  For example, you can add a 3x5 matrix with a 3x5 matrix, but you cannot add a 3x5 matrix with a 3x1 matrix. If the dimensionality is not valid, print this error message { \Cannot perform matrix addition between a-by-b matrix and c-by-d matrix", where you substitute a, b with the dimension of the input matrix A, and c,d with the dimension of the input matrix B.

4. Recall matrix multiplication here https://en.wikipedia.org/wiki/Matrix_multiplication. Write a function `matrix_mult(matrix_A, matrix_B)` that performs matrix multiplication if the dimensionality is valid.

  Note that the dimensionality is only valid if the number of columns of input matrix A is the same as the number of rows of input matrix B. Multiplying a 5x3 matrix with a 3x7 matrix is valid (resulting in a 5x7 matrix), while doing so with a 5x1 and a 7x1 matrix would not be. 

  NOTE: order matters in matrix multiplication! Even though it is valid to multiply a 5x3 matrix with a 3x7 matrix in that order, it would not be valid to multiply a 3x7 matrix with a 5x3 matrix.

  If the dimensionality is not valid, print this error message { \Cannot perform matrix multiplication between a-by-b matrix and c-by-d matrix", where you substitute a, b with the dimension of the input matrix A, and c,d with the dimension of the input matrix B.

5. Write a function `matrix_inverse(matrix_A)` that outputs the inverse matrix. 

  Determine whether the following system of equations has no solution, infinite solutions or a unique solution without solving the system. Explain your answer in `README.txt`.

``` python
A = np.array([[1,2,-1,1,2],[3,-4,0,2,3],[0,2,1,0,4],[2,2,-3,2,0],[-2,6,-1,-1,-1]])
```

### Optional Reading

[TimeSeries Graphs](https://www.datadoghq.com/blog/timeseries-metric-graphs-101/)
