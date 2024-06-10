use crate::vector::Vector;
use rayon::prelude::*;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
}

#[pymethods]
impl Matrix {
    #[new]
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let row_length = data[0].len();
        assert!(data.iter().all(|row| row.len() == row_length), "All rows must have the same length");
        Matrix { data }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.data.len(), other.data.len(), "Matrices must have the same dimensions");
        assert_eq!(self.data[0].len(), other.data[0].len(), "Matrices must have the same dimensions");
        let result: Vec<Vec<f64>> = self.data.par_iter()
            .zip(&other.data)
            .map(|(row1, row2)| row1.iter().zip(row2).map(|(a, b)| a + b).collect())
            .collect();
        Matrix::new(result)
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.data.len(), other.data.len(), "Matrices must have the same dimensions");
        assert_eq!(self.data[0].len(), other.data[0].len(), "Matrices must have the same dimensions");
        let result: Vec<Vec<f64>> = self.data.par_iter()
            .zip(&other.data)
            .map(|(row1, row2)| row1.iter().zip(row2).map(|(a, b)| a - b).collect())
            .collect();
        Matrix::new(result)
    }

    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let result: Vec<Vec<f64>> = self.data.par_iter()
            .map(|row| row.iter().map(|&x| x * scalar).collect())
            .collect();
        Matrix::new(result)
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.data[0].len(), other.data.len(), "Number of columns in the first matrix must be equal to the number of rows in the second matrix");
        let result: Vec<Vec<f64>> = self.data.par_iter()
            .map(|row| (0..other.data[0].len()).map(|j| row.iter().zip(&other.data).map(|(a, col)| a * col[j]).sum()).collect())
            .collect();
        Matrix::new(result)
    }

    pub fn mul_vector(&self, vector: &Vector) -> Vector {
        assert_eq!(self.data[0].len(), vector.data.len(), "Number of columns in the matrix must be equal to the number of elements in the vector");
        let result: Vec<f64> = self.data.par_iter()
            .map(|row| row.iter().zip(&vector.data).map(|(a, b)| a * b).sum())
            .collect();
        Vector::new(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.add(&m2);
        assert_eq!(result, Matrix::new(vec![vec![6.0, 8.0], vec![10.0, 12.0]]));
    }

    #[test]
    fn test_matrix_subtraction() {
        let m1 = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let m2 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m1.sub(&m2);
        assert_eq!(result, Matrix::new(vec![vec![4.0, 4.0], vec![4.0, 4.0]]));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m.scalar_mul(2.0);
        assert_eq!(result, Matrix::new(vec![vec![2.0, 4.0], vec![6.0, 8.0]]));
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::new(vec![vec![2.0, 0.0], vec![1.0, 2.0]]);
        let result = m1.mul(&m2);
        assert_eq!(result, Matrix::new(vec![vec![4.0, 4.0], vec![10.0, 8.0]]));
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v = Vector::new(vec![2.0, 1.0]);
        let result = m.mul_vector(&v);
        assert_eq!(result, Vector::new(vec![4.0, 10.0]));
    }
}
