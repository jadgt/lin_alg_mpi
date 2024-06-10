use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

#[pymethods]
impl Vector {
    #[new]
    pub fn new(data: Vec<f64>) -> Self {
        Vector { data }
    }

    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.data.len(), other.data.len(), "Vectors must be of the same length");
        let result: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Vector::new(result)
    }

    pub fn sub(&self, other: &Vector) -> Vector {
        assert_eq!(self.data.len(), other.data.len(), "Vectors must be of the same length");
        let result: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Vector::new(result)
    }

    pub fn scalar_mul(&self, scalar: f64) -> Vector {
        let result: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Vector::new(result)
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        assert_eq!(self.data.len(), other.data.len(), "Vectors must be of the same length");
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = v1.add(&v2);
        assert_eq!(result, Vector::new(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::new(vec![4.0, 5.0, 6.0]);
        let v2 = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = v1.sub(&v2);
        assert_eq!(result, Vector::new(vec![3.0, 3.0, 3.0]));
    }

    #[test]
    fn test_scalar_multiplication() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        let result = v.scalar_mul(2.0);
        assert_eq!(result, Vector::new(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = v1.dot(&v2);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vector_norm() {
        let v = Vector::new(vec![3.0, 4.0]);
        let result = v.norm();
        assert_eq!(result, 5.0);
    }
}
