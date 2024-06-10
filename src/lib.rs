use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod vector;
mod matrix;

pub use vector::Vector;
pub use matrix::Matrix;

#[pyfunction]
fn add_vectors(v1: Vec<f64>, v2: Vec<f64>) -> Vec<f64> {
    let vec1 = Vector::new(v1);
    let vec2 = Vector::new(v2);
    vec1.add(&vec2).data
}

#[pyfunction]
fn sub_vectors(v1: Vec<f64>, v2: Vec<f64>) -> Vec<f64> {
    let vec1 = Vector::new(v1);
    let vec2 = Vector::new(v2);
    vec1.sub(&vec2).data
}

#[pyfunction]
fn scalar_mul_vector(v: Vec<f64>, scalar: f64) -> Vec<f64> {
    let vec = Vector::new(v);
    vec.scalar_mul(scalar).data
}

#[pyfunction]
fn dot_product(v1: Vec<f64>, v2: Vec<f64>) -> f64 {
    let vec1 = Vector::new(v1);
    let vec2 = Vector::new(v2);
    vec1.dot(&vec2)
}

#[pyfunction]
fn norm_vector(v: Vec<f64>) -> f64 {
    let vec = Vector::new(v);
    vec.norm()
}

#[pyfunction]
fn add_matrices(m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mat1 = Matrix::new(m1);
    let mat2 = Matrix::new(m2);
    mat1.add(&mat2).data
}

#[pyfunction]
fn sub_matrices(m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mat1 = Matrix::new(m1);
    let mat2 = Matrix::new(m2);
    mat1.sub(&mat2).data
}

#[pyfunction]
fn scalar_mul_matrix(m: Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
    let mat = Matrix::new(m);
    mat.scalar_mul(scalar).data
}

#[pyfunction]
fn mul_matrices(m1: Vec<Vec<f64>>, m2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mat1 = Matrix::new(m1);
    let mat2 = Matrix::new(m2);
    mat1.mul(&mat2).data
}

#[pyfunction]
fn mul_matrix_vector(m: Vec<Vec<f64>>, v: Vec<f64>) -> Vec<f64> {
    let mat = Matrix::new(m);
    let vec = Vector::new(v);
    mat.mul_vector(&vec).data
}

#[pymodule]
fn linear_algebra(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(sub_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_mul_vector, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(norm_vector, m)?)?;
    m.add_function(wrap_pyfunction!(add_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(sub_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_mul_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mul_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(mul_matrix_vector, m)?)?;
    Ok(())
}
