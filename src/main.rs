extern crate rand;
extern crate ndarray_rand;
extern crate ndarray;

use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fs::File;
use std::path::PathBuf;
use ndarray::prelude::*;
use std::io;
use std::io::Read;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    W1: Array2<f64>,
    W2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
}

impl NeuralNetwork { pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            W1: Array::random((input_size, hidden_size), Uniform::new(-1.0, 1.0)),
            W2: Array::random((hidden_size, output_size), Uniform::new(-1.0, 1.0)),
            b1: Array::random(hidden_size, Uniform::new(-1.0, 1.0)),
            b2: Array::random(output_size, Uniform::new(-1.0, 1.0)),
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn feed_forward(&self, X: Array1<f64>) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), io::Error> {
        // Pass the input X through the hidden layer Z1 with weights W1 and biases b1. We multiply the input X with the weights W1 and add the biases b1.
        let Z1 = X.dot(&self.W1) + &self.b1;
        let A1 = Z1.mapv(|z| Self::sigmoid(z));

        // Pass the output of the hidden layer Z1 through the output layer Z2 with weights W2 and biases b2. We multiply the output of the hidden layer Z1 with the weights W2 and add the biases b2.
        let Z2 = A1.dot(&self.W2) + &self.b2;
        let A2 = Z2.mapv(|z| Self::sigmoid(z));

        Ok((Z1, A1, Z2, A2))
    }

    fn cost_function(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        // Clip the predicted values to avoid log(0) or log(1).
        let epsilon = 1e-15;
        let y_pred_clipped = y_pred.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        
        
        // Compute the cross-entropy loss average.
        let mut total_loss = 0.0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred_clipped.iter()) {
            let loss_for_example = if true_val == 1.0 {
                -pred_val.ln()
            } else {
                -(1.0 - pred_val).ln()
            };
            total_loss += loss_for_example;
        }

        total_loss / y_true.len() as f64
    }

}

fn read_u32_from_file(file: &mut File) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist_data(images_path: PathBuf, labels_path: PathBuf) -> Result<(Array2<f64>, Array2<f64>), io::Error> {
    let mut image_file = File::open(images_path).expect("Failed to open file");
    let mut label_file = File::open(labels_path).expect("Failed to open file");
    
    // Read header information
    let _magic_images = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_images = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_rows = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_cols = read_u32_from_file(&mut image_file).expect("Failed to read header information");

    let _magic_labels = read_u32_from_file(&mut label_file).expect("Failed to read header information");
    let num_labels = read_u32_from_file(&mut label_file).expect("Failed to read header information");

    assert_eq!(num_images, num_labels, "Number of images and labels do not match");

    let mut image_data = vec![0u8; (num_images * num_rows * num_cols) as usize];
    image_file.read_exact(&mut image_data)?;

    let images = Array2::from_shape_vec(
        (num_images as usize, (num_rows * num_cols) as usize),
        image_data.into_iter().map(|x| x as f64 / 255.0).collect()
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Read label data
    let mut label_data = vec![0u8; num_labels as usize];
    label_file.read_exact(&mut label_data)?;

    let labels = Array2::from_shape_vec(
        (num_labels as usize, 10),
        label_data.into_iter().map(|label| {
            let mut one_hot = vec![0.0; 10];
            one_hot[label as usize] = 1.0;
            one_hot
        }).flatten().collect()
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    Ok((images, labels))
}

fn main() -> Result<(), io::Error> {
    // First we load the mnist dataset and have X train which is a 2D array with 60000 rows and 784 columns.
    // Each entry in the column represents a pixel value of the image. 
    // We also have y_train which is a 2D array with 60000 rows and 10 columns. Each row is one image and each column is the label of the image. So 0-9.
    let train_images_path = PathBuf::from("C:\\Data\\other\\Projects\\Rust\\Neural-Network-Scratch\\train-images.idx3-ubyte");
    let train_labels_path = PathBuf::from("C:\\Data\\other\\Projects\\Rust\\Neural-Network-Scratch\\train-labels.idx1-ubyte");
    let test_images_path = PathBuf::from("C:\\Data\\other\\Projects\\Rust\\Neural-Network-Scratch\\t10k-images.idx3-ubyte");
    let test_labels_path = PathBuf::from("C:\\Data\\other\\Projects\\Rust\\Neural-Network-Scratch\\t10k-labels.idx1-ubyte");
    
    let (x_train, y_train) = match load_mnist_data(train_images_path, train_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load training data: {}", e);
            return Err(e);
        }
    };

    println!("X_train shape: {:?}, y_train shape: {:?}\n", x_train.shape(), y_train.shape());

    let (x_test, y_test) = match load_mnist_data(test_images_path, test_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load test data: {}", e);
            return Err(e);
        }
    };
    println!("X_test shape: {:?}, y_test shape: {:?}", x_test.shape(), y_test.shape());

    println!("\nTraining and Test data loaded successfully");

    // Now we initialize the neural network with 784 input neurons, 64 hidden neurons and 10 output neurons.
    // The weights and biases are set at random.
    let mut neural_network = NeuralNetwork::new(784, 64, 10);

    // Feed training data through the network to check if the feed forward works for now.
    let first_row = x_train.row(0).to_owned();
    let (Z1, A1, Z2, A2) = match neural_network.feed_forward(first_row) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to feed forward: {}", e);
            return Err(e);
        }
    };
    
    println!("\nFeed forward successful:");
    for i in 0..10 {
        println!("{}: {}", i, A2[i]);
    }

    println!("\nActual labels:");
    for i in 0..10 {
        println!("{}: {}", i, y_train[[0, i]]);
    }

    let mut cost = neural_network.cost_function(&y_train.row(0).to_owned(), &A2);

    println!("\nCost function: {}", cost);

    // implement back propagation.
    // Implement training function with cost function.


    // Train the network
    // let epochs = 1000;
    // nn.train(&X_train, &y_train, epochs);

    // Predict on a test sample (dummy example)
    // let test_sample = X_train.slice(s![0, ..]).to_owned();
    // let prediction = nn.predict(&test_sample);
    // println!("Predicted label: {}", prediction);
    Ok(())
}
