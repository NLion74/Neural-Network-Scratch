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

    fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
        1.0 / (1.0 + (-z).mapv(f64::exp))
    }

    fn sigmoid_derivative(z: &Array1<f64>) -> Array1<f64> {
        let sig = Self::sigmoid(z);
        sig.clone() * (1.0 - sig)
    }

    fn binary_cross_entropy_loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let epsilon = 1e-15;
        let y_pred_clipped = y_pred.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        let loss = y_true * y_pred_clipped.mapv(|p| p.ln()) + (1.0 - y_true) * (1.0 - y_pred_clipped).mapv(|p| p.ln());
        -loss.mean().unwrap()
    }

    fn feed_forward(&self, X: Array1<f64>) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if X.len() != self.input_size {
            return Err(format!(
                "Input size mismatch: expected {}, got {}",
                self.input_size,
                X.len()
            ));
        }

        // Pass the input X through the hidden layer Z1 with weights W1 and biases b1.
        let Z1 = X.dot(&self.W1) + &self.b1; // Calculate Z1: input * weights + biases
        let A1 = Self::sigmoid(&Z1); // Apply the sigmoid activation function to Z1
    
        // Pass the output of the hidden layer Z1 through the output layer Z2 with weights W2 and biases b2.
        let Z2 = A1.dot(&self.W2) + &self.b2; // Calculate Z2: A1 * W2 + b2
        let A2 = Self::sigmoid(&Z2); // Apply the sigmoid activation function to Z2
    
        Ok((Z1, A1, Z2, A2)) // Return the computed values
        }

    fn back_propagation(&mut self, X: &Array1<f64>, y: &Array1<f64>, Z1: &Array1<f64>, A1: &Array1<f64>, Z2: &Array1<f64>, A2: &Array1<f64>, learning_rate: f64) {
        // Output layer error
        let dA2 = A2 - y; 
        // Gradient at output layer
        let dZ2 = dA2.clone() * Self::sigmoid_derivative(Z2);

        // Gradients for W2 and b2
        let dW2 = A1.clone().insert_axis(Axis(1)).dot(&dZ2.clone().insert_axis(Axis(0))); // Shape: (hidden_size, output_size)
        let db2 = dZ2.clone(); // Shape: (output_size,)

        // Hidden layer gradients
        let dZ1 = dZ2.dot(&self.W2.t()) * Self::sigmoid_derivative(Z1);

        // Gradients for W1 and b1
        let dW1 = X.clone().insert_axis(Axis(1)).dot(&dZ1.clone().insert_axis(Axis(0))); // Shape: (input_size, hidden_size)
        let db1 = dZ1.clone(); // Shape: (hidden_size,)

        // Update weights and biases by multiplying by learning rate
        self.W1 -= &(learning_rate * &dW1); // Update weights W1
        self.W2 -= &(learning_rate * &dW2); // Update weights W2
        self.b1 -= &(learning_rate * &db1); // Update biases b1
        self.b2 -= &(learning_rate * &db2); // Update biases b2
    }

    pub fn train(&mut self, X_train: &Array2<f64>, y_train: &Array2<f64>, epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for i in 0..X_train.nrows() {
                let x = X_train.row(i).to_owned();
                let y = y_train.row(i).to_owned();

                // Feed forward
                let (Z1, A1, Z2, A2) = match self.feed_forward(x.clone()) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("Feed forward error: {}", e);
                        continue; // Skip this iteration
                    }
                };

                // Calculate loss
                total_loss += self.binary_cross_entropy_loss(&y, &A2);

                // Backpropagation
                self.back_propagation(&x, &y, &Z1, &A1, &Z2, &A2, learning_rate);

                if i % 10000 == 0 {
                    println!("Loss = {}", total_loss / (i + 1) as f64);
                }
            }

            // Average loss for the epoch
            total_loss /= X_train.nrows() as f64;
            println!("Epoch {}: Loss = {}", epoch + 1, total_loss);
        }
    }

    fn accuracy(&self, X: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let mut correct_predictions = 0;

        for i in 0..X.nrows() {
            let x = X.row(i).to_owned();

            // Feed forward to get the predicted output
            let (_, _, _, A2) = self.feed_forward(x).expect("Feed forward failed");

            // Get the predicted class (index of the highest value)
            let predicted_class = A2.iter().cloned().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

            // Get the actual class index from one-hot encoded labels
            let actual_class = y.row(i).iter().cloned().enumerate()
                .find(|&(_, value)| value == 1.0)
                .map(|(index, _)| index)
                .unwrap();

            // Compare with the actual class
            if predicted_class == actual_class {
                correct_predictions += 1;
            }
        }

        // Calculate accuracy as a percentage
        correct_predictions as f64 / X.nrows() as f64 * 100.0
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

    println!("\nNeural network initialized successfully");

    // Train the network for a specified number of epochs
    let epochs = 1;
    let learning_rate = 0.01;

    neural_network.train(&x_train, &y_train, epochs, learning_rate);

    println!("\nTraining complete");

    // Calculate accuracy on the test dataset
    let test_accuracy = neural_network.accuracy(&x_test, &y_test);
    println!("\nTest accuracy: {:.2}%", test_accuracy);

    Ok(())
}
