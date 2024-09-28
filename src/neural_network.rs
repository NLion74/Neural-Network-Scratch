extern crate rand;
extern crate ndarray_rand;
extern crate ndarray;
//extern crate bincode;
//extern crate serde;

//use std::fs::File;
//use std::io::{self, Write, Read};
//use std::path::PathBuf;
use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
//use crate::serde::{Serialize, Deserialize};
//use serde::{Serialize, Deserialize};

//#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    W1: Array2<f64>,
    W2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
}

impl NeuralNetwork { pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
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

    // pub fn save(&self, path: &PathBuf) -> io::Result<()> {
    //     let data = bincode::serialize(&self)?;
    //     let mut file = File::create(path)?;
    //     file.write_all(&data)?;
    //     Ok(())
    // }

    // Function to load the neural network from a file
    // pub fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
    //     let mut file = File::open(file_path)?;
    //     let mut serialized = Vec::new();
    //     file.read_to_end(&mut serialized)?; // Read the file into a vector
    //     let network: NeuralNetwork = bincode::deserialize(&serialized)?; // Deserialize the vector into the struct
    //     Ok(network)
    // }

    fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
        1.0 / (1.0 + (-z).mapv(f64::exp))
    }

    fn sigmoid_derivative(z: &Array1<f64>) -> Array1<f64> {
        let sig = Self::sigmoid(z);
        sig.clone() * (1.0 - sig)
    }

    pub fn loss_function(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        // Using binary cross-entropy loss
        let epsilon = 1e-15;
        let y_pred_clipped = y_pred.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        let loss = y_true * y_pred_clipped.mapv(|p| p.ln()) + (1.0 - y_true) * (1.0 - y_pred_clipped).mapv(|p| p.ln());
        -loss.mean().unwrap()
    }

    pub fn feed_forward(&self, X: Array1<f64>) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), String> {
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
        let Z2 = A1.dot(&self.W2) + &self.b2; // Calculate Z2: A1 * w2 + b2
        let A2 = Self::sigmoid(&Z2); // Apply the sigmoid activation function to Z2
    
        Ok((Z1, A1, Z2, A2)) // Return the computed values
        }

    pub fn back_propagation(&mut self, X: &Array1<f64>, y: &Array1<f64>, Z1: &Array1<f64>, A1: &Array1<f64>, Z2: &Array1<f64>, A2: &Array1<f64>, learning_rate: f64) {
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
        // Train the neural network for a specified number of epochs
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            // Iterate over each training example, every epoch
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

                // Add the loss for this example to the total loss for the epoch
                total_loss += self.loss_function(&y, &A2);

                // Back propagate
                self.back_propagation(&x, &y, &Z1, &A1, &Z2, &A2, learning_rate);
            }

            // Print the average loss for the epoch
            total_loss /= X_train.nrows() as f64;
            println!("Epoch {}: Loss = {}", epoch + 1, total_loss);
        }
    }

    pub fn accuracy(&self, X: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let mut correct_predictions = 0;

        // Iterate over each test example
        for i in 0..X.nrows() {
            let x = X.row(i).to_owned();

            // Feed forward to get the predicted output
            let (_, _, _, A2) = self.feed_forward(x).expect("Feed forward failed");

            // determine the index of the highest value in the output layer
            let mut max_value = A2[0];
            let mut predicted_class = 0;
            for (index, &value) in A2.iter().enumerate() {
                if value > max_value {
                    max_value = value;
                    predicted_class = index;
                }
            }

            let mut actual_class = None;
            for (index, &value) in y.row(i).iter().enumerate() {
                if value == 1.0 {
                    actual_class = Some(index);
                    break;
                }
            }
            let actual_class = actual_class.unwrap();

            // Compare with the actual class
            if predicted_class == actual_class {
                correct_predictions += 1;
            }
        }

        // Calculate accuracy as a percentage over all test examples
        correct_predictions as f64 / X.nrows() as f64 * 100.0
    }
}