extern crate rand;
extern crate ndarray_rand;
extern crate ndarray;

mod neural_network;

use neural_network::NeuralNetwork;
use ndarray::Array2;
use std::fs::File;
use std::path::PathBuf;
use std::io::{self, Write, Read};

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
    print!("Please enter the path to the MNIST dataset (defaults to './dataset'): ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");

    let path = if input.trim().is_empty() {
        "./dataset".to_string()
    } else {
        input.trim().to_string()
    };

    // We load the mnist dataset and have X train which is a 2D array with 60000 rows and 784 columns.
    // Each entry in the column represents a pixel value of the image. 
    // We also have y_train which is a 2D array with 60000 rows and 10 columns. Each row is one image and each column is the label of the image. So 0-9.
    let train_images_path = PathBuf::from(format!("{}/train-images.idx3-ubyte", path));
    let train_labels_path = PathBuf::from(format!("{}/train-labels.idx1-ubyte", path));
    let test_images_path = PathBuf::from(format!("{}/t10k-images.idx3-ubyte", path));
    let test_labels_path = PathBuf::from(format!("{}/t10k-labels.idx1-ubyte", path));
    
    let (x_train, y_train) = match load_mnist_data(train_images_path, train_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load training data: {}", e);
            return Err(e);
        }
    };

    let (x_test, y_test) = match load_mnist_data(test_images_path, test_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load test data: {}", e);
            return Err(e);
        }
    };

    println!("Successfully loaded training and test data from: '{}'", path);

    // We initialize the neural network with 784 input neurons, 64 hidden neurons and 10 output neurons.
    // The weights and biases are set at random.
    let mut neural_network = NeuralNetwork::new(784, 64, 10);

    println!("\nNeural network initialized succesfully with input size: {}, hidden size: {}, output size: {}", neural_network.input_size, neural_network.hidden_size, neural_network.output_size);
    
    let mut epochs: Option<usize> = None;
    input.clear();
    while epochs.is_none() {
        print!("For how many epochs do you wish to train the network? (positive integer): ");
        io::stdout().flush().unwrap();

        input.clear();
        io::stdin().read_line(&mut input).expect("Failed to read input");

        match input.trim().parse::<usize>() {
            Ok(n) if n > 0 => epochs = Some(n), 
            _ => println!("Invalid input. Please enter a valid positive integer."),
        }
    }

    let epochs = epochs.unwrap();
    let learning_rate = 0.01;

    println!("\nTraining neural network with {} epochs and learning rate of {}", epochs, learning_rate);

    neural_network.train(&x_train, &y_train, epochs, learning_rate);

    println!("\nTraining complete");

    // Calculate accuracy on the test dataset
    let test_accuracy = neural_network.accuracy(&x_test, &y_test);
    println!("\nTest accuracy: {:.2}%", test_accuracy);

    //let model_path = PathBuf::from(format!("./model.bin"));
    //neural_network.save(&model_path);
    
    // Add option to load and save models
    // Add user input to train the model, add options for epochs, learning rate and at auto mode which detects when the model has converged
    // Add option to predict on a single image after another model has been loaded
    // Maybe add github actions to run a test for accuracy?
    // Add option of choosing handwritten gray scale images and converting to array so it can be predicted

    Ok(())
}
