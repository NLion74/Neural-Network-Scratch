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

fn get_user_input(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush();

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");
    input.trim().to_string()
}

fn main() -> Result<(), io::Error> {
    println!("Welcome to the MNIST Neural-Network-Scratch!");
    println!("Note: For optimal performance, please compile the project in release mode, as it may run slowly in debug mode.");

    // User prompt to provide dataset path
    let mut input = String::new();
    input = get_user_input("Enter the path to the MNIST dataset (or press Enter for the default './dataset'): ");

    let path = if input.is_empty() {
        "./dataset".to_string()
    } else {
        input.to_string()
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
            eprintln!("Error loading training data: {}", e);
            return Err(e);
        }
    };
    let (x_test, y_test) = match load_mnist_data(test_images_path, test_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading test data: {}", e);
            return Err(e);
        }
    };

    println!("Successfully loaded training and test data from: '{}'", path);

    // User prompt for initializing or loading network
    let mut neural_network: NeuralNetwork;
    loop {
        input = get_user_input("\nPress Enter to initialize a new network or provide a file path to load an existing one: ");

        if input.is_empty() {
            let input_size = 784;
            let hidden_size = 64;
            let output_size = 10;
            neural_network = NeuralNetwork::new(input_size, hidden_size, output_size);

            println!("Neural network initialized successfully with input size: {}, hidden size: {}, output size: {}.", 
                neural_network.input_size, neural_network.hidden_size, neural_network.output_size);
            println!("Initial accuracy on test set: {}", neural_network.accuracy(&x_test, &y_test));
            break;
        } else {
            let model_path = PathBuf::from(input);
            neural_network = match NeuralNetwork::load(&model_path) {
                Ok(nn) => nn,
                    Err(e) => {
                        eprintln!("Failed to load the neural network from '{}': {}", model_path.display(), e);
                        continue;
                    }
            };
            println!("Neural network loaded successfully from '{}'.", model_path.display());
            println!("Input size: {}, hidden size: {}, output size: {}.", 
                neural_network.input_size, neural_network.hidden_size, neural_network.output_size);
            println!("Accuracy on test set: {}", neural_network.accuracy(&x_test, &y_test));
            break;
        }
    }

    // User prompt for training the network
    loop {
        input = get_user_input("\nPress Enter to skip training or enter the number of epochs to train (positive integer): ");

        if input.is_empty() {
            println!("Training skipped.");
            break;
        }
    
        match input.parse::<usize>() {
            Ok(epochs) if epochs > 0 => {
                let learning_rate = 0.01;
                println!("Training neural network with {} epochs and learning rate of {}...", epochs, learning_rate);
                neural_network.train(&x_train, &y_train, epochs, learning_rate);
                println!("Training complete! Final accuracy on test set: {}", neural_network.accuracy(&x_test, &y_test));
                break;
            }
            _ => {
                println!("Invalid input. Please enter a valid positive integer.");
            }
        }
    }
    
    // User prompt for saving the model
    input = get_user_input("Press Enter to skip saving the model or provide a file path to save the model: ");

    if !input.is_empty() {
        let model_path = PathBuf::from(input);
        neural_network.save(&model_path).expect("Failed to save the neural network");
    
        println!("Neural network saved successfully to '{}'.", model_path.display());
    }

    // Future improvements comments
    // Add auto mode which detects when the model has converged
    // Maybe add github actions to run a test for accuracy?
    // Add option of choosing handwritten gray scale images and converting to array so it can be predicted

    Ok(())
}
