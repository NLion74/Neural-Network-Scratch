# Rust-Neural-Network
This project is a simple neural network implementation from scratch built in Rust, designed to work with the MNIST dataset of handwritten digits. The neural network achieves an accuracy of up to ~96.8% on the test set. The project was created to build a deeper understanding of neural networks.

A pre-trained model file, model.bin, is included in the repository.
For optimal performance, compile the project in release mode, as it may run slowly in debug mode.

# Getting started
```
git clone https://github.com/NLion74/Neural-Network-Scratch
cd Neural-Network-Scratch && cargo build --release
# Linux
chmod +x ./target/release/neural-network-scratch && ./target/release/neural-network-scratch
# Windows
.\target\release\neural-network-scratch.exe
```