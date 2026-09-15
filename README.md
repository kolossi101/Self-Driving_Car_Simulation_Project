# Self-Driving Car Simulation

A deep learning project that uses **behavioral cloning and a Convolutional Neural Network (CNN)** to teach a simulated car to drive autonomously in the **Udacity Self-Driving Car Simulator**.

The project focuses on collecting driving data, applying image augmentation to improve generalization, training an end-to-end CNN to predict steering angles, and sending real-time predictions back to the simulator.

## Project Overview

The overall pipeline is:

**Camera Images → Preprocessing & Augmentation → CNN → Steering Prediction → Real-Time Simulator Control**

### Key Features

* End-to-end CNN for autonomous steering
* Training data collected from center, left, and right simulated cameras
* Recovery training using steering offsets
* Real-time image augmentation
* Handling of straight-line bias and sharp turns
* Real-time communication between the trained model and simulator
* Autonomous driving through Socket.IO and Eventlet

## Technologies

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Convolutional Neural Networks (CNN)**
* **Socket.IO**
* **Eventlet**
* **Conda**
* **Udacity Self-Driving Car Simulator**

## ENVIRONMENT SETUP

To create the exact Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

If this command does not work because Conda is not recognized, install **Anaconda3** if you do not already have it.

If Anaconda is already installed, or you have just finished installing it, open the **Anaconda PowerShell Prompt** and run:

```bash
conda --version
```

This checks that Conda is installed and working.

Then run:

```bash
conda init powershell
```

Close and reopen the VS Code project.

To verify that Conda is available in VS Code, run:

```bash
conda --version
```

You should now be able to create the environment:

```bash
conda env create -f environment.yml
```

**Skip to the next step if `conda env create -f environment.yml` worked successfully.**

### Activate the Environment

```bash
conda activate car-cnn
```

# APPROACH

The solution relies on **behavioral cloning** to train a **Convolutional Neural Network (CNN)** to steer the car autonomously in the Udacity simulator.

## Data Collection and Processing

Images are collected from three simulated front-facing cameras — **center, left, and right** — together with steering-angle telemetry in real time.

Left and right camera images are used to simulate recovery scenarios by applying a calculated steering offset to the recorded steering angle. This provides the model with examples of how to return the car toward the center of the track when it moves away from the ideal driving position.

## CNN Architecture

An end-to-end regression architecture is used.

The model begins with a **Lambda layer** to normalize pixel values, followed by **five Conv2D convolutional layers with ReLU activations** to extract spatial features such as road and lane information.

The resulting feature maps are flattened and passed through a **Dropout layer** to reduce overfitting. Fully connected layers then progressively reduce the representation:

```text
100 → 50 → 10 → 1
```

The final output is a continuous **steering-angle prediction**.

## Real-Time Control Loop

The simulator connection is implemented in `TestSimulation.py` using Python **Socket.IO** and asynchronous event handling through **Eventlet**.

The simulator streams live camera frames to the application. The frames are passed through the trained model, which immediately predicts a steering angle and sends the appropriate steering and throttle commands back to the simulator.

# MAJOR CHALLENGES AND HOW WE ADDRESSED THEM

## Challenge 1 — Overfitting to a Specific Track and Lighting

Because the simulator tracks are static, the model had a high risk of memorizing background scenery instead of learning how to identify relevant road features.

### How It Was Addressed

A real-time image augmentation pipeline was implemented in `src/augmentations.py`.

Variations of the input images are created during training through a continuous batch generator before the images reach the neural network.

Augmentation includes:

* Random horizontal flips
* Artificial shadows
* Brightness variation
* Horizontal image translations

These transformations increase the variety of training examples and encourage the model to focus on important road features rather than fixed background elements.

## Challenge 2 — Sharp Turns and Straight-Line Bias

Normal driving contains many more examples of keeping the steering wheel near `0` than making sharp turns. This creates a strong bias toward driving straight and can cause the model to perform poorly on corners.

### How It Was Addressed

Recovery examples were created using the left and right camera images with steering offsets.

Dynamic horizontal translations were also introduced during augmentation to simulate the car being positioned closer to the edge of the road.

This teaches the network how to recover from off-center positions and navigate difficult sections of the track before reaching a failure state.

# PROJECT STRUCTURE

```text
Self-Driving_Car_Simulation_Project/
│
├── src/
│   ├── config.py
│   ├── augmentations.py
│   ├── data_generator.py
│   ├── dataset.py
│   ├── image_utils.py
│   └── model.py
│
├── train.py
├── TestSimulation.py
├── environment.yml
└── README.md
```

### Main Components

| File                    | Purpose                            |
| ----------------------- | ---------------------------------- |
| `src/config.py`         | Project and training configuration |
| `src/augmentations.py`  | Image augmentation pipeline        |
| `src/data_generator.py` | Batch generation for training      |
| `src/dataset.py`        | Dataset loading and preparation    |
| `src/image_utils.py`    | Image preprocessing utilities      |
| `src/model.py`          | CNN architecture                   |
| `train.py`              | Model training                     |
| `TestSimulation.py`     | Real-time simulator control        |

# TO RUN THE CODE

Make sure you are in the **project root folder**.

The following commands can be used to run the individual modules:

```bash
python -m src.config
python -m src.augmentations
python -m src.data_generator
python -m src.dataset
python -m src.image_utils
python -m src.model
```

Then train the model:

```bash
python train.py
```

After training, run the simulator interface:

```bash
python TestSimulation.py
```

### Model Weight Issue

If `TestSimulation.py` does not run, the model specified in the script may not exist.

For example:

```python
model.load_weights('model-012.h5')
```

If `model-012.h5` does not exist, change the filename to the model with the **highest numeric value** among the available trained model files.

# TO SEE THE CODE RUN IN THE UDACITY CAR SIMULATOR

After running:

```bash
python TestSimulation.py
```

1. Open `beta_simulator.exe`.
2. In the popup window, click **Play!**
3. Once the main Udacity Car Driving Simulator screen appears, select **Autonomous Mode**.
4. The trained model should begin controlling the car automatically.

The car should now drive autonomously using the steering predictions generated by the CNN.

# RESULTS & DEMO

The trained CNN is capable of performing **real-time steering prediction and autonomous navigation** within the Udacity simulator.

### 🎥 Demo

[Watch the Self-Driving Car Simulation Demo](https://youtu.be/aMdI5PAKO64)

# WHAT I LEARNED

This project provided hands-on experience with:

* Designing and training CNN-based regression models
* Computer vision and image preprocessing
* Data augmentation and dataset balancing
* Handling model overfitting and generalization
* Real-time machine learning inference
* Socket-based communication
* Integrating a machine learning model with a simulation environment

## About

Built as a hands-on exploration of **deep learning, computer vision, and autonomous vehicle systems**, with an emphasis on connecting a trained neural network to a real-time simulation environment.
