# PoPL-Milestone-2-G22
## Group Members:
1. Tejas Bajaj (2021A7PS2510G)
2. Nikhileswar Reddy Machammagari (2021A7PS2911G)
3. Ujjwal Kumar (2021A7PS2736G)
4. Shivam Agarwal (2021A7PS2909G)

## Necessary Dependencies to be installed:
numpy - 1.23.1
tensorflow - 2.10.1
matplotlib - 3.6.3
torch - 2.1.0
torchvision - 0.16.0
pyro-ppl - 0.2.1

<!-- pls also include instructions on how to compile and run the project and obtain any results.  -->

## 1) Problem statement (original statement):
Our Problem statement is Image Analysis/Identification System using Pyro (Probabilistic) vs any Python library (like Tensorflow) (ie statistic). Our major motive behind taking this study is to show the difference in results of models in the two different paradigms, one of which had the capability of saying NO as an answer if it wasn't trained on that specific data and other (TensorFlow) will give a random output for such an input. Also Pyro model has an additional major benefit of giving accurate predictions even for samples of small data size.

POPL Angle : Pyro is a probabilistic programming library built on top of PyTorch, designed to facilitate the implementation of probabilistic models and conduct Bayesian inference. It provides tools and abstractions for implementing probabilistic models in the context of the PyTorch programming paradigm. We wanted to compare its performance with another paradigm Tensorflow: which is an open-source machine learning framework. The core paradigm of TensorFlow revolves around building and training computational graphs for machine learning models. So using pyro we built a system for image identification using Bayesian/Probabilistic model and using TensorFlow we showed it using a classical Neural Network.

The solution was existing before but we took comparitive study of how it would be different on being implemented in different paradigms. We started with the pre-existant solution for pyro and built the Tensorflow solution from scratch.  We also tweaked no. of layers used in the neural networks and saw its effect on the accuracy of prediction of the image. We also tried to use multiple different data sets to sample from them and used it to see the differences it produced in the results and showcase the advantage it has with pyro.

## 2) Software architecture: 
The software architecture of the solution involves building an Image Analysis/Identification System using two different paradigms: a Bayesian/Probabilistic model using Pyro (built on PyTorch) and a classical Neural Network using TensorFlow. The goal is to compare the performance of the models implemented in these two different paradigms.

Architecture Overview:
   -The Pyro implementation follows a probabilistic programming paradigm, where models are defined as probabilistic graphical models, and inference is performed using Pyro's probabilistic programming tools.
   -The TensorFlow implementation follows a traditional neural network paradigm, where a feedforward neural network is defined using the Keras API, and training is performed using the TensorFlow backend.

Parts Reused and Developed:
   -Reused Parts:
     -Both implementations reuse common machine learning and deep learning libraries, such as PyTorch for Pyro and TensorFlow for the TensorFlow implementation.
     -The datasets (MNIST) used for training and testing are standard datasets commonly used in machine learning.

   -Developed Parts:
     -The Pyro implementation involves the definition of a probabilistic model using Pyro's syntax, including the model and guide functions. We have altered the number of  neural networks used in the models to make it computationally less intensive. 
     -We built the TensorFlow implementation from scratch in the same structure of Neural networks as pyro code. We sampled from different datasets to show the benefit of pyro that it works well even in small dataset.

Client-Server Architecture:
   -There is no explicit indication of a client-server architecture. Both Pyro and TensorFlow implementations are standalone scripts that define, train, and evaluate models locally.

Testing Component:
   -The testing components are part of the same scripts. For example, the accuracy of predictions is evaluated on the test data within the same script after training.

Database Involvement:
   -There is no database being involved in the code developed. The models are trained and evaluated using in-memory data from the MNIST dataset.

Visualization:
   -The code snippets include visualizations using matplotlib to display loss during training and confusion matrices for evaluation.

## 3) POPL Aspects:
Probabilistic Programming with Pyro:
   -Code Reference: Lines 62-84 (Pyro Model Definition), Lines 86-112 (Pyro Guide Definition)
   -Explanation: Pyro is used for implementing a Bayesian Neural Network (BNN). The probabilistic programming paradigm is evident in the definition of priors for the model parameters (weights and biases) and the guide function, which represents the approximate posterior distribution.

Stochastic Variational Inference (SVI) for Training:
   -Code Reference: Lines 114-116 (SVI Initialization), Lines 121-125 (SVI Training Loop)
   -Explanation: SVI is used for approximate Bayesian inference. The loss is calculated using the ELBO (Evidence Lower Bound), and the model is trained using the Adam optimizer.

Uncertainty Estimation in Bayesian Neural Network:
   -Code Reference: Lines 126-145 (Function give_uncertainities())
   -Explanation: The BNN is not just used for predictions but also for estimating uncertainties. The function give_uncertainities() samples from the posterior distribution to obtain uncertainties associated with each prediction.

Model Refusal in Bayesian Neural Network:
   -Code Reference: Lines 147-176 (Function test_batch())
   -Explanation: The BNN is designed to refuse predictions when the network is uncertain about the input. The model is considered to refuse if none of the digit probabilities exceeds a certain threshold (20% chance in this case).

Classical Neural Network Implementation:
   -Code Reference: Lines 34-41 (Classical Neural Network Model Definition)
   -Explanation: The classical neural network (ANN) is implemented using TensorFlow for comparison. This represents a different paradigm where the focus is on deterministic, non-probabilistic models.

Dataset Subset Creation for Experimentation:
   -Code Reference: Lines 43-59 (Dataset Subsetting)
   -Explanation: Different fractions of the MNIST dataset are used for training and testing to analyze the impact of varying dataset sizes on the performance of both the classical and Bayesian neural networks.

Comparison Metrics and Visualization:
   -Code Reference: Lines 82, 144, 175-180 (Accuracy Calculation, Refusal Statistics)
   -Explanation: Metrics like accuracy for both classical and Bayesian models, and the statistics on model refusal, are calculated to provide a quantitative basis for comparing the two paradigms. Visualization using Matplotlib is employed to present the results in a clear and interpretable manner.

Adaptive Learning Rates with Adam Optimizer:
   -Code Reference: Lines 182-183 (Adam Optimizer Configuration)
   -Explanation: Adam optimizer is used for both the classical and Bayesian models. The learning rate is set to 0.01, showing an understanding of the importance of adaptive optimization techniques in training neural networks.

Evaluation of Both Paradigms under Different Conditions:
   -Code Reference: Lines 190-213 (Training and Evaluation Loop)
   -Explanation: The project involves systematic evaluation under different conditions, such as varying dataset sizes. This aligns with the idea of experimental design and thorough analysis required in POPL projects.

Utilizing PyTorch for Bayesian Neural Network Implementation:
    -Code Reference: Lines 55-65 (Bayesian Neural Network Model Definition)
    -Explanation: The choice of PyTorch for the Bayesian Neural Network implementation demonstrates a familiarity with multiple frameworks, aligning with the POPL principle of choosing the right tool for the job.

Experience of Difficulties:
-Implementing a Bayesian Neural Network with Pyro involves a steeper learning curve due to the probabilistic programming concepts and the need to define priors and approximate posteriors.
-Ensuring that the refusal mechanism works effectively and aligns with the probabilistic nature of the Bayesian model required careful consideration and debugging.
-Coordinating and synchronizing the dataset subsets for both TensorFlow and Pyro implementations, ensuring a fair comparison, posed challenges.
-Balancing the trade-off between model complexity and available computational resources was crucial, especially when dealing with Bayesian models that might demand more resources.

## 4) Results:
Tests Conducted:
   -Both Pyro and TensorFlow implementations underwent training and testing phases using the MNIST dataset for handwritten digit recognition.
   -Two types of predictions were tested: one where the network is forced to predict, and another where the network can refuse predictions.

Dataset Used:
   -MNIST dataset was used for training and testing. MNIST is a standard dataset for digit recognition and consists of 28x28 pixel grayscale images of handwritten digits (0-9).

Benchmarks Run:
   -Training was performed for a specified number of epochs (10 in the TensorFlow case) with batch sizes of 128.
   -The Pyro implementation used the SVI (Stochastic Variational Inference) method with the ELBO (Evidence Lower Bound) loss.
   -TensorFlow used the Adam optimizer and sparse categorical cross-entropy loss.

Graphs Shown:
   -The project includes visualizations such as loss during training, confusion matrices, and accuracy metrics.
   -Matplotlib is used for plotting the loss during training in the Pyro implementation.

Checking/Validating Results Alignment with Initial Problem Statement:
   -The problem statement focuses on comparing the performance of Pyro's probabilistic programming paradigm with TensorFlow's deep learning paradigm for image analysis.
   -The comparison is done through metrics such as accuracy, confusion matrices, and loss values during training.
   -The "Prediction when the network can refuse" scenario is specifically designed to showcase the difference in behavior between the probabilistic and deep learning paradigms.

Data-Driven Proof Points:
   -The accuracy metric is calculated for both implementations on the MNIST test dataset, providing a quantitative measure of performance.
   -Confusion matrices are shown, offering insights into the precision and recall of the models for each digit class.
   -The use of multiple datasets for training and the exploration of different neural network architectures serves as a data-driven exploration of the model's behavior.

Why should you be convinced with it is working?
   -The code outputs accuracy metrics, confusion matrices, and loss values, providing a detailed overview of the model's performance.
   -The clear comparison between Pyro's probabilistic programming and TensorFlow's deep learning, with specific scenarios like network refusal, highlights the distinct behaviors of the two paradigms.
   -The project adheres to the initial problem statement, demonstrating a well-planned approach to comparing different programming paradigms for image analysis.

## 5) Potential for future work:
Given more time, there are several potential avenues for future work and additional POPL aspects to explore:

1. Advanced Bayesian Neural Network Architectures:
   - Future Work: Experiment with more complex Bayesian Neural Network architectures. This could include hierarchical models, variational autoencoders, or ensembling techniques for improved uncertainty estimation.
   - POPL Aspect: Explore different probabilistic programming constructs and model structures to understand their impact on both performance and interpretability.

2. Model Refusal Mechanism:
   - Future Work: Refine and extend the model refusal mechanism. This could involve optimizing the threshold for refusal dynamically based on the uncertainty estimates. Additionally, exploring alternative refusal strategies beyond a simple probability threshold.
   - POPL Aspect: Investigate strategies for incorporating domain-specific knowledge into the refusal mechanism and optimizing the trade-off between model accuracy and refusal rate.

3. Benchmarking on Diverse Datasets:
   - Future Work: Extend the evaluation to other datasets beyond MNIST, especially datasets with different characteristics (e.g., CIFAR-10, ImageNet). Assess how well the Bayesian and classical models generalize across diverse data distributions.
   - POPL Aspect: Consider the generalizability of the chosen programming paradigm and model to different problem domains, emphasizing the paradigm's flexibility and applicability.

4. Hyperparameter Tuning and Sensitivity Analysis:
   - Future Work: Conduct a thorough hyperparameter tuning and sensitivity analysis for both classical and Bayesian models. This involves systematically varying hyperparameters to understand their impact on model performance.
   - POPL Aspect: Highlight the importance of choosing appropriate hyperparameters in both paradigms and their impact on the robustness and reliability of the models.

5. Probabilistic Programming Language Comparison:
   - Future Work: Extend the comparison to other probabilistic programming languages beyond Pyro. This could involve implementing the Bayesian Neural Network in languages like Stan or Edward and comparing their expressiveness and performance.
   - POPL Aspect: Assess the strengths and weaknesses of different probabilistic programming languages in the context of Bayesian modeling, emphasizing the importance of language choice in POPL.

6. Parallelization and Scalability:
   - Future Work: Investigate strategies for parallelizing the training and inference process, especially for Bayesian models that might benefit from parallel sampling. Assess the scalability of the models concerning larger datasets.
   - POPL Aspect: Explore how the chosen paradigm handles parallelization and scalability, considering the efficiency and resource utilization.

7. Interactive Probabilistic Programming:
   - Future Work: Develop interactive tools or interfaces for probabilistic programming. This could involve creating visualizations to aid users in understanding and manipulating probabilistic models.
   - POPL Aspect: Address the usability and human-computer interaction aspects of probabilistic programming, making the paradigm more accessible to a broader audience.

8. Incorporating Domain-Specific Constraints:
   - Future Work: Integrate domain-specific constraints into the Bayesian model. This might involve incorporating prior knowledge about the relationships between features or constraining certain parameters based on external information.
   - POPL Aspect: Explore how the chosen probabilistic programming paradigm facilitates the incorporation of domain-specific constraints and knowledge into the model.

9. Probabilistic Programming for Program Synthesis:
   - Future Work: Extend the project to explore the application of probabilistic programming in the domain of program synthesis. This could involve generating programs that exhibit probabilistic behavior or uncertainty.
   - POPL Aspect: Investigate how probabilistic programming can be leveraged for program synthesis, emphasizing the expressiveness and conciseness of probabilistic specifications.

10. Comparative Study with Different Classical Models:
    - Future Work: Expand the comparative study by implementing different classical models (e.g., convolutional neural networks, recurrent neural networks) and assessing their performance against the Bayesian model.
    - POPL Aspect: Emphasize the flexibility and versatility of the chosen paradigm compared to classical models in addressing various machine learning tasks.

In summary, future work could involve deepening the exploration of Bayesian modeling, refining model components, extending evaluations, and considering broader aspects of probabilistic programming in the context of specific problem domains.
