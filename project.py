import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
accuracies_ann = []
accuracies_bnn = []
accuracies_bnn_refused = []
skipped = []


###########################################################
### Classical Neural Network #############################
###########################################################
(X_train_, y_train_), (X_test_, y_test_) = tf.keras.datasets.mnist.load_data()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))   # input layer
model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))   # 1st hidden layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))   # output layer

for fraction in fractions:
    X_train = X_train_[range(int(X_train_.shape[0] * fraction))]
    y_train = y_train_[range(int(y_train_.shape[0] * fraction))]
    X_test = X_test_[range(int(X_test_.shape[0] * fraction))]
    y_test = y_test_[range(int(y_test_.shape[0] * fraction))]

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=None)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=None)
    accuracies_ann.append(accuracy*100)

print (accuracies_ann)



###########################################################
### Bayesian Neural Network ###############################
###########################################################
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

net = NN(28*28, 1024, 10)
log_softmax = nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()

def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}
    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()
    
    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

def guide(x_data, y_data):
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

def predict(x):
    sampled_models = [guide(None, None) for _ in range(10)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.argmax(mean, dim=1)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(100)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)

def test_batch(images, labels):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0

    for i in range(len(labels)):    
        all_digits_prob = []
        highted_something = False
        
        for j in range(len(classes)):
            highlight=False
            histo = []
            histo_exp = []    
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))
            
            prob = np.percentile(histo_exp, 50) #sampling median probability
            if(prob>0.2): #select if network thinks this sample is 20% chance of this being a label
                highlight = True #possibly an answer
            all_digits_prob.append(prob)

            if(highlight):            
                highted_something = True

        predicted = np.argmax(all_digits_prob)    
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                correct_predictions +=1.0
                
    return len(labels), correct_predictions, predicted_for_images 

for fraction in fractions:
    train_dataset = datasets.MNIST('mnist-data/', train=True, download=True, 
                                    transform=transforms.Compose([transforms.ToTensor(),]))
    num_samples_train = int(len(train_dataset) * fraction)
    train_loader = DataLoader(train_dataset, batch_size=128,
                                            sampler=SubsetRandomSampler(range(num_samples_train)))

    test_dataset = datasets.MNIST('mnist-data/', train=False,
                                transform=transforms.Compose([transforms.ToTensor(),]))
    num_samples_test = int(len(test_dataset) * fraction)
    test_loader = DataLoader(test_dataset, batch_size=128,
                                            sampler=SubsetRandomSampler(range(num_samples_test)))
    
    num_iterations = 5
    loss = 0

    for j in range(num_iterations):
        loss = 0
        for batch_id, data in enumerate(train_loader):
            # calculate the loss and take a gradient step
            loss += svi.step(data[0].view(-1,28*28), data[1])

    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        predicted = predict(images.view(-1,28*28))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracies_bnn.append(correct * 100 / total)

    correct = 0
    total = 0
    total_predicted_for = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels)
        total += total_minibatch
        correct += correct_minibatch
        total_predicted_for += predictions_minibatch

    skipped.append((total - total_predicted_for) / total_predicted_for)
    accuracies_bnn_refused.append(100 * correct / total_predicted_for)

# print(fractions)
# print(accuracies_ann)
# print(accuracies_bnn)
# print(skipped)
# print(accuracies_bnn_refused)

lines = {
    'Classical NN': (fractions, accuracies_ann),
    'Bayesian NN on all training data': (fractions, accuracies_bnn),
    'Bayesian NN when it refuses to train on some data': (fractions, accuracies_bnn_refused)
}
for label, (x, y) in lines.items():
    plt.plot(x, y, label = label)
plt.legend()
plt.xlabel('Percentage of Data Used')
plt.ylabel('Accuracy in %')
plt.savefig('accuracy.png')