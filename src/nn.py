import numpy as np

class FullyConnectedNN(object):

    def __init__(self, layers=(2, 3, 1), batch_size=10, epochs=20,
                    learning_rate=0.001):
        weights = []
        for layer, next_layer in zip(layers, layers[1:]):
            weights.append(np.random.rand(next_layer, layer))
        
        self.weights = weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


    def forward_batch(self, batch):
        layer_zs = []
        batch = np.transpose(batch)
        for weight in self.weights:
            print("multiplying ", batch, " with", weight)
            z = np.dot(weight, batch)
            layer_zs.append(z)
            batch = self.relu(z)
            print("result ", batch)
            
        return (batch, layer_zs)
            
            
    def compute_gradient(self, loss): 
        gradient = []
        for i, hidden_layer in enumerate(weights):
            if i == len(weights) - 1:
                pass
                # output layer weight gradients
                # how much does total loss change with respect to o1
                #d_o1 = 
            #else:
                # hidden layer weight gradients
                
            
    def update_weights(self, gradient):
        return self.weights - self.learning_rate * gradient        

    def relu(self, sums):
        return np.clip(sums, 0, None)

    # relu derivative
    def d_relu(val):
        if val > 0:
            return 1
        else:
            return 0

    def compute_loss(self, actual, output):
        # squared euclidean
        print("Output: ", output)
        print("Actual: ", actual)
        return np.sum((actual-output)**2, axis=1)

    def prepare_inputs(self, inputs):
        return np.array([sample.reshape((len(sample), 1)) for sample in inputs])

    def train(self, inputs, targets):
        training_losses = []
        validation_losses = []

        batch_size = min(self.batch_size, len(inputs))

        for i in range(self.epochs):
            print("Epoch " , i, "/", self.epochs)
            loss = compute_loss(targets, forward_batch(inputs))
            training_losses.append(loss)
            print("Loss ", loss)

            # random order
            ind = np.arange(0, len(inputs))
            np.random.shuffle(ind)
            
            # sampling mini-batches
            for i in range(0, len(ind), batch_size):
                # building batch using random indices
                training_indices = ind[i:i + batch_size]
                batch = inputs[training_indices]
                batch_labels = targets[training_indices]

                # forwarding batch through network
                outputs, layer_zs = self.forward_batch(batch)

                # compute the loss of the current batch and weights
                loss = compute_loss(batch_labels, outputs)

                # propagate backwards, change weights
                gradient = nn.compute_gradient(loss, layer_zs)
                nn.update_weights(gradient)


        loss = compute_loss(targets, forward_batch(inputs))
        training_losses.append(loss)
        print("Final loss: ", loss)
        print("All Losses: ", training_losses)


if __name__ == "__main__":
    nn = FullyConnectedNN()

    #inp = nn.prepare_inputs(np.array([1,1], [2,2]))
    inp = np.array([[1,1], [2,2]])
    actuals = np.array([[[2]], [[3]]])
    print(inp)

    pred = nn.forward_batch(inp)[0]

    print("Prediction: ", pred)
    print("Loss: ", nn.compute_loss(actuals, pred))
