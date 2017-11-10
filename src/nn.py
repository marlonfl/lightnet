import numpy as np

class FullyConnectedNN(object):

    def __init__(self, layers=(2, 3, 3, 1), batch_size=10, epochs=20,
                    learning_rate=0.0001):
        weights = []
        for layer, next_layer in zip(layers, layers[1:]):
            weights.append(np.random.rand(next_layer, layer))

        self.weights = weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, values):
        outp = values[:]
        for weight in self.weights:
            outp = self.relu(np.dot(weight, outp))

        return values

    def forward_batch(self, batch):
        outputs = []
        for sample in batch:
            outp = sample[:]
            for weight in self.weights:
                outp = self.relu(np.dot(weight, outp))
            outputs.append(outp)

        return np.array(outputs)

    def backward(self):
        pass

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
        return np.sum((actual-output)**2)

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
                outputs = self.forward_batch(batch)

                # compute the loss of the current batch and weights
                loss = compute_loss(batch_labels, outputs)

                # propagate backwards, change weights
                # todo
                # todo

        loss = compute_loss(targets, forward_batch(inputs))
        training_losses.append(loss)
        print("Final loss: ", loss)
        print("All Losses: ", training_losses)


if __name__ == "__main__":
    nn = FullyConnectedNN()

    inp = nn.prepare_inputs(np.array([[-1,1], [2,2]]))
    actuals = np.array([[[2]], [[3]]])

    pred = nn.forward_batch(inp)

    print("Prediction: ", pred)
    print("Loss: ", nn.compute_loss(actuals, pred))
