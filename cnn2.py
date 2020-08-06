import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import tsneNN2
import matplotlib.pyplot as plt
import tsne_comp
import find_perplexity

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1))
        self.max_pooling1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pooling2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.max_pooling2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def get_logits(self, x):
        x = self.conv1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.max_pooling2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return x


# Create an instance of the model
model = CNN()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 2

# how to save self-defined model ???
# checkpoint_path = "model2/{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# save_model_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

# model.save_weights(checkpoint_path.format(epoch=5))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5, callbacks=[save_model_cb])
#
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(y_test)))


class tsneNN(Model):
    def __init__(self):
        super(tsneNN, self).__init__()
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(2, activation='relu')

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)

    def get_loss(self, logits, beta):
        hidden = self.d1(logits)
        output = self.d2(hidden)
        loss = tsneNN2.tsne_loss(logits, output, beta)
        return loss

    def get_grad(self, logits, beta):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(logits, beta)
            g = tape.gradient(L, self.trainable_variables)
        return g

    def network_learn(self, input_tensor, beta):
        g = self.get_grad(input_tensor, beta)
        tf.keras.optimizers.Adam().apply_gradients(zip(g, self.trainable_variables))


tsneNNModel = tsneNN()

# # search perplexity
# train_ds_ = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(5000)
# for images, labels in train_ds_:
#     logits = model.get_logits(images)
#     find_perplexity.test_tsneNN(logits.numpy(), labels.numpy(), "logits_find_perp")
#     break

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(2000)

for epoch in range(5):
    for images, labels in train_ds:
        logits = model.get_logits(images)
        beta = tsneNN2.seach_beta(logits.numpy(), 1e-5, 40.0)
        tsneNNModel.network_learn(logits, beta)


print("Run Y1 = tsneNN(X, no_dims=2, perplexity=40) to perform t-SNE on your dataset.")
print("Run Y2 = tsne(X, no_dims=2, initialdim = 1024, perplexity=40) to perform t-SNE on your dataset.")
print("Running example on 5000 MNIST digits...")

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)

for test_images, test_labels in test_ds:
    ## NN result
    logits = model.get_logits(test_images)
    Y = tsneNNModel.call(logits).numpy()
    plt.scatter(Y[:, 0], Y[:, 1], test_labels)
    plt.savefig("result//NN1.png")
    plt.show()

    # t-SNE result, compare
    plt.clf()
    Y = tsne_comp.tsne(logits.numpy(), 2, 1024, 40.0)
    plt.scatter(Y[:, 0], Y[:, 1], test_labels)
    plt.savefig("result//tSNE1.png")
    plt.show()
    break


