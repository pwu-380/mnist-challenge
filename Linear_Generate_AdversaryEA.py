# Test script, to generate adversarial images against a previously trained linear classifier model

# I base my code on these two papers:
# http://www.evolvingai.org/files/DNNsEasilyFooled_cvpr15.pdf
# http://dl.acm.org/citation.cfm?id=2955178
# Instead of fast gradient sign it uses a genetic algorithm so it should be adaptable against all neural networks
# All code is my own, did not copy anything


import random as rand
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # This is the MNIST data

MUTATE_P = 0.1                              # Mutation probability
MUTATE_N = int(MUTATE_P * 784)              # Number of elements to mutate
A_RANGE = range(784)                        # Used to select mutation and crossover points
CROSSOVER_P = 0.6                           # Crossover probability
POPULATION_N = 50                           # Population size
POPULATION_R = range(POPULATION_N)
GENERATION_N = 1000                         # Number of generations
GENERATION_R = range(GENERATION_N)
TOURNAMENT_S = 3                            # Tournament sizes
GUASS_SCALE = 0.007                         # Determines magnitude of mutations (guassian standard deviation)
FITNESS_SENSITIVITY = 0.0000005             # Used in fitness algorithm, multiplies the distance between 2 images

sess = tf.InteractiveSession()

# Note: the following are placeholders for a tensor that will be constantly fed
x = tf.placeholder(tf.float32, shape=[None, 784])   # Feature tensor = 28x28

W = tf.Variable(tf.zeros([784, 10]))                # Weights
b = tf.Variable(tf.zeros([10]))                     # Biases

sess.run(tf.global_variables_initializer())         # Initializes TensorFlow variables and model saver
saver = tf.train.Saver()

batch = mnist.test.next_batch(1)                    # I noticed the second MNIST image is a 2 so I just hard coded it
batch = mnist.test.next_batch(1)
two = batch[0]                                      # two is our test image
population = np.zeros((POPULATION_N, 784))
for i in POPULATION_R:                              # Seeding the initial image population as the target image (two)
    population[i] = two

test = population[1]                                # Test sample from population

# Print the image
test.shape = (28, 28)
plt.imshow(test, cmap='gray')
plt.savefig("fig1.png")

children = np.zeros((POPULATION_N, 784))            # Empty child population


# Mutation function (with some probability, each pixel mutates in an amount given by guassian distribution)
def generate_mutation(chromosome):
    change_list = rand.sample(A_RANGE, MUTATE_N)

    for i in change_list:
        chromosome[i] += np.random.normal(scale=GUASS_SCALE)    # The results are somewhat sensitive to how
                                                                # the gaussian is scaled
    chromosome[chromosome < 0] = 0
    chromosome[chromosome > 1] = 1

    return chromosome


# Crossover function (two point crossover)
def crossover(chromosome1, chromosome2):
    crossover_points = rand.sample(A_RANGE, 2)

    temp = chromosome2[crossover_points[0]:crossover_points[1]]
    chromosome2[crossover_points[0]:crossover_points[1]] = chromosome1[crossover_points[0]:crossover_points[1]]
    chromosome1[crossover_points[0]:crossover_points[1]] = temp

    return chromosome1, chromosome2


# Tournament function (winner moves on with 100% probability)
def tournament(images, target_image, scores, target_score, step):
    fitness_value = np.zeros(TOURNAMENT_S)
    for i in range(TOURNAMENT_S):
        fitness_value[i] = fitness(images[i], target_image, scores[i], target_score, step)

    winner = images[np.argmax(fitness_value)]

    return winner


# Fitness function
def fitness(image, target_image, score, target_score, step):
    fitness_value = -(FITNESS_SENSITIVITY * np.linalg.norm(image - target_image) + (target_score-score))
    # The above equation is actually super sensitive to the first part, maybe worth reformulating somehow

    # If you zero out the second part, all the mutations are suppressed and zeroing out the first part causes
    # fast convergence to >99% certainty. But the balance needs to be tuned.

    return fitness_value


# Model from Tutorial 1
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Restore variables
saver.restore(sess, 'linear_model/linear_model.ckpt')

# Prints the initial classification
result = sess.run(y, feed_dict={x: batch[0]})
print("Model out:", result)
print("Actual:", batch[1])
two.shape = (784)

# Main algorithm
for i in GENERATION_R:
    scores = sess.run(y, feed_dict={x: population})     # Score the entire population = target class % confidence

    # In each generation, we hold population/2 number of tournaments, 2 at a time, each with a randomly selected
    # sample from the population (with replacement). 60% of the time, the tournament winners crossbreed and their
    # children move on to the next generation, 40% of the time the tournament winners move on the next generation.
    # Every individual that moves on is mutated.
    for j in range(POPULATION_N/2):
        selection = rand.sample(POPULATION_R, TOURNAMENT_S)
        parent1 = tournament([population[selection[0]], population[selection[1]], population[selection[2]]], two,
                             [scores[selection[0], 6], scores[selection[1], 6], scores[selection[2], 6]], 1, i)
        selection = rand.sample(POPULATION_R, TOURNAMENT_S)
        parent2 = tournament([population[selection[0]], population[selection[1]], population[selection[2]]], two,
                             [scores[selection[0], 6], scores[selection[1], 6], scores[selection[2], 6]], 1, i)

        if np.random.rand() < CROSSOVER_P:
            child1, child2 = crossover(parent1, parent2)
            children[j*2] = generate_mutation(child1)
            children[j*2+1] = generate_mutation(child2)
        else:
            children[j*2] = generate_mutation(parent1)
            children[j*2+1] = generate_mutation(parent2)

    population = children

# Take a sample from out final population, (I just took the first since they all should have converged to be similar)
# and checks the classification
test = population[1]
test.shape = (1, 784)
result = sess.run(y, feed_dict={x: test})
print ("New: ", result)

# Print image
test.shape = (28, 28)
plt.imshow(test, cmap='gray')
plt.savefig("fig2.png")

