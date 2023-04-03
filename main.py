import math
import matplotlib.pyplot as plt
import numpy as np

def main(): 
    # print("Generate an N x n matrix of samples from Bernoulli(p) ")
    # p = float(input("Enter bernoulli parameter:\n"))
    # N = int(input("Enter N:\n"))
    # n = int(input("Enter n:\n"))
    p = 0.5 
    N = 200000 
    n = 20 
    bernoulli_matrix = create_bernoullii_matrix(p,N,n) 
    result_mean = [np.mean(bernoulli_matrix[i]) for i in range(N)]
    x_epsilon_random, y_empirical_prob = empirical_prob(result_mean, 50)
    y_hoeffding_bound = [2*math.pow(math.e, (-2)*n*(math.pow(x_epsilon_random[i],2))) for i in range(len(x_epsilon_random))] 
    plot_care(x_epsilon_random, y_empirical_prob, y_hoeffding_bound) 


def empirical_prob(result_mean: list, num_of_epsilon: int ) -> tuple : 
    epsilon_random = np.linspace(0,1,num_of_epsilon)
    y_axis = [] 
    for j in range(num_of_epsilon) : 
        y = [1 if (abs(result_mean[i] - 0.5 ) > epsilon_random[j]) else 0  for i in range(len(result_mean))] 
        y_axis.append(np.mean(y)) 
    
    return epsilon_random, y_axis

def create_bernoullii_matrix(p: int,N:int ,n:int ) -> np.array:
    return np.random.binomial(1,p, size=(N,n))

def plot_care(x,y, hoeffding_bound) : 
    plt.plot(x, y, label="Empirical distribution")
    plt.plot(x, hoeffding_bound,  label="Hoeffding bound")
    plt.title("Hoeffding bound")
    plt.xlabel("epsilon")
    plt.ylabel("probability |X - 1/2| > epsilon")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()