import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    plt.plot([-0.1], [-0.5], marker='x', markersize=5, color='red') 
    A0, A1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    a0 = A0[0].reshape(100, 1)
    gauss_contour = []

    for i in range(0, 100):
      samples = np.concatenate((a0, A1[i].reshape(100, 1)), 1)
      gauss_contour.append(util.density_Gaussian([0, 0], [[beta, 0], [0, beta]], samples))
    
    plt.title('a0 and a1: Prior Distribution')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.contour(A0, A1, gauss_contour, colors='black')
    plt.savefig("prior.pdf")
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    covariance_a_inv = [[1/beta, 0], [0, 1/beta]]
    A = np.append(np.ones(shape=(len(x), 1)), x, axis=1)
    covariance_w_inv = 1/sigma2
    
    mu = np.linalg.inv((np.dot(A.T, A) * covariance_w_inv + covariance_a_inv)) @ (np.dot(A.T, z) * covariance_w_inv)
    mu = mu.reshape(2, 1).squeeze()
    Cov = np.linalg.inv(covariance_a_inv + np.dot(A.T, np.dot(A, covariance_w_inv)))

    A0, A1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

    a0 = A0[0].reshape(100, 1)
    post_contour = []

    for i in range(0, 100):
      samples = np.concatenate((a0, A1[i].reshape(100, 1)), 1)
      post_contour.append(util.density_Gaussian(mu.T, Cov, samples))
    
    plt.plot([-0.1], [-0.5], marker='x', markersize=5, color='red')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.contour(A0, A1, post_contour, colors='black')
    if len(x) == 100:
        plt.title('100 Data Samples: Posterior Distribution')
        plt.savefig("posterior100.pdf")
    elif len(x) == 5:
        plt.title('5 Data Samples: Posterior Distribution')
        plt.savefig("posterior5.pdf")
    elif len(x) == 1:
        plt.title('1 Data Sample: Posterior Distribution')
        plt.savefig("posterior1.pdf")
    plt.show()
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    A = np.append(np.ones(shape=(len(x), 1)), np.expand_dims(x, 1), axis=1)
    covariance_w = sigma2

    mean_z = np.dot(A, mu)
    covariance_z = covariance_w + np.dot(A, np.dot(Cov, A.T))
    standard_deviation_z = np.sqrt(np.diag(covariance_z))
    
    print("The mean of z is: ", mean_z)
    print("The standard deviation of z is: ", standard_deviation_z)
    print("The covariance of z is: ", covariance_z)
    
    plt.ylabel('z')
    plt.ylim([-4, 4])
    
    plt.xlabel('x')
    plt.xlim([-4, 4])

    plt.errorbar(x, mean_z, yerr=standard_deviation_z, fmt='black')
    plt.scatter(x_train, z_train, color = 'red')

    if len(x_train) == 100:
        plt.title('100 Data Samples: Prediction')
        plt.savefig("predict100.pdf")
    elif len(x_train) == 5:
        plt.title('5 Data Samples: Prediction')
        plt.savefig("predict5.pdf")
    elif len(x_train) == 1:
        plt.title('1 Data Sample: Prediction')
        plt.savefig("predict1.pdf")
    
    plt.show()
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('reg/training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # ------------------- 1 TRAINING SAMPLE -------------------
    # number of training samples used to compute posterior
    ns  = 1
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    # ------------------- 5 TRAINING SAMPLES -------------------
    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)

    # ------------------- 100 TRAINING SAMPLES -------------------
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    