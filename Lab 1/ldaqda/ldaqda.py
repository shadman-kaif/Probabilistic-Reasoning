import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    # Create count variables for male and female
    male_count, female_count = 0, 0

    # Create mean and covariance arrays for male and female
    mu_male, mu_female = np.array([0, 0]), np.array([0, 0])
    cov, cov_male, cov_female = np.array([[0,0], [0,0]]), np.array([[0,0], [0,0]]), np.array([[0,0], [0,0]])

    for i in range(x.shape[0]):
        if y[i] != 1:
            mu_female = mu_female + x[i, :]
            female_count = female_count + 1
        else:
            mu_male = mu_male + x[i, :]
            male_count = male_count + 1

    mu_female = np.divide(mu_female, female_count)
    mu_male = np.divide(mu_male, male_count)

    for i in range(x.shape[0]):
        data = np.array(x[i,:])
        if y[i] == 1:
            data = data - mu_male
            cov_male = np.add(cov_male, np.outer(data, data))
        else:
            data = data - mu_female
            cov_female = np.add(cov_female, np.outer(data, data))
            
        cov = np.add(cov, np.outer(data, data))

    cov_female = np.divide(cov_female, female_count)
    cov_male = np.divide(cov_male, male_count)
    cov = np.divide(cov, male_count+female_count)
    
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here

    lda_right_pred, lda_incorrect_pred = 0, 0
    lda_male = util.density_Gaussian(mu_male, cov, x)
    lda_female = util.density_Gaussian(mu_female, cov, x)
    
    lda_male_pred = lda_male > lda_female
    
    for i in range(lda_male_pred.shape[0]):
        if (lda_male_pred[i] and y[i]==1) or (not lda_male_pred[i] and y[i]==2):
            lda_right_pred = lda_right_pred + 1
        else:
            lda_incorrect_pred = lda_incorrect_pred + 1

    assert(lda_right_pred + lda_incorrect_pred == x.shape[0])

    mis_lda = lda_incorrect_pred / x.shape[0]

    qda_right_pred, qda_incorrect_pred = 0, 0
    qda_male = util.density_Gaussian(mu_male, cov_male, x)
    qda_female = util.density_Gaussian(mu_female, cov_female, x)

    qda_male_pred = qda_male > qda_female
    
    for i in range(qda_male_pred.shape[0]):
        if (qda_male_pred[i] and y[i] == 1) or (not qda_male_pred[i] and y[i] == 2):
            qda_right_pred = qda_right_pred + 1
        else:
            qda_incorrect_pred = qda_incorrect_pred + 1

    assert(qda_right_pred + qda_incorrect_pred == x.shape[0])

    mis_qda = qda_incorrect_pred / x.shape[0]
    
    return (mis_lda, mis_qda)

if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('ldaqda/trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('ldaqda/testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)

    # Print out the LDA and QDA misclassification rates
    print("The LDA misclassification rate is: ", mis_LDA)
    print("The QDA misclassification rate is: ", mis_QDA)

    # Calculate the m and b for the plot
    b = -0.5 * np.dot(np.dot(mu_male, np.linalg.inv(cov)), mu_male.T) + 0.5  *np.dot(np.dot(mu_female, np.linalg.inv(cov)), mu_female.T)
    m = np.dot(mu_male, np.linalg.inv(cov)) - np.dot(mu_female, np.linalg.inv(cov))
    
    weights = np.linspace(80, 280, 100)   
    heights = np.linspace(50, 80, 100)   
    H, W = np.meshgrid(heights, weights)

    male_lda, male_qda, female_lda, female_qda = [], [], [], []

    for i in weights:
        male_lda_tmp, male_qda_tmp, female_lda_tmp, female_qda_tmp = [], [], [], []
        for j in heights:
            male_qda_tmp.extend(util.density_Gaussian(mu_male, cov_male, np.array([[j, i]])))
            male_lda_tmp.extend(util.density_Gaussian(mu_male, cov, np.array([[j, i]])))
            female_qda_tmp.extend(util.density_Gaussian(mu_female, cov_female, np.array([[j, i]])))
            female_lda_tmp.extend(util.density_Gaussian(mu_female, cov, np.array([[j, i]])))
        
        male_qda.append(male_qda_tmp[:])
        male_lda.append(male_lda_tmp[:])
        female_qda.append(female_qda_tmp[:])
        female_lda.append(female_lda_tmp[:])

    # LDA Plot
    for i in range(x_train.shape[0]):
        if y_train[i] != 1:
            col = 'red' 
        else:
            col = 'blue'
        plt.scatter(x_train[i, 0], x_train[i, 1], s=10, c = col, linewidth = 0)  

    plt.contour(H, W, female_lda, colors = 'red')
    plt.contour(H, W, male_lda, colors = 'blue')

    lda_decision = np.asarray(male_lda) - np.asarray(female_lda)
    plt.contour(H, W, lda_decision, 0, colors = 'black')
    
    h1, l1 = plt.contour(H, W, female_lda, colors = 'red').legend_elements()
    h2, l2 = plt.contour(H, W, male_lda, colors = 'blue').legend_elements()
    h3, l3 = plt.contour(H, W, lda_decision, 0, colors = 'black').legend_elements()

    plt.title('Weight vs Height for LDA')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend([h1[0], h2[0], h3[0]], ['Female LDA', 'Male LDA', 'Decision Boundary'])
    plt.savefig("lda.pdf")
    plt.show()

    # QDA Plot
    for i in range(x_train.shape[0]):
        if y_train[i] != 1:
            col = 'red'
        else:
            col = 'blue'
        plt.scatter(x_train[i, 0], x_train[i, 1], s = 10, c = col, linewidth = 0)  

    plt.contour(H, W, female_qda, colors='red')
    plt.contour(H, W, male_qda, colors='blue')

    qda_decision = np.asarray(male_qda) - np.asarray(female_qda)
    plt.contour(H, W, qda_decision, 0, colors = 'black')

    h1, l1 = plt.contour(H, W, female_qda, colors = 'red').legend_elements()
    h2, l2 = plt.contour(H, W, male_qda, colors = 'blue').legend_elements()
    h3, l3 = plt.contour(H, W, qda_decision, 0, colors = 'black').legend_elements()

    plt.title('Weight vs Height for QDA')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.legend([h1[0], h2[0], h3[0]], ['Female QDA', 'Male QDA', 'Decision Boundary'])
    plt.savefig("qda.pdf")
    plt.show()
    