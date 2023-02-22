import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

# Import Counter library
from typing import Counter

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here

    # Get spam and ham files
    spam_files = file_lists_by_category[0]
    ham_files = file_lists_by_category[1]
    all_files = spam_files + ham_files

    # Create a set for the vocabulary
    vocab = set()

    for f in all_files:
        words_in_file = set(util.get_words_in_file(f))
        vocab = vocab.union(words_in_file)

    p_d, q_d = util.get_word_freq(spam_files), util.get_word_freq(ham_files)

    spam_count, ham_count = 0, 0
    
    # Add the word to count
    for word in vocab:
        q_d[word] = q_d[word] + 1
        p_d[word] = p_d[word] + 1
        ham_count = ham_count + q_d[word]
        spam_count = spam_count + p_d[word]

    # Divide the p_d and q_d by the counts
    for word in vocab:
        q_d[word] = q_d[word]/ham_count
        p_d[word] = p_d[word]/spam_count

    # Create the return tuple
    probabilities_by_category = (p_d, q_d)
    
    return probabilities_by_category

def classify_new_email(filename, probabilities_by_category, prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here

    # smoothed estimates of p_d and q_d
    p_d = probabilities_by_category[0]
    q_d = probabilities_by_category[1]

    # Create a counter
    counts = Counter()
    words = util.get_words_in_file(filename)
    for word in words:
        counts[word] += 1

    # Initially set probabilities to 0
    spam_probability, ham_probability = 0, 0

    # Add the spam and ham probabilities by the log base 10 of the estimates by the count
    for word in counts:
        if word in p_d:
            spam_probability = spam_probability + np.log10(p_d[word]) * counts[word]
            ham_probability = ham_probability + np.log10(q_d[word]) * counts[word]

    spam_probability = spam_probability + np.log10(prior_by_category[0]) 
    ham_probability = ham_probability + np.log10(prior_by_category[1])

    # If the probability of it being ham is higher than spam, classify the result as ham
    # else classify result as spam
    if ham_probability > spam_probability:
        classify_result = ("ham", [spam_probability, ham_probability])
    else:
        classify_result = ("spam", [spam_probability, ham_probability])
    
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "classifier/data/spam"
    ham_folder = "classifier/data/ham"
    test_folder = "classifier/data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    type1_error_count, type2_error_count = [], []

    epsilon_values = [0, 1e-15, 1e-10, 1e-5, 1e-2, 1, 1e2, 1e5, 1e10, 1e15, 1e20, 1e25, 1e30]
    
    for epsilon in epsilon_values:
    
        # Store the classification results
        performance_measures = np.zeros([2,2])
        
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    (1 / (epsilon + 1), 1- (1 / (epsilon + 1))))
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0],totals[0],correct[1],totals[1]))

        type1_error_count.append(totals[0] - correct[0])
        type2_error_count.append(totals[1] - correct[1])
    
    # Plot the Type1 and Type2 errors
    plt.plot(type1_error_count, type2_error_count)
    plt.ylabel("Number of Type 2 Errors")
    plt.xlabel("Number of Type 1 Errors")
    plt.ylim([-2, 53])
    plt.xlim([-2, 38])
    plt.title("Type 2 Errors vs Type 1 Errors")
    plt.savefig("nbc.pdf")
    plt.show()

 