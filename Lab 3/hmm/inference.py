import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    forward_messages[0] = rover.Distribution()
    for x in all_possible_hidden_states:
        if observations[0] is None:
            factor = 1
        else:
            factor = observation_model(x)[observations[0]]
        if prior_distribution[x] * factor != 0:
            forward_messages[0][x] = prior_distribution[x] * factor
    forward_messages[0].renormalize()

    for i in range(1, len(forward_messages)):
        dist = rover.Distribution({})
        for x in all_possible_hidden_states:
            if observations[i] is None:
                factor = 1
            else:
                factor = observation_model(x)[observations[i]]
            current = 0
            for key in forward_messages[i-1]:
                current += forward_messages[i-1][key]*transition_model(key)[x]
            if factor * current != 0:
                dist[x] = factor * current
        forward_messages[i] = dist
        forward_messages[i].renormalize()
                   
    # TODO: Compute the backward messages
    backward_messages[num_time_steps - 1] = rover.Distribution({})

    for x in all_possible_hidden_states:
        backward_messages[num_time_steps-1][x] = 1
    
    for i in range(num_time_steps - 2, -1, -1):
        dist = rover.Distribution({})
        for x in all_possible_hidden_states:
            current = 0
            for key in backward_messages[i+1]:
                if observations[i+1] is None:
                    factor = 1
                else:
                    factor = observation_model(key)[observations[i+1]]
                current += backward_messages[i+1][key] * factor * transition_model(x)[key]
            if current != 0:
                dist[x] = current
        backward_messages[i] = dist
        backward_messages[i].renormalize()
    
    # TODO: Compute the marginals 
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for x in all_possible_hidden_states:
            if forward_messages[i][x] * backward_messages[i][x] != 0:
                marginals[i][x] = forward_messages[i][x] * backward_messages[i][x]
        marginals[i].renormalize()

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here

    num_time_steps = len(observations)
    w = [None] * num_time_steps
    tracker = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    # Initialization
    w[0] = rover.Distribution({})
    for state in all_possible_hidden_states:
        if observations[0] is None:
            initial_obsmodel_val = 1
        else:
            initial_obsmodel_val = observation_model(state)[observations[0]]
        if prior_distribution[state] != 0 and initial_obsmodel_val != 0:
            w[0][state] = np.log(prior_distribution[state]) + np.log(initial_obsmodel_val)


    for i in range(1, num_time_steps):
        w[i] = rover.Distribution()
        tracker[i] = dict()
        for state in all_possible_hidden_states:
            if observations[i] is None:
                term = 1
            else:
                term = observation_model(state)[observations[i]]
            best_prev = None
            best_sum = np.NINF
            for prev_state in w[i-1]:
                if transition_model(prev_state)[state]!=0 and np.log(transition_model(prev_state)[state]) + w[i-1][prev_state] > best_sum:
                    best_sum = np.log(transition_model(prev_state)[state]) + w[i-1][prev_state]
                    best_prev = prev_state
            tracker[i][state] = best_prev
            if term!=0:
                w[i][state] = best_sum + np.log(term)

    # Find best
    curr = np.NINF
    for state in w[-1]:
        if w[-1][state] > curr:
            curr = w[-1][state]
            estimated_hidden_states[num_time_steps-1] = state 

    for i in range(num_time_steps-2, -1, -1):
        estimated_hidden_states[i] = tracker[i+1][estimated_hidden_states[i+1]]
    
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    # Marginal Errors
    marginals_count = 0
    viterbi_count = 0
    for i in range(num_time_steps):
        if estimated_states[i] == hidden_states[i]:
            viterbi_count += 1
        if marginals[i].get_mode() == hidden_states[i]:
            marginals_count += 1
    viterbi_error = 1 - viterbi_count/num_time_steps
    marginal_error = 1 - marginals_count/num_time_steps

    print(f'Forward-Backward error: {marginal_error}')
    print(f'Viterbi error: {viterbi_error}')
    
    '''for i in range(num_time_steps):
        print(marginals[i].get_mode())'''
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
