def backward_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) backward function
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    b = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        b[state, nObservations-1] = np.log(1)
    for k in range(nObservations-2, -1, -1):
        for state in states:
            logsum = -np.inf
            for nextState in states:
                temp = b[nextState, k+1] + np.log(transProbs[state, nextState]
                                                  * emissionProbs[nextState, observations[k+1]])
                if temp > -np.inf:
                    logsum = temp + np.log(1 + np.exp(logsum - temp))
            b[state, k] = logsum
    return b