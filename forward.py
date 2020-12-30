def forward_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) forward decoding
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    f = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        f[state, 0] = np.log(initProbs[state] * emissionProbs[state, observations[0]])
    for k in range(1, nObservations):
        for state in states:
            logsum = -np.inf
            for previousState in states:
                temp = f[previousState, k-1] + np.log(transProbs[previousState, state])
                if temp > -np.inf:
                    logsum = temp + np.log(1 + np.exp(logsum - temp))
            f[state, k] = np.log(emissionProbs[state, observations[k]]) + logsum
    return f