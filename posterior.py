def posterior_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) posterior function
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    f = forward_hmm(transProbs, initProbs, emissionProbs, states, observations)
    b = backward_hmm(transProbs, initProbs, emissionProbs, states, observations)
    probObservations = f[0, len(observations)-1]
    for i in range(1, len(states)):
        j = f[i, len(observations)-1]
        if j > -np.inf:
            probObservations = j + np.log(1 + np.exp(probObservations - j))
    posteriorProb = np.exp((f+b) - probObservations)
    return posteriorProb