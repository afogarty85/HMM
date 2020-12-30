def viterbi(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) viterbi decoding
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    v = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        v[state, 0] = np.log(initProbs[state] * emissionProbs[state, observations[0]])
        # iteration
        for k in range(1, nObservations):
            for state in states:
                maxi = -np.inf
                for previousState in states:
                    temp = v[previousState, k-1] + np.log(transProbs[previousState, state])
                    maxi = max(maxi, temp)
                v[state, k] = np.log(emissionProbs[state, observations[k]]) + maxi
        viterbiPath = np.repeat(np.nan, nObservations)
        for state in states:
            if max(v[:, nObservations-1]) == v[state, nObservations-1]:
                viterbiPath[nObservations-1] = state
                break
        for k in range(nObservations-2, -1, -1):
            for state in states:
                if (max(v[:, k] + np.log(transProbs[:, int(viterbiPath[k+1])]))) == v[state, k] + np.log(transProbs[state, int(viterbiPath[k+1])]):
                    viterbiPath[k] = state
                    break
    return viterbiPath