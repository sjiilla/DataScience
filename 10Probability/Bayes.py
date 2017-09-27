#Baye's rule
def bayes(prior, likelihood, evidence):
        return (prior * likelihood) / evidence
    
#prior belief is 2% and cancer test raliability is assuming that only 90% accurate
print(bayes(0.2, 0.9, (0.2*0.9 + 0.98*0.1)))