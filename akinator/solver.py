import pandas as pd
import numpy as np
from collections import Counter


def entropy(question,candidates,priors):
    """
    Now there are two ways the entropy calculation can be done :
    - The General Case : We select the candidates for a particular question and then all the prob mass for them which we are doing in the code right now
    - The Uniform Scenario Case : where all the candidates are equally likely , in our case then instead of adding the probabilities , we can just count the number of candidates
    and divide by the length of dataset to get total mass , total_prob_mass = candidates[candidates[question]==1].shape[0]/candidates.shapes[0]
    """
    selected_candidates =  candidates[candidates[question]==1]
    total_prob_mass = sum(priors[name] for name in selected_candidates['name'])

    if total_prob_mass == 0 or total_prob_mass == 1:
        return 0
    
    entropy = -(np.log2(total_prob_mass)*total_prob_mass + np.log2(1-total_prob_mass)*(1-total_prob_mass))

    return entropy


def question_picker(questions,candidates,priors):
    question_entropy = Counter()

    for question in questions:
        question_entropy[question] = entropy(question,candidates,priors)
        
    return question_entropy,max(question_entropy, key=question_entropy.get)



if __name__ == "__main__":
    df = pd.read_csv("data/entities_v1.csv")
    questions = df.columns.tolist()[1:]

    ## Assuming uniform probs
    priors = {name:1/len(df) for name in df['name'].unique()}

    print(question_picker(questions,df,priors))