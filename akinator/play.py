import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solver import question_picker
import json


class PlayAkinator:
    def __init__(self, path, priors=None):
        self.candidates = pd.read_csv(path)
        self.questions = self.candidates.columns.to_list()[1:]
        self.selected_answer = self.candidates.iloc[np.random.randint(0, len(self.candidates))]

        if priors:
            self.priors = priors.copy()
            self.prior_type = "non-uniform"
        else:
            self.priors = {name:1/len(self.candidates) for name in self.candidates['name'].unique()}
            self.prior_type = "uniform"

    def run_game(self):
        i = 0
        while len(self.candidates) > 1:
            question_entropy, best_guess = question_picker(self.questions, self.candidates, self.priors)

            print(f"Entropy Table at Step {i}",question_entropy)
            print(f"Best question: {best_guess}  (entropy={question_entropy[best_guess]:.3f})")

            info = self.get_answer(best_guess)

            self.candidates = self.candidates[self.candidates[best_guess] == info]

            if self.prior_type == "uniform":
                surviving = set(self.candidates['name'])
                self.priors = {n: 1 / len(self.candidates) for n in surviving}
            else:
                surviving = set(self.candidates['name'])
                self.priors = {k: v for k, v in self.priors.items() if k in surviving}
                total = sum(self.priors.values())
                self.priors = {k: v / total for k, v in self.priors.items()}

            self.questions.remove(best_guess)

            i += 1

        print(f"\nAnswer found: {self.candidates['name'].iloc[0]}")
        print(f"True answer:  {self.selected_answer['name']}")
        print(f"Steps taken:  {i}")


    def run_game_silently(self, true_answer):
        selected_answer = self.candidates[self.candidates['name'] == true_answer].iloc[0]

        i = 0
        while len(self.candidates) > 1 and len(self.questions) > 0:

            _, best_guess = question_picker(self.questions, self.candidates, self.priors)

            info = selected_answer[best_guess]

            self.candidates = self.candidates[self.candidates[best_guess] == info]

            if self.prior_type == "uniform":
                surviving = set(self.candidates['name'])
                self.priors = {n: 1 / len(self.candidates) for n in surviving}
            else:
                surviving = set(self.candidates['name'])
                self.priors = {k: v for k, v in self.priors.items() if k in surviving}
                total = sum(self.priors.values())
                self.priors = {k: v / total for k, v in self.priors.items()}

            self.questions.remove(best_guess)

            i += 1


        return i, self.candidates['name'].iloc[0]


    def get_answer(self, question):
        return self.selected_answer[question]


def evaluate(path,priors=None):
    version = path.split("_")[-1].replace(".csv","")
    df = pd.read_csv(path)

    results = []
    for true_animal in df['name']:
        play = PlayAkinator(path=path,priors=priors)
        steps, guessed = play.run_game_silently(true_animal)
        correct = (guessed == true_animal)
        results.append({"animal": true_animal, "steps": steps, "guessed": guessed, "correct": correct})

    res_df = pd.DataFrame(results)

    # --- insights ---
    print("=" * 60)
    print(f"Evaluated {len(res_df)} animals")
    print("=" * 60)
    print(f"Avg steps:        {res_df['steps'].mean():.2f}")
    print(f"Median steps:     {res_df['steps'].median():.1f}")
    print(f"Min / Max steps:  {res_df['steps'].min()} / {res_df['steps'].max()}")
    print(f"Theoretical min:  {np.log2(len(df)):.2f}  (log2 of pool size)")
    print(f"Correct guesses:  {res_df['correct'].sum()} / {len(res_df)}")

    print("\nHardest 5 animals (most steps):")
    print(res_df.nlargest(5, 'steps')[['animal', 'steps', 'correct']].to_string(index=False))

    print("\nEasiest 5 animals (fewest steps):")
    print(res_df.nsmallest(5, 'steps')[['animal', 'steps']].to_string(index=False))

    failures = res_df[~res_df['correct']]
    if len(failures) > 0:
        print(f"\n⚠ {len(failures)} animals were NOT guessed correctly:")
        print(failures[['animal', 'guessed', 'steps']].to_string(index=False))

    # --- plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Histogram of step counts
    bins = range(1, res_df['steps'].max() + 2)
    ax1.hist(res_df['steps'], bins=bins, edgecolor='black', alpha=0.75)
    ax1.axvline(res_df['steps'].mean(), color='red', linestyle='--', label=f"mean = {res_df['steps'].mean():.2f}")
    ax1.axvline(np.log2(len(df)), color='green', linestyle=':', label=f"log2(N) = {np.log2(len(df)):.2f}")
    ax1.set_xlabel("Steps to identify")
    ax1.set_ylabel("Animals")
    ax1.set_title("Step-count distribution")
    ax1.legend()

    # 2. Per-animal bar chart, sorted
    sorted_df = res_df.sort_values('steps')
    ax2.barh(sorted_df['animal'], sorted_df['steps'],
             color=['#E8976C' if c else 'red' for c in sorted_df['correct']])
    ax2.set_xlabel("Steps")
    ax2.set_title("Steps per animal (red = solver failed)")
    ax2.tick_params(axis='y', labelsize=6)

    plt.tight_layout()
    
    if priors:
        plt.savefig(f"plots/eval_results_{version}_non_uniform_priors.png", dpi=120, bbox_inches='tight')
    else:
        plt.savefig(f"plots/eval_results_{version}.png", dpi=120, bbox_inches='tight')

    print(f"\nPlot saved → eval_results.png")
    plt.show()

    return res_df


if __name__ == "__main__":

    # play = PlayAkinator(path='data/entities_v2.csv')
    # play.run_game()
    priors = json.load(open("data/priors_exp3.json","r"))
    print(priors)
    evaluate('data/entities_v2.csv',priors)
