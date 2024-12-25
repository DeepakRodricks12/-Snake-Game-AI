# Import:
# -------
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    """
    Plot the scores and mean scores of the episodes.

    Parameters:
        scores (list): List of scores for each episode.
        mean_scores (list): List of mean scores for each episode.
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
    plt.savefig("training_curve.png")
    plt.show()
