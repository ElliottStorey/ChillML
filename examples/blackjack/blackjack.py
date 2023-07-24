import random
import numpy as np

class Blackjack:
    def __init__(self):
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9,  10, 10, 10, 10] * 4
        self.hand = []
        self.dealer = []
        self.deal(self.dealer)
    
    def deal(self, hand):
        card = random.choice(self.deck)
        self.deck.pop(card)
        return card
        
    def hit(self):
        self.deal(self.hand)

    def stand(self):
        self.deal(self.dealer)

def generate(games):
    inputs = []
    outputs = []
    for _ in range(games):
        bj = Blackjack()
        while True:
            action = random.choice([bj.hit, bj.stand])
            if action == bj.hit:
                inputs.append(np.array([[sum(bj.hand), sum(bj.dealer), 1]]))
            else:
                inputs.append(np.array([[sum(bj.hand), sum(bj.dealer), 0]]))
            action()
            if bj.hand > 21:
                outputs.append(np.array([[0]]))
                break
            elif bj.dealer > 21:
                outputs.append(np.array([[1]]))
                break