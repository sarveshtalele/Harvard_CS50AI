from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # Basic rule for character A: A is either a knight or a knave, but not both.
    Biconditional(AKnight, Not(AKnave)),

    # The sentence A says
    # "I am both a knight and a knave" translates to And(AKnight, AKnave)
    # A is a knight if and only if their statement is true.
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # Basic rules for characters A and B
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),

    # The sentence A says
    # "We are both knaves" translates to And(AKnave, BKnave)
    # A is a knight if and only if their statement is true.
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."

# First, define what A said as a variable: "We are the same kind."
# (A is Knight and B is Knight) OR (A is Knave and B is Knave)
sentenceA = Or(
    And(AKnight, BKnight),
    And(AKnave, BKnave)
)

# Next, define what B said as a variable: "We are of different kinds."
# (A is Knight and B is Knave) OR (A is Knave and B is Knight)
sentenceB = Or(
    And(AKnight, BKnave),
    And(AKnave, BKnight)
)

# Now, build the knowledge base using these variables.
knowledge2 = And(
    # Basic rules for A and B
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),

    # Rule for A's statement
    Biconditional(AKnight, sentenceA),

    # Rule for B's statement
    Biconditional(BKnight, sentenceB)
)


# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Basic rules for all three characters
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    Biconditional(CKnight, Not(CKnave)),

    # B's first statement: "A said 'I am a knave'."
    # The content of what A said is: "I am a knave" (AKnave)
    # The logical structure of A having said that is: Biconditional(AKnight, AKnave)
    # So B's statement is that this structure is true.
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),

    # B's second statement: "C is a knave."
    Biconditional(BKnight, CKnave),

    # C's statement: "A is a knight."
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
