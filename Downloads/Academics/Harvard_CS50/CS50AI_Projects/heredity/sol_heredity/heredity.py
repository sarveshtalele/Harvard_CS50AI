import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_p = 1.0

    # Map each person to their number of genes for this "world"
    gene_counts = {
        person: (1 if person in one_gene else 2 if person in two_genes else 0)
        for person in people
    }

    for person, data in people.items():
        person_genes = gene_counts[person]
        person_has_trait = person in have_trait

        # Calculate probability of having the specified number of genes
        if data["mother"] is None:
            # Unconditional probability for people with no parents
            gene_p = PROBS["gene"][person_genes]
        else:
            # Conditional probability based on parents' genes
            mother = data["mother"]
            father = data["father"]
            mother_genes = gene_counts[mother]
            father_genes = gene_counts[father]

            # Probability of passing gene from a parent
            prob_from_parent = {
                0: PROBS["mutation"],
                1: 0.5,
                2: 1 - PROBS["mutation"]
            }

            p_from_mother = prob_from_parent[mother_genes]
            p_from_father = prob_from_parent[father_genes]

            if person_genes == 0:
                gene_p = (1 - p_from_mother) * (1 - p_from_father)
            elif person_genes == 1:
                gene_p = (p_from_mother * (1 - p_from_father)) + ((1 - p_from_mother) * p_from_father)
            else:  # person_genes == 2
                gene_p = p_from_mother * p_from_father

        # Calculate probability of having the trait given the number of genes
        trait_p = PROBS["trait"][person_genes][person_has_trait]

        # Multiply into the total joint probability
        joint_p *= gene_p * trait_p

    return joint_p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `one_gene`, `two_genes`, and `have_trait`, respectively.
    """
    for person in probabilities:
        # Determine person's gene count and trait status in this world
        num_genes = 1 if person in one_gene else 2 if person in two_genes else 0
        has_trait = person in have_trait

        # Add the joint probability to the corresponding values
        probabilities[person]["gene"][num_genes] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize the gene distribution
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total > 0:
            for gene_count in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene_count] /= gene_total

        # Normalize the trait distribution
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total > 0:
            for trait_value in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait_value] /= trait_total


if __name__ == "__main__":
    main()
