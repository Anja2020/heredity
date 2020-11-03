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
    joint_prob = 1
    zero_genes = set()

    for person in people:
        if person not in one_gene and person not in two_genes:
            zero_genes.add(person)

    # compute probability that a set of people has no gene
    for person in people:
        parent_prob = 0

        # person is parent
        if people[person]['mother'] is None:

            # compute probability that a person has one gene
            if person in one_gene:
                num_genes = 1

            # compute probability that a person has two genes
            elif person in two_genes:
                num_genes = 2

            # compute probability that a person has no gene
            else:
                num_genes = 0

            # check for trait attribute
            trait = True if person in have_trait else False

            joint_prob *= PROBS["gene"][num_genes] * PROBS["trait"][num_genes][trait]

        # person is child
        else:
            # get info about parents
            mother = people[person]["mother"]
            father = people[person]["father"]

            if person in one_gene:
                num_genes = 1

                if mother in zero_genes and father in zero_genes:
                    parent_prob = PROBS["mutation"] * (1 - PROBS["mutation"]) + (
                        1 - PROBS["mutation"]) * PROBS["mutation"]
                elif mother in one_gene and father in one_gene:
                    parent_prob = 0.5 * 0.5 + 0.5 * 0.5
                elif mother in two_genes and father in two_genes:
                    parent_prob = (
                        1 - PROBS["mutation"]) * PROBS["mutation"] + PROBS["mutation"] * (1 - PROBS["mutation"])
                elif (mother in zero_genes and father in one_gene) or (mother in one_gene and father in zero_genes):
                    parent_prob = PROBS["mutation"] * \
                        0.5 + 0.5 * (1 - PROBS["mutation"])
                elif (mother in two_genes and father in zero_genes) or (mother in zero_genes and father in two_genes):
                    parent_prob = (
                        1 - PROBS["mutation"]) * (1 - PROBS["mutation"]) + PROBS["mutation"] * PROBS["mutation"]
                elif (mother in two_genes and father in one_gene) or (mother in one_gene and father in two_genes):
                    parent_prob = (
                        1 - PROBS["mutation"]) * 0.5 + 0.5 * PROBS["mutation"]

            # compute probability that a person has two genes
            elif person in two_genes:
                num_genes = 2

                if mother in zero_genes and father in zero_genes:
                    parent_prob = PROBS["mutation"] * \
                        PROBS["mutation"]
                elif mother in one_gene and father in one_gene:
                    parent_prob = 0.5 * 0.5
                elif mother in two_genes and father in two_genes:
                    parent_prob = (
                        1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
                elif (mother in zero_genes and father in one_gene) or (mother in one_gene and father in zero_genes):
                    parent_prob = PROBS["mutation"] * 0.5
                elif (mother in two_genes and father in zero_genes) or (mother in zero_genes and father in two_genes):
                    parent_prob = (
                        1 - PROBS["mutation"]) * PROBS["mutation"]
                elif (mother in two_genes and father in one_gene) or (mother in one_gene and father in two_genes):
                    parent_prob = (1 - PROBS["mutation"]) * 0.5

            # compute probability that a person has no gene
            else:
                num_genes = 0
    
                if mother in zero_genes and father in zero_genes:
                    parent_prob = (
                        1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
                elif mother in one_gene and father in one_gene:
                    parent_prob = 0.5 * 0.5
                elif mother in two_genes and father in two_genes:
                    parent_prob = PROBS["mutation"] * \
                        PROBS["mutation"]
                elif (mother in zero_genes and father in one_gene) or (mother in one_gene and father in zero_genes):
                    parent_prob = (1 - PROBS["mutation"]) * 0.5
                elif (mother in two_genes and father in zero_genes) or (mother in zero_genes and father in two_genes):
                    parent_prob = PROBS["mutation"] * \
                        (1 - PROBS["mutation"])
                elif (mother in two_genes and father in one_gene) or (mother in one_gene and father in two_genes):
                    parent_prob = PROBS["mutation"] * 0.5

            trait = True if person in have_trait else False
            trait_prob = PROBS["trait"][num_genes][trait]

            # compute total joint probability
            joint_prob *= trait_prob * parent_prob

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # update gene distribution
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        # update trait distribution
        trait = True if person in have_trait else False
        probabilities[person]["trait"][trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        gene_prob = (probabilities[person]["gene"][0], probabilities[person]
                     ["gene"][1], probabilities[person]["gene"][2])
        trait_prob = (probabilities[person]["trait"]
                      [True], probabilities[person]["trait"][False])
        if sum(gene_prob) != 0:
            # normalize gene distribution
            probabilities[person]["gene"][0] = float(
                probabilities[person]["gene"][0])/sum(gene_prob)
            probabilities[person]["gene"][1] = float(
                probabilities[person]["gene"][1]) / sum(gene_prob)
            probabilities[person]["gene"][2] = float(
                probabilities[person]["gene"][2])/sum(gene_prob)
        if sum(trait_prob) != 0:
            # normalize trait distribution
            probabilities[person]["trait"][True] = float(
                probabilities[person]["trait"][True]) / sum(trait_prob)
            probabilities[person]["trait"][False] = float(
                probabilities[person]["trait"][False])/sum(trait_prob)


if __name__ == "__main__":
    main()
