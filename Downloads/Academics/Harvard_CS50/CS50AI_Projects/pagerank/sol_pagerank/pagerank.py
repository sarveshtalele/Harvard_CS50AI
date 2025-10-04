import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    num_pages = len(corpus)
    links = corpus[page]
    num_links = len(links)

    if num_links == 0:
        # If no outgoing links, choose randomly among all pages.
        for p in corpus:
            distribution[p] = 1 / num_pages
    else:
        # Probability of choosing a page randomly from the whole corpus
        random_prob = (1 - damping_factor) / num_pages

        # Probability of choosing a linked page
        link_prob = damping_factor / num_links

        for p in corpus:
            distribution[p] = random_prob
            if p in links:
                distribution[p] += link_prob

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {page: 0 for page in corpus}

    # First sample is chosen at random
    sample = random.choice(list(corpus.keys()))
    pagerank[sample] += 1

    # Generate the remaining n-1 samples
    for _ in range(n - 1):
        model = transition_model(corpus, sample, damping_factor)
        pages = list(model.keys())
        probabilities = list(model.values())
        sample = random.choices(pages, weights=probabilities, k=1)[0]
        pagerank[sample] += 1

    # Normalize counts to get probabilities
    for page in pagerank:
        pagerank[page] /= n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    pagerank = {page: 1 / num_pages for page in corpus}

    # Create a mapping from each page to pages that link to it for efficiency
    links_to_page = {p: set() for p in corpus}
    for p in corpus:
        for linked_page in corpus[p]:
            links_to_page[linked_page].add(p)

    while True:
        max_change = 0
        for page in pagerank:

            # First part of the formula
            random_choice_prob = (1 - damping_factor) / num_pages

            # Second part of the formula (summation)
            link_prob_sum = 0
            for linking_page in links_to_page[page]:
                num_links = len(corpus[linking_page])
                if num_links == 0:
                    # A page with no links is treated as having links to all pages
                    link_prob_sum += pagerank[linking_page] / num_pages
                else:
                    link_prob_sum += pagerank[linking_page] / num_links

            # Handle pages with no links pointing to them from the entire corpus
            for p in corpus:
                if len(corpus[p]) == 0:
                    link_prob_sum += pagerank[p] / num_pages


            new_rank = random_choice_prob + damping_factor * link_prob_sum

            # Check for convergence
            change = abs(pagerank[page] - new_rank)
            if change > max_change:
                max_change = change

            pagerank[page] = new_rank

        if max_change < 0.001:
            break

    # Normalize the final PageRank values to ensure they sum to 1
    total_rank = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_rank

    return pagerank


if __name__ == "__main__":
    main()
