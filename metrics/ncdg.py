import math
import csv
def ComputeDCG(sorted_ranks):
    dcg = 0
    for pos, rating in enumerate(sorted_ranks):
        dcg += ((2 ^ rating - 1) / math.log2(pos + 2))
    return dcg


def ComputeNDCGPerQuery(ideal_ratings, scored_ratings):
    ideal_ranks = sorted(ideal_ratings, reverse=True)
    model_ranks = [pair[1] for pair in sorted(scored_ratings, key=lambda tup: tup[0], reverse=True)]
    ideal_dcg = ComputeDCG(ideal_ranks)
    model_dcg = ComputeDCG(model_ranks)
    return model_dcg / ideal_dcg if ideal_dcg > 0 else 0

def ComputeNDCGPerQueryForPairwise(ideal_ratings, model_ranks):
    print(ideal_ratings)
    print(model_ranks)
    ideal_ranks = sorted(ideal_ratings, reverse=True)
    ideal_dcg = ComputeDCG(ideal_ranks)
    model_dcg = ComputeDCG(model_ranks)
    return model_dcg / ideal_dcg if ideal_dcg > 0 else 0


def ComputeNDCG(score_file, ndcg_pos):
    ndcg = 0
    curr_qid = -1
    ideal_ratings = []
    scored_ratings = []
    q_count = 0

    with open(score_file, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            q_id = row[0]
            doc_id = row[1]
            rating = int(row[1])
            score = float(row[1])

            if q_id != curr_qid:
                ndcg += ComputeNDCGPerQuery(ideal_ratings, scored_ratings)
                q_count += 1
                ideal_ratings = []
                scored_ratings = []

            curr_qid = q_id
            ideal_ratings.append(rating)
            scored_ratings.append((score, rating))

    ndcg += ComputeNDCGPerQuery(ideal_ratings, scored_ratings)
    q_count += 1

    return ndcg / q_count