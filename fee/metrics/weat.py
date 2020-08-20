import numpy as np
import random 
from numpy.random import *
from itertools import combinations

import numpy as np
from sympy.utilities.iterables import multiset_permutations


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_differential_association(X, Y, A, B):
    """
    Returns differential association of two sets of target words with the attribute for WEAT score.
    s(X, Y, A, B)
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: differential association (float value)
    """
    return np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B))


def weat_p_value(X, Y, A, B):
    """
    Returns one-sided p-value of the permutation test for WEAT score
    CAUTION: this function is not appropriately implemented, so it runs very slowly
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: p-value (float value)
    """
    diff_association = weat_differential_association(X, Y, A, B)
    target_words = np.concatenate((X, Y), axis=0)

    # get all the partitions of X union Y into two sets of equal size.
    idx = np.zeros(len(target_words))
    idx[:len(target_words) // 2] = 1

    partition_diff_association = []
    for i in multiset_permutations(idx):
        i = np.array(i, dtype=np.int32)
        partition_X = target_words[i]
        partition_Y = target_words[1 - i]
        partition_diff_association.append(weat_differential_association(partition_X, partition_Y, A, B))

    partition_diff_association = np.array(partition_diff_association)

    return np.sum(partition_diff_association > diff_association) / len(partition_diff_association)


def weat_score(X, Y, A, B, p_val):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)


    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))
    w_score = tmp1 / tmp2

    if p_val:
        p = weat_p_value(X, Y, A, B)
        return w_score, p

    return w_score


class WEAT(object):
    """
    Perform WEAT (Word Embedding Association Test) bias tests on a language model.
    Follows from Caliskan et al 2017 (10.1126/science.aal4230).
    Code mostly "stolen" from: https://github.com/chadaeun/weat_replication/blob/master/lib/weat.py
    """
    
    def __init__(self, E):
        self.E = E
        self.group_db = {
            'instruments' : ["bagpipe", "cello", "guitar", "lute", "trombone", "banjo", "clarinet", "harmonica", "mandolin", "trumpet", "bassoon", "drum", "harp", "oboe", "tuba", "bell", "fiddle", "harpsichord", "piano", "viola", "bongo","flute", "horn", "saxophone", "violin"],
            'weapons' : ["arrow", "club", "gun", "missile", "spear", "axe", "dagger", "harpoon", "pistol", "sword", "blade", "dynamite", "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun", "teargas", "cannon", "grenade", "mace", "slingshot", "whip"],
            'flowers' : ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose", "blue-bell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet", "carnation", "gladiola", "magnolia", "petunia", "zinnia"],
            'insects' : ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula", "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly","dragonfly", "horsefly", "roach", "weevil"],
            'pleasant' : ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter", "paradise", "vacation"],
            'unpleasant' : ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", "rotten", "vomit", "agony", "prison"],
            'math' : ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
            'arts' : ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
            'male' : ["male", "man", "boy", "brother", "he", "him", "his", "son"],
            'female' : ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"],
            'science' : ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"],
            'arts_2' : ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"],
            'male_2' : ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"],
            'female_2' : ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "he"],
            'mental_disease' : ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"],
            'physical_disease' : ["sick", "illness", "influenza", "disease", "virus", "cancer"],
            'temporary' : ["impermanent", "unstable", "variable", "fleeting", "short-term", "brief", "occasional"],
            'permanent' : ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"],
        }    

    def compute(self, *args, p_val=False):
        groups = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    groups.append(self.group_db[arg])  
                except Exception as e:
                    print("Invalid group name, available groups:\n")
                    print(list(self.group_db.keys()))
                    raise e    
            else:
                groups.append(arg)        

        target_1, target_2, attributes_1, attributes_2 = groups

        X = [self.E.v(w.lower()) for w in target_1]
        Y = [self.E.v(w.lower()) for w in target_2]
        A = [self.E.v(w.lower()) for w in attributes_1]
        B = [self.E.v(w.lower()) for w in attributes_2]

        return weat_score(X, Y, A, B, p_val)
