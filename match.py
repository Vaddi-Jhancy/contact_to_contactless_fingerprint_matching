import cv2
import numpy as np
import random
import csv
import os
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms

# CSV file to store results
CSV_FILE = "fingerprint_matching_results.csv"

# Function to load minutiae points from a file
def load_minutiae(file_path):
    return np.loadtxt(file_path, dtype=int)

# Function to compute metrics: Accuracy, Precision, Recall, F1-Score
def compute_metrics(matches, total_minutiae1, total_minutiae2):
    TP = matches
    FP = total_minutiae1 - matches  
    FN = total_minutiae2 - matches  
    TN = 0  

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (total_minutiae1 + total_minutiae2) if (total_minutiae1 + total_minutiae2) > 0 else 0

    return accuracy, precision, recall, f1_score

# Method 1: Hough Transform Matching
def hough_transform_matching(minutiae1, minutiae2):
    centroid1 = np.mean(minutiae1, axis=0)
    centroid2 = np.mean(minutiae2, axis=0)
    translation = centroid2 - centroid1
    
    transformed_minutiae1 = minutiae1 + translation
    distances = cdist(transformed_minutiae1, minutiae2, metric='euclidean')
    matches = np.sum(np.min(distances, axis=1) < 10)  

    score = matches / max(len(minutiae1), len(minutiae2))
    return matches, score

# Method 2: Genetic Algorithm Matching
def genetic_algorithm_matching(minutiae1, minutiae2):
    def fitness(individual):
        transform = np.array(individual).reshape((2, 3))
        transformed_minutiae1 = (minutiae1 @ transform[:, :2].T) + transform[:, 2]
        distances = cdist(transformed_minutiae1, minutiae2, metric='euclidean')
        matches = np.sum(np.min(distances, axis=1) < 10)
        transformation_penalty = np.linalg.norm(transform) * 0.01  
        return (matches - transformation_penalty,)  # Higher is better
        # return (matches,)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -10, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.4, ngen=200, verbose=False)
    best_ind = tools.selBest(population, k=1)[0]
    matches = fitness(best_ind)[0]
    score = matches / max(len(minutiae1), len(minutiae2))
    return matches, score

# Method 3: Core Point Matching
def core_point_matching(minutiae1, minutiae2):
    core1 = np.mean(minutiae1, axis=0)
    core2 = np.mean(minutiae2, axis=0)

    aligned_minutiae1 = minutiae1 - core1 + core2
    distances = cdist(aligned_minutiae1, minutiae2, metric='euclidean')
    matches = np.sum(np.min(distances, axis=1) < 10)

    score = matches / max(len(minutiae1), len(minutiae2))
    return matches, score

# Function to save results to CSV (Appending)
def save_results_to_csv(results):
    header = ["Method", "Score", "Accuracy", "Precision", "Recall", "F1-Score"]
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  
        writer.writerows(results)  

# Main function to perform matching and store results
def match_fingerprints(minutiae_file1, minutiae_file2):
    minutiae1 = load_minutiae(minutiae_file1)
    minutiae2 = load_minutiae(minutiae_file2)

    matches_hough, score1 = hough_transform_matching(minutiae1, minutiae2)
    matches_ga, score2 = genetic_algorithm_matching(minutiae1, minutiae2)
    matches_core, score3 = core_point_matching(minutiae1, minutiae2)

    accuracy1, precision1, recall1, f1_1 = compute_metrics(matches_hough, len(minutiae1), len(minutiae2))
    accuracy2, precision2, recall2, f1_2 = compute_metrics(matches_ga, len(minutiae1), len(minutiae2))
    accuracy3, precision3, recall3, f1_3 = compute_metrics(matches_core, len(minutiae1), len(minutiae2))

    print(f"\nHough Transform Matching:")
    print(f"Score: {score1:.3f}, Accuracy: {accuracy1:.3f}, Precision: {precision1:.3f}, Recall: {recall1:.3f}, F1-score: {f1_1:.3f}")

    print(f"\nGenetic Algorithm Matching:")
    print(f"Score: {score2:.3f}, Accuracy: {accuracy2:.3f}, Precision: {precision2:.3f}, Recall: {recall2:.3f}, F1-score: {f1_2:.3f}")

    print(f"\nCore Point Matching:")
    print(f"Score: {score3:.3f}, Accuracy: {accuracy3:.3f}, Precision: {precision3:.3f}, Recall: {recall3:.3f}, F1-score: {f1_3:.3f}")

    results = [
        ["Hough Transform", score1, accuracy1, precision1, recall1, f1_1],
        ["Genetic Algorithm", score2, accuracy2, precision2, recall2, f1_2],
        ["Core Point Matching", score3, accuracy3, precision3, recall3, f1_3]
    ]
    
    save_results_to_csv(results)

    return results

# Example usage
match_fingerprints("f2_processed_output/p14/p1/minutiae.txt", "f2_1_processed_contact_based/1_1/minutiae.txt")
