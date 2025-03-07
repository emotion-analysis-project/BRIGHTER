#!/usr/bin/env python
import os
import csv
import json
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

from sklearn.metrics import f1_score
from scipy.stats import pearsonr

# This list is for reference only, actual emotions are dynamically determined from data
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def read_csv_file(filepath: str) -> List[Dict]:
    """
    Read a CSV file and return its contents as a list of dictionaries.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries with each row's data
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def find_optimal_top_count(data: List[Dict], dev_data: List[Dict] = None) -> Tuple[int, float]:
    """
    Find the optimal number of top emotions to predict by testing different values
    and selecting the one that gives the highest F1 score.
    
    Args:
        data: List of training data samples
        dev_data: List of development data samples (if None, use data for both training and testing)
        
    Returns:
        Tuple containing:
            - Optimal number of top emotions to predict
            - F1 score achieved with that number
    """
    # Get the list of emotions present in this dataset
    available_emotions = list(data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Store the percentage of non-zero values for each emotion
    emotion_percentages = {}
    
    for emotion in available_emotions:
        # Count occurrences of 0 and 1
        counter = Counter([int(sample[emotion]) for sample in data])
        
        # Get counts for class 1
        predicted_count = counter.get(1, 0)
        
        # Calculate percentage 
        total_samples = len(data)
        predicted_percentage = (predicted_count / total_samples) * 100
        
        # Store percentage
        emotion_percentages[emotion] = predicted_percentage
    
    # Sort emotions by percentage of non-zero values (highest to lowest)
    sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Use development set for testing if provided, otherwise use training data
    test_data = dev_data if dev_data is not None else data
    
    best_f1 = 0
    best_top_count = 1
    all_scores = {}
    
    # Try different numbers of top emotions (from 1 to number of available emotions)
    for top_count in range(1, len(available_emotions) + 1):
        # Get the top N emotions
        top_emotions = [emotion for emotion, _ in sorted_emotions[:top_count]]
        
        # Set predictions: 1 for top N emotions, 0 for others
        majority_classes = {}
        for emotion, _ in sorted_emotions:
            if emotion in top_emotions:
                majority_classes[emotion] = 1
            else:
                majority_classes[emotion] = 0
        
        # Evaluate predictions
        results = evaluate_track_a(test_data, majority_classes)
        f1_score = results["macro_f1"]
        all_scores[top_count] = f1_score
        
        # Update if this is the best score so far
        if f1_score > best_f1:
            best_f1 = f1_score
            best_top_count = top_count
    
    return best_top_count, best_f1, all_scores

def get_majority_class_track_a(data: List[Dict], dev_data: List[Dict] = None, find_optimal: bool = True) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Find the optimal number of top emotions to predict (if find_optimal is True),
    and predict 1 for these top N emotions with highest percentage of non-zero values,
    0 for all others.
    
    Args:
        data: List of data samples
        dev_data: Optional development data for finding optimal top count
        find_optimal: Whether to find the optimal number of emotions or use fixed top 2
        
    Returns:
        Tuple containing:
            - Dictionary mapping each emotion to its predicted class (1 for top N emotions, 0 for others)
            - Dictionary with detailed class distribution statistics
    """
    distribution_stats = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Store the percentage of non-zero values for each emotion
    emotion_percentages = {}
    
    # Calculate stats about non-zero emotions (for informational purposes only)
    total_non_zero_emotions = 0
    instances_with_emotions = 0
    
    for sample in data:
        # Count how many emotions have class 1 in this instance
        emotions_in_instance = sum(1 for emotion in available_emotions if int(sample[emotion]) == 1)
        if emotions_in_instance > 0:
            total_non_zero_emotions += emotions_in_instance
            instances_with_emotions += 1
    
    # If no instance has any emotion, default to 0
    if instances_with_emotions == 0:
        avg_non_zero_emotions = 0
    else:
        # Calculate average (only for instances with at least one emotion)
        avg_non_zero_emotions = total_non_zero_emotions / instances_with_emotions
    
    # Get the optimal or fixed number of top emotions to predict
    if find_optimal:
        # Find optimal count by testing different values
        top_count, best_f1, all_scores = find_optimal_top_count(data, dev_data)
        optimization_method = "optimized"
    else:
        # Use fixed top 2 emotions
        fixed_top_count = 2
        top_count = min(fixed_top_count, len(available_emotions))
        optimization_method = "fixed"
        best_f1 = None
        all_scores = None
    
    for emotion in available_emotions:
        # Count occurrences of 0 and 1
        counter = Counter([int(sample[emotion]) for sample in data])
        class_counts = {k: v for k, v in counter.items()}
        
        # Get counts for class 1
        predicted_count = class_counts.get(1, 0)
        
        # Calculate percentage 
        total_samples = len(data)
        predicted_percentage = (predicted_count / total_samples) * 100
        
        # Store percentage
        emotion_percentages[emotion] = predicted_percentage
        
        # Store detailed statistics
        distribution_stats[emotion] = {
            "predicted_class": 1,  # Will be updated after finding the top emotions
            "predicted_count": predicted_count,
            "predicted_percentage": predicted_percentage,
            "total_samples": total_samples,
            "class_distribution": class_counts,
            "rank": 0  # Will be updated with the emotion's rank
        }
    
    # Sort emotions by percentage of non-zero values (highest to lowest)
    sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top N emotions
    top_emotions = [emotion for emotion, _ in sorted_emotions[:top_count]]
    
    # Set predictions: 1 for top N emotions, 0 for others
    majority_classes = {}
    for i, (emotion, percentage) in enumerate(sorted_emotions):
        rank = i + 1  # Rank starts at 1
        distribution_stats[emotion]["rank"] = rank
        
        if emotion in top_emotions:
            majority_classes[emotion] = 1
            distribution_stats[emotion]["is_top_emotion"] = True
        else:
            majority_classes[emotion] = 0
            distribution_stats[emotion]["is_top_emotion"] = False
        
        # Update the predicted class in distribution stats
        distribution_stats[emotion]["predicted_class"] = majority_classes[emotion]
    
    # Store the average emotions per instance and optimization info in the stats
    for emotion in available_emotions:
        distribution_stats[emotion]["avg_non_zero_emotions"] = avg_non_zero_emotions
        distribution_stats[emotion]["instances_with_emotions"] = instances_with_emotions
        distribution_stats[emotion]["total_instances"] = len(data)
        distribution_stats[emotion]["predicted_top_count"] = top_count
        distribution_stats[emotion]["optimization_method"] = optimization_method
        if best_f1 is not None:
            distribution_stats[emotion]["best_f1"] = best_f1
        if all_scores is not None:
            distribution_stats[emotion]["all_scores"] = all_scores
    
    return majority_classes, distribution_stats

def get_majority_intensity_track_b(data: List[Dict]) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Find the most common non-zero intensity for each emotion in Track B data.
    
    Args:
        data: List of data samples
        
    Returns:
        Tuple containing:
            - Dictionary mapping each emotion to its most common non-zero intensity
            - Dictionary with detailed intensity distribution statistics
    """
    majority_intensities = {}
    distribution_stats = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Store the percentage of non-zero intensities for each emotion
    emotion_percentages = {}
    
    for emotion in available_emotions:
        # Count occurrences of each intensity (0-3)
        counter = Counter([int(sample[emotion]) for sample in data])
        intensity_counts = {k: v for k, v in counter.items()}
        
        # Calculate percentage of non-zero intensities
        total_samples = len(data)
        non_zero_count = sum(v for k, v in intensity_counts.items() if k > 0)
        non_zero_percentage = (non_zero_count / total_samples) * 100 if non_zero_count > 0 else 0
        
        # Find the most common non-zero intensity
        non_zero_intensities = {k: v for k, v in counter.items() if k > 0}
        
        if non_zero_intensities:
            # Find the most common non-zero intensity
            most_common_intensity, most_common_count = max(non_zero_intensities.items(), key=lambda x: (x[1], x[0]))
        else:
            # If all intensities are 0, default to intensity 1
            most_common_intensity = 1
            most_common_count = 0
            
        # Store non-zero percentage
        emotion_percentages[emotion] = non_zero_percentage
        
        # Always use the most common non-zero intensity for this emotion
        majority_intensities[emotion] = most_common_intensity
        
        # Store detailed statistics
        distribution_stats[emotion] = {
            "most_common_non_zero_intensity": most_common_intensity,
            "most_common_count": most_common_count,
            "non_zero_count": non_zero_count,
            "non_zero_percentage": non_zero_percentage,
            "total_samples": total_samples,
            "intensity_distribution": intensity_counts
        }
    
    # Find the emotion with highest percentage of non-zero intensities (for informational purposes)
    if emotion_percentages:
        top_emotion = max(emotion_percentages.items(), key=lambda x: x[1])[0]
        distribution_stats[top_emotion]["is_top_emotion"] = True
    
    return majority_intensities, distribution_stats

def evaluate_track_a(test_data: List[Dict], majority_classes: Dict[str, int]) -> Dict:
    """
    Evaluate the majority baseline for Track A.
    
    Args:
        test_data: List of test samples
        majority_classes: Dictionary mapping each emotion to its majority class
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(test_data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Prepare true and predicted labels for each emotion
    y_true_all = []
    y_pred_all = []
    emotion_results = {}
    
    for emotion in available_emotions:
        # Skip if this emotion isn't in the majority_classes dictionary
        if emotion not in majority_classes:
            continue
            
        y_true = np.array([int(sample[emotion]) for sample in test_data])
        y_pred = np.array([majority_classes[emotion]] * len(test_data))
        
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
        # Calculate F1 score for this emotion (same way as regular_lms_track_ab.py)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        emotion_results[emotion] = f1
    
    # Stack arrays for all emotions
    y_true_stacked = np.vstack(y_true_all).T  # Shape becomes (n_samples, n_emotions)
    y_pred_stacked = np.vstack(y_pred_all).T
    
    # Calculate individual F1 scores using the same approach as in regular_lms_track_ab.py
    f1_indiv = f1_score(y_true_stacked, y_pred_stacked, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1_indiv))
    
    results = {
        "per_emotion_f1": emotion_results,
        "macro_f1": macro_f1,
        "majority_classes": majority_classes
    }
    
    return results

def evaluate_track_b(test_data: List[Dict], majority_intensities: Dict[str, int]) -> Dict:
    """
    Evaluate the majority baseline for Track B.
    
    Args:
        test_data: List of test samples
        majority_intensities: Dictionary mapping each emotion to its majority intensity
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(test_data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Prepare true and predicted intensities for each emotion
    emotion_results = {}
    pearson_vals = []
    
    for emotion in available_emotions:
        # Skip if this emotion isn't in the majority_intensities dictionary
        if emotion not in majority_intensities:
            continue
            
        y_true = np.array([int(sample[emotion]) for sample in test_data])
        y_pred = np.array([majority_intensities[emotion]] * len(test_data))
        
        # Calculate Pearson correlation for this emotion
        # Need to add some noise/variation to the predictions to get a non-zero correlation
        # Add very small random noise to predictions (doesn't affect integer predictions)
        y_pred_with_noise = y_pred + np.random.normal(0, 0.001, size=len(y_pred))
        
        # Calculate correlation
        if np.std(y_true) == 0:  # If true values are all the same
            correlation = 0
        else:
            try:
                correlation, _ = pearsonr(y_true, y_pred_with_noise)
                # If correlation is NaN, set to 0
                if np.isnan(correlation):
                    correlation = 0
            except:
                correlation = 0
        
        emotion_results[emotion] = correlation
        pearson_vals.append(correlation)
    
    # Calculate average correlation across emotions (matching regular_lms_track_ab.py)
    avg_correlation = float(np.mean(pearson_vals))
    
    results = {
        "per_emotion_correlation": emotion_results,
        "pearsonr_macro_overall": avg_correlation,  # Use same key as in regular_lms_track_ab.py
        "avg_correlation": avg_correlation,         # Keep original for compatibility
        "majority_intensities": majority_intensities
    }
    
    return results

def run_track_a_baseline(train_dir: str, test_dir: str, dev_dir: str = None, find_optimal: bool = True) -> Dict:
    """
    Run the majority baseline for Track A across all languages.
    Optionally find the optimal number of emotions to predict for each language.
    
    Args:
        train_dir: Directory containing training CSV files
        test_dir: Directory containing test CSV files
        dev_dir: Optional directory containing development CSV files for optimization
        find_optimal: Whether to find the optimal number of emotions (True) or use fixed top 2 (False)
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(train_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        # Skip if test file doesn't exist
        if not os.path.exists(test_path):
            print(f"Test file for {language} not found, skipping.")
            continue
        
        # Read training and test data
        train_data = read_csv_file(train_path)
        test_data = read_csv_file(test_path)
        
        # Read development data if provided and available
        dev_data = None
        if dev_dir is not None:
            dev_path = os.path.join(dev_dir, filename)
            if os.path.exists(dev_path):
                dev_data = read_csv_file(dev_path)
        
        # Get majority classes and distribution stats from training data
        majority_classes, distribution_stats = get_majority_class_track_a(train_data, dev_data, find_optimal)
        
        # Find the top emotions (with is_top_emotion = True)
        top_emotions = [emotion for emotion, stats in distribution_stats.items() 
                      if stats.get("is_top_emotion", False)]
        
        # Get stats from distribution_stats
        first_emotion = list(distribution_stats.keys())[0]
        avg_non_zero_emotions = distribution_stats[first_emotion]["avg_non_zero_emotions"]
        instances_with_emotions = distribution_stats[first_emotion]["instances_with_emotions"]
        total_instances = distribution_stats[first_emotion]["total_instances"]
        predicted_top_count = distribution_stats[first_emotion]["predicted_top_count"]
        optimization_method = distribution_stats[first_emotion]["optimization_method"]
        
        # Get optimization results if available
        best_f1 = distribution_stats[first_emotion].get("best_f1")
        all_scores = distribution_stats[first_emotion].get("all_scores")
        
        # Print distribution statistics
        print(f"\n===== Track A - {language} Class Distribution =====")
        print(f"Average non-zero emotions per instance: {avg_non_zero_emotions:.2f}")
        print(f"Instances with emotions: {instances_with_emotions}/{total_instances} ({instances_with_emotions/total_instances*100:.1f}%)")
        
        if optimization_method == "optimized":
            print(f"Optimization method: {optimization_method}")
            print(f"Optimal number of emotions: {predicted_top_count} (best F1: {best_f1:.4f})")
            
            # Print all scores if available
            if all_scores:
                print("All tested configurations:")
                for count, score in sorted(all_scores.items()):
                    print(f"  Top {count} emotions: F1 = {score:.4f}" + (" [BEST]" if count == predicted_top_count else ""))
        else:
            print(f"Using fixed top {predicted_top_count} emotions")
        
        for emotion, stats in distribution_stats.items():
            predicted_class = stats["predicted_class"]
            predicted_count = stats["predicted_count"]
            predicted_pct = stats["predicted_percentage"]
            rank = stats["rank"]
            total = stats["total_samples"]
            class_dist = stats["class_distribution"]
            is_top = stats.get("is_top_emotion", False)
            
            # Format class distribution for printing
            dist_str = ", ".join([f"Class {cls}: {count} ({count/total*100:.1f}%)" 
                               for cls, count in class_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Non-zero entries: Class 1 with {predicted_count}/{total} examples ({predicted_pct:.1f}%)")
            print(f"  Distribution: {dist_str}")
            print(f"  Rank: {rank} of {len(distribution_stats)} emotions")
            if is_top:
                print(f"  Will predict: {predicted_class} (in top {predicted_top_count} emotions)")
            else:
                print(f"  Will predict: {predicted_class} (not in top {predicted_top_count} emotions)")
        
        print(f"  Selected top {predicted_top_count} emotions: {', '.join(top_emotions)}")
        
        # Evaluate on test data
        lang_results = evaluate_track_a(test_data, majority_classes)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["top_emotions"] = top_emotions
        lang_results["avg_non_zero_emotions"] = avg_non_zero_emotions
        lang_results["instances_with_emotions"] = instances_with_emotions
        lang_results["total_instances"] = total_instances
        lang_results["predicted_top_count"] = predicted_top_count
        lang_results["optimization_method"] = optimization_method
        if best_f1 is not None:
            lang_results["best_dev_f1"] = best_f1
        if all_scores is not None:
            lang_results["all_dev_scores"] = all_scores
        
        # Add f1_macro to match the key used in regular_lms_track_ab.py
        lang_results["f1_macro"] = lang_results["macro_f1"]
        results[language] = lang_results
        
        print(f"  Test Performance: Macro-F1 = {lang_results['macro_f1']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_macro_f1 = sum(lang_results['macro_f1'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "macro_f1": avg_macro_f1,
        "f1_macro": avg_macro_f1  # Add f1_macro to match the key used in regular_lms_track_ab.py
    }
    
    print(f"Track A - Average Macro-F1 across languages: {avg_macro_f1:.4f}")
    
    return results

def run_track_b_baseline(train_dir: str, test_dir: str) -> Dict:
    """
    Run the majority baseline for Track B across all languages.
    Predict the most common non-zero intensity for each emotion.
    
    Args:
        train_dir: Directory containing training CSV files
        test_dir: Directory containing test CSV files
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(train_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        # Skip if test file doesn't exist
        if not os.path.exists(test_path):
            print(f"Test file for {language} not found, skipping.")
            continue
        
        # Read data
        train_data = read_csv_file(train_path)
        test_data = read_csv_file(test_path)
        
        # Get majority intensities and distribution stats from training data
        majority_intensities, distribution_stats = get_majority_intensity_track_b(train_data)
        
        # Find the top emotion (with is_top_emotion = True) for informational purposes
        top_emotion = next((emotion for emotion, stats in distribution_stats.items() 
                         if stats.get("is_top_emotion", False)), None)
        
        # Print distribution statistics
        print(f"\n===== Track B - {language} Intensity Distribution =====")
        for emotion, stats in distribution_stats.items():
            is_top = stats.get("is_top_emotion", False)
            most_common_intensity = stats["most_common_non_zero_intensity"]
            most_common_count = stats["most_common_count"]
            non_zero_count = stats["non_zero_count"]
            non_zero_pct = stats["non_zero_percentage"]
            total = stats["total_samples"]
            intensity_dist = stats["intensity_distribution"]
            
            # Format intensity distribution for printing
            dist_str = ", ".join([f"Intensity {intensity}: {count} ({count/total*100:.1f}%)" 
                               for intensity, count in intensity_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Distribution: {dist_str}")
            print(f"  Non-zero values: {non_zero_count}/{total} examples ({non_zero_pct:.1f}%)")
            
            if most_common_count > 0:
                print(f"  Will predict: {most_common_intensity} (most common non-zero intensity with {most_common_count} examples)")
            else:
                print(f"  No non-zero examples, defaulting to intensity: {most_common_intensity}")
            
            if is_top:
                print(f"  Note: This is the emotion with highest non-zero percentage")
        
        # Evaluate on test data
        lang_results = evaluate_track_b(test_data, majority_intensities)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["top_emotion"] = top_emotion
        results[language] = lang_results
        
        print(f"  Performance: Avg Correlation = {lang_results['pearsonr_macro_overall']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_correlation = sum(lang_results['pearsonr_macro_overall'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "avg_correlation": avg_correlation,
        "pearsonr_macro_overall": avg_correlation  # Match key used in regular_lms_track_ab.py
    }
    
    print(f"Track B - Average Correlation across languages: {avg_correlation:.4f}")
    
    return results

def run_track_c_baseline(test_dir: str, find_optimal: bool = True) -> Dict:
    """
    Run the majority baseline for Track C using test set distributions.
    Optionally find the optimal number of emotions to predict for each language.
    
    Args:
        test_dir: Directory containing test CSV files
        find_optimal: Whether to find the optimal number of emotions (True) or use fixed top 2 (False)
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(test_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        test_path = os.path.join(test_dir, filename)
        
        # Read test data
        test_data = read_csv_file(test_path)
        
        # Get majority classes and distribution stats from test data
        # For Track C, we only have test data, so we use it for both training and testing
        majority_classes, distribution_stats = get_majority_class_track_a(test_data, test_data, find_optimal)
        
        # Find the top emotions (with is_top_emotion = True)
        top_emotions = [emotion for emotion, stats in distribution_stats.items() 
                      if stats.get("is_top_emotion", False)]
        
        # Get stats from distribution_stats
        first_emotion = list(distribution_stats.keys())[0]
        avg_non_zero_emotions = distribution_stats[first_emotion]["avg_non_zero_emotions"]
        instances_with_emotions = distribution_stats[first_emotion]["instances_with_emotions"]
        total_instances = distribution_stats[first_emotion]["total_instances"]
        predicted_top_count = distribution_stats[first_emotion]["predicted_top_count"]
        optimization_method = distribution_stats[first_emotion]["optimization_method"]
        
        # Get optimization results if available
        best_f1 = distribution_stats[first_emotion].get("best_f1")
        all_scores = distribution_stats[first_emotion].get("all_scores")
        
        # Print distribution statistics
        print(f"\n===== Track C - {language} Class Distribution =====")
        print(f"Average non-zero emotions per instance: {avg_non_zero_emotions:.2f}")
        print(f"Instances with emotions: {instances_with_emotions}/{total_instances} ({instances_with_emotions/total_instances*100:.1f}%)")
        
        if optimization_method == "optimized":
            print(f"Optimization method: {optimization_method}")
            print(f"Optimal number of emotions: {predicted_top_count} (best F1: {best_f1:.4f})")
            
            # Print all scores if available
            if all_scores:
                print("All tested configurations:")
                for count, score in sorted(all_scores.items()):
                    print(f"  Top {count} emotions: F1 = {score:.4f}" + (" [BEST]" if count == predicted_top_count else ""))
        else:
            print(f"Using fixed top {predicted_top_count} emotions")
        
        for emotion, stats in distribution_stats.items():
            predicted_class = stats["predicted_class"]
            predicted_count = stats["predicted_count"]
            predicted_pct = stats["predicted_percentage"]
            rank = stats["rank"]
            total = stats["total_samples"]
            class_dist = stats["class_distribution"]
            is_top = stats.get("is_top_emotion", False)
            
            # Format class distribution for printing
            dist_str = ", ".join([f"Class {cls}: {count} ({count/total*100:.1f}%)" 
                               for cls, count in class_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Non-zero entries: Class 1 with {predicted_count}/{total} examples ({predicted_pct:.1f}%)")
            print(f"  Distribution: {dist_str}")
            print(f"  Rank: {rank} of {len(distribution_stats)} emotions")
            if is_top:
                print(f"  Will predict: {predicted_class} (in top {predicted_top_count} emotions)")
            else:
                print(f"  Will predict: {predicted_class} (not in top {predicted_top_count} emotions)")
        
        print(f"  Selected top {predicted_top_count} emotions: {', '.join(top_emotions)}")
        
        # Evaluate on test data - for Track C, it's the same as the "training" data
        lang_results = evaluate_track_a(test_data, majority_classes)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["top_emotions"] = top_emotions
        lang_results["avg_non_zero_emotions"] = avg_non_zero_emotions
        lang_results["instances_with_emotions"] = instances_with_emotions
        lang_results["total_instances"] = total_instances
        lang_results["predicted_top_count"] = predicted_top_count
        lang_results["optimization_method"] = optimization_method
        if best_f1 is not None:
            lang_results["best_f1"] = best_f1
        if all_scores is not None:
            lang_results["all_scores"] = all_scores
        
        # Add f1_macro to match the key used in regular_lms_track_c.py
        lang_results["f1_macro"] = lang_results["macro_f1"]
        results[language] = lang_results
        
        print(f"  Performance: Macro-F1 = {lang_results['macro_f1']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_macro_f1 = sum(lang_results['macro_f1'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "macro_f1": avg_macro_f1,
        "f1_macro": avg_macro_f1  # Add f1_macro to match the key used in regular_lms_track_c.py
    }
    
    print(f"Track C - Average Macro-F1 across languages: {avg_macro_f1:.4f}")
    
    return results

def save_results_to_csv(results, track, output_path):
    """
    Save results to CSV file with the specified format.
    
    Args:
        results: Dictionary of results
        track: Track identifier (A, B, C)
        output_path: Path to save the CSV file
    """
    # Get all languages except "average"
    languages = [lang for lang in results.keys() if lang != "average"]
    
    # Collect all possible emotions across all languages
    all_emotions = set()
    for language in languages:
        if track in ["A", "C"]:
            all_emotions.update(results[language]["per_emotion_f1"].keys())
        else:  # Track B
            all_emotions.update(results[language]["per_emotion_correlation"].keys())
    
    # Sort emotions to ensure consistent order
    emotions = sorted(list(all_emotions))
    
    # Check if this is a random baseline (check for presence of random_classes or random_intensities)
    is_random = "random_classes" in results[languages[0]] or "random_intensities" in results[languages[0]]
    
    # Prepare header row
    if track in ["A", "C"]:
        header = ["language", "macro_f1"]
        # Add emotion-specific F1 columns
        header.extend([f"{emotion}_f1" for emotion in emotions])
        # Removed top_emotion column and {emotion}_prediction columns
    else:  # Track B
        header = ["language", "avg_correlation"]
        # Add emotion-specific correlation columns
        header.extend([f"{emotion}_correlation" for emotion in emotions])
        # Removed top_emotion column and {emotion}_prediction columns
    
    # Prepare rows
    rows = [header]
    for language in languages:
        lang_results = results[language]
        
        if track in ["A", "C"]:
            row = [language, f"{lang_results['macro_f1']:.4f}"]
            
            # Add emotion-specific F1 scores, handling missing emotions
            for emotion in emotions:
                if emotion in lang_results["per_emotion_f1"]:
                    row.append(f"{lang_results['per_emotion_f1'][emotion]:.4f}")
                else:
                    row.append("")  # Empty for missing emotions
            
            # Removed prediction information section
                
        else:  # Track B
            row = [language, f"{lang_results['avg_correlation']:.4f}"]
            
            # Add emotion-specific correlations, handling missing emotions
            for emotion in emotions:
                if emotion in lang_results["per_emotion_correlation"]:
                    row.append(f"{lang_results['per_emotion_correlation'][emotion]:.4f}")
                else:
                    row.append("")  # Empty for missing emotions
            
            # Removed intensity information section
        
        rows.append(row)
    
    # Add average row if available
    if "average" in results:
        if track in ["A", "C"]:
            avg_row = ["AVERAGE", f"{results['average']['macro_f1']:.4f}"]
            # Fill remaining columns with empty strings
            avg_row.extend([""] * (len(header) - len(avg_row)))
        else:  # Track B
            avg_row = ["AVERAGE", f"{results['average']['avg_correlation']:.4f}"]
            # Fill remaining columns with empty strings
            avg_row.extend([""] * (len(header) - len(avg_row)))
        rows.append(avg_row)
    
    # Write to CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def get_random_predictions_track_a(data: List[Dict]) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Generate random binary predictions (0 or 1) for Track A/C data.
    
    Args:
        data: List of data samples
        
    Returns:
        Tuple containing:
            - Dictionary mapping each emotion to a random class (0 or 1)
            - Dictionary with detailed statistics
    """
    np.random.seed(42)  # For reproducibility
    distribution_stats = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Generate random predictions
    random_classes = {}
    
    for emotion in available_emotions:
        # Randomly choose 0 or 1 for each emotion
        random_class = np.random.randint(0, 2)  # Random integer: 0 or 1
        random_classes[emotion] = random_class
        
        # Count occurrences of 0 and 1 in true data for reference
        counter = Counter([int(sample[emotion]) for sample in data])
        class_counts = {k: v for k, v in counter.items()}
        
        # Get counts for class 1
        predicted_count = class_counts.get(1, 0)
        
        # Calculate percentage 
        total_samples = len(data)
        predicted_percentage = (predicted_count / total_samples) * 100
        
        # Store detailed statistics
        distribution_stats[emotion] = {
            "random_class": random_class,
            "predicted_count": predicted_count,
            "predicted_percentage": predicted_percentage,
            "total_samples": total_samples,
            "class_distribution": class_counts
        }
    
    return random_classes, distribution_stats

def get_random_predictions_track_b(data: List[Dict]) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Generate random intensity predictions (0-3) for Track B data.
    
    Args:
        data: List of data samples
        
    Returns:
        Tuple containing:
            - Dictionary mapping each emotion to a random intensity (0-3)
            - Dictionary with detailed statistics
    """
    np.random.seed(42)  # For reproducibility
    distribution_stats = {}
    
    # Get the list of emotions present in this dataset
    available_emotions = list(data[0].keys())
    available_emotions = [e for e in available_emotions if e != "id" and e != "text"]
    
    # Generate random predictions
    random_intensities = {}
    
    for emotion in available_emotions:
        # Randomly choose an intensity from 0 to 3 for each emotion
        random_intensity = np.random.randint(0, 4)  # Random integer: 0, 1, 2, or 3
        random_intensities[emotion] = random_intensity
        
        # Count occurrences of each intensity in true data for reference
        counter = Counter([int(sample[emotion]) for sample in data])
        intensity_counts = {k: v for k, v in counter.items()}
        
        # Calculate statistics about non-zero values
        total_samples = len(data)
        non_zero_count = sum(v for k, v in intensity_counts.items() if k > 0)
        non_zero_percentage = (non_zero_count / total_samples) * 100 if non_zero_count > 0 else 0
        
        # Store detailed statistics
        distribution_stats[emotion] = {
            "random_intensity": random_intensity,
            "non_zero_count": non_zero_count,
            "non_zero_percentage": non_zero_percentage,
            "total_samples": total_samples,
            "intensity_distribution": intensity_counts
        }
    
    return random_intensities, distribution_stats

def run_track_a_random_baseline(train_dir: str, test_dir: str) -> Dict:
    """
    Run a random baseline for Track A across all languages.
    Randomly predict 0 or 1 for each emotion.
    
    Args:
        train_dir: Directory containing training CSV files
        test_dir: Directory containing test CSV files
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(train_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        # Skip if test file doesn't exist
        if not os.path.exists(test_path):
            print(f"Test file for {language} not found, skipping.")
            continue
        
        # Read data
        train_data = read_csv_file(train_path)
        test_data = read_csv_file(test_path)
        
        # Get random predictions for this language
        random_classes, distribution_stats = get_random_predictions_track_a(train_data)
        
        # Print distribution statistics
        print(f"\n===== Track A - {language} Random Baseline =====")
        for emotion, stats in distribution_stats.items():
            random_class = stats["random_class"]
            predicted_count = stats["predicted_count"]
            predicted_pct = stats["predicted_percentage"]
            total = stats["total_samples"]
            class_dist = stats["class_distribution"]
            
            # Format class distribution for printing
            dist_str = ", ".join([f"Class {cls}: {count} ({count/total*100:.1f}%)" 
                               for cls, count in class_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Ground truth: Class 1 with {predicted_count}/{total} examples ({predicted_pct:.1f}%)")
            print(f"  Distribution: {dist_str}")
            print(f"  Will predict: {random_class} (randomly generated)")
        
        # Evaluate on test data
        lang_results = evaluate_track_a(test_data, random_classes)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["random_classes"] = random_classes
        # Add f1_macro to match the key used in regular_lms_track_ab.py
        lang_results["f1_macro"] = lang_results["macro_f1"]
        results[language] = lang_results
        
        print(f"  Performance: Macro-F1 = {lang_results['macro_f1']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_macro_f1 = sum(lang_results['macro_f1'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "macro_f1": avg_macro_f1,
        "f1_macro": avg_macro_f1  # Add f1_macro to match the key used in regular_lms_track_ab.py
    }
    
    print(f"Track A - Random Baseline - Average Macro-F1: {avg_macro_f1:.4f}")
    
    return results

def run_track_b_random_baseline(train_dir: str, test_dir: str) -> Dict:
    """
    Run a random baseline for Track B across all languages.
    Randomly predict intensities 0-3 for each emotion.
    
    Args:
        train_dir: Directory containing training CSV files
        test_dir: Directory containing test CSV files
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(train_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        # Skip if test file doesn't exist
        if not os.path.exists(test_path):
            print(f"Test file for {language} not found, skipping.")
            continue
        
        # Read data
        train_data = read_csv_file(train_path)
        test_data = read_csv_file(test_path)
        
        # Get random predictions for this language
        random_intensities, distribution_stats = get_random_predictions_track_b(train_data)
        
        # Print distribution statistics
        print(f"\n===== Track B - {language} Random Baseline =====")
        for emotion, stats in distribution_stats.items():
            random_intensity = stats["random_intensity"]
            non_zero_count = stats["non_zero_count"]
            non_zero_pct = stats["non_zero_percentage"]
            total = stats["total_samples"]
            intensity_dist = stats["intensity_distribution"]
            
            # Format intensity distribution for printing
            dist_str = ", ".join([f"Intensity {intensity}: {count} ({count/total*100:.1f}%)" 
                               for intensity, count in intensity_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Ground truth distribution: {dist_str}")
            print(f"  Non-zero values: {non_zero_count}/{total} examples ({non_zero_pct:.1f}%)")
            print(f"  Will predict: {random_intensity} (randomly generated)")
        
        # Evaluate on test data
        lang_results = evaluate_track_b(test_data, random_intensities)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["random_intensities"] = random_intensities
        results[language] = lang_results
        
        print(f"  Performance: Avg Correlation = {lang_results['pearsonr_macro_overall']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_correlation = sum(lang_results['pearsonr_macro_overall'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "avg_correlation": avg_correlation,
        "pearsonr_macro_overall": avg_correlation  # Match key used in regular_lms_track_ab.py
    }
    
    print(f"Track B - Random Baseline - Average Correlation: {avg_correlation:.4f}")
    
    return results

def run_track_c_random_baseline(test_dir: str) -> Dict:
    """
    Run a random baseline for Track C using test set distributions.
    Randomly predict 0 or 1 for each emotion.
    
    Args:
        test_dir: Directory containing test CSV files
        
    Returns:
        Dictionary with results for each language
    """
    results = {}
    
    # Process each language file
    for filename in os.listdir(test_dir):
        if not filename.endswith('.csv'):
            continue
            
        language = filename.split('.')[0]
        test_path = os.path.join(test_dir, filename)
        
        # Read test data
        test_data = read_csv_file(test_path)
        
        # Get random predictions for this language
        random_classes, distribution_stats = get_random_predictions_track_a(test_data)
        
        # Print distribution statistics
        print(f"\n===== Track C - {language} Random Baseline =====")
        for emotion, stats in distribution_stats.items():
            random_class = stats["random_class"]
            predicted_count = stats["predicted_count"]
            predicted_pct = stats["predicted_percentage"]
            total = stats["total_samples"]
            class_dist = stats["class_distribution"]
            
            # Format class distribution for printing
            dist_str = ", ".join([f"Class {cls}: {count} ({count/total*100:.1f}%)" 
                               for cls, count in class_dist.items()])
            
            print(f"Emotion '{emotion}':")
            print(f"  Ground truth: Class 1 with {predicted_count}/{total} examples ({predicted_pct:.1f}%)")
            print(f"  Distribution: {dist_str}")
            print(f"  Will predict: {random_class} (randomly generated)")
        
        # Evaluate on test data
        lang_results = evaluate_track_a(test_data, random_classes)
        lang_results["distribution_stats"] = distribution_stats
        lang_results["random_classes"] = random_classes
        # Add f1_macro to match the key used in regular_lms_track_c.py
        lang_results["f1_macro"] = lang_results["macro_f1"]
        results[language] = lang_results
        
        print(f"  Performance: Macro-F1 = {lang_results['macro_f1']:.4f}")
        print()
    
    # Calculate average across all languages
    avg_macro_f1 = sum(lang_results['macro_f1'] for lang_results in results.values()) / len(results)
    results["average"] = {
        "macro_f1": avg_macro_f1,
        "f1_macro": avg_macro_f1  # Add f1_macro to match the key used in regular_lms_track_c.py
    }
    
    print(f"Track C - Random Baseline - Average Macro-F1: {avg_macro_f1:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation for BRIGHTER")
    parser.add_argument("--track", type=str, required=True, choices=["A", "B", "C"], 
                        help="Track to evaluate (A, B, or C)")
    parser.add_argument("--baseline", type=str, default="majority", choices=["majority", "random"],
                        help="Type of baseline to run (majority or random)")
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                        help="Directory to save results")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="Find optimal number of emotions for each language")
    parser.add_argument("--fixed_top", type=int, default=2,
                        help="Fixed number of top emotions to predict when not optimizing")
    parser.add_argument("--dev_dir", type=str, default=None,
                        help="Directory containing development data for optimization (only for track A)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set optimization flag
    find_optimal = args.optimize
    
    # Determine output file suffix
    suffix = ""
    if args.baseline == "majority":
        suffix = "_optimal" if find_optimal else f"_top{args.fixed_top}"
    
    # Run appropriate baseline
    if args.baseline == "majority":
        # Majority baseline
        if args.track == "A":
            print(f"Running Track A majority baseline{' (optimized)' if find_optimal else f' (fixed top {args.fixed_top})'}...")
            # Set development directory
            dev_dir = args.dev_dir
            if dev_dir is None and find_optimal:
                dev_dir = "./track_a/dev"  # Default dev directory for track A
            
            results = run_track_a_baseline(
                train_dir="./track_a/train", 
                test_dir="./track_a/test",
                dev_dir=dev_dir,
                find_optimal=find_optimal
            )
            output_json_path = os.path.join(args.output_dir, f"track_a_majority{suffix}_results.json")
            output_csv_path = os.path.join(args.output_dir, f"track_a_majority{suffix}_results.csv")
            
        elif args.track == "B":
            print("Running Track B majority baseline...")
            results = run_track_b_baseline(
                train_dir="./track_b/train", 
                test_dir="./track_b/test"
            )
            output_json_path = os.path.join(args.output_dir, f"track_b_majority_results.json")
            output_csv_path = os.path.join(args.output_dir, f"track_b_majority_results.csv")
            
        elif args.track == "C":
            print(f"Running Track C majority baseline{' (optimized)' if find_optimal else f' (fixed top {args.fixed_top})'}...")
            results = run_track_c_baseline(
                test_dir="./track_c/test",
                find_optimal=find_optimal
            )
            output_json_path = os.path.join(args.output_dir, f"track_c_majority{suffix}_results.json")
            output_csv_path = os.path.join(args.output_dir, f"track_c_majority{suffix}_results.csv")
    else:
        # Random baseline
        if args.track == "A":
            print("Running Track A random baseline...")
            results = run_track_a_random_baseline(
                train_dir="./track_a/train", 
                test_dir="./track_a/test"
            )
            output_json_path = os.path.join(args.output_dir, "track_a_random_results.json")
            output_csv_path = os.path.join(args.output_dir, "track_a_random_results.csv")
            
        elif args.track == "B":
            print("Running Track B random baseline...")
            results = run_track_b_random_baseline(
                train_dir="./track_b/train", 
                test_dir="./track_b/test"
            )
            output_json_path = os.path.join(args.output_dir, "track_b_random_results.json")
            output_csv_path = os.path.join(args.output_dir, "track_b_random_results.csv")
            
        elif args.track == "C":
            print("Running Track C random baseline...")
            results = run_track_c_random_baseline(
                test_dir="./track_c/test"
            )
            output_json_path = os.path.join(args.output_dir, "track_c_random_results.json")
            output_csv_path = os.path.join(args.output_dir, "track_c_random_results.csv")
    
    # Save results in both formats
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save results as CSV
    save_results_to_csv(results, args.track, output_csv_path)
    
    print(f"Results saved to {output_json_path} and {output_csv_path}")

if __name__ == "__main__":
    main()