#Pre-requisite packages
import math
import argparse
from matplotlib.patches import Patch
import pandas as pd
import os
import logging
import numpy as np
import datetime
from numpy.random import permutation
from scipy.signal import square
from scipy.spatial.distance import squareform
from skbio.diversity import beta_diversity, alpha_diversity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import kruskal, mannwhitneyu, alpha, fisher_exact, shapiro
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.plotting import acorr_plot

#Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('evaluation.log'), logging.StreamHandler()
                    ])

#section##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Baseline & Evaluation Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

def normalize_taxonomy(name):
    return name.strip().lower()

def baseline_comparison(mock_tsv, blank_tsv, original_tsv):
    """
    Determines suspected contaminants using mock and blank reference communities.
    Identifies "other" species found in the samples but not in the mock or blank - Biologically relevant.

    :param mock_tsv:
    :param blank_tsv:
    :param original_tsv:

    :return: Dictionary with :
        positive_recall: Expected (mock) species present in the sample (grouped by genus_species with summed abundance)
        missing_expected: Expected mock species missing from the sample

        contaminants_in_sample: Suspected contaminants present in the sample (grouped with summed abundance and a contaminant level)
        Contaminant level: Low abundance (<10), Medium abundance (10-100), High abundance (>100), No abundance (<0)

        others:Species in the sample that are not found in either mock or blank (grouped with summed abundance)
    """
    logging.info("Start baseline identification for sample file %s", original_tsv)

    #Load each TSV as a Dataframe
    dfs = {
        "mock": pd.read_csv(mock_tsv, sep='\t'),
        "blank": pd.read_csv(blank_tsv, sep='\t'),
        "original": pd.read_csv(original_tsv, sep='\t')
    }

    logging.info("Cleaning up columns and introducing species columns")
    alt_names = ["abundance_sample", "raw_abundance"]
    # Always require taxonomy and abundance columns
    for name, df in dfs.items():
        if 'taxonomy' in df.columns:
            df['taxonomy'] = df["taxonomy"].apply(normalize_taxonomy)
        if "abundance" not in df.columns:
            for alt in alt_names:
                if alt in df.columns:
                    df = df.rename(columns={alt: "abundance"})
                    logging.info(f"Renamed column '{alt}' to 'abundance' in {name} Dataframe")
                    break
        for col in ['taxonomy', 'abundance']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col} in {name} Dataframe") #If the required columns are not present, throw error.

        # Convert 'abundance' to numeric and fill any invalid entries with 0
        df['abundance'] = pd.to_numeric(df['abundance'], errors='coerce').fillna(0) #Fill invalid entries with 0
        #df['taxonomy'] = df['taxonomy'].str.strip().str.lower() #Clean up.

        # If this is the blank, also process 'prevalence' if available
        if name == "blank" and "prevalence" in df.columns:
            df['prevalence'] = pd.to_numeric(df['prevalence'], errors='coerce').fillna(0)

        if df["abundance"].sum() <= 0:
            raise ValueError(f"Total abundance should not be zero in the {name} Dataframe")

        # Extract genus and species (assumes taxonomy is space-delimited)
        df["genus"] = df["taxonomy"].apply(lambda x: x.split(" ")[0].strip())
        df["species"] = df["taxonomy"].apply(lambda x: x.split(" ")[-1].strip())
        df["genus_species"] = df["genus"] + " " + df["species"]
        df["check_ids"] = df.apply(lambda row: row["genus"] if row["species"] in ["sp."] else row["genus_species"], axis=1)
        dfs[name] = df #Update the dictionary

    # Create sets of species from the mock, blank and the sample.
    expected_ids = set(dfs['mock']['check_ids'])
    blank_ids = set(dfs['blank']['check_ids'])
    original_ids = set(dfs['original']['check_ids'])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mock species ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Computes positive recall: Expected (mock) species that are present in the sample
    positive_recall_ids = expected_ids.intersection(original_ids)
    missing_expected_ids = expected_ids - original_ids

    # Get corresponding "genus species" strings for output
    dfs['original'] = dfs['original'].sort_values(by='abundance', ascending=False)
    pos_recall_info = dfs['original'][dfs['original']['check_ids'].isin(positive_recall_ids)][['genus_species', 'abundance']]
    pos_recall_info = pos_recall_info.groupby("genus_species", as_index=False).agg({"abundance": "sum"}) #Summed abundance

    missing_expected_info = dfs['mock'][dfs['mock']['check_ids'].isin(missing_expected_ids)][['genus_species']].drop_duplicates()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Contaminants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #Contaminants (contaminants_info: negative recall: Species in the blank (not found in the mock) that occur in the samples
    potential_contaminants_ids = (blank_ids.intersection(original_ids))- expected_ids
        #Find species present in both blank and sample
        # Then from those, remove any present in the mock ==> Leaving only confirmed contaminants.

    contaminants_info = dfs['original'][dfs['original']['check_ids'].isin(potential_contaminants_ids)]
    contaminants_info = contaminants_info.groupby("genus_species", as_index=False).agg({"abundance": "sum"})

    logging.info("Labelling blank contaminants based on abundance thresholds (No: 0, Low: <10, Medium: 10-100, High: >100)")
    def label_abundance(x):
        if x >= 100:
            return "High abundance contaminant"
        elif x >= 10 and x < 100:
            return "Medium abundance contaminant"
        elif x < 10 and x > 0:
            return "Low abundance contaminant"
        else:
            return "No abundance"

    contaminants_info["contaminant_level"] = contaminants_info["abundance"].apply(label_abundance)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Other species ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~ Other species: Species in the sample, not in either mock or blank - Biologically relevant ~~#
    #Further, they are not confirmed to be contaminants or not, just should be retained as not confirmed #

    ref_ids = expected_ids.union(blank_ids)
    others_ids = original_ids - ref_ids
    others_info = dfs['original'][dfs['original']['check_ids'].isin(others_ids)][['genus_species', 'abundance']]
    others_info = others_info.groupby("genus_species", as_index=False).agg({"abundance": "sum"})

    logging.info("Baseline identification complete for sample %s", original_tsv) #~Debugging
    print("Expected species present in sample (positive recall):")
    print(pos_recall_info.to_string(index=False))
    print("\nExpected species missing from sample:")
    print(missing_expected_info.to_string(index=False))
    print("\nPotential contaminants in sample (negative recall):")
    print(contaminants_info.to_string(index=False))
    print("\nOther species in sample (not in mock or blank):")
    print(others_info.to_string(index=False))

    # Return the results as a dictionary if further processing is needed
    output_dict = {
        "positive_recall": pos_recall_info,
        "missing_expected": missing_expected_info,
        "contaminants_in_sample": contaminants_info,
        "others": others_info
    }
    return output_dict

def save_results(output_dict, output_dir):
    """
    Save each dataframe in the output dictionary as an TSV file in the given directory
    """
    os.makedirs(output_dir, exist_ok=True)
    for key, df in output_dict.items():
        output_filename = os.path.join(output_dir, f"{key}_{timestamp}.tsv")
        df.to_csv(output_filename, sep='\t', index=False)
        logging.info(f"Saving results to {output_filename}")

def evaluate_decontamination(mock_tsv, blank_tsv, pre_decontam_tsv, post_decontam_tsv, prev_post_decontam_tsv, output_dir=None):
    """
    Evaluate decontamination pipeline performance.
        1. For expected (mock) species: Compares pre- and post- decontamination abundances
        2. For contaminants: Determine how many were removed or downweighted (abundance reduction)
        3: For "others": Compare species not found in mock or blank.
        3. Optionally compare new post-decontam to a previous approach (if provided).

    :param mock_tsv:
    :param blank_tsv:
    :param pre_decontam_tsv:
    :param post_decontam_tsv:
    :param prev_post_decontam_tsv:
    :param output_dir:

    :return: Exports TSV files with the comparison results if output_dir is provided.
    :returns:Returns a tuple of DataFrames:
      (expected_compare, contaminants_compare, others_compare, [compare_prev] if applicable)

    """

    logging.info("Start evaluating decontamination performance")
    # Run baseline comparison on pre and post decontaminated samples

    logging.info("Run baseline comparison for pre-decontaminated sample..")
    pre_decontam_results = baseline_comparison(mock_tsv, blank_tsv, pre_decontam_tsv)

    logging.info("Run baseline comparison for post-decontaminated sample..")
    post_decontam_results = baseline_comparison(mock_tsv, blank_tsv, post_decontam_tsv)

    #Compare expected (mock) species
    logging.info("Compare pre- and post-decontaminated samples with expected mock species")
    pre_decontam_expected = pre_decontam_results["positive_recall"]
    post_decontam_expected = post_decontam_results["positive_recall"]

    #Merge df and distinguish the pre vs. post columns with different suffixes
    expected_compare = pd.merge(pre_decontam_expected, post_decontam_expected, on="genus_species", how="outer",
                                suffixes=("_pre_decontam", "_post_decontam")).fillna(0)

    expected_compare["abundance_change"] = expected_compare["abundance_post_decontam"] - expected_compare["abundance_pre_decontam"]
    expected_compare["percent_change"] = expected_compare.apply(
        lambda row: (row["abundance_change"] / row["abundance_pre_decontam"] * 100) if row["abundance_pre_decontam"] > 0 else 0, axis=1
    )

    logging.info("Compare pre- and post-decontaminated samples for suspected blank contaminants")
    #Compare contaminants found in the samples
    pre_decontam_contaminants = pre_decontam_results["contaminants_in_sample"]
    post_decontam_contaminants = post_decontam_results["contaminants_in_sample"]
    contaminants_compare = pd.merge(pre_decontam_contaminants, post_decontam_contaminants, on="genus_species", how="outer", suffixes=("_pre_decontam", "_post_decontam")).fillna(0)

    contaminants_compare["abundance_change"] = contaminants_compare["abundance_post_decontam"] - contaminants_compare["abundance_pre_decontam"]
    contaminants_compare["percent_change"] = contaminants_compare.apply(
        lambda row: (row["abundance_change"] / row["abundance_pre_decontam"] * 100) if row["abundance_pre_decontam"] > 0 else 0, axis=1
    )

    logging.info("Other species comparison (not found in the mock or blank)")
    # Others comparison (species in sample that are not in mock or blank)
    pre_decontam_others = pre_decontam_results["others"]
    post_decontam_others = post_decontam_results["others"]
    others_compare = pd.merge(pre_decontam_others, post_decontam_others, on="genus_species",
                                    how="outer", suffixes=("_pre_decontam", "_post_decontam")).fillna(0)

    others_compare["abundance_change"] = others_compare["abundance_post_decontam"] - others_compare[
        "abundance_pre_decontam"]
    others_compare["percent_change"] = contaminants_compare.apply(
        lambda row: (row["abundance_change"] / row["abundance_pre_decontam"] * 100) if row["abundance_pre_decontam"] > 0 else 0,
        axis=1
    )

    #If there is a previous decontam approach to compare to...
    compare_decontam_approaches_contaminants = None
    compare_decontam_approaches_expected = None
    compare_decontam_approaches_other = None

    if prev_post_decontam_tsv is not None:
        prev_decontam_results = baseline_comparison(mock_tsv, blank_tsv, prev_post_decontam_tsv)
        prev_decontam_contaminants = prev_decontam_results["contaminants_in_sample"]
        new_decontam_contaminants = post_decontam_results["contaminants_in_sample"]

        compare_decontam_approaches_contaminants = pd.merge(new_decontam_contaminants, prev_decontam_contaminants, on="genus_species",
                                               how="outer", suffixes=("_new_decontam", "_prev_decontam")). fillna(0)

        compare_decontam_approaches_contaminants["abundance_change"] = compare_decontam_approaches_contaminants["abundance_new_decontam"] - compare_decontam_approaches_contaminants["abundance_prev_decontam"]
        compare_decontam_approaches_contaminants["percent_change"] = compare_decontam_approaches_contaminants.apply(
            lambda row: (row["abundance_change"] / row["abundance_prev_decontam"] * 100) if row["abundance_prev_decontam"] > 0 else "No abundance",
            axis=1
        )
        #~~Mock species~~#
        prev_decontam_expected = prev_decontam_results["positive_recall"]
        new_decontam_expected = post_decontam_results["positive_recall"]

        compare_decontam_approaches_expected = pd.merge(new_decontam_expected, prev_decontam_expected, on="genus_species",
                                               how="outer", suffixes=("_new_decontam", "_prev_decontam")). fillna(0)

        compare_decontam_approaches_expected["abundance_change"] = compare_decontam_approaches_expected["abundance_new_decontam"] - compare_decontam_approaches_expected["abundance_prev_decontam"]
        compare_decontam_approaches_expected["percent_change"] = compare_decontam_approaches_expected.apply(
            lambda row: (row["abundance_change"] / row["abundance_prev_decontam"] * 100) if row["abundance_prev_decontam"] > 0 else "No abundance",
            axis=1
        )

        #~~ Other species ~~#
        prev_decontam_other = prev_decontam_results["others"]
        new_decontam_other = post_decontam_results["others"]

        compare_decontam_approaches_other = pd.merge(new_decontam_other, prev_decontam_other, on="genus_species",
                                               how="outer", suffixes=("_new_decontam", "_prev_decontam")). fillna(0)

        compare_decontam_approaches_other["abundance_change"] = compare_decontam_approaches_other["abundance_new_decontam"] - compare_decontam_approaches_other["abundance_prev_decontam"]
        compare_decontam_approaches_other["percent_change"] = compare_decontam_approaches_other.apply(
            lambda row: (row["abundance_change"] / row["abundance_prev_decontam"] * 100) if row["abundance_prev_decontam"] > 0 else "No abundance",
            axis=1
        )

    #Print comparisons (for debugging)
    print("\nExpected (mock) species comparison (Pre vs Post decontamination):")
    print(expected_compare.to_string(index=False))

    #print("\nContaminants comparison (Pre vs Post decontamination):")
    print(contaminants_compare.to_string(index=False))

    #print("\nOther species comparison (Pre vs. Post-decontamination):")
    print(others_compare.to_string(index=False))

    if prev_post_decontam_tsv is not None: #~For debugging
        print("\nExpected (mock) species comparison (Prev vs New decontamination pipeline):")
        print(compare_decontam_approaches_expected.to_string(index=False))

        print("\nContaminants comparison (New pipeline decontamination vs Previous pipeline decontamination):")
        print(compare_decontam_approaches_contaminants.to_string(index=False))

        print("\nOther species comparison (New pipeline decontamination vs Previous pipeline decontamination):")
        print(compare_decontam_approaches_other.to_string(index=False))


    #Export results as TSV files if output_dir is provided
    if output_dir:
        logging.info("Saving the evaluation results")
        os.makedirs(output_dir, exist_ok=True)
        expected_compare.to_csv(os.path.join(output_dir, f"expected_compare_{timestamp}.tsv"), sep="\t", index=False)
        contaminants_compare.to_csv(os.path.join(output_dir, f"contaminants_comparison_{timestamp}.tsv"), sep="\t", index=False)
        others_compare.to_csv(os.path.join(output_dir, f"others_comparison_{timestamp}.tsv"), sep="\t", index=False)

        if prev_post_decontam_tsv is not None:
            compare_decontam_approaches_contaminants.to_csv(os.path.join(output_dir, f"prev_vs_new_contaminants_comparison_{timestamp}.tsv"), sep="\t", index=False)
            compare_decontam_approaches_expected.to_csv(os.path.join(output_dir, f"prev_vs_new_expected_comparison_{timestamp}.tsv"), sep="\t", index=False)
            compare_decontam_approaches_other.to_csv(os.path.join(output_dir, f"prev_vs_new_other_comparison_{timestamp}.tsv"), sep="\t", index=False)

    if prev_post_decontam_tsv is not None:
        return expected_compare, contaminants_compare, others_compare, compare_decontam_approaches_contaminants, compare_decontam_approaches_expected, compare_decontam_approaches_other
    else:
        return expected_compare, contaminants_compare, others_compare


#section##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Precision and recall calculations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

def precision_and_recall(mock_df, blank_df, decontaminated_df, output_dir):
    """
    Calculates Precision, Recall, and Youden's index
    The mock community is the reference (known composition, "true" taxa")
    Evaluate how well the decontamination removes the contaminant (blank-only) species
    Ignore the other species (not in mock or blank) in the confusion matrix.

    True positives (TP): Mock species present in the final (post-decontam) sample
    False negatives (FN): Mock species missing in the final sample
    False positives (FP): Contaminant species (blank-only) that remains in the final sample
    True negatives (TN): Contaminant species successfully removed (i.e. Absent in the final sample)
    Others: (Not in the mock or blank) are not counted in these metrics

    Youden's Index: Summary statistic for pipeline performance

    :return: Returns a dictionary with TP, FN, FP, TN, plus precision and recall.
    """
    #Convert abundance to numeric (just in case) and filter out any zero-abundance if you want presence/absence
    decontaminated_df['abundance'] = pd.to_numeric(decontaminated_df['abundance'], errors='coerce').fillna(0)

    #Create sets..
    mock_species = set(mock_df["taxonomy"])
    blank_species = set(blank_df["taxonomy"])
    final_species = set(decontaminated_df[decontaminated_df["abundance"] > 0]['taxonomy'])

    #Identify contaminant set (species in blank but not in mock)
    contaminant_species = blank_species - mock_species

    logging.info("Computing Precision and Recall metrics")
    # True positives: mock species that remain
    TP = len(mock_species.intersection(final_species))
    # False negatives: mock species that got removed
    FN = len(mock_species - final_species)
    # False positives: contaminant species that remain
    FP = len(contaminant_species.intersection(final_species))
    # True negatives: contaminant species that are removed
    TN = len(contaminant_species - final_species)

    # 8) Precision = TP / (TP + FP)
    # 9) Recall = TP / (TP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    logging.info("Calculating Youden's Index (sensitivity + specificity - 1)")
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 #True positive rate
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 #True negative rate
    youdens_index = sensitivity + specificity - 1

    results = {
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "youdens_index": youdens_index
    }

    if output_dir:
        logging.info("Saving the Precision and recall results")
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame([results])
        results_df.to_csv(os.path.join(output_dir, f"precision_recall{timestamp}.tsv"), sep="\t", index=False)
    print(results)
    return results

def fisher_exact_test(mock_df,blank_df, original_df,decontaminated_df):
    """
    Fisher's Exact Test p-value: Evaluating presence/absence of each taxon in pre- vs. post-decontaminated sample
    for the mock species or contaminants.

    Contaminant species (in blank but not in mock) that were present pre-decontamination.
    Mock species that were present pre-decontamination.

    For each group (contaminants and mock), only species present in pre_df (abundance > 0) are considered.
    Returns the contingency table, odds ratio, and p-value.
    """
    logging.info("Calculating Fisher Exact test..")

    #Define species sets
    mock_species = set(mock_df["taxonomy"])
    blank_species = set(blank_df["taxonomy"])
    pre_species = set(original_df[original_df["abundance"] > 0]["taxonomy"])
    post_species = set(decontaminated_df[decontaminated_df["abundance"] > 0]["taxonomy"])

    # Define contaminants as species in blank but not in mock, present in pre-decontamination
    contaminant_species =(blank_species - mock_species).intersection(pre_species)
    mock_present_pre = mock_species.intersection(pre_species)

    # For contaminants:
    removed_contaminants = sum([1 for s in contaminant_species if s not in post_species])
    retained_contaminants = sum([1 for s in contaminant_species if s in post_species])

    # For mock:
    removed_mock= sum([1 for s in mock_present_pre if s not in post_species])
    retained_mock = sum([1 for s in mock_present_pre if s in post_species])

    contingency_table = [[removed_contaminants, retained_contaminants],
                         [removed_mock, retained_mock]]

    oddsratio, p_value = fisher_exact(contingency_table)
    print("Contingency Table:", contingency_table)
    print("Odds Ratio:", oddsratio)
    print("Fisher's Exact p-value:", p_value) #debugging

    return contingency_table, oddsratio, p_value

#section##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Beta Diversity Functions & Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

def beta_diversity_preprocessing(original_df, decontaminated_df, mock_df, comparison_df, blank_df):
    """
    Prepares a unified feature table from multiple datasets
    The previous decontamination pipeline outputed the TSVs with "relative abundance"

    :param original_df: original sample dataframe (columns: taxonomy, abundance)
    :param decontaminated_df: decontaminated sample dataframe (columns: taxonomy, abundance)
    :param mock_df: mock community dataframe (columns: taxonomy, abundance)
    :param comparison_df: sample processed by a previous pipeline (columns: taxonomy, abundance)
    :param blank_df: blank sample dataframe (columns: taxonomy, abundance)

    :return: DataFrame with rows as taxa and columns as sample types (missing values filled with zeros).
    """
    # Create copies and set 'taxonomy' as index
    orig = original_df[["taxonomy", "abundance"]].copy().set_index("taxonomy")
    decon = decontaminated_df[["taxonomy", "abundance"]].copy().set_index("taxonomy")
    mock = mock_df[["taxonomy", "abundance"]].copy().set_index("taxonomy")
    blank = blank_df[["taxonomy", "abundance"]].copy().set_index("taxonomy")

    # Rename columns to reflect sample type
    orig.rename(columns={'abundance': 'Original'}, inplace=True)
    decon.rename(columns={'abundance': 'Decontaminated'}, inplace=True)
    mock.rename(columns={'abundance': 'Mock'}, inplace=True)
    blank.rename(columns={'abundance': 'Blank'}, inplace=True)

    if comparison_df is not None:
        comp = comparison_df[["taxonomy", "abundance"]].copy().set_index("taxonomy")
        comp.rename(columns={'abundance': 'PreviousPipeline'}, inplace=True)
        feature_table = orig.join([decon, mock, blank, comp], how='outer')
        feature_table.fillna(0, inplace=True)
        #Could use: feature_table = pd.concat([orig, decon, mock, blank, comp], axis=1) (??)
        return feature_table

    # Merge all DataFrames on taxonomy index using an outer join
    feature_table = orig.join([orig, decon, mock, blank], how='outer')
    feature_table.fillna(0, inplace=True)
    return feature_table

def beta_diversity_summary(feature_table, output_file, metric="braycurtis"):
    """
    Computes the Bray–Curtis dissimilarity matrix from a TSS-normalized feature table
    Mean and median calculated of the off-diagonal distances

    :param feature_table: pd.DataFrame with rows as taxa and columns as samples.
    :param metric: dissimilarity metric to use (default: "braycurtis").

    :return: dist_matrix: Bray–Curtis dissimilarity matrix
    """
    sample_table = feature_table.T  # rows: samples, columns: taxa
    sample_ids = sample_table.index.tolist()

    # Compute the beta diversity distance matrix
    dist_matrix = beta_diversity(metric, sample_table.values, ids=sample_ids)
    # Convert the distance matrix to a square DataFrame for display
    distance_df = pd.DataFrame(dist_matrix.data, index=dist_matrix.ids, columns=dist_matrix.ids) #Convert to square dataframe

    # Convert the square matrix to a condensed vector (off-diagonal distances)
    condensed = squareform(dist_matrix.data)
    mean_distance = np.mean(condensed)
    median_distance = np.median(condensed)

    print("Pairwise Bray–Curtis Dissimilarity Matrix:")
    print(distance_df)
    print("\nMean off-diagonal Bray–Curtis distance:", mean_distance)
    print("Median off-diagonal Bray–Curtis distance:", median_distance)

    if output_file is not None:
        logging.info("Saving the Bray-Curtis dissimilarity matrix, mean and median distance results ")
        distance_df.to_csv(output_file, sep="\t") # Save the dissimilarity matrix
        #THEN append the median/median values
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"Mean_OffDiagonal_Distance\t" + str(mean_distance) + "\n")
            f.write(f"Median_OffDiagonal_Distance\t" + str(median_distance) + "\n")
        print(f"Saved Bray-Curtis matrix and statistics to {output_file}")

    return dist_matrix


def beta_diversity_plot_pcoa(dist_matrix, sample_groups, title="PCoA (Bray-Curtis)"):
    """
    Principal coordinate analysis (PCoA) - Metric multidimensional scaling

    Simple scatter plot of first two PCoA axes
    PCoA (Principal Coordinates Analysis)

    :param: dist_matrix: dissimilarity matrix
    :param: sample_groups: list of sample names
    :param: title: plot title
    :return:
    """
    pcoa_coords = squareform(dist_matrix) #np.ndarray (square matrix of pairwise dissimilarities)

    pc1 = pcoa_coords["PC1"]
    pc2 = pcoa_coords["PC2"]

    unique_groups = sorted(list(set(sample_groups.values())))

    color_map = plt.cm.get_cmap("tab10", len(unique_groups))
    groups_to_color = {g:color_map(i) for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(7,6))
    handles = [] #store scatter plot handles to build a legend

    #Group sample IDs by their groups
    groups_to_samples = {}
    for sid in pcoa_coords.index:
        grp = sample_groups[sid]
        groups_to_samples.setdefault(grp, []).append(sid)

    for grp in unique_groups:
        # Extract points belonging to this group
        sample_ids_in_group = groups_to_samples[grp]
        xvals = pc1[sample_ids_in_group]
        yvals = pc2[sample_ids_in_group]

    #Plot together
    sc = ax.scatter(xvals, yvals, label=grp, color=groups_to_color[grp], alpha=0.8)

    #Store the handle for legend
    handles.append(sc)

    if len(sample_ids_in_group) > 2:
        points = np.column_stack((xvals, yvals))
        hull = ConvexHull(points)
        hull_vertices = hull.vertices
        #close the polygon
        hull_vertices = np.append(hull_vertices, hull.vertices[0])
        ax.plot(points[hull_vertices, 0], points[hull_vertices, 1],
                color=groups_to_color[grp], lw=1.5)
        ax.fill(points[hull_vertices, 0], points[hull_vertices, 1],
                color=groups_to_color[grp], alpha=0.1)


    for i, sample_id in enumerate(pcoa_coords.index):
        ax.text(pc1[i], pc2[i], sample_id, fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

    ax.legend(handles=handles, loc="best", title="Groups")

    plt.tight_layout()
    plt.show()



#section##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Alpha Diversity Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

def alpha_diversity_preprocessing(feature_table, output_dir, metrics=["observed_otus", "chao1", "shannon", "simpson"]):
    """
    Computes alpha diversity for each sample (column) in the feature_table (taxa as rows).
    feature_table: pd.DataFrame with shape (n_taxa, n_samples)

    Measuring: Richness (observed_otus) and Evenness (shannon and simpson)
    1: "Observed_OTUS" (Richness): Actual number of unique OTUs (taxa) that have been identified in a given sample
    2: Shannon Index: diversity measure that considers both richness and evenness - Gives more weight to rare species than the Simpson index.
    3: Simpson Index: Diversity measure that gives more weight to common species

    Evenness: A measure that indicates how evenly individuals are distributed among the different taxa.

    4: Calculate evenness (shannon/ln(richness))
    Returns: pd.DataFrame with each metric as a column, each sample as a row.

    5: Calculating statistical significance for
    """
    logging.info("Computing alpha diversity results")
    data_matrix_t = feature_table.values.T # Convert to numpy array. Rows=taxa, columns=samples, transpose
    sample_ids = feature_table.columns.tolist()

    alpha_results = {m: alpha_diversity(m, data_matrix_t, ids=sample_ids) for m in metrics}
    # For storing the results in a dictionary (metric: [values])

    alpha_df = pd.DataFrame(alpha_results, index=sample_ids)

    #Calculate eveness
    alpha_df['eveness'] = alpha_df.apply(lambda row: row['shannon'] / np.log(row['observed_otus']) if row ['observed_otus'] > 0 else 0, axis=1)

    #Visualise with a histogram and Q-Q plots
    sns.histplot(alpha_df['shannon'], kde=True)
    plt.title("Historgram of Shannon Index")
    plt.show()

    sm.qqplot(alpha_df['shannon'], line='s')
    plt.title("Q-Q plot for Shannon Index")
    plt.show()

    #Assessing normality with Shapiro-Wilk test
    stat, p = shapiro(alpha_df['shannon'])
    if p < 0.05:
        logging.info("Data is non-normally distributed")
        logging.info("Calculate Mann-Whitney U test (unpaired) independent data ")

        # groupA = alpha_df[alpha_df['group'] == 'Pre']['shannon']
        # groupB = alpha_df[alpha_df['group'] == 'Post']['shannon']
        # w_stat, w_p = wilcoxon(groupA, groupB)
        # logging.info(f"Wilcoxon test: stat={w_stat}, p={w_p}")

        # For unpaired data, use Mann-Whitney:
        # mw_stat, mw_p = mannwhitneyu(groupA, groupB)
        # logging.info(f"Mann-Whitney test: stat={mw_stat}, p={mw_p}")


    else:
        logging.info("Data do not significantly deviate from normality, parametric tests may be acceptable.")
        logging.info("Calculating Kruskal-Wallis test")
        #Compare with Kruskal–Wallis Test: Comparing more tha two groups
        #Need "shannon" and "group" columns for each sample (mock, blank, pre-, post etc.)
        groups = alpha_df.groupby('group').unique()
        group_values = [alpha_df[alpha_df['group'] == g]['shannon'] for g in groups]

        kruskal_stats, kruskal_p = kruskal(*group_values)
        print(f"Kruskal-Wallis: statistic = {kruskal_stats}, p-value={kruskal_p}")
        print(alpha_df) #debugging

        if output_dir:
            logging.info("Saving Alpha diversity and Evenness results")
            os.makedirs(output_dir, exist_ok=True)
            alpha_df.to_csv(os.path.join(output_dir, f"alpha_diversity_{timestamp}.tsv"), sep="\t", index=False)
        return alpha_df

#section##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multi-panel plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

def plot_pre_vs_post_abundance1(pre_df, post_df, group_label, transform="symlog", title="", output_file=None):
    """
    Plot abundance bar charts comparing pre- vs post-decontamination for a given group.

    Parameters:
      - pre_df: DataFrame with columns ["genus_species", "abundance"]
      - post_df: DataFrame with the same columns
      - group_label: label for the group (e.g. "Expected Mock Species")
      - transform: if "symlog", transform using log1p and plot using symlog scale.
      - title: overall title for the figure.
      - output_file: if provided, saves the figure.
    """
    sns.set_theme(style="whitegrid")

    # Merge the pre and post data on genus_species
    merged = pre_df.merge(post_df, on="genus_species", how="outer", suffixes=("_pre", "_post")).fillna(0)
    # Sort by pre abundance (or total)
    merged.sort_values("abundance_pre", ascending=False, inplace=True)

    # Create a figure with two subplots: one for pre, one for post
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for ax, col, label in zip(axes, ["abundance_pre", "abundance_post"], ["Pre-cleaning", "Post-cleaning"]):
        # Optionally apply a log1p transformation
        if transform == "symlog":
            plot_vals = np.log1p(merged[col])
            ax.set_yscale("symlog", linthresh=1e-3)
            ylabel = "log1p(abundance)"
        else:
            plot_vals = merged[col]
            ylabel = "Abundance"

        ax.bar(merged["genus_species"], plot_vals, color=sns.color_palette("tab10", n_colors=len(merged)))
        ax.set_title(f"{group_label} - {label}")
        ax.set_xlabel("Taxon")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=90)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_pre_vs_post_abundance(dataframes_to_visual_list, group_labels, transform_method="symlog", max_columns=3, panel_width=5, panel_height=5, bar_width=0.8, legend_dict=None, overall_title=None, output_dir=None):
    """
    Produce multipanel figure for list of DataFrames for visualization.

    :param dataframes_to_visual_list: list of dataframes (Require cols: "genus_species", "abundance", "abundance_change", etc. depends on context)
    :param title_labels: List of titles for each subplot
    :param transform_method: either
        "symlog": Apply np.log1p and then use a symmetric log scale on the axis
        "cubic": Cube the normalized values
    :param max_columns: Maximum number of charts per row(default 3)
    :param panel_width, panel_height: Size of each panel
    :param bar_width, panel_height: Size of each bar
    :param legend_dict: Optional dictionary mapping parameter names to colors or descriptions (displayed as a legend)
    :param overall_title: Optional title for the figure
    :param output_dir: Save figures to this path.
    :return: Bar chart plots

    """
    sns.set_theme(style="whitegrid")
    num_plots = len(dataframes_to_visual_list) #Counts number of plots to create.

    num_cols=min(num_plots, max_columns)
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(panel_width * num_cols, panel_height * num_rows), squeeze=False)

    # Flatten axes array for easy iteration.
    axes_flat = axes.flatten()

    # Loop through each DataFrame and plot
    for i, (df, title) in enumerate(zip(dataframes_to_visual_list, group_labels)):
        ax = axes_flat[i]

        #Filter out zero
        df_filtered = df[df["abundance"] > 0].copy()

        #Apply transformation..

        if transform_method == "cubic":
            df_filtered["plot_value"] = df_filtered["abundance"] ** 3
            use_symlog = False

        elif transform_method in ["log", "symlog"]:
            #Use np.log1p to avoid issues with zeros i.e. log(1 + x))
            df_filtered["plot_value"] = np.log1p(df_filtered["abundance"])
            use_symlog = True

        #Sort by the transformed value for better visualization
        df_filtered.sort_values(by="plot_value", ascending=False, inplace=True)

        #Create bar chart
        x = np.arange(len(df_filtered))

        palette = sns.color_palette("tab20", n_colors=len(df_filtered))
        ax.bar(df_filtered["genus_species"], df_filtered["plot_value"],color=palette, alpha=0.6)
        ax.set_xticks(x)

        ax.set_title(
            title,
            fontsize=12,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.4')
        )
        ax.set_xlabel("Taxonomy", fontsize=11, fontweight='bold')
        ax.set_xticklabels(df_filtered["genus_species"], rotation=90)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        if i % num_cols == 0: #Only label y-axis on first column
            if transform_method == "cubic":
                axes[0].set_ylabel("Cubic transformed relative abundance (TSS-scaled)", fontsize=11, fontweight='bold')

            elif use_symlog:
                axes[0].set_ylabel("Log transformed relative abundance" if transform_method != "none" else "Relative abundance", fontsize=11, fontweight='bold')
                # Setting the y-axis to symlog so that very small values and negatives are visualized properly
                axes[0].set_yscale("symlog", linthresh=1e-3)

    for j in range(i+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if use_symlog:
        # Set a symlog scale on the first axis and then apply to all (if desired)
        for ax in axes_flat[:n]:
            ax.set_yscale("symlog", linthresh=1e-3)

    # Optionally add an overall legend for parameters
    if legend_dict:
        # Create legend handles from the dictionary
        legend_handles = [Patch(color=color, label=str(label)) for label, color in legend_dict.items()]
        # Place legend outside the subplots
        fig.legend(handles=legend_handles, loc="upper right", title="Parameters", bbox_to_anchor=(0.95, 0.95))

    if overall_title:
        fig.suptitle(overall_title, fontsize=16)


    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"multipanel_barcharts_{timestamp}.png"))
    plt.show()
    plt.close()
    logging.info(f"Comparison plot save to {output_dir}")



def plot_multipanel_barcharts(dataframes_to_visual_list, title_labels, transform_method="symlog", max_columns=3, panel_width=5, panel_height=5, bar_width=0.8, legend_dict=None, overall_title=None, output_dir=None):
    """
    Produce multipanel figure for list of DataFrames for visualization.

    :param dataframes_to_visual_list: list of dataframes (Require cols: "genus_species", "abundance", "abundance_change", etc. depends on context)
    :param title_labels: List of titles for each subplot
    :param transform_method: either
        "symlog": Apply np.log1p and then use a symmetric log scale on the axis
        "cubic": Cube the normalized values
    :param max_columns: Maximum number of charts per row(default 3)
    :param panel_width, panel_height: Size of each panel
    :param bar_width, panel_height: Size of each bar
    :param legend_dict: Optional dictionary mapping parameter names to colors or descriptions (displayed as a legend)
    :param overall_title: Optional title for the figure
    :param output_dir: Save figures to this path.
    :return: Bar chart plots

    """
    print(dataframes_to_visual_list[0])
    sns.set_theme(style="whitegrid")
    num_plots = len(dataframes_to_visual_list) #Counts number of plots to create.

    num_cols=min(num_plots, max_columns)
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(panel_width * num_cols, panel_height * num_rows), squeeze=False)

    # Flatten axes array for easy iteration.
    axes_flat = axes.flatten()

    # Loop through each DataFrame and plot
    for i, (df, title) in enumerate(zip(dataframes_to_visual_list, title_labels)):
        ax = axes_flat[i]

        #Filter out zero
        df_filtered = df[df["abundance"] > 0].copy()

        #Apply transformation...
        if transform_method == "cubic":
            df_filtered["plot_value"] = df_filtered["abundance"] ** 3
            use_symlog = False

        elif transform_method in ["log", "symlog"]:
            #Use np.log1p to avoid issues with zeros i.e. log(1 + x))
            df_filtered["plot_value"] = np.log1p(df_filtered["abundance"])
            use_symlog = True

        #Sort by the transformed value for better visualization
        df_filtered.sort_values(by="plot_value", ascending=False, inplace=True)

        #Create bar chart
        x = np.arange(len(df_filtered))

        palette = sns.color_palette("tab20", n_colors=len(df_filtered))
        ax.bar(df_filtered["taxonomy"], df_filtered["plot_value"],color=palette, alpha=0.6)
        ax.set_xticks(x)

        ax.set_title(
            title,
            fontsize=12,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.4')
        )
        ax.set_xlabel("Taxonomy", fontsize=11, fontweight='bold')
        ax.set_xticklabels(df_filtered["taxonomy"], rotation=90)
        ax.tick_params(axis='x', rotation=90, labelsize=10)

        if i % num_cols == 0: #Only label y-axis on first column
            if transform_method == "cubic":
                axes_flat[0].set_ylabel("Cubic transformed relative abundance (TSS-scaled)", fontsize=11, fontweight='bold')

            elif use_symlog:
                axes_flat[0].set_ylabel("Log transformed relative abundance" if transform_method != "none" else "Relative abundance", fontsize=11, fontweight='bold')
                # Setting the y-axis to symlog so that very small values and negatives are visualized properly
                axes_flat[0].set_yscale("symlog", linthresh=1e-3)
            else:
                axes_flat[0].set_ylabel("Relative abundance", fontsize=11, fontweight="bold")

    for j in range(i+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])    # Remove unused axes if any


    if use_symlog:
        # Set a symlog scale on the first axis and then apply to all (if desired)
        for ax in axes_flat:
            ax.set_yscale("symlog", linthresh=1e-3)

    # Optionally add an overall legend for parameters
    if legend_dict:
        # Create legend handles from the dictionary
        legend_handles = [Patch(color=color, label=str(label)) for label, color in legend_dict.items()]
        # Place legend outside the subplots
        fig.legend(handles=legend_handles, loc="upper right", title="Parameters", bbox_to_anchor=(0.95, 0.95))

    if overall_title:
        fig.suptitle(overall_title, fontsize=16)


    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"multipanel_barcharts_{timestamp}.png"))
    plt.show()
    plt.close()
    logging.info(f"Comparison plot save to {output_dir}")


def plot_multi_approach_comparison(
        before_decontam_df,
        after_decontam_df,
        group_label,
        output_path,
        mock_df=None,
        top_n=20
):


    """
    Generate a figure comparing 'Before' vs multiple 'After' DataFrames side by side.

    :param before_df: DataFrame with columns ['taxonomy', 'abundance'] (pre-decontamination).
    :param after_dfs: list of DataFrames, each with ['taxonomy', 'adjusted_abundance'].
    :param group_labels: list of strings naming each approach (e.g. ["App1", "App2", "App3"]).
    :param output_path: where to save the figure.
    :param mock_df: optional DataFrame with ['taxonomy', 'abundance'] for reference.
    """
    # Number of subplots = 1 for the "before" data plus len(after_dfs)
    # If mock_df is provided, add one more subplot for the mock.

    sns.set_theme(style="whitegrid")
    num_plots = 1 + len(after_decontam_df) + (1 if mock_df is not None else 0)

    # Create a figure with subplots in a single row
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(5 * num_plots, 5),
        sharey=True
    )

    if num_plots == 1:
        axes = [axes]

    # Plotting "before"
    before_df_filtered = before_decontam_df[before_decontam_df['abundance'] > 0].copy()

    before_df_filtered['abundance'] = before_decontam_df['abundance'] / before_df_filtered[
        'abundance'].sum()  # Convert to "relative abundance (0 to 1, a fraction)

    before_df_filtered.sort_values(by='abundance', ascending=False, inplace=True)
    before_df_filtered = before_df_filtered.head(top_n)
    palette = sns.color_palette("Set3", n_colors=len(before_df_filtered))

    axes[0].bar(before_df_filtered['taxonomy'], before_df_filtered['abundance'], color=palette, alpha=0.6)
    axes[0].set_title(
        "Pre-decontamination",
        fontsize=12,
        bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.4')
    )
    # for spine in axes.spines.values():
    #    spine.set_linewidth(1.5)

    axes[0].set_xlabel("Taxonomy", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Relative abundance", fontsize=11, fontweight="bold")
    axes[0].set_xticklabels(before_df_filtered['taxonomy'], rotation=90)

    # axes might be a single Axes if num_plots=1, or an array of Axes
    # Make sure we handle that gracefully. If there's only 1 plot, convert to a list
    # Plotting "after" each approach
    for i, (df_after, group_label) in enumerate(zip(after_decontam_df, group_label), start=1):
        df_after_filtered = df_after[df_after['adjusted_abundance'] > 0].copy()
        df_after_filtered['adjusted_abundance'] = df_after_filtered['adjusted_abundance'] / df_after_filtered[
            'adjusted_abundance'].sum()
        df_after_filtered.sort_values(by='adjusted_abundance', ascending=False, inplace=True)
        df_after_filtered = df_after_filtered.head(top_n)
        palette = sns.color_palette("tab20", n_colors=len(before_df_filtered))

        axes[i].bar(df_after_filtered['taxonomy'], df_after_filtered['adjusted_abundance'], color=palette, alpha=0.6)
        axes[i].set_title(
            f"Post-decontamination: {group_label}",
            fontsize=12,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.4')
        )

        axes[i].set_xlabel("Taxonomy", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Relative abundance", fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(df_after_filtered['taxonomy'], rotation=90)

    # If there is mock Dataframe, plot last
    if mock_df is not None:
        idx = len(after_decontam_df) + 1  # next subplot index
        mock_df_filtered = mock_df.copy()
        mock_df_filtered['abundance'] = mock_df['abundance'] / mock_df['abundance'].sum()
        mock_df_filtered.sort_values(by='abundance', ascending=False, inplace=True)
        mock_df_filtered = mock_df_filtered.head(top_n)
        palette = sns.color_palette("tab20", n_colors=len(mock_df_filtered))

        axes[idx].bar(mock_df_filtered['taxonomy'], mock_df_filtered['abundance'], color=palette, alpha=0.6)
        axes[idx].set_title(
            "Mock Community",
            fontsize=12,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.4')
        )
        axes[idx].set_xticklabels(mock_df_filtered['taxonomy'], rotation=90)
        axes[idx].set_xlabel("Taxonomy", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Relative abundance", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()
    print(f"Comparison plot save to {output_path}")


#optimize##########################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ################################################################################################

if __name__ == "__main__":
    # File paths for the references and samples
    timestamp = datetime.datetime.now().strftime("%H%M")  # Records timestamp (hours & minutes) to append to the file name

    mock_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\zymo_standard_reduced_depth.tsv"
    blank_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\decontamination_05_11_2024-15_35_22.tsv"
    pre_decontam_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M3_Q34_microbiome.tsv"
    post_decontam_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Decontamination\Parameters 1\high_quality_decontaminated_0312.tsv"
    pairwise_bc_matrix_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Decontamination\Parameters 1\pairwise_BCmatrix_tsv.tsv"

    raw_pre_decontam_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Decontamination\Parameters 1\de_norm_pre_decontam_sample_0312.tsv"
    raw_post_decontam_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Decontamination\Parameters 1\de_norm_post_decontam_sample_0312.tsv"

    #(Optional) Previous post-decontaminated sample from your prior approach
    #Compared to another pipeline OR a previous attempt with different parameters.
    prev_post_decontam_tsv =r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M3_Q34_microbiome-OLD.tsv"
    raw_prev_post_decontam_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\30_03_2025\de_normalise_M3_Q34_microbiome-OLD.tsv"

    # For precision, recall & beta diversity analyses...
    #ALL TSS Normalised....
    original_df = pd.read_csv(pre_decontam_tsv, sep="\t")
    decontaminated_df = pd.read_csv(post_decontam_tsv, sep="\t")
    mock_df = pd.read_csv(mock_tsv, sep="\t")
    comparison_df = pd.read_csv(prev_post_decontam_tsv, sep="\t")
    blank_df = pd.read_csv(blank_tsv, sep="\t")

    #~~~~~~~~~~~~~~~~~~~~~#Start of Evaluation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Run baseline comparison on pre-decontaminated sample and save results
    #See the mock taxon, blank taxon, and those present in the pre-decontaminated_sample...
    
    pre_results = baseline_comparison(mock_tsv, blank_tsv, pre_decontam_tsv)
    #save_results(pre_results,r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Pre")
    pre_decontam_expected = pre_results["positive_recall"]
    post_results = baseline_comparison(mock_tsv, blank_tsv, post_decontam_tsv)
    post_decontam_expected = post_results["positive_recall"]

    # Run evaluation comparing pre- and post- decontamination, with an optional previous decontamination file - RAW ABUNDANCES, NOT TSS scaled
    expected_compare = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )
    contaminants_compare = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )
    others_compare = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )

    #Compare to previous approach (previous is raw abundance, not relative abundance)
    if raw_prev_post_decontam_tsv is not None:
        compare_decontam_approaches_contaminants = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        raw_prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )

        compare_decontam_approaches_expected = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        raw_prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )

        compare_decontam_approaches_other = evaluate_decontamination(
        mock_tsv, blank_tsv, raw_pre_decontam_tsv, raw_post_decontam_tsv,
        raw_prev_post_decontam_tsv,
        output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation"
    )

    #For beta diversity, prepare a unified feature table (df above)
    feature_table = beta_diversity_preprocessing(original_df, decontaminated_df, mock_df, comparison_df, blank_df)
    
    #Feature table has relative abundances (no futher normalisation required)
    dist_matrix = beta_diversity_summary(feature_table, pairwise_bc_matrix_tsv, metric="braycurtis")

    precision_and_recall(mock_df, blank_df, decontaminated_df, output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation")

    logging.info("Calculating alpha diversity...")

    alpha_diversity = alpha_diversity_preprocessing(feature_table, output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation")
    fisher_exact_test(mock_df, blank_df, original_df, decontaminated_df)

    #Plot the beta diversity into a PCOA plot..
    sample_groups = {
        "Original": "Original",
        "Decontaminated": "Decontaminated",
        "Mock": "Mock",
        "Blank": "Blank",
        "PreviousPipeline": "PreviousPipeline",
    }
    beta_diversity_plot_pcoa(dist_matrix,sample_groups)


    dataframes_to_visual_list = [mock_df, blank_df, original_df, decontaminated_df, comparison_df]
    group_labels = ["Mock", "Blank", "Pre-Decontamination", "Post-Decontamination", "Previous Decontamination pipeline"]

    #dataframes_to_visual_list = [expected_compare, contaminants_compare, others_compare]
    #dataframes_to_visual_list = [original_df, decontaminated_df]
    #group_labels = ["Pre-Decontaminated", "Post-Decontaminated"]

    #Phred_Over_30_group_label = ["Nonlinear Transformation"]
    #Phred_Under_30_group_label = ["Proportional", "Threshold>20%", "FlaggedOnly"]

    logging.info("Visualising multi-panelled bar charts")
    plot_pre_vs_post_abundance(
       pre_df= pre_decontam_expected,
        post_df= pre_decontam_expected,
        group_label="Expected Mock Species",
        transform="symlog",
        title="Pre vs Post Cleaning - Expected Species"
    )

    plot_multipanel_barcharts(dataframes_to_visual_list, group_labels, transform_method="symlog", output_dir=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Evaluation")
    logging.info("Comparison plots saved.")



