#optimize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#optimize~~~~~~~~~~~~~~ Current copy - 05/01/2025 - Working! ~~~~~~~~~~~~~~#
#optimize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#todo ~~ Current framework ~~#
# Reads input TSVs and logs information.
# Normalizes abundances using Total Sum Scaling (TSS).
# Adjusts contamination levels based on aggression factors and Phred scores.
# Merges the blank and sample datasets.
# Performs genus-level corrections for low-quality data.
# Generates output files.
# Creates barplots comparing before/after distributions.

import pandas as pd
import time
import numpy as np
import logging
import matplotlib.pyplot as plt

#todo ~~~ Setup logging ~~~#
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('decontamination.log'), logging.StreamHandler()
                    ]
) #Save to file and Print to console

#todo ~~ Initial Checking ~~#
file_path = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M1_Q23_microbiome.tsv"
df = pd.read_csv(file_path, sep='\\t', engine='python')

print("Column Names:", df.columns.tolist())  # See what columns exist
print(df.head())  # Show first few rows


def load_data(sample_tsv, blank_tsv):
    """
    Load sample and blank TSV files into pandas DataFrames.
    """
    try:
        sample_df = pd.read_csv(sample_tsv, sep='\t')
        sample_df.columns = sample_df.columns.str.strip().str.lower() #Normalize column names
        logging.info(f"Successfully loaded {sample_tsv}")

        blank_df = pd.read_csv(blank_tsv, sep='\t')
        blank_df.columns = blank_df.columns.str.strip().str.lower() #Normalize column names
        logging.info(f"Successfully loaded {blank_tsv}")
        return sample_df, blank_df

    except Exception as e:
        logging.error(f"Error loading {sample_tsv, blank_tsv}: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the DataFrame by ensuring required columns are present,
    converting abundances to numeric, and extracting genus information.
    """
    required_columns = ['taxonomy', 'abundance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    assert "abundance" in df.columns, "Missing abundance column in DataFrame"
    assert df["abundance"].sum() > 0, "Total aundance should not be zero"

    df['abundance'] = pd.to_numeric(df['abundance'], errors='coerce').fillna(0) #Fill invalid entries with 0
    df['genus'] = df['taxonomy'].apply(lambda x: x.split()[0] if pd.notnull(x) else 'Unknown') #Extracts genus name from taxonomy column (Genus-level adjustments)
    return df


def denormalize_abundance(df, total_raw_abundance, scaling_factor):
    """
    Denormalize the abundance values based on total reads.
    Scale denormalized abundances in sample AND blank by the given scaling factor, proportional to sample read depth.
    """

    df['denorm_abundance'] = (df['abundance'] * total_raw_abundance) / scaling_factor
    return df

def merge_data(sample_df, blank_df):
    """
    Merge sample and blank DataFrames on taxonomy and genus.
    Combines samples and blank data on taxonomy and genus
    """
    merged_df = pd.merge(sample_df, blank_df, on=['taxonomy', 'genus'], how='outer', suffixes=('_sample', '_blank'))
    merged_df.fillna(0, inplace=True) #Missing values with 0 - handle taxa present in only one dataset
    return merged_df


def apply_decontamination(merged_df, aggression_factor, phred_score):
    """
    Apply regression-based decontamination to adjust for contaminants.
    If Pred < 30, genus-level adjustments are applied to redistribute abundance
    instead of outright removal.
    """

    # Group by genus to ensure genus-level corrections
    genus_groups = merged_df.groupby('genus')

    def adjust_abundance(row):
        blank_abundance = row['denorm_abundance_blank']
        sample_abundance = row['denorm_abundance_sample']
        taxon_genus = row['genus']

        #High-quality data (Phred > 30) - More aggressive decontamination
        if phred_score >= 30: #For high quality data (apply aggression_factor)
            if blank_abundance < 10: #High quality, low abundance in blank
                adjustment = 2 * aggression_factor * blank_abundance #Doubles aggression
                new_abundance = sample_abundance - adjustment
                new_abundance = max(new_abundance, 0) #Ensure no negative values
                #alteration 1: ~~~~~~~ Change to handle negative values ~~~~~ ????

            else: #High quality, high abundance in blank
                adjustment = aggression_factor * blank_abundance
                new_abundance = sample_abundance - adjustment
                new_abundance = max(new_abundance, 0)  # Ensure no negative values (taxons adjusted abundance never below zero)

        else: #For low-abundance contaminants (Phred < 30) - Needs Genus-level adjustments
            #Calculate genus-wide abundance
            genus_sample_total_raw_abundance = genus_groups['denorm_abundance_sample'].sum().get(taxon_genus, 0)

            #Checks for taxa in the same genus
            #IF genus contains multiple taxa, we redistribute the contaminant adjustment proportionally


            #Ensure genus-wide abundance exists before applying genus correction
            #alteration 2: ~~ Genus level adjustment needs work!! ~~~#
            if genus_sample_total_raw_abundance > 0: #alteration ~~~~~~ Might need to be higher to be classed as multiple ~~~~~#
                # Redistribute removed abundance proportionally to other taxa in the same genus
                #This ensures that if one taxon is over-filtered, the abundance is spread across the genus-group instead of being completely removed.

                genus_adjustment = (aggression_factor * blank_abundance * 0.5) # Less aggressive for low-quality
                adjusted_proportion = (sample_abundance / genus_sample_total_raw_abundance)

                #Distribute the lost abundance among the genus-group
                #Like for E.coli and Shigella (high seq similarity, resulting in missclassification)
                #Contaminant adjustment is scaled down, removed abundance is redistributed among other taxa in the same genus
                genus_redistributed_abundance = genus_adjustment * adjusted_proportion

                # Apply genus-aware correction
                new_abundance = max(sample_abundance - genus_redistributed_abundance, 0)

            else:
                # If no genus-wide abundance to redistribute, apply normal correction
                new_abundance = max(sample_abundance - (aggression_factor * blank_abundance * 0.5), 0)
        return new_abundance

    # Apply the adjusted abundance function across all rows
    merged_df['adjusted_abundance'] = merged_df.apply(adjust_abundance, axis=1)
    print(merged_df.head())
    return merged_df
#changes


def filter_low_abundance(merged_df, threshold=0.1):
    """
    Filter out taxa with adjusted abundance below the threshold.
    Sort by descending abundance.
    """
    merged_df = merged_df[merged_df['adjusted_abundance'] > threshold]
    merged_df = merged_df.sort_values(by='adjusted_abundance', ascending=False)
    return merged_df[['taxonomy', 'adjusted_abundance']]

def plot_comparison(before_df, after_df, output_path, mock_df=None):
    #alteration 3: Need to add in beta diversity and make these more visually appealing

    """Generate side-by-side bar charts comparing pre and post decontamination, along with the mock community."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    axes[0].bar(before_df['taxonomy'], before_df['abundance'], color='blue', alpha=0.6)
    axes[0].set_title("Before Decontamination")
    axes[0].set_xticklabels(before_df['taxonomy'], rotation=90)

    axes[1].bar(after_df['taxonomy'], after_df['adjusted_abundance'], color='red', alpha=0.6)
    axes[1].set_title("After Decontamination")
    axes[1].set_xticklabels(after_df['taxonomy'], rotation=90)

    if mock_df is not None:
        print("Mock Dataframe Columns:", mock_df.columns.tolist()) #debugging
        print(mock_df.head()) # Show first few rows of mock_df

        axes[2].bar(mock_df['taxonomy'], mock_df['abundance'], color='green', alpha=0.6)
        axes[2].set_title("Mock Community")
        axes[2].set_xticklabels(mock_df['taxonomy'], rotation=90)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_results(df, output_tsv):
    """
    Save the decontaminated DataFrame to a TSV file.
    """
    df[['taxonomy', 'adjusted_abundance']].to_csv(output_tsv, sep='\t', index=False)
    #alteration 4: Need to output the file with ..._clean.tsv formatting
    # Need to change the column names just to abundance....

def decontaminate_pipeline(sample_tsv, blank_tsv, output_tsv, aggression_factor, phred_score, scaling_factor, blank_total_raw_abundance, sample_total_raw_abundance, mock_tsv=None):
    """
    Full decontamination pipeline from loading data to saving results.
    """
    start_time = time.time()
    logging.info("Script execution started.")
    sample_df, blank_df = load_data(sample_tsv, blank_tsv)
    mock_df = pd.read_csv(mock_tsv, sep='\t') if mock_tsv else None
    mock_df.columns = mock_df.columns.str.strip().str.lower() #Normalize column names - No extra spaces and consistent lowercase names

    if sample_df is None or blank_df is None:
        logging.error("Failed to load input files. Exiting.")
        return

    logging.info("Preprocessing data...")
    sample_df = preprocess_data(sample_df)
    blank_df = preprocess_data(blank_df)

    logging.info("Denormalizing abundances...")
    sample_df = denormalize_abundance(sample_df, sample_total_raw_abundance, scaling_factor)
    blank_df = denormalize_abundance(blank_df, blank_total_raw_abundance, scaling_factor)

    logging.info("Merging data...")
    merged_df = merge_data(sample_df, blank_df)

    logging.info("Applying decontamination...")
    decontaminated_df = apply_decontamination(merged_df, aggression_factor, phred_score)

    logging.info("Filtering low abundance taxa...")
    filtered_df = filter_low_abundance(decontaminated_df)

    logging.info("Saving results...")
    save_results(filtered_df, output_tsv)
    logging.info(f"Decontaminated data saved to {output_tsv}")

    plot_comparison(sample_df, filtered_df, output_tsv.replace('.tsv', '.png'), mock_df)
    logging.info("Comparison plot saved.")

    end_time = time.time()
    execution_time = round(end_time - start_time, 3)
    logging.info(f"Execution time: {execution_time} seconds")
    print(f"Execution time: {execution_time} seconds")

#alteration 5: Inputs from multiple files and input variables - Need to robustly handle them - Need to in a integer format
    #Can have a default for the threshold for filtering and aggression factor BUT the rest is user defined.
#alteration 6:


decontaminate_pipeline(
    r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M3_Q34_microbiome.tsv", #Testing M3_Q23_tsv
    r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\decontamination_05_11_2024-15_35_22.tsv", #Corresponding combined blank
    r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M3_Q34_cleaned.tsv", #Output tsv
    aggression_factor=1.1,
    phred_score=34,
    scaling_factor = 100000,
    blank_total_raw_abundance=198079,
    sample_total_raw_abundance=103025,
    mock_tsv=r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\zymo_standard_reduced_depth.tsv") #Example TSV already made
