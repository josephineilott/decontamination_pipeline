# optimize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# optimize~~~~~~~~~~~~~~ Decontamination Python script ~~~~~~~~~~~~~~#
# optimize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import os
import argparse
import pandas as pd
import time
import datetime
import logging

#Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('evaluation.log'), logging.StreamHandler()
                    ])

def load_data(input_tsv):
    """
    Load sample and blank TSV files into pandas DataFrames.
    Validate columns
    Error if files are missing.

    :param sample_tsv:
    :param blank_tsv:
    :return: sample and blank dataframes
    """
    try:
        df = pd.read_csv(input_tsv, sep='\t')
        df.columns = df.columns.str.strip().str.lower() # Normalize column names, strip white spaces, lowercase all
        logging.info(f"Successfully loaded {input_tsv}")  # Log activities
        logging.info(f"Total sample abundance: {df['abundance'].sum()}")
        return df

    except Exception as e:
        logging.error(f"Error loading {input_tsv}: {e}")  # Log error if no file found.
        exit()

def preprocess_data(df, is_blank=False):
    """
    Preprocess the DataFrame by ensuring required columns (taxonomy, abundance) are present
    Converting abundances to numeric
    Handling prevalence column in the blank (if need be)
    Create genus column

    If is_blank=True:
        - If a 'prevalence' column exists, convert to numeric and fill missing with 0.
        - If 'prevalence' does not exist, do nothing extra (remain robust).

    :param df: pd.DataFrame input
    :param is_blank: bool, if True, some blank-specific logic is applied
    :return: processed pd.DataFrame

    """
    # Always require taxonomy and abundance columns
    required_columns = ['taxonomy', 'abundance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")  # If the required columns are not present, throw error.

    # Convert 'abundance' to numeric and fill any invalid entries with 0
    df['abundance'] = pd.to_numeric(df['abundance'], errors='coerce').fillna(0)  # Fill invalid entries with 0
    assert "abundance" in df.columns, "Missing abundance column in DataFrame"
    assert "taxonomy" in df.columns, "Missing taxonomy column in DataFrame"
    assert df["abundance"].sum() > 0, "Total abundance should not be zero"

    # If this is a blank DataFrame, handle the 'prevalence' column
    if is_blank:
        if 'prevalence' in df.columns:  # If there is a prevalence column
            df['prevalence'] = pd.to_numeric(df['prevalence'], errors='coerce').fillna(0)
        df['genus'] = df['taxonomy'].apply(lambda x: x.split()[0] if pd.notnull(x) else 'Unknown')  # Extracts genus name from taxonomy column (Genus-level adjustments)

    # Create "genus" column from the taxonomy
    df['genus'] = df['taxonomy'].apply(lambda x: x.split()[0] if pd.notnull(x) else 'Unknown')  # Extracts genus name from taxonomy column (Genus-level adjustments)
    print(df)
    return df


def denormalize_abundance(df, total_raw_abundance, scaling_factor):
    """
    Denormalize the abundance values based on total reads.
    Used Total sum scaling (TSS) in previous steps
    Scale denormalized abundances in sample AND blank by the given scaling factor, proportional to sample read depth.

    :param df: pd.DataFrame input
    :param total_raw_abundance: pd.DataFrame input
    :param scaling_factor: float
    :return: pd.DataFrame output

    """
    logging.info(f"Scaling factor: {scaling_factor} ")
    df['abundance'] = (df['abundance'] * total_raw_abundance) / scaling_factor
    return df  # Return df both for the blank and the sample.

def merged_dataframes(sample_df,blank_df):
    # Merge on taxonomy (Sample, blank and mock)
    merged_df = pd.merge(sample_df, blank_df, on=['taxonomy', 'genus'], how='outer', suffixes=('_sample', '_blank'))
    merged_df.fillna(0, inplace=True)  # Missing values with 0 - handle taxa present in only one dataset
    return merged_df

def contaminant_identifier(row, mock_taxon_set, mock_genus_set, blank_genus_set, prevalence_threshold, ratio_threshold):
    """
    Merge sample and blank DataFrames on taxonomy and genus.
    Identify suspected contaminants

    A taxon is flagged as a contaminant if ALL of the following conditions are met:
      1. The taxon is NOT present in the mock community (overriding condition)
      Any taxa present in the mock are never flagged.
      2. The taxon is detected in the blank (abundance_blank > 0).
      3. Additionally, either:
           a. If a 'prevalence_blank' column exists, its value is at or above prevalence_threshold.
        OR b. The ratio of blank to sample abundance is high (i.e. blank abundance is at least a certain
              fraction of the sample abundance), ratio_threshold.
    4. Or, if the blank abundance is more than twice the sample abundance.

    Ratio threshold (e.g., if the blank abundance is at least 20% of the sample abundance)

    :param sample_df: pd.DataFrame input
    :param blank_df: pd.DataFrame input
    :param mock_df: pd.DataFrame input
    :param prevalence_threshold: float
    :return: pd.DataFrame output
    """

    logging.info(f"Prevalence threshold: {prevalence_threshold}, Ratio_threshold: {ratio_threshold}")
    # Condition 1: If the taxon is in the mock community, return False.
    if row['taxonomy'] in mock_taxon_set:
        return False

    #Condition 2: If the taxon is not directly detected/present in the blank, False
    if row['abundance_blank'] <= 0:
        if row['genus'] in blank_genus_set:
            # If sample abundance is very low (<5), flag as contaminant even if the genus is in the mock.
            if row['abundance_sample'] < 5:
                    return True
            # If this genus is present in the mock, revert to non-contaminant
            elif row['genus'] in mock_genus_set:
                return False
            # If the candidate is in the blank but the genus is not in the mock, flag as contaminant.
            else:
                return True
        else:
            return False #If the genus is not found in the blank, leave as False

    #If prevalence data is available, check it first.
    if 'prevalence_blank' in row:
        #Condition 3: If the taxon is prevalence above the blank prevalence threshold, True
        if row['prevalence_blank'] >= prevalence_threshold:
            return True

    #Condition 4: Check the abundance ratio (sample abundance divided by blank abundance).
    #If sample abundance > 0, compute the ratio.
    if row['abundance_sample'] > 0:
        ratio = row['abundance_sample'] / (row['abundance_blank'] + 1e-6)
        if ratio >= ratio_threshold:
            return True

    #Condition 5: If the blank's abundance is more than twice that of the sample, flag as contaminant.
    if row['abundance_blank'] > row['abundance_sample'] * 2:
        #If the blank’s abundance dwarfs the sample’s abundance by some ratio, likely contaminant
        return True

    #Otherwise, do not flag as contaminant.
    return False

def high_quality_sample_decontamination(df, aggression_factor):
    # Apply the row-level aggression logic to the rows
    # For a high quality, low abundance contaminant - More aggressive

    medium_aggression_condition = (
        (df['contaminant_flag'])
    )

    high_aggression_condition = (
            (df['abundance_blank'] < (0.05 * df['abundance_sample'])) &
            (df['contaminant_flag']) & ~medium_aggression_condition
    )

    #For a high quality, medium abundance contaminant - less aggressive, but still want removed.

    def high_aggression_decontamination(row, multiplier=10, alpha=8):
        """
        For high quality reads of low abundance
        Apply most aggressive approach - 2x aggression factor

        :param row: pd.DataFrame input
        :return: pd.DataFrame output

        """
        logging.info(f"High aggression approach: Multiplier={multiplier}, Alpha={alpha}, Aggression_factor={aggression_factor}")
        abundance_sample = row['abundance_sample']
        abundance_blank = row['abundance_blank']

        #ratio = abundance_blank / (abundance_sample + 1e-6)
        #correction_factor = min(1, ratio)

        #reduction = ratio ** alpha * multiplier * 2 * abundance_blank * (aggression_factor)
        # Applying a nonlinear transformation (e.g. Multiplying by a power) so that even small blank values produce a higher penalty.

        ratio = abundance_blank / (abundance_sample + 1e-6)
        correction_factor = min(1, ratio)
        reduction = correction_factor ** alpha * multiplier * 2 * abundance_blank * (10**(-aggression_factor))

        new_abundance = max(abundance_sample - reduction, 0)
        return new_abundance

    def medium_aggression_decontamination(row, multiplier=10, alpha=0.4):
        """
        High quality, high abundance contaminant taxa - Medium level
        :param row: pd.DataFrame input
        :return: pd.DataFrame output
        """
        logging.info(f"Medium aggression approach: Multiplier=10, Alpha=0.4, Aggression_factor={aggression_factor}")
        abundance_sample = row['abundance_sample']
        abundance_blank = row['abundance_blank']

        if abundance_blank == 0:
            reduction = 1
            #For those flagged as suspected low abundance contaminants BUT don't have a direct genus-species match in the blank.

        else:
            ratio = abundance_blank / (abundance_sample + 1e-6) #Small constant to ensure no division by zero
            correction_factor = min(1, ratio)
            reduction = correction_factor ** alpha * multiplier * abundance_blank * (10**(-aggression_factor))
            # Applying a nonlinear transformation (e.g. Multiplying by a power) so that even small blank values produce a higher penalty.

        new_abundance = max(abundance_sample - reduction, 0)
        return new_abundance

    df.loc[high_aggression_condition, 'abundance_sample'] = df.loc[high_aggression_condition].apply(
        high_aggression_decontamination, axis=1)

    df.loc[medium_aggression_condition, "abundance_sample"] = df.loc[medium_aggression_condition].apply(
        medium_aggression_decontamination, axis=1)

    output_dict = None
    return df, output_dict

def low_quality_sample_decontamination(df):
    """
    For low-quality samples (phred < 30), apply three different genus-level decontamination approaches.
    Returns a list of DataFrames (one for each approach) and an output dictionary.

    For each genus, we want to reduce or redistribute abundance among flagged species.
    """
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

    df["contaminant_level"] = df["abundance_blank"].apply(label_abundance)

    # Create copies of the DataFrame for the different approaches.
    df_proportional_approach_1 = df.copy()
    df_sigfraction_approach_2 = df.copy()
    df_flagonly_approach_3 = df.copy()

    def genus_correction_proportional(group):
        """
        Approach 1: For each genus group, penalize all species proportionally to the fraction flagged.
        :param group: pd.DataFrame input. Penalize proportionally unflagged and flagged.
        :return: pd.DataFrame output
        """

        flagged_mask = group["contaminant_flag"] == True
        total_genus = group["abundance_sample"].sum()
        if total_genus > 0:
            total_flagged = group.loc[flagged_mask, 'abundance_sample'].sum()
            fraction_flagged = total_flagged / total_genus #Compute proportion of the genus
        else:
            fraction_flagged = 0

        # All species in this genus lose fraction_flagged of their abundance
        group['abundance_sample'] *= (1 - fraction_flagged)
        return group

    def sig_fraction_genus_correction(group, min_flagged_fraction=0.2):
        """
        Approach 2: If flagged species are > X% of the genus abundance,
        apply a proportional penalty to *all* species in the genus.
        Otherwise, only penalize flagged species (e.g., zero them out).
        """
        logging.info(f"Proportional penalty: Minimum flagged proportion= {min_flagged_fraction}")
        flagged_mask = group['contaminant_flag'] == True
        total_genus = group['abundance_sample'].sum()
        if total_genus > 0:
            total_flagged = group.loc[flagged_mask, 'abundance_sample'].sum()
            fraction_flagged = total_flagged / total_genus
        else:
            fraction_flagged = 0

        if fraction_flagged >= min_flagged_fraction:
            # Penalize entire genus
            group['abundance_sample'] *= (1 - fraction_flagged)
        else:
            # Only penalize flagged species
            # e.g. zero out flagged, keep unflagged
            group.loc[flagged_mask, 'abundance_sample'] = 0
        return group

    def genus_correction_flagged_only(group, penalty_fraction=0.2):
        """
        Approach 3: For each genus group, adjust abundance_sample only for rows flagged as contaminants,
        based on their contaminant_level. Unflagged rows remain unchanged.

        If any species in the genus is flagged as a contaminant, then reduce the abundance of
        all species in that genus by the penalty_fraction.

        The rules are:
        - If contaminant_level is "High abundance contaminant", set abundance_sample to 0.
        - If contaminant_level is "Medium abundance contaminant", reduce abundance_sample by multiplying by (1 - penalty_fraction).
        - If contaminant_level is "Low abundance contaminant", reduce abundance_sample by multiplying by (1 - 2*penalty_fraction),
        ensuring the value does not go negative.
        - Otherwise, leave abundance_sample unchanged.

        :param group: pd.DataFrame for one genus (must include 'contaminant_flag', 'contaminant_level', and 'abundance_sample')
        :param penalty_fraction: float (if 1.0, then complete removal for medium, and double that for low)
        :return: group with adjusted 'abundance_sample'
        """
        (logging.info(f"Flagged only genus correction: Penalty fraction= {penalty_fraction}"))
        def adjust_row(row):
            if row.get("contaminant_flag", False):
                level = row.get("contaminant_level", "No abundance").lower()
                if "high abundance contaminant" in level:
                    return 0
                elif "medium abundance" in level:
                    return max(row["abundance_sample"] * (1 - penalty_fraction), 0)
                elif "low abundance contaminant" in level:
                    return max(row["abundance_sample"] * (1 - 2 * penalty_fraction), 0)
                else:
                    return 0
            else:
                return row["abundance_sample"]
        #Apply the adjustment row-by-row.
        group["abundance_sample"] = group.apply(adjust_row, axis=1)
        return group

    df_proportional_approach_1 = df_proportional_approach_1.groupby('genus', group_keys=False).apply(lambda g: genus_correction_proportional(g))
    """
    PROS: Any contamination in the genus reduces the entire genus proportionally
    CON: Any amount of flagged contamination can penalize the entire genus severely
    (if the flagged fraction is small but not trivial)
    """

    df_sigfraction_approach_2 = df_sigfraction_approach_2.groupby('genus', group_keys=False).apply(lambda g: sig_fraction_genus_correction(g, min_flagged_fraction=0.2))
    """
    group by genus, apply threshold-based correction.
    PROS: Avoids punishing the entire genus if only a small fraction is flagged
    CONS: You still might be too lenient if e.g. 19% is flagged but the 19% is truly contamination.
    The threshold is an arbitrary cutoff that needs careful tuning.
    """

    df_flagonly_approach_3 = df_flagonly_approach_3.groupby('genus', group_keys=False).apply(lambda g: genus_correction_flagged_only(g, penalty_fraction=1.0))
    """
    group by genus, only penalize flagged species.
    PROS: Minimal risk of removing real data for unflagged species.
    CONS: If classification is poor (especially for low-quality reads), unflagged species in the same genus may actually be contaminants.
    You risk not catching cross-species misclassifications inside the same genus.
    """

    three_app_output = [df_proportional_approach_1, df_sigfraction_approach_2, df_flagonly_approach_3]
    output_dict = {
        "proportional_approach_1": df_proportional_approach_1,
        "sig_fraction_approach_2": df_sigfraction_approach_2,
        "flagonly_approach_3": df_flagonly_approach_3,
    }
    return three_app_output, output_dict


def filter_low_abundance(df):
    """
    Filter out taxa with abundance below the threshold.
    - If a taxon is not flagged as a contaminant, keep it if its abundance_sample is >= 0.1.
    - If a taxon is flagged as a contaminant, keep it if its abundance_sample is >= 0.01.

    The resulting DataFrame is then sorted by descending abundance.
    Sort by descending abundance.
    """
    if df is not None:
        condition = (
        ((df['contaminant_flag'] == True) & (df['abundance_sample'] >= 1.0)) |
        ((df['contaminant_flag'] != True) & (df['abundance_sample'] >= 0.1))
        )
        #For flagged taxon, the filter is 1, if not suspected contaminant, the filter is 0.1
        df = df[condition].copy()
        df = df.sort_values(by='abundance_sample', ascending=False)
        return df[['taxonomy', 'abundance_sample']]
    else:
        return df

def re_tss_normalize(df, scaling_factor):
    """ Re-normalise the abundance using TSS for comparison
    Divide by the column sum and multiply by the user-set scaling_factor.
    TSS normalization converts raw counts into relative abundances - differences in sequencing depth are controlled for

    :param: df
    :param: scaling_factor - Factor to multiply normalized values.
    :return: df if TSS-normalised relative abundances
    """
    total = df['abundance_sample'].sum()
    if total > 0:
        df['abundance_sample'] = (df['abundance_sample'] / total) * scaling_factor
    else:
        #df['abundance'].rename('abundance', inplace=True)
        df['abundance_sample'] = 0
    return df

def log_penalty_statistics(sample_df, df_after_filter):
    """ Summary report - guard against over-removal
    Provide:
    1: Those species that got over 5% removal (random threshold) of the original abundance
    2: Those species that got no reduction BUT were close to the cutoffs and could have penalised.
    3: CSV of borderline taxa - those that lost between 5-20% of original abundance
    - Able for manual review to be treated as contaminants (e.g. or increase aggression factor)

    Compare the original sample abundances with the abundances after decontamination.
    Log the number of taxa penalized and identify borderline cases for manual review.

    :param: sample_df / original_df (pd.DataFrame): DataFrame before decontamination
    :param: df_after_filter: (pd.DataFrame): DataFrame after decontamination with 'adjusted_abundance'.
    :return: pd.DataFrame output

    """
    # Calculate how much abundance was removed per taxon:
    # Might need to be merged_df (unsure..)
    df_after_filter["penalty"] = sample_df["abundance"] - df_after_filter["abundance_sample"]

    # Calculate penalty as percentage of the original abundance
    df_after_filter['penalty_percentage'] = (df_after_filter['penalty'] / (sample_df['abundance_sample'] + 1e-6)) * 100

    # Count how many taxa have any penalty
    num_penalized = (df_after_filter['penalty'] > 0).sum()
    total_taxa = len(df_after_filter)
    logging.info(f"{num_penalized} out of {total_taxa} taxa were penalized.")

    # Identify borderline taxa that lost between 5% and 20% of their abundance
    borderline_taxa = df_after_filter[(df_after_filter['penalty_percentage'] >= 5) &
                                      (df_after_filter['penalty_percentage'] <= 20)]

    logging.info("Borderline taxa (5%-20% penalty) for manual review:")
    logging.info(borderline_taxa[['taxonomy', 'penalty_percentage']].to_string(index=False))

    # Optionally, save the borderline taxa for later review
    borderline_taxa.to_csv("borderline_taxa.tsv", sep='\t', index=False)
    return df_after_filter

def save_results(output_df, output_dict, output_dir, single_df=True):
    """
    Save each dataframe in the output dictionary as an TSV file in the given dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%H%M")  # Records timestamp (hours & minutes) to append to the file name

    if single_df:
        output_df = output_df[['taxonomy', 'abundance_sample']].copy()
        output_df.rename(columns={'abundance_sample': 'abundance'}, inplace=True)
        filename= f"high_quality_decontaminated_{timestamp}.tsv"
        output_df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)
        logging.info( f"Decontaminated sample rows: {len(output_df)}; total sample abundance: {output_df['abundance'].sum()}")
        logging.info(f"Saving results to {filename}")

    else:
        for key, output_df in output_dict.items():
            output_filename = os.path.join(output_dir, f"{key}_{timestamp}.tsv")
            output_df.to_csv(output_filename, sep='\t', index=False)
            logging.info(f"Saving results to {output_filename}")
            print(f"Save {key} to {output_filename}")
            logging.info(f"Decontaminated {key} sample rows: {len(output_df)}; total sample abundance: {output_df['abundance_sample'].sum()}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('decontamination.log'), logging.StreamHandler()]
                        )  # Save to file and Print to console
    """
    Full decontamination pipeline from loading data to saving results.
    """

    # File paths for the references and samples, and parameters
    start_time = time.time()
    logging.info("Script execution started.")

    mock_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\zymo_standard_reduced_depth.tsv"
    blank_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\decontamination_05_11_2024-15_35_22.tsv"
    sample_tsv = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Report writing\M3_Q34_microbiome.tsv"

    output_dir = r"C:\Users\Student\OneDrive - University of Bath\Desktop\Josephine's stuff\Masters\Semester 1\Research Project - Bagby Lab\Practice runs\04_04_2025\M3_Decontamination"

    #Sample pre-requisite information
    blank_total_raw_abundance = 198079
    sample_total_raw_abundance = 103025

    #input variables based on sample processing
    scaling_factor = 100000
    phred_score = 34

    #User changeable parameters
    aggression_factor = 50
    prevalence_threshold = 0.32
    ratio_threshold = 0.2

    #User changeable parameters: aggression_factor, prevalence_threshold = 0.32 (Left as 1/3 of the samples, not employed here! - Can create bias)

    logging.info("Loading TSV files..")
    sample_df = load_data(sample_tsv)
    blank_df = load_data(blank_tsv)
    mock_df = load_data(mock_tsv)

    logging.info("Preprocessing data: Standardise columns")
    sample_df = preprocess_data(sample_df)
    blank_df = preprocess_data(blank_df)
    mock_df = preprocess_data(mock_df)

    logging.info("Denormalize TSS-scaled abundances to raw abundance")
    sample_df = denormalize_abundance(sample_df, sample_total_raw_abundance, scaling_factor)
    # Saving the de-normalise, raw abundances of the pre-decontaminated sample
    timestamp = datetime.datetime.now().strftime("%H%M") #Records timestamp (hours & minutes) to append to the file name
    sample_df.to_csv(os.path.join(output_dir, f"de_norm_pre_decontam_sample_{timestamp}.tsv"), sep='\t', index=False)

    blank_df = denormalize_abundance(blank_df, blank_total_raw_abundance, scaling_factor)

    mock_taxon = set(mock_df['taxonomy']) #Taxa that match the genus and species in the mock
    mock_genus = set(mock_df['genus']) #Taxa of the same genus in the mock
    blank_genus = set(blank_df['genus']) #Taxa of the same genus in the blank

    merged_df = merged_dataframes(sample_df, blank_df)
    logging.info("Apply suspected contaminants flag...")

    merged_df["contaminant_flag"] = merged_df.apply(
        lambda row: contaminant_identifier(row, mock_taxon, mock_genus, blank_genus, prevalence_threshold, ratio_threshold), axis=1
    )
    merged_df.to_csv(os.path.join(output_dir, f"testing_flags_{timestamp}.tsv"), sep='\t', index=False)

    if phred_score > 30:
        logging.info("Apply High quality sample decontamination approach")
        decontaminated_df, output_dict = high_quality_sample_decontamination(merged_df, aggression_factor)

        logging.info("Recording borderline species taxa")
        #log_penalty_statistics(sample_df, post_decontam_filt_norm_df)

        logging.info("Filtering low abundance taxa (<0.1%) for high quality sample")
        post_decontam_filtered_df = filter_low_abundance(decontaminated_df)

        #Saving the pre-TSS normalise, filtered post-decontaminated sample.
        timestamp = datetime.datetime.now().strftime("%H%M")#Records timestamp (hours & minutes) to append to the file name
        pre_tss_filename = os.path.join(output_dir, f"de_norm_post_decontam_sample_{timestamp}.tsv")
        decontaminated_df = decontaminated_df.loc[decontaminated_df["abundance_sample"] != 0.0, ['taxonomy', 'abundance_sample']].copy()
        decontaminated_df.to_csv(pre_tss_filename, sep='\t', index=False)
        logging.info(f"Saved pre-TSS normalized high quality decontaminated sample to {pre_tss_filename}")

        logging.info("Normalise with TSS-scaling for high quality sample")
        post_decontam_filt_norm_df = re_tss_normalize(post_decontam_filtered_df, scaling_factor)

        # Save the TSS-normalised final result
        save_results(post_decontam_filt_norm_df, output_dict, output_dir, single_df=True)

    else: #if Phred score < 30 (low-quality sample)
        logging.info("Apply Low quality sample decontamination approach")
        decontaminated_list, output_dict = low_quality_sample_decontamination(merged_df)
        # decontaminated_df is a list of DataFrames for each approach

        logging.info("Filtering low abundance taxa (<0.1%) for low quality samples (each approach)")
        post_decontam_filtered_list= [filter_low_abundance(df) for df in decontaminated_list]

        # Save each filtered dataframe (pre-TSS) with unique filenames
        for i, df in enumerate(post_decontam_filtered_list):
             timestamp = datetime.datetime.now().strftime("%H%M")  # Records timestamp (hours & minutes) to append to the file name
             pre_tss_filename = os.path.join(output_dir, f"de_norm_post_decontam_sample_{i+1}_{timestamp}.tsv")
             df.to_csv(pre_tss_filename, sep='\t', index=False)
             logging.info(f"Saved pre-TSS normalised low quality decontaminated sample, approach {i+1}, to {pre_tss_filename}")

        logging.info("Normalising each filtered dataframe with TSS-scaling for low quality sample")
        post_decontam_filt_norm_list = [re_tss_normalize(df, scaling_factor) for df in post_decontam_filtered_list]

        # Save each TSS-normalised dataframe with unique filenames
        for i, df in enumerate(post_decontam_filt_norm_list):
            timestamp = datetime.datetime.now().strftime("%H%M")  # Records timestamp (hours & minutes) to append to the file name
            final_filename = os.path.join(output_dir, f"final_post_decontam_sample_approach_{i+1}_{timestamp}.tsv")
            df.to_csv(final_filename, sep='\t', index=False)
            logging.info(f"Saved TSS-normalised low quality decontaminated sample, approach {i+1}, to {final_filename}")

    end_time = time.time()
    execution_time = round(end_time - start_time, 3)
    logging.info(f"Execution time: {execution_time} seconds")
    print(f"Execution time: {execution_time} seconds")

