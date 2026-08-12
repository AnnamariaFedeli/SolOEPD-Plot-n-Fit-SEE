import pandas as pd


def excluded_channels_from_fit(data_name_list, channel_list):
    """
    Combine input DataFrames and exclude rows whose index is not in channel_list.

    Args:
        data_name_list (list[pd.DataFrame]): List of DataFrames to concatenate.
        channel_list (list[int]): Indices to retain after concatenation.

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame.
    """

    # Combine all input data
    combined_csv = pd.concat(data_name_list)
    combined_csv.reset_index(drop=True, inplace=True)

    # Remove unused column
    combined_csv = combined_csv.drop(columns='Energy_channel')

    # Identify rows to remove
    rows_to_delete = []
    for i in range(len(combined_csv)):
        if i not in channel_list:
            rows_to_delete.append(i)

    # Drop unwanted rows
    combined_csv = combined_csv.drop(rows_to_delete, axis=0)

    # Sort and reset index
    combined_csv = combined_csv.sort_values('Primary_energy')
    combined_csv.reset_index(drop=True, inplace=True)

    return combined_csv


def combine_data(
    data_name_list,
    path=None,
    sigma=3,
    rel_err=0.5,
    frac_nan_threshold=0.9,
    leave_out_1st_het_chan=False,
    fit_to='Peak',
    channels_to_exclude=None
):
    """
    Combine and filter multiple DataFrames according to significance,
    relative error, and NaN thresholds, with optional channel exclusions.

    Args:
        data_name_list (list[pd.DataFrame]): Input DataFrames.
        path (str, optional): Output CSV path.
        sigma (int, optional): Minimum significance threshold.
        rel_err (float, optional): Maximum relative error threshold.
        frac_nan_threshold (float, optional): Minimum fraction of non-NaN values.
        leave_out_1st_het_chan (bool, optional): Remove low-energy HET channels.
        fit_to (str, optional): Prefix for significance column.
        channels_to_exclude (list[int], optional): Row indices to drop.

    Returns:
        pd.DataFrame: Filtered combined DataFrame.
    """

    combined_csv = None

    # Case 1: Explicit channel exclusion
    if channels_to_exclude is not None:
        combined_csv = pd.concat(data_name_list)
        combined_csv.reset_index(drop=True, inplace=True)
        combined_csv = combined_csv.drop(columns='Energy_channel')

        combined_csv = combined_csv.drop(channels_to_exclude, axis=0)
        combined_csv = combined_csv.sort_values('Primary_energy')
        combined_csv.reset_index(drop=True, inplace=True)

    # Case 2: Remove first HET channel
    if leave_out_1st_het_chan and len(data_name_list) > 2:
        het = data_name_list[-1]

        first_het = het.index[het['Primary_energy'] < 0.7].tolist()
        het = het.drop(first_het, axis=0)
        het.reset_index(drop=True, inplace=True)

        data_name_list[-1] = het

        combined_csv = pd.concat(data_name_list)
        combined_csv.reset_index(drop=True, inplace=True)
        combined_csv = combined_csv.drop(columns='Energy_channel', errors='ignore')

    # Default case
    if combined_csv is None:
        combined_csv = pd.concat(data_name_list)
        combined_csv.reset_index(drop=True, inplace=True)
        combined_csv = combined_csv.drop(columns='Energy_channel', errors='ignore')

    # Apply significance filter
    rows_to_delete = combined_csv.index[
        combined_csv[fit_to + '_significance'] < sigma
    ].tolist()
    combined_csv = combined_csv.drop(rows_to_delete, axis=0)
    combined_csv.reset_index(drop=True, inplace=True)

    # Apply relative error filter
    if rel_err is not None:
        rows_to_delete = combined_csv.index[
            combined_csv['rel_backsub_peak_err'] > rel_err
        ].tolist()
        combined_csv = combined_csv.drop(rows_to_delete, axis=0)
        combined_csv.reset_index(drop=True, inplace=True)

    # Apply NaN fraction filter
    rows_to_delete = combined_csv.index[
        combined_csv['frac_nonan'] < frac_nan_threshold
    ].tolist()
    combined_csv = combined_csv.drop(rows_to_delete, axis=0)

    # --- Final sorting ---
    combined_csv = combined_csv.sort_values('Primary_energy')
    combined_csv.reset_index(drop=True, inplace=True)

    # Optional save 
    if path is not None:
        combined_csv.to_csv(path, sep=';')

    return combined_csv


def extract_low_sigma_rows(
    data_name_list,
    sigma=3,
    leave_out_1st_het_chan=False,
    fit_to='Peak'
):
    """
    Filter rows with significance <= sigma after combining input DataFrames.

    Args:
        data_name_list (list of pd.DataFrame): Input datasets to combine.
        sigma (int, optional): Upper threshold for significance. Defaults to 3.
        leave_out_1st_het_chan (bool, optional): Whether to exclude low-energy
            channels from the last dataset. Defaults to False.
        fit_to (str, optional): Prefix for significance column. Defaults to 'Peak'.

    Returns:
        pd.DataFrame: Filtered and combined DataFrame.
    """
    combined_csv = pd.concat(data_name_list)
    combined_csv.reset_index(drop=True, inplace=True)
    combined_csv = combined_csv.drop(columns='Energy_channel')

    if leave_out_1st_het_chan:
        het = data_name_list[-1]

        # Identify low-energy channels
        first_het = het.index[het['Primary_energy'] < 0.7].tolist()

        het = het.drop(first_het, axis=0)
        het.reset_index(drop=True, inplace=True)

        # Update original list (side effect is intentional)
        data_name_list[-1] = het

    # Filter based on significance threshold (keep <= sigma)
    significance_col = f"{fit_to}_significance"
    rows_to_delete = combined_csv.index[
        combined_csv[significance_col] > sigma
    ].tolist()

    combined_csv = combined_csv.drop(rows_to_delete, axis=0)
    combined_csv.reset_index(drop=True, inplace=True)

    return combined_csv

def extract_nan_heavy_rows(
    data_name_list,
    frac_nan_threshold=0.9,
    leave_out_1st_het_chan=False
):
    """
    Extract rows with too many NaNs by removing rows with sufficiently
    high fraction of non-NaN values.

    Args:
        data_name_list (list of pd.DataFrame): Input datasets to combine.
        frac_nan_threshold (float, optional): Threshold for fraction of
            non-NaN values. Rows ABOVE this threshold are removed, leaving
            rows with many NaNs. Defaults to 0.9.
        leave_out_1st_het_chan (bool, optional): Whether to exclude low-energy
            channels from the last dataset. Defaults to False.

    Returns:
        pd.DataFrame: Subset containing rows with many NaNs.
    """
    combined_csv = pd.concat(data_name_list)
    combined_csv.reset_index(drop=True, inplace=True)
    combined_csv = combined_csv.drop(columns='Energy_channel')

    if leave_out_1st_het_chan:
        het = data_name_list[-1]

        # Identify low-energy channels
        first_het = het.index[het['Primary_energy'] < 0.7].tolist()

        het = het.drop(first_het, axis=0)
        het.reset_index(drop=True, inplace=True)

        # Intentional side effect
        data_name_list[-1] = het

    # Remove "good" rows → keep rows with many NaNs
    rows_to_delete = combined_csv.index[
        combined_csv['frac_nonan'] > frac_nan_threshold
    ].tolist()

    combined_csv = combined_csv.drop(rows_to_delete, axis=0)
    combined_csv.reset_index(drop=True, inplace=True)

    return combined_csv


def extract_high_rel_err_rows(
    data_name_list,
    rel_err=0.5,
    leave_out_1st_het_chan=False
):
    """
    Extract rows with high relative error by removing rows below the threshold.

    Args:
        data_name_list (list of pd.DataFrame): Input datasets to combine.
        rel_err (float, optional): Threshold for relative error. Rows BELOW this
            value are removed, leaving high-error rows. Defaults to 0.5.
        leave_out_1st_het_chan (bool, optional): Whether to exclude low-energy
            channels from the last dataset. Defaults to False.

    Returns:
        pd.DataFrame: Subset containing high relative error rows.
    """
    combined_csv = pd.concat(data_name_list)
    combined_csv.reset_index(drop=True, inplace=True)
    combined_csv = combined_csv.drop(columns='Energy_channel')

    if leave_out_1st_het_chan:
        het = data_name_list[-1]

        # Identify low-energy channels
        first_het = het.index[het['Primary_energy'] < 0.7].tolist()

        het = het.drop(first_het, axis=0)
        het.reset_index(drop=True, inplace=True)

        # Intentional side effect
        data_name_list[-1] = het

    # Remove "good" rows → keep high-error ones
    rows_to_delete = combined_csv.index[
        combined_csv['rel_backsub_peak_err'] < rel_err
    ].tolist()

    combined_csv = combined_csv.drop(rows_to_delete, axis=0)
    combined_csv.reset_index(drop=True, inplace=True)

    return combined_csv


def delete_bad_data(
    data,
    sigma=3,
    rel_err=0.5,
    frac_nan_threshold=0.9,
    leave_out_1st_het_chan=False,
    fit_to='Peak',
    channels_to_exclude=None
):
    """
    Remove rows that do not meet quality criteria.

    Args:
        data (pd.DataFrame): Input dataset.
        sigma (int, optional): Minimum significance threshold.
        rel_err (float, optional): Maximum allowed relative error.
        frac_nan_threshold (float, optional): Minimum fraction of non-NaN values.
        leave_out_1st_het_chan (bool, optional): Whether to exclude low-energy channels.
        fit_to (str, optional): Prefix for significance column.
        channels_to_exclude (list, optional): Row indices to drop before filtering.

    Returns:
        pd.DataFrame: Cleaned dataset with only "good" rows.
    """

    # Drop unused column (safe version recommended below)
    data = data.drop(columns='Energy_channel')

    if channels_to_exclude is not None:
        data = data.drop(channels_to_exclude, axis=0)

    data = data.sort_values('Primary_energy')
    data.reset_index(drop=True, inplace=True)

    # Remove low significance rows
    rows_to_delete = data.index[data[f'{fit_to}_significance'] < sigma].tolist()
    data = data.drop(rows_to_delete, axis=0)
    data.reset_index(drop=True, inplace=True)

    # Remove high relative error rows
    rows_to_delete = data.index[data['rel_backsub_peak_err'] > rel_err].tolist()
    data = data.drop(rows_to_delete, axis=0)
    data.reset_index(drop=True, inplace=True)

    # Remove rows with too many NaNs
    rows_to_delete = data.index[data['frac_nonan'] < frac_nan_threshold].tolist()
    data = data.drop(rows_to_delete, axis=0)
    data.reset_index(drop=True, inplace=True)

    if leave_out_1st_het_chan:
        # Remove low-energy channels
        first_het = data.index[data['Primary_energy'] < 0.7].tolist()
        data = data.drop(first_het, axis=0)
        data.reset_index(drop=True, inplace=True)

    return data


def extract_first_het_channel(data, expected_channels=4, energy_threshold=0.7):
    """
    Extract the first HET channel.

    If the expected number of channels is present, the lowest-energy channel
    is selected based on ordering. Otherwise, a fallback using an energy
    threshold is applied.

    Args:
        data (pd.DataFrame): HET-only dataset.
        expected_channels (int, optional): Expected number of channels. Defaults to 4.
        energy_threshold (float, optional): Threshold for fallback selection. Defaults to 0.7.

    Returns:
        pd.DataFrame: Data containing only the first HET channel.
    """
    # Sort data by energy to ensure correct ordering
    data = data.sort_values('Primary_energy')
    data.reset_index(drop=True, inplace=True)

    n_channels = len(data)

    if n_channels == expected_channels:
        # Select the lowest-energy channel (first row after sorting)
        first_channel = data.iloc[[0]]
        first_channel.reset_index(drop=True, inplace=True)
    else:
        # Select channels below the fallback energy threshold
        first_channel = data[data['Primary_energy'] < energy_threshold]
        first_channel = first_channel.copy()
        first_channel.reset_index(drop=True, inplace=True)

    return first_channel

def combine_data_general(data_name_list, path):
    """
    Concatenate a list of DataFrames, reset the index, and save to CSV.

    Args:
        data_name_list (list of pd.DataFrame): Input datasets.
        path (str): Output file path.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    combined_csv = pd.concat(data_name_list)

    # Ensure a clean, continuous index after concatenation
    combined_csv.reset_index(drop=True, inplace=True)

    # Write combined data to disk
    combined_csv.to_csv(path, sep=';')

    return combined_csv
