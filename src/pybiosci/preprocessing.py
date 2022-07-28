import pandas as pd


def preprocess_feature_metadata(feature_path: str, metadata_path: str, feature_index="SampleID",
                                metadata_index="run_id", sample_in_row=True, otu_filter=None, num_seq_filter=None):
    # Reading Feature Table
    all_feature_df = pd.read_csv(feature_path, delimiter='\t')
    if sample_in_row:
        all_feature_df = all_feature_df.set_index(feature_index).T
    else:
        all_feature_df = all_feature_df.set_index(feature_index)

    # Removing the zero sum samples or affected samples
    all_feature_df = all_feature_df.fillna(0)
    all_feature_df = all_feature_df.loc[:, all_feature_df.sum(axis=0) != 0]

    # Reading the metadata and otu table
    all_metadata_df = pd.read_csv(metadata_path)
    all_metadata_df = all_metadata_df.set_index(metadata_index)
    # Removing the samples without Phenotype info
    all_metadata_df = all_metadata_df[all_metadata_df['phenotype'].notna()]

    # Filtering
    if otu_filter:
        print(f"##Reducing OTUs not seen in at least {otu_filter} samples")
        before = all_feature_df.shape[0]
        if otu_filter < 1:
            all_feature_df = all_feature_df.loc[
                             all_feature_df.apply(lambda x: sum(x > 0), axis=1) > all_feature_df.shape[1] * otu_filter,
                             :]
        else:
            all_feature_df = all_feature_df.loc[all_feature_df.apply(lambda x: sum(x > 0), axis=1) > 10, :]

        print(f"##Reduced from {before} to {all_feature_df.shape[0]}")

    # Read Count Filter
    if num_seq_filter:
        print(f"##Removing samples with less than {num_seq_filter} reads")
        before = all_feature_df.shape[1]
        all_metadata_df = all_metadata_df.loc[all_metadata_df['num_seqs'] > num_seq_filter]
        print(f"Removed samples from {before} to {all_feature_df.shape[1]}")

    # Matching the Feature and Metadata info
    sample_list = set(all_feature_df.columns).intersection(set(all_metadata_df.index))
    failed_metadata_df = all_metadata_df.loc[set(all_metadata_df.index) - set(all_feature_df.columns)]
    all_metadata_df = all_metadata_df.loc[sample_list]
    all_feature_df = all_feature_df.loc[:, sample_list]

    return all_feature_df, all_metadata_df


