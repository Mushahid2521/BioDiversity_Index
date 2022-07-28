import pandas as pd


def format_stats_file_to_df(stat_file_path: str):
    stats_ls = []
    with open(stat_file_path, 'r') as stat:
        lines = [l.strip() for l in stat.readlines()]
        stats_ls.append(lines[0])

        for line in lines:
            if not line.startswith("file"):
                stats_ls.append(line)

    columns = [l.strip() for l in stats_ls[0].split()]
    stats_dict = {col: [] for col in columns}

    for ln in stats_ls[1:]:
        splits = ln.split()
        if splits[0].__contains__("Samples"):
            splits[0] = splits[0].split("/")[1].split(".")[0] + "_1.fastq"
        for idx, col in enumerate(stats_dict):
            stats_dict[col].append(splits[idx].replace(',', ''))

    stats_df = pd.DataFrame(stats_dict)
    final_stat_dict = {'run_id': [], 'num_seqs': [], 'sum_len': [], 'min_len': [], 'avg_len': [], 'max_len': []}

    for i, item in stats_df.iterrows():
        id_ = str(item['file']).split(".")[0].split("_")[0]
        if id_ + "_2.fastq" in stats_df['file']:
            final_stat_dict['num_seqs'].append(int(item['num_seqs']) * 2)
        else:
            final_stat_dict['num_seqs'].append(int(item['num_seqs']))

        final_stat_dict['run_id'].append(id_)
        final_stat_dict['sum_len'].append(item['sum_len'])
        final_stat_dict['min_len'].append(item['min_len'])
        final_stat_dict['avg_len'].append(item['avg_len'])
        final_stat_dict['max_len'].append(item['max_len'])

    final_stat_df = pd.DataFrame(final_stat_dict)
    final_stat_df = final_stat_df.loc[~final_stat_df.run_id.duplicated(keep="first")]
    final_stat_df = final_stat_df.set_index('run_id')

    return final_stat_df


def format_se_pe_to_df(se_pe_file: str):
    library_dict = {'run_id': [], 'library_type': []}
    with open(se_pe_file, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for l in lines:
            splits = l.split()
            library_dict['run_id'].append(splits[0])
            library_dict['library_type'].append(splits[1])

    library_df = pd.DataFrame(library_dict)
    library_df = library_df.loc[~library_df.run_id.duplicated(keep="first")]
    library_df = library_df.set_index('run_id')

    return library_df


def merge_multiple_df(df_files: list):
    df_list = []
    for file in df_files:
        df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, axis=0)
    df = df.drop_duplicates(subset='run_id', keep="last")
    df = df.set_index('run_id')
    return df


def merge_other_metas(metas_list: list, stats_list: list, library_list: list):
    basic_metas = merge_multiple_df(metas_list)
    stats_metas = pd.concat([format_stats_file_to_df(file) for file in stats_list], axis=0)
    library_metas = pd.concat([format_se_pe_to_df(file) for file in library_list])

    return pd.concat([basic_metas, stats_metas, library_metas], axis=1)