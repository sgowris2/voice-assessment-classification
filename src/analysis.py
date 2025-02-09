from typing import List, Dict

import pandas as pd
from matplotlib import pyplot as plt, lines as mlines


def merge_results_with_actual_values(analysis_results: List[Dict], data_df: pd.DataFrame):
    data_df['measured_level'] = None
    data_df['actual_meets_content_level'] = None
    data_df['predicted_meets_content_level'] = None
    for i in analysis_results:
        file_id = str(i['file_id'])
        measured_reading_level = int(i['reading_level'])
        data_df.loc[data_df['file_id'] == file_id, 'measured_level'] = measured_reading_level
        actual_content_level = data_df.loc[data_df['file_id'] == file_id, 'content_level'].item()
        actual_reading_level = data_df.loc[data_df['file_id'] == file_id, 'reading_level'].item()
        data_df.loc[data_df['file_id'] == file_id, 'actual_meets_content_level'] = \
            1 if actual_content_level == actual_reading_level else 0
        data_df.loc[data_df['file_id'] == file_id, 'predicted_meets_content_level'] = \
            1 if actual_content_level == measured_reading_level else 0
    data_df = data_df.loc[(data_df['measured_level'].notna()) & (data_df['measured_level'] > 0)]
    return data_df


def visualize_metrics(results_df: pd.DataFrame, show_plots=True):
    levels = [1, 2, 3, 4, 5, 6]
    actual_level_dfs = {x: results_df[results_df['reading_level'] == x] for x in levels}
    predicted_level_dfs = {x: results_df[results_df['measured_level'] == x] for x in levels}
    total_no_samples = results_df.shape[0]
    overall_accuracy = round(
        100 * (results_df.loc[(results_df['reading_level'] == results_df['measured_level'])].shape[0])
        / total_no_samples, 0)
    overall_accuracy_within_one_level = round(
        100 * (results_df.loc[abs(results_df['reading_level'] - results_df['measured_level']) <= 1].shape[0])
        / total_no_samples, 0)
    accuracy_by_content_level = {x: _get_accuracy_by_content_level(results_df, x) for x in levels}
    precisions = {x: _get_precision_for_level(results_df, x) for x in levels}
    recalls = {x: _get_recall_for_level(results_df, x) for x in levels}
    f1_scores = {x: _get_f1_score(precisions[x], recalls[x]) for x in levels}

    content_reading_level_within_one_level_df = results_df.loc[
        abs(results_df['content_level'] - results_df['reading_level']) <= 1]
    precisions_for_crlw1l = {x: _get_precision_for_level(content_reading_level_within_one_level_df, x) for x in levels}
    recalls_for_crlw1l = {x: _get_recall_for_level(content_reading_level_within_one_level_df, x) for x in levels}
    f1_scores_for_crlw1l = {x: _get_f1_score(precisions_for_crlw1l[x], recalls_for_crlw1l[x]) for x in levels}

    samples_with_more_than_1_level_difference = \
        list(results_df.loc[abs(results_df.reading_level - results_df.measured_level) > 1]['file_id'])

    content_level_match_accuracy = (
        round(100.0 * (results_df.loc[results_df.actual_meets_content_level == results_df.predicted_meets_content_level]).shape[0]
              / (results_df.shape[0]), 0))

    print('\nTotal Samples: {}'.format(total_no_samples))
    print('Overall Accuracy: {}%'.format(overall_accuracy))
    print('Overall Accuracy Within One Level: {}%'.format(overall_accuracy_within_one_level))
    print('Content Level Matching Accuracy: {}%'.format(content_level_match_accuracy))
    print('Accuracy By Content Level: {}'.format(accuracy_by_content_level))
    print('Samples with >1 level difference: {}'.format(samples_with_more_than_1_level_difference))
    print('***************************************')
    print('Precisions: {}'.format(precisions))
    print('Recalls: {}'.format(recalls))
    print('F1 Scores: {}'.format(f1_scores))
    print('Precisions for CRLW1L: {}'.format(precisions_for_crlw1l))
    print('Recalls for CRLW1L: {}'.format(recalls_for_crlw1l))
    print('F1 Scores for CRLW1L: {}'.format(f1_scores_for_crlw1l))

    if show_plots:
        fig, axs = plt.subplots(nrows=2, ncols=3)
        for level in levels:
            axis = axs[(level - 1) // 3][(level - 1) % 3]
            axis.hist(x=list(actual_level_dfs[level]['measured_level']),
                      bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                      color='skyblue')
            axis.set_title('Actual Level {}'.format(level))
            axis.set_xlabel('Predicted Level')
            axis.set_ylabel('Frequency')
            axis.set_ylim(0, 15)
            axis.tick_params(left=False, bottom=False)
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(nrows=2, ncols=3)
        for level in levels:
            axis = axs[(level - 1) // 3][(level - 1) % 3]
            axis.hist(x=list(predicted_level_dfs[level]['reading_level']),
                      bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                      color='skyblue')
            axis.set_title('Predicted Level {}'.format(level))
            axis.set_xlabel('Actual Level')
            axis.set_ylim(0, 15)
            axis.tick_params(left=False, bottom=False)
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(nrows=2, ncols=3)
        fig.suptitle('Auto-level Mapping Results By Content Level')
        plt.title('{} Samples'.format(results_df.shape[0]))

        for level in levels:
            content_level_df = results_df.loc[results_df['content_level'] == level]
            content_level_df = content_level_df.copy()
            content_level_df['exact_match'] = content_level_df.apply(lambda x: x['reading_level'] == x['measured_level'], axis=1)
            axis = axs[(level - 1) // 3][(level - 1) % 3]
            xs = []
            ys = []
            sizes = []
            colors = []
            for i in range(1, level + 1):
                for j in range(1, level + 1):
                    size = content_level_df.loc[(content_level_df.reading_level == i) & (content_level_df.measured_level == j)].shape[0]
                    xs.append(i)
                    ys.append(j)
                    sizes.append(100*size)
                    if i == j:
                        colors.append('lightgreen')
                    elif abs(i - j) == 1:
                        colors.append('skyblue')
                    else:
                        colors.append('red')
            plt.rc('grid', linestyle="-", color='black')
            axis.grid()
            axis.scatter(x=xs, y=ys, s=sizes, c=colors)
            axis.plot([0, 7], [0, 7], 'g-')
            axis.set_title('Content Level {} (n={})'.format(level, content_level_df.shape[0]))
            axis.set_xlabel('Actual Level')
            axis.set_ylabel('Predicted Level')
            axis.set_xlim(0, 7)
            axis.set_ylim(0, 7)
            axis.tick_params(left=False, bottom=False)

        exact_match = mlines.Line2D([], [], color='lightgreen', marker='o', linestyle='None',
                                    markersize=5, label='Exact Matches')
        one_level_match = mlines.Line2D([], [], color='skyblue', marker='o', linestyle='None',
                                        markersize=5, label='Match within 1 Level')
        no_match = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                 markersize=5, label='Match outside 1 Level')
        plt.legend(handles=[exact_match, one_level_match, no_match], bbox_to_anchor=(1.1, 1.05))
        plt.tight_layout()
        plt.show()


def _get_accuracy_by_content_level(df: pd.DataFrame, level: int):
    content_level_df = df.loc[df['content_level'] == level]
    if content_level_df.shape[0] == 0:
        return None
    true_positives = content_level_df.loc[content_level_df['reading_level'] == content_level_df['measured_level']]
    return round(100 * true_positives.shape[0] / content_level_df.shape[0], 0)


def _get_precision_for_level(df: pd.DataFrame, level: int):
    true_positives = df.loc[(df['reading_level'] == level) & (df['measured_level'] == level)]
    predicted_positives = df.loc[df['measured_level'] == level]
    if predicted_positives.shape[0] == 0:
        return None
    return round(100 * true_positives.shape[0] / predicted_positives.shape[0], 0)


def _get_recall_for_level(df: pd.DataFrame, level: int):
    true_positives = df.loc[(df['reading_level'] == level) & (df['measured_level'] == level)]
    actual_positives = df.loc[df['reading_level'] == level]
    if actual_positives.shape[0] == 0:
        return None
    return round(100 * true_positives.shape[0] / actual_positives.shape[0], 0)


def _get_f1_score(precision, recall):
    if precision is None or recall is None or ((precision + recall) == 0):
        return None
    precision = precision / 100.0
    recall = recall / 100.0
    return round((2 * precision * recall) / (precision + recall), 2)
