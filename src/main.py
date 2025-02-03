from datetime import datetime
import json
import os
from typing import List, Dict

import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def initialize_model(name='gemini-1.5-pro',
                     temperature=1.0,
                     top_p=0.95,
                     top_k=40,
                     max_output_tokens=8192,
                     response_mime_type='application/json'):
    with open('../secret.yaml', 'r') as secret_file:
        secrets = yaml.load(secret_file, Loader=yaml.SafeLoader)
        gemini_api_key = secrets['GEMINI_API_KEY']
    genai.configure(api_key=gemini_api_key)

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": response_mime_type,
    }
    genai_model = genai.GenerativeModel(model_name=name, generation_config=generation_config)
    return genai_model


def upload_files(directory='../data'):
    files_to_upload = [{'name': x.replace('.wav', ''), "path": os.path.join(directory, x)} for x in
                       os.listdir(directory) if x.endswith('.wav')]
    uploaded_files = list(genai.list_files())
    uploaded_filenames = [x.display_name for x in uploaded_files]

    upload_count = 0
    for file in files_to_upload:
        if file['name'] not in uploaded_filenames:
            genai.upload_file(file['path'], name=file['name'], display_name=file['name'])
            upload_count += 1

    print('Uploaded {} files.'.format(upload_count))
    return upload_count


def create_prompt(file_id, text, content_level):

    text_prompt = _get_prompt_template().format(file_id=file_id, text=text, content_level=content_level)

    prompt = [text_prompt]

    audio_files = genai.list_files()
    for f in audio_files:
        if f.display_name == file_id:
            prompt.append(genai.get_file(f.name))

    return prompt


def merge_results_with_actual_values(analysis_results: List[Dict], level_mapping_df: pd.DataFrame):
    level_mapping_df['measured_level'] = None
    for i in analysis_results:
        file_id = str(i['file_id'])
        measured_reading_level = int(i['reading_level'])
        level_mapping_df.loc[level_mapping_df['file_id'] == file_id, 'measured_level'] = measured_reading_level
    return level_mapping_df


def visualize_metrics(results_df: pd.DataFrame, show_plots=True):

    levels = [1, 2, 3, 4, 5, 6]
    actual_level_dfs = {x: results_df[results_df['reading_level'] == x] for x in levels}
    predicted_level_dfs = {x: results_df[results_df['measured_level'] == x] for x in levels}
    total_no_samples = results_df.shape[0]
    overall_accuracy = round(100 * (results_df.loc[(results_df['reading_level'] == results_df['measured_level'])].shape[0])
                           / total_no_samples, 0)
    overall_accuracy_within_one_level = round(100 * (results_df.loc[abs(results_df['reading_level'] - results_df['measured_level']) <= 1].shape[0])
                           / total_no_samples, 0)
    precisions = {x: _get_precision_for_level(results_df, x) for x in levels}
    recalls = {x: _get_recall_for_level(results_df, x) for x in levels}
    f1_scores = {x: _get_f1_score(precisions[x], recalls[x]) for x in levels}

    print('Total Samples: {}'.format(total_no_samples))
    print('Overall Accuracy: {}'.format(overall_accuracy))
    print('Overall Accuracy Within One Level: {}'.format(overall_accuracy_within_one_level))
    print('Precisions: {}'.format(precisions))
    print('Recalls: {}'.format(recalls))
    print('F1 Scores: {}'.format(f1_scores))

    if show_plots:
        fig, axs = plt.subplots(nrows=2, ncols=3)
        for level in levels:
            axis = axs[(level - 1) // 3][(level - 1) % 3]
            axis.hist(x=list(actual_level_dfs[level]['measured_level']),
                      bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                      color='skyblue')
            axis.set_title('Actual Level {}'.format(level))
            axis.set_xlabel('Predicted Level')
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


def _get_prompt_template():

    return '''
            Here is a recording of a student reading a Hindi text (short story).
            The actual text that the student is reading is the following:
            ```
            {text}
            ```
            The level of this text is: Level {content_level}
            
            For the recording file, do the following:
            1. First, create a transcript from the recording and compare the recording transcript with the actual text.
            2. Then, assess the reading level of the student using the reading levels which are given as follows:
                ```
                Level 1 - Unable to recognize most letters.
                Level 2 - Pre-reader: Recognizes letters (like क, ख, ग, etc.). But unable to read any words.
                Level 3 - Beginner Word Reader: Reads only very short, common words with mistakes and multiple attempts.
                Level 4 - Word Reader: Reads simple words with limited mistakes. Can read simple sentences in a word-by-word fashion.
                Level 5 - Sentence Reader: Reads sentences but with limited comprehension and inaccurate intonation.
                Level 6 - Paragraph Reader: Reads sentences fluently with reasonable comprehension and mostly correct intonation.
                Level 7 - Story Reader: Reads a more complex Hindi text with fluency, comprehension, and no mistakes.
                ```
            
            Note that the reading level of a student cannot be assessed higher than the level of the text.
            
            *
            Give an answer in JSON format that looks like:
            {{"file_id": {file_id}, "reading_level": int, "transcript": string, "reason_for_reading_level": string}}
            '''


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
    if precision is None or recall is None:
        return None
    precision = precision / 100.0
    recall = recall / 100.0
    return round((2 * precision * recall) / (precision + recall), 2)


if __name__ == '__main__':

    run_model = False
    run_analysis = True
    show_plots = False
    output_filepath = '../output/{}.json'.format(datetime.strftime(datetime.now(), format('%d%b-%H%M')))
    analysis_filepath = '../output/trial.json'  # Set to None to use output_filepath

    level_mapping_df = pd.read_csv('../data/level_mapping.csv', dtype=str)
    level_mapping_df['content_level'] = level_mapping_df['content_level'].astype(int)
    level_mapping_df['reading_level'] = level_mapping_df['reading_level'].astype(int)
    if run_model:
        model = initialize_model(name='gemini-1.5-flash-8b', temperature=0.4, top_k=3, top_p=0.75)
        upload_files()
        count = 0
        results = []
        result_file_content = {
            'prompt_template': _get_prompt_template(),
            'predictions': []
        }
        for i, row in level_mapping_df.iterrows():
            count += 1
            if count <= 100:
                prompt = create_prompt(file_id=row['file_id'], text=row['text'], content_level=row['content_level'])
                result = model.generate_content(prompt)
                result_dict = json.loads(result.text)
                results.append(result_dict)
                result_file_content['predictions'] = results
                print('\n{}\n'.format(result_dict))
                with open(output_filepath, 'w') as f:
                    json.dump(result_file_content, f)

    if run_analysis:
        if analysis_filepath is None:
            analysis_filepath = output_filepath
        with open(analysis_filepath, 'r') as f:
            file_contents = json.load(f)
            results = file_contents['predictions']
        merged_df = merge_results_with_actual_values(results, level_mapping_df)
        visualize_metrics(merged_df, show_plots=show_plots)
