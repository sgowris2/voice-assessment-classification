import json
import os
from datetime import datetime
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
    precisions = {x: _get_precision_for_level(results_df, x) for x in levels}
    recalls = {x: _get_recall_for_level(results_df, x) for x in levels}
    f1_scores = {x: _get_f1_score(precisions[x], recalls[x]) for x in levels}

    content_reading_level_within_one_level_df = results_df.loc[
        abs(results_df['content_level'] - results_df['reading_level']) <= 1]
    precisions_for_crlw1l = {x: _get_precision_for_level(content_reading_level_within_one_level_df, x) for x in levels}
    recalls_for_crlw1l = {x: _get_recall_for_level(content_reading_level_within_one_level_df, x) for x in levels}
    f1_scores_for_crlw1l = {x: _get_f1_score(precisions_for_crlw1l[x], recalls_for_crlw1l[x]) for x in levels}

    content_level_match_accuracy = (
        round(100.0 * (results_df.loc[results_df.actual_meets_content_level == results_df.predicted_meets_content_level]).shape[0]
              / (results_df.shape[0]), 0))

    print('\nTotal Samples: {}'.format(total_no_samples))
    print('Overall Accuracy: {}%'.format(overall_accuracy))
    print('Overall Accuracy Within One Level: {}%'.format(overall_accuracy_within_one_level))
    print('Content Level Matching Accuracy: {}%'.format(content_level_match_accuracy))
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
            You are a reading level assessment bot that listens to recordings of students reading text in Hindi, and you need to assess the ability of the student to read.
            This is not a strict test, but more an assessment of the ability of the student. Mistakes are okay to make but you need to figure out what level a student is capable of using the recording as evidence.
            
            Here is a recording of a student reading some text in Hindi language.
            The actual text (transcript) that the student is reading is the following:
            ```
            {text}
            ```
            The content level of this text is: Level {content_level}
            
            
            For the recording file, do the following:
            1. First, create a transcript from the recording.
            2. Then, assess how what percentage of the actual text was read accurately by the student. You may ignore it if the student omits reading a small word like "है" at the end of a sentence.
            3. Then, note down differences in terms of fluency, and comprehension of the reading.
            4. Then, use this information to assign a reading_level for the student in the recording using the following rules:
                ```
                1. If the recording is not clear and any assessment is not possible, then assign reading_level of 0.
                2. The reading_level of a student can never be assigned a level higher than the content_level of the text.
                3. For texts of any level, if the student is able to read most of the text accurately and reasonable fluently with few mistakes, the reading_level of the student is the same as the content_level.                
                4. If the student is able to read the text with medium fluency but makes multiple regular mistakes, then the reading_level is 1 level less than the content_level.               
                5. If the student is struggling to read most of the text correctly, but is able to read some words and phrases correctly, then the reading_level is 2 levels or more below the content_level.
                6. For texts of content_level of 4 or more, if the student demonstrates ability to recognize letters and simple words, and is able to read short phrases, then the reading_level is Level 3. If the student is unable to read short phrases, then the reading_level is Level 2.
                7. If the student is only able to read some small basic words and letters, but is unable to read phrases, then level of the student is Level 2.
                8. If the student is unable to read most of the text correctly, including unable to recognize most words and some letters, then the level of the student is Level 1.
                ```
            
            *
            Give an answer in the following valid JSON format:
            {{
                "file_id": "{file_id}", 
                "recording_transcript": string, 
                "percent_accuracy": int,
                "recognizes_letters": bool,
                "reads_simple_words": bool,
                "reads_short_phrases": bool,
                "reads_complex_sentences": bool,
                "differences_between_actual_transcript_and_actual_recording": [list of differences as strings], 
                "reason_for_reading_level": string,
                "reading_level": integer
            }}
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
    if precision is None or recall is None or ((precision + recall) == 0):
        return None
    precision = precision / 100.0
    recall = recall / 100.0
    return round((2 * precision * recall) / (precision + recall), 2)


if __name__ == '__main__':

    run_model = False
    run_analysis = True
    show_plots = True
    output_filepath = '../output/{}.json'.format(datetime.strftime(datetime.now(), format('%d%b-%H%M')))
    analysis_filepath = '../output/04Feb-0016.json'  # Set to None to use output_filepath, else set an actual filepath

    level_mapping_df = pd.read_csv('../data/level_mapping.csv', dtype=str)
    level_mapping_df['content_level'] = level_mapping_df['content_level'].astype(int)
    level_mapping_df['reading_level'] = level_mapping_df['reading_level'].astype(int)
    if run_model:
        model = initialize_model(name='gemini-1.5-flash',
                                 temperature=1,
                                 top_k=20,
                                 top_p=0.5)
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
                try:
                    prompt = create_prompt(file_id=row['file_id'], text=row['text'], content_level=row['content_level'])
                    result = model.generate_content(prompt)
                    result_dict = json.loads(result.text)
                    results.append(result_dict)
                    result_file_content['predictions'] = results
                    print('\n{}\n'.format(result_dict))
                    with open(output_filepath, 'w') as f:
                        json.dump(result_file_content, f)
                except Exception as e:
                    print('****************\nException:\n{}\n***************'.format(e, e.__traceback__))

    if run_analysis:
        if analysis_filepath is None:
            analysis_filepath = output_filepath
        with open(analysis_filepath, 'r') as f:
            file_contents = json.load(f)
            results = file_contents['predictions']
        merged_df = merge_results_with_actual_values(results, level_mapping_df)
        visualize_metrics(merged_df, show_plots=show_plots)
