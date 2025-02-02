import json
import os

import google.generativeai as genai
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
    text_prompt = \
        '''
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
            '''.format(file_id=file_id, text=text, content_level=content_level)

    prompt = [text_prompt]

    audio_files = genai.list_files()
    for f in audio_files:
        if f.display_name == file_id:
            prompt.append(genai.get_file(f.name))

    return prompt


if __name__ == '__main__':

    model = initialize_model(name='gemini-1.5-flash-8b', temperature=0.4, top_k=3, top_p=0.75)
    upload_files()
    df = pd.read_csv('../data/level_mapping.csv', dtype=str)

    count = 0
    results = []
    for i, row in df.iterrows():
        count += 1
        if count <= 100:
            prompt = create_prompt(file_id=row['file_id'], text=row['text'], content_level=row['content_level'])
            result = model.generate_content(prompt)
            result_dict = json.loads(result.text)
            results.append(result_dict)
            print('\n{}\n'.format(result_dict))
            with open('../output/trial.json', 'w') as f:
                json.dump(results, f)
