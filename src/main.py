import os

import google.generativeai as genai
import yaml


def initialize_model(name='gemini-1.5-pro',
                     temperature=1,
                     top_p=0.95,
                     top_k=40,
                     max_output_tokens=8192,
                     response_mime_type='text/plain'):
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

    files_to_upload = [{'name': x.replace('.wav', ''), "path": os.path.join(directory, x)} for x in os.listdir(directory) if x.endswith('.wav')]
    uploaded_files = list(genai.list_files())
    uploaded_filenames = [x.display_name for x in uploaded_files]

    upload_count = 0
    for file in files_to_upload:
        if file['name'] not in uploaded_filenames:
            genai.upload_file(file['path'], name=file['name'], display_name=file['name'])
            upload_count += 1

    print('Uploaded {} files.'.format(upload_count))
    return upload_count


def create_prompt():

    prompt = [
        '''
        Here is a CSV that contains information about Hindi reading assessments of students.
        The first column of the CSV contains the filename of the voice recording for that student's assessment.
        The second column of the CSV contains the level of the content that they are reading in that recording. 
        The third column of the CSV contains the reading level that has been assigned to that student's recording. 
        Note that the reading level can never be higher than the level of the content in a recording. 
        The files with the same display_name have been uploaded into the Gemini system. 
        Learning from this data, help me assess the reading level of the recordings with the following display_names:  
        ['0241', '0246', '0253', '0254', '0271'].
        The content level of these recordings is 6. 
        Give an answer in JSON format that looks like 
        {output: [{"filename": [string], "content_level": [int], "reading_level": [int], "reason_for_reading_level": [string]}, {...}, ...]
        ''',
        '''
        CSV with Hindi reading assessments of students: \n
        filename,content_level,reading_level
        0003,6,6
        0004,6,6
        0011,6,5
        0012,6,5
        0020,5,4
        0024,5,4
        0072,4,3
        0085,4,3
        0098,5,5
        0177,4,4
        0194,2,2
        0297,6,2
        ''',
        ]

    audio_files = genai.list_files()

    for f in audio_files:
        prompt.append(genai.get_file(f.name))

    return prompt

if __name__ == '__main__':
    model = initialize_model()
    upload_files()
    prompt = create_prompt()
    result = model.generate_content(prompt)
    print(result.text)
    pass