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


def upload_files(directory='../media'):

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

if __name__ == '__main__':
    model = initialize_model()
    upload_files()


# myfile = genai.upload_file('../media/0003.wav')
# print(f"{myfile=}")
#
# model = genai.GenerativeModel("gemini-1.5-flash")
# result = model.generate_content([myfile, "Classify whether the speaker is a fluent reader of Hindi or not. Answer with a Yes or No."])
# print(f"{result.text=}")
#