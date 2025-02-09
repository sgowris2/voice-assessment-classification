import json
from datetime import datetime

from src.gemini_api_methods import initialize_model, upload_files
from src.analysis import merge_results_with_actual_values, visualize_metrics
from src.prompt_creation import create_prompt, _get_prompt_template
from src.raw_data_parsers import get_level_mapping_df

if __name__ == '__main__':

    run_model = False
    run_analysis = True
    show_plots = True
    output_filepath = '../output/{}.json'.format(datetime.strftime(datetime.now(), format('%d%b-%H%M')))
    analysis_filepath = None # Set to None to use output_filepath, else set an actual filepath

    df = get_level_mapping_df()
    if run_model:
        model = initialize_model(name='gemini-2.0-flash',
                                 temperature=0,
                                 top_k=5,
                                 top_p=0.5)
        upload_files()
        count = 0
        results = []
        result_file_content = {
            'prompt_template': _get_prompt_template(level=None)[0],
            'level_rules': _get_prompt_template(level=None)[1],
            'predictions': []
        }
        for i, row in df.iterrows():
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
        merged_df = merge_results_with_actual_values(results, df)
        visualize_metrics(merged_df, show_plots=show_plots)
