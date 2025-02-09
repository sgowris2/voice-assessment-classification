import pandas as pd
import re

def get_multiple_recordings_per_student_df():
    df = pd.read_csv('../data/voice_notes.csv')
    df.set_index('file_id', inplace=True)
    df.drop(axis=1, columns=['recording_url2', 'level2'], inplace=True)
    df = df[df.level.isin(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6'])]
    df.dropna(axis='rows', how='any', inplace=True)
    df = df[df.recording_url.str.startswith('https://wadhwani-ai-voice-data')]
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset='recording_url', keep=False, inplace=True)
    df['student'] = df.recording_url
    df.student = df['student'].apply(lambda x: x.replace('https://wadhwani-ai-voice-data.s3.ap-south-1.amazonaws.com/hindi/audio/', ''))
    df.student = df['student'].apply(lambda x: re.sub(pattern='_[0-9._]+.wav', repl='', string=x))
    df.student = df['student'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df_with_multiple_recordings = df[df.duplicated(subset=['student'], keep=False)]
    df_with_multiple_recordings = df_with_multiple_recordings.drop_duplicates(subset='recording_url', keep=False)
    return df_with_multiple_recordings


def get_level_mapping_df():
    df = pd.read_csv('../data/level_mapping.csv', dtype=str)
    df['content_level'] = df['content_level'].astype(int)
    df['reading_level'] = df['reading_level'].astype(int)
    return df
