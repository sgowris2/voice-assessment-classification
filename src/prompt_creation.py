from google import generativeai as genai


def create_prompt(file_id, text, content_level):
    text_prompt = _get_prompt_template(level=content_level).format(file_id=file_id, text=text, content_level=content_level)

    prompt = [text_prompt]

    audio_files = genai.list_files()
    for f in audio_files:
        if f.display_name == file_id:
            prompt.append(genai.get_file(f.name))

    return prompt


def _get_prompt_template(level: int or None):

    level_rules = {
        2: '''
            - If the student can recognize 80% of letters / words correctly and fluently, then reading_level is Level 2.
            - If the student does not fulfill the criteria for Level 2, then you can assign reading_level of Level 1.
            ''',
        3: '''
            - If the student is able to read 70% of the text fluently, and about 5 words without making more than 2 mistakes, without breaking the words into letters, and without stopping between words more than 2 times, then reading_level is Level 3.
            - If the student does not fulfill the criteria for Level 3, then check whether they can recognize letters correctly or not. If they can recognize more than 70% of letters correctly, then you can assign reading_level Level 2.
            - If the student does not fulfill the criteria for Level 2, then assign Level 1.
            ''',
        4: '''
            - If the student is able to read 70% of the text fluently, with 2 sentences in continuation with less than 2 mistakes, then the reading_level is Level 4.
            - If the student does not fulfill the criteria for Level 4, then check whether they can read simple words one by one, about 4 or more words without making more than 3 mistakes, without breaking the word into letters, and without stopping between words more than 3 times, then assign reading_level of Level 3.
            - If the student does not fulfill the criteria for Level 3, then check whether they can recognize letters correctly or not. If they can recognize more than 70% of letters correctly, then you can assign reading_level of Level 2.
            - If the student does not fulfill the criteria for Level 2, then assign Level 1.
            ''',
        5: '''
            - If the student is able to read more than 70% of the text, with 2 sentences in continuation with less than 4 mistakes, then the reading_level is Level 5.
            - If the student does not fulfill the criteria for Level 5, then check whether they can read a simple sentences fairly well without reading each word one-by-one, then assign reading_level of Level 4.
            - If the student does not fulfill the criteria for Level 4, then check whether they can read some simple words one by one, about 4 or more words without making more than 3 mistakes, without breaking the word into individual letters, then assign reading_level of Level 3.
            - If the student does not fulfill the criteria for Level 3, then check whether they can recognize letters correctly or not. If they can recognize some letters correctly, then you can assign reading_level of Level 2.
            - If the student does not fulfill the criteria for Level 2, then assign Level 1.
            ''',
        6: '''
            - If the student is able to read more than 60% of the text fluently, read 2-3 consecutive sentences fluently, quickly, and easily with only few, minor mistakes, then the reading_level is Level 6.
            - If the student does not fulfill the criteria for Level 6, then check whether they can read at least 1 long sentence fluently, with less than 2 mistakes, and without reading sentences in a word-by-word, broken up fashion, and without pauses between words, then assign reading_level of Level 5.
            - If the student does not fulfill the criteria for Level 5, then check whether they can read at least 3 phrases fluently without reading each word in a one-by-one, broken-up fashion, then assign reading_level of Level 4.
            - If the student does not fulfill the criteria for Level 4, then check whether they can read simple words one by one, about 4 or more words without making more than 3 mistakes, without breaking the word into individual letters, then assign reading_level of Level 3.
            - If the student does not fulfill the criteria for Level 3, then check whether they can recognize letters correctly or not. If they can recognize some letters correctly, or they read words by reading it letter by letter, then you can assign reading_level of Level 2.
            - If the student does not fulfill the criteria for Level 2, then assign Level 1. 
            '''
    }

    template =  '''
            You are a reading level assessment bot that listens to recordings of students reading text in Hindi, and you need to assess the ability of the student to read.
            This is not a strict test, but more an assessment of the ability of the student. Mistakes are okay to make but you need to figure out what reading_level a student is capable of, using the recording as evidence.
            
            Here is a recording of a student reading some text in Hindi language.
            The actual text (transcript) that the student is reading is the following:
            ```
            {{text}}
            ```
            The content_level of this text is: Level {{content_level}}
            
            For the recording file, do the following:
            1. First, create a transcript from the recording.
            2. Then, assess how many mistakes are made by the student while reading the text. You may ignore it if the student omits reading a small word like "है" at the end of a sentence.
            3. Then, calculate approximately what percentage of the text was read correctly and fluently.
            3. Then, assign a reading_level to the student based on the following rules:
            ```
                - If the recording is not clear and any assessment is not possible, then assign reading_level of 0.
                - The reading_level of a student can never be assigned a level higher than the content_level of the text.
                {level_specific_rules}
            ```
            *
            Give an answer in the following valid JSON format:
            {{{{
                "file_id": "{{file_id}}", 
                "recording_transcript": string, 
                "percent_accuracy": int,
                "reads_letters_fluently": bool,
                "reads_simple_words_fluently": bool,
                "reads_short_phrases_fluently": bool,
                "reads_simple_sentences_fluently": bool,
                "reads_paragraphs_fluently": bool,
                "differences_between_actual_transcript_and_actual_recording": [list of differences as strings], 
                "reason_for_reading_level": string,
                "content_level": integer,
                "reading_level": integer
            }}}}
            '''

    if level is None:
        return template, level_rules

    if level < 1 or level > 6:
        raise Exception('content_level of the text is not within known range.')

    return template.format(level_specific_rules=level_rules[level])
