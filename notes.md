# Project Notes

- Meeting: Feb 8, 2025
  - Presented Results
    - 55-60% exact match accuracy
    - 90%+ match accuracy within one level
    - 80% accuracy of binary classification of content_level == reading_level or not.
  - Actions
    - Using existing recordings, check accuracy when there are multiple recordings for a student.
    - If successful, suggest design of pilot with multiple recordings per student.

### Stats

- 6829 total entries
- 4621 entries with levels mapped between 1 and 6
- 4620 entries with no data missing
- 2640 entries with WadhwaniAI URLs that have some indication of Saajhedaar + Student Name
- 2052 entries that have unique recordings (not duplicates)
- 1464 recordings that do not have conflicting level mapping
- 160 students that have more than 1 recording
- 348 total recordings across the 160 students

### Next Steps
- Script to auto-download recordings of students with multiple recordings
- Run analysis on above recordings.
- Check accuracy of mapping based on highest level reached per student across recordings:
    - Actual vs. predicted accuracy across all students
    - Average RMSE across all students
- Is this accuracy significantly better than accuracy across individual recordings?