# Running Tests

## Setup
```bash
# Set Python path to include fmcore source
export PYTHONPATH="<fmcore_git_repo_path>/fmcore/src/:$PYTHONPATH"
```

## Run Tests
```bash
# Test prompt tuning functionality with different configurations
# This test demonstrates prompt tuning with classification and LLM-as-judge scenarios
python local/prompt_tuner.py

# Test credential refresh and API stability
# This test makes continuous LLM API calls with 5-minute intervals to verify credential refresh
# and monitor API stability over extended periods. Run this test for 1 hour+ using access and 
# secret keys and not using ADA to ensure the refreshing credentials runs correctly
python local/credential_refresh_test.py

# Test LLM response JSON mapping functionality
# This test verifies the JSON mapper's ability to handle various JSON formats and edge cases
# including valid JSON, malformed JSON, and complex text with embedded JSON
python local/llm_response_json_mapper_test.py
```