from fmcore.mapper.llm_response_json_mapper import LLMResponseJsonMapper

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Valid single dictionary
        '{"key": "value"}',
        
        # Valid list of dictionaries
        '[{"key1": "value1"}, {"key2": "value2"}]',
        
        # Malformed JSON (missing quotes)
        '{key: value}',
        
        # Malformed JSON (missing comma)
        '[{"key1": "value1"}{"key2": "value2"}]',
        
        # Invalid JSON (will be caught by exception)
        'not a json string',

        # Complex string with JSON embedded in text
        '''content='Let me analyze this step by step:

1. From the Amazon product data, I notice several key points about the fit:
{}
{'value': 'abc'}

Any Text

{'key': [{"reason": "reason1", "prediction": "Incorrect"}]}'''
    ]

    def test_sync():
        mapper = LLMResponseJsonMapper()
        print("\nTesting JSON mapping:")
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = mapper.map(test_case)
                print(f"\nTest case {i}:")
                print(f"Output: {result}")
            except Exception as e:
                pass

    # Run tests
    test_sync()