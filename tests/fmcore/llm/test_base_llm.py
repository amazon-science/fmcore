import pytest
from abc import ABC
from unittest.mock import patch
from langchain_core.messages import AIMessage

from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.types.enums.provider_enums import ProviderType
from tests.fmcore.llm.conftest import bedrock_llm_config
from langchain.chains.combine_documents import create_stuff_documents_chain



def test_base_llm_registry():
    """Test that BaseLLM properly registers subclasses."""
    assert TestLLM in BaseLLM.get_subclasses()
    assert BaseLLM.get_subclass(key=ProviderType.BEDROCK.name) == TestLLM

def test_base_llm_of(bedrock_llm_config):
    """Test the of() factory method."""
    with patch('fmcore.llm.base_llm.BaseLLM.get_subclass') as mock_get_subclass:
        mock_get_subclass.return_value = TestLLM
        
        llm = BaseLLM.of(llm_config=bedrock_llm_config)
        
        assert isinstance(llm, TestLLM)
        assert llm.config == bedrock_llm_config
        mock_get_subclass.assert_called_once_with(key=bedrock_llm_config.provider_params.provider_type.name)

def test_base_llm_abstract_methods():
    """Test that BaseLLM enforces implementation of abstract methods."""
    class IncompleteLLM(BaseLLM, ABC):
        aliases = [ProviderType.BEDROCK]
        
        @classmethod
        def _get_constructor_parameters(cls, *, llm_config: LLMConfig) -> dict:
            return {"config": llm_config}
    
    # Test that instantiation fails without all required methods
    with pytest.raises(TypeError):
        IncompleteLLM(config=bedrock_llm_config)

def test_base_llm_invoke_abstract():
    """Test that invoke() is properly marked as abstract."""
    assert BaseLLM.invoke.__isabstractmethod__

def test_base_llm_ainvoke_abstract():
    """Test that ainvoke() is properly marked as abstract."""
    assert BaseLLM.ainvoke.__isabstractmethod__

def test_base_llm_stream_abstract():
    """Test that stream() is properly marked as abstract."""
    assert BaseLLM.stream.__isabstractmethod__

def test_base_llm_astream_abstract():
    """Test that astream() is properly marked as abstract."""
    assert BaseLLM.astream.__isabstractmethod__

def test_base_llm_get_constructor_parameters_abstract():
    """Test that _get_constructor_parameters() is properly marked as abstract."""
    assert BaseLLM._get_constructor_parameters.__isabstractmethod__

def test_base_llm_invoke_implementation(test_messages):
    """Test the concrete implementation of invoke()."""
    llm = TestLLM(config=bedrock_llm_config)
    response = llm.invoke(test_messages)
    assert isinstance(response, AIMessage)
    assert response.content == "Test response"

@pytest.mark.asyncio
async def test_base_llm_ainvoke_implementation(test_messages):
    """Test the concrete implementation of ainvoke()."""
    llm = TestLLM(config=bedrock_llm_config)
    response = await llm.ainvoke(test_messages)
    assert isinstance(response, AIMessage)
    assert response.content == "Test async response"

def test_base_llm_stream_implementation(test_messages):
    """Test the concrete implementation of stream()."""
    llm = TestLLM(config=bedrock_llm_config)
    response_stream = llm.stream(test_messages)
    chunks = list(response_stream)
    assert len(chunks) == 1
    assert chunks[0].content == "Test stream"

@pytest.mark.asyncio
async def test_base_llm_astream_implementation(test_messages):
    """Test the concrete implementation of astream()."""
    llm = TestLLM(config=bedrock_llm_config)
    response_stream = await llm.astream(test_messages)
    chunks = [chunk async for chunk in response_stream]
    assert len(chunks) == 1
    assert chunks[0].content == "Test async stream" 