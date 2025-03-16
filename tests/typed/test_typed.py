import pytest
from pydantic import ValidationError
from fmcore.types.typed import Typed, MutableTyped


class SampleTyped(Typed):
    name: str
    value: int


class SampleMutableTyped(MutableTyped):
    name: str
    value: int


def test_typed_immutability():
    """Test that Typed instances are immutable"""
    obj = SampleTyped(name="test", value=42)

    # Verify we can't modify attributes
    with pytest.raises(ValidationError):
        obj.name = "new_name"

    # Verify original values remain unchanged
    assert obj.name == "test"
    assert obj.value == 42


def test_typed_extra_fields():
    """Test that Typed rejects extra fields"""
    with pytest.raises(ValidationError):
        SampleTyped(name="test", value=42, extra_field="not_allowed")


def test_typed_validation():
    """Test type validation in Typed"""
    # Test valid types
    obj = SampleTyped(name="test", value=42)
    assert obj.name == "test"
    assert obj.value == 42

    # Test invalid types
    with pytest.raises(ValidationError):
        SampleTyped(name=123, value="not_an_int")


def test_mutable_typed_mutability():
    """Test that MutableTyped instances are mutable"""
    obj = SampleMutableTyped(name="test", value=42)

    # Verify we can modify attributes
    obj.name = "new_name"
    obj.value = 100

    assert obj.name == "new_name"
    assert obj.value == 100


def test_mutable_typed_validation():
    """Test that MutableTyped validates on assignment"""
    obj = SampleMutableTyped(name="test", value=42)

    # Test valid assignment
    obj.name = "new_name"
    assert obj.name == "new_name"

    # Test invalid assignment
    with pytest.raises(ValidationError):
        obj.value = "not_an_int"

    # Verify original value remains unchanged after failed validation
    assert obj.value == 42


def test_mutable_typed_extra_fields():
    """Test that MutableTyped allows extra fields"""
    # Should not raise an exception
    obj = SampleMutableTyped(name="test", value=42, extra_field="allowed")
    assert obj.name == "test"
    assert obj.value == 42
