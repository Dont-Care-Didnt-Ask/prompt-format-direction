"""
Test script to demonstrate dual return type functionality.
Run this to verify that chat template support is working correctly.
"""

from format_utils import (
    build_gsm8k_few_shot_prompt,
    build_gsm8k_lora_prompt,
    FormatSpecification,
    format_as_chat_messages
)

def test_format_as_chat_messages():
    """Test the helper function for converting strings to chat messages."""
    print("=" * 80)
    print("TEST 1: format_as_chat_messages()")
    print("=" * 80)
    
    prompt_str = "What is 2+2?"
    
    # With system message
    messages = format_as_chat_messages(
        prompt_str, 
        system_message="You are a helpful assistant.",
        use_system=True
    )
    print("\nWith system message:")
    for msg in messages:
        print(f"  {msg['role']:10s}: {msg['content']}")
    
    # Without system message
    messages = format_as_chat_messages(
        prompt_str,
        system_message="",
        use_system=False
    )
    print("\nWithout system message:")
    for msg in messages:
        print(f"  {msg['role']:10s}: {msg['content']}")


def test_lora_prompt():
    """Test the LoRA prompt builder with dual return type."""
    print("\n" + "=" * 80)
    print("TEST 2: build_gsm8k_lora_prompt()")
    print("=" * 80)
    
    # Create a simple format spec
    format_spec = FormatSpecification(
        descriptor_transformation=lambda x: x,
        descriptor_transformation_str="lambda x: x",
        separator=": ",
        space="\n",
        first_descriptor="Question",
        second_descriptor="Reasoning",
        third_descriptor="Answer"
    )
    
    test_example = {
        "question": "If John has 5 apples and gives 2 to Mary, how many does he have left?"
    }
    
    # Test string return type
    print("\nString return type:")
    prompt_str = build_gsm8k_lora_prompt(
        test_example=test_example,
        format_spec=format_spec,
        reasoning_answer_separator="####",
        return_type="string"
    )
    print(f"  {repr(prompt_str)}")
    
    # Test messages return type with system message
    print("\nMessages return type (with system message):")
    prompt_msgs = build_gsm8k_lora_prompt(
        test_example=test_example,
        format_spec=format_spec,
        reasoning_answer_separator="####",
        return_type="messages",
        system_message="You are a helpful math assistant."
    )
    for msg in prompt_msgs:
        print(f"  {msg['role']:10s}: {msg['content'][:60]}...")
    
    # Test messages return type without system message
    print("\nMessages return type (without system message):")
    prompt_msgs = build_gsm8k_lora_prompt(
        test_example=test_example,
        format_spec=format_spec,
        reasoning_answer_separator="####",
        return_type="messages",
        system_message=""
    )
    for msg in prompt_msgs:
        print(f"  {msg['role']:10s}: {msg['content'][:60]}...")


def test_few_shot_prompt():
    """Test the few-shot prompt builder with dual return type."""
    print("\n" + "=" * 80)
    print("TEST 3: build_gsm8k_few_shot_prompt()")
    print("=" * 80)
    
    # Create a simple format spec
    format_spec = FormatSpecification(
        descriptor_transformation=lambda x: x.upper(),
        descriptor_transformation_str="lambda x: x.upper()",
        separator=": ",
        space=" ",
        first_descriptor="Question",
        second_descriptor="Reasoning",
        third_descriptor="Answer"
    )
    
    few_shot_examples = [
        {
            "question": "What is 1+1?",
            "answer": "1+1 equals 2.#### 2"
        },
        {
            "question": "What is 3*4?",
            "answer": "3 times 4 is 12.#### 12"
        }
    ]
    
    test_example = {
        "question": "What is 5+3?"
    }
    
    # Test string return type
    print("\nString return type:")
    prompt_str = build_gsm8k_few_shot_prompt(
        test_example=test_example,
        few_shot_examples=few_shot_examples,
        format_spec=format_spec,
        reasoning_answer_separator="####",
        return_type="string"
    )
    print(f"  {prompt_str[:100]}...")
    print(f"  ... (total length: {len(prompt_str)} chars)")
    
    # Test messages return type
    print("\nMessages return type:")
    prompt_msgs = build_gsm8k_few_shot_prompt(
        test_example=test_example,
        few_shot_examples=few_shot_examples,
        format_spec=format_spec,
        reasoning_answer_separator="####",
        return_type="messages",
        system_message="You are a helpful math assistant."
    )
    for msg in prompt_msgs:
        content_preview = msg['content'][:80].replace('\n', '\\n')
        print(f"  {msg['role']:10s}: {content_preview}...")


def test_assertions():
    """Test that assertions work correctly."""
    print("\n" + "=" * 80)
    print("TEST 4: Assertion Tests")
    print("=" * 80)
    
    format_spec = FormatSpecification(
        descriptor_transformation=lambda x: x,
        descriptor_transformation_str="lambda x: x",
        separator=": ",
        space=" ",
        first_descriptor="Q",
        second_descriptor="R",
        third_descriptor="A"
    )
    
    test_example = {"question": "Test?"}
    
    # Test invalid return_type
    print("\nTesting invalid return_type (should raise assertion error):")
    try:
        build_gsm8k_lora_prompt(
            test_example=test_example,
            format_spec=format_spec,
            reasoning_answer_separator="####",
            return_type="invalid"
        )
        print("  ❌ FAILED: Should have raised AssertionError")
    except AssertionError as e:
        print(f"  ✓ PASSED: {e}")


if __name__ == "__main__":
    print("\n" + "🧪 TESTING DUAL RETURN TYPE FUNCTIONALITY ".center(80, "="))
    
    test_format_as_chat_messages()
    test_lora_prompt()
    test_few_shot_prompt()
    test_assertions()
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 80)

