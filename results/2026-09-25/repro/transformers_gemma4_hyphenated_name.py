"""transformers 5.17.0: google/gemma-4 response_template (tokenizer.parse_response) raises on a
tool name containing '-' (vLLM's gemma4 tests use such names; the format allows them).
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
tools = [{"type": "function", "function": {"name": "get-weather", "parameters": {
    "type": "object", "properties": {"location": {"type": "string"}}}}}]
for name in ("get_weather", "get-weather"):
    out = f'<|tool_call>call:{name}{{location:<|"|>London<|"|>}}<tool_call|>'
    ids = tok.encode(out, add_special_tokens=False)
    try:
        print(name, "->", tok.parse_response(ids, prefix="", tools=tools))
    except Exception as e:
        print(name, "->", type(e).__name__, str(e).splitlines()[0])
