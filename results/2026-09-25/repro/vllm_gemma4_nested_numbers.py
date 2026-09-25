"""vLLM 0.30.0 Gemma 4: numbers/booleans nested inside objects/arrays come back as strings.

Top-level values are typed from the tool schema; values under a nested object or array
the schema does not type are returned as JSON strings ("25", "true") instead of 25/true.
"""
from vllm.parser.gemma4 import _parse_gemma4_args

args = 'location:<|"|>Tokyo<|"|>,details:{temp:25,unit:<|"|>celsius<|"|>},flags:[true,1]'
print(_parse_gemma4_args(args))
