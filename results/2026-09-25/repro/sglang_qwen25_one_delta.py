"""SGLang 0.5.20 Qwen25Detector: a complete tool call arriving in one streaming delta loses its arguments."""
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector

tools = [Tool(type="function", function=Function(name="get_weather", parameters={
    "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}}))]
text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}\n</tool_call>'

print("non-stream:", Qwen25Detector().detect_and_parse(text, tools).calls)
d = Qwen25Detector()
r = d.parse_streaming_increment(text, tools)
print("one delta :", r.calls, "| normal_text:", repr(r.normal_text))
d = Qwen25Detector()
calls = []
for i in range(0, len(text), 4):
    calls += d.parse_streaming_increment(text[i:i + 4], tools).calls
print("4-char deltas:", [(c.name, c.parameters) for c in calls])
