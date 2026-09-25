"""vLLM 0.30.0 hermes parser: a JSON string argument that contains the text "</tool_call>"
makes extract_tool_calls return no tool call; the whole call is returned as content.

The output is what Qwen3's official chat template renders for an echo(text=...) call.
"""
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

tools = [{"type": "function", "function": {"name": "echo", "parameters": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
out = '<tool_call>\n{"name": "echo", "arguments": {"text": "Use <tool_call> ... </tool_call> tags, or \\"<think>\\" blocks."}}\n</tool_call>'
tok = get_tokenizer("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
req = ChatCompletionRequest(model="Qwen/Qwen3-0.6B", messages=[{"role": "user", "content": "hi"}], tools=tools)
info = Hermes2ProToolParser(tok, req.tools).extract_tool_calls(out, req)
print("tools_called:", info.tools_called)
print("tool_calls  :", [(t.function.name, t.function.arguments) for t in info.tool_calls])
print("content     :", repr(info.content))
