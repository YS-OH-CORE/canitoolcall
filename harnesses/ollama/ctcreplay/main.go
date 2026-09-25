// ctcreplay: a long-lived JSON-lines driver for Ollama's REAL built-in output
// parsers (model/parsers, including the harmony handler), offline, no model.
//
// scripts/engines/ollama.sh copies this file into the pinned Ollama clone as
// cmd/ctcreplay and builds it there, so it links the exact parser code of
// that commit.
//
// It mirrors server/routes.go (ChatHandler) at the pin:
//
//	p := parsers.ParserForName(m.Config.Parser)
//	p.Init(req.Tools, lastMessage, req.Think)       // once per request
//	content, thinking, calls, err := p.Add(r.Content, r.Done)  // per runner event
//
// where req.Think is resolved like routes.go does for a model whose only
// thinking capability comes from its parser (nil -> true when the parser
// supports thinking), and tool calls are serialized with openai.ToToolCalls,
// exactly what Ollama's OpenAI-compatible endpoint returns.
//
// Protocol: one JSON object per line on stdin, one reply per line on stdout.
//
//	{"op":"hello"}
//	  -> {"ok":true,"ollama_commit":..,"go_version":..}
//	{"op":"describe","parser":"qwen3-coder"}
//	  -> {"ok":true,"known":bool,"preserved_tokens":[..],"has_tool_support":bool,"has_thinking_support":bool}
//	{"op":"replay","parser":..,"tools":[api.Tool..],"think":bool|string|null,"streams":[[delta..],..],"finish":bool}
//	  -> {"ok":true,"think":<effective>,"streams":[{"events":[{"content","thinking","tool_calls"}..],"error":".."}..]}
//
// Each stream is a fresh parser; the LAST delta of a stream is passed with
// done=true unless "finish" is false (callers append the runner's final,
// empty, Done event themselves).
// A parser error or panic ends that stream and is reported in "error".
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/openai"
)

// Set with -ldflags "-X main.ollamaCommit=<sha>" by scripts/engines/ollama.sh.
var ollamaCommit = "unknown"

type request struct {
	Op      string          `json:"op"`
	Parser  string          `json:"parser"`
	Tools   []api.Tool      `json:"tools"`
	Think   *api.ThinkValue `json:"think"`
	Streams [][]string      `json:"streams"`
	// Finish (default true) passes the last delta of each stream with
	// done=true. False replays a request the runner aborted before Done.
	Finish *bool `json:"finish"`
}

type event struct {
	Content   string            `json:"content"`
	Thinking  string            `json:"thinking"`
	ToolCalls []openai.ToolCall `json:"tool_calls"`
}

type streamResult struct {
	Events []event `json:"events"`
	Error  string  `json:"error,omitempty"`
}

func fail(format string, args ...any) map[string]any {
	return map[string]any{"ok": false, "error": fmt.Sprintf(format, args...)}
}

// resolveThink mirrors server/routes.go ChatHandler: a model whose parser
// supports thinking has the thinking capability, and an unset request think
// becomes true for it. Otherwise the requested value passes through.
func resolveThink(p parsers.Parser, requested *api.ThinkValue) *api.ThinkValue {
	if requested == nil && p.HasThinkingSupport() {
		return &api.ThinkValue{Value: true}
	}
	return requested
}

// add calls p.Add, turning a panic inside the engine's parser into an error.
func add(p parsers.Parser, s string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	return p.Add(s, done)
}

func replay(req request) map[string]any {
	probe := parsers.ParserForName(req.Parser)
	if probe == nil {
		return fail("unknown parser %q", req.Parser)
	}
	think := resolveThink(probe, req.Think)
	// routes.go passes the last chat message; a replay's is the user's turn.
	last := &api.Message{Role: "user"}
	finish := req.Finish == nil || *req.Finish
	streams := make([]streamResult, 0, len(req.Streams))
	for _, deltas := range req.Streams {
		p := parsers.ParserForName(req.Parser)
		p.Init(req.Tools, last, think)
		s := streamResult{Events: []event{}}
		for i, d := range deltas {
			content, thinking, calls, err := add(p, d, finish && i == len(deltas)-1)
			if err != nil {
				s.Error = err.Error()
				break
			}
			tcs := openai.ToToolCalls(calls)
			if tcs == nil {
				tcs = []openai.ToolCall{}
			}
			s.Events = append(s.Events, event{Content: content, Thinking: thinking, ToolCalls: tcs})
		}
		streams = append(streams, s)
	}
	return map[string]any{"ok": true, "think": think, "streams": streams}
}

func handle(line []byte) map[string]any {
	var req request
	if err := json.Unmarshal(line, &req); err != nil {
		return fail("bad request: %v", err)
	}
	switch req.Op {
	case "hello":
		return map[string]any{"ok": true, "ollama_commit": ollamaCommit, "go_version": runtime.Version()}
	case "describe":
		p := parsers.ParserForName(req.Parser)
		if p == nil {
			return map[string]any{"ok": true, "known": false}
		}
		tokens := p.PreservedTokens()
		if tokens == nil {
			tokens = []string{}
		}
		return map[string]any{
			"ok":                   true,
			"known":                true,
			"preserved_tokens":     tokens,
			"has_tool_support":     p.HasToolSupport(),
			"has_thinking_support": p.HasThinkingSupport(),
		}
	case "replay":
		return replay(req)
	default:
		return fail("unknown op %q", req.Op)
	}
}

func main() {
	// Parsers log recoverable failures with slog.Warn; keep stderr quiet
	// unless CTCREPLAY_LOG is set. Logging never changes a parse.
	if os.Getenv("CTCREPLAY_LOG") == "" {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	}
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1<<20), 256<<20)
	out := bufio.NewWriter(os.Stdout)
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	for in.Scan() {
		line := in.Bytes()
		if len(line) == 0 {
			continue
		}
		if err := enc.Encode(handle(line)); err != nil {
			fmt.Fprintln(os.Stderr, "encode:", err)
			os.Exit(1)
		}
		if err := out.Flush(); err != nil {
			os.Exit(1)
		}
	}
	if err := in.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "read:", err)
		os.Exit(1)
	}
}
