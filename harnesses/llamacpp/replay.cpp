// canitoolcall-llamacpp: replay raw model outputs through llama.cpp's REAL chat
// parser, offline, exactly the way llama-server (tools/server) drives it.
//
// One long-lived process per run. JSON lines on stdin -> one JSON line per
// request on stdout (stderr is free-form). Protocol version 1:
//
//   {"op":"hello"}
//     -> {"ok":true,"protocol":1,"build_info":"b<N>-<sha>","commit":"<sha>","build_number":N}
//
//   {"op":"tokenize","vocab":"<vocab-only gguf>","text":"..."}
//     -> {"ok":true,"ids":[...]}      common_tokenize(vocab, text, add_special=false, parse_special=true)
//
//   {"op":"special_ids","vocab":"<vocab-only gguf>"}
//     -> {"ok":true,"ids":[...]}      tokens with CONTROL | USER_DEFINED | UNKNOWN attributes
//
//   {"op":"replay",
//    "vocab": "<vocab-only gguf>",           // convert_hf_to_gguf.py --vocab-only
//    "template": "<jinja>" | null,           // null = the GGUF's own tokenizer.chat_template
//    "messages": [...], "tools": [...],      // OpenAI format (the request body)
//    "reasoning_format": "deepseek",         // llama-server default (common_params)
//    "chat_template_kwargs": {...},          // values are JSON values (like the request body)
//    "parallel_tool_calls": bool,            // optional; default = template caps (like the server)
//    "ids": [...],                           // generated token ids (stop token excluded)
//    "end_tokens": ["<|im_end|>", ...],      // the stop token that ended generation (first single-token one is used)
//    "nonstream": true,                      // run the non-streaming path
//    "streams": [[n_tokens_per_step, ...] | {"text": ["delta", ...]}, ...],  // one per streaming replay
//    "return_template": false}               // include the effective template source in "config"
//     -> {"ok":true,"config":{...},"text":"<detokenized>","nonstream":{...},"streams":[{...}]}
//        or {"ok":true,"config":{...},"engine_error":"..."} when llama.cpp rejects the
//        request before generation (template init/apply failure, invalid grammar trigger).
//
// Server fidelity (llama.cpp tools/server at the pinned commit):
//  * Chat params: common_chat_templates_init(model, template) and
//    common_chat_templates_apply(inputs) with the inputs built like
//    server-common.cpp oaicompat_chat_params_parse (use_jinja, parallel_tool_calls
//    from template caps, enable_thinking default = template support, kwargs merge).
//  * Parser params like server-schema.cpp: format, reasoning_format,
//    reasoning_in_content = stream && deepseek-legacy, generation_prompt, parser.
//  * Detokenization like server-context.cpp: common_token_to_piece(vocab, id,
//    special = id in preserved_tokens), preserved ids resolved like
//    server-schema.cpp (single-token only).
//  * Token processing like server_context::process_token: hold back incomplete
//    UTF-8 (validate_utf8) and partial stop strings (additional_stops), cut at a
//    full stop string, stop at an end-of-generation token.
//  * Streaming like task_result_state::update_chat_msg: append the sent text,
//    re-parse the accumulated text with is_partial=true, keep the previous message
//    when the new one is empty, compute_diffs; the final result re-parses with
//    is_partial=false and empty added text.
//  * Non-streaming: the final result parses slot.generated_text with is_partial=false;
//    an empty parse falls back to the raw text as content (to_json_oaicompat_chat).
//
// validate_utf8, find_stopping_strings and update_chat_msg are adapted from
// llama.cpp tools/server (MIT License, Copyright (c) 2023-2026 The ggml authors).
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"

#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

static const int PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------------
// Copied from tools/server/server-common.cpp (MIT).
static size_t validate_utf8(const std::string & text) {
    size_t len = text.size();
    if (len == 0) {
        return 0;
    }
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        if ((c & 0xE0) == 0xC0) {
            if (i < 2) {
                return len - i;
            }
        } else if ((c & 0xF0) == 0xE0) {
            if (i < 3) {
                return len - i;
            }
        } else if ((c & 0xF8) == 0xF0) {
            if (i < 4) {
                return len - i;
            }
        }
    }
    return len;
}

// ---------------------------------------------------------------------------
// Emulation of the server slot's text handling (server-context.cpp process_token).
struct slot_emu {
    std::vector<std::string> antiprompt;
    std::string              generated_text;
    size_t                   n_sent_text    = 0;
    bool                     has_next_token = true;
    std::string              stopping_word;
    long                     stopped_at = -1;  // index of the token that stopped generation

    // server_slot::find_stopping_strings
    size_t find_stopping_strings(const std::string & text, size_t last_token_size, bool is_full_stop) {
        size_t stop_pos = std::string::npos;
        for (const std::string & word : antiprompt) {
            size_t pos;
            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos                   = text.find(word, from_pos);
            } else {
                pos = string_find_partial_stop(text, word);
            }
            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

    // Returns true when the server would send a partial response for this token;
    // text_to_send receives the text of that response.
    bool process_token(const std::string & token_str, bool is_eog, long index, std::string & text_to_send) {
        text_to_send.clear();
        generated_text += token_str;
        has_next_token  = true;

        const bool incomplete = validate_utf8(generated_text) < generated_text.size();
        bool       sent       = false;
        if (!incomplete) {
            size_t            pos      = std::min(n_sent_text, generated_text.size());
            const std::string str_test = generated_text.substr(pos);
            bool              send_text = true;

            size_t stop_pos = find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                generated_text.erase(generated_text.begin() + pos + stop_pos, generated_text.end());
                pos = std::min(n_sent_text, generated_text.size());
            } else if (has_next_token && !is_eog) {
                stop_pos  = find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }
            if (send_text) {
                text_to_send = generated_text.substr(pos, std::string::npos);
                n_sent_text += text_to_send.size();
            }
            sent = true;
        }
        if (incomplete) {
            has_next_token = true;
        }
        if (is_eog) {
            has_next_token = false;
        }
        if (!has_next_token && stopped_at < 0) {
            stopped_at = index;
        }
        return sent;
    }
};

// ---------------------------------------------------------------------------
// Emulation of task_result_state::update_chat_msg (server-task.cpp), filter_tool_calls=false.
struct result_state {
    const common_chat_parser_params & pp;
    std::string                       generated_text;
    common_chat_msg                   chat_msg;
    std::vector<std::string>          generated_tool_call_ids;
    size_t                            n_ids = 0;

    explicit result_state(const common_chat_parser_params & pp) : pp(pp) {}

    // deterministic stand-in for the server's random gen_tool_call_id()
    std::string gen_id() { return "ctc_" + std::to_string(n_ids++); }

    common_chat_msg update(const std::string & text_added, bool is_partial, std::vector<common_chat_msg_diff> & diffs) {
        generated_text += text_added;
        auto msg_prv_copy = chat_msg;
        auto new_msg      = common_chat_parse(generated_text, is_partial, pp);
        if (!new_msg.empty()) {
            new_msg.set_tool_call_ids(generated_tool_call_ids, [this]() { return gen_id(); });
            chat_msg = new_msg;
            diffs    = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);
        }
        return chat_msg;
    }
};

// ---------------------------------------------------------------------------
static json msg_to_json(const common_chat_msg & m) {
    json tcs = json::array();
    for (const auto & tc : m.tool_calls) {
        tcs.push_back({
            { "name",      tc.name      },
            { "arguments", tc.arguments },
            { "id",        tc.id        },
        });
    }
    return {
        { "content",           m.content           },
        { "reasoning_content", m.reasoning_content },
        { "tool_calls",        tcs                 },
    };
}

static json diff_to_json(const common_chat_msg_diff & d) {
    json j = json::object();
    if (!d.reasoning_content_delta.empty()) {
        j["reasoning_content"] = d.reasoning_content_delta;
    }
    if (!d.content_delta.empty()) {
        j["content"] = d.content_delta;
    }
    if (d.tool_call_index != std::string::npos) {
        j["index"]     = (long) d.tool_call_index;
        j["name"]      = d.tool_call_delta.name;
        j["id"]        = d.tool_call_delta.id;
        j["arguments"] = d.tool_call_delta.arguments;
    }
    return j;
}

// ---------------------------------------------------------------------------
struct harness {
    std::map<std::string, llama_model *>              models;
    std::map<std::string, common_chat_templates_ptr> templates;

    ~harness() {
        templates.clear();
        for (auto & kv : models) {
            llama_model_free(kv.second);
        }
    }

    llama_model * model(const std::string & path) {
        auto it = models.find(path);
        if (it != models.end()) {
            return it->second;
        }
        auto mparams       = llama_model_default_params();
        mparams.vocab_only = true;
        llama_model * m    = llama_model_load_from_file(path.c_str(), mparams);
        if (!m) {
            throw std::runtime_error("failed to load vocab-only GGUF: " + path);
        }
        models[path] = m;
        return m;
    }

    // Like server_context::load_model: common_chat_templates_init(model, params.chat_template).
    const common_chat_templates * tmpls(const std::string & vocab_path, llama_model * m, const json & tmpl) {
        const std::string override_src = tmpl.is_string() ? tmpl.get<std::string>() : std::string();
        const std::string key          = vocab_path + '\0' + (tmpl.is_string() ? "T" + override_src : std::string("G"));
        auto              it           = templates.find(key);
        if (it != templates.end()) {
            return it->second.get();
        }
        auto t              = common_chat_templates_init(m, override_src);
        const auto * result = t.get();
        templates[key]      = std::move(t);
        return result;
    }

    json hello() {
        return {
            { "ok",           true               },
            { "protocol",     PROTOCOL_VERSION   },
            { "build_info",   llama_build_info() },
            { "commit",       llama_commit()     },
            { "build_number", llama_build_number() },
        };
    }

    json tokenize(const json & req) {
        const llama_vocab * vocab = llama_model_get_vocab(model(req.at("vocab").get<std::string>()));
        auto                ids   = common_tokenize(vocab, req.at("text").get<std::string>(), false, true);
        return {
            { "ok",  true                                 },
            { "ids", std::vector<int>(ids.begin(), ids.end()) },
        };
    }

    // llama_vocab's special-token cache: CONTROL | USER_DEFINED | UNKNOWN attributes
    // (the tokens common_tokenize(..., parse_special=true) matches atomically).
    json special_ids(const json & req) {
        const llama_vocab * vocab = llama_model_get_vocab(model(req.at("vocab").get<std::string>()));
        std::vector<int>    ids;
        for (int id = 0, n = llama_vocab_n_tokens(vocab); id < n; ++id) {
            if (llama_vocab_get_attr(vocab, id) &
                (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                ids.push_back(id);
            }
        }
        return {
            { "ok",  true },
            { "ids", ids  },
        };
    }

    json replay(const json & req) {
        const std::string   vocab_path = req.at("vocab").get<std::string>();
        llama_model *       m          = model(vocab_path);
        const llama_vocab * vocab      = llama_model_get_vocab(m);

        json out   = json::object();
        out["ok"]  = true;
        json cfg   = json::object();

        // --- chat params, as oaicompat_chat_params_parse + load_model do ---------
        common_chat_params      chat_params;
        common_reasoning_format reasoning_format =
            common_reasoning_format_from_name(req.value("reasoning_format", std::string("deepseek")));
        try {
            const common_chat_templates * t = tmpls(vocab_path, m, req.value("template", json()));
            cfg["template_override"]         = req.value("template", json()).is_string();
            cfg["gguf_has_template"]         = llama_model_chat_template(m, /* name */ nullptr) != nullptr;
            cfg["template_was_explicit"]     = common_chat_templates_was_explicit(t);

            auto caps = common_chat_templates_get_caps(t);
            json jcaps = json::object();
            for (const auto & kv : caps) {
                jcaps[kv.first] = kv.second;
            }
            cfg["caps"] = jcaps;

            const bool template_supports_thinking = common_chat_templates_support_enable_thinking(t);
            cfg["template_supports_thinking"]      = template_supports_thinking;

            common_chat_templates_inputs inputs;
            inputs.messages            = common_chat_msgs_parse_oaicompat(req.value("messages", json::array()));
            inputs.tools               = common_chat_tools_parse_oaicompat(req.value("tools", json::array()));
            inputs.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
            inputs.use_jinja           = true;
            inputs.parallel_tool_calls = caps["supports_parallel_tool_calls"];
            if (req.contains("parallel_tool_calls") && req.at("parallel_tool_calls").is_boolean()) {
                inputs.parallel_tool_calls = req.at("parallel_tool_calls").get<bool>();
            }
            inputs.add_generation_prompt = true;
            inputs.reasoning_format      = reasoning_format;
            // server default: enable_reasoning=-1 (auto) -> thinking iff the template supports it
            inputs.enable_thinking = template_supports_thinking;

            // server default kwargs (common/arg.cpp) merged with the request's kwargs
            inputs.chat_template_kwargs["preserve_reasoning"] = "true";
            // named local: iterating .items() of a temporary would dangle
            const json kwargs = req.value("chat_template_kwargs", json::object());
            for (const auto & [k, v] : kwargs.items()) {
                inputs.chat_template_kwargs[k] = v.dump();
            }
            auto et = inputs.chat_template_kwargs.find("enable_thinking");
            if (et != inputs.chat_template_kwargs.end()) {
                if (et->second == "true") {
                    inputs.enable_thinking = true;
                } else if (et->second == "false") {
                    inputs.enable_thinking = false;
                } else if (!et->second.empty() && et->second[0] == '"') {
                    throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
                }
            }
            cfg["enable_thinking"]     = inputs.enable_thinking;
            cfg["parallel_tool_calls"] = inputs.parallel_tool_calls;

            // the variant common_chat_templates_apply_jinja picks
            const std::string tool_use_src = common_chat_templates_source(t, "tool_use");
            const bool        use_tool_use = common_chat_tools_to_json_oaicompat(inputs.tools).is_array() && !tool_use_src.empty();
            cfg["template_variant"]        = use_tool_use ? "tool_use" : "default";
            if (req.value("return_template", false)) {
                // as parsed by llama.cpp (after its built-in template patches), and as stored in the GGUF
                cfg["template_source"] = use_tool_use ? tool_use_src : common_chat_templates_source(t);
                const char * stored    = llama_model_chat_template(m, /* name */ nullptr);
                cfg["gguf_template"]   = stored ? json(std::string(stored)) : json();
            }

            chat_params = common_chat_templates_apply(t, inputs);
        } catch (const std::exception & e) {
            out["config"]       = cfg;
            out["engine_error"] = std::string("chat template: ") + e.what();
            return out;
        }

        cfg["format"]             = common_chat_format_name(chat_params.format);
        cfg["reasoning_format"]   = common_reasoning_format_name(reasoning_format);
        cfg["generation_prompt"]  = chat_params.generation_prompt;
        cfg["preserved_tokens"]   = chat_params.preserved_tokens;
        cfg["additional_stops"]   = chat_params.additional_stops;
        cfg["supports_thinking"]  = chat_params.supports_thinking;
        cfg["thinking_start_tag"] = chat_params.thinking_start_tag;
        cfg["thinking_end_tags"]  = chat_params.thinking_end_tags;
        cfg["has_parser"]         = !chat_params.parser.empty();

        // --- request params, as server-schema.cpp does ---------------------------
        std::set<llama_token> preserved;
        for (const auto & tok : chat_params.preserved_tokens) {
            auto ids = common_tokenize(vocab, tok, false, true);
            if (ids.size() == 1) {
                preserved.insert(ids[0]);
            }
        }
        cfg["preserved_token_ids"] = std::vector<int>(preserved.begin(), preserved.end());
        try {
            for (const auto & trig : chat_params.grammar_triggers) {
                if (trig.type != COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    continue;
                }
                auto ids = common_tokenize(vocab, trig.value, false, true);
                if (ids.size() == 1 && !preserved.count(ids[0])) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + trig.value);
                }
            }
        } catch (const std::exception & e) {
            out["config"]       = cfg;
            out["engine_error"] = std::string("request: ") + e.what();
            return out;
        }

        common_chat_parser_params pp(chat_params);
        pp.reasoning_format     = reasoning_format;
        pp.reasoning_in_content = false;
        pp.parse_tool_calls     = true;
        if (!chat_params.parser.empty()) {
            pp.parser.load(chat_params.parser);
        }
        common_chat_parser_params pp_stream = pp;
        pp_stream.reasoning_in_content      = reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;

        // --- generated tokens ------------------------------------------------------
        std::vector<llama_token> ids;
        const json req_ids = req.value("ids", json::array());
        for (const auto & jt : req_ids) {
            ids.push_back(jt.get<int>());
        }
        std::vector<llama_token> end_ids;
        json                     end_used = nullptr;
        const json req_end_tokens = req.value("end_tokens", json::array());
        for (const auto & jt : req_end_tokens) {
            auto t = common_tokenize(vocab, jt.get<std::string>(), false, true);
            if (t.size() == 1) {
                end_ids.push_back(t[0]);
                end_used = jt.get<std::string>();
                break;
            }
        }
        cfg["end_token"] = end_used;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (auto t : ids) {
            if (t < 0 || t >= n_vocab) {
                throw std::runtime_error("token id " + std::to_string(t) + " out of range for " + vocab_path);
            }
        }
        std::vector<llama_token> all_ids = ids;
        all_ids.insert(all_ids.end(), end_ids.begin(), end_ids.end());
        std::vector<std::string> pieces;
        std::vector<bool>        eog;
        std::string              text;
        for (size_t i = 0; i < all_ids.size(); ++i) {
            auto p = common_token_to_piece(vocab, all_ids[i], preserved.count(all_ids[i]) > 0);
            pieces.push_back(p);
            eog.push_back(llama_vocab_is_eog(vocab, all_ids[i]));
            if (i < ids.size()) {
                text += p;
            }
        }
        out["text"] = text;
        {
            json eog_ids = json::array();
            for (size_t i = 0; i < ids.size(); ++i) {
                if (eog[i]) {
                    eog_ids.push_back((long) i);
                }
            }
            cfg["eog_positions"] = eog_ids;
        }
        out["config"] = cfg;

        // --- non-streaming ---------------------------------------------------------
        if (req.value("nonstream", true)) {
            slot_emu slot;
            slot.antiprompt = chat_params.additional_stops;
            std::string tts;
            for (size_t i = 0; i < all_ids.size() && slot.has_next_token; ++i) {
                slot.process_token(pieces[i], eog[i], (long) i, tts);
            }
            json ns             = json::object();
            ns["generated_text"] = slot.generated_text;
            ns["stopped_at"]     = slot.stopped_at;
            ns["stopping_word"]  = slot.stopping_word;
            try {
                result_state                      state(pp);
                std::vector<common_chat_msg_diff> diffs;
                common_chat_msg                   msg = state.update(slot.generated_text, false, diffs);
                // server_task_result_cmpl_final::to_json_oaicompat_chat: an empty parse
                // falls back to the raw generated text as content
                ns["empty_parse_fallback"] = msg.empty();
                if (msg.empty()) {
                    msg.role    = "assistant";
                    msg.content = slot.generated_text;
                }
                ns["msg"] = msg_to_json(msg);
            } catch (const std::exception & e) {
                ns["exception"] = e.what();
            }
            out["nonstream"] = ns;
        }

        // --- streaming ---------------------------------------------------------------
        // A stream spec is either token-group sizes ([n, ...], one server step per
        // group) or {"text": ["delta", ...]} (synthetic text deltas, one step each).
        // The stop token that ended generation is always its own final step.
        struct step_token {
            std::string piece;
            bool        eog;
            long        index;
        };
        json streams = json::array();
        const json req_streams = req.value("streams", json::array());
        for (const auto & spec : req_streams) {
            std::vector<std::vector<step_token>> steps;
            if (spec.is_object()) {
                for (const auto & d : spec.at("text")) {
                    steps.push_back({ { d.get<std::string>(), false, -1 } });
                }
            } else {
                size_t k = 0;
                for (const auto & g : spec) {
                    std::vector<step_token> step;
                    for (size_t j = 0, n = g.get<size_t>(); j < n; ++j, ++k) {
                        if (k >= ids.size()) {
                            throw std::runtime_error("stream groups cover more than the " + std::to_string(ids.size()) +
                                                     " tokens");
                        }
                        step.push_back({ pieces[k], eog[k], (long) k });
                    }
                    steps.push_back(step);
                }
                if (k != ids.size()) {
                    throw std::runtime_error("stream groups cover " + std::to_string(k) + " of " +
                                             std::to_string(ids.size()) + " tokens");
                }
            }
            for (size_t e = ids.size(); e < all_ids.size(); ++e) {
                steps.push_back({ { pieces[e], eog[e], (long) e } });
            }

            slot_emu slot;
            slot.antiprompt = chat_params.additional_stops;
            result_state state(pp_stream);
            json         s      = json::object();
            json         deltas = json::array();
            try {
                for (const auto & step : steps) {
                    if (!slot.has_next_token) {
                        break;
                    }
                    std::string send;
                    bool        any_sent = false;
                    for (const auto & tok : step) {
                        if (!slot.has_next_token) {
                            break;
                        }
                        std::string tts;
                        if (slot.process_token(tok.piece, tok.eog, tok.index, tts)) {
                            any_sent = true;
                            send += tts;
                        }
                    }
                    if (any_sent) {
                        std::vector<common_chat_msg_diff> diffs;
                        state.update(send, true, diffs);
                        for (const auto & d : diffs) {
                            deltas.push_back(diff_to_json(d));
                        }
                    }
                }
                // final response: in stream mode its content is empty
                std::vector<common_chat_msg_diff> diffs;
                s["msg"] = msg_to_json(state.update("", false, diffs));
                for (const auto & d : diffs) {
                    deltas.push_back(diff_to_json(d));
                }
            } catch (const std::exception & e) {
                s["exception"] = e.what();
            }
            s["deltas"]     = deltas;
            s["stopped_at"] = slot.stopped_at;
            streams.push_back(s);
        }
        out["streams"] = streams;
        return out;
    }

    json handle(const json & req) {
        const std::string op = req.value("op", std::string());
        if (op == "hello") {
            return hello();
        }
        if (op == "tokenize") {
            return tokenize(req);
        }
        if (op == "replay") {
            return replay(req);
        }
        if (op == "special_ids") {
            return special_ids(req);
        }
        if (op == "shutdown") {
            return { { "ok", true } };
        }
        throw std::runtime_error("unknown op '" + op + "'");
    }
};

int main() {
    // Keep stdout for the protocol only: anything llama.cpp prints goes to stderr.
    int   proto_fd = dup(STDOUT_FILENO);
    FILE * proto   = fdopen(proto_fd, "w");
    dup2(STDERR_FILENO, STDOUT_FILENO);

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    harness     h;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        json reply;
        bool stop = false;
        try {
            json req = json::parse(line);
            stop     = req.value("op", std::string()) == "shutdown";
            reply    = h.handle(req);
        } catch (const std::exception & e) {
            reply = {
                { "ok",    false    },
                { "error", e.what() },
            };
        }
        std::string s = reply.dump_safe();
        fputs(s.c_str(), proto);
        fputc('\n', proto);
        fflush(proto);
        if (stop) {
            break;
        }
    }
    llama_backend_free();
    return 0;
}
