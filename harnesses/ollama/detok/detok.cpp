// ctc-detok: llama-server's token rendering, for the Ollama adapter.
//
// At the pinned Ollama commit every GGUF model is served by an upstream
// llama-server subprocess (llm/server.go NewLlamaServer), built from the
// llama.cpp tag in Ollama's LLAMA_CPP_VERSION. Ollama sends its built-in
// parser's PreservedTokens() as "preserved_tokens" (llm/llama_server.go), and
// llama-server renders each sampled token as
//
//     common_token_to_piece(ctx, tok, accept_special_token(slot, tok))
//
// (tools/server/server-context.cpp), where accept_special_token is true only
// for ids in preserved_tokens (Ollama does not pass --special). The id set is
// built like tools/server/server-schema.cpp: a preserved string counts only if
// common_tokenize(vocab, s, false, true) yields exactly one id.
//
// This tool does exactly that from a vocab-only GGUF, so no model is needed.
//
// Protocol: one JSON object per line on stdin, one reply per line on stdout.
//   {"op":"hello"} -> {"ok":true,"llama_cpp_commit":..,"llama_cpp_build":..}
//   {"op":"pieces","gguf":path,"preserved":[str..],"ids":[int..]}
//       -> {"ok":true,"preserved_ids":[int..],"pieces":[[byte..]..],"eog":[bool..]}
//   {"op":"tokenize","gguf":path,"text":str}
//       -> {"ok":true,"ids":[int..]}   (llama-server tokenization: parse_special)
//   {"op":"control","gguf":path,"ids":[int..]}
//       -> {"ok":true,"control":[int..]}   (the given ids whose GGUF type is CONTROL)
// Pieces are byte arrays because a token can end inside a UTF-8 sequence.
#include "build-info.h"
#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <map>
#include <set>
#include <string>

using json = nlohmann::ordered_json;

static std::map<std::string, llama_model *> g_models;

static const llama_vocab * load_vocab(const std::string & path) {
    auto it = g_models.find(path);
    if (it == g_models.end()) {
        auto mparams       = llama_model_default_params();
        mparams.vocab_only = true;
        llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
        if (!model) {
            throw std::runtime_error("failed to load vocab GGUF: " + path);
        }
        it = g_models.emplace(path, model).first;
    }
    return llama_model_get_vocab(it->second);
}

static llama_token checked_id(const llama_vocab * vocab, const json & jt) {
    const llama_token tok     = jt.get<llama_token>();
    const int         n_vocab = llama_vocab_n_tokens(vocab);
    if (tok < 0 || tok >= n_vocab) {
        throw std::runtime_error("token id " + std::to_string(tok) + " out of range for vocab of " +
                                 std::to_string(n_vocab));
    }
    return tok;
}

static json handle(const json & req) {
    const std::string op = req.value("op", "");
    if (op == "hello") {
        return { { "ok", true }, { "llama_cpp_commit", llama_commit() }, { "llama_cpp_build", llama_build_number() } };
    }
    if (op == "pieces") {
        const llama_vocab *   vocab = load_vocab(req.at("gguf").get<std::string>());
        std::set<llama_token> preserved;
        for (const auto & t : req.value("preserved", json::array())) {
            auto ids = common_tokenize(vocab, t.get<std::string>(), false, true);
            if (ids.size() == 1) {
                preserved.insert(ids[0]);
            }
        }
        json pieces = json::array();
        json eog    = json::array();
        for (const auto & jt : req.at("ids")) {
            const llama_token tok = checked_id(vocab, jt);
            const std::string piece = common_token_to_piece(vocab, tok, preserved.count(tok) > 0);
            json              bytes = json::array();
            for (unsigned char c : piece) {
                bytes.push_back(static_cast<int>(c));
            }
            pieces.push_back(std::move(bytes));
            // llama-server stops generating after an end-of-generation token
            // (server_context::process_token), so later ids never reach Ollama.
            eog.push_back(llama_vocab_is_eog(vocab, tok));
        }
        return { { "ok", true }, { "preserved_ids", preserved }, { "pieces", pieces }, { "eog", eog } };
    }
    if (op == "tokenize") {
        const llama_vocab * vocab = load_vocab(req.at("gguf").get<std::string>());
        return { { "ok", true }, { "ids", common_tokenize(vocab, req.at("text").get<std::string>(), false, true) } };
    }
    if (op == "control") {
        const llama_vocab *   vocab = load_vocab(req.at("gguf").get<std::string>());
        std::set<llama_token> control;
        for (const auto & jt : req.at("ids")) {
            const llama_token tok = checked_id(vocab, jt);
            if (llama_vocab_is_control(vocab, tok)) {
                control.insert(tok);
            }
        }
        return { { "ok", true }, { "control", control } };
    }
    return { { "ok", false }, { "error", "unknown op: " + op } };
}

int main() {
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json reply;
        try {
            reply = handle(json::parse(line));
        } catch (const std::exception & e) {
            reply = { { "ok", false }, { "error", e.what() } };
        }
        std::cout << reply.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    }
    for (auto & kv : g_models) {
        llama_model_free(kv.second);
    }
    llama_backend_free();
    return 0;
}
