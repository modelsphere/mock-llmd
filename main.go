package main

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	tiktoken "github.com/pkoukk/tiktoken-go"
	"go.uber.org/zap"
)

var serverName string

func init() {
	serverName = os.Getenv("MOCK_SERVER_NAME")
	if serverName == "" {
		serverName = fmt.Sprintf("mock-%04x", rand.Uint32())
	}
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// MockOverrides mirrors the X-Mock-* headers but travels inside the request
// body, so gateways that strip non-standard headers (Bifrost, LiteLLM, etc.)
// can't drop the benchmark knobs on the floor. Pointer fields preserve the
// "unset" distinction — a missing key falls through to the header value or
// the built-in default instead of being interpreted as zero.
type MockOverrides struct {
	InputTokens       *int    `json:"input_tokens,omitempty"`
	OutputTokens      *int    `json:"output_tokens,omitempty"`
	CacheTokens       *int    `json:"cache_tokens,omitempty"`
	TTFTMs            *int    `json:"ttft_ms,omitempty"`
	ChunkIntervalMs   *int    `json:"chunk_interval_ms,omitempty"`
	TokensPerChunkMin *int    `json:"tokens_per_chunk_min,omitempty"`
	TokensPerChunkMax *int    `json:"tokens_per_chunk_max,omitempty"`
	HangupWithoutDone *bool   `json:"hangup_without_done,omitempty"`
	HangupAfterTokens *int    `json:"hangup_after_tokens,omitempty"`
	TokenLiteral      *string `json:"token_literal,omitempty"`
}

type ChatCompletionRequest struct {
	Model    string         `json:"model"`
	Stream   bool           `json:"stream"`
	Messages []ChatMessage  `json:"messages"`
	Mock     *MockOverrides `json:"mock,omitempty"`
}

func parseHeaderInt(r *http.Request, key string, defaultVal int) int {
	valStr := r.Header.Get(key)
	if valStr == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(valStr)
	if err != nil {
		return defaultVal
	}
	return val
}

// mockOpts is the fully-resolved set of behaviour knobs for one request:
// header defaults, overridden by the body's `mock` block, then clamped.
//
// Shared by the chat and legacy-completions handlers. Those two differ only in
// how they read the prompt and how they shape the response — every timing,
// hangup and usage knob below behaves identically on both, so resolving them
// once here keeps the two endpoints from drifting apart.
type mockOpts struct {
	inputTokensStr    string
	outputTokens      int
	chunkIntervalMs   int
	ttftMs            int
	tokensPerChunkMin int
	tokensPerChunkMax int
	cacheTokens       int
	hangupAfterTokens int
	tokenLiteral      string
	clientID          string
}

func resolveMockOpts(r *http.Request, mock *MockOverrides) mockOpts {
	// Parse custom testing headers (legacy / direct-to-mock path).
	o := mockOpts{
		inputTokensStr:  r.Header.Get("X-Mock-Input-Tokens"),
		outputTokens:    parseHeaderInt(r, "X-Mock-Output-Tokens", 50),
		chunkIntervalMs: parseHeaderInt(r, "X-Mock-Chunk-Interval-Ms", 20),
		ttftMs:          parseHeaderInt(r, "X-Mock-TTFT-Ms", 100),
		// Tokens per chunk: defaults to 1 (legacy behavior). When max > min, the
		// per-chunk count is drawn uniformly at random from [min, max] to mimic
		// real LLM servers that batch a variable number of tokens per SSE frame.
		tokensPerChunkMin: parseHeaderInt(r, "X-Mock-Tokens-Per-Chunk-Min", 5),
		// Cache hit size. Reported via OpenAI's usage.prompt_tokens_details.cached_tokens
		// — a subset of prompt_tokens. 0 omits the field entirely so tests that
		// don't exercise caching keep their existing wire shape.
		cacheTokens:       parseHeaderInt(r, "X-Mock-Cache-Tokens", 0),
		hangupAfterTokens: parseHeaderInt(r, "X-Mock-Hangup-After-Tokens", 0),
		tokenLiteral:      r.Header.Get("X-Mock-Token-Literal"),
		clientID:          r.Header.Get("X-Client-Id"),
	}
	o.tokensPerChunkMax = parseHeaderInt(r, "X-Mock-Tokens-Per-Chunk-Max", o.tokensPerChunkMin)
	if o.tokenLiteral == "" {
		o.tokenLiteral = "token"
	}
	hangupWithoutDone := r.Header.Get("X-Mock-Hangup-Without-Done") == "true" || r.Header.Get("X-Mock-Hangup-Without-Done") == "1"

	// Body overrides win over headers — gateways that strip custom headers
	// can't tamper with the body, so this is the portable benchmark path.
	if mock != nil {
		if mock.InputTokens != nil {
			o.inputTokensStr = strconv.Itoa(*mock.InputTokens)
		}
		if mock.OutputTokens != nil {
			o.outputTokens = *mock.OutputTokens
		}
		if mock.TTFTMs != nil {
			o.ttftMs = *mock.TTFTMs
		}
		if mock.ChunkIntervalMs != nil {
			o.chunkIntervalMs = *mock.ChunkIntervalMs
		}
		if mock.TokensPerChunkMin != nil {
			o.tokensPerChunkMin = *mock.TokensPerChunkMin
		}
		if mock.TokensPerChunkMax != nil {
			o.tokensPerChunkMax = *mock.TokensPerChunkMax
		}
		if mock.CacheTokens != nil {
			o.cacheTokens = *mock.CacheTokens
		}
		if mock.HangupWithoutDone != nil && *mock.HangupWithoutDone {
			hangupWithoutDone = true
		}
		if mock.HangupAfterTokens != nil {
			o.hangupAfterTokens = *mock.HangupAfterTokens
		}
		if mock.TokenLiteral != nil && *mock.TokenLiteral != "" {
			o.tokenLiteral = *mock.TokenLiteral
		}
	}

	if hangupWithoutDone && o.hangupAfterTokens <= 0 {
		o.hangupAfterTokens = o.outputTokens / 2
		if o.hangupAfterTokens <= 0 {
			o.hangupAfterTokens = 1
		}
	}

	// Clamp after all overrides land so out-of-range body values get fixed up
	// the same way header values do.
	if o.tokensPerChunkMin < 1 {
		o.tokensPerChunkMin = 1
	}
	if o.tokensPerChunkMax < o.tokensPerChunkMin {
		o.tokensPerChunkMax = o.tokensPerChunkMin
	}
	return o
}

// resolveInputTokens reports the prompt-token count: the explicit override when
// one was supplied, otherwise a real tiktoken count of the prompt text.
func resolveInputTokens(inputTokensStr, promptText, model string) int {
	if inputTokensStr != "" {
		if val, err := strconv.Atoi(inputTokensStr); err == nil {
			return val
		}
		return 10 // fallback default
	}
	inputTokens := 10
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		tkm, _ = tiktoken.GetEncoding("cl100k_base")
	}
	if tkm != nil {
		inputTokens = len(tkm.Encode(promptText, nil, nil))
	}
	zap.S().Debugf("Real token calculation: promptText='%s', inputTokens=%d", promptText, inputTokens)
	if inputTokens == 0 {
		inputTokens = 1 // Ensure at least 1 token is tracked
	}
	return inputTokens
}

// usageBlock builds the OpenAI usage object shared by both endpoints and both
// stream modes. cacheTokens is clamped to a valid subset of prompt by the
// caller — the worker computes fresh_in = max(tokens_in - cache_tokens_in, 0)
// and expects that invariant.
func usageBlock(inputTokens, outputTokens, cacheTokens int) map[string]any {
	usage := map[string]any{
		"prompt_tokens":     inputTokens,
		"completion_tokens": outputTokens,
		"total_tokens":      inputTokens + outputTokens,
	}
	if cacheTokens > 0 {
		usage["prompt_tokens_details"] = map[string]int{"cached_tokens": cacheTokens}
	}
	return usage
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Ensure it's a POST request
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
	zap.S().Infof("Received chat request from %s: model=%s, stream=%v", r.RemoteAddr, req.Model, req.Stream)

	opts := resolveMockOpts(r, req.Mock)
	outputTokens := opts.outputTokens
	chunkIntervalMs := opts.chunkIntervalMs
	ttftMs := opts.ttftMs
	tokensPerChunkMin := opts.tokensPerChunkMin
	tokensPerChunkMax := opts.tokensPerChunkMax
	cacheTokens := opts.cacheTokens
	hangupAfterTokens := opts.hangupAfterTokens
	tokenLiteral := opts.tokenLiteral
	clientID := opts.clientID

	var promptBuilder strings.Builder
	for i, m := range req.Messages {
		if i > 0 {
			promptBuilder.WriteString("\n")
		}
		promptBuilder.WriteString(m.Content)
	}
	inputTokens := resolveInputTokens(opts.inputTokensStr, promptBuilder.String(), req.Model)

	reqID := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	model := req.Model
	if model == "" {
		model = "mock-model"
	}

	// Clamp cache to a valid subset of prompt — the worker computes
	// fresh_in = max(tokens_in - cache_tokens_in, 0) and expects this invariant.
	if cacheTokens < 0 {
		cacheTokens = 0
	}
	if cacheTokens > inputTokens {
		cacheTokens = inputTokens
	}

	// Echo the model name we actually received back to the client via a
	// response header. Lets tests verify the gateway's body-rewrite (alias →
	// backend_model_name) actually changed what the upstream sees, without
	// having to parse the streamed body. Set BEFORE any Write so it lands in
	// the SSE headers too.
	w.Header().Set("X-Mock-Received-Model", req.Model)
	if serverName != "" {
		w.Header().Set("X-Mock-Server", serverName)
	}
	if echoTag := r.Header.Get("X-Test-Tag"); echoTag != "" {
		w.Header().Set("X-Mock-Received-Tag", echoTag)
	}

	if !req.Stream {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(ttftMs) * time.Millisecond):
		}

		var fullContent strings.Builder
		for i := 0; i < outputTokens; i++ {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Duration(chunkIntervalMs) * time.Millisecond):
			}
			content := " " + tokenLiteral
			if i == 0 {
				content = tokenLiteral
				if serverName != "" {
					content = fmt.Sprintf("[%s] %s", serverName, content)
				}
			}
			if clientID != "" {
				content = fmt.Sprintf("[%s] %s", clientID, content)
			}
			fullContent.WriteString(content)
		}

		w.Header().Set("Content-Type", "application/json")
		usage := usageBlock(inputTokens, outputTokens, cacheTokens)
		resp := map[string]any{
			"id":      reqID,
			"object":  "chat.completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": fullContent.String(),
					},
					"finish_reason": "stop",
				},
			},
			"usage": usage,
		}
		json.NewEncoder(w).Encode(resp)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Set required headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Simulate TTFT (Time To First Token)
	select {
	case <-r.Context().Done():
		return
	case <-time.After(time.Duration(ttftMs) * time.Millisecond):
	}

	// Send initial chunk (role)
	fmt.Fprintf(w, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n", reqID, created, model)
	flusher.Flush()

	// Stream tokens, packed into chunks of [min, max] tokens each. The first
	// token of the entire stream has no leading space; every subsequent token
	// is space-prefixed so the joined output reads as one fluent sentence
	// regardless of how chunk boundaries fall.
	emitted := 0
	firstOfStream := true
	for emitted < outputTokens {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(chunkIntervalMs) * time.Millisecond):
		}
		n := tokensPerChunkMin
		if tokensPerChunkMax > tokensPerChunkMin {
			n = tokensPerChunkMin + rand.IntN(tokensPerChunkMax-tokensPerChunkMin+1)
		}
		if emitted+n > outputTokens {
			n = outputTokens - emitted
		}
		var sb strings.Builder
		for j := 0; j < n; j++ {
			if !firstOfStream {
				sb.WriteByte(' ')
			}
			sb.WriteString(tokenLiteral)
			firstOfStream = false
		}
		content := sb.String()
		if firstOfStream && serverName != "" {
			content = fmt.Sprintf("[%s] %s", serverName, content)
		}
		if clientID != "" {
			content = fmt.Sprintf("[%s] %s", clientID, content)
		}

		fmt.Fprintf(w, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n", reqID, created, model, content)
		flusher.Flush()
		emitted += n

		if hangupAfterTokens > 0 && emitted >= hangupAfterTokens {
			zap.S().Infof("⚡ [Mock Server] Deliberately hanging up connection after %d tokens without [DONE] chunk!", emitted)
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, _ := hj.Hijack()
				_ = conn.Close()
			} else {
				panic(http.ErrAbortHandler)
			}
			return
		}
	}

	// Send finish reason chunk
	select {
	case <-r.Context().Done():
		return
	case <-time.After(time.Duration(chunkIntervalMs) * time.Millisecond):
	}
	fmt.Fprintf(w, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n", reqID, created, model)
	flusher.Flush()

	// Send usage chunk. Marshal the usage block so the cached_tokens detail
	// can be conditionally included without printf-string surgery.
	usageJSON, _ := json.Marshal(usageBlock(inputTokens, outputTokens, cacheTokens))
	fmt.Fprintf(w, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[],\"usage\":%s}\n\n", reqID, created, model, usageJSON)
	flusher.Flush()

	// Send DONE
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// CompletionRequest is the legacy /v1/completions body. `prompt` is a raw
// message because the OpenAI spec allows a string, an array of strings, or
// pre-tokenised ids — decodeCompletionPrompt handles the first two and treats
// the rest as empty, matching what the gateway's own prompt capture does.
type CompletionRequest struct {
	Model  string          `json:"model"`
	Stream bool            `json:"stream"`
	Prompt json.RawMessage `json:"prompt"`
	Mock   *MockOverrides  `json:"mock,omitempty"`
}

func decodeCompletionPrompt(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []string
	if err := json.Unmarshal(raw, &parts); err == nil {
		return strings.Join(parts, "\n")
	}
	return ""
}

// handleCompletions serves the legacy text-completion endpoint. Every mock knob
// behaves exactly as it does on /v1/chat/completions — the differences are
// confined to the wire shape: `object` is text_completion, the generated text
// rides in choices[].text rather than message.content / delta.content, and the
// id carries the cmpl- prefix real OpenAI uses.
func handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
	zap.S().Infof("Received completion request from %s: model=%s, stream=%v", r.RemoteAddr, req.Model, req.Stream)

	opts := resolveMockOpts(r, req.Mock)
	inputTokens := resolveInputTokens(opts.inputTokensStr, decodeCompletionPrompt(req.Prompt), req.Model)

	reqID := fmt.Sprintf("cmpl-mock-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	model := req.Model
	if model == "" {
		model = "mock-model"
	}

	cacheTokens := opts.cacheTokens
	if cacheTokens < 0 {
		cacheTokens = 0
	}
	if cacheTokens > inputTokens {
		cacheTokens = inputTokens
	}

	w.Header().Set("X-Mock-Received-Model", req.Model)
	if serverName != "" {
		w.Header().Set("X-Mock-Server", serverName)
	}
	if echoTag := r.Header.Get("X-Test-Tag"); echoTag != "" {
		w.Header().Set("X-Mock-Received-Tag", echoTag)
	}

	// nextText renders one chunk of n tokens, space-joining so the reassembled
	// completion reads as one fluent line regardless of chunk boundaries.
	firstOfStream := true
	nextText := func(n int) string {
		var sb strings.Builder
		for j := 0; j < n; j++ {
			if !firstOfStream {
				sb.WriteByte(' ')
			}
			sb.WriteString(opts.tokenLiteral)
			firstOfStream = false
		}
		text := sb.String()
		if opts.clientID != "" {
			text = fmt.Sprintf("[%s] %s", opts.clientID, text)
		}
		return text
	}

	if !req.Stream {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(opts.ttftMs) * time.Millisecond):
		}

		var full strings.Builder
		if serverName != "" {
			full.WriteString(fmt.Sprintf("[%s] ", serverName))
		}
		for i := 0; i < opts.outputTokens; i++ {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Duration(opts.chunkIntervalMs) * time.Millisecond):
			}
			full.WriteString(nextText(1))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      reqID,
			"object":  "text_completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{
					"index":         0,
					"text":          full.String(),
					"logprobs":      nil,
					"finish_reason": "stop",
				},
			},
			"usage": usageBlock(inputTokens, opts.outputTokens, cacheTokens),
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	select {
	case <-r.Context().Done():
		return
	case <-time.After(time.Duration(opts.ttftMs) * time.Millisecond):
	}

	emitted := 0
	for emitted < opts.outputTokens {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(opts.chunkIntervalMs) * time.Millisecond):
		}
		n := opts.tokensPerChunkMin
		if opts.tokensPerChunkMax > opts.tokensPerChunkMin {
			n = opts.tokensPerChunkMin + rand.IntN(opts.tokensPerChunkMax-opts.tokensPerChunkMin+1)
		}
		if emitted+n > opts.outputTokens {
			n = opts.outputTokens - emitted
		}
		text := nextText(n)
		if emitted == 0 && serverName != "" {
			text = fmt.Sprintf("[%s] %s", serverName, text)
		}

		chunk, _ := json.Marshal(map[string]any{
			"id":      reqID,
			"object":  "text_completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "text": text, "logprobs": nil, "finish_reason": nil},
			},
		})
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		flusher.Flush()
		emitted += n

		if opts.hangupAfterTokens > 0 && emitted >= opts.hangupAfterTokens {
			zap.S().Infof("⚡ [Mock Server] Deliberately hanging up completion after %d tokens without [DONE] chunk!", emitted)
			if hj, ok := w.(http.Hijacker); ok {
				conn, _, _ := hj.Hijack()
				_ = conn.Close()
			} else {
				panic(http.ErrAbortHandler)
			}
			return
		}
	}

	select {
	case <-r.Context().Done():
		return
	case <-time.After(time.Duration(opts.chunkIntervalMs) * time.Millisecond):
	}
	stopChunk, _ := json.Marshal(map[string]any{
		"id":      reqID,
		"object":  "text_completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{
			{"index": 0, "text": "", "logprobs": nil, "finish_reason": "stop"},
		},
	})
	fmt.Fprintf(w, "data: %s\n\n", stopChunk)
	flusher.Flush()

	// Usage rides on a final choices-empty chunk, exactly as the chat endpoint
	// does — this is what the gateway's openaiStreamParser looks for.
	usageChunk, _ := json.Marshal(map[string]any{
		"id":      reqID,
		"object":  "text_completion",
		"created": created,
		"model":   model,
		"choices": []map[string]any{},
		"usage":   usageBlock(inputTokens, opts.outputTokens, cacheTokens),
	})
	fmt.Fprintf(w, "data: %s\n\n", usageChunk)
	flusher.Flush()

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleListModels advertises the model IDs that this mock upstream serves so
// gateways which auto-route by probing /v1/models (e.g. Bifrost) can discover
// it. Defaults to "test-model" to match the DB seed in internal/db/postgres.go
// and the e2e tests. Override with MOCK_MODEL_IDS=foo,bar for other setups.
// The mock's chat handler accepts any model name in requests regardless of
// what's advertised here — this list is only for upstream discovery.
func handleListModels(w http.ResponseWriter, r *http.Request) {
	ids := []string{"test-model"}
	if env := strings.TrimSpace(os.Getenv("MOCK_MODEL_IDS")); env != "" {
		ids = ids[:0]
		for p := range strings.SplitSeq(env, ",") {
			if p = strings.TrimSpace(p); p != "" {
				ids = append(ids, p)
			}
		}
	}
	data := make([]map[string]any, len(ids))
	for i, id := range ids {
		data[i] = map[string]any{
			"id":       id,
			"object":   "model",
			"created":  0,
			"owned_by": "mock",
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   data,
	})
}

func handleMessages(w http.ResponseWriter, r *http.Request) {
	inputTokensStr := r.Header.Get("X-Mock-Input-Tokens")
	outputTokens := parseHeaderInt(r, "X-Mock-Output-Tokens", 50)
	chunkIntervalMs := parseHeaderInt(r, "X-Mock-Chunk-Interval-Ms", 20)
	ttftMs := parseHeaderInt(r, "X-Mock-TTFT-Ms", 100)
	tokensPerChunkMin := parseHeaderInt(r, "X-Mock-Tokens-Per-Chunk-Min", 5)
	tokensPerChunkMax := parseHeaderInt(r, "X-Mock-Tokens-Per-Chunk-Max", tokensPerChunkMin)
	cacheTokens := parseHeaderInt(r, "X-Mock-Cache-Tokens", 0)
	hangupWithoutDone := (r.Header.Get("X-Mock-Hangup-Without-Done") == "true" || r.Header.Get("X-Mock-Hangup-Without-Done") == "1")
	hangupAfterTokens := parseHeaderInt(r, "X-Mock-Hangup-After-Tokens", 0)
	tokenLiteral := r.Header.Get("X-Mock-Token-Literal")
	if tokenLiteral == "" {
		tokenLiteral = "token"
	}
	clientID := r.Header.Get("X-Client-Id")

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}
	zap.S().Infof("Received message request from %s: model=%s, stream=%v", r.RemoteAddr, req.Model, req.Stream)

	if req.Mock != nil {
		if req.Mock.InputTokens != nil {
			inputTokensStr = strconv.Itoa(*req.Mock.InputTokens)
		}
		if req.Mock.OutputTokens != nil {
			outputTokens = *req.Mock.OutputTokens
		}
		if req.Mock.TTFTMs != nil {
			ttftMs = *req.Mock.TTFTMs
		}
		if req.Mock.ChunkIntervalMs != nil {
			chunkIntervalMs = *req.Mock.ChunkIntervalMs
		}
		if req.Mock.TokensPerChunkMin != nil {
			tokensPerChunkMin = *req.Mock.TokensPerChunkMin
		}
		if req.Mock.TokensPerChunkMax != nil {
			tokensPerChunkMax = *req.Mock.TokensPerChunkMax
		}
		if req.Mock.CacheTokens != nil {
			cacheTokens = *req.Mock.CacheTokens
		}
		if req.Mock.HangupWithoutDone != nil && *req.Mock.HangupWithoutDone {
			hangupWithoutDone = true
		}
		if req.Mock.HangupAfterTokens != nil {
			hangupAfterTokens = *req.Mock.HangupAfterTokens
		}
		if req.Mock.TokenLiteral != nil && *req.Mock.TokenLiteral != "" {
			tokenLiteral = *req.Mock.TokenLiteral
		}
	}

	if hangupWithoutDone && hangupAfterTokens <= 0 {
		hangupAfterTokens = outputTokens / 2
		if hangupAfterTokens <= 0 {
			hangupAfterTokens = 1
		}
	}

	if tokensPerChunkMin < 1 {
		tokensPerChunkMin = 1
	}
	if tokensPerChunkMax < tokensPerChunkMin {
		tokensPerChunkMax = tokensPerChunkMin
	}

	inputTokens := 10
	if inputTokensStr != "" {
		if val, err := strconv.Atoi(inputTokensStr); err == nil {
			inputTokens = val
		}
	} else {
		var promptText strings.Builder
		for i, m := range req.Messages {
			if i > 0 {
				promptText.WriteString("\n")
			}
			promptText.WriteString(m.Content)
		}

		tkm, err := tiktoken.EncodingForModel(req.Model)
		if err != nil {
			tkm, _ = tiktoken.GetEncoding("cl100k_base")
		}
		if tkm != nil {
			inputTokens = len(tkm.Encode(promptText.String(), nil, nil))
		}
		if inputTokens == 0 {
			inputTokens = 1
		}
	}

	reqID := fmt.Sprintf("msg_mock_%d", time.Now().UnixNano())
	model := req.Model
	if model == "" {
		model = "mock-claude"
	}

	if cacheTokens < 0 {
		cacheTokens = 0
	}
	if cacheTokens > inputTokens {
		cacheTokens = inputTokens
	}

	w.Header().Set("X-Mock-Received-Model", req.Model)
	if serverName != "" {
		w.Header().Set("X-Mock-Server", serverName)
	}
	if echoTag := r.Header.Get("X-Test-Tag"); echoTag != "" {
		w.Header().Set("X-Mock-Received-Tag", echoTag)
	}

	if !req.Stream {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(ttftMs) * time.Millisecond):
		}

		var fullContent strings.Builder
		for i := 0; i < outputTokens; i++ {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Duration(chunkIntervalMs) * time.Millisecond):
			}
			content := " " + tokenLiteral
			if i == 0 {
				content = tokenLiteral
				if serverName != "" {
					content = fmt.Sprintf("[%s] %s", serverName, content)
				}
			}
			if clientID != "" {
				content = fmt.Sprintf("[%s] %s", clientID, content)
			}
			fullContent.WriteString(content)
		}

		w.Header().Set("Content-Type", "application/json")
		usage := map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		if cacheTokens > 0 {
			usage["cache_read_input_tokens"] = cacheTokens
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":    reqID,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"content": []map[string]any{
				{"type": "text", "text": fullContent.String()},
			},
			"usage": usage,
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	select {
	case <-r.Context().Done():
		return
	case <-time.After(time.Duration(ttftMs) * time.Millisecond):
	}

	msgUsage := map[string]any{
		"input_tokens": inputTokens,
	}
	if cacheTokens > 0 {
		msgUsage["cache_read_input_tokens"] = cacheTokens
	}
	msgStart := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":    reqID,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"usage": msgUsage,
		},
	}
	startBytes, _ := json.Marshal(msgStart)
	fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", startBytes)
	flusher.Flush()

	blockStart := map[string]any{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]any{
			"type": "text",
			"text": "",
		},
	}
	blockStartBytes, _ := json.Marshal(blockStart)
	fmt.Fprintf(w, "event: content_block_start\ndata: %s\n\n", blockStartBytes)
	flusher.Flush()

	emitted := 0
	firstOfStream := true
	for emitted < outputTokens {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(time.Duration(chunkIntervalMs) * time.Millisecond):
		}
		n := tokensPerChunkMin
		if tokensPerChunkMax > tokensPerChunkMin {
			n = tokensPerChunkMin + rand.IntN(tokensPerChunkMax-tokensPerChunkMin+1)
		}
		if emitted+n > outputTokens {
			n = outputTokens - emitted
		}
		var sb strings.Builder
		for j := 0; j < n; j++ {
			if !firstOfStream {
				sb.WriteByte(' ')
			}
			sb.WriteString(tokenLiteral)
			firstOfStream = false
		}
		content := sb.String()
		if firstOfStream && serverName != "" {
			content = fmt.Sprintf("[%s] %s", serverName, content)
		}
		if clientID != "" {
			content = fmt.Sprintf("[%s] %s", clientID, content)
		}

		blockDelta := map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type": "text_delta",
				"text": content,
			},
		}
		blockDeltaBytes, _ := json.Marshal(blockDelta)
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", blockDeltaBytes)
		flusher.Flush()
		emitted += n

		if hangupAfterTokens > 0 && emitted >= hangupAfterTokens {
			zap.S().Infof("⚡ [Mock Server] Deliberately hanging up connection after %d tokens without message_stop!", emitted)
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, _ := hj.Hijack()
				_ = conn.Close()
			} else {
				panic(http.ErrAbortHandler)
			}
			return
		}
	}

	blockStop := map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	}
	blockStopBytes, _ := json.Marshal(blockStop)
	fmt.Fprintf(w, "event: content_block_stop\ndata: %s\n\n", blockStopBytes)
	flusher.Flush()

	msgDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason": "end_turn",
		},
		"usage": map[string]any{
			"output_tokens": outputTokens,
		},
	}
	deltaBytes, _ := json.Marshal(msgDelta)
	fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", deltaBytes)
	flusher.Flush()

	fmt.Fprintf(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	flusher.Flush()
}

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()
	zap.ReplaceGlobals(logger)

	// Pre-warm the tokenizer dictionary to prevent startup latency on first request
	_, _ = tiktoken.GetEncoding("cl100k_base")

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/v1/completions", handleCompletions)
	mux.HandleFunc("/v1/messages", handleMessages)
	mux.HandleFunc("GET /v1/models", handleListModels)

	port := 8081
	if portStr := os.Getenv("PORT"); portStr != "" {
		if val, err := strconv.Atoi(portStr); err == nil && val > 0 {
			port = val
		}
	}
	addr := fmt.Sprintf(":%d", port)
	if envAddr := os.Getenv("LISTEN_ADDR"); envAddr != "" {
		addr = envAddr
	}
	zap.S().Infof("Mock LLM Server listening on %s...", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		zap.S().Fatalf("Server failed: %v", err)
	}
}
