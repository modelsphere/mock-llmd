package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMockServer_TokenLiteral_NonStreaming(t *testing.T) {
	reqBody := `{"model":"test-model","stream":false,"messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("X-Mock-Output-Tokens", "3")
	req.Header.Set("X-Mock-Token-Literal", "custom_word")
	req.Header.Set("X-Mock-TTFT-Ms", "0")
	req.Header.Set("X-Mock-Chunk-Interval-Ms", "0")

	rr := httptest.NewRecorder()
	handleChatCompletions(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("missing choices")
	}
	choice := choices[0].(map[string]any)
	message := choice["message"].(map[string]any)
	content := message["content"].(string)

	expectedSuffix := "custom_word custom_word custom_word"
	if !strings.HasSuffix(content, expectedSuffix) {
		t.Errorf("expected content to end with %q, got %q", expectedSuffix, content)
	}
}

func TestMockServer_TokenLiteral_Streaming(t *testing.T) {
	customWord := "hello_stream"
	reqBody := ChatCompletionRequest{
		Model:  "test-model",
		Stream: true,
		Mock: &MockOverrides{
			OutputTokens:      ptrInt(4),
			TokenLiteral:      &customWord,
			TTFTMs:            ptrInt(0),
			ChunkIntervalMs:   ptrInt(0),
			TokensPerChunkMin: ptrInt(1),
			TokensPerChunkMax: ptrInt(1),
		},
	}
	bodyBytes, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyBytes))

	rr := httptest.NewRecorder()
	handleChatCompletions(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}

	out, _ := io.ReadAll(rr.Body)
	outStr := string(out)

	if !strings.Contains(outStr, `"content":"hello_stream"`) {
		t.Errorf("expected output to contain custom token literal, got:\n%s", outStr)
	}
	if strings.Contains(outStr, `"content":"token"`) {
		t.Errorf("unexpected default 'token' found in output when overridden:\n%s", outStr)
	}
}

func ptrInt(v int) *int {
	return &v
}
