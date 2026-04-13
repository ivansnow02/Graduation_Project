// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package logic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"backend/internal/svc"
	"backend/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ChatStreamLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

// SSE streaming chat
func NewChatStreamLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ChatStreamLogic {
	return &ChatStreamLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

// openaiRequest is the request body for the OpenAI-compatible API
type openaiRequest struct {
	Model       string              `json:"model"`
	Messages    []types.ChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	Temperature float64             `json:"temperature"`
	MaxTokens   int                 `json:"max_tokens"`
}

func (l *ChatStreamLogic) ChatStream(req *types.ChatStreamReq, w http.ResponseWriter, r *http.Request) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Build messages with optional system prompt
	messages := make([]types.ChatMessage, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		messages = append(messages, types.ChatMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}
	messages = append(messages, req.Messages...)

	// Build OpenAI-compatible request
	openaiReq := openaiRequest{
		Model:       req.Model.Name,
		Messages:    messages,
		Stream:      true,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		l.writeSSEError(w, flusher, fmt.Sprintf("Failed to marshal request: %v", err))
		return
	}

	// Create HTTP request to upstream API
	baseURL := strings.TrimRight(req.Model.BaseUrl, "/")
	apiURL := baseURL + "/chat/completions"

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, apiURL, bytes.NewReader(body))
	if err != nil {
		l.writeSSEError(w, flusher, fmt.Sprintf("Failed to create request: %v", err))
		return
	}

	upstreamReq.Header.Set("Content-Type", "application/json")
	upstreamReq.Header.Set("Authorization", "Bearer "+req.Model.ApiKey)

	// Execute request
	client := &http.Client{}
	resp, err := client.Do(upstreamReq)
	if err != nil {
		l.writeSSEError(w, flusher, fmt.Sprintf("Failed to call upstream API: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		l.writeSSEError(w, flusher, fmt.Sprintf("Upstream API error (status %d): %s", resp.StatusCode, string(bodyBytes)))
		return
	}

	// Stream SSE response from upstream to client
	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size for large responses
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines
		if line == "" {
			continue
		}

		// Forward SSE data lines
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			// Check for stream end signal
			if data == "[DONE]" {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				break
			}

			// Forward the data chunk as-is
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		l.Errorf("Scanner error: %v", err)
	}
}

func (l *ChatStreamLogic) writeSSEError(w http.ResponseWriter, flusher http.Flusher, errMsg string) {
	l.Errorf("SSE Error: %s", errMsg)
	errData, _ := json.Marshal(map[string]string{"error": errMsg})
	fmt.Fprintf(w, "data: %s\n\n", string(errData))
	flusher.Flush()
}
