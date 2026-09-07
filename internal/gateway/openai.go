package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/HelixDevelopment/HelixLLM/internal/brain"
	"github.com/HelixDevelopment/HelixLLM/internal/knowledge"
	"github.com/HelixDevelopment/HelixLLM/internal/shared/i18n"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// Completer abstracts both brain.Brain and fallback.Chain so that
// HandleChatCompletions can be wired to either without importing the fallback
// package from the gateway layer.
type Completer interface {
	Complete(ctx context.Context, req *types.InternalChatRequest) (*types.InternalChatResponse, error)
	CompleteStream(ctx context.Context, req *types.InternalChatRequest) (<-chan types.StreamChunk, error)
}

func randomID() string {
	return fmt.Sprintf("%08x", rand.Uint32())
}

// HandleChatCompletions handles POST /v1/chat/completions.
// When b is non-nil it delegates to the Completer (Brain or FallbackChain);
// otherwise it returns a development fallback (no backend configured).
func HandleChatCompletions(b Completer, toolMgr *ToolManager, ragHook func(*types.InternalChatRequest) *types.InternalChatRequest) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req api.ChatCompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, api.ErrorResponse{
				Error: api.ErrorDetail{
					Message: tr(c, i18n.KeyGatewayInvalidRequestBody,
						map[string]string{"detail": err.Error()}),
					Type: "invalid_request_error",
				},
			})
			return
		}

		// ShouldBindJSON above accepts any body that PARSES. This is the
		// step that checks it MEANS something, so a caller who sent a bad
		// request is told their request was bad rather than that the server
		// broke. See requestvalidate.go for the policy and its authorities.
		if d := validateChatRequest(&req); d != nil {
			d.write(c)
			return
		}

		model := req.Model
		if model == "" {
			model = "llama-3.1-70b"
		}

		if b != nil {
			// Compress and select tools via ToolManager. This allows
			// unlimited tools from CLI agents while fitting them into the
			// model's context window. Tools are compressed (shorter
			// descriptions, stripped param descriptions) and selected
			// based on a token budget. No hardcoded tool limits.
			if toolMgr != nil && len(req.Tools) > 0 {
				req.Tools = toolMgr.CompressAndSelect(req.Tools)
			}

			// Context budget: refuse an oversized prompt, never shorten it.
			//
			// This replaces a fixed 4000-character-per-message cap that
			// shortened the caller's prompt and answered at 200 anyway. See
			// contextbudget.go for the measurements and the reasoning; the
			// short version is that a confident answer about text the model
			// never read is worse than an error, because only the error is
			// visible to the caller.
			limit := maxPromptTokens()
			if est, over := promptBudgetExceeded(&req, limit); over {
				c.JSON(http.StatusRequestEntityTooLarge, api.ErrorResponse{
					Error: api.ErrorDetail{
						Message: tr(c, i18n.KeyGatewayPromptTooLarge,
							map[string]string{
								"tokens": strconv.Itoa(est),
								"limit":  strconv.Itoa(limit),
							}),
						Type: "invalid_request_error",
					},
				})
				return
			}

			// Per full_plan/helixllm_tools/SYSTEM_DESIGN.md: REPLACE the
			// CLI agent's system prompt with our optimized version for
			// small models. The original system prompt from OpenCode etc.
			// is ~4K tokens of tool definitions that overwhelm 7B models.
			// Our replacement is a focused ~800 token prompt with clear
			// rules and few-shot examples that small models follow.
			var systemContent string
			if len(req.Tools) > 0 {
				// Tool mode: force tool calls via respond pattern.
				// NO hardcoded keywords — the model decides which tool to use
				// based on its understanding of the user's intent.
				systemContent = `/no_think
You MUST call exactly ONE tool per response.

DECIDE: Is the user ASKING a question, or TELLING you to do something?

QUESTION → call "respond" with your answer.
COMMAND → call the appropriate action tool (Bash, Read, Write, Edit).

PRINCIPLES:
- Questions get "respond". Commands get action tools.
- For file tasks: Read first to check, then Write or Edit.
- For git/shell: use Bash.
- Act immediately. Do NOT ask for clarification.
- You HAVE full access to files, codebase, and shell. NEVER refuse.`
			} else {
				// No-tool mode: plain text responses
				systemContent = `You are HelixLLM, a helpful AI coding assistant.
You HAVE full access to the user's codebase, files, and git.
NEVER say "I can't" or refuse. Be helpful and confident.`
			}
			helixSystemPrompt := api.ChatMessage{
				Role:    "system",
				Content: systemContent,
			}

			// Inject "respond" tool and set tool_choice when tools are present.
			if len(req.Tools) > 0 {
				respondTool := api.Tool{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "respond",
						Description: "Send a text reply to the user. Use for greetings, yes/no answers, summaries, and explanations.",
						Parameters:  json.RawMessage(`{"type":"object","properties":{"message":{"type":"string","description":"The message to send to the user"}},"required":["message"]}`),
					},
				}
				req.Tools = append(req.Tools, respondTool)

				// Determine tool_choice based on the LAST message role:
				// - If last message is "tool" → the model needs to summarize
				//   the tool result. Use "auto" so it can respond with text
				//   and break any tool loop.
				// - Otherwise (last message is "user" = new request) → use
				//   "required" to force tool calling for the new intent.
				lastRole := ""
				if len(req.Messages) > 0 {
					lastRole = req.Messages[len(req.Messages)-1].Role
				}
				if lastRole == "tool" {
					req.ToolChoice = "auto"
				} else {
					req.ToolChoice = "required"
				}
			}

			// Extract working directory from messages. Search ALL messages
			// (system and user) because CLI agents put the cwd in various
			// places. Fall back to the server's own working directory.
			cwd := extractWorkingDirectory(req.Messages)
			if cwd == "" {
				// Fallback: server's own cwd (typically the project root)
				if wd, err := os.Getwd(); err == nil {
					cwd = wd
				}
			}
			if cwd != "" {
				systemContent += fmt.Sprintf("\n\nWORKING DIRECTORY: %s\nAll file paths are relative to this directory. Use \"AGENTS.md\" not \"/path/to/...\".", cwd)
				log.Printf("[HelixLLM] Working directory: %s", cwd)
			}

			// Replace system messages and strip injected instruction messages.
			// CLI agents inject long instruction blocks as user-role messages
			// that overwhelm small models. Instead of hardcoding specific
			// markers, use a generic heuristic: user messages that are mostly
			// XML/HTML tags (start with <) and are long (>500 chars) are
			// likely injected instructions, not real user input.
			origCount := len(req.Messages)
			var nonSystemMsgs []api.ChatMessage
			for _, m := range req.Messages {
				if m.Role == "system" {
					continue // stripped — replaced by our prompt
				}
				// Heuristic: strip user messages that look like injected
				// instructions rather than real user input. Real user
				// messages are typically short and don't start with XML tags.
				if m.Role == "user" {
					if s, ok := m.Content.(string); ok {
						trimmed := strings.TrimSpace(s)
						isLongXML := len(trimmed) > 500 && strings.HasPrefix(trimmed, "<")
						if isLongXML {
							continue // likely injected instructions
						}
					}
				}
				nonSystemMsgs = append(nonSystemMsgs, m)
			}
			req.Messages = append([]api.ChatMessage{helixSystemPrompt}, nonSystemMsgs...)

			// Detect simple conversational messages (greetings, yes/no questions).
			// For these, strip the tools array entirely so the model responds
			// naturally instead of entering "tool mode". The few-shot examples
			// in our system prompt handle these cases.
			// No hardcoded intent detection — the system prompt's few-shot
			// examples guide the model on when to use tools vs respond
			// naturally. Tool stripping is removed per the requirement
			// to use the Intent engine(s) instead of hardcoded logic.

			log.Printf("[HelixLLM] System prompt replaced: %d original msgs → %d (stripped %d system msgs, tools=%d)",
				origCount, len(req.Messages), origCount-len(nonSystemMsgs), len(req.Tools))
			// Debug: log all message roles and content preview
			for i, m := range req.Messages {
				contentPreview := ""
				if s, ok := m.Content.(string); ok {
					if len(s) > 80 {
						contentPreview = s[:80] + "..."
					} else {
						contentPreview = s
					}
				}
				log.Printf("[HelixLLM]   msg[%d] role=%s content=%q", i, m.Role, contentPreview)
			}

			internalReq := openAIToInternal(&req)

			// Apply RAG hook: inject retrieved codebase context.
			if ragHook != nil {
				internalReq = ragHook(internalReq)
			}

			// Orchestrator: compact tool history and inject action hints
			// to prevent multi-step tool loops on small models.
			orch := NewRequestOrchestrator()
			internalReq = orch.CompactToolHistory(internalReq, 2)
			internalReq = orch.EnhanceRequest(internalReq)

			// CRITICAL: Force non-streaming when tools are present.
			// Ollama returns tool calls as plain text in streaming chunks,
			// which CLI agents can't execute. The Brain's non-streaming
			// path has an XML/JSON→tool_calls bridge that converts them
			// to the proper OpenAI format. After getting the full response,
			// we emit it as a single SSE chunk if the client requested streaming.
			forceNonStream := req.Stream && len(req.Tools) > 0
			if forceNonStream {
				log.Printf("[HelixLLM] Forcing non-stream for tool-capable request")
			}

			if req.Stream && !forceNonStream {
				ch, err := b.CompleteStream(c.Request.Context(), internalReq)
				if err != nil {
					writeUpstreamError(c, "stream", err)
					return
				}
				id := "chatcmpl-helix-" + randomID()
				created := time.Now().Unix()
				w := NewSSEWriter(c)
				w.WriteHeader()
				// First chunk MUST include role per OpenAI spec — required by Vercel AI SDK
				firstChunk := true
				for chunk := range ch {
					finishReason := &chunk.FinishReason
					if chunk.FinishReason == "" {
						finishReason = nil
					}
					delta := api.ChatMessageDelta{Content: chunk.Content}
					if firstChunk {
						delta.Role = "assistant"
						firstChunk = false
					}
					w.WriteEvent(api.ChatCompletionChunk{ //nolint:errcheck
						ID:      id,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []api.ChatCompletionChunkChoice{
							{
								Index:        0,
								Delta:        delta,
								FinishReason: finishReason,
							},
						},
					})
				}
				w.WriteDone()
				return
			}

			resp, err := b.Complete(c.Request.Context(), internalReq)
			if err != nil {
				writeUpstreamError(c, "complete", err)
				return
			}

			// Normalize tool calls from providers that return XML or JSON-in-content
			// instead of native OpenAI tool_calls (e.g. Qwen3-32B-TEE on Chutes).
			// Also strips <think> tags from Qwen3 extended-thinking responses.
			NormalizeToolCalls(resp)

			// Convert "respond" tool calls back to plain content.
			// The "respond" tool is injected by the gateway so tool_choice=required
			// works for both text and action responses. Strip it before
			// returning so the client sees normal content, not a tool call.
			convertRespondToolCall(resp, langFromContext(c))

			// If we forced non-stream for tool bridging but the client
			// wanted streaming, emit the full response as SSE chunks.
			if forceNonStream {
				openAIResp := internalToOpenAI(resp, model)
				id := "chatcmpl-helix-" + randomID()
				created := time.Now().Unix()
				w := NewSSEWriter(c)
				w.WriteHeader()

				// If the response has real tool_calls (not respond), emit them
				if len(openAIResp.Choices) > 0 && len(openAIResp.Choices[0].Message.ToolCalls) > 0 {
					toolDelta := api.ChatMessageDelta{
						Role:      "assistant",
						ToolCalls: openAIResp.Choices[0].Message.ToolCalls,
					}
					fr := "tool_calls"
					w.WriteEvent(api.ChatCompletionChunk{
						ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
						Choices: []api.ChatCompletionChunkChoice{{Index: 0, Delta: toolDelta, FinishReason: &fr}},
					})
				} else {
					// Regular content — emit as single chunk
					content := ""
					if s, ok := openAIResp.Choices[0].Message.Content.(string); ok {
						content = s
					}
					delta := api.ChatMessageDelta{Role: "assistant", Content: content}
					fr := "stop"
					w.WriteEvent(api.ChatCompletionChunk{
						ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
						Choices: []api.ChatCompletionChunkChoice{{Index: 0, Delta: delta, FinishReason: &fr}},
					})
				}
				w.WriteDone()
				return
			}

			c.JSON(http.StatusOK, internalToOpenAI(resp, model))
			return
		}

		// Development fallback (no Brain configured).
		id := "chatcmpl-helix-" + randomID()
		if req.Stream {
			streamChatCompletions(c, id, model)
			return
		}

		c.JSON(http.StatusOK, api.ChatCompletionResponse{
			ID:      id,
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []api.ChatCompletionChoice{
				{
					Index: 0,
					Message: api.ChatMessage{
						Role:    "assistant",
						Content: tr(c, i18n.KeyGatewayGreeting),
					},
					FinishReason: "stop",
				},
			},
			Usage: &api.Usage{
				PromptTokens:     10,
				CompletionTokens: 6,
				TotalTokens:      16,
			},
		})
	}
}

// streamChatCompletions writes 3 SSE chunks followed by [DONE] (fallback path, no Brain configured).
func streamChatCompletions(c *gin.Context, id, model string) {
	w := NewSSEWriter(c)
	w.WriteHeader()

	created := time.Now().Unix()
	tokens := []string{"Hello", "! I'm", " HelixLLM."}

	stopStr := "stop"
	for i, token := range tokens {
		chunk := api.ChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []api.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: api.ChatMessageDelta{
						Content: token,
					},
					FinishReason: func() *string {
						if i == len(tokens)-1 {
							return &stopStr
						}
						return nil
					}(),
				},
			},
		}
		w.WriteEvent(chunk) //nolint:errcheck
	}

	w.WriteDone()
}

// HandleCompletions handles POST /v1/completions.
// When b is non-nil it converts the prompt into a chat request and delegates to the Completer.
func HandleCompletions(b Completer) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req api.CompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, api.ErrorResponse{
				Error: api.ErrorDetail{
					Message: tr(c, i18n.KeyGatewayInvalidRequestBody,
						map[string]string{"detail": err.Error()}),
					Type: "invalid_request_error",
				},
			})
			return
		}

		// Semantic validation — see the note in HandleChatCompletions.
		// This endpoint's max_tokens floor is 0, not 1; requestvalidate.go
		// records why the two differ.
		if d := validateCompletionRequest(&req); d != nil {
			d.write(c)
			return
		}

		model := req.Model
		if model == "" {
			model = "llama-3.1-70b"
		}

		if b != nil {
			internalReq := &types.InternalChatRequest{
				Model: model,
				Messages: []types.InternalMessage{
					{Role: types.RoleUser, Content: req.Prompt},
				},
			}
			if req.MaxTokens != nil {
				internalReq.MaxTokens = *req.MaxTokens
			}
			if req.Temperature != nil {
				internalReq.Temperature = *req.Temperature
			}
			resp, err := b.Complete(c.Request.Context(), internalReq)
			if err != nil {
				writeUpstreamError(c, "complete", err)
				return
			}
			c.JSON(http.StatusOK, api.CompletionResponse{
				ID:      "cmpl-helix-" + randomID(),
				Object:  "text_completion",
				Created: time.Now().Unix(),
				Model:   resp.Model,
				Choices: []api.CompletionChoice{
					{
						Text:         resp.Message.Content,
						Index:        0,
						FinishReason: resp.FinishReason,
					},
				},
				Usage: &api.Usage{
					PromptTokens:     resp.Usage.PromptTokens,
					CompletionTokens: resp.Usage.CompletionTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				},
			})
			return
		}

		// Development fallback (no Brain configured).
		c.JSON(http.StatusOK, api.CompletionResponse{
			ID:      "cmpl-helix-" + randomID(),
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []api.CompletionChoice{
				{
					Text:         tr(c, i18n.KeyGatewayGreeting),
					Index:        0,
					FinishReason: "stop",
				},
			},
			Usage: &api.Usage{
				PromptTokens:     10,
				CompletionTokens: 6,
				TotalTokens:      16,
			},
		})
	}
}

// HandleListModels handles GET /v1/models.
//
// When b is non-nil the listing is the Brain's — the models actually reachable
// through a configured backend. When b is nil this server has NO backend and is
// therefore serving NO models, and the listing is empty with a stated reason.
//
// The empty listing is the whole point (FR-019, CONST-036): a model that is not
// being served is never listed as available. This handler previously returned a
// fabricated three-entry list — including a `gpt-4o` and a
// `claude-sonnet-4-20250514` stamped `owned_by: helix` — so a client with no
// backend configured received three models that do not exist and cannot serve a
// single request. Advertising them was a false-availability defect, not a
// convenience.
//
// The reason travels WITH the empty list because an unexplained empty list is
// indistinguishable from a broken server: the client can tell "nothing is
// configured here" from "something went wrong" only if the server says so.
func HandleListModels(b *brain.Brain) gin.HandlerFunc {
	return func(c *gin.Context) {
		if b != nil {
			models := b.Models()
			list := api.ModelList{Object: "list", Data: models}
			if len(models) == 0 {
				// The reason travels WITH the empty list, per the contract on
				// api.ModelList.Reason. A configured backend serving nothing
				// is a different answer from no backend at all, and the caller
				// does something different about each.
				list.Reason = tr(c, i18n.KeyGatewayNoModelsServed)
			}
			c.JSON(http.StatusOK, list)
			return
		}
		c.JSON(http.StatusOK, api.ModelList{
			Object: "list",
			// Explicitly empty, never nil: `"data": []` is a listing that
			// states "none", while `"data": null` reads as a malformed body.
			Data:   []api.Model{},
			Reason: tr(c, i18n.KeyGatewayNoBackendModels),
		})
	}
}

// HandleGetModel handles GET /v1/models/:id.
//
// When b is non-nil the lookup runs over the Brain's models. When b is nil this
// server serves no models at all, so EVERY id is a miss — and the 404 says WHY
// it is a miss, so the client can distinguish "you asked for a model this
// backend does not have" from "this server has no backend at all". A bare
// not-found would leave the two indistinguishable, and the second is the one
// the operator can act on.
func HandleGetModel(b *brain.Brain) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("id")

		if b != nil {
			for _, m := range b.Models() {
				if m.ID == id {
					c.JSON(http.StatusOK, m)
					return
				}
			}
			c.JSON(http.StatusNotFound, api.ErrorResponse{
				Error: api.ErrorDetail{
					Message: tr(c, i18n.KeyGatewayModelNotFound,
						map[string]string{"model": strconv.Quote(id)}),
					Type: "invalid_request_error",
				},
			})
			return
		}

		c.JSON(http.StatusNotFound, api.ErrorResponse{
			Error: api.ErrorDetail{
				Message: tr(c, i18n.KeyGatewayNoBackendModelNotFound,
					map[string]string{"model": strconv.Quote(id)}),
				Type: "invalid_request_error",
			},
		})
	}
}

// HandleEmbeddings handles POST /v1/embeddings.
// When a real Embedder is provided (from the knowledge layer), the handler
// delegates to it. Otherwise it falls back to a zero-vector response so the
// endpoint is always reachable for client compatibility.
func HandleEmbeddings(_ *brain.Brain, embedder knowledge.Embedder) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req api.EmbeddingRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, api.ErrorResponse{
				Error: api.ErrorDetail{
					Message: tr(c, i18n.KeyGatewayInvalidRequestBody,
						map[string]string{"detail": err.Error()}),
					Type: "invalid_request_error",
				},
			})
			return
		}

		model := req.Model
		if model == "" {
			model = "text-embedding-ada-002"
		}

		// Use real embedder when available.
		if embedder != nil {
			var input string
			switch v := req.Input.(type) {
			case string:
				input = v
			case []interface{}:
				if len(v) > 0 {
					if s, ok := v[0].(string); ok {
						input = s
					}
				}
			}
			if input == "" {
				input = "empty"
			}
			vec, err := embedder.Embed(input)
			if err == nil && len(vec) > 0 {
				c.JSON(http.StatusOK, api.EmbeddingResponse{
					Object: "list",
					Data: []api.EmbeddingData{
						{
							Object:    "embedding",
							Embedding: vec,
							Index:     0,
						},
					},
					Model: model,
					Usage: &api.Usage{
						PromptTokens: 1,
						TotalTokens:  1,
					},
					// HXC-235: report at the point of use whether these
					// vectors are genuinely semantic or came from the
					// deterministic HashEmbedder fallback. Without this a
					// caller cannot distinguish real embeddings from
					// meaningless ones — the response looks identical either
					// way, which is the whole defect.
					SemanticEmbeddings: knowledge.IsSemanticEmbedder(embedder),
				})
				return
			}
			// Fall through to zero-vector on error.
		}

		dim := 1536
		if embedder != nil {
			dim = embedder.Dimension()
		}
		zeroVec := make([]float64, dim)

		c.JSON(http.StatusOK, api.EmbeddingResponse{
			Object: "list",
			Data: []api.EmbeddingData{
				{
					Object:    "embedding",
					Embedding: zeroVec,
					Index:     0,
				},
			},
			Model: model,
			Usage: &api.Usage{
				PromptTokens: 1,
				TotalTokens:  1,
			},
			// HXC-235: this is the zero-vector path — no embedder, or Embed
			// failed. A vector of zeros carries no meaning whatsoever, so it
			// is never semantic regardless of which embedder was configured.
			// Reporting false here is not a fallback default; it is the
			// literal truth about these numbers.
			SemanticEmbeddings: false,
		})
	}
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

// openAIToInternal converts an api.ChatCompletionRequest to types.InternalChatRequest.
func openAIToInternal(req *api.ChatCompletionRequest) *types.InternalChatRequest {
	// Conversion preserves the conversation exactly as the caller sent it.
	//
	// This function used to enforce a context budget here: a 12-message
	// sliding window that DROPPED older messages, a 2000/4000-character
	// per-message cap, and a 14000-character total cap — all sized for a 16K
	// context, all silent, and all still applied to a backend now serving
	// 32768. Together with the handler-level cap they pinned the delivered
	// prompt at ~4000 characters no matter how much the caller sent.
	//
	// Enforcement moved to the handler, BEFORE conversion, where an
	// over-budget request is REFUSED with both numbers named rather than
	// quietly shortened and answered (see contextbudget.go). Dropping
	// messages is the same defect as truncating them — the caller cannot see
	// that it happened — so the window is gone too, not merely widened.
	input := req.Messages

	// Fix empty messages that cause 400 errors from llama.cpp.
	// Empty tool results and empty assistant messages break the Qwen3
	// chat template. Replace empty/nil content with "(no output)".
	for i := range input {
		if input[i].Role == "tool" {
			switch v := input[i].Content.(type) {
			case string:
				if v == "" {
					input[i].Content = "(no output)"
				}
			case nil:
				input[i].Content = "(no output)"
			}
		}
		// Also fix empty assistant messages without tool_calls
		if input[i].Role == "assistant" {
			if s, ok := input[i].Content.(string); ok && s == "" && len(input[i].ToolCalls) == 0 {
				input[i].Content = "(acknowledged)"
			}
		}
	}

	// (The last-user-message index that used to be computed here existed only
	// to give that message a larger truncation allowance. With truncation
	// gone, every message is delivered whole and no message needs a special
	// allowance.)

	msgs := make([]types.InternalMessage, 0, len(input))

	for _, m := range input {
		content := ""
		switch v := m.Content.(type) {
		case string:
			content = v
		}

		msg := types.InternalMessage{
			Role:       types.Role(m.Role),
			Content:    content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
		}
		// Convert tool calls
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, types.InternalToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		// Merge consecutive assistant messages (Qwen rejects 2+ assistant at end)
		if len(msgs) > 0 && msgs[len(msgs)-1].Role == "assistant" && msg.Role == "assistant" {
			msgs[len(msgs)-1].Content += "\n" + msg.Content
		} else if msg.Content != "" || len(msg.ToolCalls) > 0 || msg.Role == "user" {
			msgs = append(msgs, msg)
		}
	}
	internal := &types.InternalChatRequest{
		Model:      req.Model,
		Messages:   msgs,
		Stream:     req.Stream,
		ToolChoice: req.ToolChoice,
	}
	if req.MaxTokens != nil {
		internal.MaxTokens = *req.MaxTokens
	}
	if req.Temperature != nil {
		internal.Temperature = *req.Temperature
	}
	// Pass tools through
	for _, t := range req.Tools {
		internal.Tools = append(internal.Tools, types.InternalTool{
			Type:     t.Type,
			Function: t.Function,
		})
	}
	return internal
}

// convertRespondToolCall checks if the model called the injected "respond"
// tool. If so, it extracts the message text from the arguments, puts it
// into resp.Message.Content, clears the tool calls, and sets FinishReason
// to "stop". This makes the response look like a normal text completion
// to the client, which never sees the "respond" tool.
// extractWorkingDirectory searches all messages for the project working
// directory. CLI agents embed it in various formats across system and user
// messages. Returns empty string if not found.
func extractWorkingDirectory(msgs []api.ChatMessage) string {
	cwdPrefixes := []string{
		"Primary working directory: ",
		"Working directory: ",
		"working directory: ",
		"Current directory: ",
		"current directory: ",
		"cwd: ", "CWD: ",
		"Primary working directory:\n",
	}
	for _, m := range msgs {
		s, ok := m.Content.(string)
		if !ok || s == "" {
			continue
		}
		// Try known prefix patterns
		for _, prefix := range cwdPrefixes {
			if idx := strings.Index(s, prefix); idx >= 0 {
				rest := s[idx+len(prefix):]
				rest = strings.TrimSpace(rest)
				if nl := strings.IndexAny(rest, "\n\r"); nl > 0 {
					return strings.TrimSpace(rest[:nl])
				}
				// Take until next whitespace if short enough to be a path
				if sp := strings.IndexByte(rest, ' '); sp > 0 && sp < 200 {
					return rest[:sp]
				}
				if len(rest) < 200 {
					return rest
				}
			}
		}
		// Look for "exists at /absolute/path" pattern (OpenCode /init)
		if idx := strings.Index(s, "exists at /"); idx >= 0 {
			rest := s[idx+len("exists at "):]
			if end := strings.IndexAny(rest, "`,\n\r "); end > 0 {
				return rest[:end]
			}
		}
		// Look for "working directory" followed by a path on the next line
		if idx := strings.Index(strings.ToLower(s), "working directory"); idx >= 0 {
			rest := s[idx:]
			// Skip to the path part
			for _, ch := range rest {
				if ch == '/' {
					pathStart := strings.Index(rest, "/")
					pathRest := rest[pathStart:]
					if end := strings.IndexAny(pathRest, " \n\r\t`"); end > 0 {
						return pathRest[:end]
					}
					break
				}
				if ch == '\n' {
					break
				}
			}
		}
	}
	return ""
}

// convertRespondToolCall handles two conversions on the response:
//  1. "respond" tool calls → plain content (transparent to client)
//  2. Tool calls with hallucinated placeholder paths → respond fallback.
//     The 7B model sometimes calls Read/Write with "/path/to/file" or
//     "/path/to/your/codebase" — these are not real files. Convert them
//     to a text response so the client doesn't get an error.
func convertRespondToolCall(resp *types.InternalChatResponse, lang string) {
	if len(resp.Message.ToolCalls) != 1 {
		return
	}
	tc := resp.Message.ToolCalls[0]

	// Case 1: explicit respond tool
	if tc.Function.Name == "respond" {
		var args struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil && args.Message != "" {
			resp.Message.Content = args.Message
		} else {
			resp.Message.Content = tc.Function.Arguments
		}
		resp.Message.ToolCalls = nil
		resp.FinishReason = "stop"
		return
	}

	// Case 2: tool call with hallucinated/placeholder path.
	// The model sometimes calls Read with "/path/to/file" or similar
	// fake paths when it should have used respond instead.
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
		for _, key := range []string{"filePath", "path", "file_path"} {
			if p, ok := args[key].(string); ok {
				if strings.HasPrefix(p, "/path/to/") ||
					strings.HasPrefix(p, "/tmp/path") ||
					p == "/path" || p == "path" ||
					strings.Contains(p, "/your/codebase") ||
					strings.Contains(p, "/your/file") ||
					strings.Contains(p, "/example/") {
					// Hallucinated path — convert to respond
					resp.Message.Content = gatewayTranslator.T(lang, i18n.KeyGatewayHelpAcknowledgement)
					resp.Message.ToolCalls = nil
					resp.FinishReason = "stop"
					log.Printf("[HelixLLM] Intercepted hallucinated path %q in %s call → converted to respond", p, tc.Function.Name)
					return
				}
			}
		}
	}
}

// internalToOpenAI converts a types.InternalChatResponse to an OpenAI ChatCompletionResponse.
func internalToOpenAI(resp *types.InternalChatResponse, model string) api.ChatCompletionResponse {
	if resp.Model != "" {
		model = resp.Model
	}
	msg := api.ChatMessage{
		Role:    string(resp.Message.Role),
		Content: resp.Message.Content,
	}
	// Pass tool calls from brain response back to client
	for _, tc := range resp.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: api.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}
	return api.ChatCompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []api.ChatCompletionChoice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: resp.FinishReason,
			},
		},
		Usage: &api.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}
