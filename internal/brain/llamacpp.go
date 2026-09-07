package brain

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/HelixDevelopment/HelixLLM/internal/brain/models"
	"github.com/HelixDevelopment/HelixLLM/pkg/api"
	"github.com/HelixDevelopment/HelixLLM/pkg/types"
)

// backendErrorDetailLimit caps how much of a backend error body is carried in
// the returned error. The body is diagnostic prose, not a payload; a bound
// keeps a misbehaving or verbose backend from pushing an unbounded string
// into the log and the error chain.
const backendErrorDetailLimit = 512

// readBackendError extracts the backend's own explanation of a non-200.
//
// The status code alone loses the only part a caller can act on. llama.cpp
// answers an oversized prompt with HTTP 400 and a body that names BOTH
// numbers — e.g.
//
//	request (141440 tokens) exceeds the available context size (4096 tokens)
//
// Discarding that body collapsed a precise, actionable refusal into a bare
// "unexpected status 400", which the gateway could only render as the generic
// "the model provider failed to complete this request". The caller was left
// unable to distinguish a prompt that was simply too large from a broken
// backend.
//
// The gateway decides what of this reaches a client: internal/gateway's
// upstream-error funnel relays only the count-bearing clause of a
// context-size refusal and redacts everything else, so widening the error
// here does not widen client-facing disclosure.
//
// A body that cannot be read contributes nothing rather than failing the
// request — the status code is still reported either way.
func readBackendError(body io.Reader) string {
	if body == nil {
		return ""
	}
	b, err := io.ReadAll(io.LimitReader(body, backendErrorDetailLimit))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// LlamaCppProvider implements Provider by calling llama.cpp's OpenAI-compatible
// API at the configured base URL.
type LlamaCppProvider struct {
	baseURL  string
	models   []string
	client   *http.Client
	registry *models.Registry

	// machineName resolves this machine's own name, and exists as a field so a
	// test can drive the case where the machine cannot say what it is called.
	// That case is not exotic — os.Hostname() answers "localhost" on stock VM
	// images, live images and many containers — and it is the branch that
	// decides whether a loopback literal can escape into a published identity.
	// Without the seam it could only be exercised by renaming the host running
	// the test suite.
	machineName func() string
}

// NewLlamaCppProvider creates a new llama.cpp provider pointing at the given
// base URL (e.g. "http://localhost:8080"). The models slice lists model IDs
// that this llama.cpp instance serves.
func NewLlamaCppProvider(baseURL string, models []string) *LlamaCppProvider {
	return &LlamaCppProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		models:  models,
		client: &http.Client{
			Timeout: 5 * time.Minute, // LLM completions can be slow.
		},
		machineName: thisMachineName,
	}
}

func (p *LlamaCppProvider) Name() string { return "llamacpp" }

// ServingHost reports the machine this llama.cpp instance runs on. It is what
// makes the models it serves nameable as helixllm/<host>/<model> (FR-014,
// FR-023): without it they would be indistinguishable from a remote vendor's
// models in a user's model list.
//
// The port is deliberately dropped — the identity names the HOST, and two
// instances on one machine are told apart by the models they serve, not by a
// port number the user never sees. A base URL with no parseable host yields "",
// which leaves the models un-prefixed rather than inventing a hostname.
//
// A LOOPBACK or wildcard base URL is resolved to this machine's own name
// instead of being published verbatim, because such a URL names no machine.
// That is the ordinary case, not an exotic one: HELIX_LLM_LOCAL_RPC_HOST
// defaults to "localhost" and cmd/helixllm rewrites it to "127.0.0.1" whenever
// the embedded llama-server is used. Publishing it verbatim caused two real
// failures:
//
//   - `helixllm/127.0.0.1/<model>` names no machine an operator could find,
//     which is exactly the question FR-023 exists to answer;
//   - it is the SAME string on every machine, so two gateways on two different
//     hosts published identical identities, identical digests and identical
//     ids. The Claude Toolkit de-duplicates by that id
//     (`group_by(.provider_id) | map(.[0])`), so one host's models silently
//     replaced the other's. That collision sits one layer above the digest and
//     no amount of hashing could have prevented it.
//
// Resolving both spellings to one machine name also removes a stability trap:
// before, flipping HELIX_LLM_LOCAL_RPC_HOST between "localhost" and "127.0.0.1"
// — which the embedded-server path does implicitly — re-minted every published
// identifier and silently invalidated a user's tool configuration.
//
// A base URL that already names a real machine is returned untouched: the
// substitution repairs a URL that names nothing, it does not overwrite one that
// names something.
//
// If this machine cannot say what it is called either — os.Hostname() answers
// "localhost" on stock cloud images, live images and many containers — the
// answer is NO HOST, not the loopback literal. Falling back to the literal
// would hand back exactly the value the substitution exists to remove, on
// exactly the machines least able to tolerate it, and both failures above would
// still be live there. Reporting no host instead leaves the models un-prefixed,
// which is what this function already does for a base URL with no parseable
// host, and what every remote provider does.
//
// That makes the invariant total, and it is relied upon: naming.RetiredHosts
// records the retired loopback renderings as a CLOSED historical set on the
// grounds that this resolver can no longer emit one. Returning the literal here
// made that a hopeful assertion; returning "" makes it true by construction.
func (p *LlamaCppProvider) ServingHost() string {
	host := hostFromBaseURL(p.baseURL)
	if host == "" || !namesNoMachine(host) {
		return host
	}
	if machine := p.thisMachine(); machine != "" {
		return machine
	}
	return ""
}

// thisMachine resolves this machine's own name through the provider's seam,
// falling back to the real resolver for a provider built as a bare literal.
func (p *LlamaCppProvider) thisMachine() string {
	if p.machineName != nil {
		return p.machineName()
	}
	return thisMachineName()
}

// hostFromBaseURL extracts the host from a base URL, tolerating the bare
// "host:port" form that does not parse as an authority.
func hostFromBaseURL(baseURL string) string {
	if u, err := url.Parse(baseURL); err == nil && u.Hostname() != "" {
		return u.Hostname()
	}
	host := baseURL
	if i := strings.Index(host, "/"); i >= 0 {
		host = host[:i]
	}
	if i := strings.LastIndex(host, ":"); i > 0 {
		host = host[:i]
	}
	return host
}

// namesNoMachine reports whether a host is one of the addresses that is the
// same on every machine and therefore identifies none of them: loopback in
// either spelling, and the wildcard bind addresses.
func namesNoMachine(host string) bool {
	lower := strings.ToLower(strings.Trim(host, "[]"))
	if lower == "localhost" || strings.HasSuffix(lower, ".localhost") ||
		lower == "localhost.localdomain" {
		return true
	}
	if ip := net.ParseIP(lower); ip != nil {
		return ip.IsLoopback() || ip.IsUnspecified()
	}
	return false
}

// thisMachineName is the machine's own name, normalised the way an identity
// host must be: lower-cased (hostnames are case-insensitive, RFC 4343) and
// trimmed to the short name so the identity stays readable — a short name and
// its FQDN are the same machine, and publishing both would make one machine
// look like two.
//
// It returns "" when the machine cannot say what it is called, or calls itself
// something that names no machine either, so the caller can fall back rather
// than publish a fabricated name.
func thisMachineName() string {
	raw, err := os.Hostname()
	if err != nil {
		return ""
	}
	name := strings.ToLower(strings.TrimSpace(raw))
	if i := strings.Index(name, "."); i > 0 {
		name = name[:i]
	}
	if name == "" || namesNoMachine(name) {
		return ""
	}
	return name
}

func (p *LlamaCppProvider) Models() []string {
	if p.registry != nil {
		return p.registry.ModelNames()
	}
	return p.models
}

// Available checks whether the llama.cpp server is reachable by hitting /health.
func (p *LlamaCppProvider) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Complete sends a non-streaming chat completion to llama.cpp and returns the
// response translated to HelixLLM's internal types.
func (p *LlamaCppProvider) Complete(ctx context.Context, req *types.InternalChatRequest) (*types.InternalChatResponse, error) {
	apiReq := p.toAPIRequest(req)
	apiReq.Stream = false

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("llamacpp: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/v1/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("llamacpp: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llamacpp: send request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llamacpp: unexpected status %d: %s",
			httpResp.StatusCode, readBackendError(httpResp.Body))
	}

	var apiResp api.ChatCompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("llamacpp: decode response: %w", err)
	}

	return p.fromAPIResponse(&apiResp), nil
}

// CompleteStream sends a streaming chat completion to llama.cpp and returns a
// channel of StreamChunks. The channel is closed when the stream ends.
func (p *LlamaCppProvider) CompleteStream(ctx context.Context, req *types.InternalChatRequest) (<-chan types.StreamChunk, error) {
	apiReq := p.toAPIRequest(req)
	apiReq.Stream = true

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("llamacpp: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/v1/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("llamacpp: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llamacpp: send request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		detail := readBackendError(httpResp.Body)
		httpResp.Body.Close()
		return nil, fmt.Errorf("llamacpp: unexpected status %d: %s",
			httpResp.StatusCode, detail)
	}

	ch := make(chan types.StreamChunk, 64)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()
		p.readSSEStream(ctx, httpResp, ch)
	}()

	return ch, nil
}

// readSSEStream reads SSE lines from an HTTP response and sends StreamChunks
// on the channel.
func (p *LlamaCppProvider) readSSEStream(ctx context.Context, resp *http.Response, ch chan<- types.StreamChunk) {
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return
		}

		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			sc := types.StreamChunk{
				Content: chunk.Choices[0].Delta.Content,
			}
			if chunk.Choices[0].FinishReason != nil {
				sc.FinishReason = *chunk.Choices[0].FinishReason
			}
			ch <- sc
		}
	}
}

// toAPIRequest converts an InternalChatRequest to an OpenAI ChatCompletionRequest
// suitable for llama.cpp's API.
func (p *LlamaCppProvider) toAPIRequest(req *types.InternalChatRequest) api.ChatCompletionRequest {
	messages := make([]api.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msg := api.ChatMessage{
			Role:       string(m.Role),
			Content:    m.Content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: api.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		messages[i] = msg
	}
	apiReq := api.ChatCompletionRequest{
		Model:      req.Model,
		Messages:   messages,
		ToolChoice: req.ToolChoice,
	}
	if req.MaxTokens > 0 {
		mt := req.MaxTokens
		apiReq.MaxTokens = &mt
	}
	if req.Temperature > 0 {
		temp := req.Temperature
		apiReq.Temperature = &temp
	}
	// Pass tools to llama.cpp (requires --jinja flag for tool support)
	for _, t := range req.Tools {
		apiReq.Tools = append(apiReq.Tools, api.Tool{
			Type: t.Type,
			Function: func() api.ToolFunction {
				if fn, ok := t.Function.(api.ToolFunction); ok {
					return fn
				}
				data, _ := json.Marshal(t.Function)
				var fn api.ToolFunction
				json.Unmarshal(data, &fn) //nolint:errcheck
				return fn
			}(),
		})
	}
	return apiReq
}

// fromAPIResponse converts an OpenAI ChatCompletionResponse to an InternalChatResponse.
func (p *LlamaCppProvider) fromAPIResponse(resp *api.ChatCompletionResponse) *types.InternalChatResponse {
	result := &types.InternalChatResponse{
		ID:       resp.ID,
		Model:    resp.Model,
		Provider: types.ProviderLocal,
	}
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		// Extract content as string. The Content field is interface{} in the API
		// types; for llama.cpp it is always a string.
		content := ""
		switch v := choice.Message.Content.(type) {
		case string:
			content = v
		}
		msg := types.InternalMessage{
			Role:    types.Role(choice.Message.Role),
			Content: content,
		}
		// Pass tool calls from llama.cpp response (native format)
		// Apply sanitizeToolArgs to fix common issues (offset=0, missing fields)
		for _, tc := range choice.Message.ToolCalls {
			args := tc.Function.Arguments
			// Sanitize native tool call arguments
			var argsMap map[string]interface{}
			if err := json.Unmarshal([]byte(args), &argsMap); err == nil {
				sanitizeToolArgs(tc.Function.Name, argsMap)
				if sanitized, err := json.Marshal(argsMap); err == nil {
					args = string(sanitized)
				}
			}
			msg.ToolCalls = append(msg.ToolCalls, types.InternalToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      tc.Function.Name,
					Arguments: args,
				},
			})
		}

		// Bridge: llama.cpp with Qwen models often returns tool calls
		// as XML or JSON in content instead of the tool_calls array.
		// Parse these and convert to proper tool_calls format.
		if len(msg.ToolCalls) == 0 && strings.Contains(content, "<function>") {
			if tc := parseXMLToolCall(content); tc != nil {
				msg.ToolCalls = append(msg.ToolCalls, *tc)
				msg.Content = ""
				result.FinishReason = "tool_calls"
			}
		}
		if len(msg.ToolCalls) == 0 && strings.Contains(content, `"name"`) && strings.Contains(content, `"arguments"`) {
			if tc := parseJSONToolCall(content); tc != nil {
				msg.ToolCalls = append(msg.ToolCalls, *tc)
				msg.Content = ""
				result.FinishReason = "tool_calls"
			}
		}

		result.Message = msg
		if result.FinishReason == "" {
			result.FinishReason = choice.FinishReason
		}
	}
	if resp.Usage != nil {
		result.Usage = types.InternalUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}
	return result
}
