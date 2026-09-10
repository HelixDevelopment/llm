package brain

import (
	"errors"
	"fmt"
)

// The condition "this deployment serves nothing by that name" needs a name of
// its own, because it is the one unservable answer that is NOT temporary.
//
// # Why it is not an availability condition
//
// internal/fallback already distinguishes two kinds of "cannot serve this":
// ErrProvidersExhausted and ErrPinnedModelUnavailable are both availability
// conditions and reach the client as 503 with "retry with backoff", while
// ErrRetiredIdentifier is permanent and reaches it as 404. A model id no
// provider in this deployment offers belongs with the second: no amount of
// backoff makes `gpt-4o` appear on a llama.cpp-only box, and a correct client
// obeying a 503 retries a name that can never resolve, forever. That is the
// same reasoning completer_status.go records for the retired identifier, and
// this is its sibling for a name that was never ours to begin with.
//
// # Why a sentinel rather than a message
//
// The HTTP boundary must classify the condition without matching on message
// text — the mistake fallback.ErrProvidersExhausted was introduced to end.
// [IsModelNotFound] is that predicate.
//
// # Why the error carries the model name
//
// A 404 that does not say WHICH model was refused leaves a caller with a
// vocabulary problem and no vocabulary: it has one model id it believed in
// and no way to learn that this was the rejected one. [NotFoundModelName]
// hands the name back so the response can name it. The name is the caller's
// own input echoed back, so relaying it discloses nothing about the
// deployment's topology — unlike the upstream error text that
// upstream_error.go redacts.
var ErrModelNotFound = errors.New("no provider serves the requested model")

// ModelNotFoundError reports that a request named a model this deployment
// does not serve, and carries the name it named.
type ModelNotFoundError struct {
	// Model is the identifier exactly as the caller sent it.
	Model string
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("%s: %q", ErrModelNotFound.Error(), e.Model)
}

// Unwrap ties the typed error to the sentinel so errors.Is works for callers
// that only need the classification, while errors.As still yields the name.
func (e *ModelNotFoundError) Unwrap() error { return ErrModelNotFound }

// NewModelNotFound builds the error for a requested model name. It is
// exported because internal/fallback raises the same condition from the
// chain's own pin, and two independently-worded copies of one condition is
// exactly how the sentinel would stop meaning one thing.
func NewModelNotFound(model string) error {
	return &ModelNotFoundError{Model: model}
}

// IsModelNotFound reports whether err (or anything it wraps) is the
// unknown-model condition. Prefer it over errors.Is at call sites so the
// sentinel stays an implementation detail of this package.
func IsModelNotFound(err error) bool {
	return errors.Is(err, ErrModelNotFound)
}

// NotFoundModelName returns the model name an unknown-model error was raised
// for. ok is false for any other error, so a caller can use it as both the
// classification and the extraction in one step.
func NotFoundModelName(err error) (string, bool) {
	var e *ModelNotFoundError
	if errors.As(err, &e) {
		return e.Model, true
	}
	return "", false
}
