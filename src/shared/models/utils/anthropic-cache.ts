import type { LanguageModelV3Message, LanguageModelV3Middleware, LanguageModelV3Prompt } from '@ai-sdk/provider'

const CACHE_CONTROL = { type: 'ephemeral' as const }

function withCache(msg: LanguageModelV3Message): LanguageModelV3Message {
  return {
    ...msg,
    providerOptions: {
      ...(msg.providerOptions ?? {}),
      anthropic: {
        ...((msg.providerOptions?.anthropic as Record<string, unknown> | undefined) ?? {}),
        cacheControl: CACHE_CONTROL,
      },
    },
  }
}

/**
 * Injects Anthropic cache_control breakpoints into a prompt (LanguageModelV3Message[]).
 *
 * Called for EVERY underlying API call, including intermediate tool-use steps
 * inside the agentic loop.
 *
 * Strategy (up to 2 breakpoints):
 * 1. System message — static prefix, highest reuse value.
 * 2. Second-to-last message (any role) — covers accumulated history from
 *    previous steps; as tool calls/results are appended each step, this
 *    breakpoint advances automatically, writing an incremental cache slice.
 *
 * Example for a 3-step tool-use call:
 *   Step 1 input: [sys✓, user]              → cache: sys
 *   Step 2 input: [sys✓, user✓, asst:tool]  → cache: sys + user
 *   Step 3 input: [sys✓, user, asst:tool✓, tool_result] → cache: sys + asst:tool
 */
function injectCacheControl(prompt: LanguageModelV3Prompt): LanguageModelV3Prompt {
  if (prompt.length === 0) return prompt

  const result = [...prompt]

  // 1. System message
  const sysIdx = result.findIndex((m) => m.role === 'system')
  if (sysIdx !== -1) {
    result[sysIdx] = withCache(result[sysIdx])
  }

  // 2. Second-to-last message (any role), only if it differs from system message
  if (result.length >= 3) {
    const targetIdx = result.length - 2
    if (targetIdx !== sysIdx) {
      result[targetIdx] = withCache(result[targetIdx])
    }
  }

  return result
}

/**
 * LanguageModelV3 middleware that injects Anthropic prompt caching
 * on every underlying API call (including multi-step agentic loops).
 */
export const anthropicCacheMiddleware: LanguageModelV3Middleware = {
  specificationVersion: 'v3',
  transformParams: async ({ params }) => ({
    ...params,
    prompt: injectCacheControl(params.prompt),
  }),
}
