/**
 * CoPilot Bot — Multi-platform bot using Vercel Chat SDK.
 *
 * Handles:
 * - Account linking (prompts unlinked users to link)
 * - Message routing to CoPilot API
 * - Streaming responses back to the user
 */

import { Chat, Message } from "chat";
import type { Adapter, StateAdapter, Thread } from "chat";
import { PlatformAPI } from "./platform-api.js";
import type { Config } from "./config.js";

// Thread state persisted across messages
export interface BotThreadState {
  /** Linked AutoGPT user ID */
  userId?: string;
  /** CoPilot chat session ID for this thread */
  sessionId?: string;
  /** Pending link token (if user hasn't linked yet) */
  pendingLinkToken?: string;
}

type BotThread = Thread<BotThreadState>;

export async function createBot(config: Config, stateAdapter: StateAdapter) {
  const api = new PlatformAPI(config.autogptApiUrl);

  // Build adapters based on config
  const adapters: Record<string, Adapter> = {};

  if (config.discord) {
    const { createDiscordAdapter } = await import("@chat-adapter/discord");
    adapters.discord = createDiscordAdapter();
  }

  if (config.telegram) {
    const { createTelegramAdapter } = await import("@chat-adapter/telegram");
    adapters.telegram = createTelegramAdapter();
  }

  if (config.slack) {
    const { createSlackAdapter } = await import("@chat-adapter/slack");
    adapters.slack = createSlackAdapter();
  }

  if (Object.keys(adapters).length === 0) {
    throw new Error(
      "No adapters enabled. Set at least one of: " +
        "DISCORD_BOT_TOKEN, TELEGRAM_BOT_TOKEN, SLACK_BOT_TOKEN"
    );
  }

  const bot = new Chat<typeof adapters, BotThreadState>({
    userName: "copilot",
    adapters,
    state: stateAdapter,
    streamingUpdateIntervalMs: 500,
    fallbackStreamingPlaceholderText: "Thinking...",
  });

  // ── New mention (first message in a thread) ──────────────────────

  bot.onNewMention(async (thread, message) => {
    const adapterName = getAdapterName(thread);
    const platformUserId = message.author.userId;

    console.log(
      `[bot] New mention from ${adapterName}:${platformUserId} in ${thread.id}`
    );

    // Check if user is linked
    const resolved = await api.resolve(adapterName, platformUserId);

    if (!resolved.linked) {
      await handleUnlinkedUser(thread, message, adapterName, api);
      return;
    }

    // User is linked — subscribe and handle the message
    await thread.subscribe();
    await thread.setState({ userId: resolved.user_id });

    await handleCoPilotMessage(thread, message.text, resolved.user_id!, api);
  });

  // ── Subscribed messages (follow-ups in a thread) ─────────────────

  bot.onSubscribedMessage(async (thread, message) => {
    const state = await thread.state;

    if (!state?.userId) {
      // Somehow lost state — re-resolve
      const adapterName = getAdapterName(thread);
      const resolved = await api.resolve(adapterName, message.author.userId);

      if (!resolved.linked) {
        await handleUnlinkedUser(thread, message, adapterName, api);
        return;
      }

      await thread.setState({ userId: resolved.user_id });
      await handleCoPilotMessage(
        thread,
        message.text,
        resolved.user_id!,
        api
      );
      return;
    }

    await handleCoPilotMessage(thread, message.text, state.userId, api);
  });

  return bot;
}

// ── Helpers ──────────────────────────────────────────────────────────

/**
 * Get the adapter/platform name from a thread.
 * Thread ID format is "adapter:channel:thread".
 */
function getAdapterName(thread: BotThread): string {
  const parts = thread.id.split(":");
  return parts[0] ?? "unknown";
}

/**
 * Handle an unlinked user — create a link token and send them a prompt.
 */
async function handleUnlinkedUser(
  thread: BotThread,
  message: Message,
  platform: string,
  api: PlatformAPI
) {
  console.log(
    `[bot] Unlinked user ${platform}:${message.author.userId}, sending link prompt`
  );

  try {
    const linkResult = await api.createLinkToken({
      platform,
      platformUserId: message.author.userId,
      platformUsername: message.author.fullName ?? message.author.userName,
    });

    await thread.post(
      `👋 To use CoPilot, link your AutoGPT account first.\n\n` +
        `🔗 **Link your account:** ${linkResult.link_url}\n\n` +
        `_This link expires in 30 minutes._`
    );

    // Store the pending token so we could poll later if needed
    await thread.setState({ pendingLinkToken: linkResult.token });
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    if (errMsg.includes("409")) {
      // Already linked (race condition) — retry resolve
      const resolved = await api.resolve(platform, message.author.userId);
      if (resolved.linked) {
        await thread.subscribe();
        await thread.setState({ userId: resolved.user_id });
        await handleCoPilotMessage(
          thread,
          message.text,
          resolved.user_id!,
          api
        );
        return;
      }
    }

    console.error("[bot] Failed to create link token:", err);
    await thread.post(
      "Sorry, I couldn't set up account linking right now. Please try again later."
    );
  }
}

/**
 * Forward a message to CoPilot and stream the response back.
 */
async function handleCoPilotMessage(
  thread: BotThread,
  text: string,
  userId: string,
  api: PlatformAPI
) {
  const state = await thread.state;
  let sessionId = state?.sessionId;

  console.log(
    `[bot] Message from user ${userId.slice(-8)}: ${text.slice(0, 100)}`
  );

  await thread.startTyping();

  try {
    // Create a session if we don't have one
    if (!sessionId) {
      sessionId = await api.createChatSession(userId);
      await thread.setState({ ...state, sessionId });
      console.log(`[bot] Created session ${sessionId} for user ${userId.slice(-8)}`);
    }

    // Stream CoPilot response — collect chunks, then post.
    // We collect first because thread.post() with an empty stream
    // causes Discord "Cannot send an empty message" errors.
    const stream = api.streamChat(userId, text, sessionId);
    let response = "";
    for await (const chunk of stream) {
      response += chunk;
    }

    if (response.trim()) {
      await thread.post(response);
    } else {
      await thread.post(
        "I processed your message but didn't generate a response. Please try again."
      );
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    console.error(`[bot] CoPilot error for user ${userId.slice(-8)}:`, errMsg);
    await thread.post(
      "Sorry, I ran into an issue processing your message. Please try again."
    );
  }
}
