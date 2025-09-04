"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { ChatMessage } from "./ChatMessage";
import { StreamingMessage } from "./StreamingMessage";
import { ChatInput } from "./ChatInput";
// import { ToolCallWidget } from "./ToolCallWidget";
// import { AgentCarousel } from "./AgentCarousel";
import { AuthPromptWidget } from "./AuthPromptWidget";
import { useChatSession } from "@/hooks/useChatSession";
import { useChatStream } from "@/hooks/useChatStream";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  ChatMessage as ChatMessageType,
  StreamChunk,
} from "@/lib/autogpt-server-api/chat";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface ToolCall {
  id: string;
  name: string;
  parameters: Record<string, any>;
  status: "calling" | "executing" | "completed" | "error";
  result?: string;
  error?: string;
  timestamp?: number;
}

interface AgentCarouselData {
  type: "agent_carousel";
  query: string;
  count: number;
  agents: any[];
}

interface ContentSegment {
  type:
    | "text"
    | "tool"
    | "carousel"
    | "credentials_setup"
    | "agent_setup"
    | "auth_required";
  content: any;
  id?: string;
}

interface ChatInterfaceProps {
  className?: string;
  systemPrompt?: string;
  sessionId?: string;
}

export function ChatInterface({
  className,
  systemPrompt,
  sessionId,
}: ChatInterfaceProps) {
  const { session, messages, isLoading, error, createSession, refreshSession } =
    useChatSession(sessionId);
  const { isStreaming, sendMessage, stopStreaming } = useChatStream();
  const { user } = useSupabase();

  const [localMessages, setLocalMessages] = useState<ChatMessageType[]>([]);
  const [streamingSegments, setStreamingSegments] = useState<ContentSegment[]>(
    [],
  );
  const [messageSegments, setMessageSegments] = useState<
    Map<number, ContentSegment[]>
  >(new Map());
  const [authPrompt, setAuthPrompt] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentMessageIndexRef = useRef<number>(0);
  const currentStreamingText = useRef<string>("");

  // Initialize session on mount
  useEffect(() => {
    // Only create new session if no sessionId provided and no session exists
    if (!sessionId && !session) {
      createSession(systemPrompt);
    }

    // Check for pending agent setup after authentication
    const pendingAgentSetup = localStorage.getItem("pending_agent_setup");
    if (pendingAgentSetup && session) {
      try {
        const agentInfo = JSON.parse(pendingAgentSetup);
        // Clear the pending setup
        localStorage.removeItem("pending_agent_setup");
        // Trigger agent setup
        handleSendMessage(
          `Set up the agent "${agentInfo.name}" (ID: ${agentInfo.graph_id})`,
        );
      } catch (_e) {
        console.error("Failed to parse pending agent setup:", e);
      }
    }
  }, [session]);

  // Clear auth prompt when user logs in and send confirmation
  useEffect(() => {
    if (user && authPrompt) {
      console.log(
        "User logged in, clearing auth prompt and sending confirmation",
      );
      setAuthPrompt(null);

      // Send hidden confirmation message to backend
      if (session) {
        // This message is not shown in the UI but tells the backend the user has logged in
        sendMessage(session.id, "I have logged in now", () => {
          // Silent handler - we don't show this message or its response
        })
          .then(() => {
            console.log("Sent login confirmation to backend");
          })
          .catch((err) => {
            console.error("Failed to send login confirmation:", err);
          });
      }
    }
  }, [user, authPrompt, session, sendMessage]);

  // Sync messages and parse segments from loaded messages
  useEffect(() => {
    console.log("🔄 Syncing messages, count:", messages.length);

    // Filter out system messages
    const userAndAssistantMessages = messages.filter(
      (m) => m.role !== "SYSTEM",
    );
    setLocalMessages(userAndAssistantMessages);

    // Parse segments from loaded messages
    const newSegments = new Map<number, ContentSegment[]>();
    userAndAssistantMessages.forEach((message, index) => {
      console.log(`📨 Message ${index}:`, message);

      if (message.role === "ASSISTANT") {
        const segments: ContentSegment[] = [];
        let hasToolOrCarousel = false;

        // Check if message has tool_calls field
        if (message.tool_calls && message.tool_calls.length > 0) {
          console.log(`  🔧 Found tool_calls:`, message.tool_calls);
          hasToolOrCarousel = true;

          // Extract text content before tool calls
          if (message.content) {
            segments.push({ type: "text", content: message.content });
          }

          // Extract tool calls
          message.tool_calls.forEach((toolCall: any) => {
            segments.push({
              type: "tool",
              content: {
                id:
                  toolCall.id ||
                  `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                name: toolCall.name || toolCall.function?.name || "Unknown",
                parameters:
                  toolCall.parameters || toolCall.function?.arguments || {},
                status: "completed",
                result: toolCall.result || toolCall.output || "",
              },
              id: toolCall.id,
            });
          });
        }

        // Check if content has embedded JSON carousel data
        if (!hasToolOrCarousel && message.content) {
          // First check if the entire content is a JSON carousel
          try {
            const trimmedContent = message.content.trim();
            if (
              trimmedContent.startsWith("{") &&
              trimmedContent.includes('"type"') &&
              trimmedContent.includes('"agent_carousel"')
            ) {
              const carouselData = JSON.parse(trimmedContent);
              if (
                carouselData.type === "agent_carousel" &&
                carouselData.agents
              ) {
                console.log(
                  `  🎠 Found carousel JSON (entire content):`,
                  carouselData,
                );
                hasToolOrCarousel = true;
                segments.push({
                  type: "carousel",
                  content: carouselData,
                  id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                });
              }
            }
          } catch (_e) {
            // Not a valid JSON carousel, try pattern matching
            try {
              const jsonMatch = message.content.match(
                /\{[\s\S]*?"type"\s*:\s*"agent_carousel"[\s\S]*?\}/,
              );
              if (jsonMatch) {
                const carouselData = JSON.parse(jsonMatch[0]);
                if (
                  carouselData.type === "agent_carousel" &&
                  carouselData.agents
                ) {
                  console.log(
                    `  🎠 Found carousel JSON data (embedded):`,
                    carouselData,
                  );
                  hasToolOrCarousel = true;
                  segments.push({
                    type: "carousel",
                    content: carouselData,
                    id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                  });

                  // Remove the JSON from the text content
                  const textWithoutJson = message.content
                    .replace(jsonMatch[0], "")
                    .trim();
                  if (textWithoutJson) {
                    segments.unshift({
                      type: "text",
                      content: textWithoutJson,
                    });
                  }
                }
              }
            } catch (_e2) {
              console.log(
                `  📄 Not JSON carousel, checking for HTML content...`,
              );
            }
          }

          // If no JSON carousel found, check for HTML content
          if (
            !hasToolOrCarousel &&
            (message.content.includes("tool-call-container") ||
              message.content.includes("agent-carousel") ||
              message.content.includes("tool-result"))
          ) {
            console.log(`  📄 Found HTML content, parsing...`);

            // Parse HTML content using the same logic as streaming
            const parsed = parseMessageContent(message.content);
            if (parsed && parsed.length > 0) {
              hasToolOrCarousel = true;
              segments.push(...parsed);
            }
          }
        }

        if (segments.length > 0) {
          console.log(`  ✅ Setting segments for message ${index}:`, segments);
          newSegments.set(index, segments);
        } else if (message.content && !hasToolOrCarousel) {
          // Regular text message without tools/carousel
          console.log(`  📝 Regular text message ${index}`);
        }
      }
    });

    console.log("📊 Final message segments map:", newSegments);
    setMessageSegments(newSegments);
  }, [messages, user, session]);

  // Helper function to parse message content for embedded HTML
  const parseMessageContent = (content: string): ContentSegment[] => {
    const segments: ContentSegment[] = [];
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, "text/html");

    let textContent = "";
    let foundStructuredContent = false;

    // Check for auth required responses (but only set authPrompt if user is not logged in)
    // Support both old and new format
    if (
      !user &&
      (content.includes('"status": "auth_required"') ||
        content.includes('"type": "need_login"'))
    ) {
      try {
        const jsonMatch = content.match(
          /\{[\s\S]*?("status"\s*:\s*"auth_required"|"type"\s*:\s*"need_login")[\s\S]*?\}/,
        );
        if (jsonMatch) {
          const authData = JSON.parse(jsonMatch[0]);
          if (
            authData.status === "auth_required" ||
            authData.type === "need_login"
          ) {
            setAuthPrompt({
              message: authData.message,
              sessionId: authData.session_id || session?.id,
              agentInfo: authData.agent_info,
            });
            // Don't add this to segments since we're showing the auth widget
            return [];
          }
        }
      } catch (_e) {
        console.error("Failed to parse auth response:", e);
      }
    }

    // Look for tool call containers
    const toolContainers = doc.querySelectorAll(".tool-call-container");
    toolContainers.forEach((container) => {
      const toolHeader = container.querySelector(".tool-header");
      const toolResult = container.querySelector(".tool-result");
      if (toolHeader) {
        const toolNameMatch =
          toolHeader.textContent?.match(/Calling Tool: (\w+)/);
        if (toolNameMatch) {
          foundStructuredContent = true;

          // Extract parameters from tool-body if present
          let parameters = {};
          const toolBody = container.querySelector(".tool-body pre");
          if (toolBody?.textContent) {
            try {
              parameters = JSON.parse(toolBody.textContent);
            } catch (_e) {
              console.error("Failed to parse tool parameters:", e);
            }
          }

          // Check if the tool result is JSON (carousel or auth)
          let skipToolSegment = false;
          if (toolResult) {
            const resultText = toolResult.textContent?.trim() || "";
            if (resultText.startsWith("{") && resultText.endsWith("}")) {
              try {
                const resultJson = JSON.parse(resultText);
                if (resultJson.type === "agent_carousel") {
                  segments.push({
                    type: "carousel",
                    content: resultJson,
                    id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                  });
                  skipToolSegment = true;
                } else if (
                  resultJson.type === "auth_required" ||
                  resultJson.type === "need_login"
                ) {
                  segments.push({
                    type: "auth_required",
                    content: resultJson,
                    id: `auth-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                  });
                  skipToolSegment = true;
                }
              } catch (_e) {
                // Not JSON or failed to parse, treat as regular result
              }
            }
          }

          if (!skipToolSegment) {
            segments.push({
              type: "tool",
              content: {
                id: `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                name: toolNameMatch[1],
                parameters: parameters,
                status: "completed",
                result: toolResult?.textContent || "",
              },
            });
          }
        }
      }
    });

    // Look for carousel data in various places
    const carousels = doc.querySelectorAll(
      ".agent-carousel, .tool-result-carousel",
    );
    carousels.forEach((carousel) => {
      try {
        const data = JSON.parse(carousel.textContent || "{}");
        if (data.type === "agent_carousel") {
          foundStructuredContent = true;
          segments.push({
            type: "carousel",
            content: data,
            id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          });
        }
      } catch (_e) {
        console.error("Failed to parse carousel:", e);
      }
    });

    // Also check for standalone Tool Response divs that might contain JSON
    const allDivs = doc.querySelectorAll("div");
    allDivs.forEach((div) => {
      // Skip if already in tool container
      if (div.closest(".tool-call-container")) return;

      const divText = div.textContent?.trim() || "";
      // Check if this div starts with "Tool Response:" and contains JSON
      if (divText.startsWith("Tool Response:")) {
        const jsonPart = divText.substring("Tool Response:".length).trim();
        if (jsonPart.startsWith("{") && jsonPart.endsWith("}")) {
          try {
            const data = JSON.parse(jsonPart);
            if (data.type === "agent_carousel") {
              foundStructuredContent = true;
              // Check if we already have this carousel
              const hasCarousel = segments.some(
                (s) =>
                  s.type === "carousel" &&
                  JSON.stringify(s.content) === JSON.stringify(data),
              );
              if (!hasCarousel) {
                segments.push({
                  type: "carousel",
                  content: data,
                  id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                });
              }
            }
          } catch (_e) {
            // Not valid JSON, skip
          }
        }
      }
    });

    // Extract plain text (excluding tool/carousel content)
    const walker = document.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        const parent = node.parentElement;
        // Skip tool containers and carousels
        if (
          parent?.closest(
            ".tool-call-container, .agent-carousel, .tool-result-carousel",
          )
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        // Skip text that looks like "Tool Response: {JSON}"
        const text = node.textContent?.trim() || "";
        if (text.startsWith("Tool Response:") && text.includes("{")) {
          return NodeFilter.FILTER_REJECT;
        }
        // Skip pure JSON nodes
        if (
          text.startsWith("{") &&
          text.includes('"type"') &&
          text.includes('"agent_carousel"')
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.textContent?.trim()) {
        textContent += node.textContent;
      }
    }

    // Add text content if present
    if (textContent.trim()) {
      segments.unshift({ type: "text", content: textContent.trim() });
    } else if (!foundStructuredContent && content) {
      // Fallback: if no structured content found and no text extracted, use original content
      segments.push({ type: "text", content: content });
    }

    return segments;
  };

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [localMessages, streamingSegments]);

  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!session) {
        await createSession(systemPrompt);
        return;
      }

      // Add user message immediately
      const userMessage: ChatMessageType = {
        content: message,
        role: "USER",
        created_at: new Date().toISOString(),
      };
      setLocalMessages((prev) => [...prev, userMessage]);
      // Set to the index where the assistant message will be
      currentMessageIndexRef.current = localMessages.length + 1;

      // Clear streaming content
      setStreamingSegments([]);
      currentStreamingText.current = "";

      // Start streaming
      const collectedSegments: ContentSegment[] = [];
      let hasContent = false;

      await sendMessage(session.id, message, (chunk: StreamChunk) => {
        // Handle new SSE message types from backend
        if (chunk.type === "text" || chunk.type === "text_chunk") {
          // Accumulate text in current segment
          currentStreamingText.current += chunk.content;
          hasContent = true;

          // Update streaming segments with current text
          setStreamingSegments((prev) => {
            const newSegments = [
              ...prev.filter(
                (s) => s.type !== "text" || s.id !== "current-text",
              ),
            ];
            if (currentStreamingText.current) {
              newSegments.push({
                type: "text",
                content: currentStreamingText.current,
                id: "current-text",
              });
            }
            return newSegments;
          });
        } else if (chunk.type === "tool_call") {
          // Handle tool call notification
          const toolCall: ToolCall = {
            id: chunk.tool_id || `tool-${Date.now()}`,
            name: chunk.tool_name || "unknown",
            parameters: chunk.arguments || {},
            status: "calling",
            timestamp: Date.now(),
          };

          // Add any accumulated text as a segment first
          if (currentStreamingText.current) {
            collectedSegments.push({
              type: "text",
              content: currentStreamingText.current,
            });
            currentStreamingText.current = "";
          }

          // Add tool call as a segment
          const toolSegment: ContentSegment = {
            type: "tool",
            content: toolCall,
            id: toolCall.id,
          };
          collectedSegments.push(toolSegment);
          setStreamingSegments([...collectedSegments]);
        } else if (chunk.type === "tool_response") {
          // Handle tool response
          hasContent = true;

          // Check if result is an agent carousel
          if (chunk.result && typeof chunk.result === "object") {
            if (chunk.result.type === "agent_carousel") {
              // Add carousel as a segment
              const carouselSegment: ContentSegment = {
                type: "carousel",
                content: chunk.result,
                id: `carousel-${Date.now()}`,
              };
              collectedSegments.push(carouselSegment);
              setStreamingSegments([...collectedSegments]);
            } else if (
              chunk.result.type === "need_login" ||
              chunk.result.type === "need_credentials"
            ) {
              // Handle auth/credentials needed
              if (chunk.result.type === "need_login" && !user) {
                setAuthPrompt({
                  message: chunk.result.message || chunk.message,
                  sessionId:
                    chunk.result.session_id || chunk.session_id || session?.id,
                  agentInfo: chunk.result.agent_info || chunk.agent_info,
                });
              } else if (chunk.result.type === "need_credentials") {
                // Add credentials setup segment
                const credSegment: ContentSegment = {
                  type: "credentials_setup",
                  content: chunk.result,
                  id: `creds-${Date.now()}`,
                };
                collectedSegments.push(credSegment);
                setStreamingSegments([...collectedSegments]);
              }
            } else if (chunk.result.type === "got_agent_details") {
              // Just update the tool result, no special UI
              setStreamingSegments((prev) => {
                const updated = [...prev];
                for (let i = updated.length - 1; i >= 0; i--) {
                  if (
                    updated[i].type === "tool" &&
                    updated[i].id === chunk.tool_id
                  ) {
                    updated[i] = {
                      ...updated[i],
                      content: {
                        ...updated[i].content,
                        status: "completed",
                        result: JSON.stringify(chunk.result.details, null, 2),
                      },
                    };
                    break;
                  }
                }
                return updated;
              });
            } else if (
              chunk.result.status === "success" &&
              chunk.result.trigger_type
            ) {
              // Agent setup successful
              const setupSegment: ContentSegment = {
                type: "agent_setup",
                content: chunk.result,
                id: `setup-${Date.now()}`,
              };
              collectedSegments.push(setupSegment);
              setStreamingSegments([...collectedSegments]);
            } else {
              // Update tool with generic result
              setStreamingSegments((prev) => {
                const updated = [...prev];
                for (let i = updated.length - 1; i >= 0; i--) {
                  if (
                    updated[i].type === "tool" &&
                    updated[i].id === chunk.tool_id
                  ) {
                    updated[i] = {
                      ...updated[i],
                      content: {
                        ...updated[i].content,
                        status: "completed",
                        result:
                          typeof chunk.result === "string"
                            ? chunk.result
                            : JSON.stringify(chunk.result, null, 2),
                      },
                    };
                    break;
                  }
                }
                return updated;
              });
            }
          } else {
            // Update tool with text result
            setStreamingSegments((prev) => {
              const updated = [...prev];
              for (let i = updated.length - 1; i >= 0; i--) {
                if (
                  updated[i].type === "tool" &&
                  updated[i].id === chunk.tool_id
                ) {
                  updated[i] = {
                    ...updated[i],
                    content: {
                      ...updated[i].content,
                      status: "completed",
                      result: chunk.result || "",
                    },
                  };
                  break;
                }
              }
              return updated;
            });
          }
        } else if (chunk.type === "login_needed") {
          // Handle login needed response
          if (!user) {
            setAuthPrompt({
              message: chunk.message,
              sessionId: chunk.session_id || session?.id,
              agentInfo: chunk.agent_info,
            });
          }
        } else if (chunk.type === "stream_end") {
          // Stream has ended, finalize any remaining content
          console.log("Stream ended", chunk.summary);
        } else if (chunk.type === "html") {
          // Parse HTML content for tool calls and special widgets
          const result = handleHtmlContent(chunk.content);
          if (result?.type === "tool_call") {
            const toolCall = result.data as ToolCall;

            // Add any accumulated text as a segment first
            if (currentStreamingText.current) {
              collectedSegments.push({
                type: "text",
                content: currentStreamingText.current,
              });
              currentStreamingText.current = "";
            }

            // Add tool call as a segment
            const toolSegment: ContentSegment = {
              type: "tool",
              content: toolCall,
              id: toolCall.id,
            };
            collectedSegments.push(toolSegment);

            setStreamingSegments([...collectedSegments]);
          } else if (result?.type === "tool_executing") {
            // Update last tool segment to executing status
            setStreamingSegments((prev) => {
              const updated = [...prev];
              for (let i = updated.length - 1; i >= 0; i--) {
                if (updated[i].type === "tool") {
                  updated[i] = {
                    ...updated[i],
                    content: { ...updated[i].content, status: "executing" },
                  };
                  break;
                }
              }
              return updated;
            });
          } else if (result?.type === "tool_result") {
            // Update last tool segment with result
            setStreamingSegments((prev) => {
              const updated = [...prev];
              for (let i = updated.length - 1; i >= 0; i--) {
                if (updated[i].type === "tool") {
                  updated[i] = {
                    ...updated[i],
                    content: {
                      ...updated[i].content,
                      status: "completed",
                      result: result.data.result,
                    },
                  };
                  break;
                }
              }
              return updated;
            });
          } else if (result?.type === "agent_carousel") {
            hasContent = true;
            const carouselData = result.data as AgentCarouselData;

            // Add any accumulated text as a segment first
            if (currentStreamingText.current) {
              collectedSegments.push({
                type: "text",
                content: currentStreamingText.current,
              });
              currentStreamingText.current = "";
            }

            // Add carousel as a segment
            const carouselSegment: ContentSegment = {
              type: "carousel",
              content: carouselData,
              id: `carousel-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            };
            collectedSegments.push(carouselSegment);

            setStreamingSegments([...collectedSegments]);
          }
        } else if (chunk.type === "error") {
          console.error("Stream error:", chunk.content);
        }
      });

      // Add final assistant message
      if (hasContent || collectedSegments.length > 0) {
        // Add any remaining text
        if (currentStreamingText.current) {
          collectedSegments.push({
            type: "text",
            content: currentStreamingText.current,
          });
        }

        // Extract text content for the message
        const textContent = collectedSegments
          .filter((s) => s.type === "text")
          .map((s) => s.content)
          .join("");

        const assistantMessage: ChatMessageType = {
          content: textContent,
          role: "ASSISTANT",
          created_at: new Date().toISOString(),
        };
        const messageIndex = localMessages.length + 1;
        setLocalMessages((prev) => [...prev, assistantMessage]);

        // Store segments for this message
        if (collectedSegments.length > 0) {
          setMessageSegments((prev) => {
            const newMap = new Map(prev);
            newMap.set(messageIndex, collectedSegments);
            return newMap;
          });
        }

        setStreamingSegments([]);
        currentStreamingText.current = "";

        // Refresh the session to get persisted messages from backend
        await refreshSession();
      }
    },
    [
      session,
      sendMessage,
      createSession,
      systemPrompt,
      refreshSession,
      localMessages,
    ],
  );

  const handleHtmlContent = (
    html: string,
  ): { type: string; data: any } | null => {
    // Parse the HTML to detect tool calls and extract data
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");

    // Check for tool call containers
    const toolCallContainer = doc.querySelector(".tool-call-container");
    if (toolCallContainer) {
      const toolHeader = toolCallContainer.querySelector(".tool-header");
      const toolBody = toolCallContainer.querySelector(".tool-body");

      if (toolHeader) {
        const toolNameMatch =
          toolHeader.textContent?.match(/Calling Tool: (\w+)/);
        if (toolNameMatch) {
          const toolName = toolNameMatch[1];
          const preElement = toolBody?.querySelector("pre");
          let parameters = {};

          try {
            if (preElement?.textContent) {
              parameters = JSON.parse(preElement.textContent);
            }
          } catch (_e) {
            console.error("Failed to parse tool parameters:", e);
          }

          const toolCall: ToolCall = {
            id: `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            name: toolName,
            parameters,
            status: "calling",
            timestamp: Date.now(),
          };

          return { type: "tool_call", data: toolCall };
        }
      }
    }

    // Check for executing indicator
    const executingDiv = doc.querySelector(".tool-executing");
    if (executingDiv) {
      // Return executing status to be handled by streaming handler
      return { type: "tool_executing", data: {} };
    }

    // Check for carousel result
    const carouselDiv = doc.querySelector(".tool-result-carousel");
    if (
      carouselDiv &&
      carouselDiv.getAttribute("data-type") === "agent-carousel"
    ) {
      console.log("Found carousel div with content:", carouselDiv.textContent);
      try {
        const carouselData = JSON.parse(carouselDiv.textContent || "{}");
        if (carouselData.type === "agent_carousel") {
          console.log("Parsed carousel data successfully:", carouselData);
          return { type: "agent_carousel", data: carouselData };
        }
      } catch (_e) {
        console.error("Failed to parse carousel data:", e);
      }
    }

    // Check for regular tool result
    const resultDiv = doc.querySelector(".tool-result");
    if (resultDiv) {
      const resultContent =
        resultDiv.querySelector("div:last-child")?.textContent || "";

      // Check for auth_required response (both old and new format)
      if (
        resultContent.includes('"status": "auth_required"') ||
        resultContent.includes('"type": "need_login"')
      ) {
        try {
          const jsonMatch = resultContent.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const authData = JSON.parse(jsonMatch[0]);
            if (
              authData.status === "auth_required" ||
              authData.type === "need_login"
            ) {
              setAuthPrompt({
                message: authData.message,
                sessionId: authData.session_id || session?.id,
                agentInfo: authData.agent_info,
              });
            }
          }
        } catch (_e) {
          console.error("Failed to parse auth response:", e);
        }
      }

      // Return result to be handled by streaming handler
      return { type: "tool_result", data: { result: resultContent } };
    }

    return null;
  };

  const handleSelectAgent = (agent: any) => {
    // Send a message to set up the selected agent
    handleSendMessage(
      `I want to set up the agent "${agent.name}" (ID: ${agent.id})`,
    );
  };

  const handleGetAgentDetails = (agent: any) => {
    // Send a message to get more details about the agent
    handleSendMessage(
      `Tell me more about the agent "${agent.name}" (ID: ${agent.id})`,
    );
  };

  if (isLoading && !session) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-violet-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-red-500">Error: {error.message}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 rounded-lg bg-violet-600 px-4 py-2 text-white hover:bg-violet-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex h-screen flex-col bg-neutral-50 dark:bg-neutral-950",
        className,
      )}
    >
      {/* Header */}
      <div className="border-b border-neutral-200 bg-white px-4 py-3 dark:border-neutral-700 dark:bg-neutral-900">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            AI Agent Discovery Assistant
          </h1>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Chat with me to find and set up the perfect AI agent for your needs
          </p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl py-4">
          {localMessages.length === 0 && streamingSegments.length === 0 && (
            <div className="px-4 py-8 text-center">
              <p className="text-neutral-600 dark:text-neutral-400">
                👋 Hello! I&apos;m here to help you discover and set up AI
                agents.
              </p>
              <p className="mt-2 text-sm text-neutral-500 dark:text-neutral-500">
                Try asking: &ldquo;I need help with content creation&rdquo; or
                &ldquo;Show me automation agents&rdquo;
              </p>
            </div>
          )}

          {localMessages.map((message, index) => {
            const segments = messageSegments.get(index);
            if (segments && segments.length > 0) {
              // Use StreamingMessage for messages with segments
              return (
                <StreamingMessage
                  key={index}
                  role={message.role}
                  segments={segments}
                  onSelectAgent={handleSelectAgent}
                  onGetAgentDetails={handleGetAgentDetails}
                />
              );
            } else {
              // Use regular ChatMessage for messages without segments
              return <ChatMessage key={index} message={message} />;
            }
          })}

          {/* Streaming message with inline segments */}
          {streamingSegments.length > 0 && (
            <StreamingMessage
              role="ASSISTANT"
              segments={streamingSegments}
              onSelectAgent={handleSelectAgent}
              onGetAgentDetails={handleGetAgentDetails}
            />
          )}

          {authPrompt && (
            <div className="px-4">
              <AuthPromptWidget
                message={authPrompt.message}
                sessionId={authPrompt.sessionId}
                agentInfo={authPrompt.agentInfo}
              />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <ChatInput
        onSendMessage={handleSendMessage}
        onStopStreaming={stopStreaming}
        isStreaming={isStreaming}
        disabled={!session}
      />
    </div>
  );
}
