/**
 * What ws_api.py sends when Redis doesn't confirm a subscribe to a graph's
 * runs in time (`WSRequestFailed`). The realtime journey injects this exact
 * frame, and the client's tests feed it to the real client.
 */
export function subscribeTimedOutFrame(graphID: string) {
  return JSON.stringify({
    method: "error",
    data: null,
    success: false,
    channel: null,
    error: "Could not handle 'subscribe_graph_executions'; retry",
    request: {
      method: "subscribe_graph_executions",
      data: { graph_id: graphID },
    },
    retryable: true,
  });
}
