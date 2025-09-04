// Test script to verify SSE streaming through the proxy
const sessionId = 'test-session-id';
const message = 'Hello, test message';
const params = new URLSearchParams({
  message,
  model: 'gpt-4o',
  max_context: '50',
});

const url = `http://localhost:3001/api/proxy/api/v2/chat/sessions/${sessionId}/stream?${params}`;

console.log('Testing SSE streaming through proxy...');
console.log('URL:', url);

fetch(url, {
  method: 'GET',
  headers: {
    'Accept': 'text/event-stream',
  },
})
  .then(async (response) => {
    console.log('Response status:', response.status);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));
    
    if (!response.ok) {
      const text = await response.text();
      console.error('Error response:', text);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      console.log('Received chunk:', chunk);
    }
  })
  .catch((error) => {
    console.error('Error:', error);
  });