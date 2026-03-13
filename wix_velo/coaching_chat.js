/**
 * ABN Consulting AI Co-Navigator — Wix Velo Page Code
 *
 * SETUP INSTRUCTIONS:
 * 1. In your Wix site, create a page called "coaching-session"
 * 2. Add these elements to the page:
 *    - #chatHistory    → Repeater (displays messages)
 *    - #messageInput   → Text Input
 *    - #sendButton     → Button ("Send")
 *    - #startButton    → Button ("Start Session")
 *    - #endButton      → Button ("End Session", initially hidden)
 *    - #statusBadge    → Text element (shows alert level)
 *    - #loadingSpinner → Image or animation (hidden by default)
 * 3. Store your API key in Wix Secrets Manager as "COACHING_API_KEY"
 *    (Wix Editor → Settings → Secrets Manager)
 * 4. API_BASE is already set to the deployed Railway server URL
 * 5. For "Sign in with Google", link a button to:
 *    https://auto-gpt-production.up.railway.app/auth/google/url?redirect_to=https://www.ben-nesher.com/coaching-chat
 *    After login, the page receives ?user_id=...&name=...&email=... in the URL.
 *
 * IMPORTANT: Never expose COACHING_API_KEY directly in client-side code.
 * Use a Wix Backend (web module) to make API calls securely.
 */

// ---------------------------------------------------------------
// wix_velo/coaching-backend.jsw  (Wix Backend Web Module)
// Place this code in a NEW file: backend/coaching-backend.jsw
// ---------------------------------------------------------------
/**
 * BACKEND MODULE (backend/coaching-backend.jsw)
 * This runs server-side in Wix, keeping the API key secure.
 *
 * import { getSecret } from 'wix-secrets-backend';
 * import { fetch } from 'wix-fetch';
 *
 * const API_BASE = 'https://auto-gpt-production.up.railway.app';
 *
 * async function getApiKey() {
 *   return await getSecret('COACHING_API_KEY');
 * }
 *
 * export async function startCoachingSession(clientId, clientName) {
 *   const apiKey = await getApiKey();
 *   const res = await fetch(`${API_BASE}/coaching/session/start`, {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
 *     body: JSON.stringify({ client_id: clientId, client_name: clientName })
 *   });
 *   return res.json();
 * }
 *
 * export async function sendCoachingMessage(sessionId, message) {
 *   const apiKey = await getApiKey();
 *   const res = await fetch(`${API_BASE}/coaching/session/${sessionId}/message`, {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
 *     body: JSON.stringify({ message })
 *   });
 *   return res.json();
 * }
 *
 * export async function endCoachingSession(sessionId) {
 *   const apiKey = await getApiKey();
 *   const res = await fetch(`${API_BASE}/coaching/session/${sessionId}/end`, {
 *     method: 'POST',
 *     headers: { 'X-API-Key': apiKey }
 *   });
 *   return res.json();
 * }
 */

// ---------------------------------------------------------------
// PAGE CODE (paste this into the Wix Page Code editor)
// ---------------------------------------------------------------

import wixUsers from 'wix-users';
import wixStorage from 'wix-storage';
import {
  startCoachingSession,
  sendCoachingMessage,
  endCoachingSession
} from 'backend/coaching-backend';

// Chat message history displayed in the repeater
let chatMessages = [];
let currentSessionId = null;

$w.onReady(function () {
  // Restore session if user refreshes the page
  currentSessionId = wixStorage.session.getItem('coaching_session_id');

  $w('#endButton').hide();
  $w('#sendButton').disable();
  $w('#messageInput').disable();

  $w('#startButton').onClick(() => handleStart());
  $w('#sendButton').onClick(() => handleSend());
  $w('#endButton').onClick(() => handleEnd());

  // Allow Enter key to send message
  $w('#messageInput').onKeyPress((event) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  });

  if (currentSessionId) {
    // Resume UI if session was already started
    setSessionActive(true);
    appendMessage('system', 'Welcome back! Your session is still active. Continue your check-in below.');
  }
});

async function handleStart() {
  const user = wixUsers.currentUser;
  if (!user.loggedIn) {
    // Prompt login if using Wix Members
    wixUsers.promptLogin().then(() => handleStart()).catch(() => {});
    return;
  }

  setLoading(true);
  try {
    const clientId = user.id;
    const clientName = await user.getEmail(); // or use member profile name

    const result = await startCoachingSession(clientId, clientName);
    currentSessionId = result.session_id;
    wixStorage.session.setItem('coaching_session_id', currentSessionId);

    setSessionActive(true);
    appendMessage('navigator', result.message);
  } catch (err) {
    appendMessage('system', 'Could not start session. Please try again.');
    console.error('Session start error:', err);
  } finally {
    setLoading(false);
  }
}

async function handleSend() {
  const userMessage = $w('#messageInput').value.trim();
  if (!userMessage || !currentSessionId) return;

  $w('#messageInput').value = '';
  appendMessage('user', userMessage);
  setLoading(true);

  try {
    const result = await sendCoachingMessage(currentSessionId, userMessage);
    appendMessage('navigator', result.reply);
  } catch (err) {
    appendMessage('system', 'Message failed. Please try again.');
    console.error('Message error:', err);
  } finally {
    setLoading(false);
  }
}

async function handleEnd() {
  if (!currentSessionId) return;
  setLoading(true);

  try {
    const summary = await endCoachingSession(currentSessionId);
    wixStorage.session.removeItem('coaching_session_id');
    currentSessionId = null;
    setSessionActive(false);

    // Display alert status
    const alertEmoji = summary.alerts.level === 'red' ? '🔴' :
                       summary.alerts.level === 'yellow' ? '🟡' : '🟢';
    appendMessage('system',
      `Session complete! ${alertEmoji} ${summary.alerts.reason}\n\n` +
      `Your session summary has been saved. Adi Ben Nesher will review it before your next meeting.`
    );
  } catch (err) {
    appendMessage('system', 'Could not end session. Please try again.');
    console.error('End session error:', err);
  } finally {
    setLoading(false);
  }
}

// --- UI Helpers ---

function appendMessage(role, text) {
  const roles = { user: 'You', navigator: 'Navigator', system: 'System' };
  chatMessages.push({ _id: String(chatMessages.length), role, text, label: roles[role] || role });
  refreshChatRepeater();
}

function refreshChatRepeater() {
  $w('#chatHistory').data = chatMessages;
  $w('#chatHistory').forEachItem(($item, itemData) => {
    $item('#messageLabel').text = itemData.label;
    $item('#messageText').text = itemData.text;
    // Style differently based on role
    if (itemData.role === 'user') {
      $item('#messageBubble').style.backgroundColor = '#E3F2FD';
    } else if (itemData.role === 'navigator') {
      $item('#messageBubble').style.backgroundColor = '#F1F8E9';
    } else {
      $item('#messageBubble').style.backgroundColor = '#F5F5F5';
    }
  });
  // Scroll to bottom
  $w('#chatHistory').scrollTo();
}

function setSessionActive(active) {
  if (active) {
    $w('#startButton').hide();
    $w('#endButton').show();
    $w('#sendButton').enable();
    $w('#messageInput').enable();
  } else {
    $w('#startButton').show();
    $w('#endButton').hide();
    $w('#sendButton').disable();
    $w('#messageInput').disable();
  }
}

function setLoading(loading) {
  if (loading) {
    $w('#loadingSpinner').show();
    $w('#sendButton').disable();
  } else {
    $w('#loadingSpinner').hide();
    if (currentSessionId) {
      $w('#sendButton').enable();
    }
  }
}
