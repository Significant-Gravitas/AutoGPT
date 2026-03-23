/**
 * ABN Consulting AI Co-Navigator — Wix Page Code
 *
 * File location in Wix Editor: Pages → coaching-session (page code tab)
 *
 * REQUIRED PAGE ELEMENTS (add in Wix Editor, use these exact IDs):
 *   #chatHistory    → Repeater  (each item needs #messageBubble, #messageLabel, #messageText)
 *   #messageInput   → Text Input
 *   #sendButton     → Button  ("Send")
 *   #startButton    → Button  ("Start Session")
 *   #endButton      → Button  ("End Session") — hidden by default
 *   #loadingSpinner → Image / Lottie animation  — hidden by default
 *
 * Wix Members app must be installed so wixUsers is available.
 */

import wixUsers from 'wix-users';
import wixStorage from 'wix-storage';
import {
  startCoachingSession,
  sendCoachingMessage,
  endCoachingSession,
} from 'backend/coaching-backend';

let chatMessages = [];
let currentSessionId = null;

$w.onReady(function () {
  currentSessionId = wixStorage.session.getItem('coaching_session_id');

  $w('#endButton').hide();
  $w('#sendButton').disable();
  $w('#messageInput').disable();

  $w('#startButton').onClick(() => handleStart());
  $w('#sendButton').onClick(() => handleSend());
  $w('#endButton').onClick(() => handleEnd());

  $w('#messageInput').onKeyPress((event) => {
    if (event.key === 'Enter') handleSend();
  });

  if (currentSessionId) {
    setSessionActive(true);
    appendMessage('system', 'Welcome back! Your session is still active. Continue your check-in below.');
  }
});

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async function handleStart() {
  const user = wixUsers.currentUser;
  if (!user.loggedIn) {
    wixUsers.promptLogin().then(() => handleStart()).catch(() => {});
    return;
  }

  setLoading(true);
  try {
    const clientId = user.id;
    const clientName = await user.getEmail();

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

    const alertEmoji =
      summary.alerts.level === 'red'    ? '🔴' :
      summary.alerts.level === 'yellow' ? '🟡' : '🟢';

    appendMessage(
      'system',
      `Session complete! ${alertEmoji} ${summary.alerts.reason}\n\n` +
      `Your session summary has been saved. Adi Ben Nesher will review it before your next meeting.`,
    );
  } catch (err) {
    appendMessage('system', 'Could not end session. Please try again.');
    console.error('End session error:', err);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function appendMessage(role, text) {
  const labels = { user: 'You', navigator: 'Navigator', system: 'System' };
  chatMessages.push({
    _id: String(chatMessages.length),
    role,
    text,
    label: labels[role] || role,
  });
  refreshChatRepeater();
}

function refreshChatRepeater() {
  $w('#chatHistory').data = chatMessages;
  $w('#chatHistory').forEachItem(($item, itemData) => {
    $item('#messageLabel').text = itemData.label;
    $item('#messageText').text = itemData.text;
    const bg =
      itemData.role === 'user'      ? '#E3F2FD' :
      itemData.role === 'navigator' ? '#F1F8E9' : '#F5F5F5';
    $item('#messageBubble').style.backgroundColor = bg;
  });
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
    if (currentSessionId) $w('#sendButton').enable();
  }
}
