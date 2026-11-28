"use strict";

// The Self-hosted chat connection's startup status. Individual chats can
// instead use a linked subscription, which this runtime event does not name.
//
// When the Claude Code CLI on this machine is signed in, Self-hosted chat
// uses that sign-in and its turns count against the user's Claude plan
// (runtime/autogpt_desktop/claude_code.py). Nothing is configured for it, so
// the app has to say that it is happening. The runtime reports what it found
// once per start:
//
//   {"event": "claude_code", "state": "in_use", "cli": "...", "version": "2.1.284", "bundled": true}
//
// The line is information, never a control: the app has no Claude sign-in of
// its own and no field for a token. Signing in is done with Claude Code
// itself (`claude`), and the runtime looks only while it starts, so every
// line that asks for something ends in a restart.

// claude_code.py: IN_USE, SIGNED_OUT, NO_ANSWER, NOT_FOUND, OFF, REFUSED.
const WITH_KEYS = "Self-hosted chat connection: configure API keys or a local model in the settings file.";
const signedOut = (name) => `${WITH_KEYS} To use your Claude plan, run \`claude\`, sign in, and restart ${name}.`;
const LINES = {
  in_use: (name) =>
    `Self-hosted chat connection: Claude Code sign-in (turns count against your Claude plan). Restart ${name} after signing in or out.`,
  signed_out: signedOut,
  no_answer: (name) =>
    `${WITH_KEYS} Claude Code did not say whether it is signed in; restart ${name} to ask it again.`,
  not_found: signedOut,
  off: () => `${WITH_KEYS} Using your Claude Code sign-in is turned off.`,
  refused: () => `${WITH_KEYS} Your Claude Code sign-in does not fit these settings; the log says why.`,
};

// What the runtime last reported, or null: nothing yet, or something this
// shell does not know how to word.
function claudeCodeStatus(event) {
  if (!event || event.event !== "claude_code") return null;
  return Object.hasOwn(LINES, event.state) ? { state: event.state } : null;
}

function claudeCodeLine(status, install) {
  return status ? LINES[status.state](install.productName) : null;
}

// For the tray menu and the application menu. Disabled: there is nothing to
// click. No item until the runtime has said what it found.
function claudeCodeMenuItems({ status, install }) {
  const label = claudeCodeLine(status, install);
  return label ? [{ label, enabled: false }] : [];
}

// What the menus show, for main.js to keep: `changed` is called whenever the
// line is another one. A runtime that is being started has reported nothing
// yet, and may never (it can fail before it looks), so the line of the one
// before it goes at once.
function claudeCodeReports(changed) {
  let status = null;
  function set(next) {
    status = next;
    changed();
  }
  return {
    status: () => status,
    runtimeStarting: () => set(null),
    // Whether the event was a report (and so nobody else's to handle).
    hears(event) {
      if (!event || event.event !== "claude_code") return false;
      set(claudeCodeStatus(event));
      return true;
    },
  };
}

module.exports = { claudeCodeLine, claudeCodeMenuItems, claudeCodeReports, claudeCodeStatus };
