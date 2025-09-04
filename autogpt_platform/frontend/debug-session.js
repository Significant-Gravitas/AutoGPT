// Run this in the browser console to clear failed sessions and force reload

// Clear failed sessions list
localStorage.removeItem("failed_chat_sessions");
console.log("✅ Cleared failed sessions list");

// Clear stored session ID if needed
// localStorage.removeItem('chat_session_id');

// Clear pending session
localStorage.removeItem("pending_chat_session");
console.log("✅ Cleared pending session");

// Force reload the page
console.log("🔄 Reloading page...");
window.location.reload();
