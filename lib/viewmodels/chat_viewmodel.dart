import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:flutter/foundation.dart';
import 'mock_data.dart'; // Import the mock data

// TODO: Update whole class once we have created TaskService
class ChatViewModel with ChangeNotifier {
  List<Chat> _chats = [];

  /// Returns the current list of chats.
  List<Chat> get chats => _chats;

  /// Fetches chats from the mock data source for a specific task.
  void fetchChatsForTask(int taskId) {
    try {
      _chats = mockChats.where((chat) => chat.taskId == taskId).toList();
      notifyListeners(); // Notify listeners to rebuild UI
      print("Chats fetched successfully for task ID: $taskId");
    } catch (error) {
      print("Error fetching chats: $error");
      // TODO: Handle additional error scenarios or log them as required
    }
  }

  /// Simulates sending a chat message for a specific task.
  void sendChatMessage(int taskId, String message) {
    final userChat = Chat(
        id: _chats.length + 1,
        taskId: taskId,
        message: message,
        timestamp: DateTime.now(),
        messageType: MessageType.user);

    // For now, we'll simulate an agent's reply after the user's message
    final agentChat = Chat(
        id: _chats.length + 2,
        taskId: taskId,
        message: 'Automated reply to: $message',
        timestamp: DateTime.now().add(const Duration(seconds: 2)),
        messageType: MessageType.agent);

    _chats.addAll([userChat, agentChat]);
    notifyListeners(); // Notify UI of the new chats
    print("User chat and automated agent reply added for task ID: $taskId");
  }
}
