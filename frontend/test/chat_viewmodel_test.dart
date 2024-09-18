import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // Initialize the ChatViewModel
  // TODO: Dependency injection in view models for testing purposes when we implement services
  final viewModel = ChatViewModel();

  group('ChatViewModel', () {
    test('fetch chats for a specific task', () {
      viewModel
          .fetchChatsForTask(1); // Assuming task with ID 1 exists in mock data
      expect(viewModel.chats.isNotEmpty, true);
      expect(viewModel.chats.every((chat) => chat.taskId == 1), true);
    });

    test('send chat message for a specific task', () {
      final initialChatsLength = viewModel.chats.length;
      viewModel.sendChatMessage(1, 'Test message');
      expect(viewModel.chats.length,
          initialChatsLength + 2); // One user message and one agent reply
      expect(viewModel.chats.last.messageType,
          MessageType.agent); // Last message should be agent's reply
    });

    // TODO: Refactor to return errors when we implement service
    test('fetch chats for invalid task id', () {
      viewModel.fetchChatsForTask(
          9999); // Assuming task with ID 9999 does not exist in mock data
      expect(
          viewModel.chats.where((chat) => chat.taskId == 9999).isEmpty, true);
    });

    // TODO: Refactor to return errors when we implement service
    test('send chat message for invalid task id', () {
      final initialChatsLength = viewModel.chats.length;
      viewModel.sendChatMessage(9999, 'Invalid test message');
      expect(
          viewModel.chats.length,
          initialChatsLength +
              2); // Even for invalid tasks, we're currently adding mock replies
      expect(viewModel.chats.last.messageType,
          MessageType.agent); // Last message should be agent's reply
    });
  });
}
