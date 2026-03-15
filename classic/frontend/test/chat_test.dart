import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Chat', () {
    // Test the properties of the Chat class
    test('Chat properties', () {
      final chat = Chat(
          id: 1,
          taskId: 1,
          message: 'Test Message',
          timestamp: DateTime.now(),
          messageType: MessageType.user);

      expect(chat.id, 1);
      expect(chat.taskId, 1);
      expect(chat.message, 'Test Message');
      expect(chat.messageType, MessageType.user);
    });

    // Test Chat.fromMap method
    test('Chat.fromMap', () {
      final chat = Chat.fromMap({
        'id': 1,
        'taskId': 1,
        'message': 'Test Message',
        'timestamp': DateTime.now().toString(),
        'messageType': 'user'
      });

      expect(chat.id, 1);
      expect(chat.taskId, 1);
      expect(chat.message, 'Test Message');
      expect(chat.messageType, MessageType.user);
    });

    // Test that two Chat objects with the same properties are equal
    test('Two chats with same properties are equal', () {
      final chat1 = Chat(
          id: 3,
          taskId: 3,
          message: 'Same Message',
          timestamp: DateTime.now(),
          messageType: MessageType.agent);
      final chat2 = Chat(
          id: 3,
          taskId: 3,
          message: 'Same Message',
          timestamp: chat1.timestamp,
          messageType: MessageType.agent);

      expect(chat1, chat2);
    });

    // Test that toString() returns a string representation of the Chat
    test('toString returns string representation', () {
      final chat = Chat(
          id: 4,
          taskId: 4,
          message: 'Test toString',
          timestamp: DateTime.now(),
          messageType: MessageType.user);

      expect(chat.toString(),
          'Chat(id: 4, taskId: 4, message: Test toString, timestamp: ${chat.timestamp}, messageType: MessageType.user)');
    });
  });
}
