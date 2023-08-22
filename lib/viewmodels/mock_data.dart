import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

/// A list of mock tasks for the application.
/// TODO: Remove this file when we implement TaskService
List<Task> mockTasks = [
  Task(id: 1, title: 'Task 1'),
  Task(id: 2, title: 'Task 2'),
  Task(id: 3, title: 'Task 3'),
  // ... add more mock tasks as needed
];

/// Adds a task to the mock data.
void addTask(Task task) {
  mockTasks.add(task);
}

/// Removes a task from the mock data based on its ID.
void removeTask(int id) {
  mockTasks.removeWhere((task) => task.id == id);
}

// mock_data.dart (extend the existing mock_data.dart file)

// Sample chats for mock data
List<Chat> mockChats = [
  Chat(
      id: 1,
      taskId: 1,
      message: 'Hello Agent',
      timestamp: DateTime.now(),
      messageType: MessageType.user),
  Chat(
      id: 2,
      taskId: 1,
      message: 'Hello! How can I assist you today?',
      timestamp: DateTime.now().add(Duration(minutes: 1)),
      messageType: MessageType.agent),
  // ... add more mock chat data as required
];
