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
