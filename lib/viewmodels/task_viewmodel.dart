import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/viewmodels/mock_data.dart';
import 'package:flutter/foundation.dart';
import 'package:collection/collection.dart';

// TODO: Update whole class once we have created TaskService
class TaskViewModel with ChangeNotifier {
  List<Task> _tasks = [];
  Task? _selectedTask; // This will store the currently selected task

  /// Returns the list of tasks.
  List<Task> get tasks => _tasks;

  /// Returns the currently selected task.
  Task? get selectedTask => _selectedTask;

  /// Adds a task.
  void createTask(String title) {
    // Generate an ID (This is a simplistic approach for mock data)
    final id = _tasks.length + 1;
    final newTask = Task(id: id, title: title);

    // Add to data source
    addTask(newTask);

    // Update local tasks list and notify listeners
    _tasks.add(newTask);
    notifyListeners();
  }

  /// Deletes a task.
  void deleteTask(int id) {
    // Remove from data source
    removeTask(id);

    // Update local tasks list and notify listeners
    _tasks.removeWhere((task) => task.id == id);
    notifyListeners();
  }

  /// Fetches tasks from the data source.
  void fetchTasks() {
    try {
      _tasks = mockTasks;
      notifyListeners(); // Notify listeners to rebuild UI
      print("Tasks fetched successfully!");
    } catch (error) {
      print("Error fetching tasks: $error");
      // TODO: Handle additional error scenarios or log them as required
    }
  }

  /// Handles the selection of a task by its ID.
  void selectTask(int id) {
    final task = _tasks.firstWhereOrNull((t) => t.id == id);

    if (task != null) {
      _selectedTask = task;
      print("Selected task with ID: ${task.id} and Title: ${task.title}");
      notifyListeners(); // Notify listeners to rebuild UI
    } else {
      final errorMessage =
          "Error: Attempted to select a task with ID: $id that does not exist in the data source.";
      print(errorMessage);
      throw ArgumentError(errorMessage);
    }
  }
}
