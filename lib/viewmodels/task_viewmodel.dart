import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:flutter/foundation.dart';
import 'package:collection/collection.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/models/task_request_body.dart';

class TaskViewModel with ChangeNotifier {
  final TaskService _taskService;
  List<Task> _tasks = [];
  Task? _selectedTask; // This will store the currently selected task

  TaskViewModel(this._taskService);

  /// Returns the list of tasks.
  List<Task> get tasks => _tasks;

  /// Returns the currently selected task.
  Task? get selectedTask => _selectedTask;

  /// Adds a task and returns its ID.
  Future<String> createTask(String title) async {
    final newTask = TaskRequestBody(input: title);
    // Add to data source
    final createdTask = await _taskService.createTask(newTask);
    // Create a Task object from the created task response
    final newTaskObject =
        Task(id: createdTask['task_id'], title: createdTask['input']);

    // Update local tasks list and notify listeners
    _tasks.add(newTaskObject);
    notifyListeners();

    return newTaskObject.id; // Return the ID of the new task
  }

  /// Deletes a task.
  void deleteTask(String id) async {
    // TODO: Protocol doesn't support deleting tasks, we need to manually manage this list

    // Update local tasks list and notify listeners
    _tasks.removeWhere((task) => task.id == id);
    notifyListeners();
  }

  /// Fetches tasks from the data source.
  void fetchTasks() async {
    try {
      final fetchedTasks = await _taskService.listAllTasks();
      _tasks = fetchedTasks.map((taskMap) {
        return Task(
          id: taskMap['task_id'].toString(),
          title: taskMap['output'].toString(),
        );
      }).toList();
      notifyListeners();
      print("Tasks fetched successfully!");
    } catch (error) {
      print("Error fetching tasks: $error");
    }
  }

  /// Handles the selection of a task by its ID.
  void selectTask(String id) {
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
