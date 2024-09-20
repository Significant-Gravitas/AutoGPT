import 'dart:convert';
import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/test_suite.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:flutter/foundation.dart';
import 'package:collection/collection.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/models/task_request_body.dart';

// TODO: How will all these functions work with test suites?
class TaskViewModel with ChangeNotifier {
  final TaskService _taskService;
  final SharedPreferencesService _prefsService;

  List<Task> _tasks = [];
  List<TestSuite> _testSuites = [];
  List<dynamic> combinedDataSource = [];
  List<Task> tasksDataSource = [];

  Task? _selectedTask;
  TestSuite? _selectedTestSuite;

  bool _isWaitingForAgentResponse = false;

  bool get isWaitingForAgentResponse => _isWaitingForAgentResponse;

  TaskViewModel(this._taskService, this._prefsService);

  /// Returns the currently selected task.
  Task? get selectedTask => _selectedTask;
  TestSuite? get selectedTestSuite => _selectedTestSuite;

  /// Adds a task and returns its ID.
  Future<String> createTask(String title) async {
    _isWaitingForAgentResponse = true;
    notifyListeners();
    try {
      final newTask = TaskRequestBody(input: title);
      // Add to data source
      final createdTask = await _taskService.createTask(newTask);
      // Create a Task object from the created task response
      final newTaskObject =
          Task(id: createdTask['task_id'], title: createdTask['input']);

      fetchAndCombineData();

      final taskId = newTaskObject.id;
      print("Task $taskId created successfully!");

      return newTaskObject.id;
    } catch (e) {
      // TODO: We are bubbling up the full response. Revisit this.
      rethrow;
    } finally {
      _isWaitingForAgentResponse = false;
      notifyListeners();
    }
  }

  /// Deletes a task.
  void deleteTask(String taskId) {
    _taskService.saveDeletedTask(taskId);
    _tasks.removeWhere((task) => task.id == taskId);
    notifyListeners();
    print("Task $taskId deleted successfully!");
  }

  /// Fetches tasks from the data source.
  Future<void> fetchTasks() async {
    try {
      final tasksFromApi = await _taskService.fetchAllTasks();
      _tasks = tasksFromApi
          .where((task) => !_taskService.isTaskDeleted(task.id))
          .toList();

      _tasks = _tasks.reversed.toList();

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

  /// Deselects the currently selected task.
  void deselectTask() {
    _selectedTask = null;
    print("Deselected the current task.");
    notifyListeners(); // Notify listeners to rebuild UI
  }

  void selectTestSuite(TestSuite testSuite) {
    _selectedTestSuite = testSuite;
    notifyListeners();
  }

  void deselectTestSuite() {
    _selectedTestSuite = null;
    notifyListeners();
  }

  // Helper method to save test suites to SharedPreferences
  Future<void> _saveTestSuitesToPrefs() async {
    final testSuitesToStore =
        _testSuites.map((testSuite) => jsonEncode(testSuite.toJson())).toList();
    await _prefsService.setStringList('testSuites', testSuitesToStore);
  }

  // Adds a new test suite and saves it to SharedPreferences
  void addTestSuite(TestSuite testSuite) async {
    _testSuites.add(testSuite);
    await _saveTestSuitesToPrefs();
    notifyListeners();
    print("Test suite successfully added!");
  }

  // Fetch test suites from SharedPreferences
  Future<void> fetchTestSuites() async {
    final storedTestSuites =
        await _prefsService.getStringList('testSuites') ?? [];
    _testSuites = storedTestSuites
        .map((testSuiteMap) => TestSuite.fromJson(jsonDecode(testSuiteMap)))
        .toList();
    notifyListeners();
  }

  // The fetchAndCombineData method performs several tasks:
  // 1. It fetches the tasks and filters out deleted ones.
  // 2. It fetches the test suites from SharedPreferences.
  // 3. It combines both the tasks and test suites into a single data source according to specified logic.
  Future<void> fetchAndCombineData() async {
    // Step 1: Fetch tasks from the data source
    // This will populate the _tasks list with tasks fetched from the backend.
    await fetchTasks();

    // Step 2: Fetch test suites from SharedPreferences
    // This will populate the _testSuites list with test suites fetched from SharedPreferences.
    await fetchTestSuites();

    // Step 3: Combine into a shared data source
    // Create a map to hold test suites by their timestamp.
    Map<String, TestSuite> testSuiteMap = {};

    // Clear the existing combined data source to start fresh.
    combinedDataSource.clear();
    tasksDataSource.clear();

    // Iterate through each task to check if it's contained in any of the test suites.
    for (var task in _tasks) {
      bool found = false;

      // Iterate through each test suite.
      for (var testSuite in _testSuites) {
        // Check if the current task is contained in the current test suite.
        if (testSuite.tests.contains(task)) {
          found = true;

          // If this test suite is already in the map, add this task to its list of tasks.
          if (testSuiteMap.containsKey(testSuite.timestamp)) {
            testSuiteMap[testSuite.timestamp]!.tests.add(task);

            // Find and replace the test suite in the combined data source.
            final index = combinedDataSource.indexWhere((item) =>
                item is TestSuite && item.timestamp == testSuite.timestamp);
            if (index != -1) {
              combinedDataSource[index] = testSuiteMap[testSuite.timestamp]!;
            }
          }
          // If this test suite is not in the map, add it to the map and to the combined data source.
          else {
            final newTestSuite = TestSuite(
              timestamp: testSuite.timestamp,
              tests: [task],
            );
            testSuiteMap[testSuite.timestamp] = newTestSuite;
            combinedDataSource.add(
                newTestSuite); // Add the new test suite to the combined data source.
          }
          break; // Exit the loop as the task is found in a test suite.
        }
      }

      // If the task was not found in any test suite, add it to the combined data source.
      if (!found) {
        combinedDataSource.add(task);
        tasksDataSource.add(task);
      }
    }

    // After processing all tasks, call notifyListeners to rebuild the widgets that depend on this data.
    notifyListeners();
    print("Combined tasks and test suites successfully!");
  }
}
