import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:auto_gpt_flutter_client/models/task_request_body.dart';
import 'package:auto_gpt_flutter_client/models/task_response.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

/// Service class for performing task-related operations.
class TaskService {
  final RestApiUtility api;
  final SharedPreferencesService prefsService;
  List<String> _deletedTaskIds = [];

  TaskService(this.api, this.prefsService);

  /// Creates a new task.
  ///
  /// [taskRequestBody] is a Map representing the request body for creating a task.
  Future<Map<String, dynamic>> createTask(
      TaskRequestBody taskRequestBody) async {
    try {
      return await api.post('agent/tasks', taskRequestBody.toJson());
    } catch (e) {
      // TODO: We are bubbling up the full response. Revisit this.
      rethrow;
    }
  }

  /// Fetches a single page of tasks.
  ///
  /// [currentPage] and [pageSize] are pagination parameters.
  Future<TaskResponse> fetchTasksPage(
      {int currentPage = 1, int pageSize = 10}) async {
    try {
      final response = await api
          .get('agent/tasks?current_page=$currentPage&page_size=$pageSize');
      return TaskResponse.fromJson(response);
    } catch (e) {
      throw Exception('Failed to fetch a page of tasks: $e');
    }
  }

  /// Fetches all tasks across all pages.
  // TODO: Temporarily make page size 10000 until pagination is fixed
  Future<List<Task>> fetchAllTasks({int pageSize = 10000}) async {
    int currentPage = 1;
    List<Task> allTasks = [];

    while (true) {
      final response =
          await fetchTasksPage(currentPage: currentPage, pageSize: pageSize);
      allTasks.addAll(response.tasks);

      if (response.tasks.length < pageSize) {
        // No more tasks to fetch
        break;
      }
      currentPage++;
    }
    return allTasks;
  }

  /// Gets details about a specific task.
  ///
  /// [taskId] is the ID of the task.
  Future<Map<String, dynamic>> getTaskDetails(String taskId) async {
    try {
      return await api.get('agent/tasks/$taskId');
    } catch (e) {
      throw Exception('Failed to get task details: $e');
    }
  }

  /// Lists all artifacts for a specific task.
  ///
  /// [taskId] is the ID of the task.
  /// [currentPage] and [pageSize] are optional pagination parameters.
  Future<Map<String, dynamic>> listTaskArtifacts(String taskId,
      {int currentPage = 1, int pageSize = 10}) async {
    try {
      return await api.get(
          'agent/tasks/$taskId/artifacts?current_page=$currentPage&page_size=$pageSize');
    } catch (e) {
      throw Exception('Failed to list task artifacts: $e');
    }
  }

  Future<void> loadDeletedTasks() async {
    _deletedTaskIds = await prefsService.getStringList('deletedTasks') ?? [];
    print("Deleted tasks fetched successfully!");
  }

  void saveDeletedTask(String taskId) {
    _deletedTaskIds.add(taskId);
    prefsService.setStringList('deletedTasks', _deletedTaskIds);
    print("Task $taskId deleted successfully!");
  }

  bool isTaskDeleted(String taskId) {
    return _deletedTaskIds.contains(taskId);
  }
}
