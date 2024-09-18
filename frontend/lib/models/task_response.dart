import 'package:auto_gpt_flutter_client/models/pagination.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

class TaskResponse {
  final List<Task> tasks;
  final Pagination pagination;

  TaskResponse({required this.tasks, required this.pagination});

  factory TaskResponse.fromJson(Map<String, dynamic> json) {
    return TaskResponse(
      tasks: (json['tasks'] as List).map((taskJson) {
        var task = Task.fromMap(taskJson);
        return task;
      }).toList(),
      pagination: Pagination.fromJson(json['pagination']),
    );
  }
}
