import 'dart:async';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_step_request_body.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_task_request_body.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/api_type.dart';

class BenchmarkService {
  final RestApiUtility api;

  BenchmarkService(this.api);

  /// Creates a new benchmark task.
  ///
  /// [benchmarkTaskRequestBody] is a Map representing the request body for creating a task.
  Future<Map<String, dynamic>> createBenchmarkTask(
      BenchmarkTaskRequestBody benchmarkTaskRequestBody) async {
    try {
      return await api.post('agent/tasks', benchmarkTaskRequestBody.toJson(),
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to create a new task: $e');
    }
  }

  /// Executes a step in a specific benchmark task.
  ///
  /// [taskId] is the ID of the task.
  /// [benchmarkStepRequestBody] is a Map representing the request body for executing a step.
  Future<Map<String, dynamic>> executeBenchmarkStep(
      String taskId, BenchmarkStepRequestBody benchmarkStepRequestBody) async {
    try {
      return await api.post(
          'agent/tasks/$taskId/steps', benchmarkStepRequestBody.toJson(),
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to execute step: $e');
    }
  }

  /// Triggers an evaluation for a specific benchmark task.
  ///
  /// [taskId] is the ID of the task.
  Future<Map<String, dynamic>> triggerEvaluation(String taskId) async {
    try {
      return await api.post('agent/tasks/$taskId/evaluations', {},
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to trigger evaluation: $e');
    }
  }
}
