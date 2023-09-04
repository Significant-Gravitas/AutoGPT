import 'dart:io';
import 'package:auto_gpt_flutter_client/models/step_request_body.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

/// Service class for performing chat-related operations.
class ChatService {
  final RestApiUtility api;

  ChatService(this.api);

  /// Executes a step in a specific task.
  ///
  /// [taskId] is the ID of the task.
  /// [stepRequestBody] is a Map representing the request body for executing a step.
  Future<Map<String, dynamic>> executeStep(
      String taskId, StepRequestBody stepRequestBody) async {
    try {
      return await api.post(
          'agent/tasks/$taskId/steps', stepRequestBody.toJson());
    } catch (e) {
      throw Exception('Failed to execute step: $e');
    }
  }

  /// Gets details about a specific task step.
  ///
  /// [taskId] is the ID of the task.
  /// [stepId] is the ID of the step.
  Future<Map<String, dynamic>> getStepDetails(
      String taskId, String stepId) async {
    try {
      return await api.get('agent/tasks/$taskId/steps/$stepId');
    } catch (e) {
      throw Exception('Failed to get step details: $e');
    }
  }

  /// Lists all steps for a specific task.
  ///
  /// [taskId] is the ID of the task.
  /// [currentPage] and [pageSize] are optional pagination parameters.
  Future<Map<String, dynamic>> listTaskSteps(String taskId,
      {int currentPage = 1, int pageSize = 10}) async {
    try {
      return await api.get(
          'agent/tasks/$taskId/steps?current_page=$currentPage&page_size=$pageSize');
    } catch (e) {
      throw Exception('Failed to list task steps: $e');
    }
  }

  /// Uploads an artifact for a specific task.
  ///
  /// [taskId] is the ID of the task.
  /// [artifactFile] is the File to be uploaded.
  /// [uri] is the URI of the artifact.
  Future<Map<String, dynamic>> uploadArtifact(
      String taskId, File artifactFile, String uri) async {
    return Future.value({'status': 'Not implemented yet'});
  }

  /// Downloads a specific artifact.
  ///
  /// [taskId] is the ID of the task.
  /// [artifactId] is the ID of the artifact.
  Future<Map<String, dynamic>> downloadArtifact(
      String taskId, String artifactId) async {
    return Future.value({'status': 'Not implemented yet'});
  }
}
