import 'dart:async';
import 'package:auto_gpt_flutter_client/models/benchmark_service/report_request_body.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:auto_gpt_flutter_client/models/benchmark_service/api_type.dart';

class BenchmarkService {
  final RestApiUtility api;

  BenchmarkService(this.api);

  /// Generates a single report using POST REST API at the /reports URL.
  ///
  /// [reportRequestBody] is a Map representing the request body for generating a single report.
  Future<Map<String, dynamic>> generateSingleReport(
      ReportRequestBody reportRequestBody) async {
    try {
      return await api.post('reports', reportRequestBody.toJson(),
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to generate single report: $e');
    }
  }

  /// Generates a combined report using POST REST API at the /reports/query URL.
  ///
  /// [testRunIds] is a list of strings representing the test run IDs to be combined into a single report.
  Future<Map<String, dynamic>> generateCombinedReport(
      List<String> testRunIds) async {
    try {
      final Map<String, dynamic> requestBody = {'test_run_ids': testRunIds};
      return await api.post('reports/query', requestBody,
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to generate combined report: $e');
    }
  }
}
