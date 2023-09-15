import 'dart:async';
import 'package:auto_gpt_flutter_client/models/benchmark_service/report_request_body.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:auto_gpt_flutter_client/models/benchmark_service/api_type.dart';

class BenchmarkService {
  final RestApiUtility api;

  BenchmarkService(this.api);

  /// Generates a report using POST REST API at the /reports URL.
  ///
  /// [reportRequestBody] is a Map representing the request body for generating a report.
  Future<Map<String, dynamic>> generateReport(
      ReportRequestBody reportRequestBody) async {
    try {
      return await api.post('reports', reportRequestBody.toJson(),
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to generate report: $e');
    }
  }

  /// Polls for updates using the GET REST API at the /updates?last_update_time=TIMESTAMP URL.
  ///
  /// [lastUpdateTime] is the UNIX UTC timestamp for last update time.
  Future<Map<String, dynamic>> pollUpdates(int lastUpdateTime) async {
    try {
      return await api.get('updates?last_update_time=$lastUpdateTime',
          apiType: ApiType.benchmark);
    } catch (e) {
      throw Exception('Failed to poll updates: $e');
    }
  }
}
