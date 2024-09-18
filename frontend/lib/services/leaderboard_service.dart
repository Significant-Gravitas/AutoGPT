import 'package:auto_gpt_flutter_client/models/benchmark/api_type.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/benchmark_run.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

class LeaderboardService {
  final RestApiUtility api;

  LeaderboardService(this.api);

  /// Submits a benchmark report to the leaderboard.
  ///
  /// [benchmarkRun] is a BenchmarkRun object representing the data of a completed benchmark.
  Future<Map<String, dynamic>> submitReport(BenchmarkRun benchmarkRun) async {
    try {
      return await api.put(
        'api/reports',
        benchmarkRun.toJson(),
        apiType: ApiType.leaderboard,
      );
    } catch (e) {
      throw Exception('Failed to submit the report to the leaderboard: $e');
    }
  }
}
