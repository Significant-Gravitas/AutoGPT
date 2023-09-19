class ReportRequestBody {
  final String test;
  final String testRunId;
  final bool mock;

  ReportRequestBody(
      {required this.test, required this.testRunId, required this.mock});

  Map<String, dynamic> toJson() {
    return {
      'test': test,
      'test_run_id': testRunId,
      'mock': mock,
    };
  }
}
