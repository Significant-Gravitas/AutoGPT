class ReportRequestBody {
  final String test;
  final String testRunId;
  final bool mock;

  ReportRequestBody(this.mock, {required this.test, required this.testRunId});

  Map<String, dynamic> toJson() {
    return {
      'test': test,
      'test_run_id': testRunId,
      'mock': mock,
    };
  }
}
