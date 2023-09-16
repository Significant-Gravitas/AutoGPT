class ReportRequestBody {
  final String category;
  final List<String> tests;
  final bool mock;

  ReportRequestBody(
      {required this.category, required this.tests, required this.mock});

  Map<String, dynamic> toJson() {
    return {
      'category': category,
      'tests': tests,
      'mock': mock,
    };
  }
}
