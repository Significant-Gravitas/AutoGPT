class ReportRequestBody {
  final String category;
  final List<String> tests;

  ReportRequestBody({required this.category, required this.tests});

  Map<String, dynamic> toJson() {
    return {
      'category': category,
      'tests': tests,
    };
  }
}
