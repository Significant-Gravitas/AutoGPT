class BenchmarkStepRequestBody {
  final String? input;

  BenchmarkStepRequestBody({required this.input});

  Map<String, dynamic> toJson() {
    if (input == null) {
      return {};
    }
    return {'input': input};
  }
}
