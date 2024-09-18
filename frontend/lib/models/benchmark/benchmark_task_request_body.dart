class BenchmarkTaskRequestBody {
  final String input;
  final String evalId;

  BenchmarkTaskRequestBody({required this.input, required this.evalId});

  Map<String, dynamic> toJson() {
    return {
      'input': input,
      'eval_id': evalId,
    };
  }
}
