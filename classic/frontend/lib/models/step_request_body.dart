class StepRequestBody {
  final String input;
  final Map<String, dynamic>? additionalInput;

  StepRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() {
    Map<String, dynamic> result = {'input': input, 'additional_input': additionalInput};
    result.removeWhere((_, v) => v == null);
    return result;
  }
}
