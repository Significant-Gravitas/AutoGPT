class TaskRequestBody {
  final String input;
  final Map<String, dynamic>? additionalInput;

  TaskRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() {
    Map<String, dynamic> result = {'input': input, 'additional_input': additionalInput};
    result.removeWhere((_, v) => v == null);
    return result;
  }
}
