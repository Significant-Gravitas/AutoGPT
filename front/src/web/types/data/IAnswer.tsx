enum AnswerType {
	TEXT = "text",
}

interface IAnswer {
	type?: AnswerType;
	content: string;
	title: string;
}
export default IAnswer;
