enum AnswerType {
	TEXT = "text",
}
export enum InternalType {
	WRITE_FILE = "write_file",
}
interface IAnswer {
	type?: AnswerType
	internalType?: InternalType
	content: string
	title: string
}
export default IAnswer
