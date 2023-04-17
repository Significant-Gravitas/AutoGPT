import ErrorService from "../services/ErrorService";
import IAnswer from "../types/data/IAnswer";
import IInitData from "../types/data/IInitData";

const startScript: () => Promise<void> = async () => {
	await fetch("/api/start");
};

const killScript: () => Promise<void> = async () => {
	await fetch("/api/stop");
};

const fetchData: () => Promise<IAnswer[]> = async () => {
	const res = await fetch("/api/data");
	let data = await res.json();
	if (data === "") {
		return [];
	}
	// remove last char from data data is a string
	// remove \n
	data = data.output.replaceAll("\n", "");
	data = data.replaceAll("\u001b", "");
	// remove last comma
	data = data
		.split("")
		.reverse()
		.join("")
		.replace(",", "")
		.split("")
		.reverse()
		.join("");
	let json = [];
	try {
		json = JSON.parse(`[${data}]`);
	} catch (e) {
		console.log(data);
		debugger;
	}
	return json;
};

const downloadFile: (filename: string) => Promise<void> = async (
	filename: string,
) => {
	const res = await fetch("/api/download", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ filename }),
	});
	const blob = await res.blob();
	const url = window.URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	a.remove();
};

const createInitData = async (data: IInitData) => {
	const res = await fetch("/api/init", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(data),
	});
	return res;
};

export default {
	startScript: ErrorService.errorHandler(startScript),
	killScript: ErrorService.errorHandler(killScript),
	fetchData: ErrorService.errorHandler(fetchData),
	createInitData: ErrorService.errorHandler(createInitData),
	downloadFile: ErrorService.errorHandler(downloadFile),
};
