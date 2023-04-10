import ErrorService from "../services/ErrorService"
import IAnswer from "../types/data/IAnswer"
import IInitData from "../types/data/IInitData"

const startScript: () => Promise<void> = async () => {
  await fetch("/api/start")
}

const killScript: () => Promise<void> = async () => {
  await fetch("/api/stop")
}

const fetchData: () => Promise<IAnswer[]> = async () => {
  const res = await fetch("/api/data")
  let data = await res.json()
  // remove last char from data data is a string
  // remove \n
  data = data.output.replace("\n", "")
  data = data.slice(0, -2)
  console.log(data)
  const json = JSON.parse(`[${data}]`)
  return json
}

const createInitData = async (data: IInitData) => {
  const res = await fetch("/api/init", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
  return res
}

export default {
  startScript: ErrorService.errorHandler(startScript),
  killScript: ErrorService.errorHandler(killScript),
  fetchData: ErrorService.errorHandler(fetchData),
  createInitData: ErrorService.errorHandler(createInitData),
}
