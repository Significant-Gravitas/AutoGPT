package models

type Agent struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Author      string   `json:"author"`
	Keywords    []string `json:"keywords"`
	Categories  []string `json:"categories"`
	Graph       Graph    `json:"graph"`
}

type Graph struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Add other fields as needed
}

type AddAgentRequest struct {
	Graph      Graph    `json:"graph"`
	Author     string   `json:"author"`
	Keywords   []string `json:"keywords"`
	Categories []string `json:"categories"`
}
