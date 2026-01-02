# Employee RAG Chatbot (Node.js POC)

This is a **Proof of Concept (POC)** Employee Directory Chatbot built using **Node.js** and **Retrieval-Augmented Generation (RAG)**.

The chatbot answers **only employee-related questions** by retrieving relevant employee information from a CSV file and generating natural, human-like responses using a **free LLM via OpenRouter**.

---

## 🚀 Features

- Reads employee data from a CSV file
- Converts employee details into **semantic embeddings**
- Uses **cosine similarity** for semantic search
- Uses **DeepSeek (free model)** via OpenRouter
- Rejects non-employee-related questions
- CLI-based chatbot (no UI, no server)
- Fully local embeddings (no vector DB)

---
