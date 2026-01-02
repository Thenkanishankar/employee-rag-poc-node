import fs from "fs";
import csv from "csv-parser";
import { pipeline } from "@xenova/transformers";
import prompts from "prompts";
import cosineSimilarity from "compute-cosine-similarity";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

const sentences = [];

for await (const row of fs.createReadStream("employees.csv").pipe(csv())) {
  sentences.push(
    `${row.name} is a ${row.role} with ${row.experience} years of experience working in ${row.location}. ` +
    `Phone ${row.number}. Email ${row.email}.`
  );
}

const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
const docEmbeddings = [];
for (const text of sentences) {
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true
  })
  docEmbeddings.push([...output.data]);
}

const retrieveContext = async (query, topk = 3) => {
  const queryVector = [...(await embedder(query, {pooling: "mean", normalize: true})).data];
  const scores = docEmbeddings.map((docVec) => 
    cosineSimilarity(queryVector, docVec)
  );
  const topIndices = scores
    .map((score, index) => ({score, index}))
    .sort((a,b) => b.score - a.score)
    .slice(0, topk)
    .map((item) => item.index)
  return topIndices.map((i) => sentences[i]).join(" ");
}

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const askOpenRouter = async (context, question) => {
  const response = await axios.post(
    "https://openrouter.ai/api/v1/chat/completions",
    {
      model: "deepseek/deepseek-chat",
      messages: [
        {
          role: "system",
          content:
            "You are a friendly and helpful employee directory assistant. " +
            "Answer in a natural, conversational, human-like way. " +
            "Use short paragraphs instead of lists. " +
            "Do not use bullet points, numbering, emojis, or markdown formatting. " +
            "Keep responses simple and professional. " +
            "If the question is unrelated, reply exactly: " +
            "'I can only answer employee-related questions.'",
        },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion:\n${question}`,
        },
      ],
      temperature: 0.3,
    },
    {
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "X-Title": "Employee RAG Chatbot",
      },
    }
  );
  return response.data.choices[0].message.content;
}

console.log("Hi! I am Employee Bot. I can help you with employee-related information.");

while(true) {
  const {query} = await prompts({
    type: "text",
    name: "query",
    message: "You:"
  });

  if (!query || query.toLowerCase() === 'exit') {
    break;
  }

  const context = await retrieveContext(query);
  const answer = await askOpenRouter(context, query);
  
  console.log("\nEmployee Bot:", answer, "\n");
}

