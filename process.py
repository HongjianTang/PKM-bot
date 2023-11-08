from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
import sys
from replit import db
import json


def train():
  if len(db.keys()) < 1:
    print("The database should contain at least one key-value pair.",
          file=sys.stderr)
    return

  data = []
  for key in db.keys():
    print(f"Add data from key {key} to dataset")
    data.append(json.loads(db[key]))

  # 文本切块，后续需考虑优化md和代码场景
  textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

  docs = []
  for sets in data:
    ttt = sets["date"] + " " + " " + sets["text"]
    docs.extend(textSplitter.split_text(ttt))

  # 调用ada模型向量化文本
  print(docs)
  store = FAISS.from_texts(docs, OpenAIEmbeddings())
  faiss.write_index(store.index, "training.index")
  store.index = None

  with open("training/faiss.pkl", "wb") as f:
    pickle.dump(store, f)


def runPrompt():
  index = faiss.read_index("training.index")

  with open("training/faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  with open("prompt/prompt.txt", "r") as f:
    promptTemplate = f.read()

  prompt = Prompt(template=promptTemplate,
                  input_variables=["context", "question"])

  llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0.25))

  def onMessage(question):
    docs = store.similarity_search_with_score(question, k=5)
    contexts = []
    for i, doc in enumerate(docs):
      contexts.append(f"Context {i}:\n{doc[0].page_content}")
      answer = llmChain.predict(question=question,
                                context="\n\n".join(contexts))
    return answer

  while True:
    question = input("Ask a question > ")
    answer = onMessage(question)
    print(answer)
