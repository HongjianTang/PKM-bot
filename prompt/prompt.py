from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
import faiss
import pickle


def runPrompt():
  index = faiss.read_index("training.index")

  with open("training/faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  with open("prompt/prompt.txt", "r") as f:
    promptTemplate = f.read()

  prompt = Prompt(template=promptTemplate,
                  input_variables=["context", "question"])

  llmChain = LLMChain(prompt=prompt,
                      llm=OpenAI(model_name='gpt-4', temperature=0.25))
  llm = 

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
