# 📘 Ask Your Book — Local AI-Powered Book QA with LangChain & Ollama

This project allows you to **ask questions about any book (PDF or TXT)** using **local language models** powered by **Ollama** and **LangChain**. No internet-based LLMs, no API keys — everything runs *completely offline* on your machine.

---

## 🧠 What It Does

1. Load your book (PDF or TXT).
2. Split the content into manageable chunks.
3. Embed and store the chunks in a local Chroma vector database.
4. Query the book using a local large language model like `llama3`, and get accurate answers!

---

## 🚀 Features

- Runs entirely **offline** using [Ollama](https://ollama.ai/)
- Supports `.pdf` and `.txt` formats
- Uses `ChromaDB` for fast local vector search
- Powered by `LangChain` and `HuggingFace` embeddings
- Simple and interactive command-line interface

---

## 🔧 Requirements

- Python 3.8+
- pip
- Git

---

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/ask-your-book.git
cd ask-your-book
````

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one with:

```txt
langchain
langchain-community
langchain-text-splitters
chromadb
pypdf
requests
```

3. **Install Ollama (if not already):**

Go to [https://ollama.ai](https://ollama.ai) and follow instructions for your OS.

Then in terminal:

```bash
ollama serve
```

4. **Download a model (like llama3):**

```bash
ollama pull llama3
```

---

## 📚 Usage

Run the script:

```bash
python ask_your_book.py
```

Follow the prompts:

1. Enter the path to your `.pdf` or `.txt` book file.
2. Wait while it processes and indexes your book.
3. Ask any questions about the book!
4. Type `exit` to quit.

---

## 💬 Example

```bash
Enter file path: ./books/alice_in_wonderland.pdf

❓ Ask a question about the book:
> Who is the main character?

🤖 Answer:
The main character is Alice, a curious young girl who falls down a rabbit hole...
```

---

## 📂 Chroma Vector Store

The vector store is saved in the `chroma_store/` directory. This allows reusing the indexed book without reprocessing it every time.

---

## ❓ Troubleshooting

* **Ollama not running?**
  Make sure you've started Ollama with:

  ```bash
  ollama serve
  ```

* **No models found?**
  Download a model with:

  ```bash
  ollama pull llama3
  ```

* **PDF not loading?**
  Ensure the file path is correct and readable.

---

## ✨ Credits

* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.ai/)
* [ChromaDB](https://github.com/chroma-core/chroma)
* [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 🛡 License

MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests and issues are welcome! If you'd like to improve this tool or add more features, feel free to contribute.

```

---

Let me know if you'd like this `README.md` content saved as a file or if you want me to generate a `requirements.txt` for you too.
```
