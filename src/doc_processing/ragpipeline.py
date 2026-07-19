"""
Main RAG pipeline handling retrieval to query
"""
import os
import re 
import json
from datetime import datetime
import platform
import torch
import torch.nn.functional as F
from groq import Groq
import config


class RAGPipeline():
    def __init__(self, late_chunking, api_key, vector_store, mcq_store=None, top_k=config.TOP_K):
        self.lc = late_chunking
        self.top_k = top_k
        self.vector_store = vector_store
        self.mcq_store = mcq_store
        self.chunk_embeddings = None
        self.use_local = False

        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1
                )
            except Exception:
                print("[RAG] Groq API key invalid, falling back to local model")
                self.use_local = True
        else:
            print("[RAG] No API key provided, falling back to local model")
            self.use_local = True
    
    @staticmethod
    def _is_noise_chunk(chunk):
        """Remove the toc section chunks"""
        lines = [l.strip() for l in chunk.strip().split('\n') if l.strip()]
        if not lines:
            return True
        
        toc_lines = sum(1 for l in lines if re.search(
            r'(\.\s*){3,}|\bSection\b.+\d+$|\d+\.\d+.+\d+$', l
        ))

        if toc_lines / len(lines) > 0.3: # 3o% as threshold
            return True
        
        if len(chunk.split()) < 30:
            return True

        return False

    def _is_noise_question(self, q):
        # mostly cross reference 
        noise_patterns = [
            r'\bpage\s+\d+\b',  # 'what page discusses...'
            r'bsection\s+\d+\.\d+\b', # 'which section...' 
            r'\bchapter\s+\d+\b', # 'see chapter...'
            r'table of contents',
            r'discusses.{0,30}topic',
        ]
        text = (q["question"] + " " + q["explanation"]).lower() 
        return any(re.search(p, text, re.IGNORECASE) for p in noise_patterns)
     

    def index(self, pages, pdf_path):
        """
        Index a document into the vector store.

        Args:
            pages   : list of (page_num, text) tuples from Extraction.extract_text().
                      Pass None if the document is already known to be indexed
                      (the worker pre-checks this before calling index()).
            pdf_path: path to the source PDF, used as the collection key.
        """
        self.pdf_path = pdf_path

        if self.vector_store.is_indexed(pdf_path):
            # Document already indexed and stored in database.
            print(f"[RAGPipeline] '{pdf_path}' already indexed — loading from DB.")
            self.lc.chunks, self.lc.chunk_pages = self.vector_store.load(pdf_path)
        else:
            # Not in database index.
            if pages is None:
                raise ValueError(
                    f"pages=None but '{pdf_path}' is not in the vector store. "
                    "Pass the extracted page list to index a new document."
                )
            pages = [(p, t) for p, t in pages if not self._is_noise_chunk(t)]
            self.chunk_embeddings = self.lc.run(pages)
            self.vector_store.store(pdf_path, self.lc.chunks, self.chunk_embeddings, self.lc.chunk_pages)

    def _embed_query(self, query):
        prefixed_query = "search_query: " + query
        tokens = self.lc.tokenizer(
            prefixed_query,
            return_tensors="pt",
            return_offsets_mapping=False,
            truncation=True,
            max_length=8192
        )
        tokens = {k: v.to(self.lc.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.lc.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
        return outputs.last_hidden_state[0].mean(dim=0)

    def _retrieve(self, query):
        query_embedding = self._embed_query(query).cpu().float().tolist()
        return self.vector_store.query(self.pdf_path, query_embedding, self.top_k) 
    
    
    def _query_llm(self, prompt):
        if self.use_local:
            import ollama
            response = ollama.chat(
                model=config.LOCAL_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        else:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    def query_qna(self, question):
        retrieved_chunks = self._retrieve(question)
        context = '\n\n'.join(
            f"[{self.lc.format_page_citation(page)}] {chunk}" 
            for chunk, page in retrieved_chunks
        )

        prompt = f"""
        You are a helpful assistant. Answer the question based strictly on the context provided below.
        Each context chunk is prefixed with its page number like [N]where is N is the page number.
        Always cite the page number(s) you used at the end of your answer like: 
        (Source Page: [5]) and in case of multi page (Source Page: [4][5]  ...)
        Context: {context}
        Question: {question}
        Answer:
        """
        return self._query_llm(prompt)

                
                
    def query_mcq(self, question, num_questions=5, save_json=False, output_dir="mcq_output"):
        retrieved_chunks = self._retrieve(question)
        context = '\n\n'.join(
            f"[{self.lc.format_page_citation(page)}] {chunk}" 
            for chunk, page in retrieved_chunks
        )

        prompt = f"""
        You are a helpful assistant. Based only on the context provided below, generate {num_questions} multiple choice questions, each with EXACTLY 4 options labeled A), B), C), D) — no more, no less. Never use any other format and never should options be more than 4 and in explanation also mention difficulty.
        Each context chunk is prefixed with its page number like [N] where N is the page number.
        At the end of each explanation cite the page like: 
        (Source Page: [5]) and in case of multi page (Source Page: [4][5]  ...)

        Context: {context}

        Topic to generate MCQ about: {question}

        Format your response exactly like this:
        Question: <question here> 
        A) <option>          
        B) <option>
        C) <option>           
        D) <option>
        Correct Answer: <letter>
        Explanation: [Easy/Medium/Hard] <brief explanation based on context>
        """
        raw_response =  self._query_llm(prompt)

        if save_json:
            parsed   = self._parse_mcq_response(raw_response, question)
            unique_qs = self._deduplicate_mcqs(parsed["questions"])

            unique_qs = [q for q in unique_qs if not self._is_noise_question(q)]
            if self.mcq_store and unique_qs:
                self.mcq_store.store(self.pdf_path, unique_qs)

            # strip embeddings before writing to JSON
            for q in unique_qs:
                q.pop("embedding", None)

            parsed["questions"] = unique_qs
            self._save_mcq_json(parsed, question, output_dir)
            
        return raw_response

    def _parse_mcq_response(self, response_text, topic):
        """
        Parse the LLM's MCQ text output into a list of structured dicts.

        Each dict has:
            question      : str
            options       : { "A": str, "B": str, "C": str, "D": str }
            correct_answer: str  (e.g. "A")
            difficulty    : str  (Easy / Medium / Hard)
            explanation   : str
            source_pages  : list[str]
            embedding     : list[float]  — stripped before JSON save
        """
        blocks = re.split(r'\n(?=(?:\d+\.\s*)?Question:)', response_text.strip())
        questions = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            q = {}

            # Questions
            q_match = re.search(r'(?:\d+\.\s*)?Question:\s*(.+?)(?=\nA\))', block, re.DOTALL)
            q["question"] = q_match.group(1).strip() if q_match else ""
            q["options"] = {}
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                for letter in "ABCD":
                    if line.startswith(f"{letter})"):
                        q["options"][letter] = line[2:].strip()
                        break
            # Correct answer
            ans_match = re.search(r'Correct Answer:\s*([A-D])', block)
            q["correct_answer"] = ans_match.group(1).strip() if ans_match else ""

            # Difficulty + explanation 
            exp_match = re.search(
                r'Explanation:\s*\[(Easy|Medium|Hard)\]\s*(.+?)(?=\nQuestion:|\Z)',
                block, re.DOTALL
            )
            if exp_match:
                q["difficulty"]  = exp_match.group(1).strip()
                q["explanation"] = exp_match.group(2).strip()
            else:
                # fallback: explanation present but no bracketed difficulty tag
                exp_fallback = re.search(r'Explanation:\s*(.+?)(?=\nQuestion:|\Z)', block, re.DOTALL)
                q["difficulty"]  = ""
                q["explanation"] = exp_fallback.group(1).strip() if exp_fallback else ""

            # Source pages
            q["source_pages"] = re.findall(r'\[(\d+)\]', q["explanation"])

            # Embedding for dedup (stripped before JSON save) 
            if q["question"]:
                emb = self._embed_query(q["question"])
                q["embedding"] = emb.cpu().float().tolist()
                questions.append(q)

        return {
            "topic"       : topic,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pdf_path"    : getattr(self, "pdf_path", ""),
            "questions"   : questions,
        }

    def _deduplicate_mcqs(self, questions, similarity_threshold=0.92):
        """
        Filters out questions that are duplicates of already-stored ones.
        Two-stage check:
        1. Option fingerprint — fast exact hash match
        2. Cosine similarity on question embedding — catches paraphrases
        Returns the list of new (non-duplicate) questions only.
        """
        import torch
        import torch.nn.functional as F

        if not self.mcq_store:
            return questions

        stored_embeddings, stored_fingerprints = self.mcq_store.get_stored_embeddings(self.pdf_path)

        # convert stored embeddingsd to tensors once
        stored_tensors = [
            torch.tensor(e, dtype=torch.float32).to(self.lc.device)
            for e in stored_embeddings
        ]

        unique = []
        # also track embeddings added in THIS batch so we don't dupe within the same run
        batch_tensors      = []
        batch_fingerprints = set()

        for q in questions:
            fp = self.mcq_store.option_fingerprint(q["options"], q["correct_answer"])

            # Stage 1: option fingerprint
            if fp in stored_fingerprints or fp in batch_fingerprints:
                print(f"[Dedup] Skipped (fingerprint match): {q['question'][:60]}")
                continue

            # Stage 2: semantic similarity 
            q_tensor = torch.tensor(q["embedding"], dtype=torch.float32).to(self.lc.device)
            is_duplicate = False

            for stored_t in stored_tensors + batch_tensors:
                score = F.cosine_similarity(q_tensor.unsqueeze(0), stored_t.unsqueeze(0)).item()
                if score >= similarity_threshold:
                    print(f"[Dedup] Skipped (similarity {score:.2f}): {q['question'][:60]}")
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(q)
                batch_tensors.append(q_tensor)
                batch_fingerprints.add(fp)

        print(f"[Dedup] {len(unique)} unique / {len(questions)} total questions kept")
        return unique

    def _save_mcq_json(self, parsed, topic, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')[:50]
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath   = os.path.join(output_dir, f"mcq_{safe_topic}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"[RAGPipeline] MCQ saved → {filepath}")
        return filepath
