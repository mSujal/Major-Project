"""
Main RAG pipeline handling retrieval to query
"""
import os
import re
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from groq import Groq

import config
from src.prompts.prompts import build_mcq_prompt


class RAGPipeline:
    def __init__(
        self,
        late_chunking,
        api_key,
        vector_store,
        mcq_store=None,
        top_k=config.TOP_K,
        taxonomy_path=config.TAXONOMY_PATH,
    ):
        self.lc = late_chunking
        self.top_k = top_k
        self.vector_store = vector_store
        self.mcq_store = mcq_store
        self.chunk_embeddings = None
        self.use_local = False

        # Load taxonomy
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)

        # Groq client
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
            except Exception:
                print("[RAG] Groq API key invalid, falling back to local model")
                self.use_local = False
        else:
            print("[RAG] No API key provided, falling back to local model")
            self.use_local = False

    @staticmethod
    def _is_noise_chunk(chunk):
        """Remove table-of-contents or very short chunks."""
        lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
        if not lines:
            return True

        toc_lines = sum(
            1
            for l in lines
            if re.search(r"(\.\s*){3,}|\bSection\b.+\d+$|\d+\.\d+.+\d+$", l)
        )

        if toc_lines / len(lines) > 0.3:  # 30% threshold
            return True

        if len(chunk.split()) < 30:
            return True

        return False

    def _is_noise_question(self, q):
        """Filter out questions that are cross‑reference noise."""
        noise_patterns = [
            r"\bpage\s+\d+\b",
            r"\bsection\s+\d+\.\d+\b",
            r"\bchapter\s+\d+\b",
            r"table of contents",
            r"discusses.{0,30}topic",
        ]
        text = (q["question"] + " " + q["explanation"]).lower()
        return any(re.search(p, text, re.IGNORECASE) for p in noise_patterns)

    def index(self, pages, pdf_path):
        """
        Index a document into the vector store.

        Args:
            pages: list of (page_num, text) tuples from Extraction.extract_text().
            pdf_path: path to the source PDF, used as the collection key.
        """
        self.pdf_path = pdf_path

        if self.vector_store.is_indexed(pdf_path):
            print(f"[RAGPipeline] '{pdf_path}' already indexed — loading from DB.")
            self.lc.chunks, self.lc.chunk_pages = self.vector_store.load(pdf_path)
        else:
            if pages is None:
                raise ValueError(
                    f"pages=None but '{pdf_path}' is not in the vector store. "
                    "Pass the extracted page list to index a new document."
                )
            pages = [(p, t) for p, t in pages if not self._is_noise_chunk(t)]
            self.chunk_embeddings = self.lc.run(pages)
            self.vector_store.store(
                pdf_path, self.lc.chunks, self.chunk_embeddings, self.lc.chunk_pages
            )

    def _embed_query(self, query):
        """Embed a single query using the late‑chunking model."""
        prefixed_query = "search_query: " + query
        tokens = self.lc.tokenizer(
            prefixed_query,
            return_tensors="pt",
            return_offsets_mapping=False,
            truncation=True,
            max_length=8192,
        )
        tokens = {k: v.to(self.lc.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.lc.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
        return outputs.last_hidden_state[0].mean(dim=0)

    def _retrieve(self, query, pdf_path):
        """Retrieve top‑k context chunks for a query."""
        query_embedding = self._embed_query(query).cpu().float().tolist()
        return self.vector_store.query(pdf_path, query_embedding, self.top_k)

    def _query_llm(self, prompt):
        """Call the LLM (Groq or local) with the given prompt."""
        # if self.use_local:
        #     import ollama
        #     response = ollama.chat(
        #         model=config.LOCAL_MODEL,
        #         messages=[{"role": "user", "content": prompt}]
        #     )
        #     return response["message"]["content"]
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def query_qna(self, question, pdf_path):
        """Ask a free‑text question and get an answer with citations."""
        retrieved_chunks = self._retrieve(question, pdf_path)
        context = "\n\n".join(
            f"[{self.lc.format_page_citation(page)}] {chunk}"
            for chunk, page in retrieved_chunks
        )

        prompt = f"""
        You are a helpful assistant. Answer the question based strictly on the context provided below.
        Each context chunk is prefixed with its page number like [N] where N is the page number.
        Always cite the page number(s) you used at the end of your answer like:
        (Source Page: [5]) and in case of multi page (Source Page: [4][5] ...)
        Context: {context}
        Question: {question}
        Answer:
        """
        return self._query_llm(prompt)

    def _map_topic(self, query):
        """
        Find the best matching topic key from the taxonomy based on the query.
        Uses simple substring matching.
        """
        query_lower = query.lower()

        for key in self.taxonomy["topics"].keys():
            key_words = key.replace("_", " ")
            if key_words in query_lower or query_lower in key_words:
                return key

        return None  # fallback if no match found

    def _filter_low_quality_questions(self, questions: list, allowed_levels: list = None) -> list:
        """
        Filter out low‑quality MCQs:
        - Reject trivial page‑referencing questions.
        - Reject questions whose Bloom's level is not in the allowed set.
        - Reject questions that are too short or have very short explanations.
        """
        if not questions:
            return []

        # Patterns that indicate low‑quality / trivia
        reject_patterns = [
            r"on which page",
            r"what page",
            r"which page",
            r"who is the author",
            r"how many pages",
            r"what is the name of",
            r"list of (?:all|the) methods",
            r"recalling the list",
            r"^what is the primary purpose of$",   # too generic
        ]

        filtered = []

        for q in questions:
            text = (q.get("question", "") + " " + q.get("explanation", "")).lower()

            # 1. Reject if it matches any noise pattern
            if any(re.search(p, text, re.IGNORECASE) for p in reject_patterns):
                print(f"[Filter] Rejected low‑quality: {q['question'][:50]}...")
                continue

            # 2. Reject if explanation is too short (less than 10 words) – indicates poor quality
            if len(q.get("explanation", "").split()) < 10:
                print(f"[Filter] Rejected (explanation too short): {q['question'][:50]}...")
                continue

            # 3. Check Bloom's level if allowed_levels is provided
            if allowed_levels:
                bloom_match = re.search(r'Bloom:\s*(\w+)', q.get("explanation", ""))
                if bloom_match:
                    bloom_level = bloom_match.group(1).capitalize()
                    if bloom_level not in allowed_levels:
                        print(f"[Filter] Rejected (wrong Bloom level: {bloom_level}): {q['question'][:50]}...")
                        continue
                else:
                    # No Bloom tag – treat as invalid
                    print(f"[Filter] Rejected (missing Bloom tag): {q['question'][:50]}...")
                    continue

            # 4. Reject questions shorter than 10 words (too vague)
            if len(q.get("question", "").split()) < 10:
                print(f"[Filter] Rejected (question too short): {q['question'][:50]}...")
                continue

            filtered.append(q)

        print(f"[Filter] Kept {len(filtered)} / {len(questions)} questions")
        return filtered

    def query_mcq(self, query, pdf_path, num_questions=5, save_json=False, output_dir="mcq_output"):
        """
        Generate multiple‑choice questions for a given topic, using taxonomy‑driven prompts.
        """
        # 1. Map query to taxonomy topic
        topic_key = self._map_topic(query)
        if topic_key is None:
            print(f"[RAG] No taxonomy match for '{query}', using generic fallback")
            taxonomy_entry = {"levels": ["Remember", "Understand"], "patterns": ["General questions"]}
            subject = "the course"
            retrieval_query = query
        else:
            taxonomy_entry = self.taxonomy["topics"][topic_key]
            subject = self.taxonomy.get("subject", "the course")
            retrieval_query = topic_key.replace("_", " ")
            print(f"[RAG] Mapped query to topic: '{topic_key}'")

        # 2. Retrieve context from vector store using the mapped topic for better results
        retrieved_chunks = self._retrieve(retrieval_query, pdf_path)
        context = "\n\n".join(
            f"[{self.lc.format_page_citation(page)}] {chunk}"
            for chunk, page in retrieved_chunks
        )

        # 3. Build prompt using the generic prompt builder
        prompt = build_mcq_prompt(
            topic=retrieval_query,
            taxonomy=taxonomy_entry,
            context=context,
            num_questions=num_questions,
            subject=subject,
        )

        # 4. Call LLM
        raw_response = self._query_llm(prompt)

        # 5. Parse, filter, deduplicate
        parsed = self._parse_mcq_response(raw_response, retrieval_query, pdf_path)

        # Apply the filter with allowed levels from taxonomy
        allowed = taxonomy_entry.get("levels", None)
        parsed["questions"] = self._filter_low_quality_questions(parsed["questions"], allowed)

        unique_qs = self._deduplicate_mcqs(parsed["questions"], pdf_path)

        # Remove embeddings before JSON save
        for q in unique_qs:
            q.pop("embedding", None)
        parsed["questions"] = unique_qs

        if save_json:
            self._save_mcq_json(parsed, retrieval_query, output_dir)

        return raw_response

    def _parse_mcq_response(self, response_text, topic, pdf_path):
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
        blocks = re.split(r"\n(?=(?:\d+\.\s*)?Question:)", response_text.strip())
        questions = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            q = {}

            # Question text
            q_match = re.search(r"(?:\d+\.\s*)?Question:\s*(.+?)(?=\nA\))", block, re.DOTALL)
            q["question"] = q_match.group(1).strip() if q_match else ""

            # Options
            q["options"] = {}
            lines = block.split("\n")
            for line in lines:
                line = line.strip()
                for letter in "ABCD":
                    if line.startswith(f"{letter})"):
                        q["options"][letter] = line[2:].strip()
                        break

            # Correct answer
            ans_match = re.search(r"Correct Answer:\s*([A-D])", block)
            q["correct_answer"] = ans_match.group(1).strip() if ans_match else ""

            # Explanation and difficulty
            exp_match = re.search(
                r"Explanation:\s*\[(Easy|Medium|Hard)\]\s*(.+?)(?=\nQuestion:|\Z)",
                block,
                re.DOTALL,
            )
            if exp_match:
                q["difficulty"] = exp_match.group(1).strip()
                q["explanation"] = exp_match.group(2).strip()
            else:
                # fallback: no bracket
                exp_fallback = re.search(r"Explanation:\s*(.+?)(?=\nQuestion:|\Z)", block, re.DOTALL)
                q["difficulty"] = ""
                q["explanation"] = exp_fallback.group(1).strip() if exp_fallback else ""

            # Source pages
            q["source_pages"] = re.findall(r"\[(\d+)\]", q["explanation"])

            # Embedding for dedup
            if q["question"]:
                emb = self._embed_query(q["question"])
                q["embedding"] = emb.cpu().float().tolist()
                questions.append(q)

        return {
            "topic": topic,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pdf_path": pdf_path,
            "questions": questions,
        }

    def _deduplicate_mcqs(self, questions, pdf_path, similarity_threshold=0.92):
        """
        Filters out questions that are duplicates of already‑stored ones.
        Two‑stage check:
        1. Option fingerprint — fast exact hash match
        2. Cosine similarity on question embedding — catches paraphrases
        """
        if not self.mcq_store:
            return questions

        stored_embeddings, stored_fingerprints = self.mcq_store.get_stored_embeddings(pdf_path)

        stored_tensors = [
            torch.tensor(e, dtype=torch.float32).to(self.lc.device)
            for e in stored_embeddings
        ]

        unique = []
        batch_tensors = []
        batch_fingerprints = set()

        for q in questions:
            fp = self.mcq_store.option_fingerprint(q["options"], q["correct_answer"])

            # Stage 1: exact match
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
        """Save the generated MCQs to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        safe_topic = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"mcq_{safe_topic}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"[RAGPipeline] MCQ saved → {filepath}")
        return filepath