"""
Late chunking Implementation
"""

import os
import re
import bisect

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class LateChunking:
    def __init__(self, model_name, tokenizer_name, chunk_size=384, chunk_overlap=50,
                 window_size=8192, window_overlap=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.device = config.DEVICE
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_num_threads(os.cpu_count())

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

        self.window_size = window_size
        self.window_overlap = window_overlap
        self.window_stride = window_size - window_overlap

        self.chunks = []
        self.chunk_pages = []

    def _chunk(self, pages):
        """
        Split each page's text into chunks, tracking which page(s) each
        chunk originated from via inline <PAGE_N> markers.
        """
        self.chunks = []
        self.chunk_pages = []

        marked_pages = [(page_num, f"<PAGE_{page_num}> {text}") for page_num, text in pages]

        for page_num, marked_text in marked_pages:
            splits = self.splitter.split_text(marked_text)

            for split in splits:
                # find every page marker present in this chunk
                markers = re.findall(r'<PAGE_(\d+)>', split)

                if markers:
                    page_nums_in_chunk = [int(m) for m in markers]
                    page_start = page_nums_in_chunk[0]
                    page_end = page_nums_in_chunk[-1]
                else:
                    # fallback: no marker survived splitting, use the source page
                    page_start = page_num
                    page_end = page_num

                clean_split = re.sub(r'<PAGE_\d+>\s*', '', split).strip()

                if clean_split:
                    self.chunks.append(clean_split)
                    self.chunk_pages.append({"page_start": page_start, "page_end": page_end})

    def _tokenize_full(self, corpus):
        """Tokenize the full corpus (with retrieval prefix) in one pass, no truncation."""
        prefixed = "search_document: " + corpus
        encoded = self.tokenizer(
            prefixed,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["offset_mapping"]

    def _find_token_boundaries(self, corpus, offset_mapping):
        """
        Map each text chunk to its (start_token, end_token) index range within
        the fully tokenized corpus, so embeddings can later be pooled per chunk.
        """
        prefix_len = len("search_document: ")

        chunk_boundaries = []
        current_pos = 0
        for chunk in self.chunks:
            start_char = corpus.find(chunk, current_pos)
            if start_char == -1:
                start_char = corpus.find(chunk)
            end_char = start_char + len(chunk)
            chunk_boundaries.append((start_char + prefix_len, end_char + prefix_len))
            current_pos = end_char

        offsets = offset_mapping[0]
        # build lookup arrays once, then binary search per chunk
        tok_starts = [offsets[i][0].item() for i in range(len(offsets))]
        tok_ends = [offsets[i][1].item() for i in range(len(offsets))]

        token_boundaries = []
        for char_start, char_end in chunk_boundaries:
            # first token whose start >= char_start
            tok_start = bisect.bisect_left(tok_starts, char_start)
            if tok_start >= len(tok_starts):
                tok_start = 1
            # last token whose end <= char_end
            tok_end = bisect.bisect_right(tok_ends, char_end) - 1
            if tok_end < 0:
                tok_end = tok_start
            # skip special tokens (start=0, end=0)
            while tok_start < len(tok_starts) and tok_starts[tok_start] == 0 and tok_ends[tok_start] == 0:
                tok_start += 1
            token_boundaries.append((tok_start, tok_end))

        return token_boundaries

    def _embed_windowed(self, input_ids, token_boundaries):
        """
        Run the model over sliding windows of the full token sequence and
        mean-pool each chunk's token embeddings from whichever window covers it.
        """
        T = input_ids.shape[1]
        chunk_embeddings = [None] * len(token_boundaries)

        starts = list(range(0, max(T - self.window_size + 1, 1), self.window_stride))
        if starts[-1] + self.window_size < T:
            starts.append(max(0, T - self.window_size))

        for win_start in starts:
            win_end = min(win_start + self.window_size, T)
            win_ids = input_ids[:, win_start:win_end].to(self.device)
            win_mask = torch.ones_like(win_ids)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=win_ids,
                    attention_mask=win_mask,
                )
            win_token_embs = outputs.last_hidden_state[0]

            for ci, (tok_start, tok_end) in enumerate(token_boundaries):
                if tok_end < win_start or tok_start >= win_end:
                    continue

                local_start = max(tok_start, win_start) - win_start
                local_end = min(tok_end, win_end) - win_start

                if local_start > local_end:
                    continue

                chunk_emb = win_token_embs[local_start: local_end + 1].mean(dim=0)

                if chunk_embeddings[ci] is None:
                    chunk_embeddings[ci] = chunk_emb

        hidden = self.model.config.hidden_size
        for ci in range(len(chunk_embeddings)):
            if chunk_embeddings[ci] is None:
                print(f"[LateChunking] WARNING: chunk {ci} has no embedding — using zeros")
                chunk_embeddings[ci] = torch.zeros(hidden, device=self.device)

        return chunk_embeddings

    def run(self, pages):
        """
        Full pipeline: chunk the pages, tokenize the whole corpus, embed it in
        sliding windows, and pool per-chunk embeddings from token boundaries.

        Args:
            pages: list of (page_num, text) tuples

        Returns:
            list of per-chunk embedding tensors (self.chunk_embeddings)
        """
        corpus = "\n\n".join(text for _, text in pages)

        self._chunk(pages)
        print(f"[LateChunking] {len(self.chunks)} chunks from {len(pages)} pages")

        input_ids, offset_mapping = self._tokenize_full(corpus)
        T = input_ids.shape[1]
        n_windows = max(1, (T - self.window_overlap - 1) // self.window_stride + 1)
        print(f"[LateChunking] {T} tokens → {n_windows} window(s) "
              f"(size={self.window_size}, overlap={self.window_overlap})")

        token_boundaries = self._find_token_boundaries(corpus, offset_mapping)

        self.chunk_embeddings = self._embed_windowed(input_ids, token_boundaries)

        print(f"[LateChunking] embedded {len(self.chunk_embeddings)} chunks")

        return self.chunk_embeddings

    @staticmethod
    def format_page_citation(page_info):
        """
        Helper for downstream citation formatting.
        Accepts either a dict {"page_start", "page_end"}, a (start, end) tuple,
        or a legacy single int/page value.
        """
        if isinstance(page_info, dict):
            start = page_info["page_start"]
            end = page_info["page_end"]
        elif isinstance(page_info, tuple):
            start, end = page_info
        else:
            return f"Page: {page_info}"

        if start == end:
            return f"Page: {start}"
        return f"Pages: {start}-{end}"