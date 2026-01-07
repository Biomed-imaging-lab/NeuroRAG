import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm
from unidecode import unidecode
from dotenv import load_dotenv
from datetime import datetime

import umap  # type: ignore
from sklearn.mixture import GaussianMixture
import chromadb

from langchain_core.documents import Document
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configuration
DOCUMENTS_DIR = Path(__file__).parent.parent / 'documents'
CACHE_DIR = Path(__file__).parent.parent / '.cache' / 'raptor'
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'documents')

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_CACHE_FILE = CACHE_DIR / 'embeddings_cache.pkl'
SUMMARIES_CACHE_FILE = CACHE_DIR / 'summaries_cache.pkl'
UPLOADED_PDFS_FILE = CACHE_DIR / f'uploaded_pdfs_{COLLECTION_NAME}.json'

# Chroma Cloud configuration
CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
CHROMA_TENANT = os.getenv('CHROMA_TENANT')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE', 'neurorag')
CHROMA_BATCH_SIZE = 250  # Max records per add operation (free tier limit is 300)

# RAPTOR parameters
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50
MAX_CLUSTER_LENGTH = 3500
REDUCTION_DIMENSION = 10
CLUSTERING_THRESHOLD = 0.1

# OpenAI configuration
OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'


# Cache Management
class Cache:
  """Simple file-based cache for expensive operations with statistics."""

  def __init__(self, cache_file: Path):
    self.cache_file = cache_file
    self.cache: dict[str, Any] = {}
    self.hits = 0
    self.misses = 0
    self.load()

  def load(self):
    """Load cache from disk."""
    if self.cache_file.exists():
      try:
        with open(self.cache_file, 'rb') as f:
          self.cache = pickle.load(f)
        print(f'Loaded cache from {self.cache_file.name} ({len(self.cache)} entries)')
      except Exception as e:
        print(f'Warning: Could not load cache from {self.cache_file.name}: {e}')
        self.cache = {}

  def save(self):
    """Save cache to disk."""
    try:
      with open(self.cache_file, 'wb') as f:
        pickle.dump(self.cache, f)
    except Exception as e:
      print(f'Warning: Could not save cache to {self.cache_file.name}: {e}')

  def get(self, key: str) -> Any | None:
    """Get value from cache."""
    value = self.cache.get(key)
    if value is not None:
      self.hits += 1
    else:
      self.misses += 1
    return value

  def set(self, key: str, value: Any):
    """Set value in cache."""
    self.cache[key] = value

  def __len__(self):
    return len(self.cache)

  def stats(self) -> str:
    """Get cache statistics."""
    total = self.hits + self.misses
    hit_rate = (self.hits / total * 100) if total > 0 else 0
    return f'{len(self)} entries, {self.hits} hits / {self.misses} misses ({hit_rate:.1f}% hit rate)'


def get_text_hash(text: str) -> str:
  """Generate a hash for text to use as cache key."""
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


# Initialize caches
embeddings_cache = Cache(EMBEDDINGS_CACHE_FILE)
summaries_cache = Cache(SUMMARIES_CACHE_FILE)


# Uploaded PDFs tracking
def load_uploaded_pdfs() -> dict[str, dict[str, Any]]:
  """Load the set of PDFs already uploaded to Chroma."""
  if UPLOADED_PDFS_FILE.exists():
    try:
      with open(UPLOADED_PDFS_FILE, 'r') as f:
        data = json.load(f)
      print(f'Loaded upload tracking: {len(data)} PDFs already in collection')
      return data
    except Exception as e:
      print(f'Warning: Could not load upload tracking file: {e}')
      return {}
  return {}


def save_uploaded_pdfs(uploaded_pdfs: dict[str, dict[str, Any]]):
  """Save the set of uploaded PDFs."""
  try:
    with open(UPLOADED_PDFS_FILE, 'w') as f:
      json.dump(uploaded_pdfs, f, indent=2)
  except Exception as e:
    print(f'Warning: Could not save upload tracking file: {e}')


def mark_pdf_uploaded(
  uploaded_pdfs: dict[str, dict[str, Any]],
  pdf_name: str,
  doc_count: int,
  start_id: int,
  end_id: int,
):
  """Mark a PDF as uploaded to Chroma."""
  uploaded_pdfs[pdf_name] = {
    'uploaded_at': datetime.now().isoformat(),
    'doc_count': doc_count,
    'start_id': start_id,
    'end_id': end_id,
    'collection': COLLECTION_NAME,
  }
  save_uploaded_pdfs(uploaded_pdfs)


def load_and_process_pdf(pdf_path: Path) -> list[Document]:
  """Load a PDF and preprocess its content."""
  docs = PDFMinerLoader(str(pdf_path), concatenate_pages=False).load()

  # Convert to ASCII to remove special characters
  for doc in docs:
    doc.page_content = unidecode(doc.page_content)

  return docs


def split_documents(docs: list[Document]) -> list[Document]:
  """Split documents into smaller chunks."""
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
    separators=['.', '\uff0e', '\u3002'],
  )

  splitted_docs = text_splitter.create_documents([doc.page_content for doc in docs])

  return splitted_docs


def summarize_text(llm: ChatOpenAI, context: str) -> str:
  """Summarize a given context using the LLM with caching."""
  # Check cache first
  cache_key = get_text_hash(context)
  cached_summary = summaries_cache.get(cache_key)

  if cached_summary is not None:
    return cached_summary

  # Generate summary if not in cache
  template = """Write a comprehensive summary of the following text, including as many key details as possible:

{context}

Summary:"""

  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | llm | StrOutputParser()

  try:
    summary = chain.invoke({'context': context})
    # Cache the result and save immediately
    summaries_cache.set(cache_key, summary)
    summaries_cache.save()
    return summary
  except Exception as e:
    print(f'Warning: Summarization failed: {e}')
    return context[:500]  # Fallback to truncated text


def get_text_from_docs(docs: list[Document]) -> str:
  """Concatenate document contents into a single text."""
  text = ''
  for doc in docs:
    text += f'{" ".join(doc.page_content.splitlines())}\n\n'
  return text


def get_cached_embedding(embedding_func: OpenAIEmbeddings, text: str) -> list[float]:
  """Get embedding with caching."""
  cache_key = get_text_hash(text)
  cached_embedding = embeddings_cache.get(cache_key)

  if cached_embedding is not None:
    return cached_embedding

  # Generate embedding if not in cache
  embedding = embedding_func.embed_query(text)
  embeddings_cache.set(cache_key, embedding)
  embeddings_cache.save()  # Save immediately

  return embedding


def get_cached_embeddings_batch(
  embedding_func: OpenAIEmbeddings, texts: list[str]
) -> list[list[float]]:
  """Get embeddings for a batch of texts with caching."""
  embeddings = []
  uncached_texts = []
  uncached_indices = []

  # Check cache for each text
  for i, text in enumerate(texts):
    cache_key = get_text_hash(text)
    cached_embedding = embeddings_cache.get(cache_key)

    if cached_embedding is not None:
      embeddings.append(cached_embedding)
    else:
      embeddings.append(None)  # Placeholder
      uncached_texts.append(text)
      uncached_indices.append(i)

  # Generate embeddings for uncached texts
  if uncached_texts:
    new_embeddings = embedding_func.embed_documents(uncached_texts)

    # Cache and insert new embeddings
    for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
      cache_key = get_text_hash(text)
      embeddings_cache.set(cache_key, embedding)
      embeddings[idx] = embedding

    # Save cache after batch
    embeddings_cache.save()

  return embeddings


def global_cluster_embeddings(
  embeddings: np.ndarray,
  dim: int,
  n_neighbors: int | None = None,
  metric: str = 'cosine',
  random_state: int = 42,
) -> np.ndarray:
  """Reduce embeddings dimensionality globally using UMAP (deterministic)."""
  # Ensure float64 for better numerical stability
  embeddings = embeddings.astype(np.float64)

  if n_neighbors is None:
    n_neighbors = int((len(embeddings) - 1) ** 0.5)

  # Ensure n_neighbors is valid
  n_neighbors = min(n_neighbors, len(embeddings) - 1)
  n_neighbors = max(2, n_neighbors)  # At least 2 neighbors

  reduced_embeddings = umap.UMAP(
    n_neighbors=n_neighbors,
    n_components=dim,
    metric=metric,
    random_state=random_state,  # Make it deterministic
    n_jobs=1,  # Required for reproducibility with random_state
  ).fit_transform(embeddings)

  return reduced_embeddings


def local_cluster_embeddings(
  embeddings: np.ndarray,
  dim: int,
  num_neighbors: int = 10,
  metric: str = 'cosine',
  random_state: int = 42,
) -> np.ndarray:
  """Reduce embeddings dimensionality locally using UMAP (deterministic)."""
  # Ensure float64 for better numerical stability
  embeddings = embeddings.astype(np.float64)

  # Ensure num_neighbors is valid
  num_neighbors = min(num_neighbors, len(embeddings) - 1)
  num_neighbors = max(2, num_neighbors)  # At least 2 neighbors

  reduced_embeddings = umap.UMAP(
    n_neighbors=num_neighbors,
    n_components=dim,
    metric=metric,
    random_state=random_state,  # Make it deterministic
    n_jobs=1,  # Required for reproducibility with random_state
  ).fit_transform(embeddings)

  return reduced_embeddings


def get_optimal_clusters(
  embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 0
) -> int:
  """Determine optimal number of clusters using BIC with error handling."""
  # Ensure float64 for better numerical stability
  embeddings = embeddings.astype(np.float64)

  max_clusters = min(max_clusters, len(embeddings))
  n_clusters = np.arange(1, max_clusters)
  bics = []

  for n in n_clusters:
    try:
      # Add regularization for numerical stability
      gm = GaussianMixture(
        n_components=n,
        random_state=random_state,
        reg_covar=1e-6,  # Regularization to prevent singular covariance
      )
      gm.fit(embeddings)
      bics.append(gm.bic(embeddings))
    except ValueError:
      # If GMM fails for this n, assign a high BIC (worse fit)
      bics.append(np.inf)
      continue

  # Filter out infinite BICs
  valid_bics = [(i, bic) for i, bic in enumerate(bics) if not np.isinf(bic)]

  if not valid_bics:
    # If all failed, return 1 cluster as fallback
    return 1

  optimal_idx = min(valid_bics, key=lambda x: x[1])[0]
  optimal_clusters = n_clusters[optimal_idx]

  return optimal_clusters


def gmm_cluster(embeddings: np.ndarray, threshold: float = 0.5, random_state: int = 0):
  """Perform Gaussian Mixture Model clustering with robust error handling."""
  # Ensure float64 for better numerical stability
  embeddings = embeddings.astype(np.float64)

  n_clusters = get_optimal_clusters(embeddings, random_state=random_state)

  # Try with regularization first
  try:
    gm = GaussianMixture(
      n_components=n_clusters, random_state=random_state, reg_covar=1e-6
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters
  except ValueError:
    # If it still fails, try with more regularization
    try:
      gm = GaussianMixture(
        n_components=max(1, n_clusters // 2),  # Reduce number of clusters
        random_state=random_state,
        reg_covar=1e-4,  # More regularization
      )
      gm.fit(embeddings)
      probs = gm.predict_proba(embeddings)
      labels = [np.where(prob > threshold)[0] for prob in probs]
      return labels, max(1, n_clusters // 2)
    except ValueError:
      # Last resort: single cluster
      labels = [np.array([0]) for _ in embeddings]
      return labels, 1


def perform_clustering(
  embeddings: np.ndarray, dim: int, threshold: float
) -> list[np.ndarray]:
  """Perform hierarchical clustering on embeddings with robust error handling."""
  try:
    # Global clustering
    reduced_embeddings_global = global_cluster_embeddings(
      embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = gmm_cluster(
      reduced_embeddings_global, threshold
    )
  except Exception as e:
    print(f'Warning: Clustering failed ({e}), falling back to single cluster')
    # Fallback: put all embeddings in one cluster
    return [np.array([0]) for _ in embeddings]

  # Local clustering within each global cluster
  all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
  total_clusters = 0

  for i in range(n_global_clusters):
    global_cluster_embeddings_ = embeddings[
      np.array([i in gc for gc in global_clusters])
    ]

    if len(global_cluster_embeddings_) == 0:
      continue

    if len(global_cluster_embeddings_) <= dim + 1:
      local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
      n_local_clusters = 1
    else:
      try:
        reduced_embeddings_local = local_cluster_embeddings(
          global_cluster_embeddings_, dim
        )
        local_clusters, n_local_clusters = gmm_cluster(
          reduced_embeddings_local, threshold
        )
      except Exception as e:
        print(f'Warning: Local clustering failed ({e}), using single cluster')
        local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
        n_local_clusters = 1

    # Assign local cluster labels
    for j in range(n_local_clusters):
      local_cluster_embeddings_ = global_cluster_embeddings_[
        np.array([j in lc for lc in local_clusters])
      ]
      indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
      for idx in indices:
        all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

    total_clusters += n_local_clusters

  return all_local_clusters


def perform_raptor_clustering(
  embedding_func: OpenAIEmbeddings,
  docs: list[Document],
  max_length_in_cluster: int = MAX_CLUSTER_LENGTH,
  reduction_dimension: int = REDUCTION_DIMENSION,
  threshold: float = CLUSTERING_THRESHOLD,
  pbar: tqdm | None = None,
  max_depth: int = 10,
  current_depth: int = 0,
) -> list[list[Document]]:
  """Recursively cluster documents using RAPTOR algorithm with depth limit."""
  # Check recursion depth limit
  if current_depth >= max_depth:
    print(f'Warning: Max recursion depth ({max_depth}) reached, stopping subdivision')
    return [docs]

  # Get embeddings (with progress tracking and caching)
  embeddings_list = []
  for doc in docs:
    embeddings_list.append(get_cached_embedding(embedding_func, doc.page_content))
    if pbar:
      pbar.update(1)

  embeddings = np.array(embeddings_list)

  # Perform clustering
  try:
    clusters = perform_clustering(
      embeddings, dim=reduction_dimension, threshold=threshold
    )
  except RecursionError:
    print(
      f'Warning: Clustering failed (recursion error), falling back to single cluster'
    )
    return [docs]

  doc_clusters = []

  # Process each cluster
  for label in np.unique(np.concatenate(clusters)):
    indices = [i for i, cluster in enumerate(clusters) if label in cluster]
    cluster_docs = [docs[i] for i in indices]

    # Base case: single document clusters
    if len(cluster_docs) == 1:
      doc_clusters.append(cluster_docs)
      continue

    # Calculate total text length
    total_length = sum(len(doc.page_content) for doc in cluster_docs)

    # Recursively recluster if needed
    if total_length > max_length_in_cluster and len(cluster_docs) > 4:
      doc_clusters.extend(
        perform_raptor_clustering(
          embedding_func,
          cluster_docs,
          max_length_in_cluster,
          reduction_dimension,
          threshold,
          pbar,
          max_depth,
          current_depth + 1,
        )
      )
    else:
      doc_clusters.append(cluster_docs)

  return doc_clusters


def generate_raptor_hierarchy(
  embedding_func: OpenAIEmbeddings, llm: ChatOpenAI, docs: list[Document]
) -> list[Document]:
  """Generate RAPTOR hierarchical document representation."""
  levels = [docs]

  print(f'Starting RAPTOR clustering with {len(docs)} initial documents...')

  level_num = 1
  while True:
    prev_level = levels[-1]

    # Stop if we've reached a small enough set
    if len(prev_level) <= 4:
      print(f'Reached top level with {len(prev_level)} documents')
      break

    print(f'\nProcessing level {level_num} with {len(prev_level)} documents...')

    # Cluster documents
    print('  Clustering documents...')
    # Create progress bar for embeddings
    with tqdm(
      total=len(prev_level), desc='  Computing embeddings', leave=False
    ) as pbar:
      clusters = perform_raptor_clustering(embedding_func, prev_level, pbar=pbar)

    print(f'  Created {len(clusters)} clusters')

    # Summarize each cluster
    print('  Summarizing clusters...')
    level_docs = []
    for cluster_docs in tqdm(clusters, desc='  Summarizing', leave=False):
      text = get_text_from_docs(cluster_docs)
      summary = summarize_text(llm, text)
      level_docs.append(Document(page_content=summary))

    levels.append(level_docs)
    level_num += 1

  # Flatten all levels into a single list
  all_docs = [doc for level in levels for doc in level]
  print(f'\nGenerated {len(all_docs)} total documents across {len(levels)} levels')

  return all_docs


def process_pdf_file(
  pdf_path: Path, embedding_func: OpenAIEmbeddings, llm: ChatOpenAI
) -> list[Document]:
  """Process a single PDF file through the RAPTOR pipeline."""
  print(f'\n{"=" * 80}')
  print(f'Processing: {pdf_path.name}')
  print(f'{"=" * 80}')

  # Load and preprocess
  print('Loading PDF...')
  docs = load_and_process_pdf(pdf_path)
  print(f'Loaded {len(docs)} pages')

  # Split into chunks
  print('Splitting into chunks...')
  splitted_docs = split_documents(docs)
  print(f'Created {len(splitted_docs)} chunks')

  # Generate RAPTOR hierarchy
  raptor_docs = generate_raptor_hierarchy(embedding_func, llm, splitted_docs)

  # Add metadata
  for doc in raptor_docs:
    doc.metadata = {'source': pdf_path.name}

  return raptor_docs


def main():
  """Main execution function."""
  print('RAPTOR Document Generator')
  print('=' * 80)

  # Validate required environment variables
  required_vars = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'CHROMA_API_KEY': CHROMA_API_KEY,
    'CHROMA_TENANT': CHROMA_TENANT,
  }

  missing_vars = [var for var, value in required_vars.items() if not value]
  if missing_vars:
    print(f'\nError: Missing required environment variables: {", ".join(missing_vars)}')
    print('Please set them in your .env file or environment.')
    return

  # Initialize OpenAI components
  print('\nInitializing OpenAI components...')
  embedding_func = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
  llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

  # Get all PDF files
  pdf_files = sorted(DOCUMENTS_DIR.glob('*.pdf'))

  if not pdf_files:
    print(f'No PDF files found in {DOCUMENTS_DIR}')
    return

  print(f'Found {len(pdf_files)} PDF files')

  # Initialize Chroma Cloud database
  print('\nInitializing Chroma Cloud database...')
  chroma_client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY,
  )

  # Get or create collection
  collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={'description': 'RAPTOR hierarchical documents for neuroscience books'},
  )

  print(f'Using collection: {COLLECTION_NAME}')
  print(f'Current document count: {collection.count()}')

  # Load uploaded PDFs tracking
  uploaded_pdfs = load_uploaded_pdfs()

  # Process each PDF
  doc_id_counter = collection.count()  # Start IDs from current count
  all_raptor_docs = []
  current_pdf_tracking = {}  # Track: pdf_name -> (start_doc_idx, doc_count)
  skipped_count = 0
  processed_count = 0

  for pdf_path in pdf_files:
    pdf_name = pdf_path.name

    # Skip if already uploaded
    if pdf_name in uploaded_pdfs:
      print(f'\n{"=" * 80}')
      print(f'Skipping (already uploaded): {pdf_name}')
      print(f'  Uploaded at: {uploaded_pdfs[pdf_name]["uploaded_at"]}')
      print(f'  Documents in Chroma: {uploaded_pdfs[pdf_name]["doc_count"]}')
      print(
        f'  ID range: {uploaded_pdfs[pdf_name]["start_id"]} - {uploaded_pdfs[pdf_name]["end_id"]}'
      )
      print(f'{"=" * 80}')
      skipped_count += 1
      continue

    try:
      raptor_docs = process_pdf_file(pdf_path, embedding_func, llm)

      # Track this PDF's documents
      start_idx = len(all_raptor_docs)
      all_raptor_docs.extend(raptor_docs)
      current_pdf_tracking[pdf_name] = (start_idx, len(raptor_docs))

      processed_count += 1
      print(f'  Generated {len(raptor_docs)} RAPTOR documents')
      print(f'  Embeddings cache: {embeddings_cache.stats()}')
      print(f'  Summaries cache: {summaries_cache.stats()}')

      # Periodically add to collection to manage memory
      if len(all_raptor_docs) >= 1000:
        print(f'\nAdding {len(all_raptor_docs)} documents to Chroma...')

        # Prepare batch data
        documents = [doc.page_content for doc in all_raptor_docs]
        metadatas = [doc.metadata for doc in all_raptor_docs]
        ids = [f'doc_{doc_id_counter + i}' for i in range(len(all_raptor_docs))]

        # Generate embeddings in batches (with caching)
        print('  Generating embeddings...')
        embeddings = []
        batch_size = 100
        for i in tqdm(
          range(0, len(documents), batch_size), desc='  Embedding batches', leave=False
        ):
          batch = documents[i : i + batch_size]
          embeddings.extend(get_cached_embeddings_batch(embedding_func, batch))

        # Add to collection in batches (respect Chroma Cloud quota limits)
        print('  Uploading to Chroma...')
        chroma_batch_size = CHROMA_BATCH_SIZE
        for i in tqdm(
          range(0, len(documents), chroma_batch_size),
          desc='  Upload batches',
          leave=False,
        ):
          batch_end = min(i + chroma_batch_size, len(documents))
          collection.add(
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
          )

        # Mark PDFs as uploaded after successful upload
        batch_start_id = doc_id_counter
        for pdf_name, (start_idx, doc_count) in current_pdf_tracking.items():
          pdf_start_id = batch_start_id + start_idx
          pdf_end_id = pdf_start_id + doc_count - 1
          mark_pdf_uploaded(
            uploaded_pdfs, pdf_name, doc_count, pdf_start_id, pdf_end_id
          )
          print(f'  ✓ Marked {pdf_name} as uploaded (IDs: {pdf_start_id}-{pdf_end_id})')

        doc_id_counter += len(all_raptor_docs)
        all_raptor_docs = []  # Clear memory
        current_pdf_tracking = {}  # Clear tracking

    except Exception as e:
      print(f'Error processing {pdf_path.name}: {e}')
      import traceback

      traceback.print_exc()
      continue

  # Add remaining documents
  if all_raptor_docs:
    print(f'\nAdding final {len(all_raptor_docs)} documents to Chroma...')

    documents = [doc.page_content for doc in all_raptor_docs]
    metadatas = [doc.metadata for doc in all_raptor_docs]
    ids = [f'doc_{doc_id_counter + i}' for i in range(len(all_raptor_docs))]

    # Generate embeddings (with caching)
    print('  Generating embeddings...')
    embeddings = []
    batch_size = 100
    for i in tqdm(
      range(0, len(documents), batch_size), desc='  Embedding batches', leave=False
    ):
      batch = documents[i : i + batch_size]
      embeddings.extend(get_cached_embeddings_batch(embedding_func, batch))

    # Add to collection in batches (respect Chroma Cloud quota limits)
    print('  Uploading to Chroma...')
    chroma_batch_size = CHROMA_BATCH_SIZE
    for i in tqdm(
      range(0, len(documents), chroma_batch_size), desc='  Upload batches', leave=False
    ):
      batch_end = min(i + chroma_batch_size, len(documents))
      collection.add(
        documents=documents[i:batch_end],
        embeddings=embeddings[i:batch_end],
        metadatas=metadatas[i:batch_end],
        ids=ids[i:batch_end],
      )

    # Mark remaining PDFs as uploaded after successful upload
    batch_start_id = doc_id_counter
    for pdf_name, (start_idx, doc_count) in current_pdf_tracking.items():
      pdf_start_id = batch_start_id + start_idx
      pdf_end_id = pdf_start_id + doc_count - 1
      mark_pdf_uploaded(uploaded_pdfs, pdf_name, doc_count, pdf_start_id, pdf_end_id)
      print(f'  ✓ Marked {pdf_name} as uploaded (IDs: {pdf_start_id}-{pdf_end_id})')

  print('\n' + '=' * 80)
  print('RAPTOR document generation complete!')
  print('Documents stored in Chroma Cloud')
  print(f'Tenant: {CHROMA_TENANT}')
  print(f'Database: {CHROMA_DATABASE}')
  print(f'Collection: {COLLECTION_NAME}')
  print(f'Total documents in collection: {collection.count()}')
  print('\nProcessing Summary:')
  print(f'  Total PDFs found: {len(pdf_files)}')
  print(f'  Already uploaded (skipped): {skipped_count}')
  print(f'  Newly processed: {processed_count}')
  print(f'  Total uploaded overall: {len(uploaded_pdfs)}')
  print('\nCache Statistics:')
  print(f'  Embeddings: {embeddings_cache.stats()}')
  print(f'  Summaries: {summaries_cache.stats()}')
  print('=' * 80)


if __name__ == '__main__':
  main()
