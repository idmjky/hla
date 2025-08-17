"""
RNA Translation Efficiency Prediction Pipeline
==============================================
Predicts translation efficiency in 6 tissues using RNA sequence embeddings

Key Features:
- Uses AIDO.RNA model to generate sequence embeddings
- Handles variable-length RNA sequences via average pooling
- Tests multiple ML models (linear, tree-based, neural networks)
- Comprehensive evaluation with regression metrics
- Advanced visualizations for model comparison
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set PyTorch CUDA memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Data processing
from sklearn.model_selection import cross_val_score, KFold, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

# Neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#=====================================
# 1. EMBEDDING GENERATION
#=====================================

def inspect_csv_structure(csv_path):
    """Inspect CSV file structure to help debug data issues"""
    print(f"\nInspecting CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        if 'sequence' in df.columns:
            print(f"\nSequence column info:")
            print(f"  Non-null count: {df['sequence'].count()}")
            print(f"  Null count: {df['sequence'].isnull().sum()}")
            print(f"  Unique values: {df['sequence'].nunique()}")
            print(f"  Sample values:")
            for i, val in enumerate(df['sequence'].head(5)):
                print(f"    Row {i}: {repr(val)} (type: {type(val)})")
        else:
            print("Warning: 'sequence' column not found!")
            
    except Exception as e:
        print(f"Error reading CSV: {e}")

def check_gpu_availability():
    """Check and report GPU availability and memory"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU available: {gpu_count} device(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        return True
    else:
        print("No GPU available. Using CPU (this may be slow for large datasets)")
        return False

def estimate_safe_batch_size(gpu_memory_gb, sequence_length_estimate=1000):
    """Estimate safe batch size based on GPU memory and sequence length"""
    # Rough estimation: each sequence embedding takes ~sequence_length * 1280 * 4 bytes (float32)
    # Plus some overhead for model parameters and intermediate computations
    embedding_size_per_seq = sequence_length_estimate * 1280 * 4  # bytes
    model_overhead = 2 * 1024**3  # 2GB for model parameters and intermediate tensors
    
    available_memory = (gpu_memory_gb * 1024**3) - model_overhead
    safe_batch_size = max(1, int(available_memory / (embedding_size_per_seq * 2)))  # *2 for safety margin
    
    return min(safe_batch_size, 32)  # Cap at 32

def categorize_sequences_by_length(sequences, transcript_ids, ensembl_ids, df):
    """Categorize sequences by length for optimal processing strategy"""
    print("Categorizing sequences by length...")
    
    # Calculate sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq_lengths)
    avg_length = np.mean(seq_lengths)
    
    print(f"Sequence length statistics:")
    print(f"  Average: {avg_length:.1f}")
    print(f"  Maximum: {max_length}")
    print(f"  Minimum: {min(seq_lengths)}")
    
    # Define length categories
    short_threshold = 5000    # < 5k: GPU normal batch
    medium_threshold = 20000  # 5k-20k: GPU reduced batch  
    long_threshold = 50000    # 20k-50k: CPU processing
    # > 50k: Skip entirely
    
    # Categorize sequences
    categories = {
        'short': {'indices': [], 'sequences': [], 'transcript_ids': [], 'ensembl_ids': [], 'df_indices': []},
        'medium': {'indices': [], 'sequences': [], 'transcript_ids': [], 'ensembl_ids': [], 'df_indices': []},
        'long': {'indices': [], 'sequences': [], 'transcript_ids': [], 'ensembl_ids': [], 'df_indices': []},
        'very_long': {'indices': [], 'sequences': [], 'transcript_ids': [], 'ensembl_ids': [], 'df_indices': []}
    }
    
    for i, (seq, length) in enumerate(zip(sequences, seq_lengths)):
        if length < short_threshold:
            cat = 'short'
        elif length < medium_threshold:
            cat = 'medium'
        elif length < long_threshold:
            cat = 'long'
        else:
            cat = 'very_long'
        
        categories[cat]['indices'].append(i)
        categories[cat]['sequences'].append(seq)
        categories[cat]['transcript_ids'].append(transcript_ids[i])
        categories[cat]['ensembl_ids'].append(ensembl_ids[i])
        categories[cat]['df_indices'].append(i)
    
    # Print summary
    print(f"\nSequence categorization:")
    print(f"  Short (<{short_threshold}): {len(categories['short']['sequences'])} sequences")
    print(f"  Medium ({short_threshold}-{medium_threshold}): {len(categories['medium']['sequences'])} sequences")
    print(f"  Long ({medium_threshold}-{long_threshold}): {len(categories['long']['sequences'])} sequences")
    print(f"  Very long (>{long_threshold}): {len(categories['very_long']['sequences'])} sequences")
    
    if categories['very_long']['sequences']:
        print(f"\nWarning: {len(categories['very_long']['sequences'])} very long sequences will be skipped")
        print(f"  Longest sequence: {max_length} characters")
    
    return categories

def process_sequence_category(model, category_data, device, batch_size, pooling, category_name):
    """Process a category of sequences with appropriate strategy"""
    print(f"\nProcessing {category_name} sequences ({len(category_data['sequences'])} sequences)...")
    
    sequences = category_data['sequences']
    all_embeddings = []
    
    # Adjust batch size based on category
    if category_name == 'short':
        effective_batch_size = max(1, batch_size // 2)  # More conservative for short sequences
    elif category_name == 'medium':
        effective_batch_size = max(1, batch_size // 4)
    elif category_name == 'long':
        effective_batch_size = max(1, batch_size // 8)
    else:
        effective_batch_size = 1
    
    print(f"  Using batch size: {effective_batch_size}")
    
    # Track OOM occurrences for dynamic batch size reduction
    oom_count = 0
    max_oom_before_reduction = 3
    
    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), effective_batch_size), desc=f"Processing {category_name}"):
        try:
            batch_seqs = sequences[i:i+effective_batch_size]
            transformed_batch = model.transform({"sequences": batch_seqs})
            
            # Move to appropriate device
            if isinstance(transformed_batch, dict):
                transformed_batch = {k: v.to(device) if hasattr(v, 'to') else v 
                                   for k, v in transformed_batch.items()}
            elif hasattr(transformed_batch, 'to'):
                transformed_batch = transformed_batch.to(device)
            
            # Get embeddings
            with torch.no_grad():
                batch_embeddings = model(transformed_batch).cpu().numpy()
            
            # Pool embeddings
            pooled_embeddings = []
            for j in range(len(batch_seqs)):
                seq_embedding = batch_embeddings[j]
                
                if pooling == 'mean':
                    pooled = np.mean(seq_embedding, axis=0)
                elif pooling == 'max':
                    pooled = np.max(seq_embedding, axis=0)
                elif pooling == 'both':
                    mean_pool = np.mean(seq_embedding, axis=0)
                    max_pool = np.max(seq_embedding, axis=0)
                    pooled = np.concatenate([mean_pool, max_pool])
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")
                
                pooled_embeddings.append(pooled)
            
            # Stack and store
            batch_pooled = np.vstack(pooled_embeddings)
            all_embeddings.append(batch_pooled)
            
            # Aggressive memory clearing for all sequences
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                import gc
                gc.collect()
            
            # Additional memory clearing for problematic categories
            if category_name in ['long', 'medium'] or oom_count > 0:
                # Force garbage collection more aggressively
                gc.collect()
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                print(f"\nGPU OOM for {category_name} sequences (OOM #{oom_count}). Trying CPU...")
                
                # Reduce batch size if too many OOMs
                if oom_count >= max_oom_before_reduction and effective_batch_size > 1:
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(f"  Reduced batch size to {effective_batch_size} due to repeated OOMs")
                    oom_count = 0  # Reset counter
                
                # Move model to CPU for this batch
                model_cpu = model.cpu()
                try:
                    # Move transformed batch to CPU as well
                    if isinstance(transformed_batch, dict):
                        transformed_batch_cpu = {k: v.cpu() if hasattr(v, 'to') else v 
                                               for k, v in transformed_batch.items()}
                    elif hasattr(transformed_batch, 'to'):
                        transformed_batch_cpu = transformed_batch.cpu()
                    else:
                        transformed_batch_cpu = transformed_batch
                    
                    with torch.no_grad():
                        batch_embeddings = model_cpu(transformed_batch_cpu).numpy()
                    
                    # Process pooling
                    pooled_embeddings = []
                    for j in range(len(batch_seqs)):
                        seq_embedding = batch_embeddings[j]
                        if pooling == 'mean':
                            pooled = np.mean(seq_embedding, axis=0)
                        elif pooling == 'max':
                            pooled = np.max(seq_embedding, axis=0)
                        elif pooling == 'both':
                            mean_pool = np.mean(seq_embedding, axis=0)
                            max_pool = np.max(seq_embedding, axis=0)
                            pooled = np.concatenate([mean_pool, max_pool])
                        pooled_embeddings.append(pooled)
                    
                    batch_pooled = np.vstack(pooled_embeddings)
                    all_embeddings.append(batch_pooled)
                    
                    # Move model back to GPU
                    model = model.to(device)
                    print(f"  Successfully processed batch on CPU")
                    
                except Exception as cpu_error:
                    print(f"  CPU also failed: {cpu_error}")
                    print(f"  Skipping this batch of {len(batch_seqs)} sequences")
                    # Add zero embeddings for skipped sequences
                    zero_embeddings = np.zeros((len(batch_seqs), 1280))  # Default embedding size
                    all_embeddings.append(zero_embeddings)
                    model = model.to(device)
            else:
                raise e
    
    # Combine all batches
    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
        print(f"  Completed {category_name} sequences: {embeddings.shape}")
    else:
        embeddings = np.array([])
        print(f"  No {category_name} sequences processed")
    
    return embeddings

def generate_embeddings(csv_path, embedding_cache_path='embeddings.pkl', batch_size=32, pooling='mean'):
    """
    Generate RNA embeddings using AIDO.RNA model
    Averages embeddings along sequence length dimension to handle variable-length RNAs
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    embedding_cache_path : str
        Path to save/load cached embeddings
    batch_size : int
        Batch size for processing sequences (reduce if GPU OOM occurs)
    pooling : str
        Pooling strategy ('mean', 'max', or 'both')
        - 'mean': Average pooling along sequence length
        - 'max': Max pooling along sequence length
        - 'both': Concatenate mean and max pooling
    """
    print("="*50)
    print("STEP 1: GENERATING RNA EMBEDDINGS")
    print("="*50)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Inspect CSV structure for debugging
    inspect_csv_structure(csv_path)
    
    # Check if embeddings already exist
    if os.path.exists(embedding_cache_path):
        print(f"Loading cached embeddings from {embedding_cache_path}")
        with open(embedding_cache_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        return embeddings_data
    
    # Check for existing progress files
    progress_files = []
    base_path = embedding_cache_path.replace('.pkl', '')
    for file in os.listdir(os.path.dirname(embedding_cache_path) or '.'):
        if file.startswith(os.path.basename(base_path) + '_progress_') and file.endswith('.pkl'):
            progress_files.append(file)
    
    if progress_files:
        print(f"Found {len(progress_files)} progress files from previous run")
        # Sort by category order: short, medium, long
        category_order = {'short': 0, 'medium': 1, 'long': 2}
        progress_files.sort(key=lambda x: category_order.get(x.split('_progress_')[1].split('.')[0], 999))
        
        # Load the latest progress file
        latest_progress = progress_files[-1]
        progress_path = os.path.join(os.path.dirname(embedding_cache_path) or '.', latest_progress)
        
        with open(progress_path, 'rb') as f:
            progress_data = pickle.load(f)
        
        print(f"Found progress file: {latest_progress}")
        print(f"Already processed: {progress_data['total_processed']} sequences")
        print(f"Category: {progress_data.get('category', 'unknown')}")
        
        # We'll resume from this point
        resume_from_progress = True
        progress_data_loaded = progress_data
    else:
        resume_from_progress = False
        progress_data_loaded = None
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} sequences")
    
    # Debug: Check data types and sample values
    print(f"Sequence column data type: {df['sequence'].dtype}")
    print(f"Sample sequence values:")
    print(df['sequence'].head(10).tolist())
    
    # Clean and validate sequence data
    print("Cleaning and validating sequence data...")
    original_count = len(df)
    
    # Remove rows with missing or invalid sequences
    df = df.dropna(subset=['sequence'])
    df = df[df['sequence'].astype(str).str.strip() != '']
    
    # Convert sequences to strings and filter out non-string data
    df['sequence'] = df['sequence'].astype(str)
    df = df[df['sequence'].str.len() > 0]
    
    # Remove any sequences that are just whitespace or special characters
    df = df[~df['sequence'].str.match(r'^[\s\-\.]+$')]
    
    # Ensure we have the required columns for downstream analysis
    required_columns = ['sequence', 'selected_transcript_id', 'ensembl_gene_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    
    print(f"After cleaning: {len(df)} valid sequences (removed {original_count - len(df)} invalid entries)")
    print(f"Columns available: {list(df.columns)}")
    print(f"Using {pooling} pooling strategy")
    
    # Import AIDO.RNA model
    try:
        from modelgenerator.tasks import Embed
        model = Embed.from_config({"model.backbone": "aido_rna_650m"}).eval()
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        if torch.cuda.is_available():
            print(f"AIDO.RNA model loaded successfully on GPU: {torch.cuda.get_device_name()}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("AIDO.RNA model loaded successfully on CPU")
        use_real_model = True
    except ImportError:
        print("WARNING: AIDO.RNA not installed. Using random embeddings for demonstration.")
        print("To install: pip install modelgenerator")
        use_real_model = False
    
    # Extract sequences (already cleaned)
    sequences = df['sequence'].tolist()
    print(f"Extracted {len(sequences)} valid sequences for processing")
    
    # Verify we have the same number of sequences as rows in the dataframe
    if len(sequences) != len(df):
        raise ValueError(f"Mismatch: {len(sequences)} sequences vs {len(df)} dataframe rows")
    
    # Store transcript IDs for downstream analysis
    transcript_ids = df['selected_transcript_id'].tolist()
    ensembl_ids = df['ensembl_gene_id'].tolist()
    
    print(f"Stored {len(transcript_ids)} transcript IDs for downstream analysis")
    
    # Categorize sequences by length
    categories = categorize_sequences_by_length(sequences, transcript_ids, ensembl_ids, df)
    
    # Skip very long sequences entirely
    if categories['very_long']['sequences']:
        print(f"\nSkipping {len(categories['very_long']['sequences'])} very long sequences")
        print("These sequences will be excluded from downstream analysis")
    
    all_embeddings = []
    
    if use_real_model:
        # Get device for tensor operations
        device = next(model.parameters()).device
        
        # Monitor GPU memory and adjust batch size if needed
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU memory: {gpu_memory:.2f} GB")
            
            # Calculate safe batch size based on memory
            safe_batch_size = estimate_safe_batch_size(gpu_memory, 1000)  # Conservative estimate
            batch_size = min(batch_size, safe_batch_size)
            print(f"Adjusted batch size to {batch_size} based on GPU memory")
        
        # Process each category separately
        if resume_from_progress:
            # Resume from existing progress
            all_embeddings = [progress_data_loaded['embeddings']]
            all_transcript_ids = progress_data_loaded['transcript_ids']
            all_ensembl_ids = progress_data_loaded['ensembl_ids']
            all_df_indices = list(range(len(progress_data_loaded['embeddings'])))
            print(f"Resuming from {progress_data_loaded['category']} category with {len(progress_data_loaded['embeddings'])} embeddings")
        else:
            # Start fresh
            all_embeddings = []
            all_transcript_ids = []
            all_ensembl_ids = []
            all_df_indices = []
        
        # Save progress after each category
        def save_category_progress(category_name, embeddings, transcript_ids, ensembl_ids, df_indices, df):
            """Save progress after processing each category"""
            if len(embeddings) > 0:
                temp_embeddings = np.vstack(embeddings)
                temp_cache_path = embedding_cache_path.replace('.pkl', f'_progress_{category_name}.pkl')
                
                temp_data = {
                    'embeddings': temp_embeddings,
                    'ensembl_ids': transcript_ids,
                    'transcript_ids': transcript_ids,
                    'df': df.iloc[df_indices].reset_index(drop=True),
                    'pooling_type': pooling,
                    'embedding_dim': temp_embeddings.shape[1],
                    'category': category_name,
                    'total_processed': len(temp_embeddings)
                }
                
                with open(temp_cache_path, 'wb') as f:
                    pickle.dump(temp_data, f)
                
                print(f"  ✓ Saved {category_name} progress: {temp_embeddings.shape[0]} embeddings to {temp_cache_path}")
        
        # Process short sequences first (most efficient)
        if categories['short']['sequences'] and (not resume_from_progress or progress_data_loaded['category'] == 'short'):
            if resume_from_progress and progress_data_loaded['category'] == 'short':
                print("  Skipping short sequences (already completed)")
            else:
                short_embeddings = process_sequence_category(
                    model, categories['short'], device, batch_size, pooling, 'short'
                )
                if len(short_embeddings) > 0:
                    all_embeddings.append(short_embeddings)
                    all_transcript_ids.extend(categories['short']['transcript_ids'])
                    all_ensembl_ids.extend(categories['short']['ensembl_ids'])
                    all_df_indices.extend(categories['short']['df_indices'])
                    
                    # Save short sequences progress
                    save_category_progress('short', all_embeddings, all_transcript_ids, all_ensembl_ids, all_df_indices, df)
        
        # Process medium sequences
        if categories['medium']['sequences'] and (not resume_from_progress or progress_data_loaded['category'] in ['short', 'medium']):
            if resume_from_progress and progress_data_loaded['category'] == 'medium':
                print("  Skipping medium sequences (already completed)")
            else:
                medium_embeddings = process_sequence_category(
                    model, categories['medium'], device, batch_size, pooling, 'medium'
                )
                if len(medium_embeddings) > 0:
                    all_embeddings.append(medium_embeddings)
                    all_transcript_ids.extend(categories['medium']['transcript_ids'])
                    all_ensembl_ids.extend(categories['medium']['ensembl_ids'])
                    all_df_indices.extend(categories['medium']['df_indices'])
                    
                    # Save medium sequences progress
                    save_category_progress('medium', all_embeddings, all_transcript_ids, all_ensembl_ids, all_df_indices, df)
        
        # Process long sequences (CPU fallback)
        if categories['long']['sequences'] and (not resume_from_progress or progress_data_loaded['category'] in ['short', 'medium', 'long']):
            if resume_from_progress and progress_data_loaded['category'] == 'long':
                print("  Skipping long sequences (already completed)")
            else:
                long_embeddings = process_sequence_category(
                    model, categories['long'], device, batch_size, pooling, 'long'
                )
                if len(long_embeddings) > 0:
                    all_embeddings.append(long_embeddings)
                    all_transcript_ids.extend(categories['long']['transcript_ids'])
                    all_ensembl_ids.extend(categories['long']['ensembl_ids'])
                    all_df_indices.extend(categories['long']['df_indices'])
                    
                    # Save long sequences progress
                    save_category_progress('long', all_embeddings, all_transcript_ids, all_ensembl_ids, all_df_indices, df)
        
        # Verify data integrity before combining
        print(f"\nData integrity check before combining:")
        print(f"  Total embeddings to combine: {sum(len(emb) for emb in all_embeddings)}")
        print(f"  Total transcript IDs: {len(all_transcript_ids)}")
        print(f"  Total ensembl IDs: {len(all_ensembl_ids)}")
        print(f"  Total df indices: {len(all_df_indices)}")
        
        # Verify all lists have the same length
        total_embeddings = sum(len(emb) for emb in all_embeddings)
        assert total_embeddings == len(all_transcript_ids), f"Embedding count ({total_embeddings}) != transcript ID count ({len(all_transcript_ids)})"
        assert total_embeddings == len(all_ensembl_ids), f"Embedding count ({total_embeddings}) != ensembl ID count ({len(all_ensembl_ids)})"
        assert total_embeddings == len(all_df_indices), f"Embedding count ({total_embeddings}) != df index count ({len(all_df_indices)})"
        print("  ✓ All data arrays have matching lengths")
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            print(f"\nGenerated {pooling}-pooled embeddings with shape: {embeddings.shape}")
            print(f"  (Pooled from variable-length sequences to fixed {embeddings.shape[1]}-dim vectors)")
        else:
            embeddings = np.array([])
            print("No embeddings generated")
        
        # Update transcript IDs and ensembl IDs to match processed sequences
        transcript_ids = all_transcript_ids
        ensembl_ids = all_ensembl_ids
        
                # Update dataframe to only include processed sequences
        df = df.iloc[all_df_indices].reset_index(drop=True)
        
        # Final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    else:
        # Generate random embeddings for demonstration
        # Simulate the pooling process
        np.random.seed(42)
        embeddings = []
        embedding_dim = 1280  # AIDO.RNA 650M dimension
        
        for seq in sequences:
            seq_len = len(seq)
            # Simulate (seq_len, embedding_dim) embedding
            seq_embedding = np.random.randn(seq_len, embedding_dim)
            
            if pooling == 'mean':
                pooled = np.mean(seq_embedding, axis=0)
            elif pooling == 'max':
                pooled = np.max(seq_embedding, axis=0)
            elif pooling == 'both':
                mean_pool = np.mean(seq_embedding, axis=0)
                max_pool = np.max(seq_embedding, axis=0)
                pooled = np.concatenate([mean_pool, max_pool])
            
            embeddings.append(pooled)
        
        embeddings = np.vstack(embeddings)
        print(f"Generated demo {pooling}-pooled embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    embeddings_data = {
        'embeddings': embeddings,
        'ensembl_ids': ensembl_ids,
        'transcript_ids': transcript_ids,
        'df': df,
        'pooling_type': pooling,
        'embedding_dim': embeddings.shape[1]
    }
    
    with open(embedding_cache_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"Embeddings saved to {embedding_cache_path}")
    
    # Verify data integrity
    print("\nData integrity check:")
    print(f"  Embeddings shape: {embeddings_data['embeddings'].shape}")
    print(f"  Number of transcript IDs: {len(embeddings_data['transcript_ids'])}")
    print(f"  Number of ensembl IDs: {len(embeddings_data['ensembl_ids'])}")
    print(f"  DataFrame rows: {len(embeddings_data['df'])}")
    
    # Verify alignment
    assert len(embeddings_data['embeddings']) == len(embeddings_data['transcript_ids']), "Embeddings and transcript IDs mismatch"
    assert len(embeddings_data['embeddings']) == len(embeddings_data['ensembl_ids']), "Embeddings and ensembl IDs mismatch"
    assert len(embeddings_data['embeddings']) == len(embeddings_data['df']), "Embeddings and dataframe mismatch"
    print("  ✓ All data aligned correctly")
    
    # Print summary for downstream analysis
    print("\nDownstream analysis summary:")
    print(f"  Each embedding row corresponds to:")
    print(f"    - Transcript ID: {embeddings_data['transcript_ids'][0]} (example)")
    print(f"    - Ensembl ID: {embeddings_data['ensembl_ids'][0]} (example)")
    tissue_cols = ['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']
    available_tissues = [col for col in embeddings_data['df'].columns if col in tissue_cols]
    print(f"    - Available tissue columns: {available_tissues}")
    
    # Print processing summary
    print(f"\nProcessing summary:")
    print(f"  Original sequences: {original_count}")
    print(f"  Processed sequences: {len(embeddings_data['embeddings'])}")
    print(f"  Skipped sequences: {original_count - len(embeddings_data['embeddings'])}")
    if categories['very_long']['sequences']:
        print(f"  Very long sequences skipped: {len(categories['very_long']['sequences'])}")
    
    return embeddings_data

#=====================================
# 2. NEURAL NETWORK DEFINITION
#=====================================

class TranslationMLP(nn.Module):
    """Simple MLP for translation efficiency prediction"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super(TranslationMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Single output for regression
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

def train_neural_network(X_train, y_train, X_val, y_val, input_dim, 
                         epochs=100, lr=0.001, batch_size=64, verbose=False):
    """Train neural network model"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TranslationMLP(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).numpy()
            val_r2 = r2_score(y_val, val_pred)
        model.train()
        
        scheduler.step(val_r2)
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={total_loss:.4f}, Val R²={val_r2:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy()
    
    return model, val_pred

#=====================================
# 3. MODEL TRAINING & EVALUATION
#=====================================

def create_models():
    """Create all regression models to test"""
    models = {
        # Linear models
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        
        # Tree-based models
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1),
        
        # Other models
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance')
    }
    return models

def evaluate_models(embeddings_data, tissues=['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']):
    """Evaluate all models on all tissues"""
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("="*50)
    
    X = embeddings_data['embeddings']
    df = embeddings_data['df']
    
    # Verify data alignment for downstream analysis
    print(f"\nData alignment verification for model training:")
    print(f"  Embeddings shape: {X.shape}")
    print(f"  DataFrame shape: {df.shape}")
    
    # Verify we have transcript IDs for downstream analysis
    if 'transcript_ids' in embeddings_data:
        transcript_ids = embeddings_data['transcript_ids']
        print(f"  Transcript IDs: {len(transcript_ids)}")
        # Create a mapping for easy lookup
        transcript_to_idx = {tid: idx for idx, tid in enumerate(transcript_ids)}
        
        # Verify alignment
        assert len(X) == len(df), f"Embeddings ({len(X)}) and dataframe ({len(df)}) have different lengths"
        assert len(X) == len(transcript_ids), f"Embeddings ({len(X)}) and transcript IDs ({len(transcript_ids)}) have different lengths"
        print("  ✓ Embeddings, dataframe, and transcript IDs are properly aligned")
        
        # Verify tissue columns exist
        tissue_cols = ['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']
        available_tissues = [col for col in df.columns if col in tissue_cols]
        print(f"  Available tissue columns: {available_tissues}")
        
        if len(available_tissues) != 6:
            print(f"  ⚠️  Warning: Only {len(available_tissues)}/6 tissue columns found")
        
    else:
        print("Warning: No transcript IDs found in embeddings data")
        transcript_to_idx = {}
    
    # Initialize results storage
    results = {
        'model': [],
        'tissue': [],
        'r2_score': [],
        'rmse': [],
        'mae': [],
        'pearson_r': []
    }
    
    # Get models
    models = create_models()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train and evaluate for each tissue
    for tissue in tissues:
        print(f"\n--- Evaluating models for {tissue.upper()} ---")
        
        # Check if tissue column exists
        if tissue not in df.columns:
            print(f"  ⚠️  Warning: Tissue '{tissue}' not found in dataframe. Skipping.")
            continue
            
        y = df[tissue].values
        
        # Remove NaN values if any
        valid_idx = ~np.isnan(y)
        X_tissue = X_scaled[valid_idx]
        y_tissue = y[valid_idx]
        
        print(f"  Data for {tissue}: {len(y_tissue)} samples (removed {len(y) - len(y_tissue)} NaN values)")
        print(f"  Features shape: {X_tissue.shape}, Targets shape: {y_tissue.shape}")
        
        if len(y_tissue) == 0:
            print(f"  ⚠️  Warning: No valid data for tissue '{tissue}'. Skipping.")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tissue, y_tissue, test_size=0.2, random_state=42
        )
        
        # Evaluate each model
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            pearson, _ = pearsonr(y_test, y_pred)
            
            # Store results
            results['model'].append(model_name)
            results['tissue'].append(tissue)
            results['r2_score'].append(r2)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['pearson_r'].append(pearson)
            
            print(f"  {model_name:15s}: R²={r2:.4f}, RMSE={rmse:.4f}")
        
        # Train neural network
        print(f"  Training Neural Network...")
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        nn_model, y_pred_nn = train_neural_network(
            X_train_nn, y_train_nn, X_val_nn, y_val_nn, 
            input_dim=X_train.shape[1], epochs=50, verbose=False
        )
        
        # Evaluate NN on test set
        nn_model.eval()
        with torch.no_grad():
            y_pred = nn_model(torch.FloatTensor(X_test)).numpy()
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        pearson, _ = pearsonr(y_test, y_pred)
        
        results['model'].append('Neural Network')
        results['tissue'].append(tissue)
        results['r2_score'].append(r2)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['pearson_r'].append(pearson)
        
        print(f"  {'Neural Network':15s}: R²={r2:.4f}, RMSE={rmse:.4f}")
    
    return pd.DataFrame(results)

#=====================================
# 4. VISUALIZATION
#=====================================

def plot_model_comparison(results_df, save_path='model_comparison.png'):
    """Create comprehensive visualization of model performance"""
    print("\n" + "="*50)
    print("STEP 3: VISUALIZATION")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bar plot of R² scores by model
    ax1 = axes[0, 0]
    model_avg = results_df.groupby('model')['r2_score'].mean().sort_values(ascending=False)
    model_std = results_df.groupby('model')['r2_score'].std()
    
    bars = ax1.bar(range(len(model_avg)), model_avg.values, yerr=model_std.values, 
                   capsize=5, alpha=0.8)
    ax1.set_xticks(range(len(model_avg)))
    ax1.set_xticklabels(model_avg.index, rotation=45, ha='right')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Average R² Score Across All Tissues', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Color best performer
    best_idx = model_avg.argmax()
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)
    
    # 2. Heatmap of R² scores (Model × Tissue)
    ax2 = axes[0, 1]
    pivot_r2 = results_df.pivot(index='model', columns='tissue', values='r2_score')
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'R² Score'})
    ax2.set_title('R² Score Heatmap: Models × Tissues', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tissue', fontsize=12)
    ax2.set_ylabel('Model', fontsize=12)
    
    # 3. RMSE comparison
    ax3 = axes[1, 0]
    model_rmse = results_df.groupby('model')['rmse'].mean().sort_values()
    model_rmse_std = results_df.groupby('model')['rmse'].std()
    
    bars = ax3.barh(range(len(model_rmse)), model_rmse.values, 
                    xerr=model_rmse_std.values, capsize=5, alpha=0.8)
    ax3.set_yticks(range(len(model_rmse)))
    ax3.set_yticklabels(model_rmse.index)
    ax3.set_xlabel('RMSE (Lower is Better)', fontsize=12)
    ax3.set_title('Average RMSE Across All Tissues', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Color best performer
    bars[0].set_color('green')
    bars[0].set_alpha(1.0)
    
    # 4. Box plot of R² scores by tissue
    ax4 = axes[1, 1]
    tissue_data = [results_df[results_df['tissue'] == t]['r2_score'].values 
                   for t in results_df['tissue'].unique()]
    bp = ax4.boxplot(tissue_data, labels=results_df['tissue'].unique(), patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('R² Score', fontsize=12)
    ax4.set_xlabel('Tissue', fontsize=12)
    ax4.set_title('R² Score Distribution by Tissue', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Translation Efficiency Prediction: Model Performance Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Model comparison plot saved to {save_path}")

def plot_best_model_analysis(results_df, embeddings_data, save_path='best_model_analysis.png'):
    """Detailed analysis of the best performing model"""
    # Find best model
    best_model_name = results_df.groupby('model')['r2_score'].mean().idxmax()
    print(f"\nBest performing model: {best_model_name}")
    print(f"Average R² score: {results_df[results_df['model']==best_model_name]['r2_score'].mean():.4f}")
    
    # Prepare data
    X = embeddings_data['embeddings']
    df = embeddings_data['df']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create best model instance
    models = create_models()
    best_model = models[best_model_name]
    
    # Create visualization
    tissues = ['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, tissue in enumerate(tissues):
        ax = axes[idx]
        
        # Prepare tissue data
        y = df[tissue].values
        valid_idx = ~np.isnan(y)
        X_tissue = X_scaled[valid_idx]
        y_tissue = y[valid_idx]
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_tissue, y_tissue, test_size=0.2, random_state=42
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # Add diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)
        
        # Add confidence band
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
        predict_mean_se = std_err * np.sqrt(1/len(y_test) + (y_test - np.mean(y_test))**2 / np.sum((y_test - np.mean(y_test))**2))
        margin = 1.96 * predict_mean_se
        ax.fill_between(y_test, y_test - margin, y_test + margin, alpha=0.15, color='gray')
        
        # Labels and title
        ax.set_xlabel('Actual Translation Efficiency', fontsize=10)
        ax.set_ylabel('Predicted Translation Efficiency', fontsize=10)
        ax.set_title(f'{tissue.capitalize()}\nR²={r2:.3f}, RMSE={rmse:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Best Model ({best_model_name}): Predicted vs Actual Values', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Best model analysis saved to {save_path}")

def plot_learning_curves(results_df, embeddings_data, save_path='learning_curves.png'):
    """Plot learning curves for top models"""
    # Get top 3 models
    top_models = results_df.groupby('model')['r2_score'].mean().nlargest(3).index.tolist()
    
    X = embeddings_data['embeddings']
    df = embeddings_data['df']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    tissues = ['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']
    
    for idx, tissue in enumerate(tissues):
        ax = axes[idx // 3, idx % 3]
        
        # Prepare tissue data
        y = df[tissue].values
        valid_idx = ~np.isnan(y)
        X_tissue = X_scaled[valid_idx]
        y_tissue = y[valid_idx]
        
        # Plot learning curves for top models
        for model_name in top_models:
            models = create_models()
            model = models[model_name]
            
            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_tissue, y_tissue, 
                cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2'
            )
            
            # Plot
            ax.plot(train_sizes, np.mean(val_scores, axis=1), 
                   label=model_name, marker='o', markersize=4)
            ax.fill_between(train_sizes, 
                           np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                           np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                           alpha=0.1)
        
        ax.set_xlabel('Training Size', fontsize=10)
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title(f'{tissue.capitalize()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Learning Curves: Top 3 Models Across Tissues', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Learning curves saved to {save_path}")

#=====================================
# 5. MAIN PIPELINE
#=====================================

def run_pipeline(csv_path, output_dir='results', pooling='mean', batch_size=32):
    """
    Run the complete analysis pipeline
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    output_dir : str
        Directory to save results
    pooling : str
        Pooling strategy for embeddings ('mean', 'max', or 'both')
    batch_size : int
        Batch size for embedding generation (reduce if GPU OOM occurs)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("RNA TRANSLATION EFFICIENCY PREDICTION PIPELINE")
    print("="*50)
    
    # Step 1: Generate embeddings
    embedding_cache = os.path.join(output_dir, f'embeddings_{pooling}.pkl')
    embeddings_data = generate_embeddings(csv_path, embedding_cache, batch_size=batch_size, pooling=pooling)
    
    # Step 2: Train and evaluate models
    results_df = evaluate_models(embeddings_data)
    
    # Save results
    results_path = os.path.join(output_dir, f'model_results_{pooling}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Step 3: Visualization
    plot_model_comparison(results_df, os.path.join(output_dir, f'model_comparison_{pooling}.png'))
    plot_best_model_analysis(results_df, embeddings_data, 
                           os.path.join(output_dir, f'best_model_analysis_{pooling}.png'))
    plot_learning_curves(results_df, embeddings_data, 
                        os.path.join(output_dir, f'learning_curves_{pooling}.png'))
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    # Best model per tissue
    print("\nBest model for each tissue:")
    for tissue in results_df['tissue'].unique():
        tissue_results = results_df[results_df['tissue'] == tissue]
        best = tissue_results.loc[tissue_results['r2_score'].idxmax()]
        print(f"  {tissue:10s}: {best['model']:15s} (R²={best['r2_score']:.4f})")
    
    # Overall best model
    print("\nOverall best model (averaged across tissues):")
    avg_scores = results_df.groupby('model').agg({
        'r2_score': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'pearson_r': 'mean'
    }).round(4)
    best_overall = avg_scores['r2_score'].idxmax()
    print(f"  {best_overall}: R²={avg_scores.loc[best_overall, 'r2_score']:.4f}, "
          f"RMSE={avg_scores.loc[best_overall, 'rmse']:.4f}, "
          f"MAE={avg_scores.loc[best_overall, 'mae']:.4f}")
    
    print(f"\nPooling strategy used: {pooling}")
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(f"All results saved to: {output_dir}/")
    print("="*50)
    
    return results_df, embeddings_data

#=====================================
# 6. USAGE
#=====================================

if __name__ == "__main__":
    # Run the pipeline with mean pooling (recommended)
    csv_path = "translation_efficiency_with_mrna.csv"  # Your CSV file path
    results_df, embeddings_data = run_pipeline(
        csv_path, 
        output_dir='translation_results',
        pooling='mean',  # Options: 'mean', 'max', 'both'
        batch_size=8     # Conservative batch size for large datasets
    )
    
    # Optional: Compare different pooling strategies
    def compare_pooling_strategies(csv_path, batch_size=16):
        """Compare model performance with different pooling strategies"""
        pooling_results = {}
        
        for pooling in ['mean', 'max', 'both']:
            print(f"\n{'='*50}")
            print(f"Testing {pooling.upper()} pooling")
            print('='*50)
            
            results_df, _ = run_pipeline(
                csv_path,
                output_dir=f'results_{pooling}',
                pooling=pooling,
                batch_size=batch_size
            )
            
            # Store average R² score
            avg_r2 = results_df.groupby('model')['r2_score'].mean().max()
            pooling_results[pooling] = avg_r2
        
        print("\n" + "="*50)
        print("POOLING STRATEGY COMPARISON")
        print("="*50)
        for strategy, score in pooling_results.items():
            print(f"{strategy:10s}: Best R² = {score:.4f}")
        
        return pooling_results
    
    # Uncomment to compare pooling strategies:
    # pooling_comparison = compare_pooling_strategies(csv_path)
    
    # Optional: Export best model for production use
    def export_best_model(results_df, embeddings_data, output_path='best_model.pkl'):
        """Export the best trained model for future use"""
        best_model_name = results_df.groupby('model')['r2_score'].mean().idxmax()
        
        X = embeddings_data['embeddings']
        df = embeddings_data['df']
        
        # Train on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = create_models()
        best_model = models[best_model_name]
        
        # Train on all tissues combined (or you can train separate models)
        all_predictions = {}
        for tissue in ['brain', 'heart', 'kidney', 'liver', 'lung', 'retina']:
            y = df[tissue].values
            valid_idx = ~np.isnan(y)
            X_tissue = X_scaled[valid_idx]
            y_tissue = y[valid_idx]
            
            model = models[best_model_name]
            model.fit(X_tissue, y_tissue)
            all_predictions[tissue] = model
        
        # Save
        export_data = {
            'models': all_predictions,
            'scaler': scaler,
            'model_name': best_model_name,
            'pooling_type': embeddings_data.get('pooling_type', 'mean')
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"Best model exported to {output_path}")
        return export_data
    
    # Uncomment to export:
    export_best_model(results_df, embeddings_data, 'translation_results/best_model.pkl')