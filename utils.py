import matplotlib.pyplot as plt

def plot_dataset_stats(dataset):
    caption_lengths = [len(row['revised_caption'].split()) for row in dataset]
    durations = [row['action_duration'] for row in dataset]
    
    # Create subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Caption Word Lengths ---
    ax1.hist(caption_lengths, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Caption Word Length Distribution')
    ax1.set_xlabel('Number of Words')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # --- Plot 2: Action Durations ---
    ax2.hist(durations, bins=20, color='salmon', edgecolor='black')
    ax2.set_title('Action Duration Distribution')
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(left=0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Final layout
    plt.tight_layout()
    plt.show()