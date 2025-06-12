# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import torch
import time
from datetime import datetime
from tqdm import tqdm

# Load data with sample
data = pd.read_csv('../data/PS_20174392719_1491204439457_log.csv')
sample_size = min(len(data), 100000)  # Limit to 100k rows
data = data.sample(n=sample_size, random_state=42)

# Initialize metadata and model
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=100,
    batch_size=500,
    #cuda=torch.cuda.is_available(),
    verbose=True  # Enable verbose output
)

print(f"Using GPU: {torch.cuda.is_available()}")
start_time = time.time()
synthesizer.fit(data)
print(f"Total training time: {time.time() - start_time:.2f}s")

synthesizer.save('ctgan_model_CPU.pkl')
synthetic_data = synthesizer.sample(1000)


# synthesizer.get_loss_values()
# fig = synthesizer.get_loss_values_plot()
# fig.show()