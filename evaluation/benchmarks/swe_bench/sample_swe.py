import datasets

def sample_per_difficulty_level(data, num_samples):
    """
    Sample the specified number of samples from each difficulty level in the dataset.
    
    Args:
        data (datasets.Dataset): The dataset to sample from.
        num_samples (int): The number of samples to take from each difficulty level.
        
    Returns:
        datasets.Dataset: A new dataset containing the sampled data.
    """
    sampled_data = []
    for difficulty in data.unique('difficulty'):
        difficulty_data = data.filter(lambda x: x['difficulty'] == difficulty)
        print(f"Sampling {num_samples} samples from difficulty level: {difficulty}")
        try:
            sampled_data.append(difficulty_data.shuffle(seed=42).select(range(num_samples)))
        except:
            print(f"Not enough samples available for difficulty level {difficulty}. Sampling all available samples.")
            sampled_data.append(difficulty_data.shuffle(seed=42))

        print(f"Instance ids of difficulty level {difficulty}: {sampled_data[-1]['instance_id']}")
    
    return datasets.concatenate_datasets(sampled_data)

data = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
# Sample 5 samples from each difficulty level
sampled_data = sample_per_difficulty_level(data, 5)
# print the instance_ids of the sampled data
instance_id_list = sampled_data['instance_id']
print(instance_id_list)
