import numpy as np
import os


means = {
    (0, 1): np.array([-0.9, -1.1]),
    (0, 2): np.array([0.8, 0.75]),
    (1, 1): np.array([-1.1, 0.9]),
    (1, 2): np.array([0.9, -0.75])
}
cov_matrix = np.array([[0.75, 0], [0, 1.25]])


def generate_labels(n_samples):
    return np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])


def generate_samples(label):
    sub_dist = np.random.choice([1, 2])  
    mean = means[(label, sub_dist)]
    return np.random.multivariate_normal(mean, cov_matrix)


def generate_dataset(n_samples):
    labels = generate_labels(n_samples)
    data = np.array([generate_samples(label) for label in labels])
    return data, labels


datasets = {
    "D20_train": generate_dataset(20),
    "D200_train": generate_dataset(200),
    "D2000_train": generate_dataset(2000),
    "D10K_validate": generate_dataset(10000)
}


save_dir = "generated_datasets"
os.makedirs(save_dir, exist_ok=True)

for name, (data, labels) in datasets.items():
    np.savez_compressed(os.path.join(save_dir, f"{name}.npz"), data=data, labels=labels)
    print(f"{name} saved.")


def load_dataset(dataset_name):
    with np.load(os.path.join(save_dir, f"{dataset_name}.npz")) as data:
        return data['data'], data['labels']


loaded_data, loaded_labels = load_dataset("D20_train")
print(f"Loaded D20_train: {loaded_data.shape}, {loaded_labels.shape}")
