from datasets import load_dataset

def load_eurosat(split="train"):
    # Load dataset (downloads + caches into data/raw)
    dataset = load_dataset(
        "blanchon/EuroSAT_RGB",
        cache_dir="data/raw"
    )

    # Save a copy to disk in our own structure
    dataset.save_to_disk("data/raw/eurosat_rgb")

    return dataset[split]

if __name__ == "__main__":
    ds = load_eurosat()
    print(ds)