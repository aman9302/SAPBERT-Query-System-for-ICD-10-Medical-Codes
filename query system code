# Inference on SAPBERT code
# Aman Nair R

import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

def load_data(datafile):
    result = []

    # Load the data
    with open(datafile, 'r') as file:
        for line in file:
            columns = line.strip().split("|")
            if len(columns) >= 17:
                column_value = columns[11]
                if column_value in ["ICD10CM", "ICD10"]:
                    name = columns[13]
                    id = columns[14]
                    result.append((name, id))

    return result

all_data = load_data(datafile="/content/drive/MyDrive/UMLS/MRCONSO.RRF")

print(all_data[:10])
len(all_data)
unique_values = list(set(all_data))
print(unique_values)
len(unique_values)
ids_list = [item[0] for item in unique_values]
names_list = [item[1] for item in unique_values]
len(ids_list)
len(names_list)

def encode_labels(names_list):
    # Loading Autotokenizer and Automodel from sapbert
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # Encoding the labels
    bs = 128
    all_reps = []
    all_ids_encoded = []  # To store the encoded IDs

    for i in tqdm(np.arange(0, len(names_list), bs)):
        toks = tokenizer.batch_encode_plus(names_list[i:i+bs],
                                            padding="max_length",
                                            max_length=25,
                                            truncation=True,
                                            return_tensors="pt")

        output = model(**toks)
        cls_rep = output[0][:, 0, :].cpu().detach().numpy()

        all_reps.append(cls_rep)
        all_ids_encoded.extend(ids_list[i:i+bs])  # Appending the IDs for the corresponding batch

    all_reps_emb = np.concatenate(all_reps, axis=0)

    return all_reps_emb, all_ids_encoded

all_reps_emb, all_ids_encoded = encode_labels(names_list)

# Save the all_ids_encoded list to a file
np.savetxt("all_reps_emb.txt", all_reps_emb, delimiter=",", fmt="%d")

len(all_reps_emb)


file_path = "/content/faissdb.index"

def create_faiss_index(all_reps_emb, file_path):
    # Convert the list into a numpy array
    all_reps_emb = np.array(all_reps_emb, dtype=np.float32)

    # Get the dimensionality of the vectors
    vector_dim = all_reps_emb.shape[1]

    # Print the number of vectors
    num_vectors = all_reps_emb.shape[0]
    print("Number of vectors:", num_vectors)

    # Creating a Faiss index
    index = faiss.IndexFlatIP(vector_dim)

    # Adding the embeddings to the index
    index.add(all_reps_emb)

    # Saving the index
    faiss.write_index(index, file_path)

create_faiss_index(all_reps_emb, file_path)

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

file_path = "/content/faissdb.index"

# Load the Faiss index
index = faiss.read_index(file_path)

# Code for encoding the query using tokenizer and model
query = "tuberculosis"
query_toks = tokenizer.batch_encode_plus(
    [query],
    padding="max_length",
    max_length=25,
    truncation=True,
    return_tensors="pt"
)
query_output = model(**query_toks)
query_cls_rep = query_output[0][:, 0, :]
query_vector = query_cls_rep.cpu().detach().numpy().astype(np.float32)

def query_faiss_index(query_vector, k):
    # Perform a search on the index
    query_vector = np.array(query_vector, dtype=np.float32)
    query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    # Retrieve the corresponding texts and ICD10 codes
    texts = [names_list[index] for index in indices[0]]
    icd10_codes = [ids_list[index] for index in indices[0]]

    # Calculate cosine similarity between the query vector and retrieved vectors
    similarity = cosine_similarity(query_vector, all_reps_emb[indices[0]])

    return distances, indices, query_vector, texts, icd10_codes, similarity

k = 5

# Query the Faiss index
distances, indices, query_vector, texts, icd10_codes, similarity = query_faiss_index(query_vector, k)

# Print the results
print("Query Text:", query)
print("Top", k, "Results:")
for distance, index, text, icd10_code, sim in zip(distances[0], indices[0], texts, icd10_codes, similarity[0]):
    print("Description:", text)
    print("ICD10 Code:", icd10_code)
    print("Distance:", distance)
    print("Similarity:", sim)
    print()



