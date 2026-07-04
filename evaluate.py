from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example inputs
student_explanation = "Photosynthesis is the process by which plants make food using sunlight."
reference_concept = "Photosynthesis converts light energy into chemical energy in plants."

# Generate embeddings
student_embedding = model.encode(student_explanation)
reference_embedding = model.encode(reference_concept)